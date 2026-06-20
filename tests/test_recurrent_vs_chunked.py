#!/usr/bin/env python3
"""Equivalence test: the recurrent (decode, step-by-step) form == the chunked (parallel training)
form of routed RoLA, on RANDOM inputs. This is the canonical linear-attention check (FLA's own
suite asserts chunk == fused_recurrent == naive): a chunked kernel and its recurrent counterpart
must produce the same sequence output, so a bug in either's state bookkeeping shows up as drift.

RoLA has no bespoke recurrent kernel — the recurrent form is virtual-heads over the canonical
`fused_recurrent_simple_gla` (q/k replicated across the nc states into the head axis, v carrying the
write gate + a denominator ones-column, read-combine outside). The chunked form is the routed
`chunk_simple_gla` (shared-gram Triton). We assert max-rel-error over the full sequence across
many random seeds × {RLA, GLA} × {global, kappa, per_state} × nc.

Run on CUDA, unbuffered. Forward-only (decode has no backward).
"""
import sys
import torch

from fla_rola.ops.simple_gla import chunk_simple_gla
from fla_rola.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla
from fla_rola.ops.rola.naive import _rola_global_ref, _rola_gla_ref     # naive O(L^2) DIRECT oracle (no vh glue)

DEV = "cuda"
B, H, DQK, DV = 4, 4, 16, 16
EPS = 1e-5
KAPPA = 0.5
P = lambda *a: print(*a, flush=True)


def _mk(L, nc, gla, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    t = lambda *s: torch.randn(*s, device=DEV, generator=g, dtype=torch.float32)
    q, k = t(B, L, H, DQK).abs(), t(B, L, H, DQK).abs()       # elu+1-like: positive features
    v = t(B, L, H, DV)
    r = torch.softmax(t(B, L, H, nc), -1)
    w = torch.softmax(t(B, L, H, nc), -1)
    ld = (-torch.rand(B, L, H, nc, device=DEV, generator=g) * 0.5).clamp(min=-2.5) if gla else None
    return q, k, v, r, w, ld


def _vh_expand(q, k, v, w, ld, nc):
    """[B,L,H,*] -> virtual-head [B,L,H*nc,*]; v carries the write gate + a den ones-column."""
    Bq, L = q.shape[0], q.shape[1]
    qv = q.unsqueeze(3).expand(Bq, L, H, nc, DQK).reshape(Bq, L, H * nc, DQK)
    kv = k.unsqueeze(3).expand(Bq, L, H, nc, DQK).reshape(Bq, L, H * nc, DQK)
    v1 = torch.cat([v, torch.ones_like(v[..., :1])], -1)
    vv = (v1.unsqueeze(3) * w.unsqueeze(-1)).reshape(Bq, L, H * nc, DV + 1)
    gv = ld.reshape(Bq, L, H * nc).float() if ld is not None else None
    return qv, kv, vv, gv


def _vh_combine(o_aug, r, nc, norm):
    """o_aug:[B,L,H*nc,DV+1] per-state (num|den) -> combined [B,L,H,DV] under the given norm."""
    Bq, L = o_aug.shape[0], o_aug.shape[1]
    o = o_aug.view(Bq, L, H, nc, DV + 1)
    num, den = o[..., :DV], o[..., DV]
    if norm == 'kappa':
        r = r * (den.abs() + EPS).pow(-KAPPA)
    elif norm == 'per_state':
        r = r / (den.abs() + EPS)
    return (num * r.unsqueeze(-1)).sum(3) / ((den * r).sum(3).unsqueeze(-1) + EPS)


def _chunked(q, k, v, r, w, ld, nc, norm):
    qv, kv, vv, gv = _vh_expand(q, k, v, w, ld, nc)
    o_aug, _ = chunk_simple_gla(qv, kv, vv, g=gv, scale=1.0)
    return _vh_combine(o_aug.float(), r, nc, norm)


def _recurrent(q, k, v, r, w, ld, nc, norm):
    """Step-by-step decode: one fused_recurrent step per token, carrying the state."""
    L = q.shape[1]
    state, outs = None, []
    for t in range(L):
        s = slice(t, t + 1)
        qv, kv, vv, gv = _vh_expand(q[:, s], k[:, s], v[:, s], w[:, s],
                                    ld[:, s] if ld is not None else None, nc)
        o, state = fused_recurrent_simple_gla(qv, kv, vv, g=gv, scale=1.0,
                                              initial_state=state, output_final_state=True)
        outs.append(_vh_combine(o.float(), r[:, s], nc, norm))
    return torch.cat(outs, 1)


def _naive(q, k, v, r, w, ld, gla):
    """DIRECT O(L^2) global-norm oracle (no vh glue): O=(G∘R∘causal)@v / den. Human-verifiable
    ground truth. r=read, w=write — the oracle takes (q,k,v,wg,rg[,ld])."""
    if gla:
        return _rola_gla_ref(q, k, v, w, r, ld, normalized=True)
    return _rola_global_ref(q, k, v, w, r)


def _routed(q, k, v, r, w, ld, nc, norm):
    """The first-class chunk_rola operator — the third leg of vh == routed-kernel == naive. It owns
    the whole recipe (den pre-pass, read-gate rescale, numerator-only readout, divide); norm selects
    global/kappa/per_state. Directly exercises the public method the LM layer calls."""
    from fla_rola.ops.rola import chunk_rola
    kap = (torch.full((q.shape[0], q.shape[1], H, 1), KAPPA, device=q.device, dtype=q.dtype)
           if norm == 'kappa' else None)
    return chunk_rola(q, k, v, r=r, w=w, g=ld, norm=norm, kappa=kap, scale=1.0)


def _relmax(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-9)


def main():
    if not torch.cuda.is_available():
        P("SKIP: needs CUDA"); return 0
    P("device: " + torch.cuda.get_device_name(0))
    P(f"chunk (chunk_simple_gla) == recurrent (step fused_recurrent) == naive (direct O(L^2) oracle), "
      f"B={B} H={H} dqk=dv={DQK}, L=64 | max-rel over random seeds")
    SEEDS = range(6)                       # random-sampled inputs (reproducible)
    NCS = (16, 64)
    DVS = (16, 32, 64)                     # dv=16: un-tiled (ND_V=1); 32/64: engage the value-tiling (ND_V>=2)
    TOL = 5e-3          # clean residuals are ~1.5e-3 (routed==naive); 5e-3 = ~3x headroom, catches a
                        # ~1% uniform error (the audit's MUT-1 blind spot was the old loose 3e-2).
    res = {}
    global DV
    for DV in DVS:                         # routed kernel's value tile follows from dv -> exercises every ND_V
        for gla in (False, True):
            for nc in NCS:
                for norm in ('global', 'kappa', 'per_state'):
                    w_rc, w_cn, w_rn, w_kc = 0.0, 0.0, 0.0, 0.0   # rec-chunk, chunk-naive, rec-naive, routed-chunk
                    w_kn = 0.0                                     # routed-naive (global only)
                    for seed in SEEDS:
                        q, k, v, r, w, ld = _mk(64, nc, gla, seed)
                        o_chunk = _chunked(q, k, v, r, w, ld, nc, norm)
                        o_rec = _recurrent(q, k, v, r, w, ld, nc, norm)
                        o_routed = _routed(q, k, v, r, w, ld, nc, norm)        # the routed (split) kernel
                        w_rc = max(w_rc, _relmax(o_rec, o_chunk))
                        w_kc = max(w_kc, _relmax(o_routed, o_chunk))           # vh == routed-kernel
                        # naive direct oracle exists for the GLOBAL norm (no vh glue) — the 3-way anchor
                        # that catches a shared vh-expand/combine bug recurrent==chunked alone would miss.
                        if norm == 'global':
                            o_naive = _naive(q, k, v, r, w, ld, gla)
                            w_cn = max(w_cn, _relmax(o_chunk, o_naive))
                            w_rn = max(w_rn, _relmax(o_rec, o_naive))
                            w_kn = max(w_kn, _relmax(o_routed, o_naive))
                    tag = f"dv={DV:<3d} {'GLA' if gla else 'RLA'} nc={nc:<3d} norm={norm}"
                    if norm == 'global':
                        ok = max(w_rc, w_cn, w_rn, w_kc, w_kn) < TOL
                        P(f"  {tag:30s} rec==chunk={w_rc:.1e} chunk==naive={w_cn:.1e} rec==naive={w_rn:.1e}"
                          f" routed==chunk={w_kc:.1e} routed==naive={w_kn:.1e}  {'PASS' if ok else 'FAIL'}")
                    else:
                        ok = max(w_rc, w_kc) < TOL
                        P(f"  {tag:30s} rec==chunk={w_rc:.1e} routed==chunk={w_kc:.1e} (norm-rescale; naive=global only)"
                          f"  {'PASS' if ok else 'FAIL'}")
                    res[tag] = ok
    allok = all(res.values())
    P(f"\n-> {'ALL GREEN' if allok else 'NOT GREEN'}")
    return 0 if allok else 1


def _grad_through(fwd, q, k, v, r, w, ld, gla, coef):
    """autograd grads of (fwd(...)*coef).sum() w.r.t. (q,k,v,r,w[,ld]) — the backward source of truth."""
    ins = [x.clone().requires_grad_() for x in ([q, k, v, r, w] + ([ld] if gla else []))]
    ld_in = ins[5] if gla else None
    o = fwd(ins[0], ins[1], ins[2], ins[3], ins[4], ld_in)
    return torch.autograd.grad((o * coef).sum(), ins)


def backward_gate():
    """INTER backward: the routed kernel's analytic grad must equal autograd through TWO independent
    implementations — the naive O(L^2) oracle AND the virtual-heads chunk_simple_gla — across dv (so
    the value-tiled backward is what's checked). global norm (the naive oracle exists there). Run at
    BT=16 so the fp32 value-tiled backward fits a 99KB card (rigorous ~1e-3, not the bf16 floor)."""
    import fla_rola.ops.rola.chunk as _C
    _C._CHUNK = 16
    _C._CHUNK_FWD = 16
    global DV
    P("\n=== INTER backward | routed-kernel grad == autograd(naive oracle) == autograd(vh-chunk), norm=global, BT=16 fp32 ===")
    SEEDS = range(3)
    TOL = 8e-3          # clean ~4e-3 (routed==vhchunk, two distinct fp32 impls); 8e-3 = 2x, catches the
                        # ~2% MUT-1 class. (Backward across two impls is intrinsically looser than fwd.)
    res = {}
    for DV in (16, 32, 64):
        for gla in (False, True):
            for nc in (16, 64):
                w_kn = w_kc = w_cn = 0.0   # routed-vs-naive, routed-vs-vhchunk, vhchunk-vs-naive (grads)
                for seed in SEEDS:
                    q, k, v, r, w, ld = _mk(64, nc, gla, seed)
                    coef = torch.randn(*v.shape, device=DEV)
                    g_routed = _grad_through(lambda a, b, c, d, e, f: _routed(a, b, c, d, e, f, nc, 'global'),
                                             q, k, v, r, w, ld, gla, coef)
                    g_naive = _grad_through(lambda a, b, c, d, e, f: _naive(a, b, c, d, e, f, gla),
                                            q, k, v, r, w, ld, gla, coef)
                    g_chunk = _grad_through(lambda a, b, c, d, e, f: _chunked(a, b, c, d, e, f, nc, 'global'),
                                            q, k, v, r, w, ld, gla, coef)
                    w_kn = max(w_kn, max(_relmax(a, b) for a, b in zip(g_routed, g_naive)))
                    w_kc = max(w_kc, max(_relmax(a, b) for a, b in zip(g_routed, g_chunk)))
                    w_cn = max(w_cn, max(_relmax(a, b) for a, b in zip(g_chunk, g_naive)))
                tag = f"dv={DV:<3d} {'GLA' if gla else 'RLA'} nc={nc}"
                ok = max(w_kn, w_kc, w_cn) < TOL
                P(f"  {tag:22s} routed==naive={w_kn:.1e} routed==vhchunk={w_kc:.1e} vhchunk==naive={w_cn:.1e}  {'PASS' if ok else 'FAIL'}")
                res[tag] = ok
    return all(res.values())


if __name__ == "__main__":
    fwd_ok = main() == 0
    bwd_ok = backward_gate()
    P(f"\n=> forward {'GREEN' if fwd_ok else 'RED'} | backward {'GREEN' if bwd_ok else 'RED'}")
    sys.exit(0 if (fwd_ok and bwd_ok) else 1)
