#!/usr/bin/env python3
"""Airtight correctness harness for the routed RoLA Triton kernels — the gate for K-tiling (#86).

Three independent checks, each rigorous on its own terms:
  (1) ORACLE TRUST   — gradcheck the torch reference (_rola_global_ref) in fp64. Proves the oracle's
                       gradients are correct, so it's a valid reference for the kernel's grads.
  (2) FORWARD        — kernel (fp32) normalized readout vs oracle (fp64), rel-error bound, across K.
  (3) BACKWARD       — kernel ANALYTIC grads (fp32 autograd) vs oracle ANALYTIC grads (fp64
                       autograd), per input, across K. NEVER gradcheck-in-fp64 through the fp32
                       kernel (that fails to compile by construction — the earlier harness bug).

K spans the K<=64 SRAM regime AND beyond (the conservative-bound region the tiling makes robust).
Run on CUDA, unbuffered (`python -u`). Deterministic seeds.

Convention bridge: oracle `_rola_global_ref(q,k,v,wg,rg)` takes [B,L,H,*], returns the NORMALIZED
readout; the kernel `rola_rla_triton(q,k,v,r,w)` takes folded [BH,L,*], returns the UN-normalized
numerator — so we fold + apply the ones-column denominator trick (exactly as rola.py does).
"""
import sys
import torch

from rola import _rola_global_ref, _rola_gla_ref, _rola_perstate_den, _rola_gla_perstate_den
from fla_rola.ops.simple_gla.rola import (rola_rla_triton, rola_gla_triton,
                                          rola_perstate_den_triton, rola_perstate_den_gla_triton)

DEV = "cuda"
EPS = 1e-5
P = lambda *a: print(*a, flush=True)          # unbuffered


def _fold(t):                                  # [B,L,H,D] -> [B*H, L, D]
    B, L, H, D = t.shape
    return t.permute(0, 2, 1, 3).reshape(B * H, L, D).contiguous()


def _unfold(t, B, H):                          # [B*H, L, D] -> [B, L, H, D]
    BH, L, D = t.shape
    return t.view(B, H, L, D).permute(0, 2, 1, 3).contiguous()


def rla_triton_normalized(q, k, v, rg, wg):
    """[B,L,H,*] -> normalized [B,L,H,dv] via the kernel + ones-column denominator (rola.py's trick)."""
    B, L, H, dv = v.shape
    v1 = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)
    oa = rola_rla_triton(_fold(q), _fold(k), _fold(v1), _fold(rg), _fold(wg))   # [BH,L,dv+1]
    oa = _unfold(oa, B, H)
    return oa[..., :dv] / (oa[..., dv:dv + 1] + EPS)


def _mk(B, L, H, K, nc, dv, dtype, seed=0, with_ld=False):
    g = torch.Generator(device=DEV).manual_seed(seed)
    rnd = lambda *s: torch.randn(*s, generator=g, device=DEV, dtype=dtype)
    q = torch.nn.functional.elu(rnd(B, L, H, K)) + 1.0
    k = torch.nn.functional.elu(rnd(B, L, H, K)) + 1.0
    v = rnd(B, L, H, dv)
    rg = torch.softmax(rnd(B, L, H, nc), dim=-1)
    wg = torch.softmax(rnd(B, L, H, nc), dim=-1)
    if with_ld:
        ld = torch.log(torch.sigmoid(rnd(B, L, H, nc)))      # per-state log-decay in (-inf, 0)
        return q, k, v, rg, wg, ld
    return q, k, v, rg, wg


def gla_triton_normalized(q, k, v, rg, wg, ld):
    """[B,L,H,*] -> normalized [B,L,H,dv] via the GLA kernel + ones-column denominator (mirrors RLA)."""
    B, L, H, dv = v.shape
    v1 = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)
    # rola_gla_triton(q,k,v,r,w,ld): r=read=rg, w=write=wg
    oa = rola_gla_triton(_fold(q), _fold(k), _fold(v1), _fold(rg), _fold(wg), _fold(ld))
    oa = _unfold(oa, B, H)
    return oa[..., :dv] / (oa[..., dv:dv + 1] + EPS)


def check_oracle_grads():
    """(1) The oracle's own gradients are correct (true fp64 gradcheck of the torch reference)."""
    P("=== (1) oracle trust: fp64 gradcheck of _rola_global_ref ===")
    q, k, v, rg, wg = _mk(1, 12, 1, 8, nc=3, dv=4, dtype=torch.float64)
    ins = [t.detach().requires_grad_(True) for t in (q, k, v, rg, wg)]
    ok = torch.autograd.gradcheck(
        lambda q, k, v, rg, wg: _rola_global_ref(q, k, v, wg, rg),
        tuple(ins), eps=1e-6, atol=1e-5, rtol=1e-4)
    P(f"  oracle gradcheck (fp64): {'PASS' if ok else 'FAIL'}")
    return ok


def check_forward(Ks):
    P("=== (2) forward: kernel(fp32) vs oracle(fp64), across K ===")
    res = {}
    for K in Ks:
        q, k, v, rg, wg = _mk(2, 64, 2, K, nc=4, dv=16, dtype=torch.float64)
        ref = _rola_global_ref(q, k, v, wg, rg)
        out = rla_triton_normalized(*[t.float() for t in (q, k, v, rg, wg)]).double()
        rel = (out - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
        res[K] = rel < 2e-2
        P(f"  K={K:4d}  {'PASS' if res[K] else 'FAIL'}  rel={rel:.2e}")
    return res


def check_backward(Ks):
    P("=== (3) backward: kernel analytic grads vs oracle analytic grads, across K ===")
    res = {}
    for K in Ks:
        try:
            q, k, v, rg, wg = _mk(2, 48, 2, K, nc=4, dv=16, dtype=torch.float32)
            ik = [t.clone().requires_grad_(True) for t in (q, k, v, rg, wg)]
            rla_triton_normalized(*ik).sum().backward()
            io = [t.double().detach().requires_grad_(True) for t in (q, k, v, rg, wg)]
            _rola_global_ref(io[0], io[1], io[2], io[4], io[3]).sum().backward()
            rels = [("qkvrw"[i], (ik[i].grad.double() - io[i].grad).abs().max().item()
                     / (io[i].grad.abs().max().item() + 1e-9)) for i in range(5)]
            worst = max(r for _, r in rels)
            res[K] = worst < 3e-2
            P(f"  K={K:4d}  {'PASS' if res[K] else 'FAIL'}  worst={worst:.2e}  "
              f"({', '.join(f'{n}:{r:.1e}' for n, r in rels)})")
        except Exception as e:
            res[K] = False
            P(f"  K={K:4d}  FAIL  {type(e).__name__}: {str(e)[:90]}")
    return res


def check_oracle_grads_gla():
    """(1-GLA) fp64 gradcheck of the GLA torch reference (with per-state decay ld)."""
    P("=== (1-GLA) oracle trust: fp64 gradcheck of _rola_gla_ref ===")
    q, k, v, rg, wg, ld = _mk(1, 12, 1, 8, nc=3, dv=4, dtype=torch.float64, with_ld=True)
    ins = [t.detach().requires_grad_(True) for t in (q, k, v, rg, wg, ld)]
    ok = torch.autograd.gradcheck(
        lambda q, k, v, rg, wg, ld: _rola_gla_ref(q, k, v, wg, rg, ld, normalized=True),
        tuple(ins), eps=1e-6, atol=1e-5, rtol=1e-4)
    P(f"  GLA oracle gradcheck (fp64): {'PASS' if ok else 'FAIL'}")
    return ok


def check_forward_gla(Ks):
    P("=== (2-GLA) forward: GLA kernel(fp32) vs oracle(fp64), across K ===")
    res = {}
    for K in Ks:
        try:
            q, k, v, rg, wg, ld = _mk(2, 64, 2, K, nc=4, dv=16, dtype=torch.float64, with_ld=True)
            ref = _rola_gla_ref(q, k, v, wg, rg, ld, normalized=True)
            out = gla_triton_normalized(*[t.float() for t in (q, k, v, rg, wg, ld)]).double()
            rel = (out - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
            res[K] = rel < 2e-2
            P(f"  K={K:4d}  {'PASS' if res[K] else 'FAIL'}  rel={rel:.2e}")
        except Exception as e:
            res[K] = False
            P(f"  K={K:4d}  FAIL  {type(e).__name__}: {str(e)[:90]}")
    return res


def check_backward_gla(Ks):
    P("=== (3-GLA) backward: GLA kernel analytic grads vs oracle analytic grads, across K ===")
    res = {}
    for K in Ks:
        try:
            q, k, v, rg, wg, ld = _mk(2, 48, 2, K, nc=4, dv=16, dtype=torch.float32, with_ld=True)
            ik = [t.clone().requires_grad_(True) for t in (q, k, v, rg, wg, ld)]
            gla_triton_normalized(*ik).sum().backward()
            io = [t.double().detach().requires_grad_(True) for t in (q, k, v, rg, wg, ld)]
            _rola_gla_ref(io[0], io[1], io[2], io[4], io[3], io[5], normalized=True).sum().backward()
            rels = [("qkvrwl"[i], (ik[i].grad.double() - io[i].grad).abs().max().item()
                     / (io[i].grad.abs().max().item() + 1e-9)) for i in range(6)]
            worst = max(r for _, r in rels)
            res[K] = worst < 3e-2
            P(f"  K={K:4d}  {'PASS' if res[K] else 'FAIL'}  worst={worst:.2e}  "
              f"({', '.join(f'{n}:{r:.1e}' for n, r in rels)})")
        except Exception as e:
            res[K] = False
            P(f"  K={K:4d}  FAIL  {type(e).__name__}: {str(e)[:90]}")
    return res


# ########## DEN — per-state denominator (kappa normalization) ##########
# The den entry points take folded [BH,L,*]; the oracles take [B,L,H,*]. WRITE gate only (no read
# gate, no v). fold/unfold are differentiable (permute+reshape) so grads flow to the [B,L,H,*] leaf.
# K spans 96 (non-pow2) to exercise the padded-tail dmask in the feature-tiled kernels.

def check_forward_den(Ks):
    P("=== (4) den forward: kernel(fp32) vs oracle(fp64), across K ===")
    res = {}
    for K in Ks:
        try:
            q, k, v, rg, wg = _mk(2, 64, 2, K, nc=4, dv=16, dtype=torch.float64)
            ref = _rola_perstate_den(q, k, wg)
            out = _unfold(rola_perstate_den_triton(_fold(q.float()), _fold(k.float()),
                                                   _fold(wg.float())), 2, 2).double()
            rel = (out - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
            res[K] = rel < 2e-2
            P(f"  K={K:4d}  {'PASS' if res[K] else 'FAIL'}  rel={rel:.2e}")
        except Exception as e:
            res[K] = False
            P(f"  K={K:4d}  FAIL  {type(e).__name__}: {str(e)[:90]}")
    return res


def check_backward_den(Ks):
    P("=== (5) den backward: kernel analytic grads vs oracle analytic grads, across K ===")
    res = {}
    for K in Ks:
        try:
            q, k, v, rg, wg = _mk(2, 48, 2, K, nc=4, dv=16, dtype=torch.float32)
            ik = [t.clone().requires_grad_(True) for t in (q, k, wg)]
            _unfold(rola_perstate_den_triton(_fold(ik[0]), _fold(ik[1]), _fold(ik[2])), 2, 2).sum().backward()
            io = [t.double().detach().requires_grad_(True) for t in (q, k, wg)]
            _rola_perstate_den(io[0], io[1], io[2]).sum().backward()
            rels = [("qkw"[i], (ik[i].grad.double() - io[i].grad).abs().max().item()
                     / (io[i].grad.abs().max().item() + 1e-9)) for i in range(3)]
            worst = max(r for _, r in rels)
            res[K] = worst < 3e-2
            P(f"  K={K:4d}  {'PASS' if res[K] else 'FAIL'}  worst={worst:.2e}  "
              f"({', '.join(f'{n}:{r:.1e}' for n, r in rels)})")
        except Exception as e:
            res[K] = False
            P(f"  K={K:4d}  FAIL  {type(e).__name__}: {str(e)[:90]}")
    return res


def check_forward_den_gla(Ks):
    P("=== (4-GLA) den forward: GLA kernel(fp32) vs oracle(fp64), across K ===")
    res = {}
    for K in Ks:
        try:
            q, k, v, rg, wg, ld = _mk(2, 64, 2, K, nc=4, dv=16, dtype=torch.float64, with_ld=True)
            ref = _rola_gla_perstate_den(q, k, wg, ld)
            out = _unfold(rola_perstate_den_gla_triton(_fold(q.float()), _fold(k.float()),
                                                       _fold(wg.float()), _fold(ld.float())), 2, 2).double()
            rel = (out - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
            res[K] = rel < 2e-2
            P(f"  K={K:4d}  {'PASS' if res[K] else 'FAIL'}  rel={rel:.2e}")
        except Exception as e:
            res[K] = False
            P(f"  K={K:4d}  FAIL  {type(e).__name__}: {str(e)[:90]}")
    return res


def check_backward_den_gla(Ks):
    P("=== (5-GLA) den backward: GLA kernel analytic grads vs oracle, across K (l: the dld column) ===")
    res = {}
    for K in Ks:
        try:
            q, k, v, rg, wg, ld = _mk(2, 48, 2, K, nc=4, dv=16, dtype=torch.float32, with_ld=True)
            ik = [t.clone().requires_grad_(True) for t in (q, k, wg, ld)]
            _unfold(rola_perstate_den_gla_triton(_fold(ik[0]), _fold(ik[1]), _fold(ik[2]),
                                                 _fold(ik[3])), 2, 2).sum().backward()
            io = [t.double().detach().requires_grad_(True) for t in (q, k, wg, ld)]
            _rola_gla_perstate_den(io[0], io[1], io[2], io[3]).sum().backward()
            rels = [("qkwl"[i], (ik[i].grad.double() - io[i].grad).abs().max().item()
                     / (io[i].grad.abs().max().item() + 1e-9)) for i in range(4)]
            worst = max(r for _, r in rels)
            res[K] = worst < 3e-2
            P(f"  K={K:4d}  {'PASS' if res[K] else 'FAIL'}  worst={worst:.2e}  "
              f"({', '.join(f'{n}:{r:.1e}' for n, r in rels)})")
        except Exception as e:
            res[K] = False
            P(f"  K={K:4d}  FAIL  {type(e).__name__}: {str(e)[:90]}")
    return res


def check_den_caller_e2e():
    """(6) E2E: the exact path the deleted CLA dqk<=64 den gates now enable. At dqk=128 fold a real
    (q,k,wg[,ld]) with the model's foldd pattern and confirm the Triton den entry points match the
    eager oracle (fwd + grads), RLA and GLA. Proves the gate-free model path is correct at dqk>64."""
    P("=== (6) den caller E2E at dqk=128 (the now-ungated model path) ===")
    K = 128
    ok = {}
    # RLA
    q, k, v, rg, wg = _mk(2, 48, 2, K, nc=4, dv=16, dtype=torch.float32)
    ik = [t.clone().requires_grad_(True) for t in (q, k, wg)]
    _unfold(rola_perstate_den_triton(_fold(ik[0]), _fold(ik[1]), _fold(ik[2])), 2, 2).sum().backward()
    io = [t.double().detach().requires_grad_(True) for t in (q, k, wg)]
    _rola_perstate_den(io[0], io[1], io[2]).sum().backward()
    fwd = (_unfold(rola_perstate_den_triton(_fold(q), _fold(k), _fold(wg)), 2, 2).double()
           - _rola_perstate_den(io[0], io[1], io[2])).abs().max().item() / (
        _rola_perstate_den(io[0], io[1], io[2]).abs().max().item() + 1e-9)
    gw = max((ik[i].grad.double() - io[i].grad).abs().max().item()
             / (io[i].grad.abs().max().item() + 1e-9) for i in range(3))
    ok['RLA'] = fwd < 2e-2 and gw < 3e-2
    P(f"  RLA  {'PASS' if ok['RLA'] else 'FAIL'}  fwd={fwd:.2e} grad={gw:.2e}")
    # GLA
    q, k, v, rg, wg, ld = _mk(2, 48, 2, K, nc=4, dv=16, dtype=torch.float32, with_ld=True)
    ik = [t.clone().requires_grad_(True) for t in (q, k, wg, ld)]
    _unfold(rola_perstate_den_gla_triton(_fold(ik[0]), _fold(ik[1]), _fold(ik[2]), _fold(ik[3])), 2, 2).sum().backward()
    io = [t.double().detach().requires_grad_(True) for t in (q, k, wg, ld)]
    _rola_gla_perstate_den(io[0], io[1], io[2], io[3]).sum().backward()
    refg = _rola_gla_perstate_den(io[0], io[1], io[2], io[3])
    fwd = (_unfold(rola_perstate_den_gla_triton(_fold(q), _fold(k), _fold(wg), _fold(ld)), 2, 2).double()
           - refg).abs().max().item() / (refg.abs().max().item() + 1e-9)
    gw = max((ik[i].grad.double() - io[i].grad).abs().max().item()
             / (io[i].grad.abs().max().item() + 1e-9) for i in range(4))
    ok['GLA'] = fwd < 2e-2 and gw < 3e-2
    P(f"  GLA  {'PASS' if ok['GLA'] else 'FAIL'}  fwd={fwd:.2e} grad={gw:.2e}")
    return ok


if __name__ == "__main__":
    if not torch.cuda.is_available():
        P("SKIP: needs CUDA"); sys.exit(0)
    P("device: " + torch.cuda.get_device_name(0))
    Ks = [16, 64, 128, 256, 512]                # <=64 regime + well beyond (based/rebased + headroom)
    Ks_den = [16, 64, 96, 128, 256, 512]        # +96 (non-pow2) exercises the padded-tail dmask
    P("########## RLA ##########")
    o = check_oracle_grads(); f = check_forward(Ks); b = check_backward(Ks)
    P("########## GLA (scalar per-state decay) ##########")
    og = check_oracle_grads_gla(); fg = check_forward_gla(Ks); bg = check_backward_gla(Ks)
    P("########## DEN ##########")
    fd = check_forward_den(Ks_den); bd = check_backward_den(Ks_den)
    fdg = check_forward_den_gla(Ks_den); bdg = check_backward_den_gla(Ks_den)
    e2e = check_den_caller_e2e()
    allok = (o and all(f.values()) and all(b.values())
             and og and all(fg.values()) and all(bg.values())
             and all(fd.values()) and all(bd.values())
             and all(fdg.values()) and all(bdg.values()) and all(e2e.values()))
    P(f"\nSUMMARY RLA: oracle={o} forward={f} backward={b}")
    P(f"SUMMARY GLA: oracle={og} forward={fg} backward={bg}")
    P(f"SUMMARY DEN: fwd={fd} bwd={bd}")
    P(f"SUMMARY DEN-GLA: fwd={fdg} bwd={bdg}")
    P(f"SUMMARY DEN-E2E: {e2e}")
    P(f"-> {'ALL GREEN' if allok else 'NOT GREEN'}")
    sys.exit(0 if allok else 1)
