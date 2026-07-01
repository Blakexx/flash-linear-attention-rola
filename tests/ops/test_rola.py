# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""Correctness suite for the routed RoLA operator — restructured into TWO principled axes + autograd (#46).

The precomputed-gate RoLA Triton kernels are RETIRED (#44 convergence): routed math lives in the
in-kernel `chunk_rola_routed` / `fused_recurrent_rola`, and the O(L²) `naive_rola_*` plus fp64
materialized-gate references below are the anchors. The old public precomputed-gate `chunk_rola` helper
has been moved to rola-scratch; this suite validates the shipping interfaces.

The suite has three legs:

  1. INTRA-CHUNK CONFIG-EQUIVALENCE (`TestIntraConfigEquivalence`) — the config axis. For a FIXED random
     input, run the kernel under EVERY output-equivalent config branch and assert IDENTICAL forward AND
     grads: the chunk size (NCH=1 single-chunk vs multi-chunk inter-scan), the autotune tile counts
     (ND/ND_V/BV/BD via dv/dqk + a FORCED single-config sweep), and BUILD_R True/False (write-only snapshot
     path — bit-identical write factor). Bit-exact where the
     branch is a pure reorder/flag; fp-tight where the reduction order genuinely differs across tiles.
     Catches tile/config/flag bugs SYSTEMATICALLY — the axis the old per-feature tests covered only
     incidentally (grep 'intra' on the old file → 0).

  2. INTER-CHUNK CORRECTNESS (`TestInterCorrectness`) — for a fixed input, chunked == recurrent == naive
     for BOTH forward AND backward. ONE comprehensive parametrized grid over the FULL semantic
     cross-product: norm{raw,global,per_state,kappa} × {RLA,GLA} × bias{on,off} × routing{flat,square,tree}
     × dims{pow2 + non-pow2 dv, dqk 16/64/128}. SUBSUMES the scattered per-feature tests
     (test_kappa_routed_*/test_gla_routed_*/test_routed_bias_*/test_routed_bwd_nonpow2_dv) and closes the
     holes BY CONSTRUCTION (the GLA×bias bwd the gates caught now lives at every grid node, not one cell).

  3. AUTOGRAD (`TestAutograd`) — the fp64 gradcheck (analytic bwd == torch.autograd of the fwd), DISTINCT
     from the inter recurrent==chunked==naive leg. Anchors the naive oracle whose grads suite 2 trusts.

Plus the orthogonal STRUCTURAL gates kept verbatim (`TestStructuralGates`): per-head routing/decay
independence, signed-den consistency, the below-floor ld guard, the *_no_LNC_materialization allocation
watches, and (tests/layers/test_rola_routing_init) the init-parity.

Run:  PYTHONPATH=. pytest tests/ops/test_rola.py -q   (CUDA required; CPU is skipped).
      Mind the GLA-routed cold autotune (~40min).
"""

import pytest
import torch
import triton

import fla_rola.ops.rola.chunk as C
from fla_rola.ops.rola import fused_recurrent_rola
from fla_rola.ops.rola.naive import (
    naive_rola_gla,
    naive_rola_global,
)
from fla_rola.ops.simple_gla import chunk_simple_gla
from fla_rola.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla
from fla_rola.utils import assert_close, device

EPS = 1e-5
KAPPA = 0.5


# -----------------------------------------------------------------------------
# helpers (the original max-rel metric — do NOT swap for an RMS-rel; the gates were calibrated to it)
# -----------------------------------------------------------------------------
def _relmax(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-9)


def _routed_kwargs(h, Wr, Ww, D, b, Wg=None, b_r=None, b_w=None):
    """Build the only non-weight routed side input: scalar GLA alpha."""
    alpha = None
    if Wg is not None:
        alpha = torch.sigmoid(torch.einsum('bthm,hm->bth', h.float(), Wg.float()))
    return dict(alpha=alpha)


_GLA_BWD_TOL = 1.2e-1   # GLA decay fp32 floor (NOT a kernel bug): the gate/decay grads (drg,dwg,dld) flow
#                       through the softmax-gate × exp(cumsum(ld)) product across the chunked recurrence,
#                       whose fp32-vs-fp64 floor is ~9e-2 worst-case (the dwg grad; seed-driven, present
#                       even at nc=4 — verified identical on the unmodified rola HEAD). The fwd already
#                       documents the GLA ~3e-2 decay floor; the backward inherits + amplifies it. q,k,v
#                       grads stay ~5e-3 (the tight stride/mask gate asserted separately). Tight-grad GLA
#                       validation = GLA/RLA parity sync.


# =============================================================================
# SHARED helpers: the vh (virtual-head simple_gla) bridge + the direct O(L²) naive oracle wrappers, used
# by the chunk==recurrent==naive backbone. (was tests/test_recurrent_vs_chunked.py — same B/H/dqk=dv,
# seeds, nc/dv grid, TOL.) The vh-chunk leg is RETAINED as a fourth independent witness on the forward.
# =============================================================================
_B, _H, _DQK = 4, 4, 16


def _mk_inter(L, nc, dv, gla, seed, dqk=_DQK):
    g = torch.Generator(device=device).manual_seed(seed)

    def t(*s):
        return torch.randn(*s, device=device, generator=g, dtype=torch.float32)
    q, k = t(_B, L, _H, dqk).abs(), t(_B, L, _H, dqk).abs()         # elu+1-like: positive features
    v = t(_B, L, _H, dv)
    r = torch.softmax(t(_B, L, _H, nc), -1)
    w = torch.softmax(t(_B, L, _H, nc), -1)
    ld = (-torch.rand(_B, L, _H, nc, device=device, generator=g) * 0.5).clamp(min=-2.5) if gla else None
    return q, k, v, r, w, ld


def _vh_expand(q, k, v, w, ld, nc, dv):
    """[B,L,H,*] -> virtual-head [B,L,H*nc,*]; v carries the write gate + a den ones-column.
    dqk is read from q's true row width (not the module global) so the inter sweep covers any dqk."""
    Bq, L, dqk = q.shape[0], q.shape[1], q.shape[-1]
    qv = q.unsqueeze(3).expand(Bq, L, _H, nc, dqk).reshape(Bq, L, _H * nc, dqk)
    kv = k.unsqueeze(3).expand(Bq, L, _H, nc, dqk).reshape(Bq, L, _H * nc, dqk)
    v1 = torch.cat([v, torch.ones_like(v[..., :1])], -1)
    vv = (v1.unsqueeze(3) * w.unsqueeze(-1)).reshape(Bq, L, _H * nc, dv + 1)
    gv = ld.reshape(Bq, L, _H * nc).float() if ld is not None else None
    return qv, kv, vv, gv


def _vh_combine(o_aug, r, nc, norm, dv):
    """o_aug:[B,L,H*nc,dv+1] per-state (num|den) -> combined [B,L,H,dv] under the given norm."""
    Bq, L = o_aug.shape[0], o_aug.shape[1]
    o = o_aug.view(Bq, L, _H, nc, dv + 1)
    num, den = o[..., :dv], o[..., dv]
    # RAW signed den (the canonical convention: routed chunk, decode, and the naive oracle all use
    # raw (d+ε)). Tests run positive features (q,k=.abs(), softmax w ⇒ d>0), so
    # raw == |d|; the rescale r̃=r·(d+ε)^{−κ} | r/(d+ε) is only well-defined for d>0 (production = elu+1).
    if norm == 'kappa':
        r = r * (den + EPS).pow(-KAPPA)
    elif norm == 'per_state':
        r = r / (den + EPS)
    return (num * r.unsqueeze(-1)).sum(3) / ((den * r).sum(3).unsqueeze(-1) + EPS)


def _chunked(q, k, v, r, w, ld, nc, norm, dv):
    qv, kv, vv, gv = _vh_expand(q, k, v, w, ld, nc, dv)
    o_aug, _ = chunk_simple_gla(qv, kv, vv, g=gv, scale=1.0)
    return _vh_combine(o_aug.float(), r, nc, norm, dv)


def _recurrent(q, k, v, r, w, ld, nc, norm, dv):
    """Step-by-step decode: one fused_recurrent step per token, carrying the state."""
    L = q.shape[1]
    state, outs = None, []
    for tstep in range(L):
        s = slice(tstep, tstep + 1)
        qv, kv, vv, gv = _vh_expand(q[:, s], k[:, s], v[:, s], w[:, s],
                                    ld[:, s] if ld is not None else None, nc, dv)
        o, state = fused_recurrent_simple_gla(qv, kv, vv, g=gv, scale=1.0,
                                              initial_state=state, output_final_state=True)
        outs.append(_vh_combine(o.float(), r[:, s], nc, norm, dv))
    return torch.cat(outs, 1)


def _naive(q, k, v, r, w, ld, gla):
    """DIRECT O(L²) global-norm oracle (no vh glue). r=read, w=write."""
    if gla:
        return naive_rola_gla(q, k, v, w, r, ld, normalized=True)
    return naive_rola_global(q, k, v, w, r)


def _grad_through(fwd, q, k, v, r, w, ld, gla, coef):
    """autograd grads of (fwd(...)*coef).sum() w.r.t. (q,k,v,r,w[,ld])."""
    ins = [x.clone().requires_grad_() for x in ([q, k, v, r, w] + ([ld] if gla else []))]
    ld_in = ins[5] if gla else None
    o = fwd(ins[0], ins[1], ins[2], ins[3], ins[4], ld_in)
    return torch.autograd.grad((o * coef).sum(), ins)


def _kap(q, norm):
    return (torch.full((q.shape[0], q.shape[1], _H, 1), KAPPA, device=q.device, dtype=q.dtype)
            if norm == 'kappa' else None)


def _mk_oracle(B, L, H, K, nc, dv, dtype, seed=0, with_ld=False):
    g = torch.Generator(device=device).manual_seed(seed)

    def rnd(*s):
        return torch.randn(*s, generator=g, device=device, dtype=dtype)
    q = torch.nn.functional.elu(rnd(B, L, H, K)) + 1.0
    k = torch.nn.functional.elu(rnd(B, L, H, K)) + 1.0
    v = rnd(B, L, H, dv)
    rg = torch.softmax(rnd(B, L, H, nc), dim=-1)
    wg = torch.softmax(rnd(B, L, H, nc), dim=-1)
    if with_ld:
        # per-state log-decay in (-inf, 0), floored to the kernel's fp32-safe domain (_GLA_FLOOR=-2.5,
        # #33) so kernel and oracle test the SAME in-domain decay (the kernel raises out-of-range now).
        ld = torch.log(torch.sigmoid(rnd(B, L, H, nc))).clamp(min=-2.5)
        return q, k, v, rg, wg, ld
    return q, k, v, rg, wg


# =============================================================================
# THE SINGLE CANONICAL PER-HEAD ROUTING GROUND TRUTH.
#
# RoLA routes PER HEAD: each head h owns its own [D,d_model,b] tree router (Wr,Ww ∈ [H,D,d_model,b]).
# There is ONE ground truth for "the gates a router produces" — `_per_head_gates` — and EVERYTHING is
# anchored to it: materialized-gate references build on these explicit gates, and the fused
# `chunk_rola_routed` must compute the SAME gates IN-KERNEL from the SAME router. A shared-vs-per-head
# divergence CANNOT pass this plus the per-head-independence test below — the class of bug a fresh
# fused-only reference (re-encoding the kernel's own assumption) used to hide.
# =============================================================================
def _per_head_gates(hf, Wr, Ww, D, b, H, b_r=None, b_w=None):
    """THE canonical per-head tree gates. hf:[BH,T,d_model] (BH=(B,H) fold), Wr,Ww:[H,D,d_model,b],
    optional per-head bias b_r/b_w:[H,D,b]. Each head folds with its OWN router:
        r[...,leaf] = Π_lvl softmax(h·Wr[head,lvl] + b_r[head,lvl])[..., digit_lvl(leaf)].
    Returns the explicit folded gates [BH,T,nc] for the materialized-gate references — out-of-place
    (graph-reuse safe). This is the ONLY routing reference; every routed test anchors to it."""
    BH, T, dm = hf.shape
    hr = hf.view(BH // H, H, T, dm)                       # [B,H,T,dm]

    def _logit(W, bias, i):
        z = torch.einsum('bhtd,hdc->bhtc', hr, W[:, i].to(hf.dtype))     # per-head h·W[head,i]
        if bias is not None:
            z = z + bias[:, i].to(hf.dtype)[None, :, None, :]
        return z.reshape(BH, T, -1)
    fr = torch.stack([torch.softmax(_logit(Wr, b_r, i), -1) for i in range(D)], 0)   # [D,BH,T,b]
    fw = torch.stack([torch.softmax(_logit(Ww, b_w, i), -1) for i in range(D)], 0)
    rc, wc = [], []
    for leaf in range(b ** D):
        digs = [(leaf // (b ** (D - 1 - i))) % b for i in range(D)]
        rr, ww = fr[0][..., digs[0]], fw[0][..., digs[0]]
        for i in range(1, D):
            rr, ww = rr * fr[i][..., digs[i]], ww * fw[i][..., digs[i]]
        rc.append(rr)
        wc.append(ww)
    return torch.stack(rc, -1), torch.stack(wc, -1)


def _tree_gates_oop(hf, Wr, Ww, D, b, H=1, b_r=None, b_w=None):
    """Per-head explicit tree gates (thin alias of the canonical `_per_head_gates`; default H=1 = the
    single-router fold for the BH-as-batch op-level tests)."""
    return _per_head_gates(hf, Wr, Ww, D, b, H, b_r=b_r, b_w=b_w)


def _ld_from_Wg(hf, wf, Wg, H):
    """THE per-state log-decay ld[BH,L,nc] from the per-head decay weight Wg:[H,d_model] + the write gate
    wf — `RoLA._log_decay` (layers/rola.py) EXACTLY. The op computes this IN-KERNEL (never materialized);
    this is the test's reference materialization. #45."""
    BH, T, dm = hf.shape
    hr = hf.view(BH // H, H, T, dm)
    alpha = torch.sigmoid(torch.einsum('bhtd,hd->bht', hr, Wg)).reshape(BH, T, 1)
    return (1.0 - wf * (1.0 - alpha)).clamp(min=1e-8).log().clamp(min=-2.5)


# -----------------------------------------------------------------------------
# fp64 explicit-gate routed references (the chunked semantic anchors the fused path reproduces). These
# carry the value + den states across chunks and route on the CANONICAL per-head gates. Used by the
# INTER cross-product (suite 2) backward leg as the differentiable autograd anchor.
# -----------------------------------------------------------------------------
def _kappa_ref_chunked(q, k, v, h, Wr, Ww, kappa, D, b, norm, scale, chunk, eps=EPS):
    """Differentiable chunked reference mirroring the fused global/kappa/per_state RLA math EXACTLY
    (carries the value + den states across chunks), on the canonical per-head gates. `global` skips the
    read-gate rescale (r̃=r); the den D_i=Σ_c r^c d^c is still reduced over c. Per-head."""
    B, T, H, Kd = q.shape
    nc = b ** D

    def fold(t):
        return t.permute(0, 2, 1, 3).reshape(B * H, T, t.shape[-1])

    def unfold(t):
        return t.view(B, H, T, -1).permute(0, 2, 1, 3).contiguous()
    qf, kf, vf, hf = fold(q) * scale, fold(k), fold(v), fold(h)
    rf, wf = _per_head_gates(hf, Wr, Ww, D, b, H)
    kapf = fold(kappa) if kappa is not None else None
    BH = B * H
    Sval = qf.new_zeros(BH, nc, Kd, vf.shape[-1])
    Sden = qf.new_zeros(BH, nc, Kd)
    outs = []
    for c0 in range(0, T, chunk):
        c1 = min(c0 + chunk, T)
        Cc = c1 - c0
        qc, kc, vc = qf[:, c0:c1], kf[:, c0:c1], vf[:, c0:c1]
        rc, wc = rf[:, c0:c1], wf[:, c0:c1]
        G = torch.einsum('bid,bjd->bij', qc, kc)
        caus = torch.tril(torch.ones(Cc, Cc, device=q.device, dtype=qf.dtype))
        d = torch.einsum('bij,bjc->bic', G * caus, wc) + torch.einsum('bik,bck->bic', qc, Sden)
        if norm == 'global':
            rt = rc
        elif norm == 'per_state':
            rt = rc / (d + eps)
        else:
            rt = rc * (d + eps).pow(-kapf[:, c0:c1])
        R = torch.einsum('bic,bjc->bij', rt, wc)
        num = (torch.einsum('bij,bjv->biv', G * R * caus, vc)
               + torch.einsum('bic,bicv->biv', rt, torch.einsum('bik,bckv->bicv', qc, Sval)))
        den = (rt * d).sum(-1, keepdim=True)
        outs.append(num / (den + eps))
        Sval = Sval + torch.einsum('bjc,bjk,bjv->bckv', wc, kc, vc)
        Sden = Sden + torch.einsum('bjc,bjk->bck', wc, kc)
    return unfold(torch.cat(outs, 1))


def _routed_raw_ref(q, k, v, h, Wr, Ww, D, b, scale, b_r=None, b_w=None):
    """fp64 explicit-gate reference for the routed un-normalized RLA numerator (the math norm='raw' and
    the norm='global' numerator reproduce). Optional per-head bias threads into the gates."""
    B, T, H, Kd = q.shape

    def fold(t):
        return t.permute(0, 2, 1, 3).reshape(B * H, T, t.shape[-1])

    def unfold(t):
        return t.view(B, H, T, -1).permute(0, 2, 1, 3).contiguous()
    qf, kf, vf, hf = fold(q) * scale, fold(k), fold(v), fold(h)
    rf, wf = _per_head_gates(hf, Wr, Ww, D, b, H, b_r=b_r, b_w=b_w)
    G = torch.einsum('bid,bjd->bij', qf, kf)
    caus = torch.tril(torch.ones(T, T, device=q.device, dtype=qf.dtype))
    R = torch.einsum('bic,bjc->bij', rf, wf)
    num = torch.einsum('bij,bjv->biv', G * R * caus, vf)
    return unfold(num)


def _bias_norm_ref(q, k, v, h, Wr, Ww, kappa, D, b, norm, scale, b_r, b_w):
    """fp64 explicit-gate reference for the biased routed RLA readout (softmax(h·W + b)), all norms."""
    B, T, H, _ = q.shape

    def fold(t):
        return t.permute(0, 2, 1, 3).reshape(B * H, T, t.shape[-1])

    def unfold(t):
        return t.view(B, H, T, -1).permute(0, 2, 1, 3).contiguous()
    qf, kf, vf, hf = fold(q) * scale, fold(k), fold(v), fold(h)
    rf, wf = _per_head_gates(hf, Wr, Ww, D, b, H, b_r=b_r, b_w=b_w)
    G = qf @ kf.transpose(-1, -2)
    caus = torch.tril(torch.ones(T, T, device=q.device, dtype=qf.dtype))
    d = (G * caus) @ wf
    if norm == 'per_state':
        rt = rf / (d + EPS)
    elif norm == 'kappa':
        rt = rf * (d + EPS).pow(-fold(kappa))
    else:  # raw / global keep r̃ = r
        rt = rf
    num = (G * (rt @ wf.transpose(-1, -2)) * caus) @ vf
    if norm == 'raw':
        return unfold(num)               # un-normalized numerator
    den = (rt * d).sum(-1, keepdim=True)
    return unfold(num / (den + EPS))


def _gla_routed_ref(q, k, v, h, Wr, Ww, Wg, D, b, scale, b_r=None, b_w=None):
    """fp64 explicit-gate GLA reference for the routed un-normalized numerator — `naive_rola_gla` on the
    per-head tree gates + the per-state log-decay built from Wg the LAYER's `_log_decay` way (#45).
    Differentiable w.r.t. Wg/h/Ww (and b_r/b_w when biased). The bias threads into BOTH the gates AND the
    decay (ld is built from the BIASED write gate — the #45 bias-threading bug surfaced exactly there)."""
    B, T, H, Kd = q.shape

    def fold(t):
        return t.permute(0, 2, 1, 3).reshape(B * H, T, t.shape[-1])
    qf, kf, vf, hf = fold(q) * scale, fold(k), fold(v), fold(h)
    rf, wf = _per_head_gates(hf, Wr, Ww, D, b, H, b_r=b_r, b_w=b_w)   # [B*H, T, nc], fp64, per-head (+bias)
    ld = _ld_from_Wg(hf, wf, Wg, H)                        # ld(Wg) on the BIASED write gate — LAYER's decay

    def unf(t):
        return t.view(B * H, T, 1, -1)
    o = naive_rola_gla(unf(qf), unf(kf), unf(vf), unf(wf), unf(rf), unf(ld),
                       normalized=False).view(B * H, T, -1)
    return o.view(B, H, T, -1).permute(0, 2, 1, 3).contiguous()


def _gla_bias_norm_ref(q, k, v, h, Wr, Ww, Wg, kappa, D, b, norm, scale, b_r, b_w, chunk=None):
    """fp64 NORMALIZED GLA(+bias) reference for the kappa-path norms (global|per_state|kappa): the
    per-state decayed den + the decayed numerator on the BIASED per-head gates, ld from the LAYER's
    _log_decay on those biased write gates. Differentiable w.r.t. all of (q,k,v,h,Wr,Ww,Wg,b_r,b_w[,kappa])."""
    B, T, H, _ = q.shape

    def fold(t):
        return t.permute(0, 2, 1, 3).reshape(B * H, T, t.shape[-1])

    def unfold(t):
        return t.view(B, H, T, -1).permute(0, 2, 1, 3).contiguous()
    qf, kf, vf, hf = fold(q) * scale, fold(k), fold(v), fold(h)
    rf, wf = _per_head_gates(hf, Wr, Ww, D, b, H, b_r=b_r, b_w=b_w)
    ld = _ld_from_Wg(hf, wf, Wg, H)                                  # ld(Wg) on the BIASED write gate
    chunk = min(64, max(16, T)) if chunk is None else chunk
    d = C._perstate_den_torch(qf, kf, wf, ld, chunk, EPS)           # decayed per-state den
    if norm == 'kappa':
        rt = rf * (d + EPS).pow(-fold(kappa))
    elif norm == 'per_state':
        rt = rf / (d + EPS)
    else:  # global
        rt = rf
    num = C._rola_chunk_core(qf, kf, vf, wf, rt, ld, chunk)
    den = (rt * d).sum(-1, keepdim=True)
    return unfold(num / (den + EPS))


def _perstate_den_signed(q, k, w):
    """RAW signed per-state den d[BH,T,nc] = Σ_{j≤i}(q_i·k_j) w_j^c, folded over (B,H). No abs, no decay."""
    B, T, H, Kd = q.shape
    nc = w.shape[-1]
    qf = q.permute(0, 2, 1, 3).reshape(B * H, T, Kd)
    kf = k.permute(0, 2, 1, 3).reshape(B * H, T, Kd)
    wf = w.permute(0, 2, 1, 3).reshape(B * H, T, nc)
    G = torch.einsum('bid,bjd->bij', qf, kf)
    caus = torch.tril(torch.ones(T, T, device=q.device, dtype=q.dtype))
    return torch.einsum('bij,bjc->bic', G * caus, wf)


# Routing-tree shapes: flat (D=1,b=nc), square (D=2), tree (b=2). Crossed over the cross-product.
_ROUTE_FLAT_SQ_TREE_8 = [(1, 8), (2, 3), (3, 2)]    # flat, square(nc=9), tree(nc=8)
_ROUTE_FLAT_SQ_TREE_16 = [(1, 16), (2, 4), (4, 2)]  # flat, square(nc=16), tree(nc=16)


# =============================================================================
# SUITE 1 — INTRA-CHUNK CONFIG-EQUIVALENCE (fwd + bwd). The config axis the old suite covered only
# incidentally (grep 'intra' on the old file → 0). For a FIXED random input, run the routed kernel under
# EVERY output-equivalent config BRANCH and assert IDENTICAL forward AND grads. Bit-exact where the branch
# is a pure reorder/flag (the warp/stage pipeline knobs, BUILD_R, bias=None); fp-tight where the reduction
# order genuinely differs across tile sizes (the chunk-size reassociation, the ND/ND_V value/feature
# tiling). Catches tile/config/flag bugs systematically — a config that silently mis-tiles fires here.
# =============================================================================
class TestIntraConfigEquivalence:

    def test_kappa_fit_chunk_uses_device_smem_budget(self, monkeypatch):
        """#55 resource-fit regression: kappa fwd/bwd must share the same BT, but the bwd-state kernel's
        real BT=64 live set exceeds the sm86 hard SMEM ceiling. Fit from the reported device budget rather
        than arch names: sm86-class cards derive BT=32, while A100-class budgets can keep BT=64."""
        monkeypatch.setattr(C, '_device_smem', lambda: 101376)  # sm86/sm89 hard limit
        assert C._kappa_fit_chunk(dqk=16, dv=16, chunk=64, BC=16) == 32
        assert C._kappa_fit_chunk(dqk=16, dv=16, chunk=32, BC=16) == 32
        assert C._fit_chunk(64, C._routed_inter_state_row_bytes(BK=16, BC=16)) == 32

        monkeypatch.setattr(C, '_device_smem', lambda: 166912)  # A100 hard limit
        assert C._kappa_fit_chunk(dqk=16, dv=16, chunk=64, BC=16) == 64
        assert C._fit_chunk(64, C._routed_inter_state_row_bytes(BK=16, BC=16)) == 64

    @pytest.mark.parametrize('gla', [False, True])
    def test_kappa_sparse_checkpoints_match_dense_checkpoint_bwd(self, gla):
        """#55 proof: the production sqrt-spaced checkpoint backward must match the old dense every-chunk
        snapshot schedule. This compares the actual fused backward kernels, not only the fp64 reference."""
        if device != 'cuda':
            pytest.skip('RoLA Triton kernels require CUDA')
        B, T, Kd, V, dm, H, D, b = 1, 320, 16, 16, 24, 1, 3, 2
        g = torch.Generator(device=device).manual_seed(55)

        def rand(*shape, positive=False):
            x = torch.randn(*shape, device=device, generator=g)
            return torch.nn.functional.elu(x) + 1.0 if positive else x

        q, k = rand(B, T, Kd, positive=True), rand(B, T, Kd, positive=True)
        v, h = rand(B, T, V), rand(B, T, dm)
        Wr, Ww = rand(H, D, dm, b) * 0.4, rand(H, D, dm, b) * 0.4
        kap = torch.full((B, T), 0.5, device=device)
        Wg = torch.zeros(H, dm, device=device) if gla else None
        alpha = torch.sigmoid(torch.einsum('btm,hm->bt', h.float(), Wg.float())) if gla else None
        sel = C._build_sel(D, b, b ** D, q.device)
        chunk = 64

        num_s, den_s, ckv_s, ckd_s, _ = C._kappa_routed_fwd(
            q, k, v, h, h, Wr, Ww, None, None, alpha, kap, D, b, sel, chunk,
            global_norm=False, per_state=False, eps=EPS, H=H, save_checkpoints=True)
        num_d, den_d, ckv_d, ckd_d, _ = C._kappa_routed_fwd(
            q, k, v, h, h, Wr, Ww, None, None, alpha, kap, D, b, sel, chunk,
            global_norm=False, per_state=False, eps=EPS, H=H, save_checkpoints=True,
            checkpoint_every_override=1)
        assert _relmax(num_s, num_d) < 1e-6
        assert _relmax(den_s, den_d) < 1e-6

        out_s = num_s / (den_s.unsqueeze(-1) + EPS)
        out_d = num_d / (den_d.unsqueeze(-1) + EPS)
        if out_s.shape[1] > 0:
            out_s[:, 0].copy_(v[:, 0] * (den_s[:, 0] / (den_s[:, 0] + EPS))[:, None])
            out_d[:, 0].copy_(v[:, 0] * (den_d[:, 0] / (den_d[:, 0] + EPS))[:, None])
        do = torch.randn(num_s.shape, device=device, generator=g)
        dnum_s = do / (den_s.unsqueeze(-1) + EPS)
        dden_s = -(do * out_s).sum(-1) / (den_s + EPS)
        dnum_d = do / (den_d.unsqueeze(-1) + EPS)
        dden_d = -(do * out_d).sum(-1) / (den_d + EPS)
        gs = C._kappa_routed_bwd(
            q, k, v, h, h, h, Wr, Ww, None, None, kap, alpha, ckv_s, ckd_s, out_s, dnum_s, dden_s, D, b, sel, chunk,
            global_norm=False, per_state=False, eps=EPS, H=H, Wg=Wg)
        gd = C._kappa_routed_bwd(
            q, k, v, h, h, h, Wr, Ww, None, None, kap, alpha, ckv_d, ckd_d, out_d, dnum_d, dden_d, D, b, sel, chunk,
            global_norm=False, per_state=False, eps=EPS, H=H, Wg=Wg,
            checkpoint_window_override=1)
        names = ['dq', 'dk', 'dv', 'dhr', 'dhw', 'dhg', 'dWr', 'dWw', 'dbr', 'dbw', 'dkap'] + (['dWg'] if gla else [])
        for name, sparse, dense in zip(names, gs, gd):
            assert _relmax(sparse, dense) < 2e-4, f'{name}: sparse vs dense checkpoint rel {_relmax(sparse, dense):.2e}'

    def _mk(self, D=1, b=8, dtype=torch.float32, dv=24, dqk=16, dm=40, B=2, H=2, T=64, seed=0, grad=False):
        g = torch.Generator(device=device).manual_seed(seed)

        def mk(*s, f=False):
            x = torch.randn(*s, device=device, dtype=dtype, generator=g)
            x = (torch.nn.functional.elu(x) + 1.0) if f else x
            return x.requires_grad_() if grad else x
        q, k = mk(B, T, H, dqk, f=True), mk(B, T, H, dqk, f=True)
        v, h = mk(B, T, H, dv), mk(B, T, H, dm)
        Wr = (torch.randn(H, D, dm, b, device=device, dtype=dtype, generator=g) * 0.4)
        Ww = (torch.randn(H, D, dm, b, device=device, dtype=dtype, generator=g) * 0.4)
        if grad:
            Wr, Ww = Wr.requires_grad_(), Ww.requires_grad_()
        return q, k, v, h, Wr, Ww

    @pytest.mark.parametrize('norm', ['raw', 'global', 'per_state', 'kappa'])
    @pytest.mark.parametrize('gla', [False, True])
    def test_chunk_size_equivalence_fwd_bwd(self, norm, gla):
        """The CHUNK SIZE is a config branch: re-chunking is a pure reduction-order reassociation, so the
        forward AND every grad are INVARIANT to it. Drive `chunk_rola_routed`'s internal _routed_fwd_tiled
        / kappa-fwd / snapshot scans at NCH=1 (single chunk, the whole sequence) vs multi-chunk (the
        inter-chunk decvec/Λ/state-carry path NCH=1 leaves dead) by monkeypatching the _CHUNK_FWD/_CHUNK
        chunk-size CEILINGS (the SMEM derive `_fit_chunk` then caps under them; at T=32 the dominant tile
        fits so the ceiling IS the chunk), and assert the readout + grads match fp-tight. A bug in the
        inter-chunk carry (the GLA decay between chunks, the state hand-off) shows up as NCH=1 ≠ NCH>1.
        T=32 so the single-chunk arm runs at chunk32→NCH=1 vs the multi-chunk arm chunk16→NCH=2 (forward
        AND backward both key off these ceilings — there is no separate backward chunk constant)."""
        if device != 'cuda':
            pytest.skip('RoLA Triton kernels require CUDA')
        D, b, T = 1, 8, 32
        go = torch.randn(2, T, 2, 24, device=device)
        kap = torch.full((2, T, 2, 1), KAPPA, device=device) if norm == 'kappa' else None
        Wg_seed = torch.randn(2, 40, device=device,
                              generator=torch.Generator(device=device).manual_seed(9)) * 0.4

        def run(chunk_fwd, chunk_bwd):
            saved = (C._CHUNK_FWD, C._CHUNK)
            C._CHUNK_FWD, C._CHUNK = chunk_fwd, chunk_bwd
            try:
                q, k, v, h, Wr, Ww = self._mk(T=T, seed=0, grad=True)
                Wg = Wg_seed.clone().requires_grad_() if gla else None
                sel = [q, k, v, h, Wr, Ww] + ([Wg] if gla else [])
                o = C.chunk_rola_routed(q, k, v, h, Wr, Ww, D, b, norm=norm,
                                        kappa=kap, scale=1.0, Wg=Wg,
                                        **_routed_kwargs(h, Wr, Ww, D, b, Wg=Wg))
                grads = torch.autograd.grad((o * go).sum(), sel)
                return o.detach(), grads
            finally:
                C._CHUNK_FWD, C._CHUNK = saved
        # single-chunk (chunk=32 → NCH=1: T<=chunk) vs multi-chunk (chunk=16 → NCH=2: the inter-chunk path).
        o1, g1 = run(32, 32)
        o2, g2 = run(16, 16)
        tol = 5e-3 if not gla else _GLA_BWD_TOL   # GLA inter-chunk decay sits at the fp32 floor
        assert _relmax(o1, o2) < (5e-3 if not gla else 3e-2), f'fwd chunk-size {_relmax(o1, o2):.2e}'
        names = ['q', 'k', 'v', 'h', 'Wr', 'Ww'] + (['Wg'] if gla else [])
        rels = {n: _relmax(a, b) for n, a, b in zip(names, g1, g2)}
        # q/k/v stay tight even in GLA; the gate/decay grads carry the GLA fp32 floor.
        for n in ('q', 'k', 'v'):
            assert rels[n] < (5e-3 if not gla else 1e-2), f'{n} grad chunk-size {rels[n]:.2e} ({rels})'
        assert all(r < tol for r in rels.values()), f'grad chunk-size {rels}'

    def test_autotune_config_equivalence_fwd(self):
        """The autotune configs (warp/stage software-pipeline knobs + the inter-scan's BV value-tile) are
        OUTPUT-INVARIANT — every surviving config must produce the SAME forward. FORCE each kernel over ITS
        OWN config list one config at a time (intra over `_AT_CFGS`, inter over `_SCAN_CFGS` — the latter
        carries the BV value-tile loop, a reduction-order reassociation) and assert the readout is fp-tight
        across them. norm='raw' is the ONLY norm whose CUDA path hits these two kernels (the normalized
        norms use the fused kappa kernels), so this is the targeted force-configs arm of the tile axis. A
        config that mis-stages SMEM, mis-loops the value-tile, or reorders a reduction wrongly diverges."""
        if device != 'cuda':
            pytest.skip('RoLA Triton kernels require CUDA')
        from triton.runtime.errors import OutOfResources
        D, b = 1, 8
        # non-pow2 dv so the inter-scan BV value-tile genuinely loops (ND_V>1 for the smaller BV configs).
        q, k, v, h, Wr, Ww = self._mk(dv=48, dqk=16, seed=1)
        kw = dict(norm='raw', scale=1.0)
        worst, n_ok = 0.0, 0
        # one kernel at a time, over its OWN config list, holding the other at its full (autotuned) set.
        for kern in (C._rola_routed_fwd_intra, C._rola_routed_fwd_inter):
            full = kern.configs
            kern.cache.clear()
            try:
                route = _routed_kwargs(h, Wr, Ww, D, b)
                ref = C.chunk_rola_routed(q, k, v, h, Wr, Ww, D, b, **kw, **route).float()   # autotuner's pick
                for cfg in full:
                    kern.configs = [cfg]
                    kern.cache.clear()
                    try:
                        out = C.chunk_rola_routed(q, k, v, h, Wr, Ww, D, b, **kw, **route).float()
                    except OutOfResources:
                        continue   # the autotuner scores OOR configs as inf and skips them on this card
                    worst = max(worst, _relmax(out, ref))
                    n_ok += 1
            finally:
                kern.configs = full
                kern.cache.clear()
        assert n_ok >= 2, f'no SMEM-fitting forced configs ran ({n_ok})'
        assert worst < 5e-3, f'forced-config forward divergence {worst:.2e}'

    @pytest.mark.parametrize('dv,dqk', [(16, 16), (24, 16), (48, 64), (32, 128)])
    def test_tile_count_equivalence_fwd(self, dv, dqk):
        """The host-computed tile counts (ND=cdiv(dqk,BK), ND_V=cdiv(dv,BV), BV/BD) change with dv/dqk —
        the value/feature LOOPS are reduction-order reassociations that must leave the output invariant. We
        can't change ND_V for a FIXED dv (it's derived), so the equivalence is: the routed kernel matches
        the materialized-gate reference at EVERY (dv,dqk) tile regime — pow2 (ND_V=1), non-pow2 dv
        (BV padding), and large dqk (ND>1 feature loop). A mis-tiled loop bound diverges somewhere in this
        grid."""
        if device != 'cuda':
            pytest.skip('RoLA Triton kernels require CUDA')
        D, b = 1, 8
        for norm in ('raw', 'global', 'per_state'):
            q, k, v, h, Wr, Ww = self._mk(dv=dv, dqk=dqk, seed=2)
            kw = dict(norm=norm, scale=1.0)
            of = C.chunk_rola_routed(q, k, v, h, Wr, Ww, D, b, **kw,
                                     **_routed_kwargs(h, Wr, Ww, D, b)).float()
            if norm == 'raw':
                oe = _routed_raw_ref(q.float(), k.float(), v.float(), h.float(),
                                     Wr.float(), Ww.float(), D, b, 1.0).float()
            else:
                oe = _kappa_ref_chunked(q.float(), k.float(), v.float(), h.float(),
                                        Wr.float(), Ww.float(), None, D, b, norm, 1.0,
                                        min(64, max(16, triton.next_power_of_2(q.shape[1])))).float()
            assert _relmax(of, oe) < 5e-3, f'norm={norm} dv={dv} dqk={dqk} tile vs ref {_relmax(of, oe):.2e}'

    @pytest.mark.parametrize('norm', ['raw', 'global', 'per_state', 'kappa'])
    @pytest.mark.parametrize('D,b', [(1, 16), (2, 4), (4, 2)])
    def test_bias_none_flag_bit_identical(self, norm, D, b):
        """The b_r/b_w=None flag is a pure WRITE-ONLY-style branch: passing b_r=b_w=None must reproduce the
        implicit no-bias path BIT-for-bit (the bias dummy is never read; HAS_BIAS=False gates every load).
        fp32 + bf16, all norms, flat/square/tree. (The analogue of BUILD_R-style flag equivalence: a flag
        that changes the math when it should not fires here.)"""
        if device != 'cuda':
            pytest.skip('RoLA Triton kernels require CUDA')
        for dt in (torch.float32, torch.bfloat16):
            q, k, v, h, Wr, Ww = self._mk(D=D, b=b, dtype=dt, dv=16, seed=0)
            kappa = (torch.rand(2, 64, 2, 1, device=device, dtype=dt) * 0.5 + 0.5) if norm == 'kappa' else None
            kw = dict(norm=norm, kappa=kappa, scale=1.0)
            route = _routed_kwargs(h, Wr, Ww, D, b)
            o_implicit = C.chunk_rola_routed(q, k, v, h, Wr, Ww, D, b, **kw, **route)
            o_explicit = C.chunk_rola_routed(q, k, v, h, Wr, Ww, D, b, b_r=None, b_w=None, **kw, **route)
            diff = (o_implicit.float() - o_explicit.float()).abs().max().item()
            assert diff == 0.0, f'{norm} {dt} bias=None not bit-identical: {diff}'

    def test_build_r_snapshot_state_equivalence(self):
        """BUILD_R True/False — the write-only branch (#37): the per-chunk PRE-STATE snapshots the backward
        consumes are built by the snapshot scan with BUILD_R=False (the read factor is never built — it
        would be discarded). The snapshot STATE depends ONLY on the WRITE gate, so it must equal the
        analytic write-only recurrent state regardless of whether the read factor is built. Bit-faithful
        equivalence of the BUILD_R=False path to the canonical write-state ground truth."""
        if device != 'cuda':
            pytest.skip('RoLA Triton kernels require CUDA')
        D, b, nc = 1, 8, 8
        B, H, T, dqk, dv, dm = 2, 2, 64, 16, 24, 40
        chunk = 16
        q, k, v, h, Wr, Ww = self._mk(dv=dv, dqk=dqk, dm=dm, B=B, H=H, T=T, seed=4)

        def foldf(t):
            return t.permute(0, 2, 1, 3).reshape(B * H, T, t.shape[-1]).float().contiguous()
        sel = C._build_sel(D, b, nc, device)
        # F2b: snapshots now consume the PRECOMPUTED write logits lw:[BH,T,D,b] (the cuBLAS GEMM), not Wr/Ww.
        _, lw = C._router_logits(foldf(h), Wr.float(), Ww.float(), None, None, H)
        snap = C._routed_snapshots(foldf(q), foldf(k), foldf(v), foldf(h), lw,
                                   D, b, sel, chunk, BG=16, H=H)          # [B*H, NCH, nc, dqk, dv]
        # analytic PRE-state at each chunk boundary, write-gate-only (BUILD_R is False ⇒ read never built).
        _, wf = _per_head_gates(foldf(h), Wr.float(), Ww.float(), D, b, H)    # [BH,T,nc]
        kf, vf = foldf(k), foldf(v)
        BH, NCH = B * H, triton.cdiv(T, chunk)
        S = torch.zeros(BH, nc, dqk, dv, device=device)
        worst = 0.0
        for c in range(NCH):
            worst = max(worst, _relmax(snap[:, c], S))                  # PRE-update snapshot
            c0, c1 = c * chunk, min((c + 1) * chunk, T)
            S = S + torch.einsum('btc,btk,btv->bckv', wf[:, c0:c1], kf[:, c0:c1], vf[:, c0:c1])
        assert worst < 5e-3, f'BUILD_R=False snapshot state vs analytic write-state {worst:.2e}'

# =============================================================================
# SUITE 2 — INTER-CHUNK CORRECTNESS (fwd + bwd). For a fixed input, chunked == recurrent == naive across
# the FULL semantic cross-product. ONE comprehensive grid subsuming the scattered per-feature tests.
#
# Two coupled grids share the cross-product:
#   (A) the backbone (chunk_simple_gla vh-chunk / fused_recurrent step-decode / direct O(L²) oracle)
#       over norm × {RLA,GLA} × dims — the chunk==recurrent==naive agreement (forward) plus
#       autograd(naive)==autograd(vh-chunk) for global backward.
#   (B) the ROUTED-KERNEL grid (chunk_rola_routed, in-kernel routing) over norm × {RLA,GLA} × bias{on,off}
#       × routing{flat,square,tree} × dims — the fused output AND all grads == autograd of the fp64
#       chunked/materialized-gate reference on the CANONICAL per-head gates. SUBSUMES test_kappa_routed_*/
#       test_gla_routed_*/
#       test_routed_bias_*/test_routed_bwd_nonpow2_dv and closes the GLA×bias×routing holes BY CONSTRUCTION.
# =============================================================================
# (A) EXPLICIT-GATE backbone dims (curated (dqk,nc,dv): pow2 + non-pow2 dv/nc + large dqk — a full
# Cartesian recompiles the vh chunk_simple_gla per shape, so awkward widths are one curated axis).
_INTER_DIMS = [
    (16, 16, 16),    # pow2 baseline (the original cell)
    (16, 96, 24),    # non-pow2 nc=96 + non-pow2 dv=24 (the LM dim) at small dqk
    (16, 64, 48),    # non-pow2 dv=48 (ND_V>=2 value-tiling) at small dqk
    (64, 16, 32),    # large dqk=64 + pow2 dv
    (64, 96, 24),    # large dqk=64 + non-pow2 nc + non-pow2 dv together
    (128, 64, 24),   # large dqk=128 + non-pow2 dv=24
]


class TestInterCorrectness:
    @pytest.mark.parametrize('gla', [False, True])
    def test_kappa_production_autograd_uses_checkpoint_path(self, gla):
        """#55 inter-kernel/reference smoke: `chunk_rola_routed` uses the shipped autograd Function
        (sparse checkpoints, ctx chunk refit, optional Wg, torch divide) and matches the fp64 reference to
        the precision floor of the path."""
        if device != 'cuda':
            pytest.skip('RoLA Triton kernels require CUDA')
        B, T, H, Kd, V, dm, D, b = 1, 320, 1, 16, 16, 24, 3, 2
        scale = Kd ** -0.5
        g = torch.Generator(device=device).manual_seed(56)

        def mk(*shape, positive=False):
            x = torch.randn(*shape, device=device, dtype=torch.float64, generator=g)
            x = torch.nn.functional.elu(x) + 1.0 if positive else x
            return x.requires_grad_()

        q, k = mk(B, T, H, Kd, positive=True), mk(B, T, H, Kd, positive=True)
        v, h = mk(B, T, H, V), mk(B, T, H, dm)
        Wr = (torch.randn(H, D, dm, b, device=device, dtype=torch.float64, generator=g) * 0.4).requires_grad_()
        Ww = (torch.randn(H, D, dm, b, device=device, dtype=torch.float64, generator=g) * 0.4).requires_grad_()
        Wg = torch.zeros(H, dm, device=device, dtype=torch.float64, requires_grad=True) if gla else None
        kap = (torch.rand(B, T, H, 1, device=device, dtype=torch.float64, generator=g) * 0.4 + 0.4).requires_grad_()
        go = torch.randn(B, T, H, V, device=device, dtype=torch.float64, generator=g)
        ref_args = [q, k, v, h, Wr, Ww] + ([Wg] if gla else []) + [kap]

        def f32_leaf(t):
            return t.detach().float().requires_grad_()

        qt, kt, vt, ht = (f32_leaf(t) for t in (q, k, v, h))
        Wrt, Wwt, kapt = (f32_leaf(t) for t in (Wr, Ww, kap))
        Wgt = f32_leaf(Wg) if gla else None
        prod_args = [qt, kt, vt, ht, Wrt, Wwt] + ([Wgt] if gla else []) + [kapt]

        out = C.chunk_rola_routed(
            qt, kt, vt, ht, Wrt, Wwt, D, b, norm='kappa', kappa=kapt, scale=scale, Wg=Wgt,
            **_routed_kwargs(ht, Wrt, Wwt, D, b, Wg=Wgt))
        grads = torch.autograd.grad(out, prod_args, go.to(out.dtype))
        prod_chunk = C._kappa_fit_chunk(Kd, V, min(64, max(16, triton.next_power_of_2(T))), BC=16)
        if gla:
            ref = _gla_bias_norm_ref(q, k, v, h, Wr, Ww, Wg, kap, D, b, 'kappa', scale, None, None,
                                     chunk=prod_chunk)
        else:
            ref = _kappa_ref_chunked(q, k, v, h, Wr, Ww, kap, D, b, 'kappa', scale, chunk=prod_chunk)
        ref_grads = torch.autograd.grad(ref, ref_args, go)

        assert _relmax(out.float(), ref.float()) < (3e-2 if gla else 1e-2)
        names = ['q', 'k', 'v', 'h', 'Wr', 'Ww'] + (['Wg'] if gla else []) + ['kappa']
        for name, got, exp in zip(names, grads, ref_grads):
            tol = _GLA_BWD_TOL if gla else (2.5e-2 if name == 'kappa' else 1e-2)
            assert _relmax(got.float(), exp.float()) < tol, f'{name}: {_relmax(got.float(), exp.float()):.2e}'

    @pytest.mark.parametrize('gla', [False, True])
    def test_kappa_checkpoint_wrapper_bf16_upcast_smoke(self, gla):
        """#55 dtype smoke: bf16 q/k/v/kap enter the production checkpointed wrapper and are upcast
        internally for the saved state path. This is a branch/finite-grad guard, not an fp64 equivalence gate."""
        if device != 'cuda':
            pytest.skip('RoLA Triton kernels require CUDA')
        B, T, H, Kd, V, dm, D, b = 1, 320, 1, 16, 16, 24, 3, 2
        g = torch.Generator(device=device).manual_seed(57)

        def bf16(*shape, positive=False):
            x = torch.randn(*shape, device=device, generator=g)
            x = torch.nn.functional.elu(x) + 1.0 if positive else x
            return x.to(torch.bfloat16).requires_grad_()

        q, k = bf16(B, T, H, Kd, positive=True), bf16(B, T, H, Kd, positive=True)
        v, h = bf16(B, T, H, V), bf16(B, T, H, dm)
        Wr = (torch.randn(H, D, dm, b, device=device, generator=g) * 0.4).to(torch.bfloat16).requires_grad_()
        Ww = (torch.randn(H, D, dm, b, device=device, generator=g) * 0.4).to(torch.bfloat16).requires_grad_()
        Wg = torch.zeros(H, dm, device=device, dtype=torch.bfloat16, requires_grad=True) if gla else None
        kap = (torch.rand(B, T, H, 1, device=device, generator=g) * 0.4 + 0.4).to(torch.bfloat16).requires_grad_()
        out = C.chunk_rola_routed(q, k, v, h, Wr, Ww, D, b, norm='kappa', kappa=kap, scale=Kd ** -0.5,
                                  Wg=Wg, **_routed_kwargs(h, Wr, Ww, D, b, Wg=Wg))
        args = [q, k, v, h, Wr, Ww] + ([Wg] if gla else []) + [kap]
        grads = torch.autograd.grad(out, args, torch.randn(out.shape, device=device, dtype=out.dtype, generator=g))
        assert torch.isfinite(out.float()).all()
        for grad in grads:
            assert grad is not None
            assert torch.isfinite(grad.float()).all()

    # ---- (A) the chunk==recurrent==naive backbone ---------------------------------------------------
    @pytest.mark.parametrize('dqk,nc,dv', _INTER_DIMS)
    @pytest.mark.parametrize('gla', [False, True])
    @pytest.mark.parametrize('norm', ['global', 'kappa', 'per_state'])
    def test_backbone_fwd(self, norm, gla, dqk, nc, dv):
        """chunk (vh chunk_simple_gla) == recurrent (step fused_recurrent) == naive direct O(L²) oracle
        (global), FORWARD, L=64, over 6 seeds. TOL 5e-3
        (clean ~1.5e-3; ~3x headroom). Swept over non-pow2 dv/nc + large dqk so any padding/stride/masking
        bug in the vh-chunk fwd or the recurrent step fires somewhere in the grid."""
        if device != 'cuda':
            pytest.skip('RoLA Triton kernels require CUDA')
        L, tol = 64, 5e-3
        w_rc = w_cn = w_rn = 0.0
        for seed in range(6):
            q, k, v, r, w, ld = _mk_inter(L, nc, dv, gla, seed, dqk=dqk)
            o_chunk = _chunked(q, k, v, r, w, ld, nc, norm, dv)
            o_rec = _recurrent(q, k, v, r, w, ld, nc, norm, dv)
            w_rc = max(w_rc, _relmax(o_rec, o_chunk))            # recurrent == chunked
            if norm == 'global':
                o_naive = _naive(q, k, v, r, w, ld, gla)         # the direct-oracle anchor
                w_cn = max(w_cn, _relmax(o_chunk, o_naive))
                w_rn = max(w_rn, _relmax(o_rec, o_naive))
        assert w_rc < tol, f'recurrent==chunk max-rel {w_rc:.2e}'
        if norm == 'global':
            assert max(w_cn, w_rn) < tol, f'chunk==naive {w_cn:.2e} rec==naive {w_rn:.2e}'

    @pytest.mark.parametrize('dv', [16, 32, 64])
    @pytest.mark.parametrize('nc', [16, 64])
    @pytest.mark.parametrize('gla', [False, True])
    def test_backbone_bwd(self, gla, nc, dv):
        """INTER backward: autograd(naive oracle) == autograd(vh-chunk), norm=global,
        BT=16 fp32 (value-tiled fp32 backward fits a small card -> rigorous ~1e-3). TOL 1.2e-2 (the
        chunked + value-tiled fp32 reduction-order gap ranges to ~8e-3 and varies with the config picked,
        so 1.2e-2 clears the noise floor while still catching the ~2% MUT class)."""
        if device != 'cuda':
            pytest.skip('RoLA Triton kernels require CUDA')
        saved = (C._CHUNK, C._CHUNK_FWD)
        C._CHUNK = 16
        C._CHUNK_FWD = 16
        try:
            tol = 1.2e-2
            w_cn = 0.0
            for seed in range(3):
                q, k, v, r, w, ld = _mk_inter(64, nc, dv, gla, seed)
                coef = torch.randn(*v.shape, device=device)
                g_naive = _grad_through(lambda a, b, c, d, e, f: _naive(a, b, c, d, e, f, gla),
                                        q, k, v, r, w, ld, gla, coef)
                g_chunk = _grad_through(lambda a, b, c, d, e, f: _chunked(a, b, c, d, e, f, nc, 'global', dv),
                                        q, k, v, r, w, ld, gla, coef)
                w_cn = max(w_cn, max(_relmax(a, b) for a, b in zip(g_chunk, g_naive)))
            assert w_cn < tol, f'vhchunk==naive {w_cn:.2e}'
        finally:
            C._CHUNK, C._CHUNK_FWD = saved

    @pytest.mark.parametrize('norm', ['global', 'kappa', 'per_state'])
    @pytest.mark.parametrize('gla', [False, True])
    def test_recurrent_state_carry(self, gla, norm):
        """fused_recurrent_rola: split a sequence at t, carry the state, == process the whole (decode)."""
        if device != 'cuda':
            pytest.skip('RoLA Triton kernels require CUDA')
        nc, dv, L, t, tol = 8, 16, 64, 40, 5e-3

        def sl(x, a, b):
            return x[:, a:b] if x is not None else None
        for seed in range(4):
            q, k, v, r, w, ld = _mk_inter(L, nc, dv, gla, seed)
            kap = _kap(q, norm)
            common = dict(norm=norm, scale=1.0)
            o_full, s_full = fused_recurrent_rola(q, k, v, r=r, w=w, g=ld, kappa=kap, **common,
                                                  output_final_state=True)
            o1, s1 = fused_recurrent_rola(sl(q, 0, t), sl(k, 0, t), sl(v, 0, t), r=sl(r, 0, t),
                                          w=sl(w, 0, t), g=sl(ld, 0, t), kappa=sl(kap, 0, t), **common,
                                          output_final_state=True)
            o2, s2 = fused_recurrent_rola(sl(q, t, L), sl(k, t, L), sl(v, t, L), r=sl(r, t, L),
                                          w=sl(w, t, L), g=sl(ld, t, L), kappa=sl(kap, t, L), **common,
                                          initial_state=s1, output_final_state=True)
            assert_close('split==whole readout', o_full, torch.cat([o1, o2], 1), tol)
            assert_close('split==whole state', s_full, s2, tol)

    # ---- (B) the in-kernel ROUTED grid: fused == autograd(fp64 materialized-gate refs) ----------------
    # The full semantic cross-product. Each node feeds (h, Wr, Ww[, Wg, b_r, b_w]) to chunk_rola_routed and
    # the fp64 chunked/naive reference on the SAME canonical per-head gates, asserting the fused output AND
    # every grad agree. norm × {RLA,GLA} × bias{on,off} × routing{flat,square,tree} × (dqk,dv).
    @pytest.mark.parametrize('norm', ['raw', 'global', 'per_state', 'kappa'])
    @pytest.mark.parametrize('gla', [False, True])
    @pytest.mark.parametrize('bias', [False, True])
    @pytest.mark.parametrize('D,b', _ROUTE_FLAT_SQ_TREE_8)   # flat(8), square(9), tree(8)
    @pytest.mark.parametrize('Kd,dv', [(16, 16), (16, 24), (64, 48), (128, 32)])
    def test_routed_fwd_bwd(self, norm, gla, bias, D, b, Kd, dv):
        """THE cross-product node. fused chunk_rola_routed(norm, [Wg], [b_r,b_w]) — forward AND all grads
        (dq,dk,dv,dh,dWr,dWw[,dWg][,db_r,db_w][,dkappa]) == autograd of the fp64 explicit-gate reference on
        the CANONICAL per-head gates, flat/square/tree × {RLA,GLA} × bias{on,off} × awkward (dqk,dv)
        (non-pow2 dv 24/48 = the alloc-stride bug class; large dqk 64/128 = the tiled path). fp64 → q/k/v
        TIGHT; the gate/decay/bias grads to the documented GLA fp32 floor under decay. This SUBSUMES the
        old test_kappa_routed_autograd_fp64 / test_routed_bias_fwd_bwd / test_routed_bwd_nonpow2_dv /
        test_gla_routed_bwd_faithful / test_gla_routed_bias_bwd_faithful — the GLA×bias×routing holes the
        gates caught are now grid nodes, not one cell.

        The nc=16 / DEPTH-4 routing family (flat-16, square-16, tree-16 at D=4) is covered for the BACKWARD
        grads by the sibling `test_routed_bwd_nc16_depth4` (the old test_routed_bias_fwd_bwd gated db_r/db_w/
        dWr/dWw through depth-4 — restored there so the depth-4/nc=16 grads stay gated)."""
        if device != 'cuda':
            pytest.skip('RoLA Triton kernels require CUDA')
        self._routed_fwd_bwd_check(norm, gla, bias, D, b, Kd, dv)

    @pytest.mark.parametrize('norm', ['raw', 'global', 'per_state', 'kappa'])
    @pytest.mark.parametrize('bias', [False, True])
    @pytest.mark.parametrize('D,b', _ROUTE_FLAT_SQ_TREE_16)   # flat(16), square(16), tree-16 (DEPTH 4)
    def test_routed_bwd_nc16_depth4(self, norm, bias, D, b):
        """Restores the nc=16 / DEPTH-4 BACKWARD grad coverage the old test_routed_bias_fwd_bwd gated (the
        cross-product otherwise tops out at the nc=8 family / depth-3). Runs the SAME fused-vs-fp64-reference
        check (all grads incl. db_r/db_w/dWr/dWw through the depth-4 tree's leaf-product fold) over flat-16,
        square-16, and the depth-4 tree-16 × bias{off,on} × all norms. RLA (the bias/router-grad family the
        old depth-4 cell covered); the nc=8 grid already crosses GLA × bias × tree. Runs in the deferred
        warmed sweep — closes the depth-4/nc=16 grad hole flagged by the gate."""
        if device != 'cuda':
            pytest.skip('RoLA Triton kernels require CUDA')
        self._routed_fwd_bwd_check(norm, gla=False, bias=bias, D=D, b=b, Kd=16, dv=16)

    @pytest.mark.parametrize('norm', ['kappa', 'global', 'raw'])
    @pytest.mark.parametrize('gla', [False, True])
    def test_routed_bwd_nc256_concurrent_cb(self, gla, norm):
        """#58 regression — LOCKS the parallel nc-state-block axis in the routed BACKWARD. The standard
        routed-bwd grid (test_routed_fwd_bwd / nc16_depth4) tops out at nc≤16 → NCBLK=cdiv(nc,BC)=1, so it
        runs exactly ONE nc-state-block per (batch-head) program and NEVER exercises CONCURRENT cb programs.
        This node runs nc=256 (tree D=8,b=2 → NCBLK=16), so the backward kernels launch grid (B·H, NCBLK)
        with 16 cb-axis programs PER batch-head writing dk/dv/dq (token-indexed `tl.atomic_add`) and the
        gd*/gda/dWg/dh partials CONCURRENTLY into shared locations. norm='raw' drives the RLA-raw inter
        kernels (`_bwd_inter_state`/`_bwd_inter_read`); 'kappa'/'global' drive the kappa kernels
        (`_kappa_bwd_state`/`_kappa_bwd_read`); both paths hit `_fold_kernel` — all five #58-parallelized
        kernels. Validated against the SAME fp64 canonical per-head-gate reference as test_routed_fwd_bwd
        (dq/dk/dv to the tight 8e-3 floor, the gate/kappa grads to their bounds). FAILS
        BY CONSTRUCTION if a cross-cb atomic is reverted to a plain store: the 16 concurrent cb partials
        race / last-writer-win on the shared output, dropping ~15/16 of that gradient. Verified — reverting
        the dkappa atomic_add (a SOLE-source cross-cb reduction, unmaskable) drives the kappa grad to ~9.7e-1
        vs ~1e-3 for every untouched grad, far past the 2.5e-2 bound; this node trips. RLA + GLA. Small
        B/H/T — the point is NCBLK=16, not size."""
        if device != 'cuda':
            pytest.skip('RoLA Triton kernels require CUDA')
        self._routed_fwd_bwd_check(norm, gla=gla, bias=False, D=8, b=2, Kd=16, dv=16)

    def _routed_fwd_bwd_check(self, norm, gla, bias, D, b, Kd, dv):
        """The shared fused-vs-fp64-reference forward+backward check body (see test_routed_fwd_bwd)."""
        B, H, T, dm = 2, 2, 64, 40
        scale = Kd ** -0.5
        g = torch.Generator(device=device).manual_seed(0)

        def mk(*s, f=False):
            x = torch.randn(*s, device=device, dtype=torch.float64, generator=g)
            return ((torch.nn.functional.elu(x) + 1.0) if f else x).requires_grad_()
        q, k = mk(B, T, H, Kd, f=True), mk(B, T, H, Kd, f=True)
        v, h = mk(B, T, H, dv), mk(B, T, H, dm)
        Wr = (torch.randn(H, D, dm, b, device=device, dtype=torch.float64, generator=g) * 0.4).requires_grad_()
        Ww = (torch.randn(H, D, dm, b, device=device, dtype=torch.float64, generator=g) * 0.4).requires_grad_()
        Wg = ((torch.randn(H, dm, device=device, dtype=torch.float64, generator=g) * 0.4).requires_grad_()
              if gla else None)
        b_r = ((torch.randn(H, D, b, device=device, dtype=torch.float64, generator=g) * 0.7).requires_grad_()
               if bias else None)
        b_w = ((torch.randn(H, D, b, device=device, dtype=torch.float64, generator=g) * 0.7).requires_grad_()
               if bias else None)
        kappa = ((torch.rand(B, T, H, 1, device=device, dtype=torch.float64, generator=g) * 0.5 + 0.5)
                 .requires_grad_() if norm == 'kappa' else None)
        go = torch.randn(B, T, H, dv, device=device, dtype=torch.float64, generator=g)
        sel = ([q, k, v, h, Wr, Ww] + ([Wg] if gla else []) + ([b_r, b_w] if bias else [])
               + ([kappa] if norm == 'kappa' else []))
        names = (['q', 'k', 'v', 'h', 'Wr', 'Ww'] + (['Wg'] if gla else []) + (['b_r', 'b_w'] if bias else [])
                 + (['kappa'] if norm == 'kappa' else []))
        of = C.chunk_rola_routed(
            q.float(), k.float(), v.float(), h.float(), Wr.float(), Ww.float(), D, b, norm=norm,
            kappa=(kappa.float() if norm == 'kappa' else None), scale=scale,
            Wg=(Wg.float() if gla else None),
            b_r=(b_r.float() if bias else None), b_w=(b_w.float() if bias else None),
            **_routed_kwargs(h.float(), Wr.float(), Ww.float(), D, b,
                             Wg=(Wg.float() if gla else None),
                             b_r=(b_r.float() if bias else None),
                             b_w=(b_w.float() if bias else None)))
        gf = torch.autograd.grad(of, sel, go.float())
        # the fp64 explicit-gate reference on the canonical per-head gates (the right ref per norm/gla/bias).
        if gla:
            if norm == 'raw':
                oe = _gla_routed_ref(q, k, v, h, Wr, Ww, Wg, D, b, scale, b_r=b_r, b_w=b_w)
            else:
                oe = _gla_bias_norm_ref(q, k, v, h, Wr, Ww, Wg, kappa, D, b, norm, scale, b_r, b_w)
        else:
            if norm == 'raw':
                oe = _routed_raw_ref(q, k, v, h, Wr, Ww, D, b, scale, b_r=b_r, b_w=b_w)
            elif bias:
                oe = _bias_norm_ref(q, k, v, h, Wr, Ww, kappa, D, b, norm, scale, b_r, b_w)
            else:
                chunk = min(64, max(16, triton.next_power_of_2(T)))
                oe = _kappa_ref_chunked(q, k, v, h, Wr, Ww, kappa, D, b, norm, scale, chunk)
        ge = torch.autograd.grad(oe, sel, go)
        rels = {n: _relmax(a.float(), c.float()) for n, a, c in zip(names, gf, ge)}
        out_tol = 3e-2 if gla else 1e-2   # GLA fwd carries the decay fp32 floor; RLA is rigorous fp64
        assert _relmax(of.float(), oe.float()) < out_tol, f'out {_relmax(of.float(), oe.float()):.2e}'
        # q/k/v TIGHT (the dv=24 / bias-threading bugs spiked these to ~1e0/~2-3e-1); the gate/bias grads to
        # the rigorous fp64 ~1e-2 (RLA), or — under GLA decay — the documented GLA fp32 floor. The dkappa
        # grad gets its OWN bound: it flows through the read-rescale exponent ∂/∂κ (d+ε)^{−κ} = −ln(d+ε)·
        # (d+ε)^{−κ}, the most ill-conditioned grad in the readout — surveyed worst-case 1.73e-2 over the
        # full tree×large-dqk×bias kappa grid (all OTHER grads stay ≤4e-3 there; verified fp64, no kernel
        # bug). 2.5e-2 clears it with headroom while still tripping a real ~3% dκ error.
        tight = 8e-3
        gate_tol = _GLA_BWD_TOL if gla else 1e-2
        kappa_tol = _GLA_BWD_TOL if gla else 2.5e-2
        for n in ('q', 'k', 'v'):
            assert rels[n] < tight, f'{n} grad {rels[n]:.2e} (norm={norm} gla={gla} bias={bias} rels={rels})'
        for n, r in rels.items():
            tol_n = kappa_tol if n == 'kappa' else gate_tol
            assert r < tol_n, f'{n} grad {r:.2e} (norm={norm} gla={gla} bias={bias} rels={rels})'
    @pytest.mark.parametrize('norm', ['global', 'kappa', 'per_state'])
    @pytest.mark.parametrize('D,b', _ROUTE_FLAT_SQ_TREE_8)
    def test_routed_faithful_bf16(self, D, b, norm):
        """Faithfulness gate (the no-model-change proof): the fused bf16 output matches the chunked
        reference (the explicit-gate math the routed kernel reproduces) to the bf16 noise floor,
        flat/square/tree. RLA. (was test_kappa_routed_faithful_bf16.)"""
        if device != 'cuda':
            pytest.skip('RoLA Triton kernels require CUDA')
        B, H, T, Kd, V, dm = 2, 2, 128, 16, 24, 32
        scale = Kd ** -0.5
        g = torch.Generator(device=device).manual_seed(1)

        def fm(x):
            return torch.nn.functional.elu(x) + 1.0
        q = fm(torch.randn(B, T, H, Kd, device=device, generator=g)).to(torch.bfloat16)
        k = fm(torch.randn(B, T, H, Kd, device=device, generator=g)).to(torch.bfloat16)
        v = torch.randn(B, T, H, V, device=device, generator=g).to(torch.bfloat16)
        h = torch.randn(B, T, H, dm, device=device, generator=g).to(torch.bfloat16)
        Wr = (torch.randn(H, D, dm, b, device=device, generator=g) * 0.5).to(torch.bfloat16)
        Ww = (torch.randn(H, D, dm, b, device=device, generator=g) * 0.5).to(torch.bfloat16)
        kappa = (torch.rand(B, T, H, 1, device=device, generator=g) * 0.6).to(torch.bfloat16)
        of = C.chunk_rola_routed(q, k, v, h, Wr, Ww, D, b, norm=norm,
                                 kappa=(kappa if norm == 'kappa' else None), scale=scale,
                                 **_routed_kwargs(h, Wr, Ww, D, b))
        oe = _kappa_ref_chunked(q.float(), k.float(), v.float(), h.float(), Wr.float(), Ww.float(),
                                kappa.float(), D, b, norm, scale,
                                min(64, max(16, triton.next_power_of_2(T))))
        assert _relmax(of.float(), oe.float()) < 1e-2, \
            f'fused vs explicit out {_relmax(of.float(), oe.float()):.2e}'


# =============================================================================
# SUITE 3 — AUTOGRAD: the fp64 gradcheck (analytic bwd == torch.autograd of the fwd). DISTINCT from the
# inter recurrent==chunked==naive leg — this anchors the naive ORACLE whose grads suite 2 trusts.
# (was test_oracle_gradcheck_rla / test_oracle_gradcheck_gla.) Sweep dv (incl. non-pow2) and nc (incl.
# non-pow2) so the oracle is itself proven valid at the awkward dims.
# =============================================================================
class TestAutograd:
    @pytest.mark.parametrize('dv', [4, 24, 48])
    @pytest.mark.parametrize('nc', [3, 6])
    def test_oracle_gradcheck_rla(self, nc, dv):
        """fp64 gradcheck of the naive global-norm oracle — proves its gradients are a valid reference,
        across non-pow2 dv/nc (the oracle anchors every kernel-vs-oracle gate)."""
        if device != 'cuda':
            pytest.skip('RoLA Triton kernels require CUDA')
        q, k, v, rg, wg = _mk_oracle(1, 12, 1, 8, nc=nc, dv=dv, dtype=torch.float64)
        ins = [t.detach().requires_grad_(True) for t in (q, k, v, rg, wg)]
        assert torch.autograd.gradcheck(
            lambda q, k, v, rg, wg: naive_rola_global(q, k, v, wg, rg),
            tuple(ins), eps=1e-6, atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize('dv', [4, 24, 48])
    @pytest.mark.parametrize('nc', [3, 6])
    def test_oracle_gradcheck_gla(self, nc, dv):
        """fp64 gradcheck of the naive GLA oracle (per-state decay ld), across non-pow2 dv/nc."""
        if device != 'cuda':
            pytest.skip('RoLA Triton kernels require CUDA')
        q, k, v, rg, wg, ld = _mk_oracle(1, 12, 1, 8, nc=nc, dv=dv, dtype=torch.float64, with_ld=True)
        ins = [t.detach().requires_grad_(True) for t in (q, k, v, rg, wg, ld)]
        assert torch.autograd.gradcheck(
            lambda q, k, v, rg, wg, ld: naive_rola_gla(q, k, v, wg, rg, ld, normalized=True),
            tuple(ins), eps=1e-6, atol=1e-5, rtol=1e-4)


# =============================================================================
# STRUCTURAL GATES (kept verbatim) — orthogonal to the equivalence axes. These assert PROPERTIES the
# equivalence tests can't: per-head routing/decay independence (the shared-router-bug catcher), signed-den
# consistency + the below-floor ld guard (loud-not-silent edges), and the no-[*,L,nc]-materialization
# allocation watches (the saved-activation win). See tests/layers/test_rola_routing_init for init-parity.
# =============================================================================
class TestStructuralGates:
    # ---- per-head ROUTING independence: the assertion the OLD shared-router reference COULD NOT make ----
    @pytest.mark.parametrize('norm', ['raw', 'global', 'per_state', 'kappa'])
    @pytest.mark.parametrize('D,b', [(1, 8), (2, 3), (3, 2)])   # flat, square(nc=9), tree(nc=8)
    def test_routed_per_head_independence(self, D, b, norm):
        """Distinct per-head routers ⇒ distinct per-head routing; shared router ⇒ identical. A
        shared-across-heads kernel would FAIL the DISTINCT case (the bug the old shared reference could not
        catch). IDENTICAL inputs across all H heads, so head outputs depend ONLY on the router."""
        if device != 'cuda':
            pytest.skip('RoLA Triton kernels require CUDA')
        B, H, T, Kd, V, dm = 2, 4, 64, 16, 16, 24
        g = torch.Generator(device=device).manual_seed(5)

        def fm(x):
            return torch.nn.functional.elu(x) + 1.0
        q = fm(torch.randn(B, T, 1, Kd, device=device, generator=g)).expand(B, T, H, Kd).contiguous()
        k = fm(torch.randn(B, T, 1, Kd, device=device, generator=g)).expand(B, T, H, Kd).contiguous()
        v = torch.randn(B, T, 1, V, device=device, generator=g).expand(B, T, H, V).contiguous()
        h = torch.randn(B, T, 1, dm, device=device, generator=g).expand(B, T, H, dm).contiguous()
        kap = (torch.rand(B, T, 1, 1, device=device, generator=g) * 0.5 + 0.3).expand(B, T, H, 1).contiguous() \
            if norm == 'kappa' else None
        Wr = torch.randn(H, D, dm, b, device=device, generator=g)        # DISTINCT per head
        Ww = torch.randn(H, D, dm, b, device=device, generator=g)
        kw = dict(norm=norm, kappa=kap, scale=1.0)
        o = C.chunk_rola_routed(q, k, v, h, Wr, Ww, D, b, **kw,
                                **_routed_kwargs(h, Wr, Ww, D, b))      # [B,T,H,V]
        diffs = [(o[:, :, i] - o[:, :, j]).abs().max().item() for i in range(H) for j in range(i + 1, H)]
        assert min(diffs) > 1e-3, \
            f'per-head routing collapsed: identical inputs + DISTINCT routers gave near-identical head ' \
            f'outputs (min cross-head diff {min(diffs):.2e}) — the router is being SHARED across heads'
        # CONTROL: broadcast head-0's router to all heads ⇒ identical inputs + router ⇒ identical out.
        Wr_s = Wr[:1].expand(H, D, dm, b).contiguous()
        Ww_s = Ww[:1].expand(H, D, dm, b).contiguous()
        o_s = C.chunk_rola_routed(q, k, v, h, Wr_s, Ww_s, D, b, **kw,
                                  **_routed_kwargs(h, Wr_s, Ww_s, D, b))
        same = max((o_s[:, :, i] - o_s[:, :, 0]).abs().max().item() for i in range(H))
        assert same < 1e-4, f'shared-router control: head outputs should be identical, got {same:.2e}'

    @pytest.mark.parametrize('norm', ['raw', 'global', 'kappa'])
    @pytest.mark.parametrize('D,b', [(1, 8), (2, 3)])
    def test_routed_per_head_decay_independence(self, D, b, norm):
        """The DECAY is PER-HEAD too (#45): distinct per-head decay weights Wg ⇒ distinct head outputs;
        shared Wg (with shared routers) ⇒ identical. A shared-across-heads decay weight would COLLAPSE the
        distinct case. Inputs IDENTICAL across heads + ROUTERS shared, so the outputs depend ONLY on Wg."""
        if device != 'cuda':
            pytest.skip('RoLA Triton kernels require CUDA')
        B, H, T, Kd, V, dm = 2, 4, 64, 16, 16, 24
        g = torch.Generator(device=device).manual_seed(7)

        def fm(x):
            return torch.nn.functional.elu(x) + 1.0
        q = fm(torch.randn(B, T, 1, Kd, device=device, generator=g)).expand(B, T, H, Kd).contiguous()
        k = fm(torch.randn(B, T, 1, Kd, device=device, generator=g)).expand(B, T, H, Kd).contiguous()
        v = torch.randn(B, T, 1, V, device=device, generator=g).expand(B, T, H, V).contiguous()
        h = torch.randn(B, T, 1, dm, device=device, generator=g).expand(B, T, H, dm).contiguous()
        kap = (torch.rand(B, T, 1, 1, device=device, generator=g) * 0.5 + 0.3).expand(B, T, H, 1).contiguous() \
            if norm == 'kappa' else None
        # SHARED routers (so routing alone can't make heads differ) — the only per-head asymmetry is Wg.
        Wr = (torch.randn(1, D, dm, b, device=device, generator=g) * 0.8).expand(H, D, dm, b).contiguous()
        Ww = (torch.randn(1, D, dm, b, device=device, generator=g) * 0.8).expand(H, D, dm, b).contiguous()
        Wg = torch.randn(H, dm, device=device, generator=g) * 0.6           # DISTINCT decay weight per head
        kw = dict(norm=norm, kappa=kap, scale=1.0, Wg=Wg)
        o = C.chunk_rola_routed(q, k, v, h, Wr, Ww, D, b, **kw,
                                **_routed_kwargs(h, Wr, Ww, D, b, Wg=Wg))  # [B,T,H,V]
        diffs = [(o[:, :, i] - o[:, :, j]).abs().max().item() for i in range(H) for j in range(i + 1, H)]
        assert min(diffs) > 1e-3, \
            f'per-head DECAY collapsed: identical inputs + shared routers + DISTINCT Wg gave near-identical ' \
            f'head outputs (min cross-head diff {min(diffs):.2e}) — the decay weight is being SHARED'
        # CONTROL: broadcast head-0's Wg ⇒ identical inputs + router + decay ⇒ identical out.
        Wg_s = Wg[:1].expand(H, dm).contiguous()
        o_s = C.chunk_rola_routed(q, k, v, h, Wr, Ww, D, b, **dict(kw, Wg=Wg_s),
                                  **_routed_kwargs(h, Wr, Ww, D, b, Wg=Wg_s))
        same = max((o_s[:, :, i] - o_s[:, :, 0]).abs().max().item() for i in range(H))
        assert same < 1e-4, f'shared-Wg control: head outputs should be identical, got {same:.2e}'

    # ---- signed-den consistency (#36 F1) + below-floor ld guard (#33 F2): loud, not silently divergent --
    def test_signed_den_consistent_decode_chunk_routed(self):
        """#36 F1 — SIGNED per-state den. Decode and production tree-routed chunk compute the RAW SIGNED
        den — no path silently tl.abs()es it — so for the SAME signed input the paths agree, never silently
        DIVERGE. (1) GLOBAL norm (den summed but NOT divided) is rate-consistent. (2)
        per_state / kappa produce the SAME non-finite MASK across paths (an abs() would change WHICH entries
        blow up); the divided magnitudes near d+ε≈0 are ill-conditioned + path-amplified, so the MASK (not
        the values) is the anti-divergence gate."""
        if device != 'cuda':
            pytest.skip('RoLA Triton kernels require CUDA')
        B, H, T, Kd, V, nc, dm = 2, 2, 32, 16, 16, 8, 24
        tol = 5e-3
        gseed = torch.Generator(device=device).manual_seed(7)

        def rnd(*s):
            return torch.randn(*s, device=device, generator=gseed, dtype=torch.float32)
        # SIGNED features (no elu/abs) → the content gram φq·φk is signed → d can be ≤0.
        q, k, v = rnd(B, T, H, Kd), rnd(B, T, H, Kd), rnd(B, T, H, V)
        h = rnd(B, T, H, dm)
        D, b = 1, nc
        Wr, Ww = rnd(H, D, dm, b) * 0.5, rnd(H, D, dm, b) * 0.5           # PER-HEAD router [H,D,dm,b]
        hf = h.permute(0, 2, 1, 3).reshape(B * H, T, dm)                  # [BH,T,dm] (the routed fold layout)
        rf, wf = _tree_gates_oop(hf, Wr, Ww, D, b, H)                     # [BH,T,nc] each, PER-HEAD gates
        r = rf.view(B, H, T, nc).permute(0, 2, 1, 3).contiguous()        # -> [B,T,H,nc]
        w = wf.view(B, H, T, nc).permute(0, 2, 1, 3).contiguous()
        assert (_perstate_den_signed(q, k, w) <= 0).any(), 'no non-positive den — adjust the seed/shape'

        def run(norm, kap=None):
            c = dict(r=r, w=w, norm=norm, kappa=kap, scale=1.0)
            od = fused_recurrent_rola(q, k, v, **c, output_final_state=True)[0].float()
            orr = C.chunk_rola_routed(q, k, v, h, Wr, Ww, D, b, norm=norm, kappa=kap, scale=1.0,
                                      **_routed_kwargs(h, Wr, Ww, D, b)).float()
            return od, orr

        od, orr = run('global')
        rg = r.permute(0, 2, 1, 3).reshape(B * H, T, nc)             # [BH,T,nc]
        dg = _perstate_den_signed(q, k, w)                           # [BH,T,nc]
        Dglob = (rg * dg).sum(-1).view(B, H, T).permute(0, 2, 1)    # [B,T,H] global den
        wc = (Dglob.abs() > 0.5)[..., None].expand_as(od)           # [B,T,H,V]
        assert wc.any(), 'no well-conditioned (|den|>0.5) entries — adjust seed'
        assert_close('signed-d global routed==decode (well-cond)', od[wc], orr[wc], tol)
        for norm in ('per_state', 'kappa'):
            kap = torch.full((B, T, H, 1), KAPPA, device=device) if norm == 'kappa' else None
            od, orr = run(norm, kap)
            fd, fr_ = torch.isfinite(od), torch.isfinite(orr)
            assert torch.equal(fd, fr_), f'{norm}: routed non-finite mask differs from decode (abs() in a path?)'

    def test_decode_below_floor_ld_raises_and_clamps(self, monkeypatch):
        """#33 F2 — the GLA log-decay floor guard is LOUD. An ld below `_GLA_FLOOR` must RAISE by default on
        the explicit-ld decode op; with ROLA_GLA_FLOOR_CLAMP=1 it must instead WARN-once and clamp."""
        if device != 'cuda':
            pytest.skip('RoLA Triton kernels require CUDA')
        B, H, T, Kd, V, nc = 1, 1, 16, 16, 16, 4
        g = torch.Generator(device=device).manual_seed(3)

        def fm(x):
            return torch.nn.functional.elu(x) + 1.0
        q = fm(torch.randn(B, T, H, Kd, device=device, generator=g))
        k = fm(torch.randn(B, T, H, Kd, device=device, generator=g))
        v = torch.randn(B, T, H, V, device=device, generator=g)
        r = torch.softmax(torch.randn(B, T, H, nc, device=device, generator=g), -1)
        w = torch.softmax(torch.randn(B, T, H, nc, device=device, generator=g), -1)
        ld = torch.full((B, T, H, nc), C._GLA_FLOOR - 1.0, device=device)   # below the floor
        common = dict(r=r, w=w, g=ld, norm='global', scale=1.0)

        def call():
            return fused_recurrent_rola(q, k, v, **common, output_final_state=True)
        monkeypatch.setattr(C, '_GLA_FLOOR_CLAMP', False, raising=False)
        with pytest.raises(ValueError, match='floor'):
            call()
        monkeypatch.setattr(C, '_GLA_FLOOR_CLAMP', True, raising=False)
        monkeypatch.setattr(C, '_gla_floor_warned', False, raising=False)
        import warnings
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            out = call()
        o = out[0] if isinstance(out, tuple) else out
        assert torch.isfinite(o).all(), 'clamp-mode output must be finite'
        assert any('clamp' in str(x.message).lower() or 'floor' in str(x.message).lower() for x in rec), \
            'clamp-mode must warn'

    # ---- no-[*,L,nc]-materialization allocation watches (the saved-activation win) -------------------
    def test_raw_split_streams_match_explicit_reference(self):
        """Raw GLA must use separate read, write, and decay streams in the weights-in API."""
        if device != 'cuda':
            pytest.skip('RoLA Triton kernels require CUDA')
        torch.manual_seed(0)
        B, T, H, Kd, V, dm, D, b = 1, 24, 2, 16, 16, 5, 2, 3
        q = (torch.nn.functional.elu(torch.randn(B, T, H, Kd, device=device)) + 1.0).to(torch.bfloat16)
        k = (torch.nn.functional.elu(torch.randn(B, T, H, Kd, device=device)) + 1.0).to(torch.bfloat16)
        v = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16)
        h_read = torch.randn(B, T, H, dm, device=device, dtype=torch.bfloat16)
        h_write = torch.randn(B, T, H, dm, device=device, dtype=torch.bfloat16)
        h_decay = torch.randn(B, T, H, dm, device=device, dtype=torch.bfloat16)
        Wr = (torch.randn(H, D, dm, b, device=device) * 0.2).to(torch.bfloat16)
        Ww = (torch.randn(H, D, dm, b, device=device) * 0.2).to(torch.bfloat16)
        Wg = (torch.randn(H, dm, device=device) * 0.2).to(torch.float32)

        alpha = torch.sigmoid(torch.einsum('bthm,hm->bth', h_decay.float(), Wg))
        streamed = C.chunk_rola_routed(q, k, v, h_read, Wr, Ww, D, b, norm='raw',
                                       scale=1.0, Wg=Wg, alpha=alpha,
                                       h_w=h_write, h_g=h_decay)

        def fold(x):
            return x.permute(0, 2, 1, 3).reshape(B * H, T, x.shape[-1])

        def unfold(x):
            return x.view(B, H, T, -1).permute(0, 2, 1, 3).contiguous()

        qf, kf, vf = fold(q), fold(k), fold(v)
        hf, hwf, hgf = fold(h_read), fold(h_write), fold(h_decay)
        lr, lw = C._router_logits(hf, Wr, Ww, None, None, H, h_w=hwf)
        r = C._gates_from_factor_logits(lr, D, b).to(q.dtype)
        w = C._gates_from_factor_logits(lw, D, b).to(q.dtype)
        ld = C._ld_from_Wg_torch(hgf, w, Wg, H)
        explicit = unfold(C._rola_chunk_core(qf.float(), kf.float(), vf.float(), w.float(), r.float(), ld, 64))
        assert _relmax(streamed, explicit) < 5e-3

    @pytest.mark.parametrize('gla', [False, True])
    def test_no_grad_tiled_prefill_matches_scan(self, gla):
        """No-grad prefill uses tiled GEMM routing and should match the differentiable scan."""
        if device != 'cuda':
            pytest.skip('RoLA Triton kernels require CUDA')
        torch.manual_seed(0)
        B, T, H, Kd, V, dm, D, b = 1, 128, 2, 16, 16, 32, 3, 2
        q = (torch.nn.functional.elu(torch.randn(B, T, H, Kd, device=device)) + 1.0).to(torch.bfloat16)
        k = (torch.nn.functional.elu(torch.randn(B, T, H, Kd, device=device)) + 1.0).to(torch.bfloat16)
        v = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16)
        h = torch.randn(B, T, H, dm, device=device, dtype=torch.bfloat16)
        Wr = (torch.randn(H, D, dm, b, device=device) * 0.2).to(torch.bfloat16)
        Ww = (torch.randn(H, D, dm, b, device=device) * 0.2).to(torch.bfloat16)
        kap = (torch.rand(B, T, H, 1, device=device) * 0.5).to(torch.bfloat16)
        Wg = (torch.randn(H, dm, device=device) * 0.2).to(torch.float32) if gla else None
        route = _routed_kwargs(h, Wr, Ww, D, b, Wg=Wg)
        with torch.no_grad():
            tiled = C.chunk_rola_routed(q, k, v, h, Wr, Ww, D, b, norm='kappa',
                                        kappa=kap, scale=1.0, Wg=Wg, **route)
        scan = C.chunk_rola_routed(q, k, v, h, Wr, Ww, D, b, norm='kappa',
                                   kappa=kap, scale=1.0, Wg=Wg, **route)
        assert _relmax(tiled, scan) < 1e-2

    @pytest.mark.parametrize('norm', ['global', 'kappa', 'per_state'])
    def test_kappa_routed_no_LNC_materialization(self, norm):
        """No [*,L,nc] d / r̃ / gate buffer is ever allocated in the fused global/kappa/per_state path
        (fwd+bwd). d_model != nc != L so any [*,L,nc]-shaped allocation is unambiguous."""
        if device != 'cuda':
            pytest.skip('RoLA Triton kernels require CUDA')
        D, b, nc = 2, 3, 9
        B, H, T, Kd, V, dm = 2, 2, 96, 16, 24, 40   # T=96, nc=9, dm=40 all distinct
        scale = Kd ** -0.5

        def mk(*s, f=False):
            x = torch.randn(*s, device=device)
            return ((torch.nn.functional.elu(x) + 1.0) if f else x).to(torch.bfloat16).requires_grad_()
        q, k, v, h = mk(B, T, H, Kd, f=True), mk(B, T, H, Kd, f=True), mk(B, T, H, V), mk(B, T, H, dm)
        Wr = (torch.randn(H, D, dm, b, device=device) * 0.5).to(torch.bfloat16).requires_grad_()
        Ww = (torch.randn(H, D, dm, b, device=device) * 0.5).to(torch.bfloat16).requires_grad_()
        kappa = (torch.rand(B, T, H, 1, device=device) * 0.5).to(torch.bfloat16).requires_grad_()
        hits = []
        real_zeros, real_empty = torch.zeros, torch.empty

        def watch(fn):
            def w(*a, **kw):
                t = fn(*a, **kw)
                sh = tuple(t.shape)
                if T in sh and nc in sh:        # a [*,L,nc]-shaped d/r̃/gate buffer would trip this
                    hits.append(sh)
                return t
            return w
        torch.zeros, torch.empty = watch(real_zeros), watch(real_empty)
        try:
            o = C.chunk_rola_routed(q, k, v, h, Wr, Ww, D, b, norm=norm,
                                    kappa=(kappa if norm == 'kappa' else None), scale=scale,
                                    **_routed_kwargs(h, Wr, Ww, D, b))
            o.sum().backward()
        finally:
            torch.zeros, torch.empty = real_zeros, real_empty
        assert not hits, f'[*,L={T},nc={nc}] buffer(s) materialized: {hits}'

    def test_gla_routed_fwd_no_LNC_materialization(self):
        """The fused routed GLA forward never allocates ANY [*,L,nc] buffer — neither the routed read/write
        GATE nor (now, #45) the per-state log-decay ld: the decay is computed IN-KERNEL from Wg:[H,d_model]
        (the saved-activation win). d_model != nc != L so any [*,L,nc] alloc is unambiguous."""
        if device != 'cuda':
            pytest.skip('RoLA Triton kernels require CUDA')
        D, b, nc = 2, 3, 9
        BH, L, dqk, dv, dm = 2, 96, 16, 24, 40
        g_ = torch.Generator(device=device).manual_seed(0)

        def mk(*s, f=False):
            x = torch.randn(*s, device=device, generator=g_)
            return torch.nn.functional.elu(x) + 1.0 if f else x
        q, k = mk(BH, L, dqk, f=True), mk(BH, L, dqk, f=True)
        v, h = mk(BH, L, dv), mk(BH, L, dm)
        Wr = torch.randn(1, D, dm, b, device=device, generator=g_) * 0.4   # H=1 (BH-as-batch op-level test)
        Ww = torch.randn(1, D, dm, b, device=device, generator=g_) * 0.4
        Wg = torch.randn(1, dm, device=device, generator=g_) * 0.4         # per-head decay weight [H=1,d_model]
        sel = C._build_sel(D, b, nc, device)
        # warm the kernel (cold autotune itself calls torch.empty) BEFORE the watch.
        C._routed_fwd_tiled(q, k, v, h, Wr, Ww, D, b, sel, chunk=32, BG=16, Wg=Wg)
        hits = []
        real_zeros, real_empty = torch.zeros, torch.empty

        def watch(fn):
            def w(*a, **kw):
                t = fn(*a, **kw)
                sh = tuple(t.shape)
                if L in sh and nc in sh:
                    hits.append(sh)
                return t
            return w
        torch.zeros, torch.empty = watch(real_zeros), watch(real_empty)
        try:
            C._routed_fwd_tiled(q, k, v, h, Wr, Ww, D, b, sel, chunk=32, BG=16, Wg=Wg)
        finally:
            torch.zeros, torch.empty = real_zeros, real_empty
        assert not hits, f'[*,L={L},nc={nc}] buffer(s) materialized (gate or ld): {hits}'

    def test_gla_routed_bwd_no_LNC_materialization(self):
        """The fused routed GLA BACKWARD never allocates a [*,L,nc] routed GATE-GRAD buffer, and (now, #45)
        NEITHER ld NOR dld is [L,nc]: ld is computed in-kernel from Wg, dld is assembled PER CHUNK. The ONLY
        [*,L,nc] backward allocation is the gda decay-adjoint accumulator. Bound: <=1 (gda)."""
        if device != 'cuda':
            pytest.skip('RoLA Triton kernels require CUDA')
        D, b, nc = 2, 3, 9
        BH, L, dqk, dv, dm = 2, 96, 16, 24, 40
        g = torch.Generator(device=device).manual_seed(0)

        def mk(*s, f=False):
            x = torch.randn(*s, device=device, generator=g)
            return (torch.nn.functional.elu(x) + 1.0) if f else x
        q = mk(BH, L, dqk, f=True).requires_grad_()
        k = mk(BH, L, dqk, f=True).requires_grad_()
        v, h = mk(BH, L, dv).requires_grad_(), mk(BH, L, dm).requires_grad_()
        Wr = (torch.randn(1, D, dm, b, device=device, generator=g) * 0.4).requires_grad_()   # H=1
        Ww = (torch.randn(1, D, dm, b, device=device, generator=g) * 0.4).requires_grad_()
        Wg = (torch.randn(1, dm, device=device, generator=g) * 0.4).requires_grad_()         # per-head decay
        go = torch.randn(BH, L, dv, device=device, generator=g)
        # warm the kernels (cold autotune calls torch.empty) BEFORE the watch.
        o = C.rola_gla_routed_triton(q, k, v, h, Wr, Ww, Wg, D, b)
        torch.autograd.grad(o, [q, k, v, h, Wr, Ww, Wg], go, retain_graph=False)
        hits = []
        real_zeros, real_empty = torch.zeros, torch.empty

        def watch(fn):
            def w(*a, **kw):
                t = fn(*a, **kw)
                if L in tuple(t.shape) and nc in tuple(t.shape):
                    hits.append(tuple(t.shape))
                return t
            return w
        o = C.rola_gla_routed_triton(q, k, v, h, Wr, Ww, Wg, D, b)
        torch.zeros, torch.empty = watch(real_zeros), watch(real_empty)
        try:
            torch.autograd.grad(o, [q, k, v, h, Wr, Ww, Wg], go, retain_graph=False)
        finally:
            torch.zeros, torch.empty = real_zeros, real_empty
        assert len(hits) <= 1, f'[*,L={L},nc={nc}] backward allocations exceed gda (the only legit one): {hits}'

    # ---- the GLA in-kernel-routed forward faithfulness (V1 #30/#45): the [L,nc]-free + ld-free readout ---
    @pytest.mark.parametrize('chunk', [64, 16])  # 64 = single-chunk; 16 = multi-chunk (inter-chunk decay)
    @pytest.mark.parametrize('D,b', _ROUTE_FLAT_SQ_TREE_16)
    def test_gla_routed_fwd_faithful(self, D, b, chunk):
        """Routed GLA numerator forward (in-kernel ld from Wg, #45) == naive_rola_gla oracle on the
        tree-materialized gates + the LAYER's ld(Wg) formula — the [L,nc]-free AND ld-free fused readout
        faithful to the explicit-gate math. flat/square/tree."""
        if device != 'cuda':
            pytest.skip('RoLA Triton kernels require CUDA')
        nc = b ** D
        BH, L, dqk, dv, dm = 2, 64, 16, 32, 24
        g_ = torch.Generator(device=device).manual_seed(0)

        def mk(*s, f=False):
            x = torch.randn(*s, device=device, generator=g_)
            return torch.nn.functional.elu(x) + 1.0 if f else x
        q, k = mk(BH, L, dqk, f=True), mk(BH, L, dqk, f=True)
        v, h = mk(BH, L, dv), mk(BH, L, dm)
        Wr = torch.randn(1, D, dm, b, device=device, generator=g_) * 0.4   # H=1 (BH-as-batch op-level test)
        Ww = torch.randn(1, D, dm, b, device=device, generator=g_) * 0.4
        Wg = torch.randn(1, dm, device=device, generator=g_) * 0.4         # per-head decay weight [H=1,d_model]
        sel = C._build_sel(D, b, nc, device)
        o_routed = C._routed_fwd_tiled(q, k, v, h, Wr, Ww, D, b, sel, chunk=chunk, BG=16, Wg=Wg)
        r, w = C._tree_gates_torch(h, Wr, Ww, D, b, 1)
        ld = _ld_from_Wg(h, w, Wg, 1)   # the LAYER's ld(Wg) — the oracle the in-kernel ld must match

        def unf(t):
            return t.view(BH, L, 1, -1)
        o_naive = naive_rola_gla(unf(q), unf(k), unf(v), unf(w), unf(r), unf(ld),
                                 normalized=False).view(BH, L, dv)
        rel = ((o_routed - o_naive).norm() / (o_naive.norm() + 1e-9)).item()
        assert rel < 1e-2, f'D={D} b={b} routed-GLA-fwd(Wg) vs naive(ld(Wg)) rel {rel:.2e}'
