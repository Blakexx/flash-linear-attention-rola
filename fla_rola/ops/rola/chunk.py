# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# RoLA routing — Triton kernels (additive extension of simple_gla).
#
# Routed linear attention shares the content gram G=qkᵀ across `nc` states and modulates it by a
# routing gram R=Σ_c r_i^c w_j^c (+ optional per-state scalar decay). These kernels compute the
# *un-normalized* routed readout O = (G∘R∘causal) @ v — the FLA convention; the global denominator is
# reconstructed by the caller as Σ_c r̃ᶜ·dᶜ from the per-state den pre-pass (see `chunk_rola` at the
# bottom of this file — the norm-aware public entry point, FLA-style). Tiled
# over state-blocks (BG states/program) so only this block's slice of the Kronecker state lives in
# SRAM → scales to any nc. The content gram is formed ONCE per chunk (the FLOP win), never
# materializing the L×nc product nor replicating q/k.
#
# Ported verbatim from the verified `rola_kernels` reference (gradcheck + fp64-exact + matched vs the
# O(L²) ground truth). The readout is numerator-only (width dv, BV=next_pow2(dv)); there is no
# ones-column augmentation — the denominator is the caller's separate per-state pre-pass.

import os
import typing  # noqa: F401 (torch custom_op schema needs typing.List)

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from fla_rola.utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    autotune_cache_kwargs,
    check_shared_mem,
    input_guard,
)

# SMEM fitting follows the canonical FLA idiom: the chunk size BT is a fixed per-device constant
# (a `check_shared_mem` gate — the one structural knob the autotuner can't own, since the host-side
# Sb/dSa allocations + launch grid are sized from it before any kernel compiles, and GLA needs BT<=32
# for its fp32 decay floor), while the INNER blocks (num_warps / num_stages / BD) are autotune knobs
# that Triton prunes via OutOfResources. The denominator-split (numerator-only readout, BV=next_pow2
# (dv)) halved the per-program footprint, so the ada-class BT (64 RLA-fwd / 32 GLA+bwd) now fits down
# to RTX 30xx (measured); sm75 (T4, 64KB) drops to BT=16.
_BIG_SMEM = check_shared_mem('ada')
_WARPS = (2, 4, 8)
_STAGES = (1, 2, 3)
_AT_CFGS = [triton.Config({}, num_warps=w, num_stages=s) for w in _WARPS for s in _STAGES]
_AT_KEY = ['dqk', 'dv', 'nc']
_CHUNK_FWD = 64 if _BIG_SMEM else 16     # RLA forward
_CHUNK = 32 if _BIG_SMEM else 16         # GLA forward + all backwards (GLA fp32 decay floor caps BT<=32)

# --- Backward feature tiling: BD as an autotune knob ----------------------------------------------
# The backward grad kernels load a [BD, BG*BV] slice of the Kronecker state per feature-block into
# SRAM. Rather than model that footprint with a byte budget (fragile — it misses Triton's tl.dot
# operand staging), we expose the feature tile BD itself as an autotune knob and let the autotuner
# EMPIRICALLY pick the largest that fits: Triton's autotuner compiles each config and scores any
# OutOfResources as inf, so over-SRAM tiles are dropped against the real hardware. BD=16 always fits
# and floors the set, so a config always survives. This is how FLA's own kernels handle SRAM limits —
# no magic constant, no headroom factor; an A100/H100 lands on a big BD + deep pipeline, a 3080 Ti on
# BD=16. BG is pinned at the dot minimum (tl.dot dims must be >=16). NB: BD here tiles the GRAD
# kernels only; the scans take their own (independent) feature block — Sb is indexed by absolute
# feature row, so the two blockings need not match.
# --- Backward value tiling: BV as an autotune knob (twin of BD) -----------------------------------
# The Kronecker state/grad tiles are [*, BG*BV] and the value-scaled dot operands (WV, wg_v1, rg_g,
# dSa slices) stage to SRAM linearly in BV=value-width. At large d_v (64+) the full-width tile blows
# the per-block smem cap (99KB on sm86/sm89) and — because BV was a *fixed* host const — there was no
# smaller config to fall back to, so the autotuner had nothing that fit. Exposing BV as a knob (like
# BD) and looping cdiv(d_v, BV) value-blocks gives the autotuner the fitting options: BV=64 on an
# A100 with room, BV=32/16 (2/4 blocks) on a 99KB card. d_v is a constexpr (already an _AT_KEY), so
# cdiv(d_v,BV)=1 at d_v<=BV UNROLLS to byte-identical code — zero perf impact for the configs we run.
_BWD_BV = (16, 32, 64)                     # value-tile candidates; pruned to <= next_pow2(d_v)
_BWD_BK = (16, 32, 64, 128)               # feature-tile candidates; autotuner keeps the largest fitting
_BWD_CFGS = [triton.Config({'BD': bk}, num_warps=w, num_stages=s)   # BD-only (den kernels, BV-free)
             for bk in _BWD_BK for w in _WARPS for s in _STAGES]
# value-looped grad kernels additionally tile the value axis: BD × BV.
_BWD_CFGS_BV = [triton.Config({'BD': bk, 'BV': bv}, num_warps=w, num_stages=s)
                for bk in _BWD_BK for bv in _BWD_BV for w in _WARPS for s in _STAGES]
# GLA-recompute grad kernels (_par_grad_gla_{qr,kwv}) carry an IN-KERNEL decay-replay sub-scan
# (_recompute_{S,dS}_gla) on top of the value-looped body — a much larger kernel than the RLA grads.
# Each surviving config compiles AND benchmarks the full replay, so the cold autotune of the un-pruned
# BD×BV×warp×stage grid (the same _BWD_CFGS_BV the RLA grads use) ran ~40 min and stalled the GLA gate.
# These two kernels do NOT benefit from a deep software pipeline: the replay's wide fp32 Kronecker-state
# tensors (BG·BV·BD) already pin SMEM/registers to a single resident block (see the latency profile —
# occupancy is state-tensor-SMEM-bound, not stage-bound), so num_stages>1 only multiplies compile/bench
# cost with no perf upside, and 8 warps over-subscribes the same starved block. So this set caps
# num_stages=1 and trims warps to (2,4) — keeping the BD/BV fit knobs (the ones that actually decide
# whether a config runs at all) at full range. ~4-5x fewer configs ⇒ single-digit-minute cold compile,
# correctness unchanged (every surviving config computes identical math; stages/warps are perf-only).
_GLA_BWD_WARPS = (2, 4)
_BWD_CFGS_GLA = [triton.Config({'BD': bk, 'BV': bv}, num_warps=w, num_stages=1)
                 for bk in _BWD_BK for bv in _BWD_BV for w in _GLA_BWD_WARPS]
# scans have no BD knob (Sflat is a register carry, own fixed feature block) but DO need the BV knob.
_SCAN_CFGS = [triton.Config({'BV': bv}, num_warps=w, num_stages=s)
              for bv in _BWD_BV for w in _WARPS for s in _STAGES]


def _bv_cap(dv):
    return max(16, triton.next_power_of_2(dv))


def _prune_bv(configs, named_args, **kwargs):
    """Cap BV at next_pow2(d_v): a bigger value-tile than the value dim is pure waste (and would blow
    up the config grid). Floor 16 always survives. Mirrors _prune_bwd_bd for the value axis."""
    try:
        cap = _bv_cap(named_args['dv'])
    except Exception:
        return configs
    keep = [c for c in configs if c.kwargs.get('BV', 16) <= cap]
    return keep or [c for c in configs if c.kwargs.get('BV', 16) == 16] or configs


def _prune_bwd(configs, named_args, **kwargs):
    """Combined BD + BV prune for the grad kernels."""
    return _prune_bv(_prune_bwd_bd(configs, named_args, **kwargs), named_args, **kwargs)


def _prune_bwd_bd(configs, named_args, **kwargs):
    """Exact (not heuristic) prune: drop feature tiles larger than the padded feature dim — pure
    waste, never a fit question. SRAM-too-big tiles are pruned empirically by the autotuner
    (OutOfResources -> inf). Keeps BD<=16 as the floor so the set is never empty."""
    try:
        cap = max(16, triton.next_power_of_2(named_args['dqk']))
    except Exception:
        return configs
    keep = [c for c in configs if c.kwargs['BD'] <= cap]
    return keep or [c for c in configs if c.kwargs['BD'] == 16] or configs


@triton.autotune(configs=_AT_CFGS, key=_AT_KEY, **autotune_cache_kwargs)
@triton.jit
def _rola_fwd_intra(q_ptr, k_ptr, v_ptr, wg_ptr, rg_ptr, outa_ptr,
                    L, dqk, dv, nc,
                    sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c,
                    soa_b, soa_l, soa_v,
                    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                    BG: tl.constexpr, ND: tl.constexpr, NB: tl.constexpr):
    """Intra-chunk routed readout for one (batch, chunk) — the nc axis COLLAPSES here:
    o = (G ⊙ (r·wᵀ) ⊙ causal)·v. Content gram G built ONCE (loop BK-blocks of dqk); the FULL routing
    gram R is accumulated over state-blocks IN-KERNEL (loop NB), so there is no per-sb HBM grid — the
    routed sum is done in SRAM and the output is [B,L,dv]. SRAM bounded by [BT,BK]+[BT,BG]+[BT,BT]."""
    b = tl.program_id(0)
    t = tl.program_id(1)
    offs_t = tl.arange(0, BT)
    offs_v = tl.arange(0, BV)
    rows = t * BT + offs_t
    rmask = rows < L
    G = tl.zeros([BT, BT], dtype=tl.float32)
    for d0 in range(ND):
        offs_d = d0 * BK + tl.arange(0, BK)
        dmask = offs_d < dqk
        qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        G += tl.dot(qc, tl.trans(kc))
    R = tl.zeros([BT, BT], dtype=tl.float32)
    for sb in range(NB):
        offs_c = sb * BG + tl.arange(0, BG)
        cmask = offs_c < nc
        rgc = tl.load(rg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                      mask=rmask[:, None] & cmask[None, :], other=0.0)
        wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                      mask=rmask[:, None] & cmask[None, :], other=0.0)
        R += tl.dot(rgc, tl.trans(wgc))
    vc = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                 mask=rmask[:, None] & (offs_v[None, :] < dv), other=0.0)
    causal = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
    A = G * R * causal
    o = tl.dot(A.to(vc.dtype), vc)
    tl.store(outa_ptr + b*soa_b + rows[:, None]*soa_l + offs_v[None, :]*soa_v,
             o, mask=rmask[:, None] & (offs_v[None, :] < dv))


@triton.autotune(configs=_SCAN_CFGS, key=_AT_KEY, reset_to_zero=['outa_ptr'],
                 prune_configs_by={'early_config_prune': _prune_bv}, **autotune_cache_kwargs)
@triton.jit
def _rola_fwd_inter(q_ptr, k_ptr, v_ptr, wg_ptr, rg_ptr, outa_ptr,
                    L, dqk, dv: tl.constexpr, nc,
                    sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c,
                    soa_b, soa_n, soa_l, soa_v,
                    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                    BVF: tl.constexpr, BG: tl.constexpr, NCH: tl.constexpr):
    """Inter-chunk (state) contribution for one (batch, state-block, FEATURE-block d0). Carries this
    feature-block's slice Sd[BK, BG*BV] of the Kronecker state across chunks → SRAM bounded by BK and
    the autotuned value-tile BV (value-OUTER over cdiv(dv,BV) blocks; ND_V==1 == the un-tiled kernel).
    o_inter uses the state BEFORE this chunk's update (causal); partials over feature-blocks sum via
    atomic_add into its OWN out_inter buffer — autotunable with reset_to_zero."""
    b = tl.program_id(0)
    sb = tl.program_id(1)
    d0 = tl.program_id(2)
    ND_V = (dv + BV - 1) // BV
    offs_t = tl.arange(0, BT)
    offs_d = d0 * BK + tl.arange(0, BK)
    dmask = offs_d < dqk
    offs_c = sb * BG + tl.arange(0, BG)
    cmask = offs_c < nc
    for vb in range(ND_V):
        offs_v = vb * BV + tl.arange(0, BV)
        vmask = offs_v < dv
        Sd = tl.zeros([BK, BG * BV], dtype=tl.float32)
        for t in range(NCH):
            rows = t * BT + offs_t
            rmask = rows < L
            qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                         mask=rmask[:, None] & dmask[None, :], other=0.0)
            rgc = tl.load(rg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                          mask=rmask[:, None] & cmask[None, :], other=0.0)
            P = tl.dot(qc, Sd.to(qc.dtype))
            P3 = tl.reshape(P, [BT, BG, BV])
            o_inter = tl.sum(P3 * rgc[:, :, None], axis=1)
            # NB-fused: state-blocks atomic-accumulate the routed sum into shared [B,L,dv] (no per-sb
            # grid). Frontier-safe: each program still carries only THIS block's state Sd in registers.
            tl.atomic_add(outa_ptr + b*soa_b + rows[:, None]*soa_l + offs_v[None, :]*soa_v,
                          o_inter, mask=rmask[:, None] & vmask[None, :])
            kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                         mask=rmask[:, None] & dmask[None, :], other=0.0)
            vc = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                         mask=rmask[:, None] & vmask[None, :], other=0.0)
            wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                          mask=rmask[:, None] & cmask[None, :], other=0.0)
            WV = tl.reshape(wgc[:, :, None] * vc[:, None, :], [BT, BG * BV])
            Sd += tl.dot(tl.trans(kc), WV.to(kc.dtype))


def _fwd_tiled(q, k, v, wg, rg, chunk, BG, BK=64):
    """D-tiled numerator-only forward: smem bounded by BK (feature-block), so ANY dqk fits. Returns
    [B,L,dv] at BV=next_pow2(dv); the kappa/per_state caller forms the global den from the den
    pre-pass (Σ_c rᶜ·dᶜ)."""
    B, L, dqk = q.shape
    dv = v.shape[-1]
    nc = wg.shape[-1]
    BV = max(16, triton.next_power_of_2(dv))
    BK = min(BK, max(16, triton.next_power_of_2(dqk)))   # exact-width blocks for dqk<=64; tile beyond
    ND = triton.cdiv(dqk, BK)
    NB = triton.cdiv(nc, BG)
    NCH = triton.cdiv(L, chunk)
    q, k, v, wg, rg = [x.contiguous() for x in (q, k, v, wg, rg)]
    # Separate intra/inter output buffers (summed after) so each kernel has a non-shared output and is
    # independently autotunable: intra writes disjoint rows (store), inter atomic-accumulates over
    # feature-blocks (reset_to_zero). FLA idiom — the autotuner prunes warps/stages per device.
    # Both collapse the per-sb HBM grid → [B,L,BV]: intra collapses nc in-kernel; inter atomic-
    # accumulates the routed sum over state-blocks (the read routing can't collapse, but the SUM can
    # be done via atomic into the shared buffer — each program still carries only its own state).
    out_intra = torch.zeros(B, L, BV, device=q.device, dtype=torch.float32)
    out_inter = torch.zeros(B, L, BV, device=q.device, dtype=torch.float32)
    so_a = (out_intra.stride(0), out_intra.stride(1), out_intra.stride(2))
    so_e = (out_inter.stride(0), 0, out_inter.stride(1), out_inter.stride(2))
    base = (q.stride(0), q.stride(1), q.stride(2), v.stride(0), v.stride(1), v.stride(2),
            wg.stride(0), wg.stride(1), wg.stride(2))
    _rola_fwd_intra[(B, NCH)](q, k, v, wg, rg, out_intra, L, dqk, dv, nc, *base, *so_a,
                              BT=chunk, BK=BK, BV=BV, BG=BG, ND=ND, NB=NB)
    _rola_fwd_inter[(B, NB, ND)](q, k, v, wg, rg, out_inter, L, dqk, dv, nc, *base, *so_e,
                                 BT=chunk, BK=BK, BVF=BV, BG=BG, NCH=NCH)
    return out_intra[..., :dv] + out_inter[..., :dv]


@torch.library.custom_op("rola::readout_rla", mutates_args=())
def _readout_rla(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 wg: torch.Tensor, rg: torch.Tensor, chunk: int, BG: int) -> torch.Tensor:
    """RoLA-RLA un-normalized routed readout as an opaque custom op (Triton kernels inside → no
    Dynamo graph break; inductor fuses the surrounding glue)."""
    with torch.autocast('cuda', enabled=False):
        return _fwd_tiled(q, k, v, wg, rg, chunk=chunk, BG=BG).to(q.dtype)


@_readout_rla.register_fake
def _readout_rla_fake(q, k, v, wg, rg, chunk, BG):
    return q.new_empty((q.shape[0], q.shape[1], v.shape[-1]))


def _readout_rla_setup(ctx, inputs, output):
    q, k, v, wg, rg, chunk, BG = inputs
    ctx.save_for_backward(q, k, v, wg, rg)


@torch.library.custom_op("rola::readout_rla_bwd", mutates_args=())
def _readout_rla_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     wg: torch.Tensor, rg: torch.Tensor, grad: torch.Tensor) -> typing.List[torch.Tensor]:  # noqa: UP006
    """Opaque backward (Triton kernels inside) so inductor doesn't trace the kernel launches."""
    with torch.autocast('cuda', enabled=False):
        dq, dk, dvv, dw, dr = _bwd_split_rla(q, k, v, wg, rg, grad, chunk=_CHUNK)
    return [dq, dk, dvv, dw, dr]


@_readout_rla_bwd.register_fake
def _readout_rla_bwd_fake(q, k, v, wg, rg, grad):
    f = torch.float32
    return [q.new_empty(q.shape, dtype=f), k.new_empty(k.shape, dtype=f),
            v.new_empty(v.shape, dtype=f), wg.new_empty(wg.shape, dtype=f), rg.new_empty(rg.shape, dtype=f)]


def _readout_rla_backward(ctx, grad):
    q, k, v, wg, rg = ctx.saved_tensors
    dq, dk, dvv, dw, dr = _readout_rla_bwd(q, k, v, wg, rg, grad)
    def cast(t): return t.to(q.dtype)
    return cast(dq), cast(dk), cast(dvv), cast(dw), cast(dr), None, None


_readout_rla.register_autograd(_readout_rla_backward, setup_context=_readout_rla_setup)


@input_guard
def rola_rla_triton(q, k, v, r, w, chunk=None, BG=16):
    """Un-normalized routed RLA readout via Triton. q,k:[BH,L,K] v:[BH,L,V] r,w:[BH,L,nc]
    (r=read gate, w=write gate). Differentiable (fused Triton backward). Numerator-only ⇒ returns
    [BH,L,V] at BV=next_pow2(V); the kappa/per_state caller reconstructs the global denominator as
    Σ_c rᶜ·dᶜ from the per-state den pre-pass."""
    chunk = _CHUNK_FWD if chunk is None else min(chunk, _CHUNK_FWD)
    return _readout_rla(q, k, v, w, r, chunk, BG)


# ============================================================================
# Scalar-gated RoLA-GLA Triton kernels (shared-gram + per-state scalar decay).
#
# Decay is absorbed chunk-LOCALLY (rt=rg·e^a, wt=wg·e^{-a}, a=cumsum(ld)) so the effective weight
# keeps the shared-gram form; the Kronecker state is carried across chunks with a per-chunk decay
# (decayed scan). The per-token log-decay is floored (fp32-safe factored gram for BT≤32). Forward
# tiled over states; backward is the bespoke fused fwd-scan (dq,drg,da_rt) + reverse-scan
# (dk,dwg,dv,da_kwv,dLam) + a torch dld assembly (reverse-cumsum of da per chunk). Ported verbatim
# from the verified reference (all 6 grads incl. dld matched the torch GLA reference to fp noise).
# ============================================================================
_GLA_FLOOR = -2.5   # per-token log-decay floor (retention ≥ 8.2%/tok); fp32-safe for BT≤32

# Snapshot-GRANULARITY stride (#1): the backward boundary-state snapshots Sb/dSa are written only every
# KSNAP chunks (slot = chunk//KSNAP), 1/KSNAP the dominant backward HBM. The grad kernels read the
# nearest coarse anchor and recompute the ≤KSNAP-1 intervening chunks (a [BD,BG*BV] sub-scan — the
# SAME working set as the snapshot path, so it fits SMEM at ANY dqk; full register recompute OOM'd
# because it carried the [dqk_pad,*] feature width). Force-overridable via ROLA_SNAP_K; otherwise
# _resolve_snap_k picks the smallest K (least recompute) whose snapshot peak fits ROLA_SNAP_BUDGET_MB.
_SNAP_BUDGET_MB = 256   # default snapshot HBM target (Sb+dSa resident); overridable via ROLA_SNAP_BUDGET_MB


def _resolve_snap_k(B, NB, NCH, BD, BV, BG):
    """Snapshot-granularity stride K (#1). Priority: ROLA_SNAP_K env (force) > auto-budget. The resident
    snapshot buffers are Sb+dSa = 2·B·NB·ceil(NCH/K)·BD·BG·BV·4 bytes; raising K shrinks them as 1/K
    while paying K-1 chunks of recompute. Auto picks the SMALLEST K (least recompute) whose snapshot
    peak fits under ROLA_SNAP_BUDGET_MB (default 256): small/short shapes resolve to K=1 (no
    granularity, no recompute cost), only long-L/large-nc shapes — the ones that OOM — climb K. Capped
    at NCH (K≥NCH is a single snapshot). Returns K in [1, NCH]."""
    if os.environ.get('ROLA_SNAP_K'):
        return max(1, min(int(os.environ['ROLA_SNAP_K']), NCH))
    budget = int(os.environ.get('ROLA_SNAP_BUDGET_MB', str(_SNAP_BUDGET_MB))) * (1 << 20)
    per_slot = max(1, 2 * B * NB * BD * BG * BV * 4)         # (Sb+dSa) fp32 bytes per snapshot SLOT
    k = 1
    while k < NCH and per_slot * triton.cdiv(NCH, k) > budget:
        k += 1
    return max(1, min(k, NCH))


@triton.autotune(configs=_AT_CFGS, key=_AT_KEY, **autotune_cache_kwargs)
@triton.jit
def _rola_gla_fwd_intra(q_ptr, k_ptr, v_ptr, wg_ptr, rg_ptr, ld_ptr, outa_ptr,
                        L, dqk, dv, nc,
                        sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c,
                        soa_b, soa_n, soa_l, soa_v,
                        BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                        BG: tl.constexpr, ND: tl.constexpr):
    """GLA intra-chunk routed readout for one (batch, state-block, chunk). Same as the RLA intra but
    the routing gram uses the decayed gates rt=rg·e^a, wt=wg·e^-a (a = intra-chunk cumsum of ld). The
    [BT,BT] content gram is built by looping BK-blocks of the feature dim → SRAM bounded by BK.
    Numerator-only (output width dv, BV=next_pow2(dv))."""
    b = tl.program_id(0)
    sb = tl.program_id(1)
    t = tl.program_id(2)
    offs_t = tl.arange(0, BT)
    offs_v = tl.arange(0, BV)
    offs_c = sb * BG + tl.arange(0, BG)
    cmask = offs_c < nc
    rows = t * BT + offs_t
    rmask = rows < L
    G = tl.zeros([BT, BT], dtype=tl.float32)
    for d0 in range(ND):
        offs_d = d0 * BK + tl.arange(0, BK)
        dmask = offs_d < dqk
        qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        G += tl.dot(qc, tl.trans(kc))
    rgc = tl.load(rg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                  mask=rmask[:, None] & cmask[None, :], other=0.0)
    wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                  mask=rmask[:, None] & cmask[None, :], other=0.0)
    ldc = tl.load(ld_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                  mask=rmask[:, None] & cmask[None, :], other=0.0)
    vc = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                 mask=rmask[:, None] & (offs_v[None, :] < dv), other=0.0)
    a = tl.cumsum(ldc, axis=0)
    rt = rgc * tl.exp(a)
    wt = wgc * tl.exp(-a)
    R = tl.dot(rt, tl.trans(wt))
    causal = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
    A = G * R * causal
    o = tl.dot(A.to(vc.dtype), vc)
    tl.store(outa_ptr + b*soa_b + sb*soa_n + rows[:, None]*soa_l + offs_v[None, :]*soa_v,
             o, mask=rmask[:, None] & (offs_v[None, :] < dv))


@triton.autotune(configs=_SCAN_CFGS, key=_AT_KEY, reset_to_zero=['outa_ptr'],
                 prune_configs_by={'early_config_prune': _prune_bv}, **autotune_cache_kwargs)
@triton.jit
def _rola_gla_fwd_inter(q_ptr, k_ptr, v_ptr, wg_ptr, rg_ptr, ld_ptr, outa_ptr,
                        L, dqk, dv: tl.constexpr, nc,
                        sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c,
                        soa_b, soa_n, soa_l, soa_v,
                        BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                        BVF: tl.constexpr, BG: tl.constexpr, NCH: tl.constexpr):
    """GLA inter-chunk (state) contribution for one (batch, state-block, FEATURE-block d0). Carries
    this feature-block's slice Sd[BK, BG*BV] across chunks → SRAM bounded by BK and the autotuned
    value-tile BV (value-OUTER; ND_V==1 == un-tiled). The state decays by decvec = e^Λ (Λ = chunk-total
    ld) each chunk; the read uses rt = rg·e^a. The decay is d-independent; partials sum via atomic_add."""
    b = tl.program_id(0)
    sb = tl.program_id(1)
    d0 = tl.program_id(2)
    ND_V = (dv + BV - 1) // BV
    offs_t = tl.arange(0, BT)
    offs_d = d0 * BK + tl.arange(0, BK)
    dmask = offs_d < dqk
    offs_c = sb * BG + tl.arange(0, BG)
    cmask = offs_c < nc
    for vb in range(ND_V):
        offs_v = vb * BV + tl.arange(0, BV)
        vmask = offs_v < dv
        Sd = tl.zeros([BK, BG * BV], dtype=tl.float32)
        for t in range(NCH):
            rows = t * BT + offs_t
            rmask = rows < L
            qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                         mask=rmask[:, None] & dmask[None, :], other=0.0)
            rgc = tl.load(rg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                          mask=rmask[:, None] & cmask[None, :], other=0.0)
            ldc = tl.load(ld_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                          mask=rmask[:, None] & cmask[None, :], other=0.0)
            a = tl.cumsum(ldc, axis=0)
            rt = rgc * tl.exp(a)
            P = tl.dot(qc, Sd.to(qc.dtype))
            P3 = tl.reshape(P, [BT, BG, BV])
            o_inter = tl.sum(P3 * rt[:, :, None], axis=1)
            tl.atomic_add(outa_ptr + b*soa_b + sb*soa_n + rows[:, None]*soa_l + offs_v[None, :]*soa_v,
                          o_inter, mask=rmask[:, None] & vmask[None, :])
            kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                         mask=rmask[:, None] & dmask[None, :], other=0.0)
            vc = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                         mask=rmask[:, None] & vmask[None, :], other=0.0)
            wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                          mask=rmask[:, None] & cmask[None, :], other=0.0)
            Lam = tl.sum(tl.where(offs_t[:, None] == (BT - 1), a, 0.0), axis=0)
            w_end = wgc * tl.exp(Lam[None, :] - a)
            WV = tl.reshape(w_end[:, :, None] * vc[:, None, :], [BT, BG * BV])
            decvec = tl.reshape(tl.exp(Lam)[:, None] * tl.full([BG, BV], 1.0, tl.float32), [BG * BV])
            Sd = decvec[None, :] * Sd + tl.dot(tl.trans(kc), WV.to(kc.dtype))


def _gla_fwd(q, k, v, wg, rg, ld, chunk, BG, BK=64):
    """GLA Triton numerator-only forward: [BH,L,dv] at BV=next_pow2(dv); the kappa/per_state caller
    forms the global den from the GLA den pre-pass. D-tiled (intra d-loops the gram; inter carries
    an Sd[BK,*] slice)."""
    B, L, dqk = q.shape
    dv = v.shape[-1]
    nc = wg.shape[-1]
    ld = ld.clamp(min=_GLA_FLOOR).contiguous()
    BV = max(16, triton.next_power_of_2(dv))
    BK = min(BK, max(16, triton.next_power_of_2(dqk)))   # exact-width blocks for dqk<=64; tile beyond
    ND = triton.cdiv(dqk, BK)
    NB = triton.cdiv(nc, BG)
    NCH = triton.cdiv(L, chunk)
    q, k, v, wg, rg = [x.contiguous() for x in (q, k, v, wg, rg)]
    # separate intra/inter buffers → each kernel autotunable (see _fwd_tiled).
    out_intra = torch.zeros(B, NB, L, BV, device=q.device, dtype=torch.float32)
    out_inter = torch.zeros_like(out_intra)
    so = (out_intra.stride(0), out_intra.stride(1), out_intra.stride(2), out_intra.stride(3))
    base = (q.stride(0), q.stride(1), q.stride(2), v.stride(0), v.stride(1), v.stride(2),
            wg.stride(0), wg.stride(1), wg.stride(2))
    _rola_gla_fwd_intra[(B, NB, NCH)](q, k, v, wg, rg, ld, out_intra, L, dqk, dv, nc, *base, *so,
                                      BT=chunk, BK=BK, BV=BV, BG=BG, ND=ND)
    _rola_gla_fwd_inter[(B, NB, ND)](q, k, v, wg, rg, ld, out_inter, L, dqk, dv, nc, *base, *so,
                                     BT=chunk, BK=BK, BVF=BV, BG=BG, NCH=NCH)
    return (out_intra + out_inter)[..., :dv].sum(1)


class _RoLAGLAFn(torch.autograd.Function):
    """RoLA-GLA (scalar per-state decay) un-normalized routed readout, on folded [BH,L,*] tensors.
    Numerator-only ([.,dv]); the global denominator is the caller's separate per-state pre-pass."""
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, wg, rg, ld, chunk, BG):
        Oa = _gla_fwd(q, k, v, wg, rg, ld, chunk=chunk, BG=BG)
        ctx.save_for_backward(q, k, v, wg, rg, ld)
        ctx.BG = BG
        return Oa.to(q.dtype)

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dO):
        q, k, v, wg, rg, ld = ctx.saved_tensors
        g = dO
        dq, dk, dvv, dwg, drg, dld = _bwd_split_gla(
            q, k, v, wg, rg, ld, g, chunk=_CHUNK)
        def cast(t): return t.to(q.dtype)
        # forward args order: q, k, v, wg, rg, ld, chunk, BG
        return cast(dq), cast(dk), cast(dvv), cast(dwg), cast(drg), cast(dld), None, None


@input_guard
def rola_gla_triton(q, k, v, r, w, ld, chunk=None, BG=16):
    """Un-normalized routed GLA readout via Triton. q,k:[BH,L,K] v:[BH,L,V] r,w,ld:[BH,L,nc]
    (r=read gate, w=write gate, ld=per-state log-decay). Differentiable. Numerator-only (BV=next_pow2
    (dv)) — the kappa/per_state caller forms the global den from the per-state den pre-pass."""
    chunk = _CHUNK if chunk is None else min(chunk, _CHUNK)
    return _RoLAGLAFn.apply(q, k, v, w, r, ld, chunk, BG)


# ============================================================================
# Phase F — chunk-PARALLEL backward.
#
# The fused backward kernels serialize all gram work behind a sequential chunk scan (grid (BH,NB) —
# occupancy cliff at low nc). Restructure FLA-style:
#   K1 _scan_S    (sequential, tiny): store S_before[t] for all chunks.
#   K2 _scan_dS   (sequential, tiny): store dS_after[t] (the pre-update value used at chunk t).
#   K3a/K3b       (PARALLEL over (BH, NB, NCH)): per-chunk grad kernels with the SAME working set as
#                 the fused pair (one state tile each) — a combined single K3 was tried and REGRESSED
#                 (both state tiles + both intermediate sets live ⇒ register spill).
# Per-chunk math copied verbatim from the verified fused kernels with carries replaced by loads.
# tl.dot needs K>=16 ⇒ BG>=16.
# ============================================================================


@triton.autotune(configs=_SCAN_CFGS, key=_AT_KEY,
                 prune_configs_by={'early_config_prune': _prune_bv}, **autotune_cache_kwargs)
@triton.jit
def _scan_S(k_ptr, v_ptr, wg_ptr, ld_ptr, Sb_ptr, L, dqk, dv: tl.constexpr, nc,
            sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c,
            ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
            USE_G: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr, BV: tl.constexpr,
            BVF: tl.constexpr, BG: tl.constexpr, NCH: tl.constexpr, KSNAP: tl.constexpr):
    # Value-OUTER over cdiv(dv,BV) blocks: each block carries its own [BD, BG*BV] state slice, bounding
    # smem+regs by BV (the autotuned value-tile). ND_V==1 (BV==BVF, the fits-everywhere case the
    # autotuner picks on A100 / at d_v<=32) runs ONE full scan == the un-tiled kernel byte-for-byte.
    # Snapshot is full-width BVF (state-major g*BVF+v); a value-block writes the strided e-slice.
    # SNAPSHOT-GRANULARITY: store the boundary state only at coarse anchors (chunk t with t%KSNAP==0,
    # into slot t//KSNAP) → 1/KSNAP the snapshot HBM. The grad kernel reads the nearest coarse anchor at
    # or below t and forward-recomputes chunks [(t//KSNAP)*KSNAP, t) to rebuild the exact boundary at t.
    b = tl.program_id(0)
    sb = tl.program_id(1)
    d0 = tl.program_id(2)                       # feature-block: this program owns Sflat rows [d0*BD:]
    ND_V = (dv + BV - 1) // BV
    offs_t = tl.arange(0, BT)
    offs_d = d0 * BD + tl.arange(0, BD)
    offs_g = tl.arange(0, BG)
    offs_c = sb * BG + tl.arange(0, BG)
    dmask = offs_d < dqk
    cmask = offs_c < nc
    for vb in range(ND_V):
        offs_v = vb * BV + tl.arange(0, BV)
        vmask = offs_v < dv
        offs_e = tl.reshape(offs_g[:, None] * BVF + offs_v[None, :], [BG * BV])   # strided into BVF snapshot
        Sflat = tl.zeros([BD, BG * BV], dtype=tl.float32)
        for t in range(NCH):
            rows = t * BT + offs_t
            rmask = rows < L
            kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]
                         * sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            vc = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                         mask=rmask[:, None] & vmask[None, :], other=0.0).to(tl.float32)
            wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]
                          * sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
            if t % KSNAP == 0:
                tl.store(Sb_ptr + b*ssb_b + sb*ssb_n + (t // KSNAP)*ssb_t
                         + offs_d[:, None]*ssb_d + offs_e[None, :]*ssb_e, Sflat, mask=dmask[:, None])
            if USE_G:
                ldc = tl.load(ld_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]
                              * sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
                a = tl.cumsum(ldc, axis=0)
                Lam = tl.sum(tl.where(offs_t[:, None] == (BT - 1), a, 0.0), axis=0)
                w_end = wgc * tl.exp(Lam[None, :] - a)
                WV = tl.reshape(w_end[:, :, None] * vc[:, None, :], [BT, BG * BV])
                decvec = tl.reshape(tl.exp(Lam)[:, None] * tl.full([BG, BV], 1.0, tl.float32), [BG * BV])
                Sflat = decvec[None, :] * Sflat + tl.dot(tl.trans(kc), WV.to(kc.dtype))
            else:
                WV = tl.reshape(wgc[:, :, None] * vc[:, None, :], [BT, BG * BV])
                Sflat += tl.dot(tl.trans(kc), WV.to(kc.dtype))


@triton.autotune(configs=_SCAN_CFGS, key=_AT_KEY,
                 prune_configs_by={'early_config_prune': _prune_bv}, **autotune_cache_kwargs)
@triton.jit
def _scan_dS(q_ptr, rg_ptr, ld_ptr, g_ptr, dSa_ptr, L, dqk, dv: tl.constexpr, nc,
             sq_b, sq_l, sq_d, sg_b, sg_l, sg_c, sgr_b, sgr_l, sgr_d,
             ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
             USE_G: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr, BV: tl.constexpr,
             BVF: tl.constexpr, BG: tl.constexpr, NCH: tl.constexpr, KSNAP: tl.constexpr):
    # value-OUTER (mirror of _scan_S); ND_V==1 == the un-tiled reverse scan byte-for-byte.
    # SNAPSHOT-GRANULARITY (reverse): dSa[t] = Σ_{t'>t} contrib(t'). Anchor at the TOP of each coarse
    # block (chunk t with t%KSNAP==KSNAP-1, or the final chunk) into slot t//KSNAP — that holds Σ_{t'>top} = the
    # state entering the block from above. The grad kernel reads its block's anchor and reverse-
    # recomputes chunks (t, block_top] to rebuild the exact dSa at t. NSNAP = ceil(NCH/KSNAP).
    b = tl.program_id(0)
    sb = tl.program_id(1)
    d0 = tl.program_id(2)                       # feature-block: this program owns dS rows [d0*BD:]
    ND_V = (dv + BV - 1) // BV
    offs_t = tl.arange(0, BT)
    offs_d = d0 * BD + tl.arange(0, BD)
    offs_g = tl.arange(0, BG)
    offs_c = sb * BG + tl.arange(0, BG)
    dmask = offs_d < dqk
    cmask = offs_c < nc
    for vb in range(ND_V):
        offs_v = vb * BV + tl.arange(0, BV)
        vmask = offs_v < dv
        offs_e = tl.reshape(offs_g[:, None] * BVF + offs_v[None, :], [BG * BV])
        dS = tl.zeros([BD, BG * BV], dtype=tl.float32)
        for ti in range(NCH):
            t = NCH - 1 - ti
            rows = t * BT + offs_t
            rmask = rows < L
            qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]
                         * sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            rgc = tl.load(rg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]
                          * sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
            gc = tl.load(g_ptr + b*sgr_b + rows[:, None]*sgr_l + offs_v[None, :]
                         * sgr_d, mask=rmask[:, None] & vmask[None, :], other=0.0).to(tl.float32)
            if (t % KSNAP == KSNAP - 1) or (t == NCH - 1):
                tl.store(dSa_ptr + b*ssb_b + sb*ssb_n + (t // KSNAP)*ssb_t
                         + offs_d[:, None]*ssb_d + offs_e[None, :]*ssb_e, dS, mask=dmask[:, None])
            if USE_G:
                ldc = tl.load(ld_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]
                              * sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
                a = tl.cumsum(ldc, axis=0)
                rt = rgc * tl.exp(a)
                Lam = tl.sum(tl.where(offs_t[:, None] == (BT - 1), a, 0.0), axis=0)
                rt_g = tl.reshape(rt[:, :, None] * gc[:, None, :], [BT, BG * BV])
                decvec = tl.reshape(tl.exp(Lam)[:, None] * tl.full([BG, BV], 1.0, tl.float32), [BG * BV])
                dS = decvec[None, :] * dS + tl.dot(tl.trans(qc), rt_g.to(qc.dtype))
            else:
                rg_g = tl.reshape(rgc[:, :, None] * gc[:, None, :], [BT, BG * BV])
                dS += tl.dot(tl.trans(qc), rg_g.to(qc.dtype))


# ---------------------------------------------------------------------------------------------------
# Snapshot-GRANULARITY sub-scan recompute (device helpers, inlined by Triton). Each reconstructs the
# exact [BD, BG*BV] boundary state at chunk t for ONE feature-block (offs_d) and value-block
# (offs_v/offs_e), starting from the nearest COARSE snapshot and replaying the ≤KSNAP-1 intervening chunks
# — the substep working set is exactly [BD, BG*BV], unchanged from the snapshot path, so it fits SMEM
# at ANY dqk (the whole point: full register recompute carried [dqk_pad,*] and OOM'd; this does not).
# ---------------------------------------------------------------------------------------------------
@triton.jit
def _recompute_S_rla(k_ptr, v_ptr, wg_ptr, Sb_ptr, t, b, sb, L,
                     sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c,
                     ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
                     offs_t, offs_d, dmask, offs_c, cmask, offs_v, vmask, offs_e,
                     dqk, nc, BT: tl.constexpr, BD: tl.constexpr, BG: tl.constexpr,
                     BV: tl.constexpr, KSNAP: tl.constexpr):
    # forward anchor = (t//KSNAP)*KSNAP; replay chunks [t0, t) adding Kᵀ·WV. The loop trip count is the
    # CONSTEXPR KSNAP (not the runtime t-t0) so Triton unrolls it like the scans — a runtime-bounded
    # loop blows up codegen. Chunks tt>=t are MASKED off (rmask &= tt < t → zero contribution).
    t0 = (t // KSNAP) * KSNAP
    Sflat = tl.load(Sb_ptr + b*ssb_b + sb*ssb_n + (t // KSNAP)*ssb_t
                    + offs_d[:, None]*ssb_d + offs_e[None, :]*ssb_e, mask=dmask[:, None], other=0.0)
    for i in range(KSNAP):
        tt = t0 + i
        rows = tt * BT + offs_t
        rmask = (rows < L) & (tt < t)
        kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
        vc = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                     mask=rmask[:, None] & vmask[None, :], other=0.0).to(tl.float32)
        wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                      mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        WV = tl.reshape(wgc[:, :, None] * vc[:, None, :], [BT, BG * BV])
        Sflat += tl.dot(tl.trans(kc), WV.to(kc.dtype))
    return Sflat


@triton.jit
def _recompute_dS_rla(q_ptr, rg_ptr, g_ptr, dSa_ptr, t, b, sb, L,
                      sq_b, sq_l, sq_d, sg_b, sg_l, sg_c, sgr_b, sgr_l, sgr_d,
                      ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
                      offs_t, offs_d, dmask, offs_c, cmask, offs_v, vmask, offs_e,
                      dqk, nc, NCH, BT: tl.constexpr, BD: tl.constexpr, BG: tl.constexpr,
                      BV: tl.constexpr, KSNAP: tl.constexpr):
    # reverse anchor = top of t's coarse block = min((t//KSNAP)*KSNAP+KSNAP-1, NCH-1); the snapshot at
    # slot t//KSNAP holds Σ_{t'>top}. Replay chunks (t, top] in REVERSE adding Qᵀ·rg_g → Σ_{t'>t}. Trip
    # count is the CONSTEXPR KSNAP (unrolled); chunks tt<=t OR tt>top are MASKED off (zero contribution).
    top = (t // KSNAP) * KSNAP + (KSNAP - 1)
    if top > NCH - 1:
        top = NCH - 1
    dS = tl.load(dSa_ptr + b*ssb_b + sb*ssb_n + (t // KSNAP)*ssb_t
                 + offs_d[:, None]*ssb_d + offs_e[None, :]*ssb_e, mask=dmask[:, None], other=0.0)
    for i in range(KSNAP):
        tt = top - i
        rows = tt * BT + offs_t
        rmask = (rows < L) & (tt > t)
        qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
        rgc = tl.load(rg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                      mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        gc = tl.load(g_ptr + b*sgr_b + rows[:, None]*sgr_l + offs_v[None, :]*sgr_d,
                     mask=rmask[:, None] & vmask[None, :], other=0.0).to(tl.float32)
        rg_g = tl.reshape(rgc[:, :, None] * gc[:, None, :], [BT, BG * BV])
        dS += tl.dot(tl.trans(qc), rg_g.to(qc.dtype))
    return dS


# --- GLA snapshot-granularity recompute (decay-weighted; mirrors _recompute_*_rla + _scan_* USE_G) ---
# The replay must reproduce _scan_S/_scan_dS's decayed accumulation EXACTLY. For a MASKED chunk (tt>=t
# fwd / tt<=t or tt>top rev) all loads are zeroed (rmask), so the dot is 0 AND ldc→0 ⇒ a=0 ⇒ Lam=0 ⇒
# decvec=exp(0)=1: the carried state passes through identically. No special-casing needed.
@triton.jit
def _recompute_S_gla(k_ptr, v_ptr, wg_ptr, ld_ptr, Sb_ptr, t, b, sb, L,
                     sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c,
                     ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
                     offs_t, offs_d, dmask, offs_c, cmask, offs_v, vmask, offs_e,
                     dqk, nc, BT: tl.constexpr, BD: tl.constexpr, BG: tl.constexpr,
                     BV: tl.constexpr, KSNAP: tl.constexpr):
    # forward anchor = (t//KSNAP)*KSNAP (boundary state ENTERING that chunk); replay chunks [t0, t)
    # ascending applying Sflat = decvec*Sflat + Kᵀ·WV. Trip count is CONSTEXPR KSNAP (unrolled).
    t0 = (t // KSNAP) * KSNAP
    Sflat = tl.load(Sb_ptr + b*ssb_b + sb*ssb_n + (t // KSNAP)*ssb_t
                    + offs_d[:, None]*ssb_d + offs_e[None, :]*ssb_e, mask=dmask[:, None], other=0.0)
    for i in range(KSNAP):
        tt = t0 + i
        rows = tt * BT + offs_t
        rmask = (rows < L) & (tt < t)
        kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
        vc = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                     mask=rmask[:, None] & vmask[None, :], other=0.0).to(tl.float32)
        wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                      mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        ldc = tl.load(ld_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                      mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        a = tl.cumsum(ldc, axis=0)
        Lam = tl.sum(tl.where(offs_t[:, None] == (BT - 1), a, 0.0), axis=0)
        w_end = wgc * tl.exp(Lam[None, :] - a)
        WV = tl.reshape(w_end[:, :, None] * vc[:, None, :], [BT, BG * BV])
        decvec = tl.reshape(tl.exp(Lam)[:, None] * tl.full([BG, BV], 1.0, tl.float32), [BG * BV])
        Sflat = decvec[None, :] * Sflat + tl.dot(tl.trans(kc), WV.to(kc.dtype))
    return Sflat


@triton.jit
def _recompute_dS_gla(q_ptr, rg_ptr, ld_ptr, g_ptr, dSa_ptr, t, b, sb, L,
                      sq_b, sq_l, sq_d, sg_b, sg_l, sg_c, sgr_b, sgr_l, sgr_d,
                      ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
                      offs_t, offs_d, dmask, offs_c, cmask, offs_v, vmask, offs_e,
                      dqk, nc, NCH, BT: tl.constexpr, BD: tl.constexpr, BG: tl.constexpr,
                      BV: tl.constexpr, KSNAP: tl.constexpr):
    # reverse anchor = top of t's coarse block (slot holds Σ_{t'>top}); replay chunks (t, top] in REVERSE
    # applying dS = decvec*dS + Qᵀ·rt_g → Σ_{t'>t}. Trip count CONSTEXPR KSNAP; tt<=t OR tt>top MASKED.
    top = (t // KSNAP) * KSNAP + (KSNAP - 1)
    if top > NCH - 1:
        top = NCH - 1
    dS = tl.load(dSa_ptr + b*ssb_b + sb*ssb_n + (t // KSNAP)*ssb_t
                 + offs_d[:, None]*ssb_d + offs_e[None, :]*ssb_e, mask=dmask[:, None], other=0.0)
    for i in range(KSNAP):
        tt = top - i
        rows = tt * BT + offs_t
        rmask = (rows < L) & (tt > t)
        qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
        rgc = tl.load(rg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                      mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        gc = tl.load(g_ptr + b*sgr_b + rows[:, None]*sgr_l + offs_v[None, :]*sgr_d,
                     mask=rmask[:, None] & vmask[None, :], other=0.0).to(tl.float32)
        ldc = tl.load(ld_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                      mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        a = tl.cumsum(ldc, axis=0)
        rt = rgc * tl.exp(a)
        Lam = tl.sum(tl.where(offs_t[:, None] == (BT - 1), a, 0.0), axis=0)
        rt_g = tl.reshape(rt[:, :, None] * gc[:, None, :], [BT, BG * BV])
        decvec = tl.reshape(tl.exp(Lam)[:, None] * tl.full([BG, BV], 1.0, tl.float32), [BG * BV])
        dS = decvec[None, :] * dS + tl.dot(tl.trans(qc), rt_g.to(qc.dtype))
    return dS


@triton.autotune(configs=_BWD_CFGS_BV, key=_AT_KEY,
                 prune_configs_by={'early_config_prune': _prune_bwd}, **autotune_cache_kwargs)
@triton.jit
def _par_grad_rla_qr(q_ptr, k_ptr, v_ptr, wg_ptr, rg_ptr, g_ptr, Sb_ptr, dq_ptr, dr_ptr,
                     L, dqk: tl.constexpr, dv: tl.constexpr, nc,
                     sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c, sgr_b, sgr_l, sgr_d,
                     ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
                     sdq_b, sdq_n, sdq_l, sdq_d, sdr_b, sdr_l, sdr_c,
                     BT: tl.constexpr, BD: tl.constexpr, BV: tl.constexpr, BVF: tl.constexpr,
                     BG: tl.constexpr, NCH: tl.constexpr, KSNAP: tl.constexpr):
    # D-tiled (BD autotune knob) over the feature axis; V-tiled (BV autotune knob) over the value axis.
    # ND_V==1 (BV==BVF — what the autotuner picks where the full value tile fits) runs the un-tiled fused
    # body byte-for-byte. ND_V>=2 takes the value-OUTER path: P is value-contracted (built first), dq's
    # inter and dr's inter are value-summed (looped), each value-block's state slice read strided from
    # the BVF-wide snapshot. See [[branch-on-structure-not-thresholds]].
    ND = tl.cdiv(dqk, BD)
    ND_V = (dv + BV - 1) // BV
    b = tl.program_id(0)
    sb = tl.program_id(1)
    t = tl.program_id(2)
    offs_t = tl.arange(0, BT)
    offs_g = tl.arange(0, BG)
    offs_c = sb * BG + tl.arange(0, BG)
    cmask = offs_c < nc
    rows = t * BT + offs_t
    rmask = rows < L
    rgc = tl.load(rg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
    wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
    causal = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
    Rg = tl.dot(rgc, tl.trans(wgc))
    if ND_V == 1:
        offs_v = tl.arange(0, BV)
        offs_e = tl.arange(0, BG * BV)
        vmask = offs_v < dv
        v1 = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                     mask=rmask[:, None] & vmask[None, :], other=0.0).to(tl.float32)
        gc = tl.load(g_ptr + b*sgr_b + rows[:, None]*sgr_l + offs_v[None, :]
                     * sgr_d, mask=rmask[:, None] & vmask[None, :], other=0.0).to(tl.float32)
        P = tl.dot(gc, tl.trans(v1))
        coef = causal * Rg * P                                          # dq_intra coefficient [BT,BT]
        rg_g = tl.reshape(rgc[:, :, None] * gc[:, None, :], [BT, BG * BV])
        G = tl.zeros([BT, BT], dtype=tl.float32)
        QS = tl.zeros([BT, BG * BV], dtype=tl.float32)
        for d0b in range(ND):
            offs_d = d0b * BD + tl.arange(0, BD)
            dmask = offs_d < dqk
            qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                         mask=rmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                         mask=rmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            Sb = _recompute_S_rla(k_ptr, v_ptr, wg_ptr, Sb_ptr, t, b, sb, L,
                                  sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c,
                                  ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
                                  offs_t, offs_d, dmask, offs_c, cmask, offs_v, vmask, offs_e,
                                  dqk, nc, BT, BD, BG, BV, KSNAP)
            dq_d = tl.dot(coef.to(kc.dtype), kc) + tl.dot(rg_g.to(Sb.dtype), tl.trans(Sb))
            tl.store(dq_ptr + b*sdq_b + sb*sdq_n + rows[:, None]*sdq_l + offs_d[None, :]*sdq_d,
                     dq_d, mask=rmask[:, None] & dmask[None, :])
            G += tl.trans(tl.dot(kc, tl.trans(qc)))   # operand-shared tl.dot miscompile fix (qr)
            QS += tl.dot(qc, Sb.to(qc.dtype))
        dr_intra = tl.dot((causal * G * P).to(wgc.dtype), wgc)
        dr_inter = tl.sum(tl.reshape(QS, [BT, BG, BV]) * gc[:, None, :], axis=2)
        tl.store(dr_ptr + b*sdr_b + rows[:, None]*sdr_l + offs_c[None, :]*sdr_c,
                 dr_intra + dr_inter, mask=rmask[:, None] & cmask[None, :])
    else:
        # Phase 1: P (value-contracted) over value-blocks.
        P = tl.zeros([BT, BT], dtype=tl.float32)
        for vb in range(ND_V):
            offs_v = vb * BV + tl.arange(0, BV)
            vm = offs_v < dv
            v1 = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                         mask=rmask[:, None] & vm[None, :], other=0.0).to(tl.float32)
            gv = tl.load(g_ptr + b*sgr_b + rows[:, None]*sgr_l + offs_v[None, :]*sgr_d,
                         mask=rmask[:, None] & vm[None, :], other=0.0).to(tl.float32)
            P += tl.dot(gv, tl.trans(v1))
        coef = causal * Rg * P
        # Phase 2: dq = intra (coef·k) + inter (Σ_vb rg_g_vb · Sb_vbᵀ, value-summed); also accumulate G.
        G = tl.zeros([BT, BT], dtype=tl.float32)
        for d0b in range(ND):
            offs_d = d0b * BD + tl.arange(0, BD)
            dmask = offs_d < dqk
            qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                         mask=rmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                         mask=rmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            dq_d = tl.dot(coef.to(kc.dtype), kc)
            for vb in range(ND_V):
                offs_v = vb * BV + tl.arange(0, BV)
                vm = offs_v < dv
                gv = tl.load(g_ptr + b*sgr_b + rows[:, None]*sgr_l + offs_v[None, :]*sgr_d,
                             mask=rmask[:, None] & vm[None, :], other=0.0).to(tl.float32)
                rg_g = tl.reshape(rgc[:, :, None] * gv[:, None, :], [BT, BG * BV])
                offs_e = tl.reshape(offs_g[:, None] * BVF + offs_v[None, :], [BG * BV])
                Sb = _recompute_S_rla(k_ptr, v_ptr, wg_ptr, Sb_ptr, t, b, sb, L,
                                      sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c,
                                      ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
                                      offs_t, offs_d, dmask, offs_c, cmask, offs_v, vm, offs_e,
                                      dqk, nc, BT, BD, BG, BV, KSNAP)
                dq_d += tl.dot(rg_g.to(Sb.dtype), tl.trans(Sb))
            tl.store(dq_ptr + b*sdq_b + sb*sdq_n + rows[:, None]*sdq_l + offs_d[None, :]*sdq_d,
                     dq_d, mask=rmask[:, None] & dmask[None, :])
            G += tl.trans(tl.dot(kc, tl.trans(qc)))   # operand-shared tl.dot miscompile fix (qr)
        # Phase 3: dr = intra (value-summed via P) + inter (value-outer QS per block).
        dr_intra = tl.dot((causal * G * P).to(wgc.dtype), wgc)
        dr_inter = tl.zeros([BT, BG], dtype=tl.float32)
        for vb in range(ND_V):
            offs_v = vb * BV + tl.arange(0, BV)
            vm = offs_v < dv
            gv = tl.load(g_ptr + b*sgr_b + rows[:, None]*sgr_l + offs_v[None, :]*sgr_d,
                         mask=rmask[:, None] & vm[None, :], other=0.0).to(tl.float32)
            offs_e = tl.reshape(offs_g[:, None] * BVF + offs_v[None, :], [BG * BV])
            QS = tl.zeros([BT, BG * BV], dtype=tl.float32)
            for d0b in range(ND):
                offs_d = d0b * BD + tl.arange(0, BD)
                dmask = offs_d < dqk
                qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                             mask=rmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
                Sb = _recompute_S_rla(k_ptr, v_ptr, wg_ptr, Sb_ptr, t, b, sb, L,
                                      sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c,
                                      ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
                                      offs_t, offs_d, dmask, offs_c, cmask, offs_v, vm, offs_e,
                                      dqk, nc, BT, BD, BG, BV, KSNAP)
                QS += tl.dot(qc, Sb.to(qc.dtype))
            dr_inter += tl.sum(tl.reshape(QS, [BT, BG, BV]) * gv[:, None, :], axis=2)
        tl.store(dr_ptr + b*sdr_b + rows[:, None]*sdr_l + offs_c[None, :]*sdr_c,
                 dr_intra + dr_inter, mask=rmask[:, None] & cmask[None, :])


@triton.autotune(configs=_BWD_CFGS_BV, key=_AT_KEY,
                 prune_configs_by={'early_config_prune': _prune_bwd}, **autotune_cache_kwargs)
@triton.jit
def _par_grad_rla_kwv(q_ptr, k_ptr, v_ptr, wg_ptr, rg_ptr, g_ptr, dSa_ptr, dk_ptr, dw_ptr, dv_ptr,
                      L, dqk: tl.constexpr, dv: tl.constexpr, nc,
                      sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c, sgr_b, sgr_l, sgr_d,
                      ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
                      sdk_b, sdk_n, sdk_l, sdk_d, sdw_b, sdw_l, sdw_c, sdv_b, sdv_n, sdv_l, sdv_d,
                      BT: tl.constexpr, BD: tl.constexpr, BV: tl.constexpr, BVF: tl.constexpr,
                      BG: tl.constexpr, NCH: tl.constexpr, KSNAP: tl.constexpr):
    # D-tiled (BD) over features, V-tiled (BV) over the value axis. ND_V==1 runs the un-tiled fused body
    # byte-for-byte; ND_V>=2 is value-OUTER: P value-contracted, dk/dw value-summed (looped), dv
    # value-indexed (per block), KS rebuilt per value-block. See [[branch-on-structure-not-thresholds]].
    ND = tl.cdiv(dqk, BD)
    ND_V = (dv + BV - 1) // BV
    b = tl.program_id(0)
    sb = tl.program_id(1)
    t = tl.program_id(2)
    offs_t = tl.arange(0, BT)
    offs_g = tl.arange(0, BG)
    offs_c = sb * BG + tl.arange(0, BG)
    cmask = offs_c < nc
    rows = t * BT + offs_t
    rmask = rows < L
    rgc = tl.load(rg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
    wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
    causal = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
    Rg = tl.dot(rgc, tl.trans(wgc))
    if ND_V == 1:
        offs_v = tl.arange(0, BV)
        offs_e = tl.arange(0, BG * BV)
        vmask = offs_v < dv
        v1 = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                     mask=rmask[:, None] & vmask[None, :], other=0.0).to(tl.float32)
        gc = tl.load(g_ptr + b*sgr_b + rows[:, None]*sgr_l + offs_v[None, :]
                     * sgr_d, mask=rmask[:, None] & vmask[None, :], other=0.0).to(tl.float32)
        P = tl.dot(gc, tl.trans(v1))
        A2 = Rg * P * causal                                            # dk_intra coef [BT,BT]
        wg_v1 = tl.reshape(wgc[:, :, None] * v1[:, None, :], [BT, BG * BV])
        G = tl.zeros([BT, BT], dtype=tl.float32)
        KS = tl.zeros([BT, BG * BV], dtype=tl.float32)
        for d0b in range(ND):
            offs_d = d0b * BD + tl.arange(0, BD)
            dmask = offs_d < dqk
            qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                         mask=rmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                         mask=rmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            dSa = _recompute_dS_rla(q_ptr, rg_ptr, g_ptr, dSa_ptr, t, b, sb, L,
                                    sq_b, sq_l, sq_d, sg_b, sg_l, sg_c, sgr_b, sgr_l, sgr_d,
                                    ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
                                    offs_t, offs_d, dmask, offs_c, cmask, offs_v, vmask, offs_e,
                                    dqk, nc, NCH, BT, BD, BG, BV, KSNAP)
            dk_d = tl.dot(tl.trans(A2).to(qc.dtype), qc) + tl.dot(wg_v1.to(dSa.dtype), tl.trans(dSa))
            tl.store(dk_ptr + b*sdk_b + sb*sdk_n + rows[:, None]*sdk_l + offs_d[None, :]*sdk_d,
                     dk_d, mask=rmask[:, None] & dmask[None, :])
            G += tl.dot(qc, tl.trans(kc))
            KS += tl.dot(kc, dSa.to(kc.dtype))
        A = G * Rg * causal
        B2 = G * P * causal
        dw_intra = tl.dot(tl.trans(B2).to(rgc.dtype), rgc)
        dv_intra = tl.dot(tl.trans(A).to(gc.dtype), gc)
        KS3 = tl.reshape(KS, [BT, BG, BV])
        dw_wr = tl.sum(KS3 * v1[:, None, :], axis=2)
        dv_wr = tl.sum(wgc[:, :, None] * KS3, axis=1)
        tl.store(dw_ptr + b*sdw_b + rows[:, None]*sdw_l + offs_c[None, :]*sdw_c,
                 dw_intra + dw_wr, mask=rmask[:, None] & cmask[None, :])
        tl.store(dv_ptr + b*sdv_b + sb*sdv_n + rows[:, None]*sdv_l + offs_v[None, :]*sdv_d,
                 dv_intra + dv_wr, mask=rmask[:, None] & (offs_v[None, :] < dv))
    else:
        # Phase 1: P (value-contracted).
        P = tl.zeros([BT, BT], dtype=tl.float32)
        for vb in range(ND_V):
            offs_v = vb * BV + tl.arange(0, BV)
            vm = offs_v < dv
            v1 = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                         mask=rmask[:, None] & vm[None, :], other=0.0).to(tl.float32)
            gv = tl.load(g_ptr + b*sgr_b + rows[:, None]*sgr_l + offs_v[None, :]*sgr_d,
                         mask=rmask[:, None] & vm[None, :], other=0.0).to(tl.float32)
            P += tl.dot(gv, tl.trans(v1))
        A2 = Rg * P * causal
        # Phase 2: dk = intra (A2ᵀ·q) + inter (Σ_vb wg_v1_vb · dSa_vbᵀ); accumulate G.
        G = tl.zeros([BT, BT], dtype=tl.float32)
        for d0b in range(ND):
            offs_d = d0b * BD + tl.arange(0, BD)
            dmask = offs_d < dqk
            qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                         mask=rmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                         mask=rmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            dk_d = tl.dot(tl.trans(A2).to(qc.dtype), qc)
            for vb in range(ND_V):
                offs_v = vb * BV + tl.arange(0, BV)
                vm = offs_v < dv
                v1 = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                             mask=rmask[:, None] & vm[None, :], other=0.0).to(tl.float32)
                wg_v1 = tl.reshape(wgc[:, :, None] * v1[:, None, :], [BT, BG * BV])
                offs_e = tl.reshape(offs_g[:, None] * BVF + offs_v[None, :], [BG * BV])
                dSa = _recompute_dS_rla(q_ptr, rg_ptr, g_ptr, dSa_ptr, t, b, sb, L,
                                        sq_b, sq_l, sq_d, sg_b, sg_l, sg_c, sgr_b, sgr_l, sgr_d,
                                        ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
                                        offs_t, offs_d, dmask, offs_c, cmask, offs_v, vm, offs_e,
                                        dqk, nc, NCH, BT, BD, BG, BV, KSNAP)
                dk_d += tl.dot(wg_v1.to(dSa.dtype), tl.trans(dSa))
            tl.store(dk_ptr + b*sdk_b + sb*sdk_n + rows[:, None]*sdk_l + offs_d[None, :]*sdk_d,
                     dk_d, mask=rmask[:, None] & dmask[None, :])
            G += tl.dot(qc, tl.trans(kc))
        A = G * Rg * causal
        B2 = G * P * causal
        dw_intra = tl.dot(tl.trans(B2).to(rgc.dtype), rgc)
        dw_wr = tl.zeros([BT, BG], dtype=tl.float32)
        # Phase 3: per value-block rebuild KS_vb -> dw_wr (value-summed) + dv (value-indexed).
        for vb in range(ND_V):
            offs_v = vb * BV + tl.arange(0, BV)
            vm = offs_v < dv
            v1 = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                         mask=rmask[:, None] & vm[None, :], other=0.0).to(tl.float32)
            gv = tl.load(g_ptr + b*sgr_b + rows[:, None]*sgr_l + offs_v[None, :]*sgr_d,
                         mask=rmask[:, None] & vm[None, :], other=0.0).to(tl.float32)
            offs_e = tl.reshape(offs_g[:, None] * BVF + offs_v[None, :], [BG * BV])
            KS = tl.zeros([BT, BG * BV], dtype=tl.float32)
            for d0b in range(ND):
                offs_d = d0b * BD + tl.arange(0, BD)
                dmask = offs_d < dqk
                kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                             mask=rmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
                dSa = _recompute_dS_rla(q_ptr, rg_ptr, g_ptr, dSa_ptr, t, b, sb, L,
                                        sq_b, sq_l, sq_d, sg_b, sg_l, sg_c, sgr_b, sgr_l, sgr_d,
                                        ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
                                        offs_t, offs_d, dmask, offs_c, cmask, offs_v, vm, offs_e,
                                        dqk, nc, NCH, BT, BD, BG, BV, KSNAP)
                KS += tl.dot(kc, dSa.to(kc.dtype))
            KS3 = tl.reshape(KS, [BT, BG, BV])
            dw_wr += tl.sum(KS3 * v1[:, None, :], axis=2)
            dv_intra = tl.dot(tl.trans(A).to(gv.dtype), gv)
            dv_wr = tl.sum(wgc[:, :, None] * KS3, axis=1)
            tl.store(dv_ptr + b*sdv_b + sb*sdv_n + rows[:, None]*sdv_l + offs_v[None, :]*sdv_d,
                     dv_intra + dv_wr, mask=rmask[:, None] & vm[None, :])
        tl.store(dw_ptr + b*sdw_b + rows[:, None]*sdw_l + offs_c[None, :]*sdw_c,
                 dw_intra + dw_wr, mask=rmask[:, None] & cmask[None, :])


@triton.autotune(configs=_BWD_CFGS_GLA, key=_AT_KEY,   # slim recompute grid (stages=1, warps 2/4) — see _BWD_CFGS_GLA
                 prune_configs_by={'early_config_prune': _prune_bwd}, **autotune_cache_kwargs)
@triton.jit
def _par_grad_gla_qr(q_ptr, k_ptr, v_ptr, wg_ptr, rg_ptr, ld_ptr, g_ptr, Sb_ptr,
                     dq_ptr, drg_ptr, dart_ptr,
                     L, dqk: tl.constexpr, dv: tl.constexpr, nc,
                     sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c, sgr_b, sgr_l, sgr_d,
                     ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
                     sdq_b, sdq_n, sdq_l, sdq_d, sdr_b, sdr_l, sdr_c, sda_b, sda_l, sda_c,
                     BT: tl.constexpr, BD: tl.constexpr, BV: tl.constexpr, BVF: tl.constexpr,
                     BG: tl.constexpr, NCH: tl.constexpr, KSNAP: tl.constexpr):
    # GLA mirror of _par_grad_rla_qr: decayed gates rt=rg·eᵃ, wt=wg·e⁻ᵃ, D=rt·wtᵀ. ND_V==1 == un-tiled
    # fused body; ND_V>=2 value-OUTER (P value-contracted; dq/drt inter value-summed, looped).
    ND = tl.cdiv(dqk, BD)
    ND_V = (dv + BV - 1) // BV
    b = tl.program_id(0)
    sb = tl.program_id(1)
    t = tl.program_id(2)
    offs_t = tl.arange(0, BT)
    offs_g = tl.arange(0, BG)
    offs_c = sb * BG + tl.arange(0, BG)
    cmask = offs_c < nc
    rows = t * BT + offs_t
    rmask = rows < L
    rgc = tl.load(rg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
    wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
    ldc = tl.load(ld_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
    a = tl.cumsum(ldc, axis=0)
    ea = tl.exp(a)
    rt = rgc * ea
    wt = wgc * tl.exp(-a)
    caus = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
    D = tl.dot(rt, tl.trans(wt))
    if ND_V == 1:
        offs_v = tl.arange(0, BV)
        offs_e = tl.arange(0, BG * BV)
        vmask = offs_v < dv
        v1 = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d, mask=rmask[:, None] & vmask[None, :], other=0.0).to(tl.float32)
        gc = tl.load(g_ptr + b*sgr_b + rows[:, None]*sgr_l + offs_v[None, :]*sgr_d, mask=rmask[:, None] & vmask[None, :], other=0.0).to(tl.float32)
        P = tl.dot(gc, tl.trans(v1))
        dG = P * D * caus
        rt_g = tl.reshape(rt[:, :, None] * gc[:, None, :], [BT, BG * BV])
        G = tl.zeros([BT, BT], dtype=tl.float32)
        QS = tl.zeros([BT, BG * BV], dtype=tl.float32)
        for d0b in range(ND):
            offs_d = d0b * BD + tl.arange(0, BD)
            dmask = offs_d < dqk
            qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            Sb = _recompute_S_gla(k_ptr, v_ptr, wg_ptr, ld_ptr, Sb_ptr, t, b, sb, L,
                                  sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c,
                                  ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
                                  offs_t, offs_d, dmask, offs_c, cmask, offs_v, vmask, offs_e,
                                  dqk, nc, BT, BD, BG, BV, KSNAP)
            dq_intra = tl.dot(dG.to(kc.dtype), kc)
            dq_inter = tl.dot(rt_g.to(Sb.dtype), tl.trans(Sb))
            tl.store(dq_ptr + b*sdq_b + sb*sdq_n + rows[:, None]*sdq_l + offs_d[None, :]*sdq_d,
                     dq_intra + dq_inter, mask=rmask[:, None] & dmask[None, :])
            G += tl.dot(qc, tl.trans(kc))
            QS += tl.dot(qc, Sb.to(qc.dtype))
        dD = P * G * caus
        drt_intra = tl.dot(dD.to(wt.dtype), wt)
        drt_inter = tl.sum(tl.reshape(QS, [BT, BG, BV]) * gc[:, None, :], axis=2)
        drt = drt_intra + drt_inter
    else:
        P = tl.zeros([BT, BT], dtype=tl.float32)
        for vb in range(ND_V):
            offs_v = vb * BV + tl.arange(0, BV)
            vm = offs_v < dv
            v1 = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d, mask=rmask[:, None] & vm[None, :], other=0.0).to(tl.float32)
            gv = tl.load(g_ptr + b*sgr_b + rows[:, None]*sgr_l + offs_v[None, :]*sgr_d, mask=rmask[:, None] & vm[None, :], other=0.0).to(tl.float32)
            P += tl.dot(gv, tl.trans(v1))
        dG = P * D * caus
        G = tl.zeros([BT, BT], dtype=tl.float32)
        for d0b in range(ND):
            offs_d = d0b * BD + tl.arange(0, BD)
            dmask = offs_d < dqk
            qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            dq_d = tl.dot(dG.to(kc.dtype), kc)
            for vb in range(ND_V):
                offs_v = vb * BV + tl.arange(0, BV)
                vm = offs_v < dv
                gv = tl.load(g_ptr + b*sgr_b + rows[:, None]*sgr_l + offs_v[None, :]*sgr_d, mask=rmask[:, None] & vm[None, :], other=0.0).to(tl.float32)
                rt_g = tl.reshape(rt[:, :, None] * gv[:, None, :], [BT, BG * BV])
                offs_e = tl.reshape(offs_g[:, None] * BVF + offs_v[None, :], [BG * BV])
                Sb = _recompute_S_gla(k_ptr, v_ptr, wg_ptr, ld_ptr, Sb_ptr, t, b, sb, L,
                                      sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c,
                                      ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
                                      offs_t, offs_d, dmask, offs_c, cmask, offs_v, vm, offs_e,
                                      dqk, nc, BT, BD, BG, BV, KSNAP)
                dq_d += tl.dot(rt_g.to(Sb.dtype), tl.trans(Sb))
            tl.store(dq_ptr + b*sdq_b + sb*sdq_n + rows[:, None]*sdq_l + offs_d[None, :]*sdq_d,
                     dq_d, mask=rmask[:, None] & dmask[None, :])
            G += tl.dot(qc, tl.trans(kc))
        dD = P * G * caus
        drt_intra = tl.dot(dD.to(wt.dtype), wt)
        drt_inter = tl.zeros([BT, BG], dtype=tl.float32)
        for vb in range(ND_V):
            offs_v = vb * BV + tl.arange(0, BV)
            vm = offs_v < dv
            gv = tl.load(g_ptr + b*sgr_b + rows[:, None]*sgr_l + offs_v[None, :]*sgr_d, mask=rmask[:, None] & vm[None, :], other=0.0).to(tl.float32)
            offs_e = tl.reshape(offs_g[:, None] * BVF + offs_v[None, :], [BG * BV])
            QS = tl.zeros([BT, BG * BV], dtype=tl.float32)
            for d0b in range(ND):
                offs_d = d0b * BD + tl.arange(0, BD)
                dmask = offs_d < dqk
                qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
                Sb = _recompute_S_gla(k_ptr, v_ptr, wg_ptr, ld_ptr, Sb_ptr, t, b, sb, L,
                                      sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c,
                                      ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
                                      offs_t, offs_d, dmask, offs_c, cmask, offs_v, vm, offs_e,
                                      dqk, nc, BT, BD, BG, BV, KSNAP)
                QS += tl.dot(qc, Sb.to(qc.dtype))
            drt_inter += tl.sum(tl.reshape(QS, [BT, BG, BV]) * gv[:, None, :], axis=2)
        drt = drt_intra + drt_inter
    tl.store(drg_ptr + b*sdr_b + rows[:, None]*sdr_l + offs_c[None, :]*sdr_c, drt * ea, mask=rmask[:, None] & cmask[None, :])
    tl.store(dart_ptr + b*sda_b + rows[:, None]*sda_l + offs_c[None, :]*sda_c, drt * rt, mask=rmask[:, None] & cmask[None, :])


@triton.autotune(configs=_BWD_CFGS_GLA, key=_AT_KEY,   # slim recompute grid (stages=1, warps 2/4) — see _BWD_CFGS_GLA
                 prune_configs_by={'early_config_prune': _prune_bwd}, **autotune_cache_kwargs)
@triton.jit
def _par_grad_gla_kwv(q_ptr, k_ptr, v_ptr, wg_ptr, rg_ptr, ld_ptr, g_ptr, Sb_ptr, dSa_ptr, dart_ptr,
                      dk_ptr, dwg_ptr, dv_ptr, dld_ptr,
                      L, dqk: tl.constexpr, dv: tl.constexpr, nc,
                      sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c, sgr_b, sgr_l, sgr_d,
                      ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
                      sdk_b, sdk_n, sdk_l, sdk_d, sdw_b, sdw_l, sdw_c, sdv_b, sdv_n, sdv_l, sdv_d,
                      sda_b, sda_l, sda_c,
                      BT: tl.constexpr, BD: tl.constexpr, BV: tl.constexpr, BVF: tl.constexpr,
                      BG: tl.constexpr, NCH: tl.constexpr, KSNAP: tl.constexpr):
    # GLA mirror of _par_grad_rla_kwv + the dLam reverse-cumsum. Both ND_V branches produce dw_end[BT,BG]
    # (value-summed), ZdZ[BG] = Σ_{d,v}(Sb∘dSa), and dwt[BT,BG]; the shared tail folds dLam into da and
    # reverse-cumsums to dld. ND_V==1 == the un-tiled fused body; ND_V>=2 is value-OUTER.
    ND = tl.cdiv(dqk, BD)
    ND_V = (dv + BV - 1) // BV
    b = tl.program_id(0)
    sb = tl.program_id(1)
    t = tl.program_id(2)
    offs_t = tl.arange(0, BT)
    offs_g = tl.arange(0, BG)
    offs_c = sb * BG + tl.arange(0, BG)
    cmask = offs_c < nc
    rows = t * BT + offs_t
    rmask = rows < L
    rgc = tl.load(rg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
    wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
    ldc = tl.load(ld_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
    a = tl.cumsum(ldc, axis=0)
    ena = tl.exp(-a)
    rt = rgc * tl.exp(a)
    wt = wgc * ena
    Lam = tl.sum(tl.where(offs_t[:, None] == (BT - 1), a, 0.0), axis=0)
    w_end = wgc * tl.exp(Lam[None, :] - a)
    caus = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
    D = tl.dot(rt, tl.trans(wt))
    if ND_V == 1:
        offs_v = tl.arange(0, BV)
        offs_e = tl.arange(0, BG * BV)
        vmask = offs_v < dv
        v1 = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d, mask=rmask[:, None] & vmask[None, :], other=0.0).to(tl.float32)
        gc = tl.load(g_ptr + b*sgr_b + rows[:, None]*sgr_l + offs_v[None, :]*sgr_d, mask=rmask[:, None] & vmask[None, :], other=0.0).to(tl.float32)
        P = tl.dot(gc, tl.trans(v1))
        dG = P * D * caus
        wv1 = tl.reshape(w_end[:, :, None] * v1[:, None, :], [BT, BG * BV])
        G = tl.zeros([BT, BT], dtype=tl.float32)
        KS = tl.zeros([BT, BG * BV], dtype=tl.float32)
        ZdZ = tl.zeros([BG], dtype=tl.float32)
        for d0b in range(ND):
            offs_d = d0b * BD + tl.arange(0, BD)
            dmask = offs_d < dqk
            qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            Sb = _recompute_S_gla(k_ptr, v_ptr, wg_ptr, ld_ptr, Sb_ptr, t, b, sb, L,
                                  sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c,
                                  ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
                                  offs_t, offs_d, dmask, offs_c, cmask, offs_v, vmask, offs_e,
                                  dqk, nc, BT, BD, BG, BV, KSNAP)
            dSa = _recompute_dS_gla(q_ptr, rg_ptr, ld_ptr, g_ptr, dSa_ptr, t, b, sb, L,
                                    sq_b, sq_l, sq_d, sg_b, sg_l, sg_c, sgr_b, sgr_l, sgr_d,
                                    ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
                                    offs_t, offs_d, dmask, offs_c, cmask, offs_v, vmask, offs_e,
                                    dqk, nc, NCH, BT, BD, BG, BV, KSNAP)
            dk_intra = tl.dot(tl.trans(dG).to(qc.dtype), qc)
            dk_KV = tl.dot(wv1.to(dSa.dtype), tl.trans(dSa))
            tl.store(dk_ptr + b*sdk_b + sb*sdk_n + rows[:, None]*sdk_l + offs_d[None, :]*sdk_d,
                     dk_intra + dk_KV, mask=rmask[:, None] & dmask[None, :])
            G += tl.dot(qc, tl.trans(kc))
            KS += tl.dot(kc, dSa.to(kc.dtype))
            ZdZ += tl.sum(tl.sum(tl.reshape(Sb * dSa, [BD, BG, BV]), axis=2), axis=0)
        A = G * D * caus
        dD = P * G * caus
        dwt = tl.dot(tl.trans(dD).to(rt.dtype), rt)
        dv_intra = tl.dot(tl.trans(A).to(gc.dtype), gc)
        KS3 = tl.reshape(KS, [BT, BG, BV])
        dw_end = tl.sum(KS3 * v1[:, None, :], axis=2)
        dv_KV = tl.sum(w_end[:, :, None] * KS3, axis=1)
        tl.store(dv_ptr + b*sdv_b + sb*sdv_n + rows[:, None]*sdv_l + offs_v[None, :]*sdv_d,
                 dv_intra + dv_KV, mask=rmask[:, None] & vmask[None, :])
    else:
        P = tl.zeros([BT, BT], dtype=tl.float32)
        for vb in range(ND_V):
            offs_v = vb * BV + tl.arange(0, BV)
            vm = offs_v < dv
            v1 = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d, mask=rmask[:, None] & vm[None, :], other=0.0).to(tl.float32)
            gv = tl.load(g_ptr + b*sgr_b + rows[:, None]*sgr_l + offs_v[None, :]*sgr_d, mask=rmask[:, None] & vm[None, :], other=0.0).to(tl.float32)
            P += tl.dot(gv, tl.trans(v1))
        dG = P * D * caus
        G = tl.zeros([BT, BT], dtype=tl.float32)
        ZdZ = tl.zeros([BG], dtype=tl.float32)
        for d0b in range(ND):
            offs_d = d0b * BD + tl.arange(0, BD)
            dmask = offs_d < dqk
            qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            dk_d = tl.dot(tl.trans(dG).to(qc.dtype), qc)
            for vb in range(ND_V):
                offs_v = vb * BV + tl.arange(0, BV)
                vm = offs_v < dv
                v1 = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d, mask=rmask[:, None] & vm[None, :], other=0.0).to(tl.float32)
                wv1 = tl.reshape(w_end[:, :, None] * v1[:, None, :], [BT, BG * BV])
                offs_e = tl.reshape(offs_g[:, None] * BVF + offs_v[None, :], [BG * BV])
                Sb = _recompute_S_gla(k_ptr, v_ptr, wg_ptr, ld_ptr, Sb_ptr, t, b, sb, L,
                                      sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c,
                                      ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
                                      offs_t, offs_d, dmask, offs_c, cmask, offs_v, vm, offs_e,
                                      dqk, nc, BT, BD, BG, BV, KSNAP)
                dSa = _recompute_dS_gla(q_ptr, rg_ptr, ld_ptr, g_ptr, dSa_ptr, t, b, sb, L,
                                        sq_b, sq_l, sq_d, sg_b, sg_l, sg_c, sgr_b, sgr_l, sgr_d,
                                        ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
                                        offs_t, offs_d, dmask, offs_c, cmask, offs_v, vm, offs_e,
                                        dqk, nc, NCH, BT, BD, BG, BV, KSNAP)
                dk_d += tl.dot(wv1.to(dSa.dtype), tl.trans(dSa))
                ZdZ += tl.sum(tl.sum(tl.reshape(Sb * dSa, [BD, BG, BV]), axis=2), axis=0)
            tl.store(dk_ptr + b*sdk_b + sb*sdk_n + rows[:, None]*sdk_l + offs_d[None, :]*sdk_d,
                     dk_d, mask=rmask[:, None] & dmask[None, :])
            G += tl.dot(qc, tl.trans(kc))
        A = G * D * caus
        dD = P * G * caus
        dwt = tl.dot(tl.trans(dD).to(rt.dtype), rt)
        dw_end = tl.zeros([BT, BG], dtype=tl.float32)
        for vb in range(ND_V):
            offs_v = vb * BV + tl.arange(0, BV)
            vm = offs_v < dv
            v1 = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d, mask=rmask[:, None] & vm[None, :], other=0.0).to(tl.float32)
            gv = tl.load(g_ptr + b*sgr_b + rows[:, None]*sgr_l + offs_v[None, :]*sgr_d, mask=rmask[:, None] & vm[None, :], other=0.0).to(tl.float32)
            offs_e = tl.reshape(offs_g[:, None] * BVF + offs_v[None, :], [BG * BV])
            KS = tl.zeros([BT, BG * BV], dtype=tl.float32)
            for d0b in range(ND):
                offs_d = d0b * BD + tl.arange(0, BD)
                dmask = offs_d < dqk
                kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
                dSa = _recompute_dS_gla(q_ptr, rg_ptr, ld_ptr, g_ptr, dSa_ptr, t, b, sb, L,
                                        sq_b, sq_l, sq_d, sg_b, sg_l, sg_c, sgr_b, sgr_l, sgr_d,
                                        ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
                                        offs_t, offs_d, dmask, offs_c, cmask, offs_v, vm, offs_e,
                                        dqk, nc, NCH, BT, BD, BG, BV, KSNAP)
                KS += tl.dot(kc, dSa.to(kc.dtype))
            KS3 = tl.reshape(KS, [BT, BG, BV])
            dw_end += tl.sum(KS3 * v1[:, None, :], axis=2)
            dv_intra = tl.dot(tl.trans(A).to(gv.dtype), gv)
            dv_KV = tl.sum(w_end[:, :, None] * KS3, axis=1)
            tl.store(dv_ptr + b*sdv_b + sb*sdv_n + rows[:, None]*sdv_l + offs_v[None, :]*sdv_d,
                     dv_intra + dv_KV, mask=rmask[:, None] & vm[None, :])
    # --- shared dLam tail (value-summed inputs dw_end, ZdZ, dwt) ---
    da_wend = -dw_end * w_end
    dwgc = dwt * ena + dw_end * tl.exp(Lam[None, :] - a)
    da_wt = -dwt * wt
    tl.store(dwg_ptr + b*sdw_b + rows[:, None]*sdw_l + offs_c[None, :]*sdw_c, dwgc, mask=rmask[:, None] & cmask[None, :])
    dart = tl.load(dart_ptr + b*sda_b + rows[:, None]*sda_l + offs_c[None, :]*sda_c,
                   mask=rmask[:, None] & cmask[None, :], other=0.0)
    dlam = tl.exp(Lam) * ZdZ - tl.sum(da_wend, axis=0)
    da = dart + da_wt + da_wend
    da += tl.where(offs_t[:, None] == (BT - 1), dlam[None, :], 0.0)
    s = tl.cumsum(da, axis=0)
    dld = tl.sum(da, axis=0)[None, :] - s + da                       # reverse cumsum (tot − cumsum + da)
    tl.store(dld_ptr + b*sdw_b + rows[:, None]*sdw_l + offs_c[None, :]*sdw_c, dld, mask=rmask[:, None] & cmask[None, :])


def _alloc_split(q, v, wg, chunk, BG):
    B, L, dqk = q.shape
    dv = v.shape[-1]
    nc = wg.shape[-1]
    BD = max(16, triton.next_power_of_2(dqk))
    BV = max(16, triton.next_power_of_2(dv))
    NB = triton.cdiv(nc, BG)
    NCH = triton.cdiv(L, chunk)
    Sb = torch.empty(B, NB, NCH, BD, BG * BV, device=q.device, dtype=torch.float32)
    dSa = torch.empty_like(Sb)
    dq = torch.empty(B, NB, L, dqk, device=q.device, dtype=torch.float32)
    dk = torch.empty_like(dq)
    dvo = torch.empty(B, NB, L, BV, device=q.device, dtype=torch.float32)
    dr = torch.empty(B, L, nc, device=q.device, dtype=torch.float32)
    dw = torch.empty_like(dr)
    return BD, BV, NB, NCH, Sb, dSa, dq, dk, dvo, dr, dw


# State-block tiling knob (#3): process at most `_NB_TILE` state-blocks per backward pass, looping over
# the rest, to bound the Sb/dSa snapshot buffers — the dominant backward memory (≈55% of peak). The
# scans/grad kernels index state by `sb*BG` (program_id), so a tile is just sliced gate views + a
# tile-local Sb/dSa: identical math, summed over tiles. None = all blocks at once (byte-identical to
# the untiled path); smaller trades sb-parallelism for peak memory.
_NB_TILE = int(os.environ['ROLA_NB_TILE']) if os.environ.get('ROLA_NB_TILE') else None


def _resolve_nb_tile(nb_tile, B, NB, NCH, BD, BV, BG):
    """How many state-blocks to process per backward pass. Priority: explicit arg > ROLA_NB_TILE env >
    auto-budget. Auto keeps the resident Sb+dSa snapshots under ROLA_SB_BUDGET_MB (default 1024): small
    shapes resolve to all blocks (untiled, byte-identical to the original path), only long-L/large-nc
    shapes — the ones that OOM — tile down. Always >=1 and <=NB."""
    explicit = nb_tile if nb_tile is not None else _NB_TILE
    if explicit is not None:
        return max(1, min(int(explicit), NB))
    budget = int(os.environ.get('ROLA_SB_BUDGET_MB', '1024')) * (1 << 20)
    per_block = max(1, B * NCH * BD * BG * BV * 8)            # (Sb+dSa) fp32 bytes for ONE state-block
    return max(1, min(budget // per_block, NB))


def _bwd_split_rla(q, k, v, wg, rg, g, chunk=None, BG=16, nb_tile=None):
    chunk = _CHUNK if chunk is None else chunk
    B, L, dqk = q.shape
    dv = v.shape[-1]
    nc = wg.shape[-1]
    q, k, v, wg, rg, g = [x.contiguous() for x in (q, k, v, wg, rg, g)]
    BD = max(16, triton.next_power_of_2(dqk))
    BV = max(16, triton.next_power_of_2(dv))
    NB = triton.cdiv(nc, BG)
    NCH = triton.cdiv(L, chunk)
    KSNAP = _resolve_snap_k(B, NB, NCH, BD, BV, BG)
    NSNAP = triton.cdiv(NCH, KSNAP)
    nb_tile = _resolve_nb_tile(nb_tile, B, NB, NSNAP, BD, BV, BG)
    BK_scan = min(64, BD)
    ND_scan = triton.cdiv(dqk, BK_scan)
    sq = (q.stride(0), q.stride(1), q.stride(2))
    sv = (v.stride(0), v.stride(1), v.stride(2))
    sgr = (g.stride(0), g.stride(1), g.stride(2))
    dr = torch.empty(B, L, nc, device=q.device, dtype=torch.float32)     # full-nc, written per slice
    dw = torch.empty_like(dr)
    dq = torch.zeros(B, L, dqk, device=q.device, dtype=torch.float32)     # running sums over state-blocks
    dk = torch.zeros_like(dq)
    dvo = torch.zeros(B, L, BV, device=q.device, dtype=torch.float32)
    for j0 in range(0, NB, nb_tile):
        nbt = min(nb_tile, NB - j0)
        c0, c1 = j0 * BG, min(nc, (j0 + nbt) * BG)
        nct = c1 - c0                                                     # this tile's state width
        wg_s, rg_s, dr_s, dw_s = wg[:, :, c0:c1], rg[:, :, c0:c1], dr[:, :, c0:c1], dw[:, :, c0:c1]
        sg = (wg_s.stride(0), wg_s.stride(1), wg_s.stride(2))
        Sb = torch.empty(B, nbt, NSNAP, BD, BG * BV, device=q.device, dtype=torch.float32)
        dSa = torch.empty_like(Sb)
        dq_t = torch.empty(B, nbt, L, dqk, device=q.device, dtype=torch.float32)
        dk_t = torch.empty_like(dq_t)
        dvo_t = torch.empty(B, nbt, L, BV, device=q.device, dtype=torch.float32)
        sS = (Sb.stride(0), Sb.stride(1), Sb.stride(2), Sb.stride(3), Sb.stride(4))
        _scan_S[(B, nbt, ND_scan)](k, v, wg_s, wg_s, Sb, L, dqk, dv, nct, *sq, *sv, *sg, *sS,
                                   USE_G=False, BT=chunk, BD=BK_scan, BVF=BV, BG=BG, NCH=NCH, KSNAP=KSNAP)
        _scan_dS[(B, nbt, ND_scan)](q, rg_s, wg_s, g, dSa, L, dqk, dv, nct, *sq, *sg, *sgr, *sS,
                                    USE_G=False, BT=chunk, BD=BK_scan, BVF=BV, BG=BG, NCH=NCH, KSNAP=KSNAP)
        _par_grad_rla_qr[(B, nbt, NCH)](q, k, v, wg_s, rg_s, g, Sb, dq_t, dr_s, L, dqk, dv, nct,
                                        *sq, *sv, *sg, *sgr, *sS,
                                        dq_t.stride(0), dq_t.stride(1), dq_t.stride(2), dq_t.stride(3),
                                        dr_s.stride(0), dr_s.stride(1), dr_s.stride(2),
                                        BT=chunk, BVF=BV, BG=BG, NCH=NCH, KSNAP=KSNAP)
        _par_grad_rla_kwv[(B, nbt, NCH)](q, k, v, wg_s, rg_s, g, dSa, dk_t, dw_s, dvo_t, L, dqk, dv, nct,
                                         *sq, *sv, *sg, *sgr, *sS,
                                         dk_t.stride(0), dk_t.stride(1), dk_t.stride(2), dk_t.stride(3),
                                         dw_s.stride(0), dw_s.stride(1), dw_s.stride(2),
                                         dvo_t.stride(0), dvo_t.stride(1), dvo_t.stride(2), dvo_t.stride(3),
                                         BT=chunk, BVF=BV, BG=BG, NCH=NCH, KSNAP=KSNAP)
        dq += dq_t.sum(1)
        dk += dk_t.sum(1)
        dvo += dvo_t.sum(1)
    return dq, dk, dvo[..., :dv], dw, dr


def _bwd_split_gla(q, k, v, wg, rg, ld, g, chunk=None, BG=16, nb_tile=None):
    chunk = _CHUNK if chunk is None else chunk
    B, L, dqk = q.shape
    dv = v.shape[-1]
    nc = wg.shape[-1]
    ld = ld.clamp(min=_GLA_FLOOR)
    q, k, v, wg, rg, ld, g = [x.contiguous() for x in (q, k, v, wg, rg, ld, g)]
    BD = max(16, triton.next_power_of_2(dqk))
    BV = max(16, triton.next_power_of_2(dv))
    NB = triton.cdiv(nc, BG)
    NCH = triton.cdiv(L, chunk)
    KSNAP = _resolve_snap_k(B, NB, NCH, BD, BV, BG)
    NSNAP = triton.cdiv(NCH, KSNAP)
    nb_tile = _resolve_nb_tile(nb_tile, B, NB, NSNAP, BD, BV, BG)
    BK_scan = min(64, BD)
    ND_scan = triton.cdiv(dqk, BK_scan)
    sq = (q.stride(0), q.stride(1), q.stride(2))
    sv = (v.stride(0), v.stride(1), v.stride(2))
    sgr = (g.stride(0), g.stride(1), g.stride(2))
    drg = torch.empty(B, L, nc, device=q.device, dtype=torch.float32)    # full-nc, written per slice
    dwg = torch.empty_like(drg)
    dart = torch.empty_like(drg)                                         # da_rt (read-gate decay adjoint), qr→kwv
    dld = torch.empty_like(drg)                                          # per-token log-decay grad, in kwv
    dq = torch.zeros(B, L, dqk, device=q.device, dtype=torch.float32)    # running sums over state-blocks
    dk = torch.zeros_like(dq)
    dvo = torch.zeros(B, L, BV, device=q.device, dtype=torch.float32)
    for j0 in range(0, NB, nb_tile):
        nbt = min(nb_tile, NB - j0)
        c0, c1 = j0 * BG, min(nc, (j0 + nbt) * BG)
        nct = c1 - c0
        wg_s, rg_s, ld_s = wg[:, :, c0:c1], rg[:, :, c0:c1], ld[:, :, c0:c1]
        drg_s, dwg_s, dart_s, dld_s = drg[:, :, c0:c1], dwg[:, :, c0:c1], dart[:, :, c0:c1], dld[:, :, c0:c1]
        sg = (wg_s.stride(0), wg_s.stride(1), wg_s.stride(2))
        Sb = torch.empty(B, nbt, NSNAP, BD, BG * BV, device=q.device, dtype=torch.float32)
        dSa = torch.empty_like(Sb)
        dq_t = torch.empty(B, nbt, L, dqk, device=q.device, dtype=torch.float32)
        dk_t = torch.empty_like(dq_t)
        dvo_t = torch.empty(B, nbt, L, BV, device=q.device, dtype=torch.float32)
        sS = (Sb.stride(0), Sb.stride(1), Sb.stride(2), Sb.stride(3), Sb.stride(4))
        # GLA snapshot-granularity (parity with RLA): boundary state written only every KSNAP chunks; the
        # grad kernels read the nearest coarse anchor and DECAY-recompute the ≤KSNAP-1 intervening chunks
        # (_recompute_S_gla/_recompute_dS_gla replay Sflat=decvec*Sflat+Kᵀ·WV exactly). −1/KSNAP snapshot HBM.
        _scan_S[(B, nbt, ND_scan)](k, v, wg_s, ld_s, Sb, L, dqk, dv, nct, *sq, *sv, *sg, *sS,
                                   USE_G=True, BT=chunk, BD=BK_scan, BVF=BV, BG=BG, NCH=NCH, KSNAP=KSNAP)
        _scan_dS[(B, nbt, ND_scan)](q, rg_s, ld_s, g, dSa, L, dqk, dv, nct, *sq, *sg, *sgr, *sS,
                                    USE_G=True, BT=chunk, BD=BK_scan, BVF=BV, BG=BG, NCH=NCH, KSNAP=KSNAP)
        _par_grad_gla_qr[(B, nbt, NCH)](q, k, v, wg_s, rg_s, ld_s, g, Sb, dq_t, drg_s, dart_s, L, dqk, dv, nct,
                                        *sq, *sv, *sg, *sgr, *sS,
                                        dq_t.stride(0), dq_t.stride(1), dq_t.stride(2), dq_t.stride(3),
                                        drg_s.stride(0), drg_s.stride(1), drg_s.stride(2),
                                        dart_s.stride(0), dart_s.stride(1), dart_s.stride(2),
                                        BT=chunk, BVF=BV, BG=BG, NCH=NCH, KSNAP=KSNAP)
        _par_grad_gla_kwv[(B, nbt, NCH)](q, k, v, wg_s, rg_s, ld_s, g, Sb, dSa, dart_s, dk_t, dwg_s, dvo_t, dld_s,
                                         L, dqk, dv, nct, *sq, *sv, *sg, *sgr, *sS,
                                         dk_t.stride(0), dk_t.stride(1), dk_t.stride(2), dk_t.stride(3),
                                         dwg_s.stride(0), dwg_s.stride(1), dwg_s.stride(2),
                                         dvo_t.stride(0), dvo_t.stride(1), dvo_t.stride(2), dvo_t.stride(3),
                                         dart_s.stride(0), dart_s.stride(1), dart_s.stride(2),
                                         BT=chunk, BVF=BV, BG=BG, NCH=NCH, KSNAP=KSNAP)
        dq += dq_t.sum(1)
        dk += dk_t.sum(1)
        dvo += dvo_t.sum(1)
    return dq, dk, dvo[..., :dv], dwg, drg, dld


# ============================================================================
# Phase G — per-state DENOMINATOR kernel (for kappa / per-state normalization).
#
# d[i,c] = Σ_{j≤i} (φq_i·φk_j) w_j^c — the mass state c contributes to token i's partition
# function. Used to rescale read gates: r̃ = r·(d+ε)^{-κ(x)} (κ=0 global, κ=1 per-state, exact).
# The eager torch version retains its chunk grams for backward (VRAM blowup at LM scale); this
# is the same scan+parallel pattern as the main backward at [BD,BG] state scale (tiny buffers).
# ============================================================================
_DEN_KEY = ['dqk', 'nc']


@triton.autotune(configs=_BWD_CFGS, key=_DEN_KEY,
                 prune_configs_by={'early_config_prune': _prune_bwd_bd}, **autotune_cache_kwargs)
@triton.jit
def _den_fwd_intra(q_ptr, k_ptr, wg_ptr, ld_ptr, d_ptr, L, dqk: tl.constexpr, nc,
                   sq_b, sq_l, sq_d, sg_b, sg_l, sg_c, sd_b, sd_l, sd_c,
                   USE_G: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr, BG: tl.constexpr):
    """Intra-chunk den for one (batch, state-block, chunk). The [BT,BT] content gram is built by
    BD-blocking the feature dim (SRAM bounded by BD, not dqk). Writes its OWN d_intra buffer with
    plain tl.store (one program per chunk → disjoint rows, race-free / idempotent under autotuning);
    the inter kernel writes the cross-chunk part into d_inter and the host sums d = d_intra + d_inter.
    BD is an autotune knob (the small-SMEM feature-tile fallback); dqk: tl.constexpr so ND is
    compile-time and the loop unrolls (ND==1 → straight-line). USE_G pre-scales by e^a (row-wise,
    distributes over the intra+inter sum)."""
    ND = tl.cdiv(dqk, BD)
    b = tl.program_id(0)
    sb = tl.program_id(1)
    t = tl.program_id(2)
    offs_t = tl.arange(0, BT)
    offs_c = sb * BG + tl.arange(0, BG)
    cmask = offs_c < nc
    rows = t * BT + offs_t
    rmask = rows < L
    wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
    caus = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
    G = tl.zeros([BT, BT], dtype=tl.float32)
    for d0 in range(ND):
        offs_d = d0 * BD + tl.arange(0, BD)
        dmask = offs_d < dqk
        qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        G += tl.dot(qc, tl.trans(kc))
    if USE_G:
        ldc = tl.load(ld_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                      mask=rmask[:, None] & cmask[None, :], other=0.0)
        a = tl.cumsum(ldc, axis=0)
        wt = wgc * tl.exp(-a)
        dch = tl.exp(a) * tl.dot(G * caus, wt)
    else:
        dch = tl.dot((G * caus).to(wgc.dtype), wgc)
    tl.store(d_ptr + b*sd_b + rows[:, None]*sd_l + offs_c[None, :]*sd_c, dch, mask=rmask[:, None] & cmask[None, :])


@triton.autotune(configs=_BWD_CFGS, key=_DEN_KEY, reset_to_zero=['d_ptr'],
                 prune_configs_by={'early_config_prune': _prune_bwd_bd}, **autotune_cache_kwargs)
@triton.jit
def _den_fwd_inter(q_ptr, k_ptr, wg_ptr, ld_ptr, d_ptr, Zb_ptr, L, dqk, nc,
                   sq_b, sq_l, sq_d, sg_b, sg_l, sg_c, sd_b, sd_l, sd_c,
                   szb_b, szb_n, szb_t, szb_d, szb_c,
                   USE_G: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr, BG: tl.constexpr, NCH: tl.constexpr):
    """Inter-chunk (state) den for one (batch, state-block, FEATURE-block d0). Carries this block's
    slice Zd[BD,BG] of the den state across chunks (SRAM bounded by BD) and writes the PRE-update Zb
    snapshot the backward needs. inter uses the state BEFORE this chunk's update (causal); partials
    over feature-blocks sum via atomic_add into its OWN d_inter buffer (reset_to_zero — autotunable;
    the host sums d = d_intra + d_inter). BD is an autotune knob; the grid's feature-block count is a
    launch lambda over meta['BD']. USE_G pre-scales by e^a; the carry decays by e^{Lam}."""
    b = tl.program_id(0)
    sb = tl.program_id(1)
    d0 = tl.program_id(2)
    offs_t = tl.arange(0, BT)
    offs_d = d0 * BD + tl.arange(0, BD)
    dmask = offs_d < dqk
    offs_c = sb * BG + tl.arange(0, BG)
    cmask = offs_c < nc
    offs_g = tl.arange(0, BG)
    Zd = tl.zeros([BD, BG], dtype=tl.float32)
    for t in range(NCH):
        rows = t * BT + offs_t
        rmask = rows < L
        qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                      mask=rmask[:, None] & cmask[None, :], other=0.0)
        tl.store(Zb_ptr + b*szb_b + sb*szb_n + t*szb_t + offs_d[:, None]*szb_d + offs_g[None, :]*szb_c,
                 Zd, mask=dmask[:, None])                                        # PRE-update snapshot
        if USE_G:
            ldc = tl.load(ld_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                          mask=rmask[:, None] & cmask[None, :], other=0.0)
            a = tl.cumsum(ldc, axis=0)
            inter = tl.exp(a) * tl.dot(qc, Zd.to(qc.dtype))
        else:
            inter = tl.dot(qc, Zd.to(qc.dtype))
        tl.atomic_add(d_ptr + b*sd_b + rows[:, None]*sd_l + offs_c[None, :]*sd_c, inter, mask=rmask[:, None] & cmask[None, :])
        kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        if USE_G:
            Lam = tl.sum(tl.where(offs_t[:, None] == (BT - 1), a, 0.0), axis=0)
            w_end = wgc * tl.exp(Lam[None, :] - a)
            Zd = tl.exp(Lam)[None, :] * Zd + tl.dot(tl.trans(kc), w_end.to(kc.dtype))
        else:
            Zd += tl.dot(tl.trans(kc), wgc)


@triton.autotune(configs=_AT_CFGS, key=_DEN_KEY, **autotune_cache_kwargs)
@triton.jit
def _den_bwd_scan(q_ptr, gd_ptr, dZa_ptr, L, dqk, nc,
                  sq_b, sq_l, sq_d, sg_b, sg_l, sg_c,
                  szb_b, szb_n, szb_t, szb_d, szb_c,
                  BT: tl.constexpr, BD: tl.constexpr, BG: tl.constexpr, NCH: tl.constexpr):
    b = tl.program_id(0)
    sb = tl.program_id(1)
    d0 = tl.program_id(2)                        # feature-block: this program owns dZ rows [d0*BD:]
    offs_t = tl.arange(0, BT)
    offs_d = d0 * BD + tl.arange(0, BD)
    offs_c = sb * BG + tl.arange(0, BG)
    offs_g = tl.arange(0, BG)
    dmask = offs_d < dqk
    cmask = offs_c < nc
    dZ = tl.zeros([BD, BG], dtype=tl.float32)
    for ti in range(NCH):
        t = NCH - 1 - ti
        rows = t * BT + offs_t
        rmask = rows < L
        qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]
                     * sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        gdc = tl.load(gd_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]
                      * sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
        tl.store(dZa_ptr + b*szb_b + sb*szb_n + t*szb_t + offs_d[:, None]*szb_d + offs_g[None, :]*szb_c,
                 dZ, mask=dmask[:, None])
        dZ += tl.dot(tl.trans(qc), gdc)


@triton.autotune(configs=_BWD_CFGS, key=_DEN_KEY,
                 prune_configs_by={'early_config_prune': _prune_bwd_bd}, **autotune_cache_kwargs)
@triton.jit
def _den_grad(q_ptr, k_ptr, wg_ptr, gd_ptr, Zb_ptr, dZa_ptr, dq_ptr, dk_ptr, dw_ptr,
              L, dqk: tl.constexpr, nc,
              sq_b, sq_l, sq_d, sg_b, sg_l, sg_c,
              szb_b, szb_n, szb_t, szb_d, szb_c,
              sdq_b, sdq_n, sdq_l, sdq_d, sdw_b, sdw_l, sdw_c,
              BT: tl.constexpr, BD: tl.constexpr, BG: tl.constexpr, NCH: tl.constexpr):
    # D-tiled: dq/dk are feature-indexed (written per BD-block); dw needs the full content gram G +
    # KdZ=Σ_d kc·dZa, accumulated over the BD-block loop ([BT,BT] and [BT,BG] — bounded, independent
    # of dqk). dqk: tl.constexpr so ND is compile-time and the loop unrolls (ND==1 → straight-line,
    # no spill). BD is an autotune knob; ND follows from dqk (the perf lesson — see the main grads).
    ND = tl.cdiv(dqk, BD)
    b = tl.program_id(0)
    sb = tl.program_id(1)
    t = tl.program_id(2)
    offs_t = tl.arange(0, BT)
    offs_c = sb * BG + tl.arange(0, BG)
    offs_g = tl.arange(0, BG)
    cmask = offs_c < nc
    rows = t * BT + offs_t
    rmask = rows < L
    wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
    gdc = tl.load(gd_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
    caus = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
    P = tl.dot(gdc, tl.trans(wgc))                                   # [BT,BT]   (d-independent)
    Pc = P * caus
    G = tl.zeros([BT, BT], dtype=tl.float32)
    KdZ = tl.zeros([BT, BG], dtype=tl.float32)
    for d0b in range(ND):
        offs_d = d0b * BD + tl.arange(0, BD)
        dmask = offs_d < dqk
        qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        Zb = tl.load(Zb_ptr + b*szb_b + sb*szb_n + t*szb_t + offs_d[:, None]
                     * szb_d + offs_g[None, :]*szb_c, mask=dmask[:, None], other=0.0)
        dZa = tl.load(dZa_ptr + b*szb_b + sb*szb_n + t*szb_t +
                      offs_d[:, None]*szb_d + offs_g[None, :]*szb_c, mask=dmask[:, None], other=0.0)
        dq_d = tl.dot(Pc.to(kc.dtype), kc) + tl.dot(gdc, tl.trans(Zb.to(gdc.dtype)))
        dk_d = tl.dot(tl.trans(Pc).to(qc.dtype), qc) + tl.dot(wgc, tl.trans(dZa.to(wgc.dtype)))
        tl.store(dq_ptr + b*sdq_b + sb*sdq_n + rows[:, None]*sdq_l +
                 offs_d[None, :]*sdq_d, dq_d, mask=rmask[:, None] & dmask[None, :])
        tl.store(dk_ptr + b*sdq_b + sb*sdq_n + rows[:, None]*sdq_l +
                 offs_d[None, :]*sdq_d, dk_d, mask=rmask[:, None] & dmask[None, :])
        G += tl.dot(qc, tl.trans(kc))
        KdZ += tl.dot(kc, dZa.to(kc.dtype))
    dw = tl.dot(tl.trans(G * caus).to(gdc.dtype), gdc) + KdZ
    tl.store(dw_ptr + b*sdw_b + rows[:, None]*sdw_l + offs_c[None, :]*sdw_c, dw, mask=rmask[:, None] & cmask[None, :])


@torch.library.custom_op("rola::den", mutates_args=())
def _den_op(q: torch.Tensor, k: torch.Tensor, wg: torch.Tensor,
            chunk: int, BG: int) -> typing.List[torch.Tensor]:  # noqa: UP006
    """Per-state denominator as an opaque custom op; returns [d, Zb] (Zb saved for backward)."""
    with torch.autocast('cuda', enabled=False):
        B, L, dqk = q.shape
        nc = wg.shape[-1]
        BD = max(16, triton.next_power_of_2(dqk))
        NB = triton.cdiv(nc, BG)
        NCH = triton.cdiv(L, chunk)
        q, k, wg = q.contiguous(), k.contiguous(), wg.contiguous()
        d_intra = torch.zeros(B, L, nc, device=q.device, dtype=torch.float32)
        d_inter = torch.zeros_like(d_intra)
        Zb = torch.empty(B, NB, NCH, BD, BG, device=q.device, dtype=torch.float32)
        sq = (q.stride(0), q.stride(1), q.stride(2))
        sg = (wg.stride(0), wg.stride(1), wg.stride(2))
        sd = (d_intra.stride(0), d_intra.stride(1), d_intra.stride(2))
        sZ = (Zb.stride(0), Zb.stride(1), Zb.stride(2), Zb.stride(3), Zb.stride(4))
        _den_fwd_intra[(B, NB, NCH)](q, k, wg, wg, d_intra, L, dqk, nc, *sq, *sg, *sd,
                                     USE_G=False, BT=chunk, BG=BG)

        def grid_inter(meta):
            return (B, NB, triton.cdiv(dqk, meta['BD']))
        _den_fwd_inter[grid_inter](q, k, wg, wg, d_inter, Zb, L, dqk, nc, *sq, *sg, *sd, *sZ,
                                   USE_G=False, BT=chunk, BG=BG, NCH=NCH)
        return [d_intra + d_inter, Zb]


@_den_op.register_fake
def _den_op_fake(q, k, wg, chunk, BG):
    B, L, dqk = q.shape
    nc = wg.shape[-1]
    BD = max(16, triton.next_power_of_2(dqk))
    NB = triton.cdiv(nc, BG)
    NCH = triton.cdiv(L, chunk)
    return [q.new_empty((B, L, nc), dtype=torch.float32),
            q.new_empty((B, NB, NCH, BD, BG), dtype=torch.float32)]


@torch.library.custom_op("rola::den_bwd", mutates_args=())
def _den_bwd_op(q: torch.Tensor, k: torch.Tensor, wg: torch.Tensor, Zb: torch.Tensor,
                gd: torch.Tensor, chunk: int, BG: int) -> typing.List[torch.Tensor]:  # noqa: UP006
    with torch.autocast('cuda', enabled=False):
        B, L, dqk = q.shape
        nc = wg.shape[-1]
        BD = max(16, triton.next_power_of_2(dqk))
        NB = triton.cdiv(nc, BG)
        NCH = triton.cdiv(L, chunk)
        gd = gd.contiguous().to(q.dtype)
        dZa = torch.empty_like(Zb)
        BK_scan = min(64, BD)
        ND_scan = triton.cdiv(dqk, BK_scan)
        _den_bwd_scan[(B, NB, ND_scan)](q, gd, dZa, L, dqk, nc,
                                        q.stride(0), q.stride(1), q.stride(2), gd.stride(0), gd.stride(1), gd.stride(2),
                                        dZa.stride(0), dZa.stride(1), dZa.stride(2), dZa.stride(3), dZa.stride(4),
                                        BT=chunk, BD=BK_scan, BG=BG, NCH=NCH)
        dq = torch.empty(B, NB, L, dqk, device=q.device, dtype=torch.float32)
        dk = torch.empty_like(dq)
        dw = torch.empty(B, L, nc, device=q.device, dtype=torch.float32)
        _den_grad[(B, NB, NCH)](q, k, wg, gd, Zb, dZa, dq, dk, dw, L, dqk, nc,
                                q.stride(0), q.stride(1), q.stride(2), gd.stride(0), gd.stride(1), gd.stride(2),
                                Zb.stride(0), Zb.stride(1), Zb.stride(2), Zb.stride(3), Zb.stride(4),
                                dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3), dw.stride(0), dw.stride(1), dw.stride(2),
                                BT=chunk, BG=BG, NCH=NCH)
        return [dq.sum(1), dk.sum(1), dw]


@_den_bwd_op.register_fake
def _den_bwd_op_fake(q, k, wg, Zb, gd, chunk, BG):
    B, L, dqk = q.shape
    nc = wg.shape[-1]
    f = torch.float32
    return [q.new_empty((B, L, dqk), dtype=f), q.new_empty((B, L, dqk), dtype=f), q.new_empty((B, L, nc), dtype=f)]


def _den_setup(ctx, inputs, output):
    q, k, wg, chunk, BG = inputs
    ctx.save_for_backward(q, k, wg, output[1])
    ctx.chunk = chunk
    ctx.BG = BG


def _den_backward(ctx, grad):
    q, k, wg, Zb = ctx.saved_tensors
    grad_d = grad[0] if isinstance(grad, (list, tuple)) else grad   # list-output op: grad is [grad_d, grad_Zb]
    dq, dk, dw = _den_bwd_op(q, k, wg, Zb, grad_d, ctx.chunk, ctx.BG)
    def cast(t): return t.to(q.dtype)
    return cast(dq), cast(dk), cast(dw), None, None


_den_op.register_autograd(_den_backward, setup_context=_den_setup)


@input_guard
def rola_perstate_den_triton(q, k, w, chunk=None, BG=16):
    """Per-state denominator on folded [BH,L,*] tensors. Differentiable (Triton fwd + parallel bwd)."""
    chunk = _CHUNK if chunk is None else min(chunk, _CHUNK)
    d, _Zb = _den_op(q, k, w, chunk, BG)
    return d


@triton.autotune(configs=_AT_CFGS, key=_DEN_KEY, **autotune_cache_kwargs)
@triton.jit
def _den_gla_bwd_scan(q_ptr, gd_ptr, ld_ptr, dZa_ptr, L, dqk, nc,
                      sq_b, sq_l, sq_d, sg_b, sg_l, sg_c,
                      szb_b, szb_n, szb_t, szb_d, szb_c,
                      BT: tl.constexpr, BD: tl.constexpr, BG: tl.constexpr, NCH: tl.constexpr):
    """Reverse scan for the DECAYED den: dZa[t] = adjoint of chunk t's carry increment
    (sum over later chunks, decayed). dZ_t = e^{Lam_t} dZ_{t+1} + q^T (e^a ∘ gd)."""
    b = tl.program_id(0)
    sb = tl.program_id(1)
    d0 = tl.program_id(2)                        # feature-block: this program owns dZ rows [d0*BD:]
    offs_t = tl.arange(0, BT)
    offs_d = d0 * BD + tl.arange(0, BD)
    offs_c = sb * BG + tl.arange(0, BG)
    offs_g = tl.arange(0, BG)
    dmask = offs_d < dqk
    cmask = offs_c < nc
    dZ = tl.zeros([BD, BG], dtype=tl.float32)
    for ti in range(NCH):
        t = NCH - 1 - ti
        rows = t * BT + offs_t
        rmask = rows < L
        qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]
                     * sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        gdc = tl.load(gd_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]
                      * sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
        ldc = tl.load(ld_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]
                      * sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
        a = tl.cumsum(ldc, axis=0)
        tl.store(dZa_ptr + b*szb_b + sb*szb_n + t*szb_t + offs_d[:, None]*szb_d + offs_g[None, :]*szb_c,
                 dZ, mask=dmask[:, None])
        Lam = tl.sum(tl.where(offs_t[:, None] == (BT - 1), a, 0.0), axis=0)
        gt = gdc * tl.exp(a)
        dZ = tl.exp(Lam)[None, :] * dZ + tl.dot(tl.trans(qc), gt.to(qc.dtype))


@triton.autotune(configs=_BWD_CFGS, key=_DEN_KEY,
                 prune_configs_by={'early_config_prune': _prune_bwd_bd}, **autotune_cache_kwargs)
@triton.jit
def _den_gla_grad(q_ptr, k_ptr, wg_ptr, ld_ptr, gd_ptr, Zb_ptr, dZa_ptr,
                  dq_ptr, dk_ptr, dw_ptr, dld_ptr, L, dqk: tl.constexpr, nc,
                  sq_b, sq_l, sq_d, sg_b, sg_l, sg_c,
                  szb_b, szb_n, szb_t, szb_d, szb_c,
                  sdq_b, sdq_n, sdq_l, sdq_d, sdw_b, sdw_l, sdw_c,
                  BT: tl.constexpr, BD: tl.constexpr, BG: tl.constexpr, NCH: tl.constexpr):
    """Parallel per-chunk grads for the decayed den, incl. dld assembled IN-KERNEL:
    da from the three appearances of a (output scale e^a, intra w·e^{-a}, carry w·e^{Lam-a}),
    dLam from the carry decay (e^{Lam}·Σ Zb∘dZa, the main kernel's dLam trick) + the carry
    writes, folded into da's last row; dld = reverse cumsum of da (tot − cumsum + da)."""
    # D-tiled: dq/dk feature-indexed (per BD-block); the FOUR feature-dependent quantities
    # G[BT,BT], KdZ=Σ_d kc·dZa, QZ=Σ_d qc·Zb, ZdZ=Σ_d(Zb∘dZa) [over the D axis] are accumulated
    # across BD-blocks and consumed AFTER the loop (dch uses QZ; dLam uses ZdZ; both feed dld).
    # dqk: tl.constexpr → ND compile-time, loop unrolls (ND==1 collapses to straight-line, no spill).
    ND = tl.cdiv(dqk, BD)
    b = tl.program_id(0)
    sb = tl.program_id(1)
    t = tl.program_id(2)
    offs_t = tl.arange(0, BT)
    offs_c = sb * BG + tl.arange(0, BG)
    offs_g = tl.arange(0, BG)
    cmask = offs_c < nc
    rows = t * BT + offs_t
    rmask = rows < L
    wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
    ldc = tl.load(ld_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
    gdc = tl.load(gd_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
    a = tl.cumsum(ldc, axis=0)
    Lam = tl.sum(tl.where(offs_t[:, None] == (BT - 1), a, 0.0), axis=0)
    ea = tl.exp(a)
    wt = wgc * tl.exp(-a)
    w_end = wgc * tl.exp(Lam[None, :] - a)
    caus = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
    gt = gdc * ea
    P = tl.dot(gt, tl.trans(wt))                                     # [BT,BT]   (d-independent)
    Pc = P * caus
    G = tl.zeros([BT, BT], dtype=tl.float32)
    KdZ = tl.zeros([BT, BG], dtype=tl.float32)
    QZ = tl.zeros([BT, BG], dtype=tl.float32)
    ZdZ = tl.zeros([BG], dtype=tl.float32)
    for d0b in range(ND):
        offs_d = d0b * BD + tl.arange(0, BD)
        dmask = offs_d < dqk
        qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        Zb = tl.load(Zb_ptr + b*szb_b + sb*szb_n + t*szb_t + offs_d[:, None]
                     * szb_d + offs_g[None, :]*szb_c, mask=dmask[:, None], other=0.0)
        dZa = tl.load(dZa_ptr + b*szb_b + sb*szb_n + t*szb_t +
                      offs_d[:, None]*szb_d + offs_g[None, :]*szb_c, mask=dmask[:, None], other=0.0)
        dq_d = tl.dot(Pc.to(kc.dtype), kc) + tl.dot(gt, tl.trans(Zb))
        dk_d = tl.dot(tl.trans(Pc).to(qc.dtype), qc) + tl.dot(w_end, tl.trans(dZa))
        tl.store(dq_ptr + b*sdq_b + sb*sdq_n + rows[:, None]*sdq_l +
                 offs_d[None, :]*sdq_d, dq_d, mask=rmask[:, None] & dmask[None, :])
        tl.store(dk_ptr + b*sdq_b + sb*sdq_n + rows[:, None]*sdq_l +
                 offs_d[None, :]*sdq_d, dk_d, mask=rmask[:, None] & dmask[None, :])
        G += tl.dot(qc, tl.trans(kc))
        KdZ += tl.dot(kc, dZa.to(kc.dtype))                         # [BT,BG]   Σ_d kc·dZa
        QZ += tl.dot(qc, Zb.to(qc.dtype))                          # [BT,BG]   Σ_d qc·Zb
        ZdZ += tl.sum(Zb * dZa, axis=0)                            # [BG]      Σ_d (Zb∘dZa)
    GTg = tl.dot(tl.trans(G * caus), gt)                             # [BT(j),BG]
    dw = tl.exp(-a) * GTg + tl.exp(Lam[None, :] - a) * KdZ
    dch = ea * (tl.dot(G * caus, wt) + QZ)                           # recomputed chunk output
    da = gdc * dch - wt * GTg - w_end * KdZ
    dLam = tl.sum(w_end * KdZ, axis=0) + tl.exp(Lam) * ZdZ
    da += tl.where(offs_t[:, None] == (BT - 1), dLam[None, :], 0.0)
    s = tl.cumsum(da, axis=0)
    dld = tl.sum(da, axis=0)[None, :] - s + da                       # reverse cumsum
    tl.store(dw_ptr + b*sdw_b + rows[:, None]*sdw_l + offs_c[None, :]*sdw_c, dw, mask=rmask[:, None] & cmask[None, :])
    tl.store(dld_ptr + b*sdw_b + rows[:, None]*sdw_l + offs_c[None, :]*sdw_c, dld, mask=rmask[:, None] & cmask[None, :])


class _DenGLAFn(torch.autograd.Function):
    """Per-state denominator under per-state log-decay: d[i,c] = Σ_{j≤i} (φq_i·φk_j) w_j^c e^{Λ_ic−Λ_jc}.
    Forward: _den_fwd_intra+_den_fwd_inter USE_G=True. Backward: dedicated decayed den kernels at [BD,BG] state
    scale, value-free (_den_gla_bwd_scan + _den_gla_grad with in-kernel dld assembly) — the
    same cost class as the additive den, not a second full GLA backward."""
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, wg, ld, chunk, BG):
        B, L, dqk = q.shape
        nc = wg.shape[-1]
        BD = max(16, triton.next_power_of_2(dqk))
        NB = triton.cdiv(nc, BG)
        NCH = triton.cdiv(L, chunk)
        ld = ld.clamp(min=_GLA_FLOOR)
        q, k, wg, ld = q.contiguous(), k.contiguous(), wg.contiguous(), ld.contiguous()
        # Disjoint dual-buffer + host-sum (mirrors the BC-split fwd/bwd): intra writes its own buffer
        # with plain store, inter atomic-accumulates over feature-blocks into its own (reset_to_zero).
        # Both feature-tile-autotuned (BD knob → small-SMEM fallback); summed on the host.
        d_intra = torch.zeros(B, L, nc, device=q.device, dtype=torch.float32)
        d_inter = torch.zeros_like(d_intra)
        Zb = torch.empty(B, NB, NCH, BD, BG, device=q.device, dtype=torch.float32)
        sq = (q.stride(0), q.stride(1), q.stride(2))
        sg = (wg.stride(0), wg.stride(1), wg.stride(2))
        sd = (d_intra.stride(0), d_intra.stride(1), d_intra.stride(2))
        sZ = (Zb.stride(0), Zb.stride(1), Zb.stride(2), Zb.stride(3), Zb.stride(4))
        _den_fwd_intra[(B, NB, NCH)](q, k, wg, ld, d_intra, L, dqk, nc, *sq, *sg, *sd,
                                     USE_G=True, BT=chunk, BG=BG)
        def grid_inter(meta):
            return (B, NB, triton.cdiv(dqk, meta['BD']))
        _den_fwd_inter[grid_inter](q, k, wg, ld, d_inter, Zb, L, dqk, nc, *sq, *sg, *sd, *sZ,
                                   USE_G=True, BT=chunk, BG=BG, NCH=NCH)
        d = d_intra + d_inter
        ctx.save_for_backward(q, k, wg, ld, Zb)
        ctx.meta = (chunk, BG, BD, NB, NCH)
        return d

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, gd):
        q, k, wg, ld, Zb = ctx.saved_tensors
        chunk, BG, BD, NB, NCH = ctx.meta
        B, L, dqk = q.shape
        nc = wg.shape[-1]
        gd = gd.contiguous().to(q.dtype)
        dZa = torch.empty_like(Zb)
        # Scan is d-parallel (register carry, own feature block; e^Lam d-independent → row-separable).
        # Zb/dZa indexed by absolute feature row, independent of the grad's autotuned BD. Grad BD autotuned.
        BK_scan = min(64, BD)
        ND_scan = triton.cdiv(dqk, BK_scan)
        _den_gla_bwd_scan[(B, NB, ND_scan)](q, gd, ld, dZa, L, dqk, nc,
                                            q.stride(0), q.stride(1), q.stride(2), gd.stride(0), gd.stride(1), gd.stride(2),
                                            dZa.stride(0), dZa.stride(1), dZa.stride(2), dZa.stride(3), dZa.stride(4),
                                            BT=chunk, BD=BK_scan, BG=BG, NCH=NCH)
        dq = torch.empty(B, NB, L, dqk, device=q.device, dtype=torch.float32)
        dk = torch.empty_like(dq)
        dw = torch.empty(B, L, nc, device=q.device, dtype=torch.float32)
        dld = torch.empty_like(dw)
        _den_gla_grad[(B, NB, NCH)](q, k, wg, ld, gd, Zb, dZa, dq, dk, dw, dld, L, dqk, nc,
                                    q.stride(0), q.stride(1), q.stride(2), gd.stride(0), gd.stride(1), gd.stride(2),
                                    Zb.stride(0), Zb.stride(1), Zb.stride(2), Zb.stride(3), Zb.stride(4),
                                    dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(
                                        3), dw.stride(0), dw.stride(1), dw.stride(2),
                                    BT=chunk, BG=BG, NCH=NCH)

        def cast(t): return t.to(q.dtype)
        return cast(dq.sum(1)), cast(dk.sum(1)), cast(dw), dld.to(ld.dtype), None, None


@input_guard
def rola_perstate_den_gla_triton(q, k, w, ld, chunk=None, BG=16):
    """Per-state denominator under per-state log-decay ld:[BH,L,nc], folded tensors. Differentiable."""
    chunk = _CHUNK if chunk is None else min(chunk, _CHUNK)
    return _DenGLAFn.apply(q, k, w, ld, chunk, BG)


# ============================================================================
# `chunk_rola` — the norm-aware public entry point (FLA inlines the norm-aware entrypoint at the
# bottom of `chunk.py`, cf. `chunk_gla` in `ops/gla/chunk.py`).
#
# It is the ONE norm-aware entry point: it owns the whole recipe — the per-state denominator
# pre-pass, the read-gate rescale, the (numerator-only) shared-gram readout, and the divide — so
# callers (the LM layer) just pass `norm=...`. There is no routed branch bolted onto simple_gla and
# no normalization logic in the model.
#
# Normalization is unified on the DENOMINATOR-SPLIT: the readout is always numerator-only
# (BV=next_pow2(dv), no ones-column → ~2× occupancy, fits small smem), and the global denominator is
# reconstructed as Σ_c r̃ᶜ·dᶜ from the per-state den pre-pass `d` (the same `d` that rescales the read
# gates for kappa/per_state). global/kappa/per_state differ ONLY in how the read gates are rescaled:
#   global    : r̃ = r                      (den = Σ_c rᶜ·dᶜ — the partition function)
#   per_state : r̃ = r / (d+ε)              (≡ kappa=1, each state self-normalizes)
#   kappa     : r̃ = r · (d+ε)^(-κ(x))      (input-dependent interpolation global↔per_state)
#   raw       : numerator only, no divide   (GLA convention)
#
# Differentiable by COMPOSITION — the den pre-pass and the readout are verified autograd Functions
# and the rescale/divide are plain torch, so gradients flow automatically (no custom backward).
#
# NO-CACHE LIMITATION: `chunk_rola` is the chunk-parallel (training) path only — there is no fused
# recurrent / KV-cache decode kernel yet (the recurrent form lives in the tests as virtual-heads over
# `fused_recurrent_simple_gla`). A first-class `fused_recurrent_rola.py` is deferred.
# ============================================================================
_NORMS = ('raw', 'global', 'per_state', 'kappa')


def _rola_chunk_core(q, k, v, w, r, ld, chunk_size):
    """Eager (CPU / capability-fallback) shared-gram routed readout on folded [BH, T, *] tensors.
    ld=None ⇒ RLA (chunk-parallel cumsum scan); ld given ⇒ scalar-gated GLA (decayed scan). Returns
    the un-normalized readout [BH, T, v.shape[-1]] (numerator; the den is a separate pre-pass)."""
    BH, T, K = q.shape
    V = v.shape[-1]
    nc = w.shape[-1]
    if ld is None:
        pad = (-T) % chunk_size
        if pad:
            q, k, v, w, r = [F.pad(t, (0, 0, 0, pad)) for t in (q, k, v, w, r)]
        Tp = T + pad
        nch = Tp // chunk_size
        C = chunk_size
        qc, kc, vc = q.view(BH, nch, C, K), k.view(BH, nch, C, K), v.view(BH, nch, C, V)
        rc, wc = r.view(BH, nch, C, nc), w.view(BH, nch, C, nc)
        G = torch.einsum('bnid,bnjd->bnij', qc, kc)
        R = torch.einsum('bnic,bnjc->bnij', rc, wc)
        causal = torch.tril(torch.ones(C, C, device=q.device, dtype=q.dtype))
        A = G * R * causal
        o_intra = torch.einsum('bnij,bnjv->bniv', A, vc)
        KV = torch.einsum('bnjc,bnjd,bnjv->bncdv', wc, kc, vc)
        S_before = torch.cumsum(KV, dim=1) - KV
        M = torch.einsum('bnic,bncdv->bnidv', rc, S_before)
        o_inter = torch.einsum('bnid,bnidv->bniv', qc, M)
        return (o_intra + o_inter).reshape(BH, Tp, V)[:, :T]
    # GLA: per-state scalar decay, sequential decayed scan over chunks.
    S = q.new_zeros(BH, nc, K, V)
    outs = []
    for c0 in range(0, T, chunk_size):
        c1 = min(c0 + chunk_size, T)
        Cc = c1 - c0
        qcc, kcc, vcc = q[:, c0:c1], k[:, c0:c1], v[:, c0:c1]
        rcc, wcc, ldc = r[:, c0:c1], w[:, c0:c1], ld[:, c0:c1]
        a = torch.cumsum(ldc, dim=1)
        rt, wt = rcc * torch.exp(a), wcc * torch.exp(-a)
        G = torch.einsum('bid,bjd->bij', qcc, kcc)
        D = torch.einsum('bic,bjc->bij', rt, wt)
        causal = torch.tril(torch.ones(Cc, Cc, device=q.device, dtype=q.dtype))
        o_intra = torch.einsum('bij,bjv->biv', G * D * causal, vcc)
        M = torch.einsum('bic,bcdv->bidv', rt, S)
        o_inter = torch.einsum('bid,bidv->biv', qcc, M)
        outs.append(o_intra + o_inter)
        Lam = a[:, -1, :]
        w_end = wcc * torch.exp(Lam[:, None, :] - a)
        KV = torch.einsum('bjc,bjd,bjv->bcdv', w_end, kcc, vcc)
        S = torch.exp(Lam)[:, :, None, None] * S + KV
    return torch.cat(outs, dim=1)


def _perstate_den_torch(q, k, w, ld, chunk_size, eps=1e-5):
    """Eager per-state denominator dᵢᶜ = Σ_{j≤i} (φqᵢ·φkⱼ) wⱼᶜ [· e^{Λᵢᶜ−Λⱼᶜ} under decay], folded
    [BH,T,*] → [BH,T,nc]. CPU/capability fallback for the Triton den kernels."""
    BH, T, K = q.shape
    G = torch.einsum('bid,bjd->bij', q, k)
    causal = torch.tril(torch.ones(T, T, device=q.device, dtype=q.dtype))
    if ld is None:
        return torch.einsum('bij,bjc->bic', G * causal, w)
    A = torch.cumsum(ld, dim=1)                          # [BH,T,nc] cumulative log-decay
    s = torch.einsum('bij,bjc->bic', G * causal, w * torch.exp(-A))
    return torch.exp(A) * s


def _rola_readout(qf, kf, vf, rf, wf, gf, chunk_size):
    """Folded routed readout (numerator-only). CUDA → device-agnostic Triton kernels; else → eager
    core. The global denominator is the caller's separate per-state den pre-pass."""
    if qf.is_cuda:
        if gf is None:
            return rola_rla_triton(qf, kf, vf, rf, wf, chunk=chunk_size)
        return rola_gla_triton(qf, kf, vf, rf, wf, gf)
    return _rola_chunk_core(qf, kf, vf, wf, rf, gf, chunk_size)


@input_guard
def chunk_rola(q, k, v, r, w, g=None, norm='kappa', kappa=None, scale=None, eps=1e-5,
               initial_state=None, output_final_state=False):
    """Routed RoLA (shared-gram) readout with built-in normalization.

    Args:
        q, k:  φ-mapped queries/keys [B, T, H, K] (the feature map φ stays in the caller — the
               operator is φ-agnostic, seeing only the content gram G=φ(q)φ(k)ᵀ).
        v:     values [B, T, H, V].
        r, w:  read / write routing gates [B, T, H, nc].
        g:     per-state log-decay [B, T, H, nc] for the scalar-gated (GLA) variant, or None (RLA).
        norm:  'raw' | 'global' | 'per_state' | 'kappa'.
        kappa: per-token exponent [B, T, H, 1] (required for norm='kappa').
        scale: query scale (default 1/sqrt(K)).
    Returns:
        Normalized readout [B, T, H, V] ('raw' returns the un-normalized numerator).
    """
    if norm not in _NORMS:
        raise ValueError(f"norm must be one of {_NORMS}, got {norm!r}")
    if norm == 'kappa' and kappa is None:
        raise ValueError("norm='kappa' requires a per-token `kappa` exponent tensor [B,T,H,1]")
    B, T, H, K = q.shape
    if scale is None:
        scale = K ** -0.5
    chunk_size = min(64, max(16, triton.next_power_of_2(T)))

    def fold(t):
        return t.permute(0, 2, 1, 3).reshape(B * H, T, t.shape[-1])

    def unfold(t):
        return t.view(B, H, T, -1).permute(0, 2, 1, 3).contiguous()

    # The op owns the kernel dtype contract (the layer no longer hand-rolls it). The shared-gram
    # tl.dot needs same-dtype operands, but the feature map (q,k bf16) and the router softmax (r,w
    # fp32) disagree under autocast — so unify every folded kernel input onto ONE compute dtype here.
    # Under autocast that is the autocast dtype; otherwise the inputs' own dtype.
    compute_dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else q.dtype

    def foldc(t):
        return fold(t).to(compute_dtype)

    qf, kf, vf, rf, wf = foldc(q) * scale, foldc(k), foldc(v), foldc(r), foldc(w)
    gf = foldc(g) if g is not None else None

    if norm == 'raw':
        return unfold(_rola_readout(qf, kf, vf, rf, wf, gf, chunk_size)).to(v.dtype)

    # global / per_state / kappa: per-state den pre-pass → rescale read gates → numerator-only
    # readout → divide by the reconstructed global den Σ_c r̃ᶜ·dᶜ.
    if qf.is_cuda:
        d = (rola_perstate_den_gla_triton(qf, kf, wf, gf) if gf is not None
             else rola_perstate_den_triton(qf, kf, wf))
    else:
        d = _perstate_den_torch(qf, kf, wf, gf, chunk_size, eps)
    # Rescale read gates, then cast BACK to the compute dtype: `d` is fp32, so the rescale would
    # upcast r̃ to fp32 and break the kernel's same-dtype requirement (tl.dot(r̃, wᵀ) with w in bf16).
    if norm == 'kappa':
        rf = (rf * (d + eps).pow(-fold(kappa).to(d.dtype))).to(compute_dtype)
    elif norm == 'per_state':
        rf = (rf / (d + eps)).to(compute_dtype)
    # norm == 'global': r̃ = r (unchanged)
    num = _rola_readout(qf, kf, vf, rf, wf, gf, chunk_size)
    den = (rf * d).sum(-1, keepdim=True)
    out = unfold(num / (den + eps)).to(v.dtype)
    if not output_final_state:
        return out
    return out, _final_state(kf, vf, wf, gf, B, H)


def _final_state(kf, vf, wf, gf, B, H):
    """Final recurrent state of the chunked pass: `stateᶜ = Σ_t [e^{G_T-G_t}·]wᵗᶜ·kf_t⊗[vf_t;1]`,
    shaped `[N, H*nc, K, V+1]` (the `+1` ones-column is the per-state denominator) — byte-compatible
    with `fused_recurrent_rola`'s state so a chunked prefill hands off to recurrent decode. O(L), no
    kernel; the backward through a carried state is not provided (decode is inference)."""
    v1 = torch.cat([vf, torch.ones_like(vf[..., :1])], -1).float()      # [BH,T,V+1]
    wgt = wf.float()                                                    # [BH,T,nc]
    if gf is not None:                                                  # GLA: token t decays by Σ_{t'>t} g
        G = gf.float().cumsum(1)
        wgt = wgt * (G[:, -1:, :] - G).exp()
    state = torch.einsum('btc,btd,bte->bcde', wgt, kf.float(), v1)      # [BH, nc, K, V+1]
    return state.view(B, H * state.shape[1], state.shape[2], state.shape[3])
