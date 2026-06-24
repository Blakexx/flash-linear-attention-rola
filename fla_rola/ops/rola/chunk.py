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

from fla_rola.ops.rola.proto_tree_routing import (  # validated tree-routing fold kernels (reused verbatim by the routed backward)
    _bwd_inter_read_kernel as _proto_bwd_inter_read,
)
from fla_rola.ops.rola.proto_tree_routing import (
    _bwd_inter_state_kernel as _proto_bwd_inter_state,
)
from fla_rola.ops.rola.proto_tree_routing import (
    _bwd_intra_kernel as _proto_bwd_intra,
)
from fla_rola.ops.rola.proto_tree_routing import (
    _fold_kernel as _proto_fold,
)
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
# The scans (_scan_S/_scan_dS) are SHARED by RLA (USE_G=False) and GLA (USE_G=True) at the SAME
# (dqk,dv,nc); without USE_G in the key the autotune config-cache collides → RLA and GLA reuse each
# other's tuned warps/stages/BV. USE_G is a constexpr (so it specializes the compile regardless), but
# it must also gate config SELECTION so each variant tunes its own (the GLA decay-replay has a heavier
# SMEM profile than RLA, so the best config differs). Perf-only; no correctness change. (#22)
_SCAN_KEY = ['dqk', 'dv', 'nc', 'USE_G']
_CHUNK_FWD = 64 if _BIG_SMEM else 16     # RLA forward
_CHUNK = 32 if _BIG_SMEM else 16         # GLA forward + all backwards (GLA fp32 decay floor caps BT<=32)
_KAPPA_BWD_CHUNK = 16                     # fused kappa backward (single mega-kernel, heavy fp32 SMEM)

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


# --- CURATED warp/stage candidates, paired to the tile FOOTPRINT (FLA discipline, #22) ----------------
# The full BK×BV×WARPS(3)×STAGES(3) cross-product is 108 configs (54 survive the dqk/dv prune at the
# bench cells) — the autotuner cold-compiles EVERY survivor, so the config count is the dominant cold-
# compile multiplier (each large-dqk fp32 grad config is tens of seconds). Trimming warps/stages NAIVELY
# and GLOBALLY is INFEASIBLE — it breaks the SMEM fit: at frontier dqk the big tiles NEED 8 warps (to
# spread the wide fp32 Kronecker state across the register file) AND num_stages>=2 (the software pipeline
# LOWERS peak SMEM via buffer reuse; stages=1 RAISES it). So instead of dropping warp/stage VALUES, we
# pair them to the tile size: a SMALL tile (BD*BV<=512) doesn't need 8 warps or a 3-deep pipeline (pure
# waste there), a LARGE tile keeps the fit-critical 8-warp/stages>=2 set. Every config that is fit-
# critical at large dqk survives; only the redundant small-tile combos are cut. ~2x fewer configs,
# zero fit loss. (Mirrors how FLA ships curated lists + heuristics, not a combinatorial sweep.)
def _bwd_ws(bd, bv):
    """(num_warps, num_stages) candidates for a BD×BV grad tile, by footprint (see above)."""
    fp = bd * bv
    if fp <= 16 * 32:            # small: 2/4 warps, shallow pipeline
        return [(2, 1), (2, 2), (4, 1), (4, 2)]
    if fp <= 64 * 32:            # medium: 4/8 warps
        return [(4, 1), (4, 2), (8, 1), (8, 2)]
    return [(4, 2), (4, 3), (8, 2), (8, 3)]   # large: keep the fit-critical 8-warp/deep-pipeline set


_BWD_CFGS = [triton.Config({'BD': bk}, num_warps=w, num_stages=s)   # BD-only (den kernels, BV-free)
             for bk in _BWD_BK for (w, s) in _bwd_ws(bk, 16)]
# value-looped grad kernels additionally tile the value axis: BD × BV.
_BWD_CFGS_BV = [triton.Config({'BD': bk, 'BV': bv}, num_warps=w, num_stages=s)
                for bk in _BWD_BK for bv in _BWD_BV for (w, s) in _bwd_ws(bk, bv)]
# scans have no BD knob (Sflat is a register carry, own fixed feature block) but DO need the BV knob.
# The scan footprint is BD_scan(<=64)×BV; treat as the medium/large class by BV alone.
_SCAN_CFGS = [triton.Config({'BV': bv}, num_warps=w, num_stages=s)
              for bv in _BWD_BV for (w, s) in _bwd_ws(64, bv)]


def _bv_cap(dv):
    return max(16, triton.next_power_of_2(dv))


# Feature/value tiling for the FUSED kappa/per_state kernels and the routed numerator backward. Unlike
# the simpler numerator-only `_fwd_tiled`, these kernels build [BC*BK, BV] state slices and [BT, BC*BK]
# read/write tiles with BC=16, so the dominant fp32 tile is BC·BK·BV·4 bytes (two copies live at the
# state read+write). We tile BOTH axes: cap BK<=64 (loop ND=cdiv(dqk,BK)) AND cap the value tile BV (loop
# ND_V=cdiv(dv,BV)) so BC·BK·BV stays under a conservative budget. BK=BV=16 always survives (the floor).
# Both loops are pure reduction-order / output-block reassociations → bit-faithful. (`_kappa_bk_cap`
# keeps the value-free backward — which has no BV loop — fitting by capping BK against max(BV,BT).)
def _kappa_bk_cap(dqk, dv, chunk, bc=16):
    bv = _bv_cap(dv)
    want = min(64, max(16, triton.next_power_of_2(dqk)))
    budget = 24 * 1024                       # bytes for ONE [BC*BK, max(BV,BT)] fp32 tile (several live +
    span = bc * max(bv, chunk) * 4           # the fp32 backward operands + pipelining, within the ~100KB cap)
    bk = want
    while bk > 16 and bk * span > budget:
        bk //= 2
    return bk


def _kappa_bv_tile(dqk, dv, bc=16, bk_cap=64):
    """Value tile for the kappa FORWARD (whose [BC*BK,BV] state slice IS value-looped). Largest pow2 BV in
    [16, next_pow2(dv)] keeping BC·BK·BV·4 under budget so the read + write copies of the slice both fit."""
    bk = min(bk_cap, max(16, triton.next_power_of_2(dqk)))
    want = max(16, triton.next_power_of_2(dv))
    budget = 24 * 1024                       # one [BC*BK, BV] fp32 tile; the read+write pair + rq/wk fit
    span = bc * bk * 4
    bv = want
    while bv > 16 and bv * span > budget:
        bv //= 2
    return bv


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


# Compute-dtype contract: chunk_rola hands the kernel ops fp32 operands + a `cdt` dtype-code; the bf16
# (or fp16) round happens INSIDE the opaque op, never in the torch.compile-visible glue. This is the
# fix that lands compiled grads <0.5%: with the cast in the glue, inductor fuses the cast-backward with
# the kernel-grad-consuming region in bf16, diverging the gram grads ~1% vs eager. With the cast opaque,
# inductor sees fp32→fp32 and the bf16 round is eager-deterministic. (`_DT['fp32']` is the no-autocast /
# fp64-suite path — pass-through.)
_DT = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}


def _dtcode(compute_dtype):
    return {torch.float32: 'fp32', torch.bfloat16: 'bf16', torch.float16: 'fp16'}.get(compute_dtype, 'fp32')


@torch.library.custom_op("rola::readout_rla", mutates_args=())
def _readout_rla(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 wg: torch.Tensor, rg: torch.Tensor, chunk: int, BG: int, cdt: str) -> torch.Tensor:
    """RoLA-RLA un-normalized routed readout as an opaque custom op (Triton kernels inside → no
    Dynamo graph break; inductor fuses the surrounding glue). Casts fp32 operands → `cdt` IN-op."""
    with torch.autocast('cuda', enabled=False):
        dt = _DT[cdt]
        q, k, v, wg, rg = (t.to(dt) for t in (q, k, v, wg, rg))
        return _fwd_tiled(q, k, v, wg, rg, chunk=chunk, BG=BG).to(dt)


@_readout_rla.register_fake
def _readout_rla_fake(q, k, v, wg, rg, chunk, BG, cdt):
    return q.new_empty((q.shape[0], q.shape[1], v.shape[-1]), dtype=_DT[cdt])


def _readout_rla_setup(ctx, inputs, output):
    q, k, v, wg, rg, chunk, BG, cdt = inputs
    ctx.save_for_backward(q, k, v, wg, rg)
    ctx.cdt = cdt


@torch.library.custom_op("rola::readout_rla_bwd", mutates_args=())
def _readout_rla_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     wg: torch.Tensor, rg: torch.Tensor, grad: torch.Tensor, cdt: str) -> typing.List[torch.Tensor]:  # noqa: UP006
    """Opaque backward (Triton kernels inside) so inductor doesn't trace the kernel launches. Casts the
    fp32 saved operands → `cdt` IN-op; returns fp32 grads (kept fp32 through the compiled glue)."""
    with torch.autocast('cuda', enabled=False):
        dt = _DT[cdt]
        q, k, v, wg, rg = (t.to(dt) for t in (q, k, v, wg, rg))
        dq, dk, dvv, dw, dr = _bwd_split_rla(q, k, v, wg, rg, grad, chunk=_CHUNK)
    return [dq, dk, dvv, dw, dr]


@_readout_rla_bwd.register_fake
def _readout_rla_bwd_fake(q, k, v, wg, rg, grad, cdt):
    f = torch.float32
    return [q.new_empty(q.shape, dtype=f), k.new_empty(k.shape, dtype=f),
            v.new_empty(v.shape, dtype=f), wg.new_empty(wg.shape, dtype=f), rg.new_empty(rg.shape, dtype=f)]


def _readout_rla_backward(ctx, grad):
    q, k, v, wg, rg = ctx.saved_tensors
    # Return the kernel grads in their native fp32 (the kernel computes fp32; `register_fake` declares
    # fp32). Do NOT downcast to bf16 here: the downstream normalization GLUE (the *scale backward, the
    # den-grad fork accumulation) then stays fp32, so torch.compile can't reorder it in bf16 (the ~1%
    # compiled-vs-eager grad noise). The autograd engine downcasts to the bf16 leaf ONCE, deterministically.
    dq, dk, dvv, dw, dr = _readout_rla_bwd(q, k, v, wg, rg, grad, ctx.cdt)
    return dq, dk, dvv, dw, dr, None, None, None


_readout_rla.register_autograd(_readout_rla_backward, setup_context=_readout_rla_setup)


@input_guard
def rola_rla_triton(q, k, v, r, w, chunk=None, BG=16, compute_dtype=None):
    """Un-normalized routed RLA readout via Triton. q,k:[BH,L,K] v:[BH,L,V] r,w:[BH,L,nc]
    (r=read gate, w=write gate). Differentiable (fused Triton backward). Numerator-only ⇒ returns
    [BH,L,V] at BV=next_pow2(V); the kappa/per_state caller reconstructs the global denominator as
    Σ_c rᶜ·dᶜ from the per-state den pre-pass. `compute_dtype`: pass fp32 operands + the kernel compute
    dtype to keep the bf16 round in-op (the torch.compile grad-noise fix); default None derives the code
    from the operand dtype so direct bf16/fp16 callers keep working."""
    chunk = _CHUNK_FWD if chunk is None else min(chunk, _CHUNK_FWD)
    return _readout_rla(q, k, v, w, r, chunk, BG, _dtcode(compute_dtype or q.dtype))


# ============================================================================
# In-kernel TREE-ROUTING forward (RLA). The matching BACKWARD lives just below (`_RoLARoutedFn`).
#
# Productionizes the validated prototype `proto_tree_routing.py`: instead of taking PRECOMPUTED gates
# r,w ∈ [L,nc] and forming R = r·wᵀ, the routing gram is built IN-KERNEL from the hidden state h and the
# per-level router weights Wr,Ww ∈ [D, d_model, b] (b^D = nc), never materializing the [L,nc] gates.
#
# This is a SOURCE SWAP, not a new pipeline: the inner machinery is the *same* production RLA forward
# (`_rola_fwd_intra` collapse-intra over state-blocks, `_rola_fwd_inter` NB-fused inter scan, the same
# autotune configs, the same BG state-block SMEM-tiling). The ONLY change is the block that produced the
# [BT,BG] routing tiles:  `tl.load(rg/wg)`  →  `_build_rw_tile` (per-level softmax factors gathered to
# the BG state-block via one-hot Sel maps). The reconstructed gate tiles live transiently in SRAM at the
# state-block width BG; the dominant [L,nc] gate tensor is never allocated.
#
# The factorization (proven in the prototype, validated <1e-2 vs autograd):
#   r[:, c] = Π_lvl softmax(h·Wr[lvl])[:, digit_lvl(c)],   R = r·wᵀ = ⊙_lvl (fr_lvl·fw_lvlᵀ)
# with the per-level [BT,b] softmax factors fr,fw gathered to the nc-leaf block by Sel[lvl][b, nc].
#
# BACKWARD (router-grad fold dWr,dWw,d_h) is BELOW: `_RoLARoutedFn` (mirroring the prototype's
# `_TreeRoutedFn`) wraps this forward — backward drives the validated fold kernels (`_bwd_intra_kernel`,
# `_bwd_inter_*`, `_fold_*` from proto_tree_routing.py) at the PRODUCTION state-block width (BC=BG), and
# folds the transient [BT,nc] gate-grads into dWr/dWw/d_h in-kernel (the [L,nc] grads never materialize).
# The forward builds the SAME `sel` map the bwd needs and routes through the SAME [BT,BG] factor
# reconstruction (`_build_rw_tile`) the fold recomputes.
# ============================================================================


def _build_sel(D, b, nc, device):
    """One-hot level→leaf selection maps Sel[lvl][d, leaf] = 1 iff digit_lvl(leaf)==d (big-endian
    base-b digits, matching the prototype's _digits ordering). Tiny [D,b,nc] constant — reconstructs
    the [BT,nc] gate tile from the [BT,b] per-level factors IN-KERNEL. Carries NO sequence dimension,
    so it is NOT the [L,nc] gate; identical to proto_tree_routing._build_sel."""
    sel = torch.zeros(D, b, nc, device=device, dtype=torch.float32)
    for leaf in range(nc):
        digs = [(leaf // (b ** (D - 1 - i))) % b for i in range(D)]
        for i, d in enumerate(digs):
            sel[i, d, leaf] = 1.0
    return sel


def _routing_bias(b_r, b_w, D, b, device, dtype=torch.float32):
    """Resolve the optional per-level routing bias b_r/b_w ∈ [D,b] (the affine term of softmax(h·W+b))
    to (br, bw, has_bias) for the kernels. None ⇒ a 1-element dummy (never read; HAS_BIAS=False gates
    every load) so the kernel signature stays uniform and the bias=None path is byte-identical to the
    pre-bias kernel. Mirrors proto_tree_routing._bias_args."""
    if b_r is None and b_w is None:
        dummy = torch.zeros(1, device=device, dtype=dtype)
        return dummy, dummy, False
    if b_r is None or b_w is None:
        raise ValueError("routing bias: pass both b_r and b_w, or neither")
    if tuple(b_r.shape) != (D, b) or tuple(b_w.shape) != (D, b):
        raise ValueError(f"routing bias must be [D,b]=[{D},{b}], got {tuple(b_r.shape)}/{tuple(b_w.shape)}")
    return b_r.to(dtype).contiguous(), b_w.to(dtype).contiguous(), True


def _bias_strides(br, bw, has_bias):
    """(sbr_lvl, sbr_b, sbw_lvl, sbw_b) for the kernel launch; zeros when bias is absent."""
    if not has_bias:
        return (0, 0, 0, 0)
    return (br.stride(0), br.stride(1), bw.stride(0), bw.stride(1))


@triton.jit
def _build_rw_tile(h_ptr, wr_ptr, ww_ptr, sel_ptr, offs_c, cmask,
                   bn, rows, rmask, offs_bb, bmask, d_model,
                   sh_b, sh_l, sh_d, swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
                   ssel_lvl, ssel_b, ssel_c,
                   br_ptr, bw_ptr, sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                   D: tl.constexpr, BT: tl.constexpr, BB: tl.constexpr,
                   BG: tl.constexpr, BD: tl.constexpr, NDM: tl.constexpr,
                   HAS_BIAS: tl.constexpr):
    """Build the [BT, BG] read/write routing tiles for ONE state-block (the BG-wide nc slice offs_c),
    IN-KERNEL from h + Wr,Ww — the production stand-in for `tl.load(rg/wg)`. Ports the prototype's
    in-kernel factor construction (proto_tree_routing._build_factors), with the tile width = the
    production state-block BG (not the prototype's BC). For each level: logits = h·W (+ optional bias
    b_r/b_w ∈ [D,b], added before the softmax → softmax(h·W+b)) — loop BD-blocks of d_model so any
    d_model fits SMEM, softmax over the b branches (pad cols masked to -inf → vanish), then gather the
    [BT,b] factor to the BG leaves of THIS block via the one-hot Sel slice and Hadamard-accumulate.
    r_tile,w_tile are returned masked to cmask (nc-tail cols → 0). Transient SRAM, [BT,BG]."""
    r_tile = tl.full([BT, BG], 1.0, dtype=tl.float32)
    w_tile = tl.full([BT, BG], 1.0, dtype=tl.float32)
    for lvl in range(D):
        lr = tl.zeros([BT, BB], dtype=tl.float32)
        lw = tl.zeros([BT, BB], dtype=tl.float32)
        for dm in range(NDM):
            offs_dm = dm * BD + tl.arange(0, BD)
            mmask = offs_dm < d_model
            hc = tl.load(h_ptr + bn * sh_b + rows[:, None] * sh_l + offs_dm[None, :] * sh_d,
                         mask=rmask[:, None] & mmask[None, :], other=0.0)
            wr = tl.load(wr_ptr + lvl * swr_lvl + offs_dm[:, None] * swr_d + offs_bb[None, :] * swr_b,
                         mask=mmask[:, None] & bmask[None, :], other=0.0)
            ww = tl.load(ww_ptr + lvl * sww_lvl + offs_dm[:, None] * sww_d + offs_bb[None, :] * sww_b,
                         mask=mmask[:, None] & bmask[None, :], other=0.0)
            lr += tl.dot(hc, wr)
            lw += tl.dot(hc, ww)
        if HAS_BIAS:
            brc = tl.load(br_ptr + lvl * sbr_lvl + offs_bb * sbr_b, mask=bmask, other=0.0)
            bwc = tl.load(bw_ptr + lvl * sbw_lvl + offs_bb * sbw_b, mask=bmask, other=0.0)
            lr += brc[None, :]
            lw += bwc[None, :]
        neg = tl.full([BT, BB], float('-inf'), dtype=tl.float32)
        lr = tl.where(bmask[None, :], lr, neg)
        lw = tl.where(bmask[None, :], lw, neg)
        er = tl.exp(lr - tl.max(lr, axis=1)[:, None])
        ew = tl.exp(lw - tl.max(lw, axis=1)[:, None])
        fr = er / tl.sum(er, axis=1)[:, None]   # [BT, BB] read-gate level factor
        fw = ew / tl.sum(ew, axis=1)[:, None]   # [BT, BB] write-gate level factor
        sel = tl.load(sel_ptr + lvl * ssel_lvl + offs_bb[:, None] * ssel_b + offs_c[None, :] * ssel_c,
                      mask=bmask[:, None] & cmask[None, :], other=0.0)   # [BB, BG] one-hot
        r_tile *= tl.dot(fr, sel)
        w_tile *= tl.dot(fw, sel)
    r_tile = tl.where(cmask[None, :], r_tile, 0.0)
    w_tile = tl.where(cmask[None, :], w_tile, 0.0)
    return r_tile, w_tile


@triton.autotune(configs=_AT_CFGS, key=_SCAN_KEY, **autotune_cache_kwargs)  # _SCAN_KEY: +USE_G (RLA/GLA split)
@triton.jit
def _rola_routed_fwd_intra(q_ptr, k_ptr, v_ptr, h_ptr, wr_ptr, ww_ptr, sel_ptr, ld_ptr, outa_ptr,
                           br_ptr, bw_ptr,
                           L, dqk, dv, nc, d_model,
                           sq_b, sq_l, sq_d, sv_b, sv_l, sv_d,
                           sh_b, sh_l, sh_d, swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
                           ssel_lvl, ssel_b, ssel_c, sg_b, sg_l, sg_c, soa_b, soa_l, soa_v,
                           sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                           D: tl.constexpr, bb_: tl.constexpr, BB: tl.constexpr,
                           BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                           BG: tl.constexpr, BD: tl.constexpr,
                           ND: tl.constexpr, NB: tl.constexpr, NDM: tl.constexpr,
                           HAS_BIAS: tl.constexpr, USE_G: tl.constexpr):
    """TREE-ROUTED intra: byte-for-byte the production `_rola_fwd_intra` collapse-intra — content gram G
    built ONCE, the full routing gram R accumulated over state-blocks IN-KERNEL, A=G⊙R⊙causal, o=A·v —
    with the ONLY change being R's source: each state-block's [BT,BG] r/w tiles are BUILT from h+Wr,Ww via
    `_build_rw_tile` instead of `tl.load(rg/wg)`. nc collapses; output is [B,L,dv].
    USE_G (GLA, #30 V1): the per-block gram uses the DECAYED gates rt=rgc·e^a, wt=wgc·e^-a (a = intra-chunk
    cumsum of the per-state log-decay ld over this block's c-columns) — exactly `_rola_gla_fwd_intra`. The
    nc-collapse still holds (each block's decayed [BT,BG]·[BG,BT] gram is still a [BT,BT] partial)."""
    b = tl.program_id(0)
    t = tl.program_id(1)
    offs_t = tl.arange(0, BT)
    offs_v = tl.arange(0, BV)
    offs_bb = tl.arange(0, BB)
    bmask = offs_bb < bb_
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
        rgc, wgc = _build_rw_tile(h_ptr, wr_ptr, ww_ptr, sel_ptr, offs_c, cmask,
                                  b, rows, rmask, offs_bb, bmask, d_model,
                                  sh_b, sh_l, sh_d, swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
                                  ssel_lvl, ssel_b, ssel_c,
                                  br_ptr, bw_ptr, sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                                  D, BT, BB, BG, BD, NDM, HAS_BIAS)
        if USE_G:
            ldc = tl.load(ld_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                          mask=rmask[:, None] & cmask[None, :], other=0.0)
            a = tl.cumsum(ldc, axis=0)
            rgc = rgc * tl.exp(a)
            wgc = wgc * tl.exp(-a)
        R += tl.dot(rgc.to(tl.float32), tl.trans(wgc))
    vc = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                 mask=rmask[:, None] & (offs_v[None, :] < dv), other=0.0)
    causal = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
    A = G * R * causal
    o = tl.dot(A.to(vc.dtype), vc)
    tl.store(outa_ptr + b*soa_b + rows[:, None]*soa_l + offs_v[None, :]*soa_v,
             o, mask=rmask[:, None] & (offs_v[None, :] < dv))


@triton.autotune(configs=_SCAN_CFGS, key=_SCAN_KEY, reset_to_zero=['outa_ptr'],  # _SCAN_KEY: +USE_G
                 prune_configs_by={'early_config_prune': _prune_bv}, **autotune_cache_kwargs)
@triton.jit
def _rola_routed_fwd_inter(q_ptr, k_ptr, v_ptr, h_ptr, wr_ptr, ww_ptr, sel_ptr, ld_ptr, outa_ptr,
                           br_ptr, bw_ptr,
                           L, dqk, dv: tl.constexpr, nc, d_model,
                           sq_b, sq_l, sq_d, sv_b, sv_l, sv_d,
                           sh_b, sh_l, sh_d, swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
                           ssel_lvl, ssel_b, ssel_c, sg_b, sg_l, sg_c, soa_b, soa_n, soa_l, soa_v,
                           sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                           D: tl.constexpr, bb_: tl.constexpr, BB: tl.constexpr,
                           BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                           BVF: tl.constexpr, BG: tl.constexpr, BD: tl.constexpr,
                           NCH: tl.constexpr, NDM: tl.constexpr,
                           HAS_BIAS: tl.constexpr, USE_G: tl.constexpr):
    """TREE-ROUTED inter: byte-for-byte the production `_rola_fwd_inter` NB-fused inter scan — one
    (batch, state-block, FEATURE-block) carries Sd[BK, BG*BV] across chunks, o_inter atomic-accumulated
    into the shared [B,L,dv] buffer, value-OUTER over cdiv(dv,BV) — with the ONLY change being the read/
    write gate source: the [BT,BG] rgc/wgc tiles are BUILT from h+Wr,Ww via `_build_rw_tile` instead of
    `tl.load(rg/wg)`. Same state scan, same SMEM bound (BK + value-tile BV), same autotune/reset_to_zero.
    USE_G (GLA, #30 V1): the carried state decays by decvec=e^Λ each chunk, the read uses rt=rgc·e^a, the
    write uses w_end=wgc·e^{Λ-a} (a = intra-chunk cumsum, Λ = chunk-total ld) — exactly `_rola_gla_fwd_inter`."""
    b = tl.program_id(0)
    sb = tl.program_id(1)
    d0 = tl.program_id(2)
    ND_V = (dv + BV - 1) // BV
    offs_t = tl.arange(0, BT)
    offs_d = d0 * BK + tl.arange(0, BK)
    dmask = offs_d < dqk
    offs_bb = tl.arange(0, BB)
    bmask = offs_bb < bb_
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
            rgc, wgc = _build_rw_tile(h_ptr, wr_ptr, ww_ptr, sel_ptr, offs_c, cmask,
                                      b, rows, rmask, offs_bb, bmask, d_model,
                                      sh_b, sh_l, sh_d, swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
                                      ssel_lvl, ssel_b, ssel_c,
                                  br_ptr, bw_ptr, sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                                  D, BT, BB, BG, BD, NDM, HAS_BIAS)
            if USE_G:
                ldc = tl.load(ld_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                              mask=rmask[:, None] & cmask[None, :], other=0.0)
                a = tl.cumsum(ldc, axis=0)
                Lam = tl.sum(tl.where(offs_t[:, None] == (BT - 1), a, 0.0), axis=0)
                rt = rgc * tl.exp(a)
            else:
                rt = rgc
            P = tl.dot(qc, Sd.to(qc.dtype))
            P3 = tl.reshape(P, [BT, BG, BV])
            o_inter = tl.sum(P3 * rt[:, :, None], axis=1)
            tl.atomic_add(outa_ptr + b*soa_b + rows[:, None]*soa_l + offs_v[None, :]*soa_v,
                          o_inter, mask=rmask[:, None] & vmask[None, :])
            kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                         mask=rmask[:, None] & dmask[None, :], other=0.0)
            vc = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                         mask=rmask[:, None] & vmask[None, :], other=0.0)
            if USE_G:
                w_end = wgc * tl.exp(Lam[None, :] - a)
                WV = tl.reshape(w_end[:, :, None] * vc[:, None, :], [BT, BG * BV])
                decvec = tl.reshape(tl.exp(Lam)[:, None] * tl.full([BG, BV], 1.0, tl.float32), [BG * BV])
                Sd = decvec[None, :] * Sd + tl.dot(tl.trans(kc), WV.to(kc.dtype))
            else:
                WV = tl.reshape(wgc[:, :, None] * vc[:, None, :], [BT, BG * BV])
                Sd += tl.dot(tl.trans(kc), WV.to(kc.dtype))


def _routed_fwd_tiled(q, k, v, h, Wr, Ww, D, b, sel, chunk, BG, BK=64, b_r=None, b_w=None, ld=None):
    """D-tiled numerator-only tree-routed forward — the routed twin of `_fwd_tiled`. Identical
    intra/inter dispatch (separate buffers, intra store / inter atomic reset_to_zero, value-tiling), with
    the routing gram source swapped to (h, Wr, Ww) via the routed kernels. Optional routing bias b_r/b_w
    ∈ [D,b] (softmax(h·W+b)). Returns [B,L,dv] at BV=next_pow2(dv); the [L,nc] gates are never allocated.
    Optional per-state log-decay ld:[B,L,nc] (GLA, #30 V1) → USE_G decayed scan; ld=None is RLA (USE_G=False)."""
    B, L, dqk = q.shape
    dv = v.shape[-1]
    d_model = h.shape[-1]
    nc = b ** D
    BV = max(16, triton.next_power_of_2(dv))
    BK = min(BK, max(16, triton.next_power_of_2(dqk)))
    BD = max(16, triton.next_power_of_2(d_model))
    BB = max(16, triton.next_power_of_2(b))
    ND = triton.cdiv(dqk, BK)
    NB = triton.cdiv(nc, BG)
    NCH = triton.cdiv(L, chunk)
    NDM = triton.cdiv(d_model, BD)
    q, k, v, h, Wr, Ww = [x.contiguous() for x in (q, k, v, h, Wr, Ww)]
    use_g = ld is not None
    ld = (ld.clamp(min=_GLA_FLOOR).contiguous() if use_g
          else q.new_zeros(B, L, nc))   # USE_G=False: ld unread (kernel skips the load), pass a stub
    sg = (ld.stride(0), ld.stride(1), ld.stride(2))
    br, bw, has_bias = _routing_bias(b_r, b_w, D, b, q.device, dtype=q.dtype)
    sbias = _bias_strides(br, bw, has_bias)
    out_intra = torch.zeros(B, L, BV, device=q.device, dtype=torch.float32)
    out_inter = torch.zeros(B, L, BV, device=q.device, dtype=torch.float32)
    so_a = (out_intra.stride(0), out_intra.stride(1), out_intra.stride(2))
    so_e = (out_inter.stride(0), 0, out_inter.stride(1), out_inter.stride(2))
    base = (q.stride(0), q.stride(1), q.stride(2), v.stride(0), v.stride(1), v.stride(2))
    route = (h.stride(0), h.stride(1), h.stride(2),
             Wr.stride(0), Wr.stride(1), Wr.stride(2), Ww.stride(0), Ww.stride(1), Ww.stride(2),
             sel.stride(0), sel.stride(1), sel.stride(2))
    _rola_routed_fwd_intra[(B, NCH)](q, k, v, h, Wr, Ww, sel, ld, out_intra, br, bw, L, dqk, dv, nc, d_model,
                                     *base, *route, *sg, *so_a, *sbias,
                                     D=D, bb_=b, BB=BB, BT=chunk, BK=BK, BV=BV, BG=BG, BD=BD,
                                     ND=ND, NB=NB, NDM=NDM, HAS_BIAS=has_bias, USE_G=use_g)
    _rola_routed_fwd_inter[(B, NB, ND)](q, k, v, h, Wr, Ww, sel, ld, out_inter, br, bw, L, dqk, dv, nc, d_model,
                                        *base, *route, *sg, *so_e, *sbias,
                                        D=D, bb_=b, BB=BB, BT=chunk, BK=BK, BVF=BV, BG=BG, BD=BD,
                                        NCH=NCH, NDM=NDM, HAS_BIAS=has_bias, USE_G=use_g)
    return out_intra[..., :dv] + out_inter[..., :dv]


def _rola_rla_routed_fwd(q, k, v, h, Wr, Ww, D, b, chunk=None, BG=16):
    """Un-normalized TREE-ROUTED RLA readout via Triton (the optimized production forward) — PLAIN
    (no autograd). The routing gram is built IN-KERNEL from the hidden state h + per-level router weights
    Wr,Ww ∈ [D,d_model,b] (b^D=nc); the [L,nc] gates are never materialized.

    q,k:[BH,L,K]  v:[BH,L,V]  h:[BH,L,d_model]  Wr,Ww:[D,d_model,b].  Returns [BH,L,V] at BV=next_pow2(V).
    Flat (D=1, b=nc) is the fused equivalent of `rola_rla_triton(q,k,v,r,w)` with r,w the D=1 router's
    explicit softmax gates."""
    chunk = _CHUNK_FWD if chunk is None else min(chunk, _CHUNK_FWD)
    nc = b ** D
    sel = _build_sel(D, b, nc, q.device)
    return _routed_fwd_tiled(q, k, v, h, Wr, Ww, D, b, sel, chunk=chunk, BG=BG)


# ============================================================================
# In-kernel TREE-ROUTING BACKWARD (RLA) — the router-grad fold that makes the production routed path
# trainable end-to-end. Given d_o, compute dq,dk,dv,d_h,dWr,dWw with the [L,nc] gate grads dr,dw NEVER
# materialized: they live only as transient [BT,BG] tiles (production state-block width), gathered
# through the SAME one-hot `Sel` map + the SAME `_build_rw_tile` factor reconstruction the forward uses,
# and folded in-kernel (softmax jacobian → atomic dWr/dWw/d_h).
#
# The fold math is the PROVEN prototype backward (proto_tree_routing.py: _bwd_intra_kernel,
# _bwd_inter_state_kernel, _bwd_inter_read_kernel, _fold_kernel — validated <1e-2 vs autograd for
# flat/square/tree). Those kernels are GENERIC over the nc-block width (their `BC` constexpr); the ONLY
# adaptation is to drive them at the PRODUCTION state-block width BC=BG (not the prototype's fixed
# BC=16) and through this module's `_build_sel`, so the backward routes through byte-identical factor
# reconstruction to the production forward (`_build_rw_tile` ≡ proto `_build_factors`). dr,dw stay
# transient per-chunk [B,chunk,nc] scratch tiles (gdr/gdw), OVERWRITTEN every chunk — never [L,nc].
#
# The per-chunk pre-state snapshots S_j ∈ [B,nc,dqk,dv] the reverse-scan needs are the RECURRENT state
# (NOT the gates), built by a dedicated in-kernel snapshot scan `_rola_routed_snap` (write gates built
# in-kernel via `_build_rw_tile` — gates never materialized in the snapshot pass either). The forward
# saves these snapshots; backward consumes them. The validated fold kernels are imported at module top
# (_proto_bwd_intra / _proto_bwd_inter_state / _proto_bwd_inter_read / _proto_fold), reused VERBATIM.
# ============================================================================


@triton.jit
def _rola_routed_snap_kernel(h_ptr, k_ptr, v_ptr, wr_ptr, ww_ptr, sel_ptr, ld_ptr, s_ptr, snap_ptr,
                             br_ptr, bw_ptr,
                             L, d_model, dqk, dv, nc,
                             sh_b, sh_l, sh_d, sq_b, sq_l, sq_d, sv_b, sv_l, sv_d,
                             swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
                             ssel_lvl, ssel_b, ssel_c, sg_b, sg_l, sg_c, ss_b, ss_c, ss_k, ss_v,
                             snp_b, snp_n, snp_c, snp_k, snp_v,
                             sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                             D: tl.constexpr, b: tl.constexpr, BB: tl.constexpr,
                             BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BD: tl.constexpr,
                             BG: tl.constexpr, NCBLK: tl.constexpr, ND: tl.constexpr,
                             NCH: tl.constexpr, NDM: tl.constexpr,
                             HAS_BIAS: tl.constexpr, USE_G: tl.constexpr):
    """Per-chunk PRE-STATE snapshot scan for the routed backward. One program per batch carries the flat
    Kronecker state S[nc,dqk,dv] across chunks; BEFORE each chunk's write it copies S into snap[:,chunk]
    (the state the reverse-scan reads). The write gates are built IN-KERNEL via `_build_rw_tile` over BG
    state-blocks (the [L,nc] gates are never materialized here either). Mirrors the proto chunk kernel's
    state update (S += Σ_t wₜᶜ kₜ⊗vₜ), at the production state-block width BG.
    USE_G (GLA, #30 V1): the per-c state decays by e^{Λ_c} each chunk and the write is e^{Λ_c-a}-weighted
    (S[c] ← e^{Λ_c}S[c] + Σ w_end k⊗v) — exactly the decayed inter scan; the snapshot is still PRE-update."""
    pid_b = tl.program_id(0)
    ND_V = (dv + BV - 1) // BV       # value-blocks (BV is the value TILE; ND_V==1 ⇒ the un-tiled kernel)
    offs_t = tl.arange(0, BT)
    offs_bb = tl.arange(0, BB)
    offs_c = tl.arange(0, BG)
    bmask = offs_bb < b
    for ci in range(NCH):
        t_start = ci * BT
        rows = t_start + offs_t
        rmask = rows < L
        for cb in range(NCBLK):
            cols = cb * BG + offs_c
            cmask = cols < nc
            _, w_tile = _build_rw_tile(h_ptr, wr_ptr, ww_ptr, sel_ptr, cols, cmask,
                                       pid_b, rows, rmask, offs_bb, bmask, d_model,
                                       sh_b, sh_l, sh_d, swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
                                       ssel_lvl, ssel_b, ssel_c,
                                       br_ptr, bw_ptr, sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                                       D, BT, BB, BG, BD, NDM, HAS_BIAS)
            if USE_G:
                ldc = tl.load(ld_ptr + pid_b*sg_b + rows[:, None]*sg_l + cols[None, :]*sg_c,
                              mask=rmask[:, None] & cmask[None, :], other=0.0)
                a = tl.cumsum(ldc, axis=0)
                Lam = tl.sum(tl.where(offs_t[:, None] == (BT - 1), a, 0.0), axis=0)   # [BG] chunk-total
                w_tile = w_tile * tl.exp(Lam[None, :] - a)                            # w_end
                dec_c = tl.exp(Lam)                                                   # [BG] per-c carry
            # snapshot PRE-update state + the state write (S += Σ wᶜ k⊗v); dqk → loop BK-blocks AND value →
            # loop ND_V so the [BG*BK,BV] s_flat slice stays bounded by BK·BV. wk[BT,BG*BK] value-free.
            for d0 in range(ND):
                offs_k = d0 * BK + tl.arange(0, BK)
                kmask = offs_k < dqk
                kc = tl.load(k_ptr + pid_b * sq_b + rows[:, None] * sq_l + offs_k[None, :] * sq_d,
                             mask=rmask[:, None] & kmask[None, :], other=0.0)
                ckv = tl.reshape(cols[:, None] * dqk + offs_k[None, :], [BG * BK])
                ckmask = tl.reshape(cmask[:, None] & kmask[None, :], [BG * BK])
                wk = tl.reshape(w_tile[:, :, None] * kc[:, None, :], [BT, BG * BK])
                if USE_G:
                    # per-(c,k) carry: broadcast dec_c[BG] over the BK feature rows of each c → [BG*BK]
                    deckv = tl.reshape(dec_c[:, None] * tl.full([BG, BK], 1.0, tl.float32), [BG * BK])
                for vb in range(ND_V):
                    offs_v = vb * BV + tl.arange(0, BV)
                    vmask = offs_v < dv
                    vc = tl.load(v_ptr + pid_b * sv_b + rows[:, None] * sv_l + offs_v[None, :] * sv_d,
                                 mask=rmask[:, None] & vmask[None, :], other=0.0)
                    s_flat = tl.load(s_ptr + pid_b * ss_b + ckv[:, None] * ss_k + offs_v[None, :] * ss_v,
                                     mask=ckmask[:, None] & vmask[None, :], other=0.0)
                    tl.store(snap_ptr + pid_b * snp_b + ci * snp_n + ckv[:, None] * snp_k
                             + offs_v[None, :] * snp_v, s_flat, mask=ckmask[:, None] & vmask[None, :])
                    dS = tl.dot(tl.trans(wk).to(vc.dtype), vc)
                    s_new = (deckv[:, None] * s_flat + dS) if USE_G else (s_flat + dS)
                    tl.store(s_ptr + pid_b * ss_b + ckv[:, None] * ss_k + offs_v[None, :] * ss_v,
                             s_new, mask=ckmask[:, None] & vmask[None, :])


def _routed_snapshots(q, k, v, h, Wr, Ww, D, b, sel, chunk, BG, br=None, bw=None, has_bias=False, ld=None):
    """Build per-chunk pre-state snapshots [B, NCH, nc, dqk, dv] for the routed backward — the recurrent
    STATE (NOT the gates), via the in-kernel snapshot scan. Write gates built in-kernel (never [L,nc]).
    The write gate honors the optional routing bias (softmax(h·Ww+b_w)) so snapshots match the fwd.
    Optional per-state log-decay ld:[B,L,nc] (GLA) → USE_G decayed state recurrence."""
    B, L, dqk = q.shape
    dv = v.shape[-1]
    d_model = h.shape[-1]
    nc = b ** D
    BK = _kappa_bk_cap(dqk, dv, chunk, BG)   # feature-tile (loop ND) — bounds the [BT,BG*BK] tiles
    BV = _kappa_bv_tile(dqk, dv, BG, BK)     # value-tile (loop ND_V) so the [BG*BK,BV] state slice fits
    BD = max(16, triton.next_power_of_2(d_model))
    BB = max(16, triton.next_power_of_2(b))
    NCBLK = triton.cdiv(nc, BG)
    ND = triton.cdiv(dqk, BK)
    NDM = triton.cdiv(d_model, BD)
    NCH = triton.cdiv(L, chunk)
    if br is None:
        br, bw, has_bias = _routing_bias(None, None, D, b, q.device, dtype=q.dtype)
    sbias = _bias_strides(br, bw, has_bias)
    use_g = ld is not None
    ld = (ld.clamp(min=_GLA_FLOOR).contiguous() if use_g else q.new_zeros(B, L, nc))
    sg = (ld.stride(0), ld.stride(1), ld.stride(2))
    S = torch.zeros(B, nc, dqk, dv, device=q.device, dtype=torch.float32)
    snap = torch.zeros(B, NCH, nc, dqk, dv, device=q.device, dtype=torch.float32)
    _rola_routed_snap_kernel[(B,)](
        h, k, v, Wr, Ww, sel, ld, S, snap, br, bw,
        L, d_model, dqk, dv, nc,
        h.stride(0), h.stride(1), h.stride(2), q.stride(0), q.stride(1), q.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        Wr.stride(0), Wr.stride(1), Wr.stride(2), Ww.stride(0), Ww.stride(1), Ww.stride(2),
        sel.stride(0), sel.stride(1), sel.stride(2), *sg,
        S.stride(0), S.stride(1), S.stride(2), S.stride(3),
        snap.stride(0), snap.stride(1), snap.stride(2), snap.stride(3), snap.stride(4),
        *sbias,
        D=D, b=b, BB=BB, BT=chunk, BK=BK, BV=BV, BD=BD, BG=BG, NCBLK=NCBLK, ND=ND, NCH=NCH, NDM=NDM,
        HAS_BIAS=has_bias, USE_G=use_g, num_warps=4, num_stages=1)
    return snap


def _rola_rla_routed_bwd(q, k, v, h, Wr, Ww, do, D, b, chunk, BG, b_r=None, b_w=None, ld=None):
    """Tree-routed RLA backward at the PRODUCTION state-block width BC=BG. Drives the validated proto
    fold kernels: one intra launch (dq,dk,dv-intra + dr,dw-intra fold) and a sequential reverse state-
    adjoint scan over chunks (state-update bwd → readout bwd → router-grad fold). The gate-grads dr,dw
    live only as transient [B,chunk,nc] scratch (gdr/gdw), OVERWRITTEN each chunk — never [L,nc]. With an
    optional routing bias b_r/b_w ∈ [D,b], also folds db_r/db_w (transient, never [L,nc]). Returns
    dq,dk,dv,d_h,dWr,dWw (and db_r,db_w when biased) — all fp32.

    USE_G (GLA, #30): optional per-state log-decay ld:[B,L,nc] → the decayed routed backward — the proto
    kernels use the DECAYED gates (rt=r·eᵃ, w_end=w·e^{Λ−a}), the running dS adjoint is decayed by e^Λ
    between the state-bwd and read-bwd halves (the reverse of the forward's decvec carry), and a persistent
    gda[B,L,nc] buffer collects the per-token log-decay adjoints (dart/da_wt/da_wend/dlam) which the driver
    reverse-cumsums (intra-chunk) into dld. ld=None is byte-identical to the RLA path (USE_G=False)."""
    use_g = ld is not None
    B, L, d_model = h.shape
    dqk = q.shape[-1]
    dv = v.shape[-1]
    nc = b ** D
    BVO = max(16, triton.next_power_of_2(dv))   # full padded value width (intra bwd + buffer alloc)
    BD = max(16, triton.next_power_of_2(d_model))
    BB = max(16, triton.next_power_of_2(b))
    BC = BG                                  # PRODUCTION tile width (the adaptation; proto used fixed 16)
    # intra bwd builds only [BT,BK]/[BT,BT] tiles (no BC*BK), so it keeps the FULL feature/value width (no
    # ND/ND_V loop). The inter state/read + snapshot kernels build [BC*BK,BV]/[BT,BC*BK] tiles → cap BK
    # (loop ND) AND tile the value axis BV (loop ND_V) so the state slices fit; mirrors `_kappa_routed_bwd`.
    BK_full = max(16, triton.next_power_of_2(dqk))
    BK = _kappa_bk_cap(dqk, dv, min(chunk, _CHUNK), BC)   # feature-tile (loop ND)
    BV = _kappa_bv_tile(dqk, dv, BC, BK)                 # value-tile (loop ND_V) so [BC*BK,BV] fits
    NCBLK = triton.cdiv(nc, BC)
    ND = triton.cdiv(dqk, BK)
    NDM = triton.cdiv(d_model, BD)
    NCH = triton.cdiv(L, chunk)
    # Router-grad fold accumulation in fp32 (FLA backward idiom): the deep-Hadamard softmax jacobian
    # (dfr/fr with D levels) is precision-sensitive, so the in-kernel logit recompute (logits=h·W) and
    # all gram dots run with fp32 operands — the bf16-rounded saved inputs are upcast here. Keeps grads
    # at the bf16 NOISE floor (<1e-2) rather than the bf16-operand floor of the fold dots.
    q, k, v, h, Wr, Ww, do = (x.float().contiguous() for x in (q, k, v, h, Wr, Ww, do))
    br, bw, has_bias = _routing_bias(b_r, b_w, D, b, q.device, dtype=torch.float32)
    sbias = _bias_strides(br, bw, has_bias)
    # The fold kernels run fp32 operands (precision floor of the deep-Hadamard jacobian); fp32 doubles
    # per-program SMEM vs the proto's bf16 regime, so the backward chunk is capped at _CHUNK (32) — the
    # GLA/all-backwards device constant — independent of the forward chunk (each is just a tiling of the
    # SAME sequence; the chunked readout is chunk-size invariant). Avoids the BT=64-fp32 SMEM wall.
    chunk = min(chunk, _CHUNK)
    # The inter [BT,BC*BK] read/write tiles scale with BT; at large dqk (small BK, many ND) the fp32
    # tiles + 3D elementwise temporaries exceed the cap at BT=32, so cap BT further when BC*BK is large
    # (the chunked reverse-scan is chunk-size invariant). Cheap dims keep the full _CHUNK.
    if BK * BC * chunk * 4 > 16 * 1024:
        chunk = min(chunk, _KAPPA_BWD_CHUNK)
    NCH = triton.cdiv(L, chunk)
    sel = _build_sel(D, b, nc, q.device)
    # ld:[B,L,nc] (GLA) clamped to the fp32 decay floor; RLA passes a zero stub the kernels skip (USE_G=False).
    ld = (ld.float().clamp(min=_GLA_FLOOR).contiguous() if use_g else q.new_zeros(B, L, nc))
    sgl = (ld.stride(0), ld.stride(1), ld.stride(2))
    # per-chunk pre-state snapshots (the recurrent STATE, not gates) for the reverse-scan — recomputed
    # here at the SAME fp32 router precision as the fold, so fwd/bwd routing is bit-consistent.
    snap = _routed_snapshots(q, k, v, h, Wr, Ww, D, b, sel, chunk, BG, br, bw, has_bias,
                             ld=(ld if use_g else None))
    # dvv/dq/dk are written by the fold kernels at v's / q's row strides (sv_l=v.stride(1)=dv,
    # sq_l=q.stride(1)=dqk), so they MUST be allocated at the TRUE dv/dqk width (NOT padded BV/BK) or
    # the row layout corrupts for non-pow2 dv/dqk (e.g. dv=24→BV=32). The in-kernel [BT,BV]/[BT,BK]
    # accumulators store only the masked :dv/:dqk lanes (vmask/kmask). Mirrors `_kappa_routed_bwd`.
    dq = torch.zeros(B, L, dqk, device=q.device, dtype=torch.float32)
    dk = torch.zeros(B, L, dqk, device=q.device, dtype=torch.float32)
    dvv = torch.zeros(B, L, dv, device=q.device, dtype=torch.float32)
    dh = torch.zeros(B, L, d_model, device=q.device, dtype=torch.float32)
    dWr = torch.zeros(D, d_model, b, device=q.device, dtype=torch.float32)
    dWw = torch.zeros(D, d_model, b, device=q.device, dtype=torch.float32)
    dbr = torch.zeros(D, b, device=q.device, dtype=torch.float32)   # routing-bias grads [D,b] (transient fold)
    dbw = torch.zeros(D, b, device=q.device, dtype=torch.float32)
    # gda[B,L,nc]: persistent per-token log-decay adjoint accumulator (USE_G) — the intra kernel writes
    # its da-pieces over all chunks (parallel), the inter kernels add theirs per chunk; reverse-cumsummed
    # per chunk into dld at the end. RLA leaves it zero (a 1-col stub) and dld is unused.
    gda = torch.zeros(B, L, nc, device=q.device, dtype=torch.float32) if use_g \
        else q.new_zeros(B, 1, 1)
    sga = (gda.stride(0), gda.stride(1), gda.stride(2))
    common = dict(D=D, b=b, BB=BB, BT=chunk, BK=BK_full, BV=BVO, BD=BD, BC=BC,
                  NCBLK=NCBLK, ND=triton.cdiv(dqk, BK_full), NDM=NDM, HAS_BIAS=has_bias,
                  USE_G=use_g, num_warps=4, num_stages=1)
    _proto_bwd_intra[(B, NCH)](
        h, q, k, v, Wr, Ww, sel, ld, do, dq, dk, dvv, dh, dWr, dWw, gda,
        br, bw, dbr, dbw,
        L, d_model, dqk, dv, nc,
        h.stride(0), h.stride(1), h.stride(2), q.stride(0), q.stride(1), q.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        Wr.stride(0), Wr.stride(1), Wr.stride(2), Ww.stride(0), Ww.stride(1), Ww.stride(2),
        sel.stride(0), sel.stride(1), sel.stride(2), *sgl, *sga,
        do.stride(0), do.stride(1), do.stride(2), dh.stride(0), dh.stride(1), dh.stride(2),
        *sbias,
        **common)
    dS = torch.zeros(B, nc, dqk, dv, device=q.device, dtype=torch.float32)
    gdr = torch.zeros(B, chunk, nc, device=q.device, dtype=torch.float32)   # transient, OVERWRITTEN/chunk
    gdw = torch.zeros(B, chunk, nc, device=q.device, dtype=torch.float32)
    dld = torch.zeros(B, L, nc, device=q.device, dtype=torch.float32) if use_g else None
    fold_common = dict(D=D, b=b, BB=BB, BT=chunk, BC=BC, BD=BD, NCBLK=NCBLK, NDM=NDM,
                       HAS_BIAS=has_bias, num_warps=4, num_stages=1)
    inter_common = dict(D=D, b=b, BB=BB, BT=chunk, BK=BK, BV=BV, BD=BD, BC=BC,
                        NCBLK=NCBLK, ND=ND, NDM=NDM, HAS_BIAS=has_bias, USE_G=use_g,
                        num_warps=4, num_stages=1)
    for c in reversed(range(NCH)):
        Sj = snap[:, c].contiguous()
        # state-bwd reads dS = adjoint S_{j+1} (pre-decvec) → dk,dv,gdw + (USE_G) the carry/w_end da-pieces.
        _proto_bwd_inter_state[(B,)](
            h, k, v, Wr, Ww, sel, ld, Sj, dS, dk, dvv, gdw, gda, br, bw,
            L, d_model, dqk, dv, nc, c * chunk,
            h.stride(0), h.stride(1), h.stride(2), q.stride(0), q.stride(1), q.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            Wr.stride(0), Wr.stride(1), Wr.stride(2), Ww.stride(0), Ww.stride(1), Ww.stride(2),
            sel.stride(0), sel.stride(1), sel.stride(2), *sgl,
            dS.stride(0), dS.stride(1), dS.stride(2), dS.stride(3),
            gdr.stride(0), gdr.stride(1), gdr.stride(2), *sga, *sbias,
            **inter_common)
        if use_g:
            # decay the running dS adjoint by decvec=e^{Λ_c} (per state, broadcast over dqk×dv) — the
            # reverse of the forward's state carry S_{j+1}=e^Λ S_j + ΔS. Must run AFTER state-bwd reads
            # the S_{j+1} adjoint (and its ZdZ) and BEFORE read-bwd folds dS_read → adjoint S_j.
            rows = slice(c * chunk, min(c * chunk + chunk, L))
            Lam_c = ld[:, rows].sum(dim=1)                          # [B,nc] chunk-total per state
            dS = dS * torch.exp(Lam_c)[:, :, None, None]
        _proto_bwd_inter_read[(B,)](
            h, q, Wr, Ww, sel, ld, Sj, dS, do, dq, gdr, gda, br, bw,
            L, d_model, dqk, dv, nc, c * chunk,
            h.stride(0), h.stride(1), h.stride(2), q.stride(0), q.stride(1), q.stride(2),
            Wr.stride(0), Wr.stride(1), Wr.stride(2), Ww.stride(0), Ww.stride(1), Ww.stride(2),
            sel.stride(0), sel.stride(1), sel.stride(2), *sgl,
            Sj.stride(0), Sj.stride(1), Sj.stride(2), Sj.stride(3),
            do.stride(0), do.stride(1), do.stride(2),
            gdr.stride(0), gdr.stride(1), gdr.stride(2), *sga, *sbias,
            **inter_common)
        _proto_fold[(B,)](
            h, Wr, Ww, sel, gdr, gdw, dh, dWr, dWw, br, bw, dbr, dbw,
            L, d_model, nc, c * chunk,
            h.stride(0), h.stride(1), h.stride(2),
            Wr.stride(0), Wr.stride(1), Wr.stride(2), Ww.stride(0), Ww.stride(1), Ww.stride(2),
            sel.stride(0), sel.stride(1), sel.stride(2),
            gdr.stride(0), gdr.stride(1), gdr.stride(2), dh.stride(0), dh.stride(1), dh.stride(2),
            *sbias,
            **fold_common)
    if use_g:
        # dld = intra-chunk reverse-cumsum of the assembled da (gda): dld_t = Σ_{t'≥t in chunk} da_{t'}.
        # Done per chunk on the [B,chunk,nc] slice (a_t resets each chunk in the fwd, so it's intra-only).
        for c in range(NCH):
            r0, r1 = c * chunk, min(c * chunk + chunk, L)
            g_sl = gda[:, r0:r1]                                    # [B, len, nc]
            tot = g_sl.sum(dim=1, keepdim=True)
            dld[:, r0:r1] = tot - g_sl.cumsum(dim=1) + g_sl
    if has_bias:
        if use_g:
            return dq[..., :dqk], dk[..., :dqk], dvv[..., :dv], dh, dWr, dWw, dld, dbr, dbw
        return dq[..., :dqk], dk[..., :dqk], dvv[..., :dv], dh, dWr, dWw, dbr, dbw
    if use_g:
        return dq[..., :dqk], dk[..., :dqk], dvv[..., :dv], dh, dWr, dWw, dld
    return dq[..., :dqk], dk[..., :dqk], dvv[..., :dv], dh, dWr, dWw


class _RoLARoutedFn(torch.autograd.Function):
    """End-to-end differentiable in-kernel TREE-ROUTED RLA readout. Forward runs the OPTIMIZED production
    routed forward (`_routed_fwd_tiled`); backward drives the validated fold kernels at the production
    state-block width (BC=BG), reconstructing factors through the SAME Sel map. The [L,nc] gates AND
    their grads are never materialized (only transient [BT,BG] factor tiles + per-chunk [B,chunk,nc]
    gate-grad scratch). Mirrors proto_tree_routing._TreeRoutedFn."""
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, h, Wr, Ww, D, b, chunk, BG, b_r, b_w, ld=None):
        chunk = _CHUNK_FWD if chunk is None else min(chunk, _CHUNK_FWD)
        nc = b ** D
        sel = _build_sel(D, b, nc, q.device)
        q, k, v, h, Wr, Ww = (x.contiguous() for x in (q, k, v, h, Wr, Ww))
        ldc = ld.contiguous() if ld is not None else None
        o = _routed_fwd_tiled(q, k, v, h, Wr, Ww, D, b, sel, chunk=chunk, BG=BG, b_r=b_r, b_w=b_w, ld=ldc)
        # save the inputs (NOT gates, NOT [L,nc] grads); the per-chunk pre-state snapshots the reverse-
        # scan needs are recomputed in backward (fp32-router parity) by `_routed_snapshots`. ld saved for GLA.
        ctx.save_for_backward(q, k, v, h, Wr, Ww, b_r, b_w, ldc)
        ctx.D, ctx.b, ctx.chunk, ctx.BG = D, b, chunk, BG
        return o.to(q.dtype)

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do):
        q, k, v, h, Wr, Ww, b_r, b_w, ld = ctx.saved_tensors
        grads = _rola_rla_routed_bwd(
            q, k, v, h, Wr, Ww, do.contiguous(), ctx.D, ctx.b, ctx.chunk, ctx.BG,
            b_r=b_r, b_w=b_w, ld=ld)
        use_g = ld is not None
        if b_r is None:
            (dq, dk, dv, dh, dWr, dWw), dbr, dbw = (grads[:6]), None, None
            dld = grads[6] if use_g else None
        else:
            dq, dk, dv, dh, dWr, dWw = grads[:6]
            dld = grads[6] if use_g else None
            dbr, dbw = grads[-2], grads[-1]
        # forward arg order: q, k, v, h, Wr, Ww, D, b, chunk, BG, b_r, b_w, ld
        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dh.to(h.dtype),
                dWr.to(Wr.dtype), dWw.to(Ww.dtype), None, None, None, None,
                None if dbr is None else dbr.to(b_r.dtype),
                None if dbw is None else dbw.to(b_w.dtype),
                None if dld is None else dld.to(ld.dtype))


@input_guard
def rola_rla_routed_triton(q, k, v, h, Wr, Ww, D, b, chunk=None, BG=16, b_r=None, b_w=None):
    """Un-normalized TREE-ROUTED RLA readout via Triton — the in-kernel-routing twin of
    `rola_rla_triton`, now DIFFERENTIABLE end-to-end (fused router-grad fold; the [L,nc] gates and their
    grads are NEVER materialized). The routing gram is built IN-KERNEL from the hidden state h + per-level
    router weights Wr,Ww ∈ [D,d_model,b] (b^D=nc), with an OPTIONAL per-level bias b_r/b_w ∈ [D,b]
    (softmax(h·W+b)).

    q,k:[BH,L,K]  v:[BH,L,V]  h:[BH,L,d_model]  Wr,Ww:[D,d_model,b].  Returns [BH,L,V] at BV=next_pow2(V).
    Flat (D=1, b=nc) is the fused equivalent of `rola_rla_triton(q,k,v,r,w)` with r,w the D=1 router's
    explicit softmax gates. Grads dq,dk,dv,d_h,dWr,dWw (+db_r,db_w) match autograd to rel<1e-2 (bf16)."""
    return _RoLARoutedFn.apply(q, k, v, h, Wr, Ww, D, b, chunk, BG, b_r, b_w, None)


@input_guard
def rola_gla_routed_triton(q, k, v, h, Wr, Ww, ld, D, b, chunk=None, BG=16, b_r=None, b_w=None):
    """Un-normalized TREE-ROUTED GLA readout via Triton — `rola_rla_routed_triton` + a per-state scalar
    log-decay ld:[BH,L,nc] (GLA), DIFFERENTIABLE end-to-end ([L,nc] gates AND their grads never
    materialized). The routing gram uses the DECAYED gates (rt=r·eᵃ, w_end=w·e^{Λ−a}) exactly like
    `rola_gla_triton`, but with the routing built in-kernel from h+Wr,Ww. ld=None is the RLA path.
    Grads dq,dk,dv,d_h,dWr,dWw,dld: q/k/v TIGHT (<8e-3), the gate/decay grads to the GLA fp32 floor."""
    return _RoLARoutedFn.apply(q, k, v, h, Wr, Ww, D, b, chunk, BG, b_r, b_w, ld)


# ============================================================================
# FUSED kappa / per_state TREE-ROUTED path (RLA). The production normalization, IN-KERNEL.
#
# The plain routed path (`_RoLARoutedFn`) handles 'raw'/'global' fully in-kernel. 'kappa'/'per_state'
# rescale the READ gate by a per-(token,state) factor r̃ = r·(d+ε)^{−κ} (kappa) or r/(d+ε) (per_state),
# where d_i^c = Σ_{j≤i} (φq_i·φk_j) w_j^c is the per-state denominator (a causal w-weighted content-gram
# sum, r-INDEPENDENT). The OLD path fell back to the explicit-gate core, materializing d AND r̃ as
# [L,nc]. This fused path computes BOTH transiently per chunk and never writes any [*,L,nc] buffer:
#
# Per chunk (sequential per-batch scan, two carried states Sval^c[k,v] and Sden^c[k]):
#   d[BT,nc]  = (G⊙causal)·w  (intra)  +  q·Sden^c (inter)          # carried den state, value=ones
#   r̃[BT,nc] = r·(d+ε)^{−κ}  (kappa)  |  r/(d+ε) (per_state)        # transient SRAM, never HBM
#   num_i    = Σ_{j≤i} G_ij (Σ_c r̃_i^c w_j^c) v_j  (intra) + Σ_c r̃_i^c (q_i·Sval^c) (inter)
#   den_i    = Σ_c r̃_i^c d_i^c                                      # intra-token reduce over c
#   out_i    = num_i / (den_i + ε)
# state update (w only, r̃-independent): Sval^c += Σ wₜᶜ kₜ⊗vₜ ;  Sden^c += Σ wₜᶜ kₜ.
#
# This is the EXACT explicit-gate math (validated bit-for-bit vs `_rola_chunk_core` + den pre-pass),
# with the gates r,w AND the rescale d,r̃ living only as transient [BT,nc] SRAM tiles. The gram is
# formed nc-wide for the rescale term (the documented, accepted cost — we lose factored-gram compute
# for the read side but keep the [L,nc]-activation + tree-param wins). Backward is a reverse chunk-scan
# that recomputes r,w,d,r̃ transiently and folds the [BT,nc] gate-grads into dWr,dWw,d_h (+ dκ),
# reusing the proven softmax-jacobian fold (`_fold_level`); the gate-grads never become [L,nc] either.
# ============================================================================


@triton.jit
def _kappa_d_tile(q_ptr, sden_ptr, Gc, w_tile, cols, cmask,
                  pid_b, rows, rmask, dqk, nc, sq_b, sq_l, sq_d, ssd_b, ssd_k,
                  BT: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr, ND: tl.constexpr):
    """Per-state den d[BT,BC] = (G⊙causal)·w (intra) + q·Sden^c (inter). The inter term contracts dqk →
    accumulate over BK-feature-blocks so the [BC,BK] Sden slice stays bounded by BK<=64. Gc = G⊙causal is
    passed in (value-free, reused). BV-free; recomputed identically in each numerator value-block pass."""
    d_inter = tl.zeros([BT, BC], dtype=tl.float32)
    for d0 in range(ND):
        offs_k = d0 * BK + tl.arange(0, BK)
        kmask = offs_k < dqk
        qc = tl.load(q_ptr + pid_b*sq_b + rows[:, None]*sq_l + offs_k[None, :]*sq_d,
                     mask=rmask[:, None] & kmask[None, :], other=0.0)
        ck = tl.reshape(cols[:, None] * dqk + offs_k[None, :], [BC * BK])
        ckmask = tl.reshape(cmask[:, None] & kmask[None, :], [BC * BK])
        sden = tl.load(sden_ptr + pid_b*ssd_b + ck*ssd_k, mask=ckmask, other=0.0)
        sden2 = tl.reshape(sden, [BC, BK])
        d_inter += tl.dot(qc, tl.trans(sden2).to(qc.dtype))                         # [BT,BC]
    d_intra = tl.dot(Gc.to(w_tile.dtype), w_tile)
    return tl.where(cmask[None, :], d_intra + d_inter, 0.0)


@triton.jit
def _kappa_rescale(r_tile, d_tile, kap, cmask, GLOBAL: tl.constexpr,
                   PER_STATE: tl.constexpr, EPS: tl.constexpr):
    """r_tilde = r (global) | r/(d+eps) (per_state) | r*(d+eps)^(-kappa) (kappa). Transient [BT,BC].
    `global` is `kappa` without the rescale — the den D_i=Σ_c r^c d^c is still reduced over c by the
    caller; only the per-state read-gate rescale is skipped."""
    if GLOBAL:
        rt = r_tile
    else:
        de = d_tile + EPS
        if PER_STATE:
            rt = r_tile / de
        else:
            rt = r_tile * tl.exp(-kap[:, None] * tl.log(de))
    return tl.where(cmask[None, :], rt, 0.0)


@triton.jit
def _kappa_fwd_chunk(h_ptr, q_ptr, k_ptr, v_ptr, wr_ptr, ww_ptr, sel_ptr, kap_ptr,
                     sval_ptr, sden_ptr, num_ptr, den_ptr, br_ptr, bw_ptr,
                     L, d_model, dqk, dv, nc, t_start,
                     sh_b, sh_l, sh_d, sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sk_b, sk_l,
                     swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b, ssel_lvl, ssel_b, ssel_c,
                     ssv_b, ssv_c, ssv_k, ssv_v, ssd_b, ssd_c, ssd_k,
                     snm_b, snm_l, snm_v, sdn_b, sdn_l, sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                     GLOBAL: tl.constexpr, PER_STATE: tl.constexpr, EPS: tl.constexpr,
                     D: tl.constexpr, b: tl.constexpr, BB: tl.constexpr,
                     BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BD: tl.constexpr,
                     BC: tl.constexpr, NCBLK: tl.constexpr, ND: tl.constexpr, NDM: tl.constexpr,
                     HAS_BIAS: tl.constexpr):
    """Fused global/kappa/per_state chunk: builds r,w,d,r_tilde transiently per nc-block, accumulates num +
    den, carries Sval[k,v] AND Sden[k] across chunks. One program per batch. Mirrors the proto chunk
    kernel + the den pre-pass, with the read gate rescaled by the transient per-state den."""
    pid_b = tl.program_id(0)
    ND_V = (dv + BV - 1) // BV       # value-blocks (BV is the value TILE; ND_V==1 ⇒ the un-tiled kernel)
    offs_t = tl.arange(0, BT)
    offs_bb = tl.arange(0, BB)
    offs_c = tl.arange(0, BC)
    bmask = offs_bb < b
    rows = t_start + offs_t
    rmask = rows < L
    kap = tl.load(kap_ptr + pid_b*sk_b + rows*sk_l, mask=rmask, other=0.0)
    causal = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
    # content gram G = q·kᵀ accumulated over BK-feature-blocks (so the [BT,BK] q/k tiles stay <=64).
    G = tl.zeros([BT, BT], dtype=tl.float32)
    for d0 in range(ND):
        offs_k = d0 * BK + tl.arange(0, BK)
        kmask = offs_k < dqk
        qc = tl.load(q_ptr + pid_b*sq_b + rows[:, None]*sq_l + offs_k[None, :]*sq_d,
                     mask=rmask[:, None] & kmask[None, :], other=0.0)
        kc = tl.load(k_ptr + pid_b*sq_b + rows[:, None]*sq_l + offs_k[None, :]*sq_d,
                     mask=rmask[:, None] & kmask[None, :], other=0.0)
        G += tl.dot(qc, tl.trans(kc))
    Gc = G * causal
    # ---- Pass 1 (value-free): global den o_den = Σ_c r̃^c d^c (read carried-in Sden; do NOT update it
    # yet — Pass 2 re-reads it for the same d). The [BC*BK,*] / [BT,BC*BK] tiles avoid the BV axis here.
    o_den = tl.zeros([BT], dtype=tl.float32)
    for cb in range(NCBLK):
        cols = cb * BC + offs_c
        cmask = cols < nc
        r_tile, w_tile = _build_rw_tile(h_ptr, wr_ptr, ww_ptr, sel_ptr, cols, cmask,
                                        pid_b, rows, rmask, offs_bb, bmask, d_model,
                                        sh_b, sh_l, sh_d, swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
                                        ssel_lvl, ssel_b, ssel_c,
                                        br_ptr, bw_ptr, sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                                        D, BT, BB, BC, BD, NDM, HAS_BIAS)
        d_tile = _kappa_d_tile(q_ptr, sden_ptr, Gc, w_tile, cols, cmask,
                               pid_b, rows, rmask, dqk, nc, sq_b, sq_l, sq_d, ssd_b, ssd_k,
                               BT, BK, BC, ND)
        rt_tile = _kappa_rescale(r_tile, d_tile, kap, cmask, GLOBAL, PER_STATE, EPS)
        o_den += tl.sum(rt_tile * d_tile, axis=1)
    # ---- Pass 2 (value-tiled): numerator num = intra (A·v) + inter (Σ_c r̃^c q·Sval^c), AND the Sval
    # state write — all value-tiled over ND_V blocks so the [BC*BK,BV] state slice stays bounded by BK·BV.
    for vb in range(ND_V):
        offs_v = vb * BV + tl.arange(0, BV)
        vmask = offs_v < dv
        vc = tl.load(v_ptr + pid_b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                     mask=rmask[:, None] & vmask[None, :], other=0.0)
        o_num = tl.zeros([BT, BV], dtype=tl.float32)
        for cb in range(NCBLK):
            cols = cb * BC + offs_c
            cmask = cols < nc
            r_tile, w_tile = _build_rw_tile(h_ptr, wr_ptr, ww_ptr, sel_ptr, cols, cmask,
                                            pid_b, rows, rmask, offs_bb, bmask, d_model,
                                            sh_b, sh_l, sh_d, swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
                                            ssel_lvl, ssel_b, ssel_c,
                                            br_ptr, bw_ptr, sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                                            D, BT, BB, BC, BD, NDM, HAS_BIAS)
            d_tile = _kappa_d_tile(q_ptr, sden_ptr, Gc, w_tile, cols, cmask,
                                   pid_b, rows, rmask, dqk, nc, sq_b, sq_l, sq_d, ssd_b, ssd_k,
                                   BT, BK, BC, ND)
            rt_tile = _kappa_rescale(r_tile, d_tile, kap, cmask, GLOBAL, PER_STATE, EPS)
            # numerator intra: A = G⊙(r̃·wᵀ)⊙causal; o_num += A·v.
            Rg = tl.dot(rt_tile, tl.trans(w_tile))
            A = G * Rg * causal
            o_num += tl.dot(A.to(vc.dtype), vc)
            # numerator inter Σ_c r̃^c (q·Sval^c) + the Sval write (Sval^c += Σ wᶜ k⊗v); both index dqk →
            # loop BK-blocks. rt_tile/w_tile [BT,BC] dqk-free, reused; the [BC*BK,BV] sval slice fits.
            for d0 in range(ND):
                offs_k = d0 * BK + tl.arange(0, BK)
                kmask = offs_k < dqk
                qc = tl.load(q_ptr + pid_b*sq_b + rows[:, None]*sq_l + offs_k[None, :]*sq_d,
                             mask=rmask[:, None] & kmask[None, :], other=0.0)
                kc = tl.load(k_ptr + pid_b*sq_b + rows[:, None]*sq_l + offs_k[None, :]*sq_d,
                             mask=rmask[:, None] & kmask[None, :], other=0.0)
                ck = tl.reshape(cols[:, None] * dqk + offs_k[None, :], [BC * BK])
                ckmask = tl.reshape(cmask[:, None] & kmask[None, :], [BC * BK])
                sflat = tl.load(sval_ptr + pid_b*ssv_b + ck[:, None]*ssv_k + offs_v[None, :]*ssv_v,
                                mask=ckmask[:, None] & vmask[None, :], other=0.0)        # [BC*BK,BV]
                rq = tl.reshape(rt_tile[:, :, None] * qc[:, None, :], [BT, BC * BK])
                o_num += tl.dot(rq.to(sflat.dtype), sflat)
                wk = tl.reshape(w_tile[:, :, None] * kc[:, None, :], [BT, BC * BK])
                tl.store(sval_ptr + pid_b*ssv_b + ck[:, None]*ssv_k + offs_v[None, :]*ssv_v,
                         sflat + tl.dot(tl.trans(wk).to(vc.dtype), vc),
                         mask=ckmask[:, None] & vmask[None, :])
        tl.store(num_ptr + pid_b*snm_b + rows[:, None]*snm_l + offs_v[None, :]*snm_v,
                 o_num, mask=rmask[:, None] & vmask[None, :])
    # ---- Pass 3 (value-free): Sden state write (Sden^c += Σ wᶜ k). Done LAST so Pass 1/2's d-recompute
    # read the carried-in Sden. Loops BK-blocks; the [BC*BK] tiles are tiny.
    for cb in range(NCBLK):
        cols = cb * BC + offs_c
        cmask = cols < nc
        _r, w_tile = _build_rw_tile(h_ptr, wr_ptr, ww_ptr, sel_ptr, cols, cmask,
                                    pid_b, rows, rmask, offs_bb, bmask, d_model,
                                    sh_b, sh_l, sh_d, swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
                                    ssel_lvl, ssel_b, ssel_c,
                                    br_ptr, bw_ptr, sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                                    D, BT, BB, BC, BD, NDM, HAS_BIAS)
        for d0 in range(ND):
            offs_k = d0 * BK + tl.arange(0, BK)
            kmask = offs_k < dqk
            kc = tl.load(k_ptr + pid_b*sq_b + rows[:, None]*sq_l + offs_k[None, :]*sq_d,
                         mask=rmask[:, None] & kmask[None, :], other=0.0)
            ck = tl.reshape(cols[:, None] * dqk + offs_k[None, :], [BC * BK])
            ckmask = tl.reshape(cmask[:, None] & kmask[None, :], [BC * BK])
            wk = tl.reshape(w_tile[:, :, None] * kc[:, None, :], [BT, BC * BK])
            dsden = tl.sum(wk, axis=0)                                                  # [BC*BK]
            sden = tl.load(sden_ptr + pid_b*ssd_b + ck*ssd_k, mask=ckmask, other=0.0)   # [BC*BK]
            tl.store(sden_ptr + pid_b*ssd_b + ck*ssd_k, sden + dsden, mask=ckmask)
    tl.store(den_ptr + pid_b*sdn_b + rows*sdn_l, o_den, mask=rmask)


def _kappa_routed_fwd(q, k, v, h, Wr, Ww, kap, D, b, sel, chunk, global_norm, per_state, eps,
                      b_r=None, b_w=None):
    """Fused global/kappa/per_state tree-routed forward. Returns (num[B,L,BV], den[B,L]) and the per-chunk
    pre-state snapshots (Sval, Sden) the backward needs. Optional routing bias b_r/b_w ∈ [D,b]
    (softmax(h·W+b)). No [L,nc] gate/den/r̃ buffer is allocated."""
    B, L, dqk = q.shape
    dv = v.shape[-1]
    d_model = h.shape[-1]
    nc = b ** D
    BC = max(16, min(nc, 16))   # tl.dot needs the gram dim >=16; the nc tail is cmask'd
    BK = _kappa_bk_cap(dqk, dv, _KAPPA_BWD_CHUNK, BC)    # feature-tile (loop ND) — bounds the [BT,BC*BK] tiles
    BV = _kappa_bv_tile(dqk, dv, BC)                     # value-tile (loop ND_V) so [BC*BK,BV] fits SRAM
    BVO = max(16, triton.next_power_of_2(dv))            # num buffer width (full padded value dim)
    BD = max(16, triton.next_power_of_2(d_model))
    BB = max(16, triton.next_power_of_2(b))
    # The [BT,BC*BK] rq/wk tiles scale with BT, so the heavy-tile fwd chunk is capped like the bwd (the
    # chunked scan is chunk-size invariant). Cheap dqk/dv keep the full fwd chunk (BC*BK*BT then fits).
    chunk = chunk if BK * BC * chunk * 4 <= 24 * 1024 else min(chunk, _KAPPA_BWD_CHUNK)
    NCBLK = triton.cdiv(nc, BC)
    ND = triton.cdiv(dqk, BK)
    NDM = triton.cdiv(d_model, BD)
    NCH = triton.cdiv(L, chunk)
    Sval = torch.zeros(B, nc, dqk, dv, device=q.device, dtype=torch.float32)
    Sden = torch.zeros(B, nc, dqk, device=q.device, dtype=torch.float32)
    num = torch.zeros(B, L, BVO, device=q.device, dtype=torch.float32)   # full padded width; BV tiles it
    den = torch.zeros(B, L, device=q.device, dtype=torch.float32)
    snap_val = torch.zeros(B, NCH, nc, dqk, dv, device=q.device, dtype=torch.float32)
    snap_den = torch.zeros(B, NCH, nc, dqk, device=q.device, dtype=torch.float32)
    br, bw, has_bias = _routing_bias(b_r, b_w, D, b, q.device, dtype=q.dtype)
    sbias = _bias_strides(br, bw, has_bias)
    common = dict(GLOBAL=global_norm, PER_STATE=per_state, EPS=eps, D=D, b=b, BB=BB, BT=chunk,
                  BK=BK, BV=BV, BD=BD, BC=BC, NCBLK=NCBLK, ND=ND, NDM=NDM, HAS_BIAS=has_bias,
                  num_warps=4, num_stages=1)
    for c in range(NCH):
        snap_val[:, c].copy_(Sval)
        snap_den[:, c].copy_(Sden)
        _kappa_fwd_chunk[(B,)](
            h, q, k, v, Wr, Ww, sel, kap, Sval, Sden, num, den, br, bw,
            L, d_model, dqk, dv, nc, c * chunk,
            h.stride(0), h.stride(1), h.stride(2), q.stride(0), q.stride(1), q.stride(2),
            v.stride(0), v.stride(1), v.stride(2), kap.stride(0), kap.stride(1),
            Wr.stride(0), Wr.stride(1), Wr.stride(2), Ww.stride(0), Ww.stride(1), Ww.stride(2),
            sel.stride(0), sel.stride(1), sel.stride(2),
            Sval.stride(0), Sval.stride(1), Sval.stride(2), Sval.stride(3),
            Sden.stride(0), Sden.stride(1), Sden.stride(2),
            num.stride(0), num.stride(1), num.stride(2), den.stride(0), den.stride(1),
            *sbias,
            **common)
    return num[..., :dv], den, snap_val, snap_den


@triton.jit
def _kappa_rescale_bwd(drt_tile, r_tile, d_tile, kap, cmask,
                       GLOBAL: tl.constexpr, PER_STATE: tl.constexpr, EPS: tl.constexpr):
    """Backprop r_tilde = rescale(r, d, kappa). Given d(r_tilde) returns d(r), d(d), and the per-token
    d(kappa) contribution (summed over c). r_tilde = r (global) | r/(d+eps) (per_state) |
    r*(d+eps)^(-kappa) (kappa). global: r_tilde=r so dr=drt, dd=0, dkap=0 (no rescale path)."""
    if GLOBAL:
        dr = tl.where(cmask[None, :], drt_tile, 0.0)
        dd = dr * 0.0
        dkap = tl.zeros([drt_tile.shape[0]], dtype=tl.float32)
        return dr, dd, dkap
    de = d_tile + EPS
    if PER_STATE:
        inv = 1.0 / de
        rt = r_tile * inv                              # r_tilde
        dr = drt_tile * inv
        dd = -drt_tile * rt * inv
        dkap = tl.zeros([drt_tile.shape[0]], dtype=tl.float32)
    else:
        logde = tl.log(de)
        rt = r_tile * tl.exp(-kap[:, None] * logde)    # r_tilde
        dr = drt_tile * tl.exp(-kap[:, None] * logde)
        dd = -drt_tile * rt * (kap[:, None] / de)
        dkap = tl.sum(tl.where(cmask[None, :], -drt_tile * rt * logde, 0.0), axis=1)
    dr = tl.where(cmask[None, :], dr, 0.0)
    dd = tl.where(cmask[None, :], dd, 0.0)
    return dr, dd, dkap


@triton.jit
def _kappa_bwd_state(h_ptr, k_ptr, v_ptr, wr_ptr, ww_ptr, sel_ptr,
                     dsval_ptr, dsden_ptr, dk_ptr, dv_ptr, gdw_ptr, br_ptr, bw_ptr,
                     L, d_model, dqk, dv, nc, t_start,
                     sh_b, sh_l, sh_d, sq_b, sq_l, sq_d, sv_b, sv_l, sv_d,
                     swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b, ssel_lvl, ssel_b, ssel_c,
                     ssv_b, ssv_k, ssv_v, ssd_b, ssd_k, sgd_b, sgd_t, sgd_c,
                     sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                     D: tl.constexpr, b: tl.constexpr, BB: tl.constexpr,
                     BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BD: tl.constexpr,
                     BC: tl.constexpr, NCBLK: tl.constexpr, ND: tl.constexpr, NDM: tl.constexpr,
                     HAS_BIAS: tl.constexpr):
    """State-update backward (split #1, SMEM-bound by the dSval tile). Uses the INCOMING adjoint
    (= Sval_{j+1}, Sden_{j+1}) to backprop the two writes Sval^c += Σ wᶜ k⊗v and Sden^c += Σ wᶜ k.
    Produces dk,dv (atomic) and the state half of dw (stored to gdw). w rebuilt in-kernel; no [L,nc]."""
    pid_b = tl.program_id(0)
    ND_V = (dv + BV - 1) // BV       # value-blocks (BV is the value TILE; ND_V==1 ⇒ the un-tiled kernel)
    offs_t = tl.arange(0, BT)
    offs_bb = tl.arange(0, BB)
    offs_c = tl.arange(0, BC)
    bmask = offs_bb < b
    rows = t_start + offs_t
    rmask = rows < L
    for cb in range(NCBLK):
        cols = cb * BC + offs_c
        cmask = cols < nc
        _r, w_tile = _build_rw_tile(h_ptr, wr_ptr, ww_ptr, sel_ptr, cols, cmask,
                                    pid_b, rows, rmask, offs_bb, bmask, d_model,
                                    sh_b, sh_l, sh_d, swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
                                    ssel_lvl, ssel_b, ssel_c,
                                    br_ptr, bw_ptr, sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                                    D, BT, BB, BC, BD, NDM, HAS_BIAS)
        # dw[BT,BC] sums over dqk → accumulate across BK-feature-blocks. dk[BT,BK] is per-feature-block
        # (offs_k), stored per d0. dv[BT,BV] is per value-block (offs_v), atomic-added per (d0,vb). The
        # value-contracted Nval=v·dSvalᵀ is summed over value-blocks; the [BC*BK,BV] dSval slice stays
        # bounded by BK·BV<=64·BV. w_tile[BT,BC] / wk[BT,BC*BK] are value-free, reused across vb.
        dw = tl.zeros([BT, BC], dtype=tl.float32)
        for d0 in range(ND):
            offs_k = d0 * BK + tl.arange(0, BK)
            kmask = offs_k < dqk
            kc = tl.load(k_ptr + pid_b*sq_b + rows[:, None]*sq_l + offs_k[None, :]*sq_d,
                         mask=rmask[:, None] & kmask[None, :], other=0.0)
            ck = tl.reshape(cols[:, None] * dqk + offs_k[None, :], [BC * BK])
            ckmask = tl.reshape(cmask[:, None] & kmask[None, :], [BC * BK])
            wk = tl.reshape(w_tile[:, :, None] * kc[:, None, :], [BT, BC * BK])
            Nval = tl.zeros([BT, BC * BK], dtype=tl.float32)   # v·dSvalᵀ, value-contracted → sum over vb
            for vb in range(ND_V):
                offs_v = vb * BV + tl.arange(0, BV)
                vmask = offs_v < dv
                vc = tl.load(v_ptr + pid_b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                             mask=rmask[:, None] & vmask[None, :], other=0.0)
                dSval = tl.load(dsval_ptr + pid_b*ssv_b + ck[:, None]*ssv_k + offs_v[None, :]*ssv_v,
                                mask=ckmask[:, None] & vmask[None, :], other=0.0)
                Nval += tl.dot(vc, tl.trans(dSval).to(vc.dtype))   # [BT, BC*BK]
                dv_vb = tl.dot(wk.to(dSval.dtype), dSval)          # [BT,BV] for this value-block
                tl.atomic_add(dv_ptr + pid_b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                              dv_vb, mask=rmask[:, None] & vmask[None, :])
            Nvr = tl.reshape(Nval, [BT, BC, BK])
            dw += tl.sum(Nvr * kc[:, None, :], axis=2)
            dk_acc = tl.sum(Nvr * w_tile[:, :, None], axis=1)
            dSden = tl.load(dsden_ptr + pid_b*ssd_b + ck*ssd_k, mask=ckmask, other=0.0)
            dSden2 = tl.reshape(dSden, [BC, BK])
            dw += tl.where(cmask[None, :], tl.sum(dSden2[None, :, :] * kc[:, None, :], axis=2), 0.0)
            dk_acc += tl.dot(w_tile.to(dSden2.dtype), dSden2)
            tl.atomic_add(dk_ptr + pid_b*sq_b + rows[:, None]*sq_l + offs_k[None, :]*sq_d,
                          dk_acc, mask=rmask[:, None] & kmask[None, :])
        tl.store(gdw_ptr + pid_b*sgd_b + offs_t[:, None]*sgd_t + cols[None, :]*sgd_c,
                 dw, mask=rmask[:, None] & cmask[None, :])


@triton.jit
def _kappa_bwd_read(h_ptr, q_ptr, k_ptr, v_ptr, wr_ptr, ww_ptr, sel_ptr, kap_ptr,
                    sval_ptr, sden_ptr, dnum_ptr, dden_ptr,
                    dsval_ptr, dsden_ptr, dq_ptr, dk_ptr, dv_ptr, dkap_ptr, gdr_ptr, gdw_ptr,
                    br_ptr, bw_ptr,
                    L, d_model, dqk, dv, nc, t_start,
                    sh_b, sh_l, sh_d, sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sk_b, sk_l,
                    swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b, ssel_lvl, ssel_b, ssel_c,
                    ssv_b, ssv_k, ssv_v, ssd_b, ssd_k,
                    sdo_b, sdo_l, sdo_v, sdd_b, sdd_l, sgd_b, sgd_t, sgd_c,
                    sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                    GLOBAL: tl.constexpr, PER_STATE: tl.constexpr, EPS: tl.constexpr,
                    D: tl.constexpr, b: tl.constexpr, BB: tl.constexpr,
                    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BD: tl.constexpr,
                    BC: tl.constexpr, NCBLK: tl.constexpr, ND: tl.constexpr, NDM: tl.constexpr,
                    HAS_BIAS: tl.constexpr):
    """Readout/den/d backward (split #2, SMEM-bound by the sval snapshot tile). Recomputes r,w,d,r_tilde
    transiently, backprops den + intra num + inter-readout + the per-state-den d, producing dr,dq,dv-intra,
    dkappa, the read+intra+d half of dw (atomic-added into gdw), and the readout contribution to the
    carried dSval/dSden adjoints. dG -> dq,dk. The [L,nc] gates / d / r_tilde are never materialized."""
    pid_b = tl.program_id(0)
    ND_V = (dv + BV - 1) // BV       # value-blocks (BV is the value TILE; ND_V==1 ⇒ the un-tiled kernel)
    offs_t = tl.arange(0, BT)
    offs_bb = tl.arange(0, BB)
    offs_c = tl.arange(0, BC)
    bmask = offs_bb < b
    rows = t_start + offs_t
    rmask = rows < L
    kap = tl.load(kap_ptr + pid_b*sk_b + rows*sk_l, mask=rmask, other=0.0)
    dden = tl.load(dden_ptr + pid_b*sdd_b + rows*sdd_l, mask=rmask, other=0.0)
    causal = ((offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]).to(tl.float32)
    # content gram G = q·kᵀ accumulated over BK-feature-blocks (keeps the [BT,BK] q/k tiles <=64).
    G = tl.zeros([BT, BT], dtype=tl.float32)
    for d0 in range(ND):
        offs_k = d0 * BK + tl.arange(0, BK)
        kmask = offs_k < dqk
        qc = tl.load(q_ptr + pid_b*sq_b + rows[:, None]*sq_l + offs_k[None, :]*sq_d,
                     mask=rmask[:, None] & kmask[None, :], other=0.0)
        kc = tl.load(k_ptr + pid_b*sq_b + rows[:, None]*sq_l + offs_k[None, :]*sq_d,
                     mask=rmask[:, None] & kmask[None, :], other=0.0)
        G += tl.dot(qc, tl.trans(kc))
    Gc = G * causal
    # accumulators over BT/BV (and the [BT,BT] gram) sum ACROSS the BK/value-blocks; dq/dk (over BK) and
    # dv (over BV) are atomic-added per d0/vb-block at each producing site (their HBM offset uses offs_k/v).
    dkap_acc = tl.zeros([BT], dtype=tl.float32)
    dG = tl.zeros([BT, BT], dtype=tl.float32)
    for cb in range(NCBLK):
        cols = cb * BC + offs_c
        cmask = cols < nc
        r_tile, w_tile = _build_rw_tile(h_ptr, wr_ptr, ww_ptr, sel_ptr, cols, cmask,
                                        pid_b, rows, rmask, offs_bb, bmask, d_model,
                                        sh_b, sh_l, sh_d, swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
                                        ssel_lvl, ssel_b, ssel_c,
                                        br_ptr, bw_ptr, sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                                        D, BT, BB, BC, BD, NDM, HAS_BIAS)
        # d_tile/rt_tile are value-FREE (from G,Sden,r) → compute once, reused for every value-block.
        d_tile = _kappa_d_tile(q_ptr, sden_ptr, Gc, w_tile, cols, cmask,
                               pid_b, rows, rmask, dqk, nc, sq_b, sq_l, sq_d, ssd_b, ssd_k,
                               BT, BK, BC, ND)
        rt_tile = _kappa_rescale(r_tile, d_tile, kap, cmask, GLOBAL, PER_STATE, EPS)
        drt = dden[:, None] * d_tile          # d(r_tilde) from den
        dd = dden[:, None] * rt_tile          # d(d) from den
        # num intra: A=G*Rg*causal ; o_num += A v. dA=(dnum·vᵀ)⊙causal (value-contracted → sum over vb);
        # dv=Aᵀ·dnum (per value-block → atomic). A/Rg/dG/dw are value-free; dnum,vc loaded per vb.
        Rg = tl.dot(rt_tile, tl.trans(w_tile))
        A = G * Rg * causal
        dA = tl.zeros([BT, BT], dtype=tl.float32)
        for vb in range(ND_V):
            offs_v = vb * BV + tl.arange(0, BV)
            vmask = offs_v < dv
            vc = tl.load(v_ptr + pid_b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                         mask=rmask[:, None] & vmask[None, :], other=0.0)
            dnum = tl.load(dnum_ptr + pid_b*sdo_b + rows[:, None]*sdo_l + offs_v[None, :]*sdo_v,
                           mask=rmask[:, None] & vmask[None, :], other=0.0)
            dv_vb = tl.dot(tl.trans(A).to(dnum.dtype), dnum)   # dv[j,v]=sum_i A[i,j] dnum[i,v]
            tl.atomic_add(dv_ptr + pid_b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                          dv_vb, mask=rmask[:, None] & vmask[None, :])
            dA += tl.dot(dnum, tl.trans(vc))
        dA = dA * causal
        dRg = dA * G                          # dRg_ij = dA_ij G_ij (causal already in dA)
        dG += dA * Rg                          # dG_ij  += dA_ij Rg_ij (causal already in dA)
        drt += tl.dot(dRg.to(w_tile.dtype), w_tile)
        dw = tl.dot(tl.trans(dRg).to(rt_tile.dtype), rt_tile)
        # num inter readout: o_num += sum_c r_tilde^c (q.Sval_j^c). M=dnum·svalᵀ is value-contracted (sum
        # over vb); dq + the dSval-store are per-(BK,value)-block; drt's inter contribution sums over d0.
        for d0 in range(ND):
            offs_k = d0 * BK + tl.arange(0, BK)
            kmask = offs_k < dqk
            qc = tl.load(q_ptr + pid_b*sq_b + rows[:, None]*sq_l + offs_k[None, :]*sq_d,
                         mask=rmask[:, None] & kmask[None, :], other=0.0)
            ck = tl.reshape(cols[:, None] * dqk + offs_k[None, :], [BC * BK])
            ckmask = tl.reshape(cmask[:, None] & kmask[None, :], [BC * BK])
            rq = tl.reshape(rt_tile[:, :, None] * qc[:, None, :], [BT, BC * BK])
            M = tl.zeros([BT, BC * BK], dtype=tl.float32)
            for vb in range(ND_V):
                offs_v = vb * BV + tl.arange(0, BV)
                vmask = offs_v < dv
                dnum = tl.load(dnum_ptr + pid_b*sdo_b + rows[:, None]*sdo_l + offs_v[None, :]*sdo_v,
                               mask=rmask[:, None] & vmask[None, :], other=0.0)
                sval = tl.load(sval_ptr + pid_b*ssv_b + ck[:, None]*ssv_k + offs_v[None, :]*ssv_v,
                               mask=ckmask[:, None] & vmask[None, :], other=0.0)         # [BC*BK,BV]
                M += tl.dot(dnum, tl.trans(sval).to(dnum.dtype))   # [BT, BC*BK]
                dSval = tl.load(dsval_ptr + pid_b*ssv_b + ck[:, None]*ssv_k + offs_v[None, :]*ssv_v,
                                mask=ckmask[:, None] & vmask[None, :], other=0.0)
                tl.store(dsval_ptr + pid_b*ssv_b + ck[:, None]*ssv_k + offs_v[None, :]*ssv_v,
                         dSval + tl.dot(tl.trans(rq).to(dnum.dtype), dnum),
                         mask=ckmask[:, None] & vmask[None, :])
            Mr = tl.reshape(M, [BT, BC, BK])
            drt += tl.sum(Mr * qc[:, None, :], axis=2)
            dq_read = tl.sum(Mr * rt_tile[:, :, None], axis=1)   # [BT,BK]
            tl.atomic_add(dq_ptr + pid_b*sq_b + rows[:, None]*sq_l + offs_k[None, :]*sq_d,
                          dq_read, mask=rmask[:, None] & kmask[None, :])
        # rescale bwd (drt now complete).
        dr_resc, dd_resc, dkap_c = _kappa_rescale_bwd(drt, r_tile, d_tile, kap, cmask,
                                                      GLOBAL, PER_STATE, EPS)
        dd += dd_resc
        dkap_acc += dkap_c
        # d bwd: d = (G*causal) w + q.Sden_j.  dG/dw are dqk-free; dq + dSden contract dqk → loop d0.
        dG += tl.dot(dd.to(w_tile.dtype), tl.trans(w_tile)) * causal
        dw += tl.dot(tl.trans(Gc).to(dd.dtype), dd)
        for d0 in range(ND):
            offs_k = d0 * BK + tl.arange(0, BK)
            kmask = offs_k < dqk
            qc = tl.load(q_ptr + pid_b*sq_b + rows[:, None]*sq_l + offs_k[None, :]*sq_d,
                         mask=rmask[:, None] & kmask[None, :], other=0.0)
            ck = tl.reshape(cols[:, None] * dqk + offs_k[None, :], [BC * BK])
            ckmask = tl.reshape(cmask[:, None] & kmask[None, :], [BC * BK])
            sden = tl.load(sden_ptr + pid_b*ssd_b + ck*ssd_k, mask=ckmask, other=0.0)
            sden2 = tl.reshape(sden, [BC, BK])
            dq_d = tl.dot(dd.to(sden2.dtype), sden2)             # [BT,BK]
            tl.atomic_add(dq_ptr + pid_b*sq_b + rows[:, None]*sq_l + offs_k[None, :]*sq_d,
                          dq_d, mask=rmask[:, None] & kmask[None, :])
            dSden = tl.load(dsden_ptr + pid_b*ssd_b + ck*ssd_k, mask=ckmask, other=0.0)
            dSden_read = tl.reshape(tl.dot(tl.trans(dd).to(qc.dtype), qc), [BC * BK])
            tl.store(dsden_ptr + pid_b*ssd_b + ck*ssd_k, dSden + dSden_read, mask=ckmask)
        tl.store(gdr_ptr + pid_b*sgd_b + offs_t[:, None]*sgd_t + cols[None, :]*sgd_c,
                 dr_resc, mask=rmask[:, None] & cmask[None, :])
        tl.atomic_add(gdw_ptr + pid_b*sgd_b + offs_t[:, None]*sgd_t + cols[None, :]*sgd_c,
                      dw, mask=rmask[:, None] & cmask[None, :])
    # dG -> dq,dk: contract over BT (dG is [BT,BT]) producing [BT,BK] per feature-block → loop d0.
    for d0 in range(ND):
        offs_k = d0 * BK + tl.arange(0, BK)
        kmask = offs_k < dqk
        qc = tl.load(q_ptr + pid_b*sq_b + rows[:, None]*sq_l + offs_k[None, :]*sq_d,
                     mask=rmask[:, None] & kmask[None, :], other=0.0)
        kc = tl.load(k_ptr + pid_b*sq_b + rows[:, None]*sq_l + offs_k[None, :]*sq_d,
                     mask=rmask[:, None] & kmask[None, :], other=0.0)
        dq_g = tl.dot(dG.to(kc.dtype), kc)
        dk_g = tl.dot(tl.trans(dG).to(qc.dtype), qc)
        tl.atomic_add(dq_ptr + pid_b*sq_b + rows[:, None]*sq_l + offs_k[None, :]*sq_d,
                      dq_g, mask=rmask[:, None] & kmask[None, :])
        tl.atomic_add(dk_ptr + pid_b*sq_b + rows[:, None]*sq_l + offs_k[None, :]*sq_d,
                      dk_g, mask=rmask[:, None] & kmask[None, :])
    tl.atomic_add(dkap_ptr + pid_b*sk_b + rows*sk_l, dkap_acc, mask=rmask)


def _kappa_routed_bwd(q, k, v, h, Wr, Ww, kap, snap_val, snap_den, dnum, dden,
                      D, b, sel, chunk, global_norm, per_state, eps, b_r=None, b_w=None):
    """Reverse chunk-scan backward for the fused global/kappa/per_state path. Carries dSval,dSden
    adjoints; recomputes r,w,d,r_tilde transiently per chunk; folds the [BT,nc] gate-grads into
    dWr,dWw,dh (and db_r/db_w when a routing bias is present — transient, never [L,nc]). Returns
    dq,dk,dv,dh,dWr,dWw,dkappa (and db_r,db_w when biased) — all fp32. No [L,nc] buffer is allocated."""
    B, L, d_model = h.shape
    dqk = q.shape[-1]
    dv = v.shape[-1]
    nc = b ** D
    BC = max(16, min(nc, 16))
    BK = _kappa_bk_cap(dqk, dv, chunk, BC)   # feature-tile (loop ND) — bounds the [BT,BC*BK] tiles
    BV = _kappa_bv_tile(dqk, dv, BC, BK)     # value-tile (loop ND_V) so the [BC*BK,BV] state slices fit
    BD = max(16, triton.next_power_of_2(d_model))
    BB = max(16, triton.next_power_of_2(b))
    NCBLK = triton.cdiv(nc, BC)
    ND = triton.cdiv(dqk, BK)
    NDM = triton.cdiv(d_model, BD)
    NCH = triton.cdiv(L, chunk)
    # dvv/dq/dk are written with v's / q's row strides (sv_l=v.stride(1)=dv, sq_l=q.stride(1)=dqk), so
    # they MUST be allocated at the TRUE dv/dqk width (NOT padded BV/BK) or the row layout corrupts for
    # non-pow2 dv/dqk. The in-kernel [BT,BV]/[BT,BK] accumulators store only the masked :dv/:dqk part.
    dq = torch.zeros(B, L, dqk, device=q.device, dtype=torch.float32)
    dk = torch.zeros(B, L, dqk, device=q.device, dtype=torch.float32)
    dvv = torch.zeros(B, L, dv, device=q.device, dtype=torch.float32)
    dh = torch.zeros(B, L, d_model, device=q.device, dtype=torch.float32)
    dWr = torch.zeros(D, d_model, b, device=q.device, dtype=torch.float32)
    dWw = torch.zeros(D, d_model, b, device=q.device, dtype=torch.float32)
    dkap = torch.zeros(B, L, device=q.device, dtype=torch.float32)
    dSval = torch.zeros(B, nc, dqk, dv, device=q.device, dtype=torch.float32)
    dSden = torch.zeros(B, nc, dqk, device=q.device, dtype=torch.float32)
    br, bw, has_bias = _routing_bias(b_r, b_w, D, b, q.device, dtype=torch.float32)
    dbr = torch.zeros(D, b, device=q.device, dtype=torch.float32)   # routing-bias grads [D,b] (transient fold)
    dbw = torch.zeros(D, b, device=q.device, dtype=torch.float32)
    dnum = dnum.contiguous()
    dden = dden.contiguous()
    # transient per-chunk gate-grad scratch [B,chunk,nc], OVERWRITTEN each chunk — never [L,nc].
    gdr = torch.zeros(B, chunk, nc, device=q.device, dtype=torch.float32)
    gdw = torch.zeros(B, chunk, nc, device=q.device, dtype=torch.float32)
    sB = (q.stride(0), q.stride(1), q.stride(2))
    sV = (v.stride(0), v.stride(1), v.stride(2))
    sH = (h.stride(0), h.stride(1), h.stride(2))
    sWr = (Wr.stride(0), Wr.stride(1), Wr.stride(2))
    sWw = (Ww.stride(0), Ww.stride(1), Ww.stride(2))
    sSel = (sel.stride(0), sel.stride(1), sel.stride(2))
    sSV = (dSval.stride(0), dSval.stride(2), dSval.stride(3))   # (B, flat-k=dqk-axis, v)
    sSD = (dSden.stride(0), dSden.stride(2))                    # (B, flat-k)
    sGD = (gdr.stride(0), gdr.stride(1), gdr.stride(2))
    sBias = _bias_strides(br, bw, has_bias)
    state_common = dict(D=D, b=b, BB=BB, BT=chunk, BK=BK, BV=BV, BD=BD, BC=BC, NCBLK=NCBLK, ND=ND,
                        NDM=NDM, HAS_BIAS=has_bias, num_warps=4, num_stages=1)
    read_common = dict(GLOBAL=global_norm, PER_STATE=per_state, EPS=eps, D=D, b=b, BB=BB, BT=chunk,
                       BK=BK, BV=BV, BD=BD, BC=BC, NCBLK=NCBLK, ND=ND, NDM=NDM,
                       HAS_BIAS=has_bias, num_warps=4, num_stages=1)
    fold_common = dict(D=D, b=b, BB=BB, BT=chunk, BC=BC, BD=BD, NCBLK=NCBLK, NDM=NDM,
                       HAS_BIAS=has_bias, num_warps=4, num_stages=1)
    for c in reversed(range(NCH)):
        Sval = snap_val[:, c].contiguous()
        Sden = snap_den[:, c].contiguous()
        gdw.zero_()
        # K1: state-update bwd (reads adjoint of Sval_{j+1}; produces dk,dv + state half of dw).
        _kappa_bwd_state[(B,)](
            h, k, v, Wr, Ww, sel, dSval, dSden, dk, dvv, gdw, br, bw,
            L, d_model, dqk, dv, nc, c * chunk,
            *sH, *sB, *sV, *sWr, *sWw, *sSel, *sSV, *sSD, *sGD, *sBias, **state_common)
        # K2: readout/den/d bwd (adds dw, produces dr,dq,dv-intra,dkappa; folds dSval/dSden adjoints).
        _kappa_bwd_read[(B,)](
            h, q, k, v, Wr, Ww, sel, kap, Sval, Sden, dnum, dden,
            dSval, dSden, dq, dk, dvv, dkap, gdr, gdw, br, bw,
            L, d_model, dqk, dv, nc, c * chunk,
            *sH, *sB, *sV, kap.stride(0), kap.stride(1), *sWr, *sWw, *sSel, *sSV, *sSD,
            dnum.stride(0), dnum.stride(1), dnum.stride(2), dden.stride(0), dden.stride(1), *sGD,
            *sBias,
            **read_common)
        # fold the transient gate-grads -> dWr,dWw,dh (+db; proto fold; factor-rebuild SMEM isolated here).
        _proto_fold[(B,)](
            h, Wr, Ww, sel, gdr, gdw, dh, dWr, dWw, br, bw, dbr, dbw,
            L, d_model, nc, c * chunk, *sH, *sWr, *sWw, *sSel,
            gdr.stride(0), gdr.stride(1), gdr.stride(2), dh.stride(0), dh.stride(1), dh.stride(2),
            *sBias,
            **fold_common)
    if has_bias:
        return (dq[..., :dqk], dk[..., :dqk], dvv[..., :dv], dh, dWr, dWw, dkap, dbr, dbw)
    return (dq[..., :dqk], dk[..., :dqk], dvv[..., :dv], dh, dWr, dWw, dkap)


class _RoLARoutedKappaFn(torch.autograd.Function):
    """End-to-end differentiable FUSED kappa/per_state tree-routed RLA path. Forward runs the fused
    chunk-scan (`_kappa_routed_fwd`) returning the un-divided (num, den); backward runs the reverse
    chunk-scan, recomputing the snapshots for fp-parity and folding the transient [BT,nc] gate-grads
    into the router. The [L,nc] gates AND the per-state den d / rescaled r_tilde are NEVER materialized.
    The final divide out=num/(den+eps) is left to torch (autograd handles it)."""
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, h, Wr, Ww, kap, D, b, chunk, global_norm, per_state, eps, b_r, b_w):
        chunk = _CHUNK_FWD if chunk is None else min(chunk, _CHUNK_FWD)
        nc = b ** D
        sel = _build_sel(D, b, nc, q.device)
        q, k, v, h, Wr, Ww, kap = (x.contiguous() for x in (q, k, v, h, Wr, Ww, kap))
        num, den, _sv, _sd = _kappa_routed_fwd(q, k, v, h, Wr, Ww, kap, D, b, sel, chunk,
                                               global_norm, per_state, eps, b_r=b_r, b_w=b_w)
        ctx.save_for_backward(q, k, v, h, Wr, Ww, kap, b_r, b_w)
        ctx.D, ctx.b, ctx.chunk = D, b, chunk
        ctx.global_norm, ctx.per_state, ctx.eps = global_norm, per_state, eps
        return num.to(q.dtype), den.to(q.dtype)

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dnum, dden):
        q, k, v, h, Wr, Ww, kap, b_r, b_w = ctx.saved_tensors
        D, b, chunk = ctx.D, ctx.b, ctx.chunk
        global_norm, per_state, eps = ctx.global_norm, ctx.per_state, ctx.eps
        # recompute the per-chunk pre-state snapshots (fp-parity with the fwd scan). The backward runs
        # fp32 operands (deep-Hadamard jacobian precision floor) → ~2x SMEM vs the bf16 fwd, so cap the
        # backward chunk at _CHUNK (the all-backwards device constant); the chunked scan is chunk-size
        # invariant (each chunk is just a tiling of the same sequence). Snapshots recomputed at that chunk.
        nc = b ** D
        sel = _build_sel(D, b, nc, q.device)
        # The fused kappa backward is a single mega-kernel (two carried states + the den coupling + intra
        # num + den), so its per-program fp32 SMEM is heavier than the plain routed backward → cap the
        # backward chunk at _KAPPA_BWD_CHUNK (16) so the [BT,*] tiles fit the ada-class 100KB wall.
        chunk = min(chunk, _KAPPA_BWD_CHUNK)
        brf = None if b_r is None else b_r.float()
        bwf = None if b_w is None else b_w.float()
        q, k, v, h, Wr, Ww, kap = (x.float().contiguous() for x in (q, k, v, h, Wr, Ww, kap))
        _num, _den, snap_val, snap_den = _kappa_routed_fwd(
            q, k, v, h, Wr, Ww, kap, D, b, sel, chunk, global_norm, per_state, eps, b_r=brf, b_w=bwf)
        grads = _kappa_routed_bwd(
            q, k, v, h, Wr, Ww, kap, snap_val, snap_den, dnum.float(), dden.float(),
            D, b, sel, chunk, global_norm, per_state, eps, b_r=brf, b_w=bwf)
        if b_r is None:
            dq, dk, dv, dh, dWr, dWw, dkap = grads
            dbr = dbw = None
        else:
            dq, dk, dv, dh, dWr, dWw, dkap, dbr, dbw = grads
        q0 = ctx.saved_tensors[0]
        # forward arg order: q,k,v,h,Wr,Ww,kap,D,b,chunk,global_norm,per_state,eps,b_r,b_w
        return (dq.to(q0.dtype), dk.to(q0.dtype), dv.to(q0.dtype), dh.to(q0.dtype),
                dWr.to(Wr.dtype), dWw.to(Ww.dtype), dkap.to(q0.dtype),
                None, None, None, None, None, None,
                None if dbr is None else dbr.to(b_r.dtype),
                None if dbw is None else dbw.to(b_w.dtype))


def _kappa_routed_readout(qf, kf, vf, hf, Wr, Ww, kapf, D, b, chunk_size, global_norm, per_state, eps,
                          b_r=None, b_w=None):
    """Fused global/kappa/per_state tree-routed readout returning (num[BH,L,V], den[BH,L,1]),
    differentiable. kapf:[BH,L,1]. Optional routing bias b_r/b_w ∈ [D,b]. CUDA only (the eager fallback
    stays in the public entry)."""
    kap = kapf.reshape(qf.shape[0], qf.shape[1]).contiguous()    # [BH,L]
    num, den = _RoLARoutedKappaFn.apply(qf, kf, vf, hf, Wr, Ww, kap, D, b, chunk_size,
                                        global_norm, per_state, eps, b_r, b_w)
    return num, den.unsqueeze(-1)


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


# ----------------------------------------------------------------------------
# GLA readout as an OPAQUE custom op — the V2 compile twin of `rola::readout_rla` (#30 parity). The
# eager `_RoLAGLAFn` above stays the direct-call path; this op makes the GLA numerator readout
# torch.compile-safe (Triton kernels inside → no Dynamo graph break; the autotuner's `do_bench`→
# `torch.quantile` on symbolic shapes never traces). Mirrors the RLA op EXACTLY incl. the in-op `cdt`
# fp32-cast trick (#27): chunk_rola hands fp32 operands + a `cdt` code, the bf16/fp16 round happens
# INSIDE the op (not the compile-visible glue), so inductor sees fp32→fp32 and the compiled grads land
# at the eager-deterministic noise floor (RLA <0.5%; GLA at the same in-op-cast regime).
@torch.library.custom_op("rola::readout_gla", mutates_args=())
def _readout_gla(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 wg: torch.Tensor, rg: torch.Tensor, ld: torch.Tensor,
                 chunk: int, BG: int, cdt: str) -> torch.Tensor:
    """RoLA-GLA un-normalized routed readout as an opaque custom op. Casts fp32 operands → `cdt` IN-op
    (ld kept fp32 — the decay exp/cumsum is precision-sensitive and the kernel reads ld as fp32)."""
    with torch.autocast('cuda', enabled=False):
        dt = _DT[cdt]
        q, k, v, wg, rg = (t.to(dt) for t in (q, k, v, wg, rg))
        return _gla_fwd(q, k, v, wg, rg, ld.float(), chunk=chunk, BG=BG).to(dt)


@_readout_gla.register_fake
def _readout_gla_fake(q, k, v, wg, rg, ld, chunk, BG, cdt):
    return q.new_empty((q.shape[0], q.shape[1], v.shape[-1]), dtype=_DT[cdt])


def _readout_gla_setup(ctx, inputs, output):
    q, k, v, wg, rg, ld, chunk, BG, cdt = inputs
    ctx.save_for_backward(q, k, v, wg, rg, ld)
    ctx.cdt = cdt


@torch.library.custom_op("rola::readout_gla_bwd", mutates_args=())
def _readout_gla_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     wg: torch.Tensor, rg: torch.Tensor, ld: torch.Tensor,
                     grad: torch.Tensor, cdt: str) -> typing.List[torch.Tensor]:  # noqa: UP006
    """Opaque GLA backward (Triton inside). Casts fp32 saved operands → `cdt` IN-op; ld stays fp32.
    Returns fp32 grads (dq,dk,dv,dwg,drg,dld) — kept fp32 through the compiled glue (see RLA bwd)."""
    with torch.autocast('cuda', enabled=False):
        dt = _DT[cdt]
        q, k, v, wg, rg = (t.to(dt) for t in (q, k, v, wg, rg))
        dq, dk, dvv, dwg, drg, dld = _bwd_split_gla(q, k, v, wg, rg, ld.float(), grad, chunk=_CHUNK)
    return [dq, dk, dvv, dwg, drg, dld]


@_readout_gla_bwd.register_fake
def _readout_gla_bwd_fake(q, k, v, wg, rg, ld, grad, cdt):
    f = torch.float32
    return [q.new_empty(q.shape, dtype=f), k.new_empty(k.shape, dtype=f),
            v.new_empty(v.shape, dtype=f), wg.new_empty(wg.shape, dtype=f),
            rg.new_empty(rg.shape, dtype=f), ld.new_empty(ld.shape, dtype=f)]


def _readout_gla_backward(ctx, grad):
    q, k, v, wg, rg, ld = ctx.saved_tensors
    # native-fp32 grads (see `_readout_rla_backward`): keep the downstream normalize glue fp32.
    dq, dk, dvv, dwg, drg, dld = _readout_gla_bwd(q, k, v, wg, rg, ld, grad, ctx.cdt)
    # forward arg order: q, k, v, wg, rg, ld, chunk, BG, cdt
    return dq, dk, dvv, dwg, drg, dld, None, None, None


_readout_gla.register_autograd(_readout_gla_backward, setup_context=_readout_gla_setup)


def rola_gla_readout_op(q, k, v, r, w, ld, chunk=None, BG=16, compute_dtype=None):
    """Compile-safe GLA numerator readout (the custom-op twin of `rola_rla_triton`). fp32 operands +
    `compute_dtype` → the bf16 round stays in-op. Arg order matches `rola_gla_triton` (r=read, w=write)."""
    chunk = _CHUNK if chunk is None else min(chunk, _CHUNK)
    return _readout_gla(q, k, v, w, r, ld, chunk, BG, _dtcode(compute_dtype or q.dtype))


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


@triton.autotune(configs=_SCAN_CFGS, key=_SCAN_KEY,
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
    # dv constexpr → the value loop unrolls (the de-nest was dropped — runtime cost; see _par_grad_rla_qr).
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


@triton.autotune(configs=_SCAN_CFGS, key=_SCAN_KEY,
                 prune_configs_by={'early_config_prune': _prune_bv}, **autotune_cache_kwargs)
@triton.jit
def _scan_dS(q_ptr, rg_ptr, ld_ptr, g_ptr, dSa_ptr, L, dqk, dv: tl.constexpr, nc,
             sq_b, sq_l, sq_d, sg_b, sg_l, sg_c, sgr_b, sgr_l, sgr_d,
             ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
             USE_G: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr, BV: tl.constexpr,
             BVF: tl.constexpr, BG: tl.constexpr, NCH: tl.constexpr, KSNAP: tl.constexpr):
    # value-OUTER (mirror of _scan_S); ND_V==1 == the un-tiled reverse scan byte-for-byte.
    # dv constexpr → value loop unrolls (de-nest dropped — runtime cost; see _par_grad_rla_qr).
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
    # WV-BT-SPLIT: WV=[BT,BG*BV] is the dominant SMEM tile of the GLA decay-replay (32KB at BT32/BG16/BV16)
    # — the term that pushes GLA-qr 32KB past a 99KB (sm_86) card vs the RLA path (which reuses one tile).
    # Kᵀ·WV contracts the BT (time) axis, so split that contraction into NWV row-blocks: each block
    # materialises only a [BTS,BG*BV] sub-WV (BTS=BT/NWV) and accumulates its dot. The decay exponent is
    # the GLOBAL within-chunk prefix-sum, carried across blocks via `base` (=Σ ld of earlier blocks); Lam
    # (chunk total) is the masked full-load sum (==the old where(offs_t==BT-1,cumsum) since masked rows
    # carry ld=0). Halves the resident replay tile (32→16KB). Exact (matmul contraction split + exact
    # prefix carry); NWV=1 (BT<=16) is byte-identical to the un-split body. tl.dot needs the contracted
    # BT-block >=16 ⇒ BTS>=16 (BT32→NWV2, BT16→NWV1).
    NWV: tl.constexpr = BT // 16 if BT >= 32 else 1
    BTS: tl.constexpr = BT // NWV
    t0 = (t // KSNAP) * KSNAP
    Sflat = tl.load(Sb_ptr + b*ssb_b + sb*ssb_n + (t // KSNAP)*ssb_t
                    + offs_d[:, None]*ssb_d + offs_e[None, :]*ssb_e, mask=dmask[:, None], other=0.0)
    for i in range(KSNAP):
        tt = t0 + i
        rows = tt * BT + offs_t
        rmask = (rows < L) & (tt < t)
        ldc = tl.load(ld_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                      mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        Lam = tl.sum(ldc, axis=0)
        decvec = tl.reshape(tl.exp(Lam)[:, None] * tl.full([BG, BV], 1.0, tl.float32), [BG * BV])
        upd = tl.zeros([BD, BG * BV], dtype=tl.float32)
        base = tl.zeros([BG], dtype=tl.float32)
        for s in tl.static_range(NWV):
            srows = tt * BT + s * BTS + tl.arange(0, BTS)
            srmask = (srows < L) & (tt < t)
            sk = tl.load(k_ptr + b*sq_b + srows[:, None]*sq_l + offs_d[None, :]*sq_d,
                         mask=srmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            sv = tl.load(v_ptr + b*sv_b + srows[:, None]*sv_l + offs_v[None, :]*sv_d,
                         mask=srmask[:, None] & vmask[None, :], other=0.0).to(tl.float32)
            swg = tl.load(wg_ptr + b*sg_b + srows[:, None]*sg_l + offs_c[None, :]*sg_c,
                          mask=srmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
            sld = tl.load(ld_ptr + b*sg_b + srows[:, None]*sg_l + offs_c[None, :]*sg_c,
                          mask=srmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
            a_s = base[None, :] + tl.cumsum(sld, axis=0)
            w_end_s = swg * tl.exp(Lam[None, :] - a_s)
            WVs = tl.reshape(w_end_s[:, :, None] * sv[:, None, :], [BTS, BG * BV])
            upd += tl.dot(tl.trans(sk), WVs.to(sk.dtype))
            base += tl.sum(sld, axis=0)
        Sflat = decvec[None, :] * Sflat + upd
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
    # WV-BT-SPLIT (mirror of _recompute_S_gla): rt_g=[BT,BG*BV] is the dominant replay SMEM tile; Qᵀ·rt_g
    # contracts the BT axis, so split into NWV row-blocks of [BTS,BG*BV], carrying the within-chunk decay
    # prefix `a_s = base + cumsum` across blocks (rt = rgc·e^{a}, forward prefix). Halves the tile
    # (32→16KB); NWV=1 (BT<=16) is byte-identical to the un-split body.
    NWV: tl.constexpr = BT // 16 if BT >= 32 else 1
    BTS: tl.constexpr = BT // NWV
    top = (t // KSNAP) * KSNAP + (KSNAP - 1)
    if top > NCH - 1:
        top = NCH - 1
    dS = tl.load(dSa_ptr + b*ssb_b + sb*ssb_n + (t // KSNAP)*ssb_t
                 + offs_d[:, None]*ssb_d + offs_e[None, :]*ssb_e, mask=dmask[:, None], other=0.0)
    for i in range(KSNAP):
        tt = top - i
        rows = tt * BT + offs_t
        rmask = (rows < L) & (tt > t)
        ldc = tl.load(ld_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                      mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        Lam = tl.sum(ldc, axis=0)
        decvec = tl.reshape(tl.exp(Lam)[:, None] * tl.full([BG, BV], 1.0, tl.float32), [BG * BV])
        upd = tl.zeros([BD, BG * BV], dtype=tl.float32)
        base = tl.zeros([BG], dtype=tl.float32)
        for s in tl.static_range(NWV):
            srows = tt * BT + s * BTS + tl.arange(0, BTS)
            srmask = (srows < L) & (tt > t)
            sq = tl.load(q_ptr + b*sq_b + srows[:, None]*sq_l + offs_d[None, :]*sq_d,
                         mask=srmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            srg = tl.load(rg_ptr + b*sg_b + srows[:, None]*sg_l + offs_c[None, :]*sg_c,
                          mask=srmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
            sg_v = tl.load(g_ptr + b*sgr_b + srows[:, None]*sgr_l + offs_v[None, :]*sgr_d,
                           mask=srmask[:, None] & vmask[None, :], other=0.0).to(tl.float32)
            sld = tl.load(ld_ptr + b*sg_b + srows[:, None]*sg_l + offs_c[None, :]*sg_c,
                          mask=srmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
            a_s = base[None, :] + tl.cumsum(sld, axis=0)
            rt_s = srg * tl.exp(a_s)
            rt_g_s = tl.reshape(rt_s[:, :, None] * sg_v[:, None, :], [BTS, BG * BV])
            upd += tl.dot(tl.trans(sq), rt_g_s.to(sq.dtype))
            base += tl.sum(sld, axis=0)
        dS = decvec[None, :] * dS + upd
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
    # dqk/dv stay constexpr so the feature/value loops UNROLL. A runtime-ND de-nest (#22) was measured to
    # shrink codegen but cost +52% step-time (this kernel is compute-bound — it needs the unroll's ILP; even
    # the ND==1 fast path slowed). The codegen cut is delivered by the config curation (`_bwd_ws`, runtime-
    # neutral) instead, so the loops are left unrolled.
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
    # dqk/dv constexpr (loops unrolled) — see _par_grad_rla_qr on why de-nest was dropped (runtime cost).
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


@triton.autotune(configs=_BWD_CFGS_BV, key=_AT_KEY,
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
    # dqk/dv constexpr (loops unrolled) — see _par_grad_rla_qr on why de-nest was dropped (runtime cost).
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


@triton.autotune(configs=_BWD_CFGS_BV, key=_AT_KEY,
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
    # dqk/dv constexpr (loops unrolled) — see _par_grad_rla_qr on why de-nest was dropped (runtime cost).
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
# The den FORWARD kernels (_den_fwd_intra/_den_fwd_inter) are RLA/GLA-shared (single fn, USE_G param), so
# they key on USE_G too; the den BACKWARD kernels are separate RLA/GLA fns (no USE_G arg) → plain _DEN_KEY.
_DEN_FWD_KEY = ['dqk', 'nc', 'USE_G']


@triton.autotune(configs=_BWD_CFGS, key=_DEN_FWD_KEY,
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


@triton.autotune(configs=_BWD_CFGS, key=_DEN_FWD_KEY, reset_to_zero=['d_ptr'],
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
            chunk: int, BG: int, cdt: str) -> typing.List[torch.Tensor]:  # noqa: UP006
    """Per-state denominator as an opaque custom op; returns [d, Zb] (Zb saved for backward). Casts
    fp32 operands → `cdt` IN-op (keeps the bf16 round out of the compiled glue — see readout op)."""
    with torch.autocast('cuda', enabled=False):
        q, k, wg = (t.to(_DT[cdt]) for t in (q, k, wg))
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
def _den_op_fake(q, k, wg, chunk, BG, cdt):
    B, L, dqk = q.shape
    nc = wg.shape[-1]
    BD = max(16, triton.next_power_of_2(dqk))
    NB = triton.cdiv(nc, BG)
    NCH = triton.cdiv(L, chunk)
    return [q.new_empty((B, L, nc), dtype=torch.float32),
            q.new_empty((B, NB, NCH, BD, BG), dtype=torch.float32)]


@torch.library.custom_op("rola::den_bwd", mutates_args=())
def _den_bwd_op(q: torch.Tensor, k: torch.Tensor, wg: torch.Tensor, Zb: torch.Tensor,
                gd: torch.Tensor, chunk: int, BG: int, cdt: str) -> typing.List[torch.Tensor]:  # noqa: UP006
    with torch.autocast('cuda', enabled=False):
        dt = _DT[cdt]
        q, k, wg = (t.to(dt) for t in (q, k, wg))
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
def _den_bwd_op_fake(q, k, wg, Zb, gd, chunk, BG, cdt):
    B, L, dqk = q.shape
    nc = wg.shape[-1]
    f = torch.float32
    return [q.new_empty((B, L, dqk), dtype=f), q.new_empty((B, L, dqk), dtype=f), q.new_empty((B, L, nc), dtype=f)]


def _den_setup(ctx, inputs, output):
    q, k, wg, chunk, BG, cdt = inputs
    ctx.save_for_backward(q, k, wg, output[1])
    ctx.chunk = chunk
    ctx.BG = BG
    ctx.cdt = cdt


def _den_backward(ctx, grad):
    q, k, wg, Zb = ctx.saved_tensors
    grad_d = grad[0] if isinstance(grad, (list, tuple)) else grad   # list-output op: grad is [grad_d, grad_Zb]
    dq, dk, dw = _den_bwd_op(q, k, wg, Zb, grad_d, ctx.chunk, ctx.BG, ctx.cdt)
    # Keep the den-pre-pass grads fp32 (see `_readout_rla_backward`): the den-grad accumulates into the
    # SAME qf/kf/wf forks as the readout grad, so an fp32 sum here keeps the compiled grad <0.5%.
    return dq, dk, dw, None, None, None


_den_op.register_autograd(_den_backward, setup_context=_den_setup)


@input_guard
def rola_perstate_den_triton(q, k, w, chunk=None, BG=16, compute_dtype=None):
    """Per-state denominator on folded [BH,L,*] tensors. Differentiable (Triton fwd + parallel bwd).
    `compute_dtype`: pass fp32 operands + the kernel compute dtype to keep the bf16 round in-op (the
    torch.compile grad-noise fix); default None derives the code from the operand dtype (direct callers)."""
    chunk = _CHUNK if chunk is None else min(chunk, _CHUNK)
    d, _Zb = _den_op(q, k, w, chunk, BG, _dtcode(compute_dtype or q.dtype))
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


# ----------------------------------------------------------------------------
# GLA per-state den as an OPAQUE custom op — the V2 compile twin of `rola::den` (#30 parity). Mirrors
# the RLA den op EXACTLY (returns [d, Zb]; Zb saved for backward) with the decay `ld` threaded through
# at USE_G=True + the in-op `cdt` fp32-cast trick (ld stays fp32). Makes the GLA den torch.compile-safe.
@torch.library.custom_op("rola::den_gla", mutates_args=())
def _den_gla_op(q: torch.Tensor, k: torch.Tensor, wg: torch.Tensor, ld: torch.Tensor,
                chunk: int, BG: int, cdt: str) -> typing.List[torch.Tensor]:  # noqa: UP006
    """Per-state decayed denominator as an opaque custom op; returns [d, Zb]. Casts fp32 operands → `cdt`
    IN-op (ld kept fp32 — the e^Λ decay is precision-sensitive). USE_G=True decay pre-pass (see _DenGLAFn)."""
    with torch.autocast('cuda', enabled=False):
        dt = _DT[cdt]
        q, k, wg = (t.to(dt) for t in (q, k, wg))
        ld = ld.clamp(min=_GLA_FLOOR).float()
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
        _den_fwd_intra[(B, NB, NCH)](q, k, wg, ld, d_intra, L, dqk, nc, *sq, *sg, *sd,
                                     USE_G=True, BT=chunk, BG=BG)

        def grid_inter(meta):
            return (B, NB, triton.cdiv(dqk, meta['BD']))
        _den_fwd_inter[grid_inter](q, k, wg, ld, d_inter, Zb, L, dqk, nc, *sq, *sg, *sd, *sZ,
                                   USE_G=True, BT=chunk, BG=BG, NCH=NCH)
        return [d_intra + d_inter, Zb]


@_den_gla_op.register_fake
def _den_gla_op_fake(q, k, wg, ld, chunk, BG, cdt):
    B, L, dqk = q.shape
    nc = wg.shape[-1]
    BD = max(16, triton.next_power_of_2(dqk))
    NB = triton.cdiv(nc, BG)
    NCH = triton.cdiv(L, chunk)
    return [q.new_empty((B, L, nc), dtype=torch.float32),
            q.new_empty((B, NB, NCH, BD, BG), dtype=torch.float32)]


@torch.library.custom_op("rola::den_gla_bwd", mutates_args=())
def _den_gla_bwd_op(q: torch.Tensor, k: torch.Tensor, wg: torch.Tensor, ld: torch.Tensor,
                    Zb: torch.Tensor, gd: torch.Tensor, chunk: int, BG: int, cdt: str) -> typing.List[torch.Tensor]:  # noqa: UP006
    with torch.autocast('cuda', enabled=False):
        dt = _DT[cdt]
        q, k, wg = (t.to(dt) for t in (q, k, wg))
        ld = ld.clamp(min=_GLA_FLOOR).float()
        B, L, dqk = q.shape
        nc = wg.shape[-1]
        BD = max(16, triton.next_power_of_2(dqk))
        NB = triton.cdiv(nc, BG)
        NCH = triton.cdiv(L, chunk)
        gd = gd.contiguous().to(q.dtype)
        dZa = torch.empty_like(Zb)
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
                                    dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                                    dw.stride(0), dw.stride(1), dw.stride(2),
                                    BT=chunk, BG=BG, NCH=NCH)
        return [dq.sum(1), dk.sum(1), dw, dld]


@_den_gla_bwd_op.register_fake
def _den_gla_bwd_op_fake(q, k, wg, ld, Zb, gd, chunk, BG, cdt):
    B, L, dqk = q.shape
    nc = wg.shape[-1]
    f = torch.float32
    return [q.new_empty((B, L, dqk), dtype=f), q.new_empty((B, L, dqk), dtype=f),
            q.new_empty((B, L, nc), dtype=f), q.new_empty((B, L, nc), dtype=f)]


def _den_gla_setup(ctx, inputs, output):
    q, k, wg, ld, chunk, BG, cdt = inputs
    ctx.save_for_backward(q, k, wg, ld, output[1])
    ctx.chunk = chunk
    ctx.BG = BG
    ctx.cdt = cdt


def _den_gla_backward(ctx, grad):
    q, k, wg, ld, Zb = ctx.saved_tensors
    grad_d = grad[0] if isinstance(grad, (list, tuple)) else grad   # list-output op: grad is [grad_d, grad_Zb]
    dq, dk, dw, dld = _den_gla_bwd_op(q, k, wg, ld, Zb, grad_d, ctx.chunk, ctx.BG, ctx.cdt)
    # forward arg order: q, k, wg, ld, chunk, BG, cdt
    return dq, dk, dw, dld, None, None, None


_den_gla_op.register_autograd(_den_gla_backward, setup_context=_den_gla_setup)


def rola_perstate_den_gla_op(q, k, w, ld, chunk=None, BG=16, compute_dtype=None):
    """Compile-safe GLA per-state den (the custom-op twin of `rola_perstate_den_triton`). fp32 operands
    + `compute_dtype` → the bf16 round stays in-op; ld fp32."""
    chunk = _CHUNK if chunk is None else min(chunk, _CHUNK)
    d, _Zb = _den_gla_op(q, k, w, ld, chunk, BG, _dtcode(compute_dtype or q.dtype))
    return d


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


def _rola_readout(qf, kf, vf, rf, wf, gf, chunk_size, compute_dtype=None):
    """Folded routed readout (numerator-only). CUDA → device-agnostic Triton kernels; else → eager
    core. The global denominator is the caller's separate per-state den pre-pass. `compute_dtype` (RLA
    only) routes fp32 operands through the in-op bf16 cast (the torch.compile grad-noise fix)."""
    if qf.is_cuda:
        if gf is None:
            return rola_rla_triton(qf, kf, vf, rf, wf, chunk=chunk_size, compute_dtype=compute_dtype)
        # GLA: the compile-safe op (the in-op `cdt` cast keeps the bf16 round out of the compiled glue,
        # mirroring RLA) — fp32 operands + compute_dtype, NOT a glue-side bf16 cast.
        return rola_gla_readout_op(qf, kf, vf, rf, wf, gf, chunk=chunk_size, compute_dtype=compute_dtype)
    return _rola_chunk_core(qf, kf, vf, wf, rf, gf, chunk_size)


@input_guard
def _chunk_rola_impl(q, k, v, r, w, g=None, norm='kappa', kappa=None, scale=None, eps=1e-5,
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

    # Keep the FOLDED inputs as fp32 autograd nodes. The bf16 (compute-dtype) round happens INSIDE the
    # opaque RLA readout/den ops (`compute_dtype=` below), NOT in this torch.compile-visible glue — that
    # is the grad-noise fix: a glue-side bf16 cast lets inductor fuse the cast-backward with the kernel-
    # grad-consuming region in bf16 (~1% gram-grad noise vs eager); with the cast opaque, inductor sees
    # fp32→fp32 and the round is eager-deterministic. The GLA twin (gf≠None) NOW takes the SAME in-op
    # `cdt` cast (#30 V2: its readout/den are custom_ops too) — fp32 operands + compute_dtype, no glue
    # bf16 cast — so GLA compiles with the same <0.5% grad noise. `compute_dtype=q.dtype` (no autocast)
    # is the fp32 pass-through. ld/gf stays fp32 (the decay exp is precision-sensitive, read fp32 in-op).
    qf, kf, vf, wf = fold(q).float() * scale, fold(k).float(), fold(v).float(), fold(w).float()
    gf = fold(g).float() if g is not None else None
    cdt = compute_dtype  # the kernel compute dtype (bf16 under autocast, else q.dtype)

    if norm == 'raw':
        rout = _rola_readout(qf, kf, vf, fold(r).float(), wf, gf, chunk_size, compute_dtype=cdt)
        return unfold(rout).to(v.dtype)

    # global / per_state / kappa: per-state den pre-pass → rescale read gates → numerator-only
    # readout → divide by the reconstructed global den Σ_c r̃ᶜ·dᶜ.
    if not q.is_cuda:
        d = _perstate_den_torch(qf, kf, wf, gf, chunk_size, eps)
    elif gf is not None:
        d = rola_perstate_den_gla_op(qf, kf, wf, gf, compute_dtype=cdt)
    else:
        d = rola_perstate_den_triton(qf, kf, wf, compute_dtype=cdt)
    # Read-gate rescale r̃ = r·(d+ε)^{−κ} | r/(d+ε) | r. fp32 `rf32` feeds the den sum + final divide;
    # the readout op gets fp32 r̃ + the in-op `cdt` cast (RLA and GLA alike).
    rf32 = fold(r).float()
    if norm == 'kappa':
        rf32 = rf32 * (d + eps).pow(-fold(kappa).float())
    elif norm == 'per_state':
        rf32 = rf32 / (d + eps)
    num = _rola_readout(qf, kf, vf, rf32, wf, gf, chunk_size, compute_dtype=cdt)
    den = (rf32 * d).sum(-1, keepdim=True)
    out = unfold(num.float() / (den + eps)).to(v.dtype)
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


# ----------------------------------------------------------------------------
# `chunk_rola` — DEFAULT torch.compile path. The custom-op restructure made the RLA kernels opaque to
# Dynamo (0 graph breaks), so torch.compile only fuses the ~550 elementwise GLUE ops between them
# (autocast casts, num/den normalize, the kappa pow, fold copies) — a measured −15% (nc=64) to −26%
# (nc=256) step-time win. The fused-backward glue's grad noise (inductor reordering bf16 elementwise)
# is FIXED upstream: the gram-side bf16 round now happens INSIDE the opaque RLA readout/den ops, not in
# the compile-visible glue, so compiled grads land <0.5% vs eager (max ~0.1%). Opt out with
# ROLA_NO_COMPILE=1 (e.g. for debugging, or py<3.11 where torch.compile is unavailable).
#
# GLA (g≠None) NOW compiles too (#30 V2): its readout + den are wrapped as `rola::readout_gla` /
# `rola::den_gla` custom_ops (mirroring the RLA ops, incl. the in-op `cdt` fp32-cast trick), so the
# Triton autotuner is opaque to Dynamo (no `do_bench`→`torch.quantile`-on-symbolic-shapes trace) and
# the bf16 round stays in-op → 0 graph breaks, compiled grads at the same <0.5% noise floor as RLA.
_ROLA_NO_COMPILE = os.environ.get('ROLA_NO_COMPILE', '0') not in ('0', '', 'false', 'False')
_chunk_rola_compiled = None


def chunk_rola(q, k, v, r, w, g=None, norm='kappa', kappa=None, scale=None, eps=1e-5,
               initial_state=None, output_final_state=False):
    """Routed RoLA readout (see `_chunk_rola_impl`). torch.compile is the DEFAULT for BOTH the RLA and
    GLA paths (lazily compiled on first call; the GLA readout/den are custom_op-wrapped — #30 V2 — so
    Dynamo stays fullgraph). Set ROLA_NO_COMPILE=1 to force eager. Compile is skipped only on CPU (the
    eager fallback) and whenever Dynamo is already tracing (avoid nested-compile recursion)."""
    global _chunk_rola_compiled
    kw = dict(g=g, norm=norm, kappa=kappa, scale=scale, eps=eps,
              initial_state=initial_state, output_final_state=output_final_state)
    if _ROLA_NO_COMPILE or not q.is_cuda or torch.compiler.is_compiling():
        return _chunk_rola_impl(q, k, v, r, w, **kw)
    if _chunk_rola_compiled is None:
        _chunk_rola_compiled = torch.compile(_chunk_rola_impl)
    return _chunk_rola_compiled(q, k, v, r, w, **kw)


# ============================================================================
# `chunk_rola_routed` — public TREE-ROUTED entry point. DIFFERENTIABLE end-to-end.
#
# The in-kernel-routing counterpart of `chunk_rola`: instead of precomputed gates r,w it takes the
# hidden state h + per-level router weights Wr,Ww and builds the routing gram IN-KERNEL (the [L,nc]
# gates are never materialized). Flat (D=1, b=nc) is the strict fused equivalent of `chunk_rola`'s
# precomputed-gate path with r,w the D=1 router's explicit softmax gates — confirming this is a
# generalization that reuses the real RLA kernel.
#
# The numerator readout (`rola_rla_routed_triton`) is a full autograd.Function (`_RoLARoutedFn`):
# forward runs the optimized routed kernels, backward folds the transient [BT,nc] gate-grads into
# dWr,dWw,d_h in-kernel (the [L,nc] gates AND their grads never materialize). So 'raw'/'global' (which
# use the in-kernel numerator) are trainable WITHOUT ever materializing the gate tensor in the readout.
#
# 'kappa'/'per_state' (the PRODUCTION norm) rescale the read gate by the per-(token,state) factor
# r̃ = r·(d+ε)^{−κ} | r/(d+ε), where d_i^c = Σ_{j≤i}(φq_i·φk_j) w_j^c is the per-state denominator. This
# is now ALSO fully fused (`_RoLARoutedKappaFn`): both d AND r̃ are computed TRANSIENTLY per chunk —
# d via a carried den state Sden^c alongside the value state Sval^c, r̃ as a [BT,nc] SRAM tile — so NO
# [*,L,nc] d / r̃ / gate buffer is ever written (validated bit-for-bit vs the explicit-gate math, and
# the [L,nc]-free property is asserted in the tests). The rescale gram is formed nc-wide (the accepted
# cost — we lose factored-gram compute for the read side but keep the [L,nc]-activation + tree-param
# wins). Backward is a reverse chunk-scan that folds the transient gate-grads (+ dκ) into the router.
# RLA only here (g=None); the GLA (scalar-decay) routed twin is a further extension.
# ============================================================================


def _tree_gates_torch(hf, Wr, Ww, D, b, b_r=None, b_w=None):
    """Build the EXPLICIT folded [BH,T,nc] read/write gates from (h, Wr, Ww) the way the tree
    factorizes — r[...,leaf] = Π_lvl softmax(h·Wr[lvl] + b_r[lvl])[..., digit_lvl(leaf)]. This is the
    MATERIALIZED path the routed kernel avoids; used here only for (a) the normalized-norm den pre-pass
    and (b) the flat-equivalence validation reference. Optional per-level bias b_r/b_w ∈ [D,b]."""
    nc = b ** D

    def _logit(x, W, bias, i):
        z = x @ W[i].to(x.dtype)
        return z if bias is None else z + bias[i].to(x.dtype)
    fr = torch.stack([torch.softmax(_logit(hf, Wr, b_r, i), dim=-1) for i in range(D)], 0)  # [D,BH,T,b]
    fw = torch.stack([torch.softmax(_logit(hf, Ww, b_w, i), dim=-1) for i in range(D)], 0)
    # OUT-OF-PLACE leaf-product (column-stack), so the gates stay differentiable w.r.t. b_r/b_w (the
    # in-place `r[...,leaf] *= ` aliases a view and breaks autograd through the optional bias inputs).
    rc, wc = [], []
    for leaf in range(nc):
        digs = [(leaf // (b ** (D - 1 - i))) % b for i in range(D)]
        rr, ww = fr[0, ..., digs[0]], fw[0, ..., digs[0]]
        for i in range(1, D):
            rr, ww = rr * fr[i, ..., digs[i]], ww * fw[i, ..., digs[i]]
        rc.append(rr)
        wc.append(ww)
    return torch.stack(rc, -1), torch.stack(wc, -1)


def _rola_routed_readout(qf, kf, vf, hf, Wr, Ww, D, b, chunk_size, b_r=None, b_w=None):
    """Folded tree-routed numerator-only readout. CUDA → in-kernel routed Triton kernels (gates never
    materialized); else → eager core on explicit gates (capability fallback). Optional bias b_r/b_w."""
    if qf.is_cuda:
        return rola_rla_routed_triton(qf, kf, vf, hf, Wr, Ww, D, b, chunk=chunk_size, b_r=b_r, b_w=b_w)
    r, w = _tree_gates_torch(hf, Wr, Ww, D, b, b_r=b_r, b_w=b_w)
    return _rola_chunk_core(qf, kf, vf, w, r, None, chunk_size)


@input_guard
def chunk_rola_routed(q, k, v, h, Wr, Ww, D, b, norm='kappa', kappa=None, scale=None, eps=1e-5,
                      b_r=None, b_w=None):
    """TREE-ROUTED RoLA (in-kernel routing) readout with built-in normalization. DIFFERENTIABLE
    end-to-end. ALL norms (incl. the production 'kappa'/'per_state') are fully fused — the [L,nc]
    gates, the per-state den d, and the rescaled read gate r̃ are NEVER materialized (see header).

    Args:
        q, k:  φ-mapped queries/keys [B, T, H, K].
        v:     values [B, T, H, V].
        h:     pre-routing hidden state [B, T, H, d_model] (the router lives in the kernel).
        Wr,Ww: per-level read/write router weights [D, d_model, b]  (b^D = nc).
        D, b:  tree depth and branching (flat: D=1,b=nc; square: D=2,b=√nc; tree: b=2).
        norm:  'raw' | 'global' | 'per_state' | 'kappa'.
        kappa: per-token exponent [B,T,H,1] (required for norm='kappa').
        scale: query scale (default 1/sqrt(K)).
        b_r,b_w: OPTIONAL per-level routing bias [D, b] — the affine term of softmax(h·W + b), giving
                 the routing a non-uniform prior (default None = uniform start, backward-compatible).
    Returns:
        Normalized readout [B, T, H, V] ('raw' returns the un-normalized numerator). The [L,nc] gates
        are NEVER materialized in the readout's routing gram.
    """
    if norm not in _NORMS:
        raise ValueError(f"norm must be one of {_NORMS}, got {norm!r}")
    if norm == 'kappa' and kappa is None:
        raise ValueError("norm='kappa' requires a per-token `kappa` exponent tensor [B,T,H,1]")
    if (b_r is None) != (b_w is None):
        raise ValueError("routing bias: pass both b_r and b_w, or neither")
    B, T, H, K = q.shape
    if scale is None:
        scale = K ** -0.5
    chunk_size = min(64, max(16, triton.next_power_of_2(T)))

    def fold(t):
        return t.permute(0, 2, 1, 3).reshape(B * H, T, t.shape[-1])

    def unfold(t):
        return t.view(B, H, T, -1).permute(0, 2, 1, 3).contiguous()

    compute_dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else q.dtype

    def foldc(t):
        return fold(t).to(compute_dtype)

    qf, kf, vf, hf = foldc(q) * scale, foldc(k), foldc(v), foldc(h)
    Wr = Wr.to(compute_dtype)
    Ww = Ww.to(compute_dtype)
    if b_r is not None:
        b_r, b_w = b_r.to(compute_dtype), b_w.to(compute_dtype)

    if norm == 'raw':
        return unfold(_rola_routed_readout(qf, kf, vf, hf, Wr, Ww, D, b, chunk_size,
                                           b_r=b_r, b_w=b_w)).to(v.dtype)

    # 'kappa'/'per_state': the production read-gate rescale r̃ = r·(d+ε)^{−κ} | r/(d+ε), where the
    # per-state den d_i^c = Σ_{j≤i} (φq_i·φk_j) w_j^c. The FUSED path (`_kappa_routed_readout`) computes
    # BOTH d AND r̃ TRANSIENTLY per chunk (carrying a den state Sden^c alongside the value state) and
    # never materializes any [L,nc] gate / den / r̃ buffer — the production normalization, IN-KERNEL.
    # Differentiable end-to-end (fused router-grad fold + dκ); the un-divided (num, den) come back and
    # the divide is plain torch. 'global' keeps r̃=r so it stays the in-kernel numerator + den pre-pass.
    if norm in ('global', 'kappa', 'per_state') and qf.is_cuda:
        # `global` is `kappa` with the rescale skipped (r̃=r): same in-kernel den machinery (the carried
        # Sden^c den state + transient [BT,nc] d), den D_i=Σ_c r^c d^c, never a [L,nc] gate/den/r̃ buffer.
        kapf = fold(kappa) if norm == 'kappa' else qf.new_ones(qf.shape[0], qf.shape[1], 1)
        num, den = _kappa_routed_readout(qf, kf, vf, hf, Wr, Ww, kapf.to(compute_dtype), D, b,
                                         chunk_size, global_norm=(norm == 'global'),
                                         per_state=(norm == 'per_state'), eps=eps,
                                         b_r=b_r, b_w=b_w)
        return unfold(num.float() / (den.float() + eps)).to(v.dtype)

    # CPU/capability fallback (qf not on CUDA) for all normalized norms: the per-state den pre-pass on
    # explicit gates + the eager-core numerator. ALL CUDA normalized norms (incl. 'global') route through
    # the fused in-kernel den path above, so `_tree_gates_torch` is never on the production CUDA path.
    # The bias is threaded here too (out-of-place gates) so the fallback honors softmax(h·W+b).
    rf, wf = _tree_gates_torch(hf, Wr, Ww, D, b, b_r=b_r, b_w=b_w)
    rf, wf = rf.to(compute_dtype), wf.to(compute_dtype)
    d = _perstate_den_torch(qf, kf, wf, None, chunk_size, eps)
    if norm == 'kappa':
        rf_scaled = (rf * (d + eps).pow(-fold(kappa).to(d.dtype))).to(compute_dtype)
    elif norm == 'per_state':
        rf_scaled = (rf / (d + eps)).to(compute_dtype)
    else:  # global
        rf_scaled = rf
    num = _rola_chunk_core(qf, kf, vf, wf, rf_scaled, None, chunk_size)
    den = (rf_scaled * d).sum(-1, keepdim=True)
    return unfold(num / (den + eps)).to(v.dtype)
