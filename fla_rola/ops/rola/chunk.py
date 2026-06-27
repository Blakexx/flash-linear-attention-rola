# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# RoLA routing — Triton kernels (additive extension of simple_gla).
#
# Routed linear attention shares the content gram G=qkᵀ across `nc` states and modulates it by a
# routing gram R=Σ_c r_i^c w_j^c (+ optional per-state scalar decay). The Triton kernels here compute the
# *un-normalized* routed readout O = (G∘R∘causal) @ v — the FLA convention; the global denominator is
# reconstructed by the caller as Σ_c r̃ᶜ·dᶜ from a per-state den pre-pass (see `chunk_rola` at the
# bottom of this file — the norm-aware public entry point, FLA-style). Tiled
# over state-blocks (BG states/program) so only this block's slice of the Kronecker state lives in
# SRAM → scales to any nc. The content gram is formed ONCE per chunk (the FLOP win), never
# materializing the L×nc product nor replicating q/k.
#
# This file holds the production IN-KERNEL TREE-ROUTING kernels (RLA + GLA, fwd+bwd; the routing gram is
# built from the hidden state h + per-head router weights, so the [L,nc] gates are never materialized)
# and the pure-torch reference naive (`_rola_chunk_core` / `chunk_rola`). The readout is numerator-only
# (width dv, BV=next_pow2(dv)); there is no ones-column augmentation — the denominator is a separate
# per-state pre-pass.

import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from fla_rola.ops.rola.routed_bwd_kernels import (  # production tree-routed backward kernels (the in-kernel router-grad fold)
    _build_alpha,  # in-kernel per-state log-decay helpers (#45): alpha=sigmoid(h·Wg), ld=clamp(log(1-w(1-alpha)))
    _ld_from_w,
)
from fla_rola.ops.rola.routed_bwd_kernels import (
    _bwd_inter_read_kernel as _routed_bwd_inter_read,
)
from fla_rola.ops.rola.routed_bwd_kernels import (
    _bwd_inter_state_kernel as _routed_bwd_inter_state,
)
from fla_rola.ops.rola.routed_bwd_kernels import (
    _bwd_intra_kernel as _routed_bwd_intra,
)
from fla_rola.ops.rola.routed_bwd_kernels import (
    _fold_kernel as _routed_bwd_fold,
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
# The routed forward inter-scan is SHARED by RLA (USE_G=False) and GLA (USE_G=True) at the SAME
# (dqk,dv,nc); without USE_G in the key the autotune config-cache collides → RLA and GLA reuse each
# other's tuned warps/stages/BV. USE_G is a constexpr (so it specializes the compile regardless), but
# it must also gate config SELECTION so each variant tunes its own (the GLA decay-replay has a heavier
# SMEM profile than RLA, so the best config differs). Perf-only; no correctness change. (#22)
# nc is EXCLUDED on purpose: it only sets the host-side grid trip count `NB = cdiv(nc, BG)` (see ~L494)
# — the per-program state block is a fixed BG-wide tile, so nc changes NEITHER a kernel constexpr tile
# NOR per-program SMEM. The best warps/stages/BV is therefore nc-INVARIANT, and keying on nc forced a
# needless full re-tune at every states-per-head in the scaling sweep. Perf-only (config REUSE across
# nc); correctness is unaffected — the kernel still specializes on its real constexprs.
_SCAN_KEY = ['dqk', 'dv', 'USE_G']
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
# A100 with room, BV=32/16 (2/4 blocks) on a 99KB card. d_v is a constexpr autotune key, so
# cdiv(d_v,BV)=1 at d_v<=BV UNROLLS to byte-identical code — zero perf impact for the configs we run.
_BWD_BV = (16, 32, 64)                     # value-tile candidates; pruned to <= next_pow2(d_v)


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


# scans have no BD knob (Sflat is a register carry, own fixed feature block) but DO need the BV knob.
# The scan footprint is BD_scan(<=64)×BV; treat as the medium/large class by BV alone.
_SCAN_CFGS = [triton.Config({'BV': bv}, num_warps=w, num_stages=s)
              for bv in _BWD_BV for (w, s) in _bwd_ws(64, bv)]


def _bv_cap(dv):
    return max(16, triton.next_power_of_2(dv))


# Feature/value tiling for the FUSED kappa/per_state kernels and the routed numerator backward. Unlike
# the simpler numerator-only routed forward, these kernels build [BC*BK, BV] state slices and [BT, BC*BK]
# read/write tiles with BC=16, so the dominant fp32 tile is BC·BK·BV·4 bytes (two copies live at the
# state read+write). We tile BOTH axes: cap BK<=64 (loop ND=cdiv(dqk,BK)) AND cap the value tile BV (loop
# ND_V=cdiv(dv,BV)) so BC·BK·BV stays under a conservative budget. BK=BV=16 always survives (the floor).
# Both loops are pure reduction-order / output-block reassociations → bit-faithful. (`_kappa_bk_cap`
# keeps the value-free backward — which has no BV loop — fitting by capping BK against max(BV,BT).)
#
# CURRENT-BUDGET INVARIANT: at the 24KB budget below BOTH helpers ALWAYS return 16 for every shape
# the kernels are tested on (dqk,dv,chunk all ≤128 — verified). The BK=32/BV=32 branches are therefore
# UNTESTED. If you raise the budget (or the tested-shape envelope grows), the cap can return 32 and
# silently activate those untested tile paths — re-validate the fused kappa fwd/bwd bit-faithfulness
# (test_kappa_routed_*) BEFORE trusting BK/BV>16. The asserts in the two helpers make a >16 result LOUD.
def _kappa_bk_cap(dqk, dv, chunk, bc=16):
    bv = _bv_cap(dv)
    want = min(64, max(16, triton.next_power_of_2(dqk)))
    budget = 24 * 1024                       # bytes for ONE [BC*BK, max(BV,BT)] fp32 tile (several live +
    span = bc * max(bv, chunk) * 4           # the fp32 backward operands + pipelining, within the ~100KB cap)
    bk = want
    while bk > 16 and bk * span > budget:
        bk //= 2
    # CURRENT-BUDGET INVARIANT (see header): bk>16 activates an UNTESTED tile path. Loud, not silent.
    assert bk == 16, (
        f"_kappa_bk_cap returned BK={bk}>16 (dqk={dqk}, dv={dv}, chunk={chunk}, budget={budget}). The "
        "BK>16 path is untested — re-validate test_kappa_routed_* bit-faithfulness, then relax this assert."
    )
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
    # CURRENT-BUDGET INVARIANT (see header): bv>16 activates an UNTESTED value-tile path. Loud, not silent.
    assert bv == 16, (
        f"_kappa_bv_tile returned BV={bv}>16 (dqk={dqk}, dv={dv}, budget={budget}). The BV>16 path is "
        "untested — re-validate test_kappa_routed_* bit-faithfulness, then relax this assert."
    )
    return bv


_ROUTER_BD_BUDGET = 8192   # elements (32KB fp32) for the in-kernel router's [BD, BB] weight tile (one
#                            tile live at a time — the kappa kernels run num_stages=1, no double-buffer)
_ROUTER_BD_MAX = 256       # absolute BD cap (bounds the hc[BT,BD] tile + keeps the NDM-unroll IR small)


def _router_bd_ndm(d_model, b):
    """BD (the d_model tile width of the IN-KERNEL router build — `_build_rw_tile`/`_build_factors`/
    `_build_alpha`, all of which loop `for dm in range(NDM)`) RIGHT-SIZED to the LARGEST pow2 whose
    [BD, BB] router-weight tile fits a safe SRAM fraction, with NDM=cdiv(d_model, BD) tiling the rest.
    For small d_model this returns BD=next_pow2(d_model), NDM=1 — byte-identical to the un-tiled build.

    F5: with the old BD=next_pow2(d_model) the whole router weight was ONE block (NDM=1), so flat
    nc=64/256 at d_model=1024 (BD=1024) OOM'd. Capping BD by a FIXED byte budget keeps the tile in SRAM;
    crucially we pick the LARGEST BD that fits (small NDM) — NOT the smallest — so the constexpr `for dm in
    range(NDM)` unroll stays short (fast compile + good ILP). For the common small-b routings (BB=16) BD
    lands at the 256 cap (NDM=4 at d_model=1024); only true flat routing (BB=nc large) drives NDM up,
    because [BD,BB] is intrinsically wide there. All builders NDM-loop, so the cap only splits the d_model
    reduction into NDM blocks (output unchanged to fp tolerance)."""
    BB = max(16, triton.next_power_of_2(b))
    bd_full = max(16, triton.next_power_of_2(d_model))
    bd_cap = triton.next_power_of_2(max(1, _ROUTER_BD_BUDGET // BB))
    while bd_cap > 16 and bd_cap * BB > _ROUTER_BD_BUDGET:
        bd_cap //= 2          # round DOWN to the largest pow2 keeping [BD,BB] within budget (floor 16)
    BD = max(16, min(bd_full, bd_cap, _ROUTER_BD_MAX))
    return BD, triton.cdiv(d_model, BD)


def _prune_bv(configs, named_args, **kwargs):
    """Cap BV at next_pow2(d_v): a bigger value-tile than the value dim is pure waste (and would blow
    up the config grid). Floor 16 always survives — the routed forward inter-scan's value-axis prune."""
    try:
        cap = _bv_cap(named_args['dv'])
    except Exception:
        return configs
    keep = [c for c in configs if c.kwargs.get('BV', 16) <= cap]
    return keep or [c for c in configs if c.kwargs.get('BV', 16) == 16] or configs


# ============================================================================
# In-kernel TREE-ROUTING forward (RLA). The matching BACKWARD lives just below (`_RoLARoutedFn`).
#
# The production tree-routing forward (the validated prototype since promoted; its backward kernels now
# live in `routed_bwd_kernels.py`): instead of taking PRECOMPUTED gates
# r,w ∈ [L,nc] and forming R = r·wᵀ, the routing gram is built IN-KERNEL from the hidden state h and the
# PER-HEAD router weights Wr,Ww ∈ [H, D, d_model, b] (b^D = nc, per head; the kernel indexes head = the
# B·H fold-row % H), never materializing the [L,nc] gates.
#
# This is a SOURCE SWAP, not a new pipeline: the inner machinery is the same shared-gram RLA forward
# structure (collapse-intra over state-blocks + NB-fused inter scan, the same
# autotune configs, the same BG state-block SMEM-tiling). The ONLY change is the block that produced the
# [BT,BG] routing tiles:  `tl.load(rg/wg)`  →  `_build_rw_tile` (per-level softmax factors gathered to
# the BG state-block via one-hot Sel maps). The reconstructed gate tiles live transiently in SRAM at the
# state-block width BG; the dominant [L,nc] gate tensor is never allocated.
#
# The factorization (proven in the prototype, validated <1e-2 vs autograd):
#   r[:, c] = Π_lvl softmax(h·Wr[lvl])[:, digit_lvl(c)],   R = r·wᵀ = ⊙_lvl (fr_lvl·fw_lvlᵀ)
# with the per-level [BT,b] softmax factors fr,fw gathered to the nc-leaf block by Sel[lvl][b, nc].
#
# BACKWARD (router-grad fold dWr,dWw,d_h) is BELOW: `_RoLARoutedFn` wraps this forward — backward drives
# the validated fold kernels (`_bwd_intra_kernel`, `_bwd_inter_*`, `_fold_*` from
# `routed_bwd_kernels.py`) at the PRODUCTION state-block width (BC=BG), and
# folds the transient [BT,nc] gate-grads into dWr/dWw/d_h in-kernel (the [L,nc] grads never materialize).
# The forward builds the SAME `sel` map the bwd needs and routes through the SAME [BT,BG] factor
# reconstruction (`_build_rw_tile`) the fold recomputes.
# ============================================================================


def _build_sel(D, b, nc, device):
    """One-hot level→leaf selection maps Sel[lvl][d, leaf] = 1 iff digit_lvl(leaf)==d (big-endian
    base-b digit decomposition: leaf = Σ_i d_i·b^(D-1-i)). Tiny [D,b,nc] constant — reconstructs
    the [BT,nc] gate tile from the [BT,b] per-level factors IN-KERNEL. Carries NO sequence dimension,
    so it is NOT the [L,nc] gate."""
    sel = torch.zeros(D, b, nc, device=device, dtype=torch.float32)
    for leaf in range(nc):
        digs = [(leaf // (b ** (D - 1 - i))) % b for i in range(D)]
        for i, d in enumerate(digs):
            sel[i, d, leaf] = 1.0
    return sel


def _routing_bias(b_r, b_w, H, D, b, device, dtype=torch.float32):
    """Resolve the optional PER-HEAD routing bias b_r/b_w ∈ [H,D,b] (the affine term of softmax(h·W+b),
    per head) to (br, bw, has_bias) for the kernels. None ⇒ a 1-element dummy (never read; HAS_BIAS=False
    gates every load AND `_bias_strides`/head-stride return 0) so the kernel signature stays uniform and
    the bias=None path is byte-identical to the pre-bias kernel."""
    if b_r is None and b_w is None:
        dummy = torch.zeros(1, device=device, dtype=dtype)
        return dummy, dummy, False
    if b_r is None or b_w is None:
        raise ValueError("routing bias: pass both b_r and b_w, or neither")
    if tuple(b_r.shape) != (H, D, b) or tuple(b_w.shape) != (H, D, b):
        raise ValueError(f"routing bias must be [H,D,b]=[{H},{D},{b}], got {tuple(b_r.shape)}/{tuple(b_w.shape)}")
    return b_r.to(dtype).contiguous(), b_w.to(dtype).contiguous(), True


def _bias_strides(br, bw, has_bias):
    """(sbr_lvl, sbr_b, sbw_lvl, sbw_b) for the kernel launch; zeros when bias is absent. PER-HEAD bias
    is [H,D,b] so the level/branch strides are dims 1,2 (dim 0 is head, handled by `_router_head_strides`)."""
    if not has_bias:
        return (0, 0, 0, 0)
    return (br.stride(1), br.stride(2), bw.stride(1), bw.stride(2))


def _router_head_strides(Wr, Ww, br, bw, has_bias):
    """(H, swr_head, sww_head, sbr_head, sbw_head) — the per-head row strides of the [H,D,d_model,b]
    routers and the [H,D,b] biases. The kernels offset Wr/Ww/bias by (pid_b % H)*stride to reach THIS
    head's slice. Bias head-strides are 0 when absent (the dummy is never indexed)."""
    H = Wr.shape[0]
    sbr_head = br.stride(0) if has_bias else 0
    sbw_head = bw.stride(0) if has_bias else 0
    return (H, Wr.stride(0), Ww.stride(0), sbr_head, sbw_head)


@triton.jit
def _build_rw_tile(h_ptr, wr_ptr, ww_ptr, sel_ptr, offs_c, cmask,
                   bn, rows, rmask, offs_bb, bmask, d_model,
                   sh_b, sh_l, sh_d, swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
                   ssel_lvl, ssel_b, ssel_c,
                   br_ptr, bw_ptr, sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                   D: tl.constexpr, BT: tl.constexpr, BB: tl.constexpr,
                   BG: tl.constexpr, BD: tl.constexpr, NDM: tl.constexpr,
                   HAS_BIAS: tl.constexpr, BUILD_R: tl.constexpr = True):
    """Build the [BT, BG] read/write routing tiles for ONE state-block (the BG-wide nc slice offs_c),
    IN-KERNEL from h + Wr,Ww — the production stand-in for `tl.load(rg/wg)`. Same in-kernel factor
    construction as the backward's `_build_factors` (routed_bwd_kernels.py), at the production
    state-block width BG. For each level: logits = h·W (+ optional bias
    b_r/b_w ∈ [D,b], added before the softmax → softmax(h·W+b)) — loop BD-blocks of d_model so any
    d_model fits SMEM, softmax over the b branches (pad cols masked to -inf → vanish), then gather the
    [BT,b] factor to the BG leaves of THIS block via the one-hot Sel slice and Hadamard-accumulate.
    r_tile,w_tile are returned masked to cmask (nc-tail cols → 0). Transient SRAM, [BT,BG].

    BUILD_R (#37): when False, SKIP the read-factor (the read logits h·Wr, the read bias, the read
    softmax, and the r-side Hadamard) and return r_tile as the dummy 1-tile — for the write-only callers
    (the snapshot pass + the state-update backward) that discard the read tile, halving the per-call
    routing-build matmul/softmax. The WRITE tile is computed identically → bit-for-bit unchanged from the
    BUILD_R=True path (the skipped output was already discarded by those callers)."""
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
            ww = tl.load(ww_ptr + lvl * sww_lvl + offs_dm[:, None] * sww_d + offs_bb[None, :] * sww_b,
                         mask=mmask[:, None] & bmask[None, :], other=0.0)
            lw += tl.dot(hc, ww)
            if BUILD_R:
                wr = tl.load(wr_ptr + lvl * swr_lvl + offs_dm[:, None] * swr_d + offs_bb[None, :] * swr_b,
                             mask=mmask[:, None] & bmask[None, :], other=0.0)
                lr += tl.dot(hc, wr)
        if HAS_BIAS:
            bwc = tl.load(bw_ptr + lvl * sbw_lvl + offs_bb * sbw_b, mask=bmask, other=0.0)
            lw += bwc[None, :]
            if BUILD_R:
                brc = tl.load(br_ptr + lvl * sbr_lvl + offs_bb * sbr_b, mask=bmask, other=0.0)
                lr += brc[None, :]
        neg = tl.full([BT, BB], float('-inf'), dtype=tl.float32)
        lw = tl.where(bmask[None, :], lw, neg)
        ew = tl.exp(lw - tl.max(lw, axis=1)[:, None])
        fw = ew / tl.sum(ew, axis=1)[:, None]   # [BT, BB] write-gate level factor
        sel = tl.load(sel_ptr + lvl * ssel_lvl + offs_bb[:, None] * ssel_b + offs_c[None, :] * ssel_c,
                      mask=bmask[:, None] & cmask[None, :], other=0.0)   # [BB, BG] one-hot
        w_tile *= tl.dot(fw, sel)
        if BUILD_R:
            lr = tl.where(bmask[None, :], lr, neg)
            er = tl.exp(lr - tl.max(lr, axis=1)[:, None])
            fr = er / tl.sum(er, axis=1)[:, None]   # [BT, BB] read-gate level factor
            r_tile *= tl.dot(fr, sel)
    w_tile = tl.where(cmask[None, :], w_tile, 0.0)
    if BUILD_R:
        r_tile = tl.where(cmask[None, :], r_tile, 0.0)
    return r_tile, w_tile


@triton.autotune(configs=_AT_CFGS, key=_SCAN_KEY, **autotune_cache_kwargs)  # _SCAN_KEY: +USE_G (RLA/GLA split)
@triton.jit
def _rola_routed_fwd_intra(q_ptr, k_ptr, v_ptr, h_ptr, wr_ptr, ww_ptr, sel_ptr, wg_ptr, outa_ptr,
                           br_ptr, bw_ptr,
                           L, dqk, dv, nc, d_model, H, swr_head, sww_head, sbr_head, sbw_head,
                           sq_b, sq_l, sq_d, sv_b, sv_l, sv_d,
                           sh_b, sh_l, sh_d, swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
                           ssel_lvl, ssel_b, ssel_c, swg_head, swg_d, soa_b, soa_l, soa_v,
                           sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                           D: tl.constexpr, bb_: tl.constexpr, BB: tl.constexpr,
                           BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                           BG: tl.constexpr, BD: tl.constexpr,
                           ND: tl.constexpr, NB: tl.constexpr, NDM: tl.constexpr,
                           HAS_BIAS: tl.constexpr, USE_G: tl.constexpr, GLA_FLOOR: tl.constexpr):
    """TREE-ROUTED intra: the shared-gram collapse-intra structure — content gram G
    built ONCE, the full routing gram R accumulated over state-blocks IN-KERNEL, A=G⊙R⊙causal, o=A·v —
    with the ONLY change being R's source: each state-block's [BT,BG] r/w tiles are BUILT from h+Wr,Ww via
    `_build_rw_tile` instead of loading precomputed gates. nc collapses; output is [B,L,dv].
    USE_G (GLA, #30 V1): the per-block gram uses the DECAYED gates rt=rgc·e^a, wt=wgc·e^-a (a = intra-chunk
    cumsum of the per-state log-decay ld over this block's c-columns). The
    nc-collapse still holds (each block's decayed [BT,BG]·[BG,BT] gram is still a [BT,BT] partial)."""
    b = tl.program_id(0)
    t = tl.program_id(1)
    # Per-head router: the BH fold is (B,H) -> head = pid_b % H. Offset Wr/Ww (and the bias) to THIS
    # head's [D,d_model,b] slice; the inner `_build_rw_tile` then reads the head-correct factors.
    _hd = b % H
    wr_ptr = wr_ptr + _hd * swr_head
    ww_ptr = ww_ptr + _hd * sww_head
    br_ptr = br_ptr + _hd * sbr_head
    bw_ptr = bw_ptr + _hd * sbw_head
    wg_ptr = wg_ptr + _hd * swg_head
    offs_t = tl.arange(0, BT)
    offs_v = tl.arange(0, BV)
    offs_bb = tl.arange(0, BB)
    bmask = offs_bb < bb_
    rows = t * BT + offs_t
    rmask = rows < L
    if USE_G:                                    # per-head scalar decay gate alpha[BT] (in-kernel ld, #45)
        alpha = _build_alpha(h_ptr, wg_ptr, b, rows, rmask, d_model, sh_b, sh_l, sh_d, swg_d, BT, BD, NDM)
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
            ldc = _ld_from_w(wgc, alpha, cmask, GLA_FLOOR)
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
def _rola_routed_fwd_inter(q_ptr, k_ptr, v_ptr, h_ptr, wr_ptr, ww_ptr, sel_ptr, wg_ptr, outa_ptr,
                           br_ptr, bw_ptr,
                           L, dqk, dv: tl.constexpr, nc, d_model,
                           H, swr_head, sww_head, sbr_head, sbw_head,
                           sq_b, sq_l, sq_d, sv_b, sv_l, sv_d,
                           sh_b, sh_l, sh_d, swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
                           ssel_lvl, ssel_b, ssel_c, swg_head, swg_d, soa_b, soa_n, soa_l, soa_v,
                           sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                           D: tl.constexpr, bb_: tl.constexpr, BB: tl.constexpr,
                           BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                           BG: tl.constexpr, BD: tl.constexpr,
                           NCH: tl.constexpr, NDM: tl.constexpr,
                           HAS_BIAS: tl.constexpr, USE_G: tl.constexpr, GLA_FLOOR: tl.constexpr):
    """TREE-ROUTED inter: the NB-fused inter scan structure — one
    (batch, state-block, FEATURE-block) carries Sd[BK, BG*BV] across chunks, o_inter atomic-accumulated
    into the shared [B,L,dv] buffer, value-OUTER over cdiv(dv,BV) — with the ONLY change being the read/
    write gate source: the [BT,BG] rgc/wgc tiles are BUILT from h+Wr,Ww via `_build_rw_tile` instead of
    loading precomputed gates. Same state scan, same SMEM bound (BK + value-tile BV), same autotune/reset_to_zero.
    USE_G (GLA, #30 V1): the carried state decays by decvec=e^Λ each chunk, the read uses rt=rgc·e^a, the
    write uses w_end=wgc·e^{Λ-a} (a = intra-chunk cumsum, Λ = chunk-total ld)."""
    b = tl.program_id(0)
    sb = tl.program_id(1)
    d0 = tl.program_id(2)
    _hd = b % H                                  # per-head router slice (BH fold is (B,H))
    wr_ptr = wr_ptr + _hd * swr_head
    ww_ptr = ww_ptr + _hd * sww_head
    br_ptr = br_ptr + _hd * sbr_head
    bw_ptr = bw_ptr + _hd * sbw_head
    wg_ptr = wg_ptr + _hd * swg_head
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
                alpha = _build_alpha(h_ptr, wg_ptr, b, rows, rmask, d_model,
                                     sh_b, sh_l, sh_d, swg_d, BT, BD, NDM)
                ldc = _ld_from_w(wgc, alpha, cmask, GLA_FLOOR)
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


def _routed_fwd_tiled(q, k, v, h, Wr, Ww, D, b, sel, chunk, BG, BK=64, b_r=None, b_w=None, Wg=None):
    """D-tiled numerator-only tree-routed forward — the in-kernel-routed shared-gram readout. Identical
    intra/inter dispatch (separate buffers, intra store / inter atomic reset_to_zero, value-tiling), with
    the routing gram source swapped to (h, Wr, Ww) via the routed kernels. Optional routing bias b_r/b_w
    ∈ [D,b] (softmax(h·W+b)). Returns [B,L,dv] at BV=next_pow2(dv); the [L,nc] gates are never allocated.
    Optional per-head decay weight Wg:[H,d_model] (GLA, #45) → USE_G decayed scan, the per-state log-decay
    ld computed IN-KERNEL from Wg + the write tile (NEVER a [L,nc] ld buffer); Wg=None is RLA (USE_G=False)."""
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
    use_g = Wg is not None
    Wg = (Wg.float().contiguous() if use_g
          else q.new_zeros(Wr.shape[0], d_model))   # USE_G=False: Wg unread (no decay); pass a [H,d_model] stub
    swg = (Wg.stride(0), Wg.stride(1))               # (swg_head, swg_d) — head/d_model strides
    H = Wr.shape[0]
    br, bw, has_bias = _routing_bias(b_r, b_w, H, D, b, q.device, dtype=q.dtype)
    sbias = _bias_strides(br, bw, has_bias)
    head = _router_head_strides(Wr, Ww, br, bw, has_bias)   # (H, swr_head, sww_head, sbr_head, sbw_head)
    out_intra = torch.zeros(B, L, BV, device=q.device, dtype=torch.float32)
    out_inter = torch.zeros(B, L, BV, device=q.device, dtype=torch.float32)
    so_a = (out_intra.stride(0), out_intra.stride(1), out_intra.stride(2))
    so_e = (out_inter.stride(0), 0, out_inter.stride(1), out_inter.stride(2))
    base = (q.stride(0), q.stride(1), q.stride(2), v.stride(0), v.stride(1), v.stride(2))
    route = (h.stride(0), h.stride(1), h.stride(2),
             Wr.stride(1), Wr.stride(2), Wr.stride(3), Ww.stride(1), Ww.stride(2), Ww.stride(3),
             sel.stride(0), sel.stride(1), sel.stride(2))
    _rola_routed_fwd_intra[(B, NCH)](q, k, v, h, Wr, Ww, sel, Wg, out_intra, br, bw, L, dqk, dv, nc, d_model,
                                     *head, *base, *route, *swg, *so_a, *sbias,
                                     D=D, bb_=b, BB=BB, BT=chunk, BK=BK, BV=BV, BG=BG, BD=BD,
                                     ND=ND, NB=NB, NDM=NDM, HAS_BIAS=has_bias, USE_G=use_g, GLA_FLOOR=_GLA_FLOOR)
    _rola_routed_fwd_inter[(B, NB, ND)](q, k, v, h, Wr, Ww, sel, Wg, out_inter, br, bw, L, dqk, dv, nc, d_model,
                                        *head, *base, *route, *swg, *so_e, *sbias,
                                        D=D, bb_=b, BB=BB, BT=chunk, BK=BK, BG=BG, BD=BD,
                                        NCH=NCH, NDM=NDM, HAS_BIAS=has_bias, USE_G=use_g, GLA_FLOOR=_GLA_FLOOR)
    return out_intra[..., :dv] + out_inter[..., :dv]


def _rola_rla_routed_fwd(q, k, v, h, Wr, Ww, D, b, chunk=None, BG=16):
    """Un-normalized TREE-ROUTED RLA readout via Triton (the optimized production forward) — PLAIN
    (no autograd). The routing gram is built IN-KERNEL from the hidden state h + per-head router weights
    Wr,Ww ∈ [H,D,d_model,b] (b^D=nc, per head); the [L,nc] gates are never materialized.

    q,k:[BH,L,K]  v:[BH,L,V]  h:[BH,L,d_model]  Wr,Ww:[H,D,d_model,b].  Returns [BH,L,V] at BV=next_pow2(V).
    Flat (D=1, b=nc) is the single-level equivalent of a precomputed-gate routed readout with r,w the
    D=1 router's explicit softmax gates."""
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
# The fold math is the validated backward (routed_bwd_kernels.py: _bwd_intra_kernel,
# _bwd_inter_state_kernel, _bwd_inter_read_kernel, _fold_kernel — validated <1e-2 vs autograd for
# flat/square/tree). Those kernels are GENERIC over the nc-block width (their `BC` constexpr); the ONLY
# adaptation is to drive them at the PRODUCTION state-block width BC=BG and through this module's
# `_build_sel`, so the backward routes through byte-identical factor
# reconstruction to the production forward (`_build_rw_tile` ≡ `_build_factors`). dr,dw stay
# transient per-chunk [B,chunk,nc] scratch tiles (gdr/gdw), OVERWRITTEN every chunk — never [L,nc].
#
# The per-chunk pre-state snapshots S_j ∈ [B,nc,dqk,dv] the reverse-scan needs are the RECURRENT state
# (NOT the gates), built by a dedicated in-kernel snapshot scan `_rola_routed_snap` (write gates built
# in-kernel via `_build_rw_tile` — gates never materialized in the snapshot pass either). The forward
# saves these snapshots; backward consumes them. The validated fold kernels are imported at module top
# (_routed_bwd_intra / _routed_bwd_inter_state / _routed_bwd_inter_read / _routed_bwd_fold), reused VERBATIM.
# ============================================================================


@triton.jit
def _rola_routed_snap_kernel(h_ptr, k_ptr, v_ptr, wr_ptr, ww_ptr, sel_ptr, wg_ptr, s_ptr, snap_ptr,
                             br_ptr, bw_ptr,
                             L, d_model, dqk, dv, nc, H, swr_head, sww_head, sbr_head, sbw_head,
                             sh_b, sh_l, sh_d, sq_b, sq_l, sq_d, sv_b, sv_l, sv_d,
                             swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
                             ssel_lvl, ssel_b, ssel_c, swg_head, swg_d, ss_b, ss_c, ss_k, ss_v,
                             snp_b, snp_n, snp_c, snp_k, snp_v,
                             sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                             D: tl.constexpr, b: tl.constexpr, BB: tl.constexpr,
                             BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BD: tl.constexpr,
                             BG: tl.constexpr, NCBLK: tl.constexpr, ND: tl.constexpr,
                             NCH: tl.constexpr, NDM: tl.constexpr,
                             HAS_BIAS: tl.constexpr, USE_G: tl.constexpr, GLA_FLOOR: tl.constexpr):
    """Per-chunk PRE-STATE snapshot scan for the routed backward. One program per batch carries the flat
    Kronecker state S[nc,dqk,dv] across chunks; BEFORE each chunk's write it copies S into snap[:,chunk]
    (the state the reverse-scan reads). The write gates are built IN-KERNEL via `_build_rw_tile` over BG
    state-blocks (the [L,nc] gates are never materialized here either). Mirrors the proto chunk kernel's
    state update (S += Σ_t wₜᶜ kₜ⊗vₜ), at the production state-block width BG.
    USE_G (GLA, #30 V1): the per-c state decays by e^{Λ_c} each chunk and the write is e^{Λ_c-a}-weighted
    (S[c] ← e^{Λ_c}S[c] + Σ w_end k⊗v) — exactly the decayed inter scan; the snapshot is still PRE-update."""
    pid_b = tl.program_id(0)
    _hd = pid_b % H                              # per-head router slice (BH fold is (B,H))
    wr_ptr = wr_ptr + _hd * swr_head
    ww_ptr = ww_ptr + _hd * sww_head
    br_ptr = br_ptr + _hd * sbr_head
    bw_ptr = bw_ptr + _hd * sbw_head
    wg_ptr = wg_ptr + _hd * swg_head
    ND_V = (dv + BV - 1) // BV       # value-blocks (BV is the value TILE; ND_V==1 ⇒ the un-tiled kernel)
    offs_t = tl.arange(0, BT)
    offs_bb = tl.arange(0, BB)
    offs_c = tl.arange(0, BG)
    bmask = offs_bb < b
    for ci in range(NCH):
        t_start = ci * BT
        rows = t_start + offs_t
        rmask = rows < L
        if USE_G:                                # per-head decay gate alpha[BT], once per token-block (#45)
            alpha = _build_alpha(h_ptr, wg_ptr, pid_b, rows, rmask, d_model,
                                 sh_b, sh_l, sh_d, swg_d, BT, BD, NDM)
        for cb in range(NCBLK):
            cols = cb * BG + offs_c
            cmask = cols < nc
            _, w_tile = _build_rw_tile(h_ptr, wr_ptr, ww_ptr, sel_ptr, cols, cmask,
                                       pid_b, rows, rmask, offs_bb, bmask, d_model,
                                       sh_b, sh_l, sh_d, swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
                                       ssel_lvl, ssel_b, ssel_c,
                                       br_ptr, bw_ptr, sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                                       D, BT, BB, BG, BD, NDM, HAS_BIAS, BUILD_R=False)  # write-only: skip read
            if USE_G:
                ldc = _ld_from_w(w_tile, alpha, cmask, GLA_FLOOR)
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


def _routed_snapshots(q, k, v, h, Wr, Ww, D, b, sel, chunk, BG, br=None, bw=None, has_bias=False, Wg=None):
    """Build per-chunk pre-state snapshots [B, NCH, nc, dqk, dv] for the routed backward — the recurrent
    STATE (NOT the gates), via the in-kernel snapshot scan. Write gates built in-kernel (never [L,nc]).
    The write gate honors the optional routing bias (softmax(h·Ww+b_w)) so snapshots match the fwd.
    Optional per-head decay weight Wg:[H,d_model] (GLA, #45) → USE_G decayed state recurrence, the
    per-state log-decay computed IN-KERNEL (never a [L,nc] ld)."""
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
    H = Wr.shape[0]
    if br is None:
        br, bw, has_bias = _routing_bias(None, None, H, D, b, q.device, dtype=q.dtype)
    sbias = _bias_strides(br, bw, has_bias)
    head = _router_head_strides(Wr, Ww, br, bw, has_bias)
    use_g = Wg is not None
    Wg = (Wg.float().contiguous() if use_g else q.new_zeros(H, d_model))
    swg = (Wg.stride(0), Wg.stride(1))
    S = torch.zeros(B, nc, dqk, dv, device=q.device, dtype=torch.float32)
    snap = torch.zeros(B, NCH, nc, dqk, dv, device=q.device, dtype=torch.float32)
    _rola_routed_snap_kernel[(B,)](
        h, k, v, Wr, Ww, sel, Wg, S, snap, br, bw,
        L, d_model, dqk, dv, nc, *head,
        h.stride(0), h.stride(1), h.stride(2), q.stride(0), q.stride(1), q.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        Wr.stride(1), Wr.stride(2), Wr.stride(3), Ww.stride(1), Ww.stride(2), Ww.stride(3),
        sel.stride(0), sel.stride(1), sel.stride(2), *swg,
        S.stride(0), S.stride(1), S.stride(2), S.stride(3),
        snap.stride(0), snap.stride(1), snap.stride(2), snap.stride(3), snap.stride(4),
        *sbias,
        D=D, b=b, BB=BB, BT=chunk, BK=BK, BV=BV, BD=BD, BG=BG, NCBLK=NCBLK, ND=ND, NCH=NCH, NDM=NDM,
        HAS_BIAS=has_bias, USE_G=use_g, GLA_FLOOR=_GLA_FLOOR, num_warps=4, num_stages=1)
    return snap


def _ld_chunk(h_c, Wr, Ww, Wg, D, b, H, b_r=None, b_w=None):
    """Per-state log-decay ld[BH,len,nc] for a CHUNK slice of h (chunk-local, NEVER full [L,nc]) — the SAME
    in-kernel formula (`RoLA._log_decay`), in torch, for the driver's dS chunk-total Λ_c decay (#45). h_c is
    [BH,len,d_model] (BH=(B,H) fold). alpha=sigmoid(h·Wg[head]) per-head; ld=clamp(log(clamp(1-w(1-alpha),
    1e-8)), _GLA_FLOOR), w the per-head WRITE gate. Returns fp32 [BH,len,nc]."""
    BH, T, dm = h_c.shape
    _, wf = _tree_gates_torch(h_c.float(), Wr.float(), Ww.float(), D, b, H, b_r=b_r, b_w=b_w)  # [BH,len,nc]
    hr = h_c.float().view(BH // H, H, T, dm)
    alpha = torch.sigmoid(torch.einsum('bhtd,hd->bht', hr, Wg.float())).reshape(BH, T, 1)
    ld = (1.0 - wf * (1.0 - alpha)).clamp(min=1e-8).log()
    return ld.clamp(min=_GLA_FLOOR)


def _rola_rla_routed_bwd(q, k, v, h, Wr, Ww, do, D, b, chunk, BG, b_r=None, b_w=None, Wg=None):
    """Tree-routed RLA backward at the PRODUCTION state-block width BC=BG. Drives the validated proto
    fold kernels: one intra launch (dq,dk,dv-intra + dr,dw-intra fold) and a sequential reverse state-
    adjoint scan over chunks (state-update bwd → readout bwd → router-grad fold). The gate-grads dr,dw
    live only as transient [B,chunk,nc] scratch (gdr/gdw), OVERWRITTEN each chunk — never [L,nc]. With an
    optional routing bias b_r/b_w ∈ [D,b], also folds db_r/db_w (transient, never [L,nc]). Returns
    dq,dk,dv,d_h,dWr,dWw (and db_r,db_w when biased) — all fp32.

    USE_G (GLA, #45): optional per-head decay weight Wg:[H,d_model] → the decayed routed backward — the
    per-state log-decay ld is computed IN-KERNEL (Wg + the write gate, never a [L,nc] ld), the kernels use
    the DECAYED gates (rt=r·eᵃ, w_end=w·e^{Λ−a}), the running dS adjoint is decayed by e^Λ between the
    state-bwd and read-bwd halves, and a persistent gda[B,L,nc] buffer collects the per-token log-decay
    adjoints which the driver reverse-cumsums (intra-chunk) into a PER-CHUNK dld[B,chunk,nc] (never [L,nc]);
    the fold kernel splits dld → dWg + the extra dh + the decay's write-gate grad (dWw,dh). Returns
    dq,dk,dv,d_h,dWr,dWw,dWg (and db_r,db_w when biased). Wg=None is byte-identical to RLA (USE_G=False)."""
    use_g = Wg is not None
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
    H = Wr.shape[0]
    br, bw, has_bias = _routing_bias(b_r, b_w, H, D, b, q.device, dtype=torch.float32)
    sbias = _bias_strides(br, bw, has_bias)
    head = _router_head_strides(Wr, Ww, br, bw, has_bias)   # (H, swr_head, sww_head, sbr_head, sbw_head)
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
    # Wg:[H,d_model] (GLA) — the per-head decay weight; the kernels compute ld IN-KERNEL (never a [L,nc] ld).
    # RLA passes a [H,d_model] stub the kernels skip (USE_G=False).
    Wg = (Wg.float().contiguous() if use_g else q.new_zeros(H, d_model))
    swg = (Wg.stride(0), Wg.stride(1))
    # per-chunk pre-state snapshots (the recurrent STATE, not gates) for the reverse-scan — recomputed
    # here at the SAME fp32 router precision as the fold, so fwd/bwd routing is bit-consistent.
    snap = _routed_snapshots(q, k, v, h, Wr, Ww, D, b, sel, chunk, BG, br, bw, has_bias,
                             Wg=(Wg if use_g else None))
    # dvv/dq/dk are written by the fold kernels at v's / q's row strides (sv_l=v.stride(1)=dv,
    # sq_l=q.stride(1)=dqk), so they MUST be allocated at the TRUE dv/dqk width (NOT padded BV/BK) or
    # the row layout corrupts for non-pow2 dv/dqk (e.g. dv=24→BV=32). The in-kernel [BT,BV]/[BT,BK]
    # accumulators store only the masked :dv/:dqk lanes (vmask/kmask). Mirrors `_kappa_routed_bwd`.
    dq = torch.zeros(B, L, dqk, device=q.device, dtype=torch.float32)
    dk = torch.zeros(B, L, dqk, device=q.device, dtype=torch.float32)
    dvv = torch.zeros(B, L, dv, device=q.device, dtype=torch.float32)
    dh = torch.zeros(B, L, d_model, device=q.device, dtype=torch.float32)
    dWr = torch.zeros(H, D, d_model, b, device=q.device, dtype=torch.float32)   # per-head [H,D,d_model,b]
    dWw = torch.zeros(H, D, d_model, b, device=q.device, dtype=torch.float32)
    dWg = torch.zeros(H, d_model, device=q.device, dtype=torch.float32)   # per-head decay-weight grad (#45)
    dbr = torch.zeros(H, D, b, device=q.device, dtype=torch.float32)   # per-head routing-bias grads [H,D,b]
    dbw = torch.zeros(H, D, b, device=q.device, dtype=torch.float32)
    # gda[B,L,nc]: persistent per-token log-decay adjoint accumulator (USE_G) — the intra kernel writes
    # its da-pieces over all chunks (parallel), the inter kernels add theirs per chunk; reverse-cumsummed
    # per chunk into dld at the end. RLA leaves it zero (a 1-col stub) and dld is unused.
    gda = torch.zeros(B, L, nc, device=q.device, dtype=torch.float32) if use_g \
        else q.new_zeros(B, 1, 1)
    sga = (gda.stride(0), gda.stride(1), gda.stride(2))
    common = dict(D=D, b=b, BB=BB, BT=chunk, BK=BK_full, BV=BVO, BD=BD, BC=BC,
                  NCBLK=NCBLK, ND=triton.cdiv(dqk, BK_full), NDM=NDM, HAS_BIAS=has_bias,
                  USE_G=use_g, GLA_FLOOR=_GLA_FLOOR, num_warps=4, num_stages=1)
    _routed_bwd_intra[(B, NCH)](
        h, q, k, v, Wr, Ww, sel, Wg, do, dq, dk, dvv, dh, dWr, dWw, gda,
        br, bw, dbr, dbw,
        L, d_model, dqk, dv, nc, *head,
        h.stride(0), h.stride(1), h.stride(2), q.stride(0), q.stride(1), q.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        Wr.stride(1), Wr.stride(2), Wr.stride(3), Ww.stride(1), Ww.stride(2), Ww.stride(3),
        sel.stride(0), sel.stride(1), sel.stride(2), *swg, *sga,
        do.stride(0), do.stride(1), do.stride(2), dh.stride(0), dh.stride(1), dh.stride(2),
        *sbias,
        **common)
    dS = torch.zeros(B, nc, dqk, dv, device=q.device, dtype=torch.float32)
    gdr = torch.zeros(B, chunk, nc, device=q.device, dtype=torch.float32)   # transient, OVERWRITTEN/chunk
    gdw = torch.zeros(B, chunk, nc, device=q.device, dtype=torch.float32)
    # dld is assembled PER CHUNK from gda (chunk-local reverse-cumsum) and consumed by the fold — never [L,nc].
    dld_chunk = torch.zeros(B, chunk, nc, device=q.device, dtype=torch.float32) if use_g \
        else q.new_zeros(B, 1, 1)
    fold_common = dict(D=D, b=b, BB=BB, BT=chunk, BC=BC, BD=BD, NCBLK=NCBLK, NDM=NDM,
                       HAS_BIAS=has_bias, USE_G=use_g, GLA_FLOOR=_GLA_FLOOR, num_warps=4, num_stages=1)
    inter_common = dict(D=D, b=b, BB=BB, BT=chunk, BK=BK, BV=BV, BD=BD, BC=BC,
                        NCBLK=NCBLK, ND=ND, NDM=NDM, HAS_BIAS=has_bias, USE_G=use_g,
                        GLA_FLOOR=_GLA_FLOOR, num_warps=4, num_stages=1)
    for c in reversed(range(NCH)):
        Sj = snap[:, c].contiguous()
        # state-bwd reads dS = adjoint S_{j+1} (pre-decvec) → dk,dv,gdw + (USE_G) the carry/w_end da-pieces.
        _routed_bwd_inter_state[(B,)](
            h, k, v, Wr, Ww, sel, Wg, Sj, dS, dk, dvv, gdw, gda, br, bw,
            L, d_model, dqk, dv, nc, c * chunk, *head,
            h.stride(0), h.stride(1), h.stride(2), q.stride(0), q.stride(1), q.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            Wr.stride(1), Wr.stride(2), Wr.stride(3), Ww.stride(1), Ww.stride(2), Ww.stride(3),
            sel.stride(0), sel.stride(1), sel.stride(2), *swg,
            dS.stride(0), dS.stride(1), dS.stride(2), dS.stride(3),
            gdr.stride(0), gdr.stride(1), gdr.stride(2), *sga, *sbias,
            **inter_common)
        if use_g:
            # decay the running dS adjoint by decvec=e^{Λ_c} (per state, broadcast over dqk×dv) — the
            # reverse of the forward's state carry S_{j+1}=e^Λ S_j + ΔS. Must run AFTER state-bwd reads
            # the S_{j+1} adjoint (and its ZdZ) and BEFORE read-bwd folds dS_read → adjoint S_j. The
            # chunk-total Λ_c is the in-kernel ld's chunk-total; recompute it here from Wg + the write
            # gates (chunk-local [B,len,nc], never [L,nc]). The write gates MUST honor the routing bias
            # (softmax(h·Ww+b_w)) to match the in-kernel ld (`_build_rw_tile` runs HAS_BIAS=True) — else the
            # dS-adjoint decay uses un-biased gates while the kernels use biased ones (the #45 bias bug).
            r0, r1 = c * chunk, min(c * chunk + chunk, L)
            ld_c = _ld_chunk(h[:, r0:r1], Wr, Ww, Wg, D, b, H,
                             b_r=br if has_bias else None, b_w=bw if has_bias else None)  # [B,len,nc]
            Lam_c = ld_c.sum(dim=1)                                 # [B,nc] chunk-total per state
            dS = dS * torch.exp(Lam_c)[:, :, None, None]
        _routed_bwd_inter_read[(B,)](
            h, q, Wr, Ww, sel, Wg, Sj, dS, do, dq, gdr, gda, br, bw,
            L, d_model, dqk, dv, nc, c * chunk, *head,
            h.stride(0), h.stride(1), h.stride(2), q.stride(0), q.stride(1), q.stride(2),
            Wr.stride(1), Wr.stride(2), Wr.stride(3), Ww.stride(1), Ww.stride(2), Ww.stride(3),
            sel.stride(0), sel.stride(1), sel.stride(2), *swg,
            Sj.stride(0), Sj.stride(1), Sj.stride(2), Sj.stride(3),
            do.stride(0), do.stride(1), do.stride(2),
            gdr.stride(0), gdr.stride(1), gdr.stride(2), *sga, *sbias,
            **inter_common)
        if use_g:
            # assemble dld for THIS chunk = intra-chunk reverse-cumsum of gda (a_t resets each chunk in the
            # fwd, so it's intra-only). Chunk-local [B,chunk,nc]; consumed by the fold below, never [L,nc].
            r0, r1 = c * chunk, min(c * chunk + chunk, L)
            g_sl = gda[:, r0:r1]
            tot = g_sl.sum(dim=1, keepdim=True)
            dld_chunk.zero_()
            dld_chunk[:, :r1 - r0] = tot - g_sl.cumsum(dim=1) + g_sl
        _routed_bwd_fold[(B,)](
            h, Wr, Ww, sel, gdr, gdw, dh, dWr, dWw, br, bw, dbr, dbw,
            Wg, dWg, dld_chunk,
            L, d_model, nc, c * chunk, *head,
            h.stride(0), h.stride(1), h.stride(2),
            Wr.stride(1), Wr.stride(2), Wr.stride(3), Ww.stride(1), Ww.stride(2), Ww.stride(3),
            sel.stride(0), sel.stride(1), sel.stride(2),
            gdr.stride(0), gdr.stride(1), gdr.stride(2), dh.stride(0), dh.stride(1), dh.stride(2),
            *swg,
            *sbias,
            **fold_common)
    if has_bias:
        if use_g:
            return dq[..., :dqk], dk[..., :dqk], dvv[..., :dv], dh, dWr, dWw, dWg, dbr, dbw
        return dq[..., :dqk], dk[..., :dqk], dvv[..., :dv], dh, dWr, dWw, dbr, dbw
    if use_g:
        return dq[..., :dqk], dk[..., :dqk], dvv[..., :dv], dh, dWr, dWw, dWg
    return dq[..., :dqk], dk[..., :dqk], dvv[..., :dv], dh, dWr, dWw


class _RoLARoutedFn(torch.autograd.Function):
    """End-to-end differentiable in-kernel TREE-ROUTED RLA readout. Forward runs the OPTIMIZED production
    routed forward (`_routed_fwd_tiled`); backward drives the validated fold kernels at the production
    state-block width (BC=BG), reconstructing factors through the SAME Sel map. The [L,nc] gates AND
    their grads are never materialized (only transient [BT,BG] factor tiles + per-chunk [B,chunk,nc]
    gate-grad scratch)."""
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, h, Wr, Ww, D, b, chunk, BG, b_r, b_w, Wg=None):
        # GLA (USE_G) caps the FORWARD chunk at _CHUNK (32) — the fp32 decay floor + SMEM wall (the BT=64
        # decayed-gram fp32 tiles overflow the ada-class 99KB SMEM); RLA keeps the full _CHUNK_FWD (64).
        cap = _CHUNK if Wg is not None else _CHUNK_FWD
        chunk = cap if chunk is None else min(chunk, cap)
        nc = b ** D
        sel = _build_sel(D, b, nc, q.device)
        q, k, v, h, Wr, Ww = (x.contiguous() for x in (q, k, v, h, Wr, Ww))
        Wgc = Wg.contiguous() if Wg is not None else None
        o = _routed_fwd_tiled(q, k, v, h, Wr, Ww, D, b, sel, chunk=chunk, BG=BG, b_r=b_r, b_w=b_w, Wg=Wgc)
        # save the inputs (NOT gates, NOT [L,nc] ld/grads); the per-chunk pre-state snapshots the reverse-
        # scan needs are recomputed in backward (fp32-router parity). #45: the GLA decay's saved activation
        # is the [H,d_model] Wg (NOT a [L,nc] ld) — the saved-activation win.
        ctx.save_for_backward(q, k, v, h, Wr, Ww, b_r, b_w, Wgc)
        ctx.D, ctx.b, ctx.chunk, ctx.BG = D, b, chunk, BG
        return o.to(q.dtype)

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do):
        q, k, v, h, Wr, Ww, b_r, b_w, Wg = ctx.saved_tensors
        grads = _rola_rla_routed_bwd(
            q, k, v, h, Wr, Ww, do.contiguous(), ctx.D, ctx.b, ctx.chunk, ctx.BG,
            b_r=b_r, b_w=b_w, Wg=Wg)
        use_g = Wg is not None
        if b_r is None:
            dq, dk, dv, dh, dWr, dWw = grads[:6]
            dbr, dbw = None, None
            dWg = grads[6] if use_g else None
        else:
            dq, dk, dv, dh, dWr, dWw = grads[:6]
            dWg = grads[6] if use_g else None
            dbr, dbw = grads[-2], grads[-1]
        # forward arg order: q, k, v, h, Wr, Ww, D, b, chunk, BG, b_r, b_w, Wg
        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dh.to(h.dtype),
                dWr.to(Wr.dtype), dWw.to(Ww.dtype), None, None, None, None,
                None if dbr is None else dbr.to(b_r.dtype),
                None if dbw is None else dbw.to(b_w.dtype),
                None if dWg is None else dWg.to(Wg.dtype))


@input_guard
def rola_rla_routed_triton(q, k, v, h, Wr, Ww, D, b, chunk=None, BG=16, b_r=None, b_w=None):
    """Un-normalized TREE-ROUTED RLA readout via Triton — the in-kernel-routing shared-gram readout,
    DIFFERENTIABLE end-to-end (fused router-grad fold; the [L,nc] gates and their
    grads are NEVER materialized). The routing gram is built IN-KERNEL from the hidden state h + per-level
    router weights Wr,Ww ∈ [H,D,d_model,b] (b^D=nc, per head), with an OPTIONAL per-head bias b_r/b_w ∈ [H,D,b]
    (softmax(h·W+b)).

    q,k:[BH,L,K]  v:[BH,L,V]  h:[BH,L,d_model]  Wr,Ww:[H,D,d_model,b].  Returns [BH,L,V] at BV=next_pow2(V).
    Flat (D=1, b=nc) is the single-level equivalent of a precomputed-gate routed readout with r,w the
    D=1 router's explicit softmax gates. Grads dq,dk,dv,d_h,dWr,dWw (+db_r,db_w) match autograd to rel<1e-2 (bf16)."""
    return _RoLARoutedFn.apply(q, k, v, h, Wr, Ww, D, b, chunk, BG, b_r, b_w, None)


@input_guard
def rola_gla_routed_triton(q, k, v, h, Wr, Ww, Wg, D, b, chunk=None, BG=16, b_r=None, b_w=None):
    """Un-normalized TREE-ROUTED GLA readout via Triton — `rola_rla_routed_triton` + a per-head decay
    WEIGHT Wg:[H,d_model] (GLA), DIFFERENTIABLE end-to-end ([L,nc] gates, the per-state log-decay ld, AND
    their grads NEVER materialized — ld is computed IN-KERNEL from Wg + the write gate, #45). The routing
    gram uses the DECAYED gates (rt=r·eᵃ, w_end=w·e^{Λ−a}); Wg=None is the RLA path.
    Grads dq,dk,dv,d_h,dWr,dWw,dWg: q/k/v TIGHT (<8e-3), the gate/decay grads to the GLA fp32 floor."""
    return _RoLARoutedFn.apply(q, k, v, h, Wr, Ww, D, b, chunk, BG, b_r, b_w, Wg)


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
def _kappa_d_tile(q_ptr, sden_ptr, Gc, w_tile, ea, ena, cols, cmask,
                  pid_b, rows, rmask, dqk, sq_b, sq_l, sq_d, ssd_b, ssd_k,
                  BT: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr, ND: tl.constexpr,
                  USE_G: tl.constexpr):
    """Per-state den d[BT,BC] = (G⊙causal)·w (intra) + q·Sden^c (inter). The inter term contracts dqk →
    accumulate over BK-feature-blocks so the [BC,BK] Sden slice stays bounded by BK<=64. Gc = G⊙causal is
    passed in (value-free, reused). BV-free; recomputed identically in each numerator value-block pass.
    USE_G (GLA): the DECAYED den d_i^c = e^{a_ic}·(Σ_{j≤i}(qi·kj) w_j^c e^{-a_jc} + qi·Sden_carry^c) —
    EXACTLY `naive_rola_gla_perstate_den` (intra w decayed by e^{-a}, the carried Sden is the pre-decay
    chunk-start state, the whole sum scaled by e^a). ea/ena = e^{a},e^{-a} [BT,BC]; raw w_tile decayed here."""
    wt = (w_tile * ena) if USE_G else w_tile
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
    d_intra = tl.dot(Gc.to(wt.dtype), wt)
    d = (d_intra + d_inter) * ea if USE_G else (d_intra + d_inter)
    return tl.where(cmask[None, :], d, 0.0)


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
def _kappa_fwd_chunk(h_ptr, q_ptr, k_ptr, v_ptr, wr_ptr, ww_ptr, sel_ptr, kap_ptr, wg_ptr,
                     sval_ptr, sden_ptr, num_ptr, den_ptr, br_ptr, bw_ptr,
                     L, d_model, dqk, dv, nc, t_start,
                     H, swr_head, sww_head, sbr_head, sbw_head,
                     sh_b, sh_l, sh_d, sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sk_b, sk_l,
                     swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b, ssel_lvl, ssel_b, ssel_c,
                     swg_head, swg_d,
                     ssv_b, ssv_c, ssv_k, ssv_v, ssd_b, ssd_c, ssd_k,
                     snm_b, snm_l, snm_v, sdn_b, sdn_l, sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                     GLOBAL: tl.constexpr, PER_STATE: tl.constexpr, EPS: tl.constexpr,
                     D: tl.constexpr, b: tl.constexpr, BB: tl.constexpr,
                     BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BD: tl.constexpr,
                     BC: tl.constexpr, NCBLK: tl.constexpr, ND: tl.constexpr, NDM: tl.constexpr,
                     NDV: tl.constexpr, NDVP: tl.constexpr,
                     HAS_BIAS: tl.constexpr, USE_G: tl.constexpr, GLA_FLOOR: tl.constexpr):
    """Fused global/kappa/per_state chunk: builds r,w,d,r_tilde transiently per nc-block, accumulates num +
    den, carries Sval[k,v] AND Sden[k] across chunks. One program per batch. Mirrors the proto chunk
    kernel + the den pre-pass, with the read gate rescaled by the transient per-state den.
    USE_G (GLA): per-state log-decay ld → the DECAYED scan (mirrors `_rola_routed_fwd_inter` USE_G +
    `naive_rola_gla_perstate_den`): per nc-block a=cumsum(ld), Λ=chunk-total; the readout rt=r̃·e^a, the
    intra gram wt=w·e^{-a}, the state writes w_end=w·e^{Λ-a}, the Sval AND Sden carries decay by e^Λ, and
    the per-state den d carries decay via `_kappa_d_tile`. The kap rescale wraps the UNDECAYED r (its d is
    already decayed); the den reduction o_den=Σ_c r̃^c d^c uses that UNDECAYED r̃ (e^a is readout-only)."""
    pid_b = tl.program_id(0)
    ND_V: tl.constexpr = NDV         # value-blocks (BV is the value TILE; ND_V==1 ⇒ the un-tiled kernel) —
    #                                  a constexpr (F2a) so the cb-outer value accumulators can be unrolled.
    offs_t = tl.arange(0, BT)
    offs_bb = tl.arange(0, BB)
    offs_c = tl.arange(0, BC)
    bmask = offs_bb < b
    rows = t_start + offs_t
    rmask = rows < L
    _hd = pid_b % H                              # per-head router slice (BH fold is (B,H))
    wr_ptr = wr_ptr + _hd * swr_head
    ww_ptr = ww_ptr + _hd * sww_head
    br_ptr = br_ptr + _hd * sbr_head
    bw_ptr = bw_ptr + _hd * sbw_head
    wg_ptr = wg_ptr + _hd * swg_head
    if USE_G:                                    # per-head decay gate alpha[BT], once per chunk (#45)
        alpha = _build_alpha(h_ptr, wg_ptr, pid_b, rows, rmask, d_model,
                             sh_b, sh_l, sh_d, swg_d, BT, BD, NDM)
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
        if USE_G:
            ldc = _ld_from_w(w_tile, alpha, cmask, GLA_FLOOR)
            a = tl.cumsum(ldc, axis=0)
            ea = tl.exp(a)
            ena = tl.exp(-a)
        else:
            ea = w_tile * 0.0 + 1.0
            ena = ea
        d_tile = _kappa_d_tile(q_ptr, sden_ptr, Gc, w_tile, ea, ena, cols, cmask,
                               pid_b, rows, rmask, dqk, sq_b, sq_l, sq_d, ssd_b, ssd_k,
                               BT, BK, BC, ND, USE_G)
        rt_tile = _kappa_rescale(r_tile, d_tile, kap, cmask, GLOBAL, PER_STATE, EPS)
        o_den += tl.sum(rt_tile * d_tile, axis=1)
    # ---- Pass 2 (value-tiled): numerator num = intra (A·v) + inter (Σ_c r̃^c q·Sval^c), AND the Sval
    # state write — value-tiled over ND_V blocks so the [BC*BK,BV] state slice stays bounded by BK·BV.
    # F2a: state-block OUTER / value-block INNER (mirrors the backward's cb-outer/vb-inner structure). The
    # router build + d/rescale/A — ALL value-FREE — are computed ONCE per state-block (cb) and reused across
    # the value-blocks, instead of being rebuilt ×ND_V inside the vb loop. Each value-block keeps its OWN
    # running num accumulator: a SINGLE [BT, ND_V, BV] tile (value-block on the middle axis) persisted across
    # the unrolled cb loop. Each intra/inter term for value-block vb is added to slice vb via a masked
    # `tl.where`, in the SAME per-token order as the old vb-outer fold (per vb: cb0-intra, cb0-inter…,
    # cb1-intra, … — independent of vb's loop position) → byte-identical output. A fixed-shape 3D tile is
    # used because Triton's jit rejects Python list/tuple/append containers of per-vb accumulators.
    o_num = tl.zeros([BT, NDVP, BV], dtype=tl.float32)   # NDVP=next_pow2(ND_V) — Triton block dims pow2;
    vbidx = tl.arange(0, NDVP)                            # the pad slices [ND_V,NDVP) stay 0, never stored.
    for cb in range(NCBLK):
        cols = cb * BC + offs_c
        cmask = cols < nc
        r_tile, w_tile = _build_rw_tile(h_ptr, wr_ptr, ww_ptr, sel_ptr, cols, cmask,
                                        pid_b, rows, rmask, offs_bb, bmask, d_model,
                                        sh_b, sh_l, sh_d, swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
                                        ssel_lvl, ssel_b, ssel_c,
                                        br_ptr, bw_ptr, sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                                        D, BT, BB, BC, BD, NDM, HAS_BIAS)
        if USE_G:
            ldc = _ld_from_w(w_tile, alpha, cmask, GLA_FLOOR)
            a = tl.cumsum(ldc, axis=0)
            Lam = tl.sum(tl.where(offs_t[:, None] == (BT - 1), a, 0.0), axis=0)   # [BC] chunk-total
            ea = tl.exp(a)
            ena = tl.exp(-a)
            wt = w_tile * ena                          # intra gram write w·e^{-a}
            w_end = w_tile * tl.exp(Lam[None, :] - a)   # state write w·e^{Λ-a}
            dec_c = tl.exp(Lam)                         # [BC] per-c carry e^Λ
        else:
            ea = w_tile * 0.0 + 1.0
            ena = ea
            wt = w_tile
            w_end = w_tile
        d_tile = _kappa_d_tile(q_ptr, sden_ptr, Gc, w_tile, ea, ena, cols, cmask,
                               pid_b, rows, rmask, dqk, sq_b, sq_l, sq_d, ssd_b, ssd_k,
                               BT, BK, BC, ND, USE_G)
        rt_tile = _kappa_rescale(r_tile, d_tile, kap, cmask, GLOBAL, PER_STATE, EPS)
        rd_tile = (rt_tile * ea) if USE_G else rt_tile   # decayed read gate r̃·e^a (readout only)
        # numerator intra: A = G⊙(r̃·wᵀ)⊙causal; o_num += A·v. (GLA: decayed rd·wtᵀ.) Value-free → once/cb.
        Rg = tl.dot(rd_tile, tl.trans(wt))
        A = G * Rg * causal
        for vb in range(ND_V):
            offs_v = vb * BV + tl.arange(0, BV)
            vmask = offs_v < dv
            sel_vb = (vbidx[None, :, None] == vb)        # [1,ND_V,1] one-hot select of slice vb
            vc = tl.load(v_ptr + pid_b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                         mask=rmask[:, None] & vmask[None, :], other=0.0)
            # intra: A·v added to o_num's value-block vb (each term added in the old fold order → bit-exact).
            o_num += tl.where(sel_vb, tl.dot(A.to(vc.dtype), vc)[:, None, :], 0.0)
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
                rq = tl.reshape(rd_tile[:, :, None] * qc[:, None, :], [BT, BC * BK])
                o_num += tl.where(sel_vb, tl.dot(rq.to(sflat.dtype), sflat)[:, None, :], 0.0)
                wk = tl.reshape(w_end[:, :, None] * kc[:, None, :], [BT, BC * BK])
                if USE_G:
                    # per-(c,k) carry: broadcast dec_c[BC] over BK feature rows of each c → [BC*BK].
                    deckv = tl.reshape(dec_c[:, None] * tl.full([BC, BK], 1.0, tl.float32), [BC * BK])
                    snew = deckv[:, None] * sflat + tl.dot(tl.trans(wk).to(vc.dtype), vc)
                else:
                    snew = sflat + tl.dot(tl.trans(wk).to(vc.dtype), vc)
                tl.store(sval_ptr + pid_b*ssv_b + ck[:, None]*ssv_k + offs_v[None, :]*ssv_v,
                         snew, mask=ckmask[:, None] & vmask[None, :])
    # store the [BT, NDVP, BV] accumulator as the [BT, NDVP*BV] num row (column vb*BV+j ≡ slice [vb,j];
    # NDVP*BV == BVO, the num buffer width). The pad columns (≥ ND_V*BV ≥ dv) are masked off by vfmask.
    offs_vf = tl.arange(0, NDVP * BV)
    vfmask = offs_vf < dv
    tl.store(num_ptr + pid_b*snm_b + rows[:, None]*snm_l + offs_vf[None, :]*snm_v,
             tl.reshape(o_num, [BT, NDVP * BV]), mask=rmask[:, None] & vfmask[None, :])
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
                                    D, BT, BB, BC, BD, NDM, HAS_BIAS, BUILD_R=False)  # write-only: skip read
        if USE_G:
            ldc = _ld_from_w(w_tile, alpha, cmask, GLA_FLOOR)
            a = tl.cumsum(ldc, axis=0)
            Lam = tl.sum(tl.where(offs_t[:, None] == (BT - 1), a, 0.0), axis=0)   # [BC]
            wsd = w_tile * tl.exp(Lam[None, :] - a)        # w_end for the den state write
            dec_c = tl.exp(Lam)                            # [BC] per-c carry
        else:
            wsd = w_tile
        for d0 in range(ND):
            offs_k = d0 * BK + tl.arange(0, BK)
            kmask = offs_k < dqk
            kc = tl.load(k_ptr + pid_b*sq_b + rows[:, None]*sq_l + offs_k[None, :]*sq_d,
                         mask=rmask[:, None] & kmask[None, :], other=0.0)
            ck = tl.reshape(cols[:, None] * dqk + offs_k[None, :], [BC * BK])
            ckmask = tl.reshape(cmask[:, None] & kmask[None, :], [BC * BK])
            wk = tl.reshape(wsd[:, :, None] * kc[:, None, :], [BT, BC * BK])
            dsden = tl.sum(wk, axis=0)                                                  # [BC*BK]
            sden = tl.load(sden_ptr + pid_b*ssd_b + ck*ssd_k, mask=ckmask, other=0.0)   # [BC*BK]
            if USE_G:
                deckv = tl.reshape(dec_c[:, None] * tl.full([BC, BK], 1.0, tl.float32), [BC * BK])
                tl.store(sden_ptr + pid_b*ssd_b + ck*ssd_k, deckv * sden + dsden, mask=ckmask)
            else:
                tl.store(sden_ptr + pid_b*ssd_b + ck*ssd_k, sden + dsden, mask=ckmask)
    tl.store(den_ptr + pid_b*sdn_b + rows*sdn_l, o_den, mask=rmask)


def _kappa_routed_fwd(q, k, v, h, Wr, Ww, kap, D, b, sel, chunk, global_norm, per_state, eps,
                      b_r=None, b_w=None, Wg=None, need_snapshots=False):
    """Fused global/kappa/per_state tree-routed forward. Returns (num[B,L,BV], den[B,L], snap_val, snap_den).
    The per-chunk pre-state snapshots are ONLY built when need_snapshots=True (the BACKWARD's recompute):
    the forward pass discards them (it saves the inputs, not the snapshots — the backward regenerates its
    own), and inference has no backward at all. Building them on the forward/prefill path is a pure
    [B,NCH,nc,dqk,dv] fp32 transient written for nothing — the prefill-memory blowup. Default False →
    (..., None, None). Optional routing bias b_r/b_w ∈ [D,b] (softmax(h·W+b)). No [L,nc] gate/den/r̃ buffer.
    Optional per-head decay weight Wg:[H,d_model] (GLA, #45) → USE_G decayed scan, the per-state log-decay
    computed IN-KERNEL (clamped at _GLA_FLOOR, never a [L,nc] ld); Wg=None is RLA (USE_G=False)."""
    B, L, dqk = q.shape
    dv = v.shape[-1]
    d_model = h.shape[-1]
    nc = b ** D
    BC = 16   # tl.dot needs the gram dim >=16; the nc tail is cmask'd (was max(16,min(nc,16)) ≡ 16)
    BK = _kappa_bk_cap(dqk, dv, _KAPPA_BWD_CHUNK, BC)    # feature-tile (loop ND) — bounds the [BT,BC*BK] tiles
    BV = _kappa_bv_tile(dqk, dv, BC)                     # value-tile (loop ND_V) so [BC*BK,BV] fits SRAM
    BD, NDM = _router_bd_ndm(d_model, b)                  # F5: cap BD so the [BD,BB] router tile fits SRAM
    BB = max(16, triton.next_power_of_2(b))
    # The [BT,BC*BK] rq/wk tiles scale with BT, so the heavy-tile fwd chunk is capped like the bwd (the
    # chunked scan is chunk-size invariant). Cheap dqk/dv keep the full fwd chunk (BC*BK*BT then fits).
    chunk = chunk if BK * BC * chunk * 4 <= 24 * 1024 else min(chunk, _KAPPA_BWD_CHUNK)
    NCBLK = triton.cdiv(nc, BC)
    ND = triton.cdiv(dqk, BK)
    NCH = triton.cdiv(L, chunk)
    Sval = torch.zeros(B, nc, dqk, dv, device=q.device, dtype=torch.float32)
    Sden = torch.zeros(B, nc, dqk, device=q.device, dtype=torch.float32)
    # F7: num at the TRUE dv (not the padded next_pow2(dv)) — the in-kernel store is already vfmask'd to
    # `offs_vf < dv` (the [dv, NDVP*BV) pad lanes never touch memory), so the buffer needs only dv columns,
    # exactly like dq/dk/dvv above. Saves the BVO−dv pad columns (the [B,L,*] alloc) every forward.
    num = torch.zeros(B, L, dv, device=q.device, dtype=torch.float32)
    den = torch.zeros(B, L, device=q.device, dtype=torch.float32)
    # Snapshots = the per-chunk pre-state the BACKWARD recompute consumes — NOT the forward (which
    # discards them) and NOT inference (no backward). Allocating+writing them unconditionally was a
    # [NCH,B,nc,dqk,dv] fp32 transient built for nothing on the forward/prefill path.
    # F6: NCH is the LEADING axis so the backward's per-chunk slice `snap_*[c]` is an already-contiguous
    # [B,nc,dqk,dv]/[B,nc,dqk] view (identical strides to the contiguous dSval/dSden the kernels assume) —
    # dropping the per-chunk `.contiguous()` copy in `_kappa_routed_bwd` (output byte-identical).
    if need_snapshots:
        snap_val = torch.zeros(NCH, B, nc, dqk, dv, device=q.device, dtype=torch.float32)
        snap_den = torch.zeros(NCH, B, nc, dqk, device=q.device, dtype=torch.float32)
    else:
        snap_val = snap_den = None
    H = Wr.shape[0]
    br, bw, has_bias = _routing_bias(b_r, b_w, H, D, b, q.device, dtype=q.dtype)
    sbias = _bias_strides(br, bw, has_bias)
    head = _router_head_strides(Wr, Ww, br, bw, has_bias)
    use_g = Wg is not None
    Wg = (Wg.float().contiguous() if use_g else q.new_zeros(H, d_model))
    swg = (Wg.stride(0), Wg.stride(1))
    common = dict(GLOBAL=global_norm, PER_STATE=per_state, EPS=eps, D=D, b=b, BB=BB, BT=chunk,
                  BK=BK, BV=BV, BD=BD, BC=BC, NCBLK=NCBLK, ND=ND, NDM=NDM, NDV=triton.cdiv(dv, BV),
                  NDVP=triton.next_power_of_2(triton.cdiv(dv, BV)),
                  HAS_BIAS=has_bias, USE_G=use_g, GLA_FLOOR=_GLA_FLOOR, num_warps=4, num_stages=1)
    for c in range(NCH):
        if need_snapshots:
            snap_val[c].copy_(Sval)
            snap_den[c].copy_(Sden)
        _kappa_fwd_chunk[(B,)](
            h, q, k, v, Wr, Ww, sel, kap, Wg, Sval, Sden, num, den, br, bw,
            L, d_model, dqk, dv, nc, c * chunk, *head,
            h.stride(0), h.stride(1), h.stride(2), q.stride(0), q.stride(1), q.stride(2),
            v.stride(0), v.stride(1), v.stride(2), kap.stride(0), kap.stride(1),
            Wr.stride(1), Wr.stride(2), Wr.stride(3), Ww.stride(1), Ww.stride(2), Ww.stride(3),
            sel.stride(0), sel.stride(1), sel.stride(2), *swg,
            Sval.stride(0), Sval.stride(1), Sval.stride(2), Sval.stride(3),
            Sden.stride(0), Sden.stride(1), Sden.stride(2),
            num.stride(0), num.stride(1), num.stride(2), den.stride(0), den.stride(1),
            *sbias,
            **common)
    return num, den, snap_val, snap_den


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
def _kappa_bwd_state(h_ptr, k_ptr, v_ptr, wr_ptr, ww_ptr, sel_ptr, wg_ptr, sval_ptr, sden_ptr,
                     dsval_ptr, dsden_ptr, dk_ptr, dv_ptr, gdw_ptr, gda_ptr, br_ptr, bw_ptr,
                     L, d_model, dqk, dv, nc, t_start,
                     H, swr_head, sww_head, sbr_head, sbw_head,
                     sh_b, sh_l, sh_d, sq_b, sq_l, sq_d, sv_b, sv_l, sv_d,
                     swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b, ssel_lvl, ssel_b, ssel_c,
                     swg_head, swg_d,
                     ssv_b, ssv_k, ssv_v, ssd_b, ssd_k, sgd_b, sgd_t, sgd_c, sga_b, sga_l, sga_c,
                     sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                     D: tl.constexpr, b: tl.constexpr, BB: tl.constexpr,
                     BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BD: tl.constexpr,
                     BC: tl.constexpr, NCBLK: tl.constexpr, ND: tl.constexpr, NDM: tl.constexpr,
                     HAS_BIAS: tl.constexpr, USE_G: tl.constexpr, GLA_FLOOR: tl.constexpr):
    """State-update backward (split #1, SMEM-bound by the dSval tile). Uses the INCOMING adjoint
    (= Sval_{j+1}, Sden_{j+1}) to backprop the two writes Sval^c += Σ wᶜ k⊗v and Sden^c += Σ wᶜ k.
    Produces dk,dv (atomic) and the state half of dw (stored to gdw). w rebuilt in-kernel; no [L,nc].
    USE_G (GLA): both writes use w_end=w·e^{Λ-a}; dk/dv flow through w_end; the write routing-factor grad
    is dw_end·e^{Λ-a} (→ gdw). da-pieces: da_wend=−dw_end·w_end, and the per-chunk Λ-coupling
    dlam = e^Λ·Σ(S_j∘ds_in) − Σ_t da_wend (over BOTH the Sval and Sden carries) on the LAST row of gda.
    The ds_in here is the PRE-decvec adjoint of S_{j+1}; the driver decays dS by e^Λ AFTER this kernel."""
    pid_b = tl.program_id(0)
    ND_V = (dv + BV - 1) // BV       # value-blocks (BV is the value TILE; ND_V==1 ⇒ the un-tiled kernel)
    offs_t = tl.arange(0, BT)
    offs_bb = tl.arange(0, BB)
    offs_c = tl.arange(0, BC)
    bmask = offs_bb < b
    rows = t_start + offs_t
    rmask = rows < L
    _hd = pid_b % H                              # per-head router slice (BH fold is (B,H))
    wr_ptr = wr_ptr + _hd * swr_head
    ww_ptr = ww_ptr + _hd * sww_head
    br_ptr = br_ptr + _hd * sbr_head
    bw_ptr = bw_ptr + _hd * sbw_head
    wg_ptr = wg_ptr + _hd * swg_head            # per-head decay weight (read-only here; #45)
    if USE_G:                                   # per-head decay gate alpha[BT], once (in-kernel ld, #45)
        alpha = _build_alpha(h_ptr, wg_ptr, pid_b, rows, rmask, d_model,
                             sh_b, sh_l, sh_d, swg_d, BT, BD, NDM)
    for cb in range(NCBLK):
        cols = cb * BC + offs_c
        cmask = cols < nc
        _r, w_tile = _build_rw_tile(h_ptr, wr_ptr, ww_ptr, sel_ptr, cols, cmask,
                                    pid_b, rows, rmask, offs_bb, bmask, d_model,
                                    sh_b, sh_l, sh_d, swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
                                    ssel_lvl, ssel_b, ssel_c,
                                    br_ptr, bw_ptr, sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                                    D, BT, BB, BC, BD, NDM, HAS_BIAS, BUILD_R=False)  # write-only: skip read
        if USE_G:
            ldc = _ld_from_w(w_tile, alpha, cmask, GLA_FLOOR)
            a = tl.cumsum(ldc, axis=0)
            Lam = tl.sum(tl.where(offs_t[:, None] == (BT - 1), a, 0.0), axis=0)   # [BC] chunk-total
            wend = w_tile * tl.exp(Lam[None, :] - a)     # decayed write gate (both writes)
        else:
            wend = w_tile
        # dw[BT,BC] sums over dqk → accumulate across BK-feature-blocks. dk[BT,BK] is per-feature-block
        # (offs_k), stored per d0. dv[BT,BV] is per value-block (offs_v), atomic-added per (d0,vb). The
        # value-contracted Nval=v·dSvalᵀ is summed over value-blocks; the [BC*BK,BV] dSval slice stays
        # bounded by BK·BV<=64·BV. wend[BT,BC] / wk[BT,BC*BK] are value-free, reused across vb.
        dw = tl.zeros([BT, BC], dtype=tl.float32)        # grad w.r.t. w_end (USE_G) | w (RLA)
        ZdZ = tl.zeros([BC], dtype=tl.float32)           # Σ_{k,v}(S_j∘ds_in) over BOTH carries (USE_G)
        for d0 in range(ND):
            offs_k = d0 * BK + tl.arange(0, BK)
            kmask = offs_k < dqk
            kc = tl.load(k_ptr + pid_b*sq_b + rows[:, None]*sq_l + offs_k[None, :]*sq_d,
                         mask=rmask[:, None] & kmask[None, :], other=0.0)
            ck = tl.reshape(cols[:, None] * dqk + offs_k[None, :], [BC * BK])
            ckmask = tl.reshape(cmask[:, None] & kmask[None, :], [BC * BK])
            wk = tl.reshape(wend[:, :, None] * kc[:, None, :], [BT, BC * BK])
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
                if USE_G:
                    # Sval carry Λ-grad: Σ(Sval_j ∘ dSval_in), accumulated into ZdZ.
                    sj = tl.load(sval_ptr + pid_b*ssv_b + ck[:, None]*ssv_k + offs_v[None, :]*ssv_v,
                                 mask=ckmask[:, None] & vmask[None, :], other=0.0)
                    sd = tl.where(ckmask[:, None] & vmask[None, :], sj * dSval, 0.0)
                    ZdZ += tl.sum(tl.reshape(tl.sum(sd, axis=1), [BC, BK]), axis=1)
            Nvr = tl.reshape(Nval, [BT, BC, BK])
            dw += tl.sum(Nvr * kc[:, None, :], axis=2)
            dk_acc = tl.sum(Nvr * wend[:, :, None], axis=1)
            dSden = tl.load(dsden_ptr + pid_b*ssd_b + ck*ssd_k, mask=ckmask, other=0.0)
            dSden2 = tl.reshape(dSden, [BC, BK])
            dw += tl.where(cmask[None, :], tl.sum(dSden2[None, :, :] * kc[:, None, :], axis=2), 0.0)
            dk_acc += tl.dot(wend.to(dSden2.dtype), dSden2)
            if USE_G:
                # Sden carry Λ-grad: Σ_k(Sden_j ∘ dSden_in), accumulated into ZdZ.
                sjd = tl.load(sden_ptr + pid_b*ssd_b + ck*ssd_k, mask=ckmask, other=0.0)
                sdd = tl.where(ckmask, sjd * dSden, 0.0)
                ZdZ += tl.sum(tl.reshape(sdd, [BC, BK]), axis=1)
            tl.atomic_add(dk_ptr + pid_b*sq_b + rows[:, None]*sq_l + offs_k[None, :]*sq_d,
                          dk_acc, mask=rmask[:, None] & kmask[None, :])
        if USE_G:
            # dw is grad w.r.t. w_end → routing-factor grad dw_end·e^{Λ-a}; da_wend=−dw·w_end; dlam on last row.
            dw_tile = tl.where(cmask[None, :], dw * tl.exp(Lam[None, :] - a), 0.0)
            da_wend = tl.where(cmask[None, :], -dw * wend, 0.0)
            dlam = tl.exp(Lam) * ZdZ - tl.sum(da_wend, axis=0)                 # [BC]
            da = da_wend + tl.where(offs_t[:, None] == (BT - 1), dlam[None, :], 0.0)
            tl.atomic_add(gda_ptr + pid_b*sga_b + rows[:, None]*sga_l + cols[None, :]*sga_c,
                          tl.where(cmask[None, :], da, 0.0), mask=rmask[:, None] & cmask[None, :])
        else:
            dw_tile = dw
        tl.store(gdw_ptr + pid_b*sgd_b + offs_t[:, None]*sgd_t + cols[None, :]*sgd_c,
                 dw_tile, mask=rmask[:, None] & cmask[None, :])


@triton.jit
def _kappa_bwd_read(h_ptr, q_ptr, k_ptr, v_ptr, wr_ptr, ww_ptr, sel_ptr, kap_ptr, wg_ptr,
                    sval_ptr, sden_ptr, dnum_ptr, dden_ptr,
                    dsval_ptr, dsden_ptr, dq_ptr, dk_ptr, dv_ptr, dkap_ptr, gdr_ptr, gdw_ptr, gda_ptr,
                    br_ptr, bw_ptr,
                    L, d_model, dqk, dv, nc, t_start,
                    H, swr_head, sww_head, sbr_head, sbw_head,
                    sh_b, sh_l, sh_d, sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sk_b, sk_l,
                    swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b, ssel_lvl, ssel_b, ssel_c,
                    swg_head, swg_d,
                    ssv_b, ssv_k, ssv_v, ssd_b, ssd_k,
                    sdo_b, sdo_l, sdo_v, sdd_b, sdd_l, sgd_b, sgd_t, sgd_c, sga_b, sga_l, sga_c,
                    sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                    GLOBAL: tl.constexpr, PER_STATE: tl.constexpr, EPS: tl.constexpr,
                    D: tl.constexpr, b: tl.constexpr, BB: tl.constexpr,
                    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BD: tl.constexpr,
                    BC: tl.constexpr, NCBLK: tl.constexpr, ND: tl.constexpr, NDM: tl.constexpr,
                    HAS_BIAS: tl.constexpr, USE_G: tl.constexpr, GLA_FLOOR: tl.constexpr):
    """Readout/den/d backward (split #2, SMEM-bound by the sval snapshot tile). Recomputes r,w,d,r_tilde
    transiently, backprops den + intra num + inter-readout + the per-state-den d, producing dr,dq,dv-intra,
    dkappa, the read+intra+d half of dw (atomic-added into gdw), and the readout contribution to the
    carried dSval/dSden adjoints. dG -> dq,dk. The [L,nc] gates / d / r_tilde are never materialized.
    USE_G (GLA): the readout uses the decayed rd=r̃·e^a and the gram/den intra-write wt=w·e^{-a}; the den d
    itself carries decay (d=e^a·(Gc·wt + q·Sden)). The kap rescale wraps the UNDECAYED r̃; the den reduction
    o_den=Σ_c r̃^c d^c uses that UNDECAYED r̃ (no e^a). da-pieces (→ gda): dart=drd·rd (readout),
    da_wt=−dwt·wt (gram + den intra write, dwt the grad on the DECAYED wt), da_d_outer=dd·d (the den's own
    outer e^a). The driver reverse-cumsums gda → dld; the Sden carry's Λ-grad is handled in state-bwd."""
    pid_b = tl.program_id(0)
    ND_V = (dv + BV - 1) // BV       # value-blocks (BV is the value TILE; ND_V==1 ⇒ the un-tiled kernel)
    offs_t = tl.arange(0, BT)
    offs_bb = tl.arange(0, BB)
    offs_c = tl.arange(0, BC)
    bmask = offs_bb < b
    rows = t_start + offs_t
    rmask = rows < L
    _hd = pid_b % H                              # per-head router slice (BH fold is (B,H))
    wr_ptr = wr_ptr + _hd * swr_head
    ww_ptr = ww_ptr + _hd * sww_head
    br_ptr = br_ptr + _hd * sbr_head
    bw_ptr = bw_ptr + _hd * sbw_head
    wg_ptr = wg_ptr + _hd * swg_head            # per-head decay weight (read-only here; #45)
    if USE_G:                                   # per-head decay gate alpha[BT], once (in-kernel ld, #45)
        alpha = _build_alpha(h_ptr, wg_ptr, pid_b, rows, rmask, d_model,
                             sh_b, sh_l, sh_d, swg_d, BT, BD, NDM)
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
        if USE_G:
            ldc = _ld_from_w(w_tile, alpha, cmask, GLA_FLOOR)
            a = tl.cumsum(ldc, axis=0)
            ea = tl.exp(a)
            ena = tl.exp(-a)
            wt = w_tile * ena                  # intra gram / den-intra write w·e^{-a}
        else:
            ea = w_tile * 0.0 + 1.0
            ena = ea
            wt = w_tile
        # d_tile/rt_tile are value-FREE (from G,Sden,r) → compute once, reused for every value-block.
        d_tile = _kappa_d_tile(q_ptr, sden_ptr, Gc, w_tile, ea, ena, cols, cmask,
                               pid_b, rows, rmask, dqk, sq_b, sq_l, sq_d, ssd_b, ssd_k,
                               BT, BK, BC, ND, USE_G)
        rt_tile = _kappa_rescale(r_tile, d_tile, kap, cmask, GLOBAL, PER_STATE, EPS)   # UNDECAYED r̃
        rd_tile = (rt_tile * ea) if USE_G else rt_tile   # decayed read gate r̃·e^a (readout only)
        # For RLA (ea≡1, wt≡w, rd≡r̃) the grad folds DIRECTLY into drt/dw with the SAME accumulation order
        # as the base RLA kernel → byte-identical. For GLA the readout/gram grads collect on the DECAYED
        # gates (drd/dwt) and the da-pieces (drd·rd, −dwt·wt, dd·d) accumulate into gda.
        drt = dden[:, None] * d_tile          # d(r_tilde) from den (UNDECAYED r̃ → no e^a)
        dd = dden[:, None] * rt_tile          # d(d) from den
        drd = tl.zeros([BT, BC], dtype=tl.float32)   # GLA readout grad w.r.t. the DECAYED read gate rd
        # num intra: A=G*Rg*causal ; o_num += A v. dA=(dnum·vᵀ)⊙causal (value-contracted → sum over vb);
        # dv=Aᵀ·dnum (per value-block → atomic). A/Rg/dG/dw are value-free; dnum,vc loaded per vb.
        Rg = tl.dot(rd_tile, tl.trans(wt))
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
        if USE_G:
            drd += tl.dot(dRg.to(wt.dtype), wt)
            dwt = tl.dot(tl.trans(dRg).to(rd_tile.dtype), rd_tile)
        else:
            drt += tl.dot(dRg.to(w_tile.dtype), w_tile)
            dw = tl.dot(tl.trans(dRg).to(rt_tile.dtype), rt_tile)
        # num inter readout: o_num += sum_c rd^c (q.Sval_j^c). M=dnum·svalᵀ is value-contracted (sum over
        # vb); dq + the dSval-store are per-(BK,value)-block; the read grad's inter part sums over d0.
        for d0 in range(ND):
            offs_k = d0 * BK + tl.arange(0, BK)
            kmask = offs_k < dqk
            qc = tl.load(q_ptr + pid_b*sq_b + rows[:, None]*sq_l + offs_k[None, :]*sq_d,
                         mask=rmask[:, None] & kmask[None, :], other=0.0)
            ck = tl.reshape(cols[:, None] * dqk + offs_k[None, :], [BC * BK])
            ckmask = tl.reshape(cmask[:, None] & kmask[None, :], [BC * BK])
            rq = tl.reshape(rd_tile[:, :, None] * qc[:, None, :], [BT, BC * BK])
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
            if USE_G:
                drd += tl.sum(Mr * qc[:, None, :], axis=2)
            else:
                drt += tl.sum(Mr * qc[:, None, :], axis=2)
            dq_read = tl.sum(Mr * rd_tile[:, :, None], axis=1)   # [BT,BK]
            tl.atomic_add(dq_ptr + pid_b*sq_b + rows[:, None]*sq_l + offs_k[None, :]*sq_d,
                          dq_read, mask=rmask[:, None] & kmask[None, :])
        # readout grad → UNDECAYED r̃ (×e^a) — GLA only (RLA folded drt directly above).
        if USE_G:
            drt += drd * ea
        # rescale bwd (drt now complete; drt is grad w.r.t. the UNDECAYED r̃).
        dr_resc, dd_resc, dkap_c = _kappa_rescale_bwd(drt, r_tile, d_tile, kap, cmask,
                                                      GLOBAL, PER_STATE, EPS)
        dd += dd_resc
        dkap_acc += dkap_c
        # d bwd: d = e^a·(Gc·wt + q·Sden_j) [GLA] | (Gc·w + q·Sden_j) [RLA]. ds = dd·e^a (the inner-sum
        # grad); the outer e^a contributes da_d=dd·d. dG/dw are dqk-free; dq + dSden contract dqk → loop d0.
        ds = (dd * ea) if USE_G else dd
        dG += tl.dot(ds.to(wt.dtype), tl.trans(wt)) * causal
        if USE_G:
            dwt += tl.dot(tl.trans(Gc).to(ds.dtype), ds)        # den intra → DECAYED wt
        else:
            dw += tl.dot(tl.trans(Gc).to(ds.dtype), ds)
        for d0 in range(ND):
            offs_k = d0 * BK + tl.arange(0, BK)
            kmask = offs_k < dqk
            qc = tl.load(q_ptr + pid_b*sq_b + rows[:, None]*sq_l + offs_k[None, :]*sq_d,
                         mask=rmask[:, None] & kmask[None, :], other=0.0)
            ck = tl.reshape(cols[:, None] * dqk + offs_k[None, :], [BC * BK])
            ckmask = tl.reshape(cmask[:, None] & kmask[None, :], [BC * BK])
            sden = tl.load(sden_ptr + pid_b*ssd_b + ck*ssd_k, mask=ckmask, other=0.0)
            sden2 = tl.reshape(sden, [BC, BK])
            dq_d = tl.dot(ds.to(sden2.dtype), sden2)             # [BT,BK]
            tl.atomic_add(dq_ptr + pid_b*sq_b + rows[:, None]*sq_l + offs_k[None, :]*sq_d,
                          dq_d, mask=rmask[:, None] & kmask[None, :])
            dSden = tl.load(dsden_ptr + pid_b*ssd_b + ck*ssd_k, mask=ckmask, other=0.0)
            dSden_read = tl.reshape(tl.dot(tl.trans(ds).to(qc.dtype), qc), [BC * BK])
            tl.store(dsden_ptr + pid_b*ssd_b + ck*ssd_k, dSden + dSden_read, mask=ckmask)
        # write-gate grad: GLA — dwt is the grad w.r.t. the DECAYED wt=w·e^{-a}; routing-factor grad
        # dw=dwt·e^{-a}; da_wt=−dwt·wt; the den's outer-e^a da-piece is dd·d, the readout's is drd·rd.
        if USE_G:
            dw = dwt * ena
            da = drd * rd_tile - dwt * wt + dd * d_tile
            tl.atomic_add(gda_ptr + pid_b*sga_b + rows[:, None]*sga_l + cols[None, :]*sga_c,
                          tl.where(cmask[None, :], da, 0.0), mask=rmask[:, None] & cmask[None, :])
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
                      D, b, sel, chunk, global_norm, per_state, eps, b_r=None, b_w=None, Wg=None):
    """Reverse chunk-scan backward for the fused global/kappa/per_state path. Carries dSval,dSden
    adjoints; recomputes r,w,d,r_tilde transiently per chunk; folds the [BT,nc] gate-grads into
    dWr,dWw,dh (and db_r/db_w when a routing bias is present — transient, never [L,nc]). Returns
    dq,dk,dv,dh,dWr,dWw,dkappa,dWg (and db_r,db_w when biased) — all fp32. No [L,nc] buffer is allocated.

    USE_G (GLA, #45): optional per-head decay weight Wg:[H,d_model] → the decayed kappa backward, the
    per-state log-decay ld computed IN-KERNEL (never a [L,nc] ld). The two grad kernels run their USE_G
    paths (decayed gates rd=r̃·e^a, wt=w·e^{-a}, w_end=w·e^{Λ-a}; the den d carries decay; the da-pieces
    accumulate into a persistent gda[B,L,nc]). Between state-bwd and read-bwd BOTH carried adjoints decay
    by e^Λ (the reverse of the forward's e^Λ state carry; Λ_c recomputed chunk-locally from Wg). At each
    chunk gda is reverse-cumsummed → a PER-CHUNK dld[B,chunk,nc] (never [L,nc]); the fold splits it into
    dWg + the extra dh + the decay's write-gate grad. Wg=None is RLA (byte-identical; gda/dld unused)."""
    use_g = Wg is not None
    B, L, d_model = h.shape
    dqk = q.shape[-1]
    dv = v.shape[-1]
    nc = b ** D
    BC = 16   # tl.dot gram dim >=16; nc tail cmask'd (was max(16,min(nc,16)) ≡ 16)
    BK = _kappa_bk_cap(dqk, dv, chunk, BC)   # feature-tile (loop ND) — bounds the [BT,BC*BK] tiles
    BV = _kappa_bv_tile(dqk, dv, BC, BK)     # value-tile (loop ND_V) so the [BC*BK,BV] state slices fit
    BD, NDM = _router_bd_ndm(d_model, b)     # F5: cap BD so the [BD,BB] router tile fits SRAM (NDM-tiled)
    BB = max(16, triton.next_power_of_2(b))
    NCBLK = triton.cdiv(nc, BC)
    ND = triton.cdiv(dqk, BK)
    NCH = triton.cdiv(L, chunk)
    # dvv/dq/dk are written with v's / q's row strides (sv_l=v.stride(1)=dv, sq_l=q.stride(1)=dqk), so
    # they MUST be allocated at the TRUE dv/dqk width (NOT padded BV/BK) or the row layout corrupts for
    # non-pow2 dv/dqk. The in-kernel [BT,BV]/[BT,BK] accumulators store only the masked :dv/:dqk part.
    H = Wr.shape[0]
    dq = torch.zeros(B, L, dqk, device=q.device, dtype=torch.float32)
    dk = torch.zeros(B, L, dqk, device=q.device, dtype=torch.float32)
    dvv = torch.zeros(B, L, dv, device=q.device, dtype=torch.float32)
    dh = torch.zeros(B, L, d_model, device=q.device, dtype=torch.float32)
    dWr = torch.zeros(H, D, d_model, b, device=q.device, dtype=torch.float32)   # per-head [H,D,d_model,b]
    dWw = torch.zeros(H, D, d_model, b, device=q.device, dtype=torch.float32)
    dWg = torch.zeros(H, d_model, device=q.device, dtype=torch.float32)   # per-head decay-weight grad (#45)
    dkap = torch.zeros(B, L, device=q.device, dtype=torch.float32)
    dSval = torch.zeros(B, nc, dqk, dv, device=q.device, dtype=torch.float32)
    dSden = torch.zeros(B, nc, dqk, device=q.device, dtype=torch.float32)
    br, bw, has_bias = _routing_bias(b_r, b_w, H, D, b, q.device, dtype=torch.float32)
    dbr = torch.zeros(H, D, b, device=q.device, dtype=torch.float32)   # per-head routing-bias grads [H,D,b]
    dbw = torch.zeros(H, D, b, device=q.device, dtype=torch.float32)
    dnum = dnum.contiguous()
    dden = dden.contiguous()
    # Wg:[H,d_model] (GLA) — the per-head decay weight; ld is computed IN-KERNEL (never a [L,nc] ld). RLA
    # passes a [H,d_model] stub the kernels skip (USE_G=False).
    Wg = (Wg.float().contiguous() if use_g else q.new_zeros(H, d_model))
    swg = (Wg.stride(0), Wg.stride(1))
    # gda[B,L,nc]: persistent per-token log-decay adjoint (USE_G) — state/read kernels add their da-pieces
    # per chunk; reverse-cumsummed per chunk into a PER-CHUNK dld at the fold. RLA leaves it a 1-col stub.
    gda = torch.zeros(B, L, nc, device=q.device, dtype=torch.float32) if use_g else q.new_zeros(B, 1, 1)
    sga = (gda.stride(0), gda.stride(1), gda.stride(2))
    dld_chunk = torch.zeros(B, chunk, nc, device=q.device, dtype=torch.float32) if use_g \
        else q.new_zeros(B, 1, 1)
    # transient per-chunk gate-grad scratch [B,chunk,nc], OVERWRITTEN each chunk — never [L,nc].
    gdr = torch.zeros(B, chunk, nc, device=q.device, dtype=torch.float32)
    gdw = torch.zeros(B, chunk, nc, device=q.device, dtype=torch.float32)
    sB = (q.stride(0), q.stride(1), q.stride(2))
    sV = (v.stride(0), v.stride(1), v.stride(2))
    sH = (h.stride(0), h.stride(1), h.stride(2))
    sWr = (Wr.stride(1), Wr.stride(2), Wr.stride(3))   # per-head [H,D,d_model,b]: lvl/d/b strides (dim0=head)
    sWw = (Ww.stride(1), Ww.stride(2), Ww.stride(3))
    sSel = (sel.stride(0), sel.stride(1), sel.stride(2))
    head = _router_head_strides(Wr, Ww, br, bw, has_bias)   # (H, swr_head, sww_head, sbr_head, sbw_head)
    # snapshot (Sval_j/Sden_j) strides for the state-bwd ZdZ term — same (B, flat-k, v) layout as dSval.
    sSV = (dSval.stride(0), dSval.stride(2), dSval.stride(3))   # (B, flat-k=dqk-axis, v)
    sSD = (dSden.stride(0), dSden.stride(2))                    # (B, flat-k)
    sGD = (gdr.stride(0), gdr.stride(1), gdr.stride(2))
    sBias = _bias_strides(br, bw, has_bias)
    state_common = dict(D=D, b=b, BB=BB, BT=chunk, BK=BK, BV=BV, BD=BD, BC=BC, NCBLK=NCBLK, ND=ND,
                        NDM=NDM, HAS_BIAS=has_bias, USE_G=use_g, GLA_FLOOR=_GLA_FLOOR,
                        num_warps=4, num_stages=1)
    read_common = dict(GLOBAL=global_norm, PER_STATE=per_state, EPS=eps, D=D, b=b, BB=BB, BT=chunk,
                       BK=BK, BV=BV, BD=BD, BC=BC, NCBLK=NCBLK, ND=ND, NDM=NDM,
                       HAS_BIAS=has_bias, USE_G=use_g, GLA_FLOOR=_GLA_FLOOR, num_warps=4, num_stages=1)
    fold_common = dict(D=D, b=b, BB=BB, BT=chunk, BC=BC, BD=BD, NCBLK=NCBLK, NDM=NDM,
                       HAS_BIAS=has_bias, USE_G=use_g, GLA_FLOOR=_GLA_FLOOR, num_warps=4, num_stages=1)
    for c in reversed(range(NCH)):
        Sval = snap_val[c]    # F6: NCH-leading layout → per-chunk slice is already contiguous (no copy)
        Sden = snap_den[c]
        gdw.zero_()
        # K1: state-update bwd (reads adjoint of Sval_{j+1}/Sden_{j+1}; produces dk,dv + state half of dw +
        # the USE_G carry/w_end da-pieces, using the chunk-start snapshot Sval_j/Sden_j for the Λ-coupling).
        _kappa_bwd_state[(B,)](
            h, k, v, Wr, Ww, sel, Wg, Sval, Sden, dSval, dSden, dk, dvv, gdw, gda, br, bw,
            L, d_model, dqk, dv, nc, c * chunk, *head,
            *sH, *sB, *sV, *sWr, *sWw, *sSel, *swg, *sSV, *sSD, *sGD, *sga, *sBias, **state_common)
        if use_g:
            # decay the running adjoints by decvec=e^{Λ_c} (per state) — the reverse of the forward's
            # e^Λ state carry. AFTER state-bwd reads the S_{j+1} adjoint (+ZdZ), BEFORE read-bwd folds
            # dS_read → the adjoint of S_j. Applied to BOTH Sval and Sden carries. Λ_c is the chunk-total
            # of the in-kernel ld, recomputed chunk-locally from Wg ([B,len,nc], never [L,nc]).
            r0, r1 = c * chunk, min(c * chunk + chunk, L)
            ld_c = _ld_chunk(h[:, r0:r1], Wr, Ww, Wg, D, b, H, b_r=br if has_bias else None,
                             b_w=bw if has_bias else None)
            Lam_c = ld_c.sum(dim=1)                                 # [B,nc] chunk-total per state
            dSval = dSval * torch.exp(Lam_c)[:, :, None, None]
            dSden = dSden * torch.exp(Lam_c)[:, :, None]
        # K2: readout/den/d bwd (adds dw, produces dr,dq,dv-intra,dkappa; folds dSval/dSden adjoints).
        _kappa_bwd_read[(B,)](
            h, q, k, v, Wr, Ww, sel, kap, Wg, Sval, Sden, dnum, dden,
            dSval, dSden, dq, dk, dvv, dkap, gdr, gdw, gda, br, bw,
            L, d_model, dqk, dv, nc, c * chunk, *head,
            *sH, *sB, *sV, kap.stride(0), kap.stride(1), *sWr, *sWw, *sSel, *swg, *sSV, *sSD,
            dnum.stride(0), dnum.stride(1), dnum.stride(2), dden.stride(0), dden.stride(1), *sGD, *sga,
            *sBias,
            **read_common)
        if use_g:
            # assemble dld for THIS chunk = intra-chunk reverse-cumsum of gda (a resets each chunk in the
            # fwd, so it's intra-only). Chunk-local [B,chunk,nc]; consumed by the fold below, never [L,nc].
            r0, r1 = c * chunk, min(c * chunk + chunk, L)
            g_sl = gda[:, r0:r1]
            tot = g_sl.sum(dim=1, keepdim=True)
            dld_chunk.zero_()
            dld_chunk[:, :r1 - r0] = tot - g_sl.cumsum(dim=1) + g_sl
        # fold the transient gate-grads -> dWr,dWw,dh (+db); USE_G also splits dld_chunk → dWg + extra dh +
        # the decay's write-gate grad (added to gdw inside the fold). Factor-rebuild SMEM isolated here.
        _routed_bwd_fold[(B,)](
            h, Wr, Ww, sel, gdr, gdw, dh, dWr, dWw, br, bw, dbr, dbw,
            Wg, dWg, dld_chunk,
            L, d_model, nc, c * chunk, *head, *sH, *sWr, *sWw, *sSel,
            gdr.stride(0), gdr.stride(1), gdr.stride(2), dh.stride(0), dh.stride(1), dh.stride(2),
            *swg,
            *sBias,
            **fold_common)
    if has_bias:
        if use_g:
            return (dq[..., :dqk], dk[..., :dqk], dvv[..., :dv], dh, dWr, dWw, dkap, dWg, dbr, dbw)
        return (dq[..., :dqk], dk[..., :dqk], dvv[..., :dv], dh, dWr, dWw, dkap, dbr, dbw)
    if use_g:
        return (dq[..., :dqk], dk[..., :dqk], dvv[..., :dv], dh, dWr, dWw, dkap, dWg)
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
    def forward(ctx, q, k, v, h, Wr, Ww, kap, D, b, chunk, global_norm, per_state, eps, b_r, b_w, Wg=None):
        # GLA (USE_G) caps the FORWARD chunk at _CHUNK (32) — the fp32 decay floor + the 3-pass kappa
        # mega-kernel's SMEM wall at BT=64; RLA keeps the full _CHUNK_FWD (64). Chunk-size invariant.
        cap = _CHUNK if Wg is not None else _CHUNK_FWD
        chunk = cap if chunk is None else min(chunk, cap)
        nc = b ** D
        sel = _build_sel(D, b, nc, q.device)
        q, k, v, h, Wr, Ww, kap = (x.contiguous() for x in (q, k, v, h, Wr, Ww, kap))
        Wgc = Wg.contiguous() if Wg is not None else None
        num, den, _sv, _sd = _kappa_routed_fwd(q, k, v, h, Wr, Ww, kap, D, b, sel, chunk,
                                               global_norm, per_state, eps, b_r=b_r, b_w=b_w, Wg=Wgc)
        # #45: the GLA decay's saved activation is the [H,d_model] Wg (NOT a [L,nc] ld) — the saved-act win.
        ctx.save_for_backward(q, k, v, h, Wr, Ww, kap, b_r, b_w, Wgc)
        ctx.D, ctx.b, ctx.chunk = D, b, chunk
        ctx.global_norm, ctx.per_state, ctx.eps = global_norm, per_state, eps
        return num.to(q.dtype), den.to(q.dtype)

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dnum, dden):
        q, k, v, h, Wr, Ww, kap, b_r, b_w, Wg = ctx.saved_tensors
        use_g = Wg is not None
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
        Wgf = Wg.float().contiguous() if use_g else None
        _num, _den, snap_val, snap_den = _kappa_routed_fwd(
            q, k, v, h, Wr, Ww, kap, D, b, sel, chunk, global_norm, per_state, eps,
            b_r=brf, b_w=bwf, Wg=Wgf, need_snapshots=True)
        grads = _kappa_routed_bwd(
            q, k, v, h, Wr, Ww, kap, snap_val, snap_den, dnum.float(), dden.float(),
            D, b, sel, chunk, global_norm, per_state, eps, b_r=brf, b_w=bwf, Wg=Wgf)
        # _kappa_routed_bwd returns (dq,dk,dv,dh,dWr,dWw,dkap[,dWg][,dbr,dbw]); dWg present iff use_g,
        # dbr/dbw present iff biased. Unpack positionally so RLA (no dWg) stays byte-identical.
        dq, dk, dv, dh, dWr, dWw, dkap = grads[:7]
        idx = 7
        dWg = grads[idx] if use_g else None
        idx += 1 if use_g else 0
        if b_r is None:
            dbr = dbw = None
        else:
            dbr, dbw = grads[idx], grads[idx + 1]
        q0 = ctx.saved_tensors[0]
        # forward arg order: q,k,v,h,Wr,Ww,kap,D,b,chunk,global_norm,per_state,eps,b_r,b_w,Wg
        return (dq.to(q0.dtype), dk.to(q0.dtype), dv.to(q0.dtype), dh.to(q0.dtype),
                dWr.to(Wr.dtype), dWw.to(Ww.dtype), dkap.to(q0.dtype),
                None, None, None, None, None, None,
                None if dbr is None else dbr.to(b_r.dtype),
                None if dbw is None else dbw.to(b_w.dtype),
                None if dWg is None else dWg.to(Wg.dtype))


def _kappa_routed_readout(qf, kf, vf, hf, Wr, Ww, kapf, D, b, chunk_size, global_norm, per_state, eps,
                          b_r=None, b_w=None, Wg=None):
    """Fused global/kappa/per_state tree-routed readout returning (num[BH,L,V], den[BH,L,1]),
    differentiable. kapf:[BH,L,1]. Optional routing bias b_r/b_w ∈ [D,b]. Optional per-head decay weight
    Wg:[H,d_model] (GLA, the per-state log-decay computed IN-KERNEL); Wg=None is the RLA path
    (byte-identical). CUDA only (the eager fallback stays in the public entry)."""
    kap = kapf.reshape(qf.shape[0], qf.shape[1]).contiguous()    # [BH,L]
    num, den = _RoLARoutedKappaFn.apply(qf, kf, vf, hf, Wr, Ww, kap, D, b, chunk_size,
                                        global_norm, per_state, eps, b_r, b_w, Wg)
    return num, den.unsqueeze(-1)


# --- GLA per-token log-decay floor — a GENUINE fp32+SMEM limit, NOT a tuning artifact (#33) ---------
# The chunked GLA scan FACTORS the per-chunk decay as e^{a_i} (into the read gate) × e^{-a_j} (into the
# write gate), a=cumsum(ld) over a chunk, so the routing gram is ONE tl.dot R=(r·e^a)@(w·e^{-a})ᵀ and
# the state carry is w_end=w·e^{Λ-a}, decvec=e^Λ. The decay DIFFERENCE e^{a_i-a_j} on the causal
# triangle is in (0,1], but each FACTOR e^{±a}, e^{Λ-a} is unbounded: with ld≥FLOOR over BT rows the
# largest factor is e^{BT·|FLOOR|}. fp32 overflows at ln(FLT_MAX)=88.72, so the hard constraint is
#     BT · |FLOOR| ≲ 88.72.
# At BT=32: 32·2.5 = 80 (fp32-safe, measured gram maxerr 1.2e-7). At BT=64: 64·2.5 = 160 → e^160 = inf
# (measured: factored gram goes NaN). The differenced form (e^{a_i-a_j}, FLA simple_gla) would be
# BT-free but cannot be absorbed into the routed read/write matmul nor the cross-chunk state carry —
# it is structural to the chunked GLA recurrence shared by every FLA gated op. SEPARATELY, the fused
# kappa-routed BACKWARD mega-kernel (decayed gram + 3-pass-κ fp32 tiles co-resident) OOMs the 99KB
# Ampere SMEM at BT=64 (measured Required 102400 > 101376) — a second, independent wall that pins
# BT≤32 (bwd≤16) regardless of the floor. Both limits point at the SAME shipped operating point.
#
# FLOOR=-2.5 ⇒ per-token retention ≥ e^{-2.5} = 8.2%/tok. The production layer's ld=log(alpha_chunk)
# CAN dip below this (alpha→0, write_gate→1), so the floor is NOT a structural no-op: it would alter a
# learned decay. Per the no-silent-rewrite rule we make the floor LOUD — `_floor_ld` RAISES on
# out-of-range ld by default; opt into clamp-with-warning via ROLA_GLA_FLOOR_CLAMP=1 (e.g. training
# that tolerates the truncation). All GLA decay sites route through `_floor_ld`.
_GLA_FLOOR = -2.5   # per-token log-decay floor (retention ≥ 8.2%/tok); fp32-safe for BT≤32 (see above)
_GLA_FLOOR_CLAMP = os.environ.get('ROLA_GLA_FLOOR_CLAMP', '0') not in ('0', '', 'false', 'False')
_gla_floor_warned = False


def _floor_ld(ld):
    """Enforce the GLA log-decay floor (#33). ld below `_GLA_FLOOR` would overflow the factored fp32
    decay gram (e^{BT·|FLOOR|}); by default RAISE (no silent semantic rewrite). With
    ROLA_GLA_FLOOR_CLAMP=1, clamp to the floor and warn ONCE. Returns a tensor safe for the chunk
    kernels (dtype/contiguity left to the caller).

    Under torch.compile the data-dependent min-check is a graph break, so skip it while tracing. Only
    `chunk_rola` is compiled, so ONLY it needs the separate eager pre-guard: it runs `_guard_ld(g)` once
    BEFORE dispatching to its compiled region, then the compiled `_floor_ld` (this fn) sees
    is_compiling()==True and applies only the cheap CLAMP (clamp-mode) or pass-through (default),
    trusting that eager guard. `chunk_rola_routed` and `fused_recurrent_rola` are EAGER — they call this
    `_floor_ld` directly (is_compiling()==False), so the loud min-check/raise runs inline for them; they
    do NOT use `_guard_ld`."""
    global _gla_floor_warned
    if torch.compiler.is_compiling():
        return ld.clamp(min=_GLA_FLOOR) if _GLA_FLOOR_CLAMP else ld
    mn = ld.detach().min()
    if mn < _GLA_FLOOR:
        if not _GLA_FLOOR_CLAMP:
            raise ValueError(
                f"GLA log-decay ld below the fp32-safe floor _GLA_FLOOR={_GLA_FLOOR} "
                f"(min ld={mn.item():.4f}). The chunked GLA decay is factored e^{{±a}} and overflows "
                f"fp32 (ln FLT_MAX=88.72) once BT·|ld|≳88.72; BT≤32 needs |ld|≤2.77, so the kernel "
                f"cannot represent this decay rate. Reduce the decay (raise alpha / lower the write "
                f"gate), or set ROLA_GLA_FLOOR_CLAMP=1 to clamp ld to the floor (truncating the "
                f"learned decay) instead of raising."
            )
        if not _gla_floor_warned:
            import warnings
            warnings.warn(
                f"GLA log-decay ld clamped to _GLA_FLOOR={_GLA_FLOOR} (ROLA_GLA_FLOOR_CLAMP=1); "
                f"min ld={mn.item():.4f} truncated. This alters the learned decay rate.",
                stacklevel=2,
            )
            _gla_floor_warned = True
        return ld.clamp(min=_GLA_FLOOR)
    return ld


def _guard_ld(g):
    """EAGER floor guard for the public entrypoints (runs before the compiled region so the
    data-dependent `_floor_ld` min-check never graph-breaks inside torch.compile). Raises/warns
    identically to `_floor_ld`; the value is discarded (the compiled impl re-floors compile-safely)."""
    if g is not None:
        _floor_ld(g)



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
    """Eager (CPU reference path) shared-gram routed readout on folded [BH, T, *] tensors.
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
    [BH,T,*] → [BH,T,nc]. CPU reference path for the Triton den kernels."""
    BH, T, K = q.shape
    G = torch.einsum('bid,bjd->bij', q, k)
    causal = torch.tril(torch.ones(T, T, device=q.device, dtype=q.dtype))
    if ld is None:
        return torch.einsum('bij,bjc->bic', G * causal, w)
    A = torch.cumsum(ld, dim=1)                          # [BH,T,nc] cumulative log-decay
    s = torch.einsum('bij,bjc->bic', G * causal, w * torch.exp(-A))
    return torch.exp(A) * s


@input_guard
def _chunk_rola_impl(q, k, v, r, w, g=None, norm='kappa', kappa=None, scale=None, eps=1e-5,
                     output_final_state=False):
    """Routed RoLA (shared-gram) readout with built-in normalization — the TORCH NAIVE: the pure-torch,
    device-agnostic chunked reference on EXPLICIT precomputed gates (the ground truth). The Triton
    precomputed-gate kernels were retired in the #44 convergence (production routes through the in-kernel
    `chunk_rola_routed`); this stays as the explicit-gate reference the routed op + tests validate against,
    and as the `chunk_rola` public entry point (eager torch on every device; torch.compile-able).

    The chunked pass ALWAYS starts from a zero recurrent state — there is no `initial_state` ingestion
    (#34). The bidirectional handoff is one-directional: `output_final_state` IS honored (a chunked prefill
    emits the recurrent state for `fused_recurrent_rola` to seed via ITS `initial_state`), but the input
    counterpart is NOT accepted — `chunk_rola` RAISES on a non-None `initial_state` (the continuation-decode
    path is `fused_recurrent_rola`).

    Args:
        q, k:  φ-mapped queries/keys [B, T, H, K] (the feature map φ stays in the caller — the
               reference is φ-agnostic, seeing only the content gram G=φ(q)φ(k)ᵀ).
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

    # Torch reference: fp32 throughout (device-agnostic; no Triton, no kernel dtype contract). The
    # shared-gram eager core (`_rola_chunk_core`) and the per-state den pre-pass (`_perstate_den_torch`)
    # ARE the explicit-gate math the in-kernel routed op reproduces — this is the ground truth, fp32-exact.
    qf, kf, vf, wf = fold(q).float() * scale, fold(k).float(), fold(v).float(), fold(w).float()
    gf = fold(g).float() if g is not None else None
    if gf is not None:
        gf = _floor_ld(gf)   # enforce the fp32-safe decay floor once (#33), matching every GLA decay site

    if norm == 'raw':
        out = unfold(_rola_chunk_core(qf, kf, vf, wf, fold(r).float(), gf, chunk_size)).to(v.dtype)
        if not output_final_state:
            return out
        # raw carries NO per-state denominator, so the emitted state is [N, H*nc, K, V] (no +1 den
        # column) — exactly the layout fused_recurrent_rola(norm='raw') ingests and continues.
        return out, _final_state(kf, vf, wf, gf, B, H, raw=True)

    # global / per_state / kappa: per-state den pre-pass → rescale read gates → numerator-only
    # readout → divide by the reconstructed global den Σ_c r̃ᶜ·dᶜ.
    d = _perstate_den_torch(qf, kf, wf, gf, chunk_size, eps)
    rf32 = fold(r).float()
    if norm == 'kappa':
        rf32 = rf32 * (d + eps).pow(-fold(kappa).float())
    elif norm == 'per_state':
        rf32 = rf32 / (d + eps)
    num = _rola_chunk_core(qf, kf, vf, wf, rf32, gf, chunk_size)
    den = (rf32 * d).sum(-1, keepdim=True)
    out = unfold(num.float() / (den + eps)).to(v.dtype)
    if not output_final_state:
        return out
    return out, _final_state(kf, vf, wf, gf, B, H)


def _final_state(kf, vf, wf, gf, B, H, raw=False):
    """Final recurrent state of the chunked pass: `stateᶜ = Σ_t [e^{G_T-G_t}·]wᵗᶜ·kf_t⊗[vf_t;1]`,
    shaped `[N, H*nc, K, V+1]` (the `+1` ones-column is the per-state denominator) — byte-compatible
    with `fused_recurrent_rola`'s state so a chunked prefill hands off to recurrent decode. O(L), no
    kernel; the backward through a carried state is not provided (decode is inference).

    `raw=True` emits a [N, H*nc, K, V] state with NO den column: the raw readout is un-normalized
    (no per-state denominator is ever read), so the +1 ones-column is entirely absent — matching the
    raw decode kernel's [*,K,V] state layout (uses_v_plus_one=False)."""
    # raw: bare value tile [BH,T,V] (no ones-column). normalized: augment with the den ones-column.
    v1 = vf.float() if raw else torch.cat([vf, torch.ones_like(vf[..., :1])], -1).float()  # [BH,T,V(+1)]
    wgt = wf.float()                                                    # [BH,T,nc]
    if gf is not None:                                                  # GLA: token t decays by Σ_{t'>t} g
        # Apply the SAME GLA decay floor (`_floor_ld`) every chunked GLA decay site uses (readout/den/
        # routed). Without it the emitted final state would decay at a faster rate than the chunked
        # prefill it must hand off to (`test_recurrent_handoff`) — a prefill→decode decay-rate gap.
        G = _floor_ld(gf).float().cumsum(1)
        wgt = wgt * (G[:, -1:, :] - G).exp()
    state = torch.einsum('btc,btd,bte->bcde', wgt, kf.float(), v1)      # [BH, nc, K, V+1]
    return state.view(B, H * state.shape[1], state.shape[2], state.shape[3])


# ----------------------------------------------------------------------------
# `chunk_rola` — the explicit-precomputed-gate public entry point, now the PURE-TORCH NAIVE reference
# (`_chunk_rola_impl`): the precomputed-gate Triton kernels were retired in the #44 convergence (the
# production path is the in-kernel `chunk_rola_routed` + `fused_recurrent_rola` decode). This stays as
# the explicit-gate ground truth the routed op and the tests validate against. torch.compile wraps the
# pure-torch impl on CUDA by default (a trivial inductor fuse over the chunked einsums — no Triton
# autotuner to trace, so Dynamo stays fullgraph); ROLA_NO_COMPILE=1 forces eager. Compile is skipped on
# CPU and whenever Dynamo is already tracing (avoid nested-compile recursion).
_ROLA_NO_COMPILE = os.environ.get('ROLA_NO_COMPILE', '0') not in ('0', '', 'false', 'False')
_chunk_rola_compiled = None


def chunk_rola(q, k, v, r, w, g=None, norm='kappa', kappa=None, scale=None, eps=1e-5,
               initial_state=None, output_final_state=False):
    """Explicit-precomputed-gate RoLA readout — the PURE-TORCH naive reference (see `_chunk_rola_impl`).
    torch.compile is the DEFAULT on CUDA (lazily compiled on first call over the pure-torch chunked impl;
    no Triton, so Dynamo stays fullgraph). Set ROLA_NO_COMPILE=1 to force eager. Compile is skipped on CPU
    and whenever Dynamo is already tracing (avoid nested-compile recursion).

    `initial_state` is NOT ingested by the chunked pass (#34): seeding the inter scan from a carried
    state is decode-only territory (the chunked readout starts from zero state and provides no backward
    through a carried state). The param is kept only to RAISE loudly on a non-None value — rather than
    silently dropping it and returning wrong results — so a caller meaning to continue from a prefilled
    state is told to use `fused_recurrent_rola` (which DOES ingest `initial_state`)."""
    global _chunk_rola_compiled
    if initial_state is not None:
        raise NotImplementedError(
            "chunk_rola does not ingest an initial_state — the chunked pass always starts from a zero "
            "recurrent state (#34). output_final_state IS honored (a chunked prefill emits the state), "
            "but to CONTINUE from a prefilled state use fused_recurrent_rola(..., initial_state=...), "
            "which carries it. Passing initial_state here would have been silently ignored."
        )
    _guard_ld(g)   # eager floor guard (raises out-of-range) BEFORE the compiled region — no graph break (#33)
    kw = dict(g=g, norm=norm, kappa=kappa, scale=scale, eps=eps,
              output_final_state=output_final_state)
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


def _tree_gates_torch(hf, Wr, Ww, D, b, H, b_r=None, b_w=None):
    """Build the EXPLICIT folded [BH,T,nc] read/write gates from (h, Wr, Ww) the way the tree
    factorizes — r[...,leaf] = Π_lvl softmax(h·Wr[head,lvl] + b_r[head,lvl])[..., digit_lvl(leaf)]. The
    router is PER-HEAD (Wr,Ww ∈ [H,D,d_model,b]; bias b_r/b_w ∈ [H,D,b]): each head folds with its OWN
    weights, exactly the per-head kernel. This is the MATERIALIZED path the routed kernel avoids; used
    here only for (a) the normalized-norm den pre-pass (CPU) and (b) the flat-equivalence reference.
    `hf` is the [BH,T,d_model] fold (BH=(B,H)); H lets us route each head independently."""
    nc = b ** D
    BH, T, dm = hf.shape
    hr = hf.view(BH // H, H, T, dm)                                   # [B,H,T,dm]

    def _logit(W, bias, i):
        # per-head: h[b,head]·W[head,i] -> [B,H,T,b], refold to [BH,T,b].
        z = torch.einsum('bhtd,hdc->bhtc', hr, W[:, i].to(hf.dtype))
        if bias is not None:
            z = z + bias[:, i].to(hf.dtype)[None, :, None, :]
        return z.reshape(BH, T, -1)
    fr = torch.stack([torch.softmax(_logit(Wr, b_r, i), dim=-1) for i in range(D)], 0)  # [D,BH,T,b]
    fw = torch.stack([torch.softmax(_logit(Ww, b_w, i), dim=-1) for i in range(D)], 0)
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


def _ld_from_Wg_torch(hf, wf, Wg, H):
    """Materialize the per-state log-decay ld[BH,L,nc] from a per-head decay weight Wg:[H,d_model] + the
    explicit write gate wf:[BH,L,nc], the layer's `_log_decay` formula — for the CPU reference path ONLY
    (the CUDA path computes ld IN-KERNEL, never materializing it). #45."""
    BH, T, dm = hf.shape
    hr = hf.float().view(BH // H, H, T, dm)
    alpha = torch.sigmoid(torch.einsum('bhtd,hd->bht', hr, Wg.float())).reshape(BH, T, 1)
    ld = (1.0 - wf.float() * (1.0 - alpha)).clamp(min=1e-8).log()
    return ld.clamp(min=_GLA_FLOOR)


def _rola_routed_readout(qf, kf, vf, hf, Wr, Ww, D, b, chunk_size, b_r=None, b_w=None, Wg=None):
    """Folded tree-routed numerator-only readout. CUDA → in-kernel routed Triton kernels (gates AND the
    GLA decay ld never materialized); else → eager core on explicit gates (CPU reference path). Optional
    bias b_r/b_w. Optional per-head decay weight Wg:[H,d_model] (GLA) → the decayed routed readout; Wg=None
    is RLA."""
    if qf.is_cuda:
        if Wg is not None:
            return rola_gla_routed_triton(qf, kf, vf, hf, Wr, Ww, Wg, D, b, chunk=chunk_size,
                                          b_r=b_r, b_w=b_w)
        return rola_rla_routed_triton(qf, kf, vf, hf, Wr, Ww, D, b, chunk=chunk_size, b_r=b_r, b_w=b_w)
    r, w = _tree_gates_torch(hf, Wr, Ww, D, b, Wr.shape[0], b_r=b_r, b_w=b_w)
    ld = _ld_from_Wg_torch(hf, w, Wg, Wr.shape[0]) if Wg is not None else None
    return _rola_chunk_core(qf, kf, vf, w, r, ld, chunk_size)


@input_guard
def chunk_rola_routed(q, k, v, h, Wr, Ww, D, b, norm='kappa', kappa=None, scale=None, eps=1e-5,
                      b_r=None, b_w=None, Wg=None):
    """TREE-ROUTED RoLA (in-kernel routing) readout with built-in normalization. DIFFERENTIABLE
    end-to-end. ALL norms (incl. the production 'kappa'/'per_state') are fully fused — the [L,nc]
    gates, the per-state den d, the rescaled read gate r̃, AND (GLA) the per-state log-decay ld are
    NEVER materialized (see header).

    Args:
        q, k:  φ-mapped queries/keys [B, T, H, K].
        v:     values [B, T, H, V].
        h:     pre-routing hidden state [B, T, H, d_model] (the router lives in the kernel).
        Wr,Ww: PER-HEAD read/write router weights [H, D, d_model, b]  (b^D = nc). Each head h routes
               with its OWN [D,d_model,b] tree weights (RoLA's per-head routing — the kernel indexes
               head = (B·H fold-row) % H). d_model is the per-head router input width.
        D, b:  tree depth and branching (flat: D=1,b=nc; square: D=2,b=√nc; tree: b=2).
        norm:  'raw' | 'global' | 'per_state' | 'kappa'.
        kappa: per-token exponent [B,T,H,1] (required for norm='kappa').
        scale: query scale (default 1/sqrt(K)).
        b_r,b_w: OPTIONAL PER-HEAD routing bias [H, D, b] — the affine term of softmax(h·W + b), giving
                 the routing a non-uniform prior (default None = uniform start, backward-compatible).
        Wg:    OPTIONAL PER-HEAD decay weight [H, d_model] for the scalar-gated (GLA) variant — the layer's
               `w_g.weight`. None = RLA (byte-identical to the no-Wg path). When given, the per-state
               log-decay ld[t,c] = clamp(log(1 - w[t,c]·(1-sigmoid(h·Wg[head]))), _GLA_FLOOR) is computed
               IN-KERNEL (never a [L,nc] ld buffer — the GLA saved-activation win, #45); the decayed scan
               uses rt=r·e^a, w_end=w·e^{Λ-a} and a per-state per-chunk e^Λ state carry, with the per-state
               den d itself decayed.
    Returns:
        Normalized readout [B, T, H, V] ('raw' returns the un-normalized numerator). The [L,nc] gates
        AND the GLA per-state log-decay are NEVER materialized in the readout's routing gram.
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
    # Wg:[H,d_model] stays fp32 (the in-kernel decay exp is precision-sensitive). It is per-head (NOT folded
    # over the BH batch) — the kernel indexes head = fold-row % H, exactly like Wr/Ww. #45.
    Wgf = Wg.float() if Wg is not None else None

    if norm == 'raw':
        return unfold(_rola_routed_readout(qf, kf, vf, hf, Wr, Ww, D, b, chunk_size,
                                           b_r=b_r, b_w=b_w, Wg=Wgf)).to(v.dtype)

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
                                         b_r=b_r, b_w=b_w, Wg=Wgf)
        return unfold(num.float() / (den.float() + eps)).to(v.dtype)

    # CPU reference path (qf not on CUDA) for all normalized norms: the per-state den pre-pass on
    # explicit gates + the eager-core numerator. ALL CUDA normalized norms (incl. 'global') route through
    # the fused in-kernel den path above, so `_tree_gates_torch` is never on the production CUDA path.
    # The bias is threaded here too (out-of-place gates) so the fallback honors softmax(h·W+b).
    rf, wf = _tree_gates_torch(hf, Wr, Ww, D, b, H, b_r=b_r, b_w=b_w)
    rf, wf = rf.to(compute_dtype), wf.to(compute_dtype)
    gf = _ld_from_Wg_torch(hf, wf, Wgf, H) if Wgf is not None else None   # CPU ref: build ld from Wg (#45)
    d = _perstate_den_torch(qf, kf, wf, gf, chunk_size, eps)
    if norm == 'kappa':
        rf_scaled = (rf * (d + eps).pow(-fold(kappa).to(d.dtype))).to(compute_dtype)
    elif norm == 'per_state':
        rf_scaled = (rf / (d + eps)).to(compute_dtype)
    else:  # global
        rf_scaled = rf
    num = _rola_chunk_core(qf, kf, vf, wf, rf_scaled, gf, chunk_size)
    den = (rf_scaled * d).sum(-1, keepdim=True)
    return unfold(num / (den + eps)).to(v.dtype)
