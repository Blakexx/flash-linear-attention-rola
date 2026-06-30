# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# RoLA routing — Triton kernels (additive extension of simple_gla).
#
# Routed linear attention shares the content gram G=qk^T across `nc` states and modulates it by a
# routing gram R=Σ_c r_i^c w_j^c (+ optional per-state scalar decay). The Triton kernels here compute the
# *un-normalized* routed readout O = (G∘R∘causal) @ v — the FLA convention; the global denominator is
# reconstructed by the caller as Σ_c r̃ᶜ·dᶜ from a per-state den pre-pass. Tiled
# over state-blocks (BG states/program) so only this block's slice of the Kronecker state lives in
# SRAM -> scales to any nc. The content gram is formed ONCE per chunk (the FLOP win), never
# materializing the L x nc product nor replicating q/k.
#
# This file holds the production IN-KERNEL TREE-ROUTING kernels (RLA + GLA, fwd+bwd; the routing gram is
# built from the hidden state h + per-head router weights, so the [L,nc] gates are never materialized)
# and the private pure-torch CPU/reference helpers. The readout is numerator-only
# (width dv, BV=next_pow2(dv)); there is no ones-column augmentation — the denominator is a separate
# per-state pre-pass.

import functools
import math
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from fla_rola.ops.rola.routed_bwd_kernels import (  # production tree-routed backward kernels (the in-kernel router-grad fold)
    _build_alpha,  # in-kernel per-state log-decay helpers (#45): alpha=sigmoid(h·Wg), ld=clamp(log(1-w(1-alpha)))
    _decay_factors,  # re-anchored GLA decay factors (ea TRUE; ea_g/ena_g anchored intra-gram pair — fp32-overflow kill)
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
    Backend,
    autocast_custom_bwd,
    autocast_custom_fwd,
    autotune_cache_kwargs,
    get_all_max_shared_mem,
    input_guard,
)

# SMEM fitting (the one structural knob the autotuner can't own — the host-side Sb/dSa allocations + the
# launch grid are sized from BT BEFORE any kernel compiles) is DERIVED from the device's REAL max dynamic
# shared memory, NOT a magic byte constant. `get_all_max_shared_mem()` returns the per-device hard ceiling
# Triton enforces (OutOfResources above it): 101376 on sm86/sm89 (RTX 30xx/Ada), 166912 on A100, 232448 on
# H100. `_smem_budget()` keeps `_SMEM_SAFETY` of the SMALLEST device (size to the tightest GPU so no GPU
# OOMs) as the usable budget; every chunk/tile cap below sizes its analytic fp32-tile footprint to fit it.
# This REPLACES the old binary `check_shared_mem('ada')` gate (chunk = 64 if ada else 16): the derive is
# GRADED — a 99KB card lands on the full chunk, a 64KB card steps it down, uniformly via one formula (no
# per-dqk branch). The INNER blocks (num_warps / num_stages / BD) stay autotune knobs Triton prunes via
# OutOfResources.
#
# num_stages is NOT a budget divisor: these kappa chunk/grad kernels all run num_stages=1, so the per-tile
# footprint IS the peak (a software pipeline at stages>=2 LOWERS peak via buffer reuse, so a stages=1
# budget is the CONSERVATIVE bound — never an under-count). The analytic footprints below were calibrated
# against Triton's reported `.metadata.shared`: at the design point (dqk16,dv64,nc256,tree) the fwd chunk
# kernel measures 43KB and the bwd read kernel 61KB at chunk=64 (both < 99KB), and the footprint is
# dqk-INVARIANT (the kernels loop dqk in BK-blocks), so the derived chunk holds across the full dqk sweep.
_SMEM_SAFETY = 0.8           # fraction of the real device SMEM the host-side tile model is allowed to use
#                              (headroom for static/driver SMEM + the tl.dot operand staging the model omits)


@functools.cache
def _device_smem():
    """Real per-SM max DYNAMIC shared-memory bytes of the active CUDA device(s) — the hard ceiling Triton
    enforces. Takes the SMALLEST across visible devices (size to the tightest GPU). Falls back to the
    ADA class (101376) off-GPU (CPU/meta) so the host-side tile sizing stays deterministic."""
    vals = [v for v in get_all_max_shared_mem() if isinstance(v, int) and v > 0]
    return min(vals) if vals else Backend.ADA.value


def _smem_budget():
    return int(_device_smem() * _SMEM_SAFETY)


_WARPS = (2, 4, 8)
_STAGES = (1, 2, 3)
_AT_CFGS = [triton.Config({}, num_warps=w, num_stages=s) for w in _WARPS for s in _STAGES]
# The routed forward inter-scan is SHARED by RLA (USE_G=False) and GLA (USE_G=True) at the SAME
# (dqk,dv,nc); without USE_G in the key the autotune config-cache collides → RLA and GLA reuse each
# other's tuned warps/stages/BV. USE_G is a constexpr (so it specializes the compile regardless), but
# it must also gate config SELECTION so each variant tunes its own (the GLA decay-replay has a heavier
# SMEM profile than RLA, so the best config differs). Perf-only; no correctness change. (#22)
# BT/BK/BB/BG/BD are INCLUDED because they change the compiled tile footprint. A chunk=32 structural warmup
# or a different state-block width can otherwise cache a scan config that is over-SMEM when replayed at a
# larger chunk/state tile on sm86.
# nc is EXCLUDED on purpose: it only sets the host-side grid trip count `NB = cdiv(nc, BG)` (see ~L494)
# — the per-program state block is a fixed BG-wide tile, so nc changes NEITHER a kernel constexpr tile
# NOR per-program SMEM. The best warps/stages/BV is therefore nc-INVARIANT, and keying on nc forced a
# needless full re-tune at every states-per-head in the scaling sweep. Perf-only (config REUSE across
# nc); correctness is unaffected — the kernel still specializes on its real constexprs.
_SCAN_KEY = ['dqk', 'dv', 'BT', 'BK', 'BB', 'BG', 'BD', 'USE_G']
# `_CHUNK`/`_CHUNK_FWD` are the chunk-size CEILING — the largest BT the SMEM derive may pick. 64 is the
# GLA fp32-overflow ceiling (the chunked-decay gram e^a·e^{-a}; `_decay_factors` re-anchors each factor to
# the per-state midpoint, doubling the fp32-safe span to BT·|FLOOR|≲177, so 64·2.5=160<177 fits — the
# binding NUMERICAL limit, not a SMEM one). RLA (`_CHUNK_FWD`) has no such overflow limit; it is held to the
# same 64 because the SMEM derive (`_fit_chunk`) caps there anyway on a 99KB card and the chunked readout is
# chunk-size invariant. The ACTUAL per-device chunk is `_fit_chunk(ceiling, BK, BC)` (≤ ceiling), graded by
# `_smem_budget()` — there is no longer a separate `_KAPPA_BWD_CHUNK`: forward AND backward fit the SAME
# dominant [BT, BC·BK] fp32 chunk-tile against the SAME budget, so the cap is one uniform formula.
_CHUNK_FWD = 64                          # RLA chunk ceiling (SMEM-bound; the derive caps under it)
_CHUNK = 64                              # GLA chunk ceiling (fp32-overflow bound: 64·2.5=160 < 177)
_DTYPE_BYTES = 4                         # the kappa chunk/grad kernels stage their dominant tiles in fp32
# The state-slice region ([BC·BK, BV]) is the densest co-resident set in the fused kappa kernels: a READ
# copy + a WRITE copy of the slice plus the tl.dot operand staging — ~4 live fp32 tiles. The chunk-tile
# region ([BT, BC·BK]) has ONE dominant live copy (its model UPPER-BOUNDS the measured peak — 64KB model
# vs 43/61KB real at chunk=64 for the forward/read kernels). The kappa backward-state kernel additionally
# carries multiple BT×BT / BT×BC live regions and measures above the sm86 hard ceiling at BT=64, so its fit
# uses a calibrated BT-quadratic term below. These structural co-residence counts (read from the kernel
# source, NOT tuned numbers) divide the device budget per region. On a 99KB card: state-slice budget ~20KB
# → BK=BV=16 (the tested envelope; the >16 paths stay assert-guarded, see below), while the stricter
# kappa-backward BT model derives chunk 32.
_STATE_SLICE_COPIES = 4
_ROUTER_BD_COPIES = 2                     # router decay build: [BD,BB] weight tile + hc[BT,BD] hidden tile
_KAPPA_BWD_STATE_BT2_BYTES = 10            # measured missing live-set term:
                                           # 1024*64 + 10*64^2 = 106496 bytes on sm86


def _fit_chunk(want, row_bytes, bt2_bytes=0):
    """Largest power-of-2 chunk in [16, want] whose dominant fp32 tiles (which scale linearly with BT) fit
    the device SMEM budget. This sizes BT — the host-side knob the autotuner can't own (it sets the
    Sval/snapshot allocations + the launch grid before any kernel compiles). `row_bytes` is the kernel's
    peak SMEM PER chunk-row: the summed width (bytes) of its co-resident fp32 tiles that scale with BT,
    read from the kernel source and calibrated against Triton's `.metadata.shared`. `bt2_bytes` optionally
    models BT-quadratic live regions (BT×BT grams / adjoints) that are not captured by a row-linear tile
    width; keep it zero for kernels whose measured peak is row-linear. Floors at 16 (tl.dot gram dim ≥ 16).
    A bigger tile (large BK_full at frontier dqk) ⇒ a bigger row_bytes ⇒ a smaller chunk — the derive
    NATURALLY steps the chunk down at large dqk and up at small dqk, one formula, no per-dqk branch. The
    un-looped RLA-raw backward intra kernel uses BK_full → its row_bytes scales with dqk → it steps down
    (the fit-critical path: at dqk=128 chunk=64 needs 112KB > 99KB, so it derives down to a fitting chunk).
    The kappa backward-state path adds the calibrated BT² term so sm86 derives BT=32 instead of launching
    the measured 106KB BT=64 kernel against a 101KB hard limit."""
    budget = _smem_budget()
    chunk = max(16, want)
    while chunk > 16 and row_bytes * chunk + bt2_bytes * chunk * chunk > budget:
        chunk //= 2
    return chunk


# Per-chunk-row fp32 SMEM footprints (bytes) of the BT-scaling kernels, calibrated to `.metadata.shared`:
#   * looped kappa fwd/read: the dominant [BT, BC·BK] read/write tile (BK=16 looped, BC=16) → BC·BK·4. The
#     kappa backward-state kernel also carries enough BT×BT / BT×BC live state to overflow sm86 at BT=64,
#     so `_kappa_fit_chunk` adds `_KAPPA_BWD_STATE_BT2_BYTES` and derives BT=32 on a 99KB card.
#   * un-looped RLA-raw backward `_routed_bwd_intra`: 3 fp32 [BT, BK_full] tiles (q, k, dq/dk accumulator)
#     + 2 fp32 [BT, BVO] tiles (v, do/dv) co-resident → (3·BK_full + 2·BVO)·4. EXACT vs the measured peak
#     (chunk·(12·BK_full + 8·BVO): dqk16/dv64 704·BT, dqk128/dv64 2048·BT — verified across the sweep).
def _kappa_row_bytes(BK, BC=16):
    return BC * BK * _DTYPE_BYTES

def _routed_inter_state_row_bytes(BK, BC=16):
    # `_bwd_inter_state_kernel` keeps both wk[BT,BC*BK] and N[BT,BC*BK] live in fp32.
    return 2 * BC * BK * _DTYPE_BYTES

def _intra_row_bytes(BK_full, BVO):
    return (3 * BK_full + 2 * BVO) * _DTYPE_BYTES

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
# The state-slice budget is DERIVED: `_smem_budget() // _STATE_SLICE_COPIES` — the device's real SMEM share
# for ONE [BC·BK, BV] fp32 tile, given _STATE_SLICE_COPIES (~4) live co-resident in these kernels (read +
# write copies + tl.dot operand staging). On a 99KB card this is ~20KB ≈ the old hand-tuned 24KB constant.
#
# CURRENT-BUDGET INVARIANT: at this derived budget BOTH helpers ALWAYS return 16 for every shape the
# kernels are tested on (dqk,dv,chunk all ≤512 — verified across the full sweep on a 99KB card). The
# BK=32/BV=32 branches are therefore UNTESTED. On a LARGER-SMEM device (A100 166KB → budget ~33KB,
# H100 232KB → ~46KB) the cap CAN return 32 — which (by design) trips the asserts below, forcing the
# re-validation of the untested tile path BEFORE it silently activates. Re-validate the fused kappa
# fwd/bwd bit-faithfulness (test_kappa_routed_*), then relax the assert. The asserts make a >16 result LOUD.
def _kappa_bk_cap(dqk, dv, chunk, bc=16):
    bv = _bv_cap(dv)
    want = min(64, max(16, triton.next_power_of_2(dqk)))
    budget = _smem_budget() // _STATE_SLICE_COPIES   # device SMEM share for ONE [BC*BK,max(BV,BT)] fp32 tile
    span = bc * max(bv, chunk) * 4
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
    budget = _smem_budget() // _STATE_SLICE_COPIES   # device SMEM share for ONE [BC*BK,BV] fp32 tile
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


def _largest_pow2_leq(x):
    return 1 << (max(1, int(x)).bit_length() - 1)


def _router_bd_budget():
    """fp32 ELEMENTS for the in-kernel router decay build's [BD, BB] weight tile, DERIVED from the device
    SMEM share for the router region (`_ROUTER_BD_COPIES` co-resident tiles: [BD,BB] + hc[BT,BD]). On a
    99KB card ≈ 10K elements (~40KB) — the same BD selection the old hand-tuned 8192-element budget gave."""
    return max(16, (_smem_budget() // _ROUTER_BD_COPIES) // _DTYPE_BYTES)


def _router_bd_max():
    """Upper BD cap, DERIVED so the hc[BT, BD] fp32 hidden tile of the decay build fits the router SMEM
    share at the largest chunk (_CHUNK). With chunk now SMEM-derived up to 64, this is the binding tile —
    it caps BD below the [BD,BB]-budget's pick (e.g. 128 on a 99KB card at BT=64), NDM tiling the rest."""
    return max(16, _largest_pow2_leq((_smem_budget() // _ROUTER_BD_COPIES) // (_CHUNK * _DTYPE_BYTES)))


def _router_bd_ndm(d_model, b):
    """BD (the d_model tile width of the IN-KERNEL DECAY build — post-F2b only `_build_alpha` (the GLA
    decay h·Wg) loops `for dm in range(NDM)`; the routing factors are now precomputed logits loaded by
    `_build_rw_tile_logits`/`_build_factors`, NOT an in-kernel h·W contraction) RIGHT-SIZED to the LARGEST
    pow2 whose [BD, BB] decay-weight tile fits a safe SRAM fraction, with NDM=cdiv(d_model, BD) tiling the
    rest. For small d_model this returns BD=next_pow2(d_model), NDM=1 — byte-identical to the un-tiled build.

    F5: with the old BD=next_pow2(d_model) the whole decay weight was ONE block (NDM=1), so flat
    nc=64/256 at d_model=1024 (BD=1024) OOM'd. Capping BD by a FIXED byte budget keeps the tile in SRAM;
    crucially we pick the LARGEST BD that fits (small NDM) — NOT the smallest — so the constexpr `for dm in
    range(NDM)` unroll stays short (fast compile + good ILP). For the common small-b routings (BB=16) BD
    lands at the 256 cap (NDM=4 at d_model=1024); only true flat routing (BB=nc large) drives NDM up,
    because [BD,BB] is intrinsically wide there. `_build_alpha` NDM-loops, so the cap only splits the d_model
    decay reduction into NDM blocks (output unchanged to fp tolerance). (Pre-F2b this also sized the router
    h·W build; that contraction is now a cuBLAS GEMM in the layer, so BD/NDM gate only the decay now.)"""
    bd_budget = _router_bd_budget()
    BB = max(16, triton.next_power_of_2(b))
    bd_full = max(16, triton.next_power_of_2(d_model))
    bd_cap = triton.next_power_of_2(max(1, bd_budget // BB))
    while bd_cap > 16 and bd_cap * BB > bd_budget:
        bd_cap //= 2          # round DOWN to the largest pow2 keeping [BD,BB] within budget (floor 16)
    BD = max(16, min(bd_full, bd_cap, _router_bd_max()))
    return BD, triton.cdiv(d_model, BD)


def _prune_bv(configs, named_args, **kwargs):
    """Cap BV at next_pow2(d_v) and drop routed inter-scan tiles that cannot fit SMEM.

    The scan's dominant live fp32 tile is the readout dot product/reinterpretation `P/P3` with shape
    [BT, BG*BV]. At BT=64,BG=16,BV=32 this alone is 128 KiB, over the 99 KiB sm86/sm89 hard limit before
    Triton's dot staging and routing temporaries are counted. Pruning it host-side prevents autotune cache
    from ever selecting a launch shape that later replays as `OutOfResources`.
    """
    try:
        cap = _bv_cap(named_args['dv'])
    except Exception:
        return configs
    keep = [c for c in configs if c.kwargs.get('BV', 16) <= cap]
    try:
        BT = int(named_args['BT'])
        BG = int(named_args['BG'])
        budget = _smem_budget()
        keep_fit = [
            c for c in keep
            if BT * BG * int(c.kwargs.get('BV', 16)) * _DTYPE_BYTES <= budget
        ]
    except Exception:
        keep_fit = keep
    return keep_fit or [c for c in keep if c.kwargs.get('BV', 16) == 16] \
        or [c for c in configs if c.kwargs.get('BV', 16) == 16] or configs


# ============================================================================
# In-kernel TREE-ROUTING forward (RLA). The matching BACKWARD lives just below (`_RoLARoutedFn`).
#
# The production tree-routing forward (the validated prototype since promoted; its backward kernels now
# live in `routed_bwd_kernels.py`): instead of taking PRECOMPUTED gates r,w ∈ [L,nc] and forming R = r·wᵀ,
# the routing gram is built IN-KERNEL from the PRECOMPUTED per-level logits lr,lw ∈ [BH,L,D,b] (F2b: the
# per-head h·Wr/h·Ww d_model-contraction is a cuBLAS GEMM in the layer — `_router_logits`/`_factor_logits`),
# never materializing the [L,nc] gates.
#
# This is a SOURCE SWAP, not a new pipeline: the inner machinery is the same shared-gram RLA forward
# structure (collapse-intra over state-blocks + NB-fused inter scan, the same
# autotune configs, the same BG state-block SMEM-tiling). The ONLY change is the block that produced the
# [BT,BG] routing tiles:  `tl.load(rg/wg)`  →  `_build_rw_tile_logits` (load the per-level logits, softmax,
# then gather via one-hot Sel maps and Hadamard-fold over levels — NO in-kernel h·W contraction). The
# reconstructed gate tiles live transiently in SRAM at the state-block width BG; the dominant [L,nc] gate
# tensor is never allocated.
#
# The factorization (proven in the prototype, validated <1e-2 vs autograd):
#   r[:, c] = Π_lvl softmax(lr[lvl])[:, digit_lvl(c)],   R = r·wᵀ = ⊙_lvl (fr_lvl·fw_lvlᵀ)
# with the per-level [BT,b] softmax factors fr,fw (of the precomputed logits lr=h·Wr+b_r) gathered to the
# nc-leaf block by Sel[lvl][b, nc].
#
# BACKWARD is BELOW: `_RoLARoutedFn` wraps this forward — backward drives the validated fold kernels
# (`_bwd_intra_kernel`, `_bwd_inter_*`, `_fold_*` from `routed_bwd_kernels.py`) at the PRODUCTION
# state-block width (BC=BG), and folds the transient [BT,nc] gate-grads into the per-level LOGIT grads
# dlr/dlw in-kernel (the [L,nc] grads never materialize); dWr/dWw/d_h then flow through the cuBLAS
# GEMM-backward (autograd). The forward builds the SAME `sel` map the bwd needs; backward rebuilds the SAME
# [BT,BG] factor reconstruction (`_build_rw_tile_logits` ≡ `_build_factors`) before folding logit grads.
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


def _router_logits(h, Wr, Ww, b_r, b_w, H, build_r=True):
    """F2b: the per-level routing logits lr,lw ∈ [BH, L, D, b] = (h·Wr+b_r, h·Ww+b_w) — the cuBLAS GEMM
    that REPLACES the in-kernel h·W d_model-contraction (the design-point kernel's 72%). h is the folded
    [BH, L, d_model] (BH=(B,H)); Wr,Ww ∈ [H,D,d_model,b] per-head; bias ∈ [H,D,b] or None. Returns the
    per-head-folded logits the kernel loads (`_build_rw_tile_logits`) instead of recomputing — the routing
    bias is folded in HERE (softmax(h·W+b) input), so the kernel never sees the weights/bias. `build_r`
    False (write-only callers) returns lr=None. This is a torch einsum → cuBLAS, autograd-tracked when
    h/Wr/Ww require grad (F2b backward routes dWr/dWw/dh through this GEMM).
    For tree/square the logits are tiny ([L, 2·logₙc] / [L, 2√nc]); flat's are [L,1,nc] (=[L,nc], flat's
    inherent routing cost — a topology/config choice, not a kernel branch)."""
    BH, L, dm = h.shape
    Bb = BH // H
    hr = h.reshape(Bb, H, L, dm)

    def _lg(W, bias):
        z = torch.einsum('bhld,hkdc->bhlkc', hr, W.to(h.dtype))        # [B,H,L,D,b]
        if bias is not None:
            z = z + bias.to(h.dtype)[None, :, None]                    # [H,D,b] -> [1,H,1,D,b]
        return z.reshape(BH, L, W.shape[1], W.shape[3]).contiguous()   # [BH,L,D,b]
    lw = _lg(Ww, b_w)
    lr = _lg(Wr, b_r) if build_r else None
    return lr, lw


def _gates_from_factor_logits(logits, D, b):
    """Fold per-level factor logits [*, D, b] into the explicit leaf gates [*, nc] (the torch leaf-product
    the kernel's Sel-gather mirrors) — for the driver's CPU-side decay/Λ recompute (never the kernel's
    [L,nc]). Matches the layer's `_gates_from_logits`."""
    f = torch.softmax(logits.float(), dim=-1)                # [*, D, b]
    g = f[..., 0, :]                                         # [*, b]
    for i in range(1, D):
        g = (g.unsqueeze(-1) * f[..., i, :].unsqueeze(-2)).flatten(-2)
    return g                                                 # [*, nc]




@triton.jit
def _build_rw_tile_logits(lr_ptr, lw_ptr, sel_ptr, offs_c, cmask,
                          bn, rows, rmask, offs_bb, bmask,
                          slr_b, slr_l, slr_lvl, slr_bb, slw_b, slw_l, slw_lvl, slw_bb,
                          ssel_lvl, ssel_b, ssel_c,
                          D: tl.constexpr, BT: tl.constexpr, BB: tl.constexpr,
                          BG: tl.constexpr, BUILD_R: tl.constexpr = True):
    """F2b: build the [BT, BG] read/write routing tiles for ONE state-block (the BG-wide nc slice offs_c)
    from PRECOMPUTED per-level logits lr,lw ∈ [BH, L, D, b] — the per-head h·Wr/h·Ww d_model-contraction is
    now a cuBLAS GEMM in the layer (`_router_logits`/`_factor_logits`, full occupancy + tensor cores), so
    the kernel does ONLY softmax + the one-hot Sel gather + the Hadamard fold over levels (NO in-kernel h·W).
    The optional routing bias is ALREADY folded into the logits (the GEMM computes softmax-input h·W+b), so
    no bias arg here. The per-(token,level,branch) logits are indexed by bn (the BH fold-row, == pid; NO
    per-head offset — the logits tensor is already per-head). The [L,nc] gates are NEVER materialized — only
    this state-block's transient [BT,BG] tiles. BUILD_R (#37): when False skip the read factor (write-only
    callers), bit-identical write tile.

    The per-level softmax (stable max-subtraction), the Sel gather, and the Hadamard accumulation are
    exactly the proven shared-gram factor reconstruction (≡ the backward's `_build_factors`) — only the
    logit SOURCE is a load, not an in-kernel h·W contraction. lr/lw are loaded at their stored dtype and
    upcast to fp32 for the softmax."""
    r_tile = tl.full([BT, BG], 1.0, dtype=tl.float32)
    w_tile = tl.full([BT, BG], 1.0, dtype=tl.float32)
    neg = tl.full([BT, BB], float('-inf'), dtype=tl.float32)
    for lvl in range(D):
        lw = tl.load(lw_ptr + bn * slw_b + rows[:, None] * slw_l + lvl * slw_lvl + offs_bb[None, :] * slw_bb,
                     mask=rmask[:, None] & bmask[None, :], other=0.0).to(tl.float32)
        lw = tl.where(bmask[None, :], lw, neg)
        ew = tl.exp(lw - tl.max(lw, axis=1)[:, None])
        fw = ew / tl.sum(ew, axis=1)[:, None]   # [BT, BB] write-gate level factor
        sel = tl.load(sel_ptr + lvl * ssel_lvl + offs_bb[:, None] * ssel_b + offs_c[None, :] * ssel_c,
                      mask=bmask[:, None] & cmask[None, :], other=0.0)   # [BB, BG] one-hot
        w_tile *= tl.dot(fw, sel)
        if BUILD_R:
            lr = tl.load(lr_ptr + bn * slr_b + rows[:, None] * slr_l + lvl * slr_lvl + offs_bb[None, :] * slr_bb,
                         mask=rmask[:, None] & bmask[None, :], other=0.0).to(tl.float32)
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
def _rola_routed_fwd_intra(q_ptr, k_ptr, v_ptr, h_ptr, lr_ptr, lw_ptr, sel_ptr, wg_ptr, outa_ptr,
                           L, dqk, dv, nc, d_model, H, swg_head,
                           sq_b, sq_l, sq_d, sv_b, sv_l, sv_d,
                           sh_b, sh_l, sh_d, slo_b, slo_l, slo_lvl, slo_bb,
                           ssel_lvl, ssel_b, ssel_c, swg_d, soa_b, soa_l, soa_v,
                           D: tl.constexpr, bb_: tl.constexpr, BB: tl.constexpr,
                           BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                           BG: tl.constexpr, BD: tl.constexpr,
                           ND: tl.constexpr, NB: tl.constexpr, NDM: tl.constexpr,
                           USE_G: tl.constexpr, GLA_FLOOR: tl.constexpr):
    """TREE-ROUTED intra: the shared-gram collapse-intra structure — content gram G
    built ONCE, the full routing gram R accumulated over state-blocks IN-KERNEL, A=G⊙R⊙causal, o=A·v —
    with R's source being the PRECOMPUTED per-level logits lr,lw (F2b: the h·Wr/h·Ww d_model-contraction
    is a cuBLAS GEMM in the layer; the kernel softmax+gathers via `_build_rw_tile_logits`). nc collapses;
    output is [B,L,dv].
    USE_G (GLA, #30 V1): the per-block gram uses the DECAYED gates rt=rgc·e^a, wt=wgc·e^-a (a = intra-chunk
    cumsum of the per-state log-decay ld over this block's c-columns). The decay logit h·Wg STAYS in-kernel
    (`_build_alpha`, tiny [d_model]→1 per head). The
    nc-collapse still holds (each block's decayed [BT,BG]·[BG,BT] gram is still a [BT,BT] partial)."""
    b = tl.program_id(0)
    t = tl.program_id(1)
    # The routing logits lr/lw are already per-head ([BH,L,D,b], indexed by b=pid). Only the GLA decay
    # weight Wg:[H,d_model] needs the per-head offset (BH fold is (B,H) -> head = pid % H).
    _hd = b % H
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
        rgc, wgc = _build_rw_tile_logits(lr_ptr, lw_ptr, sel_ptr, offs_c, cmask,
                                         b, rows, rmask, offs_bb, bmask,
                                         slo_b, slo_l, slo_lvl, slo_bb, slo_b, slo_l, slo_lvl, slo_bb,
                                         ssel_lvl, ssel_b, ssel_c, D, BT, BB, BG)
        if USE_G:
            ldc = _ld_from_w(wgc, alpha, cmask, GLA_FLOOR)
            a = tl.cumsum(ldc, axis=0)
            _ea, ea_g, ena_g = _decay_factors(a)        # ANCHORED intra-gram pair (no e^{-a} overflow)
            rgc = rgc * ea_g
            wgc = wgc * ena_g
        R += tl.dot(rgc.to(tl.float32), tl.trans(wgc))
    vc = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                 mask=rmask[:, None] & (offs_v[None, :] < dv), other=0.0)
    causal = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
    # R's causal entries are finite; its (masked-out) anti-causal triangle holds +inf (e^{a_i-a_j}, i<j,
    # fp32-overflows for large BT). `tl.where` SELECTS the zero there → never inf·0=NaN; identical for finite R.
    A = G * tl.where(causal, R, 0.0)
    o = tl.dot(A.to(vc.dtype), vc)
    tl.store(outa_ptr + b*soa_b + rows[:, None]*soa_l + offs_v[None, :]*soa_v,
             o, mask=rmask[:, None] & (offs_v[None, :] < dv))


@triton.autotune(configs=_SCAN_CFGS, key=_SCAN_KEY, reset_to_zero=['outa_ptr'],  # _SCAN_KEY: +USE_G
                 prune_configs_by={'early_config_prune': _prune_bv}, **autotune_cache_kwargs)
@triton.jit
def _rola_routed_fwd_inter(q_ptr, k_ptr, v_ptr, h_ptr, lr_ptr, lw_ptr, sel_ptr, wg_ptr, outa_ptr,
                           L, dqk, dv: tl.constexpr, nc, d_model, H, swg_head,
                           sq_b, sq_l, sq_d, sv_b, sv_l, sv_d,
                           sh_b, sh_l, sh_d, slo_b, slo_l, slo_lvl, slo_bb,
                           ssel_lvl, ssel_b, ssel_c, swg_d, soa_b, soa_n, soa_l, soa_v,
                           D: tl.constexpr, bb_: tl.constexpr, BB: tl.constexpr,
                           BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                           BG: tl.constexpr, BD: tl.constexpr,
                           NCH: tl.constexpr, NDM: tl.constexpr,
                           USE_G: tl.constexpr, GLA_FLOOR: tl.constexpr):
    """TREE-ROUTED inter: the NB-fused inter scan structure — one
    (batch, state-block, FEATURE-block) carries Sd[BK, BG*BV] across chunks, o_inter atomic-accumulated
    into the shared [B,L,dv] buffer, value-OUTER over cdiv(dv,BV) — with the read/write gate source being
    the PRECOMPUTED per-level logits lr,lw via `_build_rw_tile_logits` (F2b: h·W is a cuBLAS GEMM in the
    layer). Same state scan, same SMEM bound (BK + value-tile BV), same autotune/reset_to_zero.
    USE_G (GLA, #30 V1): the carried state decays by decvec=e^Λ each chunk, the read uses rt=rgc·e^a, the
    write uses w_end=wgc·e^{Λ-a} (a = intra-chunk cumsum, Λ = chunk-total ld); the decay logit h·Wg stays
    in-kernel (`_build_alpha`)."""
    b = tl.program_id(0)
    sb = tl.program_id(1)
    d0 = tl.program_id(2)
    _hd = b % H                                  # per-head decay-weight slice (BH fold is (B,H))
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
            rgc, wgc = _build_rw_tile_logits(lr_ptr, lw_ptr, sel_ptr, offs_c, cmask,
                                             b, rows, rmask, offs_bb, bmask,
                                             slo_b, slo_l, slo_lvl, slo_bb, slo_b, slo_l, slo_lvl, slo_bb,
                                             ssel_lvl, ssel_b, ssel_c, D, BT, BB, BG)
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


def _routed_fwd_tiled(q, k, v, h, Wr, Ww, D, b, sel, chunk, BG, BK=64, b_r=None, b_w=None, Wg=None,
                      lr=None, lw=None):
    """D-tiled numerator-only tree-routed forward — the in-kernel-routed shared-gram readout. F2b: the
    per-level routing LOGITS lr,lw:[BH,L,D,b] (the cuBLAS GEMM of h·Wr+b_r / h·Ww+b_w) are precomputed —
    either passed in (the autograd Function path) or computed here from (Wr,Ww,b_r,b_w) (the plain/test
    path); the kernel only softmax+gathers. Returns [B,L,dv] at BV=next_pow2(dv); [L,nc] gates never
    allocated. Optional per-head decay weight Wg:[H,d_model] (GLA, #45) → USE_G decayed scan, the
    per-state log-decay ld computed IN-KERNEL from Wg + the write tile; Wg=None is RLA (USE_G=False)."""
    B, L, dqk = q.shape
    dv = v.shape[-1]
    d_model = h.shape[-1]
    nc = b ** D
    BV = max(16, triton.next_power_of_2(dv))
    BK = min(BK, max(16, triton.next_power_of_2(dqk)))
    BD = max(16, triton.next_power_of_2(d_model))
    BB = max(16, triton.next_power_of_2(b))
    use_g = Wg is not None
    if use_g:
        # GLA inter-scan carries several [BT, BG*BV]-scale fp32 regions (P/P3, WV, decay vectors, plus
        # dot staging). On 99KB-class GPUs the nominal BT=64 variant can compile above the hard SMEM
        # limit even when BV=16, so fit the chunk before autotune/cache selection can replay an illegal
        # launch. Chunking is output-invariant up to the same reduction-order tolerance tested below.
        chunk = _fit_chunk(chunk, 6 * BG * BV * _DTYPE_BYTES)
    ND = triton.cdiv(dqk, BK)
    NB = triton.cdiv(nc, BG)
    NCH = triton.cdiv(L, chunk)
    NDM = triton.cdiv(d_model, BD)
    if lr is None:
        # plain/test path: GEMM the logits here from the per-head router weights (bias folded in).
        q, k, v, h, Wr, Ww = [x.contiguous() for x in (q, k, v, h, Wr, Ww)]
        H = Wr.shape[0]
        lr, lw = _router_logits(h, Wr, Ww, b_r, b_w, H)
    else:
        q, k, v, h, lr, lw = [x.contiguous() for x in (q, k, v, h, lr, lw)]
        H = Wg.shape[0] if use_g else 1     # decay is per-head (Wg:[H,d_model]); RLA's stub head is unread
    Wg = (Wg.float().contiguous() if use_g
          else q.new_zeros(H, d_model))   # USE_G=False: Wg unread (no decay); pass a [H,d_model] stub
    swg = (Wg.stride(0), Wg.stride(1))               # (swg_head, swg_d) — head/d_model strides
    slo = (lw.stride(0), lw.stride(1), lw.stride(2), lw.stride(3))   # [BH,L,D,b] strides (lr==lw layout)
    out_intra = torch.zeros(B, L, BV, device=q.device, dtype=torch.float32)
    out_inter = torch.zeros(B, L, BV, device=q.device, dtype=torch.float32)
    so_a = (out_intra.stride(0), out_intra.stride(1), out_intra.stride(2))
    so_e = (out_inter.stride(0), 0, out_inter.stride(1), out_inter.stride(2))
    base = (q.stride(0), q.stride(1), q.stride(2), v.stride(0), v.stride(1), v.stride(2))
    route = (h.stride(0), h.stride(1), h.stride(2), *slo, sel.stride(0), sel.stride(1), sel.stride(2))
    _rola_routed_fwd_intra[(B, NCH)](q, k, v, h, lr, lw, sel, Wg, out_intra, L, dqk, dv, nc, d_model,
                                     H, swg[0], *base, *route, swg[1], *so_a,
                                     D=D, bb_=b, BB=BB, BT=chunk, BK=BK, BV=BV, BG=BG, BD=BD,
                                     ND=ND, NB=NB, NDM=NDM, USE_G=use_g, GLA_FLOOR=_GLA_FLOOR)
    _rola_routed_fwd_inter[(B, NB, ND)](q, k, v, h, lr, lw, sel, Wg, out_inter, L, dqk, dv, nc, d_model,
                                        H, swg[0], *base, *route, swg[1], *so_e,
                                        D=D, bb_=b, BB=BB, BT=chunk, BK=BK, BG=BG, BD=BD,
                                        NCH=NCH, NDM=NDM, USE_G=use_g, GLA_FLOOR=_GLA_FLOOR)
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
# trainable end-to-end. Given d_o, compute dq,dk,dv,d_h plus the per-level LOGIT grads dlr,dlw with the
# [L,nc] gate grads dr,dw NEVER materialized: they live only as transient [BT,BG] tiles (production
# state-block width), gathered through the SAME one-hot `Sel` map + the SAME factor reconstruction
# (`_build_factors`, loading the precomputed logits) the forward uses, and folded in-kernel (softmax
# jacobian → atomic dlr/dlw); dWr/dWw/d_h then flow through the cuBLAS GEMM-backward (F2b; autograd
# through `_router_logits`/`_factor_logits`).
#
# The fold math is the validated backward (routed_bwd_kernels.py: _bwd_intra_kernel,
# _bwd_inter_state_kernel, _bwd_inter_read_kernel, _fold_kernel — validated <1e-2 vs autograd for
# flat/square/tree). Those kernels are GENERIC over the nc-block width (their `BC` constexpr); the ONLY
# adaptation is to drive them at the PRODUCTION state-block width BC=BG and through this module's
# `_build_sel`, so the backward routes through byte-identical factor reconstruction to the production
# forward (`_build_factors` ≡ `_build_rw_tile_logits`). dr,dw stay transient per-chunk [B,chunk,nc] scratch
# tiles (gdr/gdw), OVERWRITTEN every chunk — never [L,nc].
#
# The per-chunk pre-state snapshots S_j ∈ [B,nc,dqk,dv] the reverse-scan needs are the RECURRENT state
# (NOT the gates), built by a dedicated in-kernel snapshot scan `_rola_routed_snap` (write gates built
# in-kernel via `_build_rw_tile_logits` from the precomputed write logits — gates never materialized in the
# snapshot pass either). The forward saves these snapshots; backward consumes them. The validated fold
# kernels are imported at module top (_routed_bwd_intra / _routed_bwd_inter_state / _routed_bwd_inter_read
# / _routed_bwd_fold), reused VERBATIM.
# ============================================================================


@triton.jit
def _rola_routed_snap_kernel(h_ptr, k_ptr, v_ptr, lw_ptr, sel_ptr, wg_ptr, s_ptr, snap_ptr,
                             L, d_model, dqk, dv, nc, H,
                             sh_b, sh_l, sh_d, sq_b, sq_l, sq_d, sv_b, sv_l, sv_d,
                             slo_b, slo_l, slo_lvl, slo_bb,
                             ssel_lvl, ssel_b, ssel_c, swg_head, swg_d, ss_b, ss_c, ss_k, ss_v,
                             snp_b, snp_n, snp_c, snp_k, snp_v,
                             D: tl.constexpr, b: tl.constexpr, BB: tl.constexpr,
                             BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BD: tl.constexpr,
                             BG: tl.constexpr, NCBLK: tl.constexpr, ND: tl.constexpr,
                             NCH: tl.constexpr, NDM: tl.constexpr,
                             USE_G: tl.constexpr, GLA_FLOOR: tl.constexpr):
    """Per-chunk PRE-STATE snapshot scan for the routed backward. One program per batch carries the flat
    Kronecker state S[nc,dqk,dv] across chunks; BEFORE each chunk's write it copies S into snap[:,chunk]
    (the state the reverse-scan reads). The WRITE gates are built from the PRECOMPUTED logits lw via
    `_build_rw_tile_logits` (write-only, BUILD_R=False; the [L,nc] gates are never materialized here
    either). Mirrors the proto chunk kernel's state update (S += Σ_t wₜᶜ kₜ⊗vₜ), at the BG state-block width.
    USE_G (GLA, #30 V1): the per-c state decays by e^{Λ_c} each chunk and the write is e^{Λ_c-a}-weighted
    (S[c] ← e^{Λ_c}S[c] + Σ w_end k⊗v) — exactly the decayed inter scan; the snapshot is still PRE-update."""
    pid_b = tl.program_id(0)
    _hd = pid_b % H                              # per-head decay-weight slice (BH fold is (B,H))
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
            _, w_tile = _build_rw_tile_logits(lw_ptr, lw_ptr, sel_ptr, cols, cmask,
                                              pid_b, rows, rmask, offs_bb, bmask,
                                              slo_b, slo_l, slo_lvl, slo_bb, slo_b, slo_l, slo_lvl, slo_bb,
                                              ssel_lvl, ssel_b, ssel_c,
                                              D, BT, BB, BG, BUILD_R=False)  # write-only: skip read
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


def _routed_snapshots(q, k, v, h, lw, D, b, sel, chunk, BG, H, Wg=None):
    """Build per-chunk pre-state snapshots [B, NCH, nc, dqk, dv] for the routed backward — the recurrent
    STATE (NOT the gates), via the in-kernel snapshot scan. Write gates built from the PRECOMPUTED write
    logits lw:[BH,L,D,b] (F2b — never [L,nc]; the bias is already folded into the logits).
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
    slo = (lw.stride(0), lw.stride(1), lw.stride(2), lw.stride(3))
    use_g = Wg is not None
    Wg = (Wg.float().contiguous() if use_g else q.new_zeros(H, d_model))
    swg = (Wg.stride(0), Wg.stride(1))
    S = torch.zeros(B, nc, dqk, dv, device=q.device, dtype=torch.float32)
    snap = torch.zeros(B, NCH, nc, dqk, dv, device=q.device, dtype=torch.float32)
    _rola_routed_snap_kernel[(B,)](
        h, k, v, lw, sel, Wg, S, snap,
        L, d_model, dqk, dv, nc, H,
        h.stride(0), h.stride(1), h.stride(2), q.stride(0), q.stride(1), q.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        *slo,
        sel.stride(0), sel.stride(1), sel.stride(2), swg[0], swg[1],
        S.stride(0), S.stride(1), S.stride(2), S.stride(3),
        snap.stride(0), snap.stride(1), snap.stride(2), snap.stride(3), snap.stride(4),
        D=D, b=b, BB=BB, BT=chunk, BK=BK, BV=BV, BD=BD, BG=BG, NCBLK=NCBLK, ND=ND, NCH=NCH, NDM=NDM,
        USE_G=use_g, GLA_FLOOR=_GLA_FLOOR, num_warps=4, num_stages=1)
    return snap


def _ld_chunk(h_c, lw_c, Wg, D, b, H):
    """Per-state log-decay ld[BH,len,nc] for a CHUNK slice (chunk-local, NEVER full [L,nc]) — the SAME
    in-kernel formula (`RoLA._log_decay`), in torch, for the driver's dS chunk-total Λ_c decay (#45). h_c is
    [BH,len,d_model] (BH=(B,H) fold); lw_c the chunk slice of the PRECOMPUTED write logits [BH,len,D,b]
    (F2b — bias already folded). alpha=sigmoid(h·Wg[head]) per-head; ld=clamp(log(clamp(1-w(1-alpha),
    1e-8)), _GLA_FLOOR), w the per-head WRITE gate from the logits. Returns fp32 [BH,len,nc]."""
    BH, T, dm = h_c.shape
    wf = _gates_from_factor_logits(lw_c, D, b)               # [BH,len,nc] from the write logits
    hr = h_c.float().view(BH // H, H, T, dm)
    alpha = torch.sigmoid(torch.einsum('bhtd,hd->bht', hr, Wg.float())).reshape(BH, T, 1)
    ld = (1.0 - wf * (1.0 - alpha)).clamp(min=1e-8).log()
    return ld.clamp(min=_GLA_FLOOR)


def _rola_rla_routed_bwd(q, k, v, h, lr, lw, do, D, b, chunk, BG, H, Wg=None):
    """Tree-routed RLA backward at the PRODUCTION state-block width BC=BG. F2b: the routing factors are
    rebuilt from the PRECOMPUTED logits lr,lw:[BH,L,D,b] and the router-grad fold EMITS per-level LOGIT
    grads dlr,dlw:[BH,L,D,b] (the dWr/dWw/dh/db d_model contraction is a cuBLAS GEMM-backward in torch).
    Drives the fold kernels: one intra launch (dq,dk,dv-intra + the dlr,dlw-intra fold) and a sequential
    reverse state-adjoint scan (state-update bwd → readout bwd → router-grad fold). The gate-grads dr,dw
    live only as transient [B,chunk,nc] scratch (gdr/gdw), OVERWRITTEN each chunk — never [L,nc]. Returns
    dq,dk,dv,d_h(decay-only),dlr,dlw (and dWg when GLA) — all fp32.

    USE_G (GLA, #45): optional per-head decay weight Wg:[H,d_model] → the decayed routed backward — the
    per-state log-decay ld is computed IN-KERNEL (Wg + the write gate, never a [L,nc] ld), the kernels use
    the DECAYED gates (rt=r·eᵃ, w_end=w·e^{Λ−a}), the running dS adjoint is decayed by e^Λ between the
    state-bwd and read-bwd halves, and a persistent gda[B,L,nc] buffer collects the per-token log-decay
    adjoints which the driver reverse-cumsums (intra-chunk) into a PER-CHUNK dld[B,chunk,nc] (never [L,nc]);
    the fold kernel splits dld → dWg + the decay's dh + the decay's write-gate grad (→ dlw). Wg=None is RLA."""
    use_g = Wg is not None
    B, L, d_model = h.shape
    dqk = q.shape[-1]
    dv = v.shape[-1]
    nc = b ** D
    BVO = max(16, triton.next_power_of_2(dv))   # full padded value width (intra bwd + buffer alloc)
    BD = max(16, triton.next_power_of_2(d_model))
    BB = max(16, triton.next_power_of_2(b))
    BC = BG                                  # PRODUCTION tile width (the adaptation; proto used fixed 16)
    BK_full = max(16, triton.next_power_of_2(dqk))
    BK = _kappa_bk_cap(dqk, dv, min(chunk, _CHUNK), BC)   # feature-tile (loop ND)
    BV = _kappa_bv_tile(dqk, dv, BC, BK)                 # value-tile (loop ND_V) so [BC*BK,BV] fits
    NCBLK = triton.cdiv(nc, BC)
    ND = triton.cdiv(dqk, BK)
    NDM = triton.cdiv(d_model, BD)
    NCH = triton.cdiv(L, chunk)
    # Router-grad fold accumulation in fp32 (FLA backward idiom): the deep-Hadamard softmax jacobian
    # (dfr/fr with D levels) is precision-sensitive, so the logits + all gram dots run with fp32 operands.
    q, k, v, h, lr, lw, do = (x.float().contiguous() for x in (q, k, v, h, lr, lw, do))
    slo = (lw.stride(0), lw.stride(1), lw.stride(2), lw.stride(3))
    # SMEM-derived: both the full-feature intra kernel and the looped inter-state kernel can bind BT. Intra
    # uses BK_full=next_pow2(dqk), so its footprint scales with dqk; inter-state keeps two [BT,BC*BK] fp32
    # tiles live (wk and N) and exceeds the sm86 hard ceiling at BT=64 even for dqk=dv=16. Choose the smaller
    # shared chunk so the snapshot pass, state-bwd, read-bwd, and fold all agree on the reverse-scan tiling.
    chunk = min(
        _fit_chunk(min(chunk, _CHUNK), _intra_row_bytes(BK_full, BVO)),
        _fit_chunk(min(chunk, _CHUNK), _routed_inter_state_row_bytes(BK, BC)))
    NCH = triton.cdiv(L, chunk)
    sel = _build_sel(D, b, nc, q.device)
    Wg = (Wg.float().contiguous() if use_g else q.new_zeros(H, d_model))
    swg = (Wg.stride(0), Wg.stride(1))
    # per-chunk pre-state snapshots (the recurrent STATE, not gates) for the reverse-scan — recomputed
    # here from the SAME logits so fwd/bwd routing is bit-consistent.
    snap = _routed_snapshots(q, k, v, h, lw, D, b, sel, chunk, BG, H, Wg=(Wg if use_g else None))
    dq = torch.zeros(B, L, dqk, device=q.device, dtype=torch.float32)
    dk = torch.zeros(B, L, dqk, device=q.device, dtype=torch.float32)
    dvv = torch.zeros(B, L, dv, device=q.device, dtype=torch.float32)
    dh = torch.zeros(B, L, d_model, device=q.device, dtype=torch.float32)   # decay-only dh (router dh via GEMM)
    dlr = torch.zeros(B, L, D, b, device=q.device, dtype=torch.float32)     # per-level read-logit grad (F2b)
    dlw = torch.zeros(B, L, D, b, device=q.device, dtype=torch.float32)     # per-level write-logit grad
    dWg = torch.zeros(H, d_model, device=q.device, dtype=torch.float32)     # per-head decay-weight grad (#45)
    # gda[B,L,nc]: persistent per-token log-decay adjoint accumulator (USE_G). Fold reverse-cumsums the
    # current chunk in-kernel into dld. RLA leaves it a 1-col stub.
    gda = torch.zeros(B, L, nc, device=q.device, dtype=torch.float32) if use_g \
        else q.new_zeros(B, 1, 1)
    sga = (gda.stride(0), gda.stride(1), gda.stride(2))
    _routed_bwd_intra[(B, NCH)](
        h, q, k, v, lr, lw, sel, Wg, do, dq, dk, dvv, dlr, dlw, gda,
        L, d_model, dqk, dv, nc, H,
        h.stride(0), h.stride(1), h.stride(2), q.stride(0), q.stride(1), q.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        *slo,
        sel.stride(0), sel.stride(1), sel.stride(2), swg[0], swg[1], *sga,
        do.stride(0), do.stride(1), do.stride(2),
        D=D, b=b, BB=BB, BT=chunk, BK=BK_full, BV=BVO, BD=BD, BC=BC,
        NCBLK=NCBLK, ND=triton.cdiv(dqk, BK_full), NDM=NDM,
        USE_G=use_g, GLA_FLOOR=_GLA_FLOOR, num_warps=4, num_stages=1)
    dS = torch.zeros(B, nc, dqk, dv, device=q.device, dtype=torch.float32)
    gdr = torch.zeros(B, chunk, nc, device=q.device, dtype=torch.float32)   # transient, OVERWRITTEN/chunk
    gdw = torch.zeros(B, chunk, nc, device=q.device, dtype=torch.float32)
    inter_common = dict(D=D, b=b, BB=BB, BT=chunk, BK=BK, BV=BV, BD=BD, BC=BC,
                        ND=ND, NDM=NDM, USE_G=use_g,
                        GLA_FLOOR=_GLA_FLOOR, num_warps=4, num_stages=1)
    for c in reversed(range(NCH)):
        Sj = snap[:, c].contiguous()
        # state-bwd reads dS = adjoint S_{j+1} (pre-decvec) → dk,dv,gdw + (USE_G) the carry/w_end da-pieces.
        _routed_bwd_inter_state[(B, NCBLK)](
            h, k, v, lr, lw, sel, Wg, Sj, dS, dk, dvv, gdw, gda,
            L, d_model, dqk, dv, nc, c * chunk, H,
            h.stride(0), h.stride(1), h.stride(2), q.stride(0), q.stride(1), q.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            *slo,
            sel.stride(0), sel.stride(1), sel.stride(2), swg[0], swg[1],
            dS.stride(0), dS.stride(1), dS.stride(2), dS.stride(3),
            gdr.stride(0), gdr.stride(1), gdr.stride(2), *sga,
            **inter_common)
        _routed_bwd_inter_read[(B, NCBLK)](
            h, q, lr, lw, sel, Wg, Sj, dS, do, dq, gdr, gda,
            L, d_model, dqk, dv, nc, c * chunk, H,
            h.stride(0), h.stride(1), h.stride(2), q.stride(0), q.stride(1), q.stride(2),
            *slo,
            sel.stride(0), sel.stride(1), sel.stride(2), swg[0], swg[1],
            Sj.stride(0), Sj.stride(1), Sj.stride(2), Sj.stride(3),
            do.stride(0), do.stride(1), do.stride(2),
            gdr.stride(0), gdr.stride(1), gdr.stride(2), *sga,
            **inter_common)
        _routed_bwd_fold[(B, NCBLK)](
            h, lr, lw, sel, gdr, gdw, dh, dlr, dlw,
            Wg, dWg, gda,
            L, d_model, nc, c * chunk, H,
            h.stride(0), h.stride(1), h.stride(2), *slo,
            sel.stride(0), sel.stride(1), sel.stride(2),
            gdr.stride(0), gdr.stride(1), gdr.stride(2), dh.stride(0), dh.stride(1), dh.stride(2),
            swg[0], swg[1], *sga,
            D=D, b=b, BB=BB, BT=chunk, BC=BC, BD=BD, NDM=NDM,
            USE_G=use_g, GLA_FLOOR=_GLA_FLOOR, DLD_FROM_GDA=use_g, num_warps=4, num_stages=1)
    if use_g:
        return dq[..., :dqk], dk[..., :dqk], dvv[..., :dv], dh, dlr, dlw, dWg
    return dq[..., :dqk], dk[..., :dqk], dvv[..., :dv], dh, dlr, dlw


class _RoLARoutedFn(torch.autograd.Function):
    """End-to-end differentiable in-kernel TREE-ROUTED RLA readout. Forward runs the OPTIMIZED production
    routed forward (`_routed_fwd_tiled`); backward drives the validated fold kernels at the production
    state-block width (BC=BG), reconstructing factors through the SAME Sel map. The [L,nc] gates AND
    their grads are never materialized (only transient [BT,BG] factor tiles + per-chunk [B,chunk,nc]
    gate-grad scratch)."""
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, h, lr, lw, D, b, chunk, BG, H, Wg=None):
        # F2b: the routing logits lr,lw:[BH,L,D,b] are the cuBLAS-GEMM output (h·W computed in torch by the
        # wrapper, autograd-tracked); the forward kernels only softmax+gather them. Backward emits the
        # per-level LOGIT grads dlr,dlw (autograd routes them through the GEMM → dWr,dWw,dh,db).
        cap = _CHUNK if Wg is not None else _CHUNK_FWD
        chunk = cap if chunk is None else min(chunk, cap)
        nc = b ** D
        sel = _build_sel(D, b, nc, q.device)
        q, k, v, h, lr, lw = (x.contiguous() for x in (q, k, v, h, lr, lw))
        Wgc = Wg.contiguous() if Wg is not None else None
        o = _routed_fwd_tiled(q, k, v, h, None, None, D, b, sel, chunk=chunk, BG=BG, Wg=Wgc, lr=lr, lw=lw)
        # save the logits (NOT [L,nc] gates/grads); the per-chunk pre-state snapshots are recomputed in
        # backward. #45: the GLA decay's saved activation is the [H,d_model] Wg (NOT a [L,nc] ld).
        ctx.save_for_backward(q, k, v, h, lr, lw, Wgc)
        ctx.D, ctx.b, ctx.chunk, ctx.BG, ctx.H = D, b, chunk, BG, H
        return o.to(q.dtype)

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do):
        q, k, v, h, lr, lw, Wg = ctx.saved_tensors
        grads = _rola_rla_routed_bwd(
            q, k, v, h, lr, lw, do.contiguous(), ctx.D, ctx.b, ctx.chunk, ctx.BG, ctx.H, Wg=Wg)
        use_g = Wg is not None
        dq, dk, dv, dh, dlr, dlw = grads[:6]
        dWg = grads[6] if use_g else None
        # forward arg order: q, k, v, h, lr, lw, D, b, chunk, BG, H, Wg
        return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dh.to(h.dtype),
                dlr.to(lr.dtype), dlw.to(lw.dtype), None, None, None, None, None,
                None if dWg is None else dWg.to(Wg.dtype))


@input_guard
def rola_rla_routed_triton(q, k, v, h, Wr, Ww, D, b, chunk=None, BG=16, b_r=None, b_w=None,
                           lr=None, lw=None, H=None):
    """Un-normalized TREE-ROUTED RLA readout via Triton — the shared-gram readout, DIFFERENTIABLE
    end-to-end (the [L,nc] gates and their grads are NEVER materialized). F2b: the per-level routing
    logits (h·Wr+b_r, h·Ww+b_w) are a cuBLAS GEMM (`_router_logits`, autograd-tracked) and the kernel only
    softmax+gathers them; dWr/dWw/dh/db flow through the GEMM-backward. Wr,Ww ∈ [H,D,d_model,b] per-head,
    optional per-head bias b_r/b_w ∈ [H,D,b]. lr,lw:[BH,L,D,b] may be passed precomputed (the layer's
    z-loss logits — dedup) instead of (Wr,Ww,b_r,b_w).

    q,k:[BH,L,K]  v:[BH,L,V]  h:[BH,L,d_model]  Wr,Ww:[H,D,d_model,b].  Returns [BH,L,V] at BV=next_pow2(V)."""
    if lr is None:
        H = Wr.shape[0]
        lr, lw = _router_logits(h, Wr, Ww, b_r, b_w, H)
    return _RoLARoutedFn.apply(q, k, v, h, lr, lw, D, b, chunk, BG, H, None)


@input_guard
def rola_gla_routed_triton(q, k, v, h, Wr, Ww, Wg, D, b, chunk=None, BG=16, b_r=None, b_w=None,
                           lr=None, lw=None, H=None):
    """Un-normalized TREE-ROUTED GLA readout via Triton — `rola_rla_routed_triton` + a per-head decay
    WEIGHT Wg:[H,d_model] (GLA), DIFFERENTIABLE end-to-end ([L,nc] gates, the per-state log-decay ld, AND
    their grads NEVER materialized — ld is computed IN-KERNEL from Wg + the write gate, #45). The routing
    gram uses the DECAYED gates (rt=r·eᵃ, w_end=w·e^{Λ−a}); Wg=None is the RLA path. lr,lw may be passed
    precomputed (the layer dedup)."""
    if lr is None:
        H = Wr.shape[0]
        lr, lw = _router_logits(h, Wr, Ww, b_r, b_w, H)
    return _RoLARoutedFn.apply(q, k, v, h, lr, lw, D, b, chunk, BG, H, Wg)


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
def _kappa_d_tile(q_ptr, sden_ptr, Gc, w_tile, ea, ea_g, ena_g, cols, cmask,
                  pid_b, rows, rmask, dqk, sq_b, sq_l, sq_d, ssd_b, ssd_k,
                  BT: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr, ND: tl.constexpr,
                  USE_G: tl.constexpr):
    """Per-state den d[BT,BC] = (G⊙causal)·w (intra) + q·Sden^c (inter). The inter term contracts dqk →
    accumulate over BK-feature-blocks so the [BC,BK] Sden slice stays bounded by BK<=64. Gc = G⊙causal is
    passed in (value-free, reused). BV-free; recomputed identically in each numerator value-block pass.
    USE_G (GLA): the DECAYED den d_i^c = e^{a_ic}·(Σ_{j≤i}(qi·kj) w_j^c e^{-a_jc} + qi·Sden_carry^c) —
    EXACTLY `naive_rola_gla_perstate_den` (intra w decayed by e^{-a}, the carried Sden is the pre-decay
    chunk-start state, the whole sum scaled by e^a).
    RE-ANCHORING (fp32-overflow kill): the intra gram exp(a_i)·exp(-a_j) is computed as
    exp(a_i-a_ref)·exp(a_ref-a_j) (a_ref = per-state midpoint of a) — identical product, both factors
    bounded (no e^{-a}=e^{|Λ|} blow-up for any BT). So the INTRA term uses the anchored pair (ea_g·ena_g)
    while the INTER (q·Sden, against the absolutely-decayed carry) keeps the TRUE ea=e^a (≤1, no overflow).
    ea = e^a [BT,BC] (true, inter); ea_g = e^{a-a_ref}, ena_g = e^{a_ref-a} [BT,BC] (anchored, intra)."""
    wt = (w_tile * ena_g) if USE_G else w_tile
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
    # intra scaled by the anchored ea_g (cancels ena_g's a_ref → e^{a_i-a_j}); inter by the true e^a.
    d = (d_intra * ea_g + d_inter * ea) if USE_G else (d_intra + d_inter)
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
def _kappa_fwd_scan_cb_atomic(alpha_ptr, q_ptr, k_ptr, v_ptr, lr_ptr, lw_ptr, sel_ptr, kap_ptr,
                              sval_ptr, sden_ptr, num_ptr, den_ptr,
                              snap_val_ptr, snap_den_ptr,
                              L, dqk, dv, nc,
                              sa_b, sa_l, sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sk_b, sk_l,
                              slo_b, slo_l, slo_lvl, slo_bb, ssel_lvl, ssel_b, ssel_c,
                              ssv_b, ssv_c, ssv_k, ssv_v, ssd_b, ssd_c, ssd_k,
                              snm_b, snm_l, snm_v, sdn_b, sdn_l,
                              snv_n, snv_b, snv_c, snv_k, snv_v, snd_n, snd_b, snd_c, snd_k,
                              GLOBAL: tl.constexpr, PER_STATE: tl.constexpr, EPS: tl.constexpr,
                              D: tl.constexpr, b: tl.constexpr, BB: tl.constexpr,
                              BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                              BC: tl.constexpr, NCBLK: tl.constexpr, ND: tl.constexpr,
                              NDV: tl.constexpr, NDVP: tl.constexpr,
                              USE_G: tl.constexpr, GLA_FLOOR: tl.constexpr,
                              C_LO: tl.constexpr, NCH_LOOP: tl.constexpr,
                              NEED_SNAPSHOTS: tl.constexpr, SAVE_CHECKPOINTS: tl.constexpr,
                              CHECKPOINT_EVERY: tl.constexpr):
    """State-block kappa forward scan: one launch loops over chunks inside each program."""
    pid_b = tl.program_id(0)
    cb = tl.program_id(1)
    ND_V: tl.constexpr = NDV
    offs_t = tl.arange(0, BT)
    offs_bb = tl.arange(0, BB)
    offs_c = tl.arange(0, BC)
    bmask = offs_bb < b
    cols = cb * BC + offs_c
    cmask = cols < nc
    vbidx = tl.arange(0, NDVP)
    offs_vf = tl.arange(0, NDVP * BV)
    vfmask = offs_vf < dv

    for ci in tl.range(0, NCH_LOOP):
        c_abs = C_LO + ci
        rows = c_abs * BT + offs_t
        rmask = rows < L

        if NEED_SNAPSHOTS or SAVE_CHECKPOINTS:
            write_snap = True
            snap_i = ci
            if SAVE_CHECKPOINTS:
                write_snap = (c_abs % CHECKPOINT_EVERY) == 0
                snap_i = c_abs // CHECKPOINT_EVERY
            for d0s in range(ND):
                offs_ks = d0s * BK + tl.arange(0, BK)
                kmasks = offs_ks < dqk
                cks = tl.reshape(cols[:, None] * dqk + offs_ks[None, :], [BC * BK])
                ckmasks = tl.reshape(cmask[:, None] & kmasks[None, :], [BC * BK])
                sd = tl.load(sden_ptr + pid_b*ssd_b + cks*ssd_k, mask=ckmasks, other=0.0)
                tl.store(snap_den_ptr + snap_i*snd_n + pid_b*snd_b + cks*snd_k,
                         sd, mask=write_snap & ckmasks)
                for vbs in range(ND_V):
                    offs_vs = vbs * BV + tl.arange(0, BV)
                    vmasks = offs_vs < dv
                    sf = tl.load(sval_ptr + pid_b*ssv_b + cks[:, None]*ssv_k + offs_vs[None, :]*ssv_v,
                                 mask=ckmasks[:, None] & vmasks[None, :], other=0.0)
                    tl.store(snap_val_ptr + snap_i*snv_n + pid_b*snv_b + cks[:, None]*snv_k + offs_vs[None, :]*snv_v,
                             sf, mask=write_snap & ckmasks[:, None] & vmasks[None, :])

        if USE_G:
            alpha = tl.load(alpha_ptr + pid_b * sa_b + rows * sa_l, mask=rmask, other=1.0)
        kap = tl.load(kap_ptr + pid_b*sk_b + rows*sk_l, mask=rmask, other=0.0)
        causal = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]

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

        r_tile, w_tile = _build_rw_tile_logits(lr_ptr, lw_ptr, sel_ptr, cols, cmask,
                                               pid_b, rows, rmask, offs_bb, bmask,
                                               slo_b, slo_l, slo_lvl, slo_bb, slo_b, slo_l, slo_lvl, slo_bb,
                                               ssel_lvl, ssel_b, ssel_c, D, BT, BB, BC)
        if USE_G:
            ldc = _ld_from_w(w_tile, alpha, cmask, GLA_FLOOR)
            a = tl.cumsum(ldc, axis=0)
            Lam = tl.sum(tl.where(offs_t[:, None] == (BT - 1), a, 0.0), axis=0)
            ea, ea_g, ena_g = _decay_factors(a)
            wt = w_tile * ena_g
            w_end = w_tile * tl.exp(Lam[None, :] - a)
            dec_c = tl.exp(Lam)
        else:
            ea = w_tile * 0.0 + 1.0
            ea_g = ea
            ena_g = ea
            wt = w_tile
            w_end = w_tile

        d_tile = _kappa_d_tile(q_ptr, sden_ptr, Gc, w_tile, ea, ea_g, ena_g, cols, cmask,
                               pid_b, rows, rmask, dqk, sq_b, sq_l, sq_d, ssd_b, ssd_k,
                               BT, BK, BC, ND, USE_G)
        rt_tile = _kappa_rescale(r_tile, d_tile, kap, cmask, GLOBAL, PER_STATE, EPS)
        o_den = tl.sum(rt_tile * d_tile, axis=1)
        tl.atomic_add(den_ptr + pid_b*sdn_b + rows*sdn_l, o_den, sem="relaxed", mask=rmask)

        rd_inter = (rt_tile * ea) if USE_G else rt_tile
        rd_gram = (rt_tile * ea_g) if USE_G else rt_tile
        Rg = tl.dot(rd_gram, tl.trans(wt))
        A = G * tl.where(causal, Rg, 0.0)
        o_num = tl.zeros([BT, NDVP, BV], dtype=tl.float32)
        for vb in range(ND_V):
            offs_v = vb * BV + tl.arange(0, BV)
            vmask = offs_v < dv
            sel_vb = (vbidx[None, :, None] == vb)
            vc = tl.load(v_ptr + pid_b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                         mask=rmask[:, None] & vmask[None, :], other=0.0)
            o_num += tl.where(sel_vb, tl.dot(A.to(vc.dtype), vc)[:, None, :], 0.0)
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
                                mask=ckmask[:, None] & vmask[None, :], other=0.0)
                rq = tl.reshape(rd_inter[:, :, None] * qc[:, None, :], [BT, BC * BK])
                o_num += tl.where(sel_vb, tl.dot(rq.to(sflat.dtype), sflat)[:, None, :], 0.0)
                wk = tl.reshape(w_end[:, :, None] * kc[:, None, :], [BT, BC * BK])
                if USE_G:
                    deckv = tl.reshape(dec_c[:, None] * tl.full([BC, BK], 1.0, tl.float32), [BC * BK])
                    snew = deckv[:, None] * sflat + tl.dot(tl.trans(wk).to(vc.dtype), vc)
                else:
                    snew = sflat + tl.dot(tl.trans(wk).to(vc.dtype), vc)
                tl.store(sval_ptr + pid_b*ssv_b + ck[:, None]*ssv_k + offs_v[None, :]*ssv_v,
                         snew, mask=ckmask[:, None] & vmask[None, :])

        tl.atomic_add(num_ptr + pid_b*snm_b + rows[:, None]*snm_l + offs_vf[None, :]*snm_v,
                      tl.reshape(o_num, [BT, NDVP * BV]), sem="relaxed",
                      mask=rmask[:, None] & vfmask[None, :])

        for d0 in range(ND):
            offs_k = d0 * BK + tl.arange(0, BK)
            kmask = offs_k < dqk
            kc = tl.load(k_ptr + pid_b*sq_b + rows[:, None]*sq_l + offs_k[None, :]*sq_d,
                         mask=rmask[:, None] & kmask[None, :], other=0.0)
            ck = tl.reshape(cols[:, None] * dqk + offs_k[None, :], [BC * BK])
            ckmask = tl.reshape(cmask[:, None] & kmask[None, :], [BC * BK])
            wk = tl.reshape(w_end[:, :, None] * kc[:, None, :], [BT, BC * BK])
            dsden = tl.sum(wk, axis=0)
            sden = tl.load(sden_ptr + pid_b*ssd_b + ck*ssd_k, mask=ckmask, other=0.0)
            if USE_G:
                deckv = tl.reshape(dec_c[:, None] * tl.full([BC, BK], 1.0, tl.float32), [BC * BK])
                tl.store(sden_ptr + pid_b*ssd_b + ck*ssd_k, deckv * sden + dsden, mask=ckmask)
            else:
                tl.store(sden_ptr + pid_b*ssd_b + ck*ssd_k, sden + dsden, mask=ckmask)


def _kappa_fit_chunk(dqk, dv, chunk, BC=16):
    """The SMEM-derived kappa chunk — the single source of truth shared by the forward, the #55
    checkpoint pass, and the backward reverse-scan. Idempotent: re-fitting an already-fitted chunk
    returns it, so the bwd recompute lands on the SAME chunk → NCH / snapshot alignment is guaranteed
    without threading the value through every call."""
    bk = _kappa_bk_cap(dqk, dv, min(chunk, _CHUNK_FWD), BC)
    return _fit_chunk(
        min(chunk, _CHUNK_FWD),
        _kappa_row_bytes(bk, BC),
        bt2_bytes=_KAPPA_BWD_STATE_BT2_BYTES)


def _kappa_ckpt_window(NCH):
    """#55 checkpoint window W = ⌈√NCH⌉ — shared by the forward (stores a checkpoint every W-th chunk)
    and the backward (seeds segment `seg` from ckpt[seg] at chunk seg·W). √NCH checkpoints + a W-wide
    segment recompute bound the snapshot term to O(√NCH·state) instead of the all-snapshot NCH·state
    (∝ L² when nc∝L)."""
    return max(1, math.ceil(math.sqrt(NCH)))


def _kappa_routed_fwd(q, k, v, alpha, lr, lw, kap, D, b, sel, chunk, global_norm, per_state, eps,
                      H, need_snapshots=False, state_in=None, c_lo=0, c_hi=None,
                      save_checkpoints=False, checkpoint_every_override=None):
    """Fused global/kappa/per_state tree-routed forward. Returns (num[B,L,BV], den[B,L], snap_val, snap_den).
    #55 snapshot modes (at most one): need_snapshots=True stores the DENSE per-chunk pre-states for chunks
    [c_lo,c_hi) (the backward's per-segment recompute); save_checkpoints=True stores only the SPARSE √NCH
    boundary states (the differentiable forward, for the backward to seed segments from). Both default off →
    (..., None, None) — inference / non-grad forward allocates no snapshot tensor (no prefill blowup).
    Optional precomputed per-head decay alpha:[B,L] (GLA, #45) → USE_G decayed scan. No [L,nc]
    gate/den/r̃/ld buffer is materialized."""
    B, L, dqk = q.shape
    dv = v.shape[-1]
    nc = b ** D
    BC = 16   # tl.dot needs the gram dim >=16; the nc tail is cmask'd (was max(16,min(nc,16)) ≡ 16)
    BK = _kappa_bk_cap(dqk, dv, min(chunk, _CHUNK_FWD), BC)  # feature-tile (loop ND) — bounds the [BT,BC*BK] tiles
    BV = _kappa_bv_tile(dqk, dv, BC)                     # value-tile (loop ND_V) so [BC*BK,BV] fits SRAM
    BB = max(16, triton.next_power_of_2(b))
    # The [BT,BC*BK] rq/wk tiles scale with BT → the chunk is SMEM-derived (one uniform formula, fwd+bwd):
    # the largest pow2 BT whose dominant fp32 chunk-tile fits the device budget. On a 99KB card this lands
    # at the full 64 (the design-point win: 64 launches → 16); a smaller card steps down. Chunk-invariant
    # scan, so a bit-identical readout at any chunk. Idempotent (the Function already capped to the ceiling).
    chunk = _kappa_fit_chunk(dqk, dv, chunk, BC)
    NCBLK = triton.cdiv(nc, BC)
    ND = triton.cdiv(dqk, BK)
    NCH = triton.cdiv(L, chunk)
    if state_in is None:
        Sval = torch.zeros(B, nc, dqk, dv, device=q.device, dtype=torch.float32)
        Sden = torch.zeros(B, nc, dqk, device=q.device, dtype=torch.float32)
    else:
        # #55: seed from a segment checkpoint; clone so the caller's checkpoint is never mutated.
        Sval = state_in[0].clone()
        Sden = state_in[1].clone()
    if c_hi is None:
        c_hi = NCH
    # #55: save_checkpoints (the differentiable forward) → store the sparse √NCH boundary states the
    # backward seeds its segment recomputes from. Resolved to the shared window so fwd-store/bwd-index agree.
    checkpoint_every = _kappa_ckpt_window(NCH) if save_checkpoints else None
    if checkpoint_every is not None and checkpoint_every_override is not None:
        checkpoint_every = max(1, int(checkpoint_every_override))
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
    # #55 sqrt-checkpointing: need_snapshots stores the DENSE per-chunk pre-states for chunks [c_lo,c_hi)
    # (a segment recompute, O(segment·state)); checkpoint_every stores only every W-th state (the sparse
    # Pass-1 boundary checkpoints the backward seeds segments from, O(√NCH·state)). At most one is set —
    # never the old [NCH,...] all-snapshot peak (∝ L² when nc scales with L).
    if need_snapshots:
        snap_val = torch.zeros(c_hi - c_lo, B, nc, dqk, dv, device=q.device, dtype=torch.float32)
        snap_den = torch.zeros(c_hi - c_lo, B, nc, dqk, device=q.device, dtype=torch.float32)
    elif checkpoint_every is not None:
        nseg = triton.cdiv(NCH, checkpoint_every)
        snap_val = torch.zeros(nseg, B, nc, dqk, dv, device=q.device, dtype=torch.float32)
        snap_den = torch.zeros(nseg, B, nc, dqk, device=q.device, dtype=torch.float32)
    else:
        snap_val = snap_den = None
    use_g = alpha is not None
    alpha = alpha.float().contiguous() if use_g else q.new_empty(1, 1)
    sa = (alpha.stride(0), alpha.stride(1))
    slo = (lw.stride(0), lw.stride(1), lw.stride(2), lw.stride(3))
    common = dict(GLOBAL=global_norm, PER_STATE=per_state, EPS=eps, D=D, b=b, BB=BB, BT=chunk,
                  BK=BK, BV=BV, BC=BC, NCBLK=NCBLK, ND=ND, NDV=triton.cdiv(dv, BV),
                  NDVP=triton.next_power_of_2(triton.cdiv(dv, BV)),
                  USE_G=use_g, GLA_FLOOR=_GLA_FLOOR, num_warps=4, num_stages=1)
    snap_val_arg = snap_val if snap_val is not None else Sval
    snap_den_arg = snap_den if snap_den is not None else Sden
    if snap_val is not None:
        snv = snap_val.stride()
        snd = snap_den.stride()
    else:
        snv = (0, 0, 0, 0, 0)
        snd = (0, 0, 0, 0)
    _kappa_fwd_scan_cb_atomic[(B, NCBLK)](
        alpha, q, k, v, lr, lw, sel, kap, Sval, Sden, num, den, snap_val_arg, snap_den_arg,
        L, dqk, dv, nc,
        sa[0], sa[1], q.stride(0), q.stride(1), q.stride(2),
        v.stride(0), v.stride(1), v.stride(2), kap.stride(0), kap.stride(1),
        *slo,
        sel.stride(0), sel.stride(1), sel.stride(2),
        Sval.stride(0), Sval.stride(1), Sval.stride(2), Sval.stride(3),
        Sden.stride(0), Sden.stride(1), Sden.stride(2),
        num.stride(0), num.stride(1), num.stride(2), den.stride(0), den.stride(1),
        *snv, *snd,
        C_LO=c_lo, NCH_LOOP=c_hi - c_lo,
        NEED_SNAPSHOTS=need_snapshots, SAVE_CHECKPOINTS=checkpoint_every is not None,
        CHECKPOINT_EVERY=checkpoint_every or 1,
        **common)
    # Return the SMEM-fitted `chunk` actually used: the backward's reverse-scan must drive the SAME chunk
    # as this recompute (NCH / per-chunk snapshot alignment) — threading it out is the single source of
    # truth (no re-fit-must-match-the-fit coupling).
    return num, den, snap_val, snap_den, chunk


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
def _kappa_bwd_state(h_ptr, k_ptr, v_ptr, lr_ptr, lw_ptr, sel_ptr, wg_ptr, sval_ptr, sden_ptr,
                     dsval_ptr, dsden_ptr, dk_ptr, dv_ptr, gdw_ptr, gda_ptr,
                     L, d_model, dqk, dv, nc, t_start, H,
                     sh_b, sh_l, sh_d, sq_b, sq_l, sq_d, sv_b, sv_l, sv_d,
                     slo_b, slo_l, slo_lvl, slo_bb, ssel_lvl, ssel_b, ssel_c,
                     swg_head, swg_d,
                     ssv_b, ssv_k, ssv_v, ssd_b, ssd_k, sgd_b, sgd_t, sgd_c, sga_b, sga_l, sga_c,
                     D: tl.constexpr, b: tl.constexpr, BB: tl.constexpr,
                     BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BD: tl.constexpr,
                     BC: tl.constexpr, ND: tl.constexpr, NDM: tl.constexpr,
                     USE_G: tl.constexpr, GLA_FLOOR: tl.constexpr):
    """State-update backward (split #1, SMEM-bound by the dSval tile). Grid (B·H, NCBLK): ONE program owns
    ONE nc-state-block cb=program_id(1) (was an in-program serial `for cb` loop) — the #58 occupancy widen
    (16→256 blocks). The nc-state-blocks are INDEPENDENT here: dk/dv are token-indexed `tl.atomic_add`
    (safe, now more concurrent adders), gda is cb-local-cols atomic, gdw is a cb-local (own-cols) store — no
    cross-cb accumulator, so the fan-out needs no recombine. F2b: write factors rebuilt from
    logits; the state half of dw stored to gdw. Uses the INCOMING adjoint (= Sval_{j+1}, Sden_{j+1}) to
    backprop the two writes Sval^c += Σ wᶜ k⊗v and Sden^c += Σ wᶜ k. Produces dk,dv (atomic). No [L,nc].
    USE_G (GLA): both writes use w_end=w·e^{Λ-a}; dk/dv flow through w_end; the write routing-factor grad
    is dw_end·e^{Λ-a} (→ gdw). da-pieces: da_wend=−dw_end·w_end, and the per-chunk Λ-coupling
    dlam = e^Λ·Σ(S_j∘ds_in) − Σ_t da_wend (over BOTH the Sval and Sden carries) on the LAST row of gda.
    The ds_in here is the PRE-decvec adjoint of S_{j+1}; this kernel then overwrites dS with e^Λ·ds_in,
    the adjoint of S_j's carry."""
    pid_b = tl.program_id(0)
    ND_V = (dv + BV - 1) // BV       # value-blocks (BV is the value TILE; ND_V==1 ⇒ the un-tiled kernel)
    offs_t = tl.arange(0, BT)
    offs_bb = tl.arange(0, BB)
    offs_c = tl.arange(0, BC)
    bmask = offs_bb < b
    rows = t_start + offs_t
    rmask = rows < L
    _hd = pid_b % H                              # per-head decay-weight slice (BH fold is (B,H))
    wg_ptr = wg_ptr + _hd * swg_head            # per-head decay weight (read-only here; #45)
    if USE_G:                                   # per-head decay gate alpha[BT], once (in-kernel ld, #45)
        alpha = _build_alpha(h_ptr, wg_ptr, pid_b, rows, rmask, d_model,
                             sh_b, sh_l, sh_d, swg_d, BT, BD, NDM)
    cb = tl.program_id(1)
    cols = cb * BC + offs_c
    cmask = cols < nc
    _r, w_tile = _build_rw_tile_logits(lr_ptr, lw_ptr, sel_ptr, cols, cmask,
                                       pid_b, rows, rmask, offs_bb, bmask,
                                       slo_b, slo_l, slo_lvl, slo_bb, slo_b, slo_l, slo_lvl, slo_bb,
                                       ssel_lvl, ssel_b, ssel_c, D, BT, BB, BC, BUILD_R=False)
    if USE_G:
        ldc = _ld_from_w(w_tile, alpha, cmask, GLA_FLOOR)
        a = tl.cumsum(ldc, axis=0)
        Lam = tl.sum(tl.where(offs_t[:, None] == (BT - 1), a, 0.0), axis=0)   # [BC] chunk-total
        dec = tl.exp(Lam)
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
                dec_k = tl.reshape(dec[:, None] * tl.full([BC, BK], 1.0, tl.float32), [BC * BK])
                tl.store(dsval_ptr + pid_b*ssv_b + ck[:, None]*ssv_k + offs_v[None, :]*ssv_v,
                         dSval * dec_k[:, None], mask=ckmask[:, None] & vmask[None, :])
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
            dec_k = tl.reshape(dec[:, None] * tl.full([BC, BK], 1.0, tl.float32), [BC * BK])
            tl.store(dsden_ptr + pid_b*ssd_b + ck*ssd_k, dSden * dec_k, mask=ckmask)
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
def _kappa_bwd_read(h_ptr, q_ptr, k_ptr, v_ptr, lr_ptr, lw_ptr, sel_ptr, kap_ptr, wg_ptr,
                    sval_ptr, sden_ptr, dnum_ptr, dden_ptr,
                    dsval_ptr, dsden_ptr, dq_ptr, dk_ptr, dv_ptr, dkap_ptr, gdr_ptr, gdw_ptr, gda_ptr,
                    L, d_model, dqk, dv, nc, t_start, H,
                    sh_b, sh_l, sh_d, sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sk_b, sk_l,
                    slo_b, slo_l, slo_lvl, slo_bb, ssel_lvl, ssel_b, ssel_c,
                    swg_head, swg_d,
                    ssv_b, ssv_k, ssv_v, ssd_b, ssd_k,
                    sdo_b, sdo_l, sdo_v, sdd_b, sdd_l, sgd_b, sgd_t, sgd_c, sga_b, sga_l, sga_c,
                    GLOBAL: tl.constexpr, PER_STATE: tl.constexpr, EPS: tl.constexpr,
                    D: tl.constexpr, b: tl.constexpr, BB: tl.constexpr,
                    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BD: tl.constexpr,
                    BC: tl.constexpr, ND: tl.constexpr, NDM: tl.constexpr,
                    USE_G: tl.constexpr, GLA_FLOOR: tl.constexpr):
    """Readout/den/d backward (split #2, SMEM-bound by the sval snapshot tile). Grid (B·H, NCBLK): ONE
    program owns ONE nc-state-block cb=program_id(1) (was an in-program serial `for cb` loop) — the #58
    occupancy widen (16→256 blocks). Per-block writes are cb-local (dsval/dsden own-cols stores, gdr own-cols
    store) or token-indexed atomic (dq,dv) or cb-local-cols atomic (gda,gdw). The two cross-cb reductions —
    the content-gram adjoint dG[BT,BT] and dkap[BT] — are now PER-PROGRAM PARTIALS: each program contracts
    its own dG→dq/dk and emits its dkap via the SAME post-pass `tl.atomic_add` (the recombine is the atomic,
    linear in dG/dkap so the split is exact to the fp atomic-reorder floor). Recomputes r,w,d,r_tilde
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
    _hd = pid_b % H                              # per-head decay-weight slice (BH fold is (B,H))
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
    cb = tl.program_id(1)
    cols = cb * BC + offs_c
    cmask = cols < nc
    r_tile, w_tile = _build_rw_tile_logits(lr_ptr, lw_ptr, sel_ptr, cols, cmask,
                                           pid_b, rows, rmask, offs_bb, bmask,
                                           slo_b, slo_l, slo_lvl, slo_bb, slo_b, slo_l, slo_lvl, slo_bb,
                                           ssel_lvl, ssel_b, ssel_c, D, BT, BB, BC)
    if USE_G:
        ldc = _ld_from_w(w_tile, alpha, cmask, GLA_FLOOR)
        a = tl.cumsum(ldc, axis=0)
        ea, ea_g, ena_g = _decay_factors(a)
        wt = w_tile * ena_g                # intra gram / den-intra write, ANCHORED (no e^{-a} overflow)
    else:
        ea = w_tile * 0.0 + 1.0
        ea_g = ea
        ena_g = ea
        wt = w_tile
    # d_tile/rt_tile are value-FREE (from G,Sden,r) → compute once, reused for every value-block.
    d_tile = _kappa_d_tile(q_ptr, sden_ptr, Gc, w_tile, ea, ea_g, ena_g, cols, cmask,
                           pid_b, rows, rmask, dqk, sq_b, sq_l, sq_d, ssd_b, ssd_k,
                           BT, BK, BC, ND, USE_G)
    rt_tile = _kappa_rescale(r_tile, d_tile, kap, cmask, GLOBAL, PER_STATE, EPS)   # UNDECAYED r̃
    rd_inter = (rt_tile * ea) if USE_G else rt_tile    # readout vs abs-decayed Sval carry: TRUE e^a
    rd_gram = (rt_tile * ea_g) if USE_G else rt_tile    # intra gram read: ANCHORED (pairs with wt's ena_g)
    # For RLA (ea≡ea_g≡1, wt≡w, rd≡r̃) the grad folds DIRECTLY into drt/dw with the SAME accumulation
    # order as the base RLA kernel → byte-identical. For GLA the readout/gram grads collect on the DECAYED
    # gates and the da-pieces accumulate into gda. The anchored gram read (rd_gram) and the true inter
    # read (rd_inter) carry SEPARATE grad accumulators (drd_g/drd_i); each cancels its a_ref against its
    # dual (wt's ena_g for the gram, the Sval/Sden carry for the inter) so every fold is unchanged.
    drt = dden[:, None] * d_tile          # d(r_tilde) from den (UNDECAYED r̃ → no e^a)
    dd = dden[:, None] * rt_tile          # d(d) from den
    drd_g = tl.zeros([BT, BC], dtype=tl.float32)   # readout grad w.r.t. the ANCHORED gram read rd_gram
    drd_i = tl.zeros([BT, BC], dtype=tl.float32)   # readout grad w.r.t. the TRUE inter read rd_inter
    # num intra: A=G*Rg*causal ; o_num += A v. dA=(dnum·vᵀ)⊙causal (value-contracted → sum over vb);
    # dv=Aᵀ·dnum (per value-block → atomic). A/Rg/dG/dw are value-free; dnum,vc loaded per vb. Rg's
    # anti-causal triangle is +inf at large BT → mask via where (Rg_m) so neither A nor dG forms inf·0.
    Rg = tl.dot(rd_gram, tl.trans(wt))
    Rg_m = tl.where(causal > 0.0, Rg, 0.0)
    A = G * Rg_m
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
    dG += dA * Rg_m                        # dG_ij += dA_ij Rg_ij — Rg_m (causal-masked) avoids 0·inf=NaN
    if USE_G:
        drd_g += tl.dot(dRg.to(wt.dtype), wt)
        dwt = tl.dot(tl.trans(dRg).to(rd_gram.dtype), rd_gram)
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
        rq = tl.reshape(rd_inter[:, :, None] * qc[:, None, :], [BT, BC * BK])
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
            drd_i += tl.sum(Mr * qc[:, None, :], axis=2)
        else:
            drt += tl.sum(Mr * qc[:, None, :], axis=2)
        dq_read = tl.sum(Mr * rd_inter[:, :, None], axis=1)   # [BT,BK]
        tl.atomic_add(dq_ptr + pid_b*sq_b + rows[:, None]*sq_l + offs_k[None, :]*sq_d,
                      dq_read, mask=rmask[:, None] & kmask[None, :])
    # readout grad → UNDECAYED r̃ — GLA only (RLA folded drt directly above). The gram part folds via
    # the anchored ea_g, the inter part via the true e^a; each = the true e^a contribution to ∂L/∂r̃.
    if USE_G:
        drt += drd_g * ea_g + drd_i * ea
    # rescale bwd (drt now complete; drt is grad w.r.t. the UNDECAYED r̃).
    dr_resc, dd_resc, dkap_c = _kappa_rescale_bwd(drt, r_tile, d_tile, kap, cmask,
                                                  GLOBAL, PER_STATE, EPS)
    dd += dd_resc
    dkap_acc += dkap_c
    # d bwd: d = (Gc·wt)·e^{a-ref} [intra] + (q·Sden_j)·e^a [inter] (GLA) | (Gc·w + q·Sden_j) [RLA]. The
    # inner-sum grads split: ds_intra=dd·ea_g (the ANCHORED den-intra Gc·wt), ds_inter=dd·ea (the TRUE
    # den-inter q·Sden); the outer-decay da-piece is dd·d (full d). dG/dw are dqk-free; dq + dSden → d0.
    ds_intra = (dd * ea_g) if USE_G else dd
    ds_inter = (dd * ea) if USE_G else dd
    dG += tl.dot(ds_intra.to(wt.dtype), tl.trans(wt)) * causal
    if USE_G:
        dwt += tl.dot(tl.trans(Gc).to(ds_intra.dtype), ds_intra)   # den intra → ANCHORED wt
    else:
        dw += tl.dot(tl.trans(Gc).to(ds_intra.dtype), ds_intra)
    for d0 in range(ND):
        offs_k = d0 * BK + tl.arange(0, BK)
        kmask = offs_k < dqk
        qc = tl.load(q_ptr + pid_b*sq_b + rows[:, None]*sq_l + offs_k[None, :]*sq_d,
                     mask=rmask[:, None] & kmask[None, :], other=0.0)
        ck = tl.reshape(cols[:, None] * dqk + offs_k[None, :], [BC * BK])
        ckmask = tl.reshape(cmask[:, None] & kmask[None, :], [BC * BK])
        sden = tl.load(sden_ptr + pid_b*ssd_b + ck*ssd_k, mask=ckmask, other=0.0)
        sden2 = tl.reshape(sden, [BC, BK])
        dq_d = tl.dot(ds_inter.to(sden2.dtype), sden2)       # [BT,BK]
        tl.atomic_add(dq_ptr + pid_b*sq_b + rows[:, None]*sq_l + offs_k[None, :]*sq_d,
                      dq_d, mask=rmask[:, None] & kmask[None, :])
        dSden = tl.load(dsden_ptr + pid_b*ssd_b + ck*ssd_k, mask=ckmask, other=0.0)
        dSden_read = tl.reshape(tl.dot(tl.trans(ds_inter).to(qc.dtype), qc), [BC * BK])
        tl.store(dsden_ptr + pid_b*ssd_b + ck*ssd_k, dSden + dSden_read, mask=ckmask)
    # write-gate grad: GLA — dwt is the grad w.r.t. the ANCHORED wt=w·e^{a_ref-a}; routing-factor grad
    # dw=dwt·ena_g; da_wt=−dwt·wt (the a_ref cancels → true); the den's outer-decay da-piece is dd·d,
    # the readout's splits into drd_g·rd_gram (gram) + drd_i·rd_inter (inter).
    if USE_G:
        dw = dwt * ena_g
        da = drd_g * rd_gram + drd_i * rd_inter - dwt * wt + dd * d_tile
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


def _kappa_routed_bwd(q, k, v, h, lr, lw, kap, alpha, ckpt_val, ckpt_den, out, dnum, dden,
                      D, b, sel, chunk, global_norm, per_state, eps, H, Wg=None,
                      checkpoint_window_override=None):
    """Reverse chunk-scan backward for the fused global/kappa/per_state path. Carries dSval,dSden
    adjoints; recomputes r,w,d,r_tilde transiently per chunk; folds the [BT,nc] gate-grads into
    dWr,dWw,dh (and db_r/db_w when a routing bias is present — transient, never [L,nc]). Returns
    dq,dk,dv,dh,dWr,dWw,dkappa,dWg (and db_r,db_w when biased) — all fp32. No [L,nc] buffer is allocated.

    USE_G (GLA, #45): optional per-head decay weight Wg:[H,d_model] → the decayed kappa backward, the
    per-state log-decay ld computed IN-KERNEL (never a [L,nc] ld). The two grad kernels run their USE_G
    paths (decayed gates rd=r̃·e^a, wt=w·e^{-a}, w_end=w·e^{Λ-a}; the den d carries decay; the da-pieces
    accumulate into a persistent gda[B,L,nc]). Between state-bwd and read-bwd BOTH carried adjoints decay
    by e^Λ (the reverse of the forward's e^Λ state carry; Λ_c recomputed chunk-locally from Wg). At each
    chunk the fold kernel reverse-cumsums gda in-kernel into dld, then splits it into dWg + the extra dh +
    the decay's write-gate grad. Wg=None is RLA (byte-identical; gda/dld unused)."""
    use_g = Wg is not None
    B, L, d_model = h.shape
    dqk = q.shape[-1]
    dv = v.shape[-1]
    nc = b ** D
    BC = 16   # tl.dot gram dim >=16; nc tail cmask'd (was max(16,min(nc,16)) ≡ 16)
    # #55: fit the chunk here (shared _kappa_fit_chunk, idempotent) — the Function passes the ctx ceiling
    # and the checkpoint pass + segment recomputes below all drive THIS value, so NCH/snapshot alignment
    # holds without threading the fitted chunk through every call.
    chunk = _kappa_fit_chunk(dqk, dv, chunk, BC)
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
    dq = torch.zeros(B, L, dqk, device=q.device, dtype=torch.float32)
    dk = torch.zeros(B, L, dqk, device=q.device, dtype=torch.float32)
    dvv = torch.zeros(B, L, dv, device=q.device, dtype=torch.float32)
    dh = torch.zeros(B, L, d_model, device=q.device, dtype=torch.float32)   # decay-only dh (router via GEMM)
    dlr = torch.zeros(B, L, D, b, device=q.device, dtype=torch.float32)     # per-level read-logit grad (F2b)
    dlw = torch.zeros(B, L, D, b, device=q.device, dtype=torch.float32)     # per-level write-logit grad
    dWg = torch.zeros(H, d_model, device=q.device, dtype=torch.float32)   # per-head decay-weight grad (#45)
    dkap = torch.zeros(B, L, device=q.device, dtype=torch.float32)
    dSval = torch.zeros(B, nc, dqk, dv, device=q.device, dtype=torch.float32)
    dSden = torch.zeros(B, nc, dqk, device=q.device, dtype=torch.float32)
    out = out.contiguous()
    dnum = dnum.contiguous()
    dden = dden.contiguous()
    # Wg:[H,d_model] (GLA) — the per-head decay weight; ld is computed IN-KERNEL (never a [L,nc] ld). RLA
    # passes a [H,d_model] stub the kernels skip (USE_G=False).
    Wg = (Wg.float().contiguous() if use_g else q.new_zeros(H, d_model))
    swg = (Wg.stride(0), Wg.stride(1))
    # gda[B,L,nc]: persistent per-token log-decay adjoint (USE_G) — state/read kernels add their da-pieces
    # per chunk. Fold reverse-cumsums the current chunk in-kernel into dld. RLA leaves it a 1-col stub.
    gda = torch.zeros(B, L, nc, device=q.device, dtype=torch.float32) if use_g else q.new_zeros(B, 1, 1)
    sga = (gda.stride(0), gda.stride(1), gda.stride(2))
    # transient per-chunk gate-grad scratch [B,chunk,nc], OVERWRITTEN each chunk — never [L,nc].
    gdr = torch.zeros(B, chunk, nc, device=q.device, dtype=torch.float32)
    gdw = torch.zeros(B, chunk, nc, device=q.device, dtype=torch.float32)
    sB = (q.stride(0), q.stride(1), q.stride(2))
    sV = (v.stride(0), v.stride(1), v.stride(2))
    sH = (h.stride(0), h.stride(1), h.stride(2))
    slo = (lw.stride(0), lw.stride(1), lw.stride(2), lw.stride(3))   # [BH,L,D,b] logit strides
    sSel = (sel.stride(0), sel.stride(1), sel.stride(2))
    # snapshot (Sval_j/Sden_j) strides for the state-bwd ZdZ term — same (B, flat-k, v) layout as dSval.
    sSV = (dSval.stride(0), dSval.stride(2), dSval.stride(3))   # (B, flat-k=dqk-axis, v)
    sSD = (dSden.stride(0), dSden.stride(2))                    # (B, flat-k)
    sGD = (gdr.stride(0), gdr.stride(1), gdr.stride(2))
    state_common = dict(D=D, b=b, BB=BB, BT=chunk, BK=BK, BV=BV, BD=BD, BC=BC, ND=ND,
                        NDM=NDM, USE_G=use_g, GLA_FLOOR=_GLA_FLOOR, num_warps=4, num_stages=1)
    read_common = dict(GLOBAL=global_norm, PER_STATE=per_state, EPS=eps, D=D, b=b, BB=BB, BT=chunk,
                       BK=BK, BV=BV, BD=BD, BC=BC, ND=ND, NDM=NDM,
                       USE_G=use_g, GLA_FLOOR=_GLA_FLOOR, num_warps=4, num_stages=1)
    # #55 sqrt-checkpointing: the FORWARD saved only √NCH boundary checkpoints (ckpt_val/ckpt_den, passed
    # in). Here we recompute each segment's W pre-state snapshots on demand from its checkpoint (Pass 2,
    # below) — peak snapshot mem ~2√NCH·state instead of the old all-snapshot NCH·state (∝ L² when nc∝L), at
    # NO extra forward pass (the forward already paid the scan; it just stashed √NCH states). The reverse-scan
    # order is unchanged (global c=NCH-1..0), preserving carried adjoints and GLA decay. Segment recompute
    # seeds fp32 state from the same fp32 forward checkpoints, so fp32 matches the all-snapshot path exactly;
    # bf16 remains within normal rounding-scale tolerance. W matches the forward's via _kappa_ckpt_window.
    W = _kappa_ckpt_window(NCH) if checkpoint_window_override is None else max(1, int(checkpoint_window_override))
    seg_lo = NCH   # chunk range [seg_lo, seg_hi) of the currently-loaded segment snapshots (lazy, reverse)
    seg_val = seg_den = None
    for c in reversed(range(NCH)):
        if c < seg_lo:
            # crossed into the previous segment (reverse order): recompute its W pre-states from the
            # checkpoint (Pass 2, dense O(W·state)), reused for that segment's W reverse-scan steps.
            seg = c // W
            seg_lo, seg_hi = seg * W, min(seg * W + W, NCH)
            _sn, _sd, seg_val, seg_den, _sc = _kappa_routed_fwd(
                q, k, v, alpha if use_g else None, lr, lw, kap, D, b, sel, chunk,
                global_norm, per_state, eps, H, need_snapshots=True,
                state_in=(ckpt_val[seg], ckpt_den[seg]), c_lo=seg_lo, c_hi=seg_hi)
        Sval = seg_val[c - seg_lo]    # per-chunk pre-state (recomputed), contiguous [B,nc,dqk,dv] slice
        Sden = seg_den[c - seg_lo]
        gdw.zero_()
        # K1: state-update bwd (reads adjoint of Sval_{j+1}/Sden_{j+1}; produces dk,dv + state half of dw +
        # the USE_G carry/w_end da-pieces, using the chunk-start snapshot Sval_j/Sden_j for the Λ-coupling).
        _kappa_bwd_state[(B, NCBLK)](
            h, k, v, lr, lw, sel, Wg, Sval, Sden, dSval, dSden, dk, dvv, gdw, gda,
            L, d_model, dqk, dv, nc, c * chunk, H,
            *sH, *sB, *sV, *slo, *sSel, swg[0], swg[1], *sSV, *sSD, *sGD, *sga, **state_common)
        # K2: readout/den/d bwd (adds dw, produces dr,dq,dv-intra,dkappa; folds dSval/dSden adjoints).
        _kappa_bwd_read[(B, NCBLK)](
            h, q, k, v, lr, lw, sel, kap, Wg, Sval, Sden, dnum, dden,
            dSval, dSden, dq, dk, dvv, dkap, gdr, gdw, gda,
            L, d_model, dqk, dv, nc, c * chunk, H,
            *sH, *sB, *sV, kap.stride(0), kap.stride(1), *slo, *sSel, swg[0], swg[1], *sSV, *sSD,
            dnum.stride(0), dnum.stride(1), dnum.stride(2), dden.stride(0), dden.stride(1), *sGD, *sga,
            **read_common)
        # fold the transient gate-grads -> dlr,dlw (+ decay's dh,dWg); the dWr/dWw/dh router contraction is
        # the cuBLAS GEMM-backward in torch. USE_G reverse-cumsums gda in-kernel, then splits dld → dWg +
        # decay dh + decay write-grad.
        _routed_bwd_fold[(B, NCBLK)](
            h, lr, lw, sel, gdr, gdw, dh, dlr, dlw,
            Wg, dWg, gda,
            L, d_model, nc, c * chunk, H, *sH, *slo, *sSel,
            gdr.stride(0), gdr.stride(1), gdr.stride(2), dh.stride(0), dh.stride(1), dh.stride(2),
            swg[0], swg[1], *sga,
            D=D, b=b, BB=BB, BT=chunk, BC=BC, BD=BD, NDM=NDM,
            USE_G=use_g, GLA_FLOOR=_GLA_FLOOR, DLD_FROM_GDA=use_g, num_warps=4, num_stages=1)
    if not global_norm and not per_state and L > 0:
        # Token 0 has no carried state and only its diagonal intra term, so for every state c:
        # N_0^c = d_0^c * v_0. The normalized kappa gradient can therefore be formed as the small centered
        # residual d_0^c * <dnum_0, v_0 - out_0>, instead of relying on fp32 cancellation between the
        # numerator and denominator contributions. This touches only [B,nc] token-0 gates, never [L,nc].
        with torch.no_grad():
            def first_token_gates(logits):
                factors = [torch.softmax(logits[:, 0, lvl].float(), dim=-1) for lvl in range(D)]
                leaves = []
                for leaf in range(nc):
                    digs = [(leaf // (b ** (D - 1 - lvl))) % b for lvl in range(D)]
                    g_leaf = factors[0][:, digs[0]]
                    for lvl in range(1, D):
                        g_leaf = g_leaf * factors[lvl][:, digs[lvl]]
                    leaves.append(g_leaf)
                return torch.stack(leaves, dim=-1)

            r0 = first_token_gates(lr)
            w0 = first_token_gates(lw)
            d0 = (q[:, 0].float() * k[:, 0].float()).sum(-1, keepdim=True) * w0
            rt0 = r0 * (d0 + eps).pow(-kap[:, 0, None].float())
            z0 = (dnum[:, 0].float() * (v[:, 0].float() - out[:, 0].float())).sum(-1)
            dkap[:, 0].copy_(((-rt0 * d0 * (d0 + eps).log()).sum(-1) * z0).to(dkap.dtype))
    if use_g:
        return (dq[..., :dqk], dk[..., :dqk], dvv[..., :dv], dh, dlr, dlw, dkap, dWg)
    return (dq[..., :dqk], dk[..., :dqk], dvv[..., :dv], dh, dlr, dlw, dkap)


class _RoLARoutedKappaFn(torch.autograd.Function):
    """End-to-end differentiable FUSED kappa/per_state tree-routed RLA path. Forward runs the fused
    chunk-scan (`_kappa_routed_fwd`) and returns the normalized readout. Backward runs the reverse
    chunk-scan, recomputing the snapshots for fp-parity and folding the transient [BT,nc] gate-grads
    into the router. The [L,nc] gates AND the per-state den d / rescaled r_tilde are NEVER materialized."""
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, h, lr, lw, kap, alpha, D, b, chunk, global_norm, per_state, eps, H,
                save_checkpoints=False, Wg=None):
        # F2b: lr,lw:[BH,L,D,b] are the cuBLAS-GEMM routing logits (h·W in torch by the wrapper). The kernel
        # softmax+gathers them; backward emits dlr,dlw (autograd routes them through the GEMM → dWr,dWw,dh,db).
        cap = _CHUNK if Wg is not None else _CHUNK_FWD
        chunk = cap if chunk is None else min(chunk, cap)
        nc = b ** D
        sel = _build_sel(D, b, nc, q.device)
        q_dtype = q.dtype
        h_dtype = h.dtype
        use_g = alpha is not None
        if use_g != (Wg is not None):
            raise ValueError("GLA kappa requires both precomputed `alpha` and `Wg`; RLA passes neither.")
        q, k, v, h, lr, lw, kap = (x.contiguous() for x in (q, k, v, h, lr, lw, kap))
        alphac = alpha.contiguous() if use_g else q.new_empty(1, 1)
        Wgc = Wg.contiguous() if Wg is not None else None
        # #55: the wrapper passes save_checkpoints only when grad mode is enabled and a tensor input needs
        # grad. That distinction matters for no-grad inference: parameters may still have requires_grad=True,
        # but no backward can consume checkpoints, so no checkpoint tensor should be allocated.
        if save_checkpoints:
            # Strict training policy: compute the differentiable kappa readout with the same
            # den/state-sensitive precision policy backward uses, and save sparse boundaries in that pass.
            # h/lr/lw intentionally stay in the F4 routing dtype; Wg is fp32 for decay.
            q, k, v, kap = (x.float().contiguous() for x in (q, k, v, kap))
            alphaf = alphac.float().contiguous() if use_g else None
            Wgc = Wgc.float().contiguous() if Wgc is not None else None
            num, den, ckv, ckd, _ck = _kappa_routed_fwd(q, k, v, alphaf, lr, lw, kap, D, b, sel, chunk,
                                                        global_norm, per_state, eps, H,
                                                        save_checkpoints=True)
        else:
            num, den, _sv, _sd, _ck = _kappa_routed_fwd(q, k, v, alphac if use_g else None, lr, lw, kap,
                                                        D, b, sel, chunk, global_norm, per_state, eps, H)
            ckv = ckd = None
        den_f = den.float()
        out = num.float() / (den_f.unsqueeze(-1) + eps)
        if out.shape[1] > 0:
            # At the first token there is no prior state and the causal block contains only the diagonal,
            # so num_0 = den_0 * v_0 for every normalized mode. Use that identity directly; the generic
            # tiled numerator/denominator path is algebraically equivalent but leaves a tiny fp32 mismatch
            # that gets amplified in the kappa-gradient cancellation.
            out[:, 0].copy_(v[:, 0].float() * (den_f[:, 0] / (den_f[:, 0] + eps))[:, None])
        # #45: the GLA decay's saved activation is the [H,d_model] Wg (NOT a [L,nc] ld) — the saved-act win.
        # Save fp32 out/den so backward owns the normalization and can avoid the worst dκ num/den
        # cancellation at the custom autograd boundary.
        ctx.save_for_backward(q, k, v, h, lr, lw, kap, alphac, Wgc, ckv, ckd, out, den_f)
        ctx.D, ctx.b, ctx.chunk, ctx.H = D, b, chunk, H
        ctx.global_norm, ctx.per_state, ctx.eps = global_norm, per_state, eps
        ctx.q_dtype = q_dtype
        ctx.h_dtype = h_dtype
        ctx.use_g = use_g
        return out.to(q_dtype)

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do):
        q, k, v, h, lr, lw, kap, alpha, Wg, ckpt_val, ckpt_den, out, den = ctx.saved_tensors
        use_g = ctx.use_g
        D, b, chunk, H = ctx.D, ctx.b, ctx.chunk, ctx.H
        global_norm, per_state, eps = ctx.global_norm, ctx.per_state, ctx.eps
        nc = b ** D
        sel = _build_sel(D, b, nc, q.device)
        # F4: the routing LOGITS lr/lw stay in their input dtype (bf16) — the router build is a bounded
        # softmax (the kernels upcast to fp32 internally). q/k/v/kap keep fp32 (den/decay-sensitive path).
        q, k, v, kap = (x.float().contiguous() for x in (q, k, v, kap))
        h, lr, lw = (x.contiguous() for x in (h, lr, lw))
        Wgf = Wg.float().contiguous() if use_g else None
        # #55: _kappa_routed_bwd re-fits the chunk from the ctx ceiling via the shared _kappa_fit_chunk
        # (idempotent → same chunk + NCH + window as the forward), then recomputes each segment's snapshots
        # on demand from √NCH checkpoints that match the old dense recompute for q/k/v/kap state precision.
        # Reverse-scan order and routing operand dtypes are unchanged.
        den_e = den.unsqueeze(-1) + eps
        do = do.float().contiguous()
        dnum = do / den_e
        dden = -(do * out).sum(-1) / (den + eps)
        grads = _kappa_routed_bwd(
            q, k, v, h, lr, lw, kap, alpha.float().contiguous() if use_g else None,
            ckpt_val, ckpt_den, out.float(), dnum, dden,
            D, b, sel, chunk, global_norm, per_state, eps, H, Wg=Wgf)
        # _kappa_routed_bwd returns (dq,dk,dv,dh,dlr,dlw,dkap[,dWg]); dWg present iff use_g.
        dq, dk, dv, dh, dlr, dlw, dkap = grads[:7]
        dWg = grads[7] if use_g else None
        q_dtype = ctx.q_dtype
        h_dtype = ctx.h_dtype
        # forward arg order: q,k,v,h,lr,lw,kap,alpha,D,b,chunk,global_norm,per_state,eps,H,save_checkpoints,Wg
        return (dq.to(q_dtype), dk.to(q_dtype), dv.to(q_dtype), dh.to(h_dtype),
                dlr.to(lr.dtype), dlw.to(lw.dtype), dkap.to(q_dtype), None,
                None, None, None, None, None, None, None, None,
                None if dWg is None else dWg.to(Wg.dtype))


def _kappa_routed_readout(qf, kf, vf, hf, Wr, Ww, kapf, alpha, D, b, chunk_size,
                          global_norm, per_state, eps, b_r=None, b_w=None, Wg=None,
                          lr=None, lw=None, H=None):
    """Fused global/kappa/per_state tree-routed readout returning normalized out[BH,L,V],
    differentiable. kapf:[BH,L,1]. F2b: the per-level routing logits (h·Wr+b_r, h·Ww+b_w) are a cuBLAS
    GEMM done by the caller; the kernel only softmax+gathers them; dWr/dWw/dh/db flow through those
    precomputed logits. lr,lw:[BH,L,D,b] are required. Optional precomputed alpha:[BH,L] enables GLA;
    alpha=None is the RLA path."""
    kap = kapf.reshape(qf.shape[0], qf.shape[1]).contiguous()    # [BH,L]
    if lr is None or lw is None:
        raise ValueError("chunk_rola_routed kappa path requires precomputed `rl` and `wl` logits.")
    if (alpha is None) != (Wg is None):
        raise ValueError("GLA kappa path requires both precomputed `alpha` and `Wg`; RLA passes neither.")
    ckpt_inputs = (qf, kf, vf, hf, lr, lw, kap, alpha, Wg)
    save_checkpoints = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in ckpt_inputs)
    return _RoLARoutedKappaFn.apply(qf, kf, vf, hf, lr, lw, kap, alpha, D, b, chunk_size,
                                    global_norm, per_state, eps,
                                    H if H is not None else Wr.shape[0], save_checkpoints, Wg)


# --- GLA per-token log-decay floor — a GENUINE fp32 limit, NOT a tuning artifact (#33) ---------------
# The chunked GLA scan FACTORS the per-chunk decay as e^{a_i} (into the read gate) × e^{-a_j} (into the
# write gate), a=cumsum(ld) over a chunk, so the routing gram is ONE tl.dot R=(r·e^a)@(w·e^{-a})ᵀ and
# the state carry is w_end=w·e^{Λ-a}, decvec=e^Λ. The decay DIFFERENCE e^{a_i-a_j} on the causal
# triangle is in (0,1], but each UN-anchored FACTOR e^{±a} is unbounded (e^{BT·|FLOOR|} over BT rows),
# and fp32 overflows at ln(FLT_MAX)=88.72. `_decay_factors` RE-ANCHORS each factor to the per-state span
# midpoint (both factors ≤ e^{span/2}), so the fp32-safe span DOUBLES — the constraint is now
#     BT · |FLOOR| ≲ 2·88.72 ≈ 177.
# At BT=64 (the shipped GLA forward _CHUNK, now == RLA's _CHUNK_FWD): 64·2.5 = 160 < 177 (~11% headroom,
# measured finite). The differenced form e^{a_i-a_j} would be BT-free but can't be absorbed into the
# routed read/write matmul nor the cross-chunk state carry — re-anchoring is the in-matmul-compatible
# stabilization. SEPARATELY, the fused kappa-routed BACKWARD state kernel is SMEM-bound: besides the
# dominant [BT, BC·BK] fp32 tile, it carries enough BT×BT / BT×BC live state that BT=64 measures above the
# sm86 hard ceiling. `_kappa_fit_chunk` therefore adds a calibrated BT² live-set term and derives BT=32 on
# 99KB-class cards, while larger-SMEM devices can keep BT=64 if the same budget formula says it fits. This
# BT≤64 overflow ceiling is the numerical upper bound the SMEM derive then caps under; the two limits are
# independent.
#
# FLOOR=-2.5 ⇒ per-token retention ≥ e^{-2.5} = 8.2%/tok. KEPT (160<177 fits BT=64, but the headroom is
# thin — full floor removal would need a tighter per-tile anchor). The production layer's ld=log
# (alpha_chunk) CAN dip below this (alpha→0, write_gate→1), so the floor is NOT a structural no-op: it
# would alter a learned decay. Per the no-silent-rewrite rule the floor is LOUD — `_floor_ld` RAISES on
# out-of-range ld by default; opt into clamp-with-warning via ROLA_GLA_FLOOR_CLAMP=1. All GLA decay
# sites route through `_floor_ld`.
_GLA_FLOOR = -2.5   # per-token log-decay floor (retention ≥ 8.2%/tok); re-anchored span fits BT=64 (see above)
_GLA_FLOOR_CLAMP = os.environ.get('ROLA_GLA_FLOOR_CLAMP', '0') not in ('0', '', 'false', 'False')
_gla_floor_warned = False


def _floor_ld(ld):
    """Enforce the GLA log-decay floor (#33). ld below `_GLA_FLOOR` would overflow the factored fp32
    decay gram (e^{BT·|FLOOR|}); by default RAISE (no silent semantic rewrite). With
    ROLA_GLA_FLOOR_CLAMP=1, clamp to the floor and warn ONCE. Returns a tensor safe for the chunk
    kernels (dtype/contiguity left to the caller).

    Under torch.compile the data-dependent min-check is a graph break, so skip it while tracing. The
    shipping RoLA paths call this eagerly, so the loud min-check/raise runs inline for explicit-decay
    decode and scratch/reference callers."""
    global _gla_floor_warned
    if torch.compiler.is_compiling():
        return ld.clamp(min=_GLA_FLOOR) if _GLA_FLOOR_CLAMP else ld
    mn = ld.detach().min()
    if mn < _GLA_FLOOR:
        if not _GLA_FLOOR_CLAMP:
            raise ValueError(
                f"GLA log-decay ld below the fp32-safe floor _GLA_FLOOR={_GLA_FLOOR} "
                f"(min ld={mn.item():.4f}). The chunked GLA decay is factored e^{{±a}} and overflows "
                f"fp32 (ln FLT_MAX=88.72); the re-anchored factoring doubles the safe span to BT·|ld|≲177, "
                f"and at the shipped BT=64 that needs |ld|≤2.77, so the kernel "
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


# ============================================================================
# Private materialized-gate CPU/reference helpers. The production CUDA path is `chunk_rola_routed`;
# these helpers remain for CPU fallback and fp64 reference construction without exposing a public
# precomputed-gate chunk API.
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
        # prefill it must hand off to decode — a prefill→decode decay-rate gap.
        G = _floor_ld(gf).float().cumsum(1)
        wgt = wgt * (G[:, -1:, :] - G).exp()
    state = torch.einsum('btc,btd,bte->bcde', wgt, kf.float(), v1)      # [BH, nc, K, V+1]
    return state.reshape(B, H * state.shape[1], state.shape[2], state.shape[3])


# ============================================================================
# `chunk_rola_routed` — public TREE-ROUTED entry point. DIFFERENTIABLE end-to-end.
#
# Instead of precomputed gates r,w, this entry point takes the hidden state h + per-level router weights
# Wr,Ww and builds the routing gram IN-KERNEL (the [L,nc] gates are never materialized). Flat
# (D=1, b=nc) is the single-level softmax-router case.
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


def _rola_routed_readout(qf, kf, vf, hf, Wr, Ww, D, b, chunk_size, b_r=None, b_w=None, Wg=None,
                         lr=None, lw=None, H=None):
    """Folded tree-routed numerator-only readout. CUDA → in-kernel routed Triton kernels (gates AND the
    GLA decay ld never materialized); else → eager core on explicit gates (CPU reference path). Optional
    bias b_r/b_w. Optional per-head decay weight Wg:[H,d_model] (GLA) → the decayed routed readout; Wg=None
    is RLA. lr,lw:[BH,L,D,b] may be passed precomputed (the layer dedup)."""
    if qf.is_cuda:
        if Wg is not None:
            return rola_gla_routed_triton(qf, kf, vf, hf, Wr, Ww, Wg, D, b, chunk=chunk_size,
                                          b_r=b_r, b_w=b_w, lr=lr, lw=lw, H=H)
        return rola_rla_routed_triton(qf, kf, vf, hf, Wr, Ww, D, b, chunk=chunk_size, b_r=b_r, b_w=b_w,
                                      lr=lr, lw=lw, H=H)
    if lr is None or lw is None:
        raise ValueError("chunk_rola_routed raw CPU path requires precomputed `rl` and `wl` logits.")
    r = _gates_from_factor_logits(lr, D, b)
    w = _gates_from_factor_logits(lw, D, b)
    ld = _ld_from_Wg_torch(hf, w, Wg, Wr.shape[0]) if Wg is not None else None
    return _rola_chunk_core(qf, kf, vf, w, r, ld, chunk_size)


@input_guard
def chunk_rola_routed(q, k, v, h, Wr, Ww, D, b, norm='kappa', kappa=None, scale=None, eps=1e-5,
                      b_r=None, b_w=None, Wg=None, wl=None, rl=None, alpha=None):
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
    if wl is None or rl is None:
        raise ValueError("chunk_rola_routed requires precomputed `wl` and `rl` routing logits.")
    if Wg is not None and alpha is None:
        raise ValueError("GLA chunk_rola_routed requires precomputed scalar `alpha`.")
    if Wg is None and alpha is not None:
        raise ValueError("RLA chunk_rola_routed must not receive `alpha`.")
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

    qf, kf, vf = foldc(q) * scale, foldc(k), foldc(v)
    Wr = Wr.to(compute_dtype)
    Ww = Ww.to(compute_dtype)
    if b_r is not None:
        b_r, b_w = b_r.to(compute_dtype), b_w.to(compute_dtype)
    # Wg:[H,d_model] stays fp32 (the in-kernel decay exp is precision-sensitive). It is per-head (NOT folded
    # over the BH batch) — the kernel indexes head = fold-row % H, exactly like Wr/Ww. #45.
    Wgf = Wg.float() if Wg is not None else None

    # F2b dedup: the layer computes per-level routing logits (for z-loss + router gradients) once and
    # passes wl/rl ∈ [B,T,H,D,b]. The kernels only softmax+gather; there is no fallback GEMM here.

    def fold5(t):   # [B,T,H,D,b] -> [B*H,T,D,b]
        return t.permute(0, 2, 1, 3, 4).reshape(B * H, T, t.shape[-2], t.shape[-1]).to(compute_dtype)
    lr = fold5(rl)
    lw = fold5(wl)

    def fold3(t):   # [B,T,H] -> [B*H,T]
        return t.permute(0, 2, 1).reshape(B * H, T).float().contiguous()

    alphaf = fold3(alpha) if alpha is not None else None

    # The raw CUDA GLA path and GLA backward still consume h/Wg to form decay gradients. The normalized
    # no-grad/profile path consumes precomputed alpha and can skip folding h, avoiding a huge expanded clone.
    need_hf = (not q.is_cuda) or norm == 'raw' or (torch.is_grad_enabled() and Wg is not None)
    hf = foldc(h) if need_hf else qf.new_empty(qf.shape[0], qf.shape[1], 1)

    if norm == 'raw':
        return unfold(_rola_routed_readout(qf, kf, vf, hf, Wr, Ww, D, b, chunk_size,
                                           b_r=b_r, b_w=b_w, Wg=Wgf, lr=lr, lw=lw, H=H)).to(v.dtype)

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
        out = _kappa_routed_readout(qf, kf, vf, hf, Wr, Ww, kapf.to(compute_dtype), alphaf, D, b,
                                    chunk_size, global_norm=(norm == 'global'),
                                    per_state=(norm == 'per_state'), eps=eps,
                                    b_r=b_r, b_w=b_w, Wg=Wgf, lr=lr, lw=lw, H=H)
        return unfold(out.float()).to(v.dtype)

    # CPU reference path (qf not on CUDA) for all normalized norms: the per-state den pre-pass on
    # explicit gates + the eager-core numerator. ALL CUDA normalized norms (incl. 'global') route through
    # the fused in-kernel den path above, so `_tree_gates_torch` is never on the production CUDA path.
    # The bias is threaded here too (out-of-place gates) so the fallback honors softmax(h·W+b).
    rf = _gates_from_factor_logits(lr, D, b)
    wf = _gates_from_factor_logits(lw, D, b)
    rf, wf = rf.to(compute_dtype), wf.to(compute_dtype)
    gf = None
    if alphaf is not None:
        gf = _floor_ld((1.0 - wf.float() * (1.0 - alphaf.float().unsqueeze(-1))).clamp(min=1e-8).log())
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
