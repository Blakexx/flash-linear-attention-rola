# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# RoLA tree-routed BACKWARD kernels (production). The in-kernel router-grad fold for the differentiable
# tree-routed path (`chunk_rola_routed` in chunk.py). Promoted VERBATIM from the validated prototype
# (the old proto_tree_routing.py): chunk.py's routed backward (`_rola_rla_routed_bwd` and the GLA twin)
# imports these four kernels and drives them at the PRODUCTION state-block width (BC=BG) — they are NOT
# a prototype, they are the live backward.
#
# RoLA's intra readout is  o = (G ⊙ R ⊙ causal) · v,  G = q·kᵀ (content gram over dqk), R = r·wᵀ
# (routing gram over nc states). Tree-routing builds R IN-KERNEL from the PRECOMPUTED per-level logits
# lr,lw ∈ [BH,L,D,b] (F2b: the per-head h·Wr/h·Ww d_model-contraction is a cuBLAS GEMM in the layer), so
# the [L,nc] read/write gates r,w (and their grads dr,dw) are NEVER materialized — the nc states are leaves
# of a D-level tree with branching b (b^D = nc) and the routing gram factorizes as a Hadamard of per-level
# rank-b grams  R = ⊙_i softmax(lr_i)·softmax(lw_i)ᵀ.
#
# BACKWARD chain (validated vs torch.autograd.grad(ref) to rel < 5e-3 bf16 for flat/square/tree):
#   d_o -> (intra gram bwd + inter recurrent reverse state-adjoint scan) -> dr,dw [BT,nc] TRANSIENT
#       -> tree factorization bwd (Sel gather + softmax jacobian) -> per-level LOGIT grads dlr,dlw ; plus
#          dq,dk,dv,d_h(decay-only). dWr/dWw/d_h(routing)/db then flow through the cuBLAS GEMM-backward
#          (autograd through `_router_logits`/`_factor_logits`) — NOT an in-kernel atomic h·W fold.
# The gate grads dr,dw live ONLY as transient [BT,nc] tiles, consumed in-kernel and folded — never [L,nc].
#
# Device helpers (shared by the four kernels below):
#   _build_factors : rebuilds the [BT,BC] read/write gate tiles for an nc-block by softmax+gathering the
#                    PRECOMPUTED logits (mirrors the forward `_build_rw_tile_logits`; never saved as
#                    [L,nc]). Also returns the per-level softmax factors fr/fw + selector sel (F3) so the
#                    fold can REUSE them.
#   _fold_level    : the router-grad fold for ONE tree level — takes the CACHED fr,fw,sel (no recompute;
#                    F3), gathers the transient [BT,BC] gate-grads to dfr,dfw [BT,b], applies the softmax
#                    jacobian, and atomic-adds the result into the per-level LOGIT-grad buffers dlr,dlw
#                    [BH,L,D,b] (the routing bias grad db + dWr/dWw/d_h come from the GEMM-backward). This
#                    is where dr,dw die.
# Backward kernels (one launch / per-chunk launches; see chunk.py drivers):
#   _bwd_intra_kernel       : intra gram bwd — dq,dk,dv-intra + the dr,dw-intra fold (one launch).
#   _bwd_inter_read_kernel  : inter readout bwd — dr,dq from o_inter; folds dS_read into the ds adjoint.
#   _bwd_inter_state_kernel : inter state-update bwd — dw,dk,dv from the state write (reads ds = adjoint
#                             of S_{j+1}); must run BEFORE the read kernel folds dS_read for this chunk.
#   _fold_kernel            : folds the combined inter gate-grad tiles (gdr,gdw [BT,nc]) -> dlr,dlw (+ the
#                             GLA decay's dWg/dh, which stay in-kernel).
# All four take the precomputed logits and the optional per-state log-decay (USE_G, the GLA variant), with
# the routing bias folded into the logits — exactly as the forward does.

import triton
import triton.language as tl

# --- backward kernels --------------------------------------------------------------------------
#
# In-kernel device helpers shared by the backward kernels:
#   _build_factors : rebuilds the [BT,BC] read/write gate tiles for an nc-block by softmax+gathering the
#                    PRECOMPUTED logits (mirrors the forward `_build_rw_tile_logits`; never saved as [L,nc]).
#                    Returns the per-level softmax factors fr/fw + selector sel (F3) for fold reuse.
#   _fold_level    : the router-grad fold for ONE tree level — takes the CACHED fr,fw,sel (no
#                    recompute; F3), gathers the transient [BT,BC] gate-grads to dfr,dfw [BT,b],
#                    applies the softmax jacobian, atomic-adds into the per-level LOGIT grads dlr,dlw.

@triton.jit
def _fold_level(dr_tile, dw_tile, r_tile, w_tile, fr, fw, sel,
                dlr_ptr, dlw_ptr,
                pid_b, rows, rmask, offs_bb, bmask,
                slo_b, slo_l, slo_lvl, slo_bb, lvl,
                BT: tl.constexpr, BB: tl.constexpr):
    # F2b: the router-grad fold for ONE tree level. fr,fw,sel are the per-level softmax factors + selector
    # CACHED by `_build_factors` (the SAME fp32 values). The gate-grads dr_tile,dw_tile are gathered to the
    # per-level logit grads dlr,dlw [BT,b] via the Sel gather + softmax jacobian, then EMITTED into the
    # [BH,L,D,b] dlogit buffers (atomic-add: the nc-blocks + the intra/inter folds accumulate into the same
    # per-(token,level,branch) slot). The d_model-contraction backward (dlogit -> dWr,dWw,dh,db) is now a
    # cuBLAS GEMM-backward in torch (autograd through `_router_logits`) — NOT an in-kernel atomic h·W fold.
    # The [L,nc] gate-grads never materialize; the dlogit buffer is the tiny [L,D,b] (tree/square) footprint.
    drr = dr_tile * r_tile
    dww_ = dw_tile * w_tile
    dfr = tl.dot(drr, tl.trans(sel)) / fr
    dfw = tl.dot(dww_, tl.trans(sel)) / fw
    dfr = tl.where(bmask[None, :], dfr, 0.0)
    dfw = tl.where(bmask[None, :], dfw, 0.0)
    dlr = fr * (dfr - tl.sum(fr * dfr, axis=1)[:, None])
    dlw = fw * (dfw - tl.sum(fw * dfw, axis=1)[:, None])
    off = pid_b * slo_b + rows[:, None] * slo_l + lvl * slo_lvl + offs_bb[None, :] * slo_bb
    m = rmask[:, None] & bmask[None, :]
    tl.atomic_add(dlr_ptr + off, dlr, mask=m)
    tl.atomic_add(dlw_ptr + off, dlw, mask=m)


@triton.jit
def _build_factors(lr_ptr, lw_ptr, sel_ptr, cols, cmask,
                   pid_b, rows, rmask, offs_bb, bmask,
                   slo_b, slo_l, slo_lvl, slo_bb, ssel_lvl, ssel_b, ssel_c,
                   D: tl.constexpr, BT: tl.constexpr,
                   BB: tl.constexpr, BC: tl.constexpr):
    # F2b: rebuild the [BT,BC] read/write gate tiles for an nc-block from the PRECOMPUTED per-level logits
    # lr,lw ∈ [BH,L,D,b] (the cuBLAS-GEMM stand-in for the in-kernel h·W d_model-contraction). Collects the
    # per-level softmax factors fr/fw + selector sel so the router-grad fold (`_fold_level`) REUSES them.
    # `tl.static_range` unrolls the D levels (D constexpr) so frs/fws/sels are compile-time tuples. The
    # routing bias is already folded into the logits. Mirrors the forward `_build_rw_tile_logits`.
    r_tile = tl.full([BT, BC], 1.0, dtype=tl.float32)
    w_tile = tl.full([BT, BC], 1.0, dtype=tl.float32)
    frs = ()
    fws = ()
    sels = ()
    neg = tl.full([BT, BB], float('-inf'), dtype=tl.float32)
    for lvl in tl.static_range(D):
        off = pid_b * slo_b + rows[:, None] * slo_l + lvl * slo_lvl + offs_bb[None, :] * slo_bb
        lr = tl.load(lr_ptr + off, mask=rmask[:, None] & bmask[None, :], other=0.0).to(tl.float32)
        lw = tl.load(lw_ptr + off, mask=rmask[:, None] & bmask[None, :], other=0.0).to(tl.float32)
        lr = tl.where(bmask[None, :], lr, neg)
        lw = tl.where(bmask[None, :], lw, neg)
        er = tl.exp(lr - tl.max(lr, axis=1)[:, None])
        ew = tl.exp(lw - tl.max(lw, axis=1)[:, None])
        fr = er / tl.sum(er, axis=1)[:, None]
        fw = ew / tl.sum(ew, axis=1)[:, None]
        sel = tl.load(sel_ptr + lvl * ssel_lvl + offs_bb[:, None] * ssel_b + cols[None, :] * ssel_c,
                      mask=bmask[:, None] & cmask[None, :], other=0.0)
        r_tile *= tl.dot(fr, sel)
        w_tile *= tl.dot(fw, sel)
        frs = frs + (fr,)
        fws = fws + (fw,)
        sels = sels + (sel,)
    r_tile = tl.where(cmask[None, :], r_tile, 0.0)
    w_tile = tl.where(cmask[None, :], w_tile, 0.0)
    return r_tile, w_tile, frs, fws, sels


# ============================================================================
# IN-KERNEL per-state log-decay (#45). The GLA scalar decay ld[t,c] is FUSED — computed in-kernel from a
# PER-HEAD decay weight Wg ∈ [H, d_model] (the layer's `w_g.weight`, a per-head scalar gate) + the write
# tile w that `_build_rw_tile_logits`/`_build_factors` already produces (by softmax+gathering the
# precomputed logits) — instead of being a precomputed [B,T,H,nc] INPUT (which materialized [L,nc]).
# Replicates `RoLA._log_decay` (layers/rola.py) EXACTLY:
#   alpha = sigmoid(h · Wg[head])                       # per-head scalar in (0,1); logsigmoid().exp()==sigmoid
#   ld[t,c] = clamp(log(clamp(1 - w[t,c]·(1-alpha[t]), 1e-8)), min=GLA_FLOOR)
# `_build_alpha` is the BD-blocked h·Wg sigmoid — the GENUINE in-kernel d_model contraction that REMAINS
# post-F2b (the routing h·Wr/h·Ww moved to a cuBLAS GEMM; this tiny [d_model]→1 per-head decay logit stays
# fused). wg_ptr is offset to THIS head (head = pid_b % H) by the CALLER. `_ld_from_w` turns the write
# tile into the floored ld. Defined here (the dependency-free bwd-kernel module) and imported by chunk.py
# (the forward kernels) so there is ONE definition — no per-file re-encode. RLA (USE_G=False) calls
# neither. The decay-grad fold (dWg, the extra dh, and dw_decay→dWw) lives in `_fold_kernel` below.
# ============================================================================
@triton.jit
def _build_alpha(h_ptr, wg_ptr, pid_b, rows, rmask, d_model,
                 sh_b, sh_l, sh_d, swg_d,
                 BT: tl.constexpr, BD: tl.constexpr, NDM: tl.constexpr):
    """alpha[BT] = sigmoid(Σ_d h[t,d]·Wg[head,d]) — the per-head scalar decay gate, IN-KERNEL from h + the
    head-offset Wg[d_model] row. BD-blocks d_model (NDM blocks) so any d_model fits SMEM; depends only on
    `rows` (NOT the nc-state column), so the caller computes it ONCE per token-block, reused across blocks."""
    z = tl.zeros([BT], dtype=tl.float32)
    for dm in range(NDM):
        offs_dm = dm * BD + tl.arange(0, BD)
        mmask = offs_dm < d_model
        hc = tl.load(h_ptr + pid_b * sh_b + rows[:, None] * sh_l + offs_dm[None, :] * sh_d,
                     mask=rmask[:, None] & mmask[None, :], other=0.0)
        wgc = tl.load(wg_ptr + offs_dm * swg_d, mask=mmask, other=0.0)
        z += tl.sum(hc * wgc[None, :], axis=1)
    return tl.sigmoid(z)


@triton.jit
def _ld_from_w(w_tile, alpha, cmask, GLA_FLOOR: tl.constexpr):
    """ld[BT,BC] = clamp(log(clamp(1 - w·(1-alpha), 1e-8)), min=GLA_FLOOR). `w_tile` is the write gate
    (already cmask→0 on the nc tail ⇒ m=1, ld=0 there, matching the old `tl.load(..., other=0.0)`). The
    floor matches the layer's `ld.clamp(min=_GLA_FLOOR)` and the kernels' `_floor_ld`."""
    m = 1.0 - w_tile * (1.0 - alpha[:, None])
    ld = tl.maximum(tl.log(tl.maximum(m, 1e-8)), GLA_FLOOR)
    return tl.where(cmask[None, :], ld, 0.0)


@triton.jit
def _decay_factors(a):
    """Re-anchored GLA decay factors from the per-state intra-chunk cumsum a[BT,BC] (per-token log-decay
    ld≤0 ⇒ a non-increasing down the chunk, a≤0). Returns (ea, ea_g, ena_g):

      ea    = e^a                  TRUE per-token decay (≤1) — the INTER readout (q·Sval/Sden carry) and
                                   the den's outer scale, where e^a multiplies the ABSOLUTELY-decayed
                                   cross-chunk state (no dual e^{-a} to cancel a re-anchor against).
      ea_g  = e^{a - a_ref}        ANCHORED read factor  ┐ the INTRA gram pair: ea_g·ena_g = e^{a_i-a_j},
      ena_g = e^{a_ref - a}        ANCHORED write factor ┘ IDENTICAL to e^a·e^{-a}, but with a_ref the
                                   per-state midpoint (max_t a + min_t a)/2 BOTH factors stay ≤ e^{span/2}
                                   (span = |Λ| over the chunk) instead of e^{-a}=e^{|Λ|} overflowing fp32.

    Stable-softmax / FlashAttention max-subtraction applied to the chunked-GLA decay: the product is
    mathematically invariant to a_ref (so the readout + every grad are unchanged to fp tolerance), but the
    per-tile exponents are re-anchored so they never overflow for any chunk size BT. (The decayed gram
    Rgram = (r·ea_g)·(w·ena_g)ᵀ still holds +inf in its anti-causal triangle — e^{a_i-a_j}, i<j — for large
    BT; the caller MUST `tl.where(causal, Rgram, 0)` before any 0·Rgram multiply, never `*causal`.)"""
    a_ref = 0.5 * (tl.max(a, axis=0) + tl.min(a, axis=0))   # [BC] per-state midpoint anchor
    ea = tl.exp(a)
    ea_g = tl.exp(a - a_ref[None, :])
    ena_g = tl.exp(a_ref[None, :] - a)
    return ea, ea_g, ena_g


# ============ INTRA backward kernel ============
@triton.jit
def _bwd_intra_kernel(
    h_ptr, q_ptr, k_ptr, v_ptr, lr_ptr, lw_ptr, sel_ptr, wg_ptr, do_ptr,
    dq_ptr, dk_ptr, dv_ptr, dlr_ptr, dlw_ptr, gda_ptr,
    L, d_model, dqk, dv, nc, H,
    sh_b, sh_l, sh_d, sq_b, sq_l, sq_d, sv_b, sv_l, sv_d,
    slo_b, slo_l, slo_lvl, slo_bb,
    ssel_lvl, ssel_b, ssel_c, swg_head, swg_d, sga_b, sga_l, sga_c,
    so_b, so_l, so_v,
    D: tl.constexpr, b: tl.constexpr, BB: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BD: tl.constexpr,
    BC: tl.constexpr, NCBLK: tl.constexpr, ND: tl.constexpr, NDM: tl.constexpr,
    USE_G: tl.constexpr = False, GLA_FLOOR: tl.constexpr = -2.5,
):
    # F2b: the routing factors are rebuilt from the PRECOMPUTED logits lr,lw (`_build_factors`); the
    # router-grad fold emits per-level LOGIT grads dlr,dlw (`_fold_level`) — the dWr/dWw/dh d_model
    # contraction is a cuBLAS GEMM-backward in torch. No in-kernel router-weight fold, no dh here (router
    # dh flows through the GEMM; intra carries no decay-dh — that lives in `_fold_kernel`).
    # USE_G (GLA, #30): the routing factors r_tile/w_tile feed the DECAYED intra gram rt=r·eᵃ, wt=w·e⁻ᵃ
    # (a = intra-chunk cumsum of ld over this block's c-columns), mirroring `_rola_routed_fwd_intra`.
    # The fold then consumes the routing-factor grads (drt·eᵃ, dwt·e⁻ᵃ), and the per-token log-decay
    # adjoints (dart_intra = drt·rt, da_wt = −dwt·wt) are accumulated into the persistent gda[B,L,nc]
    # buffer (the read/state kernels add their da-pieces; the driver reverse-cumsums gda → dld).
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    # The logits lr/lw and dlogits dlr/dlw are already per-head ([BH,L,D,b], indexed by pid_b). Only the
    # GLA decay weight Wg:[H,d_model] needs the per-head offset (BH fold is (B,H) -> head = pid_b % H).
    _hd = pid_b % H
    wg_ptr = wg_ptr + _hd * swg_head            # per-head decay weight (read-only here; #45)
    offs_t = tl.arange(0, BT)
    offs_v = tl.arange(0, BV)
    offs_k = tl.arange(0, BK)
    offs_bb = tl.arange(0, BB)
    offs_c = tl.arange(0, BC)
    bmask = offs_bb < b
    vmask = offs_v < dv
    kmask = offs_k < dqk
    rows = pid_t * BT + offs_t
    rmask = rows < L
    if USE_G:                                   # per-head decay gate alpha[BT], once (in-kernel ld, #45)
        alpha = _build_alpha(h_ptr, wg_ptr, pid_b, rows, rmask, d_model,
                             sh_b, sh_l, sh_d, swg_d, BT, BD, NDM)
    qc = tl.load(q_ptr + pid_b * sq_b + rows[:, None] * sq_l + offs_k[None, :] * sq_d,
                 mask=rmask[:, None] & kmask[None, :], other=0.0)
    kc = tl.load(k_ptr + pid_b * sq_b + rows[:, None] * sq_l + offs_k[None, :] * sq_d,
                 mask=rmask[:, None] & kmask[None, :], other=0.0)
    vc = tl.load(v_ptr + pid_b * sv_b + rows[:, None] * sv_l + offs_v[None, :] * sv_d,
                 mask=rmask[:, None] & vmask[None, :], other=0.0)
    doc = tl.load(do_ptr + pid_b * so_b + rows[:, None] * so_l + offs_v[None, :] * so_v,
                  mask=rmask[:, None] & vmask[None, :], other=0.0)
    G = tl.dot(qc, tl.trans(kc))
    causal = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
    dov = tl.dot(doc, tl.trans(vc)) * causal  # [BT,BT]
    # pass 1: full Rgram (USE_G: decayed gram rt·wtᵀ per block)
    Rgram = tl.zeros([BT, BT], dtype=tl.float32)
    for cb in range(NCBLK):
        cols = cb * BC + offs_c
        cmask = cols < nc
        r_tile, w_tile, _frs, _fws, _sels = _build_factors(
                                        lr_ptr, lw_ptr, sel_ptr, cols, cmask,
                                        pid_b, rows, rmask, offs_bb, bmask,
                                        slo_b, slo_l, slo_lvl, slo_bb, ssel_lvl, ssel_b, ssel_c,
                                        D, BT, BB, BC)
        if USE_G:
            ldc = _ld_from_w(w_tile, alpha, cmask, GLA_FLOOR)
            a = tl.cumsum(ldc, axis=0)
            _ea, ea_g, ena_g = _decay_factors(a)        # ANCHORED intra-gram pair (no e^{-a} overflow)
            r_tile = r_tile * ea_g
            w_tile = w_tile * ena_g
        Rgram += tl.dot(r_tile, tl.trans(w_tile))
    # Rgram's anti-causal triangle is +inf at large BT (e^{a_i-a_j}, i<j) → mask via where (Rgram_m) so
    # neither A nor dG forms 0·inf=NaN; bit-identical to `Rgram*causal` whenever Rgram is finite.
    Rgram_m = tl.where(causal, Rgram, 0.0)
    A = G * Rgram_m
    dv_acc = tl.dot(tl.trans(A).to(doc.dtype), doc)
    dRgram = dov * G
    dG = dov * Rgram_m
    dq_acc = tl.dot(dG.to(kc.dtype), kc)
    dk_acc = tl.dot(tl.trans(dG).to(qc.dtype), qc)
    tl.store(dv_ptr + pid_b * sv_b + rows[:, None] * sv_l + offs_v[None, :] * sv_d,
             dv_acc, mask=rmask[:, None] & vmask[None, :])
    tl.atomic_add(dq_ptr + pid_b * sq_b + rows[:, None] * sq_l + offs_k[None, :] * sq_d,
                  dq_acc, mask=rmask[:, None] & kmask[None, :])
    tl.atomic_add(dk_ptr + pid_b * sq_b + rows[:, None] * sq_l + offs_k[None, :] * sq_d,
                  dk_acc, mask=rmask[:, None] & kmask[None, :])
    # pass 2: per block, dr_tile/dw_tile from intra, fold (USE_G: chain through rt/wt + accumulate da)
    for cb in range(NCBLK):
        cols = cb * BC + offs_c
        cmask = cols < nc
        r_tile, w_tile, frs, fws, sels = _build_factors(
                                        lr_ptr, lw_ptr, sel_ptr, cols, cmask,
                                        pid_b, rows, rmask, offs_bb, bmask,
                                        slo_b, slo_l, slo_lvl, slo_bb, ssel_lvl, ssel_b, ssel_c,
                                        D, BT, BB, BC)
        if USE_G:
            ldc = _ld_from_w(w_tile, alpha, cmask, GLA_FLOOR)
            a = tl.cumsum(ldc, axis=0)
            _ea, ea_g, ena_g = _decay_factors(a)             # ANCHORED intra-gram pair (pure gram → both cancel)
            rt = r_tile * ea_g
            wt = w_tile * ena_g
            # grads w.r.t. the DECAYED gates rt/wt, then chain to the routing factors r/w for the fold. The
            # a_ref of the anchored pair cancels in every product below (drt·rt, dr=drt·ea_g, dw=dwt·ena_g,
            # da=drt·rt−dwt·wt), so each fold/grad is identical to the unanchored e^a/e^{-a} form.
            drt = tl.dot(dRgram.to(wt.dtype), wt)            # [BT,BC]
            dwt = tl.dot(tl.trans(dRgram).to(rt.dtype), rt)  # [BT,BC]
            dr_tile = drt * ea_g                             # routing-factor grad (→ fold)
            dw_tile = dwt * ena_g
            # da-pieces: +dart_intra (=drt·rt), −da_wt (=dwt·wt); accumulate into the persistent gda buffer.
            da = drt * rt - dwt * wt
            tl.atomic_add(gda_ptr + pid_b * sga_b + rows[:, None] * sga_l + cols[None, :] * sga_c,
                          tl.where(cmask[None, :], da, 0.0), mask=rmask[:, None] & cmask[None, :])
        else:
            dr_tile = tl.dot(dRgram.to(w_tile.dtype), w_tile)
            dw_tile = tl.dot(tl.trans(dRgram).to(r_tile.dtype), r_tile)
        for lvl in tl.static_range(D):
            _fold_level(dr_tile, dw_tile, r_tile, w_tile, frs[lvl], fws[lvl], sels[lvl],
                        dlr_ptr, dlw_ptr, pid_b, rows, rmask, offs_bb, bmask,
                        slo_b, slo_l, slo_lvl, slo_bb, lvl, BT, BB)


# ============ INTER readout backward (dr, dq from o_inter; accumulates dS_read into ds) ============
# Split from the state-update half to halve SMEM (only the readout [BT,BC*BK] tile lives here).
@triton.jit
def _bwd_inter_read_kernel(
    h_ptr, q_ptr, lr_ptr, lw_ptr, sel_ptr, wg_ptr, s_ptr, ds_ptr, do_ptr,
    dq_ptr, gdr_ptr, gda_ptr,
    L, d_model, dqk, dv, nc, t_start, H,
    sh_b, sh_l, sh_d, sq_b, sq_l, sq_d,
    slo_b, slo_l, slo_lvl, slo_bb,
    ssel_lvl, ssel_b, ssel_c, swg_head, swg_d, ss_b, ss_c, ss_k, ss_v,
    so_b, so_l, so_v, sg_b, sg_t, sg_c, sga_b, sga_l, sga_c,
    D: tl.constexpr, b: tl.constexpr, BB: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BD: tl.constexpr,
    BC: tl.constexpr, ND: tl.constexpr, NDM: tl.constexpr,
    USE_G: tl.constexpr = False, GLA_FLOOR: tl.constexpr = -2.5,
):
    # #58 grid (B·H, NCBLK): ONE program owns ONE nc-state-block cb=program_id(1) (was an in-program serial
    # `for cb` loop) — the occupancy widen (16→256 blocks). No cross-cb accumulator: dq is token-indexed
    # atomic, ds/gdr are cb-local (own-cols) stores, gda is cb-local-cols atomic → the fan-out needs no
    # recombine.
    # F2b: read factors rebuilt from logits (`_build_factors`); the read-gate grad is stored to gdr (the
    # `_fold_kernel` later folds gdr → dlogit). USE_G (GLA, #30): the inter readout uses the DECAYED read
    # gate rt=r·eᵃ (a=intra-chunk cumsum of ld). dq/dS_read flow through rt; the read routing-factor grad
    # is drt·eᵃ (→ gdr) and the read-gate decay adjoint dart_inter=drt·rt accumulates into gda.
    pid_b = tl.program_id(0)
    _hd = pid_b % H                              # per-head decay-weight slice (BH fold is (B,H))
    wg_ptr = wg_ptr + _hd * swg_head            # per-head decay weight (read-only here; #45)
    ND_V = (dv + BV - 1) // BV       # value-blocks (BV is the value TILE; ND_V==1 ⇒ the un-tiled kernel)
    offs_t = tl.arange(0, BT)
    offs_bb = tl.arange(0, BB)
    offs_c = tl.arange(0, BC)
    bmask = offs_bb < b
    rows = t_start + offs_t
    rmask = rows < L
    if USE_G:                                   # per-head decay gate alpha[BT], once (in-kernel ld, #45)
        alpha = _build_alpha(h_ptr, wg_ptr, pid_b, rows, rmask, d_model,
                             sh_b, sh_l, sh_d, swg_d, BT, BD, NDM)
    cb = tl.program_id(1)
    cols = cb * BC + offs_c
    cmask = cols < nc
    r_tile, w_tile, _frs, _fws, _sels = _build_factors(
                                    lr_ptr, lw_ptr, sel_ptr, cols, cmask,
                                    pid_b, rows, rmask, offs_bb, bmask,
                                    slo_b, slo_l, slo_lvl, slo_bb, ssel_lvl, ssel_b, ssel_c,
                                    D, BT, BB, BC)
    if USE_G:
        ldc = _ld_from_w(w_tile, alpha, cmask, GLA_FLOOR)
        a = tl.cumsum(ldc, axis=0)
        ea = tl.exp(a)
        rt_tile = r_tile * ea            # decayed read gate (used in the readout/dS_read/dq)
    else:
        rt_tile = r_tile
    # dr[BT,BC] sums over dqk → accumulate across BK-feature-blocks; dq[BT,BK] is per-block (offs_k).
    # M=do·s_flatᵀ is value-contracted → sum over vb; the dS_read store is per (BK,value)-block. The
    # [BC*BK,BV] s_flat slice stays bounded by BK·BV; rq[BT,BC*BK] is value-free, reused across vb.
    dr_inter = tl.zeros([BT, BC], dtype=tl.float32)   # grad w.r.t. rt (USE_G) | r (RLA)
    for d0 in range(ND):
        offs_k = d0 * BK + tl.arange(0, BK)
        kmask = offs_k < dqk
        qc = tl.load(q_ptr + pid_b * sq_b + rows[:, None] * sq_l + offs_k[None, :] * sq_d,
                     mask=rmask[:, None] & kmask[None, :], other=0.0)
        ckv = tl.reshape(cols[:, None] * dqk + offs_k[None, :], [BC * BK])
        ckmask = tl.reshape(cmask[:, None] & kmask[None, :], [BC * BK])
        rq = tl.reshape(rt_tile[:, :, None] * qc[:, None, :], [BT, BC * BK])
        M = tl.zeros([BT, BC * BK], dtype=tl.float32)
        for vb in range(ND_V):
            offs_v = vb * BV + tl.arange(0, BV)
            vmask = offs_v < dv
            doc = tl.load(do_ptr + pid_b * so_b + rows[:, None] * so_l + offs_v[None, :] * so_v,
                          mask=rmask[:, None] & vmask[None, :], other=0.0)
            s_flat = tl.load(s_ptr + pid_b * ss_b + ckv[:, None] * ss_k + offs_v[None, :] * ss_v,
                             mask=ckmask[:, None] & vmask[None, :], other=0.0)
            M += tl.dot(doc, tl.trans(s_flat).to(doc.dtype))  # [BT, BC*BK]
            # dS_read[c,k,v] = sum_t rt[t,c] q[t,k] do[t,v]; accumulate into ds (adjoint of S_j).
            dS_read = tl.dot(tl.trans(rq).to(doc.dtype), doc)  # [BC*BK, BV]
            dS_in = tl.load(ds_ptr + pid_b * ss_b + ckv[:, None] * ss_k + offs_v[None, :] * ss_v,
                            mask=ckmask[:, None] & vmask[None, :], other=0.0)
            tl.store(ds_ptr + pid_b * ss_b + ckv[:, None] * ss_k + offs_v[None, :] * ss_v,
                     dS_in + dS_read, mask=ckmask[:, None] & vmask[None, :])
        Mr = tl.reshape(M, [BT, BC, BK])
        dr_inter += tl.sum(Mr * qc[:, None, :], axis=2)
        dq_acc = tl.sum(Mr * rt_tile[:, :, None], axis=1)
        tl.atomic_add(dq_ptr + pid_b * sq_b + rows[:, None] * sq_l + offs_k[None, :] * sq_d,
                      dq_acc, mask=rmask[:, None] & kmask[None, :])
    if USE_G:
        # dr_inter is the grad w.r.t. rt; routing-factor grad = drt·eᵃ, da-piece dart_inter = drt·rt.
        dr_tile = tl.where(cmask[None, :], dr_inter * ea, 0.0)
        da = tl.where(cmask[None, :], dr_inter * rt_tile, 0.0)
        tl.atomic_add(gda_ptr + pid_b * sga_b + rows[:, None] * sga_l + cols[None, :] * sga_c,
                      da, mask=rmask[:, None] & cmask[None, :])
    else:
        dr_tile = tl.where(cmask[None, :], dr_inter, 0.0)
    tl.store(gdr_ptr + pid_b * sg_b + offs_t[:, None] * sg_t + cols[None, :] * sg_c,
             dr_tile, mask=rmask[:, None] & cmask[None, :])


# ============ INTER state-update backward (dw, dk, dv; uses ds = adjoint of S_{j+1}) ============
# Must run BEFORE the readout kernel writes dS_read into ds for this chunk (ds is still S_{j+1}'s adjoint).
@triton.jit
def _bwd_inter_state_kernel(
    h_ptr, k_ptr, v_ptr, lr_ptr, lw_ptr, sel_ptr, wg_ptr, sj_ptr, ds_ptr,
    dk_ptr, dv_ptr, gdw_ptr, gda_ptr,
    L, d_model, dqk, dv, nc, t_start, H,
    sh_b, sh_l, sh_d, sq_b, sq_l, sq_d, sv_b, sv_l, sv_d,
    slo_b, slo_l, slo_lvl, slo_bb,
    ssel_lvl, ssel_b, ssel_c, swg_head, swg_d, ss_b, ss_c, ss_k, ss_v,
    sg_b, sg_t, sg_c, sga_b, sga_l, sga_c,
    D: tl.constexpr, b: tl.constexpr, BB: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BD: tl.constexpr,
    BC: tl.constexpr, ND: tl.constexpr, NDM: tl.constexpr,
    USE_G: tl.constexpr = False, GLA_FLOOR: tl.constexpr = -2.5,
):
    # #58 grid (B·H, NCBLK): ONE program owns ONE nc-state-block cb=program_id(1) (was an in-program serial
    # `for cb` loop) — the occupancy widen (16→256 blocks). No cross-cb accumulator: dk/dv are token-indexed
    # atomic, gdw is a cb-local (own-cols) store, gda is cb-local-cols atomic → the fan-out needs no recombine.
    # F2b: write factors rebuilt from logits (`_build_factors`); the write-gate grad is stored to gdw (the
    # `_fold_kernel` later folds gdw → dlogit). USE_G (GLA, #30): the state write uses the DECAYED write
    # gate w_end=w·e^{Λ−a} (Λ=chunk-total ld); dk/dv flow through w_end; the write routing-factor grad +=
    # dw_end·e^{Λ−a} (→ gdw). da-pieces: −da_wend (=dw_end·w_end), and the chunk-total Λ coupling
    # dlam = e^Λ·Σ(S_j∘ds_in) − Σ_t da_wend placed on the LAST row of gda. ds_in MUST be the pre-decvec
    # adjoint of S_{j+1} (the driver applies the e^Λ state-carry decay to ds AFTER this kernel).
    pid_b = tl.program_id(0)
    _hd = pid_b % H                              # per-head decay-weight slice (BH fold is (B,H))
    wg_ptr = wg_ptr + _hd * swg_head            # per-head decay weight (read-only here; #45)
    ND_V = (dv + BV - 1) // BV       # value-blocks (BV is the value TILE; ND_V==1 ⇒ the un-tiled kernel)
    offs_t = tl.arange(0, BT)
    offs_bb = tl.arange(0, BB)
    offs_c = tl.arange(0, BC)
    bmask = offs_bb < b
    rows = t_start + offs_t
    rmask = rows < L
    if USE_G:                                   # per-head decay gate alpha[BT], once (in-kernel ld, #45)
        alpha = _build_alpha(h_ptr, wg_ptr, pid_b, rows, rmask, d_model,
                             sh_b, sh_l, sh_d, swg_d, BT, BD, NDM)
    cb = tl.program_id(1)
    cols = cb * BC + offs_c
    cmask = cols < nc
    r_tile, w_tile, _frs, _fws, _sels = _build_factors(
                                    lr_ptr, lw_ptr, sel_ptr, cols, cmask,
                                    pid_b, rows, rmask, offs_bb, bmask,
                                    slo_b, slo_l, slo_lvl, slo_bb, ssel_lvl, ssel_b, ssel_c,
                                    D, BT, BB, BC)
    if USE_G:
        ldc = _ld_from_w(w_tile, alpha, cmask, GLA_FLOOR)
        a = tl.cumsum(ldc, axis=0)
        Lam = tl.sum(tl.where(offs_t[:, None] == (BT - 1), a, 0.0), axis=0)   # [BC] chunk-total
        wend_tile = w_tile * tl.exp(Lam[None, :] - a)     # decayed write gate
    else:
        wend_tile = w_tile
    # dw[BT,BC] sums over dqk → accumulate across BK-feature-blocks; dk[BT,BK] per-block (offs_k); dv
    # [BT,BV] per value-block (offs_v, atomic). N=v·dSᵀ value-contracted → sum over vb; the [BC*BK,BV]
    # dS slice stays bounded by BK·BV. w_tile[BT,BC] / wk[BT,BC*BK] are value-free, reused across vb.
    dw_inter = tl.zeros([BT, BC], dtype=tl.float32)   # grad w.r.t. w_end (USE_G) | w (RLA)
    ZdZ = tl.zeros([BC], dtype=tl.float32)            # Σ_{k,v}(S_j ∘ ds_in), per state-column
    for d0 in range(ND):
        offs_k = d0 * BK + tl.arange(0, BK)
        kmask = offs_k < dqk
        kc = tl.load(k_ptr + pid_b * sq_b + rows[:, None] * sq_l + offs_k[None, :] * sq_d,
                     mask=rmask[:, None] & kmask[None, :], other=0.0)
        ckv = tl.reshape(cols[:, None] * dqk + offs_k[None, :], [BC * BK])
        ckmask = tl.reshape(cmask[:, None] & kmask[None, :], [BC * BK])
        wk = tl.reshape(wend_tile[:, :, None] * kc[:, None, :], [BT, BC * BK])
        N = tl.zeros([BT, BC * BK], dtype=tl.float32)
        for vb in range(ND_V):
            offs_v = vb * BV + tl.arange(0, BV)
            vmask = offs_v < dv
            vc = tl.load(v_ptr + pid_b * sv_b + rows[:, None] * sv_l + offs_v[None, :] * sv_d,
                         mask=rmask[:, None] & vmask[None, :], other=0.0)
            dS_in = tl.load(ds_ptr + pid_b * ss_b + ckv[:, None] * ss_k + offs_v[None, :] * ss_v,
                            mask=ckmask[:, None] & vmask[None, :], other=0.0)
            N += tl.dot(vc, tl.trans(dS_in).to(vc.dtype))  # [BT, BC*BK]
            dv_vb = tl.dot(wk.to(dS_in.dtype), dS_in)
            tl.atomic_add(dv_ptr + pid_b * sv_b + rows[:, None] * sv_l + offs_v[None, :] * sv_d,
                          dv_vb, mask=rmask[:, None] & vmask[None, :])
            if USE_G:
                # ZdZ_c += Σ_{k,v}(S_j[c,k,v] · ds_in[c,k,v]) — the e^Λ state-carry Λ-grad.
                sj = tl.load(sj_ptr + pid_b * ss_b + ckv[:, None] * ss_k + offs_v[None, :] * ss_v,
                             mask=ckmask[:, None] & vmask[None, :], other=0.0)
                sd = tl.where(ckmask[:, None] & vmask[None, :], sj * dS_in, 0.0)
                ZdZ += tl.sum(tl.reshape(tl.sum(sd, axis=1), [BC, BK]), axis=1)
        Nr = tl.reshape(N, [BT, BC, BK])
        dw_inter += tl.sum(Nr * kc[:, None, :], axis=2)
        dk_acc = tl.sum(Nr * wend_tile[:, :, None], axis=1)
        tl.atomic_add(dk_ptr + pid_b * sq_b + rows[:, None] * sq_l + offs_k[None, :] * sq_d,
                      dk_acc, mask=rmask[:, None] & kmask[None, :])
    if USE_G:
        # dw_inter is the grad w.r.t. w_end; routing-factor grad = dw_end·e^{Λ−a}; da_wend = −dw_end·w_end.
        dw_tile = tl.where(cmask[None, :], dw_inter * tl.exp(Lam[None, :] - a), 0.0)
        da_wend = tl.where(cmask[None, :], -dw_inter * wend_tile, 0.0)
        dlam = tl.exp(Lam) * ZdZ - tl.sum(da_wend, axis=0)                 # [BC]
        da = da_wend + tl.where(offs_t[:, None] == (BT - 1), dlam[None, :], 0.0)
        tl.atomic_add(gda_ptr + pid_b * sga_b + rows[:, None] * sga_l + cols[None, :] * sga_c,
                      tl.where(cmask[None, :], da, 0.0), mask=rmask[:, None] & cmask[None, :])
    else:
        dw_tile = tl.where(cmask[None, :], dw_inter, 0.0)
    tl.store(gdw_ptr + pid_b * sg_b + offs_t[:, None] * sg_t + cols[None, :] * sg_c,
             dw_tile, mask=rmask[:, None] & cmask[None, :])


# ============ FOLD kernel: gate-grad tiles [BT,nc] -> dWr,dWw,dh (router-grad fold) ============
@triton.jit
def _fold_kernel(
    h_ptr, lr_ptr, lw_ptr, sel_ptr, gdr_ptr, gdw_ptr,
    dh_ptr, dlr_ptr, dlw_ptr,
    wg_ptr, dwg_ptr, dld_ptr,
    L, d_model, nc, t_start, H,
    sh_b, sh_l, sh_d, slo_b, slo_l, slo_lvl, slo_bb,
    ssel_lvl, ssel_b, ssel_c, sg_b, sg_t, sg_c, sdh_b, sdh_l, sdh_d,
    swg_head, swg_d,
    D: tl.constexpr, b: tl.constexpr, BB: tl.constexpr,
    BT: tl.constexpr, BC: tl.constexpr, BD: tl.constexpr,
    NDM: tl.constexpr,
    USE_G: tl.constexpr = False, GLA_FLOOR: tl.constexpr = -2.5,
):
    # #58 grid (B·H, NCBLK): ONE program owns ONE nc-state-block cb=program_id(1) (was an in-program serial
    # `for cb` loop) — the occupancy widen (16→256 blocks). `_fold_level` atomic-adds dlr/dlw (cb-local cols
    # but the per-(token,level,branch) slot already sums across nc → already atomic, now more concurrent
    # adders). USE_G: the per-head decay-logit reduction dz[BT] is now a PER-PROGRAM PARTIAL — each program
    # folds its own dz into dWg/dh via the SAME post-pass `tl.atomic_add` (the recombine is the atomic).
    # F2b: folds the transient [BT,nc] gate-grads (gdr/gdw) into the per-level dlogit buffers dlr/dlw via
    # `_build_factors`(logits) + `_fold_level`(dlogits); the dWr/dWw/dh router contraction is a cuBLAS
    # GEMM-backward in torch. USE_G (GLA, #45): the per-state log-decay ld is computed in-kernel from Wg +
    # the write gate, so ITS gradient is folded HERE (the natural home — this kernel already rebuilds
    # w_tile and folds w → dlogit_w). dld_ptr carries the assembled ∂L/∂ld (the driver's reverse-cumsum
    # of gda) for this chunk [BT,nc]; we split it into ∂L/∂w (added to dw_tile → dlogit_w via the SAME tree
    # fold) and ∂L/∂z (z=h·Wg) → dWg, dh (the decay's dh STAYS in-kernel; h·Wg is the tiny [d_model]→1).
    pid_b = tl.program_id(0)
    _hd = pid_b % H                              # per-head decay-weight slice (BH fold is (B,H))
    wg_ptr = wg_ptr + _hd * swg_head
    dwg_ptr = dwg_ptr + _hd * swg_head
    offs_t = tl.arange(0, BT)
    offs_bb = tl.arange(0, BB)
    offs_c = tl.arange(0, BC)
    bmask = offs_bb < b
    rows = t_start + offs_t
    rmask = rows < L
    if USE_G:
        alpha = _build_alpha(h_ptr, wg_ptr, pid_b, rows, rmask, d_model,
                             sh_b, sh_l, sh_d, swg_d, BT, BD, NDM)
        dz = tl.zeros([BT], dtype=tl.float32)    # this block's ∂L/∂z PARTIAL → recombined via atomic_add into dWg/dh post-loop
    cb = tl.program_id(1)
    cols = cb * BC + offs_c
    cmask = cols < nc
    r_tile, w_tile, frs, fws, sels = _build_factors(
                                    lr_ptr, lw_ptr, sel_ptr, cols, cmask,
                                    pid_b, rows, rmask, offs_bb, bmask,
                                    slo_b, slo_l, slo_lvl, slo_bb, ssel_lvl, ssel_b, ssel_c,
                                    D, BT, BB, BC)
    dr_tile = tl.load(gdr_ptr + pid_b * sg_b + offs_t[:, None] * sg_t + cols[None, :] * sg_c,
                      mask=rmask[:, None] & cmask[None, :], other=0.0)
    dw_tile = tl.load(gdw_ptr + pid_b * sg_b + offs_t[:, None] * sg_t + cols[None, :] * sg_c,
                      mask=rmask[:, None] & cmask[None, :], other=0.0)
    if USE_G:
        # split the assembled ∂L/∂ld for this block into ∂L/∂w (→ dw_tile) and ∂L/∂z (→ dz).
        dld_tile = tl.load(dld_ptr + pid_b * sg_b + offs_t[:, None] * sg_t + cols[None, :] * sg_c,
                           mask=rmask[:, None] & cmask[None, :], other=0.0)
        one_minus_a = 1.0 - alpha[:, None]
        m = 1.0 - w_tile * one_minus_a
        active = (m > 1e-8) & (tl.log(tl.maximum(m, 1e-8)) >= GLA_FLOOR)
        dld = tl.where(cmask[None, :] & active, dld_tile, 0.0)
        dw_tile += dld * (-one_minus_a / m)                       # ∂ld/∂w = -(1-alpha)/m
        dz += tl.sum(dld * (w_tile / m), axis=1) * (alpha * (1.0 - alpha))  # ∂ld/∂alpha · ∂alpha/∂z
    for lvl in tl.static_range(D):
        _fold_level(dr_tile, dw_tile, r_tile, w_tile, frs[lvl], fws[lvl], sels[lvl],
                    dlr_ptr, dlw_ptr, pid_b, rows, rmask, offs_bb, bmask,
                    slo_b, slo_l, slo_lvl, slo_bb, lvl, BT, BB)
    if USE_G:
        # fold dz (the per-head decay-logit grad) into dWg[head,d] += Σ_t h·dz and dh[t,d] += dz·Wg[head,d].
        for dm in range(NDM):
            offs_dm = dm * BD + tl.arange(0, BD)
            mmask = offs_dm < d_model
            hc = tl.load(h_ptr + pid_b * sh_b + rows[:, None] * sh_l + offs_dm[None, :] * sh_d,
                         mask=rmask[:, None] & mmask[None, :], other=0.0)
            wgc = tl.load(wg_ptr + offs_dm * swg_d, mask=mmask, other=0.0)
            dzc = tl.where(rmask, dz, 0.0)
            tl.atomic_add(dwg_ptr + offs_dm * swg_d, tl.sum(hc * dzc[:, None], axis=0), mask=mmask)
            tl.atomic_add(dh_ptr + pid_b * sdh_b + rows[:, None] * sdh_l + offs_dm[None, :] * sdh_d,
                          dzc[:, None] * wgc[None, :], mask=rmask[:, None] & mmask[None, :])
