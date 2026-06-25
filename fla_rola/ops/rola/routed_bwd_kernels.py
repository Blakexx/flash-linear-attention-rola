# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# RoLA tree-routed BACKWARD kernels (production). The in-kernel router-grad fold for the differentiable
# tree-routed path (`chunk_rola_routed` in chunk.py). Promoted VERBATIM from the validated prototype
# (the old proto_tree_routing.py): chunk.py's routed backward (`_rola_rla_routed_bwd` and the GLA twin)
# imports these four kernels and drives them at the PRODUCTION state-block width (BC=BG) — they are NOT
# a prototype, they are the live backward.
#
# RoLA's intra readout is  o = (G ⊙ R ⊙ causal) · v,  G = q·kᵀ (content gram over dqk), R = r·wᵀ
# (routing gram over nc states). Tree-routing computes R IN-KERNEL from the hidden state h + per-level
# router weights, so the [L,nc] read/write gates r,w (and their grads dr,dw) are NEVER materialized —
# the nc states are leaves of a D-level tree with branching b (b^D = nc) and the routing gram factorizes
# as a Hadamard of per-level rank-b grams  R = ⊙_i softmax(h·Wr_i)·softmax(h·Ww_i)ᵀ.
#
# BACKWARD chain (validated vs torch.autograd.grad(ref) to rel < 5e-3 bf16 for flat/square/tree):
#   d_o -> (intra gram bwd + inter recurrent reverse state-adjoint scan) -> dr,dw [BT,nc] TRANSIENT
#       -> tree factorization bwd (Sel gather + softmax jacobian) -> dWr,dWw,d_h ; plus dq,dk,dv.
# The gate grads dr,dw live ONLY as transient [BT,nc] tiles, consumed in-kernel and folded — never [L,nc].
#
# Device helpers (shared by the four kernels below):
#   _build_factors : rebuilds the [BT,BC] read/write gate tiles for an nc-block (mirrors the forward's
#                    in-kernel factor construction; recomputed, never saved as [L,nc]).
#   _fold_level    : the router-grad fold for ONE tree level — recompute fr,fw,Sel, gather the transient
#                    [BT,BC] gate-grads to dfr,dfw [BT,b], apply the softmax jacobian to get the logit
#                    grads, and atomic-add into dWr,dWw,d_h (+ optional bias db). This is where dr,dw die.
# Backward kernels (one launch / per-chunk launches; see chunk.py drivers):
#   _bwd_intra_kernel       : intra gram bwd — dq,dk,dv-intra + the dr,dw-intra fold (one launch).
#   _bwd_inter_read_kernel  : inter readout bwd — dr,dq from o_inter; folds dS_read into the ds adjoint.
#   _bwd_inter_state_kernel : inter state-update bwd — dw,dk,dv from the state write (reads ds = adjoint
#                             of S_{j+1}); must run BEFORE the read kernel folds dS_read for this chunk.
#   _fold_kernel            : folds the combined inter gate-grad tiles (gdr,gdw [BT,nc]) -> dWr,dWw,d_h.
# All four honor the optional per-level routing bias b_r/b_w and the optional per-state log-decay (USE_G,
# the GLA variant) exactly as the forward does.

import triton
import triton.language as tl

# --- backward kernels --------------------------------------------------------------------------
#
# In-kernel device helpers shared by the backward kernels:
#   _build_factors : rebuilds the [BT,BC] read/write gate tiles for an nc-block (mirrors the
#                    forward's in-kernel factor construction; recomputed, never saved as [L,nc]).
#   _fold_level    : the router-grad fold for ONE tree level — recompute fr,fw,Sel, gather the
#                    transient [BT,BC] gate-grads to dfr,dfw [BT,b], apply the softmax jacobian to
#                    get the logit grads, and atomic-add into dWr,dWw,d_h. This is where dr,dw die.

@triton.jit
def _fold_level(dr_tile, dw_tile, r_tile, w_tile, cols, cmask,
                h_ptr, wr_ptr, ww_ptr, sel_ptr, dwr_ptr, dww_ptr, dh_ptr,
                pid_b, rows, rmask, offs_bb, bmask, d_model,
                sh_b, sh_l, sh_d, swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
                ssel_lvl, ssel_b, ssel_c, sdh_b, sdh_l, sdh_d,
                br_ptr, bw_ptr, sbr_lvl, sbr_b, sbw_lvl, sbw_b, dbr_ptr, dbw_ptr,
                lvl, BT: tl.constexpr, BB: tl.constexpr,
                BC: tl.constexpr, BD: tl.constexpr, NDM: tl.constexpr,
                HAS_BIAS: tl.constexpr):
    # recompute fr,fw for this level (incl. optional bias), fold dr_tile,dw_tile -> dWr,dWw,dh (+db).
    lr = tl.zeros([BT, BB], dtype=tl.float32)
    lw = tl.zeros([BT, BB], dtype=tl.float32)
    for dm in range(NDM):
        offs_dm = dm * BD + tl.arange(0, BD)
        mmask = offs_dm < d_model
        hc = tl.load(h_ptr + pid_b * sh_b + rows[:, None] * sh_l + offs_dm[None, :] * sh_d,
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
    fr = er / tl.sum(er, axis=1)[:, None]
    fw = ew / tl.sum(ew, axis=1)[:, None]
    sel = tl.load(sel_ptr + lvl * ssel_lvl + offs_bb[:, None] * ssel_b + cols[None, :] * ssel_c,
                  mask=bmask[:, None] & cmask[None, :], other=0.0)
    drr = dr_tile * r_tile
    dww_ = dw_tile * w_tile
    dfr = tl.dot(drr, tl.trans(sel)) / fr
    dfw = tl.dot(dww_, tl.trans(sel)) / fw
    dfr = tl.where(bmask[None, :], dfr, 0.0)
    dfw = tl.where(bmask[None, :], dfw, 0.0)
    dlr = fr * (dfr - tl.sum(fr * dfr, axis=1)[:, None])
    dlw = fw * (dfw - tl.sum(fw * dfw, axis=1)[:, None])
    for dm in range(NDM):
        offs_dm = dm * BD + tl.arange(0, BD)
        mmask = offs_dm < d_model
        hc = tl.load(h_ptr + pid_b * sh_b + rows[:, None] * sh_l + offs_dm[None, :] * sh_d,
                     mask=rmask[:, None] & mmask[None, :], other=0.0)
        wr = tl.load(wr_ptr + lvl * swr_lvl + offs_dm[:, None] * swr_d + offs_bb[None, :] * swr_b,
                     mask=mmask[:, None] & bmask[None, :], other=0.0)
        ww = tl.load(ww_ptr + lvl * sww_lvl + offs_dm[:, None] * sww_d + offs_bb[None, :] * sww_b,
                     mask=mmask[:, None] & bmask[None, :], other=0.0)
        tl.atomic_add(dwr_ptr + lvl * swr_lvl + offs_dm[:, None] * swr_d + offs_bb[None, :] * swr_b,
                      tl.dot(tl.trans(hc), dlr.to(hc.dtype)), mask=mmask[:, None] & bmask[None, :])
        tl.atomic_add(dww_ptr + lvl * sww_lvl + offs_dm[:, None] * sww_d + offs_bb[None, :] * sww_b,
                      tl.dot(tl.trans(hc), dlw.to(hc.dtype)), mask=mmask[:, None] & bmask[None, :])
        dh_blk = tl.dot(dlr.to(wr.dtype), tl.trans(wr)) + tl.dot(dlw.to(ww.dtype), tl.trans(ww))
        tl.atomic_add(dh_ptr + pid_b * sdh_b + rows[:, None] * sdh_l + offs_dm[None, :] * sdh_d,
                      dh_blk, mask=rmask[:, None] & mmask[None, :])
    if HAS_BIAS:
        # db_lvl[branch] = Σ_t dlr[t,branch] (the logit-grad summed over the chunk's tokens). Transient;
        # never an [L,nc] tensor. Pad branches (¬bmask) contribute 0 (dlr already 0 there via softmax).
        dbr = tl.sum(tl.where(rmask[:, None], dlr, 0.0), axis=0)
        dbw = tl.sum(tl.where(rmask[:, None], dlw, 0.0), axis=0)
        tl.atomic_add(dbr_ptr + lvl * sbr_lvl + offs_bb * sbr_b, dbr, mask=bmask)
        tl.atomic_add(dbw_ptr + lvl * sbw_lvl + offs_bb * sbw_b, dbw, mask=bmask)


@triton.jit
def _build_factors(h_ptr, wr_ptr, ww_ptr, sel_ptr, cols, cmask,
                   pid_b, rows, rmask, offs_bb, bmask, d_model,
                   sh_b, sh_l, sh_d, swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
                   ssel_lvl, ssel_b, ssel_c,
                   br_ptr, bw_ptr, sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                   D: tl.constexpr, BT: tl.constexpr,
                   BB: tl.constexpr, BC: tl.constexpr, BD: tl.constexpr, NDM: tl.constexpr,
                   HAS_BIAS: tl.constexpr):
    r_tile = tl.full([BT, BC], 1.0, dtype=tl.float32)
    w_tile = tl.full([BT, BC], 1.0, dtype=tl.float32)
    for lvl in range(D):
        lr = tl.zeros([BT, BB], dtype=tl.float32)
        lw = tl.zeros([BT, BB], dtype=tl.float32)
        for dm in range(NDM):
            offs_dm = dm * BD + tl.arange(0, BD)
            mmask = offs_dm < d_model
            hc = tl.load(h_ptr + pid_b * sh_b + rows[:, None] * sh_l + offs_dm[None, :] * sh_d,
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
        fr = er / tl.sum(er, axis=1)[:, None]
        fw = ew / tl.sum(ew, axis=1)[:, None]
        sel = tl.load(sel_ptr + lvl * ssel_lvl + offs_bb[:, None] * ssel_b + cols[None, :] * ssel_c,
                      mask=bmask[:, None] & cmask[None, :], other=0.0)
        r_tile *= tl.dot(fr, sel)
        w_tile *= tl.dot(fw, sel)
    r_tile = tl.where(cmask[None, :], r_tile, 0.0)
    w_tile = tl.where(cmask[None, :], w_tile, 0.0)
    return r_tile, w_tile


# ============ INTRA backward kernel ============
@triton.jit
def _bwd_intra_kernel(
    h_ptr, q_ptr, k_ptr, v_ptr, wr_ptr, ww_ptr, sel_ptr, ld_ptr, do_ptr,
    dq_ptr, dk_ptr, dv_ptr, dh_ptr, dwr_ptr, dww_ptr, gda_ptr,
    br_ptr, bw_ptr, dbr_ptr, dbw_ptr,
    L, d_model, dqk, dv, nc,
    sh_b, sh_l, sh_d, sq_b, sq_l, sq_d, sv_b, sv_l, sv_d,
    swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
    ssel_lvl, ssel_b, ssel_c, sg_b, sg_l, sg_c, sga_b, sga_l, sga_c,
    so_b, so_l, so_v, sdh_b, sdh_l, sdh_d,
    sbr_lvl, sbr_b, sbw_lvl, sbw_b,
    D: tl.constexpr, b: tl.constexpr, BB: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BD: tl.constexpr,
    BC: tl.constexpr, NCBLK: tl.constexpr, ND: tl.constexpr, NDM: tl.constexpr,
    HAS_BIAS: tl.constexpr, USE_G: tl.constexpr = False,
):
    # USE_G (GLA, #30): the routing factors r_tile/w_tile feed the DECAYED intra gram rt=r·eᵃ, wt=w·e⁻ᵃ
    # (a = intra-chunk cumsum of ld over this block's c-columns), mirroring `_rola_routed_fwd_intra`.
    # The fold then consumes the routing-factor grads (drt·eᵃ, dwt·e⁻ᵃ), and the per-token log-decay
    # adjoints (dart_intra = drt·rt, da_wt = −dwt·wt) are accumulated into the persistent gda[B,L,nc]
    # buffer (the read/state kernels add their da-pieces; the driver reverse-cumsums gda → dld).
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
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
        r_tile, w_tile = _build_factors(h_ptr, wr_ptr, ww_ptr, sel_ptr, cols, cmask,
                                        pid_b, rows, rmask, offs_bb, bmask, d_model,
                                        sh_b, sh_l, sh_d, swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
                                        ssel_lvl, ssel_b, ssel_c,
                                        br_ptr, bw_ptr, sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                                        D, BT, BB, BC, BD, NDM, HAS_BIAS)
        if USE_G:
            ldc = tl.load(ld_ptr + pid_b * sg_b + rows[:, None] * sg_l + cols[None, :] * sg_c,
                          mask=rmask[:, None] & cmask[None, :], other=0.0)
            a = tl.cumsum(ldc, axis=0)
            r_tile = r_tile * tl.exp(a)
            w_tile = w_tile * tl.exp(-a)
        Rgram += tl.dot(r_tile, tl.trans(w_tile))
    A = G * Rgram * causal
    dv_acc = tl.dot(tl.trans(A).to(doc.dtype), doc)
    dRgram = dov * G
    dG = dov * Rgram
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
        r_tile, w_tile = _build_factors(h_ptr, wr_ptr, ww_ptr, sel_ptr, cols, cmask,
                                        pid_b, rows, rmask, offs_bb, bmask, d_model,
                                        sh_b, sh_l, sh_d, swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
                                        ssel_lvl, ssel_b, ssel_c,
                                        br_ptr, bw_ptr, sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                                        D, BT, BB, BC, BD, NDM, HAS_BIAS)
        if USE_G:
            ldc = tl.load(ld_ptr + pid_b * sg_b + rows[:, None] * sg_l + cols[None, :] * sg_c,
                          mask=rmask[:, None] & cmask[None, :], other=0.0)
            a = tl.cumsum(ldc, axis=0)
            ea = tl.exp(a)
            ena = tl.exp(-a)
            rt = r_tile * ea
            wt = w_tile * ena
            # grads w.r.t. the DECAYED gates rt/wt, then chain to the routing factors r/w for the fold.
            drt = tl.dot(dRgram.to(wt.dtype), wt)            # [BT,BC]
            dwt = tl.dot(tl.trans(dRgram).to(rt.dtype), rt)  # [BT,BC]
            dr_tile = drt * ea                               # routing-factor grad (→ fold)
            dw_tile = dwt * ena
            # da-pieces: +dart_intra (=drt·rt), −da_wt (=dwt·wt); accumulate into the persistent gda buffer.
            da = drt * rt - dwt * wt
            tl.atomic_add(gda_ptr + pid_b * sga_b + rows[:, None] * sga_l + cols[None, :] * sga_c,
                          tl.where(cmask[None, :], da, 0.0), mask=rmask[:, None] & cmask[None, :])
        else:
            dr_tile = tl.dot(dRgram.to(w_tile.dtype), w_tile)
            dw_tile = tl.dot(tl.trans(dRgram).to(r_tile.dtype), r_tile)
        for lvl in range(D):
            _fold_level(dr_tile, dw_tile, r_tile, w_tile, cols, cmask,
                        h_ptr, wr_ptr, ww_ptr, sel_ptr, dwr_ptr, dww_ptr, dh_ptr,
                        pid_b, rows, rmask, offs_bb, bmask, d_model,
                        sh_b, sh_l, sh_d, swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
                        ssel_lvl, ssel_b, ssel_c, sdh_b, sdh_l, sdh_d,
                        br_ptr, bw_ptr, sbr_lvl, sbr_b, sbw_lvl, sbw_b, dbr_ptr, dbw_ptr,
                        lvl, BT, BB, BC, BD, NDM, HAS_BIAS)


# ============ INTER readout backward (dr, dq from o_inter; accumulates dS_read into ds) ============
# Split from the state-update half to halve SMEM (only the readout [BT,BC*BK] tile lives here).
@triton.jit
def _bwd_inter_read_kernel(
    h_ptr, q_ptr, wr_ptr, ww_ptr, sel_ptr, ld_ptr, s_ptr, ds_ptr, do_ptr,
    dq_ptr, gdr_ptr, gda_ptr, br_ptr, bw_ptr,
    L, d_model, dqk, dv, nc, t_start,
    sh_b, sh_l, sh_d, sq_b, sq_l, sq_d,
    swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
    ssel_lvl, ssel_b, ssel_c, sgl_b, sgl_l, sgl_c, ss_b, ss_c, ss_k, ss_v,
    so_b, so_l, so_v, sg_b, sg_t, sg_c, sga_b, sga_l, sga_c, sbr_lvl, sbr_b, sbw_lvl, sbw_b,
    D: tl.constexpr, b: tl.constexpr, BB: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BD: tl.constexpr,
    BC: tl.constexpr, NCBLK: tl.constexpr, ND: tl.constexpr, NDM: tl.constexpr,
    HAS_BIAS: tl.constexpr, USE_G: tl.constexpr = False,
):
    # USE_G (GLA, #30): the inter readout uses the DECAYED read gate rt=r·eᵃ (a=intra-chunk cumsum of ld).
    # dq/dS_read flow through rt; the read routing-factor grad is drt·eᵃ (→ gdr) and the read-gate decay
    # adjoint dart_inter=drt·rt is accumulated into the persistent gda buffer (the driver reverse-cumsums).
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
        r_tile, w_tile = _build_factors(h_ptr, wr_ptr, ww_ptr, sel_ptr, cols, cmask,
                                        pid_b, rows, rmask, offs_bb, bmask, d_model,
                                        sh_b, sh_l, sh_d, swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
                                        ssel_lvl, ssel_b, ssel_c,
                                        br_ptr, bw_ptr, sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                                        D, BT, BB, BC, BD, NDM, HAS_BIAS)
        if USE_G:
            ldc = tl.load(ld_ptr + pid_b * sgl_b + rows[:, None] * sgl_l + cols[None, :] * sgl_c,
                          mask=rmask[:, None] & cmask[None, :], other=0.0)
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
    h_ptr, k_ptr, v_ptr, wr_ptr, ww_ptr, sel_ptr, ld_ptr, sj_ptr, ds_ptr,
    dk_ptr, dv_ptr, gdw_ptr, gda_ptr, br_ptr, bw_ptr,
    L, d_model, dqk, dv, nc, t_start,
    sh_b, sh_l, sh_d, sq_b, sq_l, sq_d, sv_b, sv_l, sv_d,
    swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
    ssel_lvl, ssel_b, ssel_c, sgl_b, sgl_l, sgl_c, ss_b, ss_c, ss_k, ss_v,
    sg_b, sg_t, sg_c, sga_b, sga_l, sga_c, sbr_lvl, sbr_b, sbw_lvl, sbw_b,
    D: tl.constexpr, b: tl.constexpr, BB: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BD: tl.constexpr,
    BC: tl.constexpr, NCBLK: tl.constexpr, ND: tl.constexpr, NDM: tl.constexpr,
    HAS_BIAS: tl.constexpr, USE_G: tl.constexpr = False,
):
    # USE_G (GLA, #30): the state write uses the DECAYED write gate w_end=w·e^{Λ−a} (Λ=chunk-total ld);
    # dk/dv flow through w_end; the write routing-factor grad += dw_end·e^{Λ−a} (→ gdw). da-pieces:
    # −da_wend (=dw_end·w_end), and the chunk-total Λ coupling dlam = e^Λ·Σ(S_j∘ds_in) − Σ_t da_wend
    # placed on the LAST row of gda. ds_in MUST be the pre-decvec adjoint of S_{j+1} (the driver applies
    # the e^Λ state-carry decay to ds AFTER this kernel, before the read kernel folds dS_read).
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
        r_tile, w_tile = _build_factors(h_ptr, wr_ptr, ww_ptr, sel_ptr, cols, cmask,
                                        pid_b, rows, rmask, offs_bb, bmask, d_model,
                                        sh_b, sh_l, sh_d, swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
                                        ssel_lvl, ssel_b, ssel_c,
                                        br_ptr, bw_ptr, sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                                        D, BT, BB, BC, BD, NDM, HAS_BIAS)
        if USE_G:
            ldc = tl.load(ld_ptr + pid_b * sgl_b + rows[:, None] * sgl_l + cols[None, :] * sgl_c,
                          mask=rmask[:, None] & cmask[None, :], other=0.0)
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
    h_ptr, wr_ptr, ww_ptr, sel_ptr, gdr_ptr, gdw_ptr,
    dh_ptr, dwr_ptr, dww_ptr, br_ptr, bw_ptr, dbr_ptr, dbw_ptr,
    L, d_model, nc, t_start,
    sh_b, sh_l, sh_d, swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
    ssel_lvl, ssel_b, ssel_c, sg_b, sg_t, sg_c, sdh_b, sdh_l, sdh_d,
    sbr_lvl, sbr_b, sbw_lvl, sbw_b,
    D: tl.constexpr, b: tl.constexpr, BB: tl.constexpr,
    BT: tl.constexpr, BC: tl.constexpr, BD: tl.constexpr,
    NCBLK: tl.constexpr, NDM: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    offs_t = tl.arange(0, BT)
    offs_bb = tl.arange(0, BB)
    offs_c = tl.arange(0, BC)
    bmask = offs_bb < b
    rows = t_start + offs_t
    rmask = rows < L
    for cb in range(NCBLK):
        cols = cb * BC + offs_c
        cmask = cols < nc
        r_tile, w_tile = _build_factors(h_ptr, wr_ptr, ww_ptr, sel_ptr, cols, cmask,
                                        pid_b, rows, rmask, offs_bb, bmask, d_model,
                                        sh_b, sh_l, sh_d, swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
                                        ssel_lvl, ssel_b, ssel_c,
                                        br_ptr, bw_ptr, sbr_lvl, sbr_b, sbw_lvl, sbw_b,
                                        D, BT, BB, BC, BD, NDM, HAS_BIAS)
        dr_tile = tl.load(gdr_ptr + pid_b * sg_b + offs_t[:, None] * sg_t + cols[None, :] * sg_c,
                          mask=rmask[:, None] & cmask[None, :], other=0.0)
        dw_tile = tl.load(gdw_ptr + pid_b * sg_b + offs_t[:, None] * sg_t + cols[None, :] * sg_c,
                          mask=rmask[:, None] & cmask[None, :], other=0.0)
        for lvl in range(D):
            _fold_level(dr_tile, dw_tile, r_tile, w_tile, cols, cmask,
                        h_ptr, wr_ptr, ww_ptr, sel_ptr, dwr_ptr, dww_ptr, dh_ptr,
                        pid_b, rows, rmask, offs_bb, bmask, d_model,
                        sh_b, sh_l, sh_d, swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
                        ssel_lvl, ssel_b, ssel_c, sdh_b, sdh_l, sdh_d,
                        br_ptr, bw_ptr, sbr_lvl, sbr_b, sbw_lvl, sbw_b, dbr_ptr, dbw_ptr,
                        lvl, BT, BB, BC, BD, NDM, HAS_BIAS)
