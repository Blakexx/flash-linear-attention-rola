# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# PROTOTYPE — in-kernel "tree-routing" for routed linear attention (forward only, exploratory).
#
# RoLA's intra readout is  o = (G ⊙ R ⊙ causal) · v,  where
#   G = q·kᵀ  (content gram, contract over dqk),
#   R = r·wᵀ  (routing gram over nc states).
# Normally r,w ∈ [L,nc] are PRECOMPUTED (materialized) gates. Tree-routing instead computes R
# IN-KERNEL from the hidden state h + router weights, never materializing the [L,nc] gates.
#
# The nc states are leaves of a tree with D levels and uniform branching b (b^D = nc). The routing
# gram factorizes as a Hadamard product of per-level rank-b grams:
#
#   R = R_1 ⊙ R_2 ⊙ ... ⊙ R_D,   R_i = softmax(h·Wr_i) · softmax(h·Ww_i)ᵀ
#
# with h ∈ [L,d_model] and Wr_i, Ww_i ∈ [d_model, b]. Three shapes (all same nc = b^D):
#   flat   : D=1, b=nc        (recovers standard full routing, R_1 = softmax(h·Wr)·softmax(h·Ww)ᵀ)
#   square : D=2, b=√nc
#   tree   : D=log_b(nc), b=2  (binary)
#
# Why it works: softmax of additively-factored logits = product of softmaxes, so the per-level
# softmaxes ARE the leaf-gate factors; the Hadamard of D rank-b grams has rank up to b^D = nc (full
# capacity, the "Hadamard lift"). The [L,nc] gates are NEVER formed — only the [BT,b] per-level
# factors (SRAM) and the [BT,BT] gram. This is the activation-floor win: the dominant L×nc gate
# tensor disappears from the forward graph.
#
# This file is a standalone prototype: it does NOT touch the production chunk.py. Forward only —
# backward (router weight-grads) is out of scope for this prototype.

import torch
import triton
import triton.language as tl

# --- helpers -------------------------------------------------------------------------------------

def _digits(leaf, D, b):
    """Big-endian base-b digit decomposition of a leaf index: digit i is the level-i branch.
    leaf = sum_i d_i * b^(D-1-i). Matches the reference flat-gate ordering below."""
    out = []
    for i in range(D):
        out.append((leaf // (b ** (D - 1 - i))) % b)
    return out


# --- the prototype Triton kernel -----------------------------------------------------------------
#
# Per (batch, chunk) program tile of BT rows:
#   1. build G = q·kᵀ over dqk (loop BK-blocks),
#   2. loop d0 in range(D): load h[BT,d_model], compute f_r = softmax(h·Wr[d0]),
#      f_w = softmax(h·Ww[d0]) (each [BT,b], in SRAM), accumulate R *= (f_r·f_wᵀ) (Hadamard, init 1),
#   3. A = G ⊙ R ⊙ causal,  o = A·v,  store [B,L,dv].
#
# b is padded to BB=max(16,next_pow2(b)) for tl.dot (dims must be >=16); the pad columns are masked
# to -inf before softmax so they contribute exactly 0 to the factor (and 0 to the gram). d_model is
# looped in BD-blocks so any d_model fits SRAM.

@triton.jit
def _tree_routed_intra_kernel(
    h_ptr, q_ptr, k_ptr, v_ptr, wr_ptr, ww_ptr, o_ptr,
    L, d_model, dqk, dv, nc,
    sh_b, sh_l, sh_d, sq_b, sq_l, sq_d, sv_b, sv_l, sv_d,
    swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
    so_b, so_l, so_v,
    D: tl.constexpr, b: tl.constexpr, BB: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BD: tl.constexpr,
    ND: tl.constexpr, NDM: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    offs_t = tl.arange(0, BT)
    offs_v = tl.arange(0, BV)
    offs_bb = tl.arange(0, BB)
    bmask = offs_bb < b
    rows = pid_t * BT + offs_t
    rmask = rows < L

    # 1. content gram G = q·kᵀ (loop BK-blocks of dqk).
    G = tl.zeros([BT, BT], dtype=tl.float32)
    for d0 in range(ND):
        offs_d = d0 * BK + tl.arange(0, BK)
        dmask = offs_d < dqk
        qc = tl.load(q_ptr + pid_b * sq_b + rows[:, None] * sq_l + offs_d[None, :] * sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        kc = tl.load(k_ptr + pid_b * sq_b + rows[:, None] * sq_l + offs_d[None, :] * sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        G += tl.dot(qc, tl.trans(kc))

    # 2. routing gram R, IN-KERNEL from h + router weights — Hadamard over D levels (init ones).
    R = tl.full([BT, BT], 1.0, dtype=tl.float32)
    for lvl in range(D):
        # logits_r/w = h[BT,d_model] · W[lvl][d_model,b]  → [BT, BB] (pad cols masked).
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
        # softmax over the b axis (pad cols → -inf so they vanish from both the factor and the gram).
        neg = tl.full([BT, BB], float('-inf'), dtype=tl.float32)
        lr = tl.where(bmask[None, :], lr, neg)
        lw = tl.where(bmask[None, :], lw, neg)
        er = tl.exp(lr - tl.max(lr, axis=1)[:, None])   # softmax over b (axis=1), masked cols → 0
        ew = tl.exp(lw - tl.max(lw, axis=1)[:, None])
        fr = er / tl.sum(er, axis=1)[:, None]  # [BT, BB], read-gate level factor
        fw = ew / tl.sum(ew, axis=1)[:, None]  # [BT, BB], write-gate level factor
        R *= tl.dot(fr, tl.trans(fw))          # Hadamard accumulate the rank-b level gram

    # 3. A = G ⊙ R ⊙ causal,  o = A·v.
    vc = tl.load(v_ptr + pid_b * sv_b + rows[:, None] * sv_l + offs_v[None, :] * sv_d,
                 mask=rmask[:, None] & (offs_v[None, :] < dv), other=0.0)
    causal = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
    A = G * R * causal
    o = tl.dot(A.to(vc.dtype), vc)
    tl.store(o_ptr + pid_b * so_b + rows[:, None] * so_l + offs_v[None, :] * so_v,
             o, mask=rmask[:, None] & (offs_v[None, :] < dv))


# --- chunked forward kernel (intra + inter state-scan) -------------------------------------------
#
# Per (batch, chunk) program, given the state S carried in from before this chunk:
#   1. build factors f_r, f_w [BT,b] per level (in-kernel, same as intra),
#   2. reconstruct the [BT,nc] read/write gate tiles r,w transiently via one-hot Sel maps
#      (Sel[lvl] : [b, nc] constant, gate = Π_lvl f[:,lvl] @ Sel[lvl]; gate is [BT,nc], never [L,nc]),
#   3. intra: o_intra = (G ⊙ (r·wᵀ) ⊙ causal) · v,
#   4. inter: o_interᵢ = Σ_c rᵢᶜ (qᵢ · Sᶜ)   using the carried-in state (causal),
#   5. update: Sᶜ += Σ_t wₜᶜ (kₜ ⊗ vₜ),  written back for the next chunk.
# nc is looped in BC-blocks so the [BT,nc] gate and the [nc,dqk,dv] state never need to fit whole.

@triton.jit
def _tree_routed_chunk_kernel(
    h_ptr, q_ptr, k_ptr, v_ptr, wr_ptr, ww_ptr, sel_ptr, s_ptr, o_ptr,
    L, d_model, dqk, dv, nc, t_start,
    sh_b, sh_l, sh_d, sq_b, sq_l, sq_d, sv_b, sv_l, sv_d,
    swr_lvl, swr_d, swr_b, sww_lvl, sww_d, sww_b,
    ssel_lvl, ssel_b, ssel_c,
    ss_b, ss_c, ss_k, ss_v,
    so_b, so_l, so_v,
    D: tl.constexpr, b: tl.constexpr, BB: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, BD: tl.constexpr,
    BC: tl.constexpr, NCBLK: tl.constexpr,
    ND: tl.constexpr, NDM: tl.constexpr,
):
    pid_b = tl.program_id(0)
    offs_t = tl.arange(0, BT)
    offs_v = tl.arange(0, BV)
    offs_k = tl.arange(0, BK)
    offs_bb = tl.arange(0, BB)
    offs_c = tl.arange(0, BC)
    bmask = offs_bb < b
    vmask = offs_v < dv
    kmask = offs_k < dqk
    rows = t_start + offs_t
    rmask = rows < L

    # load q,k,v tiles for this chunk (single dqk/dv block; prototype dims are small).
    qc = tl.load(q_ptr + pid_b * sq_b + rows[:, None] * sq_l + offs_k[None, :] * sq_d,
                 mask=rmask[:, None] & kmask[None, :], other=0.0)
    kc = tl.load(k_ptr + pid_b * sq_b + rows[:, None] * sq_l + offs_k[None, :] * sq_d,
                 mask=rmask[:, None] & kmask[None, :], other=0.0)
    vc = tl.load(v_ptr + pid_b * sv_b + rows[:, None] * sv_l + offs_v[None, :] * sv_d,
                 mask=rmask[:, None] & vmask[None, :], other=0.0)

    # content gram G = q·kᵀ.
    G = tl.dot(qc, tl.trans(kc))

    # accumulate intra + inter over nc-blocks; the [BT,nc] gates are rebuilt per block (transient).
    o_intra = tl.zeros([BT, BV], dtype=tl.float32)
    o_inter = tl.zeros([BT, BV], dtype=tl.float32)
    causal = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]

    for cb in range(NCBLK):
        cbase = cb * BC
        cols = cbase + offs_c
        cmask = cols < nc
        # build r_tile, w_tile : [BT, BC] transiently from the per-level factors.
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
            neg = tl.full([BT, BB], float('-inf'), dtype=tl.float32)
            lr = tl.where(bmask[None, :], lr, neg)
            lw = tl.where(bmask[None, :], lw, neg)
            er = tl.exp(lr - tl.max(lr, axis=1)[:, None])
            ew = tl.exp(lw - tl.max(lw, axis=1)[:, None])
            fr = er / tl.sum(er, axis=1)[:, None]   # [BT, BB]
            fw = ew / tl.sum(ew, axis=1)[:, None]
            # gather to nc-leaves of THIS block: gate_lvl[:, c] = f[:, digit_lvl(c)] = f @ Sel[lvl][:, cols]
            sel = tl.load(sel_ptr + lvl * ssel_lvl + offs_bb[:, None] * ssel_b + cols[None, :] * ssel_c,
                          mask=bmask[:, None] & cmask[None, :], other=0.0)   # [BB, BC] one-hot
            r_tile *= tl.dot(fr, sel)
            w_tile *= tl.dot(fw, sel)
        r_tile = tl.where(cmask[None, :], r_tile, 0.0)
        w_tile = tl.where(cmask[None, :], w_tile, 0.0)

        # intra: accumulate this block's contribution to the routing gram, then to o.
        Rgram = tl.dot(r_tile, tl.trans(w_tile))     # [BT, BT] partial (sum over cols of this block)
        A = G * Rgram * causal
        o_intra += tl.dot(A.to(vc.dtype), vc)

        # inter readout: o_interᵢ += Σ_{c in blk} rᵢᶜ (qᵢ·Sᶜ).
        #   load S[blk] : [BC, dqk, dv] as flattened [BC*BK, BV]; build RQ[t, c*BK+k]=r[t,c]*q[t,k];
        #   o_inter += RQ @ S_flat.
        # build RQ : [BT, BC*BK] via reshape of r_tile[:,:,None]*qc[:,None,:].
        rq = r_tile[:, :, None] * qc[:, None, :]        # [BT, BC, BK]
        rq = tl.reshape(rq, [BT, BC * BK])
        # flat (c,k) row index into the contiguous S[nc,dqk,dv]: row = c*dqk+k, row stride = ss_k = dv.
        ckv = (cols[:, None] * dqk + offs_k[None, :])   # [BC, BK]
        ckv = tl.reshape(ckv, [BC * BK])
        ckmask = tl.reshape(cmask[:, None] & kmask[None, :], [BC * BK])
        s_flat = tl.load(
            s_ptr + pid_b * ss_b + ckv[:, None] * ss_k + offs_v[None, :] * ss_v,
            mask=ckmask[:, None] & vmask[None, :], other=0.0)   # [BC*BK, BV]
        o_inter += tl.dot(rq.to(s_flat.dtype), s_flat)

        # state update: Sᶜ += Σ_t wₜᶜ (kₜ⊗vₜ).  KV[t, c*BK+k] = w[t,c]*k[t,k]; S_flat += KVᵀ @ v.
        wk = w_tile[:, :, None] * kc[:, None, :]        # [BT, BC, BK]
        wk = tl.reshape(wk, [BT, BC * BK])              # [BT, BC*BK]
        dS = tl.dot(tl.trans(wk).to(vc.dtype), vc)      # [BC*BK, BV]
        s_new = s_flat + dS
        tl.store(
            s_ptr + pid_b * ss_b + ckv[:, None] * ss_k + offs_v[None, :] * ss_v,
            s_new, mask=ckmask[:, None] & vmask[None, :])

    o = o_intra + o_inter
    tl.store(o_ptr + pid_b * so_b + rows[:, None] * so_l + offs_v[None, :] * so_v,
             o, mask=rmask[:, None] & vmask[None, :])


def _build_sel(D, b, nc, device):
    """One-hot level-to-leaf selection maps Sel[lvl][d, leaf] = 1 iff digit_lvl(leaf)==d.
    Tiny [D, b, nc] constant — reconstructs the [BT,nc] gate tile from [BT,b] factors in-kernel.
    This is NOT the [L,nc] gate; it carries no sequence dimension."""
    sel = torch.zeros(D, b, nc, device=device, dtype=torch.float32)
    for leaf in range(nc):
        for i, d in enumerate(_digits(leaf, D, b)):
            sel[i, d, leaf] = 1.0
    return sel


def tree_routed_chunked(h, q, k, v, Wr, Ww, D, b, chunk=64):
    """Full chunked tree-routed forward: o = o_intra + o_inter, with a sequential per-state
    state-scan across chunks. The [L,nc] gates are NEVER materialized — only [BT,nc] gate tiles
    are reconstructed transiently in-kernel from the [BT,b] per-level factors. Forward only.

    h  : [B, L, d_model]   q,k: [B, L, dqk]   v: [B, L, dv]   Wr,Ww: [D, d_model, b]
    Returns o : [B, L, dv].
    """
    B, L, d_model = h.shape
    dqk = q.shape[-1]
    dv = v.shape[-1]
    nc = b ** D
    assert Wr.shape == (D, d_model, b) and Ww.shape == (D, d_model, b)
    BV = max(16, triton.next_power_of_2(dv))
    BK = max(16, triton.next_power_of_2(dqk))
    BD = max(16, triton.next_power_of_2(d_model))
    BB = max(16, triton.next_power_of_2(b))
    BC = min(nc, 16)
    NCBLK = triton.cdiv(nc, BC)
    ND = triton.cdiv(dqk, BK)
    NDM = triton.cdiv(d_model, BD)
    NCH = triton.cdiv(L, chunk)
    h, q, k, v, Wr, Ww = (x.contiguous() for x in (h, q, k, v, Wr, Ww))
    sel = _build_sel(D, b, nc, q.device)
    # state S : [B, nc, dqk, dv], carried across chunks (state BEFORE current chunk).
    S = torch.zeros(B, nc, dqk, dv, device=q.device, dtype=torch.float32)
    o = torch.zeros(B, L, BV, device=q.device, dtype=torch.float32)
    # S is indexed in-kernel as flat row (c*dqk+k); ss_k is the stride of that flat row = S.stride(2).
    for c in range(NCH):
        t_start = c * chunk
        _tree_routed_chunk_kernel[(B,)](
            h, q, k, v, Wr, Ww, sel, S, o,
            L, d_model, dqk, dv, nc, t_start,
            h.stride(0), h.stride(1), h.stride(2),
            q.stride(0), q.stride(1), q.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            Wr.stride(0), Wr.stride(1), Wr.stride(2),
            Ww.stride(0), Ww.stride(1), Ww.stride(2),
            sel.stride(0), sel.stride(1), sel.stride(2),
            S.stride(0), S.stride(1), S.stride(2), S.stride(3),
            o.stride(0), o.stride(1), o.stride(2),
            D=D, b=b, BB=BB, BT=chunk, BK=BK, BV=BV, BD=BD,
            BC=BC, NCBLK=NCBLK, ND=ND, NDM=NDM,
            num_warps=4, num_stages=1,
        )
    return o[..., :dv]


def tree_routed_intra(h, q, k, v, Wr, Ww, D, b, chunk=64):
    """In-kernel tree-routed intra readout: o = (G ⊙ R ⊙ causal)·v with R built from h + routers.

    h  : [B, L, d_model]   pre-routing hidden state (router lives in kernel, weights are inputs)
    q,k: [B, L, dqk]       content keys/queries (gram contracts dqk)
    v  : [B, L, dv]
    Wr : [D, d_model, b]   per-level read-gate router weights
    Ww : [D, d_model, b]   per-level write-gate router weights
    Returns o_intra : [B, L, dv].  Forward only.
    """
    B, L, d_model = h.shape
    dqk = q.shape[-1]
    dv = v.shape[-1]
    nc = b ** D
    assert Wr.shape == (D, d_model, b) and Ww.shape == (D, d_model, b)
    BV = max(16, triton.next_power_of_2(dv))
    BK = max(16, triton.next_power_of_2(dqk))
    BD = max(16, triton.next_power_of_2(d_model))
    BB = max(16, triton.next_power_of_2(b))
    ND = triton.cdiv(dqk, BK)
    NDM = triton.cdiv(d_model, BD)
    NCH = triton.cdiv(L, chunk)
    h, q, k, v, Wr, Ww = (x.contiguous() for x in (h, q, k, v, Wr, Ww))
    o = torch.zeros(B, L, BV, device=q.device, dtype=torch.float32)
    _tree_routed_intra_kernel[(B, NCH)](
        h, q, k, v, Wr, Ww, o,
        L, d_model, dqk, dv, nc,
        h.stride(0), h.stride(1), h.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        Wr.stride(0), Wr.stride(1), Wr.stride(2),
        Ww.stride(0), Ww.stride(1), Ww.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        D=D, b=b, BB=BB, BT=chunk, BK=BK, BV=BV, BD=BD, ND=ND, NDM=NDM,
        num_warps=4, num_stages=2,
    )
    return o[..., :dv]


# --- torch references ----------------------------------------------------------------------------

def _flat_gates_from_tree(h, Wr, Ww, D, b):
    """Build the EXPLICIT [B,L,nc] gates r,w from the tree (router weights + hidden).
    r[..., leaf] = Π_i softmax(h·Wr[i])[..., digit_i(leaf)], same for w. This is the materialized
    path the fused kernel avoids — used both as the validation reference and the memory baseline."""
    B, L, _ = h.shape
    nc = b ** D
    # per-level softmax factors: [D, B, L, b]
    fr = torch.stack([torch.softmax(h @ Wr[i], dim=-1) for i in range(D)], dim=0)
    fw = torch.stack([torch.softmax(h @ Ww[i], dim=-1) for i in range(D)], dim=0)
    r = torch.ones(B, L, nc, device=h.device, dtype=fr.dtype)
    w = torch.ones(B, L, nc, device=h.device, dtype=fw.dtype)
    for leaf in range(nc):
        digs = _digits(leaf, D, b)
        for i, d in enumerate(digs):
            r[..., leaf] = r[..., leaf] * fr[i, ..., d]
            w[..., leaf] = w[..., leaf] * fw[i, ..., d]
    return r, w


def ref_routed_intra(h, q, k, v, Wr, Ww, D, b):
    """Materialized reference: form r,w ∈ [L,nc] explicitly, R = r·wᵀ, o = (G⊙R⊙causal)·v.
    Returns (o_ref, R) where R is the [B,L,L] routing gram for the rank check."""
    B, L, _ = h.shape
    hf = h.float()
    r, w = _flat_gates_from_tree(hf, Wr.float(), Ww.float(), D, b)   # [B,L,nc]
    G = q.float() @ k.float().transpose(-1, -2)                       # [B,L,L]
    R = r @ w.transpose(-1, -2)                                       # [B,L,L]
    causal = torch.tril(torch.ones(L, L, device=h.device))
    A = G * R * causal
    o = A @ v.float()
    return o, R, r, w


def ref_routed_chunked(h, q, k, v, Wr, Ww, D, b, chunk=64):
    """Full chunked RoLA forward reference: o = o_intra + o_inter.

    Builds the EXPLICIT [L,nc] gates r,w from the tree factorization and runs the standard
    chunked linear-attention readout with a per-state recurrent scan:

      State (per state c):  Sᶜ = Σ_j wⱼᶜ (kⱼ ⊗ vⱼ)   (nc-wide; the recurrence does NOT factor)
      Inter readout:        o_interᵢ = Σ_c rᵢᶜ (qᵢ · Sᶜ)   using state BEFORE the chunk (causal)
      Intra readout:        within-chunk causal triangle, R = r·wᵀ

    Two equivalent views are returned so the kernel can be validated against either:
    a fully-materialized [L,L] gram path and an explicit per-state recurrent scan. They agree
    by construction; the recurrent scan is the operational definition of the inter term."""
    B, L, _ = h.shape
    nc = b ** D
    dqk = q.shape[-1]
    dv = v.shape[-1]
    hf, qf, kf, vf = h.float(), q.float(), k.float(), v.float()
    r, w = _flat_gates_from_tree(hf, Wr.float(), Ww.float(), D, b)   # [B,L,nc]
    NCH = (L + chunk - 1) // chunk
    o = torch.zeros(B, L, dv, device=h.device, dtype=torch.float32)
    # running per-state state S: [B, nc, dqk, dv], state BEFORE the current chunk (causal).
    S = torch.zeros(B, nc, dqk, dv, device=h.device, dtype=torch.float32)
    for c in range(NCH):
        s, e = c * chunk, min((c + 1) * chunk, L)
        qc, kc, vc = qf[:, s:e], kf[:, s:e], vf[:, s:e]           # [B,T,*]
        rc, wc = r[:, s:e], w[:, s:e]                              # [B,T,nc]
        T = e - s
        # --- inter: read from the state carried in BEFORE this chunk ---
        #   o_interᵢ = Σ_c rᵢᶜ (qᵢ · Sᶜ).  qS[i,c,:] = qᵢ · Sᶜ  -> [B,T,nc,dv], weight by rᵢᶜ.
        qS = torch.einsum('btk,bckv->btcv', qc, S)                 # [B,T,nc,dv]
        o_inter = torch.einsum('btc,btcv->btv', rc, qS)            # [B,T,dv]
        # --- intra: within-chunk causal triangle ---
        G = qc @ kc.transpose(-1, -2)                              # [B,T,T]
        Rgram = rc @ wc.transpose(-1, -2)                          # [B,T,T]
        causal = torch.tril(torch.ones(T, T, device=h.device))
        o_intra = (G * Rgram * causal) @ vc                        # [B,T,dv]
        o[:, s:e] = o_intra + o_inter
        # --- update state: add this chunk's writes  Sᶜ += Σ_j wⱼᶜ (kⱼ⊗vⱼ) ---
        S = S + torch.einsum('btc,btk,btv->bckv', wc, kc, vc)
    return o
