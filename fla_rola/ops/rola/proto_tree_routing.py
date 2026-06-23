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
