# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""Recurrent (decode) path for routed RoLA — a bespoke fused Triton decode kernel.

RoLA = per head, `nc` independent gated-linear-attention states: the write gate `w^c` scales the
writes into state `c`, the read gate `r^c` scales the reads, and the readout sums over `c` under the
chosen normalization. Mathematically this is `H*nc` virtual heads of `simple_gla` — but materializing
those virtual heads (the old `vh_expand`/`vh_combine` stub) replicates q/k across `H*nc` heads in HBM,
runs a generic per-virtual-head scan, applies the read-gate/kappa/norm in PyTorch glue, and CRASHES at
`B*H*nc > 65535` (the CUDA grid-z limit). ~52% of its decode latency was that PyTorch glue.

This module replaces the stub body with ONE bespoke decode kernel, **one program per (batch, REAL
head)** (grid `(B*H,)` — never `B*H*nc`), single-token-stepping:

  * the carried Kronecker state `Sᶜ[nc, dqk, dv+1]` (the `+1` column is the per-state denominator) is
    streamed from HBM in state-blocks (`BC` states × `BK` feature-rows × `BV` value-cols) — so any nc
    fits SRAM, exactly as the chunked inter tiles its state-blocks;
  * the SHARED query is read ONCE per token and reused for every state (never replicated);
  * the routing gates `r^c`/`w^c` are applied INLINE (read gate scales the readout, write gate scales
    the state update), and the kappa / per_state / global normalization is folded into the readout in
    registers — killing `vh_combine`;
  * per-state log-decay `g` (the GLA variant) decays the carried state before each write.

The signature/handoff is unchanged: the recurrent state is the RoLA Kronecker state `[N, H*nc, K, V+1]`,
layout-compatible with `chunk_rola`'s `output_final_state`, so a chunked prefill hands off bit-exactly
to this decode path.
"""

import torch
import triton
import triton.language as tl

from fla_rola.utils import input_guard

# nc / value tiling for the streamed state-block. BC states × BV value-cols per inner block; the
# feature dim dqk is loaded whole (BK=next_pow2(dqk) — dqk is small, 16/32). The combine (sum over nc +
# divide) reduces across BC-blocks in registers, so there is no cross-program reduction. NOT autotuned:
# the kernel MUTATES its state buffer in place, so the autotuner's repeated trial launches would stack
# multiple writes onto the same state — fixed launch params (num_warps/stages) instead.


@triton.jit
def _rola_decode_kernel(
    q_ptr, k_ptr, v_ptr, r_ptr, w_ptr, g_ptr, kap_ptr, s_ptr, o_ptr,
    T, dqk, dv, nc, scale, eps,
    sq_b, sq_t, sq_d, sv_b, sv_t, sv_d, sg_b, sg_t, sg_c,
    sk_b, sk_t, skap_b, skap_t,
    ss_b, ss_c, ss_k, ss_v, so_b, so_t, so_v,
    BK: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr, NCB: tl.constexpr,
    NORM: tl.constexpr, USE_G: tl.constexpr, USE_KAPPA: tl.constexpr):
    """One program per (batch, REAL head) — grid (B*H,). Steps T tokens; for each token streams the
    carried Kronecker state in nc-blocks (BC states × full feature/value tile), reading the SHARED q
    ONCE and folding the read gate + normalization inline. The per-state denominator rides in the
    `dv` (ones) column of the [dv+1]-wide value tile, so it decays/updates/reads alongside the value
    state in ONE pass (no separate den loop, no double-write). NORM: 0 global / 1 per_state / 2 kappa.
    The state buffer s_ptr is `[BH, nc, dqk, dv+1]` — handoff-compatible with chunk_rola's
    [N, H*nc, K, V+1] (last value col = the per-state denominator), mutated in place across tokens."""
    bh = tl.program_id(0)
    offs_k = tl.arange(0, BK)
    kmask = offs_k < dqk
    offs_c = tl.arange(0, BC)
    offs_v = tl.arange(0, BV)                          # full [dv+1] value tile (den rides in col dv)
    vmask = offs_v < dv
    dmask = offs_v == dv                               # the ones-column (per-state denominator)
    vsmask = offs_v < (dv + 1)                         # value cols AND the den col
    for t in range(T):
        # SHARED query/key — loaded ONCE per token, reused across every state (never replicated).
        b_q = tl.load(q_ptr + bh * sq_b + t * sq_t + offs_k * sq_d, mask=kmask, other=0.0).to(tl.float32) * scale
        b_k = tl.load(k_ptr + bh * sq_b + t * sq_t + offs_k * sq_d, mask=kmask, other=0.0).to(tl.float32)
        # value augmented with the ones-column at index dv (carries the per-state denominator).
        b_v = tl.load(v_ptr + bh * sv_b + t * sv_t + offs_v * sv_d, mask=vmask, other=0.0).to(tl.float32)
        b_v = tl.where(dmask, 1.0, b_v)                # [BV]; v at cols<dv, 1 at col dv, 0 beyond
        if USE_KAPPA:
            kap = tl.load(kap_ptr + bh * skap_b + t * skap_t).to(tl.float32)
        else:
            kap = 1.0
        num = tl.zeros([BV], dtype=tl.float32)         # Σ_c r̃ᶜ (q·Sᶜ) over the [dv+1] tile (den in col dv)
        den = 0.0                                      # Σ_c r̃ᶜ dᶜ
        for cb in range(NCB):
            cols_c = cb * BC + offs_c
            cmask = cols_c < nc
            b_r = tl.load(r_ptr + bh * sg_b + t * sg_t + cols_c * sg_c, mask=cmask, other=0.0).to(tl.float32)
            b_w = tl.load(w_ptr + bh * sg_b + t * sg_t + cols_c * sg_c, mask=cmask, other=0.0).to(tl.float32)
            # state slice S[BC, BK, BV] flattened to [BC*BK, BV] (BV spans the [dv+1] value+den tile).
            ckv = tl.reshape(cols_c[:, None] * dqk + offs_k[None, :], [BC * BK])              # [BC*BK]
            ckmask = tl.reshape(cmask[:, None] & kmask[None, :], [BC * BK])
            s_base = s_ptr + bh * ss_b
            b_s = tl.load(s_base + ckv[:, None] * ss_k + offs_v[None, :] * ss_v,
                          mask=ckmask[:, None] & vsmask[None, :], other=0.0)                  # [BC*BK, BV]
            wk2 = b_w[:, None] * b_k[None, :]                                                 # [BC, BK]
            if USE_G:
                b_g = tl.load(g_ptr + bh * sg_b + t * sg_t + cols_c * sg_c, mask=cmask, other=0.0).to(tl.float32)
                dec = tl.reshape(tl.broadcast_to(tl.exp(b_g)[:, None], [BC, BK]), [BC * BK])  # [BC*BK]
                b_s = b_s * dec[:, None]
            wk = tl.reshape(wk2, [BC * BK])                                                   # [BC*BK]
            # write current token: Sᶜ += w^c k ⊗ [v;1]  (the ones-col gives the +w^c k den update)
            b_s = b_s + wk[:, None] * b_v[None, :]
            tl.store(s_base + ckv[:, None] * ss_k + offs_v[None, :] * ss_v, b_s,
                     mask=ckmask[:, None] & vsmask[None, :])
            # SHARED read: P^c = q · Sᶜ → [BC, BV] per-state readout (col dv is the per-state den dᶜ).
            qk = tl.reshape(tl.broadcast_to(b_q[None, :], [BC, BK]), [BC * BK])               # [BC*BK]
            p = tl.sum(tl.reshape(qk[:, None] * b_s, [BC, BK, BV]), axis=1)                   # [BC, BV]
            p_den = tl.sum(tl.where(dmask[None, :], p, 0.0), axis=1)                          # [BC] per-state den
            # rescale the read gate per the norm, inline. The per-state den d^c is the canonical RAW
            # SIGNED mass Σ_{j≤t} w_j^c (φq·φk) — matching `_kappa_rescale` (chunk), `chunk_rola` torch,
            # and the naive oracle. The rescale r̃=r/(d+ε) | r·(d+ε)^{−κ} is only well-defined for d>0
            # (the kappa pow / per_state divide of a signed d is ill-posed). PRODUCTION GUARANTEES d≥0:
            # the layer is elu+1-only (φ>0) with softmax write gates (w≥0) ⇒ d≥0 structurally. We do NOT
            # tl.abs() d here: an abs would SILENTLY rewrite the normalizer for signed d, making decode
            # disagree with chunk/routed/oracle (all raw-signed). With raw d, every path behaves
            # identically (matched for d>0; identically ill-posed for signed d — loud, not divergent).
            if NORM == 1:                                                                    # per_state
                rt = b_r / (p_den + eps)
            elif NORM == 2:                                                                  # kappa
                rt = b_r * tl.exp(-kap * tl.log(p_den + eps))
            else:                                                                            # global
                rt = b_r
            rt = tl.where(cmask, rt, 0.0)
            num += tl.sum(p * rt[:, None], axis=0)
            den += tl.sum(p_den * rt)
        o = num / (den + eps)
        tl.store(o_ptr + bh * so_b + t * so_t + offs_v * so_v, o, mask=vmask)


def _decode_triton(q, k, v, r, w, g, kappa, norm, scale, eps, initial_state, output_final_state):
    """Bespoke fused decode. q,k:[BH,T,K] v:[BH,T,V] r,w:[BH,T,nc] g:[BH,T,nc]|None
    kappa:[BH,T,1]|None. State `[BH, nc, K, V+1]` carried in HBM (initial_state seeds it; mutated in
    place across tokens). Returns (o[BH,T,V], state|None)."""
    BH, T, dqk = q.shape
    dv = v.shape[-1]
    nc = r.shape[-1]
    norm_id = {'global': 0, 'per_state': 1, 'kappa': 2}[norm]
    BK = max(16, triton.next_power_of_2(dqk))
    BV = max(16, triton.next_power_of_2(dv + 1))      # full value tile INCLUDING the den (ones) column
    # state-block width: bound the resident [BC*BK, BV] register tile (~8K fp32 elems) so big dv/dqk
    # still fit — fewer states/program for fat value tiles. BC=8 measured best (smaller blocks = more
    # state-block loop iters but a lighter per-iter register tile; HBM-streaming-bound either way).
    # Floor 4 (tl ops need the BC dim).
    BC = max(4, min(8, 8192 // max(1, BK * BV)))
    nw, ns = 4, 2
    NCB = triton.cdiv(nc, BC)
    q, k, v, r, w = (x.contiguous() for x in (q, k, v, r, w))
    g = g.contiguous() if g is not None else None
    kappa = kappa.contiguous() if kappa is not None else None
    # State buffer [BH, nc, dqk, dv+1] (fp32). Seed from initial_state ([BH, nc, K, V+1]); else zeros.
    if initial_state is not None:
        S = initial_state.float().contiguous().clone()
    else:
        S = torch.zeros(BH, nc, dqk, dv + 1, device=q.device, dtype=torch.float32)
    o = torch.empty(BH, T, dv, device=q.device, dtype=torch.float32)
    # dummy ptrs for the optional inputs (triton needs a valid base even when unused).
    g_in = g if g is not None else q
    kap_in = kappa if kappa is not None else q
    sg = (r.stride(0), r.stride(1), r.stride(2))
    skap = (kap_in.stride(0), kap_in.stride(1))
    _rola_decode_kernel[(BH,)](
        q, k, v, r, w, g_in, kap_in, S, o,
        T, dqk, dv, nc, scale, eps,
        q.stride(0), q.stride(1), q.stride(2), v.stride(0), v.stride(1), v.stride(2), *sg,
        k.stride(0), k.stride(1), *skap,
        S.stride(0), S.stride(1), S.stride(2), S.stride(3), o.stride(0), o.stride(1), o.stride(2),
        BK=BK, BV=BV, BC=BC, NCB=NCB,
        NORM=norm_id, USE_G=g is not None, USE_KAPPA=kappa is not None,
        num_warps=nw, num_stages=ns)
    final_state = S.view(q.shape[0], nc, dqk, dv + 1) if output_final_state else None
    return o, final_state


@input_guard
def fused_recurrent_rola(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor | None = None,
    norm: str = 'kappa',
    kappa: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Recurrent RoLA readout (decode path) via a bespoke fused Triton kernel (one program per REAL
    head; the H*nc virtual heads are never materialized).

    Mirrors `chunk_rola`'s routed signature, plus the recurrent triad `initial_state` /
    `output_final_state` / `cu_seqlens`. `q`/`k` are the feature-mapped queries/keys (as for
    `chunk_rola`). `norm` ∈ {'global','per_state','kappa'}; 'kappa' rescales the read gate by
    `(dᶜ+eps)^{−κ}` per token (RAW signed den — the canonical convention; well-defined for d>0, which
    the production elu+1 layer guarantees). Returns `(o, final_state)`; `final_state` is `[N, H*nc, K, V+1]`
    Kronecker state (the per-state denominator carried in the `+1` column) when `output_final_state`
    else `None`, layout-compatible with `chunk_rola(output_final_state=True)` so a chunked prefill
    hands off to this decode path.
    """
    B, T, H, K = q.shape
    nc = r.shape[-1]
    dv = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    if cu_seqlens is not None:
        raise NotImplementedError("cu_seqlens (varlen) is not supported by the fused RoLA decode kernel.")

    def fold(t):
        return t.permute(0, 2, 1, 3).reshape(B * H, T, t.shape[-1]).contiguous() if t is not None else None

    qf, kf, vf, rf, wf = fold(q), fold(k), fold(v), fold(r), fold(w)
    gf = fold(g)
    if gf is not None:
        # Floor the per-token log-decay through the SAME guard the chunked GLA paths use (#33) so a
        # floored chunked prefill hands off to a decode that decays at the matching rate. Single-token
        # decay never overflows fp32, but an unfloored decode below `_GLA_FLOOR` would silently decay
        # FASTER than the (floored) prefill state it continues. Raises (or clamps, ROLA_GLA_FLOOR_CLAMP)
        # identically to the chunk path.
        from fla_rola.ops.rola.chunk import _floor_ld
        gf = _floor_ld(gf)
    # kappa is [B,T,H,1] -> [BH,T,1].
    kapf = fold(kappa) if (norm == 'kappa' and kappa is not None) else None
    # initial_state arrives as [N, H*nc, K, V+1] (== [B, H*nc, K, V+1]); view to [BH, nc, K, V+1].
    init = initial_state.view(B * H, nc, K, dv + 1) if initial_state is not None else None

    o, state = _decode_triton(qf, kf, vf, rf, wf, gf, kapf, norm, float(scale), eps,
                              init, output_final_state)
    o = o.view(B, H, T, dv).permute(0, 2, 1, 3).contiguous().to(v.dtype)        # [B,T,H,V]
    if state is not None:
        state = state.view(B, H * nc, K, dv + 1)
    return o, state
