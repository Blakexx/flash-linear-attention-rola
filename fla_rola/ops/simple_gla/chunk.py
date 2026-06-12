# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import warnings

import torch
import triton

from fla_rola.ops.common.chunk_h import chunk_bwd_dh, chunk_fwd_h
from fla_rola.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv, chunk_fwd_o
from fla_rola.ops.utils import chunk_local_cumsum, prepare_chunk_indices
from fla_rola.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


# ----------------------------------------------------------------------------
# RoLA routing (additive extension of simple_gla).
#
# Routed linear attention shares the content gram G=qkᵀ across `nc` states and
# modulates it by a routing gram R=Σ_c r_i^c w_j^c (+ optional per-state scalar
# decay). This is the *un-normalized* readout O = (G∘R∘causal) @ v — the FLA
# convention; the global denominator (ones-column) is the caller's job, exactly
# as rola.py already does it. Torch reference mirrors the verified shared-gram
# algorithms (rola_kernels.chunked_parallel / chunked_gla): the content gram is
# formed ONCE per chunk (the FLOP win), never materializing the L×nc product nor
# replicating q/k. Autograd gives exact grads for all of q,k,v,r,w,g. The Triton
# kernelization (chunk_fwd_h/chunk_fwd_o surgery) mirrors this and is gated against
# it via rola_fla_dev/check.py.
# ----------------------------------------------------------------------------
def _rola_chunk_core(q, k, v, w, r, ld, chunk_size):
    """Shared-gram routed readout on folded [BH, T, *] tensors. ld=None ⇒ RLA
    (additive, chunk-parallel cumsum scan); ld given ⇒ scalar-gated GLA (per-state
    decayed scan over chunks). Returns un-normalized O [BH, T, V]."""
    BH, T, K = q.shape
    V = v.shape[-1]
    nc = w.shape[-1]
    if ld is None:
        # RLA: vectorized chunk-parallel (mirrors chunked_parallel, no normalization).
        pad = (-T) % chunk_size
        if pad:
            q = torch.nn.functional.pad(q, (0, 0, 0, pad))
            k = torch.nn.functional.pad(k, (0, 0, 0, pad))
            v = torch.nn.functional.pad(v, (0, 0, 0, pad))
            w = torch.nn.functional.pad(w, (0, 0, 0, pad))
            r = torch.nn.functional.pad(r, (0, 0, 0, pad))
        Tp = T + pad
        nch = Tp // chunk_size
        C = chunk_size
        qc = q.view(BH, nch, C, K)
        kc = k.view(BH, nch, C, K)
        vc = v.view(BH, nch, C, V)
        rc = r.view(BH, nch, C, nc)
        wc = w.view(BH, nch, C, nc)
        G = torch.einsum('bnid,bnjd->bnij', qc, kc)
        R = torch.einsum('bnic,bnjc->bnij', rc, wc)
        causal = torch.tril(torch.ones(C, C, device=q.device, dtype=q.dtype))
        A = G * R * causal
        o_intra = torch.einsum('bnij,bnjv->bniv', A, vc)
        KV = torch.einsum('bnjc,bnjd,bnjv->bncdv', wc, kc, vc)        # [BH,nch,nc,K,V]
        S_before = torch.cumsum(KV, dim=1) - KV                       # exclusive prefix over chunks
        M = torch.einsum('bnic,bncdv->bnidv', rc, S_before)
        o_inter = torch.einsum('bnid,bnidv->bniv', qc, M)
        O = (o_intra + o_inter).reshape(BH, Tp, V)[:, :T]
        return O
    # GLA: per-state scalar decay, sequential decayed scan over chunks (mirrors chunked_gla).
    S = q.new_zeros(BH, nc, K, V)
    outs = []
    for c0 in range(0, T, chunk_size):
        c1 = min(c0 + chunk_size, T)
        Cc = c1 - c0
        qcc, kcc, vcc = q[:, c0:c1], k[:, c0:c1], v[:, c0:c1]
        rcc, wcc, ldc = r[:, c0:c1], w[:, c0:c1], ld[:, c0:c1]
        a = torch.cumsum(ldc, dim=1)                                  # chunk-local cumulative log-decay
        rt = rcc * torch.exp(a)                                       # r~ = r e^{a}
        wt = wcc * torch.exp(-a)                                      # w~ = w e^{-a}
        G = torch.einsum('bid,bjd->bij', qcc, kcc)
        D = torch.einsum('bic,bjc->bij', rt, wt)
        causal = torch.tril(torch.ones(Cc, Cc, device=q.device, dtype=q.dtype))
        Aw = G * D * causal
        o_intra = torch.einsum('bij,bjv->biv', Aw, vcc)
        M = torch.einsum('bic,bcdv->bidv', rt, S)                     # rt carries e^{a_i}
        o_inter = torch.einsum('bid,bidv->biv', qcc, M)
        outs.append(o_intra + o_inter)
        Lam = a[:, -1, :]                                             # chunk-total log-decay [BH,nc]
        w_end = wcc * torch.exp(Lam[:, None, :] - a)
        KV = torch.einsum('bjc,bjd,bjv->bcdv', w_end, kcc, vcc)
        S = torch.exp(Lam)[:, :, None, None] * S + KV
    return torch.cat(outs, dim=1)


def chunk_rola_fwd(q, k, v, r, w, g=None, scale=None, chunk_size=64):
    """RoLA routed forward. q,k:[B,T,H,K] v:[B,T,H,V] r,w:[B,T,H,nc] read/write
    gates; g:[B,T,H,nc] per-state log-decay (GLA) or None (RLA). Returns the
    un-normalized routed readout [B,T,H,V]."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    nc = r.shape[-1]
    if scale is None:
        scale = K ** -0.5

    def fold(t, d): return t.permute(0, 2, 1, 3).reshape(B * H, T, d)
    qf = fold(q, K) * scale
    kf, vf, rf, wf = fold(k, K), fold(v, V), fold(r, nc), fold(w, nc)
    ldf = fold(g, nc) if g is not None else None
    # Triton fast path on CUDA with feature dim within tl.dot block limits, on GPUs with
    # ampere-class shared memory. On small-smem devices (sm75/T4, 64KB) the Triton kernels only fit
    # at chunk=16/stages=1, where they MEASURE ~25% slower than the cuBLAS-backed torch core
    # (3.51 vs 2.82 it/s on the nc=256 MQAR cell) — so those devices dispatch to the torch core,
    # which also removes the sm75 d_v<=15 envelope. Device tiering, not a fallback: each hardware
    # class runs its measured-fastest verified implementation.
    from fla_rola.ops.simple_gla.rola import _BIG_SMEM
    if qf.is_cuda and K <= 64 and _BIG_SMEM:
        if ldf is None:
            from fla_rola.ops.simple_gla.rola import rola_rla_triton
            O = rola_rla_triton(qf, kf, vf, rf, wf, chunk=chunk_size)
        else:
            from fla_rola.ops.simple_gla.rola import rola_gla_triton
            O = rola_gla_triton(qf, kf, vf, rf, wf, ldf)
    else:
        O = _rola_chunk_core(qf, kf, vf, wf, rf, ldf, chunk_size)
    return O.reshape(B, H, T, V).permute(0, 2, 1, 3).contiguous()


def chunk_simple_gla_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h, ht = chunk_fwd_h(
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=None,
        gv=None,
        h0=initial_state,
        output_final_state=output_final_state,
        states_in_fp32=False,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
    return o, ht


def chunk_simple_gla_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_gamma: torch.Tensor,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # (SY 09/22) states_in_fp32 seems not affecting the error of dg but for safety, set to True
    h, _ = chunk_fwd_h(
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=None,
        gv=None,
        h0=initial_state,
        output_final_state=False,
        states_in_fp32=True,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dh, dh0 = chunk_bwd_dh(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=None,
        gv=None,
        do=do,
        h0=initial_state,
        dht=dht,
        scale=scale,
        states_in_fp32=True,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dq, dk, _, dg = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        h=h,
        do=do,
        dh=dh,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
    dv = chunk_bwd_dv(
        q=q,
        k=k,
        g=g,
        g_gamma=g_gamma,
        do=do,
        dh=dh,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
    return dq, dk, dv, dg, dh0


class ChunkSimpleGLAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        g_gamma,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        cu_seqlens_cpu,
    ):
        T = q.shape[1]
        chunk_size = min(64, max(16, triton.next_power_of_2(T)))

        chunk_indices = prepare_chunk_indices(
            cu_seqlens, chunk_size, cu_seqlens_cpu=cu_seqlens_cpu) if cu_seqlens is not None else None

        g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens,
                               chunk_indices=chunk_indices) if g is not None else None
        o, ht = chunk_simple_gla_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            g_gamma=g_gamma,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )
        ctx.save_for_backward(q, k, v, g, g_gamma, initial_state, chunk_indices)
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.cu_seqlens = cu_seqlens
        return o.to(q.dtype), ht

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        chunk_size, scale, cu_seqlens = ctx.chunk_size, ctx.scale, ctx.cu_seqlens
        q, k, v, g, g_gamma, initial_state, chunk_indices = ctx.saved_tensors
        dq, dk, dv, dg, dh0 = chunk_simple_gla_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            g_gamma=g_gamma,
            initial_state=initial_state,
            do=do,
            dht=dht,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )
        if g is not None:
            dg = chunk_local_cumsum(dg, chunk_size=chunk_size, reverse=True, cu_seqlens=cu_seqlens,
                                    chunk_indices=chunk_indices).to(g)
        else:
            dg = None
        return dq.to(q), dk.to(k), dv.to(v), dg, None, None, dh0, None, None, None


@torch.compiler.disable
def chunk_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    head_first: bool = False,
    r: torch.Tensor | None = None,
    w: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            Forget gates of shape `[B, T, H]`.
            Compared to GLA, the gating is head-wise instead of elementwise.
        g_gamma (torch.Tensor):
            Log decay of shape `[H]`.
            Head-wise data-independent decay is used if `g_gamma` is provided.
            Only one of `g` or `g_gamma` should be provided.
        scale (Optional[float]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
            This argument has been deprecated.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla_rola.ops.simple_gla import chunk_simple_gla
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = torch.randn(B, T, H, K, device='cuda')
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> g = F.logsigmoid(torch.randn(B, T, H, device='cuda'))
        >>> o, ht = chunk_simple_gla(
            q, k, v, g,
            initial_state=None,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_simple_gla(
            q, k, v, g,
            initial_state=None,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead.",
        )
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...].",
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if r is not None:
        # RoLA routed path: additive extension. r=read gate, w=write gate ([B,T,H,nc]);
        # g (if given) is the per-state log-decay ([B,T,H,nc]) for the scalar-gated (GLA)
        # variant, else None ⇒ RLA. Un-normalized readout; caller handles normalization.
        if w is None:
            raise ValueError("RoLA routed path requires both `r` (read) and `w` (write) gates.")
        if cu_seqlens is not None or initial_state is not None or output_final_state or g_gamma is not None:
            raise NotImplementedError(
                "RoLA routed path does not support cu_seqlens/initial_state/output_final_state/g_gamma yet.")
        if g is not None and g.shape != r.shape:
            raise ValueError(f"routed `g` must be per-state [B,T,H,nc] like r/w; got {tuple(g.shape)} vs {tuple(r.shape)}")
        T = q.shape[1]
        chunk_size = min(64, max(16, triton.next_power_of_2(T)))
        o = chunk_rola_fwd(q, k, v, r, w, g=g, scale=scale, chunk_size=chunk_size)
        return o, None
    o, final_state = ChunkSimpleGLAFunction.apply(
        q,
        k,
        v,
        g,
        g_gamma,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        cu_seqlens_cpu,
    )
    return o, final_state
