# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""Recurrent (decode) path for routed RoLA via the virtual-head reduction.

RoLA = per head, `nc` independent gated-linear-attention states: the write gate `w^c` scales the
writes into state `c`, the read gate `r^c` scales the reads, and the readout sums over `c` under the
chosen normalization. That is exactly `H*nc` virtual heads of `simple_gla` — so the recurrent form is
a thin reduction onto `fused_recurrent_simple_gla` (each token's value carries a `+1` ones-column so
the per-state denominator is scanned alongside the numerator). This is the same math `chunk_rola` is
validated against, so it is exact.

NOTE: reference/stub decode path — correct but not bandwidth-optimal (it materializes `H*nc` virtual
heads). A bespoke `fused_recurrent_rola` kernel can later replace the body without changing the
signature. The recurrent state is the RoLA Kronecker state viewed as the virtual-head linear states,
`[N, H*nc, K, V+1]`, and is layout-compatible with what a state-returning `chunk_rola` must emit so a
chunked prefill can hand its state to this decode path.
"""

import torch

from fla_rola.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla


def vh_expand(q, k, v, w, g, nc):
    """[B,T,H,*] routed inputs -> virtual-head [B,T,H*nc,*]. q/k are copied per state; v is augmented
    with a den ones-column and scaled by the write gate; decay (if any) is broadcast per state."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    qv = q.unsqueeze(3).expand(B, T, H, nc, K).reshape(B, T, H * nc, K)
    kv = k.unsqueeze(3).expand(B, T, H, nc, K).reshape(B, T, H * nc, K)
    v1 = torch.cat([v, torch.ones_like(v[..., :1])], -1)                 # ones-col carries the den
    vv = (v1.unsqueeze(3) * w.unsqueeze(-1)).reshape(B, T, H * nc, V + 1)
    gv = g.reshape(B, T, H * nc).float() if g is not None else None
    return qv, kv, vv, gv


def vh_combine(o_aug, r, nc, norm, kappa, eps):
    """Per-virtual-head (num|den) output [B,T,H*nc,V+1] -> combined [B,T,H,V] under the read gate and
    the chosen normalization (global / per_state / kappa)."""
    B, T = o_aug.shape[0], o_aug.shape[1]
    V = o_aug.shape[-1] - 1
    H = o_aug.shape[2] // nc
    o = o_aug.view(B, T, H, nc, V + 1)                                  # [B,T,H,nc,V+1]
    num, den = o[..., :V], o[..., V]
    if norm == 'per_state':
        r = r / (den.abs() + eps)
    elif norm == 'kappa':
        kap = 1.0 if kappa is None else kappa
        r = r * (den.abs() + eps).pow(-kap)
    return (num * r.unsqueeze(-1)).sum(3) / ((den * r).sum(3).unsqueeze(-1) + eps)


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
    """Recurrent RoLA readout (decode path), exact via the H*nc virtual-head reduction.

    Mirrors `chunk_rola`'s routed signature, plus the recurrent triad `initial_state` /
    `output_final_state` / `cu_seqlens`. `q`/`k` are the feature-mapped queries/keys (as for
    `chunk_rola`). Returns `(o, final_state)`; `final_state` is the `[N, H*nc, K, V+1]` virtual-head
    state when `output_final_state` else `None`.
    """
    nc = r.shape[-1]
    if scale is None:
        scale = q.shape[-1] ** -0.5
    qv, kv, vv, gv = vh_expand(q, k, v, w, g, nc)
    o_aug, final_state = fused_recurrent_simple_gla(
        qv, kv, vv, g=gv, scale=scale,
        initial_state=initial_state, output_final_state=output_final_state, cu_seqlens=cu_seqlens,
    )
    o = vh_combine(o_aug.float(), r, nc, norm, kappa, eps)
    return o.to(v.dtype), final_state
