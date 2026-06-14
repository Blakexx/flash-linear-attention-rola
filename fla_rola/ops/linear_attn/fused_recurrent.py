# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang


import torch

from fla_rola.ops.linear_attn.utils import normalize_output
from fla_rola.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla


def fused_recurrent_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    normalize: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        # Mirror chunk_linear_attn (which defaults scale before use); upstream fused_recurrent omitted
        # this, so normalize_output(q * scale) below crashes on `Tensor * None` when normalize=True
        # (do_feature_map_norm). scale cancels in the normalized ratio; defaulting keeps the
        # fused_recurrent (L<=64) and chunk (L>64) paths consistent. Upstreamable one-liner.
        scale = k.shape[-1] ** -0.5
    o, final_state = fused_recurrent_simple_gla(
        q=q,
        k=k,
        v=v,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )
    if normalize:
        o = normalize_output(q * scale, k, o)
    return o, final_state
