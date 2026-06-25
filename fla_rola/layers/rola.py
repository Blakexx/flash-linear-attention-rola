# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""RoLA — Routed Linear Attention (canonical FLA layer).

Shared q/k/v/o projections + learned dense read/write routing over `states_per_head`
recurrent states, with a feature-mapped linear-attention inner kernel. The whole
normalization recipe (per-state denominator pre-pass, read-gate rescale, shared-Gram
numerator-only readout, divide) lives in the `chunk_rola` op — the layer only
projects, routes, applies the feature map φ, and (for the scalar-GLA variant)
computes the per-state log-decay; then a single `chunk_rola` call:

    x -> (q,k,v) projections -> φ(q),φ(k) ; softmax write/read gates [B,T,H,nc]
      -> chunk_rola(qf,kf,v, r=read, w=write, g=log_decay|None, norm, kappa) -> [B,T,H,V]
      -> o_proj

Two inner kernels (both route through `chunk_rola`, the paper-shipping cells):
  * 'rla'        : un-decayed (g=None). Feature map φ ∈ {elu, hedgehog, based, rebased}.
  * 'gla_scalar' : per-state SCALAR forget gate (g=log_decay). Feature map elu.

`state_norm` ∈ {raw, global, per_state, kappa} selects the normalization (raw only for
gla_scalar). 'kappa' learns a per-head input-dependent interpolation global↔per-state
via r̃ = r·(d+ε)^{−κ(x)}, κ = σ(w_κ·x) (init ≈ global). `tie_routers=True` shares one
router for read+write (symmetric); the untied read_router is left as None (not a second
module aliasing the same tensor — that breaks HF safetensors save).

This is the single source of truth for the RoLA mixer: the zoology MQAR mixer and the
HF/LM model both wrap this layer; the perf bench instantiates it directly.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla_rola.layers.utils import get_layer_cache, update_layer_cache
from fla_rola.modules import RMSNorm, ShortConvolution
from fla_rola.ops.rola import chunk_rola, fused_recurrent_rola
from fla_rola.ops.rola.chunk import _GLA_FLOOR

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla_rola.models.utils import Cache


class RoLA(nn.Module):
    r"""Routed Linear Attention layer.

    Args:
        hidden_size (int): model dim. Default 1024.
        num_heads (int): number of heads (H). Default 8.
        head_k_dim (int): per-head query/key dim (d_qk). Default 16.
        head_v_dim (int): per-head value dim (d_v). Default 32.
        states_per_head (int): number of routed recurrent states (nc). Default 16.
        kernel (str): 'rla' (un-decayed) or 'gla_scalar' (per-state scalar decay). Default 'rla'.
        phi (str): feature map for 'rla' — 'elu' | 'hedgehog' | 'based' | 'rebased'. Default 'elu'.
                   ('gla_scalar' always uses elu.)
        state_norm (str): 'global' | 'per_state' | 'kappa' (rla) or additionally 'raw' (gla_scalar).
                          Default 'kappa'.
        tie_routers (bool): share one router for read+write (symmetric). Default False.
        tie_router_init (bool): untied routers, but read STARTS == write. Default False.
        router_bias (bool): bias on the routing projections. Default False.
        use_short_conv (bool): causal depthwise short conv (+SiLU) on q/k/v. Default False.
        conv_size (int): short-conv kernel size. Default 4.
        conv_bias (bool): bias in the short conv. Default False.
        layer_idx (int): layer index (for the KV cache). Default None.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 8,
        head_k_dim: int = 16,
        head_v_dim: int = 32,
        states_per_head: int = 16,
        kernel: str = 'rla',
        phi: str = 'elu',
        state_norm: str = 'kappa',
        tie_routers: bool = False,
        tie_router_init: bool = False,
        router_bias: bool = False,
        qk_norm: bool = False,
        router_zloss_coef: float = 0.0,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int | None = None,
        **kwargs,
    ) -> RoLA:
        super().__init__()
        assert kernel in ('rla', 'gla_scalar'), f"unsupported kernel {kernel!r}"
        assert phi == 'elu', f"only the elu feature map is supported (got {phi!r})"
        if kernel == 'rla':
            assert state_norm in ('global', 'per_state', 'kappa'), state_norm
        else:  # gla_scalar
            assert state_norm in ('raw', 'global', 'per_state', 'kappa'), state_norm

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.states_per_head = states_per_head
        self.kernel = kernel
        self.phi = phi
        self.state_norm = state_norm
        self.tie_routers = tie_routers
        self.qk_norm = qk_norm
        self.router_zloss_coef = router_zloss_coef
        self._router_aux = None                       # stashed per forward; read by get_auxiliary_loss
        self.use_short_conv = use_short_conv
        self.mode = 'chunk'   # training/prefill path; short-seq decode auto-switches to fused_recurrent
        self.conv_size = conv_size
        self.layer_idx = layer_idx

        # Feature map is elu+1 only (the fused chunk_rola path): proj_qk == feat_dim == head_k_dim.
        self.proj_qk = head_k_dim
        self.feat_dim = head_k_dim

        # 'global'/'per_state'/'kappa' carry the +feat_dim global-partition term in the recurrent
        # state; gla_scalar 'raw' does not.
        self.uses_v_plus_one = (kernel == 'rla') or (state_norm != 'raw')

        self.key_dim = num_heads * self.proj_qk
        self.value_dim = num_heads * head_v_dim

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # Optional qk-rmsnorm (per-head, pre-feature-map) — an LM-scale training stabilizer. The
        # FLA `RMSNorm` module (fp32, cf. `layers/gla.py`) is created ONLY when enabled, so the
        # default (qk_norm=False) state_dict is byte-identical to before — the MQAR cells stay valid
        # + reproducible.
        if qk_norm:
            self.q_norm = RMSNorm(self.proj_qk, eps=1e-5, dtype=torch.float32)
            self.k_norm = RMSNorm(self.proj_qk, eps=1e-5, dtype=torch.float32)

        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, bias=conv_bias, activation='silu')
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, bias=conv_bias, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, bias=conv_bias, activation='silu')

        # Routers on the residual stream -> dense softmax over states. sym (tie_routers) keeps
        # read_router=None and reuses write_router in _route — registering a second module that
        # aliases the same weight puts two keys for one tensor in the state_dict and crashes HF
        # safetensors save ("shared tensors ... not properly defined").
        self.write_router = nn.Linear(hidden_size, num_heads * states_per_head, bias=router_bias)
        if tie_routers:
            self.read_router = None
        else:
            self.read_router = nn.Linear(hidden_size, num_heads * states_per_head, bias=router_bias)
            if tie_router_init:
                self.read_router.weight.data.copy_(self.write_router.weight.data)
                if router_bias:
                    self.read_router.bias.data.copy_(self.write_router.bias.data)

        if kernel == 'gla_scalar':
            self.w_g = nn.Linear(hidden_size, num_heads, bias=False)  # per-head scalar forget gate
        if state_norm == 'kappa':
            self.w_kappa = nn.Linear(hidden_size, num_heads)
            nn.init.zeros_(self.w_kappa.weight)
            nn.init.constant_(self.w_kappa.bias, -4.0)  # start ≈ global (κ≈0.018), learn upward

    # --- feature map / routing / decay (mirror the rola.py kernels exactly) ---
    def _feature_map(self, q, k):
        return F.elu(q) + 1.0, F.elu(k) + 1.0      # elu+1 — the only supported feature map

    def _route(self, x):
        B, L = x.shape[0], x.shape[1]
        H, C = self.num_heads, self.states_per_head
        wl = self.write_router(x).view(B, L, H, C)               # write logits (pre-softmax)
        write_gates = F.softmax(wl, dim=-1)
        rl = self.read_router(x).view(B, L, H, C) if self.read_router is not None else wl  # sym reuses write
        read_gates = F.softmax(rl, dim=-1)
        if self.router_zloss_coef > 0.0:
            # ST-MoE router z-loss: penalize the log-partition magnitude of the routing logits.
            def zl(lg):
                return torch.logsumexp(lg.float(), dim=-1).square().mean()   # fp32 (bf16 loses the tail)
            self._router_aux = self.router_zloss_coef * (zl(wl) + (zl(rl) if self.read_router is not None else 0.0))
        return write_gates, read_gates

    def get_auxiliary_loss(self):
        """Router z-loss from the last forward (zoology's trainer auto-sums this across modules; the
        HF RoLAForCausalLM aggregates it explicitly). 0.0 when router_zloss_coef == 0."""
        return self._router_aux if self._router_aux is not None else 0.0

    def _log_decay(self, x, write_gates):
        B, L = x.shape[0], x.shape[1]
        H = self.num_heads
        alpha = F.logsigmoid(self.w_g(x).view(B, L, H)).exp()              # [B,L,H]
        alpha_chunk = 1.0 - write_gates * (1.0 - alpha.unsqueeze(-1))       # [B,L,H,C]
        # Floor the log-decay to the kernel's fp32-safe domain (_GLA_FLOOR=-2.5 ⇒ retention ≥ 8.2%/tok).
        # This is a DELIBERATE, documented modeling floor — the chunked GLA decay is factored e^{±a} and
        # overflows fp32 for BT·|ld|≳88.7 (see ops/rola/chunk.py `_floor_ld`, #33). The kernel RAISES on
        # ld below the floor (no silent rewrite); the layer floors explicitly here so the supported
        # decay range is an overt architectural choice, not a kernel-internal surprise.
        return alpha_chunk.clamp(min=1e-8).log().clamp(min=_GLA_FLOOR)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, None, Cache | None]:
        if attention_mask is not None:
            assert attention_mask.dim() == 2, (
                "Expected attention_mask as a [batch, seq_len] 0-1 padding matrix; "
                "[batch, seq_len, seq_len] masks are not supported."
            )
        x = hidden_states
        B, L, _ = x.shape
        H = self.num_heads
        # Short sequences (decode) take the recurrent path; training/prefill stay chunked.
        mode = 'fused_recurrent' if L <= 64 else self.mode
        last_state = get_layer_cache(self, past_key_values)
        cu_seqlens = kwargs.get('cu_seqlens')

        if self.use_short_conv:
            conv_state_q = conv_state_k = conv_state_v = None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            q, conv_state_q = self.q_conv1d(self.q_proj(x), cache=conv_state_q,
                                            output_final_state=use_cache, cu_seqlens=cu_seqlens)
            k, conv_state_k = self.k_conv1d(self.k_proj(x), cache=conv_state_k,
                                            output_final_state=use_cache, cu_seqlens=cu_seqlens)
            v, conv_state_v = self.v_conv1d(self.v_proj(x), cache=conv_state_v,
                                            output_final_state=use_cache, cu_seqlens=cu_seqlens)
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        q = rearrange(q, 'b l (h d) -> b l h d', d=self.proj_qk)
        k = rearrange(k, 'b l (h d) -> b l h d', d=self.proj_qk)
        v = rearrange(v, 'b l (h d) -> b l h d', d=self.head_v_dim)

        if self.qk_norm:                              # per-head qk-RMSNorm (fp32) before the feature map.
            q, k = self.q_norm(q), self.k_norm(k)

        qf, kf = self._feature_map(q, k)
        write_gates, read_gates = self._route(x)
        g = self._log_decay(x, write_gates) if self.kernel == 'gla_scalar' else None
        kap = (torch.sigmoid(self.w_kappa(x)).view(B, L, H, 1)
               if self.state_norm == 'kappa' else None)

        # The ops own dtype/autocast (their Triton autograd Functions carry @input_guard +
        # @autocast_custom_fwd/bwd), so the layer no longer hand-rolls the cast. `output_final_state`
        # emits the recurrent state for the KV-cache; decode seeds it back via `initial_state`.
        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        common = dict(r=read_gates, w=write_gates, g=g, norm=self.state_norm, kappa=kap, scale=1.0)
        if mode == 'fused_recurrent':
            out, recurrent_state = fused_recurrent_rola(
                qf, kf, v, **common, initial_state=recurrent_state,
                output_final_state=use_cache, cu_seqlens=cu_seqlens)
        elif mode == 'chunk':
            if recurrent_state is not None:
                raise NotImplementedError(
                    "chunk_rola has no carried initial_state yet; continuation decode uses the "
                    "fused_recurrent path (auto-selected for L<=64).")
            res = chunk_rola(qf, kf, v, **common, output_final_state=use_cache)
            out, recurrent_state = res if use_cache else (res, None)
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        update_layer_cache(
            self, past_key_values, recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
            offset=L)

        o = self.o_proj(out.reshape(B, L, H * self.head_v_dim))
        return o, None, past_key_values

    # --- state accounting (zoology Hybrid.state_size / LM matched-state read these) ---
    def get_stats(self):
        per_entry = self.feat_dim * self.head_v_dim + (self.feat_dim if self.uses_v_plus_one else 0)
        return {
            'd_qk': self.head_k_dim,
            'feat_dim': self.feat_dim,
            'd_v': self.head_v_dim,
            'n_heads': self.num_heads,
            'num_chunks': self.states_per_head,
            'state_floats': self.num_heads * self.states_per_head * per_entry,
        }

    def state_size(self, sequence_length: int = None, **kwargs) -> int:
        # Recurrent state is independent of sequence length.
        return self.get_stats()['state_floats']
