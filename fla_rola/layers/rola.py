# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""RoLA — Routed Linear Attention (canonical FLA layer).

Shared q/k/v/o projections + learned PER-HEAD read/write routing over `states_per_head`
recurrent states, with a feature-mapped linear-attention inner kernel. Routing is FACTORED
(`routing` ∈ {flat, square, tree}): each head holds per-level factor weights
Wr/Ww ∈ [H, D, hidden, b] (b**D == nc), and the leaf gate is the product over levels of the
per-level softmax. The whole normalization recipe (per-state denominator, read-gate rescale,
shared-Gram numerator-only readout, divide) AND — for the chunk path — the routing itself and
the GLA per-state log-decay live IN-KERNEL (`chunk_rola_routed`): the [L,nc] gates + ld are
NEVER materialized. The layer projects, applies φ, and dispatches one weights-in interface:

    x -> (q,k,v) projections -> φ(q),φ(k)
      CHUNK : chunk_rola_routed(qf,kf,v, h=x⊗head, Wr,Ww, D,b, Wg=w_g|None, norm, kappa) (in-kernel
              routing + decay; [L,nc] gates + ld never materialized)
      DECODE: per-token gates + per-token log-decay built IN TORCH from the SAME factor weights
              -> fused_recurrent_rola(qf,kf,v, r,w,g, norm, kappa)
      -> o_proj

flat (D=1, b=nc) is the strict equivalent of the old dense Linear(hidden, H*nc) softmax router.

Two inner kernels (both route through the in-kernel routed readout, the paper-shipping cells):
  * 'rla'        : un-decayed (Wg=None). Feature map φ ∈ {elu, hedgehog, based, rebased}.
  * 'gla_scalar' : per-state SCALAR forget gate (per-head Wg). Feature map elu.

`state_norm` ∈ {raw, global, per_state, kappa} selects the normalization (raw only for
gla_scalar). 'kappa' learns a per-head input-dependent interpolation global↔per-state
via r̃ = r·(d+ε)^{−κ(x)}, κ = σ(w_κ·x) (init ≈ global). `tie_routers=True` shares one
factor router for read+write (symmetric); the untied read_W is left as None (not a second
parameter aliasing the same tensor — that breaks HF safetensors save).

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
from fla_rola.ops.rola import chunk_rola_routed, fused_recurrent_rola
from fla_rola.ops.rola.chunk import _GLA_FLOOR


def _routing_factors(routing: str, nc: int) -> tuple[int, int]:
    """Map a routing topology + state count nc to the (D, b) factorization with b**D == nc.
      'flat'   -> (1, nc)        : one level, one softmax over all nc states (the dense router).
      'square' -> (2, sqrt(nc))  : two levels, b=sqrt(nc) each (nc must be a perfect square).
      'tree'   -> (log2(nc), 2)  : binary tree, nc must be a power of two.
    """
    if routing == 'flat':
        return 1, nc
    if routing == 'square':
        b = int(round(nc ** 0.5))
        if b * b != nc:
            raise ValueError(f"routing='square' needs a perfect-square states_per_head, got nc={nc}")
        return 2, b
    if routing == 'tree':
        D = nc.bit_length() - 1
        if nc != (1 << D):
            raise ValueError(f"routing='tree' needs a power-of-two states_per_head, got nc={nc}")
        return D, 2
    raise ValueError(f"unsupported routing {routing!r} (flat|square|tree)")

# One-time-per-process flag for the layer-side decay-floor truncation warning (#33 F4). The kernel's
# `_floor_ld` raises (or warns in clamp-mode) on out-of-range ld; but the layer floors the LEARNED decay
# silently with `.clamp(min=_GLA_FLOOR)` BEFORE the kernel ever sees it, so the kernel's loud signal is
# dead in production. We restore the signal here: warn ONCE when the learned decay actually hits the floor.
_rola_layer_floor_warned = False

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
        routing (str): routing topology — 'flat' (one softmax over nc; the dense router, default),
                       'square' (two levels of sqrt(nc)) or 'tree' (binary, nc=2^D). Selects the
                       per-head factor geometry (D, b) with b**D == nc; the layer holds per-head
                       per-level factor weights and dispatches to the in-kernel routed kernel.
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
        routing: str = 'flat',
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
        self.routing = routing
        self.route_D, self.route_b = _routing_factors(routing, states_per_head)
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

        # PER-HEAD per-level factor routers on the residual stream. RoLA routes per head: each head h
        # owns its OWN tree weights Wr/Ww ∈ [H, D, hidden, b] (b**D == nc), and the leaf gate is the
        # product over levels of softmax(h·W[head,lvl] + bias). 'flat' (D=1, b=nc) is the strict
        # equivalent of the old dense Linear(hidden, H*nc) router: W[head,0] == old per-head weight^T.
        # The factor weights live as plain nn.Parameters (NOT nn.Linear) so the per-head/per-level
        # structure is explicit and the in-kernel routed op consumes them directly. tie_routers keeps
        # read_W=None (reuse write_W) — registering a second tensor aliasing the same weight crashes HF
        # safetensors save.
        D, b = self.route_D, self.route_b
        self.write_W = nn.Parameter(torch.empty(num_heads, D, hidden_size, b))
        self._init_factor_router(self.write_W)
        if router_bias:
            self.write_b = nn.Parameter(torch.zeros(num_heads, D, b))
        else:
            self.register_parameter('write_b', None)
        if tie_routers:
            self.register_parameter('read_W', None)
            self.register_parameter('read_b', None)
        else:
            self.read_W = nn.Parameter(torch.empty(num_heads, D, hidden_size, b))
            if router_bias:
                self.read_b = nn.Parameter(torch.zeros(num_heads, D, b))
            else:
                self.register_parameter('read_b', None)
            if tie_router_init:
                self.read_W.data.copy_(self.write_W.data)
                if router_bias:
                    self.read_b.data.copy_(self.write_b.data)
            else:
                self._init_factor_router(self.read_W)

        if kernel == 'gla_scalar':
            self.w_g = nn.Linear(hidden_size, num_heads, bias=False)  # per-head scalar forget gate
        if state_norm == 'kappa':
            self.w_kappa = nn.Linear(hidden_size, num_heads)
            nn.init.zeros_(self.w_kappa.weight)
            nn.init.constant_(self.w_kappa.bias, -4.0)  # start ≈ global (κ≈0.018), learn upward

    @staticmethod
    def _init_factor_router(W: nn.Parameter):
        """Init the per-head per-level factor weights [H, D, hidden, b] like an nn.Linear(hidden, b):
        the routing logit is h·W[head,lvl] (== Linear with weight W[head,lvl]^T), so the init must be a
        transposed Linear weight with kaiming `fan_in == hidden`. We build the Linear-weight layout as a
        2-D [out, in] = [H*D*b, hidden] tensor — 2-D is essential: kaiming_uniform infers fan_in from a
        2-D weight as dim 1 (== hidden), whereas a 3-D [*, b, hidden] tensor would treat b as input
        feature-maps and inflate fan_in to b·hidden (bound √b too small). Reshape/transpose into
        [H,D,hidden,b]. flat (b=nc) thus reproduces the old dense nn.Linear(hidden, H*nc) init
        distribution exactly (std ≈ same; verified by the fresh-init parity test)."""
        H, D, hidden, b = W.shape
        wt = torch.empty(H * D * b, hidden)       # 2-D [out=H*D*b, in=hidden] -> kaiming fan_in == hidden
        nn.init.kaiming_uniform_(wt, a=5 ** 0.5)
        # [H*D*b, hidden] -> [H,D,b,hidden] -> transpose last two -> [H,D,hidden,b] (W[head,lvl] = slice^T)
        W.data.copy_(wt.view(H, D, b, hidden).transpose(-1, -2))

    # --- feature map / routing / decay (mirror the rola.py kernels exactly) ---
    def _feature_map(self, q, k):
        return F.elu(q) + 1.0, F.elu(k) + 1.0      # elu+1 — the only supported feature map

    def _factor_logits(self, x, W, bias):
        """Per-head per-level routing logits from the factor weights. x:[B,L,hidden],
        W:[H,D,hidden,b], bias:[H,D,b]|None -> logits[B,L,H,D,b] = x·W[head,lvl] (+bias)."""
        z = torch.einsum('bld,hkdc->blhkc', x, W.to(x.dtype))     # [B,L,H,D,b]
        if bias is not None:
            z = z + bias.to(x.dtype)[None, None]                  # [H,D,b] broadcast
        return z

    def _gates_from_logits(self, logits):
        """Fold per-level softmax factors into the explicit [B,L,H,nc] leaf gates (the DECODE/CPU
        reference path — the kernel builds these IN-KERNEL for the chunk path). logits:[B,L,H,D,b].
        leaf gate = Π_lvl softmax(logits[...,lvl,:])[..., digit_lvl(leaf)]."""
        D = self.route_D
        f = F.softmax(logits, dim=-1)                             # [B,L,H,D,b]
        g = f[..., 0, :]                                          # level 0 -> [B,L,H,b]
        for i in range(1, D):
            # outer product across levels: [...,b^i,1] * [...,1,b] -> [...,b^(i+1)]
            g = (g.unsqueeze(-1) * f[..., i, :].unsqueeze(-2)).flatten(-2)
        return g                                                  # [B,L,H,nc]

    def _stash_zloss(self, wl, rl):
        """ST-MoE router z-loss on the FACTOR logits — penalize Σ_lvl logsumexp(level logits)^2, the
        per-level log-partition magnitude (flat: the single softmax, == the old dense z-loss). Summed
        over the D levels of read+write. Stashed per forward; read by get_auxiliary_loss."""
        if self.router_zloss_coef <= 0.0:
            return

        def zl(lg):  # lg:[B,L,H,D,b] -> per-level logsumexpsq, summed over levels
            return torch.logsumexp(lg.float(), dim=-1).square().mean(dim=(0, 1, 2)).sum()
        self._router_aux = self.router_zloss_coef * (zl(wl) + (zl(rl) if self.read_W is not None else 0.0))

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
        ld = alpha_chunk.clamp(min=1e-8).log()
        # #33 F4: the kernel's loud raise is dead in production because we floor BEFORE the kernel sees
        # ld. Restore the signal: warn ONCE (per process) the first time a LEARNED decay actually dips
        # below the floor and gets truncated — consistent with the kernel's clamp-mode one-time warn, so
        # the truncation is loud (no silent rewrite of a learned parameter).
        global _rola_layer_floor_warned
        if not _rola_layer_floor_warned and bool((ld.detach() < _GLA_FLOOR).any()):
            import warnings
            mn = ld.detach().min().item()
            warnings.warn(
                f"RoLA learned log-decay floored to _GLA_FLOOR={_GLA_FLOOR} (min ld={mn:.4f} truncated). "
                f"This clamps the learned per-state forget gate to the kernel's fp32-safe retention floor "
                f"(≥8.2%/tok) — the learned decay rate below the floor is altered. Reduce the decay (raise "
                f"alpha / lower the write gate) if this is unintended; the floor is a deliberate modeling "
                f"choice (see ops/rola/chunk.py `_floor_ld`, #33).",
                stacklevel=2,
            )
            _rola_layer_floor_warned = True
        return ld.clamp(min=_GLA_FLOOR)

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
        kap = (torch.sigmoid(self.w_kappa(x)).view(B, L, H, 1)
               if self.state_norm == 'kappa' else None)
        D, b = self.route_D, self.route_b
        Wg = self.w_g.weight if self.kernel == 'gla_scalar' else None   # [H, hidden] per-head decay

        # Router z-loss is stashed from the (cheap, [L,nc]-free) per-level FACTOR logits on BOTH paths —
        # the chunk kernel never materializes the gates, so the layer computes the logits here purely for
        # the auxiliary loss + (decode) gate folding.
        wl = self._factor_logits(x, self.write_W, self.write_b)
        rl = self._factor_logits(x, self.read_W, self.read_b) if self.read_W is not None else wl
        self._stash_zloss(wl, rl)

        # The ops own dtype/autocast (their Triton autograd Functions carry @input_guard +
        # @autocast_custom_fwd/bwd), so the layer no longer hand-rolls the cast. `output_final_state`
        # emits the recurrent state for the KV-cache; decode seeds it back via `initial_state`.
        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        if mode == 'fused_recurrent':
            # DECODE: build the explicit per-token gates + per-token decay IN TORCH from the SAME factor
            # weights (cheap for the one/few decode tokens), then the fused recurrent kernel. One
            # weights-in interface — no separate precomputed-gate chunk signature.
            write_gates = self._gates_from_logits(wl)
            read_gates = write_gates if self.read_W is None else self._gates_from_logits(rl)
            g = self._log_decay(x, write_gates) if self.kernel == 'gla_scalar' else None
            out, recurrent_state = fused_recurrent_rola(
                qf, kf, v, r=read_gates, w=write_gates, g=g, norm=self.state_norm, kappa=kap, scale=1.0,
                initial_state=recurrent_state, output_final_state=use_cache, cu_seqlens=cu_seqlens)
        elif mode == 'chunk':
            if recurrent_state is not None:
                raise NotImplementedError(
                    "chunk_rola_routed has no carried initial_state yet; continuation decode uses the "
                    "fused_recurrent path (auto-selected for L<=64).")
            # CHUNK: in-kernel routing + decay. h is the residual stream broadcast per head ([B,L,H,hidden]);
            # the kernel folds the routing gram, the per-state den, the read-gate rescale AND (GLA) the
            # per-state log-decay IN-KERNEL — the [L,nc] gates + ld are NEVER materialized (the training
            # saved-activation win).
            h = x.unsqueeze(2).expand(B, L, H, self.hidden_size)
            out = chunk_rola_routed(
                qf, kf, v, h, self.write_W if self.read_W is None else self.read_W, self.write_W,
                D, b, norm=self.state_norm, kappa=kap, scale=1.0,
                b_r=(self.write_b if self.read_W is None else self.read_b), b_w=self.write_b, Wg=Wg)
            recurrent_state = None
            if use_cache:
                # Prefill→decode handoff (inference only): the routed readout stays [L,nc]-free, but the
                # FINAL recurrent state (a small [H*nc,K,V(+1)] tensor, independent of L) inherently needs
                # the write gates / per-state decay. Build them in TORCH from the SAME factor weights (the
                # decode-path gates) and fold via `_final_state` — this is state EMISSION, not a second
                # readout path. No backward (decode is inference); training (use_cache=False) never hits it.
                from fla_rola.ops.rola.chunk import _final_state
                write_gates = self._gates_from_logits(wl)
                gfull = self._log_decay(x, write_gates) if self.kernel == 'gla_scalar' else None

                def _fold(t):
                    return t.permute(0, 2, 1, 3).reshape(B * H, L, t.shape[-1])
                recurrent_state = _final_state(
                    _fold(kf), _fold(v), _fold(write_gates),
                    _fold(gfull) if gfull is not None else None, B, H, raw=not self.uses_v_plus_one)
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
