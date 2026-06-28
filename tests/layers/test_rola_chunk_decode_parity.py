# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""Layer-level chunk-vs-decode parity across EVERY (kernel, norm, routing).

The RoLA layer builds the op's arguments TWO different ways and they must agree:
  * chunk path  — routes, decays and normalizes IN-KERNEL from Wr/Ww/Wg (chunk_rola_routed).
  * decode path — builds explicit [B,L,H,nc] gates and per-token log-decay IN TORCH
                  (_gates_from_logits / _log_decay), then fused_recurrent_rola.
If the in-kernel routing fold, the in-kernel GLA decay, or any norm's denominator disagrees with the
torch reconstruction, prefill(chunk)->decode(recurrent) breaks SILENTLY. The model-level
test_generation only exercises kappa+flat, so the other norms (raw/global/per_state) and routings
(square/tree) had NO layer-level parity gate — this fills that matrix.

Method: process L>64 tokens at once (chunk) vs one-at-a-time with a cache (recurrent decode); the
position-t outputs must match to the chunk-vs-decode precision floor. A STRUCTURAL divergence (wrong
fold/decay/norm) is O(1) relative; the real floor is <=~5e-3 (raw GLA, un-normalized). Tolerance 2e-2
mirrors test_generation. Uses the GLOBAL ratio max|diff|/max|ref| (element-wise relatives blow up on
near-zero output elements — a metric artifact, not a divergence).

Run:  PYTHONPATH=. pytest tests/layers/test_rola_chunk_decode_parity.py -q   (CUDA required; CPU skipped).
"""
import pytest
import torch

import fla_rola.layers.rola as rola_layer_mod
from fla_rola.layers.rola import RoLA
from fla_rola.models.utils import Cache
from fla_rola.ops.rola import fused_recurrent_rola
from fla_rola.utils import device

_H, _HID, _DQK, _DV, _NC, _L, _B = 2, 64, 16, 16, 16, 96, 2
_TOL = 2e-2


def _norms_for(kernel):
    # 'raw' (un-normalized readout) is well-posed only with decay -> gla_scalar only (layer assert).
    return ['raw', 'global', 'per_state', 'kappa'] if kernel == 'gla_scalar' else ['global', 'per_state', 'kappa']


@pytest.mark.parametrize('routing', ['flat', 'square', 'tree'])
@pytest.mark.parametrize('norm', ['raw', 'global', 'per_state', 'kappa'])
@pytest.mark.parametrize('kernel', ['rla', 'gla_scalar'])
def test_chunk_decode_parity(kernel, norm, routing):
    """The layer's chunk (in-kernel) and decode (torch-gate) paths must produce the same outputs for a
    fixed input — for every kernel x norm x routing. Regression for layer<->op arg-construction
    divergence (routing fold / GLA decay / norm denominator)."""
    if device != 'cuda':
        pytest.skip('RoLA Triton kernels require CUDA')
    if norm not in _norms_for(kernel):
        pytest.skip(f"{norm} is not a valid state_norm for {kernel}")
    torch.manual_seed(0)
    m = RoLA(hidden_size=_HID, num_heads=_H, head_k_dim=_DQK, head_v_dim=_DV, layer_idx=0,
             states_per_head=_NC, kernel=kernel, state_norm=norm, routing=routing).to(device).eval()
    x = torch.randn(_B, _L, _HID, device=device)

    # chunk: whole sequence at once (L>64 -> chunk path), no grad
    with torch.no_grad():
        oc = m(x)
        oc = oc[0] if isinstance(oc, tuple) else oc

    # decode: one token at a time, recurrent, carrying the cache (L=1<=64 -> fused_recurrent)
    outs = []
    with torch.inference_mode():
        cache = Cache()
        for t in range(_L):
            ot = m(x[:, t:t + 1], past_key_values=cache, use_cache=True)
            outs.append(ot[0] if isinstance(ot, tuple) else ot)
    od = torch.cat(outs, dim=1)

    diff = (oc.float() - od.float()).abs().max()
    scale = oc.float().abs().max().clamp(min=1e-6)
    ratio = (diff / scale).item()
    assert ratio < _TOL, (f"{kernel}/{norm}/{routing}: chunk vs decode diverge — global relmax {ratio:.3e} "
                          f"(absmax {diff.item():.3e}) exceeds {_TOL}. Structural divergence in the "
                          f"in-kernel-vs-torch routing/decay/norm reconstruction.")


@pytest.mark.parametrize('name,config,kwargs', [
    ('tie_routers_tree', {'kernel': 'rla', 'state_norm': 'kappa', 'routing': 'tree'}, {'tie_routers': True}),
    ('router_bias_flat', {'kernel': 'rla', 'state_norm': 'kappa', 'routing': 'flat'}, {'router_bias': True}),
    ('qk_norm_square', {'kernel': 'rla', 'state_norm': 'per_state', 'routing': 'square'}, {'qk_norm': True}),
    ('short_conv_tree', {'kernel': 'rla', 'state_norm': 'kappa', 'routing': 'tree'},
     {'use_short_conv': True, 'conv_size': 3}),
    ('short_conv_gla_raw', {'kernel': 'gla_scalar', 'state_norm': 'raw', 'routing': 'tree'},
     {'use_short_conv': True, 'conv_size': 3}),
])
def test_decode_option_chunk_parity(name, config, kwargs):
    """Representative decode branch smokes for options outside the full kernel/norm/routing matrix."""
    if device != 'cuda':
        pytest.skip('RoLA Triton kernels require CUDA')
    torch.manual_seed(10)
    m = RoLA(hidden_size=_HID, num_heads=_H, head_k_dim=_DQK, head_v_dim=_DV, layer_idx=0,
             states_per_head=_NC, **config, **kwargs).to(device).eval()
    x = torch.randn(1, _L, _HID, device=device)

    with torch.no_grad():
        oc = m(x)
        oc = oc[0] if isinstance(oc, tuple) else oc

    outs = []
    with torch.inference_mode():
        cache = Cache()
        for t in range(_L):
            ot = m(x[:, t:t + 1], past_key_values=cache, use_cache=True)
            outs.append(ot[0] if isinstance(ot, tuple) else ot)
    od = torch.cat(outs, dim=1)

    ratio = ((oc.float() - od.float()).abs().max() / oc.float().abs().max().clamp(min=1e-6)).item()
    assert ratio < _TOL, f"{name}: chunk-vs-decode relmax {ratio:.3e} exceeds {_TOL}"


@pytest.mark.parametrize('kernel,norm,routing', [
    ('rla', 'kappa', 'tree'),
    ('gla_scalar', 'raw', 'tree'),
    ('gla_scalar', 'kappa', 'square'),
])
def test_chunk_prefill_cache_handoff_matches_decode(kernel, norm, routing):
    """Chunk prefill cache state must seed recurrent decode exactly enough to continue generation."""
    if device != 'cuda':
        pytest.skip('RoLA Triton kernels require CUDA')
    torch.manual_seed(20)
    prefix = 80
    total = 96
    m = RoLA(hidden_size=_HID, num_heads=_H, head_k_dim=_DQK, head_v_dim=_DV, layer_idx=0,
             states_per_head=_NC, kernel=kernel, state_norm=norm, routing=routing).to(device).eval()
    x = torch.randn(1, total, _HID, device=device)

    with torch.inference_mode():
        chunk_cache = Cache()
        prefix_out = m(x[:, :prefix], past_key_values=chunk_cache, use_cache=True)[0]
        chunk_state = chunk_cache[0]['recurrent_state'].float().clone()
        suffix_outs = []
        for t in range(prefix, total):
            suffix_outs.append(m(x[:, t:t + 1], past_key_values=chunk_cache, use_cache=True)[0])
        mixed_out = torch.cat([prefix_out] + suffix_outs, dim=1)

        decode_cache = Cache()
        decode_outs = []
        state_after_prefix = None
        for t in range(total):
            decode_outs.append(m(x[:, t:t + 1], past_key_values=decode_cache, use_cache=True)[0])
            if t == prefix - 1:
                state_after_prefix = decode_cache[0]['recurrent_state'].float().clone()
        decode_out = torch.cat(decode_outs, dim=1)

    state_ratio = ((chunk_state - state_after_prefix).abs().max()
                   / state_after_prefix.abs().max().clamp(min=1e-6)).item()
    out_ratio = ((mixed_out.float() - decode_out.float()).abs().max()
                 / decode_out.float().abs().max().clamp(min=1e-6)).item()
    assert state_ratio < _TOL, f"{kernel}/{norm}/{routing}: prefill state relmax {state_ratio:.3e}"
    assert out_ratio < _TOL, f"{kernel}/{norm}/{routing}: prefill->decode output relmax {out_ratio:.3e}"


def test_chunk_prefill_warns_on_learned_gla_floor():
    """Chunk prefill must keep the layer-level signal when learned GLA decay hits the fp32 floor."""
    if device != 'cuda':
        pytest.skip('RoLA Triton kernels require CUDA')
    m = RoLA(hidden_size=_HID, num_heads=_H, head_k_dim=_DQK, head_v_dim=_DV, layer_idx=0,
             states_per_head=_NC, kernel='gla_scalar', state_norm='global', routing='flat',
             router_bias=True).to(device).eval()
    with torch.no_grad():
        m.write_W.zero_()
        m.write_b.fill_(-20.0)
        m.write_b[:, :, 0] = 20.0
        m.w_g.weight.fill_(-1.0)
    x = torch.ones(1, _L, _HID, device=device)

    rola_layer_mod._rola_layer_floor_warned = False
    import warnings
    with torch.inference_mode(), warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = m(x, use_cache=True)[0]
    assert torch.isfinite(out.float()).all()
    assert any('floored' in str(w.message).lower() and '_gla_floor' in str(w.message).lower() for w in rec)


def test_decode_rejects_varlen_cu_seqlens():
    """Varlen decode is unsupported today; keep both layer plumbing and op contract loud."""
    if device != 'cuda':
        pytest.skip('RoLA Triton kernels require CUDA')
    m = RoLA(hidden_size=_HID, num_heads=_H, head_k_dim=_DQK, head_v_dim=_DV, layer_idx=0,
             states_per_head=_NC, kernel='rla', state_norm='kappa', routing='tree').to(device).eval()
    x = torch.randn(1, 2, _HID, device=device)
    cu = torch.tensor([0, 2], device=device, dtype=torch.long)
    with torch.inference_mode(), pytest.raises(NotImplementedError, match='cu_seqlens'):
        m(x, cu_seqlens=cu)

    q = torch.randn(1, 2, 1, 8, device=device)
    k = torch.randn(1, 2, 1, 8, device=device)
    v = torch.randn(1, 2, 1, 8, device=device)
    r = torch.softmax(torch.randn(1, 2, 1, 4, device=device), -1)
    w = torch.softmax(torch.randn(1, 2, 1, 4, device=device), -1)
    with pytest.raises(NotImplementedError, match='cu_seqlens'):
        fused_recurrent_rola(q, k, v, r=r, w=w, norm='global', cu_seqlens=cu)
