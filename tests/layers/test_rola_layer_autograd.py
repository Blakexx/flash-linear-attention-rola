# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""Layer-level fwd+bwd autograd gate for the RoLA dispatch threshold.

The op suite (tests/ops/test_rola.py::TestAutograd) validates chunk_rola_routed's backward, but only
at MULTI-CHUNK lengths and by calling the op directly. The LAYER (RoLA.forward) auto-selects the
recurrent DECODE kernel at L<=64, and that kernel (fused_recurrent_rola) is FORWARD-ONLY — it has no
backward. A length-ONLY dispatch (`mode = 'fused_recurrent' if L <= 64 else self.mode`) therefore
routed a GRAD-REQUIRING forward through the non-differentiable path: the autograd graph dead-ended at
the kernel, so x.grad (and every input-side param grad) came back None. That seam lives ABOVE the op
tests, so nothing caught it until the fleet GPU smoke (a RoLA fwd+bwd at L=64) failed on every host.

The fix makes the dispatch grad-aware (recurrent only when grad is disabled, i.e. inference/decode).
This gate exercises the layer END-TO-END across the L<=64 threshold: a grad-requiring fwd+bwd must
produce finite, non-None grads for the input AND every parameter (RLA and GLA), and the no-grad
short-seq path must still run (the recurrent decode fast-path is preserved for inference).

Run:  PYTHONPATH=. pytest tests/layers/test_rola_layer_autograd.py -q   (CUDA required; CPU skipped).
"""
import pytest
import torch

from fla_rola.layers.rola import RoLA
from fla_rola.utils import device


def _build(kernel='rla', routing='flat', nc=16, H=2, hidden=64):
    return RoLA(hidden_size=hidden, num_heads=H, head_k_dim=16, head_v_dim=16,
                states_per_head=nc, kernel=kernel, state_norm='global', routing=routing).to(device)


def _assert_grads_flow(m, x, o):
    """Every learnable parameter AND the input must receive a finite, non-None gradient — i.e. the
    autograd graph reaches them (a forward-only kernel in the path would leave them None)."""
    assert o.requires_grad and o.grad_fn is not None, 'forward output is detached from the graph'
    o.float().pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), 'no/NaN gradient reached the layer input'
    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    missing = [n for n, p in named if p.grad is None]
    assert not missing, f'parameters received NO gradient (graph dead-ends before them): {missing}'
    nonfinite = [n for n, p in named if not torch.isfinite(p.grad).all()]
    assert not nonfinite, f'parameters received NaN/Inf gradient: {nonfinite}'


@pytest.mark.parametrize('L', [32, 64, 128])   # 32 & 64 trip the <=64 threshold (were recurrent); 128 is chunk
def test_layer_fwd_bwd_grads_flow_rla(L):
    """Grad-requiring RLA layer fwd+bwd at and across the L<=64 threshold back-propagates to the input
    and every parameter. Regression for the length-only dispatch (the fleet-smoke failure)."""
    if device != 'cuda':
        pytest.skip('RoLA Triton kernels require CUDA')
    torch.manual_seed(0)
    m = _build('rla')
    x = torch.randn(2, L, 64, device=device, requires_grad=True)
    out = m(x)
    o = out[0] if isinstance(out, tuple) else out
    _assert_grads_flow(m, x, o)


def test_layer_fwd_bwd_grads_flow_gla_threshold():
    """Parity: the GLA (gla_scalar, per-state log-decay) layer ALSO back-propagates at L=64 — the exact
    length the dispatch sent to the no-backward recurrent kernel."""
    if device != 'cuda':
        pytest.skip('RoLA Triton kernels require CUDA')
    torch.manual_seed(0)
    m = _build('gla_scalar')
    x = torch.randn(2, 64, 64, device=device, requires_grad=True)
    out = m(x)
    o = out[0] if isinstance(out, tuple) else out
    _assert_grads_flow(m, x, o)


@pytest.mark.parametrize('routing', ['square', 'tree'])   # 'flat' covered by the threshold sweep
def test_layer_fwd_bwd_grads_flow_routings(routing):
    """The dispatch gate holds for every per-head routing topology at L=64 (RLA)."""
    if device != 'cuda':
        pytest.skip('RoLA Triton kernels require CUDA')
    torch.manual_seed(0)
    m = _build('rla', routing)
    x = torch.randn(2, 64, 64, device=device, requires_grad=True)
    out = m(x)
    o = out[0] if isinstance(out, tuple) else out
    _assert_grads_flow(m, x, o)


def test_short_seq_inference_path_preserved():
    """The fix must NOT regress decode: a no-grad L<=64 forward still runs (the recurrent inference
    fast-path is kept) and returns finite output."""
    if device != 'cuda':
        pytest.skip('RoLA Triton kernels require CUDA')
    torch.manual_seed(0)
    m = _build('rla')
    x = torch.randn(2, 32, 64, device=device)
    with torch.inference_mode():
        out = m(x)
        o = out[0] if isinstance(out, tuple) else out
    assert torch.isfinite(o).all(), 'short-seq inference (recurrent) forward produced non-finite output'
