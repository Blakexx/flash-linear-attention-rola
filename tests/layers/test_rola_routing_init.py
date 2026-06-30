# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""Fresh-init parity for the RoLA factored router (#39 P2).

The factored per-head router (write_W/read_W ∈ [H, D, hidden, b]) replaced the old dense
nn.Linear(hidden, H*nc). A `routing='flat'` layer (D=1, b=nc) must be the strict equivalent of the
old dense router — not just algebraically (gates/z-loss given equal weights) but in the FRESH-INIT
DISTRIBUTION, so existing routing='flat' runs stay reproducible. This gates the init: a fresh flat
write_W/read_W must match the init std of a reference nn.Linear(hidden, H*nc).

(Regression for the GATE-B seam: a 3-D kaiming init inflated fan_in to b·hidden, shrinking the bound
by √b ⇒ a fresh flat layer started ~uniform routing, NOT reproducing the old layer.)
"""
import pytest
import torch
import torch.nn as nn

from fla_rola.layers.rola import RoLA, _routing_factors


def _linear_ref_std(hidden, out, n=64):
    """Mean init std of nn.Linear(hidden, out).weight over n fresh inits (the distribution flat must
    match)."""
    return torch.stack([nn.Linear(hidden, out).weight.std() for _ in range(n)]).mean().item()


@pytest.mark.parametrize('nc', [8, 16, 64])
@pytest.mark.parametrize('H,hidden', [(2, 128), (8, 512)])
def test_flat_router_init_matches_dense_linear(H, hidden, nc):
    """A fresh routing='flat' layer's write_W/read_W init std == nn.Linear(hidden, H*nc) init std.
    Averaged over fresh inits so the comparison is of DISTRIBUTIONS, not a single draw. The 3-D-kaiming
    bug shrank this by √nc (e.g. flat std 0.0064 vs dense 0.0254 at nc=16) — caught here."""
    ref = _linear_ref_std(hidden, H * nc)
    ws, rs = [], []
    for _ in range(64):
        layer = RoLA(hidden_size=hidden, num_heads=H, head_k_dim=16, head_v_dim=16,
                     states_per_head=nc, kernel='rla', state_norm='global', routing='flat',
                     tie_routers=False)
        ws.append(layer.write_W.std())
        rs.append(layer.read_W.std())
    write_std = torch.stack(ws).mean().item()
    read_std = torch.stack(rs).mean().item()
    # within 10% of the reference Linear std (Monte-Carlo over 64 inits each).
    assert write_std == pytest.approx(ref, rel=0.10), f'flat write_W std {write_std:.4f} vs dense {ref:.4f}'
    assert read_std == pytest.approx(ref, rel=0.10), f'flat read_W std {read_std:.4f} vs dense {ref:.4f}'


@pytest.mark.parametrize('routing', ['flat', 'square', 'tree'])
def test_factor_router_fan_in_is_hidden(routing):
    """The factor-router init must see fan_in == hidden for EVERY topology (the kaiming bound is
    sqrt(3)*gain*sqrt(1/hidden), so std ≈ that bound / sqrt(3)). A fresh write_W std matches the
    per-level reference nn.Linear(hidden, b) std — confirming b is NOT folded into fan_in for any D."""
    H, hidden, nc = 4, 256, 16
    D, b = _routing_factors(routing, nc)
    ref = _linear_ref_std(hidden, b)        # per-level: the logit is Linear(hidden, b) per (head,level)
    stds = []
    for _ in range(64):
        layer = RoLA(hidden_size=hidden, num_heads=H, head_k_dim=16, head_v_dim=16,
                     states_per_head=nc, kernel='rla', state_norm='global', routing=routing)
        stds.append(layer.write_W.std())
    std = torch.stack(stds).mean().item()
    assert std == pytest.approx(ref, rel=0.10), f'{routing} (D={D},b={b}) write_W std {std:.4f} vs Linear(hidden,b) {ref:.4f}'


def test_tie_init_copies_and_bias_zero():
    """tie_router_init makes read_W start == write_W; router_bias starts at zeros (the old Linear bias
    default was also zero-mean small — but the layer zeros it, matching the routed-op's uniform-prior
    start). Smoke-guards the init wiring around the std fix."""
    layer = RoLA(hidden_size=64, num_heads=2, head_k_dim=16, head_v_dim=16, states_per_head=16,
                 kernel='rla', state_norm='global', routing='flat', tie_routers=False,
                 tie_router_init=True, router_bias=True)
    assert torch.equal(layer.read_W.data, layer.write_W.data)
    assert torch.count_nonzero(layer.write_b) == 0 and torch.count_nonzero(layer.read_b) == 0


@pytest.mark.parametrize('nc', [8, 64])
def test_short_conv_router_state_is_nc_independent(nc):
    """Router short-conv state lives on hidden streams before the state-count-dependent projection."""
    kwargs = dict(hidden_size=64, num_heads=2, head_k_dim=8, head_v_dim=8,
                  states_per_head=nc, kernel='rla', state_norm='global',
                  routing='flat', tie_routers=False, conv_size=3)
    base = RoLA(**kwargs)
    short = RoLA(**kwargs, use_short_conv=True)
    delta = short.get_stats()['state_floats'] - base.get_stats()['state_floats']
    expected = 3 * (short.key_dim + short.key_dim + short.value_dim + 2 * short.hidden_size)
    assert delta == expected
