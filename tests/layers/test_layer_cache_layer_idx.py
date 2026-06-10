import pytest
import torch

from fla_rola.layers.abc import ABCAttention
from fla_rola.layers.comba import Comba
from fla_rola.layers.delta_net import DeltaNet
from fla_rola.layers.gated_deltanet import GatedDeltaNet
from fla_rola.layers.gated_deltaproduct import GatedDeltaProduct
from fla_rola.layers.gla import GatedLinearAttention
from fla_rola.layers.gsa import GatedSlotAttention
from fla_rola.layers.hgrn import HGRNAttention
from fla_rola.layers.hgrn2 import HGRN2Attention
from fla_rola.layers.kda import KimiDeltaAttention
from fla_rola.layers.lightnet import LightNetAttention
from fla_rola.layers.linear_attn import LinearAttention
from fla_rola.layers.log_linear_mamba2 import LogLinearMamba2
from fla_rola.layers.mamba import Mamba
from fla_rola.layers.mamba2 import Mamba2
from fla_rola.layers.mesa_net import MesaNet
from fla_rola.layers.mom import MomAttention
from fla_rola.layers.multiscale_retention import MultiScaleRetention
from fla_rola.layers.rodimus import RodimusAttention
from fla_rola.layers.rwkv6 import RWKV6Attention
from fla_rola.layers.rwkv7 import RWKV7Attention
from fla_rola.layers.simple_gla import SimpleGatedLinearAttention
from fla_rola.utils import device


class DummyCache(list):

    def update(self, **kwargs):
        self.last_update = kwargs
        return kwargs


def maybe_build(builder):
    try:
        return builder()
    except RuntimeError as exc:
        if "LAPACK" in str(exc):
            pytest.skip(f"layer initialization requires LAPACK in this environment: {exc}")
        raise


def prepare_layer_and_inputs(builder, hidden_states):
    layer = maybe_build(builder).to(device)
    hidden_states = hidden_states.to(device)
    return layer, hidden_states


CACHE_REQUIRES_LAYER_IDX_CASES = [
    pytest.param(
        lambda: LinearAttention(hidden_size=16, num_heads=4),
        torch.randn(1, 2, 16),
        id="linear_attn",
    ),
    pytest.param(
        lambda: LightNetAttention(hidden_size=16, num_heads=4),
        torch.randn(1, 2, 16),
        id="lightnet",
    ),
    pytest.param(
        lambda: LogLinearMamba2(hidden_size=64, num_heads=1, head_dim=64),
        torch.randn(1, 1, 64),
        id="log_linear_mamba2",
    ),
    pytest.param(
        lambda: Mamba(hidden_size=16),
        torch.randn(1, 2, 16),
        id="mamba",
    ),
    pytest.param(
        lambda: Mamba2(hidden_size=16, num_heads=2, head_dim=16),
        torch.randn(1, 2, 16),
        id="mamba2",
    ),
    pytest.param(
        lambda: RWKV6Attention(hidden_size=16, num_heads=4),
        torch.randn(1, 2, 16),
        id="rwkv6",
    ),
    pytest.param(
        lambda: ABCAttention(hidden_size=16, num_heads=4),
        torch.randn(1, 2, 16),
        id="abc",
    ),
    pytest.param(
        lambda: HGRNAttention(hidden_size=16),
        torch.randn(1, 2, 16),
        id="hgrn",
    ),
    pytest.param(
        lambda: SimpleGatedLinearAttention(hidden_size=16, num_heads=4),
        torch.randn(1, 2, 16),
        id="simple_gla",
    ),
    pytest.param(
        lambda: RWKV7Attention(hidden_size=16, head_dim=None, num_heads=4),
        torch.randn(1, 2, 16),
        id="rwkv7",
    ),
    pytest.param(
        lambda: RodimusAttention(hidden_size=16),
        torch.randn(1, 2, 16),
        id="rodimus",
    ),
    pytest.param(
        lambda: MultiScaleRetention(hidden_size=16, num_heads=4),
        torch.randn(1, 2, 16),
        id="multiscale_retention",
    ),
    pytest.param(
        lambda: MomAttention(hidden_size=16, num_heads=4),
        torch.randn(1, 2, 16),
        id="mom",
    ),
    pytest.param(
        lambda: MesaNet(hidden_size=16, num_heads=4),
        torch.randn(1, 2, 16),
        id="mesa_net",
    ),
    pytest.param(
        lambda: KimiDeltaAttention(hidden_size=16, num_heads=4),
        torch.randn(1, 2, 16),
        id="kda",
    ),
    pytest.param(
        lambda: HGRN2Attention(hidden_size=16, num_heads=4, expand_ratio=None),
        torch.randn(1, 2, 16),
        id="hgrn2",
    ),
    pytest.param(
        lambda: GatedSlotAttention(hidden_size=16, num_heads=4),
        torch.randn(1, 2, 16),
        id="gsa",
    ),
    pytest.param(
        lambda: GatedLinearAttention(hidden_size=16, num_heads=4),
        torch.randn(1, 2, 16),
        id="gla",
    ),
    pytest.param(
        lambda: GatedDeltaProduct(hidden_size=16, num_heads=4),
        torch.randn(1, 2, 16),
        id="gated_deltaproduct",
    ),
    pytest.param(
        lambda: GatedDeltaNet(hidden_size=16, num_heads=4),
        torch.randn(1, 2, 16),
        id="gated_deltanet",
    ),
    pytest.param(
        lambda: DeltaNet(hidden_size=16, num_heads=4),
        torch.randn(1, 2, 16),
        id="delta_net",
    ),
    pytest.param(
        lambda: Comba(hidden_size=16, num_heads=4),
        torch.randn(1, 2, 16),
        id="comba",
    ),
]


@pytest.mark.parametrize(
    ("builder", "hidden_states"),
    CACHE_REQUIRES_LAYER_IDX_CASES,
)
def test_cache_requires_layer_idx(builder, hidden_states):
    layer, hidden_states = prepare_layer_and_inputs(builder, hidden_states)
    with pytest.raises(ValueError, match="requires `layer_idx`"):
        layer(hidden_states=hidden_states, past_key_values=DummyCache([{}]))


@pytest.mark.parametrize(
    ("builder", "hidden_states"),
    [
        pytest.param(
            lambda: LinearAttention(hidden_size=16, num_heads=4),
            torch.randn(1, 2, 16),
            id="linear_attn",
        ),
        pytest.param(
            lambda: LightNetAttention(hidden_size=16, num_heads=4),
            torch.randn(1, 2, 16),
            id="lightnet",
        ),
        pytest.param(
            lambda: Mamba(hidden_size=16),
            torch.randn(1, 2, 16),
            id="mamba",
        ),
        pytest.param(
            lambda: Mamba2(hidden_size=16, num_heads=2, head_dim=16),
            torch.randn(1, 2, 16),
            id="mamba2",
        ),
        pytest.param(
            lambda: RWKV6Attention(hidden_size=16, num_heads=4),
            torch.randn(1, 2, 16),
            id="rwkv6",
        ),
    ],
)
def test_layers_without_cache_still_work_with_layer_idx_none(builder, hidden_states):
    layer, hidden_states = prepare_layer_and_inputs(builder, hidden_states)
    output, _, returned_cache = layer(hidden_states=hidden_states)

    assert output.shape == hidden_states.shape
    assert returned_cache is None
