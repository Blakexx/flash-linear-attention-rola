# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# Routed RoLA: a first-class shared-gram linear-attention operator (read/write routing over `nc`
# states at matched total state), extending simple_gla. `chunk_rola` is the norm-aware public entry
# point; the lower-level Triton kernels are exposed for the correctness harness.
from fla_rola.ops.rola.chunk import (
    rola_gla_triton,
    rola_perstate_den_gla_triton,
    rola_perstate_den_triton,
    rola_rla_triton,
)
from fla_rola.ops.rola.interface import chunk_rola

__all__ = [
    'chunk_rola',
    'rola_rla_triton',
    'rola_gla_triton',
    'rola_perstate_den_triton',
    'rola_perstate_den_gla_triton',
]
