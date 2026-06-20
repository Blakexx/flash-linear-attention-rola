# Copyright (c) 2023-2025, RoLA authors.

"""rola_instance — named RoLA preset → fla_rola.layers.RoLA kwargs.

Ports the preset map (axes: inner kernel × feature map × normalization × router symmetry) so the
HF/LM config can name a cell. Only the canonical kernels (rla, gla_scalar — both route through the
shared-Gram chunk_rola op) are buildable as a first-class layer. The per-channel virtual-head GLA
(gla_virtual) and routed delta-rule (gdn) fold nc into virtual heads — a different code path that
stays in the rola.py shim as experimental variants; naming them here raises.
"""

ROLA_INSTANCES = (
    'rola-rla-asym', 'rola-rla-sym', 'rola-rla-asym-ps', 'rola-rla-sym-ps',
    'rola-rla-kappa-asym', 'rola-rla-kappa-sym', 'rola-rla-asym-tieinit',
    'rola-gla-scalar-sym', 'rola-gla-scalar-norm-sym',
    'rola-gla-scalar-asym', 'rola-gla-scalar-norm-asym',
    'rola-gla-kappa-sym', 'rola-gla-kappa-asym',
    'rola-hedgehog-sym', 'rola-hedgehog-asym',
    'rola-based-sym', 'rola-based-asym', 'rola-rebased-sym', 'rola-rebased-asym',
)

# experimental (virtual-head / delta) — not first-class layers; live in the rola.py shim.
_EXPERIMENTAL = ('rola-gla-sym', 'rola-gla-norm-sym', 'rola-gdn-sym')


def rola_instance(name, head_k_dim, head_v_dim, num_states, num_heads=8):
    """Return fla_rola.layers.RoLA kwargs for a named instance."""
    common = dict(head_k_dim=head_k_dim, head_v_dim=head_v_dim, num_states=num_states,
                  num_heads=num_heads, use_short_conv=False)
    sym = name.endswith('-sym')

    # RLA family (AdditiveKernel; phi = feature map; global/per_state/kappa norm).
    if name in ('rola-rla-asym', 'rola-rla-sym'):
        return dict(kernel='rla', phi='elu', state_norm='global', tie_routers=sym, **common)
    if name in ('rola-rla-asym-ps', 'rola-rla-sym-ps'):
        return dict(kernel='rla', phi='elu', state_norm='per_state', tie_routers=sym, **common)
    if name in ('rola-rla-kappa-asym', 'rola-rla-kappa-sym'):
        return dict(kernel='rla', phi='elu', state_norm='kappa', tie_routers=sym, **common)
    if name == 'rola-rla-asym-tieinit':
        return dict(kernel='rla', phi='elu', state_norm='global', tie_routers=False,
                    tie_router_init=True, **common)
    if name in ('rola-hedgehog-sym', 'rola-hedgehog-asym'):
        return dict(kernel='rla', phi='hedgehog', state_norm='global', tie_routers=sym, **common)
    if name in ('rola-based-sym', 'rola-based-asym'):
        return dict(kernel='rla', phi='based', state_norm='global', tie_routers=sym, **common)
    if name in ('rola-rebased-sym', 'rola-rebased-asym'):
        return dict(kernel='rla', phi='rebased', state_norm='global', tie_routers=sym, **common)

    # scalar-GLA family (ScalarGLAKernel; per-state scalar decay). '-norm-' => global V+1 partition.
    if name.startswith('rola-gla-scalar'):
        return dict(kernel='gla_scalar', phi='elu',
                    state_norm=('global' if '-norm-' in name else 'raw'), tie_routers=sym, **common)
    if name.startswith('rola-gla-kappa'):
        return dict(kernel='gla_scalar', phi='elu', state_norm='kappa', tie_routers=sym, **common)
    if name.startswith('rola-gla-ps'):
        return dict(kernel='gla_scalar', phi='elu', state_norm='per_state', tie_routers=sym, **common)

    if name in _EXPERIMENTAL:
        raise ValueError(
            f"{name!r} uses a virtual-head/delta kernel not supported by the canonical "
            f"fla_rola.layers.RoLA; it remains in the rola.py shim (experimental)."
        )
    raise ValueError(f"unknown RoLA instance: {name!r}")
