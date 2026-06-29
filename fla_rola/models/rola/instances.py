# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""rola_instance — named RoLA preset → fla_rola.layers.RoLA kwargs.

Initial-launch scope: only the FUSED, elu-feature-map kernels that route through the in-kernel
tree-routed `chunk_rola_routed` op — the `rla` family (global / per_state / kappa norm) and the
scalar-gated `gla_scalar` family. The alternative feature maps (hedgehog / based / rebased) and the
non-fused virtual-head GLA / routed-delta variants are not supported.
"""

ROLA_INSTANCES = (
    'rola-rla-asym', 'rola-rla-sym', 'rola-rla-asym-ps', 'rola-rla-sym-ps',
    'rola-rla-kappa-asym', 'rola-rla-kappa-sym', 'rola-rla-asym-tieinit',
    'rola-gla-scalar-sym', 'rola-gla-scalar-norm-sym',
    'rola-gla-scalar-asym', 'rola-gla-scalar-norm-asym',
    'rola-gla-kappa-sym', 'rola-gla-kappa-asym',
    'rola-rla-kappa-asym-conv', 'rola-gla-kappa-asym-conv',
)


def rola_instance(name, head_k_dim, head_v_dim, states_per_head, num_heads=8):
    """Return fla_rola.layers.RoLA kwargs for a named instance."""
    use_short_conv = name.endswith('-conv')
    if use_short_conv and name not in ROLA_INSTANCES:
        raise ValueError(f"unknown / unsupported RoLA instance: {name!r}; known: {ROLA_INSTANCES}")
    base_name = name[:-5] if use_short_conv else name
    common = dict(head_k_dim=head_k_dim, head_v_dim=head_v_dim, states_per_head=states_per_head,
                  num_heads=num_heads, use_short_conv=use_short_conv)
    sym = base_name.endswith('-sym')

    # RLA family (phi = elu; global / per_state / kappa norm).
    if base_name in ('rola-rla-asym', 'rola-rla-sym'):
        return dict(kernel='rla', phi='elu', state_norm='global', tie_routers=sym, **common)
    if base_name in ('rola-rla-asym-ps', 'rola-rla-sym-ps'):
        return dict(kernel='rla', phi='elu', state_norm='per_state', tie_routers=sym, **common)
    if base_name in ('rola-rla-kappa-asym', 'rola-rla-kappa-sym'):
        return dict(kernel='rla', phi='elu', state_norm='kappa', tie_routers=sym, **common)
    if base_name == 'rola-rla-asym-tieinit':
        return dict(kernel='rla', phi='elu', state_norm='global', tie_routers=False,
                    tie_router_init=True, **common)

    # scalar-GLA family (per-state scalar decay). '-norm-' => global V+1 partition.
    if base_name.startswith('rola-gla-scalar'):
        return dict(kernel='gla_scalar', phi='elu',
                    state_norm=('global' if '-norm-' in base_name else 'raw'), tie_routers=sym, **common)
    if base_name.startswith('rola-gla-kappa'):
        return dict(kernel='gla_scalar', phi='elu', state_norm='kappa', tie_routers=sym, **common)
    if base_name.startswith('rola-gla-ps'):
        return dict(kernel='gla_scalar', phi='elu', state_norm='per_state', tie_routers=sym, **common)

    raise ValueError(f"unknown / unsupported RoLA instance: {name!r}; known: {ROLA_INSTANCES}")
