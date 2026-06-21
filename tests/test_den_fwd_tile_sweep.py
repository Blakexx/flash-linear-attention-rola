"""#92 CONFIG-SWEEP correctness gate for the per-state-den FORWARD kernels.

The two forward den kernels `_den_fwd_intra` / `_den_fwd_inter` are now @triton.autotune'd
with a feature-tile knob BD in {16,32,64,128} (the small-SMEM fallback). On the dev card the
autotuner only ever picks the big tile, so the small-tile fallback configs are NEVER exercised
by the normal correctness suite. This test FORCES each config one at a time (by replacing the
Autotuner's `.configs` with a single config so the autotuner cannot override it — when
len(configs)==1 Triton skips pruning/benchmarking and uses configs[0] directly) and asserts the
den matches the fp64 reference oracle for:

  - both kernels (intra + inter run together for every forced config),
  - both den paths: kappa (additive, USE_G=False) AND per_state (same den fn) and
    RLA-kappa (rola_perstate_den_triton) AND GLA-kappa (rola_perstate_den_gla_triton),
  - dqk in {16,32,64,128}, a couple of nc values.

EVERY config (incl. the tiny tiles the autotuner never selects) must match fp64 — that is what
proves the small-SMEM fallback tiles are actually correct.
"""
import contextlib
import itertools

import torch

import fla_rola.ops.rola.chunk as C
from fla_rola.ops.rola.chunk import (
    rola_perstate_den_triton,
    rola_perstate_den_gla_triton,
)
from fla_rola.ops.rola.naive import (
    _rola_perstate_den,
    _rola_gla_perstate_den,
)

DEVICE = 'cuda'
DTYPE = torch.float32          # fp32 inputs; reference runs in fp64
CHUNK = C._CHUNK               # use the GLA chunk (caps both paths; den triton clamps to _CHUNK)


def _fold(t):
    """[B,L,H,*] -> [B*H,L,*] (the layout the public den fns expect)."""
    B, L, H = t.shape[:3]
    return t.permute(0, 2, 1, 3).reshape(B * H, L, t.shape[-1])


@contextlib.contextmanager
def force_config(kernel, cfg):
    """Pin a triton.autotune kernel to a single config. len(configs)==1 => the Autotuner skips
    pruning/benchmarking and uses configs[0] verbatim, so the forced tile cannot be overridden.
    Also clears the autotune cache so a previously-cached pick can't leak in."""
    saved_configs = kernel.configs
    saved_cache = dict(kernel.cache)
    kernel.configs = [cfg]
    kernel.cache.clear()
    try:
        yield
    finally:
        kernel.configs = saved_configs
        kernel.cache.clear()
        kernel.cache.update(saved_cache)


def _bd_configs(kernel):
    """Distinct configs keyed by BD (the feature-tile knob). We only need one warp/stage rep per BD:
    the BD value is what changes the SRAM footprint / tiling math the test is proving correct."""
    by_bd = {}
    for cfg in kernel.configs:
        bd = cfg.kwargs['BD']
        by_bd.setdefault(bd, cfg)
    return by_bd


def _ref_rla(q, k, w):
    d = _rola_perstate_den(q.double(), k.double(), w.double(), chunk=CHUNK)
    return _fold(d)


def _ref_gla(q, k, w, ld):
    d = _rola_gla_perstate_den(q.double(), k.double(), w.double(), ld.double(), chunk=CHUNK)
    return _fold(d)


def _run_path(name, dqk, nc):
    """Run RLA-kappa and GLA-kappa den for every forced (intra-BD x inter-BD) config and compare
    to the fp64 oracle. Returns (n_configs_tested, max_rel_err, [(cfg_desc, err), ...])."""
    torch.manual_seed(0xC0FFEE + dqk * 131 + nc)
    B, H, L = 1, 2, 3 * CHUNK + 5     # span >2 chunks + a ragged tail (exercises masking)
    q = torch.randn(B, L, H, dqk, device=DEVICE, dtype=DTYPE) * 0.3
    k = torch.randn(B, L, H, dqk, device=DEVICE, dtype=DTYPE) * 0.3
    w = torch.rand(B, L, H, nc, device=DEVICE, dtype=DTYPE)
    ld = -torch.rand(B, L, H, nc, device=DEVICE, dtype=DTYPE) * 0.1   # small negative log-decay

    qf, kf, wf, ldf = _fold(q), _fold(k), _fold(w), _fold(ld)

    ref_rla = _ref_rla(q, k, w)
    ref_gla = _ref_gla(q, k, w, ld)

    intra_cfgs = _bd_configs(C._den_fwd_intra)
    inter_cfgs = _bd_configs(C._den_fwd_inter)

    results = []
    max_err = 0.0
    n = 0
    for (bd_i, ci), (bd_j, cj) in itertools.product(intra_cfgs.items(), inter_cfgs.items()):
        with force_config(C._den_fwd_intra, ci), force_config(C._den_fwd_inter, cj):
            d_rla = rola_perstate_den_triton(qf, kf, wf, chunk=CHUNK)
            d_gla = rola_perstate_den_gla_triton(qf, kf, wf, ldf, chunk=CHUNK)
        for tag, got, ref in (('RLA', d_rla, ref_rla), ('GLA', d_gla, ref_gla)):
            # Global (Frobenius) relative error — robust to den elements crossing zero, where an
            # elementwise rel-err blows up despite tiny absolute error. fp32 kernel vs fp64 oracle.
            err = ((got.double() - ref).norm() / (ref.norm() + 1e-12)).item()
            desc = f"{name} {tag} dqk={dqk} nc={nc} intra_BD={bd_i} inter_BD={bd_j}"
            results.append((desc, err))
            max_err = max(max_err, err)
            n += 1
    return n, max_err, results


def main():
    assert torch.cuda.is_available(), "CUDA required for the den-fwd tile sweep"
    DQKS = (16, 32, 64, 128)
    NCS = (8, 24)
    TOL = 2e-3   # fp32 kernel vs fp64 oracle, relative

    all_results = []
    total = 0
    worst = 0.0
    for dqk, nc in itertools.product(DQKS, NCS):
        n, mx, res = _run_path('per_state', dqk, nc)
        all_results.extend(res)
        total += n
        worst = max(worst, mx)
        print(f"[dqk={dqk:3d} nc={nc:2d}] {n:2d} configs  max_rel_err={mx:.2e}  "
              f"{'PASS' if mx < TOL else 'FAIL'}")

    fails = [(d, e) for (d, e) in all_results if e >= TOL]
    bds = sorted(set(c.kwargs['BD'] for c in C._den_fwd_intra.configs))
    print(f"\nfeature-tile BD configs swept (intra & inter): {bds}")
    print(f"total kernel runs compared to fp64: {total}")
    print(f"global max relative error: {worst:.2e}  (tol {TOL:.0e})")
    if fails:
        print(f"\n{len(fails)} FAILING config(s):")
        for d, e in fails[:20]:
            print(f"  {d}: rel_err={e:.2e}")
        raise SystemExit(1)
    print("\nALL den-fwd tile configs match the fp64 oracle. PASS")


if __name__ == '__main__':
    main()
