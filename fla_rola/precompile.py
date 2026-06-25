"""Parallel precompile / Triton cache-warming for the RoLA autotuned kernels.

The expensive part of the *first* RoLA fwd+bwd at a new (dqk, dv, nc) shape is Triton
**codegen** (LLVM → PTX → cubin) for every surviving autotuner config — minutes, single
threaded, blocking the GPU. The kernels themselves are tiny; the compiler is the wall.

This module compiles those configs **ahead of time, CPU-only, in parallel**, so the real
run is a pure cache hit. The mechanism (proven by the de-risk research):

* ``@triton.autotune`` wraps an ``Autotuner`` whose ``.fn`` is the underlying ``JITFunction``.
  ``kernel.fn.warmup(*args, grid=grid, **meta)`` runs Triton's compile path but the launch
  is guarded behind ``if not warmup:`` in ``jit.py`` → **CPU only, no GPU kernel launch**.
* For tensor args, pass a **bare ``torch.dtype``** (e.g. ``torch.bfloat16``). Triton's
  ``warmup`` auto-wraps it in a ``MockTensor`` with ``data_ptr()==0`` → a 16-byte-aligned
  spec. Real torch tensors are (almost always) 16-byte aligned too, so the cache key
  matches and the real run hits the cache (verified: 0 recompiles, same config selected).
* Triton cache writes are atomic (``os.replace``) → safe to share one ``TRITON_CACHE_DIR``
  across worker processes.

CAVEAT: a tensor arg that is **not** 16-byte aligned (e.g. a non-contiguous / oddly-offset
view) specializes to a *different* cache key than the ``data_ptr()==0`` mock. The warmed
entry then misses and that kernel falls back to a normal (cold) compile at run time — no
error, just no speedup for that one kernel. RoLA's entrypoints fold/contiguous-ify their
operands, so in practice the mock spec matches.

Usage
-----
Before a benchmark (block until the cache is hot)::

    from fla_rola.precompile import warm_gate
    summary = warm_gate([
        {"path": "rla", "dqk": 16, "dv": 16, "nc": 8},
        {"path": "rla", "dqk": 16, "dv": 24, "nc": 64},
    ])
    print(summary)            # {'n_warmed': ..., 'wall_s': ..., 'n_workers': ...}

In tests, a session-scoped autouse fixture warms a registered shape set at session start
(see ``tests/conftest.py`` / ``register_test_shapes``).

The shape spec is a list of dicts (or ``ShapeSpec``) with keys:
    path:   'rla' (g=None) or 'gla' (scalar log-decay)         [default 'rla']
    dqk:    feature dim of q/k                                  [required]
    dv:     value dim                                           [required]
    nc:     number of states (rank)                             [required]
    norm:   'kappa' | 'raw' | 'global' | 'per_state'           [default 'kappa']
    T:      sequence length used to drive capture               [default 64]
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

# The autotuned kernels we warm, by entrypoint path. These are the attribute names of the
# Autotuner objects on ``fla_rola.ops.rola.chunk``. (Used for reporting / validation; the
# capture hook discovers whatever actually fires, so this list need not be exhaustive.)
RLA_KERNELS = (
    "_rola_fwd_intra", "_rola_fwd_inter", "_scan_S", "_scan_dS",
    "_par_grad_rla_qr", "_par_grad_rla_kwv",
    "_den_fwd_intra", "_den_fwd_inter", "_den_bwd_scan", "_den_grad",
)
GLA_KERNELS = (
    "_rola_gla_fwd_intra", "_rola_gla_fwd_inter", "_scan_S", "_scan_dS",
    "_par_grad_gla_qr", "_par_grad_gla_kwv",
    "_den_fwd_intra", "_den_fwd_inter", "_den_gla_bwd_scan", "_den_gla_grad",
)

# meta keys that are launch-controls, NOT kernel constexprs — dropped before merging a
# config's all_kwargs() into the warmup meta (the config supplies its own values for these).
_LAUNCH_META = frozenset({
    "grid", "warmup", "num_warps", "num_ctas", "num_stages",
    "num_buffers_warp_spec", "num_consumer_groups",
    "reg_dec_producer", "reg_inc_consumer", "maxnreg",
})


@dataclass(frozen=True)
class ShapeSpec:
    """One RoLA shape to warm. dqk/dv/nc are the autotuner key; the rest drive capture."""
    dqk: int
    dv: int
    nc: int
    path: str = "rla"          # 'rla' | 'gla'
    norm: str = "kappa"
    T: int = 64

    @classmethod
    def coerce(cls, spec) -> ShapeSpec:
        if isinstance(spec, cls):
            return spec
        if isinstance(spec, dict):
            return cls(**spec)
        raise TypeError(f"shape spec must be a dict or ShapeSpec, got {type(spec)!r}")

    def key(self) -> tuple:
        return (self.path, self.dqk, self.dv, self.nc, self.norm, self.T)


# ---------------------------------------------------------------------------
# Arg-descriptor capture (first-call runtime hook — the task-approved fallback)
# ---------------------------------------------------------------------------
# A static per-kernel arg table would have to be hand-maintained across 18 kernels with
# many internal call sites and stride-derived scalars; instead we drive ONE small real
# fwd+bwd per shape and intercept JITFunction.run to record, per kernel name, the arg
# "kinds" (tensor dtype vs scalar value) + the meta kwargs + grid. The warmer rebuilds
# tensors as bare torch.dtype, so this capture runs once and is GPU-free thereafter for
# every config of that shape.

@dataclass
class _Captured:
    """Per (kernel-name) record from the capture run: arg descriptors + meta + grid."""
    arg_descs: list = field(default_factory=list)   # [(kind, dtype_str|typename, value|shape)]
    meta: dict = field(default_factory=dict)
    grid: tuple = (1,)


def _make_inputs(spec: ShapeSpec, device="cuda", dtype=None):
    """Small real inputs (B=H=1) for the capture fwd+bwd at this shape's dqk/dv/nc."""
    import torch
    if dtype is None:
        dtype = torch.bfloat16
    B, H, T = 1, 1, spec.T
    g = torch.Generator(device=device).manual_seed(0)

    def mk(last):
        return torch.randn(B, T, H, last, device=device, dtype=dtype,
                           generator=g, requires_grad=True)
    q, k, v = mk(spec.dqk), mk(spec.dqk), mk(spec.dv)
    r, w = mk(spec.nc), mk(spec.nc)
    kappa = torch.rand(B, T, H, 1, device=device, dtype=dtype, generator=g)
    out = dict(q=q, k=k, v=v, r=r, w=w, kappa=kappa)
    if spec.path == "gla":
        # decay within the fp32-safe floor (see chunk._GLA_FLOOR); small negative logs.
        out["g"] = (-torch.rand(B, T, H, spec.nc, device=device, dtype=torch.float32,
                                generator=g) * 0.5).clamp(min=-2.0)
    return out


def capture_shape(spec: ShapeSpec) -> dict:
    """Run ONE small real fwd+bwd for ``spec`` on the GPU, intercepting JITFunction.run to
    record per-kernel arg descriptors. Returns {kernel_name: _Captured}. Requires CUDA.

    This is the only GPU-touching step; it runs once per shape and is cheap relative to the
    codegen it enables to be parallelized."""
    import torch
    import triton.runtime.jit as jitmod

    from fla_rola.ops.rola.chunk import chunk_rola

    spec = ShapeSpec.coerce(spec)
    if not torch.cuda.is_available():
        raise RuntimeError("capture_shape needs CUDA to drive the real fwd+bwd")

    captured: dict[str, _Captured] = {}
    orig_run = jitmod.JITFunction.run

    def patched_run(self, *args, **kwargs):
        name = getattr(self, "__name__", None)
        if name is not None and name not in captured:
            descs = []
            for a in args:
                if torch.is_tensor(a):
                    descs.append(("tensor", str(a.dtype), tuple(a.shape)))
                else:
                    descs.append(("scalar", type(a).__name__, a))
            meta = {k: v for k, v in kwargs.items() if not callable(v)}
            captured[name] = _Captured(arg_descs=descs, meta=meta,
                                       grid=kwargs.get("grid", (1,)))
        return orig_run(self, *args, **kwargs)

    jitmod.JITFunction.run = patched_run
    try:
        ins = _make_inputs(spec)
        kw = dict(norm=spec.norm, kappa=ins["kappa"])
        if "g" in ins:
            kw["g"] = ins["g"]
        out = chunk_rola(ins["q"], ins["k"], ins["v"], ins["r"], ins["w"], **kw)
        out.sum().backward()
        torch.cuda.synchronize()
    finally:
        jitmod.JITFunction.run = orig_run
    return captured


# ---------------------------------------------------------------------------
# enumerate_shapes — (kernel, shape, config) warm-list from a spec list
# ---------------------------------------------------------------------------
def _pruned_configs(at, arg_descs, meta):
    """The autotuner configs that survive early_config_prune for this shape — i.e. the ones
    the real run would actually bench. We only warm these (warming pruned-away configs is
    pure waste). nargs is built from the captured arg descriptors."""
    import torch
    nargs = {}
    for name, (kind, a, b) in zip(at.arg_names, arg_descs):
        if kind == "tensor":
            nargs[name] = torch.empty(0, dtype=getattr(torch, a.split(".")[-1]))
        else:
            nargs[name] = b
    at.nargs = nargs
    try:
        pruned = at.prune_configs(meta)
    finally:
        at.nargs = None
    return pruned


def enumerate_shapes(specs, cache_dir=None):
    """Expand a list of shape specs into a flat warm-list of work items, one per
    (kernel, shape, surviving-config). Each item is a picklable ``WarmItem`` carrying the
    kernel name + rebuildable arg descriptors + per-config warmup meta.

    Runs the capture pass (GPU) for each distinct shape, then enumerates that shape's
    pruned configs for every autotuned kernel that fired. Returns (work, capture_summary).
    """
    from triton.runtime.autotuner import Autotuner

    from fla_rola.ops.rola import chunk as ck

    if isinstance(specs, dict):
        specs = [specs]
    specs = [ShapeSpec.coerce(s) for s in specs]
    cache_dir = cache_dir or os.environ.get("TRITON_CACHE_DIR")

    work: list[WarmItem] = []
    seen_shapes = []
    seen_keys = set()
    for spec in specs:
        if spec.key() in seen_keys:
            continue
        seen_keys.add(spec.key())
        seen_shapes.append(spec)
        captured = capture_shape(spec)
        for kname, cap in captured.items():
            at = getattr(ck, kname, None)
            if not isinstance(at, Autotuner):
                continue
            # prune meta = the captured constexpr meta (the prune fns read named_args, but
            # pass the meta through defensively for any perf-model/top_k pruning).
            prune_meta = {k: v for k, v in cap.meta.items() if k not in _LAUNCH_META}
            # grid is IGNORED by warmup (only used behind `if not warmup:` in jit.py); the
            # captured grid is often an un-picklable local closure, so store a trivial tuple.
            meta = {k: v for k, v in cap.meta.items() if k != "grid"}
            for cfg in _pruned_configs(at, cap.arg_descs, prune_meta):
                work.append(WarmItem(
                    kernel=kname,
                    arg_descs=cap.arg_descs,
                    base_meta=meta,
                    config_kwargs=cfg.all_kwargs(),
                    grid=(1,),
                    cache_dir=cache_dir,
                    shape_key=spec.key(),
                ))
    summary = {
        "n_shapes": len(seen_shapes),
        "shapes": [s.key() for s in seen_shapes],
        "n_work": len(work),
    }
    return work, summary


@dataclass
class WarmItem:
    """One picklable unit of warm work: compile ONE config of ONE kernel for ONE shape."""
    kernel: str
    arg_descs: list
    base_meta: dict
    config_kwargs: dict
    grid: tuple
    cache_dir: str | None
    shape_key: tuple


# ---------------------------------------------------------------------------
# parallel_warm — spawn-pool, CPU-only codegen of the warm-list
# ---------------------------------------------------------------------------
def _worker_init():
    """Pool initializer: import the kernel module ONCE per worker so per-config calls don't
    re-pay the (heavy) import. Spawn context → fresh interpreter, no CUDA fork hazard."""
    import fla_rola.ops.rola.chunk  # noqa: F401  (warm the import)


def _warm_one(item: WarmItem):
    """Compile one config CPU-only via JITFunction.warmup. Idempotent: if the cache already
    has this kernel/config, Triton's warmup is a fast no-op (it loads the cached artifact).
    Returns (kernel, ok, seconds, pid, error_or_None)."""
    import torch
    from triton.runtime.autotuner import Autotuner

    from fla_rola.ops.rola import chunk as ck

    if item.cache_dir:
        os.environ["TRITON_CACHE_DIR"] = item.cache_dir

    t0 = time.time()
    try:
        at = getattr(ck, item.kernel)
        assert isinstance(at, Autotuner), f"{item.kernel} is not an Autotuner"
        fn = at.fn

        args = []
        for kind, a, b in item.arg_descs:
            if kind == "tensor":
                args.append(getattr(torch, a.split(".")[-1]))  # bare torch.dtype
            else:
                args.append(b)

        meta = {k: v for k, v in item.base_meta.items() if k not in _LAUNCH_META}
        meta.update(item.config_kwargs)
        fn.warmup(*args, grid=item.grid, **meta)
        return (item.kernel, True, time.time() - t0, os.getpid(), None)
    except Exception as e:  # noqa: BLE001 — report, don't crash the pool
        return (item.kernel, False, time.time() - t0, os.getpid(), repr(e))


def parallel_warm(work, n_workers=None, cache_dir=None, progress=True):
    """Compile every WarmItem in ``work`` CPU-only across a spawn ProcessPool.

    Idempotent (already-cached configs are fast no-ops). Returns a summary dict with the
    wall-clock, per-kernel counts, and any failures. ``n_workers`` defaults to the physical
    core count (os.cpu_count())."""
    work = list(work)
    if cache_dir:
        for it in work:
            it.cache_dir = cache_dir
    n_workers = n_workers or os.cpu_count() or 4
    n_workers = max(1, min(n_workers, len(work) or 1))

    t0 = time.time()
    results = []
    if not work:
        return {"n_warmed": 0, "n_failed": 0, "wall_s": 0.0, "n_workers": 0,
                "per_kernel": {}, "failures": []}

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                             initializer=_worker_init) as ex:
        for i, res in enumerate(ex.map(_warm_one, work), 1):
            results.append(res)
            if progress and (i % 16 == 0 or i == len(work)):
                print(f"[warm] {i}/{len(work)} configs "
                      f"({time.time() - t0:.1f}s elapsed)", flush=True)
    wall = time.time() - t0

    per_kernel: dict[str, int] = {}
    failures = []
    for kname, ok, _dt, _pid, err in results:
        if ok:
            per_kernel[kname] = per_kernel.get(kname, 0) + 1
        else:
            failures.append((kname, err))
    return {
        "n_warmed": sum(1 for r in results if r[1]),
        "n_failed": len(failures),
        "wall_s": wall,
        "n_workers": n_workers,
        "per_kernel": per_kernel,
        "failures": failures,
    }


def serial_warm(work, cache_dir=None, progress=False):
    """In-process serial warm of the work-list (the baseline parallel_warm is compared to).
    Same return shape as parallel_warm."""
    work = list(work)
    if cache_dir:
        for it in work:
            it.cache_dir = cache_dir
    _worker_init()
    t0 = time.time()
    results = [_warm_one(it) for it in work]
    wall = time.time() - t0
    per_kernel: dict[str, int] = {}
    failures = []
    for kname, ok, _dt, _pid, err in results:
        if ok:
            per_kernel[kname] = per_kernel.get(kname, 0) + 1
        else:
            failures.append((kname, err))
    return {"n_warmed": sum(1 for r in results if r[1]), "n_failed": len(failures),
            "wall_s": wall, "n_workers": 1, "per_kernel": per_kernel, "failures": failures}


# ---------------------------------------------------------------------------
# warm_gate — the pre-benchmark / pre-test gate
# ---------------------------------------------------------------------------
def warm_gate(specs, cache_dir=None, n_workers=None, progress=True):
    """Pre-warm the Triton cache for ``specs`` and BLOCK until ready. The reusable gate to
    call once before a benchmark loop or at test-session start.

    cache_dir defaults to $TRITON_CACHE_DIR (or Triton's default ~/.triton/cache). Returns
    a summary dict (n_shapes, n_warmed, wall_s, per_kernel, failures)."""
    cache_dir = cache_dir or os.environ.get("TRITON_CACHE_DIR")
    if cache_dir:
        os.environ["TRITON_CACHE_DIR"] = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    t0 = time.time()
    work, cap_summary = enumerate_shapes(specs, cache_dir=cache_dir)
    warm_summary = parallel_warm(work, n_workers=n_workers, cache_dir=cache_dir,
                                 progress=progress)
    warm_summary["total_wall_s"] = time.time() - t0
    warm_summary.update(cap_summary)
    return warm_summary


# ---------------------------------------------------------------------------
# Test-shape registry — the session-scoped pytest fixture warms whatever is here
# ---------------------------------------------------------------------------
# The default small representative set covering the RLA + GLA paths at the shapes the
# RoLA op tests exercise. A test module may extend this via register_test_shapes().
DEFAULT_TEST_SHAPES = [
    ShapeSpec(path="rla", dqk=16, dv=16, nc=8),
    ShapeSpec(path="rla", dqk=16, dv=24, nc=64),
    ShapeSpec(path="gla", dqk=16, dv=16, nc=8),
]

_REGISTERED: list[ShapeSpec] = list(DEFAULT_TEST_SHAPES)


def register_test_shapes(specs, replace=False):
    """Register shape specs for the session-scoped warm fixture to compile at session start.
    ``replace=True`` overwrites the default set; otherwise appends (de-duplicated)."""
    global _REGISTERED
    coerced = [ShapeSpec.coerce(s) for s in (specs if isinstance(specs, (list, tuple)) else [specs])]
    if replace:
        _REGISTERED = []
    seen = {s.key() for s in _REGISTERED}
    for s in coerced:
        if s.key() not in seen:
            _REGISTERED.append(s)
            seen.add(s.key())
    return list(_REGISTERED)


def registered_test_shapes():
    """The current registered warm shape set (used by the pytest fixture)."""
    return list(_REGISTERED)
