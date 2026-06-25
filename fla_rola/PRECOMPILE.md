# RoLA parallel cache-warming (`fla_rola.precompile`)

The first RoLA fwd+bwd at a new `(dqk, dv, nc)` shape pays Triton **codegen** for every
surviving autotuner config — single-threaded, minutes-long, blocking the GPU. The kernels
are tiny; the compiler is the wall. `fla_rola.precompile` compiles those configs **ahead of
time, CPU-only, in parallel**, so the real run is a pure cache hit.

## Mechanism

* `@triton.autotune` wraps an `Autotuner` whose `.fn` is the underlying `JITFunction`.
  `kernel.fn.warmup(*args, grid=grid, **meta)` runs Triton's compile path but the launch is
  guarded behind `if not warmup:` in `jit.py` → **CPU only, no GPU kernel launch**.
* Tensor args are passed as a **bare `torch.dtype`**; `warmup` wraps them in a `MockTensor`
  with `data_ptr()==0` → a 16-byte-aligned spec, which matches real (aligned) torch tensors,
  so the cache key matches and the real run hits the cache.
* Triton cache writes are atomic (`os.replace`) → one shared `TRITON_CACHE_DIR` is safe
  across worker processes (spawn, never fork — fork crashes on CUDA).

## API

```python
from fla_rola.precompile import warm_gate, ShapeSpec

# Pre-benchmark gate: warm + BLOCK until ready. Call once before the bench loop.
summary = warm_gate([
    {"path": "rla", "dqk": 16, "dv": 16, "nc": 8},     # dict spec
    ShapeSpec(path="gla", dqk=16, dv=24, nc=64),         # or ShapeSpec
])
# -> {'n_warmed': N, 'wall_s': ..., 'total_wall_s': ..., 'n_workers': nproc,
#     'n_shapes': ..., 'per_kernel': {...}, 'failures': []}
```

Shape spec keys: `dqk`, `dv`, `nc` (the autotuner key, required); `path` `'rla'`|`'gla'`
(default `'rla'`); `norm` `'kappa'|'raw'|'global'|'per_state'` (default `'kappa'`); `T`
(capture seqlen, default 64).

Lower-level building blocks:

```python
from fla_rola.precompile import enumerate_shapes, parallel_warm
work, cap = enumerate_shapes(specs)         # capture pass (GPU) -> picklable warm-list
summary  = parallel_warm(work, n_workers=16)  # CPU-only spawn-pool codegen
```

`warm_gate` honors `$TRITON_CACHE_DIR` (or pass `cache_dir=...`); defaults to Triton's
`~/.triton/cache`. Idempotent — already-cached configs are fast no-ops.

## In a benchmark

```python
from fla_rola.precompile import warm_gate
warm_gate(my_bench_shapes, cache_dir=os.environ["TRITON_CACHE_DIR"])
# ... then run the timed loop; every shape's kernels are already compiled.
```

## In tests

A session-scoped autouse fixture (`tests/conftest.py::warm_rola_cache`) warms the registered
shapes at session start when `ROLA_WARM_SESSION=1`:

```bash
ROLA_WARM_SESSION=1 pytest tests/ops/test_rola.py
```

Extend the warm set from a test module (at import time):

```python
from fla_rola.precompile import register_test_shapes
register_test_shapes([{"path": "rla", "dqk": 32, "dv": 32, "nc": 16}])
```

## Validation (proven)

* **Full cache hit:** warm a fresh cache for representative RLA shapes, then a real fwd+bwd
  compiles the autotuned kernels **0 times** (every selected config is pre-warmed).
* **Same choice set:** the warmer compiles *exactly* the config set the cold autotuner
  benches (cold-compile-count == warm-config-count, per kernel) — warming cannot change
  which config the autotuner can select. (Run-to-run `best_config` *does* vary on these
  sub-microsecond kernels — that is `do_bench` jitter, identical cold-vs-cold, not a warming
  artifact.)
* **CPU-only:** GPU memory is flat (±a few MiB) across the warm — no kernel launch.
* **Speedup:** parallel (16 workers) vs serial warm of the work-list.

## Caveat

A tensor arg that is **not** 16-byte aligned (an oddly-offset / non-contiguous view)
specializes to a different cache key than the `data_ptr()==0` mock, so that one kernel
misses and falls back to a normal cold compile at run time — **no error, just no speedup**
for that kernel. RoLA's entrypoints fold/contiguous-ify their operands, so the mock spec
matches in practice.
