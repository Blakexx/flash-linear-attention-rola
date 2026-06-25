# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""Tests for the parallel cache-warming utility (`fla_rola.precompile`).

Triton keeps a process-wide IN-MEMORY compiled-kernel cache, so counting compiles within a
single pytest process is contaminated by earlier tests (and by the capture pass itself). The
hermetic, deterministic signal is the on-disk cache: each compiled (kernel, signature, config)
writes exactly one ``__grp__<name>.json`` group manifest. We therefore run the cache-sensitive
assertions in FRESH subprocesses with FRESH ``TRITON_CACHE_DIR``s and count those manifests.

Gates:
  * enumerate_shapes captures the RoLA autotuned kernels and yields a picklable warm-list.
  * config-set equality: warm enumerated per-kernel == the # configs a COLD real run compiles
    (proves warming offers the autotuner the exact same choice set).
  * full cross-process HIT: warm the disk cache in one proc, then a real fwd+bwd in a SEPARATE
    fresh proc compiles the autotuned kernels 0 NEW times (mirrors 03_diagnose_hit.py).

CUDA required; CPU is skipped.
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest
import torch

from fla_rola.precompile import RLA_KERNELS, enumerate_shapes

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="precompile needs CUDA")

_KERNELS = list(dict.fromkeys(RLA_KERNELS))
_SPEC = {"path": "rla", "dqk": 16, "dv": 16, "nc": 8}
# repo root = parent of the fla_rola package dir (robust regardless of pytest cwd)
import fla_rola  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(fla_rola.__file__)))
_ENV = dict(os.environ, PYTHONPATH=_REPO_ROOT, ROLA_NO_COMPILE="1",
            ROLA_SPEC=json.dumps(_SPEC))


def _grp_counts(cache_dir):
    """Per-kernel count of compiled configs = # of __grp__<name>.json manifests on disk."""
    counts = {}
    for root, _dirs, files in os.walk(cache_dir):
        for f in files:
            if f.startswith("__grp__") and f.endswith(".json"):
                name = f[len("__grp__"):-len(".json")]
                counts[name] = counts.get(name, 0) + 1
    return counts


def _run_subproc(code, **env_extra):
    out = subprocess.run([sys.executable, "-c", code], env=dict(_ENV, **env_extra),
                         text=True, capture_output=True)
    assert out.returncode == 0, f"subprocess failed:\n{out.stderr[-3000:]}"
    for ln in out.stdout.splitlines():
        if ln.startswith("__JSON__"):
            return json.loads(ln[len("__JSON__"):])
    raise AssertionError(f"no __JSON__ result:\n{out.stdout[-2000:]}")


# The spec is passed to the subprocess via the ROLA_SPEC env var (JSON) so these code
# strings carry no format placeholders (keeps them lint-clean and unambiguous).
_REAL_RUN = '''
import os, json
from fla_rola.precompile import _make_inputs, ShapeSpec
from fla_rola.ops.rola.chunk import chunk_rola_routed
import torch
s = ShapeSpec.coerce(json.loads(os.environ["ROLA_SPEC"]))
ins = _make_inputs(s)
kw = {"Wg": ins["Wg"]} if "Wg" in ins else {}
chunk_rola_routed(ins["q"], ins["k"], ins["v"], ins["h"], ins["Wr"], ins["Ww"],
                  ins["D"], ins["b"], norm="raw", **kw).sum().backward()
torch.cuda.synchronize()
print("__JSON__" + json.dumps({"ok": True}))
'''

_WARM = '''
import os, json, tempfile
from fla_rola.precompile import enumerate_shapes, parallel_warm
cap = tempfile.mkdtemp(prefix="cap_")
os.environ["TRITON_CACHE_DIR"] = cap
target = os.environ["TARGET_CACHE"]
spec = json.loads(os.environ["ROLA_SPEC"])
work, _ = enumerate_shapes([spec], cache_dir=target)
os.environ["TRITON_CACHE_DIR"] = target
s = parallel_warm(work, cache_dir=target, n_workers=8, progress=False)
print("__JSON__" + json.dumps({"n_warmed": s["n_warmed"], "n_failed": s["n_failed"], "failures": s["failures"][:3]}))
'''


def test_enumerate_produces_picklable_workitems():
    cap = tempfile.mkdtemp(prefix="cap_")
    try:
        os.environ["TRITON_CACHE_DIR"] = cap
        work, summary = enumerate_shapes([_SPEC], cache_dir=cap)
    finally:
        os.environ.pop("TRITON_CACHE_DIR", None)
        import shutil
        shutil.rmtree(cap, ignore_errors=True)
    assert summary["n_shapes"] == 1
    fired = {w.kernel for w in work}
    for kn in ("_rola_routed_fwd_intra", "_rola_routed_fwd_inter"):
        assert kn in fired, f"{kn} not captured"
    import pickle
    pickle.loads(pickle.dumps(work))  # ProcessPool requires picklability


def test_warm_config_set_equals_cold_bench_set():
    """warm enumerated configs/kernel == the configs a COLD real run actually compiles."""
    cold_dir = tempfile.mkdtemp(prefix="cold_")
    try:
        _run_subproc(_REAL_RUN, TRITON_CACHE_DIR=cold_dir)
        cold = _grp_counts(cold_dir)
    finally:
        import shutil
        shutil.rmtree(cold_dir, ignore_errors=True)

    cap = tempfile.mkdtemp(prefix="cap_")
    try:
        os.environ["TRITON_CACHE_DIR"] = cap
        work, _ = enumerate_shapes([_SPEC], cache_dir=cap)
    finally:
        os.environ.pop("TRITON_CACHE_DIR", None)
        import shutil
        shutil.rmtree(cap, ignore_errors=True)
    from collections import Counter
    warm = Counter(w.kernel for w in work)

    for kn in _KERNELS:
        assert cold.get(kn, 0) == warm.get(kn, 0), (
            f"{kn}: cold compiled {cold.get(kn, 0)} configs, warm enumerated {warm.get(kn, 0)}")


def test_parallel_warm_gives_full_cross_process_hit():
    """Warm the disk cache (proc A), then a real fwd+bwd in a FRESH proc B compiles the
    autotuned kernels 0 NEW times — i.e. each warmed config is a cross-process disk HIT."""
    target = tempfile.mkdtemp(prefix="warm_")
    try:
        ws = _run_subproc(_WARM, TARGET_CACHE=target)
        assert ws["n_failed"] == 0, ws["failures"]
        assert ws["n_warmed"] > 0
        before = _grp_counts(target)
        # proc B: fresh interpreter (cold in-mem JIT), reads the warm DISK cache
        _run_subproc(_REAL_RUN, TRITON_CACHE_DIR=target)
        after = _grp_counts(target)
    finally:
        import shutil
        shutil.rmtree(target, ignore_errors=True)

    new = {kn: after.get(kn, 0) - before.get(kn, 0) for kn in _KERNELS}
    recompiled = {kn: n for kn, n in new.items() if n > 0}
    assert not recompiled, f"these autotuned kernels recompiled (cache MISS): {recompiled}"
