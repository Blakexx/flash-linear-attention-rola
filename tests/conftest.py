import inspect
import os
from unittest.mock import patch

import pytest
import torch

try:
    from torch.compiler import is_compiling
except ImportError:
    def is_compiling():
        return False

from fla_rola.utils import device_torch_lib

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

_ORIGINAL_EMPTY = torch.empty
_ORIGINAL_EMPTY_LIKE = torch.empty_like
_ORIGINAL_NEW_EMPTY = torch.Tensor.new_empty


def _is_called_from_fla():
    """Check if the call is from fla_rola package."""
    frame = inspect.currentframe()
    try:
        # Skip the current frame and go up the call stack
        while frame:
            frame = frame.f_back
            if frame is None:
                break

            if hasattr(frame, 'f_code') and hasattr(frame.f_code, 'co_filename'):
                filename = frame.f_code.co_filename
                # Skip conftest.py frames (where the guarded functions are defined)
                if 'conftest.py' in filename:
                    continue
                # Check if this frame is from a test file
                # Look for 'tests/' or 'test_' in the file path
                if 'tests/' in filename or 'test_' in filename:
                    return False

            module = inspect.getmodule(frame)
            if module and hasattr(module, '__name__'):
                # If call is from fla_rola package, apply guard
                if 'fla_rola' in module.__name__:
                    return True
    finally:
        del frame
    # Default to not guarding if we can't determine
    return False


def _poison(result):
    """Fill a scratch tensor with NaN. Skip requires_grad leaves: inductor's lazy init (e.g. pad_mm)
    allocates grad-leaf scratch via torch.empty, and an in-place fill_ on a leaf-that-requires-grad
    raises — so with torch.compile defaulted on, poisoning those would break compilation, not catch a
    real uninitialized-read bug. (Those tensors are inductor's, not fla buffers.)"""
    if result.requires_grad:
        return result
    if result.is_floating_point():
        result.fill_(float('nan'))
    elif result.is_complex():
        result.fill_(complex(float('nan'), float('nan')))
    return result


def _guarded_empty(*args, **kwargs):
    """Create a tensor filled with NaN instead of uninitialized values."""
    dtype = kwargs.get('dtype') or torch.get_default_dtype()

    if not (dtype.is_floating_point or dtype.is_complex):
        return _ORIGINAL_EMPTY(*args, **kwargs)

    if is_compiling() or not _is_called_from_fla():
        return _ORIGINAL_EMPTY(*args, **kwargs)

    return _poison(_ORIGINAL_EMPTY(*args, **kwargs))


def _guarded_empty_like(input, **kwargs):
    """Create a tensor filled with NaN instead of uninitialized values."""
    if is_compiling() or not _is_called_from_fla():
        return _ORIGINAL_EMPTY_LIKE(input, **kwargs)

    if kwargs.get('dtype') is None:
        kwargs['dtype'] = input.dtype

    dtype = kwargs['dtype']
    if not (dtype.is_floating_point or dtype.is_complex):
        return _ORIGINAL_EMPTY_LIKE(input, **kwargs)

    return _poison(_ORIGINAL_EMPTY_LIKE(input, **kwargs))


def _guarded_new_empty(self, *args, **kwargs):
    """Create a tensor filled with NaN instead of uninitialized values."""
    if is_compiling() or not _is_called_from_fla():
        return _ORIGINAL_NEW_EMPTY(self, *args, **kwargs)

    if kwargs.get('dtype') is None:
        kwargs['dtype'] = self.dtype

    dtype = kwargs['dtype']
    if not (dtype.is_floating_point or dtype.is_complex):
        return _ORIGINAL_NEW_EMPTY(self, *args, **kwargs)

    return _poison(_ORIGINAL_NEW_EMPTY(self, *args, **kwargs))


# -----------------------------------------------------------------------------
# Session-scoped Triton cache warm (RoLA autotuned kernels)
# -----------------------------------------------------------------------------
# Opt-in: set ROLA_WARM_SESSION=1 to parallel-precompile the registered RoLA shapes at
# session start, so the per-test first fwd+bwd is a cache HIT instead of a multi-minute
# cold autotune-codegen. A test module can extend the set with
# ``fla_rola.precompile.register_test_shapes([...])`` at import time. The warm is a no-op
# without CUDA. See fla_rola/precompile.py for the mechanism + caveats.
@pytest.fixture(scope="session", autouse=True)
def warm_rola_cache():
    if os.environ.get("ROLA_WARM_SESSION", "0") != "1" or not torch.cuda.is_available():
        yield
        return
    from fla_rola.precompile import registered_test_shapes, warm_gate
    shapes = registered_test_shapes()
    summary = warm_gate(shapes, progress=True)
    print(f"\n[warm_rola_cache] warmed {summary['n_warmed']} configs across "
          f"{summary['n_shapes']} shapes in {summary['total_wall_s']:.1f}s "
          f"({summary['n_workers']} workers, {summary['n_failed']} failed)")
    yield


@pytest.fixture(scope="function", autouse=True)
def poison_torch_memory(request):
    # Only apply the guard to ops and modules tests
    path = str(request.node.fspath)
    if 'tests/ops/' not in path and 'tests/modules/' not in path:
        yield
        return

    with patch('torch.empty', new=_guarded_empty), \
            patch('torch.empty_like', new=_guarded_empty_like), \
            patch('torch.Tensor.new_empty', new=_guarded_new_empty):
        yield
        if hasattr(device_torch_lib, 'synchronize'):
            device_torch_lib.synchronize()
