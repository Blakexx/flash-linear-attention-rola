#!/usr/bin/env python3
"""INTRA-kernel correctness: a kernel must resolve the SAME result on every execution path.

This is the half of the test strategy that needs NO oracle — it compares a kernel against different
forms of ITSELF by FORCING different autotune configs and asserting they agree. It's the only thing
that catches a bug living in one specific compiled branch/config (the autotuner picks just one per
shape, and which one is device-dependent — so a buggy non-picked config is otherwise invisible until
a different GPU selects it).

Here the axis under test is the value tile BV: at d_v=64 both BV=16 (ND_V=4) and BV=32 (ND_V=2) fit,
so we force each and assert the full fwd + bwd agree. (Locally this is bf16; the operands are the same
values grouped differently with fp32 accumulation, so agreement is ~1e-4, far under the bf16 noise
floor — a real branch bug would show as a gross mismatch. The un-tiled ND_V==1 path only fits in fp32
on a 163KB card (A100); there the same harness gives the tight ~1e-6 tiled-vs-untiled check.)

The complementary INTER-kernel test (chunk==recurrent==naive oracle) lives in
test_recurrent_vs_chunked.py / test_rola_routed_tiling.py — that validates the math; this validates
that every path computes that math identically.

Run:  PYTHONPATH=<CLA>:. python -u tests/test_kernel_self_consistency.py
"""
import sys
import torch
import triton

import fla_rola.ops.rola.chunk as C
# Shrink the chunk BT (the dominant per-program smem driver — every tile is [BT,*]) so the value-tiled
# fp32 configs fit a 99KB card and the gold ND_V=2-vs-ND_V=4 invariance runs in fp32 LOCALLY. BT is a
# correctness-preserving chunking granularity (and <= the GLA decay-floor cap), so this changes only
# smem, not math. (Reducing B/H/L/nc would NOT help — those are grid/loop dims, not per-program smem.)
C._CHUNK = 16
C._CHUNK_FWD = 16
from fla_rola.ops.rola import rola_rla_triton, rola_gla_triton

DEV = "cuda"
_VALUE_TILED = (C._scan_S, C._scan_dS, C._rola_fwd_inter, C._rola_gla_fwd_inter,
                C._par_grad_rla_qr, C._par_grad_rla_kwv, C._par_grad_gla_qr, C._par_grad_gla_kwv)
_HAS_BD = (C._par_grad_rla_qr, C._par_grad_rla_kwv, C._par_grad_gla_qr, C._par_grad_gla_kwv)


def force_bv(bv, warps, bd=16, stages=1):
    """Pin every value-tiled kernel to one (BV, num_warps) config (grad kernels also need BD)."""
    for k in _VALUE_TILED:
        kw = {'BD': bd, 'BV': bv} if k in _HAS_BD else {'BV': bv}
        k.configs = [triton.Config(dict(kw), num_warps=warps, num_stages=stages)]
        try:
            k.cache.clear()
        except Exception:
            pass


def fold(t):
    B, L, H, D = t.shape
    return t.permute(0, 2, 1, 3).reshape(B * H, L, D).contiguous()


def run(gla, bv, warps, dv, B=2, H=2, L=128, K=16, nc=16, seed=0, dt=torch.float32):
    """Forced-config forward+backward; returns (out, [grads]) as fp64."""
    force_bv(bv, warps)
    g = torch.Generator(device=DEV).manual_seed(seed)
    rf = lambda *s: torch.randn(*s, generator=g, device=DEV, dtype=torch.float64)
    q = torch.nn.functional.elu(rf(B, L, H, K)) + 1.0
    k = torch.nn.functional.elu(rf(B, L, H, K)) + 1.0
    v = rf(B, L, H, dv)
    r = torch.softmax(rf(B, L, H, nc), -1)
    w = torch.softmax(rf(B, L, H, nc), -1)
    ld = torch.log(torch.sigmoid(rf(B, L, H, nc))).clamp(min=-2.5)
    coef = rf(B, L, H, dv)
    nin = [q, k, v, r, w, ld] if gla else [q, k, v, r, w]
    kin = [fold(x.to(dt)).clone().requires_grad_() for x in nin]
    out = (rola_gla_triton(*kin) if gla else rola_rla_triton(*kin))
    gk = torch.autograd.grad((out.float() * fold(coef.to(dt)).float()).sum(), kin)
    return out.double(), [x.double() for x in gk]


def main():
    # INTRA gate: forward branches equal AND backward branches equal. At BT=16 the value-tiled fp32
    # configs fit, so we compare the actual tiling branches — ND_V=4 (BV=16) vs ND_V=2 (BV=32) — at
    # d_v=64 in fp32. Same math, different value-block grouping (fp32 accumulation) → must agree to
    # ~fp32 (1e-5), far under any tolerance. (The un-tiled ND_V=1 path at d_v=64 is intrinsically too
    # wide for fp32 on 99KB regardless of BT; its branch is covered at d_v=16 by the inter gates.)
    fails = 0
    dv = 64
    print(f"{torch.cuda.get_device_name(0)} | INTRA gate (BT=16, fp32) | d_v={dv}: ND_V=4 (BV16) vs ND_V=2 (BV32)\n")
    for gla in (False, True):
        tag = "GLA" if gla else "RLA"
        try:
            o_a, g_a = run(gla, 16, 2, dv)   # ND_V=4
            o_b, g_b = run(gla, 32, 2, dv)   # ND_V=2
        except Exception as e:
            print(f"  FAIL {tag}: {type(e).__name__}: {str(e).splitlines()[0][:55]}")
            fails += 1
            continue
        rel = lambda a, b: ((a - b).norm() / (b.norm() + 1e-12)).item()
        names = (["q", "k", "v", "r", "w", "ld"] if gla else ["q", "k", "v", "r", "w"])
        fwd = rel(o_a, o_b)
        per = {n: rel(a, b) for n, a, b in zip(names, g_a, g_b)}
        bwd = max(per.values())
        ok = max(fwd, bwd) < 1e-3
        fails += not ok
        print(f"  {'OK  ' if ok else 'FAIL'} {tag}: forward-eq={fwd:.1e}  backward-eq={bwd:.1e} "
              f"({' '.join(f'{n}{per[n]:.0e}' for n in names)})")
    print(f"\n{'PASS' if not fails else 'FAIL'}: forward branches equal AND backward branches equal (config-invariance)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
