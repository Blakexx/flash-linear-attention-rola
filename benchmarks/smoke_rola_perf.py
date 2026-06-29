#!/usr/bin/env python3
"""Quick perf smoke for the production tree-routed RoLA op (chunk_rola_routed) — a fast LOCAL signal,
NOT a benchmark and NOT a correctness check (use tests/ops/test_rola.py for correctness).

Times fwd and fwd+bwd (CUDA events, warmup, median) for RLA/GLA × {global, kappa} at a representative
shape, in the training dtype (bf16). Run on an idle GPU:  python -u benchmarks/smoke_rola_perf.py
"""
import sys

import torch

from fla_rola.ops.rola import chunk_rola_routed

DEV = "cuda"


def _median_ms(fn, leaves, bwd, reps=20, warmup=3):
    for _ in range(warmup):
        o = fn()
        if bwd:
            o.float().sum().backward()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        for x in leaves:
            if x.grad is not None:
                x.grad = None
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        o = fn()
        if bwd:
            o.float().sum().backward()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    ts.sort()
    return ts[len(ts) // 2]


def main():
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        return 0
    B, L, H, K, nc, dv = 4, 1024, 4, 16, 64, 16
    print(f"device: {torch.cuda.get_device_name(0)} | B={B} L={L} H={H} dqk=dv={dv} nc={nc} bf16 | median ms / 20")
    g = torch.Generator(device=DEV).manual_seed(0)

    def rb(*s):
        return torch.randn(*s, generator=g, device=DEV, dtype=torch.bfloat16)

    D, b = 1, nc                       # flat routing (D=1, b=nc) — the in-kernel router takes h + Wr/Ww
    dm = H * dv                        # a representative hidden width

    def routed_kwargs(h, Wr, Ww, Wg=None):
        wl = torch.einsum('bthm,hdmc->bthdc', h, Ww.to(h.dtype)).contiguous()
        rl = torch.einsum('bthm,hdmc->bthdc', h, Wr.to(h.dtype)).contiguous()
        alpha = torch.sigmoid(torch.einsum('bthm,hm->bth', h.float(), Wg.float())) if Wg is not None else None
        return dict(wl=wl, rl=rl, alpha=alpha)

    for gla in (False, True):
        q = (torch.nn.functional.elu(rb(B, L, H, K)) + 1).requires_grad_()
        k = (torch.nn.functional.elu(rb(B, L, H, K)) + 1).requires_grad_()
        v = rb(B, L, H, dv).requires_grad_()
        h = rb(B, L, H, dm).requires_grad_()
        Wr = (rb(H, D, dm, b) * 0.4).requires_grad_()
        Ww = (rb(H, D, dm, b) * 0.4).requires_grad_()
        Wg = (rb(H, dm).float() * 0.4).requires_grad_() if gla else None
        kap = torch.full((B, L, H, 1), 0.5, device=DEV, dtype=torch.bfloat16)
        leaves = [x for x in (q, k, v, h, Wr, Ww, Wg) if x is not None]
        for norm in ("global", "kappa"):
            def fn(norm=norm):
                return chunk_rola_routed(q, k, v, h, Wr, Ww, D, b, norm=norm,
                                         kappa=kap if norm == "kappa" else None, scale=1.0, Wg=Wg,
                                         **routed_kwargs(h, Wr, Ww, Wg=Wg))
            fwd = _median_ms(fn, leaves, bwd=False)
            fb = _median_ms(fn, leaves, bwd=True)
            print(f"  {'GLA' if gla else 'RLA'}/{norm:7s}  fwd={fwd:.2f}ms  fwd+bwd={fb:.2f}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
