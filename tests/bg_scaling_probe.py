#!/usr/bin/env python3
"""LOCAL scaling probe (NOT a benchmark): does a larger BG (states per block) flatten RoLA's
nc-scaling? The content gram G=qkᵀ and routing gram R are rebuilt per state-block (grid NB=nc/BG),
so bigger BG → fewer blocks → less redundant gram work + fewer grid programs. Counter-pressure:
the [BK, BG*BV] state tile grows with BG → more registers/smem (the profiled binding constraint).
Net is unknown a priori → measure. fwd+bwd median ms on the local GPU (3080 Ti), bf16.
"""
import sys
import torch
from fla_rola.ops.rola.chunk import rola_rla_triton, rola_gla_triton

DEV = "cuda"


def med_ms(fn, leaves, bwd, reps=20, warmup=4):
    for _ in range(warmup):
        o = fn()
        if bwd:
            o.float().sum().backward()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        for x in leaves:
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
    B, L, H, K, dv = 4, 1024, 4, 16, 16
    gla = "--gla" in sys.argv
    print(f"{torch.cuda.get_device_name(0)} | B={B} L={L} H={H} dqk=dv={dv} bf16 | "
          f"{'GLA' if gla else 'RLA'} | fwd+bwd median ms / peak MiB")
    g = torch.Generator(device=DEV).manual_seed(0)
    rb = lambda *s: torch.randn(*s, generator=g, device=DEV, dtype=torch.bfloat16)
    print(f"{'nc':>5} | " + " | ".join(f"BG={bg:<3d}" for bg in (16, 32, 64)))
    for nc in (16, 64, 256):
        qf = (torch.nn.functional.elu(rb(B * H, L, K)) + 1).requires_grad_()
        kf = (torch.nn.functional.elu(rb(B * H, L, K)) + 1).requires_grad_()
        vf = rb(B * H, L, dv).requires_grad_()
        rf = torch.softmax(rb(B * H, L, nc).float(), -1).bfloat16().requires_grad_()
        wf = torch.softmax(rb(B * H, L, nc).float(), -1).bfloat16().requires_grad_()
        ldf = (torch.log(torch.sigmoid(rb(B * H, L, nc).float())).clamp(min=-2.5).bfloat16()
               .requires_grad_() if gla else None)
        leaves = [x for x in (qf, kf, vf, rf, wf, ldf) if x is not None]
        cells = []
        for bg in (16, 32, 64):
            if nc % bg != 0 and nc < bg:
                cells.append("   -   ")
                continue
            try:
                fn = (lambda bg=bg: rola_gla_triton(qf, kf, vf, rf, wf, ldf, BG=bg)) if gla else \
                     (lambda bg=bg: rola_rla_triton(qf, kf, vf, rf, wf, BG=bg))
                torch.cuda.reset_peak_memory_stats()
                ms = med_ms(fn, leaves, bwd=True)
                peak = torch.cuda.max_memory_allocated() / 2**20
                cells.append(f"{ms:5.2f}/{peak:4.0f}")
            except Exception as e:
                cells.append(f"ERR:{type(e).__name__}")
        print(f"{nc:>5} | " + " | ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
