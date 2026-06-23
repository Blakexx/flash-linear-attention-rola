"""K-sweep harness for the RoLA snapshot-granularity stride (ROLA_SNAP_K).

Measures fwd+bwd step peak memory and step time at a fixed (RLA, dqk=16, dv=16, nc=256) shape
across K in {1,2,4,8,16}. Positive features / softmax gates / scalar kappa avoid the signed-den NaN.
Run once per K in its own process (env read at resolve-time; clean autotune state)."""
import os
import time

import torch

from fla_rola.ops.rola import chunk_rola

device = 'cuda'
B, H, dqk, dv, nc = 1, 1, 16, 16, 256
L = int(os.environ.get('SWEEP_L', '4096'))
EPS = 1e-5


def make_inputs(seed=0):
    g = torch.Generator(device=device).manual_seed(seed)

    def t(*s):
        return torch.randn(*s, device=device, generator=g, dtype=torch.float32)
    q = t(B, L, H, dqk).abs().requires_grad_()              # positive features
    k = t(B, L, H, dqk).abs().requires_grad_()
    v = t(B, L, H, dv).requires_grad_()
    r = torch.softmax(t(B, L, H, nc), -1).detach().requires_grad_()   # positive softmax gates
    w = torch.softmax(t(B, L, H, nc), -1).detach().requires_grad_()
    kap = torch.sigmoid(t(B, L, H, 1)).detach().requires_grad_()      # scalar kappa
    return q, k, v, r, w, kap


def step(q, k, v, r, w, kap):
    o = chunk_rola(q, k, v, r=r, w=w, g=None, norm='kappa', kappa=kap, scale=1.0)
    loss = o.float().pow(2).sum()
    loss.backward()


def main():
    K = os.environ['ROLA_SNAP_K']
    inp = make_inputs()
    # warmup (compile + autotune), then reset peak
    step(*inp)
    for x in inp:
        if x.grad is not None:
            x.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    iters = 5
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        for x in inp:
            if x.grad is not None:
                x.grad = None
        step(*inp)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters * 1e3   # ms
    peak = torch.cuda.max_memory_allocated() / (1 << 20)  # MiB
    # also dump grads for the equivalence gate
    step(*inp)
    gq = inp[0].grad.detach().clone()
    gk = inp[1].grad.detach().clone()
    gv = inp[2].grad.detach().clone()
    gr = inp[3].grad.detach().clone()
    gw = inp[4].grad.detach().clone()
    torch.save({'gq': gq, 'gk': gk, 'gv': gv, 'gr': gr, 'gw': gw},
               f'/tmp/snap_grads_K{K}_L{L}.pt')
    print(f'RESULT K={K} L={L} peak_MiB={peak:.1f} step_ms={dt:.2f}', flush=True)


if __name__ == '__main__':
    main()
