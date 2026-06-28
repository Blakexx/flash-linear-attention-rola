# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""Naive O(L^2) reference implementations for the RoLA kernel gates — the FLA `ops/<op>/naive.py`
convention (cf. `ops/gla/naive.py`, `ops/kda/naive.py`). Ground-truth O(L^2) parallel readout
(global-norm + scalar-GLA) and per-state denominator (RLA + GLA), used by tests to validate
the routed RoLA kernels. Lifted verbatim from the original CLA `rola.py` reference before it was reduced.
"""
import torch
import torch.nn.functional as F


def _fold(t):
    """[B,L,H,D] -> [B*H, L, D] (head folded into batch)."""
    return t.permute(0, 2, 1, 3).reshape(t.shape[0] * t.shape[2], t.shape[1], t.shape[-1])


def naive_rola_global(q, k, v, wg, rg, eps=1e-5):
    """O(L^2) global-norm reference. q,k:[B,L,H,dqk] v:[B,L,H,dv] wg,rg:[B,L,H,C].
    Ground truth for the first-use correctness gate."""
    L = q.shape[1]
    G = torch.einsum('bthd,bshd->bhts', q, k)            # content gram (shared)
    R = torch.einsum('bthc,bshc->bhts', rg, wg)          # routing gram
    causal = torch.tril(torch.ones(L, L, device=q.device, dtype=q.dtype))
    W = G * R * causal
    num = torch.einsum('bhts,bshv->bthv', W, v)        # [B,T,H,dv]
    den = torch.einsum('bhts->bth', W).unsqueeze(-1)   # [B,T,H,1]
    return num / (den + eps)


def naive_rola_gla(q, k, v, wg, rg, ld, eps=1e-5, normalized=False):
    """O(L^2) ref for scalar-gated RoLA-GLA (first-use gate truth). q,k:[B,L,H,dqk]
    v:[B,L,H,dv] wg,rg,ld:[B,L,H,C]. normalized=False = raw gated sum (GLA convention)."""
    B, L, H, dqk = q.shape
    q, k, v, wg, rg, ld = _fold(q), _fold(k), _fold(v), _fold(wg), _fold(rg), _fold(ld)
    A = torch.cumsum(ld, dim=1)                                   # [bh,L,C]
    G = torch.einsum('btd,bsd->bts', q, k)
    # clamp exponent ≤0: a no-op on the causal triangle (t≥s ⟹ A_t≤A_s, exact) and a guard on
    # the upper triangle (masked below) so deep decay can't overflow exp before the mask applies.
    decay = torch.exp((A[:, :, None, :] - A[:, None, :, :]).clamp(max=0.0))   # [bh,L,L,C]
    D = torch.einsum('btc,bsc,btsc->bts', rg, wg, decay)
    causal = torch.tril(torch.ones(L, L, device=q.device, dtype=q.dtype))
    W = G * D * causal
    v1 = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1) if normalized else v
    O = torch.einsum('bts,bsv->btv', W, v1)
    out = O[..., :-1] / (O[..., -1:] + eps) if normalized else O
    dv = v.shape[-1]
    return out.view(B, H, L, dv).permute(0, 2, 1, 3).contiguous()


def naive_rola_perstate_den(qf, kf, wg, chunk=64):
    """Per-state denominator d_i^c = sum_{j<=i} w_j^c (φ(q_i)·φ(k_j)) — the mass each state
    contributes to token i's global partition function. [B,L,H,*] in → [B,L,H,nc] out.
    Mirrors _rola_chunked_parallel restricted to the ones-column, with the state axis kept.
    Used by the KAPPA normalization: r̃ = r·(d+ε)^{-κ(x)} interpolates global (κ=0, exact)
    ↔ per-state (κ=1, exact — read gates sum to 1 so the outer divide collapses)."""
    B, L, H, dqk = qf.shape
    nc = wg.shape[-1]
    q, k, w = _fold(qf), _fold(kf), _fold(wg)
    pad = (-L) % chunk
    if pad:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        w = F.pad(w, (0, 0, 0, pad))
    Lp = L + pad
    n = Lp // chunk
    qc = q.view(B * H, n, chunk, dqk)
    kc = k.view(B * H, n, chunk, dqk)
    wc = w.view(B * H, n, chunk, nc)
    G = torch.einsum('bnid,bnjd->bnij', qc, kc)
    causal = torch.tril(torch.ones(chunk, chunk, device=q.device, dtype=q.dtype))
    intra = torch.einsum('bnij,bnjc->bnic', G * causal, wc)
    KZ = torch.einsum('bnjc,bnjd->bncd', wc, kc)                  # per-chunk z increments
    Z = torch.cumsum(KZ, dim=1) - KZ                              # exclusive prefix
    inter = torch.einsum('bnid,bncd->bnic', qc, Z)
    d = (intra + inter).reshape(B * H, Lp, nc)[:, :L]
    return d.view(B, H, L, nc).permute(0, 2, 1, 3)                # [B,L,H,nc]


def naive_rola_gla_perstate_den(qf, kf, wg, ld, chunk=64):
    """Per-state denominator UNDER per-state log-decay ld:
    d_i^c = sum_{j<=i} (φq_i·φk_j) w_j^c e^{Λ_ic-Λ_jc}, Λ = inclusive cumsum(ld).
    [B,L,H,*] in → [B,L,H,nc] out. Chunked (decay absorbed chunk-locally, decayed
    cross-chunk carry) — the torch fallback/oracle for the fork's GLA den kernel."""
    B, L, H, dqk = qf.shape
    nc = wg.shape[-1]
    q, k, w, g = _fold(qf), _fold(kf), _fold(wg), _fold(ld)
    pad = (-L) % chunk
    if pad:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        w = F.pad(w, (0, 0, 0, pad))
        g = F.pad(g, (0, 0, 0, pad))
    Lp = L + pad
    n = Lp // chunk
    qc = q.view(B * H, n, chunk, dqk)
    kc = k.view(B * H, n, chunk, dqk)
    wc = w.view(B * H, n, chunk, nc)
    gc = g.view(B * H, n, chunk, nc)
    a = gc.cumsum(2)                                              # chunk-local Λ [b,n,t,c]
    Lam = a[:, :, -1, :]                                          # chunk decay totals [b,n,c]
    G = torch.einsum('bnid,bnjd->bnij', qc, kc)
    causal = torch.tril(torch.ones(chunk, chunk, device=q.device, dtype=q.dtype))
    intra = torch.exp(a) * torch.einsum('bnij,bnjc->bnic', G * causal, wc * torch.exp(-a))
    w_end = wc * torch.exp(Lam.unsqueeze(2) - a)                  # writes decayed to chunk end
    KZ = torch.einsum('bnjc,bnjd->bncd', w_end, kc)               # per-chunk carry increments
    acc = torch.zeros(B * H, nc, dqk, device=q.device, dtype=q.dtype)
    Zs = []
    for i in range(n):                                            # decayed exclusive prefix
        Zs.append(acc)
        acc = torch.exp(Lam[:, i]).unsqueeze(-1) * acc + KZ[:, i]
    Z = torch.stack(Zs, 1)                                        # [b,n,c,d] state at chunk start
    inter = torch.exp(a) * torch.einsum('bnid,bncd->bnic', qc, Z)
    d = (intra + inter).reshape(B * H, Lp, nc)[:, :L]
    return d.view(B, H, L, nc).permute(0, 2, 1, 3)                # [B,L,H,nc]
