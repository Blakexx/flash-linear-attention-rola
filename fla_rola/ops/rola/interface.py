# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# First-class routed RoLA operator.
#
# `chunk_rola` is the ONE norm-aware entry point: it owns the whole recipe — the per-state
# denominator pre-pass, the read-gate rescale, the (numerator-only) shared-gram readout, and the
# divide — so callers (the LM layer) just pass `norm=...`. There is no routed branch bolted onto
# simple_gla and no normalization logic in the model.
#
# Normalization is unified on the DENOMINATOR-SPLIT: the readout is always numerator-only
# (BV=next_pow2(dv), no ones-column → ~2× occupancy, fits small smem), and the global denominator is
# reconstructed as Σ_c r̃ᶜ·dᶜ from the per-state den pre-pass `d` (the same `d` that rescales the read
# gates for kappa/per_state). global/kappa/per_state differ ONLY in how the read gates are rescaled:
#   global    : r̃ = r                      (den = Σ_c rᶜ·dᶜ — the partition function)
#   per_state : r̃ = r / (d+ε)              (≡ kappa=1, each state self-normalizes)
#   kappa     : r̃ = r · (d+ε)^(-κ(x))      (input-dependent interpolation global↔per_state)
#   raw       : numerator only, no divide   (GLA convention)
#
# Differentiable by COMPOSITION — the den pre-pass and the readout are verified autograd Functions
# and the rescale/divide are plain torch, so gradients flow automatically (no custom backward).
import torch
import triton

from fla_rola.ops.rola.chunk import (
    rola_gla_triton,
    rola_perstate_den_gla_triton,
    rola_perstate_den_triton,
    rola_rla_triton,
)

_NORMS = ('raw', 'global', 'per_state', 'kappa')


def _rola_chunk_core(q, k, v, w, r, ld, chunk_size):
    """Eager (CPU / capability-fallback) shared-gram routed readout on folded [BH, T, *] tensors.
    ld=None ⇒ RLA (chunk-parallel cumsum scan); ld given ⇒ scalar-gated GLA (decayed scan). Returns
    the un-normalized readout [BH, T, v.shape[-1]] (numerator; the den is a separate pre-pass)."""
    BH, T, K = q.shape
    V = v.shape[-1]
    nc = w.shape[-1]
    if ld is None:
        pad = (-T) % chunk_size
        if pad:
            q, k, v, w, r = [torch.nn.functional.pad(t, (0, 0, 0, pad)) for t in (q, k, v, w, r)]
        Tp = T + pad
        nch = Tp // chunk_size
        C = chunk_size
        qc, kc, vc = q.view(BH, nch, C, K), k.view(BH, nch, C, K), v.view(BH, nch, C, V)
        rc, wc = r.view(BH, nch, C, nc), w.view(BH, nch, C, nc)
        G = torch.einsum('bnid,bnjd->bnij', qc, kc)
        R = torch.einsum('bnic,bnjc->bnij', rc, wc)
        causal = torch.tril(torch.ones(C, C, device=q.device, dtype=q.dtype))
        A = G * R * causal
        o_intra = torch.einsum('bnij,bnjv->bniv', A, vc)
        KV = torch.einsum('bnjc,bnjd,bnjv->bncdv', wc, kc, vc)
        S_before = torch.cumsum(KV, dim=1) - KV
        M = torch.einsum('bnic,bncdv->bnidv', rc, S_before)
        o_inter = torch.einsum('bnid,bnidv->bniv', qc, M)
        return (o_intra + o_inter).reshape(BH, Tp, V)[:, :T]
    # GLA: per-state scalar decay, sequential decayed scan over chunks.
    S = q.new_zeros(BH, nc, K, V)
    outs = []
    for c0 in range(0, T, chunk_size):
        c1 = min(c0 + chunk_size, T)
        Cc = c1 - c0
        qcc, kcc, vcc = q[:, c0:c1], k[:, c0:c1], v[:, c0:c1]
        rcc, wcc, ldc = r[:, c0:c1], w[:, c0:c1], ld[:, c0:c1]
        a = torch.cumsum(ldc, dim=1)
        rt, wt = rcc * torch.exp(a), wcc * torch.exp(-a)
        G = torch.einsum('bid,bjd->bij', qcc, kcc)
        D = torch.einsum('bic,bjc->bij', rt, wt)
        causal = torch.tril(torch.ones(Cc, Cc, device=q.device, dtype=q.dtype))
        o_intra = torch.einsum('bij,bjv->biv', G * D * causal, vcc)
        M = torch.einsum('bic,bcdv->bidv', rt, S)
        o_inter = torch.einsum('bid,bidv->biv', qcc, M)
        outs.append(o_intra + o_inter)
        Lam = a[:, -1, :]
        w_end = wcc * torch.exp(Lam[:, None, :] - a)
        KV = torch.einsum('bjc,bjd,bjv->bcdv', w_end, kcc, vcc)
        S = torch.exp(Lam)[:, :, None, None] * S + KV
    return torch.cat(outs, dim=1)


def _perstate_den_torch(q, k, w, ld, chunk_size, eps=1e-5):
    """Eager per-state denominator dᵢᶜ = Σ_{j≤i} (φqᵢ·φkⱼ) wⱼᶜ [· e^{Λᵢᶜ−Λⱼᶜ} under decay], folded
    [BH,T,*] → [BH,T,nc]. CPU/capability fallback for the Triton den kernels."""
    BH, T, K = q.shape
    G = torch.einsum('bid,bjd->bij', q, k)
    causal = torch.tril(torch.ones(T, T, device=q.device, dtype=q.dtype))
    if ld is None:
        return torch.einsum('bij,bjc->bic', G * causal, w)
    A = torch.cumsum(ld, dim=1)                          # [BH,T,nc] cumulative log-decay
    s = torch.einsum('bij,bjc->bic', G * causal, w * torch.exp(-A))
    return torch.exp(A) * s


def _readout(qf, kf, vf, rf, wf, gf, chunk_size):
    """Folded routed readout (numerator-only). CUDA → device-agnostic Triton kernels; else → eager
    core. The global denominator is the caller's separate per-state den pre-pass."""
    if qf.is_cuda:
        if gf is None:
            return rola_rla_triton(qf, kf, vf, rf, wf, chunk=chunk_size)
        return rola_gla_triton(qf, kf, vf, rf, wf, gf)
    return _rola_chunk_core(qf, kf, vf, wf, rf, gf, chunk_size)


def chunk_rola(q, k, v, r, w, g=None, norm='kappa', kappa=None, scale=None, eps=1e-5):
    """Routed RoLA (shared-gram) readout with built-in normalization.

    Args:
        q, k:  φ-mapped queries/keys [B, T, H, K] (the feature map φ stays in the caller — the
               operator is φ-agnostic, seeing only the content gram G=φ(q)φ(k)ᵀ).
        v:     values [B, T, H, V].
        r, w:  read / write routing gates [B, T, H, nc].
        g:     per-state log-decay [B, T, H, nc] for the scalar-gated (GLA) variant, or None (RLA).
        norm:  'raw' | 'global' | 'per_state' | 'kappa'.
        kappa: per-token exponent [B, T, H, 1] (required for norm='kappa').
        scale: query scale (default 1/sqrt(K)).
    Returns:
        Normalized readout [B, T, H, V] ('raw' returns the un-normalized numerator).
    """
    if norm not in _NORMS:
        raise ValueError(f"norm must be one of {_NORMS}, got {norm!r}")
    if norm == 'kappa' and kappa is None:
        raise ValueError("norm='kappa' requires a per-token `kappa` exponent tensor [B,T,H,1]")
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    chunk_size = min(64, max(16, triton.next_power_of_2(T)))

    def fold(t):
        return t.permute(0, 2, 1, 3).reshape(B * H, T, t.shape[-1])

    def unfold(t):
        return t.view(B, H, T, -1).permute(0, 2, 1, 3).contiguous()

    qf, kf, vf, rf, wf = fold(q) * scale, fold(k), fold(v), fold(r), fold(w)
    gf = fold(g) if g is not None else None

    if norm == 'raw':
        return unfold(_readout(qf, kf, vf, rf, wf, gf, chunk_size)).to(v.dtype)

    # global / per_state / kappa: per-state den pre-pass → rescale read gates → numerator-only
    # readout → divide by the reconstructed global den Σ_c r̃ᶜ·dᶜ.
    if qf.is_cuda:
        d = (rola_perstate_den_gla_triton(qf, kf, wf, gf) if gf is not None
             else rola_perstate_den_triton(qf, kf, wf))
    else:
        d = _perstate_den_torch(qf, kf, wf, gf, chunk_size, eps)
    # Rescale read gates, then cast BACK to the gate dtype: `d` is fp32, so the rescale would upcast
    # r̃ to fp32 and break the kernel's same-dtype requirement (tl.dot(r̃, wᵀ) with w in bf16).
    gate_dtype = rf.dtype
    if norm == 'kappa':
        rf = (rf * (d + eps).pow(-fold(kappa))).to(gate_dtype)
    elif norm == 'per_state':
        rf = (rf / (d + eps)).to(gate_dtype)
    # norm == 'global': r̃ = r (unchanged)
    num = _readout(qf, kf, vf, rf, wf, gf, chunk_size)
    den = (rf * d).sum(-1, keepdim=True)
    return unfold(num / (den + eps)).to(v.dtype)
