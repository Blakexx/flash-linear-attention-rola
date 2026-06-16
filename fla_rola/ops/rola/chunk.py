# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# RoLA routing — Triton kernels (additive extension of simple_gla).
#
# Routed linear attention shares the content gram G=qkᵀ across `nc` states and modulates it by a
# routing gram R=Σ_c r_i^c w_j^c (+ optional per-state scalar decay). These kernels compute the
# *un-normalized* routed readout O = (G∘R∘causal) @ v — the FLA convention; the global denominator
# (ones-column) is the caller's job. Tiled over state-blocks (BG states/program) so only this block's
# slice of the Kronecker state lives in SRAM → scales to any nc. The content gram is formed ONCE per
# chunk (the FLOP win), never materializing the L×nc product nor replicating q/k.
#
# Ported verbatim from the verified `rola_kernels` reference (gradcheck + fp64-exact + matched vs the
# O(L²) ground truth). The kernels internally augment v with a ones-column for a denominator; this
# module returns only the first `dv` columns (the numerator) and, in the backward, pads the incoming
# grad with a zero den-column — so the math is exactly the verified kernel restricted to the real v.

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from fla_rola.utils import autotune_cache_kwargs, check_shared_mem

# SMEM fitting follows the canonical FLA idiom: the chunk size BT is a fixed per-device constant
# (a `check_shared_mem` gate — the one structural knob the autotuner can't own, since the host-side
# Sb/dSa allocations + launch grid are sized from it before any kernel compiles, and GLA needs BT<=32
# for its fp32 decay floor), while the INNER blocks (num_warps / num_stages / BD) are autotune knobs
# that Triton prunes via OutOfResources. The denominator-split (numerator-only readout, BV=next_pow2
# (dv)) halved the per-program footprint, so the ada-class BT (64 RLA-fwd / 32 GLA+bwd) now fits down
# to RTX 30xx (measured); sm75 (T4, 64KB) drops to BT=16.
_BIG_SMEM = check_shared_mem('ada')
_WARPS = (2, 4, 8)
_STAGES = (1, 2, 3)
_AT_CFGS = [triton.Config({}, num_warps=w, num_stages=s) for w in _WARPS for s in _STAGES]
_AT_KEY = ['dqk', 'dv', 'nc']
_CHUNK_FWD = 64 if _BIG_SMEM else 16     # RLA forward
_CHUNK = 32 if _BIG_SMEM else 16         # GLA forward + all backwards (GLA fp32 decay floor caps BT<=32)

# --- Backward feature tiling: BD as an autotune knob ----------------------------------------------
# The backward grad kernels load a [BD, BG*BV] slice of the Kronecker state per feature-block into
# SRAM. Rather than model that footprint with a byte budget (fragile — it misses Triton's tl.dot
# operand staging), we expose the feature tile BD itself as an autotune knob and let the autotuner
# EMPIRICALLY pick the largest that fits: Triton's autotuner compiles each config and scores any
# OutOfResources as inf, so over-SRAM tiles are dropped against the real hardware. BD=16 always fits
# and floors the set, so a config always survives. This is how FLA's own kernels handle SRAM limits —
# no magic constant, no headroom factor; an A100/H100 lands on a big BD + deep pipeline, a 3080 Ti on
# BD=16. BG is pinned at the dot minimum (tl.dot dims must be >=16). NB: BD here tiles the GRAD
# kernels only; the scans take their own (independent) feature block — Sb is indexed by absolute
# feature row, so the two blockings need not match.
_BWD_BK = (16, 32, 64, 128)               # feature-tile candidates; autotuner keeps the largest fitting
_BWD_CFGS = [triton.Config({'BD': bk}, num_warps=w, num_stages=s)
             for bk in _BWD_BK for w in _WARPS for s in _STAGES]


def _prune_bwd_bd(configs, named_args, **kwargs):
    """Exact (not heuristic) prune: drop feature tiles larger than the padded feature dim — pure
    waste, never a fit question. SRAM-too-big tiles are pruned empirically by the autotuner
    (OutOfResources -> inf). Keeps BD<=16 as the floor so the set is never empty."""
    try:
        cap = max(16, triton.next_power_of_2(named_args['dqk']))
    except Exception:
        return configs
    keep = [c for c in configs if c.kwargs['BD'] <= cap]
    return keep or [c for c in configs if c.kwargs['BD'] == 16] or configs


@triton.autotune(configs=_AT_CFGS, key=_AT_KEY, **autotune_cache_kwargs)
@triton.jit
def _rola_fwd_tiled(q_ptr, k_ptr, v_ptr, wg_ptr, rg_ptr, outa_ptr,
                    L, dqk, dv, nc,
                    sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c,
                    soa_b, soa_n, soa_l, soa_v,
                    BT: tl.constexpr, BD: tl.constexpr, BV: tl.constexpr,
                    BG: tl.constexpr, NCH: tl.constexpr):
    """One program per (batch, STATE-BLOCK of BG states). Emits the AUGMENTED partial output
    [.., dv|den] (no divide); the wrapper sums blocks (exact: num & den are linear over states)."""
    b = tl.program_id(0)
    sb = tl.program_id(1)
    dvp = dv + 1
    offs_t = tl.arange(0, BT)
    offs_d = tl.arange(0, BD)
    offs_v = tl.arange(0, BV)
    offs_c = sb * BG + tl.arange(0, BG)
    dmask = offs_d < dqk
    cmask = offs_c < nc
    Sflat = tl.zeros([BD, BG * BV], dtype=tl.float32)
    for t in range(NCH):
        rows = t * BT + offs_t
        rmask = rows < L
        qc = tl.load(q_ptr + b * sq_b + rows[:, None] * sq_l + offs_d[None, :] * sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        kc = tl.load(k_ptr + b * sq_b + rows[:, None] * sq_l + offs_d[None, :] * sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        vc = tl.load(v_ptr + b * sv_b + rows[:, None] * sv_l + offs_v[None, :] * sv_d,
                     mask=rmask[:, None] & (offs_v[None, :] < dv), other=0.0)
        vc += tl.where((offs_v[None, :] == dv) & rmask[:, None], 1.0, 0.0)
        rgc = tl.load(rg_ptr + b * sg_b + rows[:, None] * sg_l + offs_c[None, :] * sg_c,
                      mask=rmask[:, None] & cmask[None, :], other=0.0)
        wgc = tl.load(wg_ptr + b * sg_b + rows[:, None] * sg_l + offs_c[None, :] * sg_c,
                      mask=rmask[:, None] & cmask[None, :], other=0.0)
        G = tl.dot(qc, tl.trans(kc))
        R = tl.dot(rgc, tl.trans(wgc))
        causal = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
        A = G * R * causal
        o_intra = tl.dot(A.to(vc.dtype), vc)
        P = tl.dot(qc, Sflat.to(qc.dtype))
        P3 = tl.reshape(P, [BT, BG, BV])
        o_inter = tl.sum(P3 * rgc[:, :, None], axis=1)
        o = o_intra + o_inter
        tl.store(outa_ptr + b * soa_b + sb * soa_n + rows[:, None] * soa_l + offs_v[None, :] * soa_v,
                 o, mask=rmask[:, None] & (offs_v[None, :] < dvp))
        WV = tl.reshape(wgc[:, :, None] * vc[:, None, :], [BT, BG * BV])
        Sflat += tl.dot(tl.trans(kc), WV.to(kc.dtype))


def _fwd_aug(q, k, v, wg, rg, chunk, BG):
    """Triton forward, AUGMENTED [B,L,dv+1] (numerator | denominator), summed over state-blocks."""
    B, L, dqk = q.shape
    dv = v.shape[-1]
    nc = wg.shape[-1]
    BD = max(16, triton.next_power_of_2(dqk))
    BV = max(16, triton.next_power_of_2(dv + 1))
    NB = triton.cdiv(nc, BG)
    NCH = triton.cdiv(L, chunk)
    q, k, v, wg, rg = [x.contiguous() for x in (q, k, v, wg, rg)]
    out_aug = torch.zeros(B, NB, L, BV, device=q.device, dtype=torch.float32)
    _rola_fwd_tiled[(B, NB)](
        q, k, v, wg, rg, out_aug, L, dqk, dv, nc,
        q.stride(0), q.stride(1), q.stride(2), v.stride(0), v.stride(1), v.stride(2),
        wg.stride(0), wg.stride(1), wg.stride(2),
        out_aug.stride(0), out_aug.stride(1), out_aug.stride(2), out_aug.stride(3),
        BT=chunk, BD=BD, BV=BV, BG=BG, NCH=NCH)
    return out_aug[..., :dv + 1].sum(1)


@triton.autotune(configs=_AT_CFGS, key=_AT_KEY, **autotune_cache_kwargs)
@triton.jit
def _rola_fwd_intra(q_ptr, k_ptr, v_ptr, wg_ptr, rg_ptr, outa_ptr,
                    L, dqk, dv, nc,
                    sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c,
                    soa_b, soa_n, soa_l, soa_v,
                    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                    BG: tl.constexpr, ND: tl.constexpr, DEN: tl.constexpr):
    """Intra-chunk routed readout for one (batch, state-block, chunk). The [BT,BT] content gram is
    built by looping BK-blocks of the feature dim, so SRAM is bounded by [BT,BK]+[BT,BT] — NOT dqk.
    DEN=1: augment v with a ones-column (output width dv+1, num|den). DEN=0: numerator-only (width
    dv) — the kappa/per_state caller reconstructs the global den as Σ_c rᶜ·dᶜ from the den pre-pass,
    so the ones-column is redundant and BV halves to next_pow2(dv).
    Writes its OWN out_intra buffer (disjoint rows per chunk → plain store), so it is autotunable
    (num_warps/num_stages pruned by OutOfResources); the inter contribution is a separate buffer."""
    b = tl.program_id(0); sb = tl.program_id(1); t = tl.program_id(2)
    offs_t = tl.arange(0, BT); offs_v = tl.arange(0, BV)
    offs_c = sb * BG + tl.arange(0, BG); cmask = offs_c < nc
    rows = t * BT + offs_t; rmask = rows < L
    G = tl.zeros([BT, BT], dtype=tl.float32)
    for d0 in range(ND):
        offs_d = d0 * BK + tl.arange(0, BK); dmask = offs_d < dqk
        qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        G += tl.dot(qc, tl.trans(kc))
    rgc = tl.load(rg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                  mask=rmask[:, None] & cmask[None, :], other=0.0)
    wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                  mask=rmask[:, None] & cmask[None, :], other=0.0)
    vc = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                 mask=rmask[:, None] & (offs_v[None, :] < dv), other=0.0)
    if DEN:
        vc += tl.where((offs_v[None, :] == dv) & rmask[:, None], 1.0, 0.0)
    R = tl.dot(rgc, tl.trans(wgc))
    causal = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
    A = G * R * causal
    o = tl.dot(A.to(vc.dtype), vc)
    tl.store(outa_ptr + b*soa_b + sb*soa_n + rows[:, None]*soa_l + offs_v[None, :]*soa_v,
             o, mask=rmask[:, None] & (offs_v[None, :] < dv + DEN))


@triton.autotune(configs=_AT_CFGS, key=_AT_KEY, reset_to_zero=['outa_ptr'], **autotune_cache_kwargs)
@triton.jit
def _rola_fwd_inter(q_ptr, k_ptr, v_ptr, wg_ptr, rg_ptr, outa_ptr,
                    L, dqk, dv, nc,
                    sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c,
                    soa_b, soa_n, soa_l, soa_v,
                    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                    BG: tl.constexpr, NCH: tl.constexpr, DEN: tl.constexpr):
    """Inter-chunk (state) contribution for one (batch, state-block, FEATURE-block d0). Carries this
    feature-block's slice Sd[BK, BG*BV] of the Kronecker state across chunks → SRAM bounded by BK,
    not dqk. o_inter uses the state BEFORE this chunk's update (causal); partials over feature-blocks
    sum via atomic_add into its OWN out_inter buffer — autotunable with reset_to_zero (the autotuner
    zeros out_inter between benchmark trials so the atomic accumulation stays correct). DEN: as intra."""
    b = tl.program_id(0); sb = tl.program_id(1); d0 = tl.program_id(2)
    offs_t = tl.arange(0, BT); offs_v = tl.arange(0, BV)
    offs_d = d0 * BK + tl.arange(0, BK); dmask = offs_d < dqk
    offs_c = sb * BG + tl.arange(0, BG); cmask = offs_c < nc
    Sd = tl.zeros([BK, BG * BV], dtype=tl.float32)
    for t in range(NCH):
        rows = t * BT + offs_t; rmask = rows < L
        qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        rgc = tl.load(rg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                      mask=rmask[:, None] & cmask[None, :], other=0.0)
        P = tl.dot(qc, Sd.to(qc.dtype))
        P3 = tl.reshape(P, [BT, BG, BV])
        o_inter = tl.sum(P3 * rgc[:, :, None], axis=1)
        tl.atomic_add(outa_ptr + b*soa_b + sb*soa_n + rows[:, None]*soa_l + offs_v[None, :]*soa_v,
                      o_inter, mask=rmask[:, None] & (offs_v[None, :] < dv + DEN))
        kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        vc = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                     mask=rmask[:, None] & (offs_v[None, :] < dv), other=0.0)
        if DEN:
            vc += tl.where((offs_v[None, :] == dv) & rmask[:, None], 1.0, 0.0)
        wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                      mask=rmask[:, None] & cmask[None, :], other=0.0)
        WV = tl.reshape(wgc[:, :, None] * vc[:, None, :], [BT, BG * BV])
        Sd += tl.dot(tl.trans(kc), WV.to(kc.dtype))


def _fwd_aug_tiled(q, k, v, wg, rg, chunk, BG, BK=64, den=True):
    """D-tiled forward: smem bounded by BK (feature-block), so ANY dqk fits. den=True: augmented
    [B,L,dv+1] (numerator|den). den=False: numerator-only [B,L,dv] — BV halves to next_pow2(dv);
    the kappa/per_state caller forms the global den from the den pre-pass (Σ_c rᶜ·dᶜ)."""
    B, L, dqk = q.shape; dv = v.shape[-1]; nc = wg.shape[-1]
    DEN = 1 if den else 0
    dvp = dv + DEN
    BV = max(16, triton.next_power_of_2(dvp))
    BK = min(BK, max(16, triton.next_power_of_2(dqk)))   # exact-width blocks for dqk<=64; tile beyond
    ND = triton.cdiv(dqk, BK); NB = triton.cdiv(nc, BG); NCH = triton.cdiv(L, chunk)
    q, k, v, wg, rg = [x.contiguous() for x in (q, k, v, wg, rg)]
    # Separate intra/inter output buffers (summed after) so each kernel has a non-shared output and is
    # independently autotunable: intra writes disjoint rows (store), inter atomic-accumulates over
    # feature-blocks (reset_to_zero). FLA idiom — the autotuner prunes warps/stages per device.
    out_intra = torch.zeros(B, NB, L, BV, device=q.device, dtype=torch.float32)
    out_inter = torch.zeros_like(out_intra)
    so = (out_intra.stride(0), out_intra.stride(1), out_intra.stride(2), out_intra.stride(3))
    base = (q.stride(0), q.stride(1), q.stride(2), v.stride(0), v.stride(1), v.stride(2),
            wg.stride(0), wg.stride(1), wg.stride(2))
    _rola_fwd_intra[(B, NB, NCH)](q, k, v, wg, rg, out_intra, L, dqk, dv, nc, *base, *so,
                                  BT=chunk, BK=BK, BV=BV, BG=BG, ND=ND, DEN=DEN)
    _rola_fwd_inter[(B, NB, ND)](q, k, v, wg, rg, out_inter, L, dqk, dv, nc, *base, *so,
                                 BT=chunk, BK=BK, BV=BV, BG=BG, NCH=NCH, DEN=DEN)
    return (out_intra + out_inter)[..., :dvp].sum(1)


class _RoLARLAFn(torch.autograd.Function):
    """RoLA-RLA (no decay) un-normalized routed readout O = (G∘R∘causal)@v, on folded [BH,L,*] tensors.
    Forward returns the numerator only ([..,:dv]); backward pads the incoming grad with a zero
    denominator-column so the verified augmented kernels yield exactly the numerator's grads."""
    @staticmethod
    def forward(ctx, q, k, v, wg, rg, chunk, bwd_chunk, BG, den):
        dv = v.shape[-1]
        Oa = _fwd_aug_tiled(q, k, v, wg, rg, chunk=chunk, BG=BG, den=den)
        ctx.save_for_backward(q, k, v, wg, rg)
        ctx.bwd_chunk, ctx.BG, ctx.den = bwd_chunk, BG, den
        # den=False: Oa is [.,dv] (numerator). den=True: Oa is [.,dv+1]; [..,:dv] returns the readout
        # of the caller's columns (incl. a pre-augmented ones-column as the den), dropping the kernel's
        # own vestigial injected column — exactly the legacy augmented convention.
        return Oa[..., :dv].to(q.dtype)

    @staticmethod
    def backward(ctx, dO):
        q, k, v, wg, rg = ctx.saved_tensors
        # augmented path: pad incoming grad with a zero den-column ([.,dv+1]) so the kernels yield the
        # numerator's grads. numerator-only path (den=False): g IS the numerator grad ([.,dv]) — no pad.
        g = F.pad(dO.float(), (0, 1)) if ctx.den else dO.float()
        qf, kf, vf, wgf, rgf = (t.float() for t in (q, k, v, wg, rg))
        dq, dk, dvv, dw, dr = _bwd_split_rla(qf, kf, vf, wgf, rgf, g, chunk=_CHUNK, den=ctx.den)
        def cast(t): return t.to(q.dtype)
        return cast(dq), cast(dk), cast(dvv), cast(dw), cast(dr), None, None, None, None


def rola_rla_triton(q, k, v, r, w, chunk=None, bwd_chunk=16, BG=16, den=True):
    """Un-normalized routed RLA readout via Triton. q,k:[BH,L,K] v:[BH,L,V] r,w:[BH,L,nc]
    (r=read gate, w=write gate). Differentiable (fused Triton backward).
    den=True: caller passes v augmented with a ones-column ⇒ returns [num|den] (global-norm path).
    den=False: numerator-only ⇒ returns [BH,L,V] at BV=next_pow2(V) (half tiles); the kappa/per_state
    caller reconstructs the global denominator as Σ_c rᶜ·dᶜ from the per-state den pre-pass."""
    chunk = _CHUNK_FWD if chunk is None else min(chunk, _CHUNK_FWD)
    return _RoLARLAFn.apply(q, k, v, w, r, chunk, bwd_chunk, BG, den)


# ============================================================================
# Scalar-gated RoLA-GLA Triton kernels (shared-gram + per-state scalar decay).
#
# Decay is absorbed chunk-LOCALLY (rt=rg·e^a, wt=wg·e^{-a}, a=cumsum(ld)) so the effective weight
# keeps the shared-gram form; the Kronecker state is carried across chunks with a per-chunk decay
# (decayed scan). The per-token log-decay is floored (fp32-safe factored gram for BT≤32). Forward
# tiled over states; backward is the bespoke fused fwd-scan (dq,drg,da_rt) + reverse-scan
# (dk,dwg,dv,da_kwv,dLam) + a torch dld assembly (reverse-cumsum of da per chunk). Ported verbatim
# from the verified reference (all 6 grads incl. dld matched the torch GLA reference to fp noise).
# ============================================================================
_GLA_FLOOR = -2.5   # per-token log-decay floor (retention ≥ 8.2%/tok); fp32-safe for BT≤32


@triton.autotune(configs=_AT_CFGS, key=_AT_KEY, **autotune_cache_kwargs)
@triton.jit
def _rola_gla_fwd_intra(q_ptr, k_ptr, v_ptr, wg_ptr, rg_ptr, ld_ptr, outa_ptr,
                        L, dqk, dv, nc,
                        sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c,
                        soa_b, soa_n, soa_l, soa_v,
                        BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                        BG: tl.constexpr, ND: tl.constexpr, DEN: tl.constexpr):
    """GLA intra-chunk routed readout for one (batch, state-block, chunk). Same as the RLA intra but
    the routing gram uses the decayed gates rt=rg·e^a, wt=wg·e^-a (a = intra-chunk cumsum of ld). The
    [BT,BT] content gram is built by looping BK-blocks of the feature dim → SRAM bounded by BK.
    DEN=1: ones-column augment (num|den); DEN=0: numerator-only (BV halves to next_pow2(dv))."""
    b = tl.program_id(0); sb = tl.program_id(1); t = tl.program_id(2)
    offs_t = tl.arange(0, BT); offs_v = tl.arange(0, BV)
    offs_c = sb * BG + tl.arange(0, BG); cmask = offs_c < nc
    rows = t * BT + offs_t; rmask = rows < L
    G = tl.zeros([BT, BT], dtype=tl.float32)
    for d0 in range(ND):
        offs_d = d0 * BK + tl.arange(0, BK); dmask = offs_d < dqk
        qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        G += tl.dot(qc, tl.trans(kc))
    rgc = tl.load(rg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                  mask=rmask[:, None] & cmask[None, :], other=0.0)
    wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                  mask=rmask[:, None] & cmask[None, :], other=0.0)
    ldc = tl.load(ld_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                  mask=rmask[:, None] & cmask[None, :], other=0.0)
    vc = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                 mask=rmask[:, None] & (offs_v[None, :] < dv), other=0.0)
    if DEN:
        vc += tl.where((offs_v[None, :] == dv) & rmask[:, None], 1.0, 0.0)
    a = tl.cumsum(ldc, axis=0)
    rt = rgc * tl.exp(a)
    wt = wgc * tl.exp(-a)
    R = tl.dot(rt, tl.trans(wt))
    causal = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
    A = G * R * causal
    o = tl.dot(A.to(vc.dtype), vc)
    tl.store(outa_ptr + b*soa_b + sb*soa_n + rows[:, None]*soa_l + offs_v[None, :]*soa_v,
             o, mask=rmask[:, None] & (offs_v[None, :] < dv + DEN))


@triton.autotune(configs=_AT_CFGS, key=_AT_KEY, reset_to_zero=['outa_ptr'], **autotune_cache_kwargs)
@triton.jit
def _rola_gla_fwd_inter(q_ptr, k_ptr, v_ptr, wg_ptr, rg_ptr, ld_ptr, outa_ptr,
                        L, dqk, dv, nc,
                        sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c,
                        soa_b, soa_n, soa_l, soa_v,
                        BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                        BG: tl.constexpr, NCH: tl.constexpr, DEN: tl.constexpr):
    """GLA inter-chunk (state) contribution for one (batch, state-block, FEATURE-block d0). Carries
    this feature-block's slice Sd[BK, BG*BV] across chunks → SRAM bounded by BK, not dqk. The state
    decays by decvec = e^Λ (Λ = chunk-total ld) each chunk; the read uses rt = rg·e^a. The decay is
    d-independent, so each feature-block applies the same decvec; partials sum via atomic_add.
    DEN=1: ones-column augment; DEN=0: numerator-only (BV halves)."""
    b = tl.program_id(0); sb = tl.program_id(1); d0 = tl.program_id(2)
    offs_t = tl.arange(0, BT); offs_v = tl.arange(0, BV)
    offs_d = d0 * BK + tl.arange(0, BK); dmask = offs_d < dqk
    offs_c = sb * BG + tl.arange(0, BG); cmask = offs_c < nc
    Sd = tl.zeros([BK, BG * BV], dtype=tl.float32)
    for t in range(NCH):
        rows = t * BT + offs_t; rmask = rows < L
        qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        rgc = tl.load(rg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                      mask=rmask[:, None] & cmask[None, :], other=0.0)
        ldc = tl.load(ld_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                      mask=rmask[:, None] & cmask[None, :], other=0.0)
        a = tl.cumsum(ldc, axis=0)
        rt = rgc * tl.exp(a)
        P = tl.dot(qc, Sd.to(qc.dtype))
        P3 = tl.reshape(P, [BT, BG, BV])
        o_inter = tl.sum(P3 * rt[:, :, None], axis=1)
        tl.atomic_add(outa_ptr + b*soa_b + sb*soa_n + rows[:, None]*soa_l + offs_v[None, :]*soa_v,
                      o_inter, mask=rmask[:, None] & (offs_v[None, :] < dv + DEN))
        kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d,
                     mask=rmask[:, None] & dmask[None, :], other=0.0)
        vc = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                     mask=rmask[:, None] & (offs_v[None, :] < dv), other=0.0)
        if DEN:
            vc += tl.where((offs_v[None, :] == dv) & rmask[:, None], 1.0, 0.0)
        wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c,
                      mask=rmask[:, None] & cmask[None, :], other=0.0)
        Lam = tl.sum(tl.where(offs_t[:, None] == (BT - 1), a, 0.0), axis=0)
        w_end = wgc * tl.exp(Lam[None, :] - a)
        WV = tl.reshape(w_end[:, :, None] * vc[:, None, :], [BT, BG * BV])
        decvec = tl.reshape(tl.exp(Lam)[:, None] * tl.full([BG, BV], 1.0, tl.float32), [BG * BV])
        Sd = decvec[None, :] * Sd + tl.dot(tl.trans(kc), WV.to(kc.dtype))


def _gla_fwd_aug(q, k, v, wg, rg, ld, chunk, BG, BK=64, den=True):
    """GLA Triton forward. den=True: AUGMENTED [BH,L,dv+1] (num|den). den=False: numerator-only
    [BH,L,dv] at BV=next_pow2(dv) (half tiles); the kappa/per_state caller forms the global den from
    the GLA den pre-pass. D-tiled (intra d-loops the gram; inter carries an Sd[BK,*] slice)."""
    B, L, dqk = q.shape
    dv = v.shape[-1]
    nc = wg.shape[-1]
    DEN = 1 if den else 0
    dvp = dv + DEN
    ld = ld.clamp(min=_GLA_FLOOR).contiguous()
    BV = max(16, triton.next_power_of_2(dvp))
    BK = min(BK, max(16, triton.next_power_of_2(dqk)))   # exact-width blocks for dqk<=64; tile beyond
    ND = triton.cdiv(dqk, BK)
    NB = triton.cdiv(nc, BG)
    NCH = triton.cdiv(L, chunk)
    q, k, v, wg, rg = [x.contiguous() for x in (q, k, v, wg, rg)]
    # separate intra/inter buffers → each kernel autotunable (see _fwd_aug_tiled).
    out_intra = torch.zeros(B, NB, L, BV, device=q.device, dtype=torch.float32)
    out_inter = torch.zeros_like(out_intra)
    so = (out_intra.stride(0), out_intra.stride(1), out_intra.stride(2), out_intra.stride(3))
    base = (q.stride(0), q.stride(1), q.stride(2), v.stride(0), v.stride(1), v.stride(2),
            wg.stride(0), wg.stride(1), wg.stride(2))
    _rola_gla_fwd_intra[(B, NB, NCH)](q, k, v, wg, rg, ld, out_intra, L, dqk, dv, nc, *base, *so,
                                      BT=chunk, BK=BK, BV=BV, BG=BG, ND=ND, DEN=DEN)
    _rola_gla_fwd_inter[(B, NB, ND)](q, k, v, wg, rg, ld, out_inter, L, dqk, dv, nc, *base, *so,
                                     BT=chunk, BK=BK, BV=BV, BG=BG, NCH=NCH, DEN=DEN)
    return (out_intra + out_inter)[..., :dvp].sum(1)


class _RoLAGLAFn(torch.autograd.Function):
    """RoLA-GLA (scalar per-state decay) un-normalized routed readout, on folded [BH,L,*] tensors.
    Forward returns the numerator ([..,:dv]); backward pads the incoming grad with a zero den-column
    so the verified augmented kernels yield the numerator's grads (incl. dld)."""
    @staticmethod
    def forward(ctx, q, k, v, wg, rg, ld, chunk, BG, den):
        dv = v.shape[-1]
        Oa = _gla_fwd_aug(q, k, v, wg, rg, ld, chunk=chunk, BG=BG, den=den)
        ctx.save_for_backward(q, k, v, wg, rg, ld)
        ctx.BG, ctx.den = BG, den
        return Oa[..., :dv].to(q.dtype)   # see _RoLARLAFn.forward: legacy augmented convention

    @staticmethod
    def backward(ctx, dO):
        q, k, v, wg, rg, ld = ctx.saved_tensors
        g = F.pad(dO.float(), (0, 1)) if ctx.den else dO.float()   # augmented: zero den-col; else numerator-only
        def fl(t): return t.float()
        dq, dk, dvv, dwg, drg, dld = _bwd_split_gla(
            fl(q), fl(k), fl(v), fl(wg), fl(rg), fl(ld), g, chunk=_CHUNK, den=ctx.den)
        def cast(t): return t.to(q.dtype)
        # forward args order: q, k, v, wg, rg, ld, chunk, BG, den
        return cast(dq), cast(dk), cast(dvv), cast(dwg), cast(drg), cast(dld), None, None, None


def rola_gla_triton(q, k, v, r, w, ld, chunk=None, BG=16, den=True):
    """Un-normalized routed GLA readout via Triton. q,k:[BH,L,K] v:[BH,L,V] r,w,ld:[BH,L,nc]
    (r=read gate, w=write gate, ld=per-state log-decay). Differentiable. den=True: augmented [num|den];
    den=False: numerator-only (BV halves) — kappa/per_state caller forms the global den from the pre-pass."""
    chunk = _CHUNK if chunk is None else min(chunk, _CHUNK)
    return _RoLAGLAFn.apply(q, k, v, w, r, ld, chunk, BG, den)


# ============================================================================
# Phase F — chunk-PARALLEL backward.
#
# The fused backward kernels serialize all gram work behind a sequential chunk scan (grid (BH,NB) —
# occupancy cliff at low nc). Restructure FLA-style:
#   K1 _scan_S    (sequential, tiny): store S_before[t] for all chunks.
#   K2 _scan_dS   (sequential, tiny): store dS_after[t] (the pre-update value used at chunk t).
#   K3a/K3b       (PARALLEL over (BH, NB, NCH)): per-chunk grad kernels with the SAME working set as
#                 the fused pair (one state tile each) — a combined single K3 was tried and REGRESSED
#                 (both state tiles + both intermediate sets live ⇒ register spill).
# Per-chunk math copied verbatim from the verified fused kernels with carries replaced by loads.
# tl.dot needs K>=16 ⇒ BG>=16.
# ============================================================================


@triton.autotune(configs=_AT_CFGS, key=_AT_KEY, **autotune_cache_kwargs)
@triton.jit
def _scan_S(k_ptr, v_ptr, wg_ptr, ld_ptr, Sb_ptr, L, dqk, dv, nc,
            sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c,
            ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
            USE_G: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr, BV: tl.constexpr,
            BG: tl.constexpr, NCH: tl.constexpr, DEN: tl.constexpr):
    b = tl.program_id(0)
    sb = tl.program_id(1)
    d0 = tl.program_id(2)                       # feature-block: this program owns Sflat rows [d0*BD:]
    offs_t = tl.arange(0, BT)
    offs_d = d0 * BD + tl.arange(0, BD)
    offs_v = tl.arange(0, BV)
    offs_e = tl.arange(0, BG * BV)
    offs_c = sb * BG + tl.arange(0, BG)
    dmask = offs_d < dqk
    cmask = offs_c < nc
    Sflat = tl.zeros([BD, BG * BV], dtype=tl.float32)
    for t in range(NCH):
        rows = t * BT + offs_t
        rmask = rows < L
        kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]
                     * sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        vc = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                     mask=rmask[:, None] & (offs_v[None, :] < dv), other=0.0)
        if DEN:
            vc += tl.where((offs_v[None, :] == dv) & rmask[:, None], 1.0, 0.0)
        wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]
                      * sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
        tl.store(Sb_ptr + b*ssb_b + sb*ssb_n + t*ssb_t + offs_d[:, None]*ssb_d + offs_e[None, :]*ssb_e,
                 Sflat, mask=dmask[:, None])
        if USE_G:
            ldc = tl.load(ld_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]
                          * sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
            a = tl.cumsum(ldc, axis=0)
            Lam = tl.sum(tl.where(offs_t[:, None] == (BT - 1), a, 0.0), axis=0)
            w_end = wgc * tl.exp(Lam[None, :] - a)
            WV = tl.reshape(w_end[:, :, None] * vc[:, None, :], [BT, BG * BV])
            decvec = tl.reshape(tl.exp(Lam)[:, None] * tl.full([BG, BV], 1.0, tl.float32), [BG * BV])
            Sflat = decvec[None, :] * Sflat + tl.dot(tl.trans(kc), WV.to(kc.dtype))
        else:
            WV = tl.reshape(wgc[:, :, None] * vc[:, None, :], [BT, BG * BV])
            Sflat += tl.dot(tl.trans(kc), WV.to(kc.dtype))


@triton.autotune(configs=_AT_CFGS, key=_AT_KEY, **autotune_cache_kwargs)
@triton.jit
def _scan_dS(q_ptr, rg_ptr, ld_ptr, g_ptr, dSa_ptr, L, dqk, dv, nc,
             sq_b, sq_l, sq_d, sg_b, sg_l, sg_c, sgr_b, sgr_l, sgr_d,
             ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
             USE_G: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr, BV: tl.constexpr,
             BG: tl.constexpr, NCH: tl.constexpr, DEN: tl.constexpr):
    b = tl.program_id(0)
    sb = tl.program_id(1)
    d0 = tl.program_id(2)                       # feature-block: this program owns dS rows [d0*BD:]
    offs_t = tl.arange(0, BT)
    offs_d = d0 * BD + tl.arange(0, BD)
    offs_v = tl.arange(0, BV)
    offs_e = tl.arange(0, BG * BV)
    offs_c = sb * BG + tl.arange(0, BG)
    dmask = offs_d < dqk
    cmask = offs_c < nc
    vmask = offs_v < (dv + DEN)
    dS = tl.zeros([BD, BG * BV], dtype=tl.float32)
    for ti in range(NCH):
        t = NCH - 1 - ti
        rows = t * BT + offs_t
        rmask = rows < L
        qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]
                     * sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        rgc = tl.load(rg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]
                      * sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
        gc = tl.load(g_ptr + b*sgr_b + rows[:, None]*sgr_l + offs_v[None, :]
                     * sgr_d, mask=rmask[:, None] & vmask[None, :], other=0.0)
        tl.store(dSa_ptr + b*ssb_b + sb*ssb_n + t*ssb_t + offs_d[:, None]*ssb_d + offs_e[None, :]*ssb_e,
                 dS, mask=dmask[:, None])
        if USE_G:
            ldc = tl.load(ld_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]
                          * sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
            a = tl.cumsum(ldc, axis=0)
            rt = rgc * tl.exp(a)
            Lam = tl.sum(tl.where(offs_t[:, None] == (BT - 1), a, 0.0), axis=0)
            rt_g = tl.reshape(rt[:, :, None] * gc[:, None, :], [BT, BG * BV])
            decvec = tl.reshape(tl.exp(Lam)[:, None] * tl.full([BG, BV], 1.0, tl.float32), [BG * BV])
            dS = decvec[None, :] * dS + tl.dot(tl.trans(qc), rt_g.to(qc.dtype))
        else:
            rg_g = tl.reshape(rgc[:, :, None] * gc[:, None, :], [BT, BG * BV])
            dS += tl.dot(tl.trans(qc), rg_g.to(qc.dtype))


@triton.autotune(configs=_BWD_CFGS, key=_AT_KEY,
                 prune_configs_by={'early_config_prune': _prune_bwd_bd}, **autotune_cache_kwargs)
@triton.jit
def _par_grad_rla_qr(q_ptr, k_ptr, v_ptr, wg_ptr, rg_ptr, g_ptr, Sb_ptr, dq_ptr, dr_ptr,
                     L, dqk: tl.constexpr, dv, nc,
                     sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c, sgr_b, sgr_l, sgr_d,
                     ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
                     sdq_b, sdq_n, sdq_l, sdq_d, sdr_b, sdr_l, sdr_c,
                     BT: tl.constexpr, BD: tl.constexpr, BV: tl.constexpr, BG: tl.constexpr,
                     NCH: tl.constexpr, DEN: tl.constexpr):
    # D-tiled: dq is feature-indexed (written per BD-block); dr needs the full content gram G + QS,
    # accumulated over the BD-block loop ([BT,BT] and [BT,BG*BV] — bounded, independent of dqk).
    # BD is an autotune knob; the number of feature-blocks follows from it (dqk is runtime).
    ND = tl.cdiv(dqk, BD)
    b = tl.program_id(0)
    sb = tl.program_id(1)
    t = tl.program_id(2)
    offs_t = tl.arange(0, BT)
    offs_v = tl.arange(0, BV)
    offs_e = tl.arange(0, BG * BV)
    offs_c = sb * BG + tl.arange(0, BG)
    cmask = offs_c < nc
    vmask = offs_v < (dv + DEN)
    rows = t * BT + offs_t
    rmask = rows < L
    v1 = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                 mask=rmask[:, None] & (offs_v[None, :] < dv), other=0.0)
    if DEN:
        v1 += tl.where((offs_v[None, :] == dv) & rmask[:, None], 1.0, 0.0)
    rgc = tl.load(rg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
    wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
    gc = tl.load(g_ptr + b*sgr_b + rows[:, None]*sgr_l + offs_v[None, :]
                 * sgr_d, mask=rmask[:, None] & vmask[None, :], other=0.0)
    causal = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
    Rg = tl.dot(rgc, tl.trans(wgc))
    P = tl.dot(gc, tl.trans(v1))
    coef = causal * Rg * P                                          # dq_intra coefficient [BT,BT]
    rg_g = tl.reshape(rgc[:, :, None] * gc[:, None, :], [BT, BG * BV])
    G = tl.zeros([BT, BT], dtype=tl.float32)
    QS = tl.zeros([BT, BG * BV], dtype=tl.float32)
    for d0b in range(ND):
        offs_d = d0b * BD + tl.arange(0, BD)
        dmask = offs_d < dqk
        qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        Sb = tl.load(Sb_ptr + b*ssb_b + sb*ssb_n + t*ssb_t + offs_d[:, None]
                     * ssb_d + offs_e[None, :]*ssb_e, mask=dmask[:, None], other=0.0)
        dq_d = tl.dot(coef.to(kc.dtype), kc) + tl.dot(rg_g.to(Sb.dtype), tl.trans(Sb))
        tl.store(dq_ptr + b*sdq_b + sb*sdq_n + rows[:, None]*sdq_l + offs_d[None, :]*sdq_d,
                 dq_d, mask=rmask[:, None] & dmask[None, :])
        G += tl.dot(qc, tl.trans(kc))
        QS += tl.dot(qc, Sb.to(qc.dtype))
    dr_intra = tl.dot((causal * G * P).to(wgc.dtype), wgc)
    dr_inter = tl.sum(tl.reshape(QS, [BT, BG, BV]) * gc[:, None, :], axis=2)
    tl.store(dr_ptr + b*sdr_b + rows[:, None]*sdr_l + offs_c[None, :]*sdr_c,
             dr_intra + dr_inter, mask=rmask[:, None] & cmask[None, :])


@triton.autotune(configs=_BWD_CFGS, key=_AT_KEY,
                 prune_configs_by={'early_config_prune': _prune_bwd_bd}, **autotune_cache_kwargs)
@triton.jit
def _par_grad_rla_kwv(q_ptr, k_ptr, v_ptr, wg_ptr, rg_ptr, g_ptr, dSa_ptr, dk_ptr, dw_ptr, dv_ptr,
                      L, dqk: tl.constexpr, dv, nc,
                      sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c, sgr_b, sgr_l, sgr_d,
                      ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
                      sdk_b, sdk_n, sdk_l, sdk_d, sdw_b, sdw_l, sdw_c, sdv_b, sdv_n, sdv_l, sdv_d,
                      BT: tl.constexpr, BD: tl.constexpr, BV: tl.constexpr, BG: tl.constexpr,
                      NCH: tl.constexpr, DEN: tl.constexpr):
    # D-tiled: dk is feature-indexed (written per BD-block); dw/dv need the full content gram G + KS,
    # accumulated over the BD-block loop ([BT,BT] and [BT,BG*BV] — bounded, independent of dqk).
    # BD is an autotune knob; the number of feature-blocks follows from it (dqk is runtime).
    ND = tl.cdiv(dqk, BD)
    b = tl.program_id(0)
    sb = tl.program_id(1)
    t = tl.program_id(2)
    offs_t = tl.arange(0, BT)
    offs_v = tl.arange(0, BV)
    offs_e = tl.arange(0, BG * BV)
    offs_c = sb * BG + tl.arange(0, BG)
    cmask = offs_c < nc
    vmask = offs_v < (dv + DEN)
    rows = t * BT + offs_t
    rmask = rows < L
    v1 = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                 mask=rmask[:, None] & (offs_v[None, :] < dv), other=0.0)
    if DEN:
        v1 += tl.where((offs_v[None, :] == dv) & rmask[:, None], 1.0, 0.0)
    rgc = tl.load(rg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
    wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
    gc = tl.load(g_ptr + b*sgr_b + rows[:, None]*sgr_l + offs_v[None, :]
                 * sgr_d, mask=rmask[:, None] & vmask[None, :], other=0.0)
    causal = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
    Rg = tl.dot(rgc, tl.trans(wgc))
    P = tl.dot(gc, tl.trans(v1))
    A2 = Rg * P * causal                                            # dk_intra coef [BT,BT]
    wg_v1 = tl.reshape(wgc[:, :, None] * v1[:, None, :], [BT, BG * BV])
    G = tl.zeros([BT, BT], dtype=tl.float32)
    KS = tl.zeros([BT, BG * BV], dtype=tl.float32)
    for d0b in range(ND):
        offs_d = d0b * BD + tl.arange(0, BD)
        dmask = offs_d < dqk
        qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        dSa = tl.load(dSa_ptr + b*ssb_b + sb*ssb_n + t*ssb_t +
                      offs_d[:, None]*ssb_d + offs_e[None, :]*ssb_e, mask=dmask[:, None], other=0.0)
        dk_d = tl.dot(tl.trans(A2).to(qc.dtype), qc) + tl.dot(wg_v1.to(dSa.dtype), tl.trans(dSa))
        tl.store(dk_ptr + b*sdk_b + sb*sdk_n + rows[:, None]*sdk_l + offs_d[None, :]*sdk_d,
                 dk_d, mask=rmask[:, None] & dmask[None, :])
        G += tl.dot(qc, tl.trans(kc))
        KS += tl.dot(kc, dSa.to(kc.dtype))
    A = G * Rg * causal
    B2 = G * P * causal
    dw_intra = tl.dot(tl.trans(B2).to(rgc.dtype), rgc)
    dv_intra = tl.dot(tl.trans(A).to(gc.dtype), gc)
    KS3 = tl.reshape(KS, [BT, BG, BV])
    dw_wr = tl.sum(KS3 * v1[:, None, :], axis=2)
    dv_wr = tl.sum(wgc[:, :, None] * KS3, axis=1)
    tl.store(dw_ptr + b*sdw_b + rows[:, None]*sdw_l + offs_c[None, :]*sdw_c,
             dw_intra + dw_wr, mask=rmask[:, None] & cmask[None, :])
    tl.store(dv_ptr + b*sdv_b + sb*sdv_n + rows[:, None]*sdv_l + offs_v[None, :]*sdv_d,
             dv_intra + dv_wr, mask=rmask[:, None] & (offs_v[None, :] < dv))


@triton.autotune(configs=_BWD_CFGS, key=_AT_KEY,
                 prune_configs_by={'early_config_prune': _prune_bwd_bd}, **autotune_cache_kwargs)
@triton.jit
def _par_grad_gla_qr(q_ptr, k_ptr, v_ptr, wg_ptr, rg_ptr, ld_ptr, g_ptr, Sb_ptr,
                     dq_ptr, drg_ptr, dart_ptr,
                     L, dqk: tl.constexpr, dv, nc,
                     sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c, sgr_b, sgr_l, sgr_d,
                     ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
                     sdq_b, sdq_n, sdq_l, sdq_d, sdr_b, sdr_l, sdr_c, sda_b, sda_l, sda_c,
                     BT: tl.constexpr, BD: tl.constexpr, BV: tl.constexpr, BG: tl.constexpr, NCH: tl.constexpr,
                     DEN: tl.constexpr):
    # D-tiled: dq is feature-indexed (per BD-block); drg/dart need the full content gram G + QS,
    # accumulated over the BD-block loop. dG = P*D*caus is d-independent so it stays before the loop;
    # dD needs the full G so it follows. BD is an autotune knob; ND follows from dqk (runtime).
    ND = tl.cdiv(dqk, BD)
    b = tl.program_id(0)
    sb = tl.program_id(1)
    t = tl.program_id(2)
    dvp = dv + DEN
    offs_t = tl.arange(0, BT)
    offs_v = tl.arange(0, BV)
    offs_e = tl.arange(0, BG * BV)
    offs_c = sb * BG + tl.arange(0, BG)
    cmask = offs_c < nc
    rows = t * BT + offs_t
    rmask = rows < L
    v1 = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                 mask=rmask[:, None] & (offs_v[None, :] < dv), other=0.0)
    if DEN:
        v1 += tl.where((offs_v[None, :] == dv) & rmask[:, None], 1.0, 0.0)
    rgc = tl.load(rg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
    wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
    ldc = tl.load(ld_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
    gc = tl.load(g_ptr + b*sgr_b + rows[:, None]*sgr_l + offs_v[None, :]*sgr_d,
                 mask=rmask[:, None] & (offs_v[None, :] < dvp), other=0.0)
    a = tl.cumsum(ldc, axis=0)
    ea = tl.exp(a)
    rt = rgc * ea
    wt = wgc * tl.exp(-a)
    caus = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
    D = tl.dot(rt, tl.trans(wt))
    P = tl.dot(gc, tl.trans(v1))
    dG = P * D * caus
    rt_g = tl.reshape(rt[:, :, None] * gc[:, None, :], [BT, BG * BV])
    G = tl.zeros([BT, BT], dtype=tl.float32)
    QS = tl.zeros([BT, BG * BV], dtype=tl.float32)
    for d0b in range(ND):
        offs_d = d0b * BD + tl.arange(0, BD)
        dmask = offs_d < dqk
        qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        Sb = tl.load(Sb_ptr + b*ssb_b + sb*ssb_n + t*ssb_t + offs_d[:, None]
                     * ssb_d + offs_e[None, :]*ssb_e, mask=dmask[:, None], other=0.0)
        dq_intra = tl.dot(dG.to(kc.dtype), kc)
        dq_inter = tl.dot(rt_g.to(Sb.dtype), tl.trans(Sb))
        tl.store(dq_ptr + b*sdq_b + sb*sdq_n + rows[:, None]*sdq_l + offs_d[None, :]*sdq_d,
                 dq_intra + dq_inter, mask=rmask[:, None] & dmask[None, :])
        G += tl.dot(qc, tl.trans(kc))
        QS += tl.dot(qc, Sb.to(qc.dtype))
    dD = P * G * caus
    drt_intra = tl.dot(dD.to(wt.dtype), wt)
    drt_inter = tl.sum(tl.reshape(QS, [BT, BG, BV]) * gc[:, None, :], axis=2)
    drt = drt_intra + drt_inter
    tl.store(drg_ptr + b*sdr_b + rows[:, None]*sdr_l + offs_c[None, :]*sdr_c, drt * ea, mask=rmask[:, None] & cmask[None, :])
    tl.store(dart_ptr + b*sda_b + rows[:, None]*sda_l + offs_c[None, :]*sda_c, drt * rt, mask=rmask[:, None] & cmask[None, :])


@triton.autotune(configs=_BWD_CFGS, key=_AT_KEY,
                 prune_configs_by={'early_config_prune': _prune_bwd_bd}, **autotune_cache_kwargs)
@triton.jit
def _par_grad_gla_kwv(q_ptr, k_ptr, v_ptr, wg_ptr, rg_ptr, ld_ptr, g_ptr, Sb_ptr, dSa_ptr, dart_ptr,
                      dk_ptr, dwg_ptr, dv_ptr, dld_ptr,
                      L, dqk: tl.constexpr, dv, nc,
                      sq_b, sq_l, sq_d, sv_b, sv_l, sv_d, sg_b, sg_l, sg_c, sgr_b, sgr_l, sgr_d,
                      ssb_b, ssb_n, ssb_t, ssb_d, ssb_e,
                      sdk_b, sdk_n, sdk_l, sdk_d, sdw_b, sdw_l, sdw_c, sdv_b, sdv_n, sdv_l, sdv_d,
                      sda_b, sda_l, sda_c,
                      BT: tl.constexpr, BD: tl.constexpr, BV: tl.constexpr, BG: tl.constexpr, NCH: tl.constexpr,
                      DEN: tl.constexpr):
    # D-tiled: dk is feature-indexed (per BD-block); dwg/dv/dld need the full G + KS, accumulated over
    # the BD-block loop. dG = P*D*caus is d-independent (used for dk_intra in the loop); A and dD need
    # the full G so they follow. dLam is now assembled IN-KERNEL (the den-kernel trick): the BD-loop
    # also accumulates ZdZ = Σ_{d,v}(Sb∘dSa) (the carry-decay adjoint), then da = da_rt+da_wt+da_wend
    # gets dLam folded into its last row and a reverse-cumsum yields dld — no torch host-side loop.
    ND = tl.cdiv(dqk, BD)
    b = tl.program_id(0)
    sb = tl.program_id(1)
    t = tl.program_id(2)
    dvp = dv + DEN
    offs_t = tl.arange(0, BT)
    offs_v = tl.arange(0, BV)
    offs_e = tl.arange(0, BG * BV)
    offs_c = sb * BG + tl.arange(0, BG)
    cmask = offs_c < nc
    rows = t * BT + offs_t
    rmask = rows < L
    v1 = tl.load(v_ptr + b*sv_b + rows[:, None]*sv_l + offs_v[None, :]*sv_d,
                 mask=rmask[:, None] & (offs_v[None, :] < dv), other=0.0)
    if DEN:
        v1 += tl.where((offs_v[None, :] == dv) & rmask[:, None], 1.0, 0.0)
    rgc = tl.load(rg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
    wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
    ldc = tl.load(ld_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
    gc = tl.load(g_ptr + b*sgr_b + rows[:, None]*sgr_l + offs_v[None, :]*sgr_d,
                 mask=rmask[:, None] & (offs_v[None, :] < dvp), other=0.0)
    a = tl.cumsum(ldc, axis=0)
    ena = tl.exp(-a)
    rt = rgc * tl.exp(a)
    wt = wgc * ena
    Lam = tl.sum(tl.where(offs_t[:, None] == (BT - 1), a, 0.0), axis=0)
    w_end = wgc * tl.exp(Lam[None, :] - a)
    caus = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
    D = tl.dot(rt, tl.trans(wt))
    P = tl.dot(gc, tl.trans(v1))
    dG = P * D * caus
    wv1 = tl.reshape(w_end[:, :, None] * v1[:, None, :], [BT, BG * BV])
    G = tl.zeros([BT, BT], dtype=tl.float32)
    KS = tl.zeros([BT, BG * BV], dtype=tl.float32)
    ZdZ = tl.zeros([BG], dtype=tl.float32)
    for d0b in range(ND):
        offs_d = d0b * BD + tl.arange(0, BD)
        dmask = offs_d < dqk
        qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        Sb = tl.load(Sb_ptr + b*ssb_b + sb*ssb_n + t*ssb_t +
                     offs_d[:, None]*ssb_d + offs_e[None, :]*ssb_e, mask=dmask[:, None], other=0.0)
        dSa = tl.load(dSa_ptr + b*ssb_b + sb*ssb_n + t*ssb_t +
                      offs_d[:, None]*ssb_d + offs_e[None, :]*ssb_e, mask=dmask[:, None], other=0.0)
        dk_intra = tl.dot(tl.trans(dG).to(qc.dtype), qc)
        dk_KV = tl.dot(wv1.to(dSa.dtype), tl.trans(dSa))
        tl.store(dk_ptr + b*sdk_b + sb*sdk_n + rows[:, None]*sdk_l + offs_d[None, :]*sdk_d,
                 dk_intra + dk_KV, mask=rmask[:, None] & dmask[None, :])
        G += tl.dot(qc, tl.trans(kc))
        KS += tl.dot(kc, dSa.to(kc.dtype))
        ZdZ += tl.sum(tl.sum(tl.reshape(Sb * dSa, [BD, BG, BV]), axis=2), axis=0)   # Σ_{d,v}(Sb∘dSa)
    A = G * D * caus
    dD = P * G * caus
    dv_intra = tl.dot(tl.trans(A).to(gc.dtype), gc)
    dwt = tl.dot(tl.trans(dD).to(rt.dtype), rt)
    KS3 = tl.reshape(KS, [BT, BG, BV])
    dw_end = tl.sum(KS3 * v1[:, None, :], axis=2)
    dv_KV = tl.sum(w_end[:, :, None] * KS3, axis=1)
    da_wend = -dw_end * w_end
    dwg_wend = dw_end * tl.exp(Lam[None, :] - a)
    dwgc = dwt * ena + dwg_wend
    da_wt = -dwt * wt
    tl.store(dwg_ptr + b*sdw_b + rows[:, None]*sdw_l + offs_c[None, :]*sdw_c, dwgc, mask=rmask[:, None] & cmask[None, :])
    tl.store(dv_ptr + b*sdv_b + sb*sdv_n + rows[:, None]*sdv_l + offs_v[None, :]*sdv_d,
             dv_intra + dv_KV, mask=rmask[:, None] & (offs_v[None, :] < dv))
    # dLam (carry-decay adjoint) folded into da's last row, then reverse-cumsum → dld. da's read-gate
    # term da_rt comes from the qr kernel (dart); da_wt + da_wend are local. (Mirrors _den_gla_grad.)
    dart = tl.load(dart_ptr + b*sda_b + rows[:, None]*sda_l + offs_c[None, :]*sda_c,
                   mask=rmask[:, None] & cmask[None, :], other=0.0)
    dlam = tl.exp(Lam) * ZdZ - tl.sum(da_wend, axis=0)
    da = dart + da_wt + da_wend
    da += tl.where(offs_t[:, None] == (BT - 1), dlam[None, :], 0.0)
    s = tl.cumsum(da, axis=0)
    dld = tl.sum(da, axis=0)[None, :] - s + da                       # reverse cumsum (tot − cumsum + da)
    tl.store(dld_ptr + b*sdw_b + rows[:, None]*sdw_l + offs_c[None, :]*sdw_c, dld, mask=rmask[:, None] & cmask[None, :])


def _alloc_split(q, v, wg, chunk, BG, den=True):
    B, L, dqk = q.shape
    dv = v.shape[-1]
    nc = wg.shape[-1]
    BD = max(16, triton.next_power_of_2(dqk))
    BV = max(16, triton.next_power_of_2(dv + (1 if den else 0)))
    NB = triton.cdiv(nc, BG)
    NCH = triton.cdiv(L, chunk)
    Sb = torch.empty(B, NB, NCH, BD, BG * BV, device=q.device, dtype=torch.float32)
    dSa = torch.empty_like(Sb)
    dq = torch.empty(B, NB, L, dqk, device=q.device, dtype=torch.float32)
    dk = torch.empty_like(dq)
    dvo = torch.empty(B, NB, L, BV, device=q.device, dtype=torch.float32)
    dr = torch.empty(B, L, nc, device=q.device, dtype=torch.float32)
    dw = torch.empty_like(dr)
    return BD, BV, NB, NCH, Sb, dSa, dq, dk, dvo, dr, dw


def _bwd_split_rla(q, k, v, wg, rg, g, chunk=None, BG=16, den=True):
    chunk = _CHUNK if chunk is None else chunk
    DEN = 1 if den else 0
    B, L, dqk = q.shape
    dv = v.shape[-1]
    nc = wg.shape[-1]
    q, k, v, wg, rg, g = [x.contiguous() for x in (q, k, v, wg, rg, g)]
    BD, BV, NB, NCH, Sb, dSa, dq, dk, dvo, dr, dw = _alloc_split(q, v, wg, chunk, BG, den=den)
    # The scans build Sb/dSa with their OWN feature block (Sflat is a register accumulator, not a
    # loaded SRAM tile, so a fixed block is fine and independent of the grad kernels' autotuned BD —
    # Sb is indexed by absolute feature row). The grad kernels pick BD by autotune (SRAM-fit empirical).
    BK_scan = min(64, BD)
    ND_scan = triton.cdiv(dqk, BK_scan)
    sS = (Sb.stride(0), Sb.stride(1), Sb.stride(2), Sb.stride(3), Sb.stride(4))
    sq = (q.stride(0), q.stride(1), q.stride(2))
    sv = (v.stride(0), v.stride(1), v.stride(2))
    sg = (wg.stride(0), wg.stride(1), wg.stride(2))
    sgr = (g.stride(0), g.stride(1), g.stride(2))
    _scan_S[(B, NB, ND_scan)](k, v, wg, wg, Sb, L, dqk, dv, nc, *sq, *sv, *sg, *sS,
                              USE_G=False, BT=chunk, BD=BK_scan, BV=BV, BG=BG, NCH=NCH, DEN=DEN)
    _scan_dS[(B, NB, ND_scan)](q, rg, wg, g, dSa, L, dqk, dv, nc, *sq, *sg, *sgr, *sS,
                               USE_G=False, BT=chunk, BD=BK_scan, BV=BV, BG=BG, NCH=NCH, DEN=DEN)
    _par_grad_rla_qr[(B, NB, NCH)](q, k, v, wg, rg, g, Sb, dq, dr, L, dqk, dv, nc,
                                   *sq, *sv, *sg, *sgr, *sS,
                                   dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(
                                       3), dr.stride(0), dr.stride(1), dr.stride(2),
                                   BT=chunk, BV=BV, BG=BG, NCH=NCH, DEN=DEN)
    _par_grad_rla_kwv[(B, NB, NCH)](q, k, v, wg, rg, g, dSa, dk, dw, dvo, L, dqk, dv, nc,
                                    *sq, *sv, *sg, *sgr, *sS,
                                    dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(
                                        3), dw.stride(0), dw.stride(1), dw.stride(2),
                                    dvo.stride(0), dvo.stride(1), dvo.stride(2), dvo.stride(3),
                                    BT=chunk, BV=BV, BG=BG, NCH=NCH, DEN=DEN)
    return dq.sum(1), dk.sum(1), dvo.sum(1)[..., :dv], dw, dr


def _bwd_split_gla(q, k, v, wg, rg, ld, g, chunk=None, BG=16, den=True):
    chunk = _CHUNK if chunk is None else chunk
    DEN = 1 if den else 0
    B, L, dqk = q.shape
    dv = v.shape[-1]
    nc = wg.shape[-1]
    ld = ld.clamp(min=_GLA_FLOOR)
    q, k, v, wg, rg, ld, g = [x.contiguous() for x in (q, k, v, wg, rg, ld, g)]
    BD, BV, NB, NCH, Sb, dSa, dq, dk, dvo, drg, dwg = _alloc_split(q, v, wg, chunk, BG, den=den)
    dart = torch.empty(B, L, nc, device=q.device, dtype=torch.float32)   # da_rt (read-gate decay adjoint), qr→kwv
    dld = torch.empty(B, L, nc, device=q.device, dtype=torch.float32)    # per-token log-decay grad, assembled in kwv
    # Scans build Sb/dSa with their own fixed feature block (register accumulator, independent of the
    # grad kernels' autotuned BD; Sb indexed by absolute feature row).
    BK_scan = min(64, BD)
    ND_scan = triton.cdiv(dqk, BK_scan)
    sS = (Sb.stride(0), Sb.stride(1), Sb.stride(2), Sb.stride(3), Sb.stride(4))
    sq = (q.stride(0), q.stride(1), q.stride(2))
    sv = (v.stride(0), v.stride(1), v.stride(2))
    sg = (wg.stride(0), wg.stride(1), wg.stride(2))
    sgr = (g.stride(0), g.stride(1), g.stride(2))
    _scan_S[(B, NB, ND_scan)](k, v, wg, ld, Sb, L, dqk, dv, nc, *sq, *sv, *sg, *sS,
                              USE_G=True, BT=chunk, BD=BK_scan, BV=BV, BG=BG, NCH=NCH, DEN=DEN)
    _scan_dS[(B, NB, ND_scan)](q, rg, ld, g, dSa, L, dqk, dv, nc, *sq, *sg, *sgr, *sS,
                               USE_G=True, BT=chunk, BD=BK_scan, BV=BV, BG=BG, NCH=NCH, DEN=DEN)
    _par_grad_gla_qr[(B, NB, NCH)](q, k, v, wg, rg, ld, g, Sb, dq, drg, dart, L, dqk, dv, nc,
                                   *sq, *sv, *sg, *sgr, *sS,
                                   dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(
                                       3), drg.stride(0), drg.stride(1), drg.stride(2),
                                   dart.stride(0), dart.stride(1), dart.stride(2),
                                   BT=chunk, BV=BV, BG=BG, NCH=NCH, DEN=DEN)
    # kwv assembles dld IN-KERNEL (Sb + dSa → ZdZ → dLam → reverse-cumsum); dart (da_rt) comes from qr.
    _par_grad_gla_kwv[(B, NB, NCH)](q, k, v, wg, rg, ld, g, Sb, dSa, dart, dk, dwg, dvo, dld, L, dqk, dv, nc,
                                    *sq, *sv, *sg, *sgr, *sS,
                                    dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                                    dwg.stride(0), dwg.stride(1), dwg.stride(2),
                                    dvo.stride(0), dvo.stride(1), dvo.stride(2), dvo.stride(3),
                                    dart.stride(0), dart.stride(1), dart.stride(2),
                                    BT=chunk, BV=BV, BG=BG, NCH=NCH, DEN=DEN)
    return dq.sum(1), dk.sum(1), dvo.sum(1)[..., :dv], dwg, drg, dld


# ============================================================================
# Phase G — per-state DENOMINATOR kernel (for kappa / per-state normalization).
#
# d[i,c] = Σ_{j≤i} (φq_i·φk_j) w_j^c — the mass state c contributes to token i's partition
# function. Used to rescale read gates: r̃ = r·(d+ε)^{-κ(x)} (κ=0 global, κ=1 per-state, exact).
# The eager torch version retains its chunk grams for backward (VRAM blowup at LM scale); this
# is the same scan+parallel pattern as the main backward at [BD,BG] state scale (tiny buffers).
# ============================================================================
_DEN_KEY = ['dqk', 'nc']


@triton.jit
def _den_fwd_intra(q_ptr, k_ptr, wg_ptr, ld_ptr, d_ptr, L, dqk, nc,
                   sq_b, sq_l, sq_d, sg_b, sg_l, sg_c, sd_b, sd_l, sd_c,
                   USE_G: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BG: tl.constexpr, ND: tl.constexpr):
    """Intra-chunk den for one (batch, state-block, chunk). The [BT,BT] content gram is built by
    BK-blocking the feature dim (SRAM bounded by BK, not dqk). atomic_adds dch_intra into d (a
    torch.zeros target — the inter kernel adds the cross-chunk part). USE_G pre-scales by e^a (row-
    wise, distributes over the intra+inter sum)."""
    b = tl.program_id(0); sb = tl.program_id(1); t = tl.program_id(2)
    offs_t = tl.arange(0, BT)
    offs_c = sb * BG + tl.arange(0, BG); cmask = offs_c < nc
    rows = t * BT + offs_t; rmask = rows < L
    wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
    caus = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
    G = tl.zeros([BT, BT], dtype=tl.float32)
    for d0 in range(ND):
        offs_d = d0 * BK + tl.arange(0, BK); dmask = offs_d < dqk
        qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        G += tl.dot(qc, tl.trans(kc))
    if USE_G:
        ldc = tl.load(ld_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
        a = tl.cumsum(ldc, axis=0)
        wt = wgc * tl.exp(-a)
        dch = tl.exp(a) * tl.dot(G * caus, wt)
    else:
        dch = tl.dot((G * caus).to(wgc.dtype), wgc)
    tl.atomic_add(d_ptr + b*sd_b + rows[:, None]*sd_l + offs_c[None, :]*sd_c, dch, mask=rmask[:, None] & cmask[None, :])


@triton.jit
def _den_fwd_inter(q_ptr, k_ptr, wg_ptr, ld_ptr, d_ptr, Zb_ptr, L, dqk, nc,
                   sq_b, sq_l, sq_d, sg_b, sg_l, sg_c, sd_b, sd_l, sd_c,
                   szb_b, szb_n, szb_t, szb_d, szb_c,
                   USE_G: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BG: tl.constexpr, NCH: tl.constexpr):
    """Inter-chunk (state) den for one (batch, state-block, FEATURE-block d0). Carries this block's
    slice Zd[BK,BG] of the den state across chunks (SRAM bounded by BK) and writes the PRE-update Zb
    snapshot the backward needs. inter uses the state BEFORE this chunk's update (causal); partials
    over feature-blocks sum via atomic_add. USE_G pre-scales by e^a; the carry decays by e^{Lam}."""
    b = tl.program_id(0); sb = tl.program_id(1); d0 = tl.program_id(2)
    offs_t = tl.arange(0, BT)
    offs_d = d0 * BK + tl.arange(0, BK); dmask = offs_d < dqk
    offs_c = sb * BG + tl.arange(0, BG); cmask = offs_c < nc
    offs_g = tl.arange(0, BG)
    Zd = tl.zeros([BK, BG], dtype=tl.float32)
    for t in range(NCH):
        rows = t * BT + offs_t; rmask = rows < L
        qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
        tl.store(Zb_ptr + b*szb_b + sb*szb_n + t*szb_t + offs_d[:, None]*szb_d + offs_g[None, :]*szb_c,
                 Zd, mask=dmask[:, None])                                        # PRE-update snapshot
        if USE_G:
            ldc = tl.load(ld_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
            a = tl.cumsum(ldc, axis=0)
            inter = tl.exp(a) * tl.dot(qc, Zd.to(qc.dtype))
        else:
            inter = tl.dot(qc, Zd.to(qc.dtype))
        tl.atomic_add(d_ptr + b*sd_b + rows[:, None]*sd_l + offs_c[None, :]*sd_c, inter, mask=rmask[:, None] & cmask[None, :])
        kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        if USE_G:
            Lam = tl.sum(tl.where(offs_t[:, None] == (BT - 1), a, 0.0), axis=0)
            w_end = wgc * tl.exp(Lam[None, :] - a)
            Zd = tl.exp(Lam)[None, :] * Zd + tl.dot(tl.trans(kc), w_end.to(kc.dtype))
        else:
            Zd += tl.dot(tl.trans(kc), wgc)


@triton.autotune(configs=_AT_CFGS, key=_DEN_KEY, **autotune_cache_kwargs)
@triton.jit
def _den_bwd_scan(q_ptr, gd_ptr, dZa_ptr, L, dqk, nc,
                  sq_b, sq_l, sq_d, sg_b, sg_l, sg_c,
                  szb_b, szb_n, szb_t, szb_d, szb_c,
                  BT: tl.constexpr, BD: tl.constexpr, BG: tl.constexpr, NCH: tl.constexpr):
    b = tl.program_id(0)
    sb = tl.program_id(1)
    d0 = tl.program_id(2)                        # feature-block: this program owns dZ rows [d0*BD:]
    offs_t = tl.arange(0, BT)
    offs_d = d0 * BD + tl.arange(0, BD)
    offs_c = sb * BG + tl.arange(0, BG)
    offs_g = tl.arange(0, BG)
    dmask = offs_d < dqk
    cmask = offs_c < nc
    dZ = tl.zeros([BD, BG], dtype=tl.float32)
    for ti in range(NCH):
        t = NCH - 1 - ti
        rows = t * BT + offs_t
        rmask = rows < L
        qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]
                     * sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        gdc = tl.load(gd_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]
                      * sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
        tl.store(dZa_ptr + b*szb_b + sb*szb_n + t*szb_t + offs_d[:, None]*szb_d + offs_g[None, :]*szb_c,
                 dZ, mask=dmask[:, None])
        dZ += tl.dot(tl.trans(qc), gdc)


@triton.autotune(configs=_BWD_CFGS, key=_DEN_KEY,
                 prune_configs_by={'early_config_prune': _prune_bwd_bd}, **autotune_cache_kwargs)
@triton.jit
def _den_grad(q_ptr, k_ptr, wg_ptr, gd_ptr, Zb_ptr, dZa_ptr, dq_ptr, dk_ptr, dw_ptr,
              L, dqk: tl.constexpr, nc,
              sq_b, sq_l, sq_d, sg_b, sg_l, sg_c,
              szb_b, szb_n, szb_t, szb_d, szb_c,
              sdq_b, sdq_n, sdq_l, sdq_d, sdw_b, sdw_l, sdw_c,
              BT: tl.constexpr, BD: tl.constexpr, BG: tl.constexpr, NCH: tl.constexpr):
    # D-tiled: dq/dk are feature-indexed (written per BD-block); dw needs the full content gram G +
    # KdZ=Σ_d kc·dZa, accumulated over the BD-block loop ([BT,BT] and [BT,BG] — bounded, independent
    # of dqk). dqk: tl.constexpr so ND is compile-time and the loop unrolls (ND==1 → straight-line,
    # no spill). BD is an autotune knob; ND follows from dqk (the perf lesson — see the main grads).
    ND = tl.cdiv(dqk, BD)
    b = tl.program_id(0)
    sb = tl.program_id(1)
    t = tl.program_id(2)
    offs_t = tl.arange(0, BT)
    offs_c = sb * BG + tl.arange(0, BG)
    offs_g = tl.arange(0, BG)
    cmask = offs_c < nc
    rows = t * BT + offs_t
    rmask = rows < L
    wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
    gdc = tl.load(gd_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
    caus = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
    P = tl.dot(gdc, tl.trans(wgc))                                   # [BT,BT]   (d-independent)
    Pc = P * caus
    G = tl.zeros([BT, BT], dtype=tl.float32)
    KdZ = tl.zeros([BT, BG], dtype=tl.float32)
    for d0b in range(ND):
        offs_d = d0b * BD + tl.arange(0, BD)
        dmask = offs_d < dqk
        qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        Zb = tl.load(Zb_ptr + b*szb_b + sb*szb_n + t*szb_t + offs_d[:, None]
                     * szb_d + offs_g[None, :]*szb_c, mask=dmask[:, None], other=0.0)
        dZa = tl.load(dZa_ptr + b*szb_b + sb*szb_n + t*szb_t +
                      offs_d[:, None]*szb_d + offs_g[None, :]*szb_c, mask=dmask[:, None], other=0.0)
        dq_d = tl.dot(Pc.to(kc.dtype), kc) + tl.dot(gdc, tl.trans(Zb.to(gdc.dtype)))
        dk_d = tl.dot(tl.trans(Pc).to(qc.dtype), qc) + tl.dot(wgc, tl.trans(dZa.to(wgc.dtype)))
        tl.store(dq_ptr + b*sdq_b + sb*sdq_n + rows[:, None]*sdq_l +
                 offs_d[None, :]*sdq_d, dq_d, mask=rmask[:, None] & dmask[None, :])
        tl.store(dk_ptr + b*sdq_b + sb*sdq_n + rows[:, None]*sdq_l +
                 offs_d[None, :]*sdq_d, dk_d, mask=rmask[:, None] & dmask[None, :])
        G += tl.dot(qc, tl.trans(kc))
        KdZ += tl.dot(kc, dZa.to(kc.dtype))
    dw = tl.dot(tl.trans(G * caus).to(gdc.dtype), gdc) + KdZ
    tl.store(dw_ptr + b*sdw_b + rows[:, None]*sdw_l + offs_c[None, :]*sdw_c, dw, mask=rmask[:, None] & cmask[None, :])


class _DenFn(torch.autograd.Function):
    """Per-state denominator d[i,c] on folded [BH,L,*] tensors, Triton fwd + chunk-parallel bwd."""
    @staticmethod
    def forward(ctx, q, k, wg, chunk, BG):
        B, L, dqk = q.shape
        nc = wg.shape[-1]
        BD = max(16, triton.next_power_of_2(dqk))
        BK = min(64, BD)                       # exact-width blocks for dqk<=64; D-tiled beyond
        ND = triton.cdiv(dqk, BK)
        NB = triton.cdiv(nc, BG)
        NCH = triton.cdiv(L, chunk)
        q, k, wg = q.contiguous(), k.contiguous(), wg.contiguous()
        d = torch.zeros(B, L, nc, device=q.device, dtype=torch.float32)   # atomic_add target
        Zb = torch.empty(B, NB, NCH, BD, BG, device=q.device, dtype=torch.float32)
        sq = (q.stride(0), q.stride(1), q.stride(2))
        sg = (wg.stride(0), wg.stride(1), wg.stride(2))
        sd = (d.stride(0), d.stride(1), d.stride(2))
        sZ = (Zb.stride(0), Zb.stride(1), Zb.stride(2), Zb.stride(3), Zb.stride(4))
        _den_fwd_intra[(B, NB, NCH)](q, k, wg, wg, d, L, dqk, nc, *sq, *sg, *sd,
                                     USE_G=False, BT=chunk, BK=BK, BG=BG, ND=ND)
        _den_fwd_inter[(B, NB, ND)](q, k, wg, wg, d, Zb, L, dqk, nc, *sq, *sg, *sd, *sZ,
                                    USE_G=False, BT=chunk, BK=BK, BG=BG, NCH=NCH)
        ctx.save_for_backward(q, k, wg, Zb)
        ctx.meta = (chunk, BG, BD, NB, NCH)
        return d

    @staticmethod
    def backward(ctx, gd):
        q, k, wg, Zb = ctx.saved_tensors
        chunk, BG, BD, NB, NCH = ctx.meta
        B, L, dqk = q.shape
        nc = wg.shape[-1]
        gd = gd.contiguous().to(q.dtype)   # match input dtype (tl.dot requires same-dtype operands; fp32 accum regardless)
        dZa = torch.empty_like(Zb)
        # Scan is d-parallel (register carry, own feature block; Zb/dZa indexed by absolute feature
        # row so this blocking is independent of the grad's autotuned BD). Grad BD is autotuned.
        BK_scan = min(64, BD)
        ND_scan = triton.cdiv(dqk, BK_scan)
        _den_bwd_scan[(B, NB, ND_scan)](q, gd, dZa, L, dqk, nc,
                                        q.stride(0), q.stride(1), q.stride(2), gd.stride(0), gd.stride(1), gd.stride(2),
                                        dZa.stride(0), dZa.stride(1), dZa.stride(2), dZa.stride(3), dZa.stride(4),
                                        BT=chunk, BD=BK_scan, BG=BG, NCH=NCH)
        dq = torch.empty(B, NB, L, dqk, device=q.device, dtype=torch.float32)
        dk = torch.empty_like(dq)
        dw = torch.empty(B, L, nc, device=q.device, dtype=torch.float32)
        _den_grad[(B, NB, NCH)](q, k, wg, gd, Zb, dZa, dq, dk, dw, L, dqk, nc,
                                q.stride(0), q.stride(1), q.stride(2), gd.stride(0), gd.stride(1), gd.stride(2),
                                Zb.stride(0), Zb.stride(1), Zb.stride(2), Zb.stride(3), Zb.stride(4),
                                dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(
                                    3), dw.stride(0), dw.stride(1), dw.stride(2),
                                BT=chunk, BG=BG, NCH=NCH)

        def cast(t): return t.to(q.dtype)
        return cast(dq.sum(1)), cast(dk.sum(1)), cast(dw), None, None


def rola_perstate_den_triton(q, k, w, chunk=None, BG=16):
    """Per-state denominator on folded [BH,L,*] tensors. Differentiable (Triton fwd + parallel bwd)."""
    chunk = _CHUNK if chunk is None else min(chunk, _CHUNK)
    return _DenFn.apply(q, k, w, chunk, BG)


@triton.autotune(configs=_AT_CFGS, key=_DEN_KEY, **autotune_cache_kwargs)
@triton.jit
def _den_gla_bwd_scan(q_ptr, gd_ptr, ld_ptr, dZa_ptr, L, dqk, nc,
                      sq_b, sq_l, sq_d, sg_b, sg_l, sg_c,
                      szb_b, szb_n, szb_t, szb_d, szb_c,
                      BT: tl.constexpr, BD: tl.constexpr, BG: tl.constexpr, NCH: tl.constexpr):
    """Reverse scan for the DECAYED den: dZa[t] = adjoint of chunk t's carry increment
    (sum over later chunks, decayed). dZ_t = e^{Lam_t} dZ_{t+1} + q^T (e^a ∘ gd)."""
    b = tl.program_id(0)
    sb = tl.program_id(1)
    d0 = tl.program_id(2)                        # feature-block: this program owns dZ rows [d0*BD:]
    offs_t = tl.arange(0, BT)
    offs_d = d0 * BD + tl.arange(0, BD)
    offs_c = sb * BG + tl.arange(0, BG)
    offs_g = tl.arange(0, BG)
    dmask = offs_d < dqk
    cmask = offs_c < nc
    dZ = tl.zeros([BD, BG], dtype=tl.float32)
    for ti in range(NCH):
        t = NCH - 1 - ti
        rows = t * BT + offs_t
        rmask = rows < L
        qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]
                     * sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        gdc = tl.load(gd_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]
                      * sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
        ldc = tl.load(ld_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]
                      * sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
        a = tl.cumsum(ldc, axis=0)
        tl.store(dZa_ptr + b*szb_b + sb*szb_n + t*szb_t + offs_d[:, None]*szb_d + offs_g[None, :]*szb_c,
                 dZ, mask=dmask[:, None])
        Lam = tl.sum(tl.where(offs_t[:, None] == (BT - 1), a, 0.0), axis=0)
        gt = gdc * tl.exp(a)
        dZ = tl.exp(Lam)[None, :] * dZ + tl.dot(tl.trans(qc), gt.to(qc.dtype))


@triton.autotune(configs=_BWD_CFGS, key=_DEN_KEY,
                 prune_configs_by={'early_config_prune': _prune_bwd_bd}, **autotune_cache_kwargs)
@triton.jit
def _den_gla_grad(q_ptr, k_ptr, wg_ptr, ld_ptr, gd_ptr, Zb_ptr, dZa_ptr,
                  dq_ptr, dk_ptr, dw_ptr, dld_ptr, L, dqk: tl.constexpr, nc,
                  sq_b, sq_l, sq_d, sg_b, sg_l, sg_c,
                  szb_b, szb_n, szb_t, szb_d, szb_c,
                  sdq_b, sdq_n, sdq_l, sdq_d, sdw_b, sdw_l, sdw_c,
                  BT: tl.constexpr, BD: tl.constexpr, BG: tl.constexpr, NCH: tl.constexpr):
    """Parallel per-chunk grads for the decayed den, incl. dld assembled IN-KERNEL:
    da from the three appearances of a (output scale e^a, intra w·e^{-a}, carry w·e^{Lam-a}),
    dLam from the carry decay (e^{Lam}·Σ Zb∘dZa, the main kernel's dLam trick) + the carry
    writes, folded into da's last row; dld = reverse cumsum of da (tot − cumsum + da)."""
    # D-tiled: dq/dk feature-indexed (per BD-block); the FOUR feature-dependent quantities
    # G[BT,BT], KdZ=Σ_d kc·dZa, QZ=Σ_d qc·Zb, ZdZ=Σ_d(Zb∘dZa) [over the D axis] are accumulated
    # across BD-blocks and consumed AFTER the loop (dch uses QZ; dLam uses ZdZ; both feed dld).
    # dqk: tl.constexpr → ND compile-time, loop unrolls (ND==1 collapses to straight-line, no spill).
    ND = tl.cdiv(dqk, BD)
    b = tl.program_id(0)
    sb = tl.program_id(1)
    t = tl.program_id(2)
    offs_t = tl.arange(0, BT)
    offs_c = sb * BG + tl.arange(0, BG)
    offs_g = tl.arange(0, BG)
    cmask = offs_c < nc
    rows = t * BT + offs_t
    rmask = rows < L
    wgc = tl.load(wg_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
    ldc = tl.load(ld_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
    gdc = tl.load(gd_ptr + b*sg_b + rows[:, None]*sg_l + offs_c[None, :]*sg_c, mask=rmask[:, None] & cmask[None, :], other=0.0)
    a = tl.cumsum(ldc, axis=0)
    Lam = tl.sum(tl.where(offs_t[:, None] == (BT - 1), a, 0.0), axis=0)
    ea = tl.exp(a)
    wt = wgc * tl.exp(-a)
    w_end = wgc * tl.exp(Lam[None, :] - a)
    caus = (offs_t[:, None] >= offs_t[None, :]) & rmask[:, None] & rmask[None, :]
    gt = gdc * ea
    P = tl.dot(gt, tl.trans(wt))                                     # [BT,BT]   (d-independent)
    Pc = P * caus
    G = tl.zeros([BT, BT], dtype=tl.float32)
    KdZ = tl.zeros([BT, BG], dtype=tl.float32)
    QZ = tl.zeros([BT, BG], dtype=tl.float32)
    ZdZ = tl.zeros([BG], dtype=tl.float32)
    for d0b in range(ND):
        offs_d = d0b * BD + tl.arange(0, BD)
        dmask = offs_d < dqk
        qc = tl.load(q_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        kc = tl.load(k_ptr + b*sq_b + rows[:, None]*sq_l + offs_d[None, :]*sq_d, mask=rmask[:, None] & dmask[None, :], other=0.0)
        Zb = tl.load(Zb_ptr + b*szb_b + sb*szb_n + t*szb_t + offs_d[:, None]
                     * szb_d + offs_g[None, :]*szb_c, mask=dmask[:, None], other=0.0)
        dZa = tl.load(dZa_ptr + b*szb_b + sb*szb_n + t*szb_t +
                      offs_d[:, None]*szb_d + offs_g[None, :]*szb_c, mask=dmask[:, None], other=0.0)
        dq_d = tl.dot(Pc.to(kc.dtype), kc) + tl.dot(gt, tl.trans(Zb))
        dk_d = tl.dot(tl.trans(Pc).to(qc.dtype), qc) + tl.dot(w_end, tl.trans(dZa))
        tl.store(dq_ptr + b*sdq_b + sb*sdq_n + rows[:, None]*sdq_l +
                 offs_d[None, :]*sdq_d, dq_d, mask=rmask[:, None] & dmask[None, :])
        tl.store(dk_ptr + b*sdq_b + sb*sdq_n + rows[:, None]*sdq_l +
                 offs_d[None, :]*sdq_d, dk_d, mask=rmask[:, None] & dmask[None, :])
        G += tl.dot(qc, tl.trans(kc))
        KdZ += tl.dot(kc, dZa.to(kc.dtype))                         # [BT,BG]   Σ_d kc·dZa
        QZ += tl.dot(qc, Zb.to(qc.dtype))                          # [BT,BG]   Σ_d qc·Zb
        ZdZ += tl.sum(Zb * dZa, axis=0)                            # [BG]      Σ_d (Zb∘dZa)
    GTg = tl.dot(tl.trans(G * caus), gt)                             # [BT(j),BG]
    dw = tl.exp(-a) * GTg + tl.exp(Lam[None, :] - a) * KdZ
    dch = ea * (tl.dot(G * caus, wt) + QZ)                           # recomputed chunk output
    da = gdc * dch - wt * GTg - w_end * KdZ
    dLam = tl.sum(w_end * KdZ, axis=0) + tl.exp(Lam) * ZdZ
    da += tl.where(offs_t[:, None] == (BT - 1), dLam[None, :], 0.0)
    s = tl.cumsum(da, axis=0)
    dld = tl.sum(da, axis=0)[None, :] - s + da                       # reverse cumsum
    tl.store(dw_ptr + b*sdw_b + rows[:, None]*sdw_l + offs_c[None, :]*sdw_c, dw, mask=rmask[:, None] & cmask[None, :])
    tl.store(dld_ptr + b*sdw_b + rows[:, None]*sdw_l + offs_c[None, :]*sdw_c, dld, mask=rmask[:, None] & cmask[None, :])


class _DenGLAFn(torch.autograd.Function):
    """Per-state denominator under per-state log-decay: d[i,c] = Σ_{j≤i} (φq_i·φk_j) w_j^c e^{Λ_ic−Λ_jc}.
    Forward: _den_fwd_intra+_den_fwd_inter USE_G=True. Backward: dedicated decayed den kernels at [BD,BG] state
    scale, value-free (_den_gla_bwd_scan + _den_gla_grad with in-kernel dld assembly) — the
    same cost class as the additive den, not a second full GLA backward."""
    @staticmethod
    def forward(ctx, q, k, wg, ld, chunk, BG):
        B, L, dqk = q.shape
        nc = wg.shape[-1]
        BD = max(16, triton.next_power_of_2(dqk))
        BK = min(64, BD)                       # exact-width blocks for dqk<=64; D-tiled beyond
        ND = triton.cdiv(dqk, BK)
        NB = triton.cdiv(nc, BG)
        NCH = triton.cdiv(L, chunk)
        ld = ld.clamp(min=_GLA_FLOOR)
        q, k, wg, ld = q.contiguous(), k.contiguous(), wg.contiguous(), ld.contiguous()
        d = torch.zeros(B, L, nc, device=q.device, dtype=torch.float32)   # atomic_add target
        Zb = torch.empty(B, NB, NCH, BD, BG, device=q.device, dtype=torch.float32)
        sq = (q.stride(0), q.stride(1), q.stride(2))
        sg = (wg.stride(0), wg.stride(1), wg.stride(2))
        sd = (d.stride(0), d.stride(1), d.stride(2))
        sZ = (Zb.stride(0), Zb.stride(1), Zb.stride(2), Zb.stride(3), Zb.stride(4))
        _den_fwd_intra[(B, NB, NCH)](q, k, wg, ld, d, L, dqk, nc, *sq, *sg, *sd,
                                     USE_G=True, BT=chunk, BK=BK, BG=BG, ND=ND)
        _den_fwd_inter[(B, NB, ND)](q, k, wg, ld, d, Zb, L, dqk, nc, *sq, *sg, *sd, *sZ,
                                    USE_G=True, BT=chunk, BK=BK, BG=BG, NCH=NCH)
        ctx.save_for_backward(q, k, wg, ld, Zb)
        ctx.meta = (chunk, BG, BD, NB, NCH)
        return d

    @staticmethod
    def backward(ctx, gd):
        q, k, wg, ld, Zb = ctx.saved_tensors
        chunk, BG, BD, NB, NCH = ctx.meta
        B, L, dqk = q.shape
        nc = wg.shape[-1]
        gd = gd.contiguous().to(q.dtype)
        dZa = torch.empty_like(Zb)
        # Scan is d-parallel (register carry, own feature block; e^Lam d-independent → row-separable).
        # Zb/dZa indexed by absolute feature row, independent of the grad's autotuned BD. Grad BD autotuned.
        BK_scan = min(64, BD)
        ND_scan = triton.cdiv(dqk, BK_scan)
        _den_gla_bwd_scan[(B, NB, ND_scan)](q, gd, ld, dZa, L, dqk, nc,
                                            q.stride(0), q.stride(1), q.stride(2), gd.stride(0), gd.stride(1), gd.stride(2),
                                            dZa.stride(0), dZa.stride(1), dZa.stride(2), dZa.stride(3), dZa.stride(4),
                                            BT=chunk, BD=BK_scan, BG=BG, NCH=NCH)
        dq = torch.empty(B, NB, L, dqk, device=q.device, dtype=torch.float32)
        dk = torch.empty_like(dq)
        dw = torch.empty(B, L, nc, device=q.device, dtype=torch.float32)
        dld = torch.empty_like(dw)
        _den_gla_grad[(B, NB, NCH)](q, k, wg, ld, gd, Zb, dZa, dq, dk, dw, dld, L, dqk, nc,
                                    q.stride(0), q.stride(1), q.stride(2), gd.stride(0), gd.stride(1), gd.stride(2),
                                    Zb.stride(0), Zb.stride(1), Zb.stride(2), Zb.stride(3), Zb.stride(4),
                                    dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(
                                        3), dw.stride(0), dw.stride(1), dw.stride(2),
                                    BT=chunk, BG=BG, NCH=NCH)

        def cast(t): return t.to(q.dtype)
        return cast(dq.sum(1)), cast(dk.sum(1)), cast(dw), dld.to(ld.dtype), None, None


def rola_perstate_den_gla_triton(q, k, w, ld, chunk=None, BG=16):
    """Per-state denominator under per-state log-decay ld:[BH,L,nc], folded tensors. Differentiable."""
    chunk = _CHUNK if chunk is None else min(chunk, _CHUNK)
    return _DenGLAFn.apply(q, k, w, ld, chunk, BG)
