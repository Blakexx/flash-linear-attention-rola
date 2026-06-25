# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import pytest
import torch

from fla_rola.models.rola import RoLAConfig, RoLAForCausalLM
from fla_rola.utils import assert_close, device


@pytest.mark.parametrize('rola_instance', ['rola-rla-kappa-asym', 'rola-gla-kappa-asym'])
def test_generation(rola_instance):
    """Generation == forward: a chunked prefill hands its recurrent state to token-by-token recurrent
    decode (KV-cache) and reproduces the full teacher-forced forward. No padding — `chunk_rola_routed` is
    fixed-length; the varlen/unpad path lands with `chunk_rola_routed` cu_seqlens support (follow-up)."""
    if device != 'cuda':
        pytest.skip('RoLA Triton kernels require CUDA')
    torch.manual_seed(42)
    cfg = RoLAConfig(
        vocab_size=256, hidden_size=128, num_hidden_layers=2, num_heads=4,
        states_per_head=8, d_qk=16, d_v=16, max_position_embeddings=512,
        rola_instance=rola_instance, use_cache=True,
    )
    model = RoLAForCausalLM(cfg).to(device).eval()
    B, T, prefill = 1, 160, 96            # prefill>64 -> chunk; each decode step L=1 -> fused_recurrent
    ids = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    with torch.no_grad():
        ref = model(input_ids=ids, use_cache=False).logits
        out = model(input_ids=ids[:, :prefill], use_cache=True, past_key_values=None)
        logits, past = [out.logits], out.past_key_values
        for j in range(prefill, T):
            out = model(input_ids=ids[:, j:j + 1], use_cache=True, past_key_values=past)
            logits.append(out.logits)
            past = out.past_key_values
        gen = torch.cat(logits, 1)
    assert_close('generation==forward', ref[:, prefill:], gen[:, prefill:], 2e-3)
