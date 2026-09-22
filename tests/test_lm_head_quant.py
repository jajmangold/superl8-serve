# SPDX-License-Identifier: MIT
"""Tied-embedding models must quantize the LM head too (int8 dp4a), not run it as
an fp16 matmul.

On this fleet the fp16 tensor cores are firmware-gimped (~6.9 TFLOP/s); dp4a is
~6.7x faster (AGENTS.md). Profiling a tied Qwen3-0.6B decode step showed the LM
head's logits GEMM was the single biggest kernel (~9% of GPU time, one CUTLASS
`cutlass_70_wmma_tensorop_f16` call) because the tied case skipped `to_qtensor`
and fell to `F.linear` on the crippled tensor cores. Untied models already
quantize the head; this makes tied models consistent.
"""
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("superl8")

from superl8 import QTensor
from superl8serve.layers.linear import LinearW8A8
from superl8serve.models import build_model

from tests.test_models import _base_cfg, qwen3_sd  # reuse the tiny-model fixtures

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="needs the CUDA superl8 kernels")


def _build_tied():
    cfg = _base_cfg("qwen3", qk_norm=True, tie_word_embeddings=True)
    sd = qwen3_sd(cfg)
    embed_w = sd["model.embed_tokens.weight"].clone()  # keep the fp16 reference
    model = build_model(cfg, sd)
    return cfg, model, embed_w


def test_tied_lm_head_is_quantized():
    _, model, _ = _build_tied()
    # dp4a path: LMHead.proj is a LinearW8A8 over a QTensor, NOT the fp16 fallback.
    assert model.lm_head.proj is not None, "tied LM head fell back to fp16 F.linear"
    assert model.lm_head._fp16_w is None
    assert isinstance(model.lm_head.proj, LinearW8A8)
    assert isinstance(model.lm_head.proj.weight, QTensor)


def test_tied_lm_head_logits_match_fp16():
    cfg, model, embed_w = _build_tied()
    hidden = torch.randn(4, cfg.hidden_size, device="cuda", dtype=torch.float16)
    q_logits = model.compute_logits(hidden).float()
    ref_logits = F.linear(hidden, embed_w).float()  # what the fp16 tied head produced
    cos = F.cosine_similarity(q_logits, ref_logits, dim=-1).mean().item()
    assert cos > 0.999, f"quantized-head logits diverged from fp16 (cos={cos:.5f})"
    # Greedy sampling must be unaffected: top-1 token agrees on every row.
    assert torch.equal(q_logits.argmax(-1), ref_logits.argmax(-1))
