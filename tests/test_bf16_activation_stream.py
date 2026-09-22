# SPDX-License-Identifier: MIT
"""bf16 activation-stream guard for the ATTENTION path (issue #260).

`test_activation_overflow.py` pinned the dp4a LINEAR wrapper (serve #256). This file
covers the other half of the same fp16-truncation bug class: the residual/attention
stream itself. fp16 tops out at 65504; a bf16-native model (Gemma3, torch_dtype
bfloat16) carries "massive activation" residual channels reaching ~1e4-1e7, so its
forward must run the residual stream in bf16 or it overflows to inf->NaN a few layers
in (measured: Gemma-3 int8 forward went non-finite ~layer 6 when the stream was fp16).

The seam is the activation dtype, seeded by `VocabEmbedding` and carried unchanged
through norm/RoPE/attention/MLP. `ModelConfig.act_dtype()` maps the checkpoint's
`torch_dtype` to that seed. Softmax/LSE/norm reductions stay fp32 regardless.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("superl8")

import superl8
from superl8 import QTensor

from superl8serve.layers.embedding import VocabEmbedding
from superl8serve.models.config import ModelConfig

CUDA = torch.cuda.is_available()

_MASSIVE = 1e6  # well past fp16's 65504 ceiling; comfortably inside bf16's ~3.4e38 range


def test_qwen35_gated_attention_casts_only_kv_cache_write_to_fp16():
    """The sm70 paged-KV writer is fp16-only; the bf16 compute stream stays bf16."""
    from superl8serve.layers.gated_gqa_attention import _paged_cache_kv

    k = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    v = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    cached_k, cached_v = _paged_cache_kv(k, v)

    assert cached_k.dtype is torch.float16 and cached_v.dtype is torch.float16
    assert k.dtype is torch.bfloat16 and v.dtype is torch.bfloat16


def test_qwen35_gated_attention_restores_bf16_after_paged_decode():
    """The paged kernel returns fp16, but the residual stream is gate/model dtype."""
    from superl8serve.layers.gated_gqa_attention import GatedGQAAttention

    attn = object.__new__(GatedGQAAttention)
    torch.nn.Module.__init__(attn)
    attn.nh, attn.hd = 2, 8
    attn.o_proj = torch.nn.Identity()
    out = torch.randn(1, 1, 2, 8, dtype=torch.float16)
    gate = torch.randn(1, 1, 2, 8, dtype=torch.bfloat16)

    projected = attn._gate_and_project(out, gate, 1, 1)

    assert projected.dtype is torch.bfloat16


@pytest.mark.skipif(not CUDA, reason="bf16 attention fallback is CUDA-only")
def test_qwen35_bf16_varlen_fallback_preserves_stream_dtype():
    from superl8serve.layers.gated_gqa_attention import _bf16_varlen_attention

    q = torch.randn(5, 4, 32, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(5, 2, 32, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(5, 2, 32, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)

    out = _bf16_varlen_attention(q, k, v, cu, scale=32**-0.5)

    assert out.shape == q.shape
    assert out.dtype is torch.bfloat16 and torch.isfinite(out).all()


# ── config: torch_dtype -> activation dtype ──────────────────────────────────


def _cfg(torch_dtype: str) -> ModelConfig:
    return ModelConfig(
        arch="gemma3_text",
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=64,
        head_dim=16,
        torch_dtype=torch_dtype,
    )


def test_act_dtype_bf16_for_bf16_native_checkpoint():
    assert _cfg("bfloat16").act_dtype() is torch.bfloat16
    assert _cfg("torch.bfloat16").act_dtype() is torch.bfloat16


def test_act_dtype_fp16_default_and_for_fp16_checkpoint():
    assert _cfg("float16").act_dtype() is torch.float16
    # default (unspecified torch_dtype) stays fp16 -- fp16-native models are unchanged
    assert (
        ModelConfig(
            arch="qwen3",
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=64,
        ).act_dtype()
        is torch.float16
    )


def test_from_hf_captures_torch_dtype():
    cfg = ModelConfig.from_hf(
        {
            "model_type": "gemma3_text",
            "vocab_size": 64,
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "intermediate_size": 64,
            "torch_dtype": "bfloat16",
        }
    )
    assert cfg.torch_dtype == "bfloat16"
    assert cfg.act_dtype() is torch.bfloat16


# ── VocabEmbedding seeds the stream in the requested dtype (the loud negative) ─


def _int8_embed_table(vocab: int, hidden: int, dequant_mag: float) -> QTensor:
    """int8 embedding table whose per-row dequant reaches ~dequant_mag (so a looked-up
    row carries a massive-activation-sized value that overflows fp16 but not bf16)."""
    data = torch.full((vocab, hidden), 100, dtype=torch.int8)  # near int8 max (127)
    scale = torch.full((vocab,), dequant_mag / 100.0, dtype=torch.float32)  # 100*scale=mag
    return QTensor(data, scale, scheme="per_row_i8")


def test_vocab_embedding_bf16_keeps_massive_value_finite():
    """A dequant that lands at ~1e6 stays finite as bf16 but is inf as fp16 -- proving
    the out_dtype seed is load-bearing, not cosmetic (loud negative test)."""
    qt = _int8_embed_table(8, 16, _MASSIVE)
    ids = torch.zeros(1, 4, dtype=torch.long)

    emb_bf = VocabEmbedding(qt, out_dtype=torch.bfloat16)
    h_bf = emb_bf(ids)
    assert h_bf.dtype is torch.bfloat16
    assert torch.isfinite(h_bf).all(), "bf16 seed lost the massive activation to inf"
    assert h_bf.abs().max().item() > 1e5

    emb_fp = VocabEmbedding(qt, out_dtype=torch.float16)  # the pre-fix behaviour
    h_fp = emb_fp(ids)
    assert not torch.isfinite(h_fp).all(), "fp16 seed unexpectedly held 1e6 (bug not reproduced)"


def test_vocab_embedding_defaults_fp16():
    """Backwards-compatible default: no out_dtype -> fp16 seed (fp16-native models)."""
    w = torch.randn(8, 16, dtype=torch.float16)
    assert VocabEmbedding(w)(torch.zeros(1, 3, dtype=torch.long)).dtype is torch.float16


# ── the #260 crux: the int8 attention kernel accepts bf16 and stays finite ────


@pytest.mark.skipif(not CUDA, reason="attn_int8_fwd is CUDA-only")
def test_attn_int8_fwd_bf16_massive_v_stays_finite():
    """superl8.attn_int8_fwd with bf16 q/k/v carrying ~1e6 values returns finite bf16.
    This is the attention half of #260: serve feeds attention bf16 q/k/v, and the
    fp16-PV path stores the output in v's dtype -- bf16 here, so no fp16 overflow."""
    B, H, S, D = 1, 4, 64, 128
    torch.manual_seed(0)
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16) * _MASSIVE
    assert torch.isfinite(v).all()  # bf16 holds 1e6

    out = superl8.attn_int8_fwd(q, k, v, causal=True, scale=D**-0.5)

    assert out.dtype is torch.bfloat16, "attention downcast the bf16 stream to fp16"
    assert torch.isfinite(out).all(), "massive bf16 V overflowed in the attention kernel"


# ── end-to-end: a bf16-native Gemma3 model forward stays finite ──────────────


@pytest.mark.skipif(not CUDA, reason="model forward needs the CUDA superl8 kernels")
def test_gemma3_forward_runs_bf16_stream_and_stays_finite():
    """A Gemma3 (torch_dtype bfloat16) small model builds a bf16 residual stream
    end-to-end (embedding seed -> hidden), and the forward is finite. Pins that the
    whole serve stack (norm/RoPE/attn/MLP) carries bf16 without downcasting."""
    from superl8serve.models import ModelRunner, build_model
    from tests.test_models import gemma3_sd  # reuse the small-model state-dict builder

    cfg = ModelConfig(
        arch="gemma3",
        vocab_size=320,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        head_dim=64,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        rope_local_theta=1e4,
        sliding_window=64,
        sliding_window_pattern=2,
        query_pre_attn_scalar=64,
        norm_add_unit_offset=True,
        embed_scale=16.0,
        hidden_act="gelu_pytorch_tanh",
        torch_dtype="bfloat16",
    )
    model = build_model(cfg, gemma3_sd(cfg)).cuda().eval()
    assert model.model.embed_tokens.out_dtype is torch.bfloat16

    runner = ModelRunner(model, cfg, max_batch=1, max_len=32, device="cuda")
    ids = torch.arange(2, 18, device="cuda").unsqueeze(0)
    with torch.inference_mode():
        # capture the residual-stream dtype the model actually runs in
        seen = {}
        h0 = model.model.embed_tokens(ids)
        seen["stream"] = h0.dtype
        logits = runner.prefill(ids)

    assert seen["stream"] is torch.bfloat16, "Gemma3 residual stream was not bf16"
    assert torch.isfinite(logits).all(), "Gemma3 bf16 forward produced non-finite logits"
