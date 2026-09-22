# SPDX-License-Identifier: MIT
"""Multi-head Latent Attention (MLA) — Track-1 fp16 port.

Checks the weight-absorption identity the Track-2 int8 kernel depends on
(q_nope·k_nope == (W_UK^T q_nope)·c_KV), that the MLAAttention block runs
end-to-end (down/up LoRA projections on dp4a + decoupled RoPE + causal softmax),
and that decode through the latent-KV cache matches the uncached one-shot forward."""

import pytest
import torch

from superl8serve.layers.mla_attn import absorb_qk_equiv

CUDA = torch.cuda.is_available()

H, NH = 256, 4
Q_LORA, KV_LORA = 96, 64
QK_NOPE, QK_ROPE, VD = 32, 16, 32


def test_absorb_identity():
    """The nope score equals its latent-space form — the crux of MLA's absorb path."""
    torch.manual_seed(0)
    nope, kv_lora = 128, 512
    q_nope = torch.randn(4, nope)
    w_uk = torch.randn(nope, kv_lora) * 0.02  # W_UK^T : nope -> kv_lora
    c_kv = torch.randn(4, kv_lora) * 0.1
    k_nope = c_kv @ w_uk.t()  # k_nope = W_UK c_KV
    direct = (q_nope * k_nope).sum(-1)  # q_nope · k_nope
    absorbed = absorb_qk_equiv(q_nope, w_uk, c_kv)
    torch.testing.assert_close(direct, absorbed, rtol=1e-4, atol=1e-3)


def _build_mla_block(*, use_int8_absorb: bool = False):
    """Build an MLAAttention block with deterministic random quantized weights.

    Always seeds to 0 before generating weights, so weights are identical
    regardless of the `use_int8_absorb` flag — toggling the flag is the only
    difference between blocks."""
    from superl8 import QTensor

    from superl8serve.layers.mla_attn import MLAAttention
    from superl8serve.models.config import ModelConfig

    torch.manual_seed(0)
    cfg = ModelConfig(
        arch="deepseek",
        vocab_size=64,
        hidden_size=H,
        num_hidden_layers=1,
        num_attention_heads=NH,
        num_key_value_heads=NH,
        intermediate_size=128,
        max_position_embeddings=64,
        head_dim=QK_NOPE + QK_ROPE,
    )

    def qt(o, i):
        w = torch.randn(o, i, device="cuda", dtype=torch.float16) * 0.05
        s = w.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 127
        return QTensor(
            torch.round(w / s).clamp_(-127, 127).to(torch.int8),
            s.squeeze(-1).float(),
            scheme="per_row_i8",
        )

    def ones(n):
        return torch.ones(n, device="cuda", dtype=torch.float16)

    return MLAAttention(
        cfg,
        q_a_proj=qt(Q_LORA, H),
        q_a_norm=ones(Q_LORA),
        q_b_proj=qt(NH * (QK_NOPE + QK_ROPE), Q_LORA),
        kv_a_proj=qt(KV_LORA + QK_ROPE, H),
        kv_a_norm=ones(KV_LORA),
        kv_b_proj=qt(NH * (QK_NOPE + VD), KV_LORA),
        o_proj=qt(H, NH * VD),
        num_heads=NH,
        q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA,
        qk_nope_head_dim=QK_NOPE,
        qk_rope_head_dim=QK_ROPE,
        v_head_dim=VD,
        use_int8_absorb=use_int8_absorb,
    ).cuda()


@pytest.mark.skipif(not CUDA, reason="LoRA projections use the dp4a GEMM")
def test_mla_block_runs_and_matches_reference():
    blk = _build_mla_block()
    x = torch.randn(1, 7, H, device="cuda", dtype=torch.float16)
    y = blk(x, None, None, 0)
    assert y.shape == (1, 7, H) and torch.isfinite(y).all()
    # determinism
    assert torch.equal(y, blk(x, None, None, 0))


@pytest.mark.skipif(not CUDA, reason="LoRA projections use the dp4a GEMM")
def test_mla_decode_cache_matches_uncached_recompute():
    """Cached one-token-at-a-time decode must match the one-shot (no-cache) forward
    over the whole sequence — that uncached forward is MLA's own oracle (it ports the
    HF decompress path directly, see module docstring)."""
    from superl8serve.models.base import ForwardContext
    from superl8serve.models.cache import MLALatentCache

    blk = _build_mla_block()
    S = 7
    x = torch.randn(1, S, H, device="cuda", dtype=torch.float16)
    ref = blk(x, None, None, 0)

    cache = MLALatentCache(1, 1, KV_LORA + QK_ROPE, S, device="cuda")
    pos_full = torch.arange(S, device="cuda").unsqueeze(0)
    prefill_n = S - 1
    pre = blk(
        x[:, :prefill_n],
        pos_full[:, :prefill_n],
        ForwardContext(is_prefill=True, kv_cache=cache),
        0,
    )
    cache.advance(prefill_n)
    dec = blk(
        x[:, prefill_n:],
        pos_full[:, prefill_n:],
        ForwardContext(is_prefill=False, kv_cache=cache),
        0,
    )
    cache.advance(1)

    got = torch.cat([pre, dec], dim=1)
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)


def _build_mla_block_int8():
    """Thin wrapper: _build_mla_block with use_int8_absorb=True."""
    return _build_mla_block(use_int8_absorb=True)


@pytest.mark.skipif(not CUDA, reason="MLA projections use the dp4a GEMM")
def test_mla_int8_absorb_runs_and_has_correct_shape():
    """The int8 absorb decode path returns [B, 1, hidden] (already includes o_proj)."""
    from superl8serve.models.base import ForwardContext
    from superl8serve.models.cache import MLALatentCache

    blk = _build_mla_block_int8()
    S, prefill_n = 7, 6
    x = torch.randn(1, S, H, device="cuda", dtype=torch.float16)
    pos = torch.arange(S, device="cuda").unsqueeze(0)
    cache = MLALatentCache(1, 1, KV_LORA + QK_ROPE, S, device="cuda")

    pre = blk(
        x[:, :prefill_n], pos[:, :prefill_n], ForwardContext(is_prefill=True, kv_cache=cache), 0
    )
    assert pre.shape == (1, prefill_n, H) and torch.isfinite(pre).all()
    cache.advance(prefill_n)
    dec = blk(
        x[:, prefill_n:], pos[:, prefill_n:], ForwardContext(is_prefill=False, kv_cache=cache), 0
    )
    assert dec.shape == (1, 1, H) and torch.isfinite(dec).all()


@pytest.mark.skipif(not CUDA, reason="MLA projections use the dp4a GEMM")
def test_mla_int8_absorb_match_fp16_decode():
    """Int8 absorb decode output has cos similarity >= 0.99 vs fp16 reference.

    Uses the SAME block instance, toggling use_int8_absorb between runs,
    so the weights are identical and only the attention path differs."""
    from superl8serve.models.base import ForwardContext
    from superl8serve.models.cache import MLALatentCache

    blk = _build_mla_block(use_int8_absorb=False)

    S, prefill_n = 7, 6
    pos = torch.arange(S, device="cuda").unsqueeze(0)
    x = torch.randn(1, S, H, device="cuda", dtype=torch.float16)

    # Run fp16 reference
    cache_fp16 = MLALatentCache(1, 1, KV_LORA + QK_ROPE, S, device="cuda")
    blk.use_int8_absorb = False
    pre_fp16 = blk(
        x[:, :prefill_n],
        pos[:, :prefill_n],
        ForwardContext(is_prefill=True, kv_cache=cache_fp16),
        0,
    )
    cache_fp16.advance(prefill_n)
    dec_fp16 = blk(
        x[:, prefill_n:],
        pos[:, prefill_n:],
        ForwardContext(is_prefill=False, kv_cache=cache_fp16),
        0,
    )

    # Run int8 absorb on the SAME block, separate cache
    cache_i8 = MLALatentCache(1, 1, KV_LORA + QK_ROPE, S, device="cuda")
    blk.use_int8_absorb = True
    pre_i8 = blk(
        x[:, :prefill_n], pos[:, :prefill_n], ForwardContext(is_prefill=True, kv_cache=cache_i8), 0
    )
    cache_i8.advance(prefill_n)
    dec_i8 = blk(
        x[:, prefill_n:], pos[:, prefill_n:], ForwardContext(is_prefill=False, kv_cache=cache_i8), 0
    )

    pre_cos = torch.nn.functional.cosine_similarity(
        pre_fp16.float().flatten(), pre_i8.float().flatten(), dim=0
    )
    dec_cos = torch.nn.functional.cosine_similarity(
        dec_fp16.float().flatten(), dec_i8.float().flatten(), dim=0
    )
    assert pre_cos.item() >= 0.99, f"prefill cos={pre_cos.item():.6f} < 0.99"
    assert dec_cos.item() >= 0.99, f"decode cos={dec_cos.item():.6f} < 0.99"


@pytest.mark.skipif(not CUDA, reason="MLA projections use the dp4a GEMM")
def test_mla_int8_absorb_full_model_decode_match():
    """Full DeepSeek model decode with int8 absorb matches fp16 reference (cos >= 0.99)."""
    from superl8serve.models import ModelConfig, build_model
    from superl8serve.models.base import ForwardContext
    from superl8serve.models.cache import MLALatentCache

    torch.manual_seed(42)

    def _cfg(**kw):
        extra = dict(
            q_lora_rank=96,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            v_head_dim=32,
            first_k_dense_replace=1,
            num_expert_groups=2,
            topk_group=1,
            routed_scaling_factor=1.0,
            use_int8_absorb=True,
        )
        extra.update(kw)
        return ModelConfig(
            arch="deepseek",
            vocab_size=128,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=256,
            max_position_embeddings=64,
            head_dim=48,
            tie_word_embeddings=True,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=128,
            rms_norm_eps=1e-6,
            latent_attention=True,
            extra=extra,
        )

    def _sd(cfg):
        def r(*s):
            return torch.randn(*s, device="cuda", dtype=torch.float16) * 0.05

        x = cfg.extra
        H_dim, nh = cfg.hidden_size, cfg.num_attention_heads
        qk = x["qk_nope_head_dim"] + x["qk_rope_head_dim"]
        sd = {"model.embed_tokens.weight": r(cfg.vocab_size, H_dim), "model.norm.weight": r(H_dim)}
        for i in range(cfg.num_hidden_layers):
            p = f"model.layers.{i}"
            sd[f"{p}.input_layernorm.weight"] = r(H_dim)
            sd[f"{p}.post_attention_layernorm.weight"] = r(H_dim)
            a = f"{p}.self_attn"
            sd[f"{a}.q_a_proj.weight"] = r(x["q_lora_rank"], H_dim)
            sd[f"{a}.q_a_layernorm.weight"] = r(x["q_lora_rank"])
            sd[f"{a}.q_b_proj.weight"] = r(nh * qk, x["q_lora_rank"])
            sd[f"{a}.kv_a_proj_with_mqa.weight"] = r(
                x["kv_lora_rank"] + x["qk_rope_head_dim"], H_dim
            )
            sd[f"{a}.kv_a_layernorm.weight"] = r(x["kv_lora_rank"])
            sd[f"{a}.kv_b_proj.weight"] = r(
                nh * (x["qk_nope_head_dim"] + x["v_head_dim"]), x["kv_lora_rank"]
            )
            sd[f"{a}.o_proj.weight"] = r(H_dim, nh * x["v_head_dim"])
            if i >= x["first_k_dense_replace"]:
                sd[f"{p}.mlp.gate.weight"] = r(cfg.num_experts, H_dim)
                sd[f"{p}.mlp.e_score_correction_bias"] = r(cfg.num_experts)
                for e in range(cfg.num_experts):
                    sd[f"{p}.mlp.experts.{e}.gate_proj.weight"] = r(
                        cfg.moe_intermediate_size, H_dim
                    )
                    sd[f"{p}.mlp.experts.{e}.up_proj.weight"] = r(cfg.moe_intermediate_size, H_dim)
                    sd[f"{p}.mlp.experts.{e}.down_proj.weight"] = r(
                        H_dim, cfg.moe_intermediate_size
                    )
                for n in ("gate_proj", "up_proj"):
                    sd[f"{p}.mlp.shared_experts.{n}.weight"] = r(cfg.moe_intermediate_size, H_dim)
                sd[f"{p}.mlp.shared_experts.down_proj.weight"] = r(H_dim, cfg.moe_intermediate_size)
            else:
                sd[f"{p}.mlp.gate_proj.weight"] = r(cfg.intermediate_size, H_dim)
                sd[f"{p}.mlp.up_proj.weight"] = r(cfg.intermediate_size, H_dim)
                sd[f"{p}.mlp.down_proj.weight"] = r(H_dim, cfg.intermediate_size)
        return sd

    # Use dense-only for deterministic comparison (no MoE routing randomness)
    cfg_ref = _cfg(
        use_int8_absorb=False, first_k_dense_replace=2, num_experts=0, num_experts_per_tok=0
    )
    cfg_i8 = _cfg(
        use_int8_absorb=True, first_k_dense_replace=2, num_experts=0, num_experts_per_tok=0
    )

    sd = _sd(cfg_ref)
    model_fp16 = build_model(cfg_ref, sd).cuda().eval()
    model_i8 = build_model(cfg_i8, sd).cuda().eval()

    S, prefill_n = 6, 5
    ids = torch.randint(0, cfg_ref.vocab_size, (1, S), device="cuda")
    pos = torch.arange(S, device="cuda").unsqueeze(0)
    latent_dim = cfg_ref.mla_cache_dim()

    # fp16 reference
    cache_fp16 = MLALatentCache(cfg_ref.num_hidden_layers, 1, latent_dim, S, device="cuda")
    _ = model_fp16(
        ids[:, :prefill_n], pos[:, :prefill_n], ForwardContext(is_prefill=True, kv_cache=cache_fp16)
    )
    cache_fp16.advance(prefill_n)
    dec_fp16 = model_fp16(
        ids[:, prefill_n:],
        pos[:, prefill_n:],
        ForwardContext(is_prefill=False, kv_cache=cache_fp16),
    )
    logits_fp16 = model_fp16.compute_logits(dec_fp16[:, -1])

    # int8 absorb
    cache_i8 = MLALatentCache(cfg_i8.num_hidden_layers, 1, latent_dim, S, device="cuda")
    _ = model_i8(
        ids[:, :prefill_n], pos[:, :prefill_n], ForwardContext(is_prefill=True, kv_cache=cache_i8)
    )
    cache_i8.advance(prefill_n)
    dec_i8 = model_i8(
        ids[:, prefill_n:], pos[:, prefill_n:], ForwardContext(is_prefill=False, kv_cache=cache_i8)
    )
    logits_i8 = model_i8.compute_logits(dec_i8[:, -1])

    cos = torch.nn.functional.cosine_similarity(
        logits_fp16.float().flatten(), logits_i8.float().flatten(), dim=0
    )
    assert cos.item() >= 0.99, f"full model decode cos={cos.item():.6f} < 0.99"

    tok_fp16 = logits_fp16.argmax(-1).item()
    tok_i8 = logits_i8.argmax(-1).item()
    assert tok_fp16 == tok_i8, f"token mismatch: fp16={tok_fp16}, int8={tok_i8}"
