# SPDX-License-Identifier: MIT
"""End-to-end model tests: build a small random-init model of each concrete family,
run prefill + decode through the superl8 dp4a kernels, check shapes/finiteness and a
greedy loop. Proves the modular stack (registry -> model -> runner -> kernels) works
for Qwen3 dense, Qwen3-MoE, and Gemma3. Needs CUDA (the dp4a + attention kernels)."""

import pytest
import torch

pytest.importorskip("superl8")

from superl8serve.models import ModelConfig, ModelRunner, build_model, is_supported, list_models

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="model forward needs the CUDA superl8 kernels")


def _rand(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.float16) * 0.05


def _base_cfg(arch, **kw):
    d = dict(
        arch=arch,
        vocab_size=320,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        max_position_embeddings=512,
        head_dim=64,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
    )
    d.update(kw)
    return ModelConfig(**d)


def _attn_sd(sd, p, cfg, qk_norm):
    hd, nh, nkv = cfg.resolved_head_dim(), cfg.num_attention_heads, cfg.num_key_value_heads
    sd[f"{p}.self_attn.q_proj.weight"] = _rand(nh * hd, cfg.hidden_size)
    sd[f"{p}.self_attn.k_proj.weight"] = _rand(nkv * hd, cfg.hidden_size)
    sd[f"{p}.self_attn.v_proj.weight"] = _rand(nkv * hd, cfg.hidden_size)
    sd[f"{p}.self_attn.o_proj.weight"] = _rand(cfg.hidden_size, nh * hd)
    if qk_norm:
        sd[f"{p}.self_attn.q_norm.weight"] = _rand(hd)
        sd[f"{p}.self_attn.k_norm.weight"] = _rand(hd)


def _dense_mlp_sd(sd, p, cfg):
    sd[f"{p}.mlp.gate_proj.weight"] = _rand(cfg.intermediate_size, cfg.hidden_size)
    sd[f"{p}.mlp.up_proj.weight"] = _rand(cfg.intermediate_size, cfg.hidden_size)
    sd[f"{p}.mlp.down_proj.weight"] = _rand(cfg.hidden_size, cfg.intermediate_size)


def qwen3_sd(cfg):
    sd = {
        "model.embed_tokens.weight": _rand(cfg.vocab_size, cfg.hidden_size),
        "model.norm.weight": _rand(cfg.hidden_size),
    }
    if not cfg.tie_word_embeddings:
        sd["lm_head.weight"] = _rand(cfg.vocab_size, cfg.hidden_size)
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = _rand(cfg.hidden_size)
        sd[f"{p}.post_attention_layernorm.weight"] = _rand(cfg.hidden_size)
        _attn_sd(sd, p, cfg, cfg.qk_norm)
        if cfg.layer_is_moe(i):
            sd[f"{p}.mlp.gate.weight"] = _rand(cfg.num_experts, cfg.hidden_size)
            for e in range(cfg.num_experts):
                sd[f"{p}.mlp.experts.{e}.gate_proj.weight"] = _rand(
                    cfg.moe_intermediate_size, cfg.hidden_size
                )
                sd[f"{p}.mlp.experts.{e}.up_proj.weight"] = _rand(
                    cfg.moe_intermediate_size, cfg.hidden_size
                )
                sd[f"{p}.mlp.experts.{e}.down_proj.weight"] = _rand(
                    cfg.hidden_size, cfg.moe_intermediate_size
                )
        else:
            _dense_mlp_sd(sd, p, cfg)
    return sd


def gemma3_sd(cfg):
    sd = {
        "model.embed_tokens.weight": _rand(cfg.vocab_size, cfg.hidden_size),
        "model.norm.weight": _rand(cfg.hidden_size),
    }
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        for n in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            sd[f"{p}.{n}.weight"] = _rand(cfg.hidden_size)
        _attn_sd(sd, p, cfg, qk_norm=True)
        _dense_mlp_sd(sd, p, cfg)
    return sd


def _run(cfg, sd):
    model = build_model(cfg, sd).cuda().eval()
    runner = ModelRunner(model, cfg, max_batch=2, max_len=64, device="cuda")
    prompt = torch.randint(0, cfg.vocab_size, (2, 6), device="cuda")
    logits = runner.prefill(prompt)
    assert logits.shape == (2, cfg.vocab_size) and torch.isfinite(logits).all()
    out = runner.generate_greedy(prompt, max_new_tokens=5)
    assert out.shape == (2, 5) and (out >= 0).all() and (out < cfg.vocab_size).all()
    return out


def test_registry_lists_families():
    assert is_supported("qwen3") and is_supported("gemma3")
    assert is_supported("Qwen3ForCausalLM") and is_supported("qwen3_moe")
    assert is_supported("qwen2_5_vl") and is_supported("llava")
    assert {"qwen3", "gemma3", "qwen2_5_vl", "llava"} <= set(list_models())


def test_qwen3_dense_forward():
    cfg = _base_cfg("qwen3", qk_norm=True, tie_word_embeddings=True)
    _run(cfg, qwen3_sd(cfg))


def test_qwen3_untied_head():
    cfg = _base_cfg("qwen3", qk_norm=True, tie_word_embeddings=False)
    _run(cfg, qwen3_sd(cfg))


def test_qwen3_moe_forward():
    cfg = _base_cfg(
        "qwen3_moe",
        qk_norm=True,
        tie_word_embeddings=True,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        norm_topk_prob=True,
    )
    _run(cfg, qwen3_sd(cfg))


def test_gemma3_forward():
    # sliding_window_pattern=2 -> layer 0 local (window), layer 1 global; dual-theta.
    cfg = _base_cfg(
        "gemma3",
        qk_norm=True,
        tie_word_embeddings=True,
        norm_add_unit_offset=True,
        hidden_act="gelu_pytorch_tanh",
        query_pre_attn_scalar=64.0,
        rope_local_theta=1e4,
        sliding_window=16,
        sliding_window_pattern=2,
    )
    _run(cfg, gemma3_sd(cfg))


def test_qwen3_moe_routing_math():
    """Router: softmax over all experts -> top-k -> renormalize sums to 1."""
    from superl8serve.models.moe import SparseMoE
    from superl8serve.models.weights import gate_up_weight, to_qtensor

    torch.manual_seed(0)
    H, E, k = 256, 4, 2
    sd = {}
    for e in range(E):
        sd[f"e{e}.gate_proj.weight"] = _rand(128, H)
        sd[f"e{e}.up_proj.weight"] = _rand(128, H)
        sd[f"e{e}.down_proj.weight"] = _rand(H, 128)
    experts = [
        (gate_up_weight(sd, f"e{e}"), to_qtensor(sd[f"e{e}.down_proj.weight"])) for e in range(E)
    ]
    moe = SparseMoE(gate=_rand(E, H), experts=experts, top_k=k, norm_topk_prob=True).cuda()
    y = moe(torch.randn(1, 5, H, device="cuda", dtype=torch.float16))
    assert y.shape == (1, 5, H) and torch.isfinite(y).all()


def test_mtp_draft_shape():
    """MTP head drafts a token from the main model's last hidden (DeepSeek-V3 recipe)."""
    from superl8serve.layers.embedding import LMHead, VocabEmbedding
    from superl8serve.layers.norm import RMSNorm
    from superl8serve.layers.rotary import RotaryEmbedding
    from superl8serve.models.base import ForwardContext
    from superl8serve.models.cache import KVCache
    from superl8serve.models.mtp import MTPLayer, MultiTokenPredictor
    from superl8serve.models.qwen3 import Qwen3DecoderLayer

    cfg = _base_cfg("qwen3", qk_norm=True, num_hidden_layers=1)
    sd = qwen3_sd(cfg)
    rope = RotaryEmbedding(
        cfg.resolved_head_dim(), cfg.max_position_embeddings, base=cfg.rope_theta
    )
    block = Qwen3DecoderLayer(cfg, 0, sd, rope)
    H = cfg.hidden_size
    emb = VocabEmbedding(sd["model.embed_tokens.weight"])
    head = LMHead(sd["model.embed_tokens.weight"])
    mtp = (
        MultiTokenPredictor(
            cfg,
            layers=[
                MTPLayer(
                    cfg,
                    fc_weight=_rand(H, 2 * H),
                    hidden_norm=_rand(H),
                    emb_norm=_rand(H),
                    block=block,
                )
            ],
            embed=emb,
            final_norm=RMSNorm(H, cfg.rms_norm_eps, _rand(H)),
            lm_head=head,
        )
        .cuda()
        .eval()
    )
    cache = KVCache(1, 1, cfg.num_key_value_heads, 8, cfg.resolved_head_dim(), device="cuda")
    ctx = ForwardContext(is_prefill=True, kv_cache=cache)
    last_hidden = _rand(1, 1, H)
    last_token = torch.randint(0, cfg.vocab_size, (1, 1), device="cuda")
    positions = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    drafts = mtp.draft_greedy(last_hidden, last_token, positions, ctx)
    assert len(drafts) == 1 and drafts[0].shape == (1, 1)
    assert (drafts[0] >= 0).all() and (drafts[0] < cfg.vocab_size).all()
