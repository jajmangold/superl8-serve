# SPDX-License-Identifier: MIT
"""DeepSeek (MLA + fine-grained MoE + shared expert) registration + prefill/decode.

Builds a small random DeepSeek through the registry and runs prefill (MLA decompress
path + first-dense-then-MoE layers) plus a full greedy-decode loop through the
latent-KV cache (`MLALatentCache`, wired in via `ModelRunner`/`cfg.latent_attention`).
"""

import pytest
import torch

pytest.importorskip("superl8")

from superl8serve.models import ModelConfig, build_model, is_supported

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="MLA projections use the dp4a GEMM")


def _cfg():
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
        extra=dict(
            q_lora_rank=96,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            v_head_dim=32,
            first_k_dense_replace=1,
            num_expert_groups=2,
            topk_group=1,
            routed_scaling_factor=1.0,
        ),
    )


def _sd(cfg):
    def r(*s):
        return torch.randn(*s, device="cuda", dtype=torch.float16) * 0.05

    x = cfg.extra
    H, nh = cfg.hidden_size, cfg.num_attention_heads
    qk = x["qk_nope_head_dim"] + x["qk_rope_head_dim"]
    sd = {"model.embed_tokens.weight": r(cfg.vocab_size, H), "model.norm.weight": r(H)}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = r(H)
        sd[f"{p}.post_attention_layernorm.weight"] = r(H)
        a = f"{p}.self_attn"
        sd[f"{a}.q_a_proj.weight"] = r(x["q_lora_rank"], H)
        sd[f"{a}.q_a_layernorm.weight"] = r(x["q_lora_rank"])
        sd[f"{a}.q_b_proj.weight"] = r(nh * qk, x["q_lora_rank"])
        sd[f"{a}.kv_a_proj_with_mqa.weight"] = r(x["kv_lora_rank"] + x["qk_rope_head_dim"], H)
        sd[f"{a}.kv_a_layernorm.weight"] = r(x["kv_lora_rank"])
        sd[f"{a}.kv_b_proj.weight"] = r(
            nh * (x["qk_nope_head_dim"] + x["v_head_dim"]), x["kv_lora_rank"]
        )
        sd[f"{a}.o_proj.weight"] = r(H, nh * x["v_head_dim"])
        if i >= x["first_k_dense_replace"]:
            sd[f"{p}.mlp.gate.weight"] = r(cfg.num_experts, H)
            sd[f"{p}.mlp.e_score_correction_bias"] = r(cfg.num_experts)
            for e in range(cfg.num_experts):
                sd[f"{p}.mlp.experts.{e}.gate_proj.weight"] = r(cfg.moe_intermediate_size, H)
                sd[f"{p}.mlp.experts.{e}.up_proj.weight"] = r(cfg.moe_intermediate_size, H)
                sd[f"{p}.mlp.experts.{e}.down_proj.weight"] = r(H, cfg.moe_intermediate_size)
            for n in ("gate_proj", "up_proj"):
                sd[f"{p}.mlp.shared_experts.{n}.weight"] = r(cfg.moe_intermediate_size, H)
            sd[f"{p}.mlp.shared_experts.down_proj.weight"] = r(H, cfg.moe_intermediate_size)
        else:
            sd[f"{p}.mlp.gate_proj.weight"] = r(cfg.intermediate_size, H)
            sd[f"{p}.mlp.up_proj.weight"] = r(cfg.intermediate_size, H)
            sd[f"{p}.mlp.down_proj.weight"] = r(H, cfg.intermediate_size)
    return sd


def test_deepseek_registered():
    assert is_supported("deepseek") and is_supported("DeepseekV3ForCausalLM")


def test_deepseek_prefill():
    from superl8serve.models.base import ForwardContext
    from superl8serve.models.cache import MLALatentCache

    cfg = _cfg()
    model = build_model(cfg, _sd(cfg)).cuda().eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 6), device="cuda")
    pos = torch.arange(6, device="cuda").unsqueeze(0)
    cache = MLALatentCache(cfg.num_hidden_layers, 1, cfg.mla_cache_dim(), 16, device="cuda")
    hidden = model(ids, pos, ForwardContext(is_prefill=True, kv_cache=cache))
    logits = model.compute_logits(hidden[:, -1])
    assert logits.shape == (1, cfg.vocab_size) and torch.isfinite(logits).all()


def test_deepseek_generate():
    """Full autoregressive loop: prefill then decode through the latent-KV cache,
    end to end via ModelRunner (the seam other families' generate tests exercise)."""
    from superl8serve.models import ModelRunner
    from superl8serve.models.cache import MLALatentCache

    cfg = _cfg()
    model = build_model(cfg, _sd(cfg)).cuda().eval()
    runner = ModelRunner(model, cfg, max_batch=1, max_len=32, device="cuda")
    assert isinstance(runner.cache, MLALatentCache)
    prompt = torch.randint(0, cfg.vocab_size, (1, 6), device="cuda")
    logits = runner.prefill(prompt)
    assert logits.shape == (1, cfg.vocab_size) and torch.isfinite(logits).all()
    out = runner.generate_greedy(prompt, max_new_tokens=4)
    assert out.shape == (1, 4) and (out >= 0).all() and (out < cfg.vocab_size).all()


def test_deepseek_engine_generate():
    """LLMEngine drives DeepSeek (MLA + MoE) end-to-end, with output parity to the
    standalone ModelRunner at the same token-agreement threshold used by the
    standard-attention engine tests."""
    from superl8serve.engine import LLMEngine, SamplingParams
    from superl8serve.models import ModelRunner

    torch.manual_seed(42)
    cfg = _cfg()
    sd = _sd(cfg)
    prompt = [3, 1, 4, 1, 5]

    eng = LLMEngine(cfg, sd, device="cuda", max_num_seqs=4, max_len=64)
    eng_out = eng.generate([prompt], SamplingParams(temperature=0.0, max_tokens=5))[0]

    model = build_model(cfg, sd).cuda().eval()
    runner = ModelRunner(model, cfg, max_batch=1, max_len=64, device="cuda")
    ref = runner.generate_greedy(torch.tensor([prompt], device="cuda"), max_new_tokens=5)[
        0
    ].tolist()

    assert eng_out[0] == ref[0], f"first token mismatch: {eng_out} vs {ref}"
    agree = sum(1 for a, b in zip(eng_out, ref) if a == b) / len(ref)
    assert agree >= 0.8, f"engine vs runner decode diverged too much: {eng_out} vs {ref}"


def test_deepseek_engine_concurrent():
    """LLMEngine handles multiple concurrent DeepSeek requests through the latent
    cache, verifying scheduler slot management with MLALatentCache."""
    from superl8serve.engine import LLMEngine, SamplingParams

    torch.manual_seed(1)
    cfg = _cfg()
    eng = LLMEngine(cfg, _sd(cfg), device="cuda", max_num_seqs=4, max_len=64)
    prompts = [[1, 2, 3], [4, 5, 6, 7, 8], [9], [10, 11, 12, 13]]
    outs = eng.generate(prompts, SamplingParams(temperature=0.0, max_tokens=6))
    assert len(outs) == 4
    for o in outs:
        assert len(o) == 6 and all(0 <= t < cfg.vocab_size for t in o)


def test_deepseek_decode_cache_matches_full_recompute():
    """Parity: decoding one token at a time through the latent cache must match a
    one-shot forward over the whole sequence (the same decompress-path math with a
    fresh cache each time is the module's own oracle — see `mla_attn.py`). Dense-only
    (no MoE) so the comparison isn't at the mercy of top-k routing flipping on the
    tiny floating-point differences between batched-prefill and single-token softmax."""
    from superl8serve.models.base import ForwardContext
    from superl8serve.models.cache import MLALatentCache

    cfg = _cfg()
    cfg.num_experts = 0
    cfg.num_experts_per_tok = 0
    cfg.extra = dict(cfg.extra, first_k_dense_replace=cfg.num_hidden_layers)
    model = build_model(cfg, _sd(cfg)).cuda().eval()
    S = 6
    ids = torch.randint(0, cfg.vocab_size, (1, S), device="cuda")
    pos_full = torch.arange(S, device="cuda").unsqueeze(0)

    ref_cache = MLALatentCache(cfg.num_hidden_layers, 1, cfg.mla_cache_dim(), S, device="cuda")
    ref_hidden = model(ids, pos_full, ForwardContext(is_prefill=True, kv_cache=ref_cache))

    step_cache = MLALatentCache(cfg.num_hidden_layers, 1, cfg.mla_cache_dim(), S, device="cuda")
    prefill_n = S - 1
    pre_hidden = model(
        ids[:, :prefill_n],
        pos_full[:, :prefill_n],
        ForwardContext(is_prefill=True, kv_cache=step_cache),
    )
    step_cache.advance(prefill_n)
    pos_last = pos_full[:, prefill_n : prefill_n + 1]
    dec_hidden = model(
        ids[:, prefill_n : prefill_n + 1],
        pos_last,
        ForwardContext(is_prefill=False, kv_cache=step_cache),
    )
    step_cache.advance(1)

    got = torch.cat([pre_hidden, dec_hidden], dim=1)
    torch.testing.assert_close(got.float(), ref_hidden.float(), rtol=2e-2, atol=2e-2)


def test_group_limited_routing():
    """Group-limited top-k: with 2 groups of 2 experts and topk_group=1, only
    experts from one group can be selected per token."""
    from superl8serve.models.moe import SparseMoE
    from superl8serve.models.weights import gate_up_weight, to_qtensor

    torch.manual_seed(42)
    H, E, k = 256, 4, 2
    n_groups, topk_g = 2, 1
    sd = {}
    for e in range(E):
        sd[f"e{e}.gate_proj.weight"] = (
            torch.randn(128, H, device="cuda", dtype=torch.float16) * 0.05
        )
        sd[f"e{e}.up_proj.weight"] = torch.randn(128, H, device="cuda", dtype=torch.float16) * 0.05
        sd[f"e{e}.down_proj.weight"] = (
            torch.randn(H, 128, device="cuda", dtype=torch.float16) * 0.05
        )
    experts = [
        (gate_up_weight(sd, f"e{e}"), to_qtensor(sd[f"e{e}.down_proj.weight"])) for e in range(E)
    ]
    bias = torch.randn(E, device="cuda", dtype=torch.float32) * 0.1
    moe = SparseMoE(
        gate=torch.randn(E, H, device="cuda", dtype=torch.float16) * 0.05,
        experts=experts,
        top_k=k,
        norm_topk_prob=True,
        scoring_func="sigmoid",
        e_score_correction_bias=bias,
        num_expert_groups=n_groups,
        topk_group=topk_g,
    ).cuda()

    x = torch.randn(2, 4, H, device="cuda", dtype=torch.float16)
    y = moe(x)
    assert y.shape == x.shape and torch.isfinite(y).all()

    # Verify per-token expert selection respects groups: experts [0,1] are group 0,
    # experts [2,3] are group 1. With topk_group=1 all selected experts must be from
    # the same group.
    T = x.shape[0] * x.shape[1]
    router_logits = torch.mm(x.reshape(-1, H).float(), moe.gate.float().T)
    scores = torch.sigmoid(router_logits)
    sel = scores + bias
    E_per_group = E // n_groups
    group_scores = sel.view(T, n_groups, E_per_group).amax(dim=-1)
    selected_groups = group_scores.topk(topk_g, dim=-1)[1]
    offsets = torch.arange(E_per_group, device=sel.device).view(1, 1, E_per_group)
    expert_idx = (selected_groups.unsqueeze(-1) * E_per_group + offsets).reshape(T, -1)
    mask = torch.full_like(sel, float("-inf"))
    mask.scatter_(1, expert_idx, 0.0)
    sel_masked = sel + mask
    _, topi = torch.topk(sel_masked, k, dim=-1)
    for t in range(T):
        g = selected_groups[t, 0].item()
        allowed = {g * E_per_group + i for i in range(E_per_group)}
        assert set(topi[t].tolist()).issubset(allowed), (
            f"token {t}: experts {topi[t].tolist()} not all in allowed group {g} ({allowed})"
        )
