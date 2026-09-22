# SPDX-License-Identifier: MIT
"""Registration + prefill for the additional families (qwen3_next hybrid, LFM2,
GLM, Hunyuan, MiniMax). Each builds through the registry and runs a prefill forward
with random weights — proving the modular assembly. Full autoregressive decode for
the linear/conv/lightning families needs recurrent-state caching (Track 1.5)."""

import pytest
import torch

pytest.importorskip("superl8")

from superl8serve.models import ModelConfig, ModelRunner, build_model, is_supported
from superl8serve.models.base import ForwardContext
from superl8serve.models.cache import KVCache, RecurrentStateCache

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="projections use the dp4a GEMM")


def _r(*s):
    return torch.randn(*s, device="cuda", dtype=torch.float16) * 0.05


def _prefill(cfg, sd):
    model = build_model(cfg, sd).cuda().eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 6), device="cuda")
    pos = torch.arange(6, device="cuda").unsqueeze(0)
    cache = KVCache(
        cfg.num_hidden_layers,
        1,
        cfg.num_key_value_heads,
        16,
        cfg.resolved_head_dim(),
        device="cuda",
    )
    h = model(ids, pos, ForwardContext(is_prefill=True, kv_cache=cache))
    logits = model.compute_logits(h[:, -1])
    assert logits.shape == (1, cfg.vocab_size) and torch.isfinite(logits).all()


def _moe_sd(sd, p, H, E, mi, shared_name=None):
    sd[f"{p}.mlp.gate.weight"] = _r(E, H)
    for e in range(E):
        sd[f"{p}.mlp.experts.{e}.gate_proj.weight"] = _r(mi, H)
        sd[f"{p}.mlp.experts.{e}.up_proj.weight"] = _r(mi, H)
        sd[f"{p}.mlp.experts.{e}.down_proj.weight"] = _r(H, mi)
    if shared_name:
        sd[f"{p}.mlp.{shared_name}.gate_proj.weight"] = _r(mi, H)
        sd[f"{p}.mlp.{shared_name}.up_proj.weight"] = _r(mi, H)
        sd[f"{p}.mlp.{shared_name}.down_proj.weight"] = _r(H, mi)


def _qwen3_next_hybrid_cfg_sd():
    x = dict(
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
    )
    cfg = ModelConfig(
        arch="qwen3_next",
        vocab_size=64,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        head_dim=32,
        qk_norm=True,
        tie_word_embeddings=True,
        num_experts=2,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        linear_attention=True,
        full_attention_interval=2,
        shared_expert_intermediate_size=64,
        extra=x,
    )
    H, nh, nkv, hd = 128, 4, 2, 32
    nk, nv, kd, vd = 2, 4, 16, 16
    qkv_lin = 2 * nk * kd + nv * vd
    sd = {"model.embed_tokens.weight": _r(cfg.vocab_size, H), "model.norm.weight": _r(H)}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = _r(H)
        sd[f"{p}.post_attention_layernorm.weight"] = _r(H)
        if (i + 1) % 2 == 0:  # full attn
            sd[f"{p}.self_attn.q_proj.weight"] = _r(nh * hd, H)
            sd[f"{p}.self_attn.k_proj.weight"] = _r(nkv * hd, H)
            sd[f"{p}.self_attn.v_proj.weight"] = _r(nkv * hd, H)
            sd[f"{p}.self_attn.o_proj.weight"] = _r(H, nh * hd)
            sd[f"{p}.self_attn.q_norm.weight"] = _r(hd)
            sd[f"{p}.self_attn.k_norm.weight"] = _r(hd)
        else:  # linear (DeltaNet)
            la = f"{p}.linear_attn"
            sd[f"{la}.qkv_proj.weight"] = _r(qkv_lin, H)
            sd[f"{la}.z_proj.weight"] = _r(nv * vd, H)
            sd[f"{la}.out_proj.weight"] = _r(H, nv * vd)
            sd[f"{la}.conv_weight"] = _r(qkv_lin, 4)
            sd[f"{la}.A_log"] = _r(nv).float()
            sd[f"{la}.dt_bias"] = _r(nv).float()
            sd[f"{la}.beta_proj.weight"] = _r(nv, H)
            sd[f"{la}.dt_proj.weight"] = _r(nv, H)
            sd[f"{la}.norm.weight"] = _r(vd)  # gated DeltaNet norm is PER-HEAD (head_v_dim)
        _moe_sd(sd, p, H, cfg.num_experts, cfg.moe_intermediate_size, shared_name="shared_expert")
    return cfg, sd


def test_qwen3_next_hybrid_prefill():
    assert is_supported("qwen3_next")
    cfg, sd = _qwen3_next_hybrid_cfg_sd()
    _prefill(cfg, sd)


def test_qwen3_next_moe_builder_route_weight_stationary():
    """`use_weight_stationary_moe` routes the Qwen3.6 MoE through
    WeightStationaryMoE (Phase 3) with no engine change."""
    import dataclasses

    cfg, sd = _qwen3_next_hybrid_cfg_sd()
    cfg = dataclasses.replace(cfg, use_weight_stationary_moe=True)
    model = build_model(cfg, sd).cuda().eval()
    moe = model.layers[0].mlp
    from superl8serve.models.moe import WeightStationaryMoE

    assert isinstance(moe, WeightStationaryMoE)
    x = torch.randn(2, 6, cfg.hidden_size, device="cuda", dtype=torch.float16)
    out = moe(x)
    assert out.shape == (2, 6, cfg.hidden_size) and torch.isfinite(out).all()
    # skip-empty exercised: at tiny batch most experts are inactive
    assert moe.active_expert_count <= cfg.num_experts


def test_qwen3_next_moe_builder_route_transport_all_local_parity():
    """An all-local `expert_to_gpu` map routes through TransportMoELayer and stays
    parity-gated vs the plain SparseMoE path (cos > 0.99, transport contract)."""
    from superl8serve.models.moe import SparseMoE
    from superl8serve.models.moe_transport import TransportMoELayer

    import dataclasses

    torch.manual_seed(0)
    cfg, sd = _qwen3_next_hybrid_cfg_sd()
    ref_model = build_model(cfg, sd).cuda().eval()
    ref_moe = ref_model.layers[0].mlp
    assert isinstance(ref_moe, SparseMoE)

    cfg = dataclasses.replace(
        cfg,
        use_weight_stationary_moe=False,
        expert_to_gpu={e: 0 for e in range(cfg.num_experts)},
        local_gpu=0,
    )
    t_model = build_model(cfg, sd).cuda().eval()
    t_moe = t_model.layers[0].mlp
    assert isinstance(t_moe, TransportMoELayer)
    assert t_moe.transport is not None and not t_moe.transport.remote_gpus

    x = torch.randn(4, 8, cfg.hidden_size, device="cuda", dtype=torch.float16)
    out_ref = ref_moe(x)
    out_t = t_moe(x)
    cos = torch.nn.functional.cosine_similarity(out_ref.flatten(), out_t.flatten(), dim=0)
    assert cos > 0.99, f"all-local transport drift vs SparseMoE: cos={cos}"


def test_qwen3_next_hybrid_decode_matches_teacher_forced():
    """Full generate (prefill + N decode steps) must match a single teacher-forced
    forward over the whole (prompt + generated) sequence. This is the property the
    recurrent-state cache (`S`) and causal-conv tail now preserve for the linear
    (DeltaNet) layers, and the per-layer `layer_idx` fix preserves for the KV cache
    on the interleaved full-attention layers — before both fixes, decode either
    recomputed each linear layer's state from zero every step (forgetting the
    causal-conv history too) or clobbered a shared KV-cache slot across layers."""
    cfg, sd = _qwen3_next_hybrid_cfg_sd()
    model = build_model(cfg, sd).cuda().eval()
    runner = ModelRunner(model, cfg, max_batch=1, max_len=32, device="cuda")
    prompt = torch.randint(0, cfg.vocab_size, (1, 5), device="cuda")
    gen = runner.generate_greedy(prompt, max_new_tokens=4)

    full_ids = torch.cat([prompt, gen], dim=1)
    pos = torch.arange(full_ids.shape[1], device="cuda").unsqueeze(0)
    ref_cache = KVCache(
        cfg.num_hidden_layers,
        1,
        cfg.num_key_value_heads,
        32,
        cfg.resolved_head_dim(),
        device="cuda",
    )
    hidden = model(full_ids, pos, ForwardContext(is_prefill=True, kv_cache=ref_cache))
    logits = model.compute_logits(hidden)
    ref_tokens = logits[:, prompt.shape[1] - 1 : -1].argmax(-1)
    assert torch.equal(ref_tokens, gen)


def _attn_sd(sd, a, cfg, qk=None, bias=False):
    hd, nh, nkv, H = (
        cfg.resolved_head_dim(),
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.hidden_size,
    )
    o_name = (
        "out_proj" if a.endswith("self_attn") and cfg.arch in ("lfm2", "lfm2_moe") else "o_proj"
    )
    sd[f"{a}.q_proj.weight"] = _r(nh * hd, H)
    sd[f"{a}.k_proj.weight"] = _r(nkv * hd, H)
    sd[f"{a}.v_proj.weight"] = _r(nkv * hd, H)
    sd[f"{a}.{o_name}.weight"] = _r(H, nh * hd)
    if bias:
        for x in "qkv":
            sd[f"{a}.{x}_proj.bias"] = _r(nh * hd if x == "q" else nkv * hd)
    if qk:
        sd[f"{a}.{qk[0]}.weight"] = _r(hd)
        sd[f"{a}.{qk[1]}.weight"] = _r(hd)


def test_lfm2_prefill():
    cfg = ModelConfig(
        arch="lfm2",
        vocab_size=64,
        hidden_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        max_position_embeddings=64,
        head_dim=32,
        qk_norm=True,
        tie_word_embeddings=True,
        rms_norm_eps=1e-5,
        extra=dict(full_attn_idxs=[1], conv_L_cache=3),
    )
    H = cfg.hidden_size
    sd = {"model.embed_tokens.weight": _r(cfg.vocab_size, H), "model.norm.weight": _r(H)}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.operator_norm.weight"] = _r(H)
        sd[f"{p}.ffn_norm.weight"] = _r(H)
        for w, d in (("w1", cfg.intermediate_size), ("w3", cfg.intermediate_size), ("w2", H)):
            sd[f"{p}.feed_forward.{w}.weight"] = _r(d, H if w != "w2" else cfg.intermediate_size)
        if i == 1:  # attention layer
            _attn_sd(sd, f"{p}.self_attn", cfg, qk=("q_layernorm", "k_layernorm"))
        else:  # conv layer
            sd[f"{p}.conv.in_proj.weight"] = _r(3 * H, H)
            sd[f"{p}.conv.out_proj.weight"] = _r(H, H)
            sd[f"{p}.conv.conv.weight"] = _r(H, 1, 3)
    _prefill(cfg, sd)


def _lfm2_cfg_sd():
    cfg = ModelConfig(
        arch="lfm2",
        vocab_size=64,
        hidden_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        max_position_embeddings=64,
        head_dim=32,
        qk_norm=True,
        tie_word_embeddings=True,
        rms_norm_eps=1e-5,
        extra=dict(full_attn_idxs=[1], conv_L_cache=3),
    )
    H = cfg.hidden_size
    sd = {"model.embed_tokens.weight": _r(cfg.vocab_size, H), "model.norm.weight": _r(H)}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.operator_norm.weight"] = _r(H)
        sd[f"{p}.ffn_norm.weight"] = _r(H)
        for w, d in (("w1", cfg.intermediate_size), ("w3", cfg.intermediate_size), ("w2", H)):
            sd[f"{p}.feed_forward.{w}.weight"] = _r(d, H if w != "w2" else cfg.intermediate_size)
        if i == 1:  # attention layer
            _attn_sd(sd, f"{p}.self_attn", cfg, qk=("q_layernorm", "k_layernorm"))
        else:  # conv layer
            sd[f"{p}.conv.in_proj.weight"] = _r(3 * H, H)
            sd[f"{p}.conv.out_proj.weight"] = _r(H, H)
            sd[f"{p}.conv.conv.weight"] = _r(H, 1, 3)
    return cfg, sd


def test_lfm2_decode_matches_teacher_forced():
    """Full generate (prefill + N decode steps) must match a single teacher-forced
    forward over the whole (prompt + generated) sequence. This is the property the
    conv-tail state caching now preserves for ShortConv layers: decode reads the
    trailing `kernel-1` window from the previous step instead of zero-padding, so
    the stepwise output matches the batched prefill output exactly."""
    cfg, sd = _lfm2_cfg_sd()
    model = build_model(cfg, sd).cuda().eval()
    runner = ModelRunner(model, cfg, max_batch=1, max_len=32, device="cuda")
    prompt = torch.randint(0, cfg.vocab_size, (1, 5), device="cuda")
    gen = runner.generate_greedy(prompt, max_new_tokens=4)

    full_ids = torch.cat([prompt, gen], dim=1)
    pos = torch.arange(full_ids.shape[1], device="cuda").unsqueeze(0)
    ref_cache = KVCache(
        cfg.num_hidden_layers,
        1,
        cfg.num_key_value_heads,
        32,
        cfg.resolved_head_dim(),
        device="cuda",
    )
    ref_lin_cache = RecurrentStateCache()
    hidden = model(
        full_ids, pos, ForwardContext(is_prefill=True, kv_cache=ref_cache, lin_cache=ref_lin_cache)
    )
    logits = model.compute_logits(hidden)
    ref_tokens = logits[:, prompt.shape[1] - 1 : -1].argmax(-1)
    assert torch.equal(ref_tokens, gen)


def _lfm2_moe_cfg_sd():
    cfg = ModelConfig(
        arch="lfm2_moe",
        vocab_size=64,
        hidden_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,  # unused for MoE layers but kept for dense-layers compat
        max_position_embeddings=64,
        head_dim=32,
        qk_norm=True,
        tie_word_embeddings=True,
        rms_norm_eps=1e-5,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        extra=dict(full_attn_idxs=[1], conv_L_cache=3),
    )
    H, E, mi = cfg.hidden_size, cfg.num_experts, cfg.moe_intermediate_size
    sd = {"model.embed_tokens.weight": _r(cfg.vocab_size, H), "model.norm.weight": _r(H)}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.operator_norm.weight"] = _r(H)
        sd[f"{p}.ffn_norm.weight"] = _r(H)
        sd[f"{p}.feed_forward.router.weight"] = _r(E, H)
        sd[f"{p}.feed_forward.w1_experts.weight"] = _r(E * mi, H)
        sd[f"{p}.feed_forward.w3_experts.weight"] = _r(E * mi, H)
        sd[f"{p}.feed_forward.w2_experts.weight"] = _r(E * H, mi)
        if i == 1:
            _attn_sd(sd, f"{p}.self_attn", cfg, qk=("q_layernorm", "k_layernorm"))
        else:
            sd[f"{p}.conv.in_proj.weight"] = _r(3 * H, H)
            sd[f"{p}.conv.out_proj.weight"] = _r(H, H)
            sd[f"{p}.conv.conv.weight"] = _r(H, 1, 3)
    return cfg, sd


def test_lfm2_moe_prefill():
    assert is_supported("lfm2_moe")
    cfg, sd = _lfm2_moe_cfg_sd()
    _prefill(cfg, sd)


def test_lfm2_moe_decode_matches_teacher_forced():
    cfg, sd = _lfm2_moe_cfg_sd()
    model = build_model(cfg, sd).cuda().eval()
    runner = ModelRunner(model, cfg, max_batch=1, max_len=32, device="cuda")
    prompt = torch.randint(0, cfg.vocab_size, (1, 5), device="cuda")
    gen = runner.generate_greedy(prompt, max_new_tokens=4)

    full_ids = torch.cat([prompt, gen], dim=1)
    pos = torch.arange(full_ids.shape[1], device="cuda").unsqueeze(0)
    ref_cache = KVCache(
        cfg.num_hidden_layers,
        1,
        cfg.num_key_value_heads,
        32,
        cfg.resolved_head_dim(),
        device="cuda",
    )
    ref_lin_cache = RecurrentStateCache()
    hidden = model(
        full_ids, pos, ForwardContext(is_prefill=True, kv_cache=ref_cache, lin_cache=ref_lin_cache)
    )
    logits = model.compute_logits(hidden)
    ref_tokens = logits[:, prompt.shape[1] - 1 : -1].argmax(-1)
    assert torch.equal(ref_tokens, gen)


def test_glm_prefill():
    cfg = ModelConfig(
        arch="glm",
        vocab_size=64,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        max_position_embeddings=64,
        head_dim=32,
        qk_norm=True,
        partial_rotary_factor=0.5,
        tie_word_embeddings=False,
        rms_norm_eps=1e-5,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        extra=dict(first_k_dense_replace=1, routed_scaling_factor=2.5),
    )
    H = cfg.hidden_size
    sd = {
        "model.embed_tokens.weight": _r(cfg.vocab_size, H),
        "model.norm.weight": _r(H),
        "lm_head.weight": _r(cfg.vocab_size, H),
    }
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = _r(H)
        sd[f"{p}.post_attention_layernorm.weight"] = _r(H)
        _attn_sd(sd, f"{p}.self_attn", cfg, qk=("q_norm", "k_norm"), bias=True)
        if i >= 1:
            _moe_sd(
                sd, p, H, cfg.num_experts, cfg.moe_intermediate_size, shared_name="shared_experts"
            )
            sd[f"{p}.mlp.gate.e_score_correction_bias"] = _r(cfg.num_experts).float()
        else:
            for w, d in (
                ("gate_proj", cfg.intermediate_size),
                ("up_proj", cfg.intermediate_size),
                ("down_proj", H),
            ):
                sd[f"{p}.mlp.{w}.weight"] = _r(d, H if w != "down_proj" else cfg.intermediate_size)
    _prefill(cfg, sd)


def test_hunyuan_prefill():
    cfg = ModelConfig(
        arch="hunyuan",
        vocab_size=64,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        max_position_embeddings=64,
        head_dim=32,
        qk_norm=True,
        tie_word_embeddings=True,
        rms_norm_eps=1e-5,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
    )
    H = cfg.hidden_size
    sd = {"model.embed_tokens.weight": _r(cfg.vocab_size, H), "model.norm.weight": _r(H)}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = _r(H)
        sd[f"{p}.post_attention_layernorm.weight"] = _r(H)
        _attn_sd(sd, f"{p}.self_attn", cfg, qk=("query_layernorm", "key_layernorm"))
        _moe_sd(sd, p, H, cfg.num_experts, cfg.moe_intermediate_size, shared_name="shared_mlp")
        sd[f"{p}.mlp.gate.wg.weight"] = sd.pop(f"{p}.mlp.gate.weight")
    _prefill(cfg, sd)


def _minimax_cfg_sd():
    cfg = ModelConfig(
        arch="minimax",
        vocab_size=64,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=256,
        max_position_embeddings=64,
        head_dim=32,
        tie_word_embeddings=False,
        rms_norm_eps=1e-5,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        rope_theta=1e7,
        partial_rotary_factor=0.5,
        extra=dict(
            attn_type_list=[0, 1], layernorm_full_attention_alpha=1.0, layernorm_mlp_beta=1.0
        ),
    )
    H, hd, nh = cfg.hidden_size, cfg.resolved_head_dim(), cfg.num_attention_heads
    sd = {
        "model.embed_tokens.weight": _r(cfg.vocab_size, H),
        "model.norm.weight": _r(H),
        "lm_head.weight": _r(cfg.vocab_size, H),
    }
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = _r(H)
        sd[f"{p}.post_attention_layernorm.weight"] = _r(H)
        a = f"{p}.self_attn"
        if i == 0:  # lightning
            sd[f"{a}.qkv_proj.weight"] = _r(3 * nh * hd, H)
            sd[f"{a}.out_proj.weight"] = _r(H, nh * hd)
            sd[f"{a}.output_gate.weight"] = _r(nh * hd, H)
            sd[f"{a}.norm.weight"] = _r(nh * hd)
        else:  # softmax
            _attn_sd(sd, a, cfg)
        m = f"{p}.block_sparse_moe"
        sd[f"{m}.gate.weight"] = _r(cfg.num_experts, H)
        for e in range(cfg.num_experts):
            sd[f"{m}.experts.{e}.w1.weight"] = _r(cfg.moe_intermediate_size, H)
            sd[f"{m}.experts.{e}.w3.weight"] = _r(cfg.moe_intermediate_size, H)
            sd[f"{m}.experts.{e}.w2.weight"] = _r(H, cfg.moe_intermediate_size)
    return cfg, sd


def test_minimax_prefill():
    cfg, sd = _minimax_cfg_sd()
    _prefill(cfg, sd)


def test_minimax_lightning_decode_matches_teacher_forced():
    """Same decode-caching property as the qwen3_next hybrid test, for MiniMax's
    lightning-attention layer 0: full generate must match a single teacher-forced
    forward over the whole (prompt + generated) sequence."""
    cfg, sd = _minimax_cfg_sd()
    model = build_model(cfg, sd).cuda().eval()
    runner = ModelRunner(model, cfg, max_batch=1, max_len=32, device="cuda")
    prompt = torch.randint(0, cfg.vocab_size, (1, 5), device="cuda")
    gen = runner.generate_greedy(prompt, max_new_tokens=4)

    full_ids = torch.cat([prompt, gen], dim=1)
    pos = torch.arange(full_ids.shape[1], device="cuda").unsqueeze(0)
    ref_cache = KVCache(
        cfg.num_hidden_layers,
        1,
        cfg.num_key_value_heads,
        32,
        cfg.resolved_head_dim(),
        device="cuda",
    )
    hidden = model(full_ids, pos, ForwardContext(is_prefill=True, kv_cache=ref_cache))
    logits = model.compute_logits(hidden)
    ref_tokens = logits[:, prompt.shape[1] - 1 : -1].argmax(-1)
    assert torch.equal(ref_tokens, gen)


def _laurel_rank(cfg):
    return cfg.extra.get("laurel_rank", 64)


def _gemma4_sd(cfg):
    """Random state dict for Gemma4 / gemma3n with all features (PLE, AltUp, LAuReL, MatFormer)."""
    H, nh, nkv, hd = (
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.resolved_head_dim(),
    )
    ple_dim = cfg.extra.get("hidden_size_per_layer_input", 256)
    ple_vocab = cfg.extra.get("vocab_size_per_layer_input", min(cfg.vocab_size, 262144))
    K = cfg.extra.get("altup_num_inputs", 4)
    L = cfg.num_hidden_layers
    sd = {"model.embed_tokens.weight": _r(cfg.vocab_size, H), "model.norm.weight": _r(H)}
    # PLE (model-level)
    sd["model.embed_tokens_per_layer.weight"] = _r(ple_vocab, L * ple_dim)
    sd["model.per_layer_model_projection.weight"] = _r(L * ple_dim, H)
    sd["model.per_layer_projection_norm.weight"] = _r(ple_dim)
    # AltUp (model-level)
    for i in range(1, K):
        sd[f"model.altup_projections.{i - 1}.weight"] = _r(H, H)
        sd[f"model.altup_unembed_projections.{i - 1}.weight"] = _r(H, H)
    for i in range(L):
        p = f"model.layers.{i}"
        isize = (
            cfg.intermediate_size[i]
            if isinstance(cfg.intermediate_size, (list, tuple))
            else cfg.intermediate_size
        )
        # Norms
        for n in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            sd[f"{p}.{n}.weight"] = _r(H)
        # Attention
        a = f"{p}.self_attn"
        sd[f"{a}.q_proj.weight"] = _r(nh * hd, H)
        sd[f"{a}.k_proj.weight"] = _r(nkv * hd, H)
        sd[f"{a}.v_proj.weight"] = _r(nkv * hd, H)
        sd[f"{a}.o_proj.weight"] = _r(H, nh * hd)
        sd[f"{a}.q_norm.weight"] = _r(hd)
        sd[f"{a}.k_norm.weight"] = _r(hd)
        # MLP
        sd[f"{p}.mlp.gate_proj.weight"] = _r(isize, H)
        sd[f"{p}.mlp.up_proj.weight"] = _r(isize, H)
        sd[f"{p}.mlp.down_proj.weight"] = _r(H, isize)
        # LAuReL
        sd[f"{p}.laura.linear_left.weight"] = _r(_laurel_rank(cfg), H)
        sd[f"{p}.laura.linear_right.weight"] = _r(H, _laurel_rank(cfg))
        sd[f"{p}.laura.post_laurel_norm.weight"] = _r(H)
        # AltUp (per layer)
        sd[f"{p}.altup.correction_coefs.weight"] = _r(K, K)
        sd[f"{p}.altup.prediction_coefs.weight"] = _r(K * K, K)
        sd[f"{p}.altup.modality_router.weight"] = _r(K, H)
        sd[f"{p}.altup.router_norm.weight"] = _r(H)
        sd[f"{p}.altup.correct_output_scale"] = _r(H)
        # PLE (per layer)
        sd[f"{p}.per_layer_input_gate.weight"] = _r(ple_dim, H)
        sd[f"{p}.per_layer_projection.weight"] = _r(H, ple_dim)
        sd[f"{p}.post_per_layer_input_norm.weight"] = _r(H)
    return sd


def test_gemma4_prefill():
    """Gemma4 prefill with all features: AltUp, LAuReL, PLE, MatFormer."""
    assert is_supported("gemma4")
    L = 2
    ple_dim = 256
    K = 4
    cfg = ModelConfig(
        arch="gemma4",
        vocab_size=320,
        hidden_size=256,
        num_hidden_layers=L,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=[256, 384],
        max_position_embeddings=512,
        head_dim=64,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        query_pre_attn_scalar=64.0,
        rope_local_theta=1e4,
        sliding_window=16,
        sliding_window_pattern=5,
        hidden_act="gelu_pytorch_tanh",
        norm_add_unit_offset=True,
        extra=dict(
            laurel_rank=16,
            altup_num_inputs=K,
            altup_active_idx=0,
            altup_correct_scale=True,
            hidden_size_per_layer_input=ple_dim,
            vocab_size_per_layer_input=320,
            activation_sparsity_pattern=[0.0, 0.0],
        ),
    )
    sd = _gemma4_sd(cfg)
    _prefill(cfg, sd)
