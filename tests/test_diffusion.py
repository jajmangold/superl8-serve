# SPDX-License-Identifier: MIT
"""Diffusion decode tests: prove DiffusionGemma builds, registers, and generates
non-autoregressively through the engine. Needs CUDA + superl8 kernels."""
import pytest
import torch

pytest.importorskip("superl8")

from superl8serve.engine import LLMEngine, SamplingParams, DiffusionDecodeStrategy
from superl8serve.models import ModelConfig, build_model, is_supported, list_models

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="diffusion model needs CUDA superl8 kernels")


def _rand(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.float16) * 0.05


def _diffusion_cfg(**kw):
    d = dict(
        arch="diffusion_gemma", vocab_size=320, hidden_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=512,
        max_position_embeddings=128, head_dim=64, rms_norm_eps=1e-6, rope_theta=1e6,
        qk_norm=True, tie_word_embeddings=True,
        norm_add_unit_offset=True, hidden_act="gelu_pytorch_tanh",
        query_pre_attn_scalar=64.0,
        decode_strategy="diffusion",
    )
    d.update(kw)
    return ModelConfig(**d)


def _gemma_sd(cfg):
    sd = {"model.embed_tokens.weight": _rand(cfg.vocab_size, cfg.hidden_size),
          "model.norm.weight": _rand(cfg.hidden_size)}
    hd, nh, nkv = cfg.resolved_head_dim(), cfg.num_attention_heads, cfg.num_key_value_heads
    H = cfg.hidden_size
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        for n in ("input_layernorm", "post_attention_layernorm",
                  "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            sd[f"{p}.{n}.weight"] = _rand(H)
        sd[f"{p}.self_attn.q_proj.weight"] = _rand(nh * hd, H)
        sd[f"{p}.self_attn.k_proj.weight"] = _rand(nkv * hd, H)
        sd[f"{p}.self_attn.v_proj.weight"] = _rand(nkv * hd, H)
        sd[f"{p}.self_attn.o_proj.weight"] = _rand(H, nh * hd)
        sd[f"{p}.self_attn.q_norm.weight"] = _rand(hd)
        sd[f"{p}.self_attn.k_norm.weight"] = _rand(hd)
        sd[f"{p}.mlp.gate_proj.weight"] = _rand(cfg.intermediate_size, H)
        sd[f"{p}.mlp.up_proj.weight"] = _rand(cfg.intermediate_size, H)
        sd[f"{p}.mlp.down_proj.weight"] = _rand(H, cfg.intermediate_size)
    return sd


def test_diffusion_gemma_registers():
    assert is_supported("diffusion_gemma")
    assert "diffusion_gemma" in list_models()


def test_diffusion_model_builds():
    cfg = _diffusion_cfg()
    sd = _gemma_sd(cfg)
    model = build_model(cfg, sd).cuda().eval()
    assert model.config.decode_strategy == "diffusion"


def test_diffusion_engine_generates():
    torch.manual_seed(42)
    cfg = _diffusion_cfg()
    sd = _gemma_sd(cfg)
    eng = LLMEngine(cfg, sd, device="cuda", max_num_seqs=4, max_len=256,
                    num_diffusion_steps=4)
    prompt = [1, 2, 3]
    out = eng.generate([prompt], SamplingParams(temperature=0.0, max_tokens=4))[0]
    assert len(out) == 4
    assert all(0 <= t < cfg.vocab_size for t in out)


def test_diffusion_engine_deterministic():
    torch.manual_seed(7)
    cfg = _diffusion_cfg()
    sd = _gemma_sd(cfg)
    eng = LLMEngine(cfg, sd, device="cuda", max_num_seqs=4, max_len=256,
                    num_diffusion_steps=4)
    prompt = [1, 2, 3]

    torch.manual_seed(7)
    cfg2 = _diffusion_cfg()
    sd2 = _gemma_sd(cfg2)
    eng2 = LLMEngine(cfg2, sd2, device="cuda", max_num_seqs=4, max_len=256,
                     num_diffusion_steps=4)

    out1 = eng.generate([prompt], SamplingParams(temperature=0.0, max_tokens=4))[0]
    out2 = eng2.generate([prompt], SamplingParams(temperature=0.0, max_tokens=4))[0]
    assert out1 == out2


def test_diffusion_engine_multiple_requests():
    torch.manual_seed(13)
    cfg = _diffusion_cfg()
    sd = _gemma_sd(cfg)
    eng = LLMEngine(cfg, sd, device="cuda", max_num_seqs=8, max_len=256,
                    num_diffusion_steps=4)
    prompts = [[1, 2], [3, 4, 5], [6, 7, 8, 9]]
    outs = eng.generate(prompts, SamplingParams(temperature=0.0, max_tokens=3))
    assert len(outs) == 3
    for o in outs:
        assert len(o) == 3
        assert all(0 <= t < cfg.vocab_size for t in o)


def test_diffusion_outputs_all_tokens_at_once():
    """Prove diffusion generates the full output in one step (unlike AR which adds
    one token per decode step)."""
    torch.manual_seed(99)
    cfg = _diffusion_cfg()
    sd = _gemma_sd(cfg)
    eng = LLMEngine(cfg, sd, device="cuda", max_num_seqs=4, max_len=256,
                    num_diffusion_steps=4)
    prompt = [1, 2]
    seq_id = eng.add_request(prompt, SamplingParams(temperature=0.0, max_tokens=4))
    assert len(eng.sequence(seq_id).output_ids) == 0
    eng.step()
    assert len(eng.sequence(seq_id).output_ids) == 4


def test_diffusion_strategy_standalone():
    """Verify DiffusionDecodeStrategy.generate() works standalone with a model runner
    and cache, without the full engine."""
    torch.manual_seed(23)
    cfg = _diffusion_cfg(num_hidden_layers=1, max_position_embeddings=64)
    sd = _gemma_sd(cfg)
    model = build_model(cfg, sd).cuda().eval()

    from superl8serve.models.base import ForwardContext
    from superl8serve.engine.kv_cache import PagedKVCache

    cache = PagedKVCache(cfg.num_hidden_layers, 2, cfg.num_key_value_heads, 64,
                         cfg.resolved_head_dim(), device="cuda")
    slot = cache.alloc()

    strategy = DiffusionDecodeStrategy(num_steps=4, mask_id=0)

    class FakeSeq:
        pass

    seq = FakeSeq()
    seq.num_prompt = 3
    seq.prompt_ids = [1, 2, 3]
    seq.slot = slot
    seq.params = SamplingParams(max_tokens=4)

    cache.ensure_capacity([seq.slot], [seq.num_prompt + seq.params.max_tokens])
    ctx = ForwardContext(is_prefill=True, kv_cache=cache, slots=[seq.slot])

    out = strategy.generate(model, cache, "cuda", seq, ctx)
    assert len(out) == 4
    assert all(0 <= t < cfg.vocab_size for t in out)
