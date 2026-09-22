# SPDX-License-Identifier: MIT
"""Hunyuan CLA decode verification — cross-layer KV sharing must make
autoregressive decode match teacher-forced forward on a small shape
(batch=1, seq=8, 4 layers, cla_group_size=2)."""
import pytest
import torch

pytest.importorskip("superl8")

from superl8serve.models import ModelConfig, ModelRunner, build_model
from superl8serve.models.base import ForwardContext
from superl8serve.models.cache import KVCache

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="projections use the dp4a GEMM")


def _r(*s):
    return torch.randn(*s, device="cuda", dtype=torch.float16) * 0.05


def _attn_sd(sd, a, cfg, qk=None):
    hd = cfg.resolved_head_dim()
    nh, nkv, H = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size
    sd[f"{a}.q_proj.weight"] = _r(nh * hd, H)
    sd[f"{a}.k_proj.weight"] = _r(nkv * hd, H)
    sd[f"{a}.v_proj.weight"] = _r(nkv * hd, H)
    sd[f"{a}.o_proj.weight"] = _r(H, nh * hd)
    if qk:
        sd[f"{a}.{qk[0]}.weight"] = _r(hd)
        sd[f"{a}.{qk[1]}.weight"] = _r(hd)


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


def _hunyuan_cla_cfg_sd(cla_group_size=2):
    """Hunyuan with 4 layers, GQA (4 heads, 2 KV heads), 2-layer CLA groups."""
    cfg = ModelConfig(
        arch="hunyuan", vocab_size=64, hidden_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=256,
        max_position_embeddings=64, head_dim=32, qk_norm=True,
        tie_word_embeddings=True, rms_norm_eps=1e-5,
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=64,
        extra=dict(cla_group_size=cla_group_size),
    )
    H = cfg.hidden_size
    sd = {"model.embed_tokens.weight": _r(cfg.vocab_size, H),
          "model.norm.weight": _r(H)}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = _r(H)
        sd[f"{p}.post_attention_layernorm.weight"] = _r(H)
        _attn_sd(sd, f"{p}.self_attn", cfg, qk=("query_layernorm", "key_layernorm"))
        _moe_sd(sd, p, H, cfg.num_experts, cfg.moe_intermediate_size,
                shared_name="shared_mlp")
        sd[f"{p}.mlp.gate.wg.weight"] = sd.pop(f"{p}.mlp.gate.weight")
    return cfg, sd


def test_hunyuan_cla_prefill():
    """Basic prefill sanity — the CLA model can execute a forward pass."""
    cfg, sd = _hunyuan_cla_cfg_sd()
    model = build_model(cfg, sd).cuda().eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 6), device="cuda")
    pos = torch.arange(6, device="cuda").unsqueeze(0)
    cache = KVCache(cfg.num_hidden_layers, 1, cfg.num_key_value_heads, 16,
                    cfg.resolved_head_dim(), device="cuda")
    h = model(ids, pos, ForwardContext(is_prefill=True, kv_cache=cache))
    logits = model.compute_logits(h[:, -1])
    assert logits.shape == (1, cfg.vocab_size) and torch.isfinite(logits).all()


def test_hunyuan_cla_decode_matches_teacher_forced():
    """CLA decode must match teacher-forced: prefill 5 tokens, decode 4,
    and compare the 4 decode tokens against the last 4 positions of a
    teacher-forced forward over all 9 tokens."""
    cfg, sd = _hunyuan_cla_cfg_sd()
    model = build_model(cfg, sd).cuda().eval()

    runner = ModelRunner(model, cfg, max_batch=1, max_len=32, device="cuda")
    prompt = torch.randint(0, cfg.vocab_size, (1, 5), device="cuda")
    gen = runner.generate_greedy(prompt, max_new_tokens=4)

    full_ids = torch.cat([prompt, gen], dim=1)
    pos = torch.arange(full_ids.shape[1], device="cuda").unsqueeze(0)
    ref_cache = KVCache(cfg.num_hidden_layers, 1, cfg.num_key_value_heads, 32,
                        cfg.resolved_head_dim(), device="cuda")
    hidden = model(full_ids, pos, ForwardContext(is_prefill=True, kv_cache=ref_cache))
    logits = model.compute_logits(hidden)
    ref_tokens = logits[:, prompt.shape[1] - 1:-1].argmax(-1)
    assert torch.equal(ref_tokens, gen), (
        f"CLA decode {gen.tolist()} != teacher-forced {ref_tokens.tolist()}"
    )
