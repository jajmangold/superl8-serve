# SPDX-License-Identifier: MIT
"""Paged-KV engine tests (issue #15): the engine's continuous-batch decode now
commits every new token with `superl8.quantize_kv_write_paged` (int8, block-table
addressed) and reads the whole ragged running batch back with ONE
`superl8.attn_paged_decode_cached` launch, instead of looping `attn_int8_decode`
once per slot against a contiguous per-slot region.

Two things to prove, matching superl8's own paged-decode kernel test bar:
  * numerically: paged int8 decode must closely match (cos~=1) the fp16
    contiguous-cache path for the SAME weights/tokens (teacher-forced, so a
    stray argmax flip can't cascade into an unrelated comparison);
  * block-table correctness: a finished sequence's blocks must return to the
    shared pool so a later request can reuse them (`PagedKVCache.free`) --
    the whole point of paging over a fixed per-slot region.
"""
import pytest
import torch

pytest.importorskip("superl8")

from superl8serve.engine import LLMEngine, PagedKVCache, SamplingParams
from superl8serve.models import ForwardContext, ModelConfig, ModelRunner, build_model

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="engine needs the CUDA superl8 kernels")


def _cfg():
    return ModelConfig(arch="qwen3", vocab_size=256, hidden_size=128, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, intermediate_size=256,
                       max_position_embeddings=256, head_dim=32, qk_norm=True,
                       tie_word_embeddings=True)


def _sd(cfg):
    def r(*s):
        return torch.randn(*s, device="cuda", dtype=torch.float16) * 0.05
    hd, nh, nkv, H = cfg.resolved_head_dim(), cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size
    sd = {"model.embed_tokens.weight": r(cfg.vocab_size, H), "model.norm.weight": r(H)}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = r(H)
        sd[f"{p}.post_attention_layernorm.weight"] = r(H)
        sd[f"{p}.self_attn.q_proj.weight"] = r(nh * hd, H)
        sd[f"{p}.self_attn.k_proj.weight"] = r(nkv * hd, H)
        sd[f"{p}.self_attn.v_proj.weight"] = r(nkv * hd, H)
        sd[f"{p}.self_attn.o_proj.weight"] = r(H, nh * hd)
        sd[f"{p}.self_attn.q_norm.weight"] = r(hd)
        sd[f"{p}.self_attn.k_norm.weight"] = r(hd)
        sd[f"{p}.mlp.gate_proj.weight"] = r(cfg.intermediate_size, H)
        sd[f"{p}.mlp.up_proj.weight"] = r(cfg.intermediate_size, H)
        sd[f"{p}.mlp.down_proj.weight"] = r(H, cfg.intermediate_size)
    return sd


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


@torch.inference_mode()
def test_paged_engine_matches_contiguous_runner():
    """A prefill + one paged decode step, driven directly through the engine's
    `PagedKVCache` + `ForwardContext` (the same seam `EngineRunner` uses), must
    produce logits with cosine similarity ~=1 against the standalone fp16
    contiguous-cache `ModelRunner` -- fed the SAME tokens at every step, so this
    checks the paged/int8 path itself rather than autoregressive drift."""
    torch.manual_seed(0)
    cfg = _cfg()
    sd = _sd(cfg)
    prompt = [3, 1, 4, 1, 5, 9, 2, 6]
    next_tok = 7

    model = build_model(cfg, sd).cuda().eval()

    runner = ModelRunner(model, cfg, max_batch=1, max_len=64, device="cuda")
    ref_prefill = runner.prefill(torch.tensor([prompt], device="cuda"))
    ref_decode = runner.decode(torch.tensor([[next_tok]], device="cuda"))

    cache = PagedKVCache(cfg.num_hidden_layers, 1, cfg.num_key_value_heads, 64,
                         cfg.resolved_head_dim(), device="cuda")
    slot = cache.alloc()

    cache.ensure_capacity([slot], [len(prompt)])
    ids = torch.tensor([prompt], device="cuda")
    pos = torch.arange(len(prompt), device="cuda").unsqueeze(0)
    ctx = ForwardContext(is_prefill=True, kv_cache=cache, slots=[slot])
    hidden = model(ids, pos, ctx)
    eng_prefill = model.compute_logits(hidden[:, -1])

    cache.ensure_capacity([slot], [len(prompt) + 1])
    ids = torch.tensor([[next_tok]], device="cuda")
    pos = torch.tensor([[len(prompt)]], device="cuda")
    ctx = ForwardContext(is_prefill=False, kv_cache=cache, slots=[slot], slot_lengths=[len(prompt)])
    hidden = model(ids, pos, ctx)
    eng_decode = model.compute_logits(hidden[:, -1])

    assert _cos(ref_prefill, eng_prefill) > 0.999   # no quantization on the prefill logits path
    # int8 paged KV (per-token RTN scale) adds a bit more error than fp16 --
    # same looser bar superl8's own paged-decode kernel tests use.
    assert _cos(ref_decode, eng_decode) > 0.995


def test_paged_kv_recycles_blocks_across_requests():
    """A finished sequence's blocks must return to the shared pool so a later
    request can reuse them. Serially generate far more requests than the block
    pool could satisfy if blocks were ever leaked -- a leak would either exhaust
    the pool (RuntimeError) or fail the exact free-count check below.

    `enable_cuda_graph=False`: this checks `PagedKVCache.free()` itself, which is
    orthogonal to CUDA-graph decode -- graphed decode pins one scratch block
    forever by design (see `tests/test_cuda_graph.py`), which would otherwise
    make every round after the first look like a 1-block leak here."""
    torch.manual_seed(1)
    cfg = _cfg()
    sd = _sd(cfg)
    eng = LLMEngine(cfg, sd, device="cuda", max_num_seqs=2, max_len=32, enable_cuda_graph=False)
    total_blocks = eng.cache.num_blocks

    for i in range(20):
        prompt = [(i % 200) + 1, ((i + 1) % 200) + 1]
        out = eng.generate([prompt], SamplingParams(temperature=0.0, max_tokens=6))[0]
        assert len(out) == 6
        assert len(eng.cache._free_blocks) == total_blocks, \
            f"round {i}: blocks leaked, only {len(eng.cache._free_blocks)}/{total_blocks} free"
