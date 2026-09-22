# SPDX-License-Identifier: MIT
"""Production-lifecycle hardening regressions.

Two long-running-server memory-exhaustion paths caught by external review:

  BUG 1 — unbounded prefix-cache (KV-block) retention. `PagedKVCache` pinned KV
          blocks in its radix trie forever when `max_prefix_entries` was left at its
          `None` default (which the engine never overrode), so a stream of unique
          prompts eventually exhausted the block pool. The trie is now bounded by a
          pinned-BLOCK budget derived from the pool size — CPU-testable, no kernels.

  BUG 2 — `LLMEngine.generate()` retained every completed `Sequence` in `_out`.
          Offline callers loop on `generate()` and never call `forget()`, so `_out`
          grew without bound. `generate()` now auto-forgets what it collected.
"""

import pytest
import torch

from superl8serve.engine.kv_cache import PagedKVCache


# ── BUG 1: prefix-cache block budget (pure host trie/refcount logic; runs on CPU) ──


def _cpu_cache(num_blocks=64, fraction=0.25, block_size=16):
    # head_dim/heads/layers tiny — only the block bookkeeping is under test, and the
    # int8 storage tensors allocate fine on CPU.
    return PagedKVCache(
        num_layers=2,
        num_slots=4,
        num_kv_heads=2,
        max_len=64,
        head_dim=8,
        device="cpu",
        block_size=block_size,
        num_blocks=num_blocks,
        max_prefix_block_fraction=fraction,
    )


def test_prefix_budget_is_non_none_by_default():
    """The pool-derived block budget must be set even when no explicit limit is
    passed — this is the exact constructor call the engine makes."""
    c = PagedKVCache(2, 4, 2, 64, 8, device="cpu", num_blocks=64)
    assert c._max_prefix_blocks is not None
    assert c._max_prefix_blocks <= c.num_blocks


def test_unique_prompts_do_not_exhaust_the_pool():
    """Stream many unique prompts through the cache (each stored as a completed
    prefix). Pinned prefix blocks must stay under budget and the free pool must
    never dry up — the pre-fix code pinned 2 blocks per prompt forever and would
    raise 'no free blocks' from ensure_capacity within ~30 prompts."""
    c = _cpu_cache(num_blocks=64, fraction=0.25)
    budget = c._max_prefix_blocks  # 16 blocks

    for base in range(200):
        slot = c.alloc()
        # 33 unique tokens -> 3 blocks allocated, 2 block-aligned prefix entries.
        tokens = list(range(base * 100, base * 100 + 33))
        c.ensure_capacity([slot], [33])  # would raise if the pool were exhausted
        c.store_prefix(tokens, slot)
        c.free(slot)

        assert c.pinned_prefix_blocks <= budget, (
            f"prefix cache pinned {c.pinned_prefix_blocks} blocks > budget {budget} "
            f"at prompt {base}"
        )
        # Pool must still be able to admit the next request.
        assert len(c._free_blocks) > 0, f"block pool exhausted at prompt {base}"

    # Byte-budget accessor is consistent with the block count.
    assert c.prefix_block_bytes() > 0
    assert c.pinned_prefix_blocks <= budget


def test_live_inflight_blocks_are_never_evicted():
    """A block referenced by a live (un-freed) slot must survive prefix-budget
    eviction: eviction only drops the completed-prefix trie reference, never a
    block still in use by an in-flight sequence."""
    c = _cpu_cache(num_blocks=64, fraction=0.05)  # tiny budget -> aggressive eviction

    live_slot = c.alloc()
    c.ensure_capacity([live_slot], [33])
    live_blocks = list(c._slot_blocks[live_slot])
    assert live_blocks
    c.store_prefix(list(range(1000, 1033)), live_slot)
    # Slot is intentionally NOT freed -> still in flight.

    # Hammer with unique prompts to force many evictions.
    for base in range(100):
        s = c.alloc()
        c.ensure_capacity([s], [33])
        c.store_prefix(list(range(base * 50, base * 50 + 33)), s)
        c.free(s)

    # Every block the live slot holds must still be allocated (not in the free pool).
    free = set(c._free_blocks)
    for blk in live_blocks:
        assert blk not in free, f"live in-flight block {blk} was wrongly freed"
    assert c.pinned_prefix_blocks <= c._max_prefix_blocks


def test_explicit_max_prefix_blocks_overrides_fraction():
    c = PagedKVCache(
        2, 4, 2, 64, 8, device="cpu", num_blocks=100,
        max_prefix_blocks=7, max_prefix_block_fraction=0.9,
    )
    assert c._max_prefix_blocks == 7


# ── BUG 2: generate() must not retain finished Sequences in _out ───────────────


CUDA = torch.cuda.is_available()


def _tiny_cfg():
    from superl8serve.models import ModelConfig

    return ModelConfig(
        arch="qwen3",
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        max_position_embeddings=128,
        head_dim=32,
        qk_norm=True,
        tie_word_embeddings=True,
    )


def _tiny_sd(cfg):
    def r(*s):
        return torch.randn(*s, device="cuda", dtype=torch.float16) * 0.05

    hd, nh, nkv, H = (
        cfg.resolved_head_dim(),
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.hidden_size,
    )
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


@pytest.mark.skipif(not CUDA, reason="generate() needs the CUDA superl8 kernels")
def test_generate_does_not_leak_out_bookkeeping():
    """Repeated offline generate() must not grow `_out` without bound."""
    pytest.importorskip("superl8")
    from superl8serve.engine import LLMEngine, SamplingParams

    torch.manual_seed(0)
    cfg = _tiny_cfg()
    eng = LLMEngine(
        cfg, _tiny_sd(cfg), device="cuda", max_num_seqs=4, max_len=64,
        enable_cuda_graph=False,
    )
    params = SamplingParams(temperature=0.0, max_tokens=4)

    for i in range(30):
        out = eng.generate([[1, 2, 3, 4, 5, i + 6]], params)
        assert out and out[0], "generate returned no tokens"
        # The collected sequence must be forgotten -> _out does not accumulate.
        assert len(eng._out) == 0, f"_out grew to {len(eng._out)} after {i + 1} calls"

    # Returned lists survive after the Sequence is forgotten.
    outs = eng.generate([[1, 2, 3], [4, 5, 6]], params)
    assert len(outs) == 2 and all(len(o) > 0 for o in outs)
    assert len(eng._out) == 0


# ── BUG 3: packaging — hard torch-build guard (no GPU needed) ──────────────────


def test_torch_build_check_rejects_wrong_cuda(monkeypatch):
    """A torch that is not the +cu129 (CUDA 12.9) build must raise a clear
    ImportError pointing at the cu129 index — so a naive `pip install` that pulled
    a +cpu / mismatched wheel fails loudly instead of dying inside a kernel."""
    import superl8serve

    monkeypatch.setattr(torch.version, "cuda", "12.1", raising=False)
    monkeypatch.delenv("SUPERL8_SERVE_SKIP_TORCH_CHECK", raising=False)
    with pytest.raises(ImportError, match="cu129"):
        superl8serve._check_torch_build()


def test_torch_build_check_accepts_cu129(monkeypatch):
    import superl8serve

    monkeypatch.setattr(torch.version, "cuda", "12.9", raising=False)
    superl8serve._check_torch_build()  # must not raise


def test_torch_build_check_bypass_env(monkeypatch):
    import superl8serve

    monkeypatch.setattr(torch.version, "cuda", None, raising=False)
    monkeypatch.setenv("SUPERL8_SERVE_SKIP_TORCH_CHECK", "1")
    superl8serve._check_torch_build()  # bypassed -> no raise


def test_package_version_is_bumped():
    import superl8serve

    assert superl8serve.__version__ == "0.1.0"


@pytest.mark.skipif(not CUDA, reason="generate() needs the CUDA superl8 kernels")
def test_api_style_forget_still_works():
    """The long-lived API path (add_request + explicit forget) is unaffected by the
    generate() auto-forget: sequences persist until the caller forgets them."""
    pytest.importorskip("superl8")
    from superl8serve.engine import LLMEngine, SamplingParams

    torch.manual_seed(0)
    cfg = _tiny_cfg()
    eng = LLMEngine(
        cfg, _tiny_sd(cfg), device="cuda", max_num_seqs=4, max_len=64,
        enable_cuda_graph=False,
    )
    params = SamplingParams(temperature=0.0, max_tokens=4)

    sid = eng.add_request([1, 2, 3, 4, 5], params)
    while not eng.sequence(sid).is_finished(eng.eos_id):
        eng.step()
    # Still retained for the caller to read.
    assert sid in eng._out
    assert eng.sequence(sid).output_ids
    eng.forget(sid)
    assert sid not in eng._out
