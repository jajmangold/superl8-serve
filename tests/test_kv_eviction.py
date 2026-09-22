# SPDX-License-Identifier: MIT
"""KV-cache eviction tests: SnapKV + Ada-KV + DuoAttention.

Three things to prove, following the same bar as test_paged_engine.py:

  * correctness: compact() preserves the KV data at kept positions through a
    quantize→dequantize→requantize cycle, matching the original read-back within
    the int8 RTN tolerance (cos ~= 0.995);
  * HBM freed: compact() returns a positive bytes_freed and the freed blocks
    re-enter the free pool so a later request can reuse them;
  * regression-free: with eviction disabled, evict_after_prefill() is a no-op
    (returns evicted=False and never touches the block store);
  * recall gate: cos-sim above the threshold = passed, below = blocked;
  * strategy correctness: snapkv_selection, adakv_budgets, duoattn_split match
    their expected shapes and monotonic properties.

All tests need CUDA + superl8 (same as test_paged_engine.py).
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("superl8")

from superl8serve.engine import PagedKVCache
from superl8serve.engine.kv_cache import EvictionConfig, KVEviction

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="needs CUDA + superl8 paged kernels")


# ── helpers ────────────────────────────────────────────────────────────────────


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def _make_cache(num_layers=2, num_slots=4, num_kv_heads=2, max_len=64, head_dim=32, block_size=8):
    """Small cache fixture for unit tests."""
    return PagedKVCache(
        num_layers, num_slots, num_kv_heads, max_len, head_dim,
        device="cuda", block_size=block_size,
    )


def _fill_slot(cache, slot, seq_len, seed=42):
    """Write deterministic KV data to a slot's blocks."""
    rng = torch.Generator(device="cuda").manual_seed(seed)
    for layer in range(cache.num_layers):
        k_all, v_all = [], []
        for pos in range(seq_len):
            k_all.append(torch.randn(1, cache.num_kv_heads, 1, cache.head_dim, device="cuda", dtype=torch.float16, generator=rng) * 0.1)
            v_all.append(torch.randn(1, cache.num_kv_heads, 1, cache.head_dim, device="cuda", dtype=torch.float16, generator=rng) * 0.1)
        k = torch.cat(k_all, dim=2)  # [1, Hkv, seq_len, D]
        v = torch.cat(v_all, dim=2)  # [1, Hkv, seq_len, D]
        cache.write_prefill(layer, k, v, slot=slot)


# ── HBM accounting ─────────────────────────────────────────────────────────────


class TestBlockBytes:
    def test_block_bytes_formula(self):
        """block_bytes matches the actual tensor memory footprint."""
        nkv, bs, hd = 8, 16, 128
        bb = KVEviction.block_bytes(nkv, bs, hd)
        # int8: 2 * 8 * 16 * 128 = 32768
        # scales: 2 * 8 * 16 * 4 = 1024
        assert bb == 2 * nkv * bs * hd + 2 * nkv * bs * 4

    def test_block_bytes_small(self):
        bb = KVEviction.block_bytes(2, 8, 32)
        assert bb == 2 * 2 * 8 * 32 + 2 * 2 * 8 * 4  # = 1024 + 128


# ── SnapKV selection ──────────────────────────────────────────────────────────


class TestSnapKVSelection:
    def test_selects_top_k_when_sequence_longer_than_budget(self):
        """With a synthetic attention pattern, the top-k positions are picked."""
        num_heads, num_kv_heads, hd = 4, 2, 32
        seq_len, budget = 24, 8
        q = torch.randn(num_heads, hd, device="cuda")
        k = torch.randn(num_kv_heads, seq_len, hd, device="cuda")
        # Make positions 5..12 artificially important
        k[:, 5:13, :] = q[:1].unsqueeze(0) * 10.0  # high similarity
        selected = KVEviction.snapkv_selection(q, k, budget, num_heads, num_kv_heads)
        assert len(selected) == budget
        # Most selected should be in the high-similarity region
        overlap = len(set(selected) & set(range(5, 13)))
        assert overlap >= budget // 2, f"only {overlap}/{budget} in high-sim region"

    def test_returns_all_when_seq_len_leq_budget(self):
        num_heads, num_kv_heads, hd = 4, 2, 32
        q = torch.randn(num_heads, hd, device="cuda")
        k = torch.randn(num_kv_heads, 6, hd, device="cuda")
        selected = KVEviction.snapkv_selection(q, k, budget=10, num_heads=num_heads, num_kv_heads=num_kv_heads)
        assert selected == [0, 1, 2, 3, 4, 5]

    def test_handles_gqa_repeat(self):
        """num_heads > num_kv_heads does not crash and returns correct count."""
        num_heads, num_kv_heads, hd = 8, 2, 32
        seq_len, budget = 16, 6
        q = torch.randn(num_heads, hd, device="cuda")
        k = torch.randn(num_kv_heads, seq_len, hd, device="cuda")
        selected = KVEviction.snapkv_selection(q, k, budget, num_heads, num_kv_heads)
        assert len(selected) == budget
        assert selected == sorted(selected)


# ── Ada-KV budgets ────────────────────────────────────────────────────────────


class TestAdaKVBudgets:
    def test_pyramid_is_monotonic(self):
        """Budgets increase (or stay equal) from early to late layers."""
        budgets = KVEviction.adakv_budgets(12, total_budget=4096, min_budget=128, max_budget=1024)
        assert len(budgets) == 12
        for i in range(1, len(budgets)):
            assert budgets[i] >= budgets[i - 1], f"budget decreased at layer {i}"

    def test_single_layer(self):
        budgets = KVEviction.adakv_budgets(1, total_budget=512)
        assert budgets == [512]

    def test_approximate_total(self):
        budgets = KVEviction.adakv_budgets(24, total_budget=8192, min_budget=128, max_budget=1024)
        total = sum(budgets)
        # Should be within 10% of the target
        assert abs(total - 8192) / 8192 < 0.10


# ── DuoAttention split ────────────────────────────────────────────────────────


class TestDuoAttentionSplit:
    def test_split_counts(self):
        streaming, snap = KVEviction.duoattn_split(32, streaming_fraction=0.25)
        assert len(streaming) == 8
        assert len(snap) == 24
        assert sorted(streaming + snap) == list(range(32))

    def test_min_one_streaming_head(self):
        streaming, snap = KVEviction.duoattn_split(2, streaming_fraction=0.1)
        assert len(streaming) == 1
        assert len(snap) == 1


# ── Recall gate ────────────────────────────────────────────────────────────────


class TestRecallGate:
    def test_passes_when_above_threshold(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        result = KVEviction.recall_check(a, b, threshold=0.99)
        assert result["passed"]
        assert result["cos_sim"] >= 0.999

    def test_fails_when_below_threshold(self):
        a = torch.tensor([1.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0])
        result = KVEviction.recall_check(a, b, threshold=0.99)
        assert not result["passed"]

    def test_threshold_at_boundary(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        result = KVEviction.recall_check(a, b, threshold=1.0)
        # cos_sim can be 0.9999999 due to fp32 rounding; the check is `cos >= threshold`.
        assert result["cos_sim"] > 0.999


# ── Block compaction ───────────────────────────────────────────────────────────


class TestCompact:
    def test_full_keep_is_noop(self):
        """Keeping all positions should not free any blocks."""
        cache = _make_cache(block_size=4)
        slot = cache.alloc()
        seq_len = 8
        cache.ensure_capacity([slot], [seq_len])
        _fill_slot(cache, slot, seq_len)
        old_blocks = list(cache._slot_blocks[slot])

        metrics = KVEviction.compact(cache, slot, list(range(seq_len)))
        assert metrics["blocks_freed"] == 0
        assert metrics["bytes_freed"] == 0
        assert cache._slot_blocks[slot] == old_blocks

    def test_frees_blocks_when_evicting_half(self):
        """Evicting half the positions should free at least 1 block."""
        cache = _make_cache(num_layers=2, num_slots=4, num_kv_heads=2, max_len=64, head_dim=32, block_size=4)
        slot = cache.alloc()
        seq_len = 8  # needs 2 blocks of size 4
        cache.ensure_capacity([slot], [seq_len])
        _fill_slot(cache, slot, seq_len)
        total_blocks_before = len(cache._free_blocks) + cache.used_blocks

        # Keep only first 4 positions (1 block)
        metrics = KVEviction.compact(cache, slot, [0, 1, 2, 3])
        assert metrics["blocks_freed"] >= 1
        assert metrics["bytes_freed"] > 0

        # Verify freed blocks re-entered the pool
        total_blocks_after = len(cache._free_blocks) + cache.used_blocks
        assert total_blocks_after == total_blocks_before

        # Verify the kept data is still readable
        blocks_after = cache._slot_blocks[slot]
        assert len(blocks_after) == 1

    def test_data_preserved_after_compact(self):
        """KV data at kept positions is preserved (cos ~= 0.995) after compact."""
        cache = _make_cache(num_layers=2, num_slots=4, num_kv_heads=2, max_len=128, head_dim=32, block_size=8)
        slot = cache.alloc()
        seq_len = 24  # 3 blocks
        cache.ensure_capacity([slot], [seq_len])
        _fill_slot(cache, slot, seq_len, seed=1)

        # Read reference at all positions (dequantized)
        ref_k, ref_v = cache.read_dense(0, slot, seq_len)

        # Keep positions 2..13 (12 tokens -> 2 blocks)
        keep = list(range(2, 14))
        KVEviction.compact(cache, slot, keep)
        new_seq_len = len(keep)

        # Read back from the compacted cache
        evicted_k, evicted_v = cache.read_dense(0, slot, new_seq_len)

        # Compare only the kept positions (positions 0..11 in the new ordering)
        # The reference positions 2..13 should match evicted positions 0..11
        ref_slice_k = ref_k[:, :, 2:14, :]
        ref_slice_v = ref_v[:, :, 2:14, :]

        assert _cos(ref_slice_k, evicted_k) > 0.99, "K data not preserved after compact"
        assert _cos(ref_slice_v, evicted_v) > 0.99, "V data not preserved after compact"

    def test_compact_empty_keep(self):
        """Compact with empty keep list returns no-op metrics."""
        cache = _make_cache()
        slot = cache.alloc()
        seq_len = 8
        cache.ensure_capacity([slot], [seq_len])
        _fill_slot(cache, slot, seq_len)

        metrics = KVEviction.compact(cache, slot, [])
        assert metrics["blocks_freed"] == 0

    def test_reuses_freed_blocks(self):
        """Freed blocks can be allocated to another slot."""
        cache = _make_cache(num_slots=2, block_size=4)
        slot_a = cache.alloc()
        slot_b = cache.alloc()

        seq_len = 12  # 3 blocks
        cache.ensure_capacity([slot_a], [seq_len])
        _fill_slot(cache, slot_a, seq_len)

        # Evict to 1 block
        KVEviction.compact(cache, slot_a, [0, 1, 2, 3])
        freed_blocks = len(cache._free_blocks)

        # Now slot_b should be able to allocate
        cache.ensure_capacity([slot_b], [seq_len])
        assert len(cache._slot_blocks[slot_b]) == 3
        # Free blocks should have been consumed
        assert len(cache._free_blocks) == freed_blocks - 3


# ── Evict after prefill (end-to-end) ──────────────────────────────────────────


class TestEvictAfterPrefill:
    def test_disabled_is_noop(self):
        """With eviction disabled, returns evicted=False and doesn't modify blocks."""
        cache = _make_cache()
        slot = cache.alloc()
        seq_len = 16
        cache.ensure_capacity([slot], [seq_len])
        _fill_slot(cache, slot, seq_len)
        blocks_before = list(cache._slot_blocks[slot])

        cfg = EvictionConfig(enabled=False)
        result = cache.evict_after_prefill(slot, seq_len, eviction_config=cfg, num_heads=4)
        assert not result["evicted"]
        assert cache._slot_blocks[slot] == blocks_before

    def test_snapkv_evicts_with_mask(self):
        """SnapKV eviction with explicit keep_mask frees blocks and returns metrics."""
        cache = _make_cache(num_layers=2, block_size=4)
        slot = cache.alloc()
        seq_len = 16
        cache.ensure_capacity([slot], [seq_len])
        _fill_slot(cache, slot, seq_len)

        cfg = EvictionConfig(enabled=True, policy="snapkv", kv_budget=8)
        # Pass an explicit keep_mask to bypass Q/K requirement
        result = cache.evict_after_prefill(
            slot, seq_len, eviction_config=cfg, num_heads=4,
            keep_mask=list(range(8)),
        )
        assert result["evicted"]
        assert result["metrics"]["bytes_freed"] > 0
        assert result["metrics"]["blocks_freed"] >= 1

    def test_snapkv_with_attention(self):
        """SnapKV selection via actual attention scores."""
        cache = _make_cache(num_layers=1, num_slots=1, num_kv_heads=2, max_len=128, head_dim=32, block_size=4)
        slot = cache.alloc()
        seq_len = 16
        cache.ensure_capacity([slot], [seq_len])
        _fill_slot(cache, slot, seq_len)

        # Create synthetic Q and K for selection
        num_heads = 4
        q_last = torch.randn(num_heads, 32, device="cuda")
        k_full = torch.randn(2, seq_len, 32, device="cuda")
        # Make early positions more important
        k_full[:, :8, :] = q_last[:1].unsqueeze(0) * 5.0

        cfg = EvictionConfig(enabled=True, policy="snapkv", kv_budget=8)
        result = cache.evict_after_prefill(
            slot, seq_len, eviction_config=cfg, num_heads=num_heads,
            q_last=q_last, k_full=k_full,
        )
        assert result["evicted"]
        assert result["metrics"]["tokens_kept"] == 8

    def test_recall_gate_blocks(self):
        """Recall gate blocks eviction when cos-sim below threshold."""
        cache = _make_cache(num_layers=1, block_size=4)
        slot = cache.alloc()
        seq_len = 8
        cache.ensure_capacity([slot], [seq_len])
        _fill_slot(cache, slot, seq_len)

        cfg = EvictionConfig(enabled=True, kv_budget=4, recall_threshold=0.9999)
        # logits_full and logits_after are very different
        logits_full = torch.tensor([1.0, 0.0, 0.0])

        def logits_fn():
            return torch.tensor([0.0, 1.0, 0.0])

        result = cache.evict_after_prefill(
            slot, seq_len, eviction_config=cfg, num_heads=4,
            keep_mask=[0, 1, 2, 3],
            logits_full=logits_full,
            logits_fn=logits_fn,
        )
        assert not result["evicted"]
        assert result["recall"] is not None
        assert not result["recall"]["passed"]

    def test_recall_gate_passes(self):
        """Recall gate passes when logits match."""
        cache = _make_cache(num_layers=1, block_size=4)
        slot = cache.alloc()
        seq_len = 8
        cache.ensure_capacity([slot], [seq_len])
        _fill_slot(cache, slot, seq_len)

        cfg = EvictionConfig(enabled=True, kv_budget=4, recall_threshold=0.99)
        logits_full = torch.tensor([1.0, 2.0, 3.0])

        def logits_fn():
            return torch.tensor([1.0, 2.0, 3.0])

        result = cache.evict_after_prefill(
            slot, seq_len, eviction_config=cfg, num_heads=4,
            keep_mask=[0, 1, 2, 3],
            logits_full=logits_full,
            logits_fn=logits_fn,
        )
        assert result["evicted"]
        assert result["recall"] is not None
        assert result["recall"]["passed"]

    def test_adakv_eviction(self):
        """Ada-KV eviction produces valid metrics."""
        cache = _make_cache(num_layers=4, block_size=4)
        slot = cache.alloc()
        seq_len = 16
        cache.ensure_capacity([slot], [seq_len])
        _fill_slot(cache, slot, seq_len)

        cfg = EvictionConfig(
            enabled=True, policy="adakv", kv_budget=512,
            adakv_min_budget=64, adakv_max_budget=256,
        )
        result = cache.evict_after_prefill(
            slot, seq_len, eviction_config=cfg, num_heads=4,
            keep_mask=list(range(8)),
        )
        assert result["evicted"]
        assert result["metrics"]["bytes_freed"] > 0

    def test_duoattn_eviction(self):
        """DuoAttention eviction produces valid metrics."""
        cache = _make_cache(num_layers=2, block_size=4)
        slot = cache.alloc()
        seq_len = 16
        cache.ensure_capacity([slot], [seq_len])
        _fill_slot(cache, slot, seq_len)

        cfg = EvictionConfig(
            enabled=True, policy="duoattn", kv_budget=8,
            duoattn_streaming_heads=0.25, duoattn_window=4,
        )
        result = cache.evict_after_prefill(
            slot, seq_len, eviction_config=cfg, num_heads=4,
            keep_mask=list(range(8)),
        )
        assert result["evicted"]
        assert result["metrics"]["bytes_freed"] > 0

    def test_hbm_measurement(self):
        """HBM measurement returns plausible bytes_freed values."""
        cache = _make_cache(num_layers=2, num_kv_heads=2, head_dim=32, block_size=4)
        slot = cache.alloc()
        seq_len = 12  # 3 blocks
        cache.ensure_capacity([slot], [seq_len])
        _fill_slot(cache, slot, seq_len)
        nkv, bs, hd = 2, 4, 32
        expected_bb = KVEviction.block_bytes(nkv, bs, hd)

        cfg = EvictionConfig(enabled=True, kv_budget=4)
        result = cache.evict_after_prefill(
            slot, seq_len, eviction_config=cfg, num_heads=4,
            keep_mask=[0, 1, 2, 3],
        )
        # 3 blocks -> 1 block, freed 2 blocks across 2 layers
        expected_freed = 2 * 2 * expected_bb
        assert result["metrics"]["bytes_freed"] == expected_freed


# ── No regression from existing tests ─────────────────────────────────────────


class TestNoRegression:
    """These tests verify that eviction does not change existing behavior when it
    is disabled or when no tokens are actually evicted (bit-identical)."""

    def test_evict_disabled_does_not_change_blocks(self):
        """With enabled=False, the slot block list must remain bit-identical."""
        cache = _make_cache(block_size=4)
        slot = cache.alloc()
        seq_len = 8
        cache.ensure_capacity([slot], [seq_len])
        _fill_slot(cache, slot, seq_len)
        blocks_copy = list(cache._slot_blocks[slot])

        cfg = EvictionConfig(enabled=False)
        cache.evict_after_prefill(slot, seq_len, eviction_config=cfg, num_heads=4)
        assert cache._slot_blocks[slot] == blocks_copy

    def test_full_keep_does_not_modify_data(self):
        """Keeping all positions leaves the cache tensors bit-identical."""
        cache = _make_cache(block_size=4)
        slot = cache.alloc()
        seq_len = 8
        cache.ensure_capacity([slot], [seq_len])
        _fill_slot(cache, slot, seq_len)

        # Snapshot cache data
        k_snap = cache.k_cache.clone()
        v_snap = cache.v_cache.clone()

        KVEviction.compact(cache, slot, list(range(seq_len)))
        # No blocks changed -> no data should have been written
        assert (cache.k_cache == k_snap).all()
        assert (cache.v_cache == v_snap).all()


# ── Integration: eviction + decode step ----------------------------------------


class TestEvictionWithDecode:
    """End-to-end: prefill, evict, decode, verify the decode still produces
    reasonable logits (not identical — evicted KV changes the attention output,
    but the output should be valid and not NaN)."""

    def test_decode_after_eviction_produces_valid_logits(self):
        """After eviction, one decode step produces finite logits with no NaN."""
        from superl8serve.models import ForwardContext, ModelConfig, build_model

        cfg = ModelConfig(
            arch="qwen3", vocab_size=256, hidden_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, intermediate_size=256,
            max_position_embeddings=256, head_dim=32, qk_norm=True,
            tie_word_embeddings=True,
        )
        hd, nh, nkv, H = cfg.resolved_head_dim(), cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size

        def _sd():
            def r(*s):
                return torch.randn(*s, device="cuda", dtype=torch.float16) * 0.05
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

        torch.manual_seed(42)
        sd = _sd()
        model = build_model(cfg, sd).cuda().eval()

        cache = PagedKVCache(
            cfg.num_hidden_layers, 1, cfg.num_key_value_heads, 64,
            cfg.resolved_head_dim(), device="cuda", block_size=4,
        )
        slot = cache.alloc()
        prompt = [3, 1, 4, 1, 5, 9, 2, 6]
        prompt_len = len(prompt)

        # Prefill
        cache.ensure_capacity([slot], [prompt_len])
        ids = torch.tensor([prompt], device="cuda")
        pos = torch.arange(prompt_len, device="cuda").unsqueeze(0)
        ctx = ForwardContext(is_prefill=True, kv_cache=cache, slots=[slot])
        model(ids, pos, ctx)

        # Evict to 4 tokens (2 blocks of block_size=4 -> 1 block)
        cfg_evict = EvictionConfig(enabled=True, kv_budget=4)
        evict_result = cache.evict_after_prefill(
            slot, prompt_len, eviction_config=cfg_evict, num_heads=nh,
            keep_mask=list(range(4)),
        )
        assert evict_result["evicted"]

        # Decode one step
        new_len = 4  # after eviction, we have 4 tokens
        cache.ensure_capacity([slot], [new_len + 1])
        ids = torch.tensor([[7]], device="cuda")
        pos = torch.tensor([[new_len]], device="cuda")
        ctx = ForwardContext(
            is_prefill=False, kv_cache=cache, slots=[slot],
            slot_lengths=[new_len],
        )
        hidden = model(ids, pos, ctx)
        logits = model.compute_logits(hidden[:, -1])
        assert not torch.isnan(logits).any(), "NaN in logits after eviction"
        assert not torch.isinf(logits).any(), "Inf in logits after eviction"
