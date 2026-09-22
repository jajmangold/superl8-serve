# SPDX-License-Identifier: MIT
"""Prefix caching tests (issue #80): RadixAttention + chunked prefill.
Two requests with a shared prefix skip recompute of the shared portion,
and output is identical to no-cache."""

import pytest
import torch

pytest.importorskip("superl8")

from superl8serve.engine import LLMEngine, PagedKVCache, SamplingParams, Sequence
from superl8serve.models import ModelConfig

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="engine needs the CUDA superl8 kernels")


def _cfg():
    return ModelConfig(
        arch="qwen3",
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        max_position_embeddings=256,
        head_dim=32,
        qk_norm=True,
        tie_word_embeddings=True,
    )


def _sd(cfg):
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


# ── Existing correctness tests (must still pass) ────────────────────────────


def test_prefix_cache_matches_no_cache():
    """Two requests sharing a prefix must produce identical greedy output to the
    same requests run without prefix caching, and the second request should reuse
    the first's KV blocks for the shared portion."""
    torch.manual_seed(42)
    cfg = _cfg()
    sd = _sd(cfg)
    prompt_a = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    prompt_b = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25]

    params = SamplingParams(temperature=0.0, max_tokens=8)

    # Reference run: no caching (both requests processed together, fresh cache)
    torch.manual_seed(42)
    ref_eng = LLMEngine(
        cfg, _sd(cfg), device="cuda", max_num_seqs=8, max_len=64, enable_cuda_graph=False
    )
    ref_out = ref_eng.generate([prompt_a, prompt_b], params)

    # Cached run: process one at a time so prefix is stored before second request
    torch.manual_seed(42)
    eng = LLMEngine(cfg, sd, device="cuda", max_num_seqs=8, max_len=64, enable_cuda_graph=False)

    sid_a = eng.add_request(prompt_a, params)
    while not eng.sequence(sid_a).is_finished(eng.eos_id):
        eng.step()
    out_a = eng.sequence(sid_a).output_ids
    eng.forget(sid_a)

    # Before second request, verify prefix was stored
    matched, blocks = eng.cache.lookup_prefix(prompt_b)
    assert matched >= 16, f"expected prefix match >=16 tokens, got {matched}"

    sid_b = eng.add_request(prompt_b, params)
    seq_b = eng.sequence(sid_b)

    while not seq_b.is_finished(eng.eos_id):
        eng.step()
    out_b = seq_b.output_ids
    eng.forget(sid_b)

    # Output must match reference
    assert out_a == ref_out[0], f"prefix-cached A {out_a} != ref {ref_out[0]}"
    assert out_b == ref_out[1], f"prefix-cached B {out_b} != ref {ref_out[1]}"


def test_prefix_cache_block_reuse():
    """The second request must reuse the first request's physical blocks for the
    shared prefix — alloc at most one new block for the suffix."""
    torch.manual_seed(99)
    cfg = _cfg()
    sd = _sd(cfg)
    eng = LLMEngine(cfg, sd, device="cuda", max_num_seqs=8, max_len=64, enable_cuda_graph=False)
    pool_size = eng.cache.num_blocks

    # First request: a long prefix
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    params = SamplingParams(temperature=0.0, max_tokens=4)

    sid1 = eng.add_request(prompt, params)
    while not eng.sequence(sid1).is_finished(eng.eos_id):
        eng.step()
    eng.forget(sid1)

    # Second request: same prefix, different suffix
    prompt2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 30]
    sid2 = eng.add_request(prompt2, params)
    seq2 = eng.sequence(sid2)

    # Check that blocks were shared
    matched, blocks = eng.cache.lookup_prefix(prompt2)
    assert matched >= 16, f"prefix match expected >=16, got {matched}"
    assert len(blocks) > 0, "should share at least one block"

    while not seq2.is_finished(eng.eos_id):
        eng.step()
    eng.forget(sid2)

    # Verify blocks reused: at most one block is held by the prefix entry
    free_after_second = len(eng.cache._free_blocks)
    held_by_prefix = pool_size - free_after_second
    assert held_by_prefix <= 1, (
        f"at most 1 block held by prefix cache, got {held_by_prefix} missing: "
        f"{free_after_second}/{pool_size}"
    )
    # The shared block must still be reachable via lookup
    matched2, blocks2 = eng.cache.lookup_prefix(prompt2)
    assert matched2 >= 16 and len(blocks2) > 0, "prefix should still be cached"


def test_prefix_cache_lookup_no_match():
    """Lookup returns 0 for a prompt with no stored prefix."""
    torch.manual_seed(7)
    cfg = _cfg()
    sd = _sd(cfg)
    eng = LLMEngine(cfg, sd, device="cuda", max_num_seqs=4, max_len=64, enable_cuda_graph=False)
    matched, blocks = eng.cache.lookup_prefix([99, 98, 97])
    assert matched == 0
    assert blocks == []


# ── RadixAttention: compressed trie with LRU eviction ──────────────────────


def test_radix_trie_compressed_prefix_overlap():
    """RadixAttention stores common prefixes as shared edges. Three requests with
    overlapping prefixes sharing >=16 tokens (block-aligned). After A, B must find
    16-token match; after B, C must also find 16-token match."""
    torch.manual_seed(1)
    cfg = _cfg()
    sd = _sd(cfg)
    eng = LLMEngine(cfg, sd, device="cuda", max_num_seqs=4, max_len=64, enable_cuda_graph=False)

    params = SamplingParams(temperature=0.0, max_tokens=2)

    # Request A: 18 tokens
    prompt_a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    sid_a = eng.add_request(prompt_a, params)
    while not eng.sequence(sid_a).is_finished(eng.eos_id):
        eng.step()
    eng.forget(sid_a)

    # Request B: shares first 16 tokens with A, diverges at 17
    prompt_b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 100, 101]
    matched_b, blocks_b = eng.cache.lookup_prefix(prompt_b)
    assert matched_b >= 16, f"B should match >=16 tokens prefix, got {matched_b}"
    sid_b = eng.add_request(prompt_b, params)
    while not eng.sequence(sid_b).is_finished(eng.eos_id):
        eng.step()
    eng.forget(sid_b)

    # Request C: shares 17 tokens with A (diverges only at last token)
    prompt_c = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 99]
    matched_c, blocks_c = eng.cache.lookup_prefix(prompt_c)
    assert matched_c >= 16, f"C should match >=16 tokens prefix, got {matched_c}"


def test_radix_trie_eviction_lru():
    """RadixAttention evicts least-recently-used prefix entries when the cache
    exceeds capacity. Set max_prefix_entries=2, store 3 distinct prefixes, verify
    the earliest (least recently accessed) is evicted."""
    cfg = _cfg()
    sd = _sd(cfg)
    cache = PagedKVCache(
        cfg.num_hidden_layers, 4, cfg.num_key_value_heads, 64,
        cfg.resolved_head_dim(), device="cuda", block_size=16, num_blocks=32,
        max_prefix_entries=2,
    )
    # Allocate slots for three sequences
    slots = [cache.alloc() for _ in range(3)]

    # Store three distinct prefixes (each 17 tokens, need 2 blocks)
    for sidx, tokens in enumerate(
        [[1] * 17, [2] * 17, [3] * 17]
    ):
        cache.ensure_capacity([slots[sidx]], [17])
        cache.store_prefix(tokens, slots[sidx])

    # Cache only has 2 entries — the first one should be evicted (LRU: least recently stored)
    matched, blocks = cache.lookup_prefix([1] * 17)
    assert matched == 0, "first prefix should be evicted"

    # Second and third should still be present
    matched2, blocks2 = cache.lookup_prefix([2] * 17)
    assert matched2 >= 16, "second prefix should survive"

    # Access the second prefix, making it MRU
    cache.lookup_prefix([2] * 17)

    # Store a new fourth prefix — should evict the third (LRU now)
    cache.ensure_capacity([slots[2]], [17])
    cache.store_prefix([4] * 17, slots[2])
    cache.lookup_prefix([3] * 17)
    matched3, blocks3 = cache.lookup_prefix([3] * 17)
    # Now the trie has 2 entries: [4]*17 (most recent store), [2]*17 (most recent access)
    # The third [3]*17 was evicted when [4]*17 was stored
    matched3, blocks3 = cache.lookup_prefix([3] * 17)
    assert matched3 == 0, "third prefix should have been evicted for LRU"

    for s in slots:
        cache.free(s)


def test_radix_trie_eviction_clears_blocks():
    """When a prefix entry is evicted, its blocks must be returned to the free pool."""
    cfg = _cfg()
    sd = _sd(cfg)
    cache = PagedKVCache(
        cfg.num_hidden_layers, 4, cfg.num_key_value_heads, 64,
        cfg.resolved_head_dim(), device="cuda", block_size=16, num_blocks=16,
        max_prefix_entries=2,
    )
    initial_free = len(cache._free_blocks)

    slot = cache.alloc()
    cache.ensure_capacity([slot], [33])  # 3 blocks
    cache.store_prefix([1] * 33, slot)
    cache.free(slot)  # slot freed but trie still holds refs

    # All blocks pinned by trie (prefix not yet evicted).
    # 33 tokens → 2 complete blocks (0-15, 16-31) get trie entries; block 32 is freed.
    free_after_store = len(cache._free_blocks)
    blocks_pinned = initial_free - free_after_store
    assert blocks_pinned == 2, f"expected 2 pinned, got {blocks_pinned}"

    # Fill cache with another prefix → evicts first
    slot2 = cache.alloc()
    cache.ensure_capacity([slot2], [33])
    cache.store_prefix([2] * 33, slot2)
    cache.free(slot2)

    # Blocks from first prefix should be freed
    free_after_evict = len(cache._free_blocks)
    pinned_after = initial_free - free_after_evict
    assert pinned_after <= 3, f"at most 3 blocks pinned after eviction, got {pinned_after}"

    # The third prefix is still in cache
    matched, blocks = cache.lookup_prefix([2] * 33)
    assert matched >= 32


# ── Chunked prefill: bit-identical logits ──────────────────────────────────


def test_chunked_prefill_bit_identical():
    """Chunked prefill must produce identical greedy output tokens as full prefill
    for the same prompt, under temperature=0.0 (greedy argmax must match exactly).
    This tests both the non-varlen and varlen prefill paths."""
    torch.manual_seed(42)
    cfg = _cfg()
    sd = _sd(cfg)
    params = SamplingParams(temperature=0.0, max_tokens=8)

    # Full prefill (reference)
    torch.manual_seed(42)
    ref_eng = LLMEngine(
        cfg, _sd(cfg), device="cuda", max_num_seqs=8, max_len=64,
        enable_cuda_graph=False,
    )
    ref_out = ref_eng.generate(
        [[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]], params
    )[0]

    # Chunked prefill: chunk_size=8
    torch.manual_seed(42)
    eng = LLMEngine(
        cfg, sd, device="cuda", max_num_seqs=8, max_len=64,
        enable_cuda_graph=False, chunked_prefill_size=8,
    )
    sid = eng.add_request(
        [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], params
    )
    while not eng.sequence(sid).is_finished(eng.eos_id):
        eng.step()
    chunk_out = eng.sequence(sid).output_ids
    eng.forget(sid)

    assert chunk_out == ref_out, (
        f"Chunked prefill {chunk_out} != full prefill {ref_out}"
    )


def test_chunked_prefill_kv_cache_equivalence():
    """After chunked prefill, the KV cache must contain the same values as after
    full prefill. Write the same prompt using chunked and full, then measure
    per-block K/V norms. (The int8 quantized values should match.)"""

    def _kv_fingerprint(eng, layer=0):
        """Return a hash of K/V cache for *layer* across all used blocks."""
        used = set(b for blks in eng.cache._slot_blocks for b in blks)
        if not used:
            return 0.0
        k = eng.cache.k_cache[layer, list(used)].float()
        v = eng.cache.v_cache[layer, list(used)].float()
        return (k.sum() + v.sum()).item()

    torch.manual_seed(7)
    cfg = _cfg()
    sd = _sd(cfg)
    params = SamplingParams(temperature=0.0, max_tokens=2)
    prompt = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

    # Full (uncached) prefill
    torch.manual_seed(7)
    full_eng = LLMEngine(
        cfg, _sd(cfg), device="cuda", max_num_seqs=8, max_len=64, enable_cuda_graph=False,
    )
    sid_f = full_eng.add_request(prompt, params)
    while not full_eng.sequence(sid_f).is_finished(full_eng.eos_id):
        full_eng.step()
    full_fp = _kv_fingerprint(full_eng)
    full_eng.forget(sid_f)

    # Chunked prefill with chunk_size=8
    torch.manual_seed(7)
    chunk_eng = LLMEngine(
        cfg, sd, device="cuda", max_num_seqs=8, max_len=64,
        enable_cuda_graph=False, chunked_prefill_size=8,
    )
    sid_c = chunk_eng.add_request(prompt, params)
    while not chunk_eng.sequence(sid_c).is_finished(chunk_eng.eos_id):
        chunk_eng.step()
    chunk_fp = _kv_fingerprint(chunk_eng)
    chunk_eng.forget(sid_c)

    # The int8 KV values should match
    assert abs(full_fp - chunk_fp) < 1e-3, (
        f"KV cache mismatch: full={full_fp} chunked={chunk_fp}"
    )


def test_chunked_prefill_with_prefix_cache():
    """Chunked prefill combined with prefix caching: first request fills the prefix,
    second request reuses it via chunked prefill for the suffix."""
    torch.manual_seed(42)
    cfg = _cfg()
    sd = _sd(cfg)
    params = SamplingParams(temperature=0.0, max_tokens=4)

    # Reference: both requests without caching
    torch.manual_seed(42)
    ref_eng = LLMEngine(
        cfg, _sd(cfg), device="cuda", max_num_seqs=8, max_len=64,
        enable_cuda_graph=False,
    )
    prompt_a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    prompt_b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 99]
    ref_out = ref_eng.generate([prompt_a, prompt_b], params)

    # Cached + chunked prefill
    torch.manual_seed(42)
    eng = LLMEngine(
        cfg, sd, device="cuda", max_num_seqs=8, max_len=64,
        enable_cuda_graph=False, chunked_prefill_size=8,
    )

    sid_a = eng.add_request(prompt_a, params)
    while not eng.sequence(sid_a).is_finished(eng.eos_id):
        eng.step()
    out_a = eng.sequence(sid_a).output_ids
    eng.forget(sid_a)

    sid_b = eng.add_request(prompt_b, params)
    while not eng.sequence(sid_b).is_finished(eng.eos_id):
        eng.step()
    out_b = eng.sequence(sid_b).output_ids
    eng.forget(sid_b)

    assert out_a == ref_out[0], f"prefix-cached A {out_a} != ref {ref_out[0]}"
    assert out_b == ref_out[1], f"prefix-cached B {out_b} != ref {ref_out[1]}"


def test_chunked_prefill_interleaves_with_decode():
    """When one sequence is chunk-prefilling, another sequence must still be able
    to decode (TTFT for the decoding sequence should not be blocked by a full
    prefill of the first)."""
    torch.manual_seed(1)
    cfg = _cfg()
    sd = _sd(cfg)

    eng = LLMEngine(
        cfg, sd, device="cuda", max_num_seqs=4, max_len=64,
        enable_cuda_graph=False, chunked_prefill_size=8,
    )

    params_short = SamplingParams(temperature=0.0, max_tokens=4)
    params_long = SamplingParams(temperature=0.0, max_tokens=2)

    # First request: long prompt (32 tokens) — will be chunked
    long_prompt = list(range(1, 33))
    sid_long = eng.add_request(long_prompt, params_long)

    # Second request: short prompt, added immediately after
    short_prompt = [100, 101, 102, 103]
    sid_short = eng.add_request(short_prompt, params_short)

    # Run steps until short request finishes. If chunked prefill works,
    # the short request should finish without waiting for the long one's
    # full prefill to complete.
    seq_short = eng.sequence(sid_short)
    steps = 0
    while not seq_short.is_finished(eng.eos_id) and steps < 50:
        eng.step()
        steps += 1

    assert seq_short.is_finished(eng.eos_id), (
        f"Short request should finish without waiting for long prefill, "
        f"got status {seq_short.status} after {steps} steps"
    )

    # Clean up
    seq_long = eng.sequence(sid_long)
    while not seq_long.is_finished(eng.eos_id) and steps < 100:
        eng.step()
        steps += 1
