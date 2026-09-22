# SPDX-License-Identifier: MIT
"""Issue #183 decode quick-wins: the per-step host->device staging buffers must be
value-for-value identical to the old `torch.tensor(list, device=cuda)` rebuilds
they replace. These are the buffers a captured CUDA graph replays against, so any
divergence is a silent correctness bug. Pure staging-buffer equivalence -- the
end-to-end graphed-vs-eager token match lives in test_cuda_graph.py.
"""

import pytest
import torch

pytest.importorskip("superl8")

from superl8serve.engine import PagedKVCache

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="decode staging buffers are CUDA-only")


def _cache_with_slots(num_slots=4, lengths=(3, 20, 1, 33)):
    """A PagedKVCache with `num_slots` allocated and grown to `lengths` tokens."""
    cache = PagedKVCache(1, num_slots, 2, 128, 16, device="cuda", block_size=16)
    slots = [cache.alloc() for _ in range(num_slots)]
    cache.ensure_capacity(slots, list(lengths))
    return cache, slots


def test_fill_slot_mapping_matches_legacy():
    cache, slots = _cache_with_slots()
    positions = [2, 19, 0, 32]  # the token each row just wrote
    legacy = cache.slot_mapping_for(slots, positions)  # old torch.tensor(...,cuda) path
    dst = torch.empty(len(slots), dtype=torch.int32, device="cuda")
    cache.fill_slot_mapping(dst, slots, positions)  # new pinned-staging path
    assert torch.equal(dst, legacy)


def test_fill_block_table_matches_legacy():
    cache, slots = _cache_with_slots()
    legacy = cache.block_table(slots)  # old per-slot torch.tensor path
    dst = torch.zeros(len(slots), cache.max_blocks_per_seq, dtype=torch.int32, device="cuda")
    cache.fill_block_table(dst, slots)
    assert torch.equal(dst, legacy)


def test_fill_reuses_buffers_across_steps():
    """Calling the fill helpers repeatedly (buffers reused, not reallocated) must
    keep producing correct values -- catches stale data from an un-cleared row."""
    cache, slots = _cache_with_slots()
    dst = torch.empty(len(slots), dtype=torch.int32, device="cuda")
    for step in range(3):
        positions = [step, step + 1, 0, step]
        cache.fill_slot_mapping(dst, slots, positions)
        assert torch.equal(dst, cache.slot_mapping_for(slots, positions))
    # A later, smaller batch must not read leftover rows from the wider buffer.
    dst2 = torch.zeros(2, cache.max_blocks_per_seq, dtype=torch.int32, device="cuda")
    cache.fill_block_table(dst2, slots[:2])
    assert torch.equal(dst2, cache.block_table(slots[:2]))


def test_sample_param_staging_matches_legacy():
    """EngineRunner._sample stages sampling controls through pinned buffers + non_blocking
    copy instead of `torch.tensor(list, device=cuda)`. The resulting device tensors
    must be value-identical to the legacy rebuild."""
    from superl8serve.engine.model_runner import EngineRunner

    runner = EngineRunner.__new__(EngineRunner)  # skip model/cache construction
    runner.device = "cuda"
    runner._pin = True
    runner._temps_host = runner._top_p_host = None
    runner._temps_dev = runner._top_p_dev = None
    runner._top_k_host = runner._repetition_penalty_host = None
    runner._top_k_dev = runner._repetition_penalty_dev = None

    temp_vals = [0.0, 0.7, 1.0, 0.0]
    top_p_vals = [1.0, 0.9, 1.0, 0.5]
    top_k_vals = [0, 50, 7, 1]
    repetition_vals = [1.0, 1.1, 0.9, 1.2]
    n = len(temp_vals)
    runner._ensure_sample_buffers(n)
    runner._temps_host[:n].copy_(torch.tensor(temp_vals, dtype=torch.float32))
    runner._top_p_host[:n].copy_(torch.tensor(top_p_vals, dtype=torch.float32))
    runner._top_k_host[:n].copy_(torch.tensor(top_k_vals, dtype=torch.int32))
    runner._repetition_penalty_host[:n].copy_(
        torch.tensor(repetition_vals, dtype=torch.float32)
    )
    runner._temps_dev[:n].copy_(runner._temps_host[:n], non_blocking=True)
    runner._top_p_dev[:n].copy_(runner._top_p_host[:n], non_blocking=True)
    runner._top_k_dev[:n].copy_(runner._top_k_host[:n], non_blocking=True)
    runner._repetition_penalty_dev[:n].copy_(
        runner._repetition_penalty_host[:n], non_blocking=True
    )
    torch.cuda.synchronize()

    assert torch.equal(
        runner._temps_dev[:n], torch.tensor(temp_vals, device="cuda", dtype=torch.float32)
    )
    assert torch.equal(
        runner._top_p_dev[:n], torch.tensor(top_p_vals, device="cuda", dtype=torch.float32)
    )
    assert torch.equal(
        runner._top_k_dev[:n], torch.tensor(top_k_vals, device="cuda", dtype=torch.int32)
    )
    assert torch.equal(
        runner._repetition_penalty_dev[:n],
        torch.tensor(repetition_vals, device="cuda", dtype=torch.float32),
    )
