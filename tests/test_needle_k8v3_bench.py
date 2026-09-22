# SPDX-License-Identifier: MIT
"""Regression tests for the bounded Qwen3.8 needle qualification harness."""

from __future__ import annotations

import torch

from bench.needle_k8v3 import (
    HAYSTACK,
    NEEDLE,
    QUESTION,
    _embedding_needs_transpose,
    _has_deep_mtp_head,
    build_prompt,
    cache_capacity,
    chunked_prefill,
    greedy_from_prefill,
)


class _Cache:
    def __init__(self):
        self.capacity_calls = []

    def ensure_capacity(self, slots, lengths):
        self.capacity_calls.append((list(slots), list(lengths)))


class _LinearCache:
    def __init__(self):
        self.cleared = []
        self.bound = []

    def clear_slot(self, slot):
        self.cleared.append(slot)

    def bind(self, slots):
        self.bound.append(list(slots))


class _Model:
    def __init__(self):
        self.calls = []

    def __call__(self, ids, positions, ctx):
        self.calls.append(
            {
                "ids": ids.tolist(),
                "positions": positions.tolist(),
                "prefill_start": ctx.prefill_start,
                "prefill_length": ctx.prefill_length,
                "slots": list(ctx.slots),
            }
        )
        return torch.zeros((1, ids.shape[1], 4), dtype=torch.float16)


def test_chunked_prefill_bounds_each_forward_and_preserves_positions():
    model = _Model()
    cache = _Cache()
    linear = _LinearCache()

    hidden = chunked_prefill(
        model, cache, linear, slot=3, prompt_ids=list(range(10)), device="cpu", chunk_size=4
    )

    assert cache.capacity_calls == [([3], [10])]
    assert linear.cleared == [3]
    assert linear.bound == [[3]]
    assert [c["ids"][0] for c in model.calls] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert [c["positions"][0] for c in model.calls] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9],
    ]
    assert [c["prefill_length"] for c in model.calls] == [4, 8, 10]
    assert all(c["prefill_start"] == 0 and c["slots"] == [3] for c in model.calls)
    assert hidden.shape == (1, 1, 4)


def test_chunked_prefill_rejects_non_positive_chunk_size():
    for chunk_size in (0, -1):
        try:
            chunked_prefill(_Model(), _Cache(), _LinearCache(), 0, [1], "cpu", chunk_size)
        except ValueError as exc:
            assert "chunk_size" in str(exc)
        else:
            raise AssertionError("non-positive chunk_size must be rejected")


def test_build_prompt_is_exactly_the_requested_context():
    class _Tokenizer:
        _ids = {HAYSTACK: [1, 2, 3], NEEDLE: [9, 9], QUESTION: [8]}

        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return self._ids[text]

    prompt = build_prompt(_Tokenizer(), 10)

    assert len(prompt) == 10
    assert prompt[3:5] == [9, 9]


def test_embedding_transpose_guard_ignores_native_quantized_objects():
    assert not _embedding_needs_transpose(object(), hidden_size=4)
    assert _embedding_needs_transpose(torch.empty(4, 8), hidden_size=4)
    assert not _embedding_needs_transpose(torch.empty(8, 4), hidden_size=4)


def test_deep_mtp_detection_rejects_shallow_shared_head():
    assert not _has_deep_mtp_head({"model.mtp.0.eh_proj.weight": object()})
    assert _has_deep_mtp_head({"model.mtp.0.fc.weight": object()})


def test_prefill_hidden_predicts_the_first_generated_token():
    class _LogitModel:
        def compute_logits(self, hidden):
            assert hidden.shape == (1, 4)
            return torch.tensor([[0.0, 3.0, 1.0]])

    assert greedy_from_prefill(_LogitModel(), torch.zeros((1, 1, 4))) == 1


def test_cache_capacity_reserves_generated_tokens():
    assert cache_capacity(32768, 64) == 32832
