# SPDX-License-Identifier: MIT
"""Regression tests for the two-card pipeline-parallel needle gate harness
(superl8-serve#440).

CPU-only tests cover prompt construction, cache sizing, stage geometry
(full-attention layer sets per stage), boundary pack/unpack round-trips, and
output hashing. The two-GPU identity gate is a small real-pipeline test; the
full 64-layer semantic comparison runs in qualification, not CI.
"""

from __future__ import annotations

import torch

from bench.needle_k8v3 import (
    ANSWER,
    HAYSTACK,
    NEEDLE,
    QUESTION,
    build_prompt,
    cache_capacity,
)
from bench.needle_k8v3_tp import (
    sha256_ids,
    stage_full_attn_layers,
)
from superl8serve.dist.pipeline import _pack_boundary, _unpack_boundary


def _full_attn_local(cfg, offset, count):
    return stage_full_attn_layers(cfg, offset, count)


class _Cfg:
    def __init__(self, num_hidden_layers=64):
        self.num_hidden_layers = num_hidden_layers

    def attention_kind(self, i):
        # Qwen3.5 hybrid: 3 DeltaNet : 1 gated full attention.
        return "full" if (i + 1) % 4 == 0 else "linear"


def test_full_attn_layer_sets_per_stage():
    """64-layer model split 32/32: each stage's local full-attention layer set
    is exactly the 8 layers the K8V3 cache serves, and the global sets cover
    the complete 16-layer KV-bearing set."""
    cfg = _Cfg()
    global_full = [i for i in range(64) if cfg.attention_kind(i) == "full"]
    assert global_full == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]

    s0 = _full_attn_local(cfg, 0, 32)
    s1 = _full_attn_local(cfg, 32, 32)
    assert s0 == [3, 7, 11, 15, 19, 23, 27, 31]
    assert s1 == [3, 7, 11, 15, 19, 23, 27, 31]
    # Stage-local indices map back onto the global KV-bearing layer set.
    assert [i + 32 for i in s1] == [35, 39, 43, 47, 51, 55, 59, 63]
    assert sorted([i for i in s0] + [i + 32 for i in s1]) == global_full


def test_stage_geometry_covers_every_layer_exactly_once():
    cfg = _Cfg()
    n = cfg.num_hidden_layers
    bounds = [i * n // 2 for i in range(3)]
    assert bounds == [0, 32, 64]
    assert sorted(list(range(bounds[0], bounds[1])) + list(range(bounds[1], bounds[2]))) == list(range(64))
    assert len(range(bounds[0], bounds[1])) == 32
    assert len(range(bounds[1], bounds[2])) == 32


def test_prompt_construction_matches_reference():
    class _Tok:
        """Deterministic tokenizer: each distinct string maps to a fixed,
        unique range of token ids (one id per whitespace word), so the
        needle/haystack/question spans are stable across repeated encodes
        (build_prompt re-encodes its three pieces on the same tokenizer)."""

        def __init__(self):
            self._ranges = {}
            self._counter = 0

        def encode(self, s, add_special_tokens=False):
            if s not in self._ranges:
                n = len(s.split())
                self._ranges[s] = list(range(self._counter, self._counter + n))
                self._counter += n
            return self._ranges[s]

        eos_token_id = -1

    tok = _Tok()
    hay = tok.encode(HAYSTACK)  # 9 tokens
    needle = tok.encode(NEEDLE)  # 5 tokens
    q = tok.encode(QUESTION)  # 22 tokens
    ctx = 128
    prompt = build_prompt(tok, ctx)
    assert len(prompt) == ctx
    # The needle sits at exactly 50% depth and the question closes the prompt.
    half = (ctx - len(needle) - len(q)) // 2
    assert prompt[half:half + len(needle)] == needle
    assert prompt[-len(q):] == q
    assert prompt.count(needle[0]) == 1  # the passphrase sentence appears once
    assert len(build_prompt(_Tok(), 1024)) == 1024
    # Ceiling fill: a non-divisible budget still lands exactly on the target ctx.
    assert len(build_prompt(_Tok(), 1001)) == 1001
    assert hay == list(range(len(HAYSTACK.split())))  # first range is the haystack


def test_cache_capacity_includes_generated_tokens():
    assert cache_capacity(32768, 64) == 32832
    assert cache_capacity(2048, 64) == 2112
    try:
        cache_capacity(0, 64)
        raise AssertionError("ctx=0 must be rejected")
    except ValueError:
        pass


def test_output_hash_deterministic_and_content_bound():
    ids = [1, 2, 3, 4]
    assert sha256_ids(ids) == sha256_ids(list(ids))
    assert sha256_ids([1, 2, 3]) != sha256_ids([1, 2, 3, 4])


def test_answer_needle_constant():
    # The recall check is exact-substring of the planted passphrase.
    assert ANSWER == "K8V3NEEDLE7"
    assert ANSWER in (NEEDLE + QUESTION)


def test_boundary_pack_unpack_roundtrip():
    torch.manual_seed(0)
    hidden = torch.randn(1, 256, 4096, dtype=torch.float16)
    residual = torch.randn(1, 256, 4096, dtype=torch.float16)
    packed = _pack_boundary(hidden, residual)
    assert packed.shape == (1, 256, 8192)
    h2, r2 = _unpack_boundary(packed, 4096)
    assert torch.equal(h2, hidden)
    assert torch.equal(r2, residual)
    # residual=None packs to bare hidden and unpacks to None.
    packed2 = _pack_boundary(hidden, None)
    h3, r3 = _unpack_boundary(packed2, 4096)
    assert torch.equal(h3, hidden)
    assert r3 is None
