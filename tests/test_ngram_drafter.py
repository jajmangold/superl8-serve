# SPDX-License-Identifier: MIT
"""Prompt-lookup n-gram drafter + the n-gram→MTP cascade (CPU-only, no model)."""

from __future__ import annotations

from superl8serve.engine.drafters import NgramDrafter, cascade_draft


def test_ngram_proposes_continuation_of_recent_match():
    # The pattern (11,12) recurs at the end; its earlier occurrence (index 1) was
    # followed by 13,99,... so those are proposed (the run is free to ride).
    d = NgramDrafter(min_n=2, max_n=3, max_k=8)
    toks = [10, 11, 12, 13, 99, 11, 12]
    assert d.propose(toks, k=4) == [13, 99, 11, 12]


def test_ngram_longest_match_wins():
    d = NgramDrafter(min_n=1, max_n=4, max_k=8)
    # Pattern "7 8 9" recurs; the most specific (n=3) match drives the continuation.
    toks = [7, 8, 9, 42, 43, 1, 2, 7, 8, 9]
    assert d.propose(toks, k=3) == [42, 43, 1]


def test_ngram_miss_returns_empty():
    d = NgramDrafter(min_n=2, max_n=3)
    assert d.propose([1, 2, 3, 4, 5], k=4) == []  # last "4 5" never occurred before


def test_ngram_respects_k_and_max_k():
    d = NgramDrafter(min_n=2, max_n=2, max_k=2)
    toks = [1, 2, 3, 4, 5, 1, 2]  # "1 2" recurs; continuation "3 4 5..." capped at max_k=2
    assert d.propose(toks, k=8) == [3, 4]


def test_ngram_structured_json_run():
    # A repeated structured key run: after the first `KEY = [ 40 41 42 ]` the same
    # `KEY =` recurs, so the drafter proposes the bracketed value run for free.
    KEY, EQ = 200, 201
    seq = [KEY, EQ, 40, 41, 42, 99, KEY, EQ]
    out = NgramDrafter(min_n=2, max_n=4, max_k=8).propose(seq, k=4)
    assert out == [40, 41, 42, 99]  # rides the repeated structured span


def test_cascade_prefers_ngram_then_falls_back():
    ng = NgramDrafter(min_n=2, max_n=3, max_k=8)
    hits = []

    def mtp():
        hits.append(1)
        return [777]

    # Hit: n-gram fires, MTP fallback never evaluated.
    toks = [5, 6, 7, 8, 9, 5, 6]  # (5,6) recurs; earlier occ followed by 7,8,9
    assert cascade_draft(ng, mtp, toks, k=4) == [7, 8, 9, 5]
    assert hits == []
    # Miss: n-gram empty -> MTP fallback runs.
    assert cascade_draft(ng, mtp, [1, 2, 3], k=4) == [777]
    assert hits == [1]


def test_cascade_ngram_disabled_uses_mtp():
    assert cascade_draft(None, lambda: [42], [1, 2, 3], k=4) == [42]
