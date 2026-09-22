# SPDX-License-Identifier: MIT
"""Tier-0 grammar spec-decode drafter: singleton (forced-token) detection, the free
forced-run walk, non-destructive rewind, and the grammar → n-gram → MTP cascade
fall-through. CPU-only, no model and no xgrammar — the drafter is decoupled from the
XGrammar backend behind the tiny :class:`GrammarView` walk interface, so a fake grammar
(an explicit token-transition trie) exercises every path here.
"""

from __future__ import annotations

from superl8serve.engine.drafters import (
    GrammarDrafter,
    NgramDrafter,
    cascade_draft,
    drafter_config,
)


class FakeGrammarView:
    """A grammar as an explicit token-transition trie: ``allowed`` maps a committed
    path (tuple of accepted tokens) to the set of grammar-legal next tokens. Implements
    the GrammarView walk (``advance`` / ``rewind`` / ``next_singleton``) the same way
    ``structured.GrammarLogitsProcessor`` does over an XGrammar matcher, so the drafter
    can't tell the difference. A position with exactly one legal token is *forced*."""

    def __init__(self, allowed: dict[tuple[int, ...], set[int]]):
        self.allowed = allowed
        self.path: list[int] = []

    def _legal(self) -> set[int]:
        return self.allowed.get(tuple(self.path), set())

    def advance(self, token_id: int) -> bool:
        if token_id not in self._legal():
            return False
        self.path.append(token_id)
        return True

    def rewind(self, n: int) -> None:
        for _ in range(n):
            self.path.pop()

    def next_singleton(self) -> int | None:
        legal = self._legal()
        return next(iter(legal)) if len(legal) == 1 else None

    # unused by the drafter, present so _grammar_view() would recognise it
    def sync(self, input_ids):  # pragma: no cover
        pass

    def mask_row(self, row):  # pragma: no cover
        pass


# A JSON-ish forced run: after base_tok 100, the grammar forces 101,102,103 (think
# `"`, a field-name piece, `"`, `:`), then BRANCHES (a value can be many tokens).
FORCED_THEN_BRANCH = {
    (): {100},
    (100,): {101},
    (100, 101): {102},
    (100, 101, 102): {103},
    (100, 101, 102, 103): {104, 105, 106},  # branch -> forced run ends here
}


def test_singleton_forced_run_is_drafted_for_free():
    d = GrammarDrafter(FakeGrammarView(FORCED_THEN_BRANCH), max_k=8)
    assert d.propose(base_tok=100, k=8) == [101, 102, 103]


def test_forced_run_capped_at_k():
    d = GrammarDrafter(FakeGrammarView(FORCED_THEN_BRANCH), max_k=8)
    assert d.propose(base_tok=100, k=2) == [101, 102]


def test_branch_immediately_after_base_returns_empty():
    # base_tok 100 legal, but the position after it allows >1 token -> no free draft.
    view = FakeGrammarView({(): {100}, (100,): {201, 202}})
    assert GrammarDrafter(view, max_k=8).propose(base_tok=100, k=8) == []


def test_illegal_base_token_returns_empty():
    view = FakeGrammarView({(): {100}, (100,): {101}})
    assert GrammarDrafter(view, max_k=8).propose(base_tok=999, k=8) == []


def test_propose_is_non_destructive():
    # The drafter only PEEKS: after propose the walk must be exactly where it started.
    view = FakeGrammarView(FORCED_THEN_BRANCH)
    view.advance(100)  # pretend base already committed at some prior point
    before = list(view.path)
    GrammarDrafter(view, max_k=8).propose(base_tok=101, k=8)  # walks 101->102->103...
    # propose advances then rewinds by the same count -> path unchanged.
    assert view.path == before


def test_zero_k_returns_empty():
    d = GrammarDrafter(FakeGrammarView(FORCED_THEN_BRANCH), max_k=8)
    assert d.propose(base_tok=100, k=0) == []


def test_none_view_returns_empty():
    assert GrammarDrafter(None, max_k=8).propose(base_tok=100, k=8) == []


# ---- cascade: grammar tier-0 -> n-gram -> MTP -------------------------------------


def test_cascade_grammar_hit_skips_ngram_and_mtp():
    gd = GrammarDrafter(FakeGrammarView(FORCED_THEN_BRANCH), max_k=8)
    ng_hits, mtp_hits = [], []

    class SpyNgram:
        def propose(self, tokens, k):
            ng_hits.append(1)
            return [7, 7]

    def mtp():
        mtp_hits.append(1)
        return [9]

    # ctx ends with base_tok 100; grammar forces 101,102,103 for free.
    out = cascade_draft(SpyNgram(), mtp, tokens=[1, 2, 100], k=8, grammar=gd)
    assert out == [101, 102, 103]
    assert ng_hits == [] and mtp_hits == []  # free tier hit -> lower tiers never run


def test_cascade_grammar_branch_falls_through_to_ngram():
    # base_tok 100 branches immediately -> grammar yields, n-gram carries the step.
    gd = GrammarDrafter(FakeGrammarView({(): {100}, (100,): {5, 6}}), max_k=8)
    # tokens repeat "100" earlier followed by 42,43 so n-gram proposes that run.
    ng = NgramDrafter(min_n=1, max_n=2, max_k=8)
    tokens = [100, 42, 43, 99, 100]
    out = cascade_draft(ng, lambda: [777], tokens=tokens, k=4, grammar=gd)
    assert out == [42, 43, 99, 100]  # n-gram's continuation of the earlier "100"


def test_cascade_grammar_branch_and_ngram_miss_falls_to_mtp():
    gd = GrammarDrafter(FakeGrammarView({(): {100}, (100,): {5, 6}}), max_k=8)
    ng = NgramDrafter(min_n=2, max_n=3, max_k=8)  # last (1,100) never occurred before
    out = cascade_draft(ng, lambda: [777], tokens=[1, 100], k=4, grammar=gd)
    assert out == [777]


def test_cascade_no_grammar_is_unchanged_ngram_then_mtp():
    # grammar=None (unconstrained sequence): identical to the 2-tier n-gram cascade.
    ng = NgramDrafter(min_n=1, max_n=2, max_k=8)
    tokens = [5, 6, 7, 5]
    assert cascade_draft(ng, lambda: [0], tokens=tokens, k=4, grammar=None) == [6, 7, 5]


def test_ngram_preflight_only_rejects_impossible_next_token_matches():
    ng = NgramDrafter(min_n=2, max_n=3)

    assert ng.could_match_after([10, 20, 30]) is False
    # If the target's next token is 2, appending it makes the suffix [1, 2]
    # match the earlier span and prompt lookup can propose its continuation.
    assert ng.could_match_after([1, 2, 3, 1]) is True


class ForcedRunView:
    """A view exposing the ``forced_run`` primitive (the XGrammar jump-forward path):
    it returns a MULTI-token forced run at once (advancing its walk), which the drafter
    prefers over the token-by-token singleton walk. Mirrors how the real processor
    drafts the ``\"name\": \"`` structural runs of JSON in one shot."""

    def __init__(self, base_legal, run):
        self.base_legal = base_legal
        self.run = run
        self.pos = 0  # advances counted so rewind restores exactly

    def advance(self, token_id):
        if token_id != self.base_legal:
            return False
        self.pos += 1
        return True

    def rewind(self, n):
        self.pos -= n

    def forced_run(self, k):
        out = self.run[:k]
        self.pos += len(out)
        return list(out)


def test_drafter_prefers_forced_run_primitive_when_present():
    v = ForcedRunView(base_legal=100, run=[101, 102, 103, 104, 105])
    d = GrammarDrafter(v, max_k=8)
    assert d.propose(base_tok=100, k=4) == [101, 102, 103, 104]
    assert v.pos == 0  # non-destructive: base(1) + run(4) advances all rewound


def test_forced_run_illegal_base_returns_empty():
    v = ForcedRunView(base_legal=100, run=[1, 2, 3])
    assert GrammarDrafter(v, max_k=8).propose(base_tok=7, k=4) == []
    assert v.pos == 0


def test_grammar_mode_parses_from_env(monkeypatch):
    monkeypatch.setenv("SUPERL8SERVE_SPEC_DRAFTER", "grammar")
    mode, k, min_n, max_n = drafter_config()
    assert mode == "grammar"
