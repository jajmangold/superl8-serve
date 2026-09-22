# SPDX-License-Identifier: MIT
"""Speculative-decode drafters — cheap proposers that feed the shared MTP verify
path. Three are cascaded (grammar first, then n-gram, MTP fallback) by the runner.

The verify forward is a single weight-stream over ``k+1`` tokens; acceptance-length
(how many drafts survive) is what divides that stream's cost across emitted tokens.
A drafter's only job is to make the accepted prefix as long as possible for free
(no full-model forward):

  * :class:`GrammarDrafter` — **tier-0, the free/perfect drafter.** When a
    grammar / structured-output constraint is active (JSON-schema, tool-call syntax,
    raw GBNF), at many positions the grammar permits **exactly one** token — the
    structural ones: ``{``, ``"``, field names, ``:``, ``,``, closing braces. Those
    are KNOWN with certainty, so they can be committed with NO draft-model forward at
    all: strictly better than n-gram on structured output. The drafter walks the
    grammar forward from ``base_tok``, emitting each singleton-forced token until the
    grammar branches (allows >1) — then it returns and the cascade falls through to
    n-gram / MTP for the free positions. Bit-identity is by construction: a
    singleton-forced token is exactly what plain grammar-constrained greedy decode
    would emit (its mask leaves one legal token); the shared verify path re-checks it
    against the SAME grammar-masked argmax anyway.

  * :class:`NgramDrafter` — prompt-lookup / n-gram. Matches the last ``n`` tokens
    against everything generated so far and proposes the continuation that followed
    the most recent earlier match. On structured / repetitive spans (code, tool-call
    JSON, RAG quotes, edits) this proposes a long correct run at zero model cost, so
    acceptance-length climbs well past MTP's depth-1 (~2). A pure table lookup — no
    parameters, no device work.

  * The MTP head (``model.mtp``) is the fallback: on non-repetitive prose the n-gram
    lookup misses and the learned depth-1 head still lands ~90%. The engine runner
    cascades all three: grammar (free forced run) → n-gram → MTP (:func:`cascade_draft`).
"""

from __future__ import annotations

import os


class NgramDrafter:
    """Prompt-lookup n-gram drafter (Saxena 2023, "prompt lookup decoding").

    ``propose(tokens, k)`` returns up to ``k`` continuation tokens by finding the
    most recent earlier occurrence of the last ``n`` tokens (``n`` swept high→low so
    the longest, most specific match wins) and returning what followed it. Empty
    list on a miss. The match is searched over the sequence's OWN tokens so far
    (prompt + generated) — the span that actually repeats in code/JSON/editing.
    """

    def __init__(self, min_n: int = 2, max_n: int = 3, max_k: int = 8):
        # min_n: shortest pattern we trust (n==1 matches far too loosely and drafts
        #   noise); max_n: longest pattern we bother trying (diminishing returns).
        # max_k: cap on proposed continuation length (the verify cost ceiling).
        self.min_n = max(1, min_n)
        self.max_n = max(self.min_n, max_n)
        self.max_k = max_k

    def propose(self, tokens: list[int], k: int) -> list[int]:
        """Return up to ``min(k, max_k)`` draft tokens continuing ``tokens``.

        Sweeps pattern length ``n`` from ``max_n`` down to ``min_n``; for each, finds
        the LAST index ``i`` (most recent) with ``tokens[i:i+n] == tokens[-n:]`` and
        ``i+n < len(tokens)`` (so a continuation exists) and returns
        ``tokens[i+n : i+n+k]``. Longest match first = most context = best drafts."""
        k = min(k, self.max_k)
        if k <= 0:
            return []
        L = len(tokens)
        for n in range(min(self.max_n, L - 1), self.min_n - 1, -1):
            pattern = tokens[-n:]
            # Scan backward for the most recent earlier occurrence (skip the trailing
            # pattern itself: search window ends at L-n-1's start, i.e. i <= L-n-1).
            for i in range(L - n - 1, -1, -1):
                if tokens[i : i + n] == pattern:
                    cont = tokens[i + n : i + n + k]
                    if cont:
                        return cont
                    break  # match with no continuation room; try a shorter n
        return []

    def could_match_after(self, tokens: list[int]) -> bool:
        """Whether *some* next token could complete a usable n-gram match.

        The target model has not produced that token yet, but the preceding
        ``n-1`` suffix is already known. If it never appeared earlier with a token
        after it, no possible target token can make :meth:`propose` succeed, so the
        runner can stay on ordinary graphed decode without an eager probe.
        """
        L = len(tokens)
        for n in range(self.min_n, min(self.max_n, L + 1) + 1):
            prefix_len = n - 1
            suffix = tokens[-prefix_len:] if prefix_len else []
            for i in range(0, L - prefix_len):
                if tokens[i : i + prefix_len] == suffix:
                    return True
        return False


class GrammarDrafter:
    """Tier-0 spec-decode drafter: the active grammar IS the drafter.

    Wraps a :class:`GrammarView` — the small non-destructive walk interface a
    grammar/structured-output matcher exposes (:class:`superl8serve.structured.\
GrammarLogitsProcessor` implements it over an XGrammar ``GrammarMatcher``):

      * ``advance(token_id) -> bool`` — accept one token, advancing the walk one
        position; ``False`` if the token is not grammar-legal.
      * ``next_singleton() -> int | None`` — the sole grammar-legal token at the
        current position, or ``None`` when zero or >1 tokens are legal (a branch).
      * ``rewind(n)`` — undo the last ``n`` ``advance`` calls (restore state).

    ``propose(base_tok, k)`` walks forward from ``base_tok`` (the first verify token,
    already committed this step) collecting the singleton-forced continuation, then
    rewinds so the view is left exactly where it started — the drafter NEVER mutates
    the sequence's grammar state, it only peeks. Returns ``[]`` the moment the grammar
    branches (so the cascade falls through to n-gram / MTP for that free position).
    """

    def __init__(self, view, max_k: int = 8):
        self.view = view
        self.max_k = max_k

    def propose(self, base_tok: int, k: int) -> list[int]:
        k = min(k, self.max_k)
        if k <= 0 or self.view is None:
            return []
        g = self.view
        # Speculatively step over base_tok (the first verify token, already committed),
        # then collect the grammar-forced continuation. `steps` counts advances so the
        # finally-clause restores the EXACT starting state even if a walk step is
        # rejected or an exception fires — the drafter only peeks, never mutates.
        if not g.advance(base_tok):
            return []
        forced: list[int] = []
        steps = 1
        try:
            # Prefer the view's own forced-run primitive (XGrammar jump-forward: it
            # captures the MULTI-token structural runs of JSON/tool-calls, where exact
            # token singletons are rare on a BPE vocab). Fall back to the generic
            # singleton walk (used by the CPU fake-grammar tests, and any view without
            # jump-forward): draft one forced token per position until the grammar
            # branches. Either way `forced` holds only grammar-legal tokens.
            if hasattr(g, "forced_run"):
                forced = g.forced_run(k)
                steps += len(forced)
            else:
                while len(forced) < k:
                    tid = g.next_singleton()
                    if tid is None:  # grammar branches here — stop the free run
                        break
                    if not g.advance(tid):
                        break
                    forced.append(tid)
                    steps += 1
        finally:
            g.rewind(steps)
        return forced


def cascade_draft(ngram, mtp_fallback, tokens, k, *, grammar=None):
    """Cascade: grammar (free forced run) → n-gram → MTP head.

    ``grammar`` is a :class:`GrammarDrafter` (or None when no constraint is active);
    when present it proposes the singleton-forced continuation of ``tokens[-1]`` (the
    first verify token / ``base_tok``) for free — strictly the best drafter on the
    forced structural positions of JSON / tool-call output. On a grammar branch it
    returns ``[]`` and the cascade falls through: ``ngram`` (a :class:`NgramDrafter`,
    or None to skip straight to MTP), then ``mtp_fallback`` — a zero-arg callable
    returning the MTP head's draft list, evaluated lazily so the head's forward is
    skipped whenever a free tier hits. Returns the draft token list (possibly empty)."""
    if grammar is not None and tokens:
        drafts = grammar.propose(tokens[-1], k)
        if drafts:
            return drafts
    if ngram is not None:
        drafts = ngram.propose(tokens, k)
        if drafts:
            return drafts
    return mtp_fallback() if mtp_fallback is not None else []


def drafter_config():
    """Read spec-decode drafter config from the environment (engine kwargs override).

    SUPERL8SERVE_SPEC_DRAFTER: ``mtp`` (default) | ``grammar`` | ``ngram`` | ``cascade``.
                            ``cascade`` runs all tiers (grammar → n-gram → MTP);
                            ``grammar`` runs ONLY the tier-0 grammar drafter (free
                            forced runs, no draft-model forward ever) — the isolated
                            structured-output tier; ``ngram`` / ``mtp`` select a single
                            fallback tier. The grammar tier only engages on sequences
                            that actually carry a grammar/structured constraint; it is
                            a silent no-op on unconstrained sequences.

                            DEFAULT IS ``mtp``, NOT ``cascade`` -- measured on
                            V100/Qwen3.5-9B prose (superl8-serve bench/prefill_bucket_probe.py,
                            unconstrained summarization prompt, 2026-09-14): cascade's
                            n-gram tier has strict priority over MTP (``cascade_draft()``
                            returns the n-gram proposal whenever it's non-empty, MTP is
                            only tried on an n-gram MISS), so a cheap-but-often-wrong
                            n-gram guess preempts a pricier-but-more-often-right MTP
                            draft every time n-gram proposes *anything*. Measured:
                            mtp-only 32.6-34.5 tok/s decode / 70-73% accept; cascade
                            28.2-31.2 tok/s / 52-54% accept -- cascade is SLOWER than
                            MTP alone despite drafting more tokens/step, because the
                            extra drafts it wins are worse ones. Same regression class
                            already found and NOT deployed in the sibling llama.cpp
                            fork's ngram-cache + MTP combination (flight-deck memory:
                            qwen27b-ngram-cache-rs-seq-bug) -- priority-waterfall
                            arbitration between a cheap and an expensive drafter is a
                            known-bad pattern (see arXiv:2312.11462's Theorem 4.5: a
                            static "cheap first" order is provably wrong whenever the
                            cheap drafter's per-step acceptance is lower than the
                            expensive one's). ``cascade`` is left available for grammar-
                            heavy or highly-repetitive/verbatim-copy workloads where
                            n-gram's hit rate may genuinely beat MTP's, but it is no
                            longer the safe general-purpose default -- set it explicitly
                            if that's your workload.
    SUPERL8SERVE_SPEC_K:       max draft length (default 4). n-gram / grammar use up to
                            this; the MTP head is depth-1 so it caps the effective k at
                            1 on prose. Larger k helps structured workloads (long forced
                            runs), costs a few wasted verify slots on prose.
    SUPERL8SERVE_SPEC_NGRAM:   ``min_n,max_n`` (default ``2,3``).
    """
    mode = os.environ.get("SUPERL8SERVE_SPEC_DRAFTER", "mtp").lower()
    k = int(os.environ.get("SUPERL8SERVE_SPEC_K", "4"))
    ng = os.environ.get("SUPERL8SERVE_SPEC_NGRAM", "2,3")
    try:
        min_n, max_n = (int(x) for x in ng.split(","))
    except ValueError:
        min_n, max_n = 2, 3
    return mode, max(1, k), min_n, max_n
