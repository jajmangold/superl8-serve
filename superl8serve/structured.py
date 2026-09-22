# SPDX-License-Identifier: MIT
"""Constrained decoding (issue #39): XGrammar as the backend behind the sampler's
per-step logit-processor hook (`superl8serve.layers.sampler`, issue #38). Compiles an
OpenAI `response_format={"type": "json_schema", ...}` schema, or a raw grammar
(GBNF/EBNF), into an XGrammar token-mask matcher, and exposes it as a
`LogitsProcessor`: `(input_ids, logits) -> logits`.

XGrammar's own `GrammarCompiler` already caches compiled grammars by schema/grammar
string; `GrammarCompilerCache` here caches the (expensive to build) `GrammarCompiler`
itself, one per tokenizer vocab. A `GrammarMatcher` carries per-sequence progress
through the grammar though, so each in-flight request gets its own
`GrammarLogitsProcessor` instance -- never share one across requests or sequences.
"""
from __future__ import annotations

import json

import torch

try:
    import xgrammar
except ImportError:  # pragma: no cover -- exercised via the missing-dependency path
    xgrammar = None


def xgrammar_available() -> bool:
    return xgrammar is not None


def _single_set_bit(bitmask: "torch.Tensor") -> int | None:
    """If exactly one bit is set across the XGrammar next-token bitmask (a packed
    ``[1, ceil(vocab/32)]`` int32 tensor, bit==1 => token allowed), return that token
    id; otherwise ``None`` (zero or >1 tokens allowed). Exactly one set bit total <=>
    exactly one nonzero word whose value is a power of two — cheap and exact, no
    full-vocab popcount."""
    words = bitmask[0]
    nz = torch.nonzero(words, as_tuple=False).flatten()
    if nz.numel() != 1:
        return None
    wi = int(nz[0])
    w = int(words[wi]) & 0xFFFFFFFF
    if w & (w - 1):  # more than one bit set within the word
        return None
    bit = (w & -w).bit_length() - 1
    return wi * 32 + bit


def _require_xgrammar() -> None:
    if xgrammar is None:
        raise RuntimeError(
            "response_format=json_schema / grammar needs the `xgrammar` package "
            "(pip install xgrammar, or `pip install -e '.[structured]'`)."
        )


class GrammarLogitsProcessor:
    """A `LogitsProcessor` backed by one `xgrammar.GrammarMatcher`. Stateful and
    scoped to a single request: construct a fresh instance per generation (via
    `GrammarCompilerCache.for_json_schema` / `.for_grammar`), never reuse across
    sequences -- the matcher's progress through the grammar IS that sequence's
    generation state.
    """

    def __init__(self, compiled_grammar, tokenizer=None) -> None:
        # A generous rollback budget: the spec-decode drafter/verify walk speculatively
        # accepts up to (base + k drafts) tokens then rolls them all back, so the
        # matcher must retain that many undo steps (the default is small).
        self._matcher = xgrammar.GrammarMatcher(compiled_grammar, max_rollback_tokens=64)
        self._seen: int | None = None
        self._tokenizer = tokenizer  # for jump-forward string -> token drafting
        vocab_size = compiled_grammar.tokenizer_info.vocab_size
        # XGrammar fills the mask on CPU; `apply_token_bitmask_inplace` requires it on
        # the SAME device as the logits, so we keep a per-device cached copy (`_apply`)
        # — GPU logits (the engine) need a CUDA bitmask, CPU logits (tests) the CPU one.
        self._bitmask = xgrammar.allocate_token_bitmask(1, vocab_size)
        self._bitmask_dev: dict[str, "torch.Tensor"] = {}

    def _apply(self, batched: torch.Tensor) -> None:
        """Apply the freshly-filled mask to a ``[1, vocab]`` logits tensor in place,
        moving the mask onto the logits' device (cached per device)."""
        dev = batched.device
        if dev.type == "cpu":
            bm = self._bitmask
        else:
            key = str(dev)
            bm = self._bitmask_dev.get(key)
            if bm is None:
                bm = self._bitmask.to(dev)
                self._bitmask_dev[key] = bm
            else:
                bm.copy_(self._bitmask)
        xgrammar.apply_token_bitmask_inplace(batched, bm)

    def __call__(self, input_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        # `input_ids` is this sequence's full history so far (prompt + generated).
        # The grammar only constrains the generated continuation, so the first call
        # (at the prompt's last position, before any token is generated) just
        # anchors `_seen`; every later call advances the matcher by the one token
        # sampled since the previous step.
        if self._seen is None:
            self._seen = len(input_ids)
        else:
            for token_id in input_ids[self._seen:]:
                self._matcher.accept_token(token_id)
            self._seen = len(input_ids)

        # Once the grammar is satisfied (its stop token accepted), the matcher is
        # terminated and has no next-token mask — leave logits unconstrained so the
        # engine's own EOS / max_tokens logic ends the sequence (querying a terminated
        # matcher raises). In practice the stop token IS EOS, so this rarely fires.
        if self._matcher.is_terminated():
            return logits
        self._matcher.fill_next_token_bitmask(self._bitmask)
        batched = logits.unsqueeze(0).clone()
        self._apply(batched)
        return batched.squeeze(0)

    # ---- GrammarView: the non-destructive walk interface the spec-decode -------
    # tier-0 grammar drafter (`superl8serve.engine.drafters.GrammarDrafter`) and the
    # grammar-masked verify path (`EngineRunner._spec_decode_eager`) drive directly,
    # bypassing the per-step `__call__` sampler hook. `_seen` is the SHARED
    # committed-generation watermark for both paths, so a grammar request can move
    # between plain `__call__` decode and spec-decode without double-accepting: the
    # matcher's accepted-token count always equals `_seen - (prompt length)`.

    def sync(self, input_ids: list[int]) -> None:
        """Advance the matcher so it reflects every committed token in ``input_ids``
        (prompt + generated so far), exactly as ``__call__`` would. Idempotent: only
        the tokens past ``_seen`` are accepted. Call before masking/drafting a step."""
        if self._seen is None:
            self._seen = len(input_ids)
            return
        for token_id in input_ids[self._seen:]:
            self._matcher.accept_token(token_id)
        self._seen = len(input_ids)

    def mark_committed(self, n: int) -> None:
        """Set the committed-generation watermark to ``n`` (== len of the sequence's
        full token history after this step). The spec verify loop has already
        ``advance``-d the matcher over the committed tokens; this only records how
        many are committed so a later ``sync``/``__call__`` won't re-accept them."""
        self._seen = n

    def advance(self, token_id: int) -> bool:
        """Accept one token, stepping the grammar walk forward. Returns ``False`` if
        the token is not grammar-legal at the current position (walk unchanged)."""
        return bool(self._matcher.accept_token(token_id))

    def rewind(self, n: int) -> None:
        """Undo the last ``n`` :meth:`advance` steps (restore the walk state)."""
        if n > 0:
            self._matcher.rollback(n)

    def next_singleton(self) -> int | None:
        """The single grammar-legal token id at the current walk position, or ``None``
        when zero or more-than-one tokens are legal (a grammar *branch*). This is the
        forced-token test the tier-0 drafter rides: a singleton position is one the
        grammar decides with certainty, so it is committable with no model forward."""
        if self._matcher.is_terminated():
            return None
        self._matcher.fill_next_token_bitmask(self._bitmask)
        return _single_set_bit(self._bitmask)

    def forced_run(self, k: int) -> list[int]:
        """Up to ``k`` grammar-FORCED continuation tokens from the current position,
        ADVANCING the walk over them (the caller rewinds). Two forcing sources, tried
        per position:

          1. **token singleton** — exactly one vocab token is legal (rare on a BPE
             vocab, but exact);
          2. **jump-forward string** — XGrammar's ``find_jump_forward_string`` returns
             the string the grammar forces regardless of tokenization (the structural
             ``\"name\": \"`` / ``, \"`` / ``}`` runs of JSON & tool-calls). We encode it
             with the tokenizer and accept the greedy split; each token is checked with
             ``accept_token`` so only grammar-legal tokens are drafted. This is where
             the real free acceptance-length on structured output comes from.

        Empty when the grammar branches (a free value position) or is terminated. Never
        raises — an illegal encoded token just ends the run (verify re-checks anyway)."""
        out: list[int] = []
        while len(out) < k:
            if self._matcher.is_terminated():
                break
            self._matcher.fill_next_token_bitmask(self._bitmask)
            tid = _single_set_bit(self._bitmask)
            if tid is not None:
                if not self._matcher.accept_token(tid):
                    break
                out.append(tid)
                continue
            if self._tokenizer is None:
                break
            s = self._matcher.find_jump_forward_string()
            if not s:
                break
            progressed = False
            for t in self._encode(s):
                if len(out) >= k:
                    break
                if not self._matcher.accept_token(t):
                    break
                out.append(t)
                progressed = True
            if not progressed:
                break
        return out

    def _encode(self, s: str) -> list[int]:
        try:
            return self._tokenizer.encode(s, add_special_tokens=False)
        except TypeError:  # tokenizers without the kwarg
            return self._tokenizer.encode(s)

    def mask_row(self, logits_row: torch.Tensor) -> None:
        """Apply the current position's grammar mask to a single ``[vocab]`` logits
        row IN PLACE (illegal tokens -> -inf), so a later argmax is the grammar-masked
        greedy choice — bit-identical to what ``__call__`` + argmax would pick. A
        terminated matcher has no mask (grammar done) — leave the row unconstrained."""
        if self._matcher.is_terminated():
            return
        self._matcher.fill_next_token_bitmask(self._bitmask)
        self._apply(logits_row.unsqueeze(0))


class GrammarCompilerCache:
    """One `xgrammar.GrammarCompiler` per tokenizer (building it walks the whole
    vocab, so it's cached); hands out a fresh per-request `GrammarLogitsProcessor`
    for a JSON schema or a raw grammar string."""

    def __init__(self) -> None:
        self._compilers: dict[int, "xgrammar.GrammarCompiler"] = {}

    def _compiler(self, tokenizer) -> "xgrammar.GrammarCompiler":
        _require_xgrammar()
        key = id(tokenizer)
        compiler = self._compilers.get(key)
        if compiler is None:
            tokenizer_info = xgrammar.TokenizerInfo.from_huggingface(tokenizer)
            compiler = xgrammar.GrammarCompiler(tokenizer_info)
            self._compilers[key] = compiler
        return compiler

    def for_json_schema(self, tokenizer, schema: str | dict) -> GrammarLogitsProcessor:
        if isinstance(schema, dict):
            schema = json.dumps(schema)
        compiled = self._compiler(tokenizer).compile_json_schema(schema)
        return GrammarLogitsProcessor(compiled, tokenizer)

    def for_json_object(self, tokenizer) -> GrammarLogitsProcessor:
        compiled = self._compiler(tokenizer).compile_builtin_json_grammar()
        return GrammarLogitsProcessor(compiled, tokenizer)

    def for_grammar(self, tokenizer, grammar: str) -> GrammarLogitsProcessor:
        compiled = self._compiler(tokenizer).compile_grammar(grammar)
        return GrammarLogitsProcessor(compiled, tokenizer)
