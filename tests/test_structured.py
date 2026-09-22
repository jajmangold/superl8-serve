# SPDX-License-Identifier: MIT
"""Constrained-decoding tests (issue #39): XGrammar compiles a JSON schema or a
grammar (GBNF/EBNF) into a token-mask matcher; `GrammarLogitsProcessor` applies that
mask each step and advances the matcher on the token actually picked -- exactly the
`(input_ids, logits) -> logits` seam the sampler's logit-processor hook (#38, see
`tests/test_sampler.py`) already exercises.

Runs against a tiny synthetic vocabulary (a handful of single-character tokens plus
an EOS), not a real HF tokenizer/model -- same "no CUDA/superl8/model weights" spirit as
`tests/test_api.py`'s `FakeTokenizer`, and keeps these tests fast and deterministic:
greedy argmax over an all-zero logit row always picks the lowest-id allowed token, so
generation is reproducible without any actual model.
"""
from __future__ import annotations

import json

import pytest
import torch

xgrammar = pytest.importorskip("xgrammar")
jsonschema = pytest.importorskip("jsonschema")

from superl8serve.structured import GrammarCompilerCache, GrammarLogitsProcessor  # noqa: E402


def _tokenizer_info(vocab: list[str], eos_id: int) -> "xgrammar.TokenizerInfo":
    return xgrammar.TokenizerInfo(
        encoded_vocab=vocab, vocab_type=xgrammar.VocabType.RAW, stop_token_ids=[eos_id])


def _greedy_decode(processor: GrammarLogitsProcessor, vocab: list[str], eos_id: int,
                    max_steps: int = 128) -> str:
    """Simulates the engine's per-step call pattern (`model_runner.EngineRunner._sample`):
    the processor sees the full token history so far and returns masked logits; we
    pick greedy (argmax) and append, same as the sampler does at temperature 0."""
    rng = torch.Generator().manual_seed(42)
    input_ids: list[int] = []
    for _ in range(max_steps):
        logits = torch.rand(len(vocab), generator=rng) * 1e-6
        masked = processor(input_ids, logits)
        token_id = int(masked.argmax())
        if token_id == eos_id:
            return "".join(vocab[t] for t in input_ids)
        input_ids.append(token_id)
    raise AssertionError(f"did not terminate within {max_steps} steps: {input_ids!r}")


NESTED_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "address": {
            "type": "object",
            "properties": {
                "zip": {"type": "integer"},
                "active": {"type": "boolean"},
            },
            "required": ["zip", "active"],
        },
    },
    "required": ["name", "address"],
}


def test_json_schema_constrained_generation_parses_and_validates():
    """100%-parse guarantee: every token is masked to the compiled schema grammar, so
    the greedily-decoded output must be valid JSON that validates against the nested
    schema -- not just plausible-looking text."""
    vocab = list("{}[]\":,.-0123456789abcdefghijklmnopqrstuvwxyz ")
    eos_id = len(vocab)
    vocab = vocab + ["<eos>"]
    tokenizer_info = _tokenizer_info(vocab, eos_id)
    compiler = xgrammar.GrammarCompiler(tokenizer_info)
    compiled = compiler.compile_json_schema(json.dumps(NESTED_SCHEMA))
    processor = GrammarLogitsProcessor(compiled)

    text = _greedy_decode(processor, vocab, eos_id, max_steps=2048)

    instance = json.loads(text)  # must parse -- not just "look like" JSON
    jsonschema.validate(instance, NESTED_SCHEMA)


def test_gbnf_grammar_constrained_generation():
    """A raw grammar (the `grammar` extension) must constrain output the same way a
    JSON schema does -- here, to exactly one of two literal strings."""
    vocab = list("yesno")
    eos_id = len(vocab)
    vocab = vocab + ["<eos>"]
    tokenizer_info = _tokenizer_info(vocab, eos_id)
    compiler = xgrammar.GrammarCompiler(tokenizer_info)
    compiled = compiler.compile_grammar('root ::= "yes" | "no"')
    processor = GrammarLogitsProcessor(compiled)

    text = _greedy_decode(processor, vocab, eos_id)

    assert text in ("yes", "no")


def test_grammar_compiler_cache_reuses_the_compiler_per_tokenizer(monkeypatch):
    """Building a `GrammarCompiler` walks the whole vocab, so `GrammarCompilerCache`
    must build it once per tokenizer identity and reuse it across requests, not
    rebuild on every `for_json_schema`/`for_grammar` call."""
    vocab = list("ab") + ["<eos>"]
    fixed_info = _tokenizer_info(vocab, eos_id=2)
    build_calls = []

    def fake_from_huggingface(tokenizer, **kw):
        build_calls.append(tokenizer)
        return fixed_info

    monkeypatch.setattr(xgrammar.TokenizerInfo, "from_huggingface", fake_from_huggingface)

    cache = GrammarCompilerCache()
    tok_a, tok_b = object(), object()

    cache.for_json_schema(tok_a, {"type": "string"})
    cache.for_grammar(tok_a, 'root ::= "a"')
    cache.for_json_schema(tok_b, {"type": "string"})

    assert build_calls == [tok_a, tok_b]  # one build per distinct tokenizer, not per call
