# SPDX-License-Identifier: MIT
"""Qwen3.5-9B MTP speculative-decode ACCEPT-RATE gate.

The MTP draft head is only worth its verify cost if it actually accepts. Before the
prefix-KV fix the draft ran cache-free (attending only its own single token) and
accepted at ~1.3% — near chance — making spec-decode a net SLOWDOWN. qengine's inline
nextn head, given the committed context as a prefix KV cache, reaches ~83% on the same
Qwen3.5 hybrid. This test measures the real draft accept-rate on the 4-bit 9B over a
fixed prompt set and asserts it clears a bar well above chance.

Skipped unless the 4-bit 9B `.superl8` + a Qwen3.5 tokenizer dir are present and CUDA is
available (set `QWEN35_9B_SUPERL8` / `QWEN35_9B_TOK` to override the default paths).
Run in the superl8-serve test image on a single GPU, e.g.:

    docker run --rm --gpus '"device=5"' --entrypoint python3 -e CUDA_VISIBLE_DEVICES=0 \
        -v $PWD:/work -w /work -e PYTHONPATH=/work \
        superl8-serve-test:latest -m pytest tests/test_qwen35_mtp_accept.py -q
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("superl8")

import torch

CUDA = torch.cuda.is_available()
SUPERL8 = os.environ.get("QWEN35_9B_SUPERL8", os.path.join(os.environ.get("SUPERL8_WEIGHTS_DIR", "/path/to/weights"), "Qwen__Qwen3.5-9B.b4.superl8"))
TOK = os.environ.get("QWEN35_9B_TOK", os.path.join(os.environ.get("SUPERL8_WEIGHTS_DIR", "/path/to/weights"), "tok"))

_have = os.path.exists(SUPERL8) and os.path.exists(f"{TOK}/config.json")
gpu_ckpt = pytest.mark.skipif(
    not (CUDA and _have), reason="needs CUDA + Qwen3.5-9B .superl8 + a Qwen3.5 tokenizer dir"
)

# A bar comfortably above chance (1/vocab ≈ 4e-6; the cache-free draft managed ~1.3%).
# qengine hits ~83% on this exact hybrid; we require the draft to be genuinely useful.
ACCEPT_BAR = float(os.environ.get("QWEN35_9B_ACCEPT_BAR", "0.40"))

PROMPTS = [
    "Count from one to ten in words.",
    "Explain what a prime number is in one sentence.",
    "What is the capital of France, and why is it famous?",
    "List three primary colors.",
    "Summarize the water cycle in two sentences.",
]


@pytest.fixture(scope="module")
def engine_9b():
    from superl8serve.engine.llm_engine import LLMEngine
    from superl8serve.loader import checkpoint_info, load_superl8_state_dict
    from superl8serve.models.config import ModelConfig

    meta_cfg = dict(checkpoint_info(SUPERL8)["meta"]["config"])
    cfg = ModelConfig.from_hf(meta_cfg, arch="qwen3_5")
    weights = load_superl8_state_dict(SUPERL8, device="cuda")
    eng = LLMEngine(
        cfg, weights, device="cuda", max_num_seqs=2, max_len=1024, enable_cuda_graph=False
    )
    eng.runner._spec_enabled = True  # spec-decode is opt-in/off by default; measure it here
    return eng


def _encode(tok, text):
    ids = tok.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True
    )
    return ids if isinstance(ids, list) else ids.tolist()


@gpu_ckpt
def test_mtp_draft_accept_rate_above_bar(engine_9b):
    """The prefix-KV draft must accept well above chance on the real 9B — otherwise
    MTP is a net slowdown (verify cost with nothing to show)."""
    from transformers import AutoTokenizer
    from superl8serve.engine.sequence import SamplingParams

    eng = engine_9b
    mtp = getattr(eng.runner.model, "mtp", None)
    assert mtp is not None, "9B checkpoint ships an MTP head — it must be recovered"
    assert eng.runner._mtp_prefix_kv, "the prefix-KV draft path must be engaged for Qwen3.5"

    tok = AutoTokenizer.from_pretrained(TOK)
    prompts = [_encode(tok, t) for t in PROMPTS]
    params = SamplingParams(temperature=0.0, max_tokens=64)

    eng.runner.spec_stats = {"steps": 0, "drafts": 0, "accepts": 0}
    eng.generate(prompts, params)
    st = eng.runner.spec_stats
    assert st["drafts"] > 0, "spec-decode never engaged (no drafts proposed)"
    accept_rate = st["accepts"] / st["drafts"]
    assert accept_rate >= ACCEPT_BAR, (
        f"MTP draft accept-rate {accept_rate:.3f} below bar {ACCEPT_BAR:.2f} "
        f"({st['accepts']}/{st['drafts']} drafts over {st['steps']} steps) — the draft "
        f"is not attending the prefix KV cache (cache-free draft regressed?)."
    )


@gpu_ckpt
def test_mtp_spec_decode_matches_plain_greedy_mostly(engine_9b):
    """Greedy spec-decode should reproduce plain greedy ALMOST exactly. On the real
    int8 9B it is NOT bit-identical: the verify forward uses an fp16 dense fallback for
    head_dim 256 (no int8 multi-query verify kernel yet) while plain decode uses int8
    ``attn_paged_decode_cached`` — different numerics disagree on the occasional
    tie-break token, then the streams re-converge. This is a known verify-path
    limitation (spec-decode is OFF by default because of it); the future int8
    head_dim-256 verify kernel closes the gap. Here we assert the streams are ≥95%
    identical so a gross regression (draft feeding garbage into the commit) still fails.
    """
    from transformers import AutoTokenizer
    from superl8serve.engine.sequence import SamplingParams

    eng = engine_9b
    eng.runner._spec_enabled = True
    tok = AutoTokenizer.from_pretrained(TOK)
    prompts = [_encode(tok, t) for t in PROMPTS[:3]]
    params = SamplingParams(temperature=0.0, max_tokens=48)

    outs_spec = eng.generate(prompts, params)
    eng.runner._spec_enabled = False  # plain autoregressive greedy
    outs_plain = eng.generate(prompts, params)
    eng.runner._spec_enabled = True

    for i, (a, b) in enumerate(zip(outs_spec, outs_plain)):
        n = min(len(a), len(b))
        agree = sum(1 for j in range(n) if a[j] == b[j]) / max(1, n)
        assert agree >= 0.95, (
            f"prompt {i}: spec-decode agreed with plain greedy only {agree:.2%} "
            f"(expected ≥95%; gross divergence means the draft is corrupting the commit):\n"
            f"  spec ={a}\n  plain={b}"
        )
