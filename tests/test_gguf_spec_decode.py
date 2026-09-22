# SPDX-License-Identifier: MIT
"""Speculative decode on the GGUF-native path (n-gram cascade + MTP-head detection).

Two layers:

  * CPU unit tests — the model-agnostic gate: n-gram spec-decode must engage on a
    model with NO MTP head (the common quantized-GGUF case, ``model.mtp is None``).
    The pre-fix gate refused all spec-decode unless an MTP head existed, which
    blocked the free n-gram lookup on every stripped GGUF. These construct an
    ``EngineRunner`` shell (``__new__`` + the handful of attrs the gate reads) so
    they run with no model / no CUDA.

  * GPU e2e tests (perf-marked, real Qwen3-8B-Q4_K_M) — the load-bearing proof:
    n-gram spec-decode produces BIT-IDENTICAL output to plain greedy decode, plus
    the accept-rate and the spec-vs-non-spec net tok/s. Skips without the GGUF/CUDA.
"""

from __future__ import annotations

import os

import pytest
import torch

from superl8serve.engine.drafters import NgramDrafter
from superl8serve.engine.model_runner import EngineRunner
from superl8serve.engine.sequence import SamplingParams, Sequence
from superl8serve.models.cache import RecurrentStateCache

_QWEN3_8B = os.path.join(
    os.environ.get("SUPERL8_WEIGHTS_DIR", "/path/to/weights"),
    os.environ.get(
        "QWEN3_8B_GGUF",
        "text_encoders/Qwen 3/Qwen3-8B-Q4_K_M.gguf",
    ),
)
requires_qwen3 = pytest.mark.skipif(not os.path.exists(_QWEN3_8B), reason=f"missing {_QWEN3_8B}")


def _gate_runner(*, ngram, has_recurrent=False, spec_recurrent_ok=True):
    """An ``EngineRunner`` shell carrying only the attributes ``_spec_decode_allowed``
    reads — no model build, no CUDA. Mirrors the real runner's fields exactly."""
    r = EngineRunner.__new__(EngineRunner)
    r._ngram = ngram
    r.has_recurrent = has_recurrent
    r._spec_recurrent_ok = spec_recurrent_ok
    r._spec_k = 4
    r.device = "cpu"
    r.lin_cache = None
    return r


def _greedy_seq(prompt=None):
    return Sequence(0, prompt or [1, 2, 3], SamplingParams(temperature=0.0, max_tokens=8))


# ── gate: n-gram spec-decode must engage with NO MTP head ─────────────────────
def test_gate_allows_ngram_spec_without_mtp_head():
    """The core GGUF-native enablement: an MTP-less model (``mtp is None``) with an
    active n-gram drafter is a VALID spec-decode target — the lookup needs no head."""
    r = _gate_runner(ngram=NgramDrafter())
    assert r._spec_decode_allowed([_greedy_seq([1, 2, 3, 1])], mtp=None) is True
    assert r.spec_gate_reason == "allowed"


def test_gate_skips_ngram_probe_when_next_match_is_impossible():
    r = _gate_runner(ngram=NgramDrafter())
    assert r._spec_decode_allowed([_greedy_seq([10, 20, 30])], mtp=None) is False
    assert r.spec_gate_reason == "ngram_no_match"


def test_gate_blocks_when_no_drafter_at_all():
    """No MTP head AND no n-gram drafter (``SUPERL8SERVE_SPEC_DRAFTER=mtp`` on a model
    without a head) = nothing to propose → spec-decode must NOT engage (it would just
    verify base_tok every step, a pointless net slowdown)."""
    r = _gate_runner(ngram=None)
    assert r._spec_decode_allowed([_greedy_seq([1, 2, 3, 1])], mtp=None) is False


def test_gate_still_blocks_non_greedy():
    """The accept-longest-greedy-prefix rule is only bit-identical under greedy; a
    sampled (temp>0) sequence must fall back to plain decode even with a drafter."""
    r = _gate_runner(ngram=NgramDrafter())
    hot = Sequence(0, [1, 2, 3], SamplingParams(temperature=0.7, max_tokens=8))
    assert r._spec_decode_allowed([hot], mtp=None) is False


def test_gate_blocks_image_sequence():
    """A text-only spec path must never verify over spliced image embeds."""
    r = _gate_runner(ngram=NgramDrafter())
    seq = _greedy_seq()
    seq.pixel_values = torch.zeros(1, 3, 8, 8)
    assert r._spec_decode_allowed([seq], mtp=None) is False


def test_gate_blocks_recurrent_verify_that_cannot_fit(monkeypatch):
    """A one-card recurrent model must fall back to plain decode before verify
    allocates a multi-token state trajectory that cannot fit in free VRAM."""

    class _Trajectory:
        def verify_trajectory_nbytes(self, batch_rows, tokens):
            assert (batch_rows, tokens) == (1, 5)
            return 3 << 30

    r = _gate_runner(ngram=NgramDrafter(), has_recurrent=True)
    r.device = "cuda"
    r.lin_cache = _Trajectory()
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (2 << 30, 16 << 30))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 0)

    assert r._spec_decode_allowed([_greedy_seq([1, 2, 3, 1])], mtp=None) is False


def test_recurrent_verify_trajectory_size_uses_active_rows_and_tokens():
    cache = RecurrentStateCache()
    cache.enable_static_buffers(num_slots=4)
    cache.bind([0])
    cache.set_state(0, torch.zeros(1, 2, 3, dtype=torch.float32))  # 24 B / row
    cache.set_conv_tail(0, torch.zeros(1, 5, dtype=torch.float16))  # 10 B / row

    assert cache.verify_trajectory_nbytes(batch_rows=2, tokens=5) == 340


def test_zero_draft_commits_bootstrap_without_verify():
    """An n-gram miss has no tokens to verify. The bootstrap is already the same
    one-token forward as plain decode, so return its sampled base directly."""

    class _Model:
        mtp = None

        def __call__(self, ids, pos, ctx):
            return torch.zeros(1, 1, 4)

        def compute_logits(self, hidden):
            logits = torch.zeros(1, 16)
            logits[0, 7] = 1
            return logits

    class _Cache:
        def ensure_capacity(self, slots, lengths):
            pass

    class _LinCache:
        def bind(self, slots):
            pass

    class _ForbiddenVerify:
        def try_run(self, *args, **kwargs):
            pytest.fail("zero-draft step must not run verification")

    r = EngineRunner.__new__(EngineRunner)
    r.model = _Model()
    r.cache = _Cache()
    r.lin_cache = _LinCache()
    r.device = "cpu"
    r.graphed_verify = _ForbiddenVerify()
    r.has_recurrent = False
    r._mtp_prefix_kv = False
    r._drafter_mode = "ngram"
    r._compute_drafts = lambda *args: [[]]

    seq = _greedy_seq()
    seq.slot = 0
    seq.length = len(seq.prompt_ids)

    assert r._spec_decode_eager([seq], mtp=None) == [7]
    assert seq.length == len(seq.prompt_ids) + 1
    assert seq.spec_base_tok is None
    assert seq.spec_base_hidden is None
    assert seq.spec_ngram_cooldown == 0

    r._ngram = NgramDrafter()
    r._spec_recurrent_ok = True
    # A miss must not suppress the next 64 generated tokens. Prompt lookup is
    # re-evaluated against the evolving history every step, as upstream engines do.
    seq.prompt_ids = [1, 2, 3, 1]
    assert r._spec_decode_allowed([seq], mtp=None) is True
    assert seq.spec_ngram_cooldown == 0


# ── MTP-head detection on the real GGUF (config level, no GPU) ────────────────
@pytest.mark.correctness
@requires_qwen3
def test_qwen3_8b_reports_mtp_absent():
    """Qwen3-8B-Q4_K_M ships no ``nextn.*`` (num_mtp_layers==0) → the loader must
    detect MTP absent and the drafter cascade falls back to n-gram."""
    from superl8serve.gguf_native import gguf_config

    cfg = gguf_config(_QWEN3_8B)
    assert cfg.num_mtp_layers == 0


# ── e2e: n-gram spec-decode is bit-identical to greedy + accept-rate + tok/s ──
@pytest.mark.perf
@requires_qwen3
def test_gguf_ngram_spec_bit_identical_and_accept_rate(monkeypatch):
    """The deliverable: on a REAL GGUF LLM loaded natively, n-gram spec-decode
    (a) emits BYTE-IDENTICAL tokens to plain greedy decode and (b) reports a nonzero
    accept-rate + the net decode tok/s (spec vs non-spec). Needs CUDA + the 8B GGUF.

    Explicitly requests the n-gram drafter: the engine default is `mtp` (since
    2026-09-14 — cascade's n-gram-first priority regresses decode tok/s vs MTP alone
    on prose, see drafter_config()'s docstring), which wires NO n-gram drafter and
    would silently no-op spec-decode on this headless GGUF (no MTP head to fall back
    to either). That default doesn't apply to this scenario at all — n-gram isn't
    competing with MTP here, it's the only drafter option for a model with no head —
    so a real deployment of a headless GGUF wanting spec-decode must set this
    explicitly, same as this test does."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    monkeypatch.setenv("SUPERL8SERVE_SPEC_DRAFTER", "ngram")
    import time

    from superl8serve.gguf_native import load_gguf_engine

    # A repetitive prompt so the n-gram lookup has structured spans to ride.
    prompt = [
        3838,
        374,
        279,
        6722,
        315,
        9625,
        30,  # "What is the capital of France?"
        3838,
        374,
        279,
        6722,
        315,
        9625,
        30,
    ]
    n = 48
    params = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)

    # -- baseline: plain greedy (spec OFF) --
    eng0 = load_gguf_engine(_QWEN3_8B, device="cuda", max_num_seqs=2, max_len=256)
    t0 = time.perf_counter()
    base = eng0.generate([list(prompt)], params)[0]
    dt0 = time.perf_counter() - t0
    del eng0
    torch.cuda.empty_cache()

    # -- spec ON (n-gram cascade; no MTP head on this GGUF) --
    eng1 = load_gguf_engine(_QWEN3_8B, device="cuda", max_num_seqs=2, max_len=256, spec_decode=True)
    assert getattr(eng1.model, "mtp", None) is None, "Qwen3-8B GGUF must have no MTP head"
    assert eng1.runner._ngram is not None, "n-gram drafter must be active"
    t1 = time.perf_counter()
    spec = eng1.generate([list(prompt)], params)[0]
    dt1 = time.perf_counter() - t1
    st = eng1.runner.spec_stats

    assert spec == base, (
        f"spec-decode output diverged from greedy:\n base[{len(base)}]={base}\n spec[{len(spec)}]={spec}"
    )
    accept_rate = st["accepts"] / st["drafts"] if st["drafts"] else 0.0
    assert st["steps"] > 0, "spec path never ran"
    assert st["drafts"] > 0, "n-gram never proposed a draft on a repetitive prompt"
    print(
        f"\n[gguf-spec] bit-identical={spec == base} "
        f"steps={st['steps']} drafts={st['drafts']} accepts={st['accepts']} "
        f"accept_rate={accept_rate:.3f}\n"
        f"[gguf-spec] tok/s incl. prefill: non-spec={n / dt0:.2f}  spec={n / dt1:.2f}  "
        f"(spec/non-spec={dt0 / dt1:.2f}x)"
    )
