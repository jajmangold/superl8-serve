# SPDX-License-Identifier: MIT
"""Qwen3.5 MTP speculative-decode: head wiring + the safety guard.

Two tiers:
  * CPU-only: the `EngineRunner._spec_decode_allowed` guard predicate. Spec-decode
    is now SUPPORTED for the recurrent DeltaNet hybrid (greedy, text-only): the
    verify path is wired through the gated/linear mixers and the runner snapshots +
    replays the recurrent state so only accepted tokens commit. The guard still
    refuses non-greedy sampling, a missing head, and — the load-bearing safety
    contract — ANY image-carrying batch (MTP×vision is the exact interaction that
    broke in llama.cpp).
  * GPU + checkpoint (skipped otherwise): the real Qwen3.5-0.8B `.superl8` — the MTP
    head is wired and present, spec-decode text decode is bit-identical to plain
    decode, and an image request stays correct (red -> "red") with the head present.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("superl8")

import torch
import torch.nn as nn

from superl8serve.engine.model_runner import EngineRunner
from superl8serve.engine.sequence import SamplingParams, Sequence

CUDA = torch.cuda.is_available()
SUPERL8 = os.environ.get("QWEN35_VL_SUPERL8", "/models/Qwen3.5-0.8B-superl8/Qwen__Qwen3.5-0.8B.b8.superl8")
TOK = os.environ.get("QWEN35_VL_TOK", "/models/Qwen3.5-0.8B-tok")


# ── CPU-only: the guard predicate ────────────────────────────────────────────


class _PlainModel(nn.Module):
    """A non-recurrent model (plain attention) — spec-decode is safe here."""


class _RecurrentInner(nn.Module):
    is_recurrent = True
    spec_capture = True  # real GatedDeltaNet records its verify-state trajectory


class _NoCaptureRecurrent(nn.Module):
    is_recurrent = True  # recurrent but does NOT support verify-state capture


class _HybridModel(nn.Module):
    """A model carrying a recurrent (DeltaNet-style) mixer whose verify-token state
    trajectory can be captured — spec-decode is supported (the runner commits the
    accepted-prefix state directly from that trajectory, no re-decode)."""

    def __init__(self, mixer_cls=_RecurrentInner):
        super().__init__()
        self.mixer = mixer_cls()


def _runner(model) -> EngineRunner:
    # device="cpu", no CUDA graph -> __init__ builds no GPU state.
    return EngineRunner(model, cache=object(), device="cpu", enable_cuda_graph=False)


def _seq(pixel_values=None, temperature=0.0) -> Sequence:
    s = Sequence(0, [1, 2, 3], SamplingParams(temperature=temperature, max_tokens=8))
    s.pixel_values = pixel_values
    return s


_MTP = object()  # sentinel: the guard only checks `mtp is None`


def test_partial_mtp_prefix_is_treated_as_no_head():
    """A converted checkpoint may retain MTP markers without a complete block."""
    from types import SimpleNamespace

    from superl8serve.models.qwen3_5 import build_qwen3_5_mtp

    partial = {
        "mtp.fc.weight": torch.empty(1),
        "mtp.layers.0.self_attn.q_proj.weight": torch.empty(1),
    }
    cfg = SimpleNamespace(num_mtp_layers=0)

    assert build_qwen3_5_mtp(cfg, partial, None, None, None) is None


def test_guard_allows_plain_greedy_text():
    r = _runner(_PlainModel())
    assert r.has_recurrent is False
    assert r._spec_decode_allowed([_seq()], _MTP) is True


def test_guard_refuses_when_no_drafter():
    """Refusal now requires NEITHER drafter: no MTP head AND no n-gram lookup
    (the engine default, `SUPERL8SERVE_SPEC_DRAFTER=mtp`, already has no n-gram
    drafter wired on a head-less model). With nothing to propose, spec-decode
    would only verify base_tok each step — a pointless net slowdown, so the
    guard refuses."""
    r = _runner(_PlainModel())
    r._ngram = None  # no n-gram drafter either (belt-and-suspenders vs. the default)
    assert r._spec_decode_allowed([_seq()], None) is False


def test_guard_allows_ngram_without_mtp_head():
    """The GGUF-native capability: an MTP-less model (`mtp is None`) with an active
    n-gram drafter is a VALID greedy spec target — the prompt-lookup carries no head
    weights, so it drafts on any model, including a quantized GGUF whose converter
    stripped the `nextn.*` MTP tensors. Locks in the new contract.

    The engine default is `mtp` (no n-gram) since 2026-09-14 (cascade's n-gram-first
    priority was found to regress decode throughput vs MTP alone on prose — see
    drafter_config()'s docstring), so this test wires n-gram explicitly rather than
    relying on whatever the ambient default happens to be."""
    from superl8serve.engine.drafters import NgramDrafter

    r = _runner(_PlainModel())
    r._drafter_mode = "cascade"
    r._ngram = NgramDrafter(min_n=2, max_n=3, max_k=r._spec_k)
    seq = _seq()
    seq.prompt_ids = [1, 2, 3, 1]  # a next token can complete the earlier [1, ...] span
    assert r._spec_decode_allowed([seq], None) is True


def test_guard_refuses_nonzero_temperature():
    r = _runner(_PlainModel())
    assert r._spec_decode_allowed([_seq(temperature=0.7)], _MTP) is False


def test_guard_refuses_repetition_penalty():
    r = _runner(_PlainModel())
    seq = _seq(temperature=0.0)
    seq.params.repetition_penalty = 1.1
    assert r._spec_decode_allowed([seq], _MTP) is False


def test_guard_allows_recurrent_hybrid_model():
    """Qwen3.5 is a DeltaNet+gated hybrid. The verify forward records each recurrent
    layer's per-token state trajectory; after acceptance the runner commits the state
    after each row's last accepted token directly (no re-decode) — so greedy text
    spec-decode is SUPPORTED for a capture-capable recurrent hybrid."""
    r = _runner(_HybridModel())
    assert r.has_recurrent is True
    assert r._spec_recurrent_ok is True
    assert r._spec_decode_allowed([_seq()], _MTP) is True


def test_guard_refuses_noncapturing_recurrent_hybrid():
    """A recurrent hybrid whose mixer can't record its verify-state trajectory can't
    have its committed state reconstructed without a re-decode — the guard refuses it
    (plain decode instead of committing a wrong recurrent state)."""
    r = _runner(_HybridModel(mixer_cls=_NoCaptureRecurrent))
    assert r.has_recurrent is True
    assert r._spec_recurrent_ok is False
    assert r._spec_decode_allowed([_seq()], _MTP) is False


def test_guard_refuses_recurrent_hybrid_with_image():
    """The recurrent hybrid IS allowed for text, but an image on any row still
    disables spec-decode (the MTP×vision guard is independent of recurrence)."""
    r = _runner(_HybridModel())
    img = torch.zeros(1, 3, 16, 16)
    assert r._spec_decode_allowed([_seq(pixel_values=img)], _MTP) is False


def test_guard_refuses_image_batch():
    """MTP x vision guard: any sequence carrying pixel_values disables spec-decode,
    even on an otherwise-safe non-recurrent model."""
    r = _runner(_PlainModel())
    img = torch.zeros(1, 3, 16, 16)
    assert r._spec_decode_allowed([_seq(pixel_values=img)], _MTP) is False
    # ragged batch: one plain seq + one image seq -> still refused
    assert r._spec_decode_allowed([_seq(), _seq(pixel_values=img)], _MTP) is False


# ── GPU + checkpoint: real Qwen3.5-0.8B end-to-end ───────────────────────────

_have_ckpt = os.path.exists(SUPERL8) and os.path.exists(f"{TOK}/config.json")
gpu_ckpt = pytest.mark.skipif(
    not (CUDA and _have_ckpt), reason="needs CUDA + Qwen3.5-0.8B .superl8 + tokenizer dir"
)


@pytest.fixture(scope="module")
def qwen35_vl_engine():
    import json

    from superl8serve.engine.llm_engine import LLMEngine
    from superl8serve.loader import checkpoint_info, load_superl8_state_dict
    from superl8serve.models.config import ModelConfig

    meta_cfg = dict(checkpoint_info(SUPERL8)["meta"]["config"])
    hf_cfg = json.load(open(f"{TOK}/config.json"))
    meta_cfg["vision_config"] = hf_cfg["vision_config"]
    meta_cfg["image_token_id"] = hf_cfg["image_token_id"]
    cfg = ModelConfig.from_hf(meta_cfg, arch="qwen3_5_vl")
    weights = load_superl8_state_dict(SUPERL8, device="cuda")
    eng = LLMEngine(
        cfg, weights, device="cuda", max_num_seqs=2, max_len=512, enable_cuda_graph=False
    )
    return eng, cfg


@gpu_ckpt
def test_mtp_head_present(qwen35_vl_engine):
    """The MTP head is wired onto the VLM wrapper and recovered from the weights
    even though the shipped .superl8 meta baked in num_mtp_layers=0."""
    eng, _ = qwen35_vl_engine
    mtp = getattr(eng.runner.model, "mtp", None)
    assert mtp is not None
    assert mtp.num_depths() == 1
    # A hybrid model — spec-decode is now ENGAGED for greedy text (the guard allows
    # it; the runner rolls back / replays the recurrent state).
    assert eng.runner.has_recurrent is True
    greedy_text = Sequence(0, [1, 2, 3], SamplingParams(temperature=0.0, max_tokens=8))
    assert eng.runner._spec_decode_allowed([greedy_text], mtp) is True


@gpu_ckpt
def test_spec_decode_hybrid_matches_plain_mostly(qwen35_vl_engine):
    """With the MTP head present and spec-decode explicitly enabled (it is OFF by
    default), greedy spec-decode should reproduce plain greedy ALMOST exactly. It is
    NOT bit-identical on the real int8 hybrid: the verify forward uses an fp16 dense
    fallback for head_dim 256 (no int8 multi-query verify kernel yet) vs plain decode's
    int8 ``attn_paged_decode_cached`` — they disagree on the odd tie-break token, then
    re-converge. Assert ≥95% agreement so a gross regression still fails; the future
    int8 head_dim-256 verify kernel closes the gap (and re-enables spec by default)."""
    from transformers import AutoTokenizer

    eng, _ = qwen35_vl_engine
    tok = AutoTokenizer.from_pretrained(TOK)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": "Count from one to five in words."}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
    )["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    prompt = [int(x) for x in ids]
    params = SamplingParams(temperature=0.0, max_tokens=24)

    eng.runner._spec_enabled = True
    out_mtp = eng.generate([prompt], params)[0]
    eng.runner._spec_enabled = False
    out_none = eng.generate([prompt], params)[0]
    n = min(len(out_mtp), len(out_none))
    agree = sum(1 for j in range(n) if out_mtp[j] == out_none[j]) / max(1, n)
    assert agree >= 0.95, f"spec MTP decode agreed with plain only {agree:.2%} (<95%)"


@gpu_ckpt
def test_image_with_mtp_stays_correct(qwen35_vl_engine):
    """CRITICAL (llama.cpp regression): a red image through the engine, greedy, with
    the MTP head present, must still answer 'red' — and the guard must refuse
    spec-decode for the image-carrying sequence."""
    pytest.importorskip("torchvision")  # HF Qwen3.5 image processor needs it
    from transformers import AutoProcessor
    from PIL import Image

    eng, _ = qwen35_vl_engine
    proc = AutoProcessor.from_pretrained(TOK)
    tok = proc.tokenizer
    img = Image.new("RGB", (64, 64), "red")
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "What color is this image? Answer in one word."},
            ],
        }
    ]
    inp = proc.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    img_ids = inp["input_ids"][0].tolist()

    assert getattr(eng.runner.model, "mtp", None) is not None
    sid = eng.add_request(img_ids, SamplingParams(temperature=0.0, max_tokens=8))
    seq = eng.sequence(sid)
    seq.pixel_values = inp["pixel_values"].cuda()
    seq.image_grid_thw = inp["image_grid_thw"].cuda()

    # The guard must refuse spec-decode for this image sequence even if the model
    # were (hypothetically) non-recurrent — pixel_values alone disables it.
    saved = eng.runner.has_recurrent
    eng.runner.has_recurrent = False
    assert eng.runner._spec_decode_allowed([seq], eng.runner.model.mtp) is False
    eng.runner.has_recurrent = saved

    while eng.scheduler.has_work():
        eng.step()
    ans = tok.decode(eng._out[sid].output_ids, skip_special_tokens=True).lower()
    eng.forget(sid)
    assert "red" in ans, f"image+MTP answer should contain 'red', got {ans!r}"
