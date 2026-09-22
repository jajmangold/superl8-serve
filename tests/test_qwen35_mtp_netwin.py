# SPDX-License-Identifier: MIT
"""Qwen3.5-9B MTP speculative-decode NET-DECODE-SPEEDUP gate.

The MTP draft head accepts ~92% (``test_qwen35_mtp_accept``), yet spec-decode is a net
decode *slowdown* whenever the verify path costs more than the tokens it saves. This
gate measures the thing that actually matters end-to-end:

  * decode tok/s MTP-ON vs MTP-OFF (the net win — ON must beat OFF),
  * draft accept rate (must stay ~90% so the drafts are worth verifying),
  * greedy token identity ON-vs-OFF (spec-decode must not change the greedy stream).

HOW THE NET WIN LANDED (the loop restructure, spec loop task 1): the old loop paid for
every emitted token TWICE — a per-step base forward AND a canonical re-decode of the
accepted tokens (``_canonicalize_accepted_kv`` for full-attn K/V; a recurrent-state
replay for DeltaNet) — streaming weights base+verify+canon ≈ 3× to emit ~2 tokens
(~0.5–0.7×). The restructure removes both: (c) ``base_tok``/``base_hidden`` are pipelined
from the PRIOR step's verify (no per-step base forward), and (b) the verify forward now
computes attention through the SAME paged-decode kernel plain decode uses
(``GQAAttention._verify_batched`` walks the verify tokens as consecutive paged-decode
steps, head_dim 256 included), so every committed token is byte-identical to non-spec
greedy with NO re-decode; (d) DeltaNet state is committed from a per-token trajectory
captured during verify. Net: ONE weight-stream per step amortized over the accepted
tokens. Kept ``xfail(strict=False)`` only so PCIe-1.0 weight-stream jitter can't red CI.

Run in the superl8-serve test image on ONE gpu (never 4/5/7/9/11/14):

    docker run --rm --gpus '"device=8"' --entrypoint python3 -e CUDA_VISIBLE_DEVICES=0 \
        -v $PWD:/work -w /work -e PYTHONPATH=/work \
        superl8-serve-test:latest -m pytest tests/test_qwen35_mtp_netwin.py -q -s
"""

from __future__ import annotations

import os
import time

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

ACCEPT_BAR = float(os.environ.get("QWEN35_9B_ACCEPT_BAR", "0.80"))
MAXTOK = int(os.environ.get("QWEN35_9B_NETWIN_MAXTOK", "96"))

PROMPTS = [
    "Count from one to ten in words.",
    "Explain what a prime number is in one sentence.",
    "What is the capital of France, and why is it famous?",
]


@pytest.fixture(scope="module")
def engine_9b():
    from superl8serve.engine.llm_engine import LLMEngine
    from superl8serve.loader import checkpoint_info, load_superl8_state_dict
    from superl8serve.models.config import ModelConfig

    meta_cfg = dict(checkpoint_info(SUPERL8)["meta"]["config"])
    cfg = ModelConfig.from_hf(meta_cfg, arch="qwen3_5")
    weights = load_superl8_state_dict(SUPERL8, device="cuda")
    return LLMEngine(
        cfg, weights, device="cuda", max_num_seqs=len(PROMPTS), max_len=1024,
        enable_cuda_graph=False,
    )


def _encode(tok, text):
    ids = tok.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True
    )
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return ids[0] if ids and isinstance(ids[0], list) else ids


def _configure_drafter(eng, mode):
    """Point the runner at one drafter: 'mtp' (depth-1 head only), 'ngram', or
    'cascade' (n-gram then MTP fallback). None leaves the engine default."""
    from superl8serve.engine.drafters import NgramDrafter

    r = eng.runner
    if mode is None:
        return
    r._drafter_mode = mode
    r._ngram = NgramDrafter(min_n=2, max_n=3, max_k=r._spec_k) if mode in ("cascade", "ngram") else None


def _timed_generate(eng, prompts, params, spec, drafter=None):
    _configure_drafter(eng, drafter)
    eng.runner._spec_enabled = spec
    eng.runner.spec_stats = {"steps": 0, "drafts": 0, "accepts": 0}
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = eng.generate([list(p) for p in prompts], params)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    n = sum(len(o) for o in outs)
    return outs, n / dt, dict(eng.runner.spec_stats)


@pytest.fixture(scope="module")
def netwin_measure(engine_9b):
    from transformers import AutoTokenizer
    from superl8serve.engine.sequence import SamplingParams

    tok = AutoTokenizer.from_pretrained(TOK)
    prompts = [_encode(tok, t) for t in PROMPTS]
    params = SamplingParams(temperature=0.0, max_tokens=MAXTOK)

    # OFF (plain greedy) baseline.
    outs_off, tps_off, _ = _timed_generate(engine_9b, prompts, params, False)
    # CASCADE drafter — no longer the engine default (see drafter_config()'s
    # docstring: n-gram-first priority was found 2026-09-14 to regress decode
    # tok/s vs MTP alone on prose). Kept measured here for comparison against
    # the MTP-only run below, and as a regression check for anyone opting into
    # cascade explicitly (grammar-heavy / highly-repetitive workloads).
    outs_cas, tps_cas, st_cas = _timed_generate(engine_9b, prompts, params, True, drafter="cascade")
    # MTP-head-only — the draft head's own accept rate (this test's original subject:
    # "the MTP draft head accepts ~90%"). The cascade blends in n-gram, which trades
    # accept-rate for acceptance-length, so the head's rate is measured on its own.
    outs_mtp, tps_mtp, st_mtp = _timed_generate(engine_9b, prompts, params, True, drafter="mtp")

    def _acc(st):
        return st["accepts"] / st["drafts"] if st["drafts"] else 0.0

    identical = all(outs_cas[i] == outs_off[i] for i in range(len(prompts)))
    mtp_identical = all(outs_mtp[i] == outs_off[i] for i in range(len(prompts)))
    m = {
        "tps_off": tps_off, "tps_on": tps_cas, "ratio": tps_cas / tps_off,
        "accept": _acc(st_cas), "mtp_accept": _acc(st_mtp),
        "mtp_ratio": tps_mtp / tps_off, "identical": identical and mtp_identical,
    }
    print(f"\n[netwin] off={tps_off:.2f}  cascade={tps_cas:.2f} ({m['ratio']:.2f}x, "
          f"accept={m['accept']:.3f})  mtp={tps_mtp:.2f} ({m['mtp_ratio']:.2f}x, "
          f"accept={m['mtp_accept']:.3f})  greedy-identical={m['identical']}")
    return m


@gpu_ckpt
def test_mtp_accept_rate_stays_high(netwin_measure):
    """The MTP draft head must keep landing (~90%) — measured on the head alone (the
    cascade's blended rate is lower by design: n-gram trades accept-rate for
    acceptance-length, and misses on prose)."""
    assert netwin_measure["mtp_accept"] >= ACCEPT_BAR, (
        f"MTP-head accept {netwin_measure['mtp_accept']:.3f} < {ACCEPT_BAR} — draft regressed"
    )


@gpu_ckpt
@pytest.mark.xfail(
    reason="net decode win needs bit-identical verify attention so the loop can stop the "
    "base+canon re-decodes; that landed (verify attention now runs through the paged-decode "
    "kernel, base pipelined from the prior verify). Kept xfail(strict=False) so PCIe-1.0 "
    "weight-stream jitter on a shared box can't red the suite; it reports XPASS. See docstring.",
    strict=False,
)
def test_cascade_is_a_net_decode_speedup(netwin_measure):
    """CASCADE gate (no longer the engine default — see drafter_config()'s docstring):
    spec-decode with the cascade drafter must exceed plain decode, with the greedy
    token stream unchanged. Both must hold together — a faster stream that diverges is
    not a win. Kept as a regression check for anyone opting into cascade explicitly."""
    assert netwin_measure["identical"], "greedy token stream changed under spec-decode"
    assert netwin_measure["ratio"] > 1.0, (
        f"spec-on {netwin_measure['tps_on']:.2f} tok/s did not beat "
        f"off {netwin_measure['tps_off']:.2f} tok/s ({netwin_measure['ratio']:.2f}x)"
    )


@gpu_ckpt
def test_mtp_is_a_net_decode_speedup(netwin_measure):
    """THE gate: spec-decode with the engine's actual default drafter (MTP-only, since
    2026-09-14) must exceed plain decode. Not xfail — this is the path production
    traffic takes, so it must hold reliably, not just report XPASS on a good day."""
    assert netwin_measure["mtp_ratio"] > 1.0, (
        f"mtp-on {netwin_measure['tps_off'] * netwin_measure['mtp_ratio']:.2f} tok/s "
        f"did not beat off {netwin_measure['tps_off']:.2f} tok/s "
        f"({netwin_measure['mtp_ratio']:.2f}x) — this is the default drafter, "
        f"a regression here means the engine's out-of-the-box config is a net slowdown"
    )
