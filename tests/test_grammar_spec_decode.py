# SPDX-License-Identifier: MIT
"""Grammar tier-0 spec-decode: bit-identity + acceptance-length (engine level).

The tier-0 grammar drafter commits a grammar's singleton-forced tokens for free and
feeds them into the SAME verify path the MTP/n-gram cascade uses, with the verify
truth GRAMMAR-MASKED so a forced token is never rejected against an unmasked argmax.
The load-bearing contract: greedy spec-decode with a grammar constraint MUST be
bit-identical to plain (spec-off) grammar-constrained greedy decode — the forced
tokens are exactly what plain grammar decode would emit, by construction.

We drive this with a deterministic FAKE grammar (a per-position allowed-set schedule)
rather than xgrammar, so the test isolates the ENGINE's spec-vs-plain grammar plumbing
(xgrammar's own matcher is covered by tests/test_structured.py + a structured smoke).
The fake implements BOTH the sampler ``__call__`` hook (plain path) and the GrammarView
walk (spec path), so both engines apply the identical mask. Requires superl8 + CUDA.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("superl8")

from superl8serve.engine import LLMEngine, SamplingParams  # noqa: E402
from test_engine import _cfg_mtp, _no_mtp_cfg, _sd_mtp  # noqa: E402


class FakeGrammar:
    """A grammar as a per-generated-position allowed-token schedule. ``schedule[i]``
    is the set of legal ids for the i-th GENERATED token; a singleton entry is a
    grammar-FORCED position (the tier-0 drafter rides it for free); a missing entry
    means unconstrained (the model decides). Implements the LogitsProcessor ``__call__``
    (plain decode) AND the GrammarView walk (spec decode) over the same schedule, so a
    plain-decode engine and a spec-decode engine mask identically. One instance per
    request (stateful) — never share across the reference and spec engines."""

    def __init__(self, prompt_len: int, schedule: dict[int, set[int]], vocab_size: int):
        self.prompt_len = prompt_len
        self.schedule = schedule
        self.vocab_size = vocab_size
        self._pos = 0  # generated tokens the walk has committed (spec path)

    def _allowed(self, gen: int) -> set[int] | None:
        return self.schedule.get(gen)  # None => unconstrained

    def _apply_mask(self, logits_row: torch.Tensor, gen: int) -> None:
        allowed = self._allowed(gen)
        if allowed is None:
            return
        keep = torch.full_like(logits_row, float("-inf"))
        idx = torch.tensor(sorted(allowed), device=logits_row.device)
        keep[idx] = logits_row[idx]
        logits_row.copy_(keep)

    # ---- plain-decode sampler hook: (input_ids, logits) -> logits ----------------
    def __call__(self, input_ids, logits):
        gen = len(input_ids) - self.prompt_len
        out = logits.clone()
        self._apply_mask(out, gen)
        return out

    # ---- GrammarView walk (spec-decode) -----------------------------------------
    def sync(self, input_ids):
        self._pos = len(input_ids) - self.prompt_len

    def mark_committed(self, n):
        self._pos = n - self.prompt_len

    def advance(self, token_id) -> bool:
        allowed = self._allowed(self._pos)
        if allowed is not None and token_id not in allowed:
            return False
        self._pos += 1
        return True

    def rewind(self, n):
        self._pos -= n

    def next_singleton(self):
        allowed = self._allowed(self._pos)
        return next(iter(allowed)) if allowed is not None and len(allowed) == 1 else None

    def mask_row(self, logits_row):
        self._apply_mask(logits_row, self._pos)


def _schedule(vocab_size):
    """Interleave FORCED singleton runs (drafted for free by tier-0) with free
    branch positions (fall through to the model). The forced ids are fixed structural
    tokens; the runs of length >= 2 guarantee the accept path commits several tokens
    per step (n_acc >= 2), the same stress the MTP bit-identity tests demand."""
    return {
        0: {100},  # forced run 1 (len 3): 100,101,102
        1: {101},
        2: {102},
        # 3,4 free (model branches)
        5: {150},  # forced run 2 (len 4): 150,151,152,153
        6: {151},
        7: {152},
        8: {153},
        # 9,10 free
        11: {200},  # forced run 3 (len 2)
        12: {201},
    }


def _run(cfg, sd, prompt, params_factory, spec, drafter_mode="cascade", forbid_graph=False):
    eng = LLMEngine(cfg, sd, device="cuda", max_num_seqs=2, max_len=64, enable_cuda_graph=False)
    if spec:
        eng.runner._spec_enabled = True
        eng.runner._drafter_mode = drafter_mode
        eng.runner.spec_stats = {"steps": 0, "drafts": 0, "accepts": 0}
        if forbid_graph:

            class _ForbiddenGrammarGraph:
                def try_run(self, *args, **kwargs):
                    pytest.fail(
                        "grammar verification needs maskable logits; graph must be bypassed"
                    )

            eng.runner.graphed_verify = _ForbiddenGrammarGraph()
    out = eng.generate([prompt], params_factory())[0]
    stats = dict(eng.runner.spec_stats) if spec else None
    return out, stats


@pytest.mark.parametrize("drafter_mode", ["grammar", "cascade"])
def test_grammar_spec_bit_identical_to_plain_grammar_decode(drafter_mode):
    """Spec-decode (grammar tier-0) == plain grammar-constrained greedy, token-for-token."""
    torch.manual_seed(0)
    cfg = _cfg_mtp()
    sd = _sd_mtp(cfg)
    prompt = [3, 1, 4, 1, 5, 9, 2, 6]
    V = cfg.vocab_size

    def plain_params():
        return SamplingParams(
            temperature=0.0,
            max_tokens=16,
            logit_processors=[FakeGrammar(len(prompt), _schedule(V), V)],
        )

    def spec_params():
        return SamplingParams(
            temperature=0.0,
            max_tokens=16,
            logit_processors=[FakeGrammar(len(prompt), _schedule(V), V)],
        )

    # Plain grammar-constrained greedy reference (no MTP head, spec off).
    out_ref, _ = _run(_no_mtp_cfg(cfg), sd, prompt, plain_params, spec=False)

    # Spec-decode with the grammar constraint active.
    out_spec, stats = _run(
        cfg,
        sd,
        prompt,
        spec_params,
        spec=True,
        drafter_mode=drafter_mode,
        forbid_graph=True,
    )

    assert out_spec == out_ref, (
        f"[{drafter_mode}] grammar spec-decode NOT bit-identical to plain grammar decode:\n"
        f"  spec={out_spec}\n   ref={out_ref}"
    )
    # The forced runs must have been drafted+accepted: AL (tokens/step) well above 1.
    al = len(out_spec) / stats["steps"] if stats["steps"] else 1.0
    assert al > 1.3, f"[{drafter_mode}] grammar forced runs not accepted; AL={al:.2f} {stats}"


def test_grammar_forced_tokens_present_in_output():
    """Sanity: the forced structural tokens actually appear where the schedule forces
    them (so the grammar is genuinely constraining, not a no-op that trivially matches)."""
    torch.manual_seed(0)
    cfg = _cfg_mtp()
    sd = _sd_mtp(cfg)
    prompt = [3, 1, 4, 1, 5, 9, 2, 6]
    V = cfg.vocab_size

    def params():
        return SamplingParams(
            temperature=0.0,
            max_tokens=16,
            logit_processors=[FakeGrammar(len(prompt), _schedule(V), V)],
        )

    out, _ = _run(cfg, sd, prompt, params, spec=True, drafter_mode="cascade")
    assert out[0:3] == [100, 101, 102]
    assert out[5:9] == [150, 151, 152, 153]
    assert out[11:13] == [200, 201]
