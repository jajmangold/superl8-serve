# SPDX-License-Identifier: MIT
"""Regression tests for honest prefill/decode benchmark accounting."""

import pytest

from bench.metrics import env_flag, summarize_generation_steps


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_env_flag_accepts_explicit_true_values(monkeypatch, value):
    monkeypatch.setenv("SUPERL8_TEST_FLAG", value)
    assert env_flag("SUPERL8_TEST_FLAG") is True


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off"])
def test_env_flag_accepts_explicit_false_values(monkeypatch, value):
    monkeypatch.setenv("SUPERL8_TEST_FLAG", value)
    assert env_flag("SUPERL8_TEST_FLAG", default=True) is False


def test_env_flag_rejects_ambiguous_values(monkeypatch):
    monkeypatch.setenv("SUPERL8_TEST_FLAG", "maybe")
    with pytest.raises(ValueError, match="SUPERL8_TEST_FLAG"):
        env_flag("SUPERL8_TEST_FLAG")


def test_generation_summary_does_not_call_prefill_decode_throughput():
    summary = summarize_generation_steps([4.0, 0.2, 0.020, 0.018, 0.019], warmup_decode_steps=1)

    assert summary["prefill_s"] == 4.0
    assert summary["graph_capture_s"] == 0.2
    assert summary["steady_decode_s"] == [0.020, 0.018, 0.019]
    assert abs(summary["steady_decode_tok_s"] - (1.0 / 0.019)) < 1e-9
    assert summary["end_to_end_tok_s"] == 5 / sum([4.0, 0.2, 0.020, 0.018, 0.019])


def test_generation_summary_handles_short_runs():
    summary = summarize_generation_steps([1.0], warmup_decode_steps=2)
    assert summary["prefill_s"] == 1.0
    assert summary["graph_capture_s"] == 0.0
    assert summary["steady_decode_s"] == []
    assert summary["steady_decode_tok_s"] == 0.0


def test_generation_summary_counts_multi_token_spec_steps():
    times = [4.0, 0.2, 0.1, 0.1]
    emitted = [1, 1, 3, 2]
    summary = summarize_generation_steps(
        times, warmup_decode_steps=1, emitted_tokens_per_step=emitted
    )

    assert summary["steady_decode_tokens"] == 5
    assert summary["steady_decode_tok_s"] == 25.0
    assert summary["end_to_end_tok_s"] == 7 / 4.4


def test_generation_summary_rejects_misaligned_token_counts():
    with pytest.raises(ValueError, match="same length"):
        summarize_generation_steps([1.0, 0.1], emitted_tokens_per_step=[1])
