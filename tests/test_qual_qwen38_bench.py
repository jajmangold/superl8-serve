# SPDX-License-Identifier: MIT
"""Qualification harness must never publish a zero-step spec measurement."""

import pytest

from bench.qual_qwen38_tq34s import require_spec_engagement, spec_acceptance_metrics


def test_require_spec_engagement_rejects_gated_off_measurement():
    detail = {"needed": 10, "free": 3, "reusable": 2, "headroom": 1, "fits": False}
    with pytest.raises(RuntimeError, match="'needed': 10"):
        require_spec_engagement(
            {"steps": 0, "drafts": 0, "accepts": 0}, detail, "ngram_no_match"
        )


def test_spec_acceptance_metrics_use_real_steps_and_drafts():
    assert spec_acceptance_metrics(
        {"steps": 3, "drafts": 7, "accepts": 5}, emitted_tokens=8
    ) == {"al": 8 / 3, "acc_rate": 5 / 7}
