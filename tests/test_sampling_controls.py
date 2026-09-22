# SPDX-License-Identifier: MIT
"""Request-bound and construction tests for native sampling controls (#353)."""

import pytest
from pydantic import ValidationError

from superl8serve.api.request_helpers import sampling_params
from superl8serve.api.schemas import ChatCompletionRequest, CompletionRequest
from superl8serve.engine.sequence import SamplingParams


def _chat(**controls):
    return ChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            **controls,
        }
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("top_k", 0),
        ("top_k", -1),
        ("repetition_penalty", 0),
        ("repetition_penalty", -0.5),
        ("repetition_penalty", float("nan")),
        ("repetition_penalty", float("inf")),
    ],
)
def test_chat_schema_rejects_invalid_sampling_bounds(field, value):
    with pytest.raises(ValidationError):
        _chat(**{field: value})


def test_completion_schema_accepts_official_lfm_profile():
    request = CompletionRequest.model_validate(
        {
            "model": "lfm25",
            "prompt": "write",
            "temperature": 0.1,
            "top_k": 50,
            "repetition_penalty": 1.1,
        }
    )
    assert request.top_k == 50
    assert request.repetition_penalty == pytest.approx(1.1)


def test_request_helper_maps_omitted_top_k_to_native_no_op():
    params = sampling_params(0.1, 1.0, 32)
    assert params.top_k == 0
    assert params.repetition_penalty == 1.0


def test_request_helper_preserves_sampling_extensions():
    params = sampling_params(
        0.1,
        1.0,
        32,
        top_k=50,
        repetition_penalty=1.1,
    )
    assert params == SamplingParams(
        temperature=0.1,
        top_p=1.0,
        top_k=50,
        repetition_penalty=1.1,
        max_tokens=32,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_k": -1},
        {"repetition_penalty": 0},
        {"repetition_penalty": float("nan")},
        {"repetition_penalty": float("inf")},
    ],
)
def test_native_sampling_params_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)
