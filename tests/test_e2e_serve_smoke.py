# SPDX-License-Identifier: MIT
"""RC-gate e2e serve smoke test: real .superl8 model -> coherent completion.

Exercises the int8 dp4a kernels + converter + engine + sampler + API on a
real model end-to-end. ONE model, ONE short generation, CI-fast. Needs
CUDA + superl8 + model weights (set SUPERL8_WEIGHTS_DIR)."""

import os

import pytest

pytest.importorskip("superl8")
pytest.importorskip("fastapi")
pytest.importorskip("transformers")

import torch
from starlette.testclient import TestClient
from transformers import AutoTokenizer

from superl8serve.api.app import create_app
from superl8serve.api.server import load_engine
from superl8serve.loader import checkpoint_info

_WEIGHTS_DIR = os.environ.get("SUPERL8_WEIGHTS_DIR", "/path/to/weights")
_CANDIDATES = [
    "LiquidAI__LFM2.5-230M.b8.superl8",
    "LFM2-1.2B.b8.superl8",
]

CUDA = torch.cuda.is_available()


def _resolve_model():
    if not os.path.isdir(_WEIGHTS_DIR):
        pytest.skip(f"weights dir {_WEIGHTS_DIR} not found")
    for name in _CANDIDATES:
        path = os.path.join(_WEIGHTS_DIR, name)
        if os.path.isfile(path):
            return path
    pytest.skip(f"no model found in {_WEIGHTS_DIR} (tried {_CANDIDATES})")


@pytest.mark.skipif(not CUDA, reason="model engine needs CUDA")
def test_e2e_serve_smoke():
    model_path = _resolve_model()
    assert isinstance(model_path, str)

    info = checkpoint_info(model_path)
    hf_config = info.get("meta", {}).get("config", {})
    model_id = hf_config.get("_name_or_path")
    if model_id is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception:
            tokenizer = None
    else:
        tokenizer = None
    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception:
            pytest.skip(f"could not load tokenizer for {model_path}")
    assert tokenizer is not None and hasattr(tokenizer, "eos_token_id")

    engine = load_engine(
        model_path,
        device="cuda",
        max_num_seqs=1,
        max_len=512,
        eos_id=tokenizer.eos_token_id,
    )
    app = create_app(engine, tokenizer, served_model_name="e2e-smoke")

    with TestClient(app) as client:
        resp = client.post(
            "/v1/completions",
            json={
                "model": "e2e-smoke",
                "prompt": "The capital of France is",
                "temperature": 0.0,
                "max_tokens": 32,
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    text = body["choices"][0]["text"]

    # Non-empty completion
    stripped = text.strip()
    assert stripped, "completion text must be non-empty"

    # Tokens decode without error (already decoded by the API layer)
    assert body["usage"]["completion_tokens"] > 0, "must have generated at least one token"

    # Non-degenerate: more than 1 unique character
    unique = set(stripped)
    assert len(unique) >= 2, f"output appears degenerate: {text!r}"
