# SPDX-License-Identifier: MIT
"""Shared pytest configuration and fixtures for superl8-serve tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


# ── Package import path ──────────────────────────────────────────────────────
# Ensure the repo root is on sys.path so `import superl8serve` works without install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ── Markers ──────────────────────────────────────────────────────────────────
def pytest_configure(config):
    """Register custom markers (also declared in pyproject.toml [tool.pytest])."""
    config.addinivalue_line("markers", "cuda: tests that require CUDA GPU")
    config.addinivalue_line(
        "markers", "comfy_e2e: end-to-end ComfyUI integration tests"
    )


# ── Common fixtures ──────────────────────────────────────────────────────────
@pytest.fixture
def weights_dir():
    """Return the SUPERL8_WEIGHTS_DIR env var or skip the test."""
    d = os.environ.get("SUPERL8_WEIGHTS_DIR")
    if not d or not os.path.isdir(d):
        pytest.skip("SUPERL8_WEIGHTS_DIR not set or not a directory")
    return Path(d)


@pytest.fixture
def cuda_available():
    """Skip if CUDA is not available."""
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
    except ImportError:
        pytest.skip("torch not installed")
