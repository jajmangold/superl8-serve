# SPDX-License-Identifier: MIT
"""Reproducibility guards for the measured one-card Qwen3.6-35B TQ3 sidecar."""

from pathlib import Path


ROOT = Path(__file__).parents[1] / "docker" / "qwen35-tq3"


def test_tq3_runtime_is_pinned_to_sm70_and_disables_optional_ui():
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert "ed6eef39b2f934395c99b25f302d3272217e56ec" in dockerfile
    assert 'CUDA_DOCKER_ARCH=70' in dockerfile
    assert "LLAMA_BUILD_UI=OFF" in dockerfile


def test_tq3_production_profile_keeps_regressive_mtp_off():
    compose = (ROOT / "compose.yaml").read_text()

    assert "QWEN35B_ENABLE_MTP: \"0\"" in compose
    assert "QWEN35B_CACHE_TYPE_V: tq3_0" in compose
    assert "QWEN35B_BATCH_SIZE: \"32\"" in compose
    assert "QWEN35B_UBATCH_SIZE: \"32\"" in compose
