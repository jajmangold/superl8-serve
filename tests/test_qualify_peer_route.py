# SPDX-License-Identifier: MIT
"""Offline exact-pair P2P qualification probe (superl8-serve#356).

The probe is the offline, human-invoked counterpart to the fail-closed route
policy: run it on an idle pair, record the evidence, then add the pair to
``SUPERL8_P2P_VALIDATED_PAIRS``. These tests are CPU-safe — they pin the
deterministic probe pattern (no fp16 overflow / NaN) and the live-service GPU
guard (9 and 14) without touching any GPU.
"""

from __future__ import annotations

import importlib.util
import os.path

import pytest
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_mod_path = os.path.join(_here, "..", "tools", "qualify_peer_route.py")
_spec = importlib.util.spec_from_file_location("qualify_peer_route", os.path.abspath(_mod_path))
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


class TestPattern:
    def test_pattern_is_bounded_and_nan_free(self):
        x = _mod._pattern(1 << 16, "cpu")
        assert torch.isfinite(x).all()
        assert x.abs().max().item() <= 1020
        assert x.min().item() >= 0

    def test_pattern_is_deterministic(self):
        a = _mod._pattern(1 << 12, "cpu")
        b = _mod._pattern(1 << 12, "cpu")
        assert torch.equal(a, b)

    def test_pattern_no_fp16_overflow_at_large_numel(self):
        """Regression: an fp16 arange overflows past 65504 -> inf/NaN, which
        made torch.equal report a false bit-exact failure on a healthy pair."""
        x = _mod._pattern(1 << 20, "cpu")
        assert torch.isfinite(x).all()
        assert torch.equal(x, x)


class TestLiveGpuGuard:
    @pytest.mark.parametrize("pair", [(0, 9), (9, 0), (0, 14), (14, 0), (9, 14), (14, 9)])
    def test_forbidden_physical_gpu_refused(self, pair):
        src, dst = pair
        with pytest.raises(SystemExit):
            _mod.qualify_pair(src, dst, physical_src=src, physical_dst=dst)

    def test_no_forbidden_physical_gpu_passes_guard(self, monkeypatch):
        """A pair not referencing 9/14 must reach the probe path, not the guard."""
        import subprocess

        def fake_smi(*args, **kwargs):
            raise FileNotFoundError("no GPU in CPU test")

        monkeypatch.setattr(subprocess, "run", fake_smi)
        try:
            _mod.qualify_pair(0, 1, physical_src=5, physical_dst=6)
        except SystemExit as exc:
            # Must never be refused at the forbidden-GPU guard.
            assert "live-service physical GPU" not in str(exc)
        except FileNotFoundError:
            # Reached the probe path and failed for lack of CUDA/nvidia-smi —
            # exactly what we want on a CPU-only host.
            pass
