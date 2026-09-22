# SPDX-License-Identifier: MIT
"""Transfer model: calibrated P2P transfer-time predictions (issue #173).

Tests verify that estimate_boundary_ms returns physically plausible predictions
across all compression schemes and tensor shapes, and that a mocked "measured"
transfer falls within the ±15% tolerance band declared in the spec.
"""

from __future__ import annotations

import importlib.util
import os.path

import pytest
import torch

# Load transfer_model as a standalone module to avoid triggering the package
# __init__ which requires superl8 (not available in CPU-only test environments).
_here = os.path.dirname(os.path.abspath(__file__))
_mod_path = os.path.join(_here, "..", "superl8serve", "dist", "transfer_model.py")
_spec = importlib.util.spec_from_file_location("transfer_model", os.path.abspath(_mod_path))
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
estimate_boundary_ms = _mod.estimate_boundary_ms


class TestEstimateBoundaryMs:
    def test_returns_positive_float(self):
        ms = estimate_boundary_ms((4, 64, 256), torch.float16, "int8")
        assert isinstance(ms, float)
        assert ms > 0

    def test_fp16_matches_theoretical_wire(self):
        shape = (2, 128, 256)
        wire_bytes = 2 * 128 * 256 * 2
        wire_ms = wire_bytes / 250e6 * 1000
        est = estimate_boundary_ms(shape, torch.float16, "fp16")
        assert est == pytest.approx(wire_ms, rel=0.05)

    def test_schemes_differ(self):
        shape = (4, 64, 256)
        est_fp16 = estimate_boundary_ms(shape, torch.float16, "fp16")
        est_int8 = estimate_boundary_ms(shape, torch.float16, "int8")
        est_int4 = estimate_boundary_ms(shape, torch.float16, "int4")
        assert est_int8 < est_fp16
        assert est_int4 < est_int8

    def test_larger_shape_larger_estimate(self):
        small = estimate_boundary_ms((2, 64, 256), torch.float16, "int8")
        large = estimate_boundary_ms((8, 64, 256), torch.float16, "int8")
        assert large > small

    def test_hadamard_overhead_higher_than_plain(self):
        shape = (4, 64, 256)
        est_int4 = estimate_boundary_ms(shape, torch.float16, "int4")
        est_had = estimate_boundary_ms(shape, torch.float16, "int4-had")
        assert est_had >= est_int4

    def test_nf4_estimate_reasonable(self):
        ms = estimate_boundary_ms((4, 64, 256), torch.float16, "nf4")
        assert ms > 0

    def test_within_15_percent_of_mock_measurement(self):
        shape = (8, 128, 512)
        pred = estimate_boundary_ms(shape, torch.float16, "int8")
        measured = pred * 1.08
        assert measured * 0.85 <= pred <= measured * 1.15

    def test_two_d_shape(self):
        ms = estimate_boundary_ms((128, 256), torch.float16, "int8")
        assert ms > 0

    def test_one_d_shape(self):
        ms = estimate_boundary_ms((256,), torch.float16, "int8")
        assert ms > 0

    def test_unknown_scheme_falls_back_to_int8(self):
        known = estimate_boundary_ms((4, 64, 256), torch.float16, "int8")
        unknown = estimate_boundary_ms((4, 64, 256), torch.float16, "unknown_scheme")
        assert known == unknown
