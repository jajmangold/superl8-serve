# SPDX-License-Identifier: MIT
"""Tests for StagingBuffer (issue #320) — per-layer activation buffers."""

import pytest
import torch

pytest.importorskip("superl8")

from superl8serve.engine.staging import StagingBuffer

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="staging buffer needs CUDA")


class TestStagingBuffer:
    def test_shape_and_dtype(self):
        buf = StagingBuffer(8, 128, "cuda")
        assert buf.buf.shape == (8, 1, 128)
        assert buf.buf.dtype == torch.float16
        assert buf.residual.shape == (8, 1, 128)
        assert buf.residual.dtype == torch.float16

    def test_active_count_zero_by_default(self):
        buf = StagingBuffer(4, 64, "cuda")
        assert buf.active_count == 0
        assert buf.is_empty()

    def test_set_active(self):
        buf = StagingBuffer(8, 128, "cuda")
        tokens = torch.randn(3, 128, dtype=torch.float16, device="cuda")
        buf.set_active(tokens, 3)
        assert buf.active_count == 3
        assert not buf.is_empty()
        # buf is 3D: (B, 1, H)
        assert torch.allclose(buf.buf[:3, 0], tokens)

    def test_set_active_zero(self):
        buf = StagingBuffer(8, 128, "cuda")
        tokens = torch.randn(0, 128, dtype=torch.float16, device="cuda")
        buf.set_active(tokens, 0)
        assert buf.active_count == 0
        assert buf.is_empty()

    def test_set_residual(self):
        buf = StagingBuffer(8, 128, "cuda")
        res = torch.randn(3, 128, dtype=torch.float16, device="cuda")
        buf.set_residual(res, 3)
        assert torch.allclose(buf.residual[:3, 0], res)

    def test_set_active_clamps_to_count(self):
        """Only the first `count` rows should be written."""
        buf = StagingBuffer(8, 128, "cuda")
        tokens = torch.randn(5, 128, dtype=torch.float16, device="cuda")
        buf.set_active(tokens, 3)
        # rows 3..4 of buf should still be zeros
        assert buf.buf[3:].abs().sum() == 0

    def test_set_active_3d_input(self):
        """Accept 3D input (B, 1, H) directly."""
        buf = StagingBuffer(8, 128, "cuda")
        tokens = torch.randn(3, 1, 128, dtype=torch.float16, device="cuda")
        buf.set_active(tokens, 3)
        assert buf.active_count == 3
        assert torch.allclose(buf.buf[:3], tokens)

    def test_deterministic_after_reset(self):
        """After set_active(0), is_empty() returns True."""
        buf = StagingBuffer(4, 64, "cuda")
        buf.set_active(torch.randn(4, 64, device="cuda"), 4)
        assert not buf.is_empty()
        buf.active_count = 0
        assert buf.is_empty()

    def test_multiple_buffers_independent(self):
        """Two staging buffers are independent."""
        b1 = StagingBuffer(4, 64, "cuda")
        b2 = StagingBuffer(4, 64, "cuda")
        t1 = torch.randn(2, 64, device="cuda", dtype=torch.float16)
        t2 = torch.randn(3, 64, device="cuda", dtype=torch.float16)
        b1.set_active(t1, 2)
        b2.set_active(t2, 3)
        assert b1.active_count == 2
        assert b2.active_count == 3
        assert torch.allclose(b1.buf[:2, 0], t1)
        assert torch.allclose(b2.buf[:3, 0], t2)

    def test_device_consistency(self):
        buf = StagingBuffer(4, 64, "cuda")
        assert buf.buf.device.type == "cuda"
        assert buf.residual.device.type == "cuda"
