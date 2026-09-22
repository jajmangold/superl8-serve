# SPDX-License-Identifier: MIT
"""Massive-activation fp16-overflow guard (issue #255).

The comfy #115 NaN was a bf16/fp32 activation silently STORED as fp16 in the dp4a
linear wrapper: fp16 tops out at 65504, so a "massive activation" channel (Gemma has
documented residual-stream channels reaching ~1e4-1e7) overflowed to inf -> NaN. bf16
shares fp32's 8-bit exponent (same ~3.4e38 range), so a bf16-native model carries those
channels fine -- UNLESS a layer downcasts the stream back to fp16 mid-flight.

`LinearW8A8` (the dp4a GEMM wrapper) is the one op that used to hard-code an fp16 store
regardless of the activation dtype. These tests pin that it now stores in the
activation's dtype, so a bf16 stream stays finite through the int8 matmul.
"""
from __future__ import annotations

import pytest
import torch

from superl8 import QTensor
from superl8.quant.core import quantize_int8_rowwise
from superl8serve.layers.linear import LinearW8A8

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="dp4a GEMM is CUDA-only")

_D = 512
_MASSIVE = 1e7  # well past fp16's 65504 ceiling; comfortably inside bf16 range


def _int8_linear(out: int = _D, in_: int = _D) -> LinearW8A8:
    w = torch.randn(out, in_, device="cuda", dtype=torch.float16) * 0.02
    q, s = quantize_int8_rowwise(w)
    qt = QTensor(q.contiguous(), s.squeeze(-1).float().contiguous(), scheme="per_row_i8")
    return LinearW8A8(qt)


def test_bf16_massive_activation_stays_finite_through_dp4a():
    """A finite bf16 activation with ~1e7 channels must survive the int8 GEMM finite --
    the output must be bf16 (not silently fp16), else 1e7 overflows to inf/NaN."""
    lin = _int8_linear()
    x = torch.randn(8, _D, device="cuda", dtype=torch.bfloat16) * _MASSIVE
    assert torch.isfinite(x).all()  # bf16 represents 1e7 fine

    out = lin(x)

    assert out.dtype == torch.bfloat16, "dp4a wrapper downcast the bf16 stream to fp16"
    assert torch.isfinite(out).all(), "massive bf16 activation overflowed in the int8 GEMM"


def test_dp4a_output_dtype_tracks_activation_dtype():
    """The store dtype follows the activation: bf16 in -> bf16 out, fp16 in -> fp16 out.
    (fp16 in cannot even represent 1e7, so that regime is inherently lossy and not a
    wrapper bug -- we only pin the dtype threading here.)"""
    lin = _int8_linear()
    for dt in (torch.float16, torch.bfloat16):
        out = lin(torch.randn(4, _D, device="cuda", dtype=dt))
        assert out.dtype == dt


def test_bf16_and_fp16_agree_on_moderate_activations():
    """Sanity: switching the store dtype to bf16 doesn't change results in the normal
    range -- bf16 and fp16 outputs match closely for a benign activation."""
    lin = _int8_linear()
    x16 = torch.randn(4, _D, device="cuda", dtype=torch.float16)
    x_bf = x16.to(torch.bfloat16)
    o16 = lin(x16).float()
    o_bf = lin(x_bf).float()
    cos = torch.nn.functional.cosine_similarity(o16.flatten(), o_bf.flatten(), dim=0)
    assert cos > 0.99, f"bf16/fp16 dp4a diverged on benign input (cos={cos.item():.4f})"
