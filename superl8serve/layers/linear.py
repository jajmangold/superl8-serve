# SPDX-License-Identifier: MIT
"""Linear seam — W8A8 / W4A8 dp4a matmul, fed from `.superl8` weights.

The weight is int8 (`per_row_i8`) or 4-bit (`per_group_i4`, int4/NF4) straight from
the `.superl8` container — no dequant at load. At runtime the activation is quantized to
int8 per-token and the matmul runs on dp4a.

The fast path is `superl8.linear(x, qt)`: it quantizes the activation per-token to int8
and runs the dp4a GEMM (`gemm_w8a8` for `per_row_i8`, `gemm_w4a8` for `per_group_i4`
int4). It is used whenever the weight is dp4a-compatible and x is on CUDA. NF4 weights
(non-integer lookup codebook — not dp4a-able) and CPU tensors fall back to a
NUMERICALLY-CORRECT fp16 path (reconstruct the weight, torch.matmul), marked SLOW.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import superl8
from superl8 import QTensor

try:  # NF4 codebook for 4-bit weight reconstruction (present in superl8)
    from superl8.quant.lowbit import NF4_CODEBOOK
except Exception:  # pragma: no cover
    NF4_CODEBOOK = None

def _has_native_kquant(codebook: str) -> bool:
    """Whether the installed superl8 exposes this format's fused public linear op."""
    op = f"linear_{codebook.replace('_', '')}"
    return callable(getattr(superl8, op, None))


# Backward-compatible name for callers/tests from the Q4-only bring-up.
_SUPERL8_HAS_Q4K = _has_native_kquant("q4_k")


def _dequant_weight(qt: QTensor) -> torch.Tensor:
    """Reconstruct an fp16 weight [out, in] from a stored QTensor (fallback path)."""
    if qt.scheme == "per_row_i8":
        return (qt.data.float() * qt.scale.unsqueeze(-1)).to(torch.float16)
    if qt.scheme == "per_group_i4":
        packed = qt.data.to(torch.int32)                       # [out, in//2]
        lo = packed & 0xF
        hi = (packed >> 4) & 0xF
        codes = torch.stack([lo, hi], -1).reshape(qt.data.shape[0], -1)  # [out, in], 0..15
        if qt.codebook == "nf4":
            cb = torch.tensor(NF4_CODEBOOK, device=codes.device, dtype=torch.float32)
            vals = cb[codes.long()]
        else:  # signed int4 two's-complement
            vals = codes.float() - 16.0 * (codes >= 8).float()
        g = qt.group_size
        out, in_ = vals.shape
        scale = qt.scale.reshape(out, in_ // g, 1)
        return (vals.reshape(out, in_ // g, g) * scale).reshape(out, in_).to(torch.float16)
    if qt.scheme == "per_group_w3a8":
        # Uniform 3-bit dp4a MLP (issue #181): int32 Q3_K-style bit-planes [out,(in/32)*3]
        # -> int8 codes in [-4,3] -> per-group dequant. Fallback only (dp4a is the fast path).
        from superl8.quant.lowbit import unpack_w3a8_bitplanes
        in_ = (qt.data.shape[-1] // 3) * 32
        codes = unpack_w3a8_bitplanes(qt.data, in_)                 # [out, in] int8 [-4,3]
        g = qt.group_size
        out = codes.shape[0]
        scale = qt.scale.reshape(out, in_ // g, 1)
        return (codes.float().reshape(out, in_ // g, g) * scale).reshape(out, in_).to(torch.float16)
    if qt.scheme == "gguf_kquant":
        # Native GGUF k-quant with no fused dp4a kernel yet (Q5_K/Q6_K) or on CPU:
        # dequant the raw bytes via llama.cpp's own block dequant (byte-faithful, no
        # re-quant) and run an fp16 matmul. Numerically correct until the P0 kernel.
        from ..gguf_native import dequant_kquant

        return dequant_kquant(qt)
    raise ValueError(f"LinearW8A8 cannot use scheme {qt.scheme!r}")


class LinearW8A8(nn.Module):
    def __init__(self, weight: QTensor, bias: torch.Tensor | None = None):
        super().__init__()
        self.weight = weight                       # QTensor (int8 or i4), resident dp4a layout
        self.bias = bias
        self.out_features = weight.data.shape[0]
        # in_features: int8 -> data.shape[1]; i4 packs 2/byte; w3a8 packs 32 vals / 3 int32.
        if weight.scheme == "per_group_i4":
            self.in_features = weight.data.shape[1] * 2
        elif weight.scheme == "per_group_w3a8":
            self.in_features = (weight.data.shape[1] // 3) * 32
        elif weight.scheme == "gguf_kquant":
            # raw k-quant bytes [out, n_blocks*type_size]; in = n_blocks*QK. The
            # Q*_K family blocks 256 values; TQ3_4S blocks 32 (16 B/block).
            if weight.codebook == "tq3_4s":
                self.in_features = (weight.data.shape[1] // 16) * 32
            else:
                _TS = {"q2_k": 84, "q3_k": 110, "q4_k": 144, "q5_k": 176, "q6_k": 210}[
                    weight.codebook
                ]
                self.in_features = (weight.data.shape[1] // _TS) * 256
        else:
            self.in_features = weight.data.shape[1]

    def _dp4a_ok(self, x: torch.Tensor) -> bool:
        """dp4a GEMM is CUDA-only and cannot consume NF4 (non-integer codebook)."""
        if not x.is_cuda:
            return False
        qt = self.weight
        if qt.scheme == "per_row_i8":
            return True
        if qt.scheme == "per_group_w3a8":
            return True                              # uniform 3-bit dp4a (issue #181)
        if qt.scheme == "gguf_kquant":
            return _has_native_kquant(qt.codebook)
        return qt.scheme == "per_group_i4" and qt.codebook == "int4"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._dp4a_ok(x):
            # Store the dp4a result in the ACTIVATION's dtype, not an unconditional
            # fp16. A bf16-native model (Gemma has documented ~1e4-1e7 "massive
            # activation" channels) feeds bf16 here; forcing the GEMM output to fp16
            # silently truncates fp16's 65504 range and overflows those channels to
            # inf -> NaN (the comfy #115 footgun). bf16 shares fp32's exponent, so an
            # out_dtype matched to x keeps the whole residual stream finite. The dp4a
            # kernel only stores fp16/bf16, so fall back to fp16 for any other x dtype.
            out_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.float16
            return superl8.linear(x, self.weight, bias=self.bias, out_dtype=out_dtype)  # dp4a
        w = _dequant_weight(self.weight).to(x.dtype)   # SLOW fp16 fallback (NF4 / CPU)
        return torch.nn.functional.linear(x, w, self.bias)
