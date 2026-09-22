# SPDX-License-Identifier: MIT
"""NF4 quantization helpers for LoRA adapter weights (QLoRA, issue #129)."""

from __future__ import annotations

import torch

try:
    from superl8.quant.lowbit import NF4_CODEBOOK
except ImportError:
    NF4_CODEBOOK = None


def quantize_adapter_nf4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an fp16 adapter weight to NF4.

    Returns ``(codes, scale)`` where *codes* is uint8 (packed 2-per-byte) and
    *scale* is the per-block fp16 scale. Block size is 64.
    """
    w = weight.float()
    w_flat = w.flatten()
    block_size = 64
    pad = (block_size - w_flat.size(0) % block_size) % block_size
    if pad:
        w_flat = torch.cat([w_flat, w_flat.new_zeros(pad)])
    w_blocks = w_flat.view(-1, block_size)
    absmax = w_blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    normalized = w_blocks / absmax
    scale = absmax.squeeze(-1)

    cb = torch.tensor(NF4_CODEBOOK, device=weight.device, dtype=torch.float32)
    codes_flat = (normalized.unsqueeze(-1) - cb).abs().argmin(dim=-1)
    codes = codes_flat.to(torch.uint8)
    codes = codes.reshape(-1, block_size // 2)
    codes = (codes[:, ::2] << 4) | codes[:, 1::2]
    return codes.reshape(-1).contiguous(), scale.half().contiguous()


def dequantize_adapter_nf4(
    codes: torch.Tensor,
    scale: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Dequantize an NF4-quantized adapter weight back to fp16.

    Args:
        codes: Packed uint8 codes (2 values per byte).
        scale: Per-block fp16 scales.
        shape: Original (out_features, in_features) shape.

    Returns:
        fp16 weight tensor of *shape*.
    """
    cb = torch.tensor(NF4_CODEBOOK, device=codes.device, dtype=torch.float32)
    codes_u8 = codes.to(torch.uint8)
    lo = (codes_u8 >> 4).to(torch.long)
    hi = (codes_u8 & 0xF).to(torch.long)
    n = codes_u8.numel() * 2
    indices = torch.empty(n, dtype=torch.long, device=codes.device)
    indices[0::2] = lo
    indices[1::2] = hi
    vals = cb[indices]

    block_size = 64
    vals = vals.view(-1, block_size)
    s = scale.view(-1, 1).float()
    w = (vals * s).flatten()
    n_expected = shape[0] * shape[1]
    if w.numel() > n_expected:
        w = w[:n_expected]
    return w.reshape(shape).half()
