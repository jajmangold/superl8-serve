# SPDX-License-Identifier: MIT
"""Rotary position embedding (RoPE), Qwen3/Llama "rotate_half" convention.

Qwen3 applies RoPE to Q and K per head AFTER the per-head QK-norm. cos/sin are
precomputed for all positions up to `max_position` and gathered by the per-token
position ids (so prefill and paged decode share one table). Kept in fp32 for the
gather then applied in the tensor's dtype.
"""
from __future__ import annotations

import superl8
import torch
import torch.nn as nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(
        self, head_dim: int, max_position: int, base: float = 1e6, rotary_dim: int | None = None
    ):
        super().__init__()
        # Partial rotary (GLM, Qwen3-Next gated attn): rotate only the first
        # `rotary_dim` dims of each head, pass the rest through unchanged.
        rd = rotary_dim if rotary_dim is not None else head_dim
        assert rd % 2 == 0 and rd <= head_dim, "rotary_dim must be even and <= head_dim"
        self.rotary_dim = rd
        inv_freq = 1.0 / (base ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))
        t = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)                      # [max_pos, rotary_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)               # [max_pos, rotary_dim]
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def _rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if self.rotary_dim == x.shape[-1]:
            return x * cos + _rotate_half(x) * sin
        x_rot, x_pass = x[..., : self.rotary_dim], x[..., self.rotary_dim :]
        return torch.cat((x_rot * cos + _rotate_half(x_rot) * sin, x_pass), dim=-1)

    def forward(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """positions: [...] int; q, k: [..., n_heads, head_dim]. Broadcasts cos/sin
        over the head axis."""
        # Fused single-launch-per-tensor path (superl8.rope rotates in place using the
        # fp32 cos/sin tables); eager fallback on CPU / non-fp16.
        if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
            return superl8.rope(positions, q, k, self.cos, self.sin, self.rotary_dim)
        cos = self.cos[positions].unsqueeze(-2).to(q.dtype)   # [..., 1, rotary_dim]
        sin = self.sin[positions].unsqueeze(-2).to(q.dtype)
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)
