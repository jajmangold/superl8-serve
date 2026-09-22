# SPDX-License-Identifier: MIT
"""Qwen2.5-VL Vision Transformer (ViT) forward pass — int8 dp4a linears + fp16 attention.

Architecture matches the HF Qwen2.5-VL ViT (Conv2d patch embed, pre-norm LayerNorm
transformer blocks with merged QKV projection, M-RoPE 2D positional encoding, GELU-
tanh MLP, no post-transformer norm).

Every linear projection inside the blocks (QKV, O, fc1, fc2) runs on the dp4a GEMM
via ``LinearW8A8``. Attention (softmax) and norms stay fp — norms in fp32, softmax
in fp16. Single-batch prefill only; no projector / embed-merge; no dynamic resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from superl8 import QTensor

from ..layers.linear import LinearW8A8
from ..models.weights import to_qtensor


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies RoPE to Q, K.  q/k: [B, H, S, D]; cos/sin: [1, 1, S, D]."""
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _compute_2d_rope(
    grid_h: int,
    grid_w: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """2D M-RoPE cos/sin tables matching Qwen2.5-VL.

    The head dimension is split in two halves — the first half rotates with
    row positions and the second half with column positions, which is the
    standard ``Qwen2_5_VisionRotaryEmbedding`` convention for images.
    """
    d = head_dim // 4
    inv_freq = 1.0 / (base ** (torch.arange(0, d, dtype=torch.float32, device=device) / d))

    h_idx = torch.arange(grid_h, device=device).unsqueeze(1).expand(-1, grid_w).reshape(-1)
    w_idx = torch.arange(grid_w, device=device).unsqueeze(0).expand(grid_h, -1).reshape(-1)

    freqs_h = h_idx[:, None].float() * inv_freq[None, :]  # [S, d]
    freqs_w = w_idx[:, None].float() * inv_freq[None, :]  # [S, d]

    freqs = torch.cat([freqs_h, freqs_w], dim=-1)  # [S, 2d] = [S, head_dim//2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [S, head_dim]

    cos = emb.cos().to(dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, S, head_dim]
    sin = emb.sin().to(dtype).unsqueeze(0).unsqueeze(0)
    return cos, sin


class ViTAttention(nn.Module):
    """Multi-head self-attention with int8 dp4a QKV + O projections, fp16 softmax."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        qkv_weight: QTensor,
        qkv_bias: torch.Tensor | None,
        o_weight: QTensor,
        o_bias: torch.Tensor | None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.qkv_proj = LinearW8A8(qkv_weight, qkv_bias)
        self.o_proj = LinearW8A8(o_weight, o_bias)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        qkv = self.qkv_proj(x)  # [B, S, 3*hidden]
        q, k, v = qkv.split(self.num_heads * self.head_dim, dim=-1)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = _apply_rotary_pos_emb(q, k, cos, sin)  # fp16 RoPE

        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)  # fp16 softmax
        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(out)  # int8 dp4a


class ViTMLP(nn.Module):
    """Two-layer MLP with int8 dp4a projections and GELU-tanh activation."""

    def __init__(
        self,
        fc1_weight: QTensor,
        fc1_bias: torch.Tensor | None,
        fc2_weight: QTensor,
        fc2_bias: torch.Tensor | None,
    ):
        super().__init__()
        self.fc1 = LinearW8A8(fc1_weight, fc1_bias)
        self.fc2 = LinearW8A8(fc2_weight, fc2_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        return self.fc2(x)


class ViTBlock(nn.Module):
    """One ViT transformer block: pre-norm attn + mlp with residuals."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        head_dim: int,
        layer_norm_eps: float,
        qkv_weight: QTensor,
        qkv_bias: torch.Tensor | None,
        o_weight: QTensor,
        o_bias: torch.Tensor | None,
        fc1_weight: QTensor,
        fc1_bias: torch.Tensor | None,
        fc2_weight: QTensor,
        fc2_bias: torch.Tensor | None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attn = ViTAttention(
            hidden_size, num_heads, head_dim, qkv_weight, qkv_bias, o_weight, o_bias
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = ViTMLP(fc1_weight, fc1_bias, fc2_weight, fc2_bias)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Qwen2.5-VL vision tower — int8 dp4a linears for QKV/MLP, fp16 attention.

    Loads quantized weights and runs a single-batch static-shape ViT forward on
    ``pixel_values`` [B, 3, H, W], producing patch embeddings [B, S, hidden].

    Constructor accepts the resolved ``VisionConfig`` plus a ``weights`` dict keyed
    like ``blocks.0.attn.qkv.weight`` (QSUPERL8Tensor for linears, raw tensors for
    embeddings / norms / biases).
    """

    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        head_dim: int,
        in_channels: int,
        layer_norm_eps: float,
        weights: dict,
        *,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rope_base = rope_base

        self.patch_embed = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=False
        )
        if "patch_embed.weight" in weights:
            pw = weights["patch_embed.weight"]
            self.patch_embed.weight = nn.Parameter(pw.detach().clone())

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            p = f"blocks.{i}"
            self.blocks.append(
                ViTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                    head_dim=head_dim,
                    layer_norm_eps=layer_norm_eps,
                    qkv_weight=weights[f"{p}.attn.qkv.weight"],
                    qkv_bias=weights.get(f"{p}.attn.qkv.bias"),
                    o_weight=weights[f"{p}.attn.proj.weight"],
                    o_bias=weights.get(f"{p}.attn.proj.bias"),
                    fc1_weight=weights[f"{p}.mlp.fc1.weight"],
                    fc1_bias=weights.get(f"{p}.mlp.fc1.bias"),
                    fc2_weight=weights[f"{p}.mlp.fc2.weight"],
                    fc2_bias=weights.get(f"{p}.mlp.fc2.bias"),
                )
            )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(pixel_values)  # [B, hidden, gh, gw]
        x = x.flatten(2).transpose(1, 2)  # [B, S, hidden]
        S = x.shape[1]
        gh = pixel_values.shape[2] // self.patch_size
        gw = pixel_values.shape[3] // self.patch_size
        cos, sin = _compute_2d_rope(gh, gw, self.head_dim, x.device, x.dtype, base=self.rope_base)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return x
