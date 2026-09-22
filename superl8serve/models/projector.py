# SPDX-License-Identifier: MIT
"""Multimodal projectors -- map ViT patch embeddings into the LLM embedding space."""
from __future__ import annotations
import torch.nn as nn


class Qwen2_5_VLPatchMerger(nn.Module):
    """Spatial patch merger + GELU MLP projector for Qwen2.5-VL."""

    def __init__(self, vit_hidden_size, llm_hidden_size, spatial_merge_size=2, eps=1e-6):
        super().__init__()
        self.spatial_merge_unit = spatial_merge_size * spatial_merge_size
        hidden_size = vit_hidden_size * self.spatial_merge_unit
        self.ln_q = nn.LayerNorm(vit_hidden_size, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, llm_hidden_size),
        )

    def forward(self, x):
        B, S, D = x.shape
        smu = self.spatial_merge_unit
        return self.mlp(self.ln_q(x).reshape(B, S // smu, smu * D))


class LLaVAProjector(nn.Module):
    """2-layer GELU MLP projector for LLaVA / LLaVA-NeXT."""

    def __init__(self, vit_hidden_size, llm_hidden_size, bias=True):
        super().__init__()
        self.linear_1 = nn.Linear(vit_hidden_size, llm_hidden_size, bias=bias)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(llm_hidden_size, llm_hidden_size, bias=bias)

    def forward(self, x):
        return self.linear_2(self.act(self.linear_1(x)))
