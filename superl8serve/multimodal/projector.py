# SPDX-License-Identifier: MIT
"""Multimodal projector — maps vision-tower patch embeddings into LLM embedding space.

Two projector families:
  * ``MLPProjector``  (2-layer GELU, Qwen2.5-VL / Qwen2-VL style).
  * ``LinearProjector`` (single linear layer, LLaVA / LLaVA-Next style).

Helpers:
  * ``embed_merge`` — splice projected patch embeddings at image-token positions.
  * ``compute_mrope_position_ids`` — 3D (temporal, height, width) position IDs
    for Qwen2.5-VL mrope.

All modules run in fp16; the projector is the accuracy-sensitive bridge between the
int8 vision tower and the fp16/fp32 LLM, and its two matmuls are tiny (<5% of prefill).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPProjector(nn.Module):
    """2-layer GELU MLP projector (Qwen2.5-VL / Qwen2-VL).

    ``vision_hidden_size`` -> ``llm_hidden_size`` -> ``llm_hidden_size``,
    with GELU-tanh in between.
    """

    def __init__(self, vision_hidden_size: int, llm_hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(vision_hidden_size, llm_hidden_size, bias=True)
        self.fc2 = nn.Linear(llm_hidden_size, llm_hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.fc2(x)
        return x


class LinearProjector(nn.Module):
    """Single linear projection (LLaVA / LLaVA-Next).

    ``vision_hidden_size`` -> ``llm_hidden_size``, no activation.
    """

    def __init__(self, vision_hidden_size: int, llm_hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(vision_hidden_size, llm_hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def build_projector(
    vision_hidden_size: int,
    llm_hidden_size: int,
    arch: str,
    weights: dict | None = None,
    prefix: str = "",
) -> nn.Module:
    """Factory: create a projector for the given architecture.

    Args:
        vision_hidden_size: ViT output dimension.
        llm_hidden_size: LLM hidden dimension.
        arch: Model architecture key (``"qwen2_5_vl"``, ``"qwen2_vl"``,
              ``"llava"``, ``"llava_next"``).
        weights: Optional state-dict slice whose keys match the projector's
                 parameter names (e.g. ``fc1.weight``, ``fc1.bias``, etc.).
                 When provided the weights are copied into the module.
        prefix: Key prefix to strip when looking up weights (e.g. ``"merger."``).

    Returns:
        A projector ``nn.Module`` (MLP or linear).
    """
    if arch in ("qwen2_5_vl", "qwen2_vl"):
        proj = MLPProjector(vision_hidden_size, llm_hidden_size)
    else:
        proj = LinearProjector(vision_hidden_size, llm_hidden_size)

    if weights is not None:
        proj.load_state_dict(_slice_weights(weights, prefix))
    return proj


def _slice_weights(weights: dict, prefix: str) -> dict:
    """Extract a sub-dict whose keys have the given prefix stripped."""
    prefix_len = len(prefix)
    out = {}
    for k, v in weights.items():
        if k.startswith(prefix):
            out[k[prefix_len:]] = v
    return out


def embed_merge(
    token_embeds: torch.Tensor,
    vision_embeds: torch.Tensor,
    image_token_id: int,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Replace image-token rows in ``token_embeds`` with ``vision_embeds``.

    Args:
        token_embeds: ``[1, seq_len, llm_hidden]`` — the raw token embedding
                      lookup (non-image tokens are already placed).
        vision_embeds: ``[1, num_patches, llm_hidden]`` — projected patch
                       embeddings from the vision tower + projector.
        image_token_id: The integer token id used as the image placeholder
                        (e.g. ``151654`` for Qwen2.5-VL).
        input_ids: ``[1, seq_len]`` — the token ids used to look up
                   ``token_embeds``.

    Returns:
        ``[1, seq_len, llm_hidden]`` — merged embedding with projected patches
        inserted at each image-token position.
    """
    mask = input_ids == image_token_id
    n_img_toks = mask.sum().item()
    if n_img_toks == 0:
        return token_embeds
    n_patches = vision_embeds.shape[1]
    if n_img_toks != n_patches:
        # If there are fewer image tokens than patches, we tile or truncate;
        # in the standard case they match exactly.
        vision_embeds = vision_embeds[:, :n_img_toks]

    merged = token_embeds.clone()
    merged[mask] = vision_embeds
    return merged


def compute_mrope_position_ids(
    seq_len: int,
    num_patches: int,
    grid_h: int,
    grid_w: int,
    image_start_pos: int,
    device: torch.device,
) -> torch.Tensor:
    """3D MRoPE position IDs for Qwen2.5-VL.

    Produces a ``[3, 1, seq_len]`` tensor where:
      - ``[0]`` = temporal position (0 for image patches, text pos for tokens).
      - ``[1]`` = height position in the patch grid (0..grid_h-1).
      - ``[2]`` = width position in the patch grid (0..grid_w-1).

    Text tokens get identical values in all three dims (standard 1D position).
    Image-token patches get a temporal offset of 0 with their 2D grid position.

    Args:
        seq_len: Total sequence length (text + image tokens).
        num_patches: Number of image patches (grid_h * grid_w).
        grid_h: Patch grid height (after any spatial merge).
        grid_w: Patch grid width.
        image_start_pos: Position in the sequence where the image tokens begin.
        device: Target device.

    Returns:
        ``[3, 1, seq_len]`` long tensor.
    """
    pos_1d = torch.arange(seq_len, device=device)
    temporal = pos_1d.clone()
    height = pos_1d.clone()
    width = pos_1d.clone()

    # Height and width indices for each patch in raster order
    h_idx = torch.arange(grid_h, device=device).repeat_interleave(grid_w)
    w_idx = torch.arange(grid_w, device=device).tile(grid_h)

    patch_end = min(image_start_pos + num_patches, seq_len)
    n_p = patch_end - image_start_pos
    temporal[image_start_pos:patch_end] = 0
    height[image_start_pos:patch_end] = h_idx[:n_p]
    width[image_start_pos:patch_end] = w_idx[:n_p]

    return torch.stack([temporal, height, width], dim=0).unsqueeze(1)
