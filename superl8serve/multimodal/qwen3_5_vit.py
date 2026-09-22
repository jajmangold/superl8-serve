# SPDX-License-Identifier: MIT
"""Qwen3.5-VL vision tower — fp16, oracle-faithful.

Qwen3.5's ViT differs from the Qwen2.5-VL tower in ``multimodal/vit.py`` on several
axes that all bear on numerics:

  * **Conv3d patch embed** over pre-flattened patches ``[num_patches, C*t*p*p]``
    (temporal_patch_size 2), not a Conv2d over ``[B, 3, H, W]``.
  * **Learned, bilinearly-interpolated absolute position embedding** (``pos_embed``,
    a ``[num_position_embeddings, hidden]`` table) added to the patch embeds — on top
    of the per-head 2D rotary used inside attention.
  * A **norm → MLP merger** that groups each ``spatial_merge_size**2`` window of
    patches and projects to the LLM hidden size (``out_hidden_size``).

Rather than re-derive the bilinear pos-embed interpolation / merge-window ordering
(and risk drifting from the reference), this tower wraps the upstream
``transformers`` ``Qwen3_5VisionModel`` and runs it in **fp16**. The tower runs once
per image, so fp16 is cheap; an int8 dp4a re-implementation (mirroring ``vit.py``) is
a deferred follow-up. Weights load from the checkpoint's ``model.visual.*`` tensors,
which the ``.superl8`` stores raw (fp16), matched to the raw HF tower at cos ≈ 1.0.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..models.config import VisionConfig


def _build_hf_vision_config(vcfg: VisionConfig):
    """Reconstruct the upstream ``Qwen3_5VisionConfig`` from the resolved
    ``VisionConfig``. Prefers the verbatim ``raw`` HF sub-dict (future-proof against
    fields serve does not model), falling back to the parsed scalars."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

    if vcfg.raw:
        return Qwen3_5VisionConfig(**vcfg.raw)
    return Qwen3_5VisionConfig(
        depth=vcfg.num_layers,
        hidden_size=vcfg.hidden_size,
        num_heads=vcfg.num_attention_heads,
        intermediate_size=vcfg.intermediate_size,
        patch_size=vcfg.patch_size,
        spatial_merge_size=vcfg.spatial_merge_size,
        temporal_patch_size=vcfg.temporal_patch_size,
        in_channels=vcfg.in_channels,
        hidden_act=vcfg.hidden_act,
        num_position_embeddings=vcfg.num_position_embeddings,
        out_hidden_size=vcfg.out_hidden_size or vcfg.hidden_size,
    )


class Qwen3_5VisionTower(nn.Module):
    """fp16 Qwen3.5 vision tower. ``forward(pixel_values, grid_thw)`` returns merged
    patch embeddings ``[num_merged_tokens, out_hidden_size]`` ready to splice into the
    LLM token stream at the image-placeholder positions.

    ``num_merged_tokens = sum(t*gh*gw) / spatial_merge_size**2`` — one row per LLM
    image placeholder token.
    """

    def __init__(self, vcfg: VisionConfig, weights: dict, *, dtype=torch.float16):
        super().__init__()
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

        hf_cfg = _build_hf_vision_config(vcfg)
        model = Qwen3_5VisionModel(hf_cfg)
        vis_sd = _strip_visual_prefix(weights)
        if not vis_sd:
            raise KeyError(
                "no `model.visual.*` weights found for the Qwen3.5 vision tower — the "
                "checkpoint was converted text-only (vision stripped). Supply a raw HF "
                "vision checkpoint (see build_qwen3_5_vl)."
            )
        missing, unexpected = model.load_state_dict(vis_sd, strict=False)
        # `pos_embed` / rotary buffers are non-persistent; a few unmatched buffers are
        # fine, but a missing *parameter* means the checkpoint layout drifted.
        real_missing = [k for k in missing if "inv_freq" not in k]
        if real_missing:
            raise KeyError(f"vision tower missing weights: {real_missing[:8]}")
        self.model = model.to(dtype).eval()
        self.spatial_merge_size = hf_cfg.spatial_merge_size
        self.out_hidden_size = hf_cfg.out_hidden_size

    @torch.inference_mode()
    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        p = self.model.parameters().__next__()
        pixel_values = pixel_values.to(device=p.device, dtype=p.dtype)
        grid_thw = grid_thw.to(p.device)
        out = self.model(pixel_values, grid_thw=grid_thw)
        return out.pooler_output  # [num_merged, out_hidden_size]


def _strip_visual_prefix(weights: dict) -> dict:
    """Pull ``model.visual.*`` (or bare ``visual.*``) tensors out of a checkpoint
    state dict, stripping the prefix to the HF ``Qwen3_5VisionModel`` key names, and
    unwrapping any raw ``QTensor`` to its plain fp16 tensor."""
    out: dict = {}
    for k, v in weights.items():
        for pref in ("model.visual.", "visual."):
            if k.startswith(pref):
                t = v if torch.is_tensor(v) else getattr(v, "data", v)
                out[k[len(pref):]] = t
                break
    return out
