# SPDX-License-Identifier: MIT
"""Qwen3.5-VL — the vision-language wrapper over the int8 `qwen3_5_text` backbone.

Composes:
  * the **int8 dp4a** Qwen3.5 text backbone (`models/qwen3_5.py`, loaded from the
    shipped `.superl8`), and
  * an **fp16** Qwen3.5 vision tower (`multimodal/qwen3_5_vit.py`, wrapping the
    upstream `Qwen3_5VisionModel`), loaded from the checkpoint's `model.visual.*`
    tensors (which the `.superl8` stores raw / fp16).

On an image prefill the tower turns `pixel_values` (+ `image_grid_thw`) into merged
patch embeddings and splices them into the token stream at the image-placeholder
positions; decode then proceeds through the text backbone unchanged.

The vision tower runs **once per image** so fp16 is cheap; an int8 dp4a ViT
(mirroring `multimodal/vit.py`) is a deferred follow-up — the accuracy gate, not
ideology, decides where int8 is worth it, and the ViT is a tiny slice of a decode.

Weight sources:
  * text + vision both in the checkpoint (Qwen3.5-0.8B `.superl8`) — the common case.
  * vision stripped by a text-only conversion (some 9B `.superl8`s) — set the env var
    `SUPERL8_VISION_WEIGHTS` to a raw HF safetensors file/dir to load the tower from.
"""

from __future__ import annotations

import glob
import os

import torch
import torch.nn as nn

from ..multimodal.qwen3_5_vit import Qwen3_5VisionTower, _strip_visual_prefix
from .base import ForwardContext
from .config import ModelConfig
from .qwen3_5 import build_qwen3_5
from .registry import register_model


def _load_sidecar_vision_weights() -> dict:
    """Load raw `visual.*` weights from `$SUPERL8_VISION_WEIGHTS` (a safetensors file or a
    directory of shards). Used when the `.superl8` was converted text-only."""
    path = os.environ.get("SUPERL8_VISION_WEIGHTS")
    if not path:
        return {}
    from safetensors.torch import load_file

    files = [path] if os.path.isfile(path) else sorted(
        glob.glob(os.path.join(path, "*.safetensors*"))
    )
    sd: dict = {}
    for f in files:
        part = load_file(f)
        for k, v in part.items():
            if "visual" in k:
                sd[k] = v
    return sd


class Qwen3_5VLForCausalLM(nn.Module):
    """VLM wrapper: fp16 vision tower + int8 Qwen3.5 text backbone in one forward."""

    def __init__(self, cfg: ModelConfig, weights: dict):
        super().__init__()
        self.config = cfg
        vcfg = cfg.vision_config
        assert vcfg is not None, "qwen3_5_vl requires a vision_config"

        # int8 text backbone (unwraps `model.language_model.*`, drops vision).
        self.lm = build_qwen3_5(cfg, weights)

        # fp16 vision tower — prefer in-checkpoint visual weights, else a sidecar.
        vis_weights = weights if _strip_visual_prefix(weights) else _load_sidecar_vision_weights()
        self.vision_tower = Qwen3_5VisionTower(vcfg, vis_weights)

        self.image_token_id = cfg.image_token_id or vcfg.image_token_id
        self.spatial_merge_size = vcfg.spatial_merge_size

        # Expose the text backbone's MTP speculative-decode head so the engine's
        # `getattr(model, "mtp")` dispatch reaches it for TEXT decode. Image requests
        # are guarded off in the engine (EngineRunner.decode) — see the guard there:
        # spec-decode's multi-token verify forward is unsafe for this hybrid family
        # (DeltaNet recurrent-state corruption) and doubly so mixed with vision.
        self.mtp = self.lm.mtp

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, ctx: ForwardContext
    ) -> torch.Tensor:
        h = self.lm.embed_tokens(input_ids)

        if ctx.is_prefill and ctx.pixel_values is not None:
            mask = input_ids == self.image_token_id
            if mask.any():
                if ctx.image_grid_thw is None:
                    raise ValueError(
                        "Qwen3.5-VL prefill has pixel_values but no image_grid_thw in "
                        "the ForwardContext — the vision tower needs the patch grid."
                    )
                img_embeds = self.vision_tower(ctx.pixel_values, ctx.image_grid_thw)
                n_img = int(mask.sum().item())
                if img_embeds.shape[0] != n_img:  # defensive: match count exactly
                    img_embeds = img_embeds[:n_img]
                h[mask] = img_embeds.to(h.dtype)

        residual = None
        for layer in self.lm.layers:
            h, residual = layer(h, positions, ctx, residual)
        h, _ = self.lm.norm(h, residual)
        return h

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm.compute_logits(hidden)


@register_model("qwen3_5_vl", "Qwen3_5ForConditionalGeneration")
def build_qwen3_5_vl(cfg: ModelConfig, weights: dict) -> Qwen3_5VLForCausalLM:
    return Qwen3_5VLForCausalLM(cfg, weights)
