# SPDX-License-Identifier: MIT
"""Multimodal VLM wrapper — composes a vision tower (ViT), projector, and
language model (a registered CausalLM) so that an image prefill produces the
correct merged embedding before the first decode step.

Architectures supported:
  * Qwen2.5-VL / Qwen2-VL — MLP projector + MRoPE position IDs
  * LLaVA / LLaVA-Next — linear projector, simple insert

The wrapper's ``forward`` intercepts the token-embedding lookup: it replaces
image-token rows with projected patch embeddings from the vision tower, then
continues through the LLM's transformer layers.

Pixel values arrive via ``ForwardContext.pixel_values`` (set by the runner when
the sequence carries image data). Image preprocessing (#76) feeds this field.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from ..multimodal.projector import build_projector, embed_merge
from ..multimodal.vit import VisionTransformer
from .base import ForwardContext
from .config import ModelConfig
from .registry import build_model, register_model


# Map multimodal arch -> text-model arch used for the underlying LLM.
_TEXT_ARCH_MAP = {
    "qwen2_5_vl": "qwen3",
    "qwen2_vl": "qwen3",
    "llava": "qwen3",
    "llava_next": "qwen3",
}


class MultimodalCausalLM(nn.Module):
    """VLM wrapper: ViT + projector + LLM in one forward pass.

    The underlying language model (``lm``) is a ``ForCausalLM`` instance
    (e.g. ``Qwen3ForCausalLM``).  Its internal ``.model`` submodule provides
    ``embed_tokens``, ``layers``, and ``norm``; we intercept the embedding
    lookup to splice in projected vision embeddings, then forward through the
    LM's transformer layers.
    """

    def __init__(
        self,
        vision_tower: VisionTransformer,
        projector: nn.Module,
        lm: nn.Module,
        image_token_id: int,
        arch: str,
    ):
        super().__init__()
        self.config = lm.config
        self.vision_tower = vision_tower
        self.projector = projector
        self.lm = lm
        self.image_token_id = image_token_id
        self.arch = arch

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, ctx: ForwardContext
    ) -> torch.Tensor:
        lm_model = self.lm.model

        h = lm_model.embed_tokens(input_ids)

        if ctx.is_prefill and ctx.pixel_values is not None:
            mask = input_ids == self.image_token_id
            if mask.any():
                vit_out = self.vision_tower(ctx.pixel_values)
                proj_out = self.projector(vit_out)
                h = embed_merge(h, proj_out, self.image_token_id, input_ids)

        residual = None
        for layer in lm_model.layers:
            h, residual = layer(h, positions, ctx, residual)
        h, _ = lm_model.norm(h, residual)
        return h

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm.compute_logits(hidden)


# weight extraction helpers


def _load_vit_weights(sd: dict, prefix: str = "visual.") -> dict:
    vit_weights = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            vit_weights[k[len(prefix) :]] = v
    return vit_weights


def _load_projector_weights(sd: dict, arch: str) -> dict:
    proj_weights = {}
    if arch in ("qwen2_5_vl", "qwen2_vl"):
        mapping = {
            "merger.0.weight": "fc1.weight",
            "merger.0.bias": "fc1.bias",
            "merger.2.weight": "fc2.weight",
            "merger.2.bias": "fc2.bias",
        }
    else:
        mapping = {
            "model.mm_projector.weight": "proj.weight",
            "model.mm_projector.bias": "proj.bias",
        }
    for src, dst in mapping.items():
        if src in sd:
            proj_weights[dst] = sd[src]
    return proj_weights


def _build_vision_tower(vcfg, sd: dict) -> VisionTransformer:
    vit_weights = _load_vit_weights(sd)
    return VisionTransformer(
        hidden_size=vcfg.hidden_size,
        patch_size=vcfg.patch_size,
        num_layers=vcfg.num_layers,
        num_heads=vcfg.num_attention_heads,
        intermediate_size=vcfg.intermediate_size,
        head_dim=vcfg.head_dim,
        in_channels=vcfg.in_channels,
        layer_norm_eps=vcfg.layer_norm_eps,
        weights=vit_weights,
    )


def _build_projector(vcfg, llm_hidden_size: int, arch: str, sd: dict) -> nn.Module:
    proj_weights = _load_projector_weights(sd, arch)
    prefix = "merger." if arch in ("qwen2_5_vl", "qwen2_vl") else "model.mm_projector."
    return build_projector(
        vcfg.hidden_size,
        llm_hidden_size,
        arch,
        weights=proj_weights,
        prefix=prefix if proj_weights else "",
    )


def build_multimodal_model(cfg: ModelConfig, weights: dict) -> MultimodalCausalLM:
    vcfg = cfg.vision_config
    assert vcfg is not None, "vision_config required for multimodal model"

    vision_tower = _build_vision_tower(vcfg, weights)
    projector = _build_projector(vcfg, cfg.hidden_size, cfg.arch, weights)

    text_cfg = copy.copy(cfg)
    text_cfg.arch = _TEXT_ARCH_MAP.get(cfg.arch, "qwen3")
    lm = build_model(text_cfg, weights)

    img_tok = cfg.image_token_id or vcfg.image_token_id

    return MultimodalCausalLM(
        vision_tower=vision_tower,
        projector=projector,
        lm=lm,
        image_token_id=img_tok,
        arch=cfg.arch,
    )


@register_model("qwen2_5_vl", "Qwen2_5VLForConditionalGeneration")
def build_qwen2_5_vl(cfg: ModelConfig, weights: dict) -> MultimodalCausalLM:
    return build_multimodal_model(cfg, weights)


@register_model("qwen2_vl", "Qwen2VLForConditionalGeneration")
def build_qwen2_vl(cfg: ModelConfig, weights: dict) -> MultimodalCausalLM:
    return build_multimodal_model(cfg, weights)


@register_model("llava", "LlavaForConditionalGeneration")
def build_llava(cfg: ModelConfig, weights: dict) -> MultimodalCausalLM:
    return build_multimodal_model(cfg, weights)


@register_model("llava_next", "LlavaNextForConditionalGeneration")
def build_llava_next(cfg: ModelConfig, weights: dict) -> MultimodalCausalLM:
    return build_multimodal_model(cfg, weights)
