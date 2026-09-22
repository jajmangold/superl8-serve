# SPDX-License-Identifier: MIT
"""DiffusionGemma — a Gemma-style model with bidirectional (non-causal) attention for
LLaDA / diffusion-based non-autoregressive text generation.

Reuses every shared layer; differs only in per-attn settings (causal=False) so the
same dp4a kernel (`attn_int8_fwd(causal=False)`) supplies full-bidirectional masking.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..layers.embedding import LMHead, VocabEmbedding
from ..layers.gqa_attention import GQAAttention
from ..layers.mlp import GatedMLP
from ..layers.norm import RMSNorm
from ..layers.rotary import RotaryEmbedding
from .base import ForwardContext
from .config import ModelConfig
from .registry import register_model
from .weights import gate_up_weight, qkv_weight, to_qtensor


def _gemma_norm(cfg, w):
    return RMSNorm(w.shape[-1], cfg.rms_norm_eps, w, add_unit_offset=True)


class DiffusionGemmaDecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int, sd: dict,
                 rope_global: RotaryEmbedding, rope_local: RotaryEmbedding):
        super().__init__()
        self.layer_idx = layer_idx
        p = f"model.layers.{layer_idx}"
        hd = cfg.resolved_head_dim()
        is_global = cfg.layer_is_global(layer_idx)
        rope = rope_global if is_global else rope_local
        window = -1 if is_global else int(cfg.sliding_window or -1)
        scale = (cfg.query_pre_attn_scalar ** -0.5) if cfg.query_pre_attn_scalar else hd ** -0.5
        self.self_attn = GQAAttention(
            num_heads=cfg.num_attention_heads, num_kv_heads=cfg.num_key_value_heads,
            head_dim=hd, qkv_proj=qkv_weight(sd, f"{p}.self_attn"),
            o_proj=to_qtensor(sd[f"{p}.self_attn.o_proj.weight"]), scale=scale, rope=rope,
            q_norm=sd.get(f"{p}.self_attn.q_norm.weight"),
            k_norm=sd.get(f"{p}.self_attn.k_norm.weight"),
            rms_norm_eps=cfg.rms_norm_eps, window_left=window,
            causal=False,
        )
        self.input_layernorm = _gemma_norm(cfg, sd[f"{p}.input_layernorm.weight"])
        self.post_attention_layernorm = _gemma_norm(cfg, sd[f"{p}.post_attention_layernorm.weight"])
        self.pre_feedforward_layernorm = _gemma_norm(cfg, sd[f"{p}.pre_feedforward_layernorm.weight"])
        self.post_feedforward_layernorm = _gemma_norm(cfg, sd[f"{p}.post_feedforward_layernorm.weight"])
        self.mlp = GatedMLP(gate_up_weight(sd, f"{p}.mlp"),
                            to_qtensor(sd[f"{p}.mlp.down_proj.weight"]), act=cfg.hidden_act)

    def forward(self, x, positions, ctx, residual=None):
        h = self.input_layernorm(x)
        h = self.self_attn(h, positions, ctx, self.layer_idx)
        h = self.post_attention_layernorm(h)
        x = x + h
        h = self.pre_feedforward_layernorm(x)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        x = x + h
        return x, None


class DiffusionGemmaModel(nn.Module):
    def __init__(self, cfg: ModelConfig, sd: dict):
        super().__init__()
        self.config = cfg
        scale = cfg.embed_scale or (cfg.hidden_size ** 0.5)
        self.embed_tokens = VocabEmbedding(sd["model.embed_tokens.weight"], embed_scale=scale, out_dtype=cfg.act_dtype())
        rope_global = RotaryEmbedding(cfg.resolved_head_dim(), cfg.max_position_embeddings,
                                      base=cfg.rope_theta)
        rope_local = RotaryEmbedding(cfg.resolved_head_dim(), cfg.max_position_embeddings,
                                     base=cfg.rope_local_theta or 1e4)
        self.layers = nn.ModuleList(
            [DiffusionGemmaDecoderLayer(cfg, i, sd, rope_global, rope_local)
             for i in range(cfg.num_hidden_layers)])
        self.norm = _gemma_norm(cfg, sd["model.norm.weight"])

    def forward(self, input_ids, positions, ctx):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h, _ = layer(h, positions, ctx, None)
        return self.norm(h)


class DiffusionGemmaForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig, sd: dict):
        super().__init__()
        self.config = cfg
        self.model = DiffusionGemmaModel(cfg, sd)
        self.lm_head = LMHead(to_qtensor(sd["model.embed_tokens.weight"]),
                              logit_softcap=cfg.final_logit_softcap)

    def forward(self, input_ids, positions, ctx: ForwardContext):
        return self.model(input_ids, positions, ctx)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)


@register_model("diffusion_gemma", "difussiongemmaforcausallm", "DiffusionGemmaForCausalLM")
def build_diffusion_gemma(cfg: ModelConfig, weights: dict) -> DiffusionGemmaForCausalLM:
    cfg.norm_add_unit_offset = True
    cfg.decode_strategy = "diffusion"
    return DiffusionGemmaForCausalLM(cfg, weights)
