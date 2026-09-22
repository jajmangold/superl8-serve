# SPDX-License-Identifier: MIT
"""Gemma3 (text) — the second concrete architecture; proves the design is modular
(reuses every shared layer, differs only in config + decoder-layer assembly).

Verified against HF modeling_gemma3.py:
  * 5-local:1-global sliding-window attention — layer i is global iff (i+1)%6==0;
    local layers use a 1e4 RoPE base + `sliding_window` mask, global use 1e6.
  * per-head QK-norm (over head_dim) pre-RoPE; scale = query_pre_attn_scalar**-0.5
    (NOT head_dim**-0.5 on the 27b); head_dim 256 (4b) / 128 (27b).
  * RMSNorm is (1+w), eps 1e-6; FOUR norms per layer with post_attention and
    post_feedforward norms applied to the sub-block output BEFORE the residual add
    (the Gemma sandwich), not fused into the residual like Qwen.
  * GeGLU (gelu_pytorch_tanh) MLP; embeddings scaled by sqrt(hidden); tied head.
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


class Gemma3DecoderLayer(nn.Module):
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
        )
        self.input_layernorm = _gemma_norm(cfg, sd[f"{p}.input_layernorm.weight"])
        self.post_attention_layernorm = _gemma_norm(cfg, sd[f"{p}.post_attention_layernorm.weight"])
        self.pre_feedforward_layernorm = _gemma_norm(cfg, sd[f"{p}.pre_feedforward_layernorm.weight"])
        self.post_feedforward_layernorm = _gemma_norm(cfg, sd[f"{p}.post_feedforward_layernorm.weight"])
        self.mlp = GatedMLP(gate_up_weight(sd, f"{p}.mlp"),
                            to_qtensor(sd[f"{p}.mlp.down_proj.weight"]), act=cfg.hidden_act)

    def forward(self, x, positions, ctx, residual=None):
        # Gemma sandwich: norm the sub-block OUTPUT before adding the residual.
        h = self.input_layernorm(x)
        h = self.self_attn(h, positions, ctx, self.layer_idx)
        h = self.post_attention_layernorm(h)
        x = x + h
        h = self.pre_feedforward_layernorm(x)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        x = x + h
        return x, None


class Gemma3Model(nn.Module):
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
            [Gemma3DecoderLayer(cfg, i, sd, rope_global, rope_local)
             for i in range(cfg.num_hidden_layers)])
        self.norm = _gemma_norm(cfg, sd["model.norm.weight"])

    def forward(self, input_ids, positions, ctx):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h, _ = layer(h, positions, ctx, None)
        return self.norm(h)


class Gemma3ForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig, sd: dict):
        super().__init__()
        self.config = cfg
        self.model = Gemma3Model(cfg, sd)
        # Gemma ties the head to the (unscaled) embedding table; quantize it to
        # int8 dp4a (the fp16 logits GEMM runs on the fleet's crippled tensor
        # cores). The embedding lookup keeps its own fp16 table.
        self.lm_head = LMHead(to_qtensor(sd["model.embed_tokens.weight"]),
                              logit_softcap=cfg.final_logit_softcap)

    def forward(self, input_ids, positions, ctx: ForwardContext):
        return self.model(input_ids, positions, ctx)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)


@register_model("gemma3", "gemma3_text", "Gemma3ForCausalLM", "Gemma3ForConditionalGeneration")
def build_gemma3(cfg: ModelConfig, weights: dict) -> Gemma3ForCausalLM:
    cfg.norm_add_unit_offset = True
    return Gemma3ForCausalLM(cfg, weights)
