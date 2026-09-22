# SPDX-License-Identifier: MIT
"""GLM-4.5 / GLM-4.6 (glm4_moe) — GQA (partial RoPE 0.5, QK-norm, QKV bias) + sigmoid
MoE with correction bias, shared expert, and dense-then-MoE layers. Registered `glm`.

Per HF `glm4_moe`: 2 norms/layer (no sandwich); q/k/v_proj carry bias, o_proj none;
partial_rotary_factor 0.5; QK-norm on 4.5/4.6; MoE router = sigmoid + top-k selection
biased by mlp.gate.e_score_correction_bias, weights = unbiased sigmoid, ×
routed_scaling_factor; first_k_dense_replace dense layers then MoE; 1 shared expert.
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
from .moe import SparseMoE
from .registry import register_model
from .weights import gate_up_weight, merge_qtensor, to_qtensor


def _attn(cfg, sd, p, rope):
    hd = cfg.resolved_head_dim()
    a = f"{p}.self_attn"
    qkv = merge_qtensor([sd[f"{a}.q_proj.weight"], sd[f"{a}.k_proj.weight"], sd[f"{a}.v_proj.weight"]])
    qkv_bias = None
    if f"{a}.q_proj.bias" in sd:
        qkv_bias = torch.cat([sd[f"{a}.q_proj.bias"], sd[f"{a}.k_proj.bias"], sd[f"{a}.v_proj.bias"]])
    return GQAAttention(
        num_heads=cfg.num_attention_heads, num_kv_heads=cfg.num_key_value_heads, head_dim=hd,
        qkv_proj=qkv, o_proj=to_qtensor(sd[f"{a}.o_proj.weight"]), scale=hd ** -0.5, rope=rope,
        q_norm=sd.get(f"{a}.q_norm.weight"), k_norm=sd.get(f"{a}.k_norm.weight"),
        rms_norm_eps=cfg.rms_norm_eps, qkv_bias=qkv_bias)


def _moe(cfg, sd, p):
    experts = [(gate_up_weight(sd, f"{p}.mlp.experts.{e}"),
                to_qtensor(sd[f"{p}.mlp.experts.{e}.down_proj.weight"])) for e in range(cfg.num_experts)]
    shared = None
    if f"{p}.mlp.shared_experts.gate_proj.weight" in sd:
        shared = (gate_up_weight(sd, f"{p}.mlp.shared_experts"),
                  to_qtensor(sd[f"{p}.mlp.shared_experts.down_proj.weight"]))
    return SparseMoE(gate=sd[f"{p}.mlp.gate.weight"], experts=experts, top_k=cfg.num_experts_per_tok,
                     norm_topk_prob=cfg.norm_topk_prob, act=cfg.hidden_act, shared_expert=shared,
                     scoring_func="sigmoid", e_score_correction_bias=sd.get(f"{p}.mlp.gate.e_score_correction_bias"),
                     routed_scaling_factor=cfg.extra.get("routed_scaling_factor", 1.0))


class GlmDecoderLayer(nn.Module):
    def __init__(self, cfg, i, sd, rope):
        super().__init__()
        p = f"model.layers.{i}"
        self.self_attn = _attn(cfg, sd, p, rope)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, sd[f"{p}.input_layernorm.weight"])
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps,
                                                sd[f"{p}.post_attention_layernorm.weight"])
        first_dense = cfg.extra.get("first_k_dense_replace", 0)
        if cfg.is_moe() and i >= first_dense:
            self.mlp = _moe(cfg, sd, p)
        else:
            self.mlp = GatedMLP(gate_up_weight(sd, f"{p}.mlp"),
                                to_qtensor(sd[f"{p}.mlp.down_proj.weight"]), act=cfg.hidden_act)

    def forward(self, x, positions, ctx, residual):
        if residual is None:
            residual, h = x, self.input_layernorm(x)
        else:
            h, residual = self.input_layernorm(x, residual)
        h = self.self_attn(h, positions, ctx, 0)
        h, residual = self.post_attention_layernorm(h, residual)
        return self.mlp(h), residual


class GlmForCausalLM(nn.Module):
    def __init__(self, cfg, sd):
        super().__init__()
        self.config = cfg
        self.embed_tokens = VocabEmbedding(sd["model.embed_tokens.weight"], out_dtype=cfg.act_dtype())
        rope = RotaryEmbedding(cfg.resolved_head_dim(), cfg.max_position_embeddings,
                               base=cfg.rope_theta, rotary_dim=cfg.rotary_dim())
        self.layers = nn.ModuleList([GlmDecoderLayer(cfg, i, sd, rope) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, sd["model.norm.weight"])
        lm_w = sd["model.embed_tokens.weight"] if cfg.tie_word_embeddings else sd["lm_head.weight"]
        self.lm_head = LMHead(to_qtensor(lm_w))

    def forward(self, input_ids, positions, ctx: ForwardContext):
        h = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            h, residual = layer(h, positions, ctx, residual)
        h, _ = self.norm(h, residual)
        return h

    def compute_logits(self, hidden):
        return self.lm_head(hidden)


@register_model("glm", "glm4_moe", "glm4", "Glm4MoeForCausalLM")
def build_glm(cfg: ModelConfig, weights: dict) -> GlmForCausalLM:
    if cfg.partial_rotary_factor == 1.0:
        cfg.partial_rotary_factor = 0.5     # GLM default
    return GlmForCausalLM(cfg, weights)
