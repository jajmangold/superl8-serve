# SPDX-License-Identifier: MIT
"""MiniMax-Text-01 — hybrid lightning (linear) + softmax attention, softmax MoE.
Registered `minimax`.

Per HF `minimax_text_01`: `attn_type_list` selects lightning (0) vs softmax (1) per
layer (7:1). Lightning = data-independent fixed-decay linear attention (ALiBi
slopes) with a sigmoid output gate + RMSNorm. Softmax layers = GQA, partial RoPE
(rotary_dim 64), theta 1e7, no QK-norm. MoE = softmax top-2, no shared expert
(w1/w3/w2 expert naming). Postnorm residual scaling: residual = beta*h + alpha*sub(norm(h)).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..layers.embedding import LMHead, VocabEmbedding
from ..layers.gqa_attention import GQAAttention
from ..layers.linear_attn import (
    LinearW8A8,
    RMSNorm,
    _lightning_attn_dispatch,
    lightning_slopes,
)
from ..layers.rotary import RotaryEmbedding
from .base import ForwardContext
from .config import ModelConfig
from .moe import SparseMoE
from .registry import register_model
from .weights import merge_qtensor, to_qtensor


class LightningAttention(nn.Module):
    is_recurrent = True  # carries per-slot decode state via ctx.lin_cache

    def __init__(self, cfg, *, qkv_proj, out_proj, output_gate, norm_gain, num_heads, head_dim, slopes):
        super().__init__()
        self.nh, self.hd = num_heads, head_dim
        self.qkv_proj = LinearW8A8(qkv_proj)
        self.out_proj = LinearW8A8(out_proj)
        self.output_gate = LinearW8A8(output_gate)
        self.norm = RMSNorm(num_heads * head_dim, cfg.rms_norm_eps, norm_gain)
        self.register_buffer("slopes", slopes, persistent=False)

    def forward(self, x, positions, ctx, layer_idx):
        B, L, _ = x.shape
        cache = ctx.lin_cache if ctx is not None else None
        state = cache.get_state(layer_idx) if cache is not None else None
        q, k, v = self.qkv_proj(x).view(B, L, 3, self.nh, self.hd).unbind(2)
        o, state = _lightning_attn_dispatch(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            self.slopes, state=state)
        if cache is not None:
            cache.set_state(layer_idx, state)
        o = o.transpose(1, 2).reshape(B, L, self.nh * self.hd)
        o = self.norm(o * torch.sigmoid(self.output_gate(x)))
        return self.out_proj(o)


def _softmax_attn(cfg, sd, p, rope):
    hd = cfg.resolved_head_dim()
    a = f"{p}.self_attn"
    return GQAAttention(
        num_heads=cfg.num_attention_heads, num_kv_heads=cfg.num_key_value_heads, head_dim=hd,
        qkv_proj=merge_qtensor([sd[f"{a}.q_proj.weight"], sd[f"{a}.k_proj.weight"], sd[f"{a}.v_proj.weight"]]),
        o_proj=to_qtensor(sd[f"{a}.o_proj.weight"]), scale=hd ** -0.5, rope=rope, rms_norm_eps=cfg.rms_norm_eps)


class MiniMaxDecoderLayer(nn.Module):
    def __init__(self, cfg, i, sd, rope):
        super().__init__()
        self.layer_idx = i
        p = f"model.layers.{i}"
        self.is_lightning = cfg.extra.get("attn_type_list", [])[i] == 0 if cfg.extra.get("attn_type_list") \
            else (i + 1) % cfg.extra.get("softmax_every", 8) != 0
        if self.is_lightning:
            a = f"{p}.self_attn"
            hd = cfg.resolved_head_dim()
            self.attn = LightningAttention(
                cfg, qkv_proj=to_qtensor(sd[f"{a}.qkv_proj.weight"]),
                out_proj=to_qtensor(sd[f"{a}.out_proj.weight"]),
                output_gate=to_qtensor(sd[f"{a}.output_gate.weight"]),
                norm_gain=sd[f"{a}.norm.weight"], num_heads=cfg.num_attention_heads, head_dim=hd,
                slopes=lightning_slopes(cfg.num_attention_heads))
        else:
            self.attn = _softmax_attn(cfg, sd, p, rope)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, sd[f"{p}.input_layernorm.weight"])
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps,
                                                sd[f"{p}.post_attention_layernorm.weight"])
        experts = [(merge_qtensor([sd[f"{p}.block_sparse_moe.experts.{e}.w1.weight"],
                                   sd[f"{p}.block_sparse_moe.experts.{e}.w3.weight"]]),
                    to_qtensor(sd[f"{p}.block_sparse_moe.experts.{e}.w2.weight"]))
                   for e in range(cfg.num_experts)]
        self.mlp = SparseMoE(gate=sd[f"{p}.block_sparse_moe.gate.weight"], experts=experts,
                             top_k=cfg.num_experts_per_tok, norm_topk_prob=cfg.norm_topk_prob, act="silu")
        self.alpha = cfg.extra.get("layernorm_full_attention_alpha", 1.0)
        self.beta = cfg.extra.get("layernorm_mlp_beta", 1.0)

    def forward(self, x, positions, ctx, residual):
        attn_out = self.attn(self.input_layernorm(x), positions, ctx, self.layer_idx)
        h = self.beta * x + self.alpha * attn_out
        h = self.beta * h + self.alpha * self.mlp(self.post_attention_layernorm(h))
        return h, None


class MiniMaxForCausalLM(nn.Module):
    def __init__(self, cfg, sd):
        super().__init__()
        self.config = cfg
        self.embed_tokens = VocabEmbedding(sd["model.embed_tokens.weight"], out_dtype=cfg.act_dtype())
        rope = RotaryEmbedding(cfg.resolved_head_dim(), cfg.max_position_embeddings,
                               base=cfg.rope_theta, rotary_dim=cfg.rotary_dim())
        self.layers = nn.ModuleList([MiniMaxDecoderLayer(cfg, i, sd, rope) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, sd["model.norm.weight"])
        lm_w = sd["model.embed_tokens.weight"] if cfg.tie_word_embeddings else sd["lm_head.weight"]
        self.lm_head = LMHead(to_qtensor(lm_w))

    def forward(self, input_ids, positions, ctx: ForwardContext):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h, _ = layer(h, positions, ctx, None)
        return self.norm(h)

    def compute_logits(self, hidden):
        return self.lm_head(hidden)


@register_model("minimax", "minimax_text_01", "MiniMaxText01ForCausalLM", "MiniMaxForCausalLM")
def build_minimax(cfg: ModelConfig, weights: dict) -> MiniMaxForCausalLM:
    return MiniMaxForCausalLM(cfg, weights)
