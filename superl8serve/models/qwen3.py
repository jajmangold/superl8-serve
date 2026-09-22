# SPDX-License-Identifier: MIT
"""Qwen3 dense + Qwen3-MoE — the first concrete registered architectures.

Verified against HF modeling_qwen3.py / modeling_qwen3_moe.py:
  * GQA attention with per-head RMSNorm on Q,K (over head_dim) applied BEFORE RoPE;
    no QKV bias; explicit `head_dim` (128) decoupled from hidden//heads; RoPE theta
    1e6; scale = head_dim**-0.5; no sliding window.
  * SwiGLU MLP (silu); MoE layers replace it with softmax->top-k->renorm routing
    over `num_experts` experts (no shared expert in plain Qwen3-MoE).
  * plain RMSNorm (w, not 1+w), eps 1e-6; input_layernorm / post_attention_layernorm
    around each sub-block (standard pre-norm + residual); tied embeddings on the
    small dense sizes.

Registered under `qwen3` and `qwen3_moe` (MoE branch auto-selected per layer).
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
from .mtp import build_mtp
from .registry import register_model
from .weights import gate_up_weight, qkv_weight, to_qtensor


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int, sd: dict, rope: RotaryEmbedding):
        super().__init__()
        self.layer_idx = layer_idx
        p = f"model.layers.{layer_idx}"
        hd = cfg.resolved_head_dim()
        scale = (cfg.query_pre_attn_scalar ** -0.5) if cfg.query_pre_attn_scalar else hd ** -0.5
        self.self_attn = GQAAttention(
            num_heads=cfg.num_attention_heads, num_kv_heads=cfg.num_key_value_heads,
            head_dim=hd, qkv_proj=qkv_weight(sd, f"{p}.self_attn"),
            o_proj=to_qtensor(sd[f"{p}.self_attn.o_proj.weight"]), scale=scale, rope=rope,
            q_norm=sd.get(f"{p}.self_attn.q_norm.weight") if cfg.qk_norm else None,
            k_norm=sd.get(f"{p}.self_attn.k_norm.weight") if cfg.qk_norm else None,
            rms_norm_eps=cfg.rms_norm_eps,
        )
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps,
                                       sd[f"{p}.input_layernorm.weight"])
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps,
                                                sd[f"{p}.post_attention_layernorm.weight"])
        if cfg.layer_is_moe(layer_idx):
            experts = [(gate_up_weight(sd, f"{p}.mlp.experts.{e}"),
                        to_qtensor(sd[f"{p}.mlp.experts.{e}.down_proj.weight"]))
                       for e in range(cfg.num_experts)]
            shared = None
            if cfg.shared_expert_intermediate_size:
                shared = (gate_up_weight(sd, f"{p}.mlp.shared_expert"),
                          to_qtensor(sd[f"{p}.mlp.shared_expert.down_proj.weight"]))
            self.mlp = SparseMoE(
                gate=sd[f"{p}.mlp.gate.weight"], experts=experts, top_k=cfg.num_experts_per_tok,
                norm_topk_prob=cfg.norm_topk_prob, act=cfg.hidden_act, shared_expert=shared,
                shared_expert_gate=sd.get(f"{p}.mlp.shared_expert_gate.weight"),
            )
        else:
            self.mlp = GatedMLP(gate_up_weight(sd, f"{p}.mlp"),
                                to_qtensor(sd[f"{p}.mlp.down_proj.weight"]), act=cfg.hidden_act)

    def forward(self, x, positions, ctx, residual):
        # First layer has no running residual yet; seed it from the input.
        if residual is None:
            residual = x
            h = self.input_layernorm(x)
        else:
            h, residual = self.input_layernorm(x, residual)
        h = self.self_attn(h, positions, ctx, self.layer_idx)
        h, residual = self.post_attention_layernorm(h, residual)
        h = self.mlp(h)
        return h, residual


class Qwen3Model(nn.Module):
    def __init__(self, cfg: ModelConfig, sd: dict):
        super().__init__()
        self.config = cfg
        self.embed_tokens = VocabEmbedding(sd["model.embed_tokens.weight"],
                                           embed_scale=cfg.embed_scale or 1.0, out_dtype=cfg.act_dtype())
        rope = RotaryEmbedding(cfg.resolved_head_dim(), cfg.max_position_embeddings,
                               base=cfg.rope_theta, rotary_dim=cfg.rotary_dim())
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(cfg, i, sd, rope) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, sd["model.norm.weight"],
                            add_unit_offset=cfg.norm_add_unit_offset)

    def forward(self, input_ids, positions, ctx):
        h = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            h, residual = layer(h, positions, ctx, residual)
        h, _ = self.norm(h, residual)
        return h


def _qwen3_mtp_decoder(cfg, depth_idx, prefix, sd, rope):
    """Build one decoder block for an MTP depth (same structure as Qwen3DecoderLayer
    but without the input/post-attention norms — those come from MTPLayer)."""
    hd = cfg.resolved_head_dim()
    scale = (cfg.query_pre_attn_scalar ** -0.5) if cfg.query_pre_attn_scalar else hd ** -0.5
    attn = GQAAttention(
        num_heads=cfg.num_attention_heads, num_kv_heads=cfg.num_key_value_heads,
        head_dim=hd, qkv_proj=qkv_weight(sd, f"{prefix}.self_attn"),
        o_proj=to_qtensor(sd[f"{prefix}.self_attn.o_proj.weight"]), scale=scale, rope=rope,
        q_norm=sd.get(f"{prefix}.self_attn.q_norm.weight") if cfg.qk_norm else None,
        k_norm=sd.get(f"{prefix}.self_attn.k_norm.weight") if cfg.qk_norm else None,
        rms_norm_eps=cfg.rms_norm_eps,
    )
    class _MTPBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps,
                                           sd[f"{prefix}.input_layernorm.weight"])
            self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps,
                                                    sd[f"{prefix}.post_attention_layernorm.weight"])
            self.self_attn = attn
            self.mlp = GatedMLP(gate_up_weight(sd, f"{prefix}.mlp"),
                                to_qtensor(sd[f"{prefix}.mlp.down_proj.weight"]), act=cfg.hidden_act)
        def forward(self, x, positions, ctx, residual):
            B, S, Hd = x.shape
            if residual is None:
                residual = x
                h = self.input_layernorm(x)
            else:
                h, residual = self.input_layernorm(x, residual)
            # Use cache-free draft attention so MTP never corrupts the main KV
            qkv = self.self_attn.qkv_proj(h)
            q, k, v = qkv.split([self.self_attn.nh * self.self_attn.hd,
                                 self.self_attn.nkv * self.self_attn.hd,
                                 self.self_attn.nkv * self.self_attn.hd], dim=-1)
            q = q.view(B, S, self.self_attn.nh, self.self_attn.hd)
            k = k.view(B, S, self.self_attn.nkv, self.self_attn.hd)
            v = v.view(B, S, self.self_attn.nkv, self.self_attn.hd)
            if self.self_attn.q_norm is not None:
                q = self.self_attn.q_norm(q)
                k = self.self_attn.k_norm(k)
            q, k = self.self_attn.rope(positions, q, k)
            h = GQAAttention._draft_attn(q.transpose(1, 2), k.transpose(1, 2),
                                          v.transpose(1, 2), scale=self.self_attn.scale)
            h = self.self_attn.o_proj(h.to(x.dtype))
            h, residual = self.post_attention_layernorm(h, residual)
            h = self.mlp(h)
            return h, residual
    return _MTPBlock()


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig, sd: dict):
        super().__init__()
        self.config = cfg
        self.model = Qwen3Model(cfg, sd)
        lm_w = sd["model.embed_tokens.weight"] if cfg.tie_word_embeddings else sd["lm_head.weight"]
        # Quantize the head to int8 dp4a even when tied: the fp16 logits GEMM runs
        # on the fleet's crippled tensor cores (~6.9 TFLOP/s) and was the single
        # biggest decode kernel. The embedding lookup keeps its own fp16 table.
        head_w = to_qtensor(lm_w)
        self.lm_head = LMHead(head_w, logit_softcap=cfg.final_logit_softcap)
        rope = RotaryEmbedding(cfg.resolved_head_dim(), cfg.max_position_embeddings,
                               base=cfg.rope_theta, rotary_dim=cfg.rotary_dim())
        self.mtp = build_mtp(cfg, sd, self.model.embed_tokens, self.model.norm,
                             self.lm_head, rope, _qwen3_mtp_decoder)

    def forward(self, input_ids, positions, ctx: ForwardContext):
        return self.model(input_ids, positions, ctx)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)


@register_model("qwen3", "qwen3_moe", "qwen3moe", "Qwen3ForCausalLM", "Qwen3MoeForCausalLM")
def build_qwen3(cfg: ModelConfig, weights: dict) -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM(cfg, weights)
