# SPDX-License-Identifier: MIT
"""Tencent Hunyuan (hunyuan_v1_moe, e.g. Hunyuan-A13B) — GQA + QK-norm + softmax MoE
with a shared expert. Registered `hunyuan`.

Per HF: QK-norm named query_layernorm/key_layernorm (over head_dim, pre-RoPE), no
attention bias, rope_theta 1e4; MoE router `mlp.gate.wg` fp32 softmax -> top-k ->
renorm, shared expert `mlp.shared_mlp`, all layers MoE (A13B). Plain RMSNorm eps
1e-5, tied embeddings.

CLA (cross-layer KV sharing, Hunyuan-Large) is enabled via
`ModelConfig.extra["cla_group_size"]`. Layers `i % cla_group_size == 0` are KV
heads that write K,V to the shared cache; subsequent layers in a group compute Q
only and read K,V from the group head's cache slot.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import superl8

from ..layers.embedding import LMHead, VocabEmbedding
from ..layers.gqa_attention import GQAAttention
from ..layers.linear import LinearW8A8
from ..layers.norm import RMSNorm
from ..layers.rotary import RotaryEmbedding
from .base import ForwardContext
from .config import ModelConfig
from .moe import SparseMoE
from .registry import register_model
from .weights import gate_up_weight, merge_qtensor, to_qtensor


class HunyuanDecoderLayer(nn.Module):
    def __init__(self, cfg, i, sd, rope):
        super().__init__()
        self.layer_idx = i
        p = f"model.layers.{i}"
        a = f"{p}.self_attn"
        hd = cfg.resolved_head_dim()
        self.self_attn = GQAAttention(
            num_heads=cfg.num_attention_heads, num_kv_heads=cfg.num_key_value_heads, head_dim=hd,
            qkv_proj=merge_qtensor([sd[f"{a}.q_proj.weight"], sd[f"{a}.k_proj.weight"], sd[f"{a}.v_proj.weight"]]),
            o_proj=to_qtensor(sd[f"{a}.o_proj.weight"]), scale=hd ** -0.5, rope=rope,
            q_norm=sd.get(f"{a}.query_layernorm.weight"), k_norm=sd.get(f"{a}.key_layernorm.weight"),
            rms_norm_eps=cfg.rms_norm_eps)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, sd[f"{p}.input_layernorm.weight"])
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps,
                                                sd[f"{p}.post_attention_layernorm.weight"])
        experts = [(gate_up_weight(sd, f"{p}.mlp.experts.{e}"),
                    to_qtensor(sd[f"{p}.mlp.experts.{e}.down_proj.weight"])) for e in range(cfg.num_experts)]
        shared = None
        if f"{p}.mlp.shared_mlp.gate_proj.weight" in sd:
            shared = (gate_up_weight(sd, f"{p}.mlp.shared_mlp"),
                      to_qtensor(sd[f"{p}.mlp.shared_mlp.down_proj.weight"]))
        gate_w = sd.get(f"{p}.mlp.gate.wg.weight", sd.get(f"{p}.mlp.gate.weight"))
        self.mlp = SparseMoE(gate=gate_w, experts=experts, top_k=cfg.num_experts_per_tok,
                             norm_topk_prob=cfg.norm_topk_prob, act=cfg.hidden_act,
                             shared_expert=shared, scoring_func="softmax")

    def forward(self, x, positions, ctx, residual):
        if residual is None:
            residual, h = x, self.input_layernorm(x)
        else:
            h, residual = self.input_layernorm(x, residual)
        h = self.self_attn(h, positions, ctx, self.layer_idx)
        h, residual = self.post_attention_layernorm(h, residual)
        return self.mlp(h), residual


class HunyuanCLADecoderLayer(nn.Module):
    """Decoder layer for the non-KV-head role in a CLA group.

    Computes only the Q projection and reads K,V from the group head's
    cache slot (``kv_layer_idx``).  The group head (``i % cla_group_size == 0``)
    uses the regular `HunyuanDecoderLayer` instead.
    """

    def __init__(self, cfg, i, sd, rope, kv_layer_idx):
        super().__init__()
        self.kv_layer_idx = kv_layer_idx
        p = f"model.layers.{i}"
        a = f"{p}.self_attn"
        hd = cfg.resolved_head_dim()
        nh, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.nh, self.hd = nh, hd
        self.scale = hd ** -0.5
        self.rope = rope

        self.q_proj = LinearW8A8(to_qtensor(sd[f"{a}.q_proj.weight"]))
        self.o_proj = LinearW8A8(to_qtensor(sd[f"{a}.o_proj.weight"]))

        qn = sd.get(f"{a}.query_layernorm.weight")
        self.q_norm = RMSNorm(hd, cfg.rms_norm_eps, qn) if qn is not None else None

        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, sd[f"{p}.input_layernorm.weight"])
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps,
                                                sd[f"{p}.post_attention_layernorm.weight"])
        experts = [(gate_up_weight(sd, f"{p}.mlp.experts.{e}"),
                    to_qtensor(sd[f"{p}.mlp.experts.{e}.down_proj.weight"])) for e in range(cfg.num_experts)]
        shared = None
        if f"{p}.mlp.shared_mlp.gate_proj.weight" in sd:
            shared = (gate_up_weight(sd, f"{p}.mlp.shared_mlp"),
                      to_qtensor(sd[f"{p}.mlp.shared_mlp.down_proj.weight"]))
        gate_w = sd.get(f"{p}.mlp.gate.wg.weight", sd.get(f"{p}.mlp.gate.weight"))
        self.mlp = SparseMoE(gate=gate_w, experts=experts, top_k=cfg.num_experts_per_tok,
                             norm_topk_prob=cfg.norm_topk_prob, act=cfg.hidden_act,
                             shared_expert=shared, scoring_func="softmax")

    def forward(self, x, positions, ctx, residual):
        if residual is None:
            residual, h = x, self.input_layernorm(x)
        else:
            h, residual = self.input_layernorm(x, residual)

        B, S, _ = h.shape
        q = self.q_proj(h)
        q = q.view(B, S, self.nh, self.hd)
        if self.q_norm is not None:
            q = self.q_norm(q)
        q, _ = self.rope(positions, q, torch.empty_like(q[:, :, :1]))
        q = q.transpose(1, 2).contiguous()

        cache = ctx.kv_cache
        if ctx.is_prefill:
            l = self.kv_layer_idx
            k = cache.k[l, :, :, :S].contiguous()
            v = cache.v[l, :, :, :S].contiguous()
            out = superl8.attn_int8_fwd(q, k, v, causal=True, scale=self.scale)
        else:
            l = self.kv_layer_idx
            p = cache.length
            k_all = cache.k[l, :, :, :p + 1].contiguous()
            v_all = cache.v[l, :, :, :p + 1].contiguous()
            out = superl8.attn_int8_decode(q, k_all, v_all, scale=self.scale)

        out = out.transpose(1, 2).reshape(B, S, self.nh * self.hd)
        h = self.o_proj(out)
        h, residual = self.post_attention_layernorm(h, residual)
        return self.mlp(h), residual


class HunyuanForCausalLM(nn.Module):
    def __init__(self, cfg, sd):
        super().__init__()
        self.config = cfg
        self.cla_group_size = cfg.extra.get("cla_group_size", 1)
        self.embed_tokens = VocabEmbedding(sd["model.embed_tokens.weight"], out_dtype=cfg.act_dtype())
        rope = RotaryEmbedding(cfg.resolved_head_dim(), cfg.max_position_embeddings, base=cfg.rope_theta)
        layers = []
        for i in range(cfg.num_hidden_layers):
            if self.cla_group_size > 1 and i % self.cla_group_size != 0:
                kv_head = i - (i % self.cla_group_size)
                layers.append(HunyuanCLADecoderLayer(cfg, i, sd, rope, kv_head))
            else:
                layers.append(HunyuanDecoderLayer(cfg, i, sd, rope))
        self.layers = nn.ModuleList(layers)
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


@register_model("hunyuan", "hunyuan_v1_moe", "HunYuanMoEV1ForCausalLM")
def build_hunyuan(cfg: ModelConfig, weights: dict) -> HunyuanForCausalLM:
    if cfg.rope_theta == 1e6:
        cfg.rope_theta = 1e4     # Hunyuan default
    return HunyuanForCausalLM(cfg, weights)
