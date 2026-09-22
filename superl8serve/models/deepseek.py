# SPDX-License-Identifier: MIT
"""DeepSeek-V2/V3/V4 — MLA attention + fine-grained MoE (+ shared expert), over the
Track-1 fp16 MLA backend. Registered `deepseek`.

Per HF DeepseekV3: MLA (q_a/q_b LoRA, kv_a_with_mqa/kv_b, decoupled RoPE) + a MoE
FFN on most layers (first `first_k_dense_replace` layers are dense), with a shared
expert alongside the routed ones. MLA dims come from the config extras (q_lora_rank,
kv_lora_rank, qk_nope/rope_head_dim, v_head_dim). Routing detail (sigmoid gate +
group-limited top-k + routed_scaling_factor) is approximated by the shared SparseMoE
softmax router for now — noted for refinement; the attention + shapes are faithful.
"""

from __future__ import annotations

import torch.nn as nn

from ..layers.embedding import LMHead, VocabEmbedding
from ..layers.mla_attn import MLAAttention
from ..layers.mlp import GatedMLP
from ..layers.norm import RMSNorm
from .base import ForwardContext
from .config import ModelConfig
from .moe import SparseMoE
from .registry import register_model
from .weights import gate_up_weight, to_qtensor


def _mla(cfg: ModelConfig, sd: dict, p: str) -> MLAAttention:
    x = cfg.extra
    return MLAAttention(
        cfg,
        q_a_proj=to_qtensor(sd[f"{p}.q_a_proj.weight"]),
        q_a_norm=sd[f"{p}.q_a_layernorm.weight"],
        q_b_proj=to_qtensor(sd[f"{p}.q_b_proj.weight"]),
        kv_a_proj=to_qtensor(sd[f"{p}.kv_a_proj_with_mqa.weight"]),
        kv_a_norm=sd[f"{p}.kv_a_layernorm.weight"],
        kv_b_proj=to_qtensor(sd[f"{p}.kv_b_proj.weight"]),
        o_proj=to_qtensor(sd[f"{p}.o_proj.weight"]),
        num_heads=cfg.num_attention_heads,
        q_lora_rank=x["q_lora_rank"],
        kv_lora_rank=x["kv_lora_rank"],
        qk_nope_head_dim=x["qk_nope_head_dim"],
        qk_rope_head_dim=x["qk_rope_head_dim"],
        v_head_dim=x["v_head_dim"],
        rope_theta=cfg.rope_theta,
        max_pos=cfg.max_position_embeddings,
        use_int8_absorb=cfg.extra.get("use_int8_absorb", True),
    )


class DeepseekDecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, i: int, sd: dict):
        super().__init__()
        self.layer_idx = i
        p = f"model.layers.{i}"
        self.self_attn = _mla(cfg, sd, f"{p}.self_attn")
        self.input_layernorm = RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, sd[f"{p}.input_layernorm.weight"]
        )
        self.post_attention_layernorm = RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, sd[f"{p}.post_attention_layernorm.weight"]
        )
        first_dense = cfg.extra.get("first_k_dense_replace", 0)
        if cfg.is_moe() and i >= first_dense:
            experts = [
                (
                    gate_up_weight(sd, f"{p}.mlp.experts.{e}"),
                    to_qtensor(sd[f"{p}.mlp.experts.{e}.down_proj.weight"]),
                )
                for e in range(cfg.num_experts)
            ]
            shared = None
            if f"{p}.mlp.shared_experts.gate_proj.weight" in sd:
                shared = (
                    gate_up_weight(sd, f"{p}.mlp.shared_experts"),
                    to_qtensor(sd[f"{p}.mlp.shared_experts.down_proj.weight"]),
                )
            self.mlp = SparseMoE(
                gate=sd[f"{p}.mlp.gate.weight"],
                experts=experts,
                top_k=cfg.num_experts_per_tok,
                norm_topk_prob=cfg.norm_topk_prob,
                act=cfg.hidden_act,
                shared_expert=shared,
                scoring_func="sigmoid",
                e_score_correction_bias=sd.get(f"{p}.mlp.e_score_correction_bias"),
                routed_scaling_factor=cfg.extra.get("routed_scaling_factor", 1.0),
                num_expert_groups=cfg.extra.get("num_expert_groups"),
                topk_group=cfg.extra.get("topk_group"),
            )
        else:
            self.mlp = GatedMLP(
                gate_up_weight(sd, f"{p}.mlp"),
                to_qtensor(sd[f"{p}.mlp.down_proj.weight"]),
                act=cfg.hidden_act,
            )

    def forward(self, x, positions, ctx, residual):
        if residual is None:
            residual = x
            h = self.input_layernorm(x)
        else:
            h, residual = self.input_layernorm(x, residual)
        h = self.self_attn(h, positions, ctx, self.layer_idx)
        h, residual = self.post_attention_layernorm(h, residual)
        h = self.mlp(h)
        return h, residual


class DeepseekForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig, sd: dict):
        super().__init__()
        self.config = cfg
        self.embed_tokens = VocabEmbedding(sd["model.embed_tokens.weight"], out_dtype=cfg.act_dtype())
        self.layers = nn.ModuleList(
            [DeepseekDecoderLayer(cfg, i, sd) for i in range(cfg.num_hidden_layers)]
        )
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


@register_model(
    "deepseek", "deepseek_v3", "deepseek_v2", "DeepseekV3ForCausalLM", "DeepseekV2ForCausalLM"
)
def build_deepseek(cfg: ModelConfig, weights: dict) -> DeepseekForCausalLM:
    return DeepseekForCausalLM(cfg, weights)
