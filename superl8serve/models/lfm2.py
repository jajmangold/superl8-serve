# SPDX-License-Identifier: MIT
"""LiquidAI LFM2 — hybrid short-conv + GQA attention. Registered `lfm2`.

Per HF `modeling_lfm2.py`: `full_attn_idxs` selects attention layers; the rest are
double-gated short-conv (ShortConv, LIV, depthwise k=3) token mixers. Every layer
also has a SwiGLU FFN (w1/w3/w2 naming). Attention is GQA with per-head QK-norm
(q_layernorm/k_layernorm) before RoPE, theta 1e6, no bias, output proj named
`out_proj`. Plain RMSNorm eps 1e-5, tied embeddings. Layer wiring:
`h += mixer(operator_norm(h)); h += ffn(ffn_norm(h))`.
"""

from __future__ import annotations

import torch.nn as nn

from ..layers.embedding import LMHead, VocabEmbedding
from ..layers.gqa_attention import GQAAttention
from ..layers.linear_attn import ShortConv
from ..layers.mlp import GatedMLP
from ..layers.norm import RMSNorm
from ..layers.rotary import RotaryEmbedding
from .base import ForwardContext
from .config import ModelConfig
from .registry import register_model
from .weights import merge_qtensor, to_qtensor


def _swiglu(sd, p):
    # LFM2 names gate=w1, up=w3, down=w2
    gate_up = merge_qtensor([sd[f"{p}.feed_forward.w1.weight"], sd[f"{p}.feed_forward.w3.weight"]])
    return GatedMLP(gate_up, to_qtensor(sd[f"{p}.feed_forward.w2.weight"]), act="silu")


def _has_moe_weights(sd: dict, p: str) -> bool:
    return f"{p}.feed_forward.w1_experts.weight" in sd


def _lfm2_moe(sd: dict, p: str, cfg):
    from .moe import SparseMoE

    router = sd[f"{p}.feed_forward.router.weight"]
    w1_exp = sd[f"{p}.feed_forward.w1_experts.weight"]
    w3_exp = sd[f"{p}.feed_forward.w3_experts.weight"]
    w2_exp = sd[f"{p}.feed_forward.w2_experts.weight"]

    num_experts = cfg.num_experts or router.shape[0]
    mi = cfg.moe_intermediate_size or (
        w1_exp.shape[-1] if w1_exp.dim() == 3 else w1_exp.shape[0] // num_experts
    )
    H = cfg.hidden_size

    experts = []
    for e in range(num_experts):
        if w1_exp.dim() == 3:
            e_w1, e_w3, e_w2 = w1_exp[e], w3_exp[e], w2_exp[e]
        else:
            e_w1 = w1_exp[e * mi : (e + 1) * mi]
            e_w3 = w3_exp[e * mi : (e + 1) * mi]
            e_w2 = w2_exp[e * H : (e + 1) * H]
        gate_up = merge_qtensor([e_w1, e_w3])
        experts.append((gate_up, to_qtensor(e_w2)))

    return SparseMoE(
        gate=router,
        experts=experts,
        top_k=cfg.num_experts_per_tok,
        norm_topk_prob=False,
        scoring_func="sigmoid",
    )


class Lfm2Layer(nn.Module):
    def __init__(self, cfg: ModelConfig, i: int, sd: dict, rope: RotaryEmbedding, is_attn: bool):
        super().__init__()
        p = f"model.layers.{i}"
        hd = cfg.resolved_head_dim()
        if is_attn:
            a = f"{p}.self_attn"
            self.mixer = GQAAttention(
                num_heads=cfg.num_attention_heads,
                num_kv_heads=cfg.num_key_value_heads,
                head_dim=hd,
                qkv_proj=merge_qtensor(
                    [sd[f"{a}.q_proj.weight"], sd[f"{a}.k_proj.weight"], sd[f"{a}.v_proj.weight"]]
                ),
                o_proj=to_qtensor(sd[f"{a}.out_proj.weight"]),
                scale=hd**-0.5,
                rope=rope,
                q_norm=sd.get(f"{a}.q_layernorm.weight"),
                k_norm=sd.get(f"{a}.k_layernorm.weight"),
                rms_norm_eps=cfg.rms_norm_eps,
            )
        else:
            c = f"{p}.conv"
            self.mixer = ShortConv(
                cfg.hidden_size,
                in_proj=to_qtensor(sd[f"{c}.in_proj.weight"]),
                out_proj=to_qtensor(sd[f"{c}.out_proj.weight"]),
                conv_weight=sd[f"{c}.conv.weight"],
                kernel=cfg.extra.get("conv_L_cache", 3),
            )
        self.operator_norm = RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, sd[f"{p}.operator_norm.weight"]
        )
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, sd[f"{p}.ffn_norm.weight"])
        self.feed_forward = _lfm2_moe(sd, p, cfg) if _has_moe_weights(sd, p) else _swiglu(sd, p)

    def forward(self, x, positions, ctx, layer_idx):
        x = x + self.mixer(self.operator_norm(x), positions, ctx, layer_idx)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class Lfm2ForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig, sd: dict):
        super().__init__()
        self.config = cfg
        # Which layers are full attention vs short-conv. Older LFM2 configs list the
        # indices directly (`full_attn_idxs`); current HF configs (LFM2.5, LFM2-2.6B)
        # instead carry `layer_types` = ["conv"|"full_attention", ...] per layer.
        # Support both so a real HF checkpoint loads without a hand-authored config.
        attn_idxs = set(cfg.extra.get("full_attn_idxs") or [])
        if not attn_idxs and cfg.extra.get("layer_types"):
            attn_idxs = {i for i, t in enumerate(cfg.extra["layer_types"]) if t == "full_attention"}
        self.embed_tokens = VocabEmbedding(sd["model.embed_tokens.weight"], out_dtype=cfg.act_dtype())
        rope = RotaryEmbedding(
            cfg.resolved_head_dim(), cfg.max_position_embeddings, base=cfg.rope_theta
        )
        self.layers = nn.ModuleList(
            [Lfm2Layer(cfg, i, sd, rope, i in attn_idxs) for i in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            sd["model.embedding_norm.weight"]
            if "model.embedding_norm.weight" in sd
            else sd["model.norm.weight"],
        )
        # LFM2 ties the LM head to the embedding and names the flag `tie_embedding`
        # (not the HF-standard `tie_word_embeddings`), so `cfg.tie_word_embeddings`
        # can read False even though no separate `lm_head.weight` exists. Fall back to
        # the embedding whenever an untied head isn't actually present in the weights.
        lm_w = sd["lm_head.weight"] if "lm_head.weight" in sd else sd["model.embed_tokens.weight"]
        self.lm_head = LMHead(to_qtensor(lm_w))

    def forward(self, input_ids, positions, ctx: ForwardContext):
        h = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            h = layer(h, positions, ctx, i)
        return self.norm(h)

    def compute_logits(self, hidden):
        return self.lm_head(hidden)


@register_model("lfm2", "lfm2_moe", "Lfm2ForCausalLM")
def build_lfm2(cfg: ModelConfig, weights: dict) -> Lfm2ForCausalLM:
    return Lfm2ForCausalLM(cfg, weights)
