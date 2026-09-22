# SPDX-License-Identifier: MIT
"""Qwen3-Next (= Qwen3.5/3.6 backbone) — HYBRID linear + full attention, over the
Track-1 backends. Registered `qwen3_next`.

`attention_kind(i)` picks the backend per layer: full-attention layers (every
`full_attention_interval`-th) use GQAAttention (gated attn, partial-rotary);
the rest use GatedDeltaNetAttention (linear). FFN is ultra-sparse MoE WITH a shared
expert. This is the demonstration that the modular per-layer AttentionBackend seam
handles a hybrid backbone with ZERO engine change.

Prefill-tested with random weights (proves the hybrid assembly). Real-checkpoint
loading needs the exact HF fused linear-attn weight names (in_proj_qkvz / in_proj_ba
splits) mapped in the converter — noted; and full autoregressive decode needs
recurrent-state caching for the linear layers (Track 1.5 / the Track-2 kernel).
"""

from __future__ import annotations

import torch.nn as nn

from ..layers.embedding import LMHead, VocabEmbedding
from ..layers.gqa_attention import GQAAttention
from ..layers.linear_attn import GatedDeltaNetAttention
from ..layers.mlp import GatedMLP
from ..layers.norm import RMSNorm
from ..layers.rotary import RotaryEmbedding
from .base import ForwardContext
from .config import ModelConfig
from .moe import SparseMoE, WeightStationaryMoE
from .mtp import build_mtp
from .registry import register_model
from .weights import gate_up_weight, merge_qtensor, qkv_weight, to_qtensor


def _full_attn(cfg, sd, p, rope):
    hd = cfg.resolved_head_dim()
    return GQAAttention(
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=hd,
        qkv_proj=qkv_weight(sd, f"{p}.self_attn"),
        o_proj=to_qtensor(sd[f"{p}.self_attn.o_proj.weight"]),
        scale=hd**-0.5,
        rope=rope,
        q_norm=sd.get(f"{p}.self_attn.q_norm.weight"),
        k_norm=sd.get(f"{p}.self_attn.k_norm.weight"),
        rms_norm_eps=cfg.rms_norm_eps,
    )


def _linear_attn(cfg, sd, p):
    x = cfg.extra
    la = f"{p}.linear_attn"
    return GatedDeltaNetAttention(
        cfg,
        qkv_proj=to_qtensor(sd[f"{la}.qkv_proj.weight"]),
        out_proj=to_qtensor(sd[f"{la}.out_proj.weight"]),
        conv_weight=sd[f"{la}.conv_weight"],
        a_log=sd[f"{la}.A_log"],
        dt_bias=sd[f"{la}.dt_bias"],
        beta_proj=None,
        gate_proj=None,
        gate_beta_proj=merge_qtensor([
            to_qtensor(sd[f"{la}.dt_proj.weight"]),
            to_qtensor(sd[f"{la}.beta_proj.weight"]),
        ]),
        z_proj=to_qtensor(sd[f"{la}.z_proj.weight"]),
        norm_gain=sd[f"{la}.norm.weight"],
        num_k_heads=x["linear_num_key_heads"],
        num_v_heads=x["linear_num_value_heads"],
        key_dim=x["linear_key_head_dim"],
        value_dim=x["linear_value_head_dim"],
        conv_kernel=x.get("linear_conv_kernel_dim", 4),
    )


def _build_moe(cfg: ModelConfig, sd: dict, p: str):
    """Build the MoE for one decoder layer, honoring the weight-stationary and
    multi-GPU expert-sharding seams.

    Selection order (each step degrades cleanly to the next):
      1. ``cfg.expert_to_gpu`` set  -> TransportMoELayer over a weight-stationary
         base (or sparse if ``use_weight_stationary_moe`` is off), routing only
         remote experts over the compressed transport seam.
      2. ``cfg.use_weight_stationary_moe`` -> WeightStationaryMoE (per-expert
         staging buffers + skip-empty, graph-capturable expert GEMMs).
      3. default -> SparseMoE (today's resident behavior).

    An all-local transport map (expert_to_gpu[e] == local_gpu for every e) is
    bit-identical to the base MoE — the transport parity contract (tested in
    tests/test_moe_transport.py) is the regression gate.
    """
    from .moe_transport import MoETransport, TransportMoELayer

    experts = [
        (
            gate_up_weight(sd, f"{p}.mlp.experts.{e}"),
            to_qtensor(sd[f"{p}.mlp.experts.{e}.down_proj.weight"]),
        )
        for e in range(cfg.num_experts)
    ]
    shared = None
    if f"{p}.mlp.shared_expert.gate_proj.weight" in sd:
        shared = (
            gate_up_weight(sd, f"{p}.mlp.shared_expert"),
            to_qtensor(sd[f"{p}.mlp.shared_expert.down_proj.weight"]),
        )

    base_kwargs = dict(
        gate=sd[f"{p}.mlp.gate.weight"],
        experts=experts,
        top_k=cfg.num_experts_per_tok,
        norm_topk_prob=cfg.norm_topk_prob,
        act=cfg.hidden_act,
        shared_expert=shared,
        shared_expert_gate=sd.get(f"{p}.mlp.shared_expert_gate.weight"),
    )
    expert_devices = None
    if cfg.expert_to_gpu:
        # Per-expert owning device. Weights were already placed there at load
        # time (gguf_state_dict expert_device), so this is informational for
        # buffer allocation only.
        local_gpu = cfg.local_gpu if cfg.local_gpu is not None else 0
        expert_devices = [f"cuda:{cfg.expert_to_gpu.get(e, local_gpu)}" for e in range(cfg.num_experts)]
    if cfg.use_weight_stationary_moe:
        base: SparseMoE | WeightStationaryMoE = WeightStationaryMoE(
            **base_kwargs, expert_devices=expert_devices
        )
    else:
        base = SparseMoE(**base_kwargs)

    if cfg.expert_to_gpu:
        local_gpu = cfg.local_gpu if cfg.local_gpu is not None else 0
        transport = MoETransport(expert_to_gpu=cfg.expert_to_gpu, local_gpu=local_gpu)
        return TransportMoELayer(base, transport=transport)
    return base


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, i: int, sd: dict, rope: RotaryEmbedding):
        super().__init__()
        self.layer_idx = i
        p = f"model.layers.{i}"
        self.kind = cfg.attention_kind(i)
        self.attn = (
            _full_attn(cfg, sd, p, rope) if self.kind == "full" else _linear_attn(cfg, sd, p)
        )
        self.input_layernorm = RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, sd[f"{p}.input_layernorm.weight"]
        )
        self.post_attention_layernorm = RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, sd[f"{p}.post_attention_layernorm.weight"]
        )
        self.mlp = _build_moe(cfg, sd, p)

    def forward(self, x, positions, ctx, residual):
        if residual is None:
            residual, h = x, self.input_layernorm(x)
        else:
            h, residual = self.input_layernorm(x, residual)
        h = self.attn(h, positions, ctx, self.layer_idx)
        h, residual = self.post_attention_layernorm(h, residual)
        return self.mlp(h), residual


def _qwen3_next_mtp_decoder(cfg, depth_idx, prefix, sd, rope):
    """Full-attention MTP block for Qwen3-Next (MTP layers always use dense
    full attention, not the hybrid linear/full pattern)."""
    hd = cfg.resolved_head_dim()
    attn = GQAAttention(
        num_heads=cfg.num_attention_heads, num_kv_heads=cfg.num_key_value_heads,
        head_dim=hd, qkv_proj=qkv_weight(sd, f"{prefix}.self_attn"),
        o_proj=to_qtensor(sd[f"{prefix}.self_attn.o_proj.weight"]), scale=hd**-0.5, rope=rope,
        q_norm=sd.get(f"{prefix}.self_attn.q_norm.weight"),
        k_norm=sd.get(f"{prefix}.self_attn.k_norm.weight"),
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
                residual, h = x, self.input_layernorm(x)
            else:
                h, residual = self.input_layernorm(x, residual)
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
            return self.mlp(h), residual
    return _MTPBlock()


class Qwen3NextForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig, sd: dict):
        super().__init__()
        self.config = cfg
        self.embed_tokens = VocabEmbedding(sd["model.embed_tokens.weight"], out_dtype=cfg.act_dtype())
        rope = RotaryEmbedding(
            cfg.resolved_head_dim(),
            cfg.max_position_embeddings,
            base=cfg.rope_theta,
            rotary_dim=cfg.rotary_dim(),
        )
        self.layers = nn.ModuleList(
            [Qwen3NextDecoderLayer(cfg, i, sd, rope) for i in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, sd["model.norm.weight"])
        lm_w = sd["model.embed_tokens.weight"] if cfg.tie_word_embeddings else sd["lm_head.weight"]
        self.lm_head = LMHead(to_qtensor(lm_w))
        self.mtp = build_mtp(cfg, sd, self.embed_tokens, self.norm,
                             self.lm_head, rope, _qwen3_next_mtp_decoder)

    def forward(self, input_ids, positions, ctx: ForwardContext):
        h = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            h, residual = layer(h, positions, ctx, residual)
        h, _ = self.norm(h, residual)
        return h

    def compute_logits(self, hidden):
        return self.lm_head(hidden)


@register_model("qwen3_next", "qwen3next", "Qwen3NextForCausalLM")
def build_qwen3_next(cfg: ModelConfig, weights: dict) -> Qwen3NextForCausalLM:
    return Qwen3NextForCausalLM(cfg, weights)
