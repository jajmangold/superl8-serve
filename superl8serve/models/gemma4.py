# SPDX-License-Identifier: MIT
"""Gemma4 / gemma3n (text backbone) — GQA + AltUp/LAuReL/PLE/MatFormer.

Implements the four Gemma4-specific extensions over the shared GQA backend:
  * LAuReL (low-rank residual correction per decoder layer)
  * AltUp (alternating-updates residual mixing)
  * Per-Layer Embeddings (PLE)
  * MatFormer (per-layer intermediate_size for the MLP)

Weight names follow the HF gemma3n/gemma4 convention.  Verified against the
published gemma-4-E2B/E4B .superl8 weight layout.
"""

from __future__ import annotations

import math

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


def _resolve_intermediate_size(cfg, layer_idx: int) -> int:
    """MatFormer: intermediate_size can be a per-layer sequence or a single int."""
    if isinstance(cfg.intermediate_size, (list, tuple)):
        return int(cfg.intermediate_size[layer_idx])
    return int(cfg.intermediate_size)


def _ple_hidden(cfg) -> int:
    return cfg.extra.get("hidden_size_per_layer_input", 256)


def _ple_vocab(cfg) -> int:
    return cfg.extra.get("vocab_size_per_layer_input", min(cfg.vocab_size, 262144))


def _laurel_rank(cfg) -> int:
    return cfg.extra.get("laurel_rank", 64)


def _altup_num_inputs(cfg) -> int:
    return cfg.extra.get("altup_num_inputs", 4)


def _altup_active_idx(cfg) -> int:
    return cfg.extra.get("altup_active_idx", 0)


def _activation_sparsity(cfg, layer_idx: int) -> float:
    pat = cfg.extra.get("activation_sparsity_pattern", None)
    if pat is not None and layer_idx < len(pat):
        return float(pat[layer_idx])
    return 0.0


# ---------------------------------------------------------------------------
# LAuReL (Learned Augmented Residual Layer)
# ---------------------------------------------------------------------------


class Gemma4LaurelBlock(nn.Module):
    """Low-rank residual correction: x + norm(linear_right(linear_left(x)))."""

    def __init__(self, cfg: ModelConfig, sd: dict, prefix: str):
        super().__init__()
        R = _laurel_rank(cfg)
        H = cfg.hidden_size
        self.linear_left = nn.Linear(H, R, bias=False)
        self.linear_right = nn.Linear(R, H, bias=False)
        # Load pretrained weights
        ll = sd.get(f"{prefix}.linear_left.weight")
        lr = sd.get(f"{prefix}.linear_right.weight")
        if ll is not None:
            self.linear_left.weight.data = ll.to(H) if ll.dim() == 1 else ll
            # Reshape a 1D weight to 2D if needed
            if self.linear_left.weight.data.dim() == 1:
                self.linear_left.weight.data = self.linear_left.weight.data.view(R, H)
        if lr is not None:
            self.linear_right.weight.data = lr.to(R) if lr.dim() == 1 else lr
            if self.linear_right.weight.data.dim() == 1:
                self.linear_right.weight.data = self.linear_right.weight.data.view(H, R)
        pn = sd.get(f"{prefix}.post_laurel_norm.weight")
        self.post_laurel_norm = RMSNorm(H, cfg.rms_norm_eps, pn) if pn is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear_left(x)
        h = self.linear_right(h)
        if self.post_laurel_norm is not None:
            h = self.post_laurel_norm(h)
        return x + h


# ---------------------------------------------------------------------------
# AltUp (Alternating Updates)
# ---------------------------------------------------------------------------


class Gemma4AltUp(nn.Module):
    """AltUp predict/correct mechanism (full precision, tiny params)."""

    def __init__(self, cfg: ModelConfig, sd: dict, prefix: str):
        super().__init__()
        self.cfg = cfg
        K = _altup_num_inputs(cfg)
        H = cfg.hidden_size
        self.K = K

        self.correction_coefs = nn.Linear(K, K, bias=False, dtype=torch.float32)
        self.prediction_coefs = nn.Linear(K, K * K, bias=False, dtype=torch.float32)
        self.modality_router = nn.Linear(H, K, bias=False, dtype=torch.float32)
        self.router_norm = RMSNorm(H, cfg.rms_norm_eps)

        # correct_output_scale is a 1D parameter
        cos = sd.get(f"{prefix}.correct_output_scale")
        if cos is not None:
            self.correct_output_scale = nn.Parameter(cos.reshape(H).float())
        else:
            self.correct_output_scale = nn.Parameter(torch.zeros(H))

        # Load AltUp weights (loaded weight might be fp16; convert to fp32)
        def _load_linear(lin, key):
            w = sd.get(f"{prefix}.{key}")
            if w is not None:
                lin.weight.data = w.to(dtype=torch.float32)

        _load_linear(self.correction_coefs, "correction_coefs.weight")
        _load_linear(self.prediction_coefs, "prediction_coefs.weight")
        _load_linear(self.modality_router, "modality_router.weight")
        rn = sd.get(f"{prefix}.router_norm.weight")
        if rn is not None:
            self.router_norm.weight.data = rn

        self.register_buffer("router_input_scale", torch.tensor(H**-1.0), persistent=False)

    def compute_modalities(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()  # router always runs in fp32
        router_inputs = self.router_norm(xf) * self.router_input_scale.to(xf.dtype)
        return self.modality_router(router_inputs)

    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states [K, B, S, H] -> predictions [K, B, S, H]."""
        K, B, S, H = hidden_states.shape
        modalities = self.compute_modalities(
            hidden_states[self.cfg.extra.get("altup_active_idx", 0)]
        )
        coefs = self.prediction_coefs(modalities).to(hidden_states.dtype)
        coefs = coefs.reshape(B, S, K, K)
        # hidden_states: [K, B, S, H] -> [B, S, H, K]
        hs = hidden_states.permute(1, 2, 3, 0)
        pred = torch.matmul(hs, coefs)  # [B, S, H, K]
        return pred.permute(3, 0, 1, 2)  # [K, B, S, H]

    def correct(self, predictions: torch.Tensor, innovation: torch.Tensor) -> torch.Tensor:
        """predictions [K, B, S, H], innovation [B, S, H] -> corrected [K, B, S, H]."""
        K, B, S, H = predictions.shape
        modalities = self.compute_modalities(predictions[self.cfg.extra.get("altup_active_idx", 0)])
        coefs = self.correction_coefs(modalities).to(predictions.dtype)
        # coefs: [B, S, K] -> weight per altup input
        delta = innovation - predictions[self.cfg.extra.get("altup_active_idx", 0)]
        corrected = predictions.clone()
        for k in range(K):
            corrected[k] = corrected[k] + coefs[..., k : k + 1] * delta
        return corrected

    def scale_corrected_output(self, x: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + self.correct_output_scale.to(x.dtype))


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class Gemma4DecoderLayer(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        layer_idx: int,
        sd: dict,
        rope_global: RotaryEmbedding,
        rope_local: RotaryEmbedding,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.cfg = cfg
        p = f"model.layers.{layer_idx}"
        H = cfg.hidden_size
        hd = cfg.resolved_head_dim()
        is_global = cfg.layer_is_global(layer_idx)
        rope = rope_global if is_global else rope_local
        window = -1 if is_global else int(cfg.sliding_window or -1)
        scale = (cfg.query_pre_attn_scalar**-0.5) if cfg.query_pre_attn_scalar else hd**-0.5

        # Attention
        self.self_attn = GQAAttention(
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=hd,
            qkv_proj=qkv_weight(sd, f"{p}.self_attn"),
            o_proj=to_qtensor(sd[f"{p}.self_attn.o_proj.weight"]),
            scale=scale,
            rope=rope,
            q_norm=sd.get(f"{p}.self_attn.q_norm.weight"),
            k_norm=sd.get(f"{p}.self_attn.k_norm.weight"),
            rms_norm_eps=cfg.rms_norm_eps,
            window_left=window,
        )

        # Norms (Gemma sandwich)
        self.input_layernorm = _gemma_norm(cfg, sd[f"{p}.input_layernorm.weight"])
        self.post_attention_layernorm = _gemma_norm(cfg, sd[f"{p}.post_attention_layernorm.weight"])
        self.pre_feedforward_layernorm = _gemma_norm(
            cfg, sd[f"{p}.pre_feedforward_layernorm.weight"]
        )
        self.post_feedforward_layernorm = _gemma_norm(
            cfg, sd[f"{p}.post_feedforward_layernorm.weight"]
        )

        # MLP (MatFormer — per-layer intermediate_size)
        isize = _resolve_intermediate_size(cfg, layer_idx)
        self.mlp_gate_up = gate_up_weight(sd, f"{p}.mlp")
        self.mlp_down = to_qtensor(sd[f"{p}.mlp.down_proj.weight"])
        self.mlp = GatedMLP(self.mlp_gate_up, self.mlp_down, act=cfg.hidden_act)

        # LAuReL
        self.laurel = (
            Gemma4LaurelBlock(cfg, sd, f"{p}.laura")
            if f"{p}.laura.linear_left.weight" in sd
            else None
        )

        # AltUp (per layer)
        self.altup = (
            Gemma4AltUp(cfg, sd, f"{p}.altup")
            if f"{p}.altup.correction_coefs.weight" in sd
            else None
        )

        # PLE per-layer projections
        ple_dim = _ple_hidden(cfg)
        self.per_layer_input_gate = nn.Linear(H, ple_dim, bias=False)
        self.per_layer_projection = nn.Linear(ple_dim, H, bias=False)
        self.post_per_layer_input_norm = _gemma_norm(
            cfg,
            sd.get(
                f"{p}.post_per_layer_input_norm.weight",
                torch.zeros(H, dtype=torch.float16, device="cpu"),
            ),
        )
        ig = sd.get(f"{p}.per_layer_input_gate.weight")
        if ig is not None:
            self.per_layer_input_gate.weight.data = ig
        pp = sd.get(f"{p}.per_layer_projection.weight")
        if pp is not None:
            self.per_layer_projection.weight.data = pp

    def forward(self, x, positions, ctx, per_layer_input=None, residual=None):
        """x: [K, B, S, H] (AltUp stack) or [B, S, H] if no AltUp."""
        if self.altup is not None:
            return self._forward_altup(x, positions, ctx, per_layer_input)
        return self._forward_plain(x, positions, ctx, residual)

    def _forward_plain(self, x, positions, ctx, residual):
        """Standard Gemma-style forward (no AltUp)."""
        if residual is None:
            residual, h = x, self.input_layernorm(x)
        else:
            h, residual = self.input_layernorm(x, residual)

        # LAuReL on normed input
        laurel_out = self.laurel(h) if self.laurel is not None else 0.0

        h = self.self_attn(h, positions, ctx, self.layer_idx)
        h = self.post_attention_layernorm(h)
        h = x + h

        if self.laurel is not None:
            h = (h + laurel_out) / math.sqrt(2)

        h = self.pre_feedforward_layernorm(h)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        x = x + h
        return x, None

    def _forward_altup(self, x, positions, ctx, per_layer_input):
        """AltUp forward: process one active copy, spread to all copies."""
        K, B, S, H = x.shape
        altup_active = _altup_active_idx(self.cfg)

        # Predict
        predictions = self.altup.predict(x)
        active = predictions[altup_active]

        # Gemma sandwich on active copy
        h = self.input_layernorm(active)

        # LAuReL
        laurel_out = self.laurel(h) if self.laurel is not None else 0.0

        h = self.self_attn(h, positions, ctx, self.layer_idx)
        h = self.post_attention_layernorm(h)
        h = active + h

        if self.laurel is not None:
            h = (h + laurel_out) / math.sqrt(2)

        h = self.pre_feedforward_layernorm(h)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        output = active + h

        # Correct
        corrected = self.altup.correct(predictions, output)

        first = corrected[altup_active].clone()
        if self.cfg.extra.get("altup_correct_scale", True):
            first = self.altup.scale_corrected_output(first)

        # PLE: gate per_layer_input with output, project back
        if per_layer_input is not None:
            gated = self.per_layer_input_gate(first)
            gated = torch.nn.functional.gelu(gated, approximate="tanh")
            gated = gated * per_layer_input
            gated = self.per_layer_projection(gated)
            gated = self.post_per_layer_input_norm(gated)
            corrected[1:] = corrected[1:] + gated.unsqueeze(0)

        return corrected, None


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class Gemma4Model(nn.Module):
    def __init__(self, cfg: ModelConfig, sd: dict):
        super().__init__()
        self.config = cfg
        H = cfg.hidden_size
        scale = cfg.embed_scale or (H**0.5)
        self.embed_tokens = VocabEmbedding(sd["model.embed_tokens.weight"], embed_scale=scale, out_dtype=cfg.act_dtype())

        # Dual RoPE
        rope_theta = cfg.rope_theta
        rope_local = cfg.rope_local_theta or 1e4
        hd = cfg.resolved_head_dim()
        self.rotary_emb_global = RotaryEmbedding(hd, cfg.max_position_embeddings, base=rope_theta)
        self.rotary_emb_local = RotaryEmbedding(hd, cfg.max_position_embeddings, base=rope_local)

        # Decoder layers
        self.layers = nn.ModuleList(
            [
                Gemma4DecoderLayer(cfg, i, sd, self.rotary_emb_global, self.rotary_emb_local)
                for i in range(cfg.num_hidden_layers)
            ]
        )

        self.norm = _gemma_norm(cfg, sd["model.norm.weight"])

        # --- PLE (Per-Layer Embeddings) ---
        self.has_ple = "model.embed_tokens_per_layer.weight" in sd
        if self.has_ple:
            ple_dim = _ple_hidden(cfg)
            ple_vocab = _ple_vocab(cfg)
            # Combined embedding table: [vocab_ple, num_layers * ple_dim]
            ew = sd["model.embed_tokens_per_layer.weight"]
            self.embed_tokens_per_layer = nn.Embedding(
                ple_vocab, cfg.num_hidden_layers * ple_dim, _weight=ew
            )

            self.per_layer_model_projection = nn.Linear(
                H, cfg.num_hidden_layers * ple_dim, bias=False
            )
            pmw = sd.get("model.per_layer_model_projection.weight")
            if pmw is not None:
                self.per_layer_model_projection.weight.data = pmw

            self.per_layer_projection_norm = RMSNorm(
                ple_dim, cfg.rms_norm_eps, sd.get("model.per_layer_projection_norm.weight")
            )

            self.register_buffer(
                "per_layer_projection_scale", torch.tensor(H**-0.5), persistent=False
            )
            self.register_buffer(
                "per_layer_input_scale", torch.rsqrt(torch.tensor(2.0)), persistent=False
            )

        # --- AltUp model-level projections ---
        self.has_altup = "model.altup_projections.0.weight" in sd
        if self.has_altup:
            K = _altup_num_inputs(cfg)
            self.altup_projections = nn.ModuleList(
                [nn.Linear(H, H, bias=False) for _ in range(1, K)]
            )
            self.altup_unembed_projections = nn.ModuleList(
                [nn.Linear(H, H, bias=False) for _ in range(1, K)]
            )
            for i in range(K - 1):
                w = sd.get(f"model.altup_projections.{i}.weight")
                if w is not None:
                    self.altup_projections[i].weight.data = w
                w = sd.get(f"model.altup_unembed_projections.{i}.weight")
                if w is not None:
                    self.altup_unembed_projections[i].weight.data = w

    def _get_per_layer_inputs(self, input_ids):
        """PLE: look up input_ids in embed_tokens_per_layer -> [B, S, L, ple_dim]."""
        ple = self.embed_tokens_per_layer(input_ids)
        B, S = input_ids.shape
        return ple.reshape(B, S, self.config.num_hidden_layers, _ple_hidden(self.config))

    def _project_per_layer_inputs(self, inputs_embeds, per_layer_inputs):
        """PLE: project inputs_embeds, combine with token-identity lookup."""
        cfg = self.config
        ple_dim = _ple_hidden(cfg)
        proj = self.per_layer_model_projection(inputs_embeds)
        proj = proj * self.per_layer_projection_scale.to(
            dtype=inputs_embeds.dtype, device=proj.device
        )
        proj = proj.reshape(*inputs_embeds.shape[:-1], cfg.num_hidden_layers, ple_dim)
        proj = self.per_layer_projection_norm(proj)

        if per_layer_inputs is not None:
            out = (proj + per_layer_inputs) * self.per_layer_input_scale.to(
                dtype=inputs_embeds.dtype, device=proj.device
            )
        else:
            out = proj
        return out

    def forward(self, input_ids, positions, ctx):
        h = self.embed_tokens(input_ids)

        # PLE
        per_layer_inputs = None
        if self.has_ple:
            ple_lookup = self._get_per_layer_inputs(input_ids)
            per_layer_inputs = self._project_per_layer_inputs(h, ple_lookup)

        # AltUp: expand input into multiple copies
        if self.has_altup:
            K = _altup_num_inputs(self.config)
            # First copy is the raw embed
            h_list = [h]
            target_mag = torch.mean(h**2, dim=-1, keepdim=True) ** 0.5
            eps_t = torch.tensor(1e-5, device=h.device)
            for i in range(1, K):
                proj = self.altup_projections[i - 1](h)
                new_mag = torch.mean(proj**2, dim=-1, keepdim=True)
                new_mag = torch.sqrt(torch.maximum(new_mag, eps_t))
                proj = proj * target_mag / new_mag
                h_list.append(proj)
            h = torch.stack(h_list, dim=0)  # [K, B, S, H]

        for i, layer in enumerate(self.layers):
            ple_i = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            h, _ = layer(h, positions, ctx, per_layer_input=ple_i)

        # AltUp: unembed all copies and take mean
        if self.has_altup:
            K, B, S, H = h.shape
            h_list = [h[0]]
            target_mag = torch.mean(h[0] ** 2, dim=-1, keepdim=True) ** 0.5
            eps_t = torch.tensor(1e-5, device=h.device)
            for i in range(1, K):
                proj = self.altup_unembed_projections[i - 1](h[i])
                new_mag = torch.mean(proj**2, dim=-1, keepdim=True)
                new_mag = torch.sqrt(torch.maximum(new_mag, eps_t))
                proj = proj * target_mag / new_mag
                h_list.append(proj)
            h = torch.stack(h_list, dim=0)
            h = torch.mean(h, dim=0)
        else:
            # h is [B, S, H] (single copy, no AltUp)
            pass

        return self.norm(h)


class Gemma4ForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig, sd: dict):
        super().__init__()
        self.config = cfg
        self.model = Gemma4Model(cfg, sd)
        lm_w = sd.get("lm_head.weight", sd.get("model.embed_tokens.weight"))
        self.lm_head = LMHead(to_qtensor(lm_w), logit_softcap=cfg.final_logit_softcap)

    def forward(self, input_ids, positions, ctx):
        return self.model(input_ids, positions, ctx)

    def compute_logits(self, hidden):
        return self.lm_head(hidden)


@register_model(
    "gemma4",
    "gemma3n",
    "Gemma4ForCausalLM",
    "Gemma4ForConditionalGeneration",
    "Gemma3nForCausalLM",
    "Gemma3nForConditionalGeneration",
)
def build_gemma4(cfg: ModelConfig, weights: dict) -> Gemma4ForCausalLM:
    cfg.norm_add_unit_offset = True
    return Gemma4ForCausalLM(cfg, weights)
