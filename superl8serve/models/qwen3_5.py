# SPDX-License-Identifier: MIT
"""Qwen3.5 (real `Qwen/Qwen3.5-9B`) — HYBRID Gated-DeltaNet (linear) + GATED full
attention, with a DENSE SwiGLU MLP.

Distinct from `qwen3_next` (the 80B-A3B MoE backbone) on two axes that matter for
the 9B checkpoint:

  1. **Dense MLP.** Qwen3.5-9B has no experts (`num_experts == 0`,
     `intermediate_size == 12288`); every layer is a plain SwiGLU `GatedMLP`. The
     builder still routes to `SparseMoE` when a config DOES carry experts, so the
     larger MoE Qwen3.5/3.6 variants reuse this same assembly.
  2. **Gated full attention** (`attn_output_gate == True`). The full-attention
     layers emit a per-head output gate (q_proj -> query|gate) and multiply the
     attention output by `sigmoid(gate)` before o_proj — see
     `layers/gated_gqa_attention.py`. The linear (DeltaNet) layers are identical to
     `qwen3_next`.

`attention_kind(i)` picks the backend per layer from `layer_types` (3 linear : 1
full, `full_attention_interval == 4`). RoPE is partial-rotary 0.25 over head_dim 256,
theta 1e7. Registered under the HF `model_type`/arch of the (VLM-wrapped) release;
this builder serves the TEXT backbone (vision tower is out of scope here).
"""

from __future__ import annotations

import torch.nn as nn

from ..layers.embedding import LMHead, VocabEmbedding
from ..layers.gated_gqa_attention import GatedGQAAttention
from ..layers.gqa_attention import GQAAttention
from ..layers.linear_attn import GatedDeltaNetAttention
from ..layers.mlp import GatedMLP
from ..layers.norm import RMSNorm
from ..layers.rotary import RotaryEmbedding
from .base import ForwardContext
from .config import ModelConfig
from .moe import SparseMoE
from .mtp import MTPLayer, MultiTokenPredictor
from .registry import register_model
from .weights import gate_up_weight, merge_qtensor, qkv_weight, to_qtensor


def _gated_full_attn(cfg, sd, p, rope):
    """Full-attention layer WITH output gate. HF stores the fused query+gate in
    `self_attn.q_proj.weight` ([2*nh*hd, H]); merging q|k|v yields the
    [2*nh*hd | nkv*hd | nkv*hd] projection GatedGQAAttention expects."""
    hd = cfg.resolved_head_dim()
    return GatedGQAAttention(
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=hd,
        qkv_gate_proj=qkv_weight(sd, f"{p}.self_attn"),
        o_proj=to_qtensor(sd[f"{p}.self_attn.o_proj.weight"]),
        scale=hd**-0.5,
        rope=rope,
        q_norm=sd.get(f"{p}.self_attn.q_norm.weight"),
        k_norm=sd.get(f"{p}.self_attn.k_norm.weight"),
        rms_norm_eps=cfg.rms_norm_eps,
        # Qwen3.5's Qwen3_5RMSNorm is ZERO-CENTERED (gain = 1 + weight, weight init 0),
        # incl. q_norm/k_norm. (The gated DeltaNet output norm is standard, unchanged.)
        qk_unit_offset=True,
    )


def _la_weight(sd, la, native, *aliases):
    """Resolve a DeltaNet weight by its serve-native name, falling back to the raw
    HF names some `.superl8` checkpoints carry un-remapped. Qwen3.5-0.8B ships the
    *separate*-projection HF layout (`in_proj_qkv/z/b/a`, `conv1d`) that convert.py's
    remap (written for Qwen3-Next's *fused* `in_proj_qkvz/ba`) doesn't rewrite, so
    accept both. Pure renames — the int8 weights are identical."""
    for name in (native, *aliases):
        key = f"{la}.{name}"
        if key in sd:
            return sd[key]
    raise KeyError(f"{la}.{native} (or aliases {aliases!r})")


def _la_dims(sd, la, x):
    """DeltaNet head dims live in cfg.extra, but a round-tripped `.superl8` meta drops
    the `extra` overflow, so recover them from the weight shapes when absent. Uses
    the standard Gated-DeltaNet invariants (num_k_heads == num_v_heads, key_dim ==
    value_dim for this family): num_v_heads = A_log length, value_dim = per-head norm
    gain length, and qkv_out = 2*num_k*key_dim + num_v*value_dim pins key_dim."""
    keys = (
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
    )
    if all(k in x for k in keys):
        return {k: x[k] for k in keys}  # full config survived — no derivation needed

    def _shape(w):  # weights may be plain tensors or int8 QTensors (no `.shape`)
        return tuple(getattr(w, "logical_shape", None) or w.shape)

    nv = int(_shape(_la_weight(sd, la, "A_log"))[0])
    vd = int(_shape(_la_weight(sd, la, "norm.weight"))[0])
    qkv_out = int(_shape(_la_weight(sd, la, "qkv_proj.weight", "in_proj_qkv.weight"))[0])
    nk = nv  # this family ties key/value head counts
    kd = (qkv_out - nv * vd) // (2 * nk)
    ck = int(_shape(_la_weight(sd, la, "conv_weight", "conv1d.weight"))[-1])
    return dict(
        linear_num_key_heads=x.get("linear_num_key_heads", nk),
        linear_num_value_heads=x.get("linear_num_value_heads", nv),
        linear_key_head_dim=x.get("linear_key_head_dim", kd),
        linear_value_head_dim=x.get("linear_value_head_dim", vd),
        linear_conv_kernel_dim=x.get("linear_conv_kernel_dim", ck),
    )


def _linear_attn(cfg, sd, p):
    x = cfg.extra
    la = f"{p}.linear_attn"
    d = _la_dims(sd, la, x)
    # conv1d.weight is [dim, 1, kernel]; the kernel wants [dim, kernel].
    conv_w = _la_weight(sd, la, "conv_weight", "conv1d.weight")
    if conv_w.dim() == 3:
        conv_w = conv_w.squeeze(1)
    return GatedDeltaNetAttention(
        cfg,
        qkv_proj=to_qtensor(_la_weight(sd, la, "qkv_proj.weight", "in_proj_qkv.weight")),
        out_proj=to_qtensor(_la_weight(sd, la, "out_proj.weight")),
        conv_weight=conv_w,
        a_log=_la_weight(sd, la, "A_log"),
        dt_bias=_la_weight(sd, la, "dt_bias"),
        beta_proj=None,
        gate_proj=None,
        gate_beta_proj=merge_qtensor(
            [
                to_qtensor(_la_weight(sd, la, "dt_proj.weight", "in_proj_a.weight")),
                to_qtensor(_la_weight(sd, la, "beta_proj.weight", "in_proj_b.weight")),
            ]
        ),
        z_proj=to_qtensor(_la_weight(sd, la, "z_proj.weight", "in_proj_z.weight")),
        norm_gain=_la_weight(sd, la, "norm.weight"),
        num_k_heads=d["linear_num_key_heads"],
        num_v_heads=d["linear_num_value_heads"],
        key_dim=d["linear_key_head_dim"],
        value_dim=d["linear_value_head_dim"],
        conv_kernel=d["linear_conv_kernel_dim"],
    )


def _build_mlp(cfg, sd, p):
    """Dense SwiGLU (Qwen3.5-9B) or ultra-sparse MoE + shared expert (larger MoE
    variants). Selected purely by whether the config carries experts."""
    if not cfg.is_moe():
        return GatedMLP(
            gate_up_weight(sd, f"{p}.mlp"),
            to_qtensor(sd[f"{p}.mlp.down_proj.weight"]),
            act=cfg.hidden_act,
        )
    from .moe import WeightStationaryMoE
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
        local_gpu = cfg.local_gpu if cfg.local_gpu is not None else 0
        expert_devices = [
            f"cuda:{cfg.expert_to_gpu.get(e, local_gpu)}" for e in range(cfg.num_experts)
        ]
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


def _qwen3_5_mtp_decoder(cfg, prefix, sd, rope):
    """One MTP-depth decoder block for Qwen3.5: GATED full attention + SwiGLU MLP,
    with Qwen3.5's zero-centered (add_unit_offset) norms. Mirrors
    ``qwen3.py::_qwen3_mtp_decoder`` but uses ``GatedGQAAttention`` (fused
    query|gate ``q_proj``) and the qk_unit_offset q/k-norm.

    The attention is dense/fp16 (``GQAAttention._draft_attn`` on the query half — the
    int8 paged-decode kernel is head-dim {32,64,128} only, this family is 256) and
    NEVER touches the main model's paged KV cache OR the DeltaNet recurrent state in
    ``ctx.lin_cache``. The block is split into ``project`` (fused qkv+gate, qk-norm,
    RoPE) and ``attend`` (dense attention over an externally supplied K/V + gate +
    o_proj + residual) so the engine can give the draft a PREFIX K/V cache — the
    head's own K/V over the committed context, primed at prefill and each accepted
    step (mirrors qengine's inline nextn head; Haru-neo/qengine, Apache-2.0). Without
    it the draft attends only its own single token and predicts noise (~1% accept).
    """
    attn = _gated_full_attn(cfg, sd, prefix, rope)

    class _MTPBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = attn
            self.input_layernorm = RMSNorm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                sd[f"{prefix}.input_layernorm.weight"],
                add_unit_offset=True,
            )
            self.post_attention_layernorm = RMSNorm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                sd[f"{prefix}.post_attention_layernorm.weight"],
                add_unit_offset=True,
            )
            self.mlp = GatedMLP(
                gate_up_weight(sd, f"{prefix}.mlp"),
                to_qtensor(sd[f"{prefix}.mlp.down_proj.weight"]),
                act=cfg.hidden_act,
            )

        def project(self, x, positions):
            """fc-output hidden ``x`` [B,S,H] -> (q, gate, k, v, residual), q/k RoPE'd,
            qk-norm applied. ``k``/``v`` are token-major [B,S,nkv,hd] ready to append to
            the MTP prefix-KV cache; ``residual`` (== ``x``, the block-entry residual)
            is threaded into :meth:`attend`."""
            residual, h = x, self.input_layernorm(x)
            a = self.self_attn
            q, gate, k, v = a._project(h)  # gated projection + qk-norm (pre-RoPE)
            q, k = a.rope(positions, q, k)
            return q, gate, k, v, residual

        def attend(self, q, gate, k_all, v_all, residual, B, S):
            """Dense attention of the S query tokens over ``k_all``/``v_all`` (the
            prefix-KV cache PLUS this step's K/V, token-major [B,N,nkv,hd]), then the
            sigmoid output gate, o_proj, and BOTH residual adds. Returns the full block
            hidden [B,S,H] (attn-residual + mlp), so the head's final norm sees the
            complete residual stream — the earlier code dropped the post-MLP residual."""
            a = self.self_attn
            out = GQAAttention._draft_attn(
                q.transpose(1, 2), k_all.transpose(1, 2), v_all.transpose(1, 2), scale=a.scale
            )
            out = out.view(B, S, a.nh, a.hd)
            h = a._gate_and_project(out, gate, B, S)  # sigmoid gate + o_proj
            h, residual = self.post_attention_layernorm(h, residual)
            return self.mlp(h) + residual

        def forward(self, x, positions, ctx, residual):
            # Cache-free single-token path (fallback / non-prefix-KV callers). residual
            # arg kept for signature parity; MTP always enters with residual=None.
            B, S, _ = x.shape
            q, gate, k, v, res = self.project(x, positions)
            return self.attend(q, gate, k, v, res, B, S), None

    return _MTPBlock()


def _resolve_mtp_prefix(sd: dict) -> str | None:
    """The Qwen3.5-0.8B `.superl8` ships the MTP head at ROOT level (`mtp.*`), a sibling
    of the text backbone that lives under `model.language_model.*`; a differently
    nested VLM checkpoint may carry it as `model.mtp.*` (after language_model unwrap).
    Accept either. Returns the prefix or None if no MTP head is present."""
    for cand in ("mtp", "model.mtp", "model.language_model.mtp"):
        if f"{cand}.fc.weight" in sd:
            return cand
    return None


def build_qwen3_5_mtp(cfg, sd, embed, lm_head, rope):
    """Build the Qwen3.5 MTP speculative-decode head if the checkpoint ships one.

    The shipped 0.8B layout is a SINGLE depth: a shared `fc` (2H->H), two pre-fc
    RMSNorms, a gated-full-attn + MLP decoder block (`mtp.layers.0.*`), and the
    head's OWN final norm (`mtp.norm.weight`) applied before the SHARED int8 LM head.
    All norms are zero-centered (Qwen3_5RMSNorm), forced on here since the config
    does not carry `norm_add_unit_offset`.

    Depth count is recovered from the WEIGHTS, not the config: the shipped 0.8B
    `.superl8` meta baked in `num_mtp_layers=0` (its converter didn't read the nested
    `text_config.mtp_num_hidden_layers`), so gating on the config would silently
    skip a head that is physically present — same weight-recovery pattern as
    `_la_dims`. A fresh conversion from the raw HF config carries the count and it
    is honored as an upper bound."""
    mp = _resolve_mtp_prefix(sd)
    if mp is None:
        return None  # no MTP head shipped in this checkpoint
    shared_required = (
        f"{mp}.fc.weight",
        f"{mp}.pre_fc_norm_hidden.weight",
        f"{mp}.pre_fc_norm_embedding.weight",
        f"{mp}.norm.weight",
    )
    if not all(name in sd for name in shared_required):
        return None

    def _depth_complete(depth: int) -> bool:
        p = f"{mp}.layers.{depth}"
        required = (
            f"{p}.input_layernorm.weight",
            f"{p}.post_attention_layernorm.weight",
            f"{p}.self_attn.q_proj.weight",
            f"{p}.self_attn.k_proj.weight",
            f"{p}.self_attn.v_proj.weight",
            f"{p}.self_attn.o_proj.weight",
            f"{p}.mlp.gate_proj.weight",
            f"{p}.mlp.up_proj.weight",
            f"{p}.mlp.down_proj.weight",
        )
        return all(name in sd for name in required)

    n_present = 0
    while _depth_complete(n_present):
        n_present += 1
    n_depths = min(n_present, cfg.num_mtp_layers) if cfg.num_mtp_layers > 0 else n_present
    layers = []
    for d in range(n_depths):
        layers.append(
            MTPLayer(
                cfg,
                fc_weight=sd[f"{mp}.fc.weight"],
                hidden_norm=sd[f"{mp}.pre_fc_norm_hidden.weight"],
                emb_norm=sd[f"{mp}.pre_fc_norm_embedding.weight"],
                block=_qwen3_5_mtp_decoder(cfg, f"{mp}.layers.{d}", sd, rope),
                norm_add_unit_offset=True,
            )
        )
    if not layers:
        return None
    # Qwen3.5's MTP carries its OWN final norm (not the main model's) before the
    # shared LM head — use it so draft logits match the head's training.
    mtp_norm = RMSNorm(
        cfg.hidden_size, cfg.rms_norm_eps, sd[f"{mp}.norm.weight"], add_unit_offset=True
    )
    return MultiTokenPredictor(
        cfg, layers=layers, embed=embed, final_norm=mtp_norm, lm_head=lm_head
    )


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, i: int, sd: dict, rope: RotaryEmbedding):
        super().__init__()
        self.layer_idx = i
        p = f"model.layers.{i}"
        self.kind = cfg.attention_kind(i)
        self.attn = (
            _gated_full_attn(cfg, sd, p, rope) if self.kind == "full" else _linear_attn(cfg, sd, p)
        )
        # Qwen3.5 RMSNorm is zero-centered (Gemma-style gain = 1 + weight).
        self.input_layernorm = RMSNorm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            sd[f"{p}.input_layernorm.weight"],
            add_unit_offset=True,
        )
        self.post_attention_layernorm = RMSNorm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            sd[f"{p}.post_attention_layernorm.weight"],
            add_unit_offset=True,
        )
        self.mlp = _build_mlp(cfg, sd, p)

    def forward(self, x, positions, ctx, residual):
        if residual is None:
            residual, h = x, self.input_layernorm(x)
        else:
            h, residual = self.input_layernorm(x, residual)
        h = self.attn(h, positions, ctx, self.layer_idx)
        h, residual = self.post_attention_layernorm(h, residual)
        return self.mlp(h), residual


def _unwrap_vlm_text_backbone(sd: dict) -> dict:
    """Real Qwen3.5 ships as a VLM wrapper (`Qwen3_5ForConditionalGeneration`):
    published `.superl8`s store the text backbone under `model.language_model.*` and
    the vision tower under `model.visual.*`. To serve text we strip the
    `model.language_model.` prefix down to `model.` and drop the vision tower.
    No-op for a plain (already-unwrapped) text checkpoint."""
    if not any(k.startswith("model.language_model.") for k in sd):
        return sd
    pref = "model.language_model."
    out = {}
    for k, v in sd.items():
        if k.startswith("model.visual.") or ".visual." in k:
            continue  # vision tower — not needed to serve the text model
        out[("model." + k[len(pref) :]) if k.startswith(pref) else k] = v
    return out


class Qwen3_5ForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig, sd: dict):
        super().__init__()
        self.config = cfg
        sd = _unwrap_vlm_text_backbone(sd)
        self.embed_tokens = VocabEmbedding(
            sd["model.embed_tokens.weight"], out_dtype=cfg.act_dtype()
        )
        rope = RotaryEmbedding(
            cfg.resolved_head_dim(),
            cfg.max_position_embeddings,
            base=cfg.rope_theta,
            rotary_dim=cfg.rotary_dim(),
        )
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(cfg, i, sd, rope) for i in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, sd["model.norm.weight"], add_unit_offset=True
        )  # zero-centered (Qwen3_5RMSNorm)
        lm_w = sd["model.embed_tokens.weight"] if cfg.tie_word_embeddings else sd["lm_head.weight"]
        self.lm_head = LMHead(to_qtensor(lm_w))
        # MTP speculative-decode head (present in the shipped .superl8; None if absent).
        self.mtp = build_qwen3_5_mtp(cfg, sd, self.embed_tokens, self.lm_head, rope)

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
    "qwen3_5",
    "qwen3.5",
    "qwen3_5_text",  # text backbone of the VLM wrapper — published .superl8s (e.g.
    "qwen3_5_text_config",  # Qwen3.5-0.8B-superl8) carry this as their meta arch
    "qwen3_5_moe",  # Qwen3.6-35B-A3B: same hybrid backbone, SparseMoE MLP branch
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5ForCausalLM",
    # NOTE: `Qwen3_5ForConditionalGeneration` (the raw HF VLM arch) is claimed by
    # models/qwen3_5_vl.py, which composes THIS text backbone with the vision tower.
    # A text-only checkpoint routes here via its `qwen3_5_text` meta arch.
    "Qwen3_5TextModel",
)
def build_qwen3_5(cfg: ModelConfig, weights: dict) -> Qwen3_5ForCausalLM:
    return Qwen3_5ForCausalLM(cfg, weights)
