# SPDX-License-Identifier: MIT
"""ModelConfig — the union of architectural axes across supported families.

One flat dataclass whose fields are a superset of the HF `config.json` knobs we
care about, so a new model family is expressed as *values*, not code. A concrete
model's `build()` reads only the fields it needs; unused fields stay at their
defaults. `from_hf(dict)` maps a HuggingFace text config (or the `text_config`
sub-dict of a multimodal one) onto this schema.

Axes captured: GQA dims, norm style (plain vs Gemma (1+w)), activation, RoPE
(single- or dual-theta for Gemma local/global), sliding-window pattern, QK-norm,
attention/query scaling + soft-caps, MoE routing, MTP depth, decode strategy
(autoregressive vs diffusion), and multimodal VLM detection (VisionConfig for
Qwen2-VL/Qwen2.5-VL, LLaVA/LLaVA-Next).
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Architecture keys that carry a vision backbone. Any model_type matching
# this set triggers is_multimodal and vision_config parsing in from_hf().
_MULTIMODAL_ARCHS = frozenset(
    {
        "qwen2_vl",
        "qwen2_5_vl",
        "llava",
        "llava_next",
        "qwen3_5_vl",
    }
)


# Gated-DeltaNet hybrid families and their standard full-attention stride (3 linear
# : 1 full => every 4th layer is full softmax attention). A REAL HF config marks this
# with `layer_types`/`attn_type_list`, but a round-tripped `.superl8` meta carries only
# the flattened scalars — and older converters dumped the dataclass defaults
# (linear_attention=False, full_attention_interval=0), losing the pattern. When the
# arch is a known hybrid but nothing derived a pattern, fall back to the family stride
# so the checkpoint still reconstructs its DeltaNet/full layout instead of building
# every layer as full attention (which KeyErrors on the DeltaNet layers' missing
# q_proj at load).
_HYBRID_LINEAR_DEFAULT_INTERVAL = {
    "qwen3_5_text": 4,
    "qwen3_5_vl": 4,  # VLM wrapper's text backbone is the same DeltaNet/full hybrid
    "qwen3_5_moe": 4,  # Qwen3.6-35B-A3B uses the same 3:1 hybrid text backbone
    "qwen3_6_text": 4,
    "qwen3_next": 4,
}


@dataclass
class VisionConfig:
    """Configuration for a vision backbone (ViT / SigLIP) within a VLM.

    Extracted from the ``vision_config`` sub-dict of a multimodal HF config.json.
    Qwen2-VL family stores the layer count as ``depth``; LLaVA family uses
    ``num_hidden_layers``. ``spatial_merge_size`` is Qwen-specific (defaults to 1
    for LLaVA which doesn't use it).
    """

    hidden_size: int
    patch_size: int
    num_layers: int
    image_token_id: int
    spatial_merge_size: int = 1
    num_attention_heads: int = 0
    intermediate_size: int = 0
    in_channels: int = 3
    layer_norm_eps: float = 1e-6
    hidden_act: str = "gelu_pytorch_tanh"
    # Qwen3.5-VL extras (Conv3d patch embed + learned interpolated pos-embed + a
    # norm→MLP merger that projects to the LLM hidden size). Absent for Qwen2.5-VL.
    temporal_patch_size: int = 2
    num_position_embeddings: int = 0  # learned pos-embed table size (Qwen3.5)
    out_hidden_size: int = 0  # merger output dim (== LLM hidden_size); 0 => hidden_size
    deepstack_visual_indexes: tuple[int, ...] = ()
    # The verbatim HF `vision_config` sub-dict — lets a tower reconstruct the exact
    # upstream config (incl. fields serve doesn't model yet) instead of guessing.
    raw: dict = field(default_factory=dict)

    @property
    def head_dim(self) -> int:
        if self.num_attention_heads <= 0:
            return 0
        return self.hidden_size // self.num_attention_heads


@dataclass
class ModelConfig:
    arch: str  # registry key, e.g. "qwen3"
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    max_position_embeddings: int = 32768
    head_dim: int | None = None  # explicit; else hidden // heads
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6  # global-layer theta
    rope_local_theta: float | None = None  # Gemma local-layer theta (dual-RoPE)
    tie_word_embeddings: bool = False

    # norm / embedding style
    norm_add_unit_offset: bool = False  # Gemma (1+w) RMSNorm
    embed_scale: float | None = None  # Gemma sqrt(hidden) embedding scale
    qk_norm: bool = False  # per-head RMSNorm on Q,K (Qwen3/Gemma3)
    qkv_bias: bool = False  # Qwen2 had it; Qwen3/Gemma3 do not

    # attention scaling / caps
    query_pre_attn_scalar: float | None = None  # Gemma: scale = scalar**-0.5
    attn_logit_softcap: float | None = None  # Gemma2 (removed in Gemma3)
    final_logit_softcap: float | None = None

    # partial rotary (GLM, Qwen3-Next gated attn): rotate only factor*head_dim dims
    partial_rotary_factor: float = 1.0

    # sliding window (Gemma3 local layers, Mistral). pattern P => global every Pth.
    sliding_window: int | None = None
    sliding_window_pattern: int | None = None  # e.g. 6 => layers where (i+1)%6==0 are global

    # hybrid linear-attention backbone (Qwen3-Next / Qwen3.5/3.6, MiniMax lightning):
    # a subset of layers use linear attention; the rest are full softmax attention.
    linear_attention: bool = False  # this family has linear-attn layers
    full_attention_interval: int = 0  # full-attn every Nth layer (0 => none/all)

    # multi-head latent attention (DeepSeek-V2/V3/V4): compressed latent KV
    latent_attention: bool = False

    # MoE (Qwen3-MoE)
    num_experts: int = 0  # 0 => dense
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = True
    shared_expert_intermediate_size: int = 0  # 0 => no shared expert
    decoder_sparse_step: int = 1  # every Nth layer is MoE (else dense)
    mlp_only_layers: tuple[int, ...] = ()  # layer indices forced dense
    use_weight_stationary_moe: bool = False  # Phase 3: per-expert cycling with skip-empty
    # Multi-GPU expert sharding (Phase 5 / #330-333): expert_id -> owning GPU
    # index. When set, MoE layers wrap the base MoE in TransportMoELayer; the
    # owning-GPU index for this process is `local_gpu`. All-local maps degrade
    # to the base MoE (bit-identical) per the transport parity contract.
    expert_to_gpu: dict[int, int] | None = None
    local_gpu: int | None = None

    # MTP (multi-token prediction; Qwen3-Next / DeepSeek-V3 style)
    num_mtp_layers: int = 0

    # decode strategy
    decode_strategy: str = "autoregressive"  # or "diffusion"

    # quantization intent (how weights were stored in the .superl8)
    weight_bits: int = 8

    # activation/residual-stream compute dtype. Mirrors the checkpoint's HF
    # `torch_dtype`. bf16-native models (Gemma3, most Qwen3) carry "massive
    # activation" residual channels that reach ~1e4-1e7 — well past fp16's 65504
    # ceiling — so their forward MUST run the residual stream in bf16 (fp32's
    # exponent range) or it overflows to inf->NaN mid-stack (issue #260, the same
    # fp16-truncation bug class as comfy #115 / serve #256). fp16-native
    # checkpoints stay fp16. Softmax/LSE/norm stay fp32 regardless (never quantized).
    torch_dtype: str = "float16"

    # multimodal vision-language fields (populated from VLM configs only)
    is_multimodal: bool = False
    vision_config: VisionConfig | None = None
    image_token_id: int | None = None

    extra: dict = field(default_factory=dict)  # arch-specific overflow

    def resolved_head_dim(self) -> int:
        return self.head_dim or (self.hidden_size // self.num_attention_heads)

    def mla_cache_dim(self) -> int:
        """Latent-KV cache width for MLA (DeepSeek): kv_lora_rank + qk_rope_head_dim,
        the per-token c_KV + k_pe that decode up-projects and attends over."""
        return self.extra["kv_lora_rank"] + self.extra["qk_rope_head_dim"]

    def is_moe(self) -> bool:
        return self.num_experts > 0

    def act_dtype(self):
        """Torch dtype for the activation/residual stream. bf16 for bf16-native
        checkpoints (their massive-activation channels overflow fp16), else fp16.
        The int8 dp4a matmuls are unaffected (activations are quantized to int8
        regardless); only the store/residual dtype tracks this — perf-neutral on
        this fleet (bf16 elementwise uses the healthy half2 CUDA-core pipe, not the
        dead tensor cores). Softmax/LSE/norm reductions still run in fp32."""
        import torch

        return torch.bfloat16 if str(self.torch_dtype).endswith("bfloat16") else torch.float16

    def layer_is_global(self, layer_idx: int) -> bool:
        """Full-attention layer? True for dense models; for Gemma3 sliding-window
        models, only every `sliding_window_pattern`-th layer is global."""
        if not self.sliding_window or not self.sliding_window_pattern:
            return True
        return (layer_idx + 1) % self.sliding_window_pattern == 0

    def attention_kind(self, layer_idx: int) -> str:
        """Per-layer attention backend selector — the axis that makes hybrids work.
        'linear'  : Gated-DeltaNet / lightning linear attention (Qwen3-Next, MiniMax)
        'latent'  : MLA compressed-KV (DeepSeek)
        'sliding' : local windowed softmax attention (Gemma3 local layers)
        'full'    : standard GQA softmax attention (the superl8 dp4a default)"""
        if self.latent_attention:
            return "latent"
        if self.linear_attention and self.full_attention_interval:
            # full attn every Nth layer, linear (DeltaNet) on the rest
            return "full" if (layer_idx + 1) % self.full_attention_interval == 0 else "linear"
        if self.sliding_window and self.sliding_window_pattern:
            return "full" if self.layer_is_global(layer_idx) else "sliding"
        return "full"

    def rotary_dim(self) -> int:
        return int(self.resolved_head_dim() * self.partial_rotary_factor)

    def layer_is_moe(self, layer_idx: int) -> bool:
        if not self.is_moe() or layer_idx in self.mlp_only_layers:
            return False
        return (layer_idx + 1) % self.decoder_sparse_step == 0

    @classmethod
    def from_hf(cls, hf: dict, *, arch: str | None = None) -> "ModelConfig":
        """Map a HuggingFace text config dict onto ModelConfig. Accepts either a
        top-level text config or one nested under `text_config` (multimodal wrappers
        like Gemma3-it). Some wrappers duplicate a handful of fields at the top level
        (e.g. `hidden_size`) without duplicating the rest (e.g. `num_attention_heads`),
        so we can't gate the dive on a single key's absence — merge `text_config` over
        the top level instead, so its (authoritative) text-model axes always win."""
        c = dict(hf)
        archs = c.get("architectures") or []
        model_type = c.get("model_type", "")
        if c.get("text_config"):
            c = {**c, **dict(c["text_config"])}
        resolved_arch = arch or model_type or (archs[0] if archs else "unknown")
        # Qwen3.5 ships as a single `qwen3_5` / `Qwen3_5ForConditionalGeneration` arch
        # that is a VLM iff a vision tower is configured. Route a raw HF config with a
        # `vision_config` to the multimodal builder; a text-only `.superl8` (whose meta
        # arch is `qwen3_5_text`, no vision_config) stays on the text backbone.
        if resolved_arch in ("qwen3_5", "qwen3.5", "qwen3_5forconditionalgeneration") and hf.get(
            "vision_config"
        ):
            resolved_arch = "qwen3_5_vl"
        n_heads = c["num_attention_heads"]

        # ── hybrid linear-attention derivation (Qwen3-Next / Qwen3.5/3.6, MiniMax) ──
        # A real HF config marks the per-layer hybrid pattern with EITHER a
        # `layer_types` list ("linear_attention"/"full_attention", the Qwen3-Next
        # family) OR an int `attn_type_list` (0=lightning/linear, 1=softmax, MiniMax)
        # — NOT the scalar `linear_attention`/`full_attention_interval` that
        # `attention_kind()` reads (synthetic tests set those by hand; LFM2 reads
        # `layer_types` out of `extra` directly so it never needed this). Derive the
        # scalars here so a REAL checkpoint reproduces the right hybrid layer pattern.
        # `c.get("linear_attention")` is OR-ed in so a round-tripped `.superl8` dump
        # (which already carries the flattened scalar, not `layer_types`) still loads.
        layer_types = c.get("layer_types") or []
        attn_type_list = c.get("attn_type_list") or []
        has_linear = bool(
            c.get("linear_attention", False)
            or ("linear_attention" in layer_types)
            or (0 in attn_type_list)
        )
        full_interval = int(c.get("full_attention_interval", 0) or 0)
        if has_linear and not full_interval:
            full_layers = [i for i, t in enumerate(layer_types) if t == "full_attention"] or [
                i for i, t in enumerate(attn_type_list) if t == 1
            ]
            # attention_kind uses (i+1) % interval == 0, so the first full-attn layer
            # sits at index (interval-1); recover the stride from that first full layer.
            if full_layers:
                full_interval = full_layers[0] + 1

        # Family-default fallback for a round-tripped hybrid whose pattern metadata
        # was lost in conversion (linear_attention=False/interval=0, no layer_types).
        # A known hybrid arch is a DeltaNet/full hybrid by definition, so apply the
        # family's standard 3:1 stride rather than misbuild every layer as full attn.
        if not (has_linear and full_interval) and resolved_arch in _HYBRID_LINEAR_DEFAULT_INTERVAL:
            has_linear = True
            full_interval = _HYBRID_LINEAR_DEFAULT_INTERVAL[resolved_arch]

        # RoPE knobs may be flat (older configs, round-tripped dumps) or nested under
        # `rope_parameters` (Qwen3.5/3.6). Read the nested dict first, else fall back
        # to the flat key — this is where the correct rope_theta (1e7) and the
        # partial-rotary factor (0.25 for the Qwen3-Next gated-attn layers) live.
        rope_params = c.get("rope_parameters") or {}
        rope_theta = rope_params.get("rope_theta", c.get("rope_theta", 1e6))
        partial_rotary = rope_params.get(
            "partial_rotary_factor", c.get("partial_rotary_factor", 1.0)
        )
        # The first fleet Qwen3.6-35B-A3B conversion serialized ModelConfig defaults
        # instead of the nested text_config values. Recover only that identifiable
        # legacy signature; a modern header with rope_parameters remains authoritative.
        legacy_qwen36_moe = resolved_arch == "qwen3_5_moe" and not rope_params
        if legacy_qwen36_moe and rope_theta == 1e6 and partial_rotary == 1.0:
            rope_theta, partial_rotary = 1e7, 0.25

        # Multimodal VLM detection — read vision_config and expose on ModelConfig.
        if resolved_arch in _MULTIMODAL_ARCHS:
            v_raw = hf.get("vision_config") or {}
            img_tok = hf.get("image_token_id") or hf.get("image_token_index")
            n_layers = v_raw.get("depth") or v_raw.get("num_hidden_layers", 0)
            # Qwen3.5 vision uses `num_heads`; Qwen2.5-VL / LLaVA use `num_attention_heads`.
            n_vheads = v_raw.get("num_attention_heads") or v_raw.get("num_heads", 0)
            vision_cfg = (
                VisionConfig(
                    hidden_size=v_raw.get("hidden_size", 0),
                    patch_size=v_raw.get("patch_size", 0),
                    num_layers=n_layers,
                    image_token_id=img_tok or 0,
                    spatial_merge_size=v_raw.get("spatial_merge_size", 1),
                    num_attention_heads=n_vheads,
                    intermediate_size=v_raw.get("intermediate_size", 0),
                    in_channels=v_raw.get("in_channels", v_raw.get("in_chans", 3)),
                    layer_norm_eps=v_raw.get("layer_norm_eps", 1e-6),
                    hidden_act=v_raw.get("hidden_act", "gelu_pytorch_tanh"),
                    temporal_patch_size=v_raw.get("temporal_patch_size", 2),
                    num_position_embeddings=v_raw.get("num_position_embeddings", 0),
                    out_hidden_size=v_raw.get("out_hidden_size", 0),
                    deepstack_visual_indexes=tuple(v_raw.get("deepstack_visual_indexes", []) or []),
                    raw=dict(v_raw),
                )
                if v_raw
                else None
            )
        else:
            vision_cfg = None

        return cls(
            arch=resolved_arch,
            vocab_size=c["vocab_size"],
            hidden_size=c["hidden_size"],
            num_hidden_layers=c["num_hidden_layers"],
            num_attention_heads=n_heads,
            num_key_value_heads=c.get("num_key_value_heads", n_heads),
            intermediate_size=c.get("intermediate_size", 0),
            max_position_embeddings=c.get("max_position_embeddings", 32768),
            head_dim=c.get("head_dim"),
            hidden_act=c.get("hidden_act", "silu"),
            rms_norm_eps=c.get("rms_norm_eps", 1e-6),
            rope_theta=rope_theta,
            rope_local_theta=c.get("rope_local_base_freq"),
            tie_word_embeddings=c.get("tie_word_embeddings", False),
            torch_dtype=str(c.get("torch_dtype", "bfloat16" if legacy_qwen36_moe else "float16")),
            qk_norm=c.get("qk_norm", False),
            qkv_bias=c.get("attention_bias", False),
            partial_rotary_factor=partial_rotary,
            linear_attention=has_linear,
            full_attention_interval=full_interval,
            query_pre_attn_scalar=c.get("query_pre_attn_scalar"),
            attn_logit_softcap=c.get("attn_logit_softcapping"),
            final_logit_softcap=c.get("final_logit_softcapping"),
            sliding_window=c.get("sliding_window"),
            sliding_window_pattern=c.get("sliding_window_pattern"),
            num_experts=c.get("num_experts", 0),
            num_experts_per_tok=c.get("num_experts_per_tok", 0),
            moe_intermediate_size=c.get("moe_intermediate_size", 0),
            norm_topk_prob=c.get("norm_topk_prob", True),
            shared_expert_intermediate_size=c.get("shared_expert_intermediate_size", 0),
            decoder_sparse_step=c.get("decoder_sparse_step", 1),
            mlp_only_layers=tuple(c.get("mlp_only_layers", []) or []),
            use_weight_stationary_moe=c.get("use_weight_stationary_moe", False),
            # expert_to_gpu/local_gpu are serve-time topology overrides, usually
            # injected via `dataclasses.replace` at the API seam, not the HF
            # config. If a round-tripped dump carries them, preserve them.
            expert_to_gpu=c.get("expert_to_gpu"),
            local_gpu=c.get("local_gpu"),
            # MTP depth: DeepSeek/Qwen3-Next spell it `num_nextn_predict_layers`;
            # the Qwen3.5 release uses `mtp_num_hidden_layers` — accept either.
            num_mtp_layers=c.get("num_nextn_predict_layers") or c.get("mtp_num_hidden_layers", 0),
            is_multimodal=vision_cfg is not None,
            vision_config=vision_cfg,
            image_token_id=vision_cfg.image_token_id if vision_cfg else None,
            # arch-specific overflow = every unknown HF key, PLUS a nested `extra`
            # dict if this config is a round-tripped `.superl8` dump (convert.py stores
            # cfg.extra nested so it survives; merge it back, authoritative).
            extra={
                **{k: v for k, v in c.items() if k not in _KNOWN_HF_KEYS and k != "extra"},
                **(c.get("extra") or {}),
            },
        )


_KNOWN_HF_KEYS = {
    "extra",  # nested arch-overflow from a round-tripped .superl8 dump (merged separately)
    "text_config",
    "architectures",
    "model_type",
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "max_position_embeddings",
    "head_dim",
    "hidden_act",
    "rms_norm_eps",
    "rope_theta",
    "rope_parameters",  # nested {rope_theta, partial_rotary_factor} (Qwen3.5/3.6)
    "partial_rotary_factor",
    "full_attention_interval",  # derived into the linear_attention/interval scalars
    "rope_local_base_freq",
    "tie_word_embeddings",
    "torch_dtype",  # -> activation/residual-stream dtype (bf16 overflow guard, #260)
    "attention_bias",
    "qk_norm",
    "query_pre_attn_scalar",
    "attn_logit_softcapping",
    "final_logit_softcapping",
    "sliding_window",
    "sliding_window_pattern",
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "norm_topk_prob",
    "shared_expert_intermediate_size",
    "decoder_sparse_step",
    "mlp_only_layers",
    "use_weight_stationary_moe",
    "expert_to_gpu",
    "local_gpu",
    "num_nextn_predict_layers",
    "mtp_num_hidden_layers",  # Qwen3.5 MTP depth (== num_nextn_predict_layers)
    # VLM / multimodal keys — consumed by from_hf, not leaked into extra
    "vision_config",
    "image_token_id",
    "image_token_index",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
}
