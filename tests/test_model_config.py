# SPDX-License-Identifier: MIT
"""ModelConfig.from_hf: plain text configs and multimodal wrappers that nest the
text-model axes under `text_config` (Gemma3-it and friends)."""
import pytest

pytest.importorskip("superl8")   # importing superl8serve.models runs its __init__, which needs superl8

from superl8serve.models import ModelConfig


def _text_cfg(**overrides):
    cfg = dict(
        model_type="gemma3_text", vocab_size=262144, hidden_size=2560,
        num_hidden_layers=34, num_attention_heads=8, num_key_value_heads=4,
        intermediate_size=10240, head_dim=256, query_pre_attn_scalar=256,
        sliding_window=1024, sliding_window_pattern=6,
    )
    cfg.update(overrides)
    return cfg


def test_from_hf_plain_text_config():
    cfg = ModelConfig.from_hf(_text_cfg())
    assert cfg.arch == "gemma3_text"
    assert cfg.num_attention_heads == 8
    assert cfg.num_key_value_heads == 4
    assert cfg.hidden_size == 2560


def test_from_hf_dives_into_text_config_when_top_level_is_bare():
    """The historical case: a multimodal wrapper whose top level has no text-model
    fields at all, only `text_config` + a vision/model-type wrapper."""
    hf = {
        "architectures": ["Gemma3ForConditionalGeneration"],
        "model_type": "gemma3",
        "text_config": _text_cfg(),
        "vision_config": {"hidden_size": 1152, "model_type": "siglip_vision_model"},
    }
    cfg = ModelConfig.from_hf(hf)
    assert cfg.num_attention_heads == 8
    assert cfg.num_key_value_heads == 4
    assert cfg.hidden_size == 2560
    assert cfg.arch == "gemma3"                  # top-level model_type wins for arch


def test_from_hf_merges_text_config_even_when_top_level_partially_duplicates_fields():
    """Regression: some multimodal layouts duplicate a FEW fields at the top level
    (e.g. `hidden_size`, for AutoConfig probing) without duplicating the rest. The old
    guard (`"text_config" in c and "hidden_size" not in c`) skipped the dive whenever
    `hidden_size` leaked to the top level, then KeyError'd on `num_attention_heads`."""
    hf = {
        "architectures": ["Gemma3ForConditionalGeneration"],
        "model_type": "gemma3",
        "hidden_size": 999,                       # stale/partial top-level duplicate
        "text_config": _text_cfg(),
        "vision_config": {"hidden_size": 1152, "model_type": "siglip_vision_model"},
    }
    cfg = ModelConfig.from_hf(hf)
    assert cfg.num_attention_heads == 8           # would KeyError before the fix
    assert cfg.hidden_size == 2560                # text_config wins over the stale 999
    assert cfg.arch == "gemma3"


def test_from_hf_unknown_keys_land_in_extra():
    hf = {"text_config": _text_cfg(foo_bar=123)}
    cfg = ModelConfig.from_hf(hf)
    assert cfg.extra.get("foo_bar") == 123


# ── Multimodal VLM configs ──────────────────────────────────────────────


def _qwen2_5_vl_cfg(**overrides):
    """Simulated Qwen2.5-VL-7B config.json (HF hub shape)."""
    cfg = {
        "model_type": "qwen2_5_vl",
        "architectures": ["Qwen2_5_VLForConditionalGeneration"],
        "vocab_size": 152064,
        "hidden_size": 3584,                           # text hidden_size
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "intermediate_size": 18944,
        "max_position_embeddings": 32768,
        "vision_config": {
            "depth": 32,
            "hidden_size": 3584,
            "patch_size": 14,
            "spatial_merge_size": 2,
        },
        "image_token_id": 151655,
    }
    cfg.update(overrides)
    return cfg


def test_from_hf_qwen2_5_vl():
    """Qwen2.5-VL: is_multimodal=True, vision_config parsed, text axes intact."""
    hf = _qwen2_5_vl_cfg()
    cfg = ModelConfig.from_hf(hf)
    # Multimodal fields
    assert cfg.is_multimodal is True
    assert cfg.vision_config is not None
    assert cfg.vision_config.hidden_size == 3584
    assert cfg.vision_config.patch_size == 14
    assert cfg.vision_config.num_layers == 32
    assert cfg.vision_config.image_token_id == 151655
    assert cfg.vision_config.spatial_merge_size == 2
    assert cfg.image_token_id == 151655
    # Text-model axes (from text_config merge or top-level fallback)
    assert cfg.arch == "qwen2_5_vl"
    assert cfg.vocab_size == 152064
    assert cfg.hidden_size == 3584
    assert cfg.num_hidden_layers == 28
    assert cfg.num_attention_heads == 28
    assert cfg.num_key_value_heads == 4
    # VLM keys are consumed, not leaked into extra
    assert "vision_config" not in cfg.extra
    assert "image_token_id" not in cfg.extra


def test_from_hf_qwen2_5_vl_uses_text_config_subdict():
    """Qwen2.5-VL config that nests text axes under text_config (HF hub style)."""
    text = {
        "vocab_size": 152064, "hidden_size": 3584, "num_hidden_layers": 28,
        "num_attention_heads": 28, "num_key_value_heads": 4,
        "intermediate_size": 18944, "max_position_embeddings": 32768,
    }
    hf = {
        "model_type": "qwen2_5_vl",
        "architectures": ["Qwen2_5_VLForConditionalGeneration"],
        "text_config": text,
        "vision_config": {
            "depth": 32, "hidden_size": 3584, "patch_size": 14,
            "spatial_merge_size": 2,
        },
        "image_token_id": 151655,
    }
    cfg = ModelConfig.from_hf(hf)
    assert cfg.is_multimodal is True
    assert cfg.vision_config.hidden_size == 3584
    assert cfg.vision_config.num_layers == 32
    assert cfg.hidden_size == 3584
    assert cfg.num_hidden_layers == 28
    assert cfg.image_token_id == 151655


def test_from_hf_llava():
    """LLaVA: uses num_hidden_layers (not depth), image_token_index mapped."""
    text = {
        "vocab_size": 32000, "hidden_size": 4096, "num_hidden_layers": 32,
        "num_attention_heads": 32, "num_key_value_heads": 8,
        "intermediate_size": 11008,
    }
    hf = {
        "model_type": "llava",
        "architectures": ["LlavaForConditionalGeneration"],
        "text_config": text,
        "vision_config": {
            "hidden_size": 1024, "patch_size": 14, "num_hidden_layers": 24,
        },
        "image_token_index": 32001,
    }
    cfg = ModelConfig.from_hf(hf)
    assert cfg.is_multimodal is True
    assert cfg.vision_config is not None
    assert cfg.vision_config.hidden_size == 1024
    assert cfg.vision_config.patch_size == 14
    assert cfg.vision_config.num_layers == 24
    assert cfg.vision_config.image_token_id == 32001
    assert cfg.vision_config.spatial_merge_size == 1   # default for LLaVA
    assert cfg.image_token_id == 32001
    # Text axes intact
    assert cfg.arch == "llava"
    assert cfg.hidden_size == 4096
    assert cfg.num_hidden_layers == 32


def test_from_hf_llava_next():
    """LLaVA-Next: same pattern as LLaVA."""
    text = {
        "vocab_size": 32000, "hidden_size": 4096, "num_hidden_layers": 32,
        "num_attention_heads": 32, "num_key_value_heads": 8,
        "intermediate_size": 11008,
    }
    hf = {
        "model_type": "llava_next",
        "architectures": ["LlavaNextForConditionalGeneration"],
        "text_config": text,
        "vision_config": {
            "hidden_size": 1024, "patch_size": 14, "num_hidden_layers": 24,
        },
        "image_token_index": 32001,
    }
    cfg = ModelConfig.from_hf(hf)
    assert cfg.is_multimodal is True
    assert cfg.vision_config.num_layers == 24
    assert cfg.vision_config.spatial_merge_size == 1
    assert cfg.image_token_id == 32001


def _qwen3_5_vl_cfg(**overrides):
    """Synthetic Qwen3.5-VL config.json: top-level VLM wrapper with nested
    text_config (hybrid Gated-DeltaNet) + vision_config + MTP head."""
    n_layers = 24
    interval = 4
    lt = ["full_attention" if (i + 1) % interval == 0 else "linear_attention"
          for i in range(n_layers)]
    text = dict(
        vocab_size=248320, hidden_size=1024,
        num_hidden_layers=n_layers, num_attention_heads=8, num_key_value_heads=2,
        intermediate_size=3584, head_dim=256, full_attention_interval=interval,
        layer_types=lt, linear_num_key_heads=16, linear_num_value_heads=16,
        linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
        rope_parameters={"rope_theta": 10000000, "partial_rotary_factor": 0.25,
                         "rope_type": "default"},
        num_nextn_predict_layers=2,
    )
    hf = dict(
        model_type="qwen3_5_vl",
        architectures=["Qwen3_5VLForConditionalGeneration"],
        text_config=text,
        vision_config={
            "depth": 32,
            "hidden_size": 3584,
            "patch_size": 14,
            "spatial_merge_size": 2,
        },
        image_token_id=151655,
    )
    hf.update(overrides)
    return hf


def test_from_hf_qwen3_5_vl():
    """Qwen3.5-VL: is_multimodal=True, vision axes parsed, hybrid linear-attention
    derived from layer_types, MTP presence registered, text axes intact."""
    hf = _qwen3_5_vl_cfg()
    cfg = ModelConfig.from_hf(hf)
    # Multimodal fields
    assert cfg.is_multimodal is True
    assert cfg.vision_config is not None
    assert cfg.vision_config.hidden_size == 3584
    assert cfg.vision_config.patch_size == 14
    assert cfg.vision_config.num_layers == 32
    assert cfg.vision_config.spatial_merge_size == 2
    assert cfg.image_token_id == 151655
    # Text axes
    assert cfg.arch == "qwen3_5_vl"
    assert cfg.vocab_size == 248320
    assert cfg.hidden_size == 1024
    assert cfg.num_hidden_layers == 24
    assert cfg.num_attention_heads == 8
    assert cfg.num_key_value_heads == 2
    # Hybrid linear attention derived from layer_types
    assert cfg.linear_attention is True
    assert cfg.full_attention_interval == 4
    # MTP
    assert cfg.num_mtp_layers == 2
    # Nested rope
    assert cfg.rope_theta == 10000000
    assert abs(cfg.partial_rotary_factor - 0.25) < 1e-9
    # Linear head dims survive into extra
    assert cfg.extra["linear_num_key_heads"] == 16
    # attention_kind must match the source layer_types exactly
    lt = _qwen3_5_vl_cfg()["text_config"]["layer_types"]
    for i, t in enumerate(lt):
        want = "full" if t == "full_attention" else "linear"
        assert cfg.attention_kind(i) == want, (i, t, cfg.attention_kind(i))
    # VLM keys consumed, not leaked into extra
    assert "vision_config" not in cfg.extra
    assert "image_token_id" not in cfg.extra


def test_from_hf_qwen3_5_vl_roundtrips_scalars_when_text_config_absent():
    """A round-tripped .superl8 dump (flat metadata, no text_config sub-dict) still
    recovers the hybrid scalars + vision cfg when the top-level fields are set."""
    hf = _qwen3_5_vl_cfg()
    # Flatten the text_config into the top level (as a real .superl8 dump would)
    text = hf.pop("text_config")
    for k, v in text.items():
        hf[k] = v
    cfg = ModelConfig.from_hf(hf, arch="qwen3_5_vl")
    assert cfg.is_multimodal is True
    assert cfg.vision_config is not None
    assert cfg.linear_attention is True
    assert cfg.full_attention_interval == 4
    assert cfg.num_mtp_layers == 2


def test_from_hf_text_only_configs_unaffected():
    """Existing text-only configs still parse with is_multimodal=False."""
    cfg = ModelConfig.from_hf(_text_cfg())
    assert cfg.is_multimodal is False
    assert cfg.vision_config is None
    assert cfg.image_token_id is None

    # Gemma3 wrapper (text_model has vision_config but arch isn't VLM)
    hf = {
        "architectures": ["Gemma3ForConditionalGeneration"],
        "model_type": "gemma3",
        "text_config": _text_cfg(),
        "vision_config": {"hidden_size": 1152, "model_type": "siglip_vision_model"},
    }
    cfg = ModelConfig.from_hf(hf)
    assert cfg.is_multimodal is False     # gemma3 is not in _MULTIMODAL_ARCHS
    assert cfg.vision_config is None
    assert cfg.num_attention_heads == 8   # text axes still work


# ── hybrid linear-attention derivation (Qwen3-Next / Qwen3.5/3.6, MiniMax) ──────
# A real HF config declares the hybrid layer pattern with `layer_types` /
# `attn_type_list`, NOT the scalar linear_attention/full_attention_interval that
# attention_kind() reads. from_hf must derive those (the #197 real-loading gate).


def _qwen3_5_text_cfg(interval=4, n_layers=24, **ov):
    """The text_config sub-dict of a real Qwen3.5 config.json (values from the
    on-box Qwen3.5-0.8B checkpoint)."""
    lt = ["full_attention" if (i + 1) % interval == 0 else "linear_attention"
          for i in range(n_layers)]
    cfg = dict(
        model_type="qwen3_5_text", vocab_size=248320, hidden_size=1024,
        num_hidden_layers=n_layers, num_attention_heads=8, num_key_value_heads=2,
        intermediate_size=3584, head_dim=256, full_attention_interval=interval,
        layer_types=lt, linear_num_key_heads=16, linear_num_value_heads=16,
        linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
        rope_parameters={"rope_theta": 10000000, "partial_rotary_factor": 0.25,
                         "rope_type": "default"},
    )
    cfg.update(ov)
    return cfg


def test_from_hf_qwen3_5_derives_hybrid_pattern():
    """Real Qwen3.5 wrapper (arch qwen3_5, text axes nested) -> the scalar flags +
    nested rope are derived so attention_kind() reproduces the layer_types list."""
    hf = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
          "text_config": _qwen3_5_text_cfg()}
    cfg = ModelConfig.from_hf(hf, arch="qwen3_next")
    assert cfg.linear_attention is True
    assert cfg.full_attention_interval == 4
    # nested rope must win over the 1e6 default (gated-attn layers need theta 1e7 + 0.25)
    assert cfg.rope_theta == 10000000
    assert abs(cfg.partial_rotary_factor - 0.25) < 1e-9
    # attention_kind must match the source layer_types exactly
    lt = _qwen3_5_text_cfg()["layer_types"]
    for i, t in enumerate(lt):
        want = "full" if t == "full_attention" else "linear"
        assert cfg.attention_kind(i) == want, (i, t, cfg.attention_kind(i))
    # linear head dims survive into extra (the builder reads them there)
    assert cfg.extra["linear_num_key_heads"] == 16


def test_from_hf_derives_interval_when_only_layer_types_present():
    """If full_attention_interval is absent, recover the stride from layer_types."""
    tc = _qwen3_5_text_cfg()
    tc.pop("full_attention_interval")
    cfg = ModelConfig.from_hf({"model_type": "qwen3_5", "text_config": tc})
    assert cfg.linear_attention is True
    assert cfg.full_attention_interval == 4


def test_from_hf_minimax_attn_type_list():
    """MiniMax marks lightning(0)/softmax(1) per layer via attn_type_list; the scalar
    linear_attention flag is derived, and attn_type_list survives into extra (the
    MiniMax builder reads it there for the per-layer decision)."""
    atl = [0, 0, 0, 0, 0, 0, 0, 1] * 2  # 7:1 lightning:softmax, 16 layers
    hf = dict(model_type="minimax_text_01", vocab_size=32000, hidden_size=1024,
              num_hidden_layers=16, num_attention_heads=8, num_key_value_heads=8,
              intermediate_size=4096, attn_type_list=atl, num_experts=8,
              num_experts_per_tok=2)
    cfg = ModelConfig.from_hf(hf)
    assert cfg.linear_attention is True
    assert cfg.full_attention_interval == 8   # first softmax(1) at index 7 -> stride 8
    assert cfg.extra["attn_type_list"] == atl


def test_from_hf_roundtrip_dump_preserves_scalar_flags():
    """A round-tripped .superl8 dump carries the FLATTENED scalars (linear_attention
    bool, full_attention_interval int) but NOT layer_types — from_hf must keep the
    stored scalars instead of resetting them to False/0 (the server load path)."""
    dump = dict(arch="qwen3_next", model_type="qwen3_next", vocab_size=248320,
                hidden_size=1024, num_hidden_layers=24, num_attention_heads=8,
                num_key_value_heads=2, intermediate_size=3584, head_dim=256,
                linear_attention=True, full_attention_interval=4,
                partial_rotary_factor=0.25, rope_theta=10000000)
    cfg = ModelConfig.from_hf(dump, arch="qwen3_next")
    assert cfg.linear_attention is True
    assert cfg.full_attention_interval == 4
    assert abs(cfg.partial_rotary_factor - 0.25) < 1e-9
    assert cfg.rope_theta == 10000000


def test_from_hf_dense_model_unaffected_by_derivation():
    """A plain dense config gets linear_attention=False / interval=0 and every layer
    is 'full' — the derivation must not fire without layer_types/attn_type_list."""
    cfg = ModelConfig.from_hf(_text_cfg(sliding_window=None, sliding_window_pattern=None))
    assert cfg.linear_attention is False
    assert cfg.full_attention_interval == 0
    assert cfg.attention_kind(0) == "full"


def _roundtripped_qwen35_meta(**overrides):
    """A .superl8 round-tripped meta config for Qwen3.5: it carries the flattened
    scalars (NOT `layer_types`/`attn_type_list`), and older converters recorded
    the hybrid pattern as the dataclass defaults (linear_attention=False,
    full_attention_interval=0) — losing the 3:1 DeltaNet/full layout."""
    cfg = dict(
        arch="qwen3_5_text", model_type=None, vocab_size=151936, hidden_size=1024,
        num_hidden_layers=24, num_attention_heads=16, num_key_value_heads=2,
        intermediate_size=3072, head_dim=128,
        linear_attention=False, full_attention_interval=0,  # the lossy round-trip
    )
    cfg.update(overrides)
    return cfg


def test_qwen35_roundtripped_meta_recovers_hybrid_pattern():
    """Regression: a round-tripped Qwen3.5 .superl8 whose meta lost the hybrid pattern
    must still classify layers 3:1 (DeltaNet linear + every-4th full), not all-full.
    Pre-fix this returned 'full' for every layer -> KeyError on q_proj at load."""
    cfg = ModelConfig.from_hf(_roundtripped_qwen35_meta(), arch="qwen3_5_text")
    kinds = [cfg.attention_kind(i) for i in range(cfg.num_hidden_layers)]
    # 3 linear : 1 full, full every 4th layer (indices 3,7,11,...).
    assert kinds[0] == "linear" and kinds[1] == "linear" and kinds[2] == "linear"
    assert kinds[3] == "full" and kinds[7] == "full"
    assert kinds.count("full") == 6 and kinds.count("linear") == 18


def test_real_layer_types_still_wins_over_family_default():
    """If the config DOES carry layer_types, it must be honored, not overridden
    by the family default."""
    lt = ["linear_attention"] * 24
    lt[5] = lt[11] = lt[17] = lt[23] = "full_attention"  # a non-default stride-6 layout
    cfg = ModelConfig.from_hf(_roundtripped_qwen35_meta(layer_types=lt), arch="qwen3_5_text")
    assert cfg.full_attention_interval == 6
    assert cfg.attention_kind(5) == "full" and cfg.attention_kind(3) == "linear"
