# SPDX-License-Identifier: MIT
"""Qwen3.5-VL bring-up: config routing + vision-tower composition.

Three tiers:
  * CPU-only: `from_hf` routes a `qwen3_5` config-with-vision to the `qwen3_5_vl`
    builder and parses the Qwen3.5 vision axes; the registry claims the arch; the
    `visual.*` weight slicer works.
  * GPU + checkpoint (skipped otherwise): build the real `qwen3_5_vl` model from the
    shipped Qwen3.5-0.8B `.superl8` and prove a solid-red / green / blue image is named
    correctly end-to-end, and that the fp16 tower matches the transformers oracle.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("superl8")

import torch

from superl8serve.models.config import ModelConfig
from superl8serve.models.registry import is_supported

CUDA = torch.cuda.is_available()
SUPERL8 = os.environ.get(
    "QWEN35_VL_SUPERL8", "/models/Qwen3.5-0.8B-superl8/Qwen__Qwen3.5-0.8B.b8.superl8"
)
TOK = os.environ.get("QWEN35_VL_TOK", "/models/Qwen3.5-0.8B-tok")
RAW = os.environ.get("QWEN35_VL_RAW", "/archives/qwen35b/build-qwen35-0.8b/hf")


def _mini_qwen3_5_vl_config() -> dict:
    """A minimal Qwen3.5 HF config dict carrying a `vision_config` — enough to
    exercise `from_hf`'s multimodal routing without any weights."""
    return {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "image_token_id": 248056,
        "vocab_size": 248320,
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "intermediate_size": 3584,
        "vision_config": {
            "depth": 12,
            "hidden_size": 768,
            "num_heads": 12,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "intermediate_size": 3072,
            "num_position_embeddings": 2304,
            "out_hidden_size": 1024,
            "in_channels": 3,
        },
    }


# ── CPU-only: config routing + registry ──────────────────────────────────


def test_registry_claims_qwen3_5_vl():
    assert is_supported("qwen3_5_vl")
    # The raw HF VLM arch routes to the multimodal builder (not the text backbone).
    assert is_supported("Qwen3_5ForConditionalGeneration")


def test_from_hf_routes_and_parses_vision():
    cfg = ModelConfig.from_hf(_mini_qwen3_5_vl_config())
    assert cfg.arch == "qwen3_5_vl"
    assert cfg.is_multimodal and cfg.vision_config is not None
    v = cfg.vision_config
    assert v.num_layers == 12
    assert v.num_attention_heads == 12  # parsed from `num_heads`
    assert v.patch_size == 16
    assert v.spatial_merge_size == 2
    assert v.temporal_patch_size == 2
    assert v.num_position_embeddings == 2304
    assert v.out_hidden_size == 1024
    assert cfg.image_token_id == 248056
    # The text backbone must still reconstruct as the DeltaNet/full hybrid.
    assert cfg.linear_attention and cfg.full_attention_interval == 4
    # verbatim vision sub-dict preserved for exact tower reconstruction
    assert cfg.vision_config.raw["hidden_size"] == 768


def test_text_only_meta_stays_text():
    """A text-only `.superl8` meta (arch `qwen3_5_text`, no vision_config) must NOT be
    rerouted to the VLM builder."""
    cfg = ModelConfig.from_hf(
        {
            "model_type": "qwen3_5_text",
            "vocab_size": 100,
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
        },
        arch="qwen3_5_text",
    )
    assert cfg.arch == "qwen3_5_text"
    assert not cfg.is_multimodal


def test_strip_visual_prefix():
    from superl8serve.multimodal.qwen3_5_vit import _strip_visual_prefix

    sd = {
        "model.visual.patch_embed.proj.weight": torch.zeros(2),
        "visual.blocks.0.norm1.weight": torch.zeros(2),
        "model.language_model.layers.0.input_layernorm.weight": torch.zeros(2),
    }
    out = _strip_visual_prefix(sd)
    assert set(out) == {"patch_embed.proj.weight", "blocks.0.norm1.weight"}


# ── GPU + checkpoint: real end-to-end proof ───────────────────────────────

_have_ckpt = os.path.exists(SUPERL8) and os.path.exists(f"{TOK}/config.json")
gpu_ckpt = pytest.mark.skipif(
    not (CUDA and _have_ckpt), reason="needs CUDA + Qwen3.5-0.8B .superl8 + tokenizer dir"
)


@pytest.fixture(scope="module")
def vlm_and_proc():
    import json

    pytest.importorskip("transformers")
    pytest.importorskip("PIL")
    from transformers import AutoProcessor

    from superl8serve.loader import checkpoint_info, load_superl8_state_dict
    from superl8serve.models.registry import build_model

    meta_cfg = dict(checkpoint_info(SUPERL8)["meta"]["config"])
    hf_cfg = json.load(open(f"{TOK}/config.json"))
    meta_cfg["vision_config"] = hf_cfg["vision_config"]
    meta_cfg["image_token_id"] = hf_cfg["image_token_id"]
    cfg = ModelConfig.from_hf(meta_cfg, arch="qwen3_5_vl")
    weights = load_superl8_state_dict(SUPERL8, device="cuda")
    model = build_model(cfg, weights).to("cuda").eval()
    proc = AutoProcessor.from_pretrained(TOK)
    return model, proc, cfg


@gpu_ckpt
@pytest.mark.parametrize("color", ["red", "green", "blue"])
def test_color_named_correctly(vlm_and_proc, color):
    from PIL import Image

    from superl8serve.models.runner import ModelRunner

    model, proc, cfg = vlm_and_proc
    img = Image.new("RGB", (64, 64), color)
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "What color is this image? Answer in one word."},
    ]}]
    inputs = proc.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    input_ids = inputs["input_ids"].cuda()
    pv = inputs["pixel_values"].cuda()
    thw = inputs["image_grid_thw"].cuda()
    runner = ModelRunner(model, cfg, max_batch=1, max_len=input_ids.shape[1] + 16, device="cuda")
    out = runner.generate_greedy(input_ids, 8, pixel_values=pv, image_grid_thw=thw)
    txt = proc.tokenizer.decode(out[0].tolist(), skip_special_tokens=True).lower()
    assert color in txt, f"{color} image -> {txt!r}"


@gpu_ckpt
@pytest.mark.skipif(not os.path.exists(RAW), reason="needs raw HF weights for the oracle")
def test_vision_tower_matches_oracle(vlm_and_proc):
    import glob

    from PIL import Image
    from transformers import Qwen3_5ForConditionalGeneration

    model, proc, cfg = vlm_and_proc
    std = "/tmp/hf_std_oracle"
    os.makedirs(std, exist_ok=True)
    for f in glob.glob(f"{RAW}/*"):
        b = os.path.basename(f).replace(
            "model.safetensors-00001-of-00001.safetensors", "model.safetensors"
        )
        dst = f"{std}/{b}"
        if not os.path.exists(dst):
            os.symlink(f, dst)
    full = Qwen3_5ForConditionalGeneration.from_pretrained(std, dtype=torch.float16).cuda().eval()

    img = Image.new("RGB", (64, 64), "red")
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "x"}]}]
    inp = proc.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    pv = inp["pixel_values"].cuda()
    thw = inp["image_grid_thw"].cuda()
    with torch.inference_mode():
        ref = full.model.get_image_features(pv.half(), image_grid_thw=thw).pooler_output[0]
        ours = model.vision_tower(pv, thw)
    cos = torch.nn.functional.cosine_similarity(ref.float().flatten(), ours.float().flatten(), dim=0)
    assert cos.item() > 0.99, f"tower vs oracle cosine {cos.item()}"
