# SPDX-License-Identifier: MIT
"""End-to-end proof that Qwen3.5-VL vision works on the superl8-serve engine.

Builds the registered `qwen3_5_vl` model (int8 dp4a text backbone + fp16 Qwen3.5
vision tower) from a `.superl8` checkpoint, feeds solid red / green / blue 64x64 images
with "what color is this?", and asserts the answer names the color. Also compares the
vision tower's merged image embeddings against the raw-HF `Qwen3_5ForConditionalGeneration`
oracle (cosine).

Run (real GPU, in the superl8 Docker image):
    python3 -m pip install -e ".[serve]" --no-build-isolation -q
    python3 -m pip install "transformers==5.13.1" pillow -q
    CUDA_VISIBLE_DEVICES=5 python3 scripts/prove_qwen3_5_vl.py \
        --superl8 /models/Qwen3.5-0.8B-superl8/Qwen__Qwen3.5-0.8B.b8.superl8 \
        --tok  /models/Qwen3.5-0.8B-tok \
        --raw  /archives/qwen35b/build-qwen35-0.8b/hf   # optional, for the oracle
"""

from __future__ import annotations

import argparse
import json

import torch
from PIL import Image


def build_vlm(superl8_path: str, tok_dir: str, device: str):
    from superl8serve.loader import checkpoint_info, load_superl8_state_dict
    from superl8serve.models.config import ModelConfig
    from superl8serve.models.registry import build_model

    meta_cfg = dict(checkpoint_info(superl8_path)["meta"]["config"])
    hf_cfg = json.load(open(f"{tok_dir}/config.json"))
    # Inject the vision axes (a text-only `.superl8` meta omits them) and route to the VLM.
    meta_cfg["vision_config"] = hf_cfg["vision_config"]
    meta_cfg["image_token_id"] = hf_cfg["image_token_id"]
    cfg = ModelConfig.from_hf(meta_cfg, arch="qwen3_5_vl")
    assert cfg.arch == "qwen3_5_vl" and cfg.vision_config is not None
    weights = load_superl8_state_dict(superl8_path, device=device)
    model = build_model(cfg, weights).to(device).eval()
    return model, cfg


@torch.inference_mode()
def answer_color(model, proc, color: str, device: str, img_tok: int) -> str:
    from superl8serve.models.runner import ModelRunner

    img = Image.new("RGB", (64, 64), color)
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "What color is this image? Answer in one word."},
    ]}]
    inputs = proc.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(device)
    pv = inputs["pixel_values"].to(device)
    thw = inputs["image_grid_thw"].to(device)
    n_img = int((input_ids == img_tok).sum().item())

    runner = ModelRunner(
        model, model.config, max_batch=1, max_len=input_ids.shape[1] + 16, device=device
    )
    out = runner.generate_greedy(input_ids, 8, pixel_values=pv, image_grid_thw=thw)
    txt = proc.tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
    print(f"[{color}] n_img_tok={n_img} grid={thw.tolist()} -> {txt!r}")
    return txt


@torch.inference_mode()
def oracle_cosine(model, proc, raw_dir: str, device: str) -> float:
    import glob
    import os

    from transformers import Qwen3_5ForConditionalGeneration

    std = "/tmp/hf_std"
    os.makedirs(std, exist_ok=True)
    for f in glob.glob(f"{raw_dir}/*"):
        b = os.path.basename(f).replace(
            "model.safetensors-00001-of-00001.safetensors", "model.safetensors"
        )
        dst = f"{std}/{b}"
        if not os.path.exists(dst):
            os.symlink(f, dst)
    full = Qwen3_5ForConditionalGeneration.from_pretrained(std, dtype=torch.float16).to(device).eval()

    img = Image.new("RGB", (64, 64), "red")
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "x"}]}]
    inp = proc.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    pv = inp["pixel_values"].to(device)
    thw = inp["image_grid_thw"].to(device)
    ref = full.model.get_image_features(pv.half(), image_grid_thw=thw).pooler_output[0]
    ours = model.vision_tower(pv, thw)
    cos = torch.nn.functional.cosine_similarity(ref.float().flatten(), ours.float().flatten(), dim=0)
    return cos.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--superl8", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--raw", default=None, help="raw HF dir for the oracle cosine (optional)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from transformers import AutoProcessor

    model, cfg = build_vlm(args.superl8, args.tok, args.device)
    proc = AutoProcessor.from_pretrained(args.tok)
    img_tok = cfg.image_token_id

    results = {c: answer_color(model, proc, c, args.device, img_tok) for c in ("red", "green", "blue")}
    for color, txt in results.items():
        assert color in txt.lower(), f"FAIL: {color} image -> {txt!r} (expected to contain {color!r})"
    print("PASS: red/green/blue all answered correctly.")

    if args.raw:
        cos = oracle_cosine(model, proc, args.raw, args.device)
        print(f"vision-tower vs raw-HF oracle cosine: {cos:.6f}")
        assert cos > 0.99, f"vision tower cosine {cos} below 0.99"
        print("PASS: vision tower matches the transformers oracle.")


if __name__ == "__main__":
    main()
