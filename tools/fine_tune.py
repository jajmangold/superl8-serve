#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Minimal fine-tune script: load a .superl8 base, inject LoRA, train N steps (issue #129).

Usage:
    python tools/fine_tune.py --model /path/to/model.superl8 --output ./adapters.superl8

Stops before optimizer + checkpoint save (next rung). Saves adapter weights only.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from superl8serve import build_model, load_superl8_state_dict
from superl8serve.models.config import ModelConfig
from superl8serve.training.lora import LoRAConfig, inject_lora


def _build_fake_data(vocab_size: int, seq_len: int, num_seqs: int, device: str):
    """Random next-token-prediction data."""
    x = torch.randint(0, vocab_size, (num_seqs, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (num_seqs, seq_len), device=device)
    return x, targets


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune on .superl8 base")
    parser.add_argument("--model", required=True, help="Path to .superl8 checkpoint")
    parser.add_argument("--output", default="./adapters", help="Output prefix for adapters")
    parser.add_argument("--steps", type=int, default=10, help="Training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument(
        "--target-modules",
        nargs="*",
        default=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        help="Target module name suffixes",
    )
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()

    torch.set_default_device(args.device)

    info = load_superl8_state_dict(args.model, device="cpu")
    if not info:
        info = {}
    header = {}
    sd = {}

    from superl8 import FQReader

    with FQReader(args.model) as r:
        header.update(r.header)

    sd = load_superl8_state_dict(args.model, device=args.device)

    cfg = ModelConfig(
        arch=header.get("arch", "qwen3"),
        hidden_size=header.get("hidden_size", 2048),
        num_hidden_layers=header.get("num_hidden_layers", 4),
        num_attention_heads=header.get("num_attention_heads", 8),
        num_key_value_heads=header.get("num_key_value_heads", 8),
        head_dim=header.get("head_dim", 128),
        intermediate_size=header.get("intermediate_size", 8192),
        rms_norm_eps=header.get("rms_norm_eps", 1e-6),
        max_position_embeddings=header.get("max_position_embeddings", 1024),
        rope_theta=header.get("rope_theta", 1e6),
        vocab_size=header.get("vocab_size", 32768),
        tie_word_embeddings=header.get("tie_word_embeddings", False),
    )

    model = build_model(cfg, sd)
    model.eval()

    lora_cfg = LoRAConfig(r=args.lora_r, alpha=args.lora_alpha, target_modules=args.target_modules)
    model = inject_lora(model, lora_cfg)

    lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
    print(f"Trainable LoRA parameters: {sum(p.numel() for p in lora_params)}")

    opt = torch.optim.AdamW(lora_params, lr=args.lr)

    vocab_size = cfg.vocab_size
    num_seqs = 2
    seq_len = 64

    for step in range(args.steps):
        x, targets = _build_fake_data(vocab_size, seq_len, num_seqs, args.device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.flatten(0, -2).float(), targets.flatten(0, -1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()
        print(f"step {step:>3d}  loss {loss.item():.4f}")

    adapter_state = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Module) and hasattr(mod, "lora_A"):
            prefix = name.replace(".", "_")
            adapter_state[f"{prefix}.lora_A"] = mod.lora_A.data.cpu().half()
            adapter_state[f"{prefix}.lora_B"] = mod.lora_B.data.cpu().half()

    torch.save(adapter_state, f"{args.output}_adapters.pt")
    print(f"Saved adapter weights to {args.output}_adapters.pt")


if __name__ == "__main__":
    main()
