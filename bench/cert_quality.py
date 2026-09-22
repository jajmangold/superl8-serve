# SPDX-License-Identifier: MIT
"""cert_quality — perplexity + top-1 next-token accuracy for a `.superl8` checkpoint,
measured through the REAL superl8-serve int8/int4 engine forward (not a fakequant
simulation) on a fixed multi-passage text set.

Reference note (honest): a co-located fp16 HF reference for the deployed Qwen3.5-9B
is NOT available on this box — the raw fp16 safetensors are not materialized
locally (xet blobs absent, HF hub offline) and a 9B fp16 forward (~18 GB) does not
fit a 16 GB V100. So for the 9B we anchor against the near-lossless 8-bit (`b8`)
checkpoint run through the SAME engine path: the `b4 - b8` delta is the on-box
fidelity cost of 4-bit weights. For configs that DO have an fp16 anchor (the 0.8B),
use tools/fakequant_ppl.py instead.

Usage (inside the superl8 container, pinned to a free Volta):
    python3 bench/cert_quality.py \
        --ckpt /path/to/weights/Qwen__Qwen3.5-9B.b4.superl8 \
        --ref  /path/to/weights/Qwen__Qwen3.5-9B.b8.superl8 \
        --tokenizer /path/to/Qwen3.5-0.8B-tok
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer

from superl8serve.engine import LLMEngine
from superl8serve.loader import checkpoint_info, load_superl8_state_dict
from superl8serve.models import ModelConfig
from superl8serve.models.base import ForwardContext

# Fixed evaluation set — general-knowledge / expository prose, deterministic.
PASSAGES = [
    ("The transformer architecture, introduced in 2017, replaced recurrent networks for most "
     "sequence modeling tasks. Its core mechanism, self-attention, lets every token attend to "
     "every other token in the sequence. The capital of France is Paris, and water boils at one "
     "hundred degrees Celsius at sea level. Photosynthesis converts sunlight, water, and carbon "
     "dioxide into glucose and oxygen inside the chloroplasts of green plants."),
    ("Perplexity measures how well a probability model predicts a sample; a lower perplexity "
     "indicates the model is better at predicting the next token in a held-out sequence of "
     "natural language text. Quantizing a neural network to eight-bit integers reduces its memory "
     "footprint and can accelerate inference on hardware that lacks fast half-precision units, "
     "at the cost of a small, measurable increase in output error."),
    ("The mitochondria is the powerhouse of the cell, generating adenosine triphosphate through "
     "oxidative phosphorylation. In economics, supply and demand determine the market price of a "
     "good: when demand rises while supply is fixed, the price tends to increase. The speed of "
     "light in a vacuum is approximately three hundred thousand kilometres per second, a universal "
     "constant that appears throughout modern physics."),
    ("A binary search algorithm repeatedly halves a sorted array to locate a target value in "
     "logarithmic time. The Great Barrier Reef, off the coast of Australia, is the largest living "
     "structure on Earth and is visible from space. Shakespeare wrote thirty-seven plays, including "
     "tragedies such as Hamlet and Macbeth, and comedies such as Twelfth Night. Rain forms when "
     "water vapour in rising air cools, condenses onto particles, and falls as droplets."),
]


def load_engine(path: str, device: str = "cuda") -> tuple[LLMEngine, ModelConfig]:
    meta = checkpoint_info(path)["meta"]["config"]
    cfg = ModelConfig.from_hf(meta, arch=meta.get("arch") or None)
    w = load_superl8_state_dict(path, device=device)
    if not getattr(cfg, "qk_norm", False) and any(".q_norm.weight" in n for n in w):
        try:
            cfg.qk_norm = True
        except Exception:
            cfg = dataclasses.replace(cfg, qk_norm=True)
    eng = LLMEngine(cfg, w, device=device, max_num_seqs=2, max_len=4096)
    return eng, cfg


def score(eng: LLMEngine, ids: list[int]) -> tuple[float, float, int]:
    """Return (sum_nll, correct_top1, n_targets) for one passage via a single
    prefill forward through the real engine model."""
    r = eng.runner
    cache = r.cache
    n = len(ids)
    cache.ensure_capacity([0], [n])
    r.lin_cache.bind([0])
    x = torch.tensor([ids], device="cuda")
    pos = torch.arange(n, device="cuda").unsqueeze(0)
    ctx = ForwardContext(is_prefill=True, kv_cache=cache, lin_cache=r.lin_cache,
                         slots=[0], prefill_start=0)
    with torch.inference_mode():
        h = r.model(x, pos, ctx)
        lg = r.model.compute_logits(h)  # [1, T, V]
    lp = torch.log_softmax(lg[0, :-1].float(), dim=-1)
    tgt = torch.tensor(ids[1:], device=lp.device)
    nll = -lp[torch.arange(len(tgt)), tgt]
    correct = int((lg[0, :-1].argmax(-1) == tgt).sum())
    return float(nll.sum()), correct, len(tgt)


def evaluate(path: str, tokenizer, device: str = "cuda") -> dict:
    eng, cfg = load_engine(path, device)
    tok = tokenizer
    tot_nll = 0.0
    tot_correct = 0
    tot_n = 0
    per_passage = []
    for p in PASSAGES:
        ids = tok.encode(p)
        s_nll, corr, n = score(eng, ids)
        tot_nll += s_nll
        tot_correct += corr
        tot_n += n
        per_passage.append({"tokens": len(ids), "ppl": float(torch.exp(torch.tensor(s_nll / n))),
                            "top1": corr / n})
    import math
    return {
        "ckpt": Path(path).name,
        "weight_bits": cfg.weight_bits,
        "n_tokens_scored": tot_n,
        "ppl": math.exp(tot_nll / tot_n),
        "top1_acc": tot_correct / tot_n,
        "per_passage": per_passage,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint under test (e.g. the deployed b4)")
    ap.add_argument("--ref", default=None, help="high-fidelity reference checkpoint (e.g. b8)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    out = {"under_test": evaluate(a.ckpt, tok)}
    print("[under-test]", json.dumps(out["under_test"], indent=2))
    if a.ref:
        # free the first engine's GPU memory before loading the reference
        torch.cuda.empty_cache()
        out["reference"] = evaluate(a.ref, tok)
        print("[reference ]", json.dumps(out["reference"], indent=2))
        u, rf = out["under_test"], out["reference"]
        out["delta_vs_ref"] = {
            "ref": rf["ckpt"],
            "ppl_pct": (u["ppl"] - rf["ppl"]) / rf["ppl"] * 100.0,
            "top1_pts": (u["top1_acc"] - rf["top1_acc"]) * 100.0,
        }
        print("[delta     ]", json.dumps(out["delta_vs_ref"], indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
