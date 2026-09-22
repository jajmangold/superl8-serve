# SPDX-License-Identifier: MIT
"""E2E validation of a low-rank PP-boundary wire codec (#184).

Per-boundary SQNR/cosine do NOT predict end-to-end token fidelity for LM PP
boundaries: a low-rank projection's structured (subspace-deleting) error breaks
next-token prediction even at a single boundary, while int8/int4 (bounded
elementwise noise) keep generation coherent. This script measures that directly,
and is the evidence behind ``docs/lowrank-codec-findings.md`` §2–§3.

It installs a codec as a real PP boundary via a forward hook on the boundary
decoder layer (round-trips its hidden-state output), greedy-generates, and
compares to the un-hooked single-GPU reference: logit cosine, top-1 next-token
agreement, and exact-completion count. Includes an identity control (must score
top-1 = 1.0) that proves the hook mechanism is lossless.

Usage (needs a GPU + a cached HF model):
    HF_HUB_OFFLINE=1 python3 bench/lowrank_e2e_probe.py --model Qwen/Qwen3-1.7B-Base
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

_CALIB = (
    "The history of science is gradual discovery. Proofs establish truth. "
    "Economies grow with productivity. The ocean covers the planet. Language "
    "shapes perception. Engineers design bridges. Music combines rhythm and "
    "melody. Cells divide to sustain tissue. Democracy needs informed citizens. "
    "Light travels fast. "
) * 20

_PROMPTS = [
    "The capital of France is",
    "Water is made of hydrogen and",
    "The opposite of hot is",
    "In 1969 humans first landed on the",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--new-tokens", type=int, default=20)
    ap.add_argument("--ranks", type=int, nargs="+", default=[256, 512, 1024, 2047])
    args = ap.parse_args()

    import superl8
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).cuda().eval()
    d = model.config.hidden_size
    k = model.config.num_hidden_layers // 2

    # Calibration: capture the boundary-layer output over calib text, fit an SVD basis.
    cap: list[torch.Tensor] = []

    def _grab(_m, _i, o):
        hs = o[0] if isinstance(o, tuple) else o
        cap.append(hs.detach().float().cpu().reshape(-1, d))

    h = model.model.layers[k].register_forward_hook(_grab)
    cids = tok(_CALIB, return_tensors="pt").input_ids.cuda()[:, :3000]
    with torch.no_grad():
        model(cids)
    h.remove()
    calib = torch.cat(cap, 0)
    mean = calib.mean(0)
    Vt = torch.linalg.svd(calib - mean, full_matrices=False)[2]  # (rank, d), variance-ordered
    mean_g = mean.cuda()

    def make_hook(kind: str, r: int | None = None):
        U = None if r is None else Vt[:r].T.contiguous().cuda()

        def _hook(_m, _i, o):
            hs = o[0] if isinstance(o, tuple) else o
            dt = hs.dtype
            if kind == "identity":
                return o
            if kind == "int8":
                c = superl8.compress_activation(hs.to(torch.float16), scheme="int8")
                hs2 = superl8.decompress_activation(c).to(dt)
            elif kind == "int4-had":
                c = superl8.compress_activation(hs.to(torch.float16), scheme="int4-had", group_size=128)
                hs2 = superl8.decompress_activation(c).to(dt)
            elif kind == "lowrank":  # pure projection through the calibration basis
                hs2 = (((hs.float() - mean_g) @ U) @ U.T + mean_g).to(dt)
            else:
                raise ValueError(kind)
            return (hs2,) + tuple(o[1:]) if isinstance(o, tuple) else hs2

        return _hook

    def generate(cfg):
        handle = None if cfg is None else model.model.layers[k].register_forward_hook(make_hook(*cfg))
        texts, logits = [], []
        with torch.no_grad():
            for p in _PROMPTS:
                ii = tok(p, return_tensors="pt").input_ids.cuda()
                out = model.generate(
                    ii,
                    max_new_tokens=args.new_tokens,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=tok.eos_token_id,
                )
                texts.append(tok.decode(out.sequences[0][ii.shape[1] :], skip_special_tokens=True))
                logits.append(torch.stack(out.scores, 0)[:, 0, :].float().cpu())
        if handle is not None:
            handle.remove()
        return texts, logits

    ref_t, ref_l = generate(None)

    def report(cfg, label):
        t, lg = generate(cfg)
        coss, top1 = [], []
        for a, b in zip(ref_l, lg):
            n = min(a.shape[0], b.shape[0])
            coss.append(float(torch.nn.functional.cosine_similarity(a[:n].flatten(), b[:n].flatten(), dim=0)))
            top1.append(float((a[:n].argmax(-1) == b[:n].argmax(-1)).float().mean()))
        exact = sum(t[i].strip() == ref_t[i].strip() for i in range(len(t)))
        print(f"{label:36s} logit_cos={np.mean(coss):.4f}  top1={np.mean(top1):.3f}  exact={exact}/{len(t)}")

    print(f"model={args.model} d={d} boundary_layer={k}")
    report(("identity",), "CONTROL identity passthrough")
    report(("int8",), "int8 boundary")
    report(("int4-had",), "int4-had boundary (shipped 3.76x)")
    for r in args.ranks:
        report(("lowrank", r), f"low-rank projection r={r} ({d / r:.2f}x)")


if __name__ == "__main__":
    main()
