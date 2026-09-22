# SPDX-License-Identifier: MIT
"""Measure the low-rank wire codec on REAL transformer boundary activations (#184).

Loads a small causal LM, captures a residual-stream boundary tensor
(``hidden_states[L//2]`` — a representative 2-way PP split point), fits a
calibration SVD basis on 70 % of the tokens, and reports reconstruction cos /
SQNR on the held-out 30 % across a sweep of ranks. Also reports the plain
per-row int8 baseline (the transport doctrine default) for comparison.

This is the evidence behind ``docs/lowrank-codec-negative-result.md``: low-rank
projection cannot reach the near-lossless bar (35 dB / 0.999 cos) at a
compressive rank, and is dominated by int8 on the accuracy-per-byte frontier.

Usage (needs a GPU + a cached HF model):
    HF_HUB_OFFLINE=1 python3 bench/lowrank_activation_probe.py --model Qwen/Qwen3-1.7B-Base
"""

from __future__ import annotations

import argparse

import torch

from superl8serve.dist.lowrank import LowRankCodec, fit_basis

_PROSE = (
    "The theory of relativity reshaped space and time. Distributed consensus "
    "protocols like Raft ensure replicas agree despite failures. Photosynthesis "
    "converts light into chemical energy in chloroplasts. Markets reacted after "
    "the central bank raised rates. Recursive algorithms need careful base cases. "
    "Maritime trade routes connected civilizations across oceans. The immune "
    "system distinguishes self from non-self. Quantum entanglement links particle "
    "states. Compilers translate source through lexing and parsing. Tectonic "
    "plates drift to raise mountains. "
) * 6


def _cos_sqnr(x: torch.Tensor, xr: torch.Tensor) -> tuple[float, float]:
    xf = x.float().flatten()
    xrf = xr.float().flatten()
    cos = float(torch.nn.functional.cosine_similarity(xf, xrf, dim=0, eps=1e-12))
    noise = (xf - xrf).pow(2).sum()
    signal = xf.pow(2).sum()
    return cos, float(10.0 * torch.log10(signal / noise))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument(
        "--ranks", type=int, nargs="+", default=[128, 160, 224, 288, 352, 400, 512]
    )
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16, output_hidden_states=True
        )
        .cuda()
        .eval()
    )
    ids = tok(_PROSE, return_tensors="pt").input_ids.cuda()[:, : args.max_tokens]
    with torch.no_grad():
        out = model(ids)

    n_layers = len(out.hidden_states)
    d = out.hidden_states[0].shape[-1]
    layer = n_layers // 2
    X = out.hidden_states[layer].squeeze(0).float().cpu()
    T = X.shape[0]
    cal, test = X[: int(T * 0.7)], X[int(T * 0.7) :]

    xc = X - X.mean(0)
    sv = torch.linalg.svdvals(xc)
    cum = torch.cumsum(sv**2, 0) / (sv**2).sum()
    r99 = int((cum < 0.99).sum()) + 1
    r9999 = int((cum < 0.9999).sum()) + 1

    print(f"model={args.model}  d={d}  layer={layer}/{n_layers - 1}  tokens={T}")
    print(f"centered energy rank: 99%={r99}  99.99%={r9999}  (of d={d})")

    import superl8

    xt = test.half().cuda()

    # Shipped superl8 codecs on the real boundary activation.
    print("\n# shipped codecs")
    shipped = []  # (name, ratio, sqnr, cos)
    for sch, gs in [("int8", None), ("int4", 128), ("int4-had", 128), ("nf4", 128)]:
        c = superl8.compress_activation(xt, scheme=sch, group_size=gs)
        cos, sqnr = _cos_sqnr(xt, superl8.decompress_activation(c))
        wire = c.payload.numel() * c.payload.element_size() + c.scales.numel() * c.scales.element_size()
        ratio = (xt.numel() * 2) / wire
        shipped.append((sch, ratio, sqnr, cos))
        print(f"  {sch:9s} ratio={ratio:5.2f}x  sqnr={sqnr:5.1f}dB  cos={cos:.4f}")

    # Low-rank sweep (int8 latent). ratio = 2d / (r + n_raw*2); basis is amortized.
    print("\n# low-rank sweep (int8 latent, 0.5% raw)")
    lr = []  # (r, ratio, sqnr, cos)
    for r in args.ranks:
        if r >= min(cal.shape[0], d):
            continue
        basis = fit_basis(cal, r=r, raw_fraction=0.005)
        cos, sqnr = _cos_sqnr(xt, LowRankCodec(basis).decode(LowRankCodec(basis).encode(xt)))
        n_raw = len(basis.raw_indices)
        ratio = 2 * d / (r + n_raw * 2)
        lr.append((r, ratio, sqnr, cos))
        print(f"  r={r:4d} ({100 * r / d:4.1f}%) ratio={ratio:5.2f}x  sqnr={sqnr:5.1f}dB  cos={cos:.4f}")

    # ISO-QUALITY: max compression at a fidelity floor (the bandwidth-bound metric).
    print("\n# ISO-QUALITY: max compression at a fidelity floor (per-boundary)")
    print("  NOTE: per-boundary SQNR/cos does NOT predict E2E fidelity for LMs —")
    print("        see bench/lowrank_e2e_probe.py and docs/lowrank-codec-findings.md.")
    for fsq, fco in [(14, 0.98), (18, 0.98), (20, 0.98), (24, 0.98)]:
        sh = [t for t in shipped if t[2] >= fsq and t[3] >= fco]
        sh_best = max(sh, key=lambda t: t[1]) if sh else None
        ok = [t for t in lr if t[2] >= fsq and t[3] >= fco]
        lr_best = min(ok, key=lambda t: t[0]) if ok else None  # min rank = max compression
        s = f"{sh_best[0]} {sh_best[1]:.2f}x" if sh_best else "none"
        lo = f"r={lr_best[0]} {lr_best[1]:.2f}x@{lr_best[2]:.1f}dB" if lr_best else "none"
        adv = f"{lr_best[1] / sh_best[1]:.2f}x more" if (lr_best and sh_best) else "-"
        print(f"  floor SQNR>={fsq}dB cos>={fco}: shipped={s:16s} low-rank={lo:22s} ({adv})")


if __name__ == "__main__":
    main()
