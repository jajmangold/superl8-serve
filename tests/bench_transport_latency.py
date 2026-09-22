#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Transport wire-codec latency benchmark: raw-fp16 vs int8/int4 over the PP seam.

Measures per-boundary latency of superl8.transport send/recv under raw-fp16
passthrough, int8, and int4 wire codecs on a real 2-GPU P2P (PCIe-1.0-x1) hop.

Reports total latency, compression overhead, decompression overhead, and
effective bandwidth for 4/8/16/32 token activations. Validates measured int4
latency against the calibrated transfer-time model (``estimate_boundary_ms``)
within +/-15 % and affirms the SQNR/cos guard never degrades reconstruction
below cos > 0.995 on the measured shape range.

Usage
-----
    python3 tests/bench_transport_latency.py [--src 0] [--dst 1] [--trials 20]

Output
------
Prints a markdown table:

    token-count | fp16-latency-ms | int4-latency-ms |
    compression-overhead-ms | decompression-overhead-ms | effective-MB/s | speedup

And int8 latencies in a separate column for reference.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from superl8serve.dist import recv, send
from superl8serve.dist.codec_quality import verify_boundary_fidelity
from superl8serve.dist.transfer_model import estimate_boundary_ms

_WARMUP = 5
_TRIALS = 20
_HIDDEN = 256  # D4 proxy hidden dim

# Scheme config: (group_size, table-label).
_SCHEMES: dict[str, tuple[int | None, str]] = {
    "fp16": (None, "fp16-latency-ms"),
    "int8": (None, "int8-latency-ms"),
    "int4": (128, "int4-latency-ms"),
}


def _activation(num_tokens: int, hidden: int, device: int) -> torch.Tensor:
    with torch.cuda.device(device):
        return torch.randn(1, num_tokens, hidden, dtype=torch.float16, device="cuda")


def bench_one(
    src: int,
    dst: int,
    num_tokens: int,
    scheme: str,
    group_size: int | None,
    trials: int = _TRIALS,
    warmup: int = _WARMUP,
) -> dict:
    """Run *trials* send/recv round-trips src->dst, return aggregate timing dict."""
    shape = (1, num_tokens, _HIDDEN)
    fp16_equiv_bytes = int(torch.Size(shape).numel()) * 2

    # warmup
    x = _activation(num_tokens, _HIDDEN, src)
    for _ in range(warmup):
        h = send(x, dst=dst, scheme=scheme, group_size=group_size)
        recv(h)
    torch.cuda.synchronize()

    totals: list[float] = []
    comps: list[float] = []
    p2ps: list[float] = []
    deps: list[float] = []
    wire_bytes = 0

    for _ in range(trials):
        x = _activation(num_tokens, _HIDDEN, src)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        h = send(x, dst=dst, scheme=scheme, group_size=group_size)
        recv(h)
        torch.cuda.synchronize()
        totals.append((time.perf_counter() - t0) * 1000.0)
        comps.append(h.compress_elapsed_ms)
        p2ps.append(h.p2p_elapsed_ms)
        deps.append(h.decompress_elapsed_ms if h.decompress_elapsed_ms is not None else 0.0)
        wire_bytes = h.on_wire_bytes

    def _avg(vals: list[float]) -> float:
        return sum(vals) / max(len(vals), 1)

    total_ms = _avg(totals)
    effective_mb_s = (fp16_equiv_bytes / 1e6) / (total_ms / 1000.0)

    return {
        "num_tokens": num_tokens,
        "scheme": scheme,
        "total_ms": total_ms,
        "compress_ms": _avg(comps),
        "p2p_ms": _avg(p2ps),
        "decompress_ms": _avg(deps),
        "on_wire_bytes": wire_bytes,
        "fp16_equiv_bytes": fp16_equiv_bytes,
        "effective_mb_s": effective_mb_s,
    }


def _fmt(v: float) -> str:
    return f"{v:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Transport wire-codec latency benchmark")
    ap.add_argument("--src", type=int, default=0)
    ap.add_argument("--dst", type=int, default=1)
    ap.add_argument("--trials", type=int, default=_TRIALS)
    ap.add_argument("--warmup", type=int, default=_WARMUP)
    ap.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32],
        help="token counts to benchmark",
    )
    a = ap.parse_args()

    src, dst = a.src, a.dst
    ngpus = torch.cuda.device_count()
    if ngpus < max(src, dst) + 1:
        print(
            f"ERROR: need >= {max(src, dst) + 1} GPU(s), got {ngpus}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── 1. Fidelity gate ──────────────────────────────────────────────────
    print("=== SQNR / cosine fidelity check (must never drop below cos > 0.995) ===")
    all_fidelity_ok = True
    for nt in a.tokens:
        x = _activation(nt, _HIDDEN, src)
        for scheme in _SCHEMES:
            gs = _SCHEMES[scheme][0]
            sqnr, cos = verify_boundary_fidelity(x, scheme, group_size=gs)
            ok = cos >= 0.995
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_fidelity_ok = False
            print(f"  {nt:>3d} tok  {scheme:>6s}  SQNR={sqnr:>8.1f} dB  cos={cos:.6f}  [{status}]")
    if all_fidelity_ok:
        print(">>> All fidelity checks PASS (cos >= 0.995 for every shape/scheme)\n")
    else:
        print(">>> WARNING: some fidelity checks FAILED (cos < 0.995)\n")

    # ── 2. Latency benchmark ──────────────────────────────────────────────
    results: dict[tuple[int, str], dict] = {}

    for nt in a.tokens:
        for scheme in _SCHEMES:
            gs = _SCHEMES[scheme][0]
            r = bench_one(src, dst, nt, scheme, gs, trials=a.trials, warmup=a.warmup)
            results[(nt, scheme)] = r

    # Print table matching acceptance-criteria columns.
    header = (
        "token-count | fp16-latency-ms | int8-latency-ms | int4-latency-ms | "
        "compression-overhead-ms | decompression-overhead-ms | effective-MB/s | speedup"
    )
    sep = "|" + "---|" * 8
    print("=== Latency / bandwidth table ===")
    print(header)
    print(sep)

    for nt in a.tokens:
        fp16_r = results[(nt, "fp16")]
        int8_r = results[(nt, "int8")]
        int4_r = results[(nt, "int4")]

        codec_overhead = int4_r["compress_ms"] + int4_r["decompress_ms"]
        speedup = fp16_r["total_ms"] / int4_r["total_ms"] if int4_r["total_ms"] > 0 else float("inf")

        print(
            f"{nt:>11d} | "
            f"{_fmt(fp16_r['total_ms']):>15s} | "
            f"{_fmt(int8_r['total_ms']):>15s} | "
            f"{_fmt(int4_r['total_ms']):>14s} | "
            f"{_fmt(codec_overhead):>22s} | "
            f"{_fmt(int4_r['decompress_ms']):>23s} | "
            f"{int4_r['effective_mb_s']:>13.1f} | "
            f"{speedup:>7.2f}x"
        )

    # ── 3. Model validation ───────────────────────────────────────────────
    print("\n=== Model validation (int4 vs estimate_boundary_ms ±15%) ===")
    all_model_ok = True
    for nt in a.tokens:
        r = results[(nt, "int4")]
        shape = (1, nt, _HIDDEN)
        predicted = estimate_boundary_ms(shape, torch.float16, "int4")
        measured = r["total_ms"]
        ratio = measured / predicted
        ok = 0.85 <= ratio <= 1.15
        if not ok:
            all_model_ok = False
        status = "PASS" if ok else "FAIL"
        print(
            f"  {nt:>3d} tok  predicted={predicted:>8.3f} ms  "
            f"measured={measured:>8.3f} ms  ratio={ratio:.3f}  [{status}]"
        )
    if all_model_ok:
        print(">>> All model validation checks PASS (within ±15%)")
    else:
        print(">>> WARNING: some model validation checks FAILED (outside ±15%)")

    if not all_fidelity_ok or not all_model_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
