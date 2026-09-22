#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded exact-pair P2P route qualification probe (superl8-serve#356).

Qualifies ONE (src, dst) GPU pair for direct P2P on the CMP 100-210 fleet by
performing a small, deterministic fp16 round-trip and asserting bit-exactness
plus a non-zero payload (the PHB failure mode arrives zero-filled). It is the
offline, human-invoked counterpart to the fail-closed route policy in
``superl8serve.dist.peer_routes``: run it on an idle pair, record the evidence, and
only then add the pair to ``SUPERL8_P2P_VALIDATED_PAIRS``.

Container-ordinal contract
--------------------------
Start the container with ``--gpus '"device=A,B"'`` where A/B are PHYSICAL host
GPU indices. Inside the container CUDA remaps them to ordinals 0/1 in the given
order, and nvidia-smi reports those same remapped ordinals. The probe therefore
takes --src/--dst as CONTAINER ordinals (0/1) plus --physical-src/--physical-dst
as the OPERATOR-KNOWN host indices, and records the container-visible UUID list
(``nvidia-smi`` order) separately. It refuses to run when a mapped physical index
is a live-service GPU (9 and 14 on the current fleet) and never sweeps devices.

IMPORTANT — ``SUPERL8_P2P_VALIDATED_PAIRS`` uses the same container/torch CUDA
ordinals as this probe's --src/--dst, NOT the physical indices: for the
``--gpus '"device=5,6"'`` shape below the serving env must be
``SUPERL8_P2P_VALIDATED_PAIRS=0,1``. Physical indices/UUIDs are provenance only.
This tool never sets the env; it only qualifies the pair and records evidence.

Usage (run inside the prebuilt sm70 container so torch/CUDA match the fleet):

    docker run --rm -e PYTHONPATH="$PWD" -v "$PWD:$PWD" -w "$PWD" \
      --gpus '"device=5,6"' superl8-built:sm70 \
      python3 tools/qualify_peer_route.py --src 0 --dst 1 \
        --physical-src 5 --physical-dst 6 --json evidence.json

Exit code 0 = p2p-safe, 2 = unusable (zero-fill / non-bit-exact), 3 = refused.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import torch

# Live-service physical GPU host indices that must never be probed
# (GPU9 = superl8 LFM2.5, GPU14 = Z-Image).
_FORBIDDEN = {9, 14}


def _pattern(numel: int, device: str) -> torch.Tensor:
    """Deterministic, NaN-free fp16 pattern.

    fp16 max is 65504 and only represents integers exactly up to 2048, so an
    fp16 arange of 2^20 overflows to inf/NaN (breaking bit-exact comparison).
    Build a bounded int32 pattern in [0, 1020] (exactly representable in fp16)
    and cast: no overflow, no rounding.
    """
    return (torch.arange(numel, dtype=torch.int32, device=device) % 1021).to(torch.float16)


def _visible_uuids() -> list[str]:
    """Container-visible GPU UUIDs in CUDA/nvidia-smi order."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line.strip() for line in out.strip().splitlines() if line.strip()]


def qualify_pair(
    src: int,
    dst: int,
    *,
    numel: int = 1 << 20,
    physical_src: int | None = None,
    physical_dst: int | None = None,
    json_path: str | None = None,
) -> dict:
    """Run the deterministic round-trip probe for one container (src, dst) pair."""
    if src == dst:
        raise SystemExit("refusing to probe a same-device pair")
    # Pure arg-based guard first: refuse live-service GPUs before any device
    # query, so the guard is exercised identically on CPU-only hosts.
    for phys in (physical_src, physical_dst):
        if phys is not None and phys in _FORBIDDEN:
            raise SystemExit(
                f"refusing to probe live-service physical GPU {phys} in forbidden set {_FORBIDDEN}"
            )
    if torch.cuda.device_count() < 2:
        raise SystemExit(
            "container exposes < 2 GPUs; start with --gpus '\"device=A,B\"' so the "
            "pair is container ordinals 0/1"
        )

    uuids = _visible_uuids()
    names = [torch.cuda.get_device_properties(i).name for i in range(torch.cuda.device_count())]
    evidence: dict = {
        "probe": "superl8-serve#356 exact-pair P2P qualification",
        "container_src_ordinal": src,
        "container_dst_ordinal": dst,
        "physical_src": physical_src,
        "physical_dst": physical_dst,
        "container_visible_uuids": uuids,
        "container_visible_device_names": names,
        "numel": numel,
        "pattern": "arange(int32) % 1021 -> fp16",
        "raw_can_device_access_peer": None,
        "fp16_roundtrip_bit_exact": None,
        "payload_nonzero": None,
        "verdict": None,
    }

    try:
        evidence["raw_can_device_access_peer"] = bool(
            torch.cuda.can_device_access_peer(src, dst)
        )
    except (RuntimeError, AttributeError) as exc:
        evidence["raw_can_device_access_peer"] = f"error: {exc}"

    with torch.cuda.device(src):
        x = _pattern(numel, "cuda")
    with torch.cuda.device(dst):
        # Exactly the copy path send() uses for P2P: device-to-device .to(dst).
        src_t = x.to(dst, copy=True, non_blocking=False)
    torch.cuda.synchronize()

    xr = src_t.cpu()
    x_orig = x.cpu()
    evidence["fp16_roundtrip_bit_exact"] = bool(torch.equal(x_orig, xr))
    evidence["payload_nonzero"] = bool(xr.abs().max().item() != 0.0)
    evidence["verdict"] = (
        "p2p-safe"
        if evidence["fp16_roundtrip_bit_exact"] and evidence["payload_nonzero"]
        else "unusable-zero-fill"
    )

    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(evidence, fh, indent=2, sort_keys=True)
    return evidence


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=int, required=True, help="container ordinal source GPU (0/1)")
    ap.add_argument("--dst", type=int, required=True, help="container ordinal destination GPU (0/1)")
    ap.add_argument("--physical-src", type=int, default=None, help="physical host source GPU index")
    ap.add_argument("--physical-dst", type=int, default=None, help="physical host destination GPU index")
    ap.add_argument("--json", type=str, default=None, help="write JSON evidence to this path")
    a = ap.parse_args()
    ev = qualify_pair(
        a.src,
        a.dst,
        physical_src=a.physical_src,
        physical_dst=a.physical_dst,
        json_path=a.json,
    )
    print(json.dumps(ev, indent=2, sort_keys=True))
    if ev["verdict"] != "p2p-safe":
        sys.exit(2)


if __name__ == "__main__":
    main()
