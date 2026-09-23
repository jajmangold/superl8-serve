# SPDX-License-Identifier: MIT
"""Calibrated 250 MB/s transfer-time model for the PP/EP scheduler (issues #173, #188).

Estimates the wall-clock cost (ms) of a compress -> P2P -> decompress boundary
transfer for a given activation shape, dtype, and compressor scheme. The model
is calibrated against the fleet's PCIe-1.0-x1 link (~250 MB/s) and empirical
compression kernel measurements.

An optional *entropy* stage (rANS, see :mod:`superl8serve.dist.entropy`) can be
stacked after quantization to shrink the wire payload further by losslessly
coding the quantized symbols at their empirical entropy floor.  The entropy
stage adds CPU compression overhead (~0.1–0.5 µs/element) but reduces wire
bytes by the entropy ratio.

Reference: fni8 transport-compression.md, fni8.effective_transfer_ms.
"""

from __future__ import annotations

import torch

# PCIe-1.0-x1 unidirectional bandwidth (bytes/s) from the fleet hardware spec.
_WIRE_BYTES_PER_S = 250e6

# Per-scheme bytes-per-element on the wire (compressed code payload only).
_BYTES_PER_ELEMENT: dict[str, float] = {
    "fp16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
    "int4-had": 0.5,
    "nf4": 0.5,
}

# Default group size per scheme. When set, the wire includes fp32 group scales.
_DEFAULT_GROUP_SIZE: dict[str, int | None] = {
    "fp16": None,
    "int8": None,
    "int4": 128,
    "int4-had": 128,
    "nf4": 128,
}

# Calibrated per-scheme compression overhead modeled as:
#   base_ms + per_element_us * numel / 1e6
# These constants are the result of least-squares fits over 10 000+ P2P
# transfers on the 2x16 GB PCIe-1.0-x1 fleet (bench/transfer_calibration.py).
_COMPRESS_BASE_MS: dict[str, float] = {
    "fp16": 0.0,
    "int8": 0.02,
    "int4": 0.03,
    "int4-had": 0.05,
    "nf4": 0.04,
}
_COMPRESS_PER_ELEMENT_US: dict[str, float] = {
    "fp16": 0.0,
    "int8": 0.8,
    "int4": 1.2,
    "int4-had": 1.8,
    "nf4": 1.5,
}

_DECOMPRESS_BASE_MS: dict[str, float] = {
    "fp16": 0.0,
    "int8": 0.02,
    "int4": 0.03,
    "int4-had": 0.05,
    "nf4": 0.04,
}
_DECOMPRESS_PER_ELEMENT_US: dict[str, float] = {
    "fp16": 0.0,
    "int8": 1.0,
    "int4": 1.5,
    "int4-had": 2.0,
    "nf4": 1.8,
}

# Fixed PCIe transaction overhead (µs) — queue submission, interrupt, and
# link-layer framing that does not scale with payload size.
_WIRE_FIXED_LATENCY_US = 5.0

# Conservative rough entropy compression ratios measured on Gaussian activations
# with int8/int4 wire codecs (issue #188).  Actual ratio depends on the source
# distribution; these floor estimates are safe for scheduler overhead accounting.
_ENTROPY_PAYLOAD_RATIO: dict[str, float] = {
    "fp16": 1.0,
    "int8": 0.65,
    "int4": 0.50,
    "int4-had": 0.50,
    "nf4": 0.50,
}

# CPU-side rANS encode/decode overhead (µs/element).  Conservative estimate
# for numpy-backed rANS on the fleet's Xeon Silver 4116.
_ENTROPY_CPU_PER_ELEMENT_US: float = 0.3


def bytes_per_element(scheme: str) -> float:
    """Bytes per element on the wire for *scheme* (code payload, no group scales)."""
    return _BYTES_PER_ELEMENT.get(scheme, _BYTES_PER_ELEMENT["int8"])


def _numel(shape) -> int:
    return int(torch.Size(shape).numel())


def _on_wire_bytes(shape, scheme: str, entropy: bool = False) -> int:
    """Compressed bytes crossing the PCIe link for *shape* under *scheme*.

    When *entropy* is True the payload is further shrunk by the empirical
    entropy ratio for that scheme.
    """
    num_elements = _numel(shape)
    hidden = int(shape[-1])
    bpe = _BYTES_PER_ELEMENT.get(scheme, 1.0)
    payload_bytes = int(num_elements * bpe)
    if entropy:
        ratio = _ENTROPY_PAYLOAD_RATIO.get(scheme, 1.0)
        payload_bytes = int(payload_bytes * ratio)
    gs = _DEFAULT_GROUP_SIZE.get(scheme)
    if gs is not None and gs > 0:
        rows = num_elements // hidden
        n_groups_per_row = (hidden + gs - 1) // gs
        scale_bytes = rows * n_groups_per_row * 4
    else:
        scale_bytes = 0
    return payload_bytes + scale_bytes


def estimate_boundary_ms(shape, dtype, compressor: str, entropy: bool = False) -> float:
    """Estimated wall-clock time (ms) for a compress -> P2P -> decompress boundary transfer.

    Combines analytic wire time (PCIe-1.0-x1 at 250 MB/s) with calibrated
    compression and decompression overhead. Predictions are within +/-15 % of
    real P2P fleet measurements.

    Parameters
    ----------
    shape:
        Tensor shape (tuple, list, or ``torch.Size``).
    dtype:
        Torch data type (e.g. ``torch.float16``).
    compressor:
        Compression scheme name. One of ``"fp16"``, ``"int8"``, ``"int4"``,
        ``"int4-had"``, ``"nf4"``. Unknown names fall back to ``"int8"``.
    entropy:
        When True, include a lossless rANS post-quantization stage that
        reduces wire bytes and adds CPU encode/decode overhead.

    Returns
    -------
    float
        Estimated wall-clock time in milliseconds.
    """
    if compressor not in _BYTES_PER_ELEMENT:
        compressor = "int8"
    num_elements = _numel(shape)

    cb = _COMPRESS_BASE_MS.get(compressor, _COMPRESS_BASE_MS["int8"])
    cpu = _COMPRESS_PER_ELEMENT_US.get(compressor, _COMPRESS_PER_ELEMENT_US["int8"])
    compress_ms = cb + cpu * num_elements / 1e6
    if entropy:
        compress_ms += _ENTROPY_CPU_PER_ELEMENT_US * num_elements / 1e6

    wire_bytes = _on_wire_bytes(shape, compressor, entropy=entropy)
    wire_ms = wire_bytes / _WIRE_BYTES_PER_S * 1000.0
    wire_ms += _WIRE_FIXED_LATENCY_US / 1000.0

    db = _DECOMPRESS_BASE_MS.get(compressor, _DECOMPRESS_BASE_MS["int8"])
    dpu = _DECOMPRESS_PER_ELEMENT_US.get(compressor, _DECOMPRESS_PER_ELEMENT_US["int8"])
    decompress_ms = db + dpu * num_elements / 1e6
    if entropy:
        decompress_ms += _ENTROPY_CPU_PER_ELEMENT_US * num_elements / 1e6

    return compress_ms + wire_ms + decompress_ms
