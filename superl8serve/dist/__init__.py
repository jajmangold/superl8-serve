# SPDX-License-Identifier: MIT
"""Engine-boundary transport: compress -> (entropy) -> P2P -> decompress (issues #81, #185, #188).

The pure-Python transport seam that ships activations between engine ranks on the
PCIe-1.0-x1 fleet. Bandwidth is the constraint, so the wire always carries the
*compressed* payload (int8/int4/nf4 codes + fp32 group scales), never fp16. The
codec seam is kept clean: `send` compresses on the source device and copies the
codes to the destination; `recv` decompresses on the destination. `accuracy_report`
gates a round-trip against per-scheme SQNR / cosine bars.

When both GPUs have direct peer access the code copy uses ``cudaMemcpyPeer`` (P2P);
otherwise it stages through pinned host RAM (see ``_can_use_p2p``).

An optional *entropy* stage (rANS, see :mod:`superl8serve.dist.entropy`) can be stacked
after quantization to shrink the wire payload further by losslessly coding the
quantized symbols at their empirical entropy floor. It is opt-in via
``send(x, dst, scheme="int8", entropy=True)`` (default ``entropy=False``); the handle
carries the rANS metadata so ``recv`` inverts the coding before dequantization.

Compression is delegated to `fni8.compress_activation` / `fni8.decompress_activation`;
this module only owns the transfer + accounting, not the quant math.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import superl8
from superl8 import Compressed

from superl8serve.dist.entropy import decode as rans_decode
from superl8serve.dist.entropy import encode as rans_encode
from superl8serve.dist.peer_routes import peer_route_allowed

# Modeled PCIe-1.0-x1 unidirectional bandwidth (bytes/s) used for the analytic
# `theoretical_wire_ms`. Kept as a module constant so the wire model is explicit.
_WIRE_BYTES_PER_S = 250e6

# Per-scheme acceptance bars (SQNR in dB, cosine similarity). Measured minima over
# Gaussian activations sit comfortably above these (int8 ~43 dB, int4/had ~18.5 dB,
# nf4 ~20 dB); the bars are set below those so the accuracy gate flags genuine
# regressions, not run-to-run noise. fp16 is lossless (bit-exact round-trip).
_BARS: dict[str, tuple[float, float]] = {
    "fp16": (float("inf"), 1.0),
    "int8": (35.0, 0.999),
    "int4": (14.0, 0.98),
    "int4-had": (14.0, 0.98),
    "nf4": (16.0, 0.985),
}


def accuracy_report(original: torch.Tensor, reconstructed: torch.Tensor, scheme: str) -> dict:
    """SQNR (dB) and cosine similarity of a round-trip, gated against `scheme` bars.

    Follows the repo numerics convention: compute the error in fp32 against the fp16
    reference (`SQNR = 10*log10(signal_power / noise_power)`). A bit-exact round-trip
    reports `sqnr_db == inf`. Returns a dict with the raw metrics, the bars, and the
    pass/fail booleans the transport tests assert on.
    """
    x = original.detach().float().flatten()
    xr = reconstructed.detach().float().flatten()

    noise = x - xr
    noise_power = noise.pow(2).sum()
    if noise_power.item() == 0.0:
        sqnr_db = float("inf")
    else:
        signal_power = x.pow(2).sum()
        sqnr_db = float(10.0 * torch.log10(signal_power / noise_power))

    cos = float(torch.nn.functional.cosine_similarity(x, xr, dim=0, eps=1e-12))

    sqnr_bar, cos_bar = _BARS.get(scheme, _BARS["int8"])
    return {
        "scheme": scheme,
        "sqnr_db": sqnr_db,
        "sqnr_bar": sqnr_bar,
        "sqnr_pass": sqnr_db >= sqnr_bar,
        "cos": cos,
        "cos_bar": cos_bar,
        "cos_pass": cos >= cos_bar,
    }


@dataclass
class TransferHandle:
    """Everything the receiver needs to reconstruct a transferred activation.

    `d_payload`/`d_scales` are the compressed codes as they land on the destination
    device; the remaining fields carry the shape/scheme metadata for decompression and
    the measured timings for the wire model. `decompress_elapsed_ms` is filled in by
    `recv`.
    """

    src_device: int
    dst_device: int
    scheme: str
    shape: torch.Size
    dtype: torch.dtype
    group_size: int | None
    d_payload: torch.Tensor
    d_scales: torch.Tensor
    compress_elapsed_ms: float
    p2p_elapsed_ms: float
    decompress_elapsed_ms: float | None = None
    used_p2p: bool = False
    entropy_freqs: np.ndarray | None = None
    entropy_n: int | None = None
    entropy_payload_dtype: torch.dtype | None = None
    entropy_payload_shape: tuple[int, ...] | None = None

    @property
    def on_wire_bytes(self) -> int:
        """Bytes actually crossing the link: packed codes + fp32 group scales.

        When the entropy (rANS) stage is active the payload is the compressed
        byte stream, and the normalized frequency table must also cross the wire
        so the receiver can invert the coding — so it is counted here. The table
        is transmitted as uint16 (freqs sum to 2**12 < 2**16), i.e. 2 bytes/entry.
        """
        payload_bytes = self.d_payload.numel() * self.d_payload.element_size()
        scale_bytes = self.d_scales.numel() * self.d_scales.element_size()
        freq_bytes = 0
        if self.entropy_freqs is not None:
            freq_bytes = int(self.entropy_freqs.size) * 2
        return payload_bytes + scale_bytes + freq_bytes

    @property
    def theoretical_wire_ms(self) -> float:
        """Analytic transfer time on the modeled PCIe link (ms)."""
        return self.on_wire_bytes / _WIRE_BYTES_PER_S * 1000.0


def _elapsed_ms(fn):
    """Run `fn` on the current CUDA device, returning (result, elapsed_ms)."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    end.synchronize()
    return result, start.elapsed_time(end)


def _can_use_p2p(src: int, dst: int) -> bool:
    """True only for a *validated* direct device→device (``cudaMemcpyPeer``) copy.

    ``torch.cuda.can_device_access_peer`` alone is NOT trustworthy on this fleet:
    the PHB hop 6->14 reports peer support yet the copy silently zero-fills the
    destination (fni8-serve#356). P2P is therefore gated by an explicit validated
    pair set (``FNI8_P2P_VALIDATED_PAIRS``, whose values are torch CUDA ordinals
    as seen by the serving process — not physical ``nvidia-smi`` indices) plus the
    live capability probe, and fails closed to pinned host staging for every
    unvalidated / unknown route.
    """
    allowed, _ = peer_route_allowed(src, dst)
    return allowed


def send(
    x: torch.Tensor,
    dst: int,
    *,
    scheme: str = "int8",
    group_size: int | None = None,
    entropy: bool = False,
) -> TransferHandle:
    """Compress `x` on its device and copy the codes to device `dst`.

    Returns a `TransferHandle` the receiver hands to `recv`. In-process here (the P2P
    is the fleet's PCIe path); the codes are copied on dedicated streams so the payload
    and scale transfers can overlap. `p2p_elapsed_ms` measures the code copy, not the
    fp16 activation.

    When `src != dst` and the two GPUs support direct peer access the copy uses
    ``cudaMemcpyPeer`` (P2P) instead of staging through host RAM. Peer access is checked
    and enabled transparently; when unavailable the copy falls back to host staging.

    When *entropy* is True (opt-in; default False) the quantized payload is losslessly
    rANS-coded before the copy — composes with any scheme and with either the P2P or
    host-staging transfer path. ``recv`` inverts the entropy coding before dequantization.
    """
    src = x.device.index if x.is_cuda else torch.cuda.current_device()

    with torch.cuda.device(src):
        c, compress_ms = _elapsed_ms(lambda: fni8.compress_activation(x, scheme=scheme, group_size=group_size))

    # Optional entropy (rANS) stage: losslessly code the raw *bytes* of the compressed
    # payload (whatever its dtype — int8 codes, packed-int32 nibbles, fp16). Record the
    # original dtype/shape so `recv` reconstructs the exact tensor bit-for-bit.
    payload_tensor = c.payload
    entropy_freqs: np.ndarray | None = None
    entropy_n: int | None = None
    entropy_payload_dtype: torch.dtype | None = None
    entropy_payload_shape: tuple[int, ...] | None = None
    if entropy:
        entropy_payload_dtype = c.payload.dtype
        entropy_payload_shape = tuple(c.payload.shape)
        np_bytes = c.payload.detach().cpu().contiguous().view(torch.uint8).numpy().ravel()
        encoded_bytes, entropy_freqs = rans_encode(np_bytes)
        entropy_n = int(np_bytes.size)
        payload_tensor = torch.frombuffer(bytearray(encoded_bytes), dtype=torch.uint8)

    # Decide transfer strategy: P2P when src != dst and peers can access each other's
    # memory directly; host-staging fallback otherwise.
    use_p2p = _can_use_p2p(src, dst)

    # Copy the compressed codes src -> dst. Double-buffered: payload and scales ride
    # separate streams so their transfers overlap. `copy=True` forces a real copy even
    # when src == dst, so the timing reflects a genuine transfer.
    with torch.cuda.device(dst):
        payload_stream = torch.cuda.Stream()
        scale_stream = torch.cuda.Stream()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if use_p2p:
            # Direct device-to-device copy — PyTorch's .to() uses cudaMemcpyPeer
            # when peer access is enabled, avoiding a round-trip through host RAM.
            # (With entropy on, payload_tensor is a small CPU byte stream, so its
            # copy is H2D; the device-resident scales still ride the P2P path.)
            with torch.cuda.stream(payload_stream):
                d_payload = payload_tensor.to(dst, copy=True, non_blocking=True)
            with torch.cuda.stream(scale_stream):
                d_scales = c.scales.to(dst, copy=True, non_blocking=True)
        else:
            # Host-staging fallback: copy through pinned CPU memory. Used when
            # src == dst or when the GPU pair lacks direct peer access.
            # Allocate pinned host staging buffers explicitly (Issue #185)
            # and copy GPU → pinned CPU synchronously, then async to dst GPU.
            cpu_payload = torch.empty(payload_tensor.shape, dtype=payload_tensor.dtype, pin_memory=True)
            cpu_payload.copy_(payload_tensor)
            cpu_scales = torch.empty(c.scales.shape, dtype=c.scales.dtype, pin_memory=True)
            cpu_scales.copy_(c.scales)
            with torch.cuda.stream(payload_stream):
                d_payload = cpu_payload.to(dst, copy=True, non_blocking=True)
            with torch.cuda.stream(scale_stream):
                d_scales = cpu_scales.to(dst, copy=True, non_blocking=True)
        end.record(payload_stream)
        end.synchronize()
        payload_stream.synchronize()
        scale_stream.synchronize()
        p2p_ms = start.elapsed_time(end)

    return TransferHandle(
        src_device=src,
        dst_device=dst,
        scheme=scheme,
        shape=c.shape,
        dtype=c.dtype,
        group_size=c.group_size,
        d_payload=d_payload,
        d_scales=d_scales,
        compress_elapsed_ms=compress_ms,
        p2p_elapsed_ms=p2p_ms,
        used_p2p=use_p2p,
        entropy_freqs=entropy_freqs,
        entropy_n=entropy_n,
        entropy_payload_dtype=entropy_payload_dtype,
        entropy_payload_shape=entropy_payload_shape,
    )


def recv(handle: TransferHandle) -> torch.Tensor:
    """Decompress the transferred codes on the destination device.

    Reconstructs the `fni8.Compressed` payload from the handle and runs the decoder,
    returning the fp16 activation on `handle.dst_device`. When *entropy* coding was
    applied during `send`, the rANS stage is inverted (losslessly) first, so the
    reconstructed activation is identical to the non-entropy path. Records
    `handle.decompress_elapsed_ms` as a side effect.
    """
    payload = handle.d_payload

    if handle.entropy_freqs is not None and handle.entropy_n is not None:
        # Invert the rANS stage byte-for-byte, then reinterpret the recovered
        # bytes back into the original payload dtype/shape (recorded by `send`).
        payload_cpu = payload.cpu().contiguous().numpy().tobytes()
        decoded = rans_decode(payload_cpu, handle.entropy_freqs, handle.entropy_n)
        byte_tensor = torch.from_numpy(decoded.copy()).to(handle.d_payload.device)
        target_dtype = handle.entropy_payload_dtype or torch.uint8
        payload = byte_tensor.view(target_dtype)
        if handle.entropy_payload_shape is not None:
            payload = payload.reshape(handle.entropy_payload_shape)

    c = Compressed(
        scheme=handle.scheme,
        shape=tuple(handle.shape),
        dtype=handle.dtype,
        group_size=handle.group_size,
        payload=payload,
        scales=handle.d_scales,
        d=handle.shape[-1],
    )
    with torch.cuda.device(handle.dst_device):
        xr, decompress_ms = _elapsed_ms(lambda: fni8.decompress_activation(c))
    handle.decompress_elapsed_ms = decompress_ms
    return xr
