# SPDX-License-Identifier: MIT
"""Low-rank projection ("VAE-ish") wire codec — EVALUATED AND REJECTED (#184).

Fits an SVD basis on calibration activations, projects boundary activations to
r-dim latent, int8-quantizes the coefficients, and transmits outlier channels
raw.  Decode: int8-dequant → inverse projection → reinsert raw channels.

STATUS: this is a self-contained, mechanically-correct reference implementation
retained for reproducibility, but it is **not wired into the transport path**.

On the *per-boundary* iso-quality axis (max compression at a fidelity floor — the
right metric for a bandwidth-bound link) low-rank actually wins: at the shipped
int4 gate it compresses ~6× more than the best shipped codec. It is nonetheless
**rejected** because that axis does not predict end-to-end LM fidelity: the
residual stream's next-token signal concentrates in the lowest-variance
directions, which a variance-ordered projection discards first — so even a single
low-rank PP boundary collapses generation (top-1 next-token agreement ~2 % at
r=400 / 20 dB per-boundary), while int8/int4 stay coherent. See
``docs/lowrank-codec-findings.md`` and reproduce with
``bench/lowrank_activation_probe.py`` (iso-quality) and
``bench/lowrank_e2e_probe.py`` (end-to-end).

Reference: Transport issue #184.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LowRankBasis:
    """Trained SVD basis for the low-rank wire codec.

    Attributes:
        U:  (d, r) orthonormal basis — top-r right singular vectors (fp32 CPU).
        mean: (d,) per-channel mean of calibration activations (fp32 CPU).
        raw_indices: sorted list of channel indices transmitted raw.
        r: rank of the projection.
        d: original hidden dimension.
    """

    U: torch.Tensor
    mean: torch.Tensor
    raw_indices: list[int]
    r: int
    d: int

    @classmethod
    def from_state(
        cls, U: torch.Tensor, mean: torch.Tensor, raw_indices: list[int]
    ) -> LowRankBasis:
        r = U.shape[-1]
        d = U.shape[-2]
        return cls(U=U, mean=mean, raw_indices=sorted(raw_indices), r=r, d=d)


# ── internal int8 helpers ──────────────────────────────────────────────────


def _int8_scale(t: torch.Tensor) -> tuple[torch.Tensor, float]:
    scale = float(t.abs().max()) / 127.0
    if scale == 0.0:
        return torch.zeros_like(t, dtype=torch.int8), 0.0
    q = torch.clamp(torch.round(t / scale), -128, 127).to(torch.int8)
    return q, scale


def _int8_dequant(q: torch.Tensor, scale: float) -> torch.Tensor:
    if scale == 0.0:
        return torch.zeros_like(q, dtype=torch.float32)
    return q.float() * scale


# ── basis fitting (calibration) ────────────────────────────────────────────


def fit_basis(
    activations: torch.Tensor,
    r: int = 224,
    raw_fraction: float = 0.005,
    *,
    seed: int = 42,
) -> LowRankBasis:
    """Fit a LowRankBasis from calibration *activations* via SVD.

    The SVD is computed on CPU with deterministic algorithms so the basis is
    bit-identical across runs.  The returned basis tensors live on CPU.

    Args:
        activations: (N, d) fp32 calibration activation vectors.
        r: target rank for the projection (default 224 → ≈4x at Qwen2-0.5B).
        raw_fraction: fraction of channels with largest mean magnitude that
            are transmitted raw instead of projected (default 0.005 ≈ 0.5 %).
        seed: random seed for deterministic SVD fallback.

    Returns:
        Fitted LowRankBasis.
    """
    N, d = activations.shape
    if r > min(N, d):
        r = min(N, d)

    x = activations.detach().float().cpu()
    mean = x.mean(dim=0)
    x_centered = x - mean

    with torch.no_grad():
        torch.manual_seed(seed)
        _, _, Vt = torch.linalg.svd(x_centered, full_matrices=False)

    # Vt shape: (min(N,d), d).  Top-r right singular vectors as columns.
    U_basis = Vt[:r, :].T.contiguous()

    # Identify raw channels: largest mean magnitude across calibration tokens.
    channel_mag = x.abs().mean(dim=0)
    num_raw = max(1, int(d * raw_fraction))
    _, raw_idx = torch.topk(channel_mag, k=num_raw)
    raw_indices = sorted(raw_idx.tolist())

    return LowRankBasis(U=U_basis, mean=mean, raw_indices=raw_indices, r=r, d=d)


# ── encode / decode ────────────────────────────────────────────────────────


class LowRankCodec:
    """Low-rank projection wire codec.

    Composes with the transport ``send``/``recv`` seam by providing ``encode``
    and ``decode`` methods.  Deterministic: same basis + same input → bit-identical
    codes across runs.
    """

    def __init__(self, basis: LowRankBasis):
        self.basis = basis

    def encode(self, x: torch.Tensor) -> dict:
        """Compress *x* (..., d) fp16 → low-rank codes + raw channels.

        Returns a dict with:
            latent:      (..., r) int8 — quantised projection coefficients.
            scale:       float — per-tensor int8 scale for the latent.
            raw_values:  (..., num_raw) fp16 — raw channel values.
            raw_indices: list[int] — which channels are transmitted raw.
            shape:       original shape of x.
        """
        basis = self.basis
        device = x.device

        U = basis.U.to(device, dtype=torch.float32)
        mean = basis.mean.to(device, dtype=torch.float32)

        xf = x.float()
        x_centered = xf - mean

        latent_fp32 = x_centered @ U  # (..., r)
        # Per-channel int8 quant on the r latent dims: a single per-tensor scale
        # was dominated by outlier latent dims, capping recon SQNR (~16 dB) and
        # making the rank irrelevant. One scale per latent dim (like per-row
        # weight quant) lets each dim use its full int8 range.
        amax = latent_fp32.abs().reshape(-1, latent_fp32.shape[-1]).amax(0)  # (r,)
        scale = (amax / 127.0).clamp_min(1e-12)  # (r,) fp32
        qlatent = torch.clamp(torch.round(latent_fp32 / scale), -128, 127).to(torch.int8)

        raw_values = xf[..., basis.raw_indices].contiguous()

        return {
            "latent": qlatent,
            "scale": scale,
            "raw_values": raw_values.half(),
            "raw_indices": basis.raw_indices,
            "shape": x.shape,
        }

    def decode(self, codes: dict) -> torch.Tensor:
        """Decompress *codes* → fp16 reconstruction."""
        basis = self.basis

        q = codes["latent"]
        scale = codes["scale"]
        scale = scale.to(q.device) if torch.is_tensor(scale) else scale
        latent_fp32 = q.float() * scale  # per-channel (r,) broadcast over leading dims
        device = latent_fp32.device

        U = basis.U.to(device, dtype=torch.float32)
        mean = basis.mean.to(device, dtype=torch.float32)

        x_recon = latent_fp32 @ U.T + mean

        raw_values = codes["raw_values"].float()
        x_recon[..., basis.raw_indices] = raw_values

        return x_recon.half()
