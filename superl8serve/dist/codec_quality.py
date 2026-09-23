# SPDX-License-Identifier: MIT
"""Wire-codec accuracy gate: compress/decompress activations and verify reconstruction fidelity.

Part of D9 rung 3 (#77, #175). Provides ``verify_boundary_fidelity`` which
round-trips an activation through the fni8 wire codec and returns SQNR / cosine
similarity so callers can assert the compression never corrupts model output.
"""

from __future__ import annotations

import torch

import superl8


def verify_boundary_fidelity(
    activation: torch.Tensor,
    compressor: str,
    group_size: int | None = None,
) -> tuple[float, float]:
    """Compress *activation* via *compressor*, decompress, and return (sqnr_db, cosine).

    Parameters
    ----------
    activation:
        fp16 activation tensor on the CUDA device.
    compressor:
        Wire-codec scheme name (e.g. ``"int8"``, ``"int4"``, ``"int4-had"``,
        ``"nf4"``, ``"fp16"``) matching ``fni8.compress_activation``.  Call
        :func:`select_wire_scheme` to let the accuracy gate choose the scheme.
    group_size:
        Group size for per-group quantization schemes. Defaults to ``128`` for
        int4-compatible schemes, ``None`` for per-tensor.

    Returns
    -------
    (sqnr_db, cosine):
        Reconstruction signal-to-noise ratio in dB and cosine similarity between
        the original and reconstructed activation, both computed in fp32.

    Raises
    ------
    RuntimeError
        If the compressor scheme is not recognised by ``fni8.compress_activation``.
    """
    if group_size is None and compressor in ("int4", "int4-had", "nf4"):
        group_size = 128

    c = fni8.compress_activation(activation, scheme=compressor, group_size=group_size)
    reconstructed = fni8.decompress_activation(c)

    x = activation.detach().float().flatten()
    xr = reconstructed.detach().float().flatten()

    noise = x - xr
    noise_power = noise.pow(2).sum()
    if noise_power.item() == 0.0:
        sqnr_db = float("inf")
    else:
        signal_power = x.pow(2).sum()
        sqnr_db = float(10.0 * torch.log10(signal_power / noise_power))

    cos = float(torch.nn.functional.cosine_similarity(x, xr, dim=0, eps=1e-12))

    return sqnr_db, cos


def select_wire_scheme(
    activation: torch.Tensor,
    group_size: int | None = None,
) -> str:
    """Select wire-codec scheme by gating int4 fidelity on *activation*.

    Compresses and decompresses *activation* with int4, then checks SQNR and
    cosine against the acceptance bars (14 dB / 0.98).  Returns ``"int4"`` when
    both metrics clear the bar and ``"int8"`` otherwise, so the PP boundary
    always uses the densest codec that the data tolerates.

    Parameters
    ----------
    activation:
        fp16 activation tensor on the CUDA device (same as
        :func:`verify_boundary_fidelity`).
    group_size:
        Group size forwarded to :func:`verify_boundary_fidelity`.  ``None``
        uses the scheme default (128 for int4).

    Returns
    -------
    ``"int4"`` or ``"int8"``.
    """
    sqnr, cos = verify_boundary_fidelity(activation, "int4", group_size=group_size)
    return "int4" if sqnr >= 14.0 and cos >= 0.98 else "int8"
