# SPDX-License-Identifier: MIT
"""Low-rank wire codec: the r-dim latent must be int8-quantized PER-CHANNEL, not
per-tensor. A single per-tensor scale is dominated by outlier latent dims and caps
reconstruction SQNR (~16 dB on real activations) regardless of rank — making the
whole low-rank projection pointless (rank buys nothing). Per-channel quant lets each
latent dim use its full int8 range, so recon SQNR rises monotonically with rank."""

import pytest

torch = pytest.importorskip("torch")
from superl8serve.dist.lowrank import LowRankCodec, fit_basis


def _calib(n=4096, d=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    # activation-like: a few high-variance (outlier) channels + a low-variance tail
    x = torch.randn(n, d, generator=g)
    x[:, :8] *= 30.0  # massive-activation channels
    return x


def test_recon_sqnr_rises_with_rank():
    """Per-channel latent quant: reconstruction SQNR must increase with rank.
    (With the old per-tensor scale it was flat — the regression this guards.)"""
    calib = _calib()
    probe = calib[:512]
    sqnrs = []
    for r in (32, 128, 400):
        cd = LowRankCodec(fit_basis(calib, r=r, raw_fraction=0.01))
        rec = cd.decode(cd.encode(probe.half())).float()
        noise = (probe - rec).pow(2).mean()
        sig = probe.pow(2).mean()
        sqnrs.append(10 * torch.log10(sig / noise).item())
    # SQNR must rise with rank (the per-channel fix). With the old per-tensor scale it
    # was flat (outlier latent dim caps it). Monotonic non-decreasing + a clear net gain.
    assert sqnrs[1] >= sqnrs[0] - 0.5 and sqnrs[2] >= sqnrs[1] - 0.5, f"non-monotonic: {sqnrs}"
    assert sqnrs[-1] > sqnrs[0] + 5.0, f"SQNR flat with rank -> latent quant not per-channel: {sqnrs}"


def test_roundtrip_shape_and_finite():
    calib = _calib()
    cd = LowRankCodec(fit_basis(calib, r=128, raw_fraction=0.01))
    x = calib[:64].half()
    rec = cd.decode(cd.encode(x))
    assert rec.shape == x.shape
    assert torch.isfinite(rec).all()
