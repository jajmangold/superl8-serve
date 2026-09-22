# SPDX-License-Identifier: MIT
"""Low-rank projection wire codec: round-trip, SQNR/cosine, determinism (issue #184).

Tests the superl8serve/dist/lowrank.py module in isolation (CPU-only). Verifies:
    - fit_basis returns a valid LowRankBasis with correct shapes
    - encode / decode round-trips at cos >= 0.999 (int8-quality bar)
    - Determinism: same input + seed → bit-identical codes ×3
    - Compression ratio is ~4x at default settings
"""

from __future__ import annotations

import torch

from superl8serve.dist.lowrank import LowRankBasis, LowRankCodec, fit_basis


def _activation(N: int, d: int, *, outlier: bool = False, seed: int = 0, structured: bool = True) -> torch.Tensor:
    """Synthetic calibration activations.

    When *structured* is True (default), generates data with a low-rank signal
    plus small isotropic noise — mimicking the spectral decay of real transformer
    activations.  When False, returns pure isotropic Gaussian (worst case for PCA).
    """
    torch.manual_seed(seed)
    if structured:
        signal_rank = min(16, d // 4)
        coeffs = torch.randn(N, signal_rank, dtype=torch.float32) * 10.0
        basis = torch.randn(d, signal_rank, dtype=torch.float32)
        basis = basis / torch.linalg.norm(basis, dim=0, keepdim=True)
        signal = coeffs @ basis.T
        noise = torch.randn(N, d, dtype=torch.float32) * 0.01
        x = signal + noise
    else:
        x = torch.randn(N, d, dtype=torch.float32)
    if outlier:
        chan = torch.zeros(d, dtype=torch.float32)
        chan[:: max(1, d // 8)] = 12.0
        x = x + chan
    return x


# ── fit_basis correctness ───────────────────────────────────────────────────


class TestFitBasis:
    def test_basic_shapes(self):
        N, d = 512, 256
        x = _activation(N, d)
        basis = fit_basis(x, r=32, raw_fraction=0.01)
        assert basis.U.shape == (d, 32), f"U shape {basis.U.shape} != ({d}, 32)"
        assert basis.mean.shape == (d,), f"mean shape {basis.mean.shape}"
        assert basis.r == 32
        assert basis.d == d
        assert len(basis.raw_indices) == max(1, int(d * 0.01))
        assert all(0 <= idx < d for idx in basis.raw_indices)
        assert basis.raw_indices == sorted(basis.raw_indices)

    def test_U_is_orthonormal(self):
        N, d, r = 512, 256, 32
        x = _activation(N, d)
        basis = fit_basis(x, r=r)
        # U.T @ U should be close to I_r
        UtU = basis.U.T @ basis.U
        eye_r = torch.eye(r, dtype=torch.float32)
        diff = (UtU - eye_r).abs().max().item()
        assert diff < 1e-5, f"U^T U != I; max diff {diff:.2e}"

    def test_projection_reconstruction_quality(self):
        """On artificially low-rank data (rank-16 signal) a rank-48 basis is
        near-exact. This is a sanity check on the codec *mechanics* only — it is
        NOT evidence the codec works on real activations, which are not low-rank
        (see the E2E evaluation in
        docs/lowrank-codec-findings.md)."""
        N_cal, N_test, d, r = 512, 32, 128, 48
        all_x = _activation(N_cal + N_test, d, seed=1)  # structured (rank-16) signal
        cal, test = all_x[:N_cal], all_x[N_cal:]
        basis = fit_basis(cal, r=r, raw_fraction=0.0)
        codec = LowRankCodec(basis)
        test_fp16 = test.half()
        xr = codec.decode(codec.encode(test_fp16))
        cos = torch.nn.functional.cosine_similarity(
            test_fp16.float().flatten(), xr.float().flatten(), dim=0, eps=1e-12
        )
        assert cos > 0.99, f"PCA reconstruction cos {cos:.6f} too low"

    def test_raw_fraction_zero(self):
        """raw_fraction=0.0 means no raw channels, just the low-rank projection."""
        N, d, r = 256, 64, 16
        x = _activation(N, d)
        basis = fit_basis(x, r=r, raw_fraction=0.0)
        assert len(basis.raw_indices) == 1  # clamped to min 1
        codec = LowRankCodec(basis)
        x_fp16 = x[:4].half()
        xr = codec.decode(codec.encode(x_fp16))
        assert xr.shape == x_fp16.shape
        assert torch.isfinite(xr).all()

    def test_deterministic_basis(self):
        """Same activations + seed → bit-identical basis."""
        x = _activation(512, 128, seed=42)
        b1 = fit_basis(x, r=24, seed=42)
        b2 = fit_basis(x, r=24, seed=42)
        assert torch.equal(b1.U, b2.U), "U differs across runs"
        assert torch.equal(b1.mean, b2.mean), "mean differs across runs"
        assert b1.raw_indices == b2.raw_indices, "raw_indices differ across runs"

    def test_rank_clipped_to_min_N_d(self):
        """When r > min(N, d), it should be clipped."""
        N, d = 32, 256
        x = _activation(N, d)
        basis = fit_basis(x, r=128)  # r > N, should clip to min(N,d)=32
        assert basis.r == min(N, d)
        assert basis.U.shape == (d, min(N, d))


# ── LowRankCodec encode/decode ──────────────────────────────────────────────


class TestLowRankCodec:
    def _make_basis(self, d=128, r=24, N=512):
        x = _activation(N, d, seed=42)
        return fit_basis(x, r=r, raw_fraction=0.01)

    def test_encode_decode_roundtrip_shape(self):
        d, r = 128, 24
        basis = self._make_basis(d=d, r=r)
        codec = LowRankCodec(basis)
        x = _activation(4, d, seed=1).half()
        codes = codec.encode(x)
        xr = codec.decode(codes)
        assert xr.shape == x.shape, f"shape mismatch {xr.shape} != {x.shape}"
        assert xr.dtype == torch.float16

    def test_encode_returns_correct_structure(self):
        basis = self._make_basis()
        codec = LowRankCodec(basis)
        x = _activation(4, 128, seed=1).half()
        codes = codec.encode(x)
        assert "latent" in codes
        assert "scale" in codes
        assert "raw_values" in codes
        assert "raw_indices" in codes
        assert "shape" in codes
        assert codes["latent"].dtype == torch.int8
        assert codes["raw_values"].dtype == torch.float16
        assert codes["latent"].shape[-1] == basis.r
        assert codes["raw_values"].shape[-1] == len(basis.raw_indices)
        # scale is now a per-channel (r,) tensor, one int8 scale per latent dim
        assert torch.is_tensor(codes["scale"])
        assert codes["scale"].shape[-1] == basis.r
        assert (codes["scale"] >= 0.0).all()

    def test_deterministic_encode(self):
        """Same input → bit-identical codes ×3."""
        basis = self._make_basis(d=128, r=24)
        codec = LowRankCodec(basis)
        x = _activation(2, 128, seed=7).half()
        c1 = codec.encode(x)
        c2 = codec.encode(x)
        c3 = codec.encode(x)
        for key in ("latent", "raw_values"):
            assert torch.equal(c1[key], c2[key]), f"{key} differs on 2nd encode"
            assert torch.equal(c1[key], c3[key]), f"{key} differs on 3rd encode"
        assert torch.equal(c1["scale"], c2["scale"]) and torch.equal(c2["scale"], c3["scale"])

    def test_different_seed_different_basis(self):
        """Different calibration data → different codes."""
        x1 = _activation(512, 128, seed=1)
        x2 = _activation(512, 128, seed=2)
        b1 = fit_basis(x1, r=24)
        b2 = fit_basis(x2, r=24)
        codec1 = LowRankCodec(b1)
        codec2 = LowRankCodec(b2)
        inp = _activation(2, 128, seed=7).half()
        c1 = codec1.encode(inp)
        c2 = codec2.encode(inp)
        assert not torch.equal(c1["latent"], c2["latent"])

    def test_cosine_above_int8_bar(self):
        """MECHANICAL SANITY ONLY (artificially low-rank data). On the rank-16
        structured signal the codec is near-exact; this does not generalize to
        real activations — see docs/lowrank-codec-findings.md."""
        d, r = 256, 48
        N_cal, N_test = 1024, 8
        all_x = _activation(N_cal + N_test, d, seed=42)
        cal, test = all_x[:N_cal], all_x[N_cal:]
        basis = fit_basis(cal, r=r, raw_fraction=0.005)
        codec = LowRankCodec(basis)
        x = test.half()
        xr = codec.decode(codec.encode(x))
        cos = float(
            torch.nn.functional.cosine_similarity(
                x.float().flatten(), xr.float().flatten(), dim=0, eps=1e-12
            )
        )
        assert cos >= 0.999, f"cos {cos:.6f} < 0.999"

    def test_sqnr_above_35dB(self):
        """MECHANICAL SANITY ONLY (artificially low-rank data). SQNR clears 35 dB
        only because the signal genuinely lives in a rank-16 subspace; real
        boundary activations do not — see docs/lowrank-codec-findings.md."""
        d, r = 256, 48
        N_cal, N_test = 1024, 8
        all_x = _activation(N_cal + N_test, d, seed=42)
        cal, test = all_x[:N_cal], all_x[N_cal:]
        basis = fit_basis(cal, r=r, raw_fraction=0.005)
        codec = LowRankCodec(basis)
        x = test.half()
        xr = codec.decode(codec.encode(x))
        xf = x.float().flatten()
        xrf = xr.float().flatten()
        noise = xf - xrf
        noise_power = noise.pow(2).sum()
        signal_power = xf.pow(2).sum()
        sqnr_db = float(10.0 * torch.log10(signal_power / noise_power))
        assert sqnr_db >= 35.0, f"SQNR {sqnr_db:.1f} dB < 35 dB"

    def test_outlier_activations(self):
        """Outlier activations should still round-trip well (raw channel helps)."""
        d, r = 256, 32
        N_cal, N_test = 1024, 4
        all_x = _activation(N_cal + N_test, d, outlier=True, seed=42)
        cal, test = all_x[:N_cal], all_x[N_cal:]
        basis = fit_basis(cal, r=r, raw_fraction=0.01)
        codec = LowRankCodec(basis)
        x = test.half()
        xr = codec.decode(codec.encode(x))
        xf = x.float().flatten()
        xrf = xr.float().flatten()
        noise = xf - xrf
        noise_power = noise.pow(2).sum()
        signal_power = xf.pow(2).sum()
        sqnr_db = float(10.0 * torch.log10(signal_power / noise_power))
        cos = float(
            torch.nn.functional.cosine_similarity(xf, xrf, dim=0, eps=1e-12)
        )
        assert cos >= 0.98, f"outlier cos {cos:.6f} < 0.98"
        assert sqnr_db >= 20.0, f"outlier SQNR {sqnr_db:.1f} dB < 20 dB"

    def test_batch_dimension_handling(self):
        """Handle various batch shapes: (d,), (1, d), (B, d), (B, S, d)."""
        basis = self._make_basis(d=64, r=16)
        codec = LowRankCodec(basis)
        for shape in [(64,), (1, 64), (4, 64), (2, 3, 64)]:
            x = torch.randn(*shape, dtype=torch.float16)
            xr = codec.decode(codec.encode(x))
            assert xr.shape == x.shape, f"shape {shape}: {xr.shape} != {x.shape}"

    def test_compression_ratio(self):
        """At default r=224 with d=2560, compression should be ~4x or better."""
        d = 2560
        r = 224
        N = 512
        cal = _activation(N, d, seed=42)
        basis = fit_basis(cal, r=r, raw_fraction=0.005)
        codec = LowRankCodec(basis)
        x = _activation(4, d, seed=7).half()
        codes = codec.encode(x)
        original_bytes = x.numel() * x.element_size()  # fp16
        compressed_bytes = codes["latent"].numel() * codes["latent"].element_size()
        compressed_bytes += codes["raw_values"].numel() * codes["raw_values"].element_size()
        compressed_bytes += 4  # fp32 scale
        ratio = original_bytes / compressed_bytes
        assert ratio >= 3.0, f"compression ratio {ratio:.1f}x < 3x (expected ~4x)"
        assert ratio <= 50.0, f"compression ratio {ratio:.1f}x implausibly high"

    def test_raw_channels_are_exact(self):
        """Values at raw channel indices should match exactly (fp16 passthrough)."""
        d, r = 128, 24
        basis = self._make_basis(d=d, r=r)
        codec = LowRankCodec(basis)
        x = _activation(4, d, seed=7).half()
        xr = codec.decode(codec.encode(x))
        for idx in basis.raw_indices:
            assert torch.equal(x[..., idx], xr[..., idx]), (
                f"raw channel {idx} differs"
            )


# ── LowRankBasis.from_state ──────────────────────────────────────────────────


class TestLowRankBasis:
    def test_from_state(self):
        U = torch.randn(256, 32)
        mean = torch.randn(256)
        raw_indices = [1, 5, 10]
        basis = LowRankBasis.from_state(U, mean, raw_indices)
        assert basis.d == 256
        assert basis.r == 32
        assert basis.raw_indices == raw_indices
        assert torch.equal(basis.U, U)
        assert torch.equal(basis.mean, mean)

    def test_from_state_sorts_raw_indices(self):
        basis = LowRankBasis.from_state(
            torch.randn(128, 16), torch.randn(128), [10, 1, 5]
        )
        assert basis.raw_indices == [1, 5, 10]


# ── Evaluation lives end-to-end, not in a per-boundary unit test (issue #184) ──
#
# The tests above verify only the codec MECHANICS. The actual go/no-go decision
# for the low-rank wire codec is END-TO-END, and it cannot run in CI (it needs a
# real model): per-boundary SQNR/cos does NOT predict LM token fidelity, and a
# controlled generation test shows a low-rank PP boundary collapses next-token
# prediction even when its per-boundary SQNR/cos look healthy. The authoritative
# evaluation and reproduction are:
#   - docs/lowrank-codec-findings.md          (iso-quality table + E2E kill)
#   - bench/lowrank_activation_probe.py       (per-boundary iso-quality)
#   - bench/lowrank_e2e_probe.py              (end-to-end generation + controls)
# Low-rank is therefore NOT wired into the transport path; this module is a
# mechanically-correct reference only.


def _sqnr_cos(x: torch.Tensor, xr: torch.Tensor) -> tuple[float, float]:
    xf = x.float().flatten()
    xrf = xr.float().flatten()
    noise = (xf - xrf).pow(2).sum()
    signal = xf.pow(2).sum()
    sqnr = float(10.0 * torch.log10(signal / noise))
    cos = float(torch.nn.functional.cosine_similarity(xf, xrf, dim=0, eps=1e-12))
    return sqnr, cos


class TestLowRankIsLossyAtCompressiveRank:
    """Mechanical guard: on non-rigged (full-rank) data a compressive-rank
    projection is lossy — it is not a near-lossless codec. This is axis-neutral
    (no fidelity-vs-ratio claim); the accept/reject call is made end-to-end in
    the findings doc, not here."""

    def test_compressive_rank_is_lossy_on_fullrank_data(self):
        d, r = 256, 64  # r = d/4: a compressive operating point
        all_x = _activation(1200, d, seed=3, structured=False)  # isotropic, full-rank
        cal, test = all_x[:1024], all_x[1024:]
        codec = LowRankCodec(fit_basis(cal, r=r, raw_fraction=0.005))
        x = test.half()
        sqnr, cos = _sqnr_cos(x, codec.decode(codec.encode(x)))
        # Lossy (finite SQNR, cos below the near-lossless 0.999) — a projection to
        # r = d/4 cannot be near-lossless on full-rank activations.
        assert sqnr < 35.0 and cos < 0.999, f"expected lossy; got {sqnr:.1f} dB / cos {cos:.5f}"
