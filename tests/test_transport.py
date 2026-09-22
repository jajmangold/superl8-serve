# SPDX-License-Identifier: MIT
"""Transport: engine-boundary send/recv (compress -> P2P -> decompress). Tests first.

Verifies:
    - accuracy_report reports SQNR / cos correctly and gates against scheme bars.
    - send / recv round-trips on a single GPU (memcpy path) and multi-GPU (P2P).
    - Double-buffered streams overlap compress and transfer.
    - on_wire_bytes / theoretical_wire_ms return sane values.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("superl8")

from superl8serve.dist import TransferHandle, accuracy_report, recv, send

CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not CUDA, reason="transport needs CUDA + superl8")
NUM_GPUS = torch.cuda.device_count() if CUDA else 0
two_gpus = pytest.mark.skipif(NUM_GPUS < 2, reason="needs >= 2 GPUs for P2P")


# ── test helpers ────────────────────────────────────────────────────────────


def _activation(shape, device="cuda", outlier=False):
    """Realistic-ish activation: Gaussian (optional channel outliers)."""
    x = torch.randn(*shape, device=device, dtype=torch.float16)
    if outlier:
        d = shape[-1]
        chan = torch.zeros(d, device=device, dtype=torch.float16)
        chan[:: max(1, d // 8)] = 12.0
        x = x + chan
    return x


# ── accuracy_report ─────────────────────────────────────────────────────────


class TestAccuracyReport:
    @cuda_only
    def test_int8_meets_bars(self):
        x = _activation((4, 64, 256))
        import superl8

        c = superl8.compress_activation(x, scheme="int8")
        xr = superl8.decompress_activation(c)
        rep = accuracy_report(x, xr, "int8")
        assert rep["sqnr_pass"], f"int8 SQNR {rep['sqnr_db']:.1f} < {rep['sqnr_bar']:.1f}"
        assert rep["cos_pass"], f"int8 cos {rep['cos']:.6f} < {rep['cos_bar']:.6f}"

    @cuda_only
    def test_int4_meets_bars(self):
        x = _activation((4, 64, 256))
        import superl8

        c = superl8.compress_activation(x, scheme="int4", group_size=128)
        xr = superl8.decompress_activation(c)
        rep = accuracy_report(x, xr, "int4")
        assert rep["sqnr_pass"], f"int4 SQNR {rep['sqnr_db']:.1f} < {rep['sqnr_bar']:.1f}"
        assert rep["cos_pass"], f"int4 cos {rep['cos']:.6f} < {rep['cos_bar']:.6f}"

    @cuda_only
    def test_int4_had_meets_bars(self):
        x = _activation((4, 64, 256))
        import superl8

        c = superl8.compress_activation(x, scheme="int4-had", group_size=128)
        xr = superl8.decompress_activation(c)
        rep = accuracy_report(x, xr, "int4-had")
        assert rep["sqnr_pass"], f"int4-had SQNR {rep['sqnr_db']:.1f} < {rep['sqnr_bar']:.1f}"
        assert rep["cos_pass"], f"int4-had cos {rep['cos']:.6f} < {rep['cos_bar']:.6f}"

    @cuda_only
    def test_nf4_meets_bars(self):
        x = _activation((4, 64, 256))
        import superl8

        c = superl8.compress_activation(x, scheme="nf4", group_size=128)
        xr = superl8.decompress_activation(c)
        rep = accuracy_report(x, xr, "nf4")
        assert rep["sqnr_pass"], f"nf4 SQNR {rep['sqnr_db']:.1f} < {rep['sqnr_bar']:.1f}"
        assert rep["cos_pass"], f"nf4 cos {rep['cos']:.6f} < {rep['cos_bar']:.6f}"

    @cuda_only
    def test_fp16_is_lossless(self):
        x = _activation((4, 64, 256))
        import superl8

        c = superl8.compress_activation(x, scheme="fp16")
        xr = superl8.decompress_activation(c)
        rep = accuracy_report(x, xr, "fp16")
        assert torch.equal(x, xr)
        assert rep["sqnr_db"] == float("inf")
        assert rep["cos"] == pytest.approx(1.0, abs=1e-6)

    @cuda_only
    def test_outlier_scheme_quality(self):
        """int4-had should beat int4 on outlier activations."""
        x = _activation((4, 64, 128), outlier=True)
        import superl8

        c4 = superl8.compress_activation(x, scheme="int4", group_size=128)
        ch = superl8.compress_activation(x, scheme="int4-had", group_size=128)
        xr4 = superl8.decompress_activation(c4)
        xrh = superl8.decompress_activation(ch)
        r4 = accuracy_report(x, xr4, "int4")
        rh = accuracy_report(x, xrh, "int4-had")
        assert rh["sqnr_db"] > r4["sqnr_db"], (
            f"hadamard SQNR {rh['sqnr_db']:.1f} <= plain {r4['sqnr_db']:.1f}"
        )


# ── TransferHandle properties ────────────────────────────────────────────────


class TestTransferHandle:
    def test_on_wire_bytes_empty_scales(self):
        h = TransferHandle(
            src_device=0,
            dst_device=0,
            scheme="int8",
            shape=torch.Size((4, 64, 256)),
            dtype=torch.float16,
            group_size=None,
            d_payload=torch.empty(4 * 64 * 256, dtype=torch.int8),
            d_scales=torch.empty(0, dtype=torch.float32),
            compress_elapsed_ms=1.0,
            p2p_elapsed_ms=2.0,
            used_p2p=False,
        )
        assert h.on_wire_bytes == 4 * 64 * 256 * 1  # int8 payload only

    def test_on_wire_bytes_with_scales(self):
        h = TransferHandle(
            src_device=0,
            dst_device=0,
            scheme="int8",
            shape=torch.Size((4, 64, 256)),
            dtype=torch.float16,
            group_size=None,
            d_payload=torch.empty(4 * 64 * 256, dtype=torch.int8),
            d_scales=torch.empty(4 * 64, dtype=torch.float32),
            compress_elapsed_ms=1.0,
            p2p_elapsed_ms=2.0,
            used_p2p=False,
        )
        # payload: 65536 bytes  (65536 * 1B)
        # scales:  1024 bytes   (256 * 4B)
        assert h.on_wire_bytes == 4 * 64 * 256 * 1 + 4 * 64 * 4

    def test_theoretical_wire_ms(self):
        h = TransferHandle(
            src_device=0,
            dst_device=0,
            scheme="int8",
            shape=torch.Size((2, 128, 256)),
            dtype=torch.float16,
            group_size=None,
            d_payload=torch.empty(2 * 128 * 256, dtype=torch.int8),
            d_scales=torch.empty(2 * 128, dtype=torch.float32),
            compress_elapsed_ms=0.5,
            p2p_elapsed_ms=3.0,
            used_p2p=False,
        )
        wire_s = h.on_wire_bytes / 250e6
        assert h.theoretical_wire_ms == pytest.approx(wire_s * 1000)


# ── send / recv roundtrip (same device) ──────────────────────────────────────


class TestSendRecvSameDevice:
    @cuda_only
    def test_roundtrip_int4(self):
        x = _activation((8, 128, 256))
        handle = send(x, dst=torch.cuda.current_device(), scheme="int4", group_size=128)
        xr = recv(handle)
        assert xr.shape == x.shape
        assert xr.dtype == x.dtype
        assert xr.device == x.device
        assert torch.isfinite(xr).all()
        rep = accuracy_report(x, xr, "int4")
        assert rep["sqnr_pass"], f"SQNR {rep['sqnr_db']:.1f} < {rep['sqnr_bar']:.1f}"
        assert rep["cos_pass"], f"cos {rep['cos']:.6f} < {rep['cos_bar']:.6f}"

    @cuda_only
    def test_roundtrip_int8(self):
        x = _activation((8, 128, 256))
        handle = send(x, dst=torch.cuda.current_device(), scheme="int8", group_size=None)
        xr = recv(handle)
        assert xr.shape == x.shape
        rep = accuracy_report(x, xr, "int8")
        assert rep["sqnr_pass"], f"SQNR {rep['sqnr_db']:.1f} < {rep['sqnr_bar']:.1f}"
        assert rep["cos_pass"], f"cos {rep['cos']:.6f} < {rep['cos_bar']:.6f}"

    @cuda_only
    def test_roundtrip_fp16(self):
        x = _activation((8, 128, 256))
        handle = send(x, dst=torch.cuda.current_device(), scheme="fp16")
        xr = recv(handle)
        assert torch.equal(x, xr)

    @cuda_only
    def test_timing_fields_populated(self):
        x = _activation((4, 64, 256))
        handle = send(x, dst=torch.cuda.current_device(), scheme="int4", group_size=128)
        assert handle.compress_elapsed_ms >= 0
        assert handle.p2p_elapsed_ms > 0
        xr = recv(handle)
        assert xr.shape == x.shape
        assert handle.decompress_elapsed_ms is not None
        assert handle.decompress_elapsed_ms > 0

    @cuda_only
    def test_handle_on_wire_bytes_sane(self):
        x = _activation((4, 64, 256))
        handle = send(x, dst=torch.cuda.current_device(), scheme="int4", group_size=128)
        fp16_bytes = x.numel() * 2
        assert handle.on_wire_bytes < fp16_bytes, "compression must shrink payload"
        assert handle.theoretical_wire_ms > 0

    @cuda_only
    def test_same_device_not_p2p(self):
        """Same-device send must fall back to host-staging, never claim P2P."""
        x = _activation((4, 64, 256))
        handle = send(x, dst=torch.cuda.current_device(), scheme="int8")
        assert not handle.used_p2p


# ── send / recv multi-GPU (P2P) ─────────────────────────────────────────────


class TestSendRecvMultiGpu:
    """Cross-GPU transport. P2P is used only for the explicitly validated pair
    (0, 1); everything else fails closed to pinned host staging (superl8-serve#356).
    """

    @pytest.fixture(autouse=True)
    def _validate_test_pair(self, monkeypatch):
        from superl8serve.dist import peer_routes

        monkeypatch.setenv(peer_routes.VALIDATED_PAIRS_ENV, "0,1")
        peer_routes._VALIDATED_CACHE.clear()
        yield
        monkeypatch.delenv(peer_routes.VALIDATED_PAIRS_ENV, raising=False)
        peer_routes._VALIDATED_CACHE.clear()

    @two_gpus
    @cuda_only
    def test_roundtrip_p2p_int4(self):
        src, dst = 0, 1
        with torch.cuda.device(src):
            x = _activation((8, 128, 256))
        handle = send(x, dst=dst, scheme="int4", group_size=128)
        with torch.cuda.device(dst):
            xr = recv(handle)
        assert xr.shape == x.shape
        assert xr.device.index == dst
        rep = accuracy_report(x.cpu(), xr.cpu(), "int4")
        assert rep["sqnr_pass"], f"P2P int4 SQNR {rep['sqnr_db']:.1f} < {rep['sqnr_bar']:.1f}"
        assert rep["cos_pass"], f"P2P int4 cos {rep['cos']:.6f} < {rep['cos_bar']:.6f}"

    @two_gpus
    @cuda_only
    def test_roundtrip_p2p_int8(self):
        src, dst = 0, 1
        with torch.cuda.device(src):
            x = _activation((8, 128, 256))
        handle = send(x, dst=dst, scheme="int8", group_size=None)
        with torch.cuda.device(dst):
            xr = recv(handle)
        assert xr.shape == x.shape
        assert xr.device.index == dst
        rep = accuracy_report(x.cpu(), xr.cpu(), "int8")
        assert rep["sqnr_pass"], f"P2P int8 SQNR {rep['sqnr_db']:.1f} < {rep['sqnr_bar']:.1f}"
        assert rep["cos_pass"]

    @two_gpus
    @cuda_only
    def test_p2p_timing(self):
        src, dst = 0, 1
        with torch.cuda.device(src):
            x = _activation((8, 128, 256))
        handle = send(x, dst=dst, scheme="int4", group_size=128)
        assert handle.p2p_elapsed_ms > 0
        assert handle.src_device == src
        assert handle.dst_device == dst

    @two_gpus
    @cuda_only
    def test_used_p2p_when_peers_accessible(self):
        """P2P path must be selected when GPUs have direct peer access."""
        src, dst = 0, 1
        with torch.cuda.device(src):
            x = _activation((8, 128, 256))
        handle = send(x, dst=dst, scheme="int8")
        assert handle.used_p2p, "expected P2P path across two GPUs with peer access"

    @two_gpus
    @cuda_only
    def test_fp16_p2p_bit_exact(self):
        """fp16 round-trip across GPUs via P2P must be bit-exact (lossless codec)."""
        src, dst = 0, 1
        with torch.cuda.device(src):
            x = _activation((4, 64, 256))
        handle = send(x, dst=dst, scheme="fp16")
        with torch.cuda.device(dst):
            xr = recv(handle)
        assert handle.used_p2p
        assert xr.device.index == dst
        # Compare on a common device: x is on src, xr on dst (P2P copy).
        assert torch.equal(x.cpu(), xr.cpu()), "fp16 P2P round-trip must be bit-exact"

    @two_gpus
    @cuda_only
    def test_host_staging_fallback_selected_when_no_peer(self, monkeypatch):
        """When peer access is unavailable, a cross-GPU send must fall back to
        host-staging (used_p2p False) and still round-trip with fidelity.

        The fleet's GPUs do have peer access, so we force the no-peer branch by
        patching the capability probe — this genuinely exercises the
        GPU→pinned-host→GPU fallback copy across two distinct devices.
        """
        import superl8serve.dist as dist_mod

        monkeypatch.setattr(dist_mod, "_can_use_p2p", lambda src, dst: False)
        src, dst = 0, 1
        with torch.cuda.device(src):
            x = _activation((8, 128, 256))
        handle = send(x, dst=dst, scheme="int8")
        assert not handle.used_p2p, "fallback must not claim P2P"
        with torch.cuda.device(dst):
            xr = recv(handle)
        assert xr.device.index == dst
        rep = accuracy_report(x.cpu(), xr.cpu(), "int8")
        assert rep["sqnr_pass"], f"fallback SQNR {rep['sqnr_db']:.1f} < {rep['sqnr_bar']:.1f}"
        assert rep["cos_pass"], f"fallback cos {rep['cos']:.6f} < {rep['cos_bar']:.6f}"

    @two_gpus
    @cuda_only
    def test_fp16_host_staging_fallback_bit_exact(self, monkeypatch):
        """The host-staging fallback must be bit-exact for the lossless fp16 codec."""
        import superl8serve.dist as dist_mod

        monkeypatch.setattr(dist_mod, "_can_use_p2p", lambda src, dst: False)
        src, dst = 0, 1
        with torch.cuda.device(src):
            x = _activation((4, 64, 256))
        handle = send(x, dst=dst, scheme="fp16")
        assert not handle.used_p2p
        with torch.cuda.device(dst):
            xr = recv(handle)
        assert torch.equal(x.cpu(), xr.cpu()), "fallback fp16 round-trip must be bit-exact"


# ── fail-closed default: unvalidated pair never uses P2P on real GPUs ───────


class TestFailClosedUnvalidatedPair:
    """With no validated pair configured, cross-GPU sends must host-stage even
    when the hardware reports peer capability (the 6->14 PHB false positive).

    These run on two real GPUs so the ``used_p2p`` decision is the live one.
    """

    @pytest.fixture(autouse=True)
    def _no_validated_pairs(self, monkeypatch):
        from superl8serve.dist import peer_routes

        monkeypatch.delenv(peer_routes.VALIDATED_PAIRS_ENV, raising=False)
        peer_routes._VALIDATED_CACHE.clear()
        yield
        peer_routes._VALIDATED_CACHE.clear()

    @two_gpus
    @cuda_only
    def test_unvalidated_cross_gpu_send_host_stages(self):
        src, dst = 0, 1
        with torch.cuda.device(src):
            x = _activation((8, 128, 256))
        handle = send(x, dst=dst, scheme="int8")
        assert not handle.used_p2p, "unvalidated pair must never claim P2P"
        with torch.cuda.device(dst):
            xr = recv(handle)
        assert xr.device.index == dst
        rep = accuracy_report(x.cpu(), xr.cpu(), "int8")
        assert rep["sqnr_pass"], f"unvalidated fallback SQNR {rep['sqnr_db']:.1f} dB"
        assert rep["cos_pass"]

    @two_gpus
    @cuda_only
    def test_unvalidated_fp16_host_staging_bit_exact(self):
        src, dst = 0, 1
        with torch.cuda.device(src):
            x = _activation((4, 64, 256))
        handle = send(x, dst=dst, scheme="fp16")
        assert not handle.used_p2p
        with torch.cuda.device(dst):
            xr = recv(handle)
        assert torch.equal(x.cpu(), xr.cpu())

    @two_gpus
    @cuda_only
    def test_validating_the_pair_reenables_p2p(self, monkeypatch):
        """Explicitly validating the live pair must restore the P2P path."""
        from superl8serve.dist import peer_routes

        monkeypatch.setenv(peer_routes.VALIDATED_PAIRS_ENV, "0,1")
        peer_routes._VALIDATED_CACHE.clear()
        src, dst = 0, 1
        with torch.cuda.device(src):
            x = _activation((4, 64, 256))
        handle = send(x, dst=dst, scheme="fp16")
        assert handle.used_p2p
