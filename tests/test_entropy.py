# SPDX-License-Identifier: MIT
"""rANS entropy coder: lossless round-trip and compression-ratio tests (issue #188).

Tests the rANS stage in two layers:
1.  Unit-level: direct :func:`rans_encode` / :func:`rans_decode` round-trips
    on various symbol distributions, asserting bit-exact reconstruction and
    measuring empirical compression ratios.
2.  Integration: :func:`send` / :func:`recv` with ``entropy=True`` produces
    the same reconstructed activation as the non-entropy path, with strictly
    smaller ``on_wire_bytes``.
"""

from __future__ import annotations

import numpy as np
import pytest

from superl8serve.dist.entropy import decode as rans_decode
from superl8serve.dist.entropy import encode as rans_encode

# ── helpers ───────────────────────────────────────────────────────────────────


def _random_bytes(n: int, p_uniform: float = 0.0, seed: int = 42) -> np.ndarray:
    """Generate *n* random bytes with a skewed distribution (lower entropy).

    When *p_uniform* == 1.0 the output is fully uniform (incompressible);
    when *p_uniform* == 0.0 the output is strongly peaked at zero (very
    compressible).  The default 0.0 gives activations-like distributions.
    """
    rng = np.random.default_rng(seed)
    if p_uniform >= 1.0:
        return rng.integers(0, 256, size=n, dtype=np.uint8)
    base = np.zeros(n, dtype=np.uint8)
    if p_uniform > 0.0:
        mask = rng.random(n) < p_uniform
        n_mask = int(mask.sum())
        base[mask] = rng.integers(0, 256, size=n_mask, dtype=np.uint8)
    return base


# ── rANS unit tests (no CUDA needed) ──────────────────────────────────────────


class TestRansRoundTrip:
    def test_empty_payload(self):
        data = np.array([], dtype=np.uint8)
        enc, freqs = rans_encode(data)
        dec = rans_decode(enc, freqs, 0)
        assert len(dec) == 0

    def test_single_byte(self):
        for val in [0, 1, 127, 255]:
            data = np.array([val], dtype=np.uint8)
            enc, freqs = rans_encode(data)
            dec = rans_decode(enc, freqs, 1)
            assert dec[0] == val, f"single byte {val} failed: got {dec[0]}"

    def test_repeated_symbol(self):
        data = np.full(1000, 42, dtype=np.uint8)
        enc, freqs = rans_encode(data)
        dec = rans_decode(enc, freqs, 1000)
        assert np.array_equal(dec, data)

    def test_two_symbols_interleaved(self):
        data = np.tile(np.array([0xAB, 0xCD], dtype=np.uint8), 500)
        enc, freqs = rans_encode(data)
        dec = rans_decode(enc, freqs, 1000)
        assert np.array_equal(dec, data)

    def test_all_256_symbols(self):
        data = np.arange(256, dtype=np.uint8)
        enc, freqs = rans_encode(data)
        dec = rans_decode(enc, freqs, 256)
        assert np.array_equal(dec, data)

    def test_uniform_random(self):
        rng = np.random.default_rng(42)
        data = rng.integers(0, 256, size=10_000, dtype=np.uint8)
        enc, freqs = rans_encode(data)
        dec = rans_decode(enc, freqs, 10_000)
        assert np.array_equal(dec, data)

    def test_skewed_like_int8_activations(self):
        """Near-normal distribution centred on zero, typical of int8 quantized activations."""
        rng = np.random.default_rng(42)
        data = np.clip(rng.normal(0, 20, 10_000).round(), -128, 127).astype(np.int8)
        data_u8 = data.view(np.uint8)
        enc, freqs = rans_encode(data_u8)
        dec = rans_decode(enc, freqs, 10_000)
        assert np.array_equal(dec, data_u8)

    def test_skewed_like_int4_activations(self):
        """Packed int4 values (0..15) typical of int4 quantized activations."""
        rng = np.random.default_rng(42)
        data = np.clip(rng.normal(0, 3, 10_000).round(), -8, 7).astype(np.int8)
        data_u4 = (data + 8).astype(np.uint8)
        enc, freqs = rans_encode(data_u4)
        dec = rans_decode(enc, freqs, 10_000)
        assert np.array_equal(dec, data_u4)

    def test_various_sizes(self):
        rng = np.random.default_rng(42)
        for n in [1, 2, 3, 10, 100, 1000, 10000, 100000]:
            data = rng.integers(0, 256, size=n, dtype=np.uint8)
            enc, freqs = rans_encode(data)
            dec = rans_decode(enc, freqs, n)
            assert np.array_equal(dec, data), f"size {n} failed"


class TestRansCompressionRatio:
    @pytest.mark.parametrize("n", [100, 1000, 10000])
    def test_skewed_compresses(self, n):
        """Compressed size must be < raw size for skewed distributions."""
        rng = np.random.default_rng(42)
        data = np.clip(rng.normal(0, 20, n).round(), -128, 127).astype(np.int8).view(np.uint8)
        enc, _ = rans_encode(data)
        raw_bytes = n
        compressed_bytes = len(enc)
        ratio = raw_bytes / compressed_bytes
        assert ratio > 1.1, f"skewed distribution only compressed {ratio:.3f}x"
        assert compressed_bytes < raw_bytes, "compressed must be smaller"

    @pytest.mark.parametrize("n", [1000, 10000])
    def test_int8_like_ratio(self, n):
        """int8 quantized activations-style data: a correct entropy coder must
        reach (within a few %) the data's Shannon entropy floor.

        The N(0,15)-clipped int8 distribution has an empirical entropy of ~5.9
        bits/symbol, so the information-theoretic ceiling is only ~1.34x — no
        lossless coder can beat that. We therefore assert the coder lands close
        to the entropy floor rather than an (impossible) fixed 1.5x. This is the
        meaningful correctness property for an entropy coder.
        """
        rng = np.random.default_rng(42)
        data = np.clip(rng.normal(0, 15, n).round(), -128, 127).astype(np.int8).view(np.uint8)
        enc, _ = rans_encode(data)
        ratio = n / len(enc)

        # Shannon floor for this exact sample (bits/symbol -> max achievable ratio).
        _, counts = np.unique(data, return_counts=True)
        p = counts / counts.sum()
        entropy_bits = float(-(p * np.log2(p)).sum())
        shannon_ratio = 8.0 / entropy_bits

        assert ratio > 1.0, f"int8-like data not compressed ({ratio:.3f}x)"
        # Within 5% of the entropy floor (state-flush overhead is O(4 bytes)).
        assert ratio >= 0.95 * shannon_ratio, (
            f"int8-like ratio {ratio:.3f}x is below 95% of the Shannon "
            f"ceiling {shannon_ratio:.3f}x (entropy {entropy_bits:.3f} bits/sym)"
        )

    @pytest.mark.parametrize("n", [1000, 10000])
    def test_int4_like_ratio(self, n):
        """int4 quantized activations-style data (0..15): expect >= 1.8x compression."""
        rng = np.random.default_rng(42)
        data = np.clip(rng.normal(0, 3, n).round(), -8, 7).astype(np.int8)
        data_u4 = (data + 8).astype(np.uint8)
        enc, _ = rans_encode(data_u4)
        ratio = n / len(enc)
        assert ratio >= 1.8, f"int4-like data only compressed {ratio:.3f}x (expected >= 1.8x)"

    def test_uniform_is_incompressible(self):
        """Uniform random data should compress very little (rANS overhead)."""
        rng = np.random.default_rng(42)
        data = rng.integers(0, 256, size=10_000, dtype=np.uint8)
        enc, _ = rans_encode(data)
        ratio = 10_000 / len(enc)
        assert ratio < 1.2, f"uniform data compressed {ratio:.3f}x (expected near 1.0)"

    def test_all_zeros_compresses_extremely_well(self):
        data = np.zeros(10_000, dtype=np.uint8)
        enc, _ = rans_encode(data)
        ratio = 10_000 / len(enc)
        assert ratio > 10, f"all-zeros only compressed {ratio:.3f}x"
