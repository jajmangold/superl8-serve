# SPDX-License-Identifier: MIT
"""rANS (range Asymmetric Numeral System) entropy coder (issue #188).

Losslessly compresses quantized code payloads using their empirical symbol
distribution. Designed as a composable post-quantization stage: encodes the
int8/uint8 payload produced by the wire codec into a smaller byte stream that
the decoder reconstructs bit-exactly before dequantization.

Implementation notes
--------------------
This is a static, byte-normalized 32-bit rANS (the ryg_rans design):

* Symbol frequencies are **normalized so they sum to exactly ``M = 2**SCALE_BITS``**
  (a power of two). Every symbol that actually occurs keeps ``freq >= 1`` so it
  stays decodable. This is the invariant the earlier draft got wrong — it used the
  raw counts (whose sum is not a power of two) as ``M`` and renormalized against
  ``M`` rather than the per-symbol frequency, which corrupted low-entropy inputs
  (e.g. an all-42s buffer) and could overflow the 4-byte final-state flush.
* The coder state lives in the normalization interval ``[L, L << 8)`` with
  ``L = 2**23``, so it always fits in 32 bits and flushes/loads as exactly 4 bytes.
* Encode renormalizes against ``freq << (23 - SCALE_BITS + 8)`` (i.e. the standard
  ``x_max = ((L >> SCALE_BITS) << 8) * freq``); decode inverts it exactly.

Reference: J. Duda, "Asymmetric Numeral Systems", 2009.
           Fabian Giesen, "rANS notes" / ryg_rans reference implementation.
"""

from __future__ import annotations

import numpy as np

_ALPHABET_SIZE = 256

# Total normalized frequency mass M = 2**SCALE_BITS. 12 bits (M = 4096) gives
# ample headroom over the 256-symbol alphabet (every symbol can carry freq >= 1
# with room for a skewed distribution) while keeping the freq table small.
_SCALE_BITS = 12
_M = 1 << _SCALE_BITS
_MASK = _M - 1

# Normalization interval lower bound L = 2**23; state stays in [L, L << 8) so it
# is always a 32-bit value flushed as 4 bytes.
_L = 1 << 23
_STATE_BYTES = 4

# Precomputed factor for the encode renorm bound: x_max(s) = _X_MAX_FACTOR * freq(s).
#   x_max = ((L >> SCALE_BITS) << 8) * freq
_X_MAX_FACTOR = (_L >> _SCALE_BITS) << 8


def _normalize_freqs(counts: np.ndarray) -> np.ndarray:
    """Scale raw symbol counts to normalized frequencies summing to exactly ``_M``.

    Every symbol with a nonzero count is guaranteed ``freq >= 1`` (so it remains
    decodable); the total is corrected to land on ``_M`` exactly by nudging the
    highest-count symbols. The returned table is int32, length 256.
    """
    counts = counts.astype(np.int64)
    total = int(counts.sum())
    freqs = np.zeros(_ALPHABET_SIZE, dtype=np.int64)
    if total == 0:
        return freqs.astype(np.int32)

    present = counts > 0
    # Proportional allocation (floor), then bump every present symbol to >= 1.
    scaled = (counts * _M) // total
    freqs[present] = np.maximum(scaled[present], 1)

    # Correct the rounding drift so the frequencies sum to exactly _M. Distribute
    # the difference over the most frequent symbols, never dropping any below 1.
    diff = _M - int(freqs.sum())
    if diff != 0:
        order = np.argsort(-counts)  # symbols by descending raw count
        n_present = int(present.sum())
        i = 0
        # order[:n_present] are exactly the present symbols (zero-count sort last).
        while diff != 0:
            s = int(order[i % n_present])
            if diff > 0:
                freqs[s] += 1
                diff -= 1
            elif freqs[s] > 1:
                freqs[s] -= 1
                diff += 1
            i += 1

    return freqs.astype(np.int32)


def _cumulative(freqs: np.ndarray) -> np.ndarray:
    """Exclusive prefix sums (``cum[s]`` = start slot of symbol ``s``), length 257."""
    cum = np.zeros(_ALPHABET_SIZE + 1, dtype=np.int64)
    np.cumsum(freqs.astype(np.int64), out=cum[1:])
    return cum


def encode(symbols: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Encode a 1-D uint8 symbol array with byte-normalized static rANS.

    Returns
    -------
    (encoded_bytes, freq_table):
        The compressed byte stream and the 256-element normalized frequency
        table (int32, summing to ``2**SCALE_BITS``) required for decoding.
    """
    symbols = np.ascontiguousarray(symbols, dtype=np.uint8).ravel()
    if symbols.size == 0:
        return b"", np.zeros(_ALPHABET_SIZE, dtype=np.int32)

    counts = np.bincount(symbols, minlength=_ALPHABET_SIZE).astype(np.int64)
    freqs = _normalize_freqs(counts)
    cum = _cumulative(freqs)

    freqs_i = freqs.tolist()
    cum_i = cum.tolist()

    state = _L
    out = bytearray()

    # Encode in reverse so the decoder recovers symbols in forward order.
    for s in symbols[::-1].tolist():
        f = freqs_i[s]
        start = cum_i[s]
        x_max = _X_MAX_FACTOR * f
        while state >= x_max:
            out.append(state & 0xFF)
            state >>= 8
        state = ((state // f) << _SCALE_BITS) + (state % f) + start

    # Flush the final 32-bit state, low byte first.
    for _ in range(_STATE_BYTES):
        out.append(state & 0xFF)
        state >>= 8

    return bytes(out), freqs


def decode(encoded: bytes, freqs: np.ndarray, n: int) -> np.ndarray:
    """Decode *n* symbols from a rANS byte stream produced by :func:`encode`.

    Returns a 1-D uint8 numpy array of length *n*. ``freqs`` must be the
    normalized table returned by :func:`encode` (summing to ``2**SCALE_BITS``).
    """
    if n == 0:
        return np.empty(0, dtype=np.uint8)

    freqs = np.asarray(freqs, dtype=np.int64)
    cum = _cumulative(freqs)
    # slot -> symbol lookup (length _M): the inverse of the CDF.
    slot_to_sym = np.repeat(
        np.arange(_ALPHABET_SIZE, dtype=np.uint8), freqs.astype(np.int64)
    )
    assert slot_to_sym.size == _M, "normalized frequencies must sum to 2**SCALE_BITS"

    freqs_i = freqs.tolist()
    cum_i = cum.tolist()
    slot_to_sym_i = slot_to_sym.tolist()

    buf = bytes(encoded)
    pos = len(buf) - _STATE_BYTES

    # Load the initial 32-bit state (low byte was written first).
    state = 0
    for i in range(_STATE_BYTES):
        state |= buf[pos + i] << (8 * i)

    result_list = [0] * n

    for i in range(n):
        slot = state & _MASK
        s = slot_to_sym_i[slot]
        f = freqs_i[s]
        start = cum_i[s]
        state = f * (state >> _SCALE_BITS) + slot - start
        while state < _L:
            pos -= 1
            state = (state << 8) | buf[pos]
        result_list[i] = s

    return np.asarray(result_list, dtype=np.uint8)
