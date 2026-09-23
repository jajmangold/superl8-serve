# SPDX-License-Identifier: MIT
"""Fail-closed GPU peer-route selection for the fni8 transport (fni8-serve#356).

``torch.cuda.can_device_access_peer()`` is NOT trustworthy on the CMP 100-210
fleet: physical GPU 6->14 reports peer support (a PHB hop) yet the
``cudaMemcpyPeer`` copy silently zero-fills the destination. This module owns the
*route selection* decision so that a transport only attempts P2P when the
(src, dst) pair is explicitly validated, and fails closed to pinned host staging
for every unvalidated, unknown, or same-device route.

Ordinal contract
----------------
``src``/``dst`` here — and therefore every value in ``FNI8_P2P_VALIDATED_PAIRS`` —
are the **torch CUDA device ordinals exactly as seen by the serving process**
(``tensor.device.index`` / ``stage._device.index``). They are NOT physical host
``nvidia-smi`` indices. A container launched with ``--gpus '"device=5,6"'`` maps
physical 5/6 to ordinals 0/1, so the serving env must be ``FNI8_P2P_VALIDATED_PAIRS=0,1``
for that shape; an all-visible process that keeps physical indices may write ``5,6``.
Physical indices/UUIDs are recorded as qualification provenance only
(``tools/qualify_peer_route.py``, ``docs/evidence/356``), never as routing input.

Policy
------
A route may use P2P only when ALL of:

1. ``src != dst`` (same-device copies are staged, never P2P).
2. the unordered pair ``{src, dst}`` is present in the validated set; and
3. ``torch.cuda.can_device_access_peer(src, dst)`` is True.

The validated set is configuration, not a runtime discovery result: it comes from
``FNI8_P2P_VALIDATED_PAIRS`` (a ``;``-separated list of ``src,dst`` pairs), and an
empty/unset value means *no* pair is P2P-eligible. Nothing here sweeps devices or
probes live GPUs at runtime — qualification is an offline, exact-pair step (see
``tools/qualify_peer_route.py``).

This keeps the fleet's staged-only doctrine: direct same-switch P2P stays enabled
for the validated same-switch pair (physical 5/6 = ordinals 0/1 in a restricted
container), while the PHB hop physical 6->14 (which reports capability but
corrupts) always host-stages.
"""

from __future__ import annotations

import os

import torch

# Environment variable naming the validated (src, dst) pairs eligible for P2P.
# Format: "0,1;1,2" — pairs separated by ';', indices by ','. Values are torch
# CUDA ordinals as seen by the serving process (NOT physical nvidia-smi indices).
# Empty/unset = fail closed.
VALIDATED_PAIRS_ENV = "FNI8_P2P_VALIDATED_PAIRS"

# Parsed validated-set cache, keyed by the exact env value so a change
# invalidates it automatically (also used by tests via monkeypatch).
_VALIDATED_CACHE: dict[str, frozenset[tuple[int, int]]] = {}

_ROUTE_P2P = "validated-p2p"
_ROUTE_SAME_DEVICE = "same-device"
_ROUTE_NOT_VALIDATED = "not-validated"
_ROUTE_NO_CAPABILITY = "no-peer-capability"


def _normalize_pair(a: int, b: int) -> tuple[int, int]:
    if a < 0 or b < 0:
        raise ValueError(f"GPU index must be >= 0: ({a}, {b})")
    if a == b:
        raise ValueError(f"a peer pair cannot reference the same GPU: ({a}, {b})")
    return (a, b) if a < b else (b, a)


def parse_validated_pairs(spec: str | None) -> frozenset[tuple[int, int]]:
    """Parse the ``FNI8_P2P_VALIDATED_PAIRS`` spec into a set of unordered pairs.

    Pairs are separated by ``;`` and each pair is ``src,dst``. Order within a pair
    and across pairs is irrelevant (selection is direction-agnostic). Whitespace is
    tolerated around tokens. A ``None``/empty spec is the empty set (fail closed).
    Malformed specs raise ``ValueError`` rather than silently ignoring a route.
    """
    if not spec or not spec.strip():
        return frozenset()
    chunks = [c.strip() for c in spec.split(";")]
    if all(not c for c in chunks):
        return frozenset()
    if any(not c for c in chunks):
        raise ValueError(
            "malformed P2P validated-pairs spec: empty ';'-separated element"
        )
    pairs: set[tuple[int, int]] = set()
    for chunk in chunks:
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                f"malformed P2P validated pair {chunk!r}; expected 'src,dst'"
            )
        try:
            a, b = (int(parts[0]), int(parts[1]))
        except ValueError as exc:
            raise ValueError(
                f"malformed P2P validated pair {chunk!r}; indices must be integers"
            ) from exc
        pairs.add(_normalize_pair(a, b))
    return frozenset(pairs)


def validated_pairs() -> frozenset[tuple[int, int]]:
    """The validated P2P-eligible pair set from the environment (cached)."""
    spec = os.environ.get(VALIDATED_PAIRS_ENV, "")
    if spec not in _VALIDATED_CACHE:
        _VALIDATED_CACHE[spec] = parse_validated_pairs(spec)
    return _VALIDATED_CACHE[spec]


def _torch_can_peer(src: int, dst: int) -> bool:
    return bool(torch.cuda.can_device_access_peer(src, dst))


def peer_capability(src: int, dst: int) -> bool:
    """Probe ``cudaDeviceCanAccessPeer``, treating any error as ``False``.

    A failure to *query* capability must never widen the route set, so every
    exception collapses to ``False`` (fail closed).
    """
    try:
        return _torch_can_peer(src, dst)
    except (RuntimeError, AttributeError):
        return False


def select_route(
    src: int,
    dst: int,
    *,
    validated: frozenset[tuple[int, int]],
    capability: bool | None,
) -> tuple[bool, str]:
    """Pure, deterministic route-selection predicate.

    Parameters
    ----------
    src, dst:
        Source / destination **torch CUDA device ordinals as seen by the serving
        process** (not physical ``nvidia-smi`` indices).
    validated:
        The validated pair set (see :func:`validated_pairs`).
    capability:
        The ``cudaDeviceCanAccessPeer`` result, or ``None`` when unknown. An
        unknown capability never selects P2P (fail closed).

    Returns
    -------
    (allowed, reason):
        ``allowed`` is True only when the route may use P2P; ``reason`` is a
        stable code for the decision (``validated-p2p``, ``same-device``,
        ``not-validated``, ``no-peer-capability``) for telemetry / diagnostics.
    """
    if src == dst:
        return False, _ROUTE_SAME_DEVICE
    pair = _normalize_pair(src, dst)
    if pair not in validated:
        return False, _ROUTE_NOT_VALIDATED
    if capability is not True:
        return False, _ROUTE_NO_CAPABILITY
    return True, _ROUTE_P2P


def peer_route_allowed(
    src: int, dst: int, *, capability: bool | None = None
) -> tuple[bool, str]:
    """Runtime route-selection entry point: env validated-set + live capability.

    ``capability`` may be injected for deterministic tests; when ``None`` the live
    ``torch.cuda.can_device_access_peer`` probe is used (and any probe error fails
    closed to host staging).
    """
    if capability is None:
        capability = peer_capability(src, dst)
    return select_route(
        src, dst, validated=validated_pairs(), capability=capability
    )
