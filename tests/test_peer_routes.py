# SPDX-License-Identifier: MIT
"""Fail-closed peer-route selection for the superl8 transport (superl8-serve#356).

torch.cuda.can_device_access_peer() is not trustworthy on the CMP 100-210 fleet:
physical GPU 6->14 reports peer support (PHB hop) yet cudaMemcpyPeer silently
zero-fills the destination. A route may use P2P only when its (src, dst) pair is
explicitly validated. Unvalidated / unknown / same-device routes fail closed to
pinned host staging.

These tests are deterministic and CPU-only: the parser and the pure selection
predicate never touch CUDA, so the fail-closed policy is fully unit-covered
without the fleet. The known fleet evidence is encoded here directly:

    - same-switch 5 -> 6: validated, P2P allowed
    - PHB 6 -> 14: reports capability, must still fall back to host staging
"""

from __future__ import annotations

import pytest

from superl8serve.dist import peer_routes

_ENV = peer_routes.VALIDATED_PAIRS_ENV


def _clean_env(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    peer_routes._VALIDATED_CACHE.clear()


# ── ordinal contract: env values are torch CUDA ordinals, not physical indices ─


class TestOrdinalContract:
    """The env spec is consumed by ``send()`` using ``tensor.device.index`` /
    ``stage._device.index`` — torch CUDA ordinals as seen by the serving process.
    Guard the module docstring so this contract can never be silently rewritten
    (superl8-serve#356 regression: examples implied physical ``nvidia-smi`` indices)."""

    def test_module_docstring_uses_ordinal_terminology(self):
        doc = peer_routes.__doc__
        assert "ordinals" in doc.lower(), "module docstring must state the ordinal contract"
        assert "not physical" in doc.lower(), (
            "module docstring must warn that env values are NOT physical nvidia-smi indices"
        )

    def test_env_comment_example_is_ordinal_form(self):
        source = open(peer_routes.__file__, encoding="utf-8").read()
        assert '"0,1;1,2"' in source, (
            "env format comment must show ordinal-form example (0,1), not physical 5,6"
        )
        assert "CUDA ordinals" in source


# ── parser: deterministic, CPU-only ─────────────────────────────────────────


class TestParseValidatedPairs:
    @pytest.mark.parametrize("spec", [None, "", "  ", ";", ";;"])
    def test_empty_spec_is_empty_set(self, spec):
        assert peer_routes.parse_validated_pairs(spec) == frozenset()

    def test_single_pair(self):
        assert peer_routes.parse_validated_pairs("5,6") == frozenset({(5, 6)})

    def test_multiple_pairs(self):
        assert peer_routes.parse_validated_pairs("5,6;6,7;5,7") == frozenset(
            {(5, 6), (6, 7), (5, 7)}
        )

    def test_pair_order_normalized(self):
        """Reversed order must normalize so selection is direction-agnostic."""
        assert peer_routes.parse_validated_pairs("6,5") == frozenset({(5, 6)})

    def test_whitespace_tolerated(self):
        assert peer_routes.parse_validated_pairs(" 5, 6 ; 6,7 ") == frozenset({(5, 6), (6, 7)})

    @pytest.mark.parametrize(
        "bad",
        ["5", "a,b", "5,", ",5", "5,6,7", "5,6;", "5,6;x", "5,6;7,8,9", "-1,5"],
    )
    def test_malformed_spec_raises(self, bad):
        with pytest.raises(ValueError):
            peer_routes.parse_validated_pairs(bad)

    def test_duplicate_pairs_collapsed(self):
        assert peer_routes.parse_validated_pairs("5,6;6,5") == frozenset({(5, 6)})

    def test_same_device_pair_rejected(self):
        with pytest.raises(ValueError):
            peer_routes.parse_validated_pairs("5,5")


# ── pure selection predicate: fail-closed ───────────────────────────────────


class TestSelectRoute:
    def test_same_device_never_p2p_even_when_validated_and_capable(self):
        allowed, reason = peer_routes.select_route(
            5, 5, validated=frozenset({(5, 6)}), capability=True
        )
        assert not allowed
        assert reason == "same-device"

    def test_unvalidated_pair_never_p2p_even_when_capability_true(self):
        """THE regression: a false-positive capability must not enable P2P.

        Fleet evidence: GPU 6 -> 14 reports peer support but zero-fills. With the
        pair absent from the validated set, selection must fall back to host
        staging regardless of what can_device_access_peer claims.
        """
        allowed, reason = peer_routes.select_route(
            6, 14, validated=frozenset({(5, 6)}), capability=True
        )
        assert not allowed
        assert reason == "not-validated"

    def test_validated_pair_with_true_capability_uses_p2p(self):
        allowed, reason = peer_routes.select_route(
            5, 6, validated=frozenset({(5, 6)}), capability=True
        )
        assert allowed
        assert reason == "validated-p2p"

    def test_validated_pair_with_false_capability_host(self):
        allowed, _ = peer_routes.select_route(
            5, 6, validated=frozenset({(5, 6)}), capability=False
        )
        assert not allowed

    def test_validated_pair_with_unknown_capability_fails_closed(self):
        """An unspecified capability (None) must not default to P2P."""
        allowed, reason = peer_routes.select_route(
            5, 6, validated=frozenset({(5, 6)}), capability=None
        )
        assert not allowed
        assert reason == "no-peer-capability"

    def test_unvalidated_unknown_capability_host(self):
        allowed, reason = peer_routes.select_route(
            6, 14, validated=frozenset(), capability=None
        )
        assert not allowed
        assert reason == "not-validated"

    def test_empty_validated_set_fails_closed_everywhere(self):
        """Default configuration (nothing validated) disables all cross-GPU P2P."""
        for src, dst, cap in [(0, 1, True), (5, 6, True), (6, 14, True), (1, 0, True)]:
            allowed, reason = peer_routes.select_route(src, dst, validated=frozenset(), capability=cap)
            assert not allowed
            assert reason == "not-validated"

    def test_reversed_route_uses_same_validation(self):
        """Direction-agnostic: 6 -> 5 is governed by the same (5, 6) pair."""
        allowed, _ = peer_routes.select_route(
            6, 5, validated=frozenset({(5, 6)}), capability=True
        )
        assert allowed


# ── deterministic chained / multi-hop coverage (matches fleet topology) ──────


class TestChainedMultiHop:
    def test_same_switch_chain_all_validated(self):
        """5 -> 6 -> 7 with both pairs validated: every hop may use P2P."""
        validated = frozenset({(5, 6), (6, 7)})
        h1, _ = peer_routes.select_route(5, 6, validated=validated, capability=True)
        h2, _ = peer_routes.select_route(6, 7, validated=validated, capability=True)
        assert h1 and h2

    def test_partially_validated_chain_hosts_unvalidated_hop(self):
        """Only the same-switch hop is validated; the PHB hop must host-stage."""
        validated = frozenset({(5, 6)})
        h1, _ = peer_routes.select_route(5, 6, validated=validated, capability=True)
        h2, reason2 = peer_routes.select_route(6, 14, validated=validated, capability=True)
        assert h1
        assert not h2
        assert reason2 == "not-validated"

    def test_three_stage_cross_switch_chain(self):
        """A 3-stage pipeline 0->1->2 where only the first hop is validated."""
        validated = frozenset({(0, 1)})
        h1, _ = peer_routes.select_route(0, 1, validated=validated, capability=True)
        h2, reason2 = peer_routes.select_route(1, 2, validated=validated, capability=True)
        assert h1
        assert not h2
        assert reason2 == "not-validated"

    def test_capability_false_breaks_otherwise_validated_chain(self):
        validated = frozenset({(5, 6), (6, 7)})
        h1, _ = peer_routes.select_route(5, 6, validated=validated, capability=True)
        h2, reason2 = peer_routes.select_route(6, 7, validated=validated, capability=False)
        assert h1
        assert not h2
        assert reason2 == "no-peer-capability"


# ── configuration seam (env var) ────────────────────────────────────────────


class TestEnvConfiguration:
    def test_env_sets_validated_pairs(self, monkeypatch):
        _clean_env(monkeypatch)
        monkeypatch.setenv(_ENV, "5,6;6,7")
        assert peer_routes.validated_pairs() == frozenset({(5, 6), (6, 7)})

    def test_env_changes_invalidate_cache(self, monkeypatch):
        _clean_env(monkeypatch)
        monkeypatch.setenv(_ENV, "5,6")
        assert peer_routes.validated_pairs() == frozenset({(5, 6)})
        monkeypatch.setenv(_ENV, "7,8")
        assert peer_routes.validated_pairs() == frozenset({(7, 8)})

    def test_unset_env_is_fail_closed(self, monkeypatch):
        _clean_env(monkeypatch)
        assert peer_routes.validated_pairs() == frozenset()

    def test_malformed_env_raises(self, monkeypatch):
        _clean_env(monkeypatch)
        monkeypatch.setenv(_ENV, "bogus")
        with pytest.raises(ValueError):
            peer_routes.validated_pairs()

    def test_env_drives_full_selection(self, monkeypatch):
        _clean_env(monkeypatch)
        monkeypatch.setenv(_ENV, "5,6")
        allowed, reason = peer_routes.peer_route_allowed(
            6, 14, capability=True
        )
        assert not allowed
        assert reason == "not-validated"
        allowed, _ = peer_routes.peer_route_allowed(5, 6, capability=True)
        assert allowed


# ── capability probe wrapper: never lets an exception enable P2P ─────────────


class TestPeerCapability:
    def test_returns_false_when_torch_raises(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("probe unavailable")

        monkeypatch.setattr(peer_routes, "_torch_can_peer", boom)
        assert peer_routes.peer_capability(0, 1) is False

    def test_returns_false_on_attribute_error(self, monkeypatch):
        def missing(*args, **kwargs):
            raise AttributeError("no such API")

        monkeypatch.setattr(peer_routes, "_torch_can_peer", missing)
        assert peer_routes.peer_capability(0, 1) is False

    def test_propagates_true(self, monkeypatch):
        monkeypatch.setattr(peer_routes, "_torch_can_peer", lambda a, b: True)
        assert peer_routes.peer_capability(0, 1) is True
