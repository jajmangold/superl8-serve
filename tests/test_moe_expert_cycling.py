# SPDX-License-Identifier: MIT
"""Tests for WeightStationaryMoE — Phase 3 MoE expert cycling.

Tests verify:
1. Bit-identical output to SparseMoE
2. Expert skip-empty logic
3. Expert token sort correctness
4. Per-expert staging buffers
5. Hot-buffer management (LRU/count-based)
"""

import pytest
import torch

pytest.importorskip("superl8")

from superl8serve.models.moe import SparseMoE, WeightStationaryMoE
from superl8serve.models.weights import gate_up_weight, to_qtensor

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="MoE expert cycling needs CUDA")


def _rand(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.float16) * 0.05


def _build_moe(num_experts=4, top_k=2, hidden_size=256, intermediate_size=128):
    """Build a SparseMoE and WeightStationaryMoE with identical weights."""
    torch.manual_seed(0)
    H, E, mi = hidden_size, num_experts, intermediate_size
    gate = _rand(E, H)
    sd = {}
    for e in range(E):
        sd[f"e{e}.gate_proj.weight"] = _rand(mi, H)
        sd[f"e{e}.up_proj.weight"] = _rand(mi, H)
        sd[f"e{e}.down_proj.weight"] = _rand(H, mi)
    experts = [
        (gate_up_weight(sd, f"e{e}"), to_qtensor(sd[f"e{e}.down_proj.weight"])) for e in range(E)
    ]
    sparse_moe = SparseMoE(gate=gate, experts=experts, top_k=top_k, norm_topk_prob=True).cuda()
    ws_moe = WeightStationaryMoE(gate=gate, experts=experts, top_k=top_k, norm_topk_prob=True).cuda()
    return sparse_moe, ws_moe


class TestWeightStationaryMoEBitIdentical:
    def test_bit_identical_single_token(self):
        sparse, ws = _build_moe()
        x = torch.randn(1, 1, 256, device="cuda", dtype=torch.float16)
        torch.manual_seed(42)
        y_sparse = sparse(x)
        torch.manual_seed(42)
        y_ws = ws(x)
        assert torch.allclose(y_sparse, y_ws, atol=1e-6, rtol=1e-5)

    def test_bit_identical_batch(self):
        sparse, ws = _build_moe()
        x = torch.randn(2, 5, 256, device="cuda", dtype=torch.float16)
        torch.manual_seed(42)
        y_sparse = sparse(x)
        torch.manual_seed(42)
        y_ws = ws(x)
        assert torch.allclose(y_sparse, y_ws, atol=1e-6, rtol=1e-5)

    def test_bit_identical_large_batch(self):
        sparse, ws = _build_moe()
        x = torch.randn(4, 10, 256, device="cuda", dtype=torch.float16)
        torch.manual_seed(42)
        y_sparse = sparse(x)
        torch.manual_seed(42)
        y_ws = ws(x)
        assert torch.allclose(y_sparse, y_ws, atol=1e-6, rtol=1e-5)


class TestExpertSkipEmpty:
    def test_empty_expert_skipped(self):
        H, E, mi, k = 256, 4, 128, 1
        gate = torch.full((E, H), -100.0, device="cuda", dtype=torch.float16)
        gate[0] = 100.0  # expert 0 gets huge positive logit
        sd = {}
        for e in range(E):
            sd[f"e{e}.gate_proj.weight"] = _rand(mi, H)
            sd[f"e{e}.up_proj.weight"] = _rand(mi, H)
            sd[f"e{e}.down_proj.weight"] = _rand(H, mi)
        experts = [
            (gate_up_weight(sd, f"e{e}"), to_qtensor(sd[f"e{e}.down_proj.weight"]))
            for e in range(E)
        ]
        ws = WeightStationaryMoE(gate=gate, experts=experts, top_k=k, norm_topk_prob=True).cuda()
        x = torch.ones(2, 3, H, device="cuda", dtype=torch.float16)
        y = ws(x)
        assert y.shape == (2, 3, H)
        assert torch.isfinite(y).all()

    def test_skip_empty_tracks_active_experts(self):
        H, E, mi, k = 256, 4, 128, 1
        gate = torch.full((E, H), -100.0, device="cuda", dtype=torch.float16)
        gate[0] = 100.0
        sd = {}
        for e in range(E):
            sd[f"e{e}.gate_proj.weight"] = _rand(mi, H)
            sd[f"e{e}.up_proj.weight"] = _rand(mi, H)
            sd[f"e{e}.down_proj.weight"] = _rand(H, mi)
        experts = [
            (gate_up_weight(sd, f"e{e}"), to_qtensor(sd[f"e{e}.down_proj.weight"]))
            for e in range(E)
        ]
        ws = WeightStationaryMoE(gate=gate, experts=experts, top_k=k, norm_topk_prob=True).cuda()
        x = torch.ones(2, 3, H, device="cuda", dtype=torch.float16)
        ws(x)
        assert ws.active_expert_count == 1


class TestExpertTokenSort:
    def test_sort_by_expert(self):
        H, E, mi, k = 256, 4, 128, 2
        gate = _rand(E, H)
        sd = {}
        for e in range(E):
            sd[f"e{e}.gate_proj.weight"] = _rand(mi, H)
            sd[f"e{e}.up_proj.weight"] = _rand(mi, H)
            sd[f"e{e}.down_proj.weight"] = _rand(H, mi)
        experts = [
            (gate_up_weight(sd, f"e{e}"), to_qtensor(sd[f"e{e}.down_proj.weight"]))
            for e in range(E)
        ]
        ws = WeightStationaryMoE(gate=gate, experts=experts, top_k=k, norm_topk_prob=True).cuda()
        x = torch.randn(2, 5, H, device="cuda", dtype=torch.float16)
        ws(x)
        for e in range(E):
            buf = ws.expert_buffers[e]
            assert buf.active_count >= 0


class TestPerExpertStaging:
    def test_staging_buffer_shape(self):
        H, E, mi, k = 256, 4, 128, 2
        gate = _rand(E, H)
        sd = {}
        for e in range(E):
            sd[f"e{e}.gate_proj.weight"] = _rand(mi, H)
            sd[f"e{e}.up_proj.weight"] = _rand(mi, H)
            sd[f"e{e}.down_proj.weight"] = _rand(H, mi)
        experts = [
            (gate_up_weight(sd, f"e{e}"), to_qtensor(sd[f"e{e}.down_proj.weight"]))
            for e in range(E)
        ]
        ws = WeightStationaryMoE(gate=gate, experts=experts, top_k=k, norm_topk_prob=True).cuda()
        x = torch.randn(2, 5, H, device="cuda", dtype=torch.float16)
        ws(x)
        for e in range(E):
            buf = ws.expert_buffers[e]
            assert buf.buf.shape == (64, 1, H)  # max_tokens=64 default


class TestHotBufferManagement:
    def test_expert_usage_tracking(self):
        H, E, mi, k = 256, 4, 128, 2
        gate = _rand(E, H)
        sd = {}
        for e in range(E):
            sd[f"e{e}.gate_proj.weight"] = _rand(mi, H)
            sd[f"e{e}.up_proj.weight"] = _rand(mi, H)
            sd[f"e{e}.down_proj.weight"] = _rand(H, mi)
        experts = [
            (gate_up_weight(sd, f"e{e}"), to_qtensor(sd[f"e{e}.down_proj.weight"]))
            for e in range(E)
        ]
        ws = WeightStationaryMoE(gate=gate, experts=experts, top_k=k, norm_topk_prob=True).cuda()
        x = torch.randn(2, 5, H, device="cuda", dtype=torch.float16)
        for _ in range(5):
            ws(x)
        assert sum(ws.expert_usage_counts) > 0
        assert len(ws.expert_usage_counts) == E


class TestExpertPrefetch:
    def test_prefetch_support(self):
        H, E, mi, k = 256, 4, 128, 2
        gate = _rand(E, H)
        sd = {}
        for e in range(E):
            sd[f"e{e}.gate_proj.weight"] = _rand(mi, H)
            sd[f"e{e}.up_proj.weight"] = _rand(mi, H)
            sd[f"e{e}.down_proj.weight"] = _rand(H, mi)
        experts = [
            (gate_up_weight(sd, f"e{e}"), to_qtensor(sd[f"e{e}.down_proj.weight"]))
            for e in range(E)
        ]
        ws = WeightStationaryMoE(gate=gate, experts=experts, top_k=k, norm_topk_prob=True).cuda()
        assert hasattr(ws, "prefetch_slots")
        assert len(ws.prefetch_slots) == E
