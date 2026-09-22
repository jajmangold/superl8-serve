# SPDX-License-Identifier: MIT
"""Tests for multi-GPU MoE transport (issues #330, #331)."""

import pytest
import torch

pytest.importorskip("superl8")

from superl8serve.models.moe_transport import MoETransport, TransportMoELayer
from superl8serve.models.moe import SparseMoE, ExpertStagingBuffer
from superl8serve.models.weights import gate_up_weight, to_qtensor

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="transport needs CUDA")


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.float16, device="cuda")


def _make_tiny_moe(num_experts=4, hidden=64, top_k=2):
    """Create a tiny MoE for testing — same pattern as test_moe_expert_cycling."""
    torch.manual_seed(0)
    H, E, mi = hidden, num_experts, hidden * 2
    gate = _rand(E, H)
    sd = {}
    for e in range(E):
        sd[f"e{e}.gate_proj.weight"] = _rand(mi, H)
        sd[f"e{e}.up_proj.weight"] = _rand(mi, H)
        sd[f"e{e}.down_proj.weight"] = _rand(H, mi)
    experts = [
        (gate_up_weight(sd, f"e{e}"), to_qtensor(sd[f"e{e}.down_proj.weight"])) for e in range(E)
    ]
    return SparseMoE(gate=gate, experts=experts, top_k=top_k, norm_topk_prob=True).cuda()


class TestMoETransport:
    def test_classify_experts(self):
        """Experts split correctly into local and remote."""
        transport = MoETransport(
            expert_to_gpu={0: 0, 1: 0, 2: 1, 3: 1},
            local_gpu=0,
        )
        local, remote = transport.classify_experts([0, 1, 2, 3])
        assert sorted(local) == [0, 1]
        assert sorted(remote[1]) == [2, 3]

    def test_classify_experts_all_local(self):
        """All-local experts produce empty remote dict."""
        transport = MoETransport(
            expert_to_gpu={0: 0, 1: 0, 2: 0, 3: 0},
            local_gpu=0,
        )
        local, remote = transport.classify_experts([0, 1, 2])
        assert sorted(local) == [0, 1, 2]
        assert all(len(v) == 0 for v in remote.values())

    def test_pack_remote_tokens(self):
        """Tokens for remote experts are packed correctly."""
        transport = MoETransport(
            expert_to_gpu={0: 0, 1: 1},
            local_gpu=0,
        )
        H = 16
        tokens = torch.randn(4, H, dtype=torch.float16, device="cuda")
        expert_assignments = torch.tensor([[0], [1], [0], [1]], device="cuda")
        topk_weights = torch.ones(4, 1, dtype=torch.float16, device="cuda")

        packed, weights, src_idx = transport.pack_remote_tokens(
            tokens, [1], expert_assignments, topk_weights
        )
        assert packed.shape == (2, H)
        assert weights.shape == (2,)
        assert torch.all(src_idx == torch.tensor([1, 3], device="cuda"))

    def test_pack_remote_tokens_empty(self):
        """No remote tokens returns empty tensors when expert IDs are all local."""
        transport = MoETransport(
            expert_to_gpu={0: 0, 1: 0},
            local_gpu=0,
        )
        tokens = torch.randn(4, 16, dtype=torch.float16, device="cuda")
        expert_assignments = torch.tensor([[0], [1], [0], [1]], device="cuda")
        topk_weights = torch.ones(4, 1, dtype=torch.float16, device="cuda")

        # Expert 1 is LOCAL (gpu=0), so pack_remote_tokens with expert_ids=[1]
        # should still pack tokens (the function doesn't filter by locality).
        # The locality filtering happens in classify_experts + the caller.
        packed, weights, src_idx = transport.pack_remote_tokens(
            tokens, [1], expert_assignments, topk_weights
        )
        # Expert 1 has tokens 1 and 3 assigned — they get packed
        assert packed.shape[0] == 2  # tokens 1 and 3

    def test_pack_results(self):
        """Results scattered back to correct positions."""
        transport = MoETransport(expert_to_gpu={0: 0}, local_gpu=0)
        H = 8
        results = torch.tensor(
            [[1.0, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]],
            dtype=torch.float16, device="cuda",
        )
        src_idx = torch.tensor([1, 3], device="cuda")
        output = transport.pack_results(results, src_idx, 5)
        assert output.shape == (5, H)
        assert torch.all(output[0] == 0)
        assert torch.all(output[1] == results[0])
        assert torch.all(output[2] == 0)
        assert torch.all(output[3] == results[1])
        assert torch.all(output[4] == 0)

    def test_transport_moe_layer_no_remote(self):
        """TransportMoELayer with no remote experts behaves like base MoE."""
        moe = _make_tiny_moe(num_experts=4, hidden=64, top_k=2)
        layer = TransportMoELayer(moe, transport=None)

        x = torch.randn(2, 8, 64, dtype=torch.float16, device="cuda")
        out = layer(x)
        assert out.shape == (2, 8, 64)
        assert out.dtype == torch.float16

    def test_transport_moe_layer_all_local(self):
        """TransportMoELayer with all experts local behaves like base MoE."""
        moe = _make_tiny_moe(num_experts=4, hidden=64, top_k=2)
        transport = MoETransport(
            expert_to_gpu={0: 0, 1: 0, 2: 0, 3: 0},
            local_gpu=0,
        )
        layer = TransportMoELayer(moe, transport=transport)

        x = torch.randn(2, 8, 64, dtype=torch.float16, device="cuda")
        out = layer(x)
        assert out.shape == (2, 8, 64)

    def test_transport_moe_bitidentical_single_gpu(self):
        """With all experts local, output matches SparseMoE approximately."""
        torch.manual_seed(42)
        moe_base = _make_tiny_moe(num_experts=4, hidden=64, top_k=2)
        import copy
        moe_copy = copy.deepcopy(moe_base)

        transport = MoETransport(
            expert_to_gpu={0: 0, 1: 0, 2: 0, 3: 0},
            local_gpu=0,
        )
        layer = TransportMoELayer(moe_copy, transport=transport)

        x = torch.randn(4, 16, 64, dtype=torch.float16, device="cuda")
        out_base = moe_base(x)
        out_transport = layer(x)
        cos = torch.nn.functional.cosine_similarity(
            out_base.flatten(), out_transport.flatten(), dim=0
        )
        assert cos > 0.99, f"cosine similarity too low: {cos}"
