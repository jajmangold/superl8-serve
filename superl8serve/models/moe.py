# SPDX-License-Identifier: MIT
"""Sparse MoE FFN — Qwen3-MoE routing, with an optional shared expert.

Routing (Qwen3-MoE, verified against HF): softmax over ALL experts (fp32) -> top-k
-> renormalize the k weights (`norm_topk_prob`). Each expert is a GatedMLP on the
superl8 dp4a GEMM. A shared expert (Qwen3-Next / Qwen2-MoE) runs on every token in
parallel and is added in; plain Qwen3-MoE has none.

v0 loops experts in Python (correct, simple). A batched grouped-GEMM that runs all
active experts in one launch is the eventual superl8 kernel (moe grouped-GEMM), noted
in the coverage matrix; the win is at high expert counts.

``WeightStationaryMoE`` is the Phase 3 weight-stationary variant: expert token sort
(host-side, eager) + per-expert staging buffers + skip-empty + expert prefetch slots
+ hot-buffer usage tracking.  The expert GEMMs themselves are graph-capturable (each
expert processes a contiguous token slice); only the router + sort + scatter is eager.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from superl8 import QTensor

from ..layers.mlp import GatedMLP

_DEFAULT_MAX_TOKENS_PER_EXPERT = 64


def _shared_gate_param(gate: torch.Tensor | None) -> torch.Tensor | None:
    """Normalize the shared-expert gate to a [1, hidden] linear weight.

    GGUF stores `ffn_gate_inp_shexp` as a flat [hidden] row (llama.cpp drops the
    trailing singleton dim); HF keeps it [1, hidden]. F.linear(x, w) needs
    [1, hidden], so a 1-D raw gate is reshaped here — format-agnostic.
    """
    if gate is None:
        return None
    if gate.dim() == 1:
        return gate.reshape(1, -1)
    return gate


class SparseMoE(nn.Module):
    def __init__(
        self,
        *,
        gate: torch.Tensor,  # router weight [num_experts, hidden] fp16
        experts: list[tuple[QTensor, QTensor]],  # per-expert (gate_up, down)
        top_k: int,
        norm_topk_prob: bool = True,
        act: str = "silu",
        shared_expert: tuple[QTensor, QTensor] | None = None,
        shared_expert_gate: torch.Tensor | None = None,  # [1, hidden] fp16 or None
        scoring_func: str = "softmax",  # "softmax" (Qwen/Hunyuan/MiniMax) | "sigmoid" (GLM/DeepSeek/LFM2)
        e_score_correction_bias: torch.Tensor | None = None,  # [E] fp32, sigmoid selection bias
        routed_scaling_factor: float = 1.0,
        num_expert_groups: int
        | None = None,  # partition experts into groups for group-limited top-k
        topk_group: int | None = None,  # number of groups to select (used with num_expert_groups)
    ):
        super().__init__()
        self.gate = nn.Parameter(gate)
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.num_expert_groups = num_expert_groups
        self.topk_group = topk_group
        self.register_buffer(
            "e_score_bias",
            e_score_correction_bias.float() if e_score_correction_bias is not None else None,
            persistent=False,
        )
        self.experts = nn.ModuleList([GatedMLP(gu, dn, act=act) for gu, dn in experts])
        self.shared = GatedMLP(*shared_expert, act=act) if shared_expert else None
        self.shared_gate = (
            nn.Parameter(_shared_gate_param(shared_expert_gate)) if shared_expert_gate is not None else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape
        xf = x.reshape(-1, H)  # [T, H]
        # Router softmax is fp32 (load-bearing); cache the fixed gate in fp32 so
        # the per-forward `gate.float()` (a [E, H] materialize per layer) is done
        # once, not 40x/step.
        gate32 = getattr(self, "_gate_f32", None)
        if gate32 is None:
            gate32 = self.gate.detach().float()
            self._gate_f32 = gate32
        router_logits = F.linear(xf.float(), gate32)  # [T, E]
        if self.scoring_func == "sigmoid":
            scores = torch.sigmoid(router_logits)  # GLM/DeepSeek/LFM2
            sel = scores + self.e_score_bias if self.e_score_bias is not None else scores
        else:
            scores = F.softmax(router_logits, dim=-1)  # softmax over ALL experts, fp32
            sel = scores

        if self.num_expert_groups is not None and self.topk_group is not None:
            T = B * S
            experts_per_group = sel.shape[-1] // self.num_expert_groups
            group_scores = sel.view(T, self.num_expert_groups, experts_per_group).amax(dim=-1)
            selected_groups = group_scores.topk(self.topk_group, dim=-1)[1]
            offsets = torch.arange(experts_per_group, device=sel.device).view(
                1, 1, experts_per_group
            )
            expert_idx = (selected_groups.unsqueeze(-1) * experts_per_group + offsets).reshape(
                T, -1
            )
            mask = torch.full_like(sel, float("-inf"))
            mask.scatter_(1, expert_idx, 0.0)
            sel = sel + mask

        if self.scoring_func == "sigmoid":
            _, topi = torch.topk(sel, self.top_k, dim=-1)  # select by biased (and masked) score
            topw = scores.gather(-1, topi)  # weight by UN-biased score
        else:
            topw, topi = torch.topk(sel, self.top_k, dim=-1)
        if self.norm_topk_prob:
            topw = topw / topw.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        topw = (topw * self.routed_scaling_factor).to(x.dtype)

        out = torch.zeros_like(xf)
        # gather tokens per expert (only run experts that got routed to)
        for e in range(len(self.experts)):
            mask = topi == e  # [T, k]
            if not mask.any():
                continue
            tok_idx, slot = mask.nonzero(as_tuple=True)  # which tokens, which slot
            contrib = self.experts[e](xf[tok_idx].unsqueeze(1)).squeeze(1)  # [n, H]
            out.index_add_(0, tok_idx, contrib * topw[tok_idx, slot].unsqueeze(-1))

        if self.shared is not None:
            shared_out = self.shared(x).reshape(-1, H)
            if self.shared_gate is not None:
                g = torch.sigmoid(F.linear(xf.float(), self.shared_gate.float())).to(x.dtype)
                shared_out = shared_out * g
            out = out + shared_out
        return out.reshape(B, S, H)


class ExpertStagingBuffer:
    """Per-expert activation buffer for weight-stationary MoE.

    Holds the tokens routed to one expert, ready for a graph-capturable GEMM.
    The buffer shape is ``(max_tokens, 1, hidden_dim)`` — each expert sees a
    contiguous slice of tokens that does not change across replays.
    """

    __slots__ = ("buf", "weight_buf", "active_count", "expert_id")

    def __init__(self, max_tokens: int, hidden_dim: int, expert_id: int, device: str):
        self.buf = torch.zeros(max_tokens, 1, hidden_dim, dtype=torch.float16, device=device)
        self.weight_buf = torch.zeros(max_tokens, dtype=torch.float16, device=device)
        self.active_count: int = 0
        self.expert_id = expert_id

    def set_tokens(
        self, tokens: torch.Tensor, weights: torch.Tensor, count: int
    ) -> None:
        """Copy *tokens* into ``buf`` and set ``active_count``."""
        if count > 0:
            t = tokens[:count].to(self.buf.dtype)
            if t.dim() == 2:
                t = t.unsqueeze(1)
            self.buf[:count].copy_(t)
            self.weight_buf[:count].copy_(weights[:count])
        self.active_count = count

    def is_empty(self) -> bool:
        return self.active_count == 0


class WeightStationaryMoE(nn.Module):
    """Phase 3 weight-stationary MoE with per-expert cycling.

    Key differences from ``SparseMoE``:
    1. **Expert token sort**: tokens are sorted by expert on the host (eager, data-dependent).
    2. **Per-expert staging**: each expert gets an ``ExpertStagingBuffer`` that holds a
       contiguous token slice ready for graph-capturable GEMM.
    3. **Expert skip-empty**: experts with zero tokens are not loaded/computed.
    4. **Expert prefetch slots**: while expert *i* computes, the next expert's weights can
       be prefetched (kernel-level concern; Python exposes the slot).
    5. **Hot-buffer management**: usage counts track which experts are most frequently active.
    """

    def __init__(
        self,
        *,
        gate: torch.Tensor,
        experts: list[tuple[QTensor, QTensor]],
        top_k: int,
        norm_topk_prob: bool = True,
        act: str = "silu",
        shared_expert: tuple[QTensor, QTensor] | None = None,
        shared_expert_gate: torch.Tensor | None = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float = 1.0,
        num_expert_groups: int | None = None,
        topk_group: int | None = None,
        max_tokens_per_expert: int = _DEFAULT_MAX_TOKENS_PER_EXPERT,
        expert_devices: list[str] | None = None,
    ):
        super().__init__()
        self.gate = nn.Parameter(gate)
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.num_expert_groups = num_expert_groups
        self.topk_group = topk_group
        self.register_buffer(
            "e_score_bias",
            e_score_correction_bias.float() if e_score_correction_bias is not None else None,
            persistent=False,
        )
        self.experts = nn.ModuleList([GatedMLP(gu, dn, act=act) for gu, dn in experts])
        self.shared = GatedMLP(*shared_expert, act=act) if shared_expert else None
        self.shared_gate = (
            nn.Parameter(_shared_gate_param(shared_expert_gate)) if shared_expert_gate is not None else None
        )
        num_e = len(self.experts)
        hidden_dim = gate.shape[1]

        # Per-expert staging buffers (host-side ownership, device tensors). Only
        # experts resident on THIS process's primary device get a buffer — with a
        # multi-GPU shard the remote experts live on peer GPUs and are computed via
        # direct peer calls (TransportMoELayer), never through these buffers.
        primary_dev = (expert_devices[0] if expert_devices else "cuda")
        self.expert_buffers: list[ExpertStagingBuffer | None] = [
            ExpertStagingBuffer(max_tokens_per_expert, hidden_dim, e, primary_dev)
            if (expert_devices is None or expert_devices[e] == primary_dev)
            else None
            for e in range(num_e)
        ]

        # Expert prefetch slots — one per expert, used by the kernel to overlap
        # weight-HBM-to-SRAM transfer of expert i+1 while expert i computes.
        self.prefetch_slots: list[int] = list(range(num_e))

        # Hot-buffer management: count-based usage tracking.
        self.expert_usage_counts: list[int] = [0] * num_e
        self.active_expert_count: int = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape
        xf = x.reshape(-1, H)  # [T, H]
        T = B * S

        # ── Eager: router + top-k (data-dependent, not graph-capturable) ──────
        gate32 = getattr(self, "_gate_f32", None)
        if gate32 is None:
            gate32 = self.gate.detach().float()
            self._gate_f32 = gate32
        router_logits = F.linear(xf.float(), gate32)  # [T, E]
        if self.scoring_func == "sigmoid":
            scores = torch.sigmoid(router_logits)
            sel = scores + self.e_score_bias if self.e_score_bias is not None else scores
        else:
            scores = F.softmax(router_logits, dim=-1)
            sel = scores

        if self.num_expert_groups is not None and self.topk_group is not None:
            experts_per_group = sel.shape[-1] // self.num_expert_groups
            group_scores = sel.view(T, self.num_expert_groups, experts_per_group).amax(dim=-1)
            selected_groups = group_scores.topk(self.topk_group, dim=-1)[1]
            offsets = torch.arange(experts_per_group, device=sel.device).view(
                1, 1, experts_per_group
            )
            expert_idx = (selected_groups.unsqueeze(-1) * experts_per_group + offsets).reshape(
                T, -1
            )
            mask = torch.full_like(sel, float("-inf"))
            mask.scatter_(1, expert_idx, 0.0)
            sel = sel + mask

        if self.scoring_func == "sigmoid":
            _, topi = torch.topk(sel, self.top_k, dim=-1)
            topw = scores.gather(-1, topi)
        else:
            topw, topi = torch.topk(sel, self.top_k, dim=-1)
        if self.norm_topk_prob:
            topw = topw / topw.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        topw = (topw * self.routed_scaling_factor).to(x.dtype)

        # ── Eager: expert token sort (host-side) ─────────────────────────────
        num_e = len(self.experts)
        active_experts: list[int] = []
        for e in range(num_e):
            mask = topi == e  # [T, k]
            if not mask.any():
                continue
            active_experts.append(e)
            tok_idx, slot = mask.nonzero(as_tuple=True)

            # Sort tokens into contiguous per-expert staging buffer.
            buf = self.expert_buffers[e]
            buf.set_tokens(xf[tok_idx], topw[tok_idx, slot], len(tok_idx))

        self.active_expert_count = len(active_experts)
        for e in active_experts:
            self.expert_usage_counts[e] += 1

        # ── Graph-capturable: per-expert GEMM (each expert processes a contiguous slice) ──
        out = self._scatter_expert_outputs(active_experts, topi, topw, xf)

        if self.shared is not None:
            shared_out = self.shared(x).reshape(-1, H)
            if self.shared_gate is not None:
                g = torch.sigmoid(F.linear(xf.float(), self.shared_gate.float())).to(x.dtype)
                shared_out = shared_out * g
            out = out + shared_out
        return out.reshape(B, S, H)

    def _scatter_expert_outputs(
        self,
        active_experts: list[int],
        topi: torch.Tensor,
        topw: torch.Tensor,
        xf: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter per-expert outputs back to token positions.

        Uses the same logic as SparseMoE: for each active expert, find which
        original tokens were routed to it and accumulate their weighted output.
        This ensures bit-identical output to SparseMoE.
        """
        out = torch.zeros_like(xf)
        for e in active_experts:
            mask = topi == e  # [T, k]
            tok_idx, slot = mask.nonzero(as_tuple=True)
            buf = self.expert_buffers[e]
            n = buf.active_count
            contrib = self.experts[e](buf.buf[:n]).squeeze(1)  # [n, H]
            out.index_add_(0, tok_idx, contrib * topw[tok_idx, slot].unsqueeze(-1))
        return out
