# SPDX-License-Identifier: MIT
"""Multi-GPU MoE with compressed transport (issues #330, #331).

Extends WeightStationaryMoE with cross-GPU token routing via the existing
transport compression seam (superl8.compress_activation / send / recv). Each GPU
owns a contiguous block of layers + local experts. At MoE boundaries, the
router sorts tokens by destination GPU, compresses remote tokens (int4 + entropy),
and sends them via P2P. The receiving GPU decompresses and processes its local
experts.

Wire budget (9B, 2-way MoE PP, batch=128, hidden=3584):
  Int4 compressed: 128 × 1.8 KB = 230 KB → 0.9 ms (2.3% of 40 ms compute)
  With entropy: 128 × 1.2 KB = 150 KB → 0.6 ms (1.5% of compute)

Key design: expert weights stay on the GPU that owns them. Only activations
(compressed) cross the wire. Skip-empty across GPUs via staging buffer
active_count.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dist import send, recv
from .moe import SparseMoE, WeightStationaryMoE

log = logging.getLogger(__name__)


class MoETransport:
    """Manages compressed cross-GPU token routing for MoE layers.

    Each MoE layer that straddles a GPU boundary gets a MoETransport instance
    that knows which experts live on which GPU and handles the compressed
    send/recv of tokens and results.
    """

    def __init__(
        self,
        expert_to_gpu: dict[int, int],
        local_gpu: int,
        default_scheme: str = "int4",
    ):
        """
        Args:
            expert_to_gpu: maps expert_id -> GPU index that owns it
            local_gpu: this process's GPU index
            default_scheme: compression scheme for transport
        """
        self.expert_to_gpu = {int(k): int(v) for k, v in expert_to_gpu.items()}
        self.local_gpu = local_gpu
        self.default_scheme = default_scheme
        # Pre-compute local vs remote expert sets (from the NORMALIZED map — the
        # raw parameter may be string-keyed from JSON).
        self.local_experts = {
            e for e, g in self.expert_to_gpu.items() if g == local_gpu
        }
        self.remote_gpus = sorted(
            set(g for g in self.expert_to_gpu.values() if g != local_gpu)
        )

    def classify_experts(
        self, active_experts: list[int]
    ) -> tuple[list[int], dict[int, list[int]]]:
        """Split active experts into local and remote (by GPU).

        Returns:
            local_active: list of active expert IDs that are local
            remote_active: dict mapping gpu_id -> list of active expert IDs
        """
        local_active = []
        remote_active: dict[int, list[int]] = {g: [] for g in self.remote_gpus}
        for e in active_experts:
            if e in self.local_experts:
                local_active.append(e)
            else:
                gpu = self.expert_to_gpu[e]
                remote_active[gpu].append(e)
        return local_active, remote_active

    def pack_remote_tokens(
        self,
        tokens: torch.Tensor,
        expert_ids: list[int],
        expert_assignments: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack tokens destined for remote experts into a contiguous tensor.

        Args:
            tokens: [T, H] all tokens
            expert_ids: which remote experts have tokens
            expert_assignments: [T, k] which expert each token slot maps to
            topk_weights: [T, k] routing weights

        Returns:
            packed_tokens: [N_remote, H] contiguous tokens for remote GPU
            packed_weights: [N_remote] routing weights for each token
            packed_src_indices: [N_remote] original token indices (for scatter-back)
        """
        device = tokens.device
        mask = torch.zeros(tokens.shape[0], dtype=torch.bool, device=device)
        for e in expert_ids:
            mask |= (expert_assignments == e).any(dim=1)

        if not mask.any():
            return (
                torch.empty(0, tokens.shape[-1], dtype=tokens.dtype, device=device),
                torch.empty(0, dtype=tokens.dtype, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )

        src_indices = mask.nonzero(as_tuple=True)[0]
        packed_tokens = tokens[src_indices]
        # Take the weight from the first matching slot
        packed_weights = topk_weights[src_indices, 0]
        return packed_tokens, packed_weights, src_indices

    def send_tokens(
        self, tokens: torch.Tensor, dst_gpu: int, scheme: str | None = None
    ):
        """Compress and send tokens to a remote GPU."""
        if tokens.numel() == 0:
            return None
        scheme = scheme or self.default_scheme
        return send(tokens, dst=dst_gpu, scheme=scheme)

    def recv_tokens(self, handle):
        """Receive and decompress tokens from a remote GPU."""
        if handle is None:
            return None
        return recv(handle)

    def pack_results(
        self,
        results: torch.Tensor,
        src_indices: torch.Tensor,
        total_tokens: int,
    ) -> torch.Tensor:
        """Scatter remote results back to their original token positions.

        Args:
            results: [N_remote, H] processed results from remote GPU
            src_indices: [N_remote] original token positions
            total_tokens: total number of tokens (for output shape)

        Returns:
            output: [total_tokens, H] with results scattered to correct positions
        """
        H = results.shape[-1]
        output = torch.zeros(total_tokens, H, dtype=results.dtype, device=results.device)
        if results.numel() > 0:
            output[src_indices] = results
        return output


class TransportMoELayer(nn.Module):
    """MoE layer with cross-GPU transport for remote expert routing.

    Wraps an existing SparseMoE or WeightStationaryMoE and adds compressed
    cross-GPU token routing. For single-GPU use, transport_dst=None makes
    this behave identically to the base MoE.

    The forward pass:
    1. Router computes expert assignments (same as base MoE)
    2. Classify experts as local vs remote
    3. Process local experts (base MoE logic)
    4. Compress + send remote tokens to owning GPU
    5. Receive remote results and scatter back
    6. Combine local + remote outputs
    """

    def __init__(
        self,
        base_moe: SparseMoE | WeightStationaryMoE,
        transport: MoETransport | None = None,
    ):
        super().__init__()
        self.base_moe = base_moe
        self.transport = transport

    def forward(self, x: torch.Tensor, ctx=None) -> torch.Tensor:
        B, S, H = x.shape
        xf = x.reshape(-1, H)
        T = B * S

        if self.transport is None or not self.transport.remote_gpus:
            # No remote experts — delegate entirely to base MoE
            return self.base_moe.forward(x)

        # ── Router (same as base MoE) ──────────────────────────────────
        gate = self.base_moe.gate
        topk = self.base_moe.top_k
        scoring_func = self.base_moe.scoring_func
        norm_topk_prob = self.base_moe.norm_topk_prob
        routed_scaling_factor = self.base_moe.routed_scaling_factor

        # The router softmax is fp32 (numerically load-bearing), but the gate
        # weight itself is fixed — cast it to fp32 ONCE (cached on the base MoE)
        # instead of `gate.float()` every layer (a [256, 2048] fp32 materialize
        # × 40 layers/step).
        gate32 = getattr(self.base_moe, "_gate_f32", None)
        if gate32 is None:
            gate32 = gate.float().detach()
            self.base_moe._gate_f32 = gate32
        router_logits = F.linear(xf.float(), gate32)
        if scoring_func == "sigmoid":
            scores = torch.sigmoid(router_logits)
            sel = scores + (
                self.base_moe.e_score_bias
                if self.base_moe.e_score_bias is not None
                else scores
            )
        else:
            scores = F.softmax(router_logits, dim=-1)
            sel = scores

        if self.base_moe.num_expert_groups is not None and self.base_moe.topk_group is not None:
            experts_per_group = sel.shape[-1] // self.base_moe.num_expert_groups
            group_scores = sel.view(T, self.base_moe.num_expert_groups, experts_per_group).amax(dim=-1)
            selected_groups = group_scores.topk(self.base_moe.topk_group, dim=-1)[1]
            offsets = torch.arange(experts_per_group, device=sel.device).view(1, 1, experts_per_group)
            expert_idx = (selected_groups.unsqueeze(-1) * experts_per_group + offsets).reshape(T, -1)
            mask = torch.full_like(sel, float("-inf"))
            mask.scatter_(1, expert_idx, 0.0)
            sel = sel + mask

        if scoring_func == "sigmoid":
            _, topi = torch.topk(sel, topk, dim=-1)
            topw = scores.gather(-1, topi)
        else:
            topw, topi = torch.topk(sel, topk, dim=-1)
        if norm_topk_prob:
            topw = topw / topw.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        topw = (topw * routed_scaling_factor).to(x.dtype)

        # ── Sort (token, slot) pairs by expert ─────────────────────────
        # topi.reshape(-1) is row-major: token i owns rows [i*topk, (i+1)*topk).
        # One device-side torch.unique(return_counts=True) gives the sorted
        # active expert set AND their counts in a single op — replacing the old
        # torch.unique().tolist() + argsort + bincount.cumsum().tolist() (2
        # device→host syncs/layer, 80/step). unique is sorted, so each expert's
        # rows are one contiguous slice of the (token,slot) sequence in expert
        # order; spans fall out of cumsum(counts) with ONE .tolist().
        flat_tok = torch.arange(T, device=xf.device).repeat_interleave(topk)
        flat_exp = topi.reshape(-1)
        # RAW per-(token, slot) weight. A token hitting k experts contributes
        # k rows, each with its own slot's weight; index_add_ sums them.
        flat_w = topw.reshape(-1)
        active_exp, counts = torch.unique(flat_exp, return_counts=True)  # sorted, [A]
        counts_l = counts.tolist()  # the ONE host sync
        active_experts = active_exp.tolist()
        local_active, remote_active = self.transport.classify_experts(active_experts)

        # Span boundaries per EXPERT ID (length E, most entries empty) so
        # _gather's `starts[e]`/`ends[e]` index by expert id as before.
        num_e = len(self.base_moe.experts)
        starts = [0] * num_e
        ends = [0] * num_e
        offset = 0
        for i, e in enumerate(active_experts):
            starts[e] = offset
            offset += counts_l[i]
            ends[e] = offset

        # (token,slot) rows are in flat_tok/flat_w order; the unique() sorted
        # the EXPERT ids, but the rows themselves are NOT yet in that order.
        # Build the per-expert contiguous row views by matching each row's
        # expert to its position in the sorted active set (device-side).
        sort_idx = torch.searchsorted(active_exp, flat_exp)  # [T*k] pos in active
        # Group rows by expert: rows sharing sort_idx are contiguous after a
        # stable sort by sort_idx — equivalent to argsort(flat_exp) but reusing
        # the unique output. tok_sorted/w_sorted are the row gather.
        perm = torch.argsort(sort_idx, stable=True)
        tok_sorted = flat_tok[perm]
        w_sorted = flat_w[perm]
        # per-expert row spans in the SORTED rows: expert e's rows occupy
        # [starts[e], ends[e]) in sorted order.
        out = torch.zeros_like(xf)

        def _gather(experts):
            """Collect the sorted rows for `experts`; spans are (expert,
            start, end) offsets into the RETURNED rows, not the global sort."""
            spans, chunks, off = [], [], 0
            for e in experts:
                s, t = starts[e], ends[e]
                if s == t:
                    continue
                chunks.append(torch.arange(s, t, device=xf.device))
                spans.append((e, off, off + (t - s)))
                off += t - s
            if not spans:
                return None, None, None
            rows = torch.cat(chunks)
            return tok_sorted[rows], w_sorted[rows], spans

        # ── Remote experts: one bulk hop per device ────────────────────
        # Enqueue ALL remote work before local experts: the .to() copy and
        # expert kernels queue on the remote device's default stream and run
        # concurrently with the local compute below. The copy-back .to()
        # event-syncs the two default streams, so the final index_add_
        # observes completed remote results without a manual synchronize.
        pending = []
        for group in remote_active.values():
            tok_r, w_r, spans = _gather(group)
            if tok_r is None:
                continue
            dev = self.base_moe.experts[spans[0][0]].gate_up_proj.weight.data.device
            xg = xf[tok_r].to(dev, non_blocking=True)
            yg = torch.empty_like(xg)
            # superl8's dp4a/k-quant kernels launch on the AMBIENT CUDA device
            # (cudaGetDevice); guard the whole remote batch once.
            with torch.cuda.device(dev):
                for e, s, t in spans:
                    yg[s:t] = self.base_moe.experts[e](
                        xg[s:t].unsqueeze(1)
                    ).squeeze(1)
            pending.append((tok_r, w_r, yg))

        # ── Local experts (rank-0 spans): same slicing, no device hop ──
        tok_l, w_l, spans_l = _gather(local_active)
        for eid, s, e in spans_l or []:
            exp = self.base_moe.experts[eid]
            tok_idx = tok_l[s:e]
            y = exp(xf[tok_idx].unsqueeze(1)).squeeze(1)
            out.index_add_(0, tok_idx, y * w_l[s:e].unsqueeze(-1))

        # ── Copy-back: drain remote results (yg is on the remote device) ──
        for tok_r, w_r, yg in pending:
            yg = yg.to(xf.device, non_blocking=True)
            out.index_add_(0, tok_r, yg * w_r.unsqueeze(-1))

        # ── Shared expert (always runs on all tokens) ──────────────────
        if self.base_moe.shared is not None:
            shared_out = self.base_moe.shared(x).reshape(-1, H)
            if self.base_moe.shared_gate is not None:
                g = torch.sigmoid(F.linear(xf.float(), self.base_moe.shared_gate.float())).to(x.dtype)
                shared_out = shared_out * g
            out = out + shared_out

        return out.reshape(B, S, H)
