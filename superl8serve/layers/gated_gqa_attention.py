# SPDX-License-Identifier: MIT
"""GatedGQAAttention — full softmax attention with an output gate (Qwen3-Next /
Qwen3.5 `attn_output_gate`).

Identical to the shared `GQAAttention` full-attention block, with ONE addition: the
query projection also emits a per-(head, head_dim) gate. Following the HF
`Qwen3NextAttention` reference, `q_proj` outputs `num_heads * head_dim * 2`; the
last dim is reshaped to `[.., num_heads, 2*head_dim]` and `chunk(2, -1)`-split into
the query and the gate. QK-norm (pre-RoPE) applies to the QUERY only, RoPE to the
query; the gate is untouched until after attention, where the output is elementwise
multiplied by `sigmoid(gate)` BEFORE `o_proj`.

This lives in a separate module (not folded into `GQAAttention`) so the gate feature
is additive and does not perturb the shared block that every non-gated family uses.
It implements the two paths the standalone `ModelRunner` drives — non-varlen prefill
(`attn_int8_fwd`) and simple single-sequence decode (`attn_int8_decode`) — plus
varlen prefill for the engine's packed prefill. The engine's paged/continuous-batch
*decode* path (slot_mapping / slot_lengths) reuses `GQAAttention._decode_batched`
(the shared paged-decode kernels) on the query half and applies the sigmoid output
gate via the same `_gate_and_project` the standalone paths use — so gated Qwen3.5
decodes under the continuous-batching engine, not just the standalone ModelRunner.

Softmax/LSE stay fp32 inside the superl8 kernels; only the gate multiply is fp16, on
the healthy half2 CUDA-core pipe (never the dead Volta tensor cores).
"""

from __future__ import annotations

import torch
import torch.nn as nn

import superl8
from superl8 import QTensor

from .gqa_attention import GQAAttention
from .linear import LinearW8A8
from .norm import RMSNorm
from .rotary import RotaryEmbedding


def _paged_cache_kv(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapt Qwen3.5's bf16 stream to the sm70 paged-KV writer's fp16 ABI."""
    if k.dtype is torch.bfloat16:
        return k.half(), v.half()
    return k, v


def _bf16_varlen_attention(q, k, v, cu_seqlens, *, scale, window_left=-1):
    """Use the bf16-capable attention kernel per packed sequence on sm70."""
    bounds = cu_seqlens.detach().cpu().tolist()
    outputs = []
    for start, end in zip(bounds, bounds[1:]):
        qs = q[start:end].transpose(0, 1).unsqueeze(0)
        ks = k[start:end].transpose(0, 1).unsqueeze(0)
        vs = v[start:end].transpose(0, 1).unsqueeze(0)
        out = superl8.attn_int8_fwd(qs, ks, vs, causal=True, scale=scale, window_left=window_left)
        outputs.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


class GatedGQAAttention(nn.Module):
    def __init__(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        qkv_gate_proj: QTensor,  # merged [2*nh*hd | nkv*hd | nkv*hd]
        o_proj: QTensor,
        scale: float,
        rope: RotaryEmbedding,
        q_norm: torch.Tensor | None = None,
        k_norm: torch.Tensor | None = None,
        rms_norm_eps: float = 1e-6,
        qk_unit_offset: bool = False,
        window_left: int = -1,
        causal: bool = True,
    ):
        super().__init__()
        self.nh, self.nkv, self.hd = num_heads, num_kv_heads, head_dim
        self.scale = scale
        self.window_left = window_left
        self.causal = causal
        self.qkv_gate_proj = LinearW8A8(qkv_gate_proj)
        self.o_proj = LinearW8A8(o_proj)
        self.rope = rope
        self.q_norm = (
            RMSNorm(head_dim, rms_norm_eps, q_norm, add_unit_offset=qk_unit_offset)
            if q_norm is not None
            else None
        )
        self.k_norm = (
            RMSNorm(head_dim, rms_norm_eps, k_norm, add_unit_offset=qk_unit_offset)
            if k_norm is not None
            else None
        )

    def _project(self, x):
        """x: [B, S, hidden] -> (q, gate, k, v) each head-shaped, plus applies QK-norm
        (pre-RoPE) to q/k. q/gate: [B,S,nh,hd]; k/v: [B,S,nkv,hd]. gate is RAW (no
        norm, no RoPE)."""
        B, S, _ = x.shape
        qkvg = self.qkv_gate_proj(x)
        qg, k, v = qkvg.split(
            [2 * self.nh * self.hd, self.nkv * self.hd, self.nkv * self.hd], dim=-1
        )
        qg = qg.view(B, S, self.nh, 2 * self.hd)
        q, gate = qg[..., : self.hd], qg[..., self.hd :]  # chunk(2, -1): query | gate
        q = q.contiguous()
        k = k.view(B, S, self.nkv, self.hd)
        v = v.view(B, S, self.nkv, self.hd)
        if self.q_norm is not None:  # per-head RMSNorm, pre-RoPE, QUERY only (+ K)
            q = self.q_norm(q)
            k = self.k_norm(k)
        return q, gate, k, v

    def forward(self, x, positions, ctx, layer_idx: int) -> torch.Tensor:
        B, S, _ = x.shape
        q, gate, k, v = self._project(x)
        q, k = self.rope(positions, q, k)

        if ctx.is_prefill:
            if ctx.cu_seqlens is not None:
                q_v = q.reshape(B * S, self.nh, self.hd)
                k_v = k.reshape(B * S, self.nkv, self.hd)
                v_v = v.reshape(B * S, self.nkv, self.hd)
                cache_k, cache_v = _paged_cache_kv(k_v, v_v)
                ctx.kv_cache.write_prefill_varlen(layer_idx, ctx.slot_mapping, cache_k, cache_v)
                max_seqlen = int((ctx.cu_seqlens[1:] - ctx.cu_seqlens[:-1]).max().item())
                if q_v.dtype is torch.bfloat16:
                    out_v = _bf16_varlen_attention(
                        q_v,
                        k_v,
                        v_v,
                        ctx.cu_seqlens,
                        scale=self.scale,
                        window_left=self.window_left,
                    )
                else:
                    out_v = superl8.attn_int8_varlen(
                        q_v,
                        k_v,
                        v_v,
                        ctx.cu_seqlens,
                        ctx.cu_seqlens,
                        max_seqlen,
                        max_seqlen,
                        causal=self.causal,
                        scale=self.scale,
                    )
                out = out_v.reshape(B, S, self.nh, self.hd)  # token-major [B,S,H,D]
                return self._gate_and_project(out, gate, B, S)
            # Non-varlen prefill: kernels want [B, H, S, D].
            qt = q.transpose(1, 2).contiguous()
            kt = k.transpose(1, 2).contiguous()
            vt = v.transpose(1, 2).contiguous()
            slot = ctx.slots[0] if ctx.slots is not None else None
            cache_k, cache_v = _paged_cache_kv(kt, vt)
            ctx.kv_cache.write_prefill(
                layer_idx,
                cache_k,
                cache_v,
                slot=slot,
                start=ctx.prefill_start,
                positions=positions,
            )
            # Chunked prefill: current Q attends accumulated + current K/V.
            # superl8#286 aligns rectangular causal masks bottom-right, so there is
            # no need to recompute zero-padded query rows for the whole prefix.
            prefix_len = getattr(ctx, "prefill_length", None)
            if prefix_len is not None and prefix_len > S:
                # The compressed paged cache is the authoritative prefix store.
                # Reconstruct only this layer while it is active instead of retaining
                # a second growing fp16 K/V pair for every full-attention layer.
                kt_all, vt_all = ctx.kv_cache.read_dense(
                    layer_idx, slot, prefix_len, dtype=qt.dtype
                )
                out = superl8.attn_int8_fwd(
                    qt,
                    kt_all,
                    vt_all,
                    causal=self.causal,
                    scale=self.scale,
                    window_left=self.window_left,
                )
            else:
                out = superl8.attn_int8_fwd(
                    qt, kt, vt, causal=self.causal, scale=self.scale, window_left=self.window_left
                )
            out = out.transpose(1, 2)  # [B,H,S,D] -> [B,S,H,D]
            return self._gate_and_project(out, gate, B, S)

        if getattr(ctx, "is_verify", False):
            # Spec-decode verify (S = 1 + num_drafts tokens per row). Reuse the shared
            # ``GQAAttention._verify_batched`` as an unbound method exactly like the
            # decode path below borrows ``_decode_batched`` — it only touches
            # attributes both blocks share (``scale``) and ``ctx``/``cache``, and runs
            # on the QUERY half (``q``); the GATE is held aside and applied after. It
            # returns RAW ``[B, H, S, D]`` (heads not merged, no o_proj); transpose to
            # token-major ``[B, S, nh, hd]`` so the SAME ``_gate_and_project`` the
            # standalone/decode paths use applies the sigmoid output gate PER verify
            # token before the single o_proj. Committing every verify token's K/V
            # (accepted-token KV, #235) is done inside ``_verify_batched``.
            cache_k, cache_v = _paged_cache_kv(k, v)
            out = GQAAttention._verify_batched(self, q, cache_k, cache_v, ctx, layer_idx)
            out = out.transpose(1, 2)  # [B,H,S,D] -> [B,S,nh,hd]
            return self._gate_and_project(out, gate, B, S)

        if ctx.slot_lengths is not None or ctx.slot_mapping is not None:
            # Engine paged/continuous-batch decode. Reuse the shared batched-decode
            # kernels verbatim (paged int8 write + one `attn_paged_decode_cached`
            # launch, or the CUDA-graph static path) by borrowing GQAAttention's
            # `_decode_batched` as an unbound method — it only touches attributes
            # GatedGQAAttention shares (`scale`, `window_left`) and `ctx.kv_cache`,
            # and runs on the QUERY half (q here is the query, gate is held aside).
            # It returns [B, H, 1, D]; transpose to token-major [B, 1, H, D] so the
            # SAME `_gate_and_project` the standalone paths use applies the sigmoid
            # output gate before o_proj — identical gate math to the ModelRunner path.
            cache_k, cache_v = _paged_cache_kv(k, v)
            out = GQAAttention._decode_batched(self, q, cache_k, cache_v, ctx, layer_idx)
            out = out.transpose(1, 2)  # [B,H,1,D] -> [B,1,H,D] = [B,S,nh,hd]
            return self._gate_and_project(out, gate, B, S)

        # Simple single-sequence decode: kernels want [B, H, 1, D].
        qt = q.transpose(1, 2).contiguous()
        kt = k.transpose(1, 2).contiguous()
        vt = v.transpose(1, 2).contiguous()
        k_all, v_all = ctx.kv_cache.append_decode(layer_idx, kt, vt)
        k_all, v_all = self._window(k_all, v_all)
        out = superl8.attn_int8_decode(qt, k_all, v_all, scale=self.scale)
        out = out.transpose(1, 2)  # [B,H,1,D] -> [B,1,H,D]
        return self._gate_and_project(out, gate, B, S)

    def _gate_and_project(self, out, gate, B, S):
        """out, gate: [B, S, nh, hd]. Apply sigmoid gate, merge heads, o_proj."""
        # Paged decode is fp16-only on sm70, while bf16-native models keep their
        # residual stream (and projection input) in bf16. Restore that model dtype
        # after the paged kernel instead of leaking fp16 into the next RMSNorm.
        out = out.to(gate.dtype) * torch.sigmoid(gate)
        out = out.reshape(B, S, self.nh * self.hd)
        return self.o_proj(out)

    def _window(self, k_all, v_all):
        if self.window_left >= 0 and k_all.shape[2] > self.window_left:
            k_all = k_all[:, :, -self.window_left :].contiguous()
            v_all = v_all[:, :, -self.window_left :].contiguous()
        return k_all, v_all
