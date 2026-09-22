# SPDX-License-Identifier: MIT
"""GQAAttention — the shared full/sliding softmax-attention block on superl8 dp4a.

Config-driven so Qwen3, Qwen3-MoE, Gemma3, GLM, Hunyuan all reuse it unchanged:
  * merged QKV projection (int8 dp4a), split to GQA heads;
  * optional per-head RMSNorm on Q and K over head_dim, applied BEFORE RoPE
    (Qwen3 / Gemma3 QK-norm);
  * RoPE with per-layer theta (Gemma3 local vs global) and partial-rotary (GLM);
  * softmax scale = query_pre_attn_scalar**-0.5 (Gemma) or head_dim**-0.5;
  * sliding-window local layers via superl8's window_left (prefill) / cache slice (decode).

This is the `full`/`sliding` AttentionBackend. `linear` (DeltaNet) and `latent`
(MLA) backends are separate — see layers/linear_attn.py and layers/mla_attn.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import superl8
from superl8 import QTensor

from .linear import LinearW8A8
from .norm import RMSNorm
from .rotary import RotaryEmbedding

# NOTE: spec-decode verify attention (``_verify_batched``) no longer uses a dedicated
# int8 verify kernel — it runs the verify tokens through the SAME paged-decode kernel
# plain decode uses (bit-identical, head_dim 256 included). No probe / dim gate needed.


class GQAAttention(nn.Module):
    def __init__(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        qkv_proj: QTensor,
        o_proj: QTensor,
        scale: float,
        rope: RotaryEmbedding,
        q_norm: torch.Tensor | None = None,
        k_norm: torch.Tensor | None = None,
        rms_norm_eps: float = 1e-6,
        window_left: int = -1,
        qkv_bias=None,
        o_bias=None,
        causal: bool = True,
    ):
        super().__init__()
        self.nh, self.nkv, self.hd = num_heads, num_kv_heads, head_dim
        self.scale = scale
        self.window_left = window_left
        self.causal = causal
        self.qkv_proj = LinearW8A8(qkv_proj, qkv_bias)
        self.o_proj = LinearW8A8(o_proj, o_bias)
        self.rope = rope
        self.q_norm = RMSNorm(head_dim, rms_norm_eps, q_norm) if q_norm is not None else None
        self.k_norm = RMSNorm(head_dim, rms_norm_eps, k_norm) if k_norm is not None else None

    def forward(self, x, positions, ctx, layer_idx: int) -> torch.Tensor:
        """x: [B, S, hidden]; positions: [B, S] or [S]. Returns [B, S, hidden]."""
        B, S, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.nh * self.hd, self.nkv * self.hd, self.nkv * self.hd], dim=-1)
        q = q.view(B, S, self.nh, self.hd)
        k = k.view(B, S, self.nkv, self.hd)
        v = v.view(B, S, self.nkv, self.hd)
        if self.q_norm is not None:  # per-head RMSNorm, pre-RoPE
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rope(positions, q, k)
        # q, k, v are [B, S, H, D] here. The transpose to the kernels' [B, H, S, D]
        # layout is now done ONLY in the two branches that genuinely need it
        # (non-varlen prefill, simple decode). The varlen-prefill and decode-batched
        # hot paths take token-major [total_tok, H, D], which is a zero-copy reshape
        # of [B, S, H, D] (Decode Lever 2: no more transpose().contiguous() churn).

        if ctx.is_prefill:
            if ctx.cu_seqlens is not None:
                # Varlen batched prefill: [1, S, H, D] -> [total_tok, H, D] via reshape
                # (B == 1 for the packed varlen layout; token-major, heads interleaved,
                # the flash_attn_varlen convention). No transpose, no contiguous copy.
                q_v = q.reshape(B * S, self.nh, self.hd)  # [total_tok, H, D]
                k_v = k.reshape(B * S, self.nkv, self.hd)  # [total_tok, Hkv, D]
                v_v = v.reshape(B * S, self.nkv, self.hd)  # [total_tok, Hkv, D]
                ctx.kv_cache.write_prefill_varlen(layer_idx, ctx.slot_mapping, k_v, v_v)
                max_seqlen = int((ctx.cu_seqlens[1:] - ctx.cu_seqlens[:-1]).max().item())
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
                # out_v is [total_tok, H, D] == [B, S, H, D] (B==1) -> merge heads by reshape.
                out = out_v.reshape(B, S, self.nh * self.hd)
                return self.o_proj(out)
            else:
                # Non-varlen prefill genuinely needs [B, H, S, D].
                q = q.transpose(1, 2).contiguous()
                k = k.transpose(1, 2).contiguous()
                v = v.transpose(1, 2).contiguous()
                slot = ctx.slots[0] if ctx.slots is not None else None
                ctx.kv_cache.write_prefill(
                    layer_idx, k, v, slot=slot, start=ctx.prefill_start, positions=positions,
                )
                # Chunked prefill: use accumulated fp16 K/V + current K/V with
                # attn_int8_fwd (same kernel as full prefill) by zero-padding Q
                # to match the accumulated K/V length.
                prefix_len = getattr(ctx, "prefill_length", None)
                if prefix_len is not None and prefix_len > S:
                    k_all, v_all = ctx.kv_cache.read_dense(
                        layer_idx, slot, prefix_len, dtype=q.dtype
                    )
                    out = superl8.attn_int8_fwd(
                        q, k_all, v_all,
                        causal=self.causal, scale=self.scale, window_left=self.window_left,
                    )
                else:
                    out = superl8.attn_int8_fwd(
                        q, k, v, causal=self.causal, scale=self.scale, window_left=self.window_left
                    )
        elif getattr(ctx, "is_verify", False):
            # Spec-decode verify: returns RAW [B, H, S, D]; the shared tail below
            # merges heads and applies o_proj exactly once (do NOT o_proj here).
            out = self._verify_batched(q, k, v, ctx, layer_idx)
        elif ctx.slot_lengths is not None or ctx.slot_mapping is not None:
            # Decode-batched hot path: takes [B, S=1, H, D] directly (no transpose here).
            out = self._decode_batched(q, k, v, ctx, layer_idx)  # engine continuous batch
            # (or CUDA-graph static path)
        else:
            # Simple decode genuinely needs [B, H, S, D].
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            k_all, v_all = ctx.kv_cache.append_decode(layer_idx, k, v)  # [B,Hkv,N,D]
            k_all, v_all = self._window(k_all, v_all)
            out = superl8.attn_int8_decode(q, k_all, v_all, scale=self.scale)

        out = out.transpose(1, 2).reshape(B, S, self.nh * self.hd)
        return self.o_proj(out)

    def _window(self, k_all, v_all):
        if self.window_left >= 0 and k_all.shape[2] > self.window_left:
            k_all = k_all[:, :, -self.window_left :].contiguous()
            v_all = v_all[:, :, -self.window_left :].contiguous()
        return k_all, v_all

    def _decode_batched(self, q, k, v, ctx, layer_idx):
        """Continuous-batch decode: rows have different KV lengths (already batched
        GEMMs upstream). The paged int8 cache commits every row's new token with ONE
        `quantize_kv_write_paged` call and reads the whole ragged batch back with ONE
        `attn_paged_decode_cached` launch -- no more per-slot Python loop.

        Sliding-window layers are the one gap the paged-decode kernel doesn't cover
        (no window parameter yet), so they fall back to a per-slot dequantized read
        + `attn_int8_decode`, same as before this PR.

        Inputs q, k, v are [B, S=1, H, D] (token-major, as they leave RoPE). The one
        new token's K/V is sliced with a zero-copy index; q is transposed to the
        kernels' [B, H_q, 1, D] here (a size-1 permute — the write/attn kernels make
        their inputs contiguous internally)."""
        cache = ctx.kv_cache
        k_new, v_new = k[:, 0, :, :], v[:, 0, :, :]  # [B,Hkv,D]: the one new token
        q = q.transpose(1, 2)  # [B,S=1,H,D] -> [B,H,1,D] for the decode kernels
        if self.window_left < 0:
            if ctx.slot_mapping is not None:
                # CUDA-graph decode (engine/cuda_graph.py): slot_mapping/block_tables/
                # context_lens are persistent device buffers refreshed via `copy_`
                # before replay, and max_context_len is a compile-time bucket int --
                # no fresh per-layer tensor allocation and no `.item()` sync, so this
                # whole call is capturable.
                cache.write_decode_static(layer_idx, ctx.slot_mapping, k_new, v_new)
                return cache.decode_attn_static(
                    layer_idx,
                    q,
                    ctx.block_tables,
                    ctx.context_lens,
                    ctx.max_context_len,
                    scale=self.scale,
                )
            cache.write_decode(layer_idx, ctx.slots, ctx.slot_lengths, k_new, v_new)
            return cache.decode_attn(layer_idx, q, ctx.slots, ctx.slot_lengths, scale=self.scale)
        outs = []
        for b, (slot, n) in enumerate(zip(ctx.slots, ctx.slot_lengths)):
            cache.write_decode(layer_idx, [slot], [n], k_new[b : b + 1], v_new[b : b + 1])
            kb, vb = cache.read_dense(layer_idx, slot, n + 1, window=self.window_left)
            outs.append(superl8.attn_int8_decode(q[b : b + 1], kb, vb, scale=self.scale))
        return torch.cat(outs, dim=0)

    @staticmethod
    def _draft_attn(q, k, v, *, scale):
        """Cache-free single-step attention for MTP drafting.

        Computes attention without reading/writing any external KV cache.
        q, k, v are fp16 in [B, H, S, D] / [B, Hkv, S, D] layout (after
        transpose). Used by MTP heads during the draft phase so they never
        corrupt the main model's paged cache.
        """
        bs, H, S, D = q.shape
        Hkv = k.shape[1]
        if H != Hkv:
            rep = H // Hkv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        scores = (q @ k.transpose(-2, -1)) * scale  # [B, H, S, S]
        attn = torch.softmax(scores, dim=-1)
        return (attn @ v).transpose(1, 2).reshape(bs, S, H * D)

    def _verify_batched(self, q, k, v, ctx, layer_idx):
        """Spec-decode verify: S = 1 + num_drafts tokens per batch row, attention
        computed BIT-IDENTICALLY to plain paged decode.

        The verify forward's GEMMs (qkv / o / mlp) run over all S tokens in ONE
        weight-read — that is the whole point (one weight-stream amortized over the
        accepted tokens). Only the ATTENTION must match decode exactly, and it does
        here: we walk the S verify tokens in causal order, writing each token's K/V to
        the paged int8 store and reading it back through the SAME
        ``attn_paged_decode_cached`` kernel plain decode uses. So verify token ``t``
        attends the committed prefix plus verify tokens ``0..t`` exactly as ``t`` back-
        to-back decode steps would — every committed token's logits/K/V are then
        byte-identical to non-spec greedy (no separate verify kernel, no re-decode /
        canonicalization, no dequant→requant of the prefix). ``attn_paged_decode_cached``
        already supports head_dim 256, so this needs no dedicated int8 verify kernel.

        Writing all S positions (including the rejected tail past ``n_acc``) is
        harmless: those positions are never read (reads gate on each row's committed
        length) and are overwritten by the next step.

        Sliding-window layers (``window_left >= 0``) keep the dequantized-read fallback
        below — the paged-decode kernel has no window parameter.

        Returns RAW ``[B, H, S, D]`` (heads not merged, no ``o_proj``); the forward
        tail merges heads and applies ``o_proj`` exactly once.
        """
        cache = ctx.kv_cache
        B, S = q.shape[0], q.shape[1]
        vsm = ctx.verify_slot_mapping.view(B, S)  # [B, S] paged slot per (row, verify pos)
        slots = ctx.slots
        prefix = ctx.slot_lengths  # committed prefix length per row (positions 0..L)

        # CUDA-graph-capturable verify (engine/cuda_graph.py `GraphedVerify`): when the
        # runner supplies persistent device `block_tables`/`context_lens` + a
        # compile-time `max_context_len` bucket (exactly the decode-graph contract),
        # walk the S verify tokens through the SAME static paged-decode primitives the
        # graphed decode uses — no python-list `slots`/`lengths`, no `.item()` sync — so
        # the whole verify forward is capturable. `context_lens` is the committed prefix
        # length per row (L+1); verify token t writes at that row's slot for position
        # L+1+t and then attends context length (L+1)+(t+1) — a device-tensor add by the
        # loop-constant `t+1` (S is fixed, so this loop is unrolled at capture). Bit-
        # identical to the eager list path below (same kernels, same context lengths).
        if getattr(ctx, "block_tables", None) is not None and self.window_left < 0:
            outs = []
            for t in range(S):
                cache.write_decode_static(
                    layer_idx, vsm[:, t].contiguous(), k[:, t], v[:, t]
                )
                ctx_len_t = ctx.context_lens + (t + 1)  # [B] int32, in-graph add
                q_t = q[:, t : t + 1].transpose(1, 2)  # [B,1,H,D] -> [B,H,1,D]
                outs.append(
                    cache.decode_attn_static(
                        layer_idx,
                        q_t,
                        ctx.block_tables,
                        ctx_len_t,
                        ctx.max_context_len,
                        scale=self.scale,
                    )
                )
            return torch.cat(outs, dim=2)  # [B, H, S, D]

        if self.window_left < 0:
            outs = []  # per t: [B, H, 1, D]
            for t in range(S):
                # 1) commit verify token t's K/V at its paged slot (== a decode write).
                cache.write_decode_static(
                    layer_idx, vsm[:, t].contiguous(), k[:, t], v[:, t]
                )
                # 2) attend query t over prefix + verify tokens 0..t (context length
                #    prefix + t + 1); write position = prefix + t, exactly as the t-th
                #    consecutive decode step would see it.
                lengths_t = [p + t for p in prefix]
                q_t = q[:, t : t + 1].transpose(1, 2)  # [B,1,H,D] -> [B,H,1,D]
                outs.append(cache.decode_attn(layer_idx, q_t, slots, lengths_t, scale=self.scale))
            return torch.cat(outs, dim=2)  # [B, H, S, D]

        # Sliding-window fallback: dequantized read + causal zero-pad-Q attn_int8_fwd
        # per row (attn_paged_decode_cached has no window param). Commit K/V first.
        k_t = k.transpose(1, 2).contiguous()  # [B, Hkv, S, D]
        v_t = v.transpose(1, 2).contiguous()
        q_t = q.transpose(1, 2).contiguous()  # [B, H, S, D]
        D = q_t.shape[-1]
        Hkv = k_t.shape[1]
        k_flat = k_t.permute(0, 2, 1, 3).reshape(B * S, Hkv, D)
        v_flat = v_t.permute(0, 2, 1, 3).reshape(B * S, Hkv, D)
        cache.write_decode_static(layer_idx, ctx.verify_slot_mapping, k_flat, v_flat)
        outs = []
        for b in range(B):
            kb, vb = cache.read_dense(layer_idx, slots[b], prefix[b], window=self.window_left)
            k_all = torch.cat([kb, k_t[b:b + 1]], dim=2)
            v_all = torch.cat([vb, v_t[b:b + 1]], dim=2)
            total = k_all.shape[2]
            q_pad = k_all.new_zeros(1, q_t.shape[1], total, D)
            q_pad[:, :, -S:, :] = q_t[b:b + 1]
            out_b = superl8.attn_int8_fwd(q_pad, k_all, v_all, causal=True, scale=self.scale)
            outs.append(out_b[:, :, -S:, :])
        return torch.cat(outs, dim=0)  # [B, H, S, D]
