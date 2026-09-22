# SPDX-License-Identifier: MIT
"""Attention seam — routes to the `superl8` int8 dp4a kernels.

Replaces nano-vllm's fp16 flash-attention:
  * prefill  -> superl8.attn_int8_fwd (causal int8 dp4a; varlen path is superl8.attn_int8_varlen)
  * decode   -> superl8.attn_int8_decode (split-KV) / attn_decode_cached (int8 KV cache)

This is the single-sequence seam used by the standalone `ModelRunner`. The engine's
continuous-batch decode goes through `GQAAttention._decode_batched` instead, which
reads paged-KV (block-table, int8 quantize-on-write) via `PagedKVCache` -- see
`superl8serve/engine/kv_cache.py`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import superl8


class Fni8Attention(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int, *,
                 scale: float | None = None, kv_cache_dtype: str = "int8"):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale or head_dim ** -0.5
        self.kv_cache_dtype = kv_cache_dtype

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                *, is_prefill: bool) -> torch.Tensor:
        """q,k,v: [B, H, S, D] fp16. Prefill = causal over the sequence; decode = M=1
        against the cache (here k/v are the full cache slice)."""
        if is_prefill:
            return superl8.attn_int8_fwd(q, k, v, causal=True, scale=self.scale)
        # decode: one query row against the cached K/V.
        if self.kv_cache_dtype == "int8":
            k_i8, k_scale, v_i8, v_scale = superl8.quantize_kv_cache(k, v, rotate=False)
            return superl8.attn_decode_cached(q, k_i8, k_scale, v_i8, v_scale, scale=self.scale)
        return superl8.attn_int8_decode(q, k, v, scale=self.scale)
