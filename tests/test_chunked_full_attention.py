# SPDX-License-Identifier: MIT
"""Qwen3.5 chunked full attention must not recompute padded prefix queries."""

from types import SimpleNamespace

import torch
import torch.nn as nn

import superl8
from superl8serve.layers.gated_gqa_attention import GatedGQAAttention


class _KVCache:
    def __init__(self, k_all, v_all):
        self.k_all = k_all
        self.v_all = v_all
        self.read_calls = []

    def write_prefill(self, *args, **kwargs):
        pass

    def read_dense(self, layer, slot, length, **kwargs):
        self.read_calls.append((layer, slot, length, kwargs))
        return self.k_all[:, :, :length], self.v_all[:, :, :length]


def test_chunked_prefill_passes_only_current_queries(monkeypatch):
    attn = GatedGQAAttention.__new__(GatedGQAAttention)
    nn.Module.__init__(attn)
    attn.nh, attn.nkv, attn.hd = 4, 2, 8
    attn.scale, attn.window_left, attn.causal = 0.125, -1, True

    current = 3
    q = torch.randn(1, current, attn.nh, attn.hd)
    gate = torch.randn_like(q)
    k = torch.randn(1, current, attn.nkv, attn.hd)
    v = torch.randn_like(k)
    attn._project = lambda _x: (q, gate, k, v)
    attn.rope = lambda _positions, q_in, k_in: (q_in, k_in)
    attn._gate_and_project = lambda out, _gate, _b, _s: out

    prefix = 5
    k_prev = torch.randn(1, attn.nkv, prefix, attn.hd)
    v_prev = torch.randn_like(k_prev)
    cache = _KVCache(
        torch.cat([k_prev, k.transpose(1, 2)], dim=2),
        torch.cat([v_prev, v.transpose(1, 2)], dim=2),
    )
    ctx = SimpleNamespace(
        is_prefill=True,
        cu_seqlens=None,
        slots=[0],
        kv_cache=cache,
        slot_mapping=None,
        prefill_start=prefix,
        prefill_length=prefix + current,
        acc_kv_buffer=[],
    )
    seen = {}

    def fake_attn(q_in, k_in, v_in, **kwargs):
        seen.update(q=q_in, k=k_in, v=v_in, kwargs=kwargs)
        return q_in

    monkeypatch.setattr(superl8, "attn_int8_fwd", fake_attn)
    out = attn(torch.empty(1, current, 1), torch.arange(prefix, prefix + current), ctx, 0)

    assert seen["q"].shape == (1, attn.nh, current, attn.hd)
    assert seen["k"].shape == seen["v"].shape == (1, attn.nkv, prefix + current, attn.hd)
    assert seen["kwargs"]["causal"] is True
    assert out.shape == (1, current, attn.nh, attn.hd)
    assert cache.read_calls == [(0, 0, prefix + current, {"dtype": q.dtype})]
    assert ctx.acc_kv_buffer == [], "chunked prefill must not retain duplicate fp K/V"
