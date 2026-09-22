# SPDX-License-Identifier: MIT
"""GPU smoke for the K8V3 paged cache decode path (superl8-serve#410).

Validates on one pinned card that the K8V3 write+dequant-decode path is
shaped correctly and finite end-to-end (int8 K + 3-bit Lloyd-Max V written via
the cache, decoded through the fp16-input `attn_int8_decode` fallback). Not an
accuracy test — the needle eval is. Run:

    CUDA_VISIBLE_DEVICES=8 python3 bench/k8v3_gpu_smoke.py
"""
from __future__ import annotations

import time

import torch

import superl8

from superl8serve.engine.kv_cache import PagedKVCache

HD, NKV, BS = 256, 4, 16
KV_LAYERS = [i for i in range(64) if (i + 1) % 4 == 0]


def main() -> None:
    dev = "cuda"
    print(f"device: {dev} {torch.cuda.get_device_name(0)}")
    cache = PagedKVCache(
        64, 2, NKV, 4096, HD, device=dev, block_size=BS, kv_layers=KV_LAYERS, v_quant="lloydmax3"
    )
    cache.num_blocks = cache.num_blocks  # touch to silence lint
    s0, s1 = cache.alloc(), cache.alloc()
    cache.ensure_capacity([s0, s1], [64, 32])
    t0 = time.perf_counter()
    for layer in (3, 7, 63):
        k = torch.randn(1, NKV, 64, HD, dtype=torch.float16, device=dev) * 0.5
        v = torch.randn(1, NKV, 64, HD, dtype=torch.float16, device=dev) * 0.5
        cache.write_prefill(layer, k, v, slot=s0)
        k2 = torch.randn(1, NKV, 32, HD, dtype=torch.float16, device=dev) * 0.5
        v2 = torch.randn(1, NKV, 32, HD, dtype=torch.float16, device=dev) * 0.5
        cache.write_prefill(layer, k2, v2, slot=s1)
    torch.cuda.synchronize()
    print(f"prefill write done in {time.perf_counter() - t0:.2f}s")

    # one decode step over the ragged batch (both slots), lloydmax3 fallback path
    for layer in (3, 7, 63):
        q = torch.randn(2, 24, 1, HD, dtype=torch.float16, device=dev)
        out = cache.decode_attn(layer, q, [s0, s1], [64, 32], scale=HD**-0.5)
        assert out.shape == (2, 24, 1, HD), out.shape
        assert torch.isfinite(out).all()
    torch.cuda.synchronize()
    print("decode_attn (lloydmax3 fallback) OK, shapes", out.shape)

    # int8 (K8V8) same-storage comparison path still routes to the fused kernel
    c8 = PagedKVCache(64, 1, NKV, 4096, HD, device=dev, block_size=BS, kv_layers=KV_LAYERS, v_quant="int8")
    s = c8.alloc()
    c8.ensure_capacity([s], [64])
    c8.write_prefill(3, torch.randn(1, NKV, 64, HD, dtype=torch.float16, device=dev),
                     torch.randn(1, NKV, 64, HD, dtype=torch.float16, device=dev), slot=s)
    out8 = c8.decode_attn(3, torch.randn(1, 24, 1, HD, dtype=torch.float16, device=dev), [s], [64], scale=HD**-0.5)
    assert torch.isfinite(out8).all()
    print("k8v8 int8 fused decode OK", out8.shape)

    # memory accounting check
    bb = superl8serve_kv_block_bytes(NKV, BS, HD)
    print(f"k8v3 block bytes: {bb} ({bb/1024:.2f} KiB); 262k single-slot: {16 * (262144//BS) * bb / 2**30:.2f} GiB")


def superl8serve_kv_block_bytes(nkv, bs, hd):
    from superl8serve.engine.kv_cache import KVEviction

    return KVEviction.block_bytes(nkv, bs, hd, v_quant="lloydmax3")


if __name__ == "__main__":
    main()
