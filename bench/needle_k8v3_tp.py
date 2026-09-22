#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Two-card pipeline-parallel full-model Qwen3.8 K8V3 vs K8V8 needle gate
(superl8-serve#440).

The one-card exact32k full-model semantic needle is peak-memory blocked
(15.73 GiB process misses a 32 MiB Q/K temporary; #437 evidence) and the
reduced 16-full-attention-layer scope is not a valid semantic oracle (both
formats emit dots; #437 evidence). This harness runs the COMPLETE 64-layer
qwen3_5 hybrid on TWO cards as a 32/32 pipeline-parallel split — the
already-supported topology in `superl8serve/dist/pipeline.py` (and the
community-standard layer split for this fleet; qengine / llama.cpp
`--split-mode layer`) — so the K8V3 vs K8V8 recall decision comes from
complete-model evidence only.

  * the model is assembled once on CPU from the production GGUF remap chain
    (`needle_k8v3._load_and_remap`), split into stage groups, and moved
    stage-local (one copy of shared RoPE tables via `make_pipeline`);
  * chunked prefill mirrors production `EngineRunner._prefill_chunked`
    (absolute positions, monotonically increasing `prefill_length`, paged
    cache as the sole accumulated K/V store, persistent recurrent bind);
  * the cross-GPU boundary carries (hidden, residual) fp16 — the lossless
    transport scheme — so the two-card result is the single-card math with
    a bit-exact wire, not a lossy codec comparison;
  * the KV cache is per-stage PagedKVCache with the full-attention layer set
    (8 per stage) in the same K8V3 (int8 K + 3-bit Lloyd-Max V) or K8V8
    (int8 K + int8 V) format the production engine serves;
  * greedy decode is argmax with the exact same prompt/model/sampling for
    both formats; the harness records per-card peak VRAM, card UUIDs,
    output ids + hash, recall, prefill/decode wall, and wire overhead.

Modes:
  two-card (default): complete 64-layer model on `--devices` (32/32 split).
  one-card:           production `load_gguf_engine` full model on the FIRST
                      `--devices` entry — the single-card reference arm used
                      to prove the two-card topology is output-identical.
  probe:              build only; report per-card residency (no prefill).

Usage:
  python3 bench/needle_k8v3_tp.py --mode two-card --ctx 32768 --format k8v3 --devices 11,14
  python3 bench/needle_k8v3_tp.py --mode one-card --ctx 2048 --format k8v3 --devices 11
  python3 bench/needle_k8v3_tp.py --mode probe --ctx 32768 --format k8v3 --devices 11,14
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from needle_k8v3 import (
    ANSWER,
    GGUF,
    TOKENIZER_DIR,
    build_full,
    build_prompt,
    cache_capacity,
    chunked_prefill,
    decode_step,
    greedy_from_prefill,
)

from superl8serve.dist import recv, send
from superl8serve.dist.pipeline import (
    _pack_boundary,
    _unpack_boundary,
    make_pipeline,
)
from superl8serve.engine.kv_cache import PagedKVCache
from superl8serve.models.base import ForwardContext

WIRE_SCHEME = "fp16"  # lossless (bit-exact round-trip) boundary codec
BS = 16


def sha256_ids(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def stage_full_attn_layers(cfg, offset: int, count: int) -> list[int]:
    """Local indices (0..count-1) of full-attention layers in a stage holding
    original layers [offset, offset+count). Global index `i` is full-attn iff
    `cfg.attention_kind(i) == "full"`; a stage's K8V3/K8V8 cache is addressed
    with LOCAL indices (LayerRemappedCache subtracts the stage offset)."""
    return [i - offset for i in range(offset, offset + count) if cfg.attention_kind(i) == "full"]


def _stage_cache(cfg, device: str, max_len: int, cache_format: str, num_layers: int) -> PagedKVCache:
    local_kv = stage_full_attn_layers(cfg, 0, num_layers)
    return PagedKVCache(
        num_layers,
        1,
        cfg.num_key_value_heads,
        max_len,
        cfg.resolved_head_dim(),
        device=device,
        block_size=BS,
        kv_layers=local_kv,
        v_quant="lloydmax3" if cache_format == "k8v3" else "int8",
    )


def build_two_card(cfg, devices: tuple[int, ...], max_len: int, cache_format: str):
    """Full 64-layer model as a 32/32 pipeline split. Weights are assembled once
    on CPU via the production GGUF remap chain, split into stage groups, and
    moved stage-local by `make_pipeline`; the per-stage KV caches are replaced
    with format-aware caches over the stage's full-attention layer subset."""
    from needle_k8v3 import _load_and_remap

    cfg, weights = _load_and_remap(cfg, "cpu")
    stages = make_pipeline(
        cfg, weights, devices=devices, max_num_seqs=1, max_len=max_len, block_size=BS
    )
    n = len(stages)
    bounds = [i * cfg.num_hidden_layers // n for i in range(n + 1)]
    for st, start, end in zip(stages, bounds[:-1], bounds[1:]):
        st.kv_cache = _stage_cache(cfg, str(st._device), max_len, cache_format, end - start)
    return cfg, stages


def pp_chunked_prefill(stages, slot, prompt_ids, chunk_size, hidden_size):
    """Chunked prefill across both stages; (hidden, residual) crosses the
    boundary losslessly once per chunk. Mirrors production
    `_prefill_chunked`: absolute positions, prefill_length monotonic, paged
    cache sole K/V store, recurrent state bound once per slot."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    s0, s1 = stages
    n = len(prompt_ids)
    for st in stages:
        st.kv_cache.ensure_capacity([slot], [n])
        st.lin_cache.clear_slot(slot)
        st.lin_cache.bind([slot])
    hidden = residual = None
    wire_bytes = 0
    wire_ms = 0.0
    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        ids = prompt_ids[chunk_start:chunk_end]
        with torch.cuda.device(s0._device):
            ids_t = torch.tensor([ids], device=str(s0._device))
            pos = torch.arange(chunk_start, chunk_end, device=str(s0._device)).unsqueeze(0)
            ctx = ForwardContext(
                is_prefill=True,
                kv_cache=s0._remap(s0.kv_cache),
                lin_cache=s0.lin_cache,
                slots=[slot],
                prefill_start=0,
                prefill_length=chunk_end,
            )
            hidden = s0.embed(ids_t)
            residual = None
            for layer in s0.layers:
                hidden, residual = layer(hidden, pos, ctx, residual)
            packed = _pack_boundary(hidden.contiguous(), residual)
            t0 = time.perf_counter()
            handle = send(packed, dst=s1._device.index, scheme=WIRE_SCHEME)
            hidden, residual = _unpack_boundary(recv(handle), hidden_size)
            wire_ms += (time.perf_counter() - t0) * 1000.0
            wire_bytes += packed.numel() * packed.element_size()
        with torch.cuda.device(s1._device):
            pos = torch.arange(chunk_start, chunk_end, device=str(s1._device)).unsqueeze(0)
            ctx = ForwardContext(
                is_prefill=True,
                kv_cache=s1._remap(s1.kv_cache),
                lin_cache=s1.lin_cache,
                slots=[slot],
                prefill_start=0,
                prefill_length=chunk_end,
            )
            for layer in s1.layers:
                hidden, residual = layer(hidden, pos, ctx, residual)
    with torch.cuda.device(s1._device):
        hidden, _ = s1.norm(hidden, residual)
    return hidden, wire_bytes, wire_ms


def pp_decode_step(stages, slot, length, last_token, hidden_size):
    """One greedy decode step through both stages; the boundary carries the
    token's (hidden, residual) losslessly."""
    s0, s1 = stages
    for st in stages:
        st.kv_cache.ensure_capacity([slot], [length + 1])
        st.lin_cache.bind([slot])
    with torch.cuda.device(s0._device):
        ids = torch.tensor([[last_token]], device=str(s0._device))
        pos = torch.tensor([[length]], device=str(s0._device))
        ctx = ForwardContext(
            is_prefill=False,
            kv_cache=s0._remap(s0.kv_cache),
            lin_cache=s0.lin_cache,
            slots=[slot],
            slot_lengths=[length],
        )
        hidden = s0.embed(ids)
        residual = None
        for layer in s0.layers:
            hidden, residual = layer(hidden, pos, ctx, residual)
        packed = _pack_boundary(hidden.contiguous(), residual)
        t0 = time.perf_counter()
        handle = send(packed, dst=s1._device.index, scheme=WIRE_SCHEME)
        hidden, residual = _unpack_boundary(recv(handle), hidden_size)
        wire_ms = (time.perf_counter() - t0) * 1000.0
        wire_bytes = packed.numel() * packed.element_size()
    with torch.cuda.device(s1._device):
        pos = torch.tensor([[length]], device=str(s1._device))
        ctx = ForwardContext(
            is_prefill=False,
            kv_cache=s1._remap(s1.kv_cache),
            lin_cache=s1.lin_cache,
            slots=[slot],
            slot_lengths=[length],
        )
        for layer in s1.layers:
            hidden, residual = layer(hidden, pos, ctx, residual)
        hidden, _ = s1.norm(hidden, residual)
        logits = s1.lm_head(hidden[:, -1])
    return int(logits.argmax().item()), wire_bytes, wire_ms


def nvidia_uuids(devices: tuple[int, ...]) -> dict:
    """Map CUDA device index -> (uuid, name) via nvidia-smi.

    Inside a container CUDA_VISIBLE_DEVICES renumbers the visible devices; the
    env maps the harness's CUDA index back to the HOST index nvidia-smi reports
    (``CUDA_VISIBLE_DEVICES=11,14`` -> CUDA 0 = host 11, CUDA 1 = host 14).
    """
    out = {}
    try:
        rows = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError, ValueError):
        return out
    host = {}
    for row in rows:
        parts = [p.strip() for p in row.split(",")]
        if len(parts) == 3 and parts[0].isdigit():
            host[int(parts[0])] = {"uuid": parts[1], "name": parts[2]}
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cvd_order = [int(x) for x in cvd.split(",") if x.strip()] if cvd else None
    for d in devices:
        host_idx = cvd_order[d] if cvd_order and d < len(cvd_order) else d
        out[d] = host.get(host_idx, {"uuid": None, "name": None, "host_index": host_idx})
        if out[d]["name"]:
            out[d]["host_index"] = host_idx
    return out


def memory_report(devices: tuple[int, ...]) -> dict:
    torch.cuda.synchronize()
    report = {}
    for d in devices:
        report[str(d)] = {
            "max_alloc_gib": round(torch.cuda.max_memory_allocated(d) / 2**30, 3),
            "max_reserved_gib": round(torch.cuda.max_memory_reserved(d) / 2**30, 3),
            "cur_alloc_gib": round(torch.cuda.memory_allocated(d) / 2**30, 3),
        }
    return report


def run_two_card(cfg, devices, tok, prompt, max_tokens, chunk_size, cache_format):
    max_len = cache_capacity(len(prompt), max_tokens)
    t0 = time.perf_counter()
    cfg, stages = build_two_card(cfg, devices, max_len, cache_format)
    torch.cuda.synchronize()
    build_s = time.perf_counter() - t0
    s0, s1 = stages
    assert len(s0.layers) == cfg.num_hidden_layers // 2
    assert len(s1.layers) == cfg.num_hidden_layers - len(s0.layers)

    slot0, slot1 = s0.kv_cache.alloc(), s1.kv_cache.alloc()
    assert slot0 == slot1, f"stage cache slots diverged: {slot0} != {slot1}"
    slot = slot0

    t0 = time.perf_counter()
    hidden, wire_bytes, wire_ms = pp_chunked_prefill(
        stages, slot, prompt, chunk_size, cfg.hidden_size
    )
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    with torch.cuda.device(s1._device):
        logits = s1.lm_head(hidden[:, -1])
        first = int(logits.argmax().item())
    out_toks = [first]
    length = len(prompt)
    last = first
    wire_bytes_dec = 0
    wire_ms_dec = 0.0
    for _ in range(max(0, max_tokens - 1)):
        if last == tok.eos_token_id:
            break
        last, wb, wm = pp_decode_step(stages, slot, length, last, cfg.hidden_size)
        length += 1
        out_toks.append(last)
        wire_bytes_dec += wb
        wire_ms_dec += wm
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return {
        "mode": "two-card",
        "format": cache_format,
        "devices": list(devices),
        "layers_per_stage": [len(s0.layers), len(s1.layers)],
        "full_attn_per_stage": [len(stage_full_attn_layers(cfg, 0, len(s0.layers))),
                                len(stage_full_attn_layers(cfg, len(s0.layers), len(s1.layers)))],
        "model_build_s": round(build_s, 2),
        "prompt_tokens": len(prompt),
        "prefill_s": round(prefill_s, 2),
        "decode_s": round(elapsed, 2),
        "decode_tok_s": round(len(out_toks) / max(elapsed, 1e-9), 3),
        "output": tok.decode(out_toks, skip_special_tokens=True),
        "output_ids": out_toks,
        "output_sha256": sha256_ids(out_toks),
        "recall": ANSWER in tok.decode(out_toks, skip_special_tokens=True),
        "wire_bytes": wire_bytes + wire_bytes_dec,
        "wire_ms": round(wire_ms + wire_ms_dec, 2),
        "memory": memory_report(devices),
    }


def run_one_card(cfg, devices, tok, prompt, max_tokens, chunk_size, cache_format):
    device = devices[0]
    dev = f"cuda:{device}"
    max_len = cache_capacity(len(prompt), max_tokens)
    t0 = time.perf_counter()
    engine = build_full(cfg, dev, cache_format, max_len)
    model, cache, lin_cache = engine.model, engine.cache, engine.lin_cache
    torch.cuda.synchronize()
    build_s = time.perf_counter() - t0

    slot = cache.alloc()
    t0 = time.perf_counter()
    hidden = chunked_prefill(model, cache, lin_cache, slot, prompt, dev, chunk_size)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    first = greedy_from_prefill(model, hidden)
    out_toks = [first]
    length = len(prompt)
    last = first
    for _ in range(max(0, max_tokens - 1)):
        if last == tok.eos_token_id:
            break
        last = decode_step(model, cache, lin_cache, slot, length, last, dev)
        length += 1
        out_toks.append(last)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return {
        "mode": "one-card",
        "format": cache_format,
        "devices": list(devices),
        "layers": cfg.num_hidden_layers,
        "model_build_s": round(build_s, 2),
        "prompt_tokens": len(prompt),
        "prefill_s": round(prefill_s, 2),
        "decode_s": round(elapsed, 2),
        "decode_tok_s": round(len(out_toks) / max(elapsed, 1e-9), 3),
        "output": tok.decode(out_toks, skip_special_tokens=True),
        "output_ids": out_toks,
        "output_sha256": sha256_ids(out_toks),
        "recall": ANSWER in tok.decode(out_toks, skip_special_tokens=True),
        "wire_bytes": 0,
        "wire_ms": 0.0,
        "memory": memory_report((device,)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["two-card", "one-card", "probe"], default="two-card")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--format", choices=["k8v3", "k8v8"], default="k8v3")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--devices", default=os.environ.get("QUAL_DEVICES", "11,14"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if args.chunk_size <= 0:
        ap.error("--chunk-size must be greater than zero")
    if args.ctx <= 0 or args.max_tokens <= 0:
        ap.error("--ctx and --max-tokens must be greater than zero")
    devices = tuple(int(d) for d in args.devices.split(",") if d.strip())
    if not devices:
        ap.error("--devices must name at least one CUDA index")
    if args.mode == "two-card" and len(devices) < 2:
        ap.error("--mode two-card needs at least two devices")
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA available — two-card/one-card gates require CUDA")
    for d in devices:
        if d >= torch.cuda.device_count():
            ap.error(f"device {d} out of range (have {torch.cuda.device_count()})")

    torch.manual_seed(args.seed)
    # Eager, no graphs: matches the qualification runs and keeps the identity
    # comparison between one-card and two-card arms free of graph capture.
    os.environ["SUPERL8SERVE_CUDA_GRAPH"] = "0"
    os.environ["SUPERL8SERVE_LAYER_GRAPH"] = "0"

    from transformers import AutoTokenizer

    print(f"[tp-needle] mode={args.mode} format={args.format} ctx={args.ctx} "
          f"devices={devices} chunk={args.chunk_size}", flush=True)
    uuids = nvidia_uuids(devices)
    print(f"[tp-needle] cards: {json.dumps(uuids)}", flush=True)
    assert os.path.exists(GGUF), f"GGUF not found: {GGUF}"

    from superl8serve.gguf_native import gguf_config

    cfg = gguf_config(GGUF)
    if args.mode == "probe":
        t0 = time.perf_counter()
        _cfg, _stages = build_two_card(cfg, devices, cache_capacity(args.ctx, args.max_tokens), args.format)
        torch.cuda.synchronize()
        print(f"[tp-needle] probe build in {time.perf_counter() - t0:.1f}s", flush=True)
        print(json.dumps({"mode": "probe", "devices": list(devices), "uuids": uuids,
                          "memory": memory_report(devices),
                          "cfg": {"hidden": _cfg.hidden_size, "layers": _cfg.num_hidden_layers,
                                  "nkv": _cfg.num_key_value_heads, "hd": _cfg.resolved_head_dim()}}))
        return

    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    prompt = build_prompt(tok, args.ctx)
    print(f"[tp-needle] prompt tokens: {len(prompt)} (target {args.ctx}) "
          f"sha256={sha256_ids(prompt)}", flush=True)
    print(f"[tp-needle] gguf={os.path.basename(GGUF)} ({os.path.getsize(GGUF)} bytes)", flush=True)

    if args.mode == "two-card":
        result = run_two_card(cfg, devices, tok, prompt, args.max_tokens, args.chunk_size, args.format)
    else:
        result = run_one_card(cfg, devices, tok, prompt, args.max_tokens, args.chunk_size, args.format)

    print(f"[tp-needle] OUTPUT: {result['output']!r}")
    print(f"[tp-needle] RECALL: {result['recall']}  decoded {len(result['output_ids'])} tokens")
    print(f"[tp-needle] prefill {result['prefill_s']}s decode {result['decode_tok_s']} tok/s "
          f"wire {result['wire_bytes']} B in {result['wire_ms']}ms")
    print(f"[tp-needle] memory: {json.dumps(result['memory'])}")
    if args.json:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
