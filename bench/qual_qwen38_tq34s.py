#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Native Qwen3.8-27B TQ3_4S serve qualification harness (superl8-serve#412).

Serves `Qwen3.8-27B-MTP-TQ3_4S.gguf` on the superl8 path (`load_gguf_engine`, fused
`gemm_tq34s` kernels resident, K8V3/K8V8 cache on the 16 full-attention layers)
and measures the three acceptance gates on ONE free CMP 100-210 card:

  * --mode smoke  thinking + tool-style completions through the engine (chat
                  template applied by the HF tokenizer);
  * --mode perf   decode tok/s (short + long context) and prefill tok/s, per
                  cache_format (int8 / k8v3) and graph mode, with decode-only
                  timing fenced by a device sync per step (the prefill step is
                  excluded from decode tok/s);
  * --mode spec   speculative-decode A/B (off vs the n-gram/tree cascade drafter
                  — the artifact carries only the SHALLOW shared-head nextn, so
                  the deep MTP head is not buildable; the superl8 spec path is the
                  `attn_tree_fwd` n-gram verify). Greedy bit-identity is checked.

Container run (superl8-built image with the fused kernels baked in):
    docker run --rm --gpus '"device=8"' --entrypoint bash -e CUDA_VISIBLE_DEVICES=0 \
        -v $PWD:/work -v /path/to/qwen38-tq3:/qwen38-gguf:ro \
        -v /path/to/model-src:/qwen38-model-src:ro \
        -w /work -e PYTHONPATH=/work local/superl8-built:tq34s \
        bash -lc 'pip install -q -e ".[dev,convert,serve,structured]" --no-build-isolation && \
                  python3 bench/qual_qwen38_tq34s.py --mode smoke'
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time

import torch

GGUF = os.environ.get("QWEN38_GGUF", "/qwen38-gguf/Qwen3.8-27B-MTP-TQ3_4S.gguf")
TOKENIZER_DIR = os.environ.get("QWEN38_TOK", "/qwen38-model-src")
DEVICE = os.environ.get("QUAL_DEVICE", "cuda")

# Fleet caveat (hard rule): CMP 100-210 / GV100 / sm_70 / 16 GiB HBM2, dp4a int
# pipe. The pinned card is chosen by the caller via CUDA_VISIBLE_DEVICES.
import torch as _t  # noqa: E402


def _device_label() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return f"cuda:{_t.cuda.current_device()} {_t.cuda.get_device_name(0)}"


def _load(
    cache_format: str,
    max_len: int,
    spec_decode: bool,
    cuda_graph: bool,
    chunked_prefill_size: int = 0,
):
    from superl8serve.gguf_native import load_gguf_engine

    # `load_gguf_engine` does not expose enable_cuda_graph; graphs are selected by
    # the engine's env defaults (SUPERL8SERVE_CUDA_GRAPH / SUPERL8SERVE_LAYER_GRAPH).
    # Set them here so this harness's `--cuda-graph`/QUAL_CUDA_GRAPH flag is honored.
    if not cuda_graph:
        os.environ["SUPERL8SERVE_CUDA_GRAPH"] = "0"
        os.environ["SUPERL8SERVE_LAYER_GRAPH"] = "0"
    return load_gguf_engine(
        GGUF,
        device=DEVICE,
        max_num_seqs=1,
        max_len=max_len,
        spec_decode=spec_decode,
        cache_format=cache_format,
        chunked_prefill_size=chunked_prefill_size,
    )


def _encode_prompt(tok, messages) -> list[int]:
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return ids[0] if ids and isinstance(ids[0], list) else ids


def smoke(tok, max_tokens: int, cuda_graph: bool, chunked_prefill_size: int) -> dict:
    from superl8serve.engine.sequence import SamplingParams

    # max_len 4096 OOMs the card (model 12.7 GiB + dense-27B graph workspace
    # ~15.6 GiB); 2048 fits with headroom on one 16 GiB card.
    engine = _load(
        "k8v8", 1024, spec_decode=False, cuda_graph=cuda_graph,
        chunked_prefill_size=chunked_prefill_size,
    )
    cases = {
        "thinking": [
            {"role": "user", "content": "A farmer has 17 sheep and all but 9 run away. How many are left? Think step by step."}
        ],
        "tool": [
            {"role": "system", "content": "You have access to functions: get_weather(city: str). Output a function call."},
            {"role": "user", "content": "What is the weather in Paris? Call the function."},
        ],
    }
    out = {}
    for name, msgs in cases.items():
        prompt = _encode_prompt(tok, msgs)
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        t0 = time.perf_counter()
        ids = engine.generate([prompt], params)[0]
        wall = time.perf_counter() - t0
        text = tok.decode(ids, skip_special_tokens=True)
        out[name] = {"prompt_tokens": len(prompt), "out_tokens": len(ids), "wall_s": round(wall, 2), "text": text}
        print(f"\n=== {name} smoke (prompt {len(prompt)} tok, {len(ids)} out, {wall:.2f}s) ===")
        print(text)
    return out


def _decode_tps(engine, prompt, max_tokens, tok) -> tuple[list[int], float, float]:
    """One run: prefill time + decode-only tok/s (per-step device sync). Returns
    (output_ids, prefill_s, decode_tps)."""
    from superl8serve.engine.sequence import SamplingParams

    sid = engine.add_request(list(prompt), SamplingParams(temperature=0.0, max_tokens=max_tokens))
    prefill_s = 0.0
    decode_wall = 0.0
    n_decode = 0
    while engine.scheduler.has_work():
        batch, is_prefill = engine.scheduler.schedule()
        if not batch:
            continue
        if is_prefill:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            toks = engine.runner.prefill(batch)
            torch.cuda.synchronize()
            prefill_s = time.perf_counter() - t0
            if toks is not None:
                for seq, tokid in zip(batch, toks):
                    seq.output_ids.append(int(tokid))
        else:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            toks = engine.runner.decode(batch)
            torch.cuda.synchronize()
            decode_wall += time.perf_counter() - t0
            if toks is not None:
                for seq, tokid in zip(batch, toks):
                    seq.output_ids.append(int(tokid))
            n_decode += 1
        engine.scheduler.postprocess(batch, is_prefill)
    out = list(engine._out[sid].output_ids)
    engine.forget(sid)
    prefill_tok_s = len(prompt) / prefill_s if prefill_s > 0 else 0.0
    decode_toks = len(out)
    decode_tps = decode_toks / decode_wall if decode_wall > 0 else 0.0
    return out, prefill_tok_s, decode_tps


def _reset(engine):
    engine.scheduler.waiting.clear()
    engine.scheduler.running.clear()
    engine._out.clear()


def require_spec_engagement(
    stats: dict, gate_detail: dict | None = None, gate_reason: str | None = None
) -> None:
    """Reject timings where the safety gate silently kept speculation off."""
    if stats.get("steps", 0) <= 0 or stats.get("drafts", 0) <= 0:
        raise RuntimeError(
            "speculative decode did not engage "
            f"(steps={stats.get('steps', 0)}, drafts={stats.get('drafts', 0)}, "
            f"accepts={stats.get('accepts', 0)}, gate_reason={gate_reason}, "
            f"gate_memory={gate_detail}); "
            "refusing to publish plain-decode noise"
        )


def spec_acceptance_metrics(
    stats: dict,
    emitted_tokens: int,
    gate_detail: dict | None = None,
    gate_reason: str | None = None,
) -> dict[str, float]:
    require_spec_engagement(stats, gate_detail, gate_reason)
    return {
        "al": emitted_tokens / stats["steps"],
        "acc_rate": stats["accepts"] / stats["drafts"],
    }


def perf(tok, ctxs: list[int], repeats: int, cuda_graph: bool, chunked_prefill_size: int) -> dict:
    from superl8serve.engine.sequence import SamplingParams

    results = {}
    for fmt in ("k8v8", "k8v3"):
        # The KV cache is pre-sized to max_len, so cap it at the largest context
        # this format can physically hold on a 16 GiB card (k8v8 int8 V is 32
        # KiB/token/layer; k8v3's 3-bit V is 22.5 KiB/token/layer).
        run_ctxs = [c for c in ctxs if not (fmt == "k8v8" and c > 16384)]
        if not run_ctxs:
            print(f"[perf] skip {fmt}: no context in range fits one 16 GiB card", flush=True)
            continue
        # The timing loop decodes 64 tokens after the prompt.  Reserve room for
        # those output tokens as well as the requested prompt context; otherwise
        # an exact-size cache reports "no free blocks" before the measurement.
        engine = _load(
            fmt, max(run_ctxs) + 64, spec_decode=False, cuda_graph=cuda_graph,
            chunked_prefill_size=chunked_prefill_size,
        )
        _reset(engine)
        # warmup
        prompt = _encode_prompt(tok, [{"role": "user", "content": "Say hello."}])
        engine.generate([prompt], SamplingParams(temperature=0.0, max_tokens=8))
        _reset(engine)
        for ctx in run_ctxs:
            base = _encode_prompt(tok, [{"role": "user", "content": "The quick brown fox jumps over the lazy dog. "}])
            prompt = []
            while len(prompt) < ctx:
                prompt.extend(base)
            prompt = prompt[:ctx]
            dec_tps = []
            pref_tps = []
            out = None
            for _ in range(repeats):
                _reset(engine)
                out, pf, dt = _decode_tps(engine, prompt, 64, tok)
                dec_tps.append(dt)
                pref_tps.append(pf)
            key = f"{fmt}_ctx{ctx}"
            results[key] = {
                "format": fmt,
                "ctx": ctx,
                "cuda_graph": cuda_graph,
                "prefill_tok_s": round(statistics.median(pref_tps), 1),
                "decode_tok_s": round(statistics.median(dec_tps), 2),
                "decode_tok_s_all": [round(x, 2) for x in dec_tps],
                "out": out,
            }
            print(f"[perf] {fmt} ctx={ctx}: prefill {results[key]['prefill_tok_s']} tok/s, "
                  f"decode {results[key]['decode_tok_s']} tok/s (graph={cuda_graph})", flush=True)
        del engine
        torch.cuda.empty_cache()
    return results


def spec(
    tok,
    max_tokens: int,
    repeats: int,
    cuda_graph: bool,
    spec_k: int,
    chunked_prefill_size: int,
    cache_format: str,
) -> dict:
    from superl8serve.engine.drafters import NgramDrafter

    # Prompt lookup is an input-grounded accelerator, not a general prose drafter.
    # Qualify only the issue's repetitive structured workload; creative prose misses
    # correctly fall back to ordinary decode and are not a spec benchmark.
    STRUCTURED = (
        "Repeat this JSON exactly three times as a list:\n"
        '{"name": "widget", "price": 9.99, "tags": ["a", "b", "c"], "in_stock": true}'
    )

    engine = _load(
        cache_format, 8192, spec_decode=False, cuda_graph=cuda_graph,
        chunked_prefill_size=chunked_prefill_size,
    )
    r = engine.runner
    results = {}
    for label, text in (("structured", STRUCTURED),):
        prompt = _encode_prompt(tok, [{"role": "user", "content": text}])
        row = {}
        for mode in ("off", "on"):
            r._spec_enabled = mode == "on"
            if mode == "on":
                r._drafter_mode = "cascade"
                r._spec_k = spec_k
                r._ngram = NgramDrafter(min_n=2, max_n=4, max_k=spec_k)
            _reset(engine)
            r.spec_stats = {"steps": 0, "drafts": 0, "accepts": 0}
            r.spec_gate_reason = None
            # independent warmup for this mode
            _decode_tps(engine, prompt, 8, tok)
            _reset(engine)
            r.spec_stats = {"steps": 0, "drafts": 0, "accepts": 0}
            r.spec_gate_reason = None
            tps_samples = []
            out = None
            stats = None
            for _ in range(repeats):
                _reset(engine)
                out, _, dt = _decode_tps(engine, prompt, max_tokens, tok)
                tps_samples.append(dt)
                stats = dict(r.spec_stats)
            metrics = (
                spec_acceptance_metrics(
                    stats, len(out), r.spec_gate_memory, r.spec_gate_reason
                )
                if mode == "on"
                else {"al": 1.0, "acc_rate": 0.0}
            )
            row[mode] = {
                "tps": round(statistics.median(tps_samples), 2),
                "tps_all": [round(x, 2) for x in tps_samples],
                "al": round(metrics["al"], 3),
                "acc_rate": round(metrics["acc_rate"], 3),
                "out": out,
            }
            print(f"[spec] {label} {mode}: {row[mode]['tps']} tok/s "
                  f"(AL {row[mode]['al']}, acc-rate {row[mode]['acc_rate']})", flush=True)
        row["identical"] = row["on"]["out"] == row["off"]["out"]
        row["speedup"] = round(row["on"]["tps"] / row["off"]["tps"], 3) if row["off"]["tps"] else 0.0
        results[label] = row
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "perf", "spec"], required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--ctxs", type=int, nargs="*", default=[2048, 32768])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cuda-graph", action="store_true", default=None)
    ap.add_argument("--spec-k", type=int, default=4)
    ap.add_argument("--spec-cache-format", choices=["k8v8", "k8v3"], default="k8v3")
    ap.add_argument(
        "--chunked-prefill", type=int,
        default=int(os.environ.get("QUAL_CHUNKED_PREFILL", "0")),
        help="max prompt tokens per prefill forward (0 disables chunking)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    graph = args.cuda_graph if args.cuda_graph is not None else os.environ.get("QUAL_CUDA_GRAPH", "1") == "1"
    if args.chunked_prefill < 0:
        ap.error("--chunked-prefill must be >= 0")

    from transformers import AutoTokenizer

    print(f"[qual] {_device_label()}  model={os.path.basename(GGUF)}  "
          f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}", flush=True)
    assert os.path.exists(GGUF), f"GGUF not found: {GGUF}"
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    if args.mode == "smoke":
        out = smoke(tok, args.max_tokens, graph, args.chunked_prefill)
    elif args.mode == "perf":
        out = perf(tok, args.ctxs, args.repeats, graph, args.chunked_prefill)
    else:
        out = spec(
            tok,
            args.max_tokens,
            args.repeats,
            graph,
            args.spec_k,
            args.chunked_prefill,
            args.spec_cache_format,
        )

    if args.json:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
