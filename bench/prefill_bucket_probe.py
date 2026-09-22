# SPDX-License-Identifier: BSD-3-Clause
"""Direct-engine (no HTTP) probe of CUDA-graph prefill/decode throughput at
increasing concurrency, to find where it peaks or breaks. Bypasses the API
server entirely to remove HTTP/uvicorn overhead as a variable.

Usage:
    python3 -m bench.prefill_bucket_probe --model /path/to/model.b8.superl8 \
        --tokenizer Qwen/Qwen3.5-9B --buckets 1,2,4,8,16,32,64
"""
from __future__ import annotations

import argparse
import time
import traceback

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--buckets", default="1,2,4,8,16,32")
    ap.add_argument("--max-tokens", type=int, default=50)
    ap.add_argument("--spec-decode", action="store_true")
    ap.add_argument("--prefill-buckets", default=None,
                     help="Override GraphedPrefill's hardcoded prompt-length "
                          "buckets (comma-separated), e.g. for finer-grained "
                          "padding-waste testing.")
    ap.add_argument("--cache-format", choices=["int8", "k8v8", "k8v3"],
                     default="int8")
    args = ap.parse_args()

    buckets = tuple(int(x) for x in args.buckets.split(","))
    max_seqs = max(buckets)

    print(f"torch {torch.__version__}, CUDA {torch.cuda.is_available()}, "
          f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

    from transformers import AutoTokenizer
    from superl8serve.api.server import load_engine
    from superl8serve.engine.sequence import SamplingParams

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    t0 = time.time()
    engine = load_engine(
        args.model, device="cuda", max_num_seqs=max_seqs, max_len=2048,
        cuda_graph_batch_buckets=buckets,
        spec_decode=args.spec_decode or None,
        cache_format=args.cache_format,
    )
    print(f"[load] {time.time() - t0:.1f}s, max_num_seqs={max_seqs}, buckets={buckets}")

    if args.prefill_buckets:
        from superl8serve.engine.cuda_graph import GraphedPrefill
        pb = tuple(int(x) for x in args.prefill_buckets.split(","))
        engine.runner.graphed_prefill = GraphedPrefill(
            engine.runner.model, engine.runner.cache, device=engine.runner.device,
            lin_cache=engine.runner.lin_cache,
            seq_buckets=pb, max_graphs=max(16, len(pb)),
        )
        print(f"[graphed_prefill override] seq_buckets={pb}")

    gp = getattr(engine.runner, "graphed_prefill", None)
    print(f"[graphed_prefill] present={gp is not None} "
          f"seq_buckets={getattr(gp, 'seq_buckets', 'n/a')} "
          f"supported={getattr(gp, 'supported', 'n/a')}")

    passage = (
        "The history of distributed computing systems traces back to the early "
        "mainframe era, when time-sharing systems first allowed multiple users to "
        "interact with a single powerful machine concurrently. As networking "
        "technology matured through the 1970s and 1980s, researchers began exploring "
        "how independent computers could cooperate on shared tasks."
    ) * 2
    msg = [{"role": "user", "content": "Summarize in two sentences.\n\n" + passage}]
    prompt_ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_dict=False)
    print(f"[prompt] {len(prompt_ids)} tokens")

    params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    for n in buckets:
        prompts = [list(prompt_ids) for _ in range(n)]
        # warmup (captures the graph for this bucket if not already captured)
        try:
            engine.generate(prompts[:1], params)
        except Exception as e:
            print(f"[bucket={n}] WARMUP FAILED: {e!r}")
            traceback.print_exc()
            continue

        # run it 3 times back to back -- is it steady-state slow, or a
        # one-time cost on the first "real" call after warmup?
        for rep in range(3):
            torch.cuda.synchronize()
            t0 = time.time()
            try:
                outs = engine.generate(prompts, params)
            except Exception as e:
                print(f"[bucket={n}] GENERATE FAILED (rep {rep}): {e!r}")
                traceback.print_exc()
                break
            torch.cuda.synchronize()
            dt = time.time() - t0

            total_out_tokens = sum(len(o) for o in outs)
            total_prefill_tokens = len(prompt_ids) * n
            decode_tokens = total_out_tokens - n
            gp = getattr(engine.runner, "graphed_prefill", None)
            caps = getattr(gp, "_capture_count", "n/a")
            reps_ = getattr(gp, "_replay_count", "n/a")
            misses = getattr(gp, "_miss_counts", "n/a")
            seq_buckets = getattr(gp, "seq_buckets", "n/a")
            graphs_cached = list(getattr(gp, "_graphs", {}).keys())
            resolved_bucket = gp._next_bucket(len(prompt_ids)) if gp is not None else "n/a"
            spec = getattr(engine.runner, "spec_stats", {})
            accept_rate = (spec.get("accepts", 0) / spec["drafts"]) if spec.get("drafts") else 0.0
            print(
                f"[bucket={n:3d} rep={rep}] wall={dt:6.2f}s  "
                f"prefill_tokens={total_prefill_tokens:6d}  "
                f"out_tokens={total_out_tokens:5d}  "
                f"approx_decode_tok/s={decode_tokens / dt:8.1f}  "
                f"approx_total_tok/s={total_out_tokens / dt:8.1f}  "
                f"captures={caps} replays={reps_} misses={misses}\n"
                f"    seq_buckets={seq_buckets} resolved_bucket={resolved_bucket} "
                f"cached_graph_keys={graphs_cached}\n"
                f"    spec_stats={spec} accept_rate={accept_rate:.2%}"
            )


if __name__ == "__main__":
    main()
