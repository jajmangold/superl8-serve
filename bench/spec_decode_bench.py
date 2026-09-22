# SPDX-License-Identifier: MIT
"""Spec-decode CONTROLLED net-speedup benchmark: off / MTP / n-gram / cascade, on a
STRUCTURED prompt (code/JSON — where n-gram shines) and PROSE (where MTP carries).

Why this harness is written the way it is (review #250 follow-up — bench integrity)
-----------------------------------------------------------------------------------
The previous version reported 2.69× structured / 1.39× prose but the measurement was
CONTAMINATED and those numbers were NOT controlled:
  (a) it reused ONE engine for every mode, so the persistent RadixAttention prefix
      cache let each later mode REUSE the prefill KV produced by an earlier run of the
      SAME prompt (and inherit warmed kernels/allocations from earlier modes);
  (b) it always ran plain `off` FIRST, then MTP/n-gram/cascade in fixed order, so the
      ordering itself biased the comparison;
  (c) it timed the WHOLE `generate()` call INCLUDING prefill, so a mode that skipped
      prefill via the shared prefix cache looked faster for a reason unrelated to
      speculative decoding;
  (d) it ran EAGER (`enable_cuda_graph=False`); the real deployment runs decode under
      CUDA graphs, which speeds up the *baseline* and therefore SHRINKS the honest
      spec-decode speedup. 2.69× was an eager number.

This harness fixes all four:
  * **prefix cache disabled** (`_disable_prefix_cache`) — every mode recomputes prefill
    from scratch; no cross-run KV reuse;
  * **independent warmup per mode** — each mode warms its OWN kernels/allocations before
    it is timed, so no mode inherits another's warm state;
  * **randomized mode order** per prompt — ordering cannot bias the result;
  * **decode-only timing** — prefill is excluded; we sum wall time over decode steps
    only (each fenced with a device sync) and report steady-state decode tok/s, plus
    acceptance-length (AL) and accepted-tokens/step separately for the spec modes;
  * **regime is explicit** — `BENCH_CUDA_GRAPH` (default 1 = graphs-on, the deployment
    regime) is printed in the header; run with 0 to reproduce the old eager regime.
The valuable part of the old bench — greedy **bit-identity** vs plain decode — is kept.

Run in the superl8-serve test image on ONE FREE gpu (NEVER GPU4 — the live server):
    docker run --rm --gpus '"device=7"' --entrypoint python3 -e CUDA_VISIBLE_DEVICES=0 \
        -v $PWD:/work -w /work -e PYTHONPATH=/work \
        superl8-serve-test:latest bench/spec_decode_bench.py
"""
from __future__ import annotations

import os
import random
import statistics
import time

import torch

from superl8serve.engine.drafters import NgramDrafter
from superl8serve.engine.llm_engine import LLMEngine
from superl8serve.engine.sequence import SamplingParams
from superl8serve.loader import checkpoint_info, load_superl8_state_dict
from superl8serve.models.config import ModelConfig

SUPERL8 = os.environ.get("QWEN35_9B_SUPERL8", os.path.join(os.environ.get("SUPERL8_WEIGHTS_DIR", "/path/to/weights"), "Qwen__Qwen3.5-9B.b4.superl8"))
TOK = os.environ.get("QWEN35_9B_TOK", os.path.join(os.environ.get("SUPERL8_WEIGHTS_DIR", "/path/to/weights"), "tok"))
MAXTOK = int(os.environ.get("BENCH_MAXTOK", "256"))
SPEC_K = int(os.environ.get("BENCH_SPEC_K", "6"))
MAXLEN = int(os.environ.get("BENCH_MAXLEN", "1024"))
REPEATS = int(os.environ.get("BENCH_REPEATS", "3"))
CUDA_GRAPH = os.environ.get("BENCH_CUDA_GRAPH", "1") == "1"
SEED = int(os.environ.get("BENCH_SEED", "0"))

MODES = ("off", "mtp", "ngram", "cascade")

STRUCTURED = (
    "Repeat this JSON exactly three times as a list:\n"
    '{"name": "widget", "price": 9.99, "tags": ["a", "b", "c"], "in_stock": true}'
)
PROSE = "Explain, in a short paragraph, why the sky appears blue during the day."


def encode(tok, text):
    ids = tok.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True
    )
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return ids[0] if ids and isinstance(ids[0], list) else ids


def _disable_prefix_cache(eng):
    """Kill cross-run KV reuse: make the RadixAttention prefix cache a no-op so every
    run recomputes prefill from scratch and no mode inherits an earlier run's prefix.
    This is the core integrity fix — without it, running the same prompt across modes
    on one engine lets later modes skip prefill via the shared prefix.
    """
    eng.cache.lookup_prefix = lambda token_ids: (0, [])
    eng.cache.store_prefix = lambda token_ids, slot: None


def set_mode(runner, mode, k):
    """Configure the runner for one drafter mode (bypasses env config, as the old
    bench did). `off` disables spec entirely; the spec modes select the drafter."""
    runner._spec_enabled = mode != "off"
    if mode == "off":
        return
    runner._drafter_mode = mode
    runner._spec_k = k
    runner._ngram = NgramDrafter(min_n=2, max_n=4, max_k=k) if mode in ("cascade", "ngram") else None
    # ngram-only: null the MTP fallback so a miss proposes nothing that step; cascade
    # keeps the MTP fallback. The runner reads `_drafter_mode == "ngram"` for this.
    runner._ngram_only = mode == "ngram"


def _reset_between_runs(eng):
    """Return the engine to a clean per-run state WITHOUT rebuilding the model or the
    CUDA-graph workspace (both stay warm — the point is that every mode starts equally
    warm, not cold). Finished sequences already freed their slots/blocks in
    `scheduler.postprocess`; the graph's never-freed scratch slot is left untouched.
    Recurrent/MTP per-slot state is re-initialised by the next prefill (prefill
    `clear_slot`+`bind`s the slot it allocates), so no stale trajectory leaks across
    runs and no explicit cache wipe is needed here."""
    eng.scheduler.waiting.clear()
    eng.scheduler.running.clear()
    eng._out.clear()


def controlled_generate(eng, prompt, max_tokens):
    """Replicates `LLMEngine.step` for the non-diffusion path but times ONLY the decode
    steps (prefill excluded). Each decode step is fenced with a device sync so the wall
    time reflects real GPU work. Returns (output_ids, decode_wall_s, n_decode_steps).

    Spec-decode commits multiple tokens per decode step INTERNALLY (runner.decode
    returns None then), so decode-token count comes from the final output length, not
    from step count — AL = decode_tokens / n_decode_steps."""
    sid = eng.add_request(list(prompt), SamplingParams(temperature=0.0, max_tokens=max_tokens))
    decode_wall = 0.0
    n_decode_steps = 0
    while eng.scheduler.has_work():
        batch, is_prefill = eng.scheduler.schedule()
        if not batch:
            continue
        if is_prefill:
            toks = eng.runner.prefill(batch)
            if toks is not None:
                for seq, tok in zip(batch, toks):
                    seq.output_ids.append(int(tok))
        else:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            toks = eng.runner.decode(batch)
            torch.cuda.synchronize()
            decode_wall += time.perf_counter() - t0
            if toks is not None:  # plain decode returns tokens; spec appends internally
                for seq, tok in zip(batch, toks):
                    seq.output_ids.append(int(tok))
            n_decode_steps += 1
        eng.scheduler.postprocess(batch, is_prefill)
    out = list(eng._out[sid].output_ids)
    # Drop bookkeeping so `_out` stays bounded across the many runs of this bench.
    eng.forget(sid)
    return out, decode_wall, n_decode_steps


def measure(eng, prompt, mode, k, repeats):
    """One drafter mode: independent warmup, then `repeats` decode-only timed runs.
    Returns dict with median decode tok/s, output ids, AL, accept-rate, accepted/step."""
    r = eng.runner
    set_mode(r, mode, k)
    # Independent warmup for THIS mode (kernels + allocations + any graph capture).
    _reset_between_runs(eng)
    controlled_generate(eng, prompt, MAXTOK)

    tps_samples = []
    out = None
    stats = None
    for _ in range(repeats):
        _reset_between_runs(eng)
        r.spec_stats = {"steps": 0, "drafts": 0, "accepts": 0}
        out, decode_wall, n_steps = controlled_generate(eng, prompt, MAXTOK)
        decode_tokens = len(out) - 1  # first token is emitted by prefill
        tps_samples.append(decode_tokens / decode_wall if decode_wall > 0 else 0.0)
        stats = dict(r.spec_stats)

    decode_tokens = len(out) - 1
    steps = stats["steps"] if stats["steps"] else n_steps
    al = decode_tokens / steps if steps else 1.0
    acc_rate = (stats["accepts"] / stats["drafts"]) if stats["drafts"] else 0.0
    acc_per_step = (stats["accepts"] / stats["steps"]) if stats["steps"] else 0.0
    return {
        "out": out,
        "tps": statistics.median(tps_samples),
        "tps_all": tps_samples,
        "al": al,
        "acc_rate": acc_rate,
        "acc_per_step": acc_per_step,
        "stats": stats,
    }


def main():
    from transformers import AutoTokenizer

    random.seed(SEED)
    torch.manual_seed(SEED)

    meta_cfg = dict(checkpoint_info(SUPERL8)["meta"]["config"])
    cfg = ModelConfig.from_hf(meta_cfg, arch="qwen3_5")
    weights = load_superl8_state_dict(SUPERL8, device="cuda")
    eng = LLMEngine(
        cfg, weights, device="cuda", max_num_seqs=1, max_len=MAXLEN,
        enable_cuda_graph=CUDA_GRAPH,
    )
    _disable_prefix_cache(eng)
    tok = AutoTokenizer.from_pretrained(TOK)

    regime = "CUDA-GRAPHS ON (deployment)" if CUDA_GRAPH else "EAGER (matches old 2.69× regime)"
    print(f"\n### superl8-serve spec-decode CONTROLLED bench — {regime}")
    print(f"### model={os.path.basename(SUPERL8)}  max_new={MAXTOK}  spec_k={SPEC_K}  "
          f"repeats={REPEATS}  prefix_cache=DISABLED  timing=DECODE-ONLY  seed={SEED}")

    for label, text in (("STRUCTURED (JSON)", STRUCTURED), ("PROSE", PROSE)):
        prompt = encode(tok, text)
        # Randomized mode order so ordering cannot bias the comparison.
        order = list(MODES)
        random.shuffle(order)
        results = {}
        for mode in order:
            results[mode] = measure(eng, prompt, mode, SPEC_K, REPEATS)

        ref = results["off"]["out"]
        tps_off = results["off"]["tps"]
        print(f"\n=== {label} — prompt {len(prompt)} tok, order={order} ===")
        print(f"  {'drafter':<10} {'dec tok/s':>10} {'speedup':>8} {'AL':>6} "
              f"{'acc/step':>9} {'acc-rate':>9}  identical")
        for mode in MODES:
            m = results[mode]
            spd = m["tps"] / tps_off if tps_off else 0.0
            ident = m["out"] == ref if mode != "off" else True
            al = "-" if mode == "off" else f"{m['al']:.2f}"
            aps = "-" if mode == "off" else f"{m['acc_per_step']:.2f}"
            acr = "-" if mode == "off" else f"{m['acc_rate']:.3f}"
            print(f"  {mode:<10} {m['tps']:>10.2f} {spd:>7.2f}x {al:>6} "
                  f"{aps:>9} {acr:>9}  {ident}")


if __name__ == "__main__":
    main()
