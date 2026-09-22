# SPDX-License-Identifier: MIT
"""Grammar tier-0 spec-decode E2E benchmark: how much faster does grammar-constrained
JSON-schema / tool-call output get when the grammar itself is the tier-0 drafter?

On structured output the grammar forces exactly one token at every structural position
(``{``, ``"``, field names, ``:``, ``,``, closing braces). The tier-0 grammar drafter
commits those forced runs with NO draft-model forward and feeds them into the shared
(grammar-masked) verify path, so acceptance-length jumps on exactly the heavily-forced
JSON/tool workloads. We compare, on the SAME grammar-constrained prompt:

  * off      — spec off, grammar on   (plain grammar-constrained greedy: the baseline)
  * grammar  — spec on, tier-0 grammar drafter ONLY (free forced runs, no MTP/n-gram)
  * cascade  — spec on, grammar → n-gram → MTP (the full stack)

and report decode tok/s, acceptance-length (AL = committed tokens / spec step), draft
accept-rate, and greedy bit-identity vs the baseline (MUST hold — the forced tokens are
exactly what plain grammar decode emits).

Run in the superl8-serve test image on ONE free gpu:
    docker run --rm --gpus '"device=5"' --entrypoint python3 -e CUDA_VISIBLE_DEVICES=0 \
        -v $PWD:/work -w /work -e PYTHONPATH=/work \
        -e QWEN35_9B_SUPERL8=/path/to/Qwen3.5-9B.b4.superl8 \
        superl8-serve-test:latest bench/grammar_spec_bench.py
"""
from __future__ import annotations

import os
import time

import torch

from superl8serve.engine.drafters import NgramDrafter
from superl8serve.engine.llm_engine import LLMEngine
from superl8serve.engine.sequence import SamplingParams
from superl8serve.loader import checkpoint_info, load_superl8_state_dict
from superl8serve.models.config import ModelConfig
from superl8serve.structured import GrammarCompilerCache
from superl8serve.tool_calls import forced_tool_schema

SUPERL8 = os.environ.get("QWEN35_9B_SUPERL8", os.path.join(os.environ.get("SUPERL8_WEIGHTS_DIR", "/path/to/weights"), "Qwen__Qwen3.5-9B.b4.superl8"))
TOK = os.environ.get("QWEN35_9B_TOK", os.path.join(os.environ.get("SUPERL8_WEIGHTS_DIR", "/path/to/weights"), "tok"))
MAXTOK = int(os.environ.get("BENCH_MAXTOK", "160"))
SPEC_K = int(os.environ.get("BENCH_SPEC_K", "8"))
MAXLEN = int(os.environ.get("BENCH_MAXLEN", "1024"))

# ---- workload 1: a nested JSON schema (many forced structural positions) ----------
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "price": {"type": "number"},
        "in_stock": {"type": "boolean"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "dimensions": {
            "type": "object",
            "properties": {
                "width": {"type": "integer"},
                "height": {"type": "integer"},
                "depth": {"type": "integer"},
            },
            "required": ["width", "height", "depth"],
        },
    },
    "required": ["name", "price", "in_stock", "tags", "dimensions"],
}
JSON_PROMPT = (
    "Emit a JSON object describing a product: a wooden desk that costs 189.5, is in "
    "stock, tagged office/wood/brown, and is 120x75x60."
)

# ---- workload 2: a forced tool-call (OpenAI tools -> single JSON call) -------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    "days": {"type": "integer"},
                },
                "required": ["city", "unit", "days"],
            },
        },
    }
]
TOOL_PROMPT = "What is the 3-day weather in Paris in celsius? Call the tool."


def encode(tok, text):
    ids = tok.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True
    )
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return ids[0] if ids and isinstance(ids[0], list) else ids


def configure(runner, mode, k):
    runner._drafter_mode = mode if mode != "off" else "cascade"
    runner._spec_k = k
    runner._ngram = (
        NgramDrafter(min_n=2, max_n=4, max_k=k) if mode in ("cascade", "ngram") else None
    )


def run(eng, prompt, mode, make_proc, k):
    r = eng.runner
    r._spec_enabled = mode != "off"
    if mode != "off":
        configure(r, mode, k)
    r.spec_stats = {"steps": 0, "drafts": 0, "accepts": 0}
    # A FRESH grammar processor per run — the matcher is stateful per request.
    params = SamplingParams(temperature=0.0, max_tokens=MAXTOK, logit_processors=[make_proc()])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = eng.generate([list(prompt)], params)[0]
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    st = dict(r.spec_stats)
    al = (len(out) / st["steps"]) if st["steps"] else 1.0
    acc = (st["accepts"] / st["drafts"]) if st["drafts"] else 0.0
    return out, len(out) / dt, al, acc, st


def bench_workload(eng, tok, label, prompt, make_proc):
    ref, tps_off, _, _, _ = run(eng, prompt, "off", make_proc, SPEC_K)
    print(f"\n=== {label} — prompt {len(prompt)} tok, max_new {MAXTOK}, k={SPEC_K} ===")
    text = tok.decode(ref)
    print(f"  output ({len(ref)} tok): {text[:200]!r}")
    print(f"  {'drafter':<10} {'tok/s':>8} {'speedup':>8} {'AL':>6} {'accept':>7}  identical")
    print(f"  {'off':<10} {tps_off:>8.2f} {'1.00x':>8} {'1.00':>6} {'-':>7}  -")
    for mode in ("grammar", "cascade"):
        out, tps, al, acc, st = run(eng, prompt, mode, make_proc, SPEC_K)
        ident = out == ref
        print(
            f"  {mode:<10} {tps:>8.2f} {tps / tps_off:>7.2f}x {al:>6.2f} {acc:>7.3f}  {ident}"
            f"   (steps={st['steps']} drafts={st['drafts']} acc={st['accepts']})"
        )
        if not ident:
            print("    !! NOT bit-identical — spec diverged from plain grammar decode")


def main():
    from transformers import AutoTokenizer

    print(f"loading {SUPERL8} ...")
    meta_cfg = dict(checkpoint_info(SUPERL8)["meta"]["config"])
    cfg = ModelConfig.from_hf(meta_cfg, arch="qwen3_5")
    weights = load_superl8_state_dict(SUPERL8, device="cuda")
    tok = AutoTokenizer.from_pretrained(TOK)
    # Pass EOS so a completed grammar (which forces its stop token = EOS) actually ends
    # the sequence instead of decoding to max_tokens past a terminated matcher.
    eos_id = tok.eos_token_id
    eng = LLMEngine(cfg, weights, device="cuda", max_num_seqs=1, max_len=MAXLEN,
                    enable_cuda_graph=False, eos_id=eos_id)
    print(f"eos_id={eos_id}")
    grammars = GrammarCompilerCache()

    # workload 1: JSON schema
    bench_workload(
        eng, tok, "JSON-SCHEMA", encode(tok, JSON_PROMPT),
        lambda: grammars.for_json_schema(tok, JSON_SCHEMA),
    )
    # workload 2: forced tool call
    _, tool_schema = forced_tool_schema(TOOLS, "required")
    bench_workload(
        eng, tok, "TOOL-CALL", encode(tok, TOOL_PROMPT),
        lambda: grammars.for_json_schema(tok, tool_schema),
    )


if __name__ == "__main__":
    main()
