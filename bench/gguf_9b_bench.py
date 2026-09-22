# SPDX-License-Identifier: BSD-3-Clause
"""GGUF-native Qwen3.5-9B decode benchmark — measures tok/s with CUDA graphs.

Bare-docker run (NEVER docker compose, NEVER GPU 4):
    docker run --rm --gpus '"device=11"' --entrypoint bash -e CUDA_VISIBLE_DEVICES=0 \
        -v $PWD:/work -v /path/to/fused_ni8:/fused_ni8 \
        -v /path/to/flint8_work:/flint8_work \
        -w /work -e PYTHONPATH=/work:/fused_ni8 \
        superl8-serve-test:latest bench/gguf_9b_bench.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

# ── Step 0: install gguf if missing ─────────────────────────────────────────
try:
    import gguf  # noqa: F401
except ImportError:
    print("[setup] installing gguf-py ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gguf", "-q"])
    print("[setup] gguf installed.")

import torch

from bench.metrics import env_flag, summarize_generation_steps

print(
    f"torch {torch.__version__}, CUDA {torch.cuda.is_available()}, "
    f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}"
)

# ── Step 1: verify superl8.linear_q4k imports ──────────────────────────────────
import superl8  # noqa: E402

assert hasattr(superl8, "linear_q4k"), "superl8.linear_q4k NOT available — fused kernel missing"
print("[check] superl8.linear_q4k available (fused Q4_K dp4a kernel)")

from superl8serve.layers.linear import _SUPERL8_HAS_Q4K  # noqa: E402

print(f"[check] _SUPERL8_HAS_Q4K = {_SUPERL8_HAS_Q4K}")

# ── Step 2: load GGUF native ────────────────────────────────────────────────
GGUF_PATH = os.environ.get("SUPERL8_GGUF_PATH", "/flint8_work/Qwen3.5-9B-UD-Q4_K_XL.gguf")
MODEL_LABEL = os.environ.get("SUPERL8_MODEL_LABEL", os.path.basename(GGUF_PATH))
TARGET_TOK_S = float(os.environ.get("SUPERL8_TARGET_TOK_S", "69"))
MAX_LEN = int(os.environ.get("SUPERL8_MAX_LEN", "2048"))
assert os.path.exists(GGUF_PATH), f"GGUF not found: {GGUF_PATH}"

from superl8serve.gguf_native import (  # noqa: E402
    _native_kquant_types,
    gguf_config,
    load_gguf_engine,
)

t0 = time.perf_counter()
cfg = gguf_config(GGUF_PATH)
t_cfg = time.perf_counter() - t0
print(f"[load] gguf_config: {t_cfg:.2f}s")
print(f"[load] arch={cfg.arch}, layers={cfg.num_hidden_layers}, hidden={cfg.hidden_size}")
print(
    f"[load] heads={cfg.num_attention_heads}, kv_heads={cfg.num_key_value_heads}, "
    f"head_dim={cfg.resolved_head_dim()}"
)
print(
    f"[load] linear_attention={cfg.linear_attention}, "
    f"full_attn_interval={cfg.full_attention_interval}"
)
print(f"[load] num_experts={cfg.num_experts}, vocab={cfg.vocab_size}")

# Check CUDA graph support
print(f"[load] is_moe={cfg.is_moe()}, latent_attention={cfg.latent_attention}")
for i in range(min(5, cfg.num_hidden_layers)):
    print(f"[load] layer {i}: {cfg.attention_kind(i)}")

PYTORCH_ALLOC_CONF = "expandable_segments:True"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", PYTORCH_ALLOC_CONF)

t0 = time.perf_counter()
SPEC_DECODE = env_flag("SUPERL8_SPEC_DECODE")
engine = load_gguf_engine(
    GGUF_PATH,
    device="cuda",
    max_num_seqs=1,
    max_len=MAX_LEN,
    spec_decode=SPEC_DECODE,
)
t_load = time.perf_counter() - t0
print(f"[load] engine built in {t_load:.2f}s")

# Report native vs fallback path
native_types = _native_kquant_types()
print(f"[load] native_types={','.join(native_types) or '(none — int8 fallback)'}")
print(f"[load] spec_decode={SPEC_DECODE}, max_len={MAX_LEN}")

# CUDA graph status
runner = engine.runner
gd = getattr(runner, "graphed", None)
if gd is not None:
    print(f"[graph] GraphedDecode supported={gd.supported}, reason={gd._unsupported_reason}")
else:
    print("[graph] no GraphedDecode on runner")

# ── Step 3: benchmark decode ────────────────────────────────────────────────
PROMPT_LEN = int(os.environ.get("SUPERL8_PROMPT_LEN", "256"))
NEW_TOKENS = int(os.environ.get("SUPERL8_NEW_TOKENS", "128"))
TOKENIZER_PATH = os.environ.get("SUPERL8_TOKENIZER")
PROMPT_TEXT = os.environ.get("SUPERL8_PROMPT_TEXT", "The capital of France is")

# The default repeated-token prompt is intentionally a timing-only workload. A supplied
# tokenizer switches the run into a natural-prompt correctness check and decodes the
# output text; token diversity alone is not evidence of language coherence.
tokenizer = None
if TOKENIZER_PATH:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    prompt_ids = tokenizer.encode(PROMPT_TEXT, add_special_tokens=False)
    PROMPT_LEN = len(prompt_ids)
else:
    prompt_ids = [1] * PROMPT_LEN

from superl8serve.engine.sequence import SamplingParams  # noqa: E402

print(f"\n[bench] prompt_len={PROMPT_LEN}, new_tokens={NEW_TOKENS}")

# Warmup: run a short generation to warm up CUDA graphs / allocations
print("[bench] warming up ...")
warmup_ids = prompt_ids[:32]
wid = engine.add_request(warmup_ids, SamplingParams(max_tokens=8, temperature=0.0))
while engine.scheduler.has_work():
    engine.step()
engine.forget(wid)
torch.cuda.synchronize()
print("[bench] warmup done")

# Actual benchmark — time each scheduler step so prefill and lazy graph capture
# cannot be mislabeled as steady decode.
torch.cuda.reset_peak_memory_stats()
request_id = engine.add_request(
    prompt_ids, SamplingParams(max_tokens=NEW_TOKENS, temperature=0.0, ignore_eos=True)
)
step_times = []
emitted_tokens_per_step = []
while engine.scheduler.has_work():
    before = len(engine._out[request_id].output_ids)
    t0 = time.perf_counter()
    engine.step()
    torch.cuda.synchronize()
    step_times.append(time.perf_counter() - t0)
    emitted_tokens_per_step.append(len(engine._out[request_id].output_ids) - before)
output_ids = list(engine._out[request_id].output_ids)
summary = summarize_generation_steps(
    step_times,
    warmup_decode_steps=1,
    emitted_tokens_per_step=emitted_tokens_per_step if SPEC_DECODE else None,
)
peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
spec_stats = dict(runner.spec_stats)
spec_active = SPEC_DECODE and spec_stats["steps"] > 0

# ── Step 4: report ──────────────────────────────────────────────────────────
print(f"\n[debug] output_ids ({len(output_ids)} tokens): {output_ids[:30]}")
if tokenizer is not None:
    print(f"[debug] prompt: {PROMPT_TEXT!r}")
    print(f"[debug] decoded output: {tokenizer.decode(output_ids)!r}")

print(f"\n{'=' * 60}")
print(f"RESULTS: {MODEL_LABEL} GGUF-native decode")
print(f"{'=' * 60}")
print(f"  Steady decode tok/s:           {summary['steady_decode_tok_s']:.1f}")
print(f"  End-to-end tok/s:              {summary['end_to_end_tok_s']:.1f}")
print(f"  Prefill:                       {summary['prefill_s']:.3f}s")
print(f"  Lazy graph capture:            {summary['graph_capture_s']:.3f}s")
print(f"  Target:                        {TARGET_TOK_S:g}")
print(
    f"  steady vs target:              {summary['steady_decode_tok_s'] / TARGET_TOK_S * 100:.1f}%"
)
print(f"  Output tokens:                 {len(output_ids)}")
print(f"  Total time:                    {summary['total_s']:.2f}s")
print(f"  Peak VRAM:                     {peak_vram:.2f} GB")
print(f"  VRAM < 16GB:                   {'YES' if peak_vram < 16.0 else 'NO — OVER BUDGET'}")
print(f"  Native k-quant types:           {','.join(native_types) or 'none'}")
print(f"  CUDA graphs:                   {'ON' if gd and gd.supported else 'OFF'}")
print(
    "  Speculative decode:            "
    f"{'ACTIVE' if spec_active else 'REQUESTED, SAFE FALLBACK' if SPEC_DECODE else 'OFF'}"
)
if SPEC_DECODE:
    print(
        "  Spec steps/drafts/accepts:      "
        f"{spec_stats['steps']}/{spec_stats['drafts']}/{spec_stats['accepts']}"
    )

# Degeneracy check only. Language coherence requires a tokenizer + natural prompt and
# is reported as decoded text above for inspection/oracle comparison.
if output_ids:
    unique = len(set(output_ids))
    print(f"  Unique tokens:                 {unique}/{len(output_ids)}")
    nondegenerate = unique > len(output_ids) * 0.1
    print(f"  Non-degenerate ids:            {'YES' if nondegenerate else 'NO'}")
    print(f"  Language coherence checked:    {'YES (decoded above)' if tokenizer else 'NO'}")
else:
    print("  Non-degenerate ids:            NO — empty output!")
    print("  Language coherence checked:    NO")

# Verbatim output if short enough
if len(output_ids) <= 200:
    print(f"  Raw output ids: {output_ids}")
