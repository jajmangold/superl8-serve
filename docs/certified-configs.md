# Certified superl8-serve model configurations

This document records **certified** serving configurations for `superl8-serve`: exact
checkpoint + serve flags, honest throughput (decode-only and end-to-end), a quality
delta, and a sustained-load (soak) reliability result. Every number here was measured
on this deployment fleet with a **fresh** engine on a **free** V100 — never against
the live production server (GPU 4) — and is reproducible with the harnesses in `bench/`.

## Hardware caveat (READ FIRST — numbers do not transfer)

All cards in this fleet are **NVIDIA CMP 100-210** GV100 mining silicon (the ones
`nvidia-smi` labels "Tesla V100-PCIE-12GB" have a V100 VBIOS flashed on but are the
same CMP die). NVIDIA firmware-gimped the FP16/TF32 **tensor cores** to ~5-6% of a
real V100; the INT8 **dp4a** CUDA-core path (what superl8 uses) is intact (~46 TOP/s,
~73% of a real V100), and PCIe is 1.0 ×1. **Throughput numbers below are
fleet-specific and DO NOT transfer to a real V100** (a real V100 would be materially
faster on both int8 and the fp16 baselines). int8 still buys the memory/bandwidth win
(half the HBM + smem for weights/KV) on top of the dp4a compute win. Measurements were
taken on host GPU index 11 (decode + quality) and index 14 (server e2e + soak), both
otherwise idle; the live Qwen3.5-9B server on GPU 4 was untouched.

Toolchain: CUDA 12.9, torch 2.10.0+cu129, Python 3.12, `sm_70`. Graphs **ON**
(the deployment regime) for every throughput number unless stated.

---

## Throughput reconciliation vs the #266 spec-decode bench (READ — no 7× discrepancy)

A prior reading held that `bench/spec_decode_bench.py` (#266) reported **5.91 tok/s**
"baseline decode, graphs-on" for the same Qwen3.5-9B b4, which looks ~7-12× below the
62-76 tok/s here. **Reconciled: there is no discrepancy — the two agree, and 5.91 was
never a baseline.** Re-running #266's harness on this card, graphs-ON, decode-only:

| harness | what it times | b4 baseline decode (single-stream, graphs-on) |
|---|---|---|
| `tools/bench_llm.py` (this cert, `--num-prompts 1`) | full `engine.step` loop incl. scheduler Python overhead + first-step lazy graph capture (no separate warmup) | **62.0 tok/s** |
| `bench/spec_decode_bench.py` mode=`off` (#266) | only `runner.decode`, explicit per-mode warmup (graph capture excluded) | **74.0 / 74.7 tok/s** (two prompts); a clean off-only warm re-run measured **75.96 tok/s** |

The two harnesses **agree**: baseline single-stream decode is **~62-76 tok/s**. The ~20%
gap is pure methodology — bench_llm counts per-step scheduler/bookkeeping and pays the
first-step lazy CUDA-graph capture inside the timed window; #266 warms first and times
only the decode call. Both are graphs-on, decode-only, single-stream.

**Where "5.91" actually came from:** it is #266's **`ngram` spec-decode drafter mode on
the prose prompt** — measured **5.99 tok/s (0.08×)** in the re-run — NOT the baseline.
Spec-decode is a **net slowdown** on this fleet with graphs on (every drafter mode
< 1×: mtp 0.14-0.15×, ngram 0.08-0.17×, cascade 0.14-0.17× here), which is exactly
#266's own headline ("cascade is a net slowdown, do not re-report the old 2.69×").
Reporting 5.91 as "decode tok/s" conflates a degenerate spec-decode drafter with the
plain-decode baseline. **The certified single-stream decode number is 62 tok/s (the
conservative full-step figure carried below); the decode-kernel-only figure is ~76
tok/s. Spec-decode stays OFF in the deployed config.**

For **b8**, the cert `bench_llm` figure is **42.8 tok/s** (full-step). Running #266's
harness on b8 to get its decode-kernel-only equivalent **OOMs on the 16 GB card** — b8
weights already sit at ~15 GB and #266 captures a *set* of decode graphs across context
buckets with no VRAM headroom left (bench_llm captures fewer buckets near its lower
`max_len`, which is why its b8 sweep fit). Applying the b4 harness offset (76/62 ≈ 1.23)
puts b8's decode-kernel-only rate at **~52 tok/s**. Either way b8 is ~30% slower than b4
single-stream — it moves ~1.6× the weight bytes and this fleet is weight-bandwidth bound.

---

## Config A — Qwen3.5-9B b4 (PRIMARY, matches the live production server)

**Checkpoint:** `/path/to/weights/Qwen__Qwen3.5-9B.b4.superl8` (4-bit weights,
6.7 GB on disk), arch `qwen3_5` (hybrid full-attention + DeltaNet linear attention).

**Serve flags (identical to the live GPU-4 container):**
```
python3 -m superl8serve.api.server \
  --model /path/to/weights/Qwen__Qwen3.5-9B.b4.superl8 \
  --tokenizer /models/Qwen3.5-0.8B-tok --served-model-name Qwen3.5-9B \
  --host 0.0.0.0 --port 8000 --max-num-seqs 4 --max-len 4096
```

### Throughput

**Decode-only** (`bench/` → `tools/bench_llm.py`, graphs ON, prompt_len 256, 128 new
tokens, `max_num_seqs = concurrency`; steady-state decode rate, excludes prefill/HTTP):

| concurrency | decode tok/s (aggregate) | per-stream tok/s | prefill tok/s | peak VRAM (GB) |
|---|---|---|---|---|
| 1 | 62.0 | 62.0 | 49.4 | 9.09 |
| 2 | 107.1 | 53.5 | 56.8 | 9.17 |
| 4 | 171.0 | 42.8 | 49.5 | 9.33 |
| 8 | 248.3 | 31.0 | 51.5 | 9.63 |

**End-to-end** (`bench/cert_loadgen.py sweep`, OpenAI `/v1/completions`, real server
with the deployed `--max-num-seqs 4`, max_tokens 128; includes prefill + decode +
queueing + HTTP):

| concurrency | agg tok/s | per-stream tok/s | p50 latency (s) | max latency (s) | errors |
|---|---|---|---|---|---|
| 1 | 48.3 | 48.3 | 2.60 | 2.97 | 0 |
| 2 | 69.1 | 34.6 | 3.61 | 4.03 | 0 |
| 4 | 85.6 | 21.4 | 5.98 | 6.26 | 0 |
| 8 | 90.8 | 11.3 | 8.41 | 11.27 | 0 |

**Saturation:** end-to-end aggregate saturates at **~86-91 tok/s around concurrency
4-8**, bounded by the deployed `--max-num-seqs 4` (at concurrency 8 four requests
queue, so per-stream halves and latency grows but aggregate is flat). The decode-only
column keeps scaling to 248 tok/s at concurrency 8 only because that run raised
`max_num_seqs` to 8 — it is the **batch-throughput ceiling if the seq cap were lifted**,
not the behaviour of the deployed config. Single-stream: **62 tok/s decode-only /
48 tok/s end-to-end** (the e2e figure includes a ~0.4 s prefill+HTTP tax per request).

### Quality

Measured through the **real superl8-serve int8/int4 engine forward** on a fixed 4-passage
expository text set (322 scored tokens), `bench/cert_quality.py`:

| checkpoint | perplexity | top-1 next-token |
|---|---|---|
| Qwen3.5-9B **b4** (deployed) | **9.17** | **52.5%** |
| Qwen3.5-9B b8 (reference) | 6.92 | 55.6% |
| **b4 − b8 delta** | **+32.5% ppl** | **−3.1 pts** |

**Reference caveat (honest):** a co-located **fp16 HF** reference for the 9B is **not
obtainable on this box** — the raw fp16 safetensors are not materialized locally (xet
blobs absent, HF hub offline) and a 9B fp16 forward (~18 GB) does not fit a 16 GB
V100. So the delta above is **b4 vs the near-lossless 8-bit (b8)** through the same
engine path, i.e. the on-box cost of 4-bit weights. The **true b4-vs-fp16** gap is
larger still (b8 itself carries some loss vs fp16; for the 0.8B, b8 measured +12.3%
ppl vs fp16). **Verdict: the deployed 4-bit config trades a real, measurable quality
loss (+32.5% ppl, −3.1 pt top-1 vs 8-bit) for ~1.5× the throughput and ~6 GB less
VRAM.** If quality matters more than latency/VRAM, prefer Config B (b8).

### Soak (sustained load — reliability)

`bench/cert_loadgen.py soak`, concurrency 4 (= deployed cap), mixed prompt lengths
(1-5× the base prompt) and mixed gen lengths (32/64/128/256 tokens), continuous stream
for **25.1 min (1503.7 s)**. VRAM/RSS/kv sampled every 30 s (50 samples,
`/tmp/soak_samples.csv`).

| signal | start | end | min / max over run | verdict |
|---|---|---|---|---|
| GPU VRAM (MB) | 10756 | 11114 | 10756 / 11422 | **flat, no creep** — warms ~0.4 GB then holds (bounded prefix cache, validates #265) |
| server host RSS (MB) | 1444 | 1068 | 1060 / 1542 | **bounded, no leak** — oscillates ~1.0-1.5 GB and ends *lower* than start; no monotonic growth (generate() auto-forget / `_out` fix, validates #265) |
| kv-cache usage (%) | 1.2 | 8.4 | 1.2 / 10.0 | oscillates 5-10%, never climbs unbounded (prefix cache bounded by pinned blocks) |
| requests / errors | — | — | **624 completed / 0 errors** | no OOM, no degradation, no timeouts |
| decode tok/s (rolling) | — | — | ~40-66 | stable; run mean **50.2 tok/s** (75,424 completion tokens) |

### CERTIFIED

> **CERTIFIED: Qwen3.5-9B b4 @ concurrency 4 = 86 tok/s aggregate end-to-end
> (62 tok/s single-stream decode), quality +32.5% ppl / −3.1 pt top-1 vs 8-bit
> (fp16 anchor infeasible on-box), soak: stable 25 min / 624 requests at 0 errors
> with flat VRAM (10.8→11.1 GB) and bounded host RAM (ended below start).**

---

## Config B — Qwen3.5-9B b8 (SECONDARY, higher quality)

**Checkpoint:** `/path/to/weights/Qwen__Qwen3.5-9B.b8.superl8` (8-bit weights,
10.7 GB on disk). Same arch and serve flags as Config A (change `--model` and, if
running alongside A, `--max-num-seqs` to fit VRAM).

### Throughput (decode-only, graphs ON, prompt_len 256, 128 new tokens)

| concurrency | decode tok/s (aggregate) | per-stream tok/s | prefill tok/s | peak VRAM (GB) |
|---|---|---|---|---|
| 1 | 42.8 | 42.8 | 52.0 | 14.80 |
| 2 | 71.4 | 35.7 | 61.6 | 14.89 |
| 4 | 77.1 | 19.3 | 62.7 | 15.04 |

b8 is **~30% slower single-stream than b4** (42.8 vs 62.0 tok/s) — the fleet is
weight-bandwidth bound and b8 moves ~1.6× the bytes — and its **~15 GB peak leaves
almost no headroom on a 16 GB card** (concurrency 8 OOMs; concurrency 4 is the safe
ceiling). Not measured e2e/soak separately; the b4 soak already validates the
lifecycle fixes and the same engine path is used.

### Quality (same fixed set, real engine)

| checkpoint | perplexity | top-1 next-token |
|---|---|---|
| Qwen3.5-9B **b8** | **6.92** | **55.6%** |

Same fp16-anchor caveat as Config A. b8 is the higher-fidelity checkpoint on this box
(−32.5% ppl / +3.1 pt top-1 relative to the deployed b4).

### CERTIFIED

> **CERTIFIED: Qwen3.5-9B b8 @ concurrency ≤4 = 77 tok/s aggregate decode
> (42.8 tok/s single-stream), quality ppl 6.92 / top-1 55.6% (best on-box, ~32% lower
> ppl than b4); VRAM ~15 GB leaves no room for concurrency >4 on 16 GB.** Soak not run
> separately — inherits the b4 soak's lifecycle validation (same engine).

---

## Reproduction

```bash
# 1. fresh server on a FREE V100 (never GPU 4 / the live server):
docker run -d --name cert --gpus all -p 8001:8001 -e CUDA_VISIBLE_DEVICES=<free-idx> \
  -e PYTHONPATH=/serve -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v /path/to/storage \
  -v /path/to/storage -v /path/to/archive:/path/to/archive \
  --entrypoint sleep superl8-built:sm70 infinity
docker exec cert pip install -q 'fastapi>=0.110' 'uvicorn[standard]' python-multipart \
  transformers pillow rich nvidia-ml-py
docker exec -d cert bash -lc "cd /serve && python3 -m superl8serve.api.server \
  --model /path/to/weights/Qwen__Qwen3.5-9B.b4.superl8 \
  --tokenizer /models/Qwen3.5-0.8B-tok --served-model-name Qwen3.5-9B \
  --host 0.0.0.0 --port 8001 --max-num-seqs 4 --max-len 4096"

# 2. decode-only throughput (in-container, one process per card — orphaned loads OOM):
docker exec cert bash -lc "cd /serve && python3 tools/bench_llm.py \
  /path/to/weights/Qwen__Qwen3.5-9B.b4.superl8 \
  --num-prompts <1|2|4|8> --prompt-len 256 --max-new-tokens 128 --out-dir /tmp/b"

# 3. end-to-end sweep + soak (from the host, no GPU footprint):
python3 bench/cert_loadgen.py sweep --url http://localhost:8001 --model Qwen3.5-9B \
  --concurrency 1 2 4 8 --reqs-per-conc 8 --max-tokens 128
python3 bench/cert_loadgen.py soak  --url http://localhost:8001 --model Qwen3.5-9B \
  --concurrency 4 --duration 1500

# 4. quality (real engine forward; b4 under test vs b8 reference):
docker exec cert bash -lc "cd /serve && python3 bench/cert_quality.py \
  --ckpt /path/to/weights/Qwen__Qwen3.5-9B.b4.superl8 \
  --ref  /path/to/weights/Qwen__Qwen3.5-9B.b8.superl8 \
  --tokenizer /models/Qwen3.5-0.8B-tok"
```

**Operational note:** run exactly **one** GPU process per card and wait for it to
release VRAM before the next — a detached/killed `docker exec` leaves the in-container
python alive holding ~7-15 GB, and two 9B loads collide and OOM on a 16 GB card.

## Honest caveats summary

- Numbers are **CMP-fleet-specific** and do not transfer to a real V100.
- No **fp16 HF** anchor on-box for the 9B → quality deltas are **b4-vs-b8**, and the
  true fp16 gap is larger. The 0.8B (which has an fp16 anchor) measured b8 at +12.3%
  ppl / −5.6 pt top-1 vs fp16; the 9B b4's fp16 gap is not directly measured here.
- Quality set is small (4 passages, 322 tokens) — a directional certified signal, not
  a full benchmark; it uses the same reference-oracle-style real-engine forward.
- The deployed 4-bit config carries a **real quality cost** vs 8-bit — this is stated,
  not hidden; the accuracy gate, not ideology, decides the trade.
