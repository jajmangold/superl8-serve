# superl8-serve #351 — batched recurrent prefill qualification evidence (2026-08-11)

## A/B setup

- Base server (GPU6): `superl8-serve-346-lfm25-native-gguf` source (serial recurrent
  prefill), loopback `:18101`, container `content-factory-lfm25-351-base-gpu6`.
- Candidate server (GPU5): this worktree (`superl8-serve-351-lfm25-batched-prefill`),
  loopback `:18102`, container `content-factory-lfm25-351-cand-gpu5`.
- Image `superl8-serve-ggufspec:latest` (no rebuild), model
  `LiquidAI_LFM2.5-2.6B-Q6_K.gguf` SHA-256
  `499c120820935273c5eec587333ce18e0d4369911bd795b537f041ab4b20052f`, tokenizer at
  `runtime/hot-models/lfm25-2.6b-superl8/tokenizer`, `max_num_seqs=128`, `max_len=1024`,
  whole-step graph (`SUPERL8SERVE_LAYER_GRAPH=0`).
- Loadgen: stdlib streaming loadgen (14-token prompt, greedy temp 0, 256 outputs,
  client-observed TTFT + host wall agg tok/s + `GET /metrics` avg_ttft_ms).
  JSON evidence files: `base-b*.json` / `cand-b*.json`.

## Headline: batch-128 TTFT

| batch | base TTFT (s) | candidate TTFT (s) | base agg tok/s | candidate agg tok/s |
|---:|---:|---:|---:|---:|
| 1 | 0.141 | 0.107 | 115.7 | 117.4 |
| 16 | 1.40 | 0.68 | 444.8 | 442.0 |
| 64 | 5.49 | 1.12 | 513.6 | 590.9 |
| 128 | **11.16** (max 11.28) | **1.79** (max 1.85) | 612.1 | 734.4 |

Batch-128 TTFT reduced **6.2×** (11.16 s → 1.79 s), exceeding the 3× acceptance bar.
HBM peak ~9.6 GB unchanged; 0 HTTP errors across all runs.

## Correctness blocker → superl8#241

Candidate varlen prefill diverged (degenerate "landlords…" loops) at batch **31/63/95**
with identical prompts; **16/64/128** stayed bit-identical. Root cause is in the superl8
library, not this worktree:

- `superl8.ops.attn_int8_varlen` subtracts ONE GLOBAL K-mean over all packed sequences
  (`ops.py:282`); serial `attn_int8_fwd` smooths K per sequence. Not softmax-invariant
  under per-row int8 quantization for non-power batches.
- Real-model per-layer trace: B63 first divergence at **attention layer 13**; all conv
  layers + attention 2/5/9 bit-identical; B64 all 32 layers bit-identical.
- A per-sequence K-mean restores **100% token-identical** varlen-vs-serial at
  B31/63/95/64 (0 diverging layers).

Tracked as **superl8#241**, fixed in **superl8 PR #242** (Python-only prologue change,
`_varlen_smooth_k`; 28/28 focused tests; equal-length perf ratio 1.040 ≤5%; ragged
run-vectorized 1.355 ms).

## Final qualification after superl8 #242 merged

The final gate used the same cached image and exact GPU5 UUID, with only the merged
#242 per-sequence K-mean prologue applied to the image-matched `ops.py` (overlay
SHA-256 `d07ccbe8a80c95eb6cec9585b99d18a9d1298b37d46b79b63b3f8e69f1b552ee`).
This avoids both a CUDA rebuild and mixing the cached extension with unrelated newer
Python wrappers. The overlay passed 11 focused CUDA varlen-vs-serial tests before the
model server started.

- Ragged real-model serial-vs-varlen output: zero mismatches at B1/16/64/128.
- B128 TTFT: 11.16 s serial baseline -> 1.603 s average / 1.628 s max (**6.96x**).
- B128, 14-token prompts, 256 outputs: 32,768 tokens in 32.099 s = **1,020.9 tok/s**,
  zero errors, 9,749 MiB peak HBM. This preserves the #350 1,021-1,125 tok/s
  whole-step decode profile while removing nearly all prefill amortization loss.
- Mixed-load fairness: a 128-token stream completed under four waves of 16 one-token
  requests; p95 ITL stayed 8 ms, one bounded 1.188 s prefill interruption occurred,
  and all 64 short requests completed (no starvation or errors).
- Disconnect cancellation was re-run and exposed a pre-existing API lifecycle gap:
  after the client closed following token one, generation continued for 6.39 s and
  grew from 1 to 46 KV blocks. This is isolated from the prefill change and is tracked
  as P0 `superl8-serve#363`; production readiness remains gated on that fix.

Compact receipts: `final-parity.json`, `final-ttft.json`, `final-b128-256.json`,
`final-fairness.json`, and `final-cancellation.json`.

## Status

#351 implementation (EngineRunner varlen dispatch gated on `varlen_prefill_safe`,
segmented ShortConv conv, per-slot clear/bind) is complete and benchmarked at 6.96×
TTFT. The former correctness blocker is resolved: superl8 PR #242 merged as
`ddee547372606cb815eca91c64f1a41bc633f7ab`. Final qualification overlaid that
merged superl8 Python prologue on the existing cached sm_70 runtime. The exact ragged
B1/16/64/128 parity, TTFT, end-to-end throughput, HBM, cancellation, fairness, and
decode-regression gates are now complete. Evidence probe files
(`cand-b32/64/96-probe`, `base-b*-probe*`, `cand-nograph-*`) remain historical
localization evidence rather than final qualification results. The prefill change is
qualified; the distinct production cancellation blocker is `superl8-serve#363`.

GPUs 9 and 14 (live services) were never touched; all temp servers used GPU5/6.
