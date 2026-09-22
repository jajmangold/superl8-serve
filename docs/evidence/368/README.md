# superl8-serve #368 — configurable B256/B512 CUDA-graph batch buckets (short context)

## Scope

Qualifies the new CLI/config seam (`--cuda-graph-batch-buckets`) that exposes bounded
decode CUDA-graph batch buckets up to 512 without changing the built-in default
(1,2,4,...,128). Whole-step graphs were swept at B128/B256/B512 over the HTTP server
and in offline exact-batch decode on a short-context workload. No claim is made beyond
short-context qualification.

## Setup (2026-08-11)

- Checkpoint: `LiquidAI_LFM2.5-2.6B-Q6_K.gguf`
  sha256 `499c120820935273c5eec587333ce18e0d4369911bd795b537f041ab4b20052f`
- superl8 `ddee547` — swizzled W8 load-time transcode
- superl8-serve `ffc0788` (base) + this branch (`perf/368-lfm25-large-graph-buckets`)
- Whole-step CUDA graphs; `max_len=16`; one-token prompts; 12 concurrent outputs
- GPU4: BIOS-modded CMP reporting as V100. GPU9 untouched (production server
  unaffected, per the issue).

## Results

- HTTP (short arrivals fail to sustain full batches — the scheduler never fills the
  widest bucket):
  - B128 median **1136.663 tok/s**
  - B256 median **1148.839 tok/s**
  - B512 median **1156.599 tok/s**
- Offline exact-B512 (6144 tokens / 12 steps, zero errors), trials
  `2371.468 / 2362.894 / 2361.576`, median **2362.894 tok/s**.

Compact receipts: `batch-sweep.json`.

## Interpretation

Wider graph buckets raise HTTP throughput only marginally (~2% B128→B512) and stay
well short of the 5k tok/s target — this disproves reaching 5k from graph buckets
alone. The HTTP path is bound by short-arrival batch coalescing, which is a separate
follow-up. Offline exact-B512 (median 2362.894 tok/s) confirms the graph itself
scales, so graph dispatch is not the primary remaining serve-path bottleneck.

## Validation

- Focused suite (`tests/test_cuda_graph_buckets.py`): the initial exhaustive form
  passed 46 tests; after removing redundant cases, the final compact suite passed
  **34/34**.
- The pre-compaction combined CUDA-graph run passed 72 tests with one reproducible
  flaky recapture test; `origin/main` passed the same test once, so this branch
  neither caused nor resolved it.

## AMD0 bounded review — findings and overseer disposition

- Duplication finding (CLI validation vs graph bucket filtering): **false** — the CLI
  rejects `> max_num_seqs` at the argparse boundary while the graph objects filter
  against cache slot count; distinct boundaries, both required.
- GGUF transpose and MTP findings: refer to unchanged code, rejected as unrelated.
- Integration coverage is layered (parse → propagation → graph filtering) rather than
  a full real-model constructor; accepted as an explicit scope note, not a defect.
- Empty programmatic tuple preserves existing semantics (graphs unsupported → eager
  fallback) and the banner reports the effective list; no behavior change.
