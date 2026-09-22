# CUDA-graph-capturable spec-decode verify (issue #259 / task #100)

## Problem (measured, #266)

Base decode is CUDA-graphed (`GraphedDecode`, `engine/cuda_graph.py`) — ~1
`cudaGraphLaunch`/step. But the spec-decode **verify** forward ran 100% eager
(`EngineRunner._spec_decode_eager`), so on this dispatch-bound CMP fleet the per-step
launch overhead of the multi-token verify made every spec mode a **net decode slowdown
graphs-on**, despite high acceptance and byte-identical output:

| workload (graphs-on) | off | MTP | n-gram | cascade |
|---|---|---|---|---|
| STRUCTURED | 1.00× | 0.73× | 0.85× | 0.91× |
| PROSE | 1.00× | 0.69× | 0.40× | 0.65× |

The drafters are correct and acceptance is healthy (structured cascade AL 4.23,
accept-rate 0.80). The blocker was entirely the non-graph-capturable verify path.

## What was made graph-capturable

`GraphedVerify` (sibling of `GraphedDecode`) captures the multi-token verify forward,
reusing the decode-graph contract exactly:

- **Fixed shape.** The verify tensor is `[B, S]` with `S = k + 1`. `S` is made a
  compile-time constant per graph by padding every row's drafts to a fixed `k`
  (rejected pads are never accepted — the accept loop only walks each row's real
  drafts). Bucketed by `(batch, context, S)`, like decode buckets.
- **Static verify attention.** `GQAAttention._verify_batched` gained a static branch:
  when the runner supplies persistent device `block_tables`/`context_lens` + a
  compile-time `max_context_len` bucket, it walks the `S` verify tokens through the
  SAME static paged-decode primitives (`write_decode_static` + `decode_attn_static`)
  the graphed decode uses — no python-list `slots`/`lengths`, no `.item()` sync. Verify
  token `t` writes at its slot for position `L+1+t` and attends context length
  `(L+1)+(t+1)` — a device-tensor add by the loop-constant `t+1` (the S-loop is
  unrolled at capture). Byte-identical context lengths to the eager list path.
- **Hybrid recurrent state inside the graph.** DeltaNet / short-conv layers already run
  the verify as `S` consecutive `L=1` fused decode steps (graph-capturable, no
  `cudaFuncSetAttribute`) and record a per-token state/conv-tail trajectory. That
  trajectory now lives in fixed-address graph-pool tensors; after replay the runner
  commits, per row, the state after that row's last **accepted** token (`commit_verify`
  over the captured trajectory) — the recurrent analogue of the paged-KV accepted-token
  canonicalizer, with no snapshot/restore/replay.
- **Persistent inputs / fixed-address recurrent buffers.** `ids`/`pos`/
  `verify_slot_mapping`/`block_table`/`context_lens` and the `slot_idx` row→slot index
  are persistent device buffers refreshed via `copy_` before each replay (`bind_graph`),
  identical to the decode path.

### What stays eager (off the captured critical path)

Drafting (n-gram/MTP), the accept-longest-greedy-prefix loop (host argmax compare),
EOS/budget truncation, the MTP prefix-KV extension, and the final recurrent-state
commit-by-accept-length. All variable/host-side; none is the dispatch bottleneck.

## Adaptive verify width (the prose gate)

A **fixed** full-`spec_k` width regresses low-acceptance prose: a `spec_k+1`-wide verify
to emit ~1 token is wasted compute the graph can't hide (measured n-gram prose
0.40×→0.32× at fixed K=6). The width is therefore **bucketed up to the actual longest
draft this step** (`_draft_bucket` → a handful of reusable S-graphs, e.g. {2,3,5,7}),
matching eager's variable width while staying capturable: MTP (draft 1) and low-hit
prose n-gram collapse to `S=2`; structured cascade keeps the full width. This both
fixes the prose regression and tightens the structured win.

## Bit-identity (the hard gate)

The captured verify uses the same kernels and the same per-position context lengths as
the eager verify, so accepted tokens + committed K/V + committed recurrent state are
byte-identical. `SUPERL8SERVE_VERIFY_GRAPH=0` forces the eager path; the two agree.

- `tests/test_graph_verify.py` — graphed vs eager verify, true-tokens (dense full-attn
  and recurrent LFM2) + committed recurrent state, plus capture/determinism.
- `bench/spec_decode_bench.py` `identical` column — spec-with-graphed-verify == plain
  greedy, byte-for-byte, on the real Qwen3.5-9B DeltaNet hybrid (`identical=True`
  throughout the results below).

## Re-measured (Qwen3.5-9B b4, real V100 GPU9/11/14 — never GPU4; graphs-on, decode-only, controlled #266 harness, greedy bit-identical)

| workload | drafter | eager verify (#266) | **graphed verify** |
|---|---|---|---|
| STRUCTURED | MTP | 0.73× | **1.22×** |
| STRUCTURED | n-gram | 0.85× | **1.29×** |
| STRUCTURED | cascade | 0.91× | **1.61×** |
| PROSE | MTP | 0.69× | **1.12×** |
| PROSE | n-gram | 0.40× | **0.69×** |
| PROSE | cascade | 0.65× | **1.07×** |

**MTP + fused-DeltaNet spec-decode now net-wins graphs-on**: the deployed **cascade**
drafter goes from a net loss (0.91× structured / 0.65× prose) to **1.61× structured /
1.07× prose**, and MTP wins both workloads (1.22× / 1.12×). Honest caveat: **n-gram-ONLY
on prose stays <1× (0.69×)** — n-gram is the wrong drafter for prose (accept-rate 0.22,
AL 1.03), so it drafts and rejects regardless; graphing still lifted it from 0.40×. The
production cascade drafter (n-gram + MTP fallback) wins both. Fleet-specific numbers;
they do not transfer to a real V100.
