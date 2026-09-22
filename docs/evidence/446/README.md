# Qwen TQ3 low-memory graph replay evidence

Issue: superl8-serve#446  
Date: 2026-08-16  
Result: useful optimization, but the issue's 80% GPU-busy acceptance gate did not pass.

## Contract

- Checkpoint: `/path/to/storage
  (`b734f643665a9d1f356cfcca0ef65572eb08ab9ab25c29b1ddf4ac602548e2cb`)
- Source tensors: `/path/to/storage
- Image: `superl8-built:sm70` (`sha256:f0dfa2388e9fd6ddae37afddbb52da0d0a299d70f335a91a9ca081a4a8158292`)
- Device: the quiet `Tesla V100-PCIE-12GB`-labelled card pinned as UUID
  `GPU-71be175a-2ae4-735d-78e1-6eeeea51fc53`
  (CUDA measured physical capacity: 15.77 GiB)
- One card, one slot, context 256, 32 generated tokens, native compact TQ3.
- K8V3 and K8V8 were tested. Greedy output IDs were identical between eager and graph replay.

## Implementation and red/green checks

PyTorch permits graph captures to share a private memory pool when the graphs are
replayed in the same order and are not concurrent. The 64 Qwen layer graphs and
the final norm/logits graph meet that contract: they replay in strict layer order
with stream hand-off. Sharing the pool reduced the tiny-model capture reserve from
30 MiB across four pools to 24 MiB in one pool (20%). Persistent pinned CPU staging
also removes pageable input copies and per-step temporary tensor allocation.

The new CUDA test failed before the change with four distinct graph pools and passes
with one. The focused parity suite passes for one and two sequences, including the
pipelined-versus-sequential contract. The sibling superl8#303 test first reproduced the
K8V3 `cb.cpu()` capture failure, then proved capture/replay after keeping the codebook
on device (superl8 PR #304).

The complete `tests/test_cuda_graph.py` suite passed: 28 tests in 10.54 seconds on
the pinned card. CI then caught two stale pipeline consumers of the expanded capture
tuple; those consumers now explicitly ignore the single-device-only host staging
fields and retain their existing cross-device ordering contract. The two-stage
regressions pass locally. The three-stage test still fails on the selected card sets,
but fails identically on unmodified baseline source; CI remains the authoritative
runner-topology gate. Ruff, `git diff --check`, and Trailmark structural diff passed.

References:

- <https://docs.pytorch.org/docs/stable/generated/torch.cuda.graph.html>
- <https://docs.pytorch.org/docs/main/notes/cuda.html>
- <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html>
- <https://github.com/1CatAI/1Cat-vLLM/blob/main/RELEASE.md>
- <https://github.com/Haru-neo/qengine>

## Exact model results

Median tokens/second from quiet-card repeats:

| Cache | Eager | Shared graph pool | + pinned staging | Final gain |
|---|---:|---:|---:|---:|
| K8V8 | 7.81 | 11.13 | not rerun | +42.5% |
| K8V3 | 5.12 | 9.19 | 10.00 | +95.4% |

K8V3 final repeats were 8.35, 10.02, and 10.00 tok/s (median 10.00).
Its median prefill was 133.66 tok/s, versus 137.2 eager, a 2.6% regression.
All runs completed within the physical 15.77 GiB card; the prior distinct-pool graph
capture did not.

## Node-level Nsight result and decision

The node-traced profile recorded 120,848 GPU intervals and 4.6388 seconds of GPU
union over 6.1686 seconds wall time: **75.2% GPU active**. This misses the issue's
required 80%, despite removing most of the original host starvation. Node tracing
also adds measurable graph-launch overhead, so this is a conservative profile, but
the acceptance threshold is not relaxed after measurement.

Dominant device work in that trace was:

| Kernel family | Total |
|---|---:|
| TQ3 decode GEMM | 1.630 s |
| gated delta/GDN state update | 0.891 s |
| TQ3 RHT prepass | 0.717 s |
| TQ3 prefill GEMMs | 0.799 s |
| RMSNorm | 0.458 s |

There were 2,048 `cudaGraphLaunch` calls for 33 output IDs, approximately 62 graph
launches per token. The patch therefore lands as a nearly 2x safe improvement, but
#446 is a measured no-go against its full gate: 10.00 tok/s is below its 35 tok/s
stretch target and the superl8#270 40 tok/s umbrella. Even removing every remaining
idle interval projects only about 13 tok/s, so a C++ persistent loop cannot close
the gap by itself. The selected next architecture branch is graph-fed multi-token
verification/topology that amortizes weight passes; isolated Python orchestration
tweaks are no longer justified.

Raw Nsight reports remain host-local under `/tmp/opencode/superl8-serve-446/` because
binary profiler artifacts are not source-controlled. The node-level report SHA-256
is `37cf88a8d9dcad11cd4f15cd1e387bda482f069321b222942fac5b60e7914d66`.

## Adversarial review disposition

AMD0 correctly prompted removal of the remaining recurrent `slot_idx` temporary
tensor allocation; the final patch fills its persistent pinned staging tensor in
place. Its other findings were rejected against source and the installed PyTorch
2.10 API: capture buffers are sized by the `(batch_bucket, context_bucket)` cache
key, graph objects retain their pool, and PyTorch exposes no `graph_pool_destroy`
API. The documented pool-sharing contract requires identical replay order and no
concurrent replay, both enforced by `_replay_layer_graphs` and covered by the full
CUDA graph suite.
