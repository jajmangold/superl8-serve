# superl8-serve #371 — LFM2 native-cache fused ShortConv decode

## Scope

Wire `superl8.causal_conv1d_decode` into LFM2/LFM2.5 `ShortConv` for single-token
CUDA decode without changing the resident cache layout. The kernel consumes and
returns `[B,D,K-1]`, so the cache crosses the backend boundary without a transpose
or copy. Prefill, CPU, `K>8`, verify trajectories, disabled fusion, and older superl8
builds retain the eager path.

## Setup (2026-08-11)

- Checkpoint: `LiquidAI_LFM2.5-2.6B-Q6_K.gguf`
  - sha256 `499c120820935273c5eec587333ce18e0d4369911bd795b537f041ab4b20052f`
- superl8: PR #252 / commit `e858b5f` over `ddee547`
- superl8-serve: issue #371 branch over `d7dc977` (merged large graph buckets)
- Device: GPU11, BIOS-modded CMP reporting as a 16 GiB V100/sm_70
- Whole-step CUDA graphs, buckets through B512, `max_len=16`
- Exact offline admission: all 512 one-token prompts admitted before stepping
- Timed region: 12 decode steps / 6,144 generated tokens, EOS ignored
- GPU9 production and QA GPUs 0/1/14 untouched

## Results

The Q6_K file was deliberately transcoded once at load to resident per-row W8,
matching the previous qualification runtime:

- trial 1 (cold clocks/caches): **2,217.742 tok/s**
- trial 2: **2,935.038 tok/s**
- trial 3: **2,921.604 tok/s**
- three-trial median: **2,921.604 tok/s**

Previous exact-B512 median on the same W8 runtime was **2,362.894 tok/s**. The
new median is a **23.6% aggregate improvement**. Warm trials correspond to about
174.4–175.2 ms per step.

As a control, leaving the file on the native Q6_K kernel path measured
`1,259.587 / 1,693.743 / 1,695.986 tok/s`; steady native-Q6 performance is well
below the W8 transcode lane and is not the production batch-throughput choice.

## Validation

- CPU/mock affected suites: **38 passed, 17 skipped** (CUDA-gated)
- GPU11 new fused-path selection: **9 passed**
- The sibling kernel suite: **12 passed, 1 missing-baseline skip**, covering B1,
  B512, exact tail roll, K=1, fp16, bf16, and fp32
- Ruff on changed Python/test files: clean
- Trailmark structural diff: four added call edges, none removed

## AMD0 bounded review — overseer dispositions

- Claimed missing multi-token guard: false; the production condition explicitly
  requires `L == 1`, and fallback tests cover `L>1`.
- Claimed mock-oracle mismatch: false; it implements the kernel's documented fp32
  dot and exact tail roll, while sibling CUDA tests and the real GPU integration
  independently validate kernel numerics.
- Claimed A/B flag toggle invalidity: false; the comparison intentionally uses the
  same stateless block, weights, input, and empty tail so only dispatch changes.

## Remaining gap

At B512, 5,000 tok/s requires at most 102.4 ms/step; the warm result remains about
175 ms/step. The prior exact Nsys trace assigns 136.807 ms/step to 121 W8 GEMMs.
That separately scoped residual is tracked by superl8#253; attention and scheduling
are not the dominant exact-batch bottlenecks.
