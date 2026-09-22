<!-- SPDX-License-Identifier: MIT -->
# Qwen3.6-27B Q3_K speculative-decode guard — 2026-07-16

Fleet-specific measurement on host GPU 11, reported by `nvidia-smi` as a Tesla
V100-PCIE-12GB but physically a CMP 100-210 (GV100). CUDA 12.9.1, PyTorch
2.10.0+cu129, batch 1, CUDA graphs enabled, native GGUF Q3_K path. GPU 8 is the
pinned live-server card and was not touched.

Checkpoint:
`/path/to/models/Qwen3.6-27B-Q3_K_S.gguf`

## Reproduction

`bench/gguf_9b_bench.py` with `SUPERL8_SPEC_DECODE=1`, `SUPERL8_MAX_LEN=512`,
`SUPERL8_NEW_TOKENS=32`, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Results

| Workload | Implementation | Steady tok/s | Peak GiB | Spec steps/drafts/accepts |
|---|---:|---:|---:|---:|
| synthetic IDs, prompt 256 | pre-fix: verify every n-gram miss | 10.7 | 14.14 | 31/0/0 |
| synthetic IDs, prompt 256 | zero-draft fallback + cooldown | 13.8 | 13.90 | 0/0/0 |
| natural text, prompt 11 | preflight + ordinary graph fallback | 19.6 | 13.87 | 0/0/0 |

Natural prompt: `Write a concise explanation of why the sky appears blue.`

Decoded prefix:

> The sky appears blue due to a phenomenon called **Rayleigh scattering**.

The pre-fix synthetic run spent one two-token recurrent verification pass per
emitted token despite receiving zero drafts, producing the apparent GGUF slowdown.
The guarded path never enters speculative verification when no next-token n-gram
match is possible. A plausible match gets one probe; an actual miss uses ordinary
graphed decode for the following 64 tokens. Recurrent trajectory allocation is also
refused when its measured live-buffer footprint plus 64 MiB workspace exceeds
available and allocator-reclaimable VRAM.

The 19.6 tok/s natural run is 91.9% of the separate 21.3 tok/s Qwen27 baseline from
PR #311; short-run variance and prompt/context buckets differ, so it is not presented
as a kernel-level regression comparison.
