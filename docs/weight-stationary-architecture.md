# Weight-Stationary Runtime Architecture

## Goal

Maximize **throughput** by minimizing **weight movement per output token**, not by minimizing single-request latency.

## Revised Direction (2026-07-25)

The universal codebook hypothesis is **falsified**. Shared scale/codebooks across layers destroy
model quality (perplexity 8,094,770 vs 21.22 baseline). Scalar PQ also failed end-to-end inference
(perplexity 25.52 → 300.55, KL 1.6061, token accuracy 60%), and its corrected index entropy is
7.0 bits rather than the initially reported 0.05. Per-row INT8 + scale is near-lossless and simple.

The weight-stationary runtime hypothesis **remains intact**. The innovation moves from compression to execution.

### What the spikes proved

| Hypothesis | Result | Evidence |
|-----------|--------|----------|
| Shared prototype across models | NO-GO | Residuals remained full-sized, dense, and not low-rank |
| Naive head/layer weight sharing | NO-GO | PPL ~12 → 28–114 million |
| Scalar/vector PQ on weights | NO-GO | PPL 25.52 → 300.55; KL 1.6061; token accuracy 60% |
| Shared scale/codebook | NO-GO | PPL 21.22 → 8,094,770 |
| Per-row INT8 + scale | GO | PPL 21.19 in the scale/codebook spike |
| NF4 quantization | Historical measured result | See `docs/compression-spike.md`; not the native format |

### Pareto frontier (CMP 100-210)

| Format | PPL | tok/s | Size | Quality | Speedup |
|--------|-----|-------|------|---------|---------|
| FP16 | 6.21 | 12.0 | 1503 MB | 100% | 1.0x |
| **W8A8 (superl8)** | **~6.25** | **136.2** | **752 MB** | **~99.4%** | **11.4x** |
| AWQ Q4 | ~6.27 | ~120 | 376 MB | ~99% | ~10x |
| NF4 | 7.88 | 31.5 | 559 MB | 73% | 2.6x |

**W8A8 with superl8 engine (CUDA graphs) achieves 11.4x speedup despite only 2x compression. Execution engine matters more than weight format.**

## Core Execution Model

The scheduler is **layer-centric**, not request-centric.

Each layer owns a queue of waiting activations.

Execution loop:

1. Select the next layer with pending work.
2. Load that layer's W8A8 weights from HBM.
3. Process every waiting activation via dp4a.
4. Store outputs into the next layer's queue.
5. Repeat.

Skip empty layers entirely.

## Weight Format: W8A8 as Native Format

W8A8 (int8 per-row scale + int8 index) is the native weight format:

```
weight = scale × index
```

- Index: int8 (1 byte per weight)
- Scale: fp32 (one per row)
- Reconstruction: multiply scale × index
- Quality: near-lossless (PPL 21.19 vs 21.22 baseline)

This is NOT a new representation. It's what superl8 already does. The runtime treats W8A8 as the native format for all operations.

### Why W8A8 wins

1. **dp4a is the fastest path on CMP 100-210**: tensor cores are firmware-limited, dp4a is not
2. **Per-row scale adapts to each row's distribution**: no shared codebook needed
3. **1 byte per weight**: half the memory of FP16
4. **Near-lossless**: 99.4% quality retention
5. **Already production-proven**: superl8 has been running this for months

### What about Q4/NF4?

Q4 (AWQ) and NF4 are viable for storage-constrained scenarios:
- Q4: 4x compression, 99% quality, good for model distribution
- NF4: 2.69x compression, 73% quality, acceptable for non-critical paths

But for GPU inference, W8A8 is strictly better: 11.4x speedup with better quality than NF4.

## Pipeline

Assign contiguous layer groups to GPUs.

Example:

* GPU0: layers 0–7
* GPU1: layers 8–15
* GPU2: layers 16–23
* GPU3: layers 24–31

Weights stay resident (W8A8 in HBM).

Only activations move between GPUs.

## Scheduling

Maintain a priority queue of active layer stations.

Prioritize using:

* queue size
* latency deadline
* weight residency
* expert locality

Allow small delays (≈1 ms) to build microbatches.

## MoE

Each expert owns its own queue.

Only experts with waiting activations execute.

Keep hot experts resident (W8A8 weights in HBM).

## GDN / Sliding Window / Mamba

Keep recurrent state or KV local to the layer owner.

Only hidden activations move between stages.

Avoid moving recurrent state.

## Kernels

Use fused kernels throughout:

* W8A8 DP4A (native format)
* fused attention (int8 dp4a)
* fused MoE routing
* fused recurrent updates
* CUDA graphs
* async DMA

Avoid intermediate buffers.

## Speculative Decode

Support:

* DFlash (parallel block drafting)
* Multi-token prediction
* Block verification

Verify multiple candidate tokens in one layer sweep.

## Storage Hierarchy

```
GPU HBM (16 GB)
├── W8A8 weights (hot, resident)
├── KV/GDN state
├── Activations (staging buffers)
└── Scratch (MoE sorting, transport)

Host DDR4 (64-128 GB)
├── Cold weights (NF4/Q4 for offloading)
├── Model metadata
└── LoRA adapters

Disk/Optane (TB)
├── Model archive
└── Checkpoints
```

W8A8 weights stay resident in HBM. Cold weights can be offloaded to DDR4 as NF4/Q4.

## Long-Term Direction

Optimize for:

* **Execution engine** (not weight representation)
* Active parameter reduction (MoE)
* Bounded-state architectures (GDN / Mamba)
* Speculative decoding
* Layer-major scheduling
* Compressed transport (P2P, int4+entropy)

## Success Metrics

Primary:

* aggregate tokens/sec
* bytes moved per output token
* GPU occupancy
* pipeline utilization

Secondary:

* single-request latency

## What Changed (2026-07-25)

Removed: universal codebook, prototype layers, hierarchical codebooks, fused lookup+DP4A.

Added: W8A8 as native format, Pareto frontier analysis, revised storage hierarchy.

The runtime should behave like a weight-stationary dataflow engine using W8A8 as the native format, not a conventional request-by-request inference server.

## References

- `docs/train-model-design.md` — original design brief
- `docs/codebook-spike-evidence.md` — dated provenance and correction chain
- `docs/compression-spike.md` — historical NF4 spike; codebook next step retired
- `docs/shared-prototype-spike.md` — cross-model prototype evidence (NO-GO for production representation)
- `docs/pq-codebook-spike.md` — initial PQ spike (**superseded / incorrect**)
- `docs/pq-rigorous-spike.md` — canonical PQ correction (NO-GO)
- `docs/weight-sharing-compression-spike.md` — naive sharing spike (NO-GO)
- `docs/scale-codebook-spike.md` — scale+codebook spike (NO-GO)
- [Pareto frontier source record](https://git.python-bull.ts.net/content-factory/content-factory/src/commit/070c9c3812629759aa983c33e97986def75418e9/docs/pareto-frontier-spike.md) — W8A8 wins; immutable control-plane source
- `docs/universal-engine-design.md` — original universal engine design (superseded)
