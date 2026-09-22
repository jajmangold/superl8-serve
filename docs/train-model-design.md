# Weight-Stationary Layer Scheduler

The goal is to maximize throughput by minimizing weight movement instead of minimizing latency.

## Core idea

The execution unit is the **layer**, not the request.

Each layer owns a queue of waiting activations.

When a layer executes:

1. Load or activate that layer's weights.
2. Process every activation currently waiting.
3. Store the resulting activations into the next layer's queue.
4. Move to the next layer that has pending work.

Empty layers are skipped entirely.

## Scheduling

The scheduler is event-driven.

Maintain a priority queue of active layers.

Priority should consider:

* queue size
* oldest waiting activation
* whether weights are already resident
* expert locality
* latency deadline

Small queues may wait briefly (≈0.5–2 ms) to form larger microbatches.

## MoE

Each expert has its own queue.

Only experts with queued activations execute.

When an expert runs:

* load expert once
* process all waiting activations
* unload if necessary

Hot experts remain resident whenever possible.

## GDN / Sliding Window

Keep recurrent state or KV cache resident with the layer that owns it.

Only hidden activations move between pipeline stages.

Do not move recurrent state unnecessarily.

## Pipeline

Each GPU owns a contiguous block of layers.

Example:

* GPU0: layers 0–7
* GPU1: layers 8–15
* GPU2: layers 16–23
* GPU3: layers 24–31
* GPU4: layers 32–39
* GPU5: layers 40–47

Weights stay resident.

Only activations cross GPUs.

## Kernels

Use fused kernels wherever possible:

* W4A8/W8A8 DP4A GEMM
* fused dequant + matmul
* fused MoE routing
* fused attention
* fused recurrent updates

Avoid intermediate buffers.

## Speculative decoding

Support multi-token verification.

Queue speculative blocks together so one layer pass verifies multiple tokens.

## Future compression

Architecture should support:

* shared prototype layers
* streamed layer deltas
* LoRA deltas
* PQ/codebook residuals
* sparse residuals

The execution engine should treat all of these as alternate weight providers behind the same interface.

## Objective

Optimize for:

* maximum aggregate tokens/sec
* minimum bytes moved per output token
* maximum weight reuse
* high GPU occupancy
* asynchronous execution
* coarse pipeline parallelism

Latency is secondary to throughput.

---

# Appendix A: Bandwidth Analysis (CMP 100-210 fleet)

Hardware: GV100/sm_70, 16 GB HBM2 at 829 GB/s, PCIe 1.0 x1 at ~250 MB/s,
int8 dp4a at ~46 TOP/s.

## Decode is memory-bound

| Model | Batch | Weight reads/step | Load time | Compute time | Ratio |
|-------|-------|-------------------|-----------|--------------|-------|
| 0.6B (28L, ~12 MB/L) | 1 | 347 MB | 386 µs | 1.2 µs | 322:1 |
| 0.6B | 128 | 347 MB | 386 µs | 159 µs | 2.4:1 |
| 0.6B | 1024 | 347 MB | 386 µs | 1.27 ms | 0.3:1 |
| 9B (32L, ~255 MB/L) | 1 | 8.16 GB | 9.07 ms | 94.5 µs | 96:1 |
| 9B | 128 | 8.16 GB | 9.07 ms | 12.1 ms | 0.75:1 |

At batch=1, 99.7% of decode time is weight loading. Crossover to compute-bound
happens around batch=256–512 (0.6B) or batch=128–256 (9B).

## Per-layer weight sizes (int8)

| Model | QKV | O | Gate+Up | Down | Total/layer |
|-------|-----|---|---------|------|-------------|
| 0.6B (H=1024) | 3 MB | 1 MB | 5.6 MB | 2.8 MB | 12.4 MB |
| 9B (H=3584) | 38.5 MB | 12.8 MB | 135.8 MB | 67.9 MB | 255 MB |

## Staging buffer memory

`N_tokens × hidden_dim × 2 bytes` per buffer.

| Model | Batch=1 | Batch=128 | Batch=1024 |
|-------|---------|-----------|------------|
| 0.6B (H=1024) | 2 KB | 256 KB | 2 MB |
| 9B (H=3584) | 7 KB | 896 KB | 7.2 MB |

32 buffers for 32-layer model: 28 MB (9B, batch=128). Negligible.

## PP wire transfer (PCIe 1.0 x1)

Hidden state per token (int8 codec via `superl8.transport`): `hidden_dim × 1 byte`.

| Model | Batch | Transfer size | Transfer time | Compute/GPU | Wire % |
|-------|-------|---------------|---------------|-------------|--------|
| 9B 4-way PP | 128 | 448 KB | 1.8 ms | 20–40 ms | 4.5–9% |

Wire is hidden behind compute. Staging buffers make PP asynchronous — each GPU
runs at its own pace, tokens accumulate at boundaries.

## MoE expert weight cycling

| Metric | Value (9B, 64 experts) |
|--------|------------------------|
| Expert weight (gate_up+down) | ~22 MB |
| Active experts per step (top-2, batch=128) | ~10–15 of 64 |
| Weight loaded per MoE layer | ~220 MB (15.6% of total) |
| Expert load time (HBM) | 26.6 µs per expert |
| Expert compute (18 tokens) | ~53 µs |
| Prefetch hiding threshold | compute > 26.6 µs |

---

# Appendix B: CUDA Graph Strategy

## Current: whole-step graph

`GraphedDecode` captures ALL layers in one `cudaGraphLaunch`. One launch per
decode step. Works for dense, non-MoE models.

## Proposed: per-layer graphs

N separate graphs, one per layer. Each reads from `staging_buf[i]`, writes to
`staging_buf[i+1]`. Benefits:

* Per-layer weight residency control (pin in L2 while tokens pass)
* Inter-layer pipelining via CUDA streams (load i+1 while computing i)
* MoE expert GEMMs become graph-capturable (per-expert subgraph)
* Skip-empty: if `staging_buf[i].active_count == 0`, skip the entire layer

Cost: N_layers × N_buckets graphs. For 28L × 8 batch buckets = 224. Each pins
~1 MB workspace. Total: ~224 MB. Acceptable on 16 GB cards.

## Graph-capturable vs eager

| Component | Capturable? | Notes |
|-----------|-------------|-------|
| Dense layer (QKV+O+MLP) | Yes | Static shapes per bucket |
| GDN layer | Yes | Fused recurrent, already capturable |
| MoE router + sort | No | Data-dependent (`mask.nonzero()`) |
| MoE expert GEMM | Yes | Per-expert subgraph, contiguous tokens |
| MoE scatter + combine | Yes | Static after sort |
| Skip-empty check | Eager | Host-side active_count read |

---

# Appendix C: Speculative Decoding Integration

## Current cascade (superl8-serve)

```
Grammar tier-0 → n-gram → MTP head → verify
```

MTP drafts sequentially (1 token per forward). Verify runs all tokens through
full model in one pass.

## Proposed cascade

```
Grammar tier-0 → n-gram → DFlash → MTP fallback → verify
```

DFlash (Z Lab, 2026) drafts an entire block of 8–16 tokens in ONE parallel
forward via block diffusion. KV injection reads hidden states from staging
buffers at each layer boundary — nearly free since staging buffers already
contain the computed hidden states.

### DFlash in the weight-stationary model

1. DFlash draft model fires as a lightweight pre-car (constant-time, ~0.5 ms)
2. KV injection: `draft_kv[i] = linear_proj(staging_buf[i].hidden)`
3. Block diffusion forward: 5 draft layers → 16 draft tokens
4. Draft tokens queue at layer 0's staging buffer
5. Main model processes base + 16 drafts through layer-cycling

### DFlash vs MTP

| Aspect | MTP | DFlash |
|--------|-----|--------|
| Draft generation | Sequential | Parallel (one forward) |
| Draft latency (k=16) | 16 × forward | 1 × forward |
| Context conditioning | Prefix-KV | KV injection from staging |
| Throughput gain | ~1.3× | ~3–4× |

---

# Appendix D: Implementation Phases

## Phase 0: Design validation

* Benchmark baseline on GPU 6 (CMP 100-210)
* Profile per-layer weight load times
* Validate skip-empty concept

## Phase 1: Per-layer staging buffers

* `StagingBuffer` class (persistent per-layer activation tensors)
* Per-layer CUDA graph capture (replaces whole-step `GraphedDecode`)
* Skip-empty: check active_count before firing layer graph
* Bit-identical output (653 tests pass)
* Fallback: `SUPERL8SERVE_LAYER_GRAPH=0`

## Phase 2: Inter-layer pipelining

* Double-buffer staging (load i+1 while computing i)
* CUDA stream overlap
* L2 persistence hints for current layer weights

## Phase 3: MoE expert cycling

* Expert token sort + per-expert staging
* Per-expert CUDA graph capture
* Expert prefetch (overlap compute i with load i+1)
* Hot-buffer LRU for frequently-used experts

## Phase 4: DFlash integration

* DFlash draft model loading (`.superl8` or GGUF)
* KV injection from staging buffers
* Cascade integration (grammar → n-gram → DFlash → MTP)
* Acceptance length measurement

## Phase 5: Async PP

* Staging buffer transfer via `superl8.transport` (int8 codec)
* Async producer-consumer between PP stages
* Skip-empty across PP boundaries (no work = no wire transfer)
* Natural batch accumulation at staging boundaries

---

# Appendix E: Open Questions

1. **L2 residency for large models.** 9B layers are 255 MB — far exceeds V100's
   6 MB L2. Partial pinning? Profile L2 hit rates per layer type.

2. **Scheduler complexity.** Event-driven priority queue with microbatch
   formation (0.5–2 ms wait). Does the wait hurt latency-critical single-stream?

3. **Skip-empty overhead.** Per-layer active_count check is a host-device sync
   point if read from GPU. Keep a shadow counter on host? Atomic?

4. **DFlash draft model training cost.** Per target model. Can draft models be
   shared across similar architectures?

5. **Expert sort overhead.** Host-side sort for 128 tokens × 64 experts.
   Fast enough on CPU? Or device-side radix sort needed?

6. **Interaction with prefix cache.** Staging buffers are per-step, not
   prefix-cached. Lifecycle conflict with RadixAttention?

7. **Future compression interface.** LoRA deltas, PQ residuals, sparse
   residuals — the weight provider interface must be abstract enough to
   support all of these without per-format code paths.
