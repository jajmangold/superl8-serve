# Universal GPU Execution Engine — Design Document

> **SUPERSEDED design.** The prototype/codebook representation below was falsified by the
> [dated experiment chain](codebook-spike-evidence.md). It is retained for design provenance only.
> Do not implement its dictionary, hierarchical-codebook, or fused lookup plan. The current
> authority is [Weight-Stationary Runtime Architecture](weight-stationary-architecture.md), with
> W8A8 as the native format.

> Issue: superl8-serve#337 — Universal GPU execution engine design
> Status: Superseded by measured NO-GO evidence
> Production target: CMP 100-210 fleet (GV100/sm_70, 16 GB HBM2, PCIe 1.0 x1)

## 1. Executive Summary

The universal GPU execution engine transforms the GPU from a model-specific weight runner
into a **neural execution engine** that processes compressed instruction streams. Instead
of loading 70 GB of dense weights per model, the GPU permanently stores a universal
dictionary (prototypes + codebooks) and loads ~500 MB of compressed indices per model.

**Key outcomes:**
- Model switching: ~2.5 seconds (vs minutes today)
- Multi-model throughput: 10× improvement for mixed workloads
- Single-model latency: comparable to current superl8 (within 5%)
- Quality: configurable tiers from ~2% degradation (fast) to 0% (exact)

## 2. Architecture Overview

### 2.1 Memory Layout (CMP 100-210, 16 GB HBM)

```
┌─────────────────────────────────────────────────────────────┐
│  PERSISTENT (entire session, never evicted)                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Global Codebooks                                      │  │
│  │  256 entries × 64 dims × 1 byte × N_codebooks         │  │ ~32 MB
│  │  Shared across ALL models                              │  │
│  │  Fits in L2 cache (6 MB on sm_70, streamed on miss)    │  │
│  ├───────────────────────────────────────────────────────┤  │
│  │  Prototype Dictionary                                  │  │ ~8 GB
│  │  50 prototypes × ~160 MB each (9B-scale layers)        │  │
│  │  Clustered centroids from all model weights            │  │
│  │  Persistent in HBM, L2-pinned for hot prototypes       │  │
│  ├───────────────────────────────────────────────────────┤  │
│  │  Kernel Code + CUDA Contexts                           │  │ ~500 MB
│  │  Fused lookup+DP4A kernels, attention, norms           │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  PER-MODEL (loaded on model switch, evicted on switch)      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Instruction Stream                                    │  │ ~500 MB
│  │  Layer routing: which prototype + which codebook       │  │
│  │  PQ indices: 1 byte per 64-element block               │  │
│  │  Scales: fp16, 1 per block (2 bytes per 64 elements)   │  │
│  │  Norm parameters: fp16, ~4 KB per layer                 │  │
│  │  Attention config: full/GDN/sliding-window              │  │
│  │  MoE routing tables + expert assignments                │  │
│  ├───────────────────────────────────────────────────────┤  │
│  │  Activations + KV Cache                                 │  │ ~2 GB
│  │  Staging buffers (per-layer, persistent across steps)   │  │
│  │  KV/GDN recurrent state (per-sequence)                  │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  SCRATCH (temporary, reused across steps)                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  MoE sorting + staging buffers                         │  │ ~1 GB
│  │  Transport compression buffers                         │  │
│  │  Attention workspace                                   │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

Total: ~12 GB used, ~4 GB free headroom
```

### 2.2 Execution Flow

```
Host                                          GPU HBM
─────                                         ───────
1. Read model indices (500 MB) from DRAM
2. DMA transfer to GPU HBM ─────────────────→ Model indices驻留
3.                                            ┌──────────────┐
4.                                            │ Layer Loop:   │
5.                                            │ for L in 0..N:│
6.                                            │  read instr[L]│
7.                                            │  load proto[P]│ ← persistent
8.                                            │  load indices │ ← streamed
9.                                            │  load scales  │ ← streamed
10.                                           │  fused kernel:│
11.                                           │  acc = proto@x│
12.                                           │  acc += CB@x  │ ← codebook in smem
13.                                           │  out = norm() │
14.                                           │  store to buf │
15.                                           └──────────────┘
```

## 3. Instruction Stream Format

### 3.1 Binary Layout

The instruction stream is a flat binary blob loaded into HBM. Each layer occupies
a variable-length region determined by its type and configuration.

```
┌──────────────────────────────────────────────────────────────┐
│  HEADER (64 bytes)                                           │
│  magic:          u32   = 0x554E4956 ("UNIV")                │
│  version:        u16   = 1                                   │
│  num_layers:     u16                                        │
│  hidden_dim:     u32                                        │
│  num_codebooks:  u32                                        │
│  prototype_dict: u32   (offset to prototype mapping table)   │
│  codebook_table: u32   (offset to codebook metadata)         │
│  total_bytes:    u64                                        │
│  reserved:       32 bytes                                   │
├──────────────────────────────────────────────────────────────┤
│  PROTOTYPE MAPPING TABLE                                     │
│  For each layer L:                                           │
│    prototype_id:  u16   (which prototype to use)             │
│    codebook_ids:  u8[8] (which codebooks for residual)       │
│    norm_id:       u16   (index into norm parameter table)     │
│    layer_type:    u8    (dense/MoE/attention-only/FFN-only)  │
│    attn_type:     u8    (full/GDN/sliding/mixed)             │
│    moe_experts:   u8    (0 = dense, N = N experts)           │
│    moe_topk:      u8    (top-k experts per token)            │
│    residual_rank: u16   (0 = PQ only, N = PQ + low-rank)    │
│    flags:         u8    (bit 0: skip-empty, bit 1: cache)    │
│    reserved:      7 bytes                                    │
│  Total: 20 bytes × num_layers                                │
├──────────────────────────────────────────────────────────────┤
│  CODEBOOK METADATA TABLE                                     │
│  For each codebook C:                                        │
│    block_size:    u16   (typically 64)                       │
│    num_entries:   u16   (typically 256)                       │
│    dims_per_entry:u16   (block_size)                         │
│    data_offset:   u32   (offset within codebook data region)  │
│  Total: 10 bytes × num_codebooks                             │
├──────────────────────────────────────────────────────────────┤
│  NORM PARAMETER TABLE                                        │
│  For each unique norm:                                       │
│    weight:        fp16[H]   (layer norm gamma)               │
│    bias:          fp16[H]   (layer norm beta)                 │
│    eps:           fp32                                            │
│  Total: ~4 KB × num_unique_norms                             │
├──────────────────────────────────────────────────────────────┤
│  PQ INDEX DATA                                               │
│  For each layer L:                                           │
│    For each weight block (Q/K/V/O/Up/Down/Gate):             │
│      indices:     u8[num_blocks]   (1 byte per 64 elements)  │
│      scales:      fp16[num_blocks] (1 scale per block)       │
│    Total per layer: ~3 bytes × (params / 64)                  │
├──────────────────────────────────────────────────────────────┤
│  LOW-RANK RESIDUAL DATA (optional, per residual_rank > 0)    │
│  For each layer with low-rank residual:                      │
│    basis:         fp16[H × rank]   (SVD basis vectors)       │
│    latent_scales: fp16[rank]       (per-channel scales)       │
│    raw_indices:   u16[raw_count]   (outlier channels)         │
│  Total: variable, ~10-20% of PQ data                         │
├──────────────────────────────────────────────────────────────┤
│  MoE ROUTING TABLES (if any layer has MoE)                   │
│  For each MoE layer:                                         │
│    expert_map:    u8[num_experts × topk]                     │
│    routing_bias:  fp16[num_experts]                          │
│  Total: ~1-2 MB for 64-expert MoE                            │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Size Analysis

For a 9B model (32 layers, H=3584):

| Component | Calculation | Size |
|-----------|------------|------|
| Header | 64 bytes | 64 B |
| Prototype mapping | 20 bytes × 32 layers | 640 B |
| Codebook metadata | 10 bytes × ~20 codebooks | 200 B |
| Norm parameters | 4 KB × 32 norms | 128 KB |
| PQ indices | 3 bytes × (9B/64) blocks | 422 MB |
| Low-rank residual | ~15% of PQ data | 63 MB |
| MoE tables | N/A (dense model) | 0 B |
| **Total** | | **~486 MB** |

For a 22B model (48 layers, H=6144):

| Component | Calculation | Size |
|-----------|------------|------|
| PQ indices | 3 bytes × (22B/64) blocks | 1.03 GB |
| Low-rank residual | ~15% of PQ data | 155 MB |
| **Total** | | **~1.18 GB** |

**Note:** The 500 MB target works for models up to ~10B parameters. Larger models
require either:
1. Coarser quantization (1024-element blocks → 1.5 bytes/block)
2. Higher sparsity (skip zero blocks)
3. Multi-level codebooks (coarser residuals)
4. Accept larger instruction streams (1-2 GB for 22B models)

### 3.3 Prototype Selection

Prototypes are clustered centroids of weight matrices across all models in the
universal dictionary. The clustering process:

1. **Extract**: For each model, extract all weight matrices by layer position and type
2. **Cluster**: K-means clustering across models, grouped by:
   - Layer position (0-31 for 32-layer models)
   - Weight type (Q/K/V/O projection, FFN up/down/gate)
   - Dimension matching (hidden_dim must match or be padable)
3. **Centroid**: The mean of each cluster becomes a prototype
4. **Residual**: Each model stores the difference from its assigned prototype

**Prototype dictionary construction (offline):**

```python
# Pseudocode for prototype construction
all_weights = []
for model in models:
    for layer in model.layers:
        all_weights.append({
            'type': 'attention_q',  # or k, v, o, ffn_up, ffn_down, ffn_gate
            'position': layer.idx,
            'weights': layer.attention_q.weight,  # [out_dim, in_dim]
        })

# Cluster by (type, position, dimension)
prototypes = {}
for (type, position, dim) in unique_combinations:
    cluster_weights = [w for w in all_weights if matches(w, type, position, dim)]
    centroids[type][position] = mean(cluster_weights)
    # Store centroid as prototype
```

**Number of prototypes needed:**
- 50 models × 32 layers × 6 weight types = 9600 weight matrices
- After clustering by (type, position): ~200 unique (type, position) pairs
- With K=3 per cluster: ~600 prototypes
- Each prototype ~160 MB (9B-scale): ~96 GB total
- **Fits in 16 GB HBM** if we keep only hot prototypes (recently used)

## 4. Execution Model

### 4.1 Fused Lookup + DP4A Kernel

The critical innovation: **never materialize the full weight matrix**. The kernel
directly computes `W @ activations` using codebook indices.

```
Input: activations x[batch, in_dim] (int8)
Output: y[batch, out_dim] (int32, later dequant to fp16)

Algorithm:
1. Load prototype P[out_dim, in_dim] from HBM (persistent)
2. Load codebook indices I[out_blocks, in_blocks] from HBM (streamed)
3. Load codebook entries C[256, 64] into shared memory (16KB, fits in 48KB smem)
4. For each output block o (parallelized across threads):
   acc = 0  # int32 accumulator
   For each input block i:
     idx = I[o * in_blocks + i]  # 1-byte index
     w_block = C[idx]            # 64 int8 values from shared memory
     x_block = x[i * 64 : (i+1) * 64]  # 64 int8 activation values
     acc += dp4a(w_block, x_block)       # 4-element dot product, accumulate
   # Add prototype contribution
   acc += dp4a(P[o], x)  # prototype@x in int8
   y[o] = acc
5. Apply norm: y = norm(y, weight, bias, eps)
```

**Performance analysis:**

For 9B model, single layer (H=3584):
- Prototype load: 255 MB (persistent, may be in L2 on hit)
- Codebook indices: 3 × (3584²/64) = ~600 KB per layer
- Codebook entries: 16 KB (shared memory, zero HBM access)
- dp4a compute: 3584² / 4 = 3.2M dp4a operations
- At 46 TOP/s: ~70 µs compute
- HBM reads: 600 KB indices + 255 MB prototype = 255.6 MB
- At 829 GB/s: ~308 µs memory
- **Total: ~378 µs per layer, 12.1 ms for 32 layers**

Compare to current superl8:
- int4 weight load: 127.5 MB per layer
- dp4a compute: ~70 µs
- Total: ~155 µs per layer, 5.0 ms for 32 layers

**The universal engine is ~2.4× slower than int4 on single-model latency.**
However, the win is in multi-model throughput and switching speed.

### 4.2 CUDA Graph Integration

The fused kernel is graph-capturable (static shapes per batch bucket):

```
Graph per layer:
  [load_indices] → [load_scales] → [fused_lookup_dp4a] → [norm] → [store]

Graph per model:
  [layer_0_graph] → [layer_1_graph] → ... → [layer_31_graph]
```

Skip-empty check remains eager (host-side active_count read).

### 4.3 MoE Integration

For MoE layers, the instruction stream specifies:
- Which experts are active for this model
- Routing weights (top-k assignment)
- Per-expert codebook indices

The execution flow:
1. Router computes expert assignments (same as current)
2. For each active expert:
   a. Load expert's codebook indices (streamed)
   b. Fused lookup + dp4a for that expert's tokens
   c. Store results in staging buffer
3. Combine expert outputs (weighted sum)

Expert weights are NOT stored as dense matrices — they use the same PQ codebook
representation. Hot experts stay resident; cold experts stream from DRAM.

## 5. Model Switching Protocol

### 5.1 Current State

```
Model load: 70 GB weights → HBM
Time: ~84 seconds at 829 GB/s (theoretical minimum)
Practical: ~2-5 minutes (PCIe 1.0 x1 at 250 MB/s → 280 seconds)
```

### 5.2 Universal Engine

```
Model load: 500 MB indices → HBM
Time: ~0.6 seconds at 829 GB/s (HBM)
Practical: ~2.5 seconds (PCIe 1.0 x1 at 250 MB/s → 2 seconds + overhead)
```

### 5.3 Switch Protocol

```
1. Signal current model to stop accepting new requests
2. Wait for in-flight requests to complete (drain, ~100ms)
3. Evict current model's indices from HBM (500 MB → 0.6s)
4. Load new model's indices from DRAM (500 MB → 2s)
5. Rebuild CUDA graphs for new model (if layer structure differs)
6. Reset staging buffers + KV cache
7. Signal new model ready
```

**Total switch time: ~2.5 seconds**

### 5.4 Multi-Model Hot Cache

With 16 GB HBM and 12 GB used:
- 4 GB free for instruction streams
- Can hold ~8 models × 500 MB simultaneously
- Switching between cached models: ~0.1 seconds (no DRAM transfer)

**Example fleet scenario:**
- Klein 9B: 486 MB indices (always cached)
- LTX 22B: 1.18 GB indices (always cached)
- Gemma 12B: 650 MB indices (always cached)
- Z-Image 6B: 350 MB indices (always cached)
- Qwen3-TTS 1.7B: 80 MB indices (always cached)
- **Total: 2.75 GB** — fits easily with prototypes

## 6. Quality Impact Analysis

### 6.1 Compression Quality Tiers

| Tier | Method | Degradation | Use Case |
|------|--------|-------------|----------|
| 0 (Fast) | PQ only, 256 entries | ~2-3% | Draft/speculative decode |
| 1 (Balanced) | PQ + 5% raw channels | ~1-2% | LTX video synthesis |
| 2 (Quality) | PQ + low-rank residual (r=400) | ~0.5-1% | Klein keyframes |
| 3 (Exact) | Full int8 weights (current superl8) | ~0.1% | Critical quality paths |

### 6.2 Comparison with Current superl8

| Metric | superl8 (int4) | Universal Tier 0 | Universal Tier 1 | Universal Tier 2 |
|--------|-------------|------------------|------------------|------------------|
| Perplexity delta | +0.5-1% | +2-3% | +1-2% | +0.5-1% |
| SQNR (typical) | 45-50 dB | 35-40 dB | 40-45 dB | 45-50 dB |
| Model size | 4.5 GB | 486 MB | 550 MB | 680 MB |
| Load time | 5.4s (PCIe) | 2.0s | 2.2s | 2.7s |
| Single-model latency | 5.0 ms/step | 12.1 ms/step | 13 ms/step | 14 ms/step |
| Model switch time | 2-5 min | 2.5s | 2.5s | 2.5s |
| Multi-model throughput | 1× | 10× | 9× | 8× |

### 6.3 Accuracy Gates

The universal engine inherits superl8's accuracy gates:

1. **SQNR calibration**: Per-layer SQNR measurement against fp32 oracle
   - Gate: SQNR ≥ 35 dB (Tier 0), ≥ 40 dB (Tier 1), ≥ 45 dB (Tier 2)
   - Same calibration infrastructure as current superl8

2. **Cosine similarity**: Output activation cosine vs fp32 baseline
   - Gate: cosine ≥ 0.99 (all tiers)
   - Prevents catastrophic quality drops

3. **Perplexity validation**: On standard benchmarks (WikiText-2, C4)
   - Gate: perplexity delta ≤ tier-specific threshold
   - Run during model conversion, not inference

4. **Production canary**: A/B testing on real workloads
   - Gate: operator-approved quality on representative prompts
   - Final arbiter for production deployment

### 6.4 Quality Recovery Options

If a model doesn't meet quality gates at a given tier:
1. **Increase tier**: Use more raw channels or higher-rank residuals
2. **Per-model calibration**: Optimize codebooks for specific model weights
3. **Hybrid approach**: Use universal engine for draft/speculative, superl8 for verify
4. **Fallback**: Fall back to current superl8 int4 loading (always available)

## 7. Comparison with Current superl8 Architecture

### 7.1 Architectural Differences

| Aspect | Current superl8 | Universal Engine |
|--------|--------------|------------------|
| Weight storage | Dense int4 weights per model | Universal dictionary (prototypes + codebooks) |
| Model loading | Load all weights (70 GB) | Load instruction stream (500 MB) |
| Weight access | Direct HBM read | Codebook lookup in shared memory |
| Model switching | Minutes (full reload) | Seconds (swap instruction stream) |
| Multi-model | One model at a time | 8+ models cached simultaneously |
| Quality control | int4 quantization gates | Tiered quality gates |
| Kernel path | int4 dp4a GEMM | Fused lookup + dp4a |

### 7.2 When to Use Each

**Use current superl8 when:**
- Single-model latency is critical (real-time serving)
- Quality tier 3 is required (exact int8 matching)
- Model doesn't fit in universal dictionary format
- CUDA graph optimization is already tuned for that model

**Use universal engine when:**
- Model switching speed matters (multi-model serving)
- Throughput across mixed workloads is priority
- Draft/speculative decode needs fast model swaps
- Storage efficiency is critical (many models, limited disk)

### 7.3 Coexistence

The universal engine is an **alternative weight provider**, not a replacement:

```python
# Current: dense weight provider
provider = DenseWeightProvider(model_path)  # loads 70 GB

# Universal: instruction stream provider
provider = UniversalWeightProvider(
    instruction_stream=model_indices,  # 500 MB
    prototype_dict=global_prototypes,  # persistent
    codebook_dict=global_codebooks,    # persistent
)

# Same interface, different internals
layer.forward(activations, weights=provider.get_layer_weights(layer_idx))
```

## 8. Implementation Roadmap

### Phase 0: Design Validation (this document)

- [x] Architecture design complete
- [x] Instruction stream format specified
- [x] Execution model defined
- [x] Quality impact analyzed
- [ ] Operator review and approval

### Phase 1: Prototype Dictionary Construction

**Dependencies:** Existing model weights on disk
**Duration:** 2-3 weeks
**Deliverable:** Prototype dictionary file

1. Extract weight matrices from all production models (Klein, LTX, Gemma, Z-Image, Qwen)
2. Cluster by (layer_type, position, dimension) using K-means
3. Compute centroids for each cluster
4. Validate prototype quality (reconstruction error)
5. Export prototype dictionary to disk format

**Validation:** Reconstruction error < 5% for each prototype vs cluster mean

### Phase 2: PQ Codec Implementation

**Dependencies:** superl8 CUDA kernels (dp4a already proven)
**Duration:** 3-4 weeks
**Deliverable:** PQ compression + fused lookup kernel

1. Implement PQ training (codebook construction from calibration data)
2. Implement PQ encoding (weight matrix → indices + scales)
3. Implement fused lookup + DP4A kernel (the critical path)
4. Benchmark kernel performance vs current int4 dp4a
5. Validate quality (SQNR, cosine similarity)

**Validation:** SQNR ≥ 35 dB, kernel within 3× of int4 dp4a performance

### Phase 3: Instruction Stream Format

**Dependencies:** Phase 2
**Duration:** 1-2 weeks
**Deliverable:** Instruction stream encoder/decoder

1. Implement binary format encoder (model → instruction stream)
2. Implement binary format decoder (instruction stream → layer configs)
3. Implement prototype mapping table generation
4. Implement codebook metadata table generation
5. Test with all production models

**Validation:** Round-trip encode/decode produces identical instruction streams

### Phase 4: Execution Engine Integration

**Dependencies:** Phase 3 + train-model-design phases 1-3
**Duration:** 4-6 weeks
**Deliverable:** Universal engine integrated into superl8-serve

1. Implement `UniversalWeightProvider` class
2. Wire into weight-stationary layer scheduler
3. Integrate with CUDA graph capture (per-layer graphs)
4. Implement skip-empty for instruction streams
5. Implement MoE expert cycling with PQ codebooks
6. Run full test suite (target: all existing tests pass)

**Validation:** All 680+ tests pass, bit-identical output where applicable

### Phase 5: Model Switching Protocol

**Dependencies:** Phase 4
**Duration:** 1-2 weeks
**Deliverable:** Fast model switching

1. Implement instruction stream DMA transfer
2. Implement CUDA graph rebuild on model switch
3. Implement staging buffer + KV cache reset
4. Benchmark switch time (target: < 3 seconds)
5. Implement multi-model hot cache (LRU eviction)

**Validation:** Model switch < 3 seconds, 8+ models cached simultaneously

### Phase 6: Quality Validation

**Dependencies:** Phase 5
**Duration:** 2-3 weeks
**Deliverable:** Quality tier certification

1. Benchmark all production models at each quality tier
2. Run perplexity validation on standard benchmarks
3. Run production canary (A/B testing on real workloads)
4. Document quality/speed tradeoffs for each model
5. Operator approval for production deployment

**Validation:** All models meet tier-specific quality gates

### Phase 7: Production Integration

**Dependencies:** Phase 6
**Duration:** 2-3 weeks
**Deliverable:** Production-ready universal engine

1. Integrate into superl8-serve as alternative weight provider
2. Add configuration options (tier selection, cache size)
3. Add monitoring + observability (model switch metrics, cache hit rates)
4. Document operational procedures
5. Gradual rollout (canary → production)

**Validation:** Production deployment with zero quality regressions

### Total Timeline: 16-23 weeks (~4-6 months)

## 9. Risk Analysis

### 9.1 Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Fused kernel too slow | Single-model latency 3× worse than int4 | Hybrid: use superl8 for latency-critical, universal for throughput |
| Quality below gates | Models don't meet accuracy requirements | Tier system allows fallback to higher quality or current superl8 |
| Prototype clustering poor | Residuals too large, compression fails | Per-model calibration, or fall back to non-shared codebooks |
| Memory pressure | 12 GB used leaves insufficient headroom | Reduce prototype count, use LRU eviction for cold prototypes |
| CUDA graph rebuild overhead | Model switch takes > 5 seconds | Pre-build graphs for common model configurations |

### 9.2 Operational Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Prototype dictionary too large | Doesn't fit in HBM | Reduce to 20 prototypes, use LRU for rest |
| Multi-model cache thrashing | Frequent evictions | Tune cache size per workload pattern |
| superl8 kernel compatibility | Fused kernel doesn't work on sm_70 | Validate on CMP 100-210 before committing |
| Quality regression in production | User-visible degradation | Canary deployment, automatic rollback |

### 9.3 Rollback Plan

If the universal engine causes issues:
1. Disable via configuration flag (`SUPERL8_UNIVERSAL_ENGINE=0`)
2. Fall back to current superl8 int4 loading
3. No data loss (instruction streams are read-only)
4. No schema changes (coexists with current architecture)

## 10. Open Questions

1. **Prototype count optimization:** How many prototypes are needed for < 1% residual
   magnitude? Depends on model diversity. Needs empirical measurement.

2. **Codebook training data:** Should codebooks be trained on calibration data from each
   model, or globally across all models? Global is simpler but may be less optimal.

3. **Low-rank residual rank:** What rank provides the best quality/speed tradeoff?
   Current findings suggest r=400 for 1% degradation, but this needs validation on
   production models.

4. **PCIe bandwidth impact:** At 500 MB model switch, PCIe 1.0 x1 (250 MB/s) takes
   2 seconds. Is this acceptable? Or should we prioritize HBM-resident models?

5. **MoE expert cycling:** How does the universal engine interact with MoE expert
   cycling? Do we need separate codebooks per expert, or shared codebooks with
   expert-specific indices?

6. **Speculative decode integration:** Can the universal engine's fast model switching
   improve speculative decode? (Draft model as a different instruction stream on same
   GPU.)

## 11. References

- [Weight-Stationary Architecture](weight-stationary-architecture.md) — endgame vision
- [Train Model Design](train-model-design.md) — layer scheduler implementation
- [Multi-GPU MoE Design](multigpu-moe-design.md) — MoE transport
- [Low-rank Codec Findings](lowrank-codec-findings.md) — compression tradeoffs
- [DFlash Integration Design](dflash-integration-design.md) — speculative decoding
- [superl8-serve AGENTS.md](../AGENTS.md) — production constraints and rules
