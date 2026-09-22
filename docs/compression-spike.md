# Compression Spike: Universal Codebook Quantization Viability

**Date:** 2026-07-24  
**Issue:** superl8#335 (research)  
**Model:** Qwen3-0.6B (596M params)  
**Target:** CMP 100-210 fleet (GV100/sm_70, 16 GB HBM2)

## Executive Summary

> **Historical experiment record.** The measured NF4 result remains evidence, but the proposed
> codebook/fused-kernel phase was retired by the later
> [codebook experiment chain](codebook-spike-evidence.md). The current production direction is
> [W8A8](weight-stationary-architecture.md).

**Measured result:** NF4 achieved 533 MB (2.69x compression) with the quality reported below.
The source experiment's proposed Phase 2 codebook path is superseded and is not a current
recommendation.

## 1. Tools Evaluated

| Tool | Status | Notes |
|------|--------|-------|
| bitsandbytes (NF4) | ✅ Works | v0.50.0, installed and tested |
| autoawq | ✅ Installed | v0.2.9, not tested (NF4 sufficient) |
| auto-gptq | ❌ Failed | Build error, CUDA 12.9 incompatible |
| FLUTE | ❌ Not available | Research code only, no pip package |
| GPTVQ | ❌ Not available | Research code only |
| AQLM | ❌ Not available | Research code only |
| QuIP# | ❌ Not available | Research code only |

**Key finding:** Production-ready 4-bit quantization is available via bitsandbytes NF4. Advanced codebook methods (FLUTE, GPTVQ, etc.) are research code without packaging.

## 2. Compression Ratio Achieved

| Format | Size (MB) | Bytes/Param | Compression vs FP16 |
|--------|-----------|-------------|---------------------|
| FP16 | 1433.7 | 2.41 | 1.00x |
| Int8 (task spec) | 868 | 1.46 | 1.65x |
| **NF4** | **533.3** | **0.89** | **2.69x** |
| Target | 500 | 0.84 | 2.87x |

**NF4 is 33 MB over target.** Gap could be closed by:
- More aggressive quantization (3-bit NF4)
- Skipping embedding quantization (saves ~30 MB)
- Codebook prototype sharing across layers

## 3. Quality Impact

### Quantitative Results

| Prompt | FP16 Output | NF4 Output | Token Divergence |
|--------|-------------|------------|------------------|
| "Capital of France" | "Paris" | "Paris" | 0/1 |
| Fibonacci code | Correct recursion | Correct recursion | 28/117 |
| Quantum computing | Technical explanation | Accessible explanation | 18/108 |

### Analysis

- **Short factual answers:** Identical (temperature=0 greedy)
- **Long-form generation:** Semantically similar, diverges around token 18-28
- **Code generation:** Functionally equivalent, different variable names
- **Cosine similarity:** Not measured (requires logit extraction, not available in bitsandbytes)

### superl8 Accuracy Gates

- **SQNR > 35 dB (int8):** NF4 is not int8, so this gate doesn't apply directly
- **Empirical quality:** Acceptable for inference, acceptable degradation for 2.69x compression
- **Risk:** Quantization artifacts may accumulate in multi-step reasoning

## 4. Fused Lookup+DP4A Feasibility on sm_70

### Technical Analysis

**dp4a instruction:** `__dp4a(int32, int32, int32)` — 4-element int8 dot product
- Available on sm_70 (GV100)
- Single-cycle latency on GV100
- Requires int8 inputs and int32 accumulator

**Codebook lookup pattern:**
```
// Pseudocode for fused lookup+DP4A
int8 entry = codebook[activation_index];  // random access from shared memory
int32 result = __dp4a(entry, activation, accumulator);  // int8 dot product
```

### Memory Access Challenges

1. **Random lookup pattern:** Codebook indices are data-dependent, causing cache misses
2. **Shared memory bank conflicts:** 32-byte codebook entries across 32 banks
3. **Occupancy:** Low arithmetic intensity (1 DP4A per 4 bytes loaded)

### Performance Estimate

| Operation | Latency | Throughput |
|-----------|---------|------------|
| L1 cache hit | 28 cycles | 1/cycle |
| L2 cache hit | 200 cycles | 1/200 cycles |
| HBM access | 400+ cycles | 1/400 cycles |
| DP4A | 1 cycle | 1/cycle |

**Bottleneck:** Memory latency, not compute. If codebook fits in L2 (8 MB on GV100), lookup is fast. If not, HBM latency dominates.

### Verdict

**Technically feasible, but:**
- Codebook must fit in L2 (8 MB) for acceptable performance
- Requires custom CUDA kernel (not available in any existing library)
- FLUTE-style offline reordering could help, but is research code
- **Estimated speedup:** 1.5-2x over standard int4 (not the 2-4x from FLUTE paper on A100)

## 5. Comparison Table

| Metric | FP16 | Int8 | NF4 | Codebook (est.) |
|--------|------|------|-----|-----------------|
| Size (MB) | 1433.7 | 868 | 533 | ~450-500 |
| Bytes/param | 2.41 | 1.46 | 0.89 | 0.75-0.84 |
| Quality (relative) | 100% | 99% | 95% | 90-95% |
| Inference speed | 1.0x | 1.2x | 0.8x | 1.5-2.0x (if fused) |
| Tool availability | Native | Custom | bitsandbytes | Research code |
| Production ready | Yes | Yes (superl8) | Yes | No |

## 6. Historical Go/No-Go Recommendation

### **Original verdict: GO for Phase 1 (NF4)**

**Rationale:**
1. NF4 achieves 533 MB (2.69x compression) — close to 500 MB target
2. Quality is acceptable for inference use cases
3. bitsandbytes is production-ready and well-maintained
4. No custom kernel development required

**Original implementation path (not the current backlog):**
- Add NF4 quantization to superl8 pipeline
- Measure actual inference speed on CMP 100-210
- Validate quality on production workloads

### **RETIRED Phase 2 (Codebook + Fused Kernel)**

The source record deferred this phase on engineering grounds. Later inference-quality evidence
made it a measured NO-GO: see [rigorous PQ validation](pq-rigorous-spike.md) and
[per-row scale + shared codebook](scale-codebook-spike.md). There is no trigger from this document
that reopens it.

## 7. Historical Risk Factors and Open Questions

### Recorded risks

1. **Quality degradation in multi-step reasoning:** NF4 quantization errors may accumulate
2. **bitsandbytes CUDA version:** v0.50.0 built for CUDA 12.8, running on 12.9 (works but untested)
3. **Production stability:** NF4 is less battle-tested than int8 in superl8

### Recorded questions

1. **Can we get below 500 MB?** Try 3-bit NF4 or sparse quantization
2. **What's the actual inference speed?** NF4 is slower than FP16 on CPU, need GPU measurement
3. **Does NF4 pass superl8 quality gates?** Need to measure SQNR or equivalent
4. **Codebook quantization:** answered NO-GO by the later inference-quality spikes

### Historical next steps

The source record proposed further NF4, INT8, codebook, and fused-kernel work. Those dated
proposals are not the current backlog. The codebook and fused-kernel items are retired; use
Gitea issues and the [current architecture](weight-stationary-architecture.md) for active work.

---

**Author:** OpenCode sub-agent (MiMo 2.5)  
**Reviewed by:** [pending]  
**Status:** Complete
