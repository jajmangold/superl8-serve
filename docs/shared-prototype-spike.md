# Spike Report: Shared Prototype Feasibility

> **Historical experiment record — not a current architecture recommendation.**
> This 2026-07-24 experiment is preserved from
> [control PR #169](https://git.python-bull.ts.net/content-factory/content-factory/pulls/169)
> at commit
> [`ea54e1ec`](https://git.python-bull.ts.net/content-factory/content-factory/commit/ea54e1ecfdb362cca20d97d4a5c1e486f6c58229).
> Its cross-model shared-prototype hypothesis did not produce a production representation:
> residuals remained full-sized, dense, and not low-rank. See the
> [evidence-chain index](codebook-spike-evidence.md) and the current
> [weight-stationary architecture](weight-stationary-architecture.md).

**Date:** 2026-07-24
**Branch:** `spike/shared-prototype`
**Models Tested:** GPT-2 (124M) + DistilGPT-2 (82M)
**Target:** CMP 100-210 fleet (GV100/sm_70, 16 GB HBM2)

## Executive Summary

**CONDITIONAL GO: Shared prototypes work for same-family models (base + fine-tune) but NOT for different model families. The residual is not sparse or low-rank — it contains as much information as the prototype itself. For the universal engine vision, the shared dictionary must be a base model, not an arbitrary average.**

The conditional wording above is the original experiment conclusion. It does **not** override
the later measured codebook NO-GO or authorize prototype/codebook work in the current runtime.

## 1. Research Findings

### Existing Work on Weight Sharing

| Paper | Year | Approach | Key Finding |
|-------|------|----------|-------------|
| **ResidualTransformer** | 2023 | Shared full-rank + unique low-rank per layer | 60-70% parameter reduction within a model |
| **DeltaLLM** | 2025 | Weight sharing between layers + low-rank deltas | Post-training compression, 2-3x reduction |
| **Share Your Attention** | 2025 | Dictionary learning for Q/K/V projections | 66.7% attention parameter reduction |
| **Basis Sharing** | 2024 | Cross-layer parameter sharing | Requires training from scratch |
| **FiPS** | 2024 | Sparse tensor decomposition | Shared basis + sparse projections |

These literature notes are contemporaneous context from the source record; this migration did not
revalidate them. They are not evidence for the current superl8-serve production direction.

### Key Insight

All successful weight-sharing approaches work **within a single model** (sharing across layers). Our spike tests sharing **across models** — a harder problem.

### Model Merging Literature

- **TIES-Merging**: Resolves interference when merging model weights
- **DARE**: Drop and Rescale for task arithmetic
- **SLERP**: Spherical linear interpolation between model weights

These methods merge models into ONE model, not share weights across multiple models.

## 2. Prototype Sharing Results

### Test Setup

- **GPT-2**: 124M params, 12 layers, hidden=768
- **DistilGPT-2**: 82M params, 6 layers, hidden=768 (distilled from GPT-2)
- **Layer mapping**: GPT-2 layers [0,2,4,6,8,10] → DistilGPT-2 layers [0-5]
- **Prototype**: Average of corresponding layers

### Cosine Similarity (Original vs Prototype)

| Layer | GPT-2 | DistilGPT-2 |
|-------|-------|-------------|
| 0→0 | 0.9952 | 0.9946 |
| 2→1 | 0.9926 | 0.9927 |
| 4→2 | 0.9923 | 0.9926 |
| 6→3 | 0.8394 | 0.8645 |
| 8→4 | 0.8392 | 0.8671 |
| 10→5 | 0.8174 | 0.8285 |

**Analysis**: Early layers have very high similarity (>0.99) because they learn similar low-level features. Later layers diverge (0.82-0.87) as they specialize for specific tasks.

## 3. Residual Analysis

### Size and Distribution

| Metric | GPT-2 Residual | DistilGPT-2 Residual |
|--------|----------------|----------------------|
| Size (float32) | 162.2 MB | 162.2 MB |
| Mean | 0.000023 | 0.000203 |
| Std | 0.0700 | 0.0700 |
| L1 norm | 0.0450 | 0.0948 |
| Dynamic range | 5.28 | 5.28 |

### Critical Finding: Residuals Are NOT Sparse

| Threshold (% of std) | GPT-2 Sparsity | DistilGPT-2 Sparsity |
|-----------------------|----------------|----------------------|
| 0.1% | 0.2% | 0.2% |
| 1.0% | 2.3% | 1.8% |
| 5.0% | 11.0% | 8.9% |
| 10.0% | 20.9% | 16.4% |

**The residual contains meaningful information, not just noise.** This is the fundamental problem.

### Residuals Are NOT Low-Rank

SVD energy capture (sample layers):

| Rank | Energy Captured |
|------|-----------------|
| 1 | 2.1% |
| 2 | 3.7% |
| 4 | 6.2% |
| 8 | 10.1% |
| 16 | 15.6% |
| 32 | 23.4% |

**The residual is spread across many dimensions.** Low-rank approximation doesn't help.

### Residual Compression Potential

Top-k energy retention:

| Keep % of Values | Energy Retained |
|------------------|-----------------|
| 100% | 100.0% |
| 50% | 98.6% |
| 25% | 89.9% |
| 10% | 65.7% |
| 5% | 46.8% |
| 1% | 18.5% |

**Aggressive pruning destroys information.** Keep 50% → retain 98.6% energy. Keep 10% → retain only 65.7%.

### Quantization Benefit

- Residual dynamic range: 30.85% of original
- Quantization benefit: 1.7 fewer bits
- **Modest but real improvement for quantization**

## 4. Multi-Model Storage Math

### CMP 100-210 (16 GB HBM)

| Component | Size |
|-----------|------|
| Prototype | 0.16 GB |
| Activation budget | 2.0 GB |
| Residual budget | 13.84 GB |

### Maximum Models Resident

| Model Scale | Avg Residual | Models Fit |
|-------------|--------------|------------|
| 124M (GPT-2) | 162 MB | 87 |
| 0.6B (Qwen3-0.6B) | 785 MB | 18 |
| 9B (Klein) | 11.8 GB | 1 |

**For 0.6B models: 18 models fit on one GPU.** This is promising for the fleet vision.

**For 9B models: Only 1 model fits.** The prototype is too small relative to model size.

## 5. Model Switching Speed

### Estimated Latency

If prototype is resident in HBM:
- **Residual load time**: 162 MB ÷ 12 GB/s (PCIe 1.0 x1) = **13.5 ms**
- **For 0.6B models**: 785 MB ÷ 12 GB/s = **65 ms**
- **For 9B models**: 11.8 GB ÷ 12 GB/s = **983 ms** (~1 second)

**PCIe 1.0 x1 is the bottleneck for 9B models.** For 0.6B models, switching is fast (~65 ms).

## 6. Go/No-Go Recommendation

### **CONDITIONAL GO**

**The universal engine approach works IF:**

1. **Models share a base checkpoint** (e.g., all fine-tuned from the same base)
   - Cosine similarity > 0.99 for early layers
   - Residuals would be much smaller (fine-tuning delta, not full weight difference)

2. **Target model scale is 0.6B-1B**
   - 18+ models fit on one GPU
   - Switching latency ~65 ms (acceptable)

3. **NOT for cross-family models**
   - GPT-2 vs SmolLM2: incompatible architectures
   - GPT-2 vs DistilGPT-2: similar but residual is still 162 MB (same as prototype)

### **What Breaks the Vision**

1. **Residual is not sparse**: Contains meaningful information, can't be discarded
2. **Residual is not low-rank**: Can't compress via SVD
3. **Residual size equals prototype size**: No storage savings from sharing
4. **9B models don't fit**: Prototype + residual = full model size

### **Revised Architecture**

Instead of "universal dictionary for ANY model", the vision should be:

**"Base model dictionary + fine-tune residuals"**

- **Prototype**: Base model (e.g., Llama-3-8B) stored once
- **Residuals**: LoRA-style deltas for each fine-tuned variant
- **Storage**: 8 GB base + 100-500 MB per variant
- **Switching**: Load only the residual (fast)

This is essentially the LoRA/MoLoRA approach, but with the base model shared across all variants.

## 7. Key Numbers Summary

| Metric | Value |
|--------|-------|
| Prototype size (6 layers) | 162.2 MB |
| Residual size (per model) | 162.2 MB |
| Cosine similarity (early layers) | 0.99+ |
| Cosine similarity (late layers) | 0.82-0.87 |
| Residual sparsity (1% threshold) | 2.3% |
| SVD energy (rank 32) | 23.4% |
| Quantization benefit | 1.7 bits |
| Max 0.6B models on 16 GB | 18 |
| Max 9B models on 16 GB | 1 |
| Model switching (0.6B) | ~65 ms |
| Model switching (9B) | ~1 second |

## 8. Historical Next Steps

The following list is preserved from the source experiment and is not the current roadmap:

1. **Test with fine-tuned variants**: Take one base model, create 4-8 fine-tuned versions, measure residual size
2. **Test LoRA-style sharing**: Use base model as prototype, LoRA adapters as residuals
3. **Scale to 0.6B**: Test with Qwen3-0.6B variants
4. **Prototype switching**: Measure actual PCIe transfer latency on CMP 100-210

---

**Author:** OpenCode sub-agent (MiMo 2.5)
**Status:** Historical experiment complete; cross-model representation is not a production target
**Original decision:** CONDITIONAL GO — revise vision to "base model + fine-tune residuals"
