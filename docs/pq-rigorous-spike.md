# Rigorous PQ Validation Spike

> **Canonical correction for the initial PQ experiment.**
> This 2026-07-25 experiment is preserved from
> [control PR #172](https://git.python-bull.ts.net/content-factory/content-factory/pulls/172)
> at commit
> [`935a051a`](https://git.python-bull.ts.net/content-factory/content-factory/commit/935a051a0308ce67c20c028fd5a0d879b9528c8d).
> It supersedes [the 2026-07-24 PQ spike](pq-codebook-spike.md). Its corrected measurements and
> **NO-GO** verdict govern this experiment chain.

**Date:** 2026-07-25
**Model:** Qwen3-0.6B (28 layers, 0.6B params, 311 tensors, 751M params)
**Status:** **NO-GO** — PQ reconstruction destroys model quality despite good cosine similarity

## Executive Summary

The previous PQ spike (2026-07-24) reported "CONDITIONAL GO" with 4x compression and 0.9997 cosine similarity. This spike runs the decisive experiments that determine whether codebook-based layer representation is actually useful.

**Result: The model is USELESS after PQ reconstruction.** Perplexity increases by 11.8x (25.5 → 300.6), token accuracy drops to 60%, and KL divergence is 1.61. The previous spike's optimistic conclusions were based on:
1. Weight cosine similarity (a poor proxy for model quality)
2. A bug in entropy measurement (reported 0.05 bits/index, actual is 7.0 bits)
3. Never running inference to verify the model actually works

## Experiment 1: Model Quality After Reconstruction

**The most important test. Weight cosine means nothing if the model doesn't work.**

### Method
- PQ-quantize all 196 weight matrices (skip lm_head and embed_tokens due to size)
- Reconstruct full model from PQ representation
- Compare: logit KL divergence, token accuracy, perplexity on WikiText-2

### Results

| Metric | Value | Assessment |
|--------|-------|------------|
| Avg weight cosine | 0.994144 | Looks good (deceptive) |
| Min weight cosine | 0.986698 | Still "high" |
| Avg relative L2 error | 0.1076 | 10.8% error per weight |
| Token accuracy | 60.0% (3/5) | **FAIL** — model produces wrong tokens |
| Avg KL divergence | 1.6061 | **FAIL** — distributions are very different |
| Original perplexity | 25.52 | Baseline |
| PQ perplexity | 300.55 | **CATASTROPHIC** — 11.8x worse |
| Perplexity increase | 1077.6% | Model is unusable |

### Per-prompt breakdown

| Prompt | KL | Token Match | Original | PQ |
|--------|-----|-------------|----------|-----|
| "The quick brown fox jumps over" | 0.7713 | MATCH | ✓ | ✓ |
| "In a distant galaxy, scientists discovered" | 0.2651 | MATCH | ✓ | ✓ |
| "The meaning of life is" | 5.0941 | **DIFF** | " a" | "," |
| "def fibonacci(n):" | 0.2452 | MATCH | ✓ | ✓ |
| "Once upon a time, in a land far away," | 1.6546 | **DIFF** | " there" | " a" |

### Why cosine similarity is misleading

A cosine similarity of 0.994 on a 1024-dim vector means each dimension has ~0.01% relative error. But neural networks are extremely sensitive to weight perturbations because:
1. Weights interact multiplicatively through matrix multiplication
2. Small errors compound through 28 layers (error accumulation)
3. Softmax amplifies small logit differences exponentially
4. The PQ reconstruction error is **structured** (not random noise), creating systematic biases

For reference, typical INT4 quantization (Q4_K_M) of a 7B model:
- Weight cosine: ~0.99 (similar to our PQ)
- Perplexity increase: ~2-5% (vs our 1077%)
- Token accuracy: ~99% (vs our 60%)

The difference: INT4 uses per-block scaling that preserves relative structure. PQ's k-means clustering destroys the fine-grained weight relationships that matter for inference.

## Experiment 2: Entropy Verification

**The 0.05 bits/index number from the previous spike was WRONG.**

### Per-tensor entropy (exact, not binned)

| Metric | Min | Max | Mean | Std |
|--------|-----|-----|------|-----|
| Raw weight entropy (exact) | 5.07 bits | 10.82 bits | 9.14 bits | 1.42 |
| PQ index entropy (exact) | 3.71 bits | 7.95 bits | 7.00 bits | 0.89 |
| Codebook utilization | 49.6% | 100.0% | 73.8% | 18.2% |

### Key findings

1. **Index entropy is 7.0 bits, NOT 0.05 bits.** The previous spike's 0.05 bits was a bug — likely caused by computing entropy on a single column of indices or using wrong bin sizes. With k=256 clusters, the maximum entropy is 8 bits. An entropy of 7.0 bits means the indices are nearly uniformly distributed (high information content), which is the opposite of what the previous spike claimed.

2. **Codebook utilization is healthy (73.8%).** Most of the 256 codebook entries are actually used. This confirms the indices carry real information.

3. **The entropy reduction ratio is 0.77x** (index entropy / raw entropy), meaning PQ indices retain 77% of the raw weight entropy. This is NOT the "12.5% of raw" claimed by the previous spike.

4. **LayerNorm weights have entropy_ratio > 1.0** — the PQ indices actually have MORE entropy than the raw weights. This happens because LayerNorm weights are very concentrated (low entropy), but PQ forces them into 256 bins (high entropy). The quantization is adding information, not removing it.

### True bit rate calculation

| Component | Size |
|-----------|------|
| Original (FP16) | 880.9 MB |
| Raw PQ (8-bit indices + codebooks) | 222.1 MB |
| True bits/weight (raw PQ) | 4.03 bits |
| Entropy-coded PQ | 220.1 MB |
| True bits/weight (entropy-coded) | 3.99 bits |

The previous spike claimed "0.05 bits/index" → ~0.1 bits/weight. The actual number is 4.03 bits/weight. The previous spike was off by **40x**.

## Experiment 3: Vector PQ on DP4A-aligned blocks

**Scalar PQ is best. Vector PQ degrades badly.**

### Results

| Block Size | Cosine | Compression | Cos/Scalar | Assessment |
|------------|--------|-------------|------------|------------|
| Scalar (baseline) | 0.996250 | 2.56x | 1.0000 | Best quality |
| Vector PQ bs=4 | 0.964486 | 4.95x | 0.9681 | 3.2% worse |
| Vector PQ bs=8 | 0.883298 | 9.48x | 0.8866 | 11.3% worse |
| Vector PQ bs=16 | 0.781863 | 17.13x | 0.7848 | 21.5% worse |
| Vector PQ bs=32 | 0.688039 | 25.80x | 0.6906 | 31.2% worse |

### Why vector PQ fails

Vector PQ treats each block of N weights as a single vector and clusters them in N-dimensional space. The problem:
1. **Curse of dimensionality:** k-means in high dimensions needs exponentially more clusters to maintain quality
2. **Block size vs codebook size:** With bs=32 and k=256, each cluster must cover a huge volume of 32D space
3. **Weight structure is per-dimension:** Neural network weights have structure along individual dimensions (row/column), not along random blocks of 32 consecutive weights

The previous spike suggested "true multi-dimensional PQ would achieve 6-8x compression." In reality, it's **worse** than scalar PQ at every block size tested.

## Experiment 4: Entropy-coded bit rate

**Entropy coding provides minimal improvement.**

### Results

| Metric | Value |
|--------|-------|
| Original (FP16) | 880.9 MB |
| Raw PQ | 222.1 MB (3.97x) |
| Entropy-coded PQ | 220.1 MB (4.00x) |
| Improvement from entropy coding | 3.2% |

### Reference bit rates

| Method | Bits/Weight | Compression |
|--------|-------------|-------------|
| Q4_K_M (GGUF) | ~4.5 | 3.6x |
| Int8 (superl8) | ~8.1 | 2.0x |
| Raw PQ | 4.03 | 3.97x |
| Entropy-coded PQ | 3.99 | 4.00x |

Entropy coding barely helps because:
1. Index entropy is 7.0 bits (high) — there's little redundancy to exploit
2. The codebook overhead (256 × 2 × 4 bytes = 2 KB per tensor) is small but fixed
3. With 4.03 bits/weight, PQ is already close to the entropy limit for this configuration

## Experiment 5: Fused lookup vs Q4 dequant speed

**PQ lookup is faster in Python, but irrelevant given the quality failure.**

### Results

| Weight Matrix | Q4 Dequant | PQ Lookup | Speedup |
|---------------|------------|-----------|---------|
| layers.0.mlp.down_proj (1024×3072) | 63.1 ms | 43.8 ms | 1.44x |
| layers.0.mlp.gate_proj (3072×1024) | 58.5 ms | 43.4 ms | 1.35x |
| layers.0.mlp.up_proj (3072×1024) | 58.9 ms | 43.2 ms | 1.36x |
| layers.1.mlp.down_proj (1024×3072) | 59.5 ms | 43.7 ms | 1.36x |
| layers.1.mlp.gate_proj (3072×1024) | 59.5 ms | 43.0 ms | 1.38x |
| **Average** | | | **1.38x** |

### Interpretation

PQ lookup is ~1.38x faster than Q4 dequant in Python because:
1. PQ lookup is a simple array index: `codebook[indices[i]]`
2. Q4 dequant requires: read nibbles → shift/mask → multiply by scale → convert to int8

However, this is a Python-level benchmark measuring algorithmic overhead. At the CUDA kernel level:
- Q4 dequant: ~1 cycle per weight (simple shift+multiply)
- PQ lookup: ~2-3 cycles per weight (index load + table lookup + cache miss)

The Python speedup likely reverses at the kernel level due to cache behavior. But it's moot — the model quality is too bad for PQ to be useful.

## Comparison with Previous Spike

| Metric | Previous Spike (2026-07-24) | This Spike (2026-07-25) |
|--------|---------------------------|------------------------|
| Weight cosine | 0.9997 | 0.9941 |
| Compression ratio | 4.0x | 3.97x |
| Index entropy | 0.05 bits | **7.0 bits** |
| Entropy reduction | 12.5% of raw | **77% of raw** |
| Codebook utilization | Not measured | 73.8% |
| Model quality | **Not tested** | **CATASTROPHIC** (11.8x PPL) |
| Token accuracy | **Not tested** | **60%** |
| KL divergence | **Not tested** | **1.61** |
| Verdict | CONDITIONAL GO | **NO-GO** |

### What the previous spike got wrong

1. **Entropy measurement was buggy.** The 0.05 bits/index claim is off by 140x. The actual index entropy is 7.0 bits. This invalidates the claim that "weights are extremely low-entropy" and "true entropy limit is much lower."

2. **Cosine similarity is the wrong metric.** The previous spike optimized for cosine similarity (0.9997) without ever testing whether the model produces correct outputs. Cosine similarity measures vector alignment, not functional equivalence.

3. **"Residual only 2.5% of norm" is misleading.** The residual is small in L2 norm, but the error is structured (not random), which matters more for neural network quality.

4. **Never ran inference.** The most critical test — does the model work after reconstruction? — was never performed.

## Go/No-Go

### **NO-GO for codebook-based layer representation**

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Model produces correct outputs | >95% token accuracy | 60% | **FAIL** |
| Perplexity increase | <5% | 1077% | **FAIL** |
| KL divergence | <0.1 | 1.61 | **FAIL** |
| Compression ratio | >4x | 3.97x | PASS |
| Entropy bits/weight | <1 bit | 4.03 bits | **FAIL** |

### Root cause

Product quantization fundamentally cannot represent neural network weights accurately enough for inference because:

1. **Neural networks are not vector quantization-friendly.** Weights have fine-grained structure along individual dimensions that k-means destroys.

2. **Errors compound exponentially through layers.** A 0.6% weight error per layer becomes a 16% error after 28 layers (1.006^28 ≈ 1.18).

3. **Softmax is amplifying small differences.** Even small logit differences become large probability differences after softmax.

4. **The information-theoretic argument is wrong.** Yes, weights have low entropy in a Shannon sense. But neural networks need the weights to be *accurate*, not just *compressible*. A weight can have low entropy (many repeated values) but still need those values to be precise.

### What would need to change for a GO

The source experiment recorded the following possible research changes; they are historical
counterfactuals, not the current superl8-serve roadmap:

1. **Use quantization-aware training (QAT):** Fine-tune the model with PQ quantization in the loop. This would learn weights that are PQ-friendly.
2. **Use larger codebooks:** k=256 (8-bit) is not enough. Try k=1024 (10-bit) or k=4096 (12-bit).
3. **Use residual PQ:** After first-pass PQ, encode the residual with a second PQ codebook.
4. **Use PQ only for storage, not inference:** Store weights in PQ format for compression, but dequantize to FP16 before inference.
5. **Apply only to non-critical layers:** LayerNorm weights might tolerate PQ better.

## Implications for Weight-Stationary Architecture

The weight-stationary architecture vision (GPU stores prototypes+codebooks, host stores indices) is **not viable** with current PQ techniques. The quality degradation is too severe.

### What still works

1. **NF4 quantization:** 533 MB (2.69x compression) with acceptable quality (from compression-spike.md)
2. **INT8 quantization (superl8):** Current production path, ~2x compression, excellent quality
3. **Q4_K_M GGUF:** ~4x compression, ~2-5% perplexity increase, production-proven

### What doesn't work

1. **PQ for inference:** Quality is catastrophic
2. **Vector PQ:** Worse than scalar PQ at every block size
3. **Shared codebooks across layers:** Adds complexity without quality benefit
4. **Depth interpolation:** Fails completely (cosine 0.50 with 14 anchors)

## Current Disposition

1. **Close this experiment direction for production inference.**
2. **Keep W8A8 as the native production direction.**
3. Treat NF4/PQ cold-storage possibilities as separate work requiring a new issue and fresh
   quality gates; this historical record does not authorize them.
4. See [weight-stationary-architecture.md](weight-stationary-architecture.md) for the current
   runtime architecture.

## Artifacts

These paths are historical runtime evidence from the source experiment and may no longer exist:

- Results: `/tmp/pq-rigorous/results/fast_results.json` (experiments 2-5)
- Results: `/tmp/pq-rigorous/results/exp1_results.json` (experiment 1)
- Results: `/tmp/pq-rigorous/results/perplexity_results.json` (perplexity)
- Scripts: `/tmp/pq-rigorous/run_optimized.py`, `run_exp1.py`, `run_perplexity.py`
- Previous spike (superseded): `docs/pq-codebook-spike.md`
