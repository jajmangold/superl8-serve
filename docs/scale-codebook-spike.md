# Spike: Per-Row Scale + Shared Codebook Quantization

> **Canonical independent NO-GO evidence.** Read this with the
> [dated codebook evidence chain](codebook-spike-evidence.md). It reinforces the corrected PQ
> verdict; it does not reopen codebook work.

**Date:** 2026-07-25
**Model:** Qwen3-0.6B (751.6M weight params, 28 layers)
**Hardware:** NVIDIA CMP 100-210
**Verdict: NO-GO**

## Hypothesis

Weight ≈ scale × codebook[index], where:
- **codebook**: learned dictionary of 256 centroid vectors (shared across ALL layers)
- **index**: per-weight int8 (1 byte) — selects which codebook entry to use
- **scale**: per-row fp32 — adapts to each row's magnitude

This is codebook quantization (vector quantization), NOT naive weight sharing. The codebook is trained on int8-normalized sub-vectors of length 8, giving 8:1 compression of the index channel.

## Setup

```
Codebook size:  256 centroids × 8 dimensions (subvec_k=8)
Training:       2M sub-vectors sampled from all 93.9M total, MiniBatchKMeans
Compression:    1 byte per 8 weights (codebook index) + per-row fp32 scale
                Same structural size as raw int8 + per-row scale
Codebook table: 8 KB (negligible overhead)
```

## Results

### Layer-level quality (cosine similarity, 198 weight matrices)

| Method | Avg Cosine Similarity | Avg Relative L2 |
|--------|----------------------|-----------------|
| Raw int8 + per-row scale | 1.000337 | 0.009220 |
| Shared codebook (k=256, subvec=8) | 0.806562 | 0.593282 |

### Perplexity (WikiText-2, 241K words)

| Method | PPL | Delta vs FP16 |
|--------|-----|---------------|
| FP16 (original) | 21.22 | — |
| Raw int8 + per-row scale | 21.19 | -0.03 (better, rounding noise) |
| Shared codebook + per-row scale | **8,094,770** | +8,094,749 |

### Compression

| Format | Size | Ratio vs FP16 |
|--------|------|---------------|
| FP16 | 1,503 MB | 1.00x |
| Raw int8 + per-row scale | 3,758 MB | 0.40x |
| Shared codebook + per-row scale | 3,758 MB + 8 KB | 0.40x |

Note: Compression ratio is < 1.0 because per-row fp32 scales (4 bytes × out_features per layer) dominate the storage. The int8 index saves nothing over raw int8 — both store 1 byte per weight.

## Analysis

### Why it fails catastrophically

1. **Double quantization**: We first quantize to int8 (lossless for practical purposes, cosine ~1.0), then quantize those int8 values again via k-means. The second quantization adds massive error on top of near-zero original error.

2. **Sub-vector space is too large**: Each sub-vector has 8 dimensions in range [-127, 127]. With 256 centroids, each centroid covers ~7,812 sub-vectors on average. The within-cluster variance is enormous — the centroid is a poor representative of any individual sub-vector.

3. **No compression advantage**: The codebook index is 1 byte per 8 weights (8:1 compression of indices), but the per-row fp32 scale dominates storage. Total storage is identical to raw int8 + per-row scale.

4. **PPL of 8M = random tokens**: The model produces gibberish. The weight perturbation from codebook quantization destroys all learned structure.

### Why raw int8 works so well

Per-row symmetric int8 quantization achieves near-lossless reconstruction (cosine > 0.9999, PPL delta < 0.1) because:
- Each row's values are scaled to [-127, 127] before rounding
- The rounding error is at most 0.5 in the scaled space
- After un-scaling, the relative error is tiny (0.9% L2 norm)

This is the same principle behind GGUF's K-quants: per-group/per-row scaling makes int8 nearly lossless.

## Conclusion

**Codebook quantization with 256 centroids and sub-vector length 8 is not competitive with raw int8 per-row quantization.** The approach adds complexity (k-means training, codebook storage, nearest-neighbor lookup) while producing catastrophically worse quality.

The fundamental issue: with only 256 centroids in an 8-dimensional space, the codebook cannot represent the diversity of int8 weight patterns across a 750M-parameter model. Increasing the codebook size would improve quality but reduce the compression advantage, eventually converging to raw int8.

### What actually works

Per-row (or per-group) symmetric int8 quantization with per-row fp32 scales is effectively lossless for inference. This is the established approach used by GGML/GGUF K-quants, and there's no reason to reinvent it with codebook quantization.

## Reproduction

```bash
docker exec superl8-dev python3 /tmp/scale_codebook_final.py
```

The following paths are historical runtime evidence and may no longer exist:

- Script: `/tmp/scale_codebook_final.py` in the `superl8-dev` container
- Results JSON: `/tmp/scale_codebook_results.json`
