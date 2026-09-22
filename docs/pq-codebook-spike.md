# PQ Codebook Layer Representation Spike

> **SUPERSEDED / INCORRECT — do not use this document's GO conclusion or entropy claim.**
> This 2026-07-24 experiment is preserved from
> [control PR #171](https://git.python-bull.ts.net/content-factory/content-factory/pulls/171)
> at commit
> [`a73a45cd`](https://git.python-bull.ts.net/content-factory/content-factory/commit/a73a45cde76583c68d94910b461eba67d1e8f97f).
> The next-day [rigorous validation](pq-rigorous-spike.md) corrected the reported
> 0.05 bits/index to **7.0 bits/index** and measured PPL **25.52 → 300.55**,
> KL divergence **1.6061**, and token accuracy **60%**. The corrected verdict is **NO-GO**.

**Date:** 2026-07-24
**Model:** Qwen3-0.6B (28 layers, 0.6B params, q_proj: 2048×1024)
**Original status (superseded):** CONDITIONAL GO — scalar quantization works, true PQ likely better

## Question

Can product quantization (PQ) of individual weight matrices achieve better compression than prototype+residual approaches? The previous spike found large residuals, but the user pointed out: **norm ≠ entropy**. A large residual can still be highly compressible via codebooks if it has low entropy.

## Method

Four experiments on Qwen3-0.6B's q_proj weights (28 layers, 2048×1024 each):

1. **Per-layer PQ:** Scalar quantize each dimension into 256 bins (8-bit codes)
2. **Cross-layer shared codebook:** Pool all sub-vectors across layers, shared 256-level codebook
3. **Entropy analysis:** Raw weight entropy vs PQ index entropy, mutual information between dimensions
4. **Depth interpolation:** Linear interpolation between anchor layers

## Key Results

### Experiment 1: Per-layer PQ — EXCELLENT quality

| Metric | Value |
|--------|-------|
| Avg cosine similarity | **0.999686** |
| Min cosine similarity | 0.999451 |
| Avg relative L2 error | 0.0248 |
| Avg compression ratio | **4.0x** |
| Avg raw weight entropy | 6.03 bits |
| Avg PQ index entropy | **0.05 bits (incorrect; corrected to 7.0)** |
| Entropy reduction | **0.008x (invalid)** |
| Avg residual ratio | 0.0248 |

**Every layer achieves >0.999 cosine similarity at 4x compression.** The residual after quantization is only 2.5% of the original weight norm — far smaller than the 54% found in the prototype+residual approach.

This reconstruction-only observation proved insufficient: the rigorous follow-up showed that high
weight cosine did not preserve model quality.

### Experiment 2: Cross-layer Sharing — ALMOST AS GOOD

| Metric | Per-layer PQ | Shared Codebook |
|--------|-------------|-----------------|
| Avg cosine | 0.999686 | **0.999296** |
| Storage | 58.73 MB | **58.72 MB** |
| Codebook overhead | 0.00 MB | 0.00 MB |

A shared codebook across all 28 layers loses only 0.0004 cosine similarity vs per-layer codebooks. Storage is essentially identical because the codebook overhead is negligible (8 dimensions × 3 floats × 8 bytes = 192 bytes).

The original conclusion was that cross-layer sharing was viable. The rigorous follow-up invalidated
that inference by testing reconstructed-model behavior.

### Experiment 3: Entropy Analysis — INVALID RESULT

| Metric | Originally reported value |
|--------|---------------------------|
| Avg raw weight entropy (256 bins) | 6.03 bits |
| Avg raw weight entropy (1024 bins) | 7.12 bits |
| Avg PQ index entropy | **0.05 bits (incorrect)** |
| Entropy reduction ratio | **0.008x (invalid)** |
| Avg mutual info between dimensions | 0.106 bits |
| Avg residual ratio | 0.0248 |

The source document claimed that norm ≠ entropy and that the weights were extremely compressible.
The corrected exact measurement is **7.0 bits/index**, not 0.05. See
[Rigorous PQ Validation Spike](pq-rigorous-spike.md#experiment-2-entropy-verification).

### Experiment 4: Depth Interpolation — DOES NOT WORK

| Anchors | Avg Cosine | Min Cosine | Compression |
|---------|-----------|-----------|-------------|
| 2 | 0.072 | -0.001 | 14.0x |
| 3 | 0.107 | -0.001 | 9.3x |
| 4 | 0.143 | -0.002 | 7.0x |
| 6 | 0.214 | -0.004 | 4.7x |
| 8 | 0.286 | -0.003 | 3.5x |
| 14 | 0.500 | -0.004 | 2.0x |

**Depth interpolation fails completely.** Even with 14 anchors (half the layers), cosine similarity is only 0.50. The smooth depth gradient (0.08→0.95 cosine between layers) does NOT mean layers are linearly interpolable.

The weight space has complex nonlinear structure. Adjacent layers are similar (high cosine) but the path between them is not a straight line in weight space.

## Comparison with Previous Spikes

| Approach | Compression | Cosine | Residual |
|----------|------------|--------|----------|
| Prototype+residual (k=8) | 3.5x | 0.946 | 54% of norm |
| Within-model prototype (k=20) | 1.4x | 0.953 | 45% of norm |
| **Per-layer scalar PQ** | **4.0x** | **0.9997** | **2.5% of norm** |
| **Shared codebook** | **4.0x** | **0.9993** | **2.5% of norm** |

The original comparison concluded that scalar PQ was dramatically better. That conclusion was
based on reconstruction proxies, not inference, and is superseded.

## Original Explanation (Superseded)

The source record proposed four reasons for the apparent result:

1. Weight distributions are peaked.
2. Each dimension appeared to have low entropy.
3. Adjacent dimensions appeared relatively independent.
4. Scalar PQ avoided the large residual created by the prototype approximation.

The entropy premise was measured incorrectly and model inference was not tested. Preserve these
points only as the rationale that led to the follow-up, not as current findings.

## Original True-PQ Hypothesis (Disproved)

The source record proposed that multi-dimensional PQ might capture correlations and improve
compression by 10–20%. The rigorous follow-up found the opposite: vector PQ degraded more than
scalar PQ at every tested block size.

## Go/No-Go

**Original verdict: CONDITIONAL GO — superseded.**

The originally listed GO criteria used compression, cosine similarity, and the incorrect entropy
measurement. They did not include an inference-quality gate. The decisive follow-up failed all
quality gates:

| Criterion | Corrected result |
|-----------|------------------|
| Token accuracy | 60% |
| Perplexity | 25.52 → 300.55 |
| KL divergence | 1.6061 |
| PQ index entropy | 7.0 bits/index |
| Corrected verdict | **NO-GO** |

## Historical Next Step

The original recommendation was to test multi-dimensional PQ. That test was completed by the
[rigorous validation](pq-rigorous-spike.md), which found worse quality at every vector block size.
There is no active next step from this document.

## Artifacts

These paths are historical runtime evidence from the source experiment and may no longer exist:

- Results: `/tmp/pq-spike/results/pq_results.json`
- Script: `/tmp/pq-spike/run_minimal.py`
- Model: Qwen3-0.6B q_proj weights (28 layers × 2048×1024)
