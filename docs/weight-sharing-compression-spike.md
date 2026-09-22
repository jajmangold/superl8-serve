# Weight-Sharing Compression Spike — Qwen3-0.6B

> **Historical experiment record — NO-GO.**
> This 2026-07-24 experiment is preserved from
> [control PR #173](https://git.python-bull.ts.net/content-factory/content-factory/pulls/173)
> at commit
> [`a979a2e4`](https://git.python-bull.ts.net/content-factory/content-factory/commit/a979a2e41aa7d8f25bf470cb86da7745718040ac).
> It establishes only that the tested naive, no-retraining sharing methods fail; it does not
> recommend a current implementation. See the [evidence-chain index](codebook-spike-evidence.md)
> and [weight-stationary architecture](weight-stationary-architecture.md).

**Date:** 2026-07-24
**Model:** Qwen/Qwen3-0.6B (596M params, 28 layers, 16 heads, 8 KV heads, hidden=1024, head_dim=64)
**Hardware:** NVIDIA GPU (CUDA), float32

---

## Goal

Determine if naive weight-sharing (no retraining) can reduce model size while preserving acceptable perplexity on a 0.6B-class language model. This is a go/no-go spike for the content-factory model compression workstream.

## Method

4 experiments, each evaluated on 4 text domains (technical, narrative, code, dialogue):

| # | Experiment | Description | Parameter Count |
|---|-----------|-------------|-----------------|
| 1 | **Baseline** | Unmodified Qwen3-0.6B | 596.0M |
| 2 | **Intra-head sharing** | Average Q/K/V/O projections across attention heads within each layer (all heads get same weights) | 596.0M (same count, but shared weights don't reduce storage without retraining) |
| 3 | **Cross-layer sharing (4)** | Share one master set of weights per group of 4 consecutive layers | 265.7M (−55.4%) |
| 4 | **Hybrid** | Cross-layer 4 + low-rank shared residual (rank=16, initialized near zero) | 265.7M (−55.4%) |

## Results

```
Experiment                   Avg PPL   Δ Baseline       Params
------------------------------------------------------------
Baseline                       11.97         0.00       596.0M
Intra-head Sharing        28393984.79 +28393972.82       596.0M
Cross-layer (4)           114406179.00 +114406167.03     265.7M
Hybrid (4+LoRA)           114406179.00 +114406167.03     265.7M
```

Per-domain breakdown:

```
Domain       Baseline    Intra-head   Cross-layer  Hybrid
technical    14.83       17,895,433   31,945,594   31,945,594
narrative    19.05       54,484,924   184,011,709  184,011,709
code         1.19        19,227,067   168,122,057  168,122,057
dialogue     12.82       21,968,515   73,545,356   73,545,356
```

## Verdict: NO-GO

**Naive weight sharing without retraining catastrophically degrades perplexity** — from 12 → millions. This is expected: transformer weights are learned as a coordinated set. Averaging across heads destroys the differentiation that allows each head to specialize. Sharing across layers eliminates the progression of increasingly abstract representations.

## Analysis

### Why it failed
1. **Head specialization is real.** Different attention heads learn distinct patterns (positional, syntactic, semantic). Averaging them produces a "jack of all trades, master of none" that can't attend coherently.
2. **Layer progression is essential.** Transformers compute increasingly abstract representations layer by layer. Making layers 1-4 share weights destroys the depth hierarchy.
3. **Cross-entropy loss amplifies small errors.** Even modest degradation in logits compounds exponentially through the softmax → perplexity calculation.
4. **The hybrid residual had zero effect** because it was initialized near zero (scaled identity × 0.01) — it's essentially a pass-through. Without training, it adds nothing.

### What would be needed

The following alternatives are preserved as contemporaneous analysis, not active recommendations:

- **Retraining after sharing**: The real test is weight-sharing + continued pretraining/fine-tuning. Literature (e.g., Albert, MobileBERT) shows 10-40% compression is achievable with 1-5% perplexity loss IF you retrain.
- **Structured pruning**: Remove entire heads/layers rather than sharing weights. Less destructive than averaging.
- **Quantization**: 4-bit/8-bit quantization is a more practical compression path for inference.
- **Knowledge distillation**: Train a smaller model to mimic the larger one.

The literature statement above is inherited from the source record and was not revalidated during
this documentation migration.

## Current Disposition

Weight-sharing compression without retraining is a measured **NO-GO**. W8A8 remains the native
production format. Any retraining, pruning, distillation, or alternate quantization investigation
requires its own issue and quality gates.

## Files

These paths are historical runtime evidence from the source experiment and may no longer exist:

- Script: `/tmp/qwen3-compression-spike.py`
- Model cache: `~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B`
