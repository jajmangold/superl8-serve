# DFlash Integration Design — Weight-Stationary Decode (Phase 4)

**Issue:** #323
**Status:** Design (research + design, not implementation)
**Author:** OpenCode sub-agent (MiMo 2.5)
**Date:** 2026-07-24

---

## 1. Executive Summary

DFlash (Z Lab, 2026) replaces autoregressive drafting (MTP/EAGLE) with a lightweight block
diffusion model that generates an entire block of draft tokens in ONE forward pass. This design
document covers how DFlash integrates with superl8-serve's weight-stationary layer scheduler
(Phase 1: staging buffers + per-layer graphs, issue #320).

**Key insight:** DFlash's draft model is a 5-layer block diffusion transformer that runs as a
"pre-car" before the main layer train. It reads target model hidden states from staging
buffers via KV injection, generates 8-16 draft tokens in parallel, and queues them at
layer 0's staging buffer for the main model to verify.

---

## 2. DFlash Architecture Summary

### 2.1 Paper Reference

- **Paper:** [arXiv:2602.06036](https://arxiv.org/abs/2602.06036) (ICML 2026)
- **Code:** [github.com/z-lab/dflash](https://github.com/z-lab/dflash)
- **Models:** [huggingface.co/collections/z-lab/dflash](https://huggingface.co/collections/z-lab/dflash)

### 2.2 Draft Model Architecture

| Property | Value |
|----------|-------|
| Layers | 5 (8 for Coder models) |
| Block size | 16 tokens (10 for LLaMA) |
| Hidden size | Same as target model |
| Attention heads | Same as target model |
| Embedding/LM head | **Shared** with target model (frozen during training) |
| Parameters | ~5 layers x target_hidden^2 x 4 (QKV+O+MLP) ~ 25-50 MB |
| Weight format | Separate checkpoint (HuggingFace safetensors) |

### 2.3 Block Diffusion Drafting

Unlike autoregressive drafters (MTP depth-1, EAGLE-3 sequential), DFlash generates all
gamma tokens in parallel:

```
T_draft = t_parallel  (constant, independent of gamma)
```

For block size 16, this means 16 draft tokens in ONE forward pass -- equivalent to 16
sequential MTP steps but with constant latency.

### 2.4 KV Injection Mechanism

**Source:** `dflash/model.py` (z-lab/dflash, MIT license)

The KV injection is the core innovation. During prefill, target model hidden states are
extracted from 5 uniformly sampled layers and projected into the draft model's KV cache
at EVERY draft layer:

```python
# 1. Extract hidden states from target model (during prefill)
layer_ids = [1, 6, 11, 16, 21]  # 5 layers, uniformly sampled
target_hidden = torch.cat([hidden_states[i] for i in layer_ids], dim=-1)  # [B, S, 5*H]

# 2. Project to draft hidden size (fused projection + norm)
fc = nn.Linear(5 * H, H, bias=False)  # cross-layer fusion
hidden_norm = RMSNorm(H)
target_hidden = hidden_norm(fc(target_hidden))  # [B, S, H]

# 3. In each draft attention layer:
k_ctx = self.k_proj(target_hidden)  # [B, S, Hkv, D]
v_ctx = self.v_proj(target_hidden)  # [B, S, Hkv, D]
k_noise = self.k_proj(hidden_states)  # draft token K
v_noise = self.v_proj(hidden_states)  # draft token V
k = torch.cat([k_ctx, k_noise], dim=1)  # [B, S+block_size, Hkv, D]
v = torch.cat([v_ctx, v_noise], dim=1)  # [B, S+block_size, Hkv, D]
# RoPE applied AFTER concatenation
q, k = apply_rotary_pos_emb(q, k, cos, sin)
# KV cache stores concatenated [target_ctx + draft_noise]
```

**Key properties:**
- Target context is **persistent** across drafting iterations (stored in draft KV cache)
- Draft tokens are **masked** during diffusion (noise positions)
- Bidirectional attention within block, causal across blocks
- KV injection reads from staging buffers at each layer boundary

### 2.5 Training Details

- Shared embedding/LM head with target (frozen)
- Random anchor positions for block construction
- Loss weighting: exponential decay w_k = exp(-(k-1)/gamma) for early positions
- Trained on 800K samples from Nemotron + CodeAlpaca
- Efficient long-context: fix blocks/seq, random anchors per epoch

---

## 3. Integration with Weight-Stationary Layer Scheduler

### 3.1 Current Architecture (Phase 1)

```
Prefill -> staging_buf[0] -> Layer 0 graph -> staging_buf[1] -> ... -> staging_buf[N] -> logits
                                    |
                              skip-empty (host-side check)
```

**Key invariant:** `StagingBuffer.active_count` is host-side; `is_empty()` is a Python
branch with zero GPU sync. `GraphedDecodeLayers` captures one CUDA graph per decoder layer.

### 3.2 DFlash Integration Points

```
                    +---------------------------------------------+
                    |              DFlash Draft Model              |
                    |  (5-layer block diffusion, KV-injected)     |
                    |                                             |
                    |  Reads: target_hidden from staging_buf[i]   |
                    |  at layers sampled during prefill            |
                    |  Writes: draft tokens -> staging_buf[0]      |
                    +---------------------------------------------+
                                         |
                                         v
+----------+    +----------+    +----------+    +----------+
| staging  |--->| Layer 0  |--->| staging  |--->| Layer 1  |---> ...
| buf[0]   |    |  graph   |    | buf[1]   |    |  graph   |
| (drafts  |    |          |    |          |    |          |
|  queued) |    +----------+    +----------+    +----------+
+----------+
```

### 3.3 Pre-Car Pattern

DFlash draft runs as a **pre-car** before the main layer train:

1. **Prefill phase:** Run target model prefill, extract hidden states from 5 uniformly
   sampled layers, project through `fc` + `hidden_norm`, store in draft KV cache
2. **Draft phase:** Run 5-layer draft model forward pass, generate block_size tokens
   in ONE pass (block diffusion)
3. **Queue phase:** Write draft tokens to `staging_buf[0]` with `set_active(drafts, k)`
4. **Verify phase:** Main model processes base + k drafts through layer-cycling
   (existing `GraphedDecodeLayers` path)
5. **Accept phase:** Accept longest greedy prefix, commit tokens, update staging buffers

### 3.4 KV Injection from Staging Buffers

The KV injection reads from staging buffers at each layer boundary during prefill:

```python
# During prefill: extract hidden states from target model
# These are the SAME hidden states that staging_buf[i] holds
target_hidden_layers = []
for layer_id in target_layer_ids:  # [1, 6, 11, 16, 21]
    # Run target model up to layer_id, read staging_buf[layer_id].buf
    target_hidden_layers.append(staging_buf[layer_id].buf)

# Concatenate and project
target_hidden = torch.cat(target_hidden_layers, dim=-1)  # [B, S, 5*H]
target_hidden = draft_model.hidden_norm(draft_model.fc(target_hidden))  # [B, S, H]

# Store in draft KV cache (persistent across drafting iterations)
draft_kv_cache.inject_context(target_hidden)
```

### 3.5 Staging Buffer Protocol for Drafts

Draft tokens queue at layer 0's staging buffer:

```python
# After DFlash draft generates k tokens
staging_buf[0].set_active(draft_tokens, k)  # [k, 1, H] -> staging_buf[0].buf
staging_buf[0].active_count = k  # host-side, no GPU sync

# GraphedDecodeLayers picks them up automatically
# Layer 0 graph processes all k tokens in one pass
# Layer 1 graph processes surviving tokens, etc.
```

---

## 4. Cascade Flow with DFlash

### 4.1 Current Cascade

```
grammar (tier-0, free forced run) -> n-gram (prompt-lookup) -> MTP (depth-1 fallback) -> verify
```

### 4.2 Proposed Cascade with DFlash

```
grammar (tier-0, free forced run) -> n-gram (prompt-lookup) -> DFlash (block diffusion) -> MTP (fallback) -> verify
```

**Rationale:**
- Grammar: free, perfect on structured output (unchanged)
- N-gram: free, excellent on repetitive spans (unchanged)
- DFlash: replaces MTP as primary drafter on non-repetitive prose
  - Generates 8-16 tokens in ONE forward pass (vs MTP's 1)
  - Higher acceptance length (~6.5 vs ~3.0 for MTP)
  - KV injection from staging buffers provides rich context
- MTP: fallback when DFlash misses or on very short drafts
  - DFlash may produce 0 drafts on degenerate inputs
  - MTP depth-1 is still useful for single-token proposals

### 4.3 Draft Selection Logic

```python
def cascade_draft_with_dflash(ngram, dflash, mtp_fallback, tokens, k, *, grammar=None):
    # Tier 0: grammar (free forced run)
    if grammar is not None:
        drafts = grammar.propose(tokens[-1], k)
        if drafts:
            return drafts

    # Tier 1: n-gram (free prompt-lookup)
    if ngram is not None:
        drafts = ngram.propose(tokens, k)
        if drafts:
            return drafts

    # Tier 2: DFlash (block diffusion, one forward pass)
    if dflash is not None:
        drafts = dflash.propose(tokens, k)  # returns up to k draft tokens
        if drafts:
            return drafts

    # Tier 3: MTP fallback (depth-1)
    return mtp_fallback() if mtp_fallback is not None else []
```

### 4.4 When MTP Still Fires

MTP fires as fallback when:
1. DFlash draft model is not loaded (no checkpoint available)
2. DFlash produces 0 drafts (degenerate input, all masked positions rejected)
3. DFlash acceptance is below threshold (heuristic: if DFlash avg tau < 2.0)
4. Sequence is on cooldown after DFlash miss (n-gram cooldown pattern)
5. Model doesn't support DFlash (no draft checkpoint for this architecture)

---

## 5. Weight Format and Loading Strategy

### 5.1 Draft Model Checkpoint

DFlash draft models are published on HuggingFace as separate safetensors checkpoints:

```
z-lab/Qwen3.5-9B-DFlash/
  config.json          # Qwen3Config + dflash_config
  model.safetensors    # 5-layer draft weights (~25-50 MB)
  tokenizer.json       # (shared with target, not stored)
  generation_config.json
```

**Key config fields:**
```json
{
  "dflash_config": {
    "target_layer_ids": [1, 6, 11, 16, 21],
    "block_size": 16,
    "mask_token_id": 151665,
    "num_target_layers": 28
  }
}
```

### 5.2 Weight Conversion Strategy

**Phase 1: Direct safetensors loading**
- Load draft model as-is from HuggingFace
- Use `transformers.AutoModel.from_pretrained()` with `trust_remote_code=True`
- Store in VRAM alongside target model
- Overhead: ~25-50 MB VRAM per model

**Phase 2: GGUF conversion (future)**
- Convert draft model to GGUF format
- Load via `superl8.gguf_import` (existing path)
- Benefits: smaller footprint, faster loading, format consistency
- Challenges: block diffusion attention not standard GGUF op

**Phase 3: .superl8 native (future)**
- Convert to .superl8 format with custom op registration
- Best performance, native kernel fusion
- Requires superl8 backend support for draft model ops

### 5.3 VRAM Budget

| Component | Size | Notes |
|-----------|------|-------|
| Target model (27B Q3_K) | ~14 GB | Existing load |
| DFlash draft (5-layer) | ~25-50 MB | New addition |
| Draft KV cache | ~5-10 MB | Block_size x layers x Hkv x D |
| Staging buffers | ~10 MB | Already allocated (Phase 1) |
| **Total overhead** | **~40-60 MB** | <0.4% of 16 GB HBM2 |

---

## 6. Kernel Requirements

### 6.1 Fused KV Injection Kernel

**Location:** `superl8` repo (not superl8-serve -- per AGENTS.md rules)

**Interface:**
```python
superl8.dflash_kv_inject(
    target_hidden: torch.Tensor,  # [B, S, 5*H] fp16
    fc_weight: torch.Tensor,      # [5*H, H] fp16
    k_proj_weight: torch.Tensor,  # [H, Hkv, D] fp16
    v_proj_weight: torch.Tensor,  # [H, Hkv, D] fp16
    k_cache: torch.Tensor,        # draft model KV cache
    v_cache: torch.Tensor,
    cache_slot_mapping: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    positions: torch.Tensor,
) -> None  # writes K/V directly into draft cache
```

**What it fuses:**
1. Linear projection (fc): `5*H -> H`
2. RMSNorm (hidden_norm)
3. K/V projection: `H -> Hkv x D`
4. RoPE application
5. KV cache write (paged)

**Performance target:** <100us for batch=8, S=2048, H=3584 (27B model)

### 6.2 Block Diffusion Attention Kernel

**Location:** `superl8` repo

**Interface:**
```python
superl8.dflash_block_attn(
    q: torch.Tensor,        # [B, block_size, n_heads, D] fp16
    k: torch.Tensor,        # [B, S+block_size, n_kv_heads, D] fp16
    v: torch.Tensor,        # [B, S+block_size, n_kv_heads, D] fp16
    mask: torch.Tensor,     # [block_size, S+block_size] bool
    scale: float,
) -> torch.Tensor           # [B, block_size, n_heads, D] fp16
```

**What it does:**
- Bidirectional attention within block (all draft tokens attend to each other)
- Causal attention across blocks (draft tokens attend to target context)
- KV injection context is prefix (positions 0..S-1)
- Draft tokens are positions S..S+block_size-1

### 6.3 Existing Kernels Reused

- `superl8.quantize_kv_write_paged`: draft KV cache writes (existing)
- `superl8.attn_paged_decode_cached`: verify attention (existing)
- `superl8.quantize_kv_cache`: verify cache construction (existing)

### 6.4 Kernel Development Sequence

1. **First:** Fused KV injection kernel (highest impact, enables end-to-end)
2. **Second:** Block diffusion attention kernel (enables draft model forward)
3. **Third:** GGUF/.superl8 conversion support (optimization, not blocking)

---

## 7. Implementation Phases

### Phase 4a: Draft Model Loading (superl8-serve)

- [ ] Add `DFlashDraftModel` wrapper class
- [ ] Load draft checkpoint from HuggingFace
- [ ] Store alongside target model in VRAM
- [ ] Expose `propose(tokens, k)` interface
- [ ] Integrate with cascade in `drafters.py`

### Phase 4b: KV Injection (superl8 + superl8-serve)

- [ ] Implement fused KV injection kernel in superl8
- [ ] Wire kernel to staging buffer reads
- [ ] Extract target hidden during prefill
- [ ] Store in draft KV cache
- [ ] Validate: draft KV matches reference implementation

### Phase 4c: Block Diffusion Forward (superl8 + superl8-serve)

- [ ] Implement block diffusion attention kernel
- [ ] Wire to draft model forward pass
- [ ] Generate draft tokens in ONE pass
- [ ] Validate: draft tokens match reference implementation

### Phase 4d: Integration with Layer Scheduler

- [ ] Queue draft tokens at staging_buf[0]
- [ ] Verify through GraphedDecodeLayers
- [ ] Accept longest greedy prefix
- [ ] Update MTP cache for fallback
- [ ] Telemetry: DFlash acceptance rate, speedup

### Phase 4e: Cascade Integration

- [ ] Add DFlash tier to cascade_draft
- [ ] Configure via SUPERL8SERVE_SPEC_DRAFTER=dflash
- [ ] Add DFlash-specific cooldown logic
- [ ] Benchmark: DFlash vs MTP vs n-gram

---

## 8. Acceptance Criteria

### Correctness

- [ ] DFlash draft tokens are bit-identical to reference implementation (z-lab/dflash)
- [ ] KV injection produces same K/V as reference (cosine similarity > 0.999)
- [ ] Acceptance length tau matches paper claims (>5.0 for Qwen3.5-9B)
- [ ] No regression on existing spec-decode modes (grammar, n-gram, MTP)

### Performance

- [ ] DFlash draft latency <2ms for block_size=16, batch=8
- [ ] End-to-end speedup >3.0x over autoregressive baseline
- [ ] No regression on n-gram/grammar cascade paths
- [ ] VRAM overhead <100 MB (draft model + KV cache + staging)

### Integration

- [ ] SUPERL8SERVE_SPEC_DRAFTER=dflash enables DFlash tier
- [ ] Fallback to MTP when DFlash unavailable or misses
- [ ] Staging buffer protocol correctly queues draft tokens
- [ ] Per-layer skip-empty works with draft tokens in staging_buf[0]

---

## 9. Open Questions

### 9.1 Draft Model Placement

**Question:** Should the DFlash draft model be a separate `nn.Module` loaded alongside
the target, or should it be a sub-module of the target model?

**Trade-offs:**
- Separate module: cleaner separation, easier to hot-swap, but requires manual KV
  injection wiring
- Sub-module: tighter integration, automatic weight sharing, but harder to disable

**Recommendation:** Separate module loaded by `EngineRunner.__init__`, wired to
staging buffers via a `DFlashDrafter` class in `drafters.py`.

### 9.2 KV Injection Timing

**Question:** When exactly should KV injection happen -- during prefill, or lazily
on first draft?

**Options:**
- A: During prefill (extract hidden from 5 layers, project, cache) -- simpler, but
  requires running target model with `output_hidden_states=True`
- B: Lazily on first draft (extract from staging buffers) -- more complex, but avoids
  changing prefill path

**Recommendation:** Option A (during prefill) -- the staging buffers already hold
hidden states from the target model; we just need to read them at the right layers.

### 9.3 Block Diffusion vs Autoregressive Drafting

**Question:** Should DFlash replace MTP entirely, or coexist as an optional tier?

**Recommendation:** Coexist. DFlash is opt-in via `SUPERL8SERVE_SPEC_DRAFTER=dflash`.
When enabled, the cascade is grammar -> n-gram -> DFlash -> MTP -> verify. When
disabled, the existing cascade (grammar -> n-gram -> MTP -> verify) is unchanged.

### 9.4 Draft Model Quantization

**Question:** Should the DFlash draft model be quantized (int8/int4) or kept in fp16?

**Trade-offs:**
- FP16: higher quality drafts, but ~50 MB VRAM per model
- INT8: half VRAM (~25 MB), but may reduce acceptance length
- INT4: quarter VRAM (~12 MB), but likely reduces acceptance significantly

**Recommendation:** Start with FP16 (reference quality), benchmark INT8 in Phase 4e.

### 9.5 Multi-Model Support

**Question:** How should DFlash scale across multiple target models (Qwen3.5-9B,
Qwen3.5-27B, LFM2)?

**Recommendation:** Each target model gets its own draft checkpoint. The `EngineRunner`
loads the draft model matching the target model's architecture. If no draft checkpoint
exists for a model, DFlash tier is silently skipped (MTP fallback).

---

## 10. References

1. Chen, Liang, Liu. "DFlash: Block Diffusion for Flash Speculative Decoding." ICML 2026.
   arXiv:2602.06036
2. z-lab/dflash. GitHub. https://github.com/z-lab/dflash
3. Issue #320: per-layer staging buffers + skip-empty. superl8-serve.
4. Li et al. "EAGLE-3: Scaling Up Inference Acceleration of Large Language Models via
   Training-Time Test." 2025.
5. Cai et al. "Medusa: Simple LLM Inference Acceleration Framework with Multiple
   Decoding Heads." 2024.
