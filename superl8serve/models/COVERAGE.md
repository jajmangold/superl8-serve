<!-- SPDX-License-Identifier: MIT -->
# Model coverage matrix

superl8-serve is **modular by construction**: a model family is a `ModelConfig` +
a thin `models/<family>.py` assembly over shared layers, registered with
`@register_model(...)`. The engine, runner, and kernels never change per family.
Support splits by **attention backend** — that's the axis that decides whether a
family runs on today's superl8 dp4a kernels or needs a new one.

Legend (**status** column): ✅ int8 decode wired end-to-end · 🟩 registered +
prefill-tested (decode path still torch / needs a kernel) · 🟡 config ready, needs real
weights · 🟧 scaffold · ⛔ one sub-path still needs a new superl8 kernel (annotated in notes;
the rest of the family is int8). A cell may pair symbols (e.g. ✅⛔ = int8 decode wired, but
a named prefill/secondary path is still torch — see notes).

**The `released` column ≠ "decodes".** ✅ under `released` only means a `.superl8` checkpoint
exists / is published for the family. Whether the family actually **decodes int8 end-to-end
is governed solely by the `status` column** — only a ✅ status means full int8 decode today
(🟩 = prefill only; ⛔ = one named sub-path is still torch/fp16, called out in that row's
notes). Do not read a released-✅ as "supported for generation."

| family | released | attention backend | MLP | status | notes |
|---|---|---|---|---|---|
| **Qwen3** dense | ✅ | GQA (full) | SwiGLU | ✅ | QK-norm pre-RoPE, explicit head_dim=128, θ=1e6 |
| **Qwen3-MoE** | ✅ | GQA (full) | top-k MoE (no shared) | ✅ | softmax→top-k→renorm; 128 experts / top-8 |
| **Gemma3** (text) | ✅ | GQA (full + **sliding**) | GeGLU | ✅ | 5-local:1-global, dual-θ RoPE, (1+w) norm, √d embed, scale=`qpas^-0.5` |
| **LFM2 / LFM2-MoE** | ✅ | hybrid **short-conv** + GQA | SwiGLU (w1/w3/w2) | ✅ | GQA halves decode int8 (`attn_int8_decode`); ShortConv **decode** routed through fused `superl8.causal_conv1d_decode` (L==1, CUDA, K<=8, no tail transpose/copy; auto-degrades to eager for prefill/CPU/older superl8 — #371); `full_attn_idxs`; double-gated conv(k=3); QK-norm; sigmoid MoE router |
| **GLM-4.5/4.6** | ✅ | GQA + **partial RoPE 0.5** + QKV-bias | sigmoid MoE + shared | ✅ | GQA → int8 decode wired (`attn_int8_decode`/`attn_paged_decode_cached`); e_score_correction_bias select, routed_scaling 2.5, first-k-dense |
| **Hunyuan** (A13B) | ✅ | GQA + QK-norm | softmax MoE + shared | ✅ | GQA → int8 decode wired; `mlp.gate.wg`, `shared_mlp`; CLA (Large) not modeled |
| **Qwen3-Next / Qwen3.5 / Qwen3.6** | ✅ | **hybrid: Gated DeltaNet (linear) + full** | ultra-sparse MoE + shared | ✅ | int8 **decode** wired both halves: gated/full GQA (`attn_int8_decode`) + DeltaNet L==1 step (`superl8.deltanet_recurrent_decode`, auto-degrades). Exact fp32 **prefill** is wired through `superl8.deltanet_gated_chunk_fwd` (auto-degrades, `_DND_PREFILL_GATED`); the ungated int8 chunk path remains an explicit approximation/opt-in fallback (`_DND_PREFILL_INT8`). MoE builder selects `WeightStationaryMoE` (`use_weight_stationary_moe`) or `TransportMoELayer` (`expert_to_gpu`/`local_gpu`, superl8-serve#389) with an all-local parity gate (cos > 0.99 vs `SparseMoE`). |
| **DeepSeek-V3/V4** | V3 ✅ | **MLA (latent KV)** | fine MoE + shared | ✅ | int8 **absorb decode** wired (`superl8.mla_decode_absorb_int8`, `use_int8_absorb=True` default) — folds `W_UK`/`W_UV`. MLA **prefill** is fp32 einsum **by design** (numerics contract; no int8 MLA-prefill kernel) |
| **MiniMax-Text** | ✅ | **lightning (linear)** + softmax hybrid | softmax MoE | ✅⛔ | softmax half → int8 decode wired (GQA). Lightning half: prefill → int8 dispatch wired (`superl8.lightning_attn_int8_fwd` via `_lightning_attn_dispatch`; activates for decay-free case, correctness gate cos at least 0.99 vs fp32 ref). MiniMax ALiBi slopes produce non-trivial per-head decay — int8 kernel is un-gated (decay-free), so MiniMax prefill falls through to fp32 scalar reference. Decode L==1 always fp32 (no graph-capturable lightning decode kernel). postnorm alpha/beta scaling |
| **DiffusionGemma** | ✅ (post-cutoff) | **bidirectional** over canvas | GeGLU MoE | 🟡 | `attn_int8_fwd(causal=False)`; needs DiffusionDecodeStrategy |
| **Gemma4 / gemma3n** | ✅ | GQA + AltUp/LAuReL/PLE/MatFormer | GeGLU | 🟧 | residual-mixing + per-layer-embeddings are new modules |

**Registered + full autoregressive decode** (`tests/test_more_models.py`,
`test_engine.py`, `test_deepseek.py`): LFM2, GLM, Hunyuan, MiniMax, Qwen3-Next all
build through the registry, prefill on the superl8 dp4a kernels, AND decode token-by-token
through the `RecurrentStateCache` (Gated-DeltaNet / lightning recurrent state + the
short-conv trailing window carried across steps). Decode is validated two ways: the
stepwise output matches a single teacher-forced forward over prompt+generated
(`test_*_decode_matches_teacher_forced`), and the same tokens come out through the
continuous-batching engine (`test_engine_qwen3_next_hybrid_matches_runner`). The
recurrent state is keyed **per slot**, so several divergent-family sequences decode
**concurrently** through one engine without corrupting each other's scan
(`test_engine_qwen3_next_concurrent_recurrent_decode`) — that's what makes them serve,
not just single-request generate. DeepSeek decodes through `MLALatentCache`, and its
decode step runs the Track-2 int8 **absorb** kernel (`superl8.mla_decode_absorb_int8`),
which folds `W_UK`/`W_UV` so the O(N) per-step up-projection GEMM disappears.

Real divergent-family **checkpoints** now load end-to-end: the converter persists the
`extra` dict (each family's hybrid layer map — LFM2 `layer_types`, MiniMax
`attn_type_list`, Qwen3-Next linear head dims, DeepSeek `kv_lora_rank`) into the
`.superl8` meta, which the old dump silently dropped. LFM2 is verified from a real HF
checkpoint (convert → int8 `.superl8` → tokenize → 32-token decode). Qwen3-Next / MiniMax
real-checkpoint loading additionally needs `from_hf` to derive the
`linear_attention` / `full_attention_interval` scalar flags from the HF config (the
synthetic-config tests set them by hand) — a small `from_hf` follow-up, tracked
separately; the decode path and cache are complete.
Generative multimodal (Z-Image, Qwen-Image, LTX, Wan, Qwen3-TTS) → see the
diffusion-pipeline plan; the DiT backbone reuses these kernels, the pipeline is new.

## What's proven today (`tests/test_models.py`, GPU)

Registry → `build_model` → `ModelRunner` → prefill + greedy decode through the
superl8 dp4a GEMM + int8 attention kernels, for **Qwen3 dense, Qwen3 untied-head,
Qwen3-MoE, and Gemma3** (sliding-window + dual-θ RoPE + Gemma sandwich norms), plus
MoE routing math. Adding Qwen3/Gemma → **zero** engine changes: pure proof of the
modular seam.

## MTP (Qwen3-Next / DeepSeek-V3 style) — `models/mtp.py`

Shared embed + final-norm + LM head; per depth: two input RMSNorms + `fc(2H→H)` +
one decoder block. Draft `k` tokens, **verify in one causal forward** —
`superl8.attn_int8_verify` already ships that kernel (chain mask). Head + greedy draft
are built; the accept/verify loop lands with the engine (it needs the main model's
KV cache).

## The remaining linear/MLA kernel gaps (superl8 PRs, not serve code)

The big decode kernels have **landed and are wired**: MLA int8 **absorb decode**
(`superl8.mla_decode_absorb_int8`, DeepSeek) and Gated-DeltaNet int8 **decode**
(`superl8.deltanet_recurrent_decode`, Qwen3-Next/3.5/3.6) both ship and are called from
`mla_attn.py` / `linear_attn.py`. What is still open:

1. **Gated DeltaNet prefill** (Qwen3-Next/3.5/3.6): the correctness path is the exact fp32
   `superl8.deltanet_gated_chunk_fwd` (auto-degrades, `_DND_PREFILL_GATED`). The older
   `superl8.deltanet_chunk_int8_fwd` route is ungated and remains an explicit approximation/opt-in path;
   it is not described as exact end-to-end. Decode remains int8 through
   `superl8.deltanet_recurrent_decode`.
2. **Lightning linear attention** (MiniMax): the recurrence runs pure-torch at **both**
   prefill and decode (`lightning_attention`). `superl8.lightning_attn_int8_fwd` exists but is
   unwired for prefill, and there is **no graph-capturable lightning *decode* kernel** at
   all — the genuinely-missing piece.
3. **MLA int8 *prefill*** (DeepSeek-V3/V4): prefill is an fp32 einsum **by design**
   (softmax/LSE/PV numerics contract), not a missing kernel. MoE + MTP are already covered.

Diffusion decoding (DiffusionGemma / LLaDA) needs **no new kernel** — bidirectional
attention is `attn_int8_fwd(causal=False)` — only a `DiffusionDecodeStrategy`
(iterative masked denoising, prefix/block KV policy) in the engine.
