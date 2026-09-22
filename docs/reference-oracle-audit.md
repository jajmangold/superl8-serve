<!-- SPDX-License-Identifier: MIT -->
# Reference-oracle eligibility audit

**Question:** which superl8-serve model families can be validated with the reusable
llama.cpp/GGUF fp32 reference-oracle harness (`tools/reference_oracle/`), and
which need a different oracle?

The harness is **text-decoder only** — it captures a stock llama.cpp fp32 forward
(`-ngl 0`) op-by-op. A family qualifies iff its architecture is in llama.cpp's
arch registry *and* a GGUF of the checkpoint exists (or can be produced with
`convert_hf_to_gguf.py`). This audit enumerates every registered superl8-serve text
family (`superl8serve/models/*.py` + `COVERAGE.md`) and classifies it:

- **(A) already-covered** — the flint8 #105 work already dumps this graph.
- **(B) eligible & would benefit** — llama.cpp supports the arch; we lacked or
  mistrusted an fp reference for it. Turnkey once you point the harness at a GGUF.
- **(C) not llama.cpp** — image/video DiTs, VLMs, audio, diffusion decoders.
  Out of scope; they need a different oracle. Named here so the boundary is clear.

llama.cpp arch strings verified against the on-box checkout
(`/path/to/local commit `69d8e4be4`; archs present:
`qwen3 qwen3moe qwen35 qwen35moe qwen3next qwen3vl qwen3vlmoe gemma3 gemma3n
gemma4 glm4 glm4moe deepseek deepseek2 deepseek2-ocr hunyuan-dense hunyuan-moe
lfm2 lfm2moe minimax-m2 …`).

## Classification table

| superl8-serve family | serve arch key | llama.cpp arch | GGUF on HF? | bucket | how to validate |
|---|---|---|---|---|---|
| **Qwen3.5 / Qwen3.6** (hybrid) | `qwen3_5` | `qwen35`, `qwen35moe` | on-box + HF (unsloth Qwen3.5 GGUF) | **A** | `dump.sh` the GGUF; already the #105 reference graph |
| **Qwen3-Next** (hybrid) | `qwen3_next` | `qwen3next` | HF (unsloth/bartowski) | **A** | same graph family as Qwen3.5; `dump.sh` + `compare` |
| **Qwen3 dense** | `qwen3` | `qwen3` | abundant (Qwen3-*-GGUF) | **B** | `dump.sh` Qwen3-*-GGUF; check `Qcur/Kcur/Vcur/__fattn__/ffn_out/l_out` |
| **Qwen3-MoE** | `qwen3_moe` | `qwen3moe` | abundant | **B** | as Qwen3 dense; also covers `ffn_moe_*` routing tensors |
| **Gemma3** (text) | `gemma3` | `gemma3` | google + bartowski `gemma-3-*-GGUF` | **B** | `dump.sh`; validates dual-θ RoPE, sliding vs global, sandwich norms |
| **LFM2 / LFM2-MoE** | `lfm2` | `lfm2`, `lfm2moe` | LiquidAI GGUFs (on-box) | **B** | **verified on-box** (worked example below): short-conv + GQA hybrid |
| **GLM-4.5 / 4.6** | `glm4_moe` | `glm4moe` | unsloth/bartowski `GLM-4.5-GGUF` | **B** | `dump.sh`; validates partial-RoPE 0.5, QKV-bias, sigmoid MoE + shared |
| **Hunyuan-A13B** | `hunyuan` | `hunyuan-moe` | tencent + unsloth GGUF | **B** | `dump.sh`; validates QK-norm + softmax MoE + shared_mlp |
| **Gemma4 / gemma3n** | `gemma4` | `gemma4`, `gemma3n` | google/unsloth `gemma-3n-*-GGUF` (3n); gemma4 as released | **B** | `dump.sh`; validates AltUp/LAuReL/PLE/MatFormer residual mixing |
| **DeepSeek-V3** | `deepseek_v3` | `deepseek2` | unsloth `DeepSeek-V3-GGUF` | **B** | `dump.sh`; validates MLA latent KV + fine MoE + shared (the fp16 MLA path) |
| **DeepSeek-V4** | `deepseek` | *(none yet)* | — | **B\*** | no `deepseek3/4` arch in llama.cpp yet; validate V3 as the MLA proxy until upstreamed |
| **MiniMax-Text-01** (lightning) | `minimax_text_01` | *(none — see note)* | — | **C\*** | llama.cpp's `minimax-m2` is a **different** (MoE, non-lightning) model; the lightning MiniMax-Text-01 has no llama.cpp graph → needs a torch/HF oracle |
| **DiffusionGemma-26B-A4B** | `diffusion_gemma` | *(none — block-diffusion decoder)* | — | **C** | non-autoregressive canvas decoder; not a llama.cpp graph. **Owned by another agent.** |
| **Qwen2.5-VL / Qwen2-VL / LLaVA** | `qwen2_5_vl` … | (`qwen3vl` exists for text tower only) | vision GGUFs partial | **C** | vision encoder + projector are not in the text decoder graph; validate the LLM tower separately if desired |
| Generative multimodal (Z-Image, Qwen-Image, LTX, Wan, Qwen3-TTS) | (diffusion pipeline) | — | — | **C** | image/video/audio DiTs; reuse the superl8 GEMM but need a diffusion/DiT oracle |

`B\*` / `C\*` = qualified: eligible in principle but the *specific* checkpoint has
no matching llama.cpp graph today; the "how to validate" column states the proxy.

## Summary

- **A (already covered):** Qwen3.5/3.6, Qwen3-Next — the #105 graph family.
- **B (eligible & would benefit, turnkey):** Qwen3 dense, Qwen3-MoE, Gemma3, LFM2/
  LFM2-MoE, GLM-4.5/4.6, Hunyuan-A13B, Gemma4/gemma3n, DeepSeek-V3. Every one of
  these is a family where a per-model "is our fp reference trustworthy?" fight has
  happened or would; each now has a from-scratch fp32 oracle for the cost of one
  `dump.sh` against a public GGUF.
- **C (need a different oracle):** MiniMax-Text-01 lightning (no llama.cpp arch —
  `minimax-m2` ≠ Text-01), DeepSeek-V4 (not upstreamed; use V3 as MLA proxy),
  DiffusionGemma (block-diffusion decoder, other agent), VLMs, and all generative
  multimodal DiTs.

## Worked example (bucket B, verified on-box)

`LFM2.5-1.2B-Nova` (`lfm2`, Q8_0), CPU fp32, prompt `"The capital of France is
Paris."` (7 tokens) → **441 activation tensors** dumped, arch auto-detected from
the GGUF (short-conv + GQA hybrid — a completely different graph than Qwen3.5),
proving the dumper is arch-agnostic with zero code changes.

For a **cross-oracle** number with first-divergence, the Qwen3.5-9B (`qwen35`,
bucket A) run: 217 tensors, re-dump determinism = **cos 1.000000 / relL1 0.0**
everywhere; a candidate with 5%·σ noise injected at `__fattn__-15` →
`reconcile.py compare` reports **`FIRST DIVERGENCE: layer 15`** (cos 0.998766)
while every other attention layer stays at cos 1.000000. See
`tools/reference_oracle/README.md`.
