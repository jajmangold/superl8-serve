<!-- SPDX-License-Identifier: MIT -->
# LLM zoo benchmarks

Output of `tools/bench_llm.py` (see `tools/bench.sh` for the containerized, GPU-pinned
wrapper): one `<model>.json` per checkpoint plus an aggregated `results.md` table with
prefill/decode tok/s, time-to-first-token, and peak VRAM for each `.superl8` LLM run
through `LLMEngine`.

Tracking issue: #17.

## Status

No numbers have been recorded here yet — see issue #17 for why (this Claude Code
session's shell was sandboxed away from the GPU/mount/`docker exec` access needed to
drive a real run; `docker ps` did confirm the fleet is real and busy with forge/quant
jobs). The harness is ready to run; a follow-up pass with GPU + mount + `docker exec`
access (or a human running `tools/bench.sh` directly) should populate this directory.

## Known scope gap: no fp16 baseline

superl8-serve doesn't have an fp16 execution path today — every quantizable linear is
routed through `to_qtensor` (`superl8serve/models/weights.py`), int8-only. This harness
can only report a static weight-quantization SQNR against the original HF checkpoint
(`--hf-dir`), not a decode-time int8-vs-fp16 latency/perplexity delta. That needs a
second reference implementation (e.g. a `transformers` fp16 forward) and isn't
implemented here.

## Which models are actually unblocked right now

Per `superl8serve/models/COVERAGE.md`'s family-glue status (full prefill **and** decode
through the int8 dp4a kernels, which is what `LLMEngine.generate()` needs):

- **Qwen3 dense** (e.g. `Qwen3-8B`) — ✅ ready
- **Qwen3-MoE** (e.g. `Qwen3-30B-A3B`) — ✅ ready
- **Gemma3** (text) — ✅ ready
- **DeepSeek-V3** — ✅ decode works, but MLA attention is a fp16 fallback (Track-2 int8
  kernel not landed) — note this caveat if benchmarked
- **Gemma4 family** (`gemma-4-*`) — 🟧 scaffold only (AltUp/LAuReL/PLE/MatFormer not
  modeled yet); a `.superl8` existing for these on the archive does **not** mean the
  family's glue is done — stays blocked regardless
- GLM-4.5, Hunyuan, LFM2, MiniMax, Qwen3-Next — registered and prefill-tested, but
  decode needs recurrent/latent-state caching not yet wired into the engine; not
  runnable through `LLMEngine.generate()` yet
