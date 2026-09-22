# SuperL8 Serve

**An LLM inference server that doesn't give up on your GPU.**

Most inference servers assume you have Ampere or newer. SuperL8 Serve is built from the ground up for Volta and CMP hardware — using [SuperL8](https://github.com/jajmangold/superl8)'s DP4A INT8 kernels to get real throughput on cards everyone else skipped past.

Drop in a GGUF checkpoint, point it at a HuggingFace model, and serve an OpenAI-compatible API. No tensor cores required.

## Features

- **OpenAI-compatible API** — `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/rerank`. Drop-in replacement for your existing client code.
- **INT8 dp4a compute** — W8A8 quantized inference using SuperL8's `__dp4a` kernels. Optimized for hardware where INT8 throughput blows past fp16.
- **Continuous batching** — dynamic request scheduling with idle coalescing. No wasted GPU cycles between requests.
- **CUDA graph capture** — graphed decode for consistent low-latency throughput. No graph-break surprises.
- **Paged KV cache** — memory-efficient attention with int8, k8v3, and k8v8 cache formats.
- **Speculative decode** — n-gram cascade + MTP head drafters for net decode speedup.
- **Structured outputs** — JSON schema and grammar-constrained generation via XGrammar.
- **Tool calls** — native `tool_choice="auto"` / `"required"` with Hermes/Qwen and LFM2 parsers.
- **GGUF native loading** — load GGUF k-quant checkpoints directly (Q2_K through Q6_K). No conversion step.
- **HF conversion** — convert any HuggingFace checkpoint to `.superl8` format with `python -m superl8serve.convert`.
- **Multi-GPU** — pipeline parallelism and MoE-expert parallelism via `superl8.transport`.
- **VLM support** — Qwen3.5-VL image inputs via the OpenAI vision API.

## Install

```bash
pip install https://github.com/jajmangold/superl8-serve/releases/download/v0.1.0/superl8_serve-0.1.0-py3-none-any.whl
```

Requires [SuperL8](https://github.com/jajmangold/superl8) for the CUDA kernels.

## Quick start

```bash
python -m superl8serve.api.server \
  --model /path/to/your-model.superl8 \
  --tokenizer Qwen/Qwen3-8B
```

### Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
resp = client.chat.completions.create(
    model="my-model",
    messages=[{"role": "user", "content": "Explain quantum computing in one sentence."}],
)
print(resp.choices[0].message.content)
```

### Docker

```bash
docker build -f docker/Dockerfile.runtime -t superl8-serve:latest .
docker run --rm --gpus all -p 8000:8000 \
  -v /path/to/models:/models:ro \
  -e SUPERL8_MODEL=/models/your-model.superl8 \
  -e SUPERL8_TOKENIZER=/models/your-tokenizer \
  superl8-serve:latest
```

## Supported models

| Family | INT8 decode (end-to-end) |
| --- | --- |
| Qwen3 dense/MoE, Gemma3 | full prefill + decode |
| GLM-4.5/4.6, Hunyuan, LFM2 | INT8 decode (GQA) |
| Qwen3-Next / 3.5 / 3.6 (Gated-DeltaNet hybrid) | INT8 decode (GQA + DeltaNet kernel) |
| DeepSeek-V3/V4 (MLA) | INT8 absorb decode |
| MiniMax-Text (lightning) | softmax half decodes INT8 |

See `superl8serve/models/COVERAGE.md` for the full family matrix.

## Performance

Measured on V100-labelled CMP fleet hardware:

| Model | Prefill | Decode | VRAM |
|---|---|---|---|
| Qwen3-8B | 69.5 tok/s | 80.2 tok/s | 13.5 GiB |
| Qwen3.6-27B Q3_K_S | — | 21.3 tok/s | 14.3 GiB |
| Qwen3-0.6B | — | — | — |

See `bench/` for raw JSON benchmark data.

## Development

```bash
git clone https://github.com/jajmangold/superl8-serve.git
cd superl8-serve
pip install -e ".[dev,convert,serve,structured]"
ruff check .
pytest
```

Set `SUPERL8_WEIGHTS_DIR` to point at a directory containing model weight files for tests that require real checkpoints.

## Related repos

- [**SuperL8**](https://github.com/jajmangold/superl8) — the CUDA kernels that power this server. INT8 DP4A FlashAttention-2 and GEMM for Volta GPUs.
- [**ComfyUI-SuperL8**](https://github.com/jajmangold/ComfyUI-superl8) — ComfyUI nodes for INT8 quantized diffusion DiTs. Same kernels, different workload.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
