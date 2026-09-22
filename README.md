# SuperL8 Serve

OpenAI-compatible INT8 (W8A8) inference server for quantized LLMs.

## Features

- **INT8 dp4a compute** -- W8A8 quantized inference using INT8 `__dp4a` kernels, optimized for hardware where INT8 throughput exceeds fp16
- **OpenAI-compatible API** -- drop-in replacement for `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/rerank`
- **Continuous batching** -- dynamic request scheduling with configurable idle coalescing
- **CUDA graph capture** -- graphed decode for consistent low-latency throughput
- **Paged KV cache** -- memory-efficient attention with configurable cache formats (int8, k8v3, k8v8)
- **Speculative decode** -- n-gram cascade + MTP head drafters for net decode speedup
- **Structured outputs** -- JSON schema and grammar-constrained generation via XGrammar
- **Tool calls** -- native `tool_choice="auto"` / `"required"` with Hermes/Qwen and LFM2 parsers
- **GGUF native loading** -- load GGUF k-quant checkpoints directly, preserving supported quant types
- **HF conversion** -- convert any HuggingFace checkpoint to `.superl8` format
- **Multi-GPU** -- pipeline parallelism and MoE-expert parallelism via `superl8.transport`
- **VLM support** -- Qwen3.5-VL image inputs via the OpenAI vision API

## Quick start

```bash
pip install -e ".[serve]"
python -m superl8serve.api.server --model /path/to/your-model.superl8 --tokenizer Qwen/Qwen3-8B
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

## Development

```bash
pip install -e ".[dev,convert,serve,structured]"
ruff check .
pytest
```

Set `SUPERL8_WEIGHTS_DIR` to point at a directory containing model weight files for tests that require real checkpoints.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
