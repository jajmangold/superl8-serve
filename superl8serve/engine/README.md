# engine/ — serving loop (port target)

These modules implement the request lifecycle. The design follows nano-vllm (MIT);
port them here, re-implemented (not vendored verbatim), swapping the compute path to
the `superl8serve/layers` seams.

Port checklist:

- [x] `sequence.py`      — request/sequence state, token ids, block table
- [ ] `block_manager.py` — paged-KV block allocation; here the blocks hold **int8** KV
      (+ per-block scales) instead of fp16 — half the cache footprint
- [x] `scheduler.py`     — continuous batching: admit/prefill/decode scheduling, preemption
- [x] `model_runner.py`  — builds the model from a `.superl8` checkpoint (via
      `superl8serve.load_superl8_checkpoint`), runs prefill/decode through `layers/`
- [x] `llm_engine.py`    — top-level `generate()` loop + sampler

Fleet-specific deltas vs nano-vllm:
- KV cache is int8 (block_manager sizes for int8 + scales).
- Paged decode needs `superl8`'s decode kernel to accept block tables (kernel TODO).
- Multi-GPU is PP + MoE-EP with `superl8.transport`, not TP (config forbids TP>1).
- Weights load zero-transform from `.superl8` (no GGUF/safetensors dequant).
