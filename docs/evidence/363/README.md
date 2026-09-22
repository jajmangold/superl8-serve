# superl8-serve #363 — disconnected-generation cancellation qualification

## Scope

This evidence qualifies cancellation of an OpenAI-compatible streaming request after
the client disconnects. The candidate is based on merged `origin/main` at `9374102`
(including the #351 batched recurrent-prefill work) and uses the existing
`superl8-serve-ggufspec:latest` image without a rebuild.

The live model was `LiquidAI_LFM2.5-2.6B-Q6_K.gguf` (SHA-256
`499c120820935273c5eec587333ce18e0d4369911bd795b537f041ab4b20052f`) on exact
GPU5 UUID `GPU-68309a3d-8ef1-8660-8634-6953b027b34a`, `max_num_seqs=128`,
`max_len=1024`, whole-step CUDA graphs, and the same image-matched superl8 #242 Python
overlay used by #351.

## Result

- A real TCP client closed immediately after the first SSE event of an 800-token
  completion. The server reached `running=0`, `waiting=0`, `inflight=0`, and
  `finish_reasons.cancelled=1` in **0.5176 s**.
- Active paged-KV state was reclaimed. One block remained as the intentional reusable
  prompt-prefix cache; no scheduler slot or inflight request remained.
- A normal four-token completion succeeded immediately afterward, proving the worker
  thread survived and remained usable.
- B128 x 256 produced 32,768 tokens in 31.9393 s: **1,025.9 aggregate tok/s**, zero
  errors. This preserves the #351 qualification result of 1,020.9 tok/s.
- Mixed-load fairness completed a 256-token long request in 3.4953 s while four waves
  of 16 one-token jobs all completed; short-request p95 was 1.2494 s and there were
  zero errors.

Compact receipts are `final-cancellation.json`, `final-throughput.json`, and
`final-fairness.json`.

## Failure found during qualification

The first live disconnect reached the new worker cancellation path but exposed a
PyTorch lifecycle bug: CUDA-graph recurrent buffers are inference tensors, and
`RecurrentStateCache.clear_slot()` attempted an in-place `zero_()` outside inference
mode. That raised an exception and killed the engine worker. The final implementation
executes `clear_slot()` under `torch.inference_mode()` and includes a CPU regression
that reproduces the former exception with a static inference buffer.

Cancellation also refreshes the running/waiting gauges explicitly. No engine step
follows a cancellation, so relying on step telemetry left those operator-facing
gauges stale.

## Static validation

- Ruff: all changed production and test files pass.
- Pytest: **52 passed, 1 skipped** across API, batch, metrics, scheduler, and recurrent
  cache lifecycle suites.
- Trailmark: cancellation adds the expected worker, scheduler, engine, recurrent-cache,
  and queue-depth nodes; hot-path complexity only increases from 3 to 4 in worker
  intake/stream.
- AMD0 bounded review was completed. Its initial full-diff pass reported no findings.
  Later candidate findings were checked against source and rejected: `output_ids`
  belong to a forgotten Sequence object rather than reusable slot state; the FIFO
  worker inbox serializes completion/cancel; and the fake scheduler does remove the
  cancelled item before asserting `(0, 0)` depth.

GPU5 was stopped after qualification and returned to 7 MiB idle usage. GPUs 0/1, 9,
and 14 were not touched.
