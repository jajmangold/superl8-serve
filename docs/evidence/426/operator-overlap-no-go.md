# superl8-serve#426 — cross-request DP4A/GDN overlap no-go

Date: 2026-08-16

## Decision

Do not add a two-stream scheduler route in this issue. The isolated kernels can
overlap, but the current engine has no request-safe operator-stage boundary at
which to issue request A's DP4A projection concurrently with request B's GDN.
Adding that boundary is scheduler/model surgery, not a benchmark-harness change.

The production route remains serialized/batched. No image rebuild is warranted.

## Measured ceiling and current whole-engine evidence

- Pinned fleet microbenchmark: TQ3 prefill x20 plus exact fp32 GDN x2 improved
  from 71.014 ms serial to 47.608 ms on independent streams (1.492x). The
  half2 GDN variant reached 1.452x but was 4.5% slower standalone, so exact fp32
  remains the only eligible GDN path.
- superl8#300 whole-model profiling measured decode at about 31% GPU busy, with
  roughly 21.4 ms GPU work and 47 ms host/launch gaps per token. That makes
  launch reduction valuable, but does not prove that two shared-model request
  forwards can execute concurrently without state races.
- The matched control remains superl8#299: Qwen3.8, K8V8, prompt 512, chunked
  prefill 256, eager, approximately 7.36 decode tok/s on a V100-labelled fleet
  card.

The 1.492x result is therefore an arithmetic ceiling, not an engine speedup.

## Structural blocker

The scheduler returns one phase per iteration:

- a newly admitted prefill batch with `is_prefill=True`, or
- the complete running decode batch with `is_prefill=False`.

It cannot return a prefill batch and a decode batch together. `ModelRunner`
then invokes the entire model once for that phase. Qwen3.8's model forward loops
through all 64 layers, and each decoder layer executes its attention/GDN path
and MLP without yielding an operator-stage handle to the scheduler.

Concurrent calls on two CUDA streams are not a safe harness-only shortcut:
`ModelRunner._decode_eager` calls `lin_cache.bind(slots)` before the whole-model
forward. The recurrent cache's bound rows are shared mutable engine state.
Launching another request forward before the first finishes can rebind that
state while layers are still gathering/scattering it. KV slots are distinct,
but the bind operation and the model's layer traversal are not request-local
execution objects.

Consequently, an engine A/B would first need all of the following production
interfaces:

1. mixed prefill/decode admission from the scheduler;
2. resumable per-layer or per-operator model execution;
3. explicit request-local recurrent bindings instead of mutable `bind(slots)`;
4. dependency events connecting each request's projection, GDN, KV, conv tail,
   and residual state;
5. tail-latency and memory backpressure policy.

Those changes exceed #426's prototype-before-scheduler-surgery fence. Building
them merely to run the acceptance benchmark would make the experiment the
production implementation it is supposed to falsify.

## Research disposition

- SGLang exposes mixed chunked prefill and overlap schedulers, but those are
  explicit scheduler architectures rather than transparent stream wrappers:
  https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md
- vLLM's single-instance prefill/decode disaggregation proposal likewise adds
  process/stream scheduling and layer pausing; it is not a drop-in CUDA-stream
  switch: https://github.com/vllm-project/vllm/issues/27093
- CUDA permits concurrent kernels only on independent streams with available
  resources; it does not establish state safety or guaranteed overlap:
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#concurrent-kernel-execution

## Closure criteria

Close #426 as a scoped no-go. Preserve the 1.492x kernel ceiling as evidence for
a future mixed-phase execution design, but do not claim aggregate throughput,
prefill latency, or decode-p95 improvements because no safe engine route exists
to measure. Any future attempt must begin with a separate red-first execution-
stage API issue and retain the original >=10% aggregate / <=5% decode-p95 gate.
