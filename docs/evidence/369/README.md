# Issue 369 — bounded idle-start coalescing

## Design

The opt-in bulk policy collects generation requests only on an idle-to-active
transition. The worker uses one absolute monotonic deadline, processes every inbox
item in FIFO order on the existing engine-owner thread, and stops collecting when
the configured generation target is reached. Once a decode is active, the loop does
not wait for arrivals. The default (`idle_coalesce_ms=0`) takes the original path.

CLI validation happens before tokenizer or model loading. The resolved target is
bounded by the positive `max_num_seqs` scheduler capacity. Telemetry records the
configured policy, collection events and wait time, effective first-step batch,
target/deadline outcomes, and p50/p95 TTFT.

## Correctness validation

- 106 focused and affected tests passed in the repository's Python 3.12
  `superl8-built:sm70` image: idle coalescing, metrics, API hardening, API, and batch
  endpoints.
- Ruff, formatting, and `git diff --check` passed.
- Tests cover default-off behavior, target and deadline exits, three-step active
  decode without an added wait, FIFO cancellation during collection, capacity and
  type validation before model load, batching telemetry, and TTFT percentiles.
- The first full CI attempt exposed worker-thread leakage from the new concurrent
  tests: pytest completed locally but remained in interpreter teardown until the
  20-minute runner limit. `EngineWorker.close()` now joins a drained owner thread,
  the test fixture closes every worker, and the same affected suite exits normally
  in 16.27 seconds.

## AMD0 adversarial review

The bounded Qwen3.5 review reported four candidates. Source verification accepted
one: `resolve_idle_coalescing` now rejects a non-positive or non-integer
`max_num_seqs` even when the optional policy is disabled, before model loading. The
other three were not source-supported: the only wait is bounded and precedes the
first decode step; `_coalesce_idle_start` passes encode and cancel items to
`_intake`; and TTFT percentiles are intentionally in latency telemetry while the
batching section contains policy/outcome telemetry.

The final lifecycle diff received a second bounded review. Its two candidates were
not source-supported: a timeout exception is the fail-closed outcome if a supposedly
drained worker cannot join and does not abandon the thread or pending state; and the
coalescing inbox read is bounded by the remaining absolute deadline while `_STOP`
wakes it immediately and is preserved for the next idle iteration.

## Performance qualification

Qualified on GPU1 (`GPU-299f91bd-13ff-f86f-9a35-959c8efe6af6`, CMP 100-210,
120 W) with the exact LFM2.5 Q6_K checkpoint and a forced resident-W8 runtime:

- superl8 `d7b4785482d523ed914867775647502f0e34915d`
- superl8-serve `f8318e8f89497642d6acf7adbce567dbcb70a23c`
- image `sha256:6bb651d217f7f982f25a3dcb52f10b022096160e977acf2719caccfcad0a999b`
- B512 graphs, `max_len=256`, 2,000 ms idle deadline, target 512

The fixed 512-request, 192-token HTTP run completed 98,304 tokens in 59.62 s at
1,648.8 tok/s with zero errors. It reached an effective batch of 512; live decode
telemetry reached 2,128.2 tok/s. Post-soak p50/p95 TTFT were 6,036/7,881 ms.

Seven consecutive B512 cycles completed 688,128 tokens in 408.62 s at a weighted
1,684.0 tok/s (1,669.4–1,719.1), with zero errors. Across 417 one-second samples,
HBM peaked at 9,835 MiB and temperature at 65 C. The final 120 samples rose only
0.67 C (63.15 to 63.82 C half means), passing the thermal-stability gate. The
candidate ended running with zero restarts/OOMs; the protected GPU9 endpoint stayed
reachable. Runtime receipts are under
`runtime/qualification/infra-582-lfm25/w8-2000ms/`.
