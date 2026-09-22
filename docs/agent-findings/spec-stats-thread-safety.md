# spec_stats Thread-Safety Audit

## Summary

**Conclusion**: `spec_stats` is **SAFE** under concurrent decode.

**Reason**: The design enforces single-threaded access through the `EngineWorker` pattern.

---

## Analysis

### Mutation Sites

The `spec_stats` dict is initialized and mutated in `superl8serve/engine/model_runner.py`:

```python
# Line 131: Initialization
self.spec_stats = {"steps": 0, "drafts": 0, "accepts": 0}

# Line 825-826: Telemetry during decode
self.spec_stats["drafts"] += actual_len[b]
self.spec_stats["accepts"] += min(n_draft_acc, n_acc - 1)

# Line 846: Step counter
self.spec_stats["steps"] += 1
```

All mutations occur during the decode step execution in the runner.

### Concurrency Model

The `EngineWorker` (in `superl8serve/api/runtime.py`) is the single owner of the `LLMEngine` and its `runner`:

1. **Single Worker Thread**: `EngineWorker` creates exactly one background thread (`_run()` method) that drains the request inbox
2. **Serialization Guarantee**: All calls to `engine.step()`, `engine.add_request()`, `engine.cancel()`, etc. are routed through this single worker thread
3. **No Direct Access**: No code path allows direct access to the engine or runner outside the worker thread

```python
# From superl8serve/api/runtime.py (docstring)
"""LLMEngine.step() and its Scheduler are plain Python state (a deque + a list),
not safe to call concurrently from multiple threads. One dedicated thread drains a
request queue and drives the scheduler, so concurrent HTTP requests still share
the engine's continuous-batching loop -- exactly what LLMEngine.generate() does
for a list of prompts, just fed incrementally over time..."""
```

### Enforcing the Invariant

The invariant is enforced by:

1. **Worker Thread Ownership**: `EngineWorker._run()` is the sole entry point for engine operations:
   ```python
   def _run(self) -\u2192 None:
       while True:
           # All engine operations happen here, serialized
           if self._pending:
               self.engine.step()
               self._dispatch()
   ```

2. **Queue-Based Request Handling**: HTTP requests arrive via `submit()` and are queued in `_inbox`. The worker thread processes them one-at-a-time.

3. **No Multi-Worker Per App**: The FastAPI app creates exactly one `EngineWorker`:
   ```python
   # From superl8serve/api/app.py
   worker = EngineWorker(
       engine,
       stats=stats,
       idle_coalesce_ms=idle_coalesce_ms,
       idle_coalesce_target=idle_coalesce_target,
   )
   ```

### Why It's Safe

The **single-threaded access invariant** makes `spec_stats` safe:

- Python's GIL guarantees atomicity for simple operations
- The dict mutations (`+=` operations) are safe when serialized by a single thread
- No lock is needed because there's only one thread accessing the data

The design trades potential parallelism for correctness and simplicity. The worker thread's sequential processing ensures that concurrent HTTP requests are serialized at the engine boundary.

---

## Verification

To verify the thread-safety invariant holds, add a test that:
1. Sends multiple concurrent requests
2. Verifies the engine only runs on one thread
3. Confirms spec_stats values are consistent

See: `tests/test_engine.py::test_mtp_spec_decode_bit_identical_concurrent_greedy()` for an existing concurrent test that exercises the engine with multiple sequences.

---

## Conclusion

**No bug found.** The `spec_stats` dict is safely mutated under concurrent decode because the `EngineWorker` enforces single-threaded access to the engine and runner.

The invariant that makes it safe:
- **Engine singleton**: One `LLMEngine` instance per serving process
- **Worker thread serialization**: One dedicated thread processes all engine operations
- **Queue-based admission**: HTTP requests are serialized through the worker's inbox queue

No changes are required.
