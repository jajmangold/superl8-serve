# superl8-serve test suite

## Test tiers

| Tier | Requires | Skip condition |
|---|---|---|
| CPU-only | Nothing extra | Always runs |
| CUDA-required | CUDA GPU | Skipped when `torch.cuda.is_available()` is False |
| GPU-weights-required | CUDA GPU + model checkpoints | Skipped when `SUPERL8_WEIGHTS_DIR` is unset |

## Running tests

```bash
# Full suite (CPU-only + CUDA tests)
pytest tests/ -ra

# CPU-only tests (no GPU, no weights)
pytest tests/ -k "not cuda" -ra

# CUDA tests only (needs GPU, skips weight-dependent tests unless SUPERL8_WEIGHTS_DIR is set)
pytest tests/ -m cuda -ra

# E2E serve smoke test (needs GPU + weights)
pytest tests/test_e2e_serve_smoke.py -ra
```

## Environment variables

| Variable | Purpose |
|---|---|
| `SUPERL8_WEIGHTS_DIR` | Path to directory containing model weight files. Tests that load real checkpoints use the `weights_dir` fixture, which skips the test if this is unset or not a directory. |

## Fake doubles pattern for CPU tests

Many CPU-only tests mock the CUDA-dependent layers with lightweight fakes:

- **`_i8(out, in_)`** — creates a fake `QTensor` with random int8 weights and per-row scales. Used to test checkpoint round-trips and `LinearW8A8` construction without real GEMM.
- **`_i4(out, in_, g)`** — same pattern for 4-bit grouped quantized weights.
- **Round-trip tests** (`test_smoke.py`) — save a fake checkpoint with `save_superl8()`, load it back with `load_superl8_checkpoint()`, and verify the weight tensor is intact and the `LinearW8A8` layer produces finite output.
- **Shard tests** — verify partial checkpoint loading via the `shard` parameter.

These tests prove the integration seams (loader, linear layer, config) work end-to-end without needing a GPU or real weights.

## Adding new tests

- CPU-only tests: no special requirements. Use `_i8`/`_i4` helpers for fake weights.
- CUDA tests: use `@pytest.mark.cuda` or `@pytest.mark.skipif(not torch.cuda.is_available(), ...)`.
- Weight-dependent tests: use the `weights_dir` fixture — it returns a `Path` and auto-skips if `SUPERL8_WEIGHTS_DIR` is unset.
