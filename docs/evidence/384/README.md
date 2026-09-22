# Issue 384 validation evidence

## Contract

- GGUF loads preserve supported native K-quants by default.
- `--gguf-force-w8` explicitly selects one-time K-quant to resident per-row W8
  transcoding for throughput-oriented deployments.
- The flag is rejected for non-GGUF checkpoints before checkpoint inspection.
- Startup telemetry identifies `checkpoint-default` or `forced-per-row-w8`.

## Automated validation

The affected loader, policy, and CUDA-bucket suites ran in the project test image:

```text
45 passed, 9 skipped in 13.20s
```

The skips are existing hardware/model-gated tests. Static validation also passed:

```text
ruff check --no-cache ...
All checks passed!
git diff --check
```

## AMD0 bounded review

AMD0 returned four candidates. Each was checked against the exact source:

1. **Rejected:** it claimed non-GGUF validation follows checkpoint loading. The
   `gguf_force_w8` guard is immediately before `checkpoint_info`, so the contract
   is fail-before-load.
2. **Rejected:** it claimed `getattr(engine, "weight_runtime", ...)` in
   `build_banner` mutates or duplicates the attribute. `getattr` only reads it.
   The final implementation assigns the runtime once in `load_gguf_engine`.
3. **Rejected as a defect:** an empty `native_types` tuple is the existing loader's
   documented selector for its dequantize-once plus `per_row_i8` path.
4. **Accepted as a validation observation:** the policy unit test proves selection
   of that existing path, not numerical transcoding itself. Existing loader tests
   cover the `per_row_i8` conversion, and infra issue 582 provides the required
   real-model, live-GPU throughput and output qualification before promotion.

No unattended review finding was accepted as merge authority.

## Structural diff

`trailmark diff origin/main . --repo "$PWD"` reported 4 added nodes, 8 modified
nodes, 17 added edges, and no removed edges. The only complexity changes are the
single explicit policy branch in `load_engine` (6 to 7) and selection/telemetry in
`load_gguf_engine` (16 to 18).
