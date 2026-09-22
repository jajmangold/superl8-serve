# P2P route validation evidence — superl8-serve#356

## Summary

The superl8 transport previously trusted `torch.cuda.can_device_access_peer()` alone
to decide direct P2P (`cudaMemcpyPeer`). On the CMP 100-210 fleet that probe is a
false positive for cross-PHB routes: physical GPU 6 → 14 reports peer support yet
the device-to-device copy silently zero-fills the destination (both fp16 and int8
payloads). This change makes route selection fail-closed via an explicit validated
pair set, so:

- direct same-switch P2P remains available **only** for explicitly validated pairs;
- every unvalidated / unknown / same-device route uses pinned host staging;
- chained/multi-hop decisions are deterministic per-boundary.

## Reference material

- NVIDIA Data Center GPU Driver Release Notes 570.211.01 ("Disable GPU initiated RO
  traffic on Ada Lovelace and older GPUs"): documents that P2P over PCIe can silently
  corrupt on platforms that do not guarantee posted-transaction ordering even when
  `cudaDeviceCanAccessPeer` returns true — the exact 6→14 symptom.
- CUDA Programming Guide §3.4.2.5: bare-metal Linux P2P requires IOMMU disabled to
  prevent silent device memory corruption.
- PyTorch issue #84803: cross-GPU `.to()` silently returning zeroed tensors.

## Policy (`superl8serve/dist/peer_routes.py`)

A route may use P2P only when **all** of:

1. `src != dst`;
2. the unordered pair `{src, dst}` is in the validated set; and
3. `torch.cuda.can_device_access_peer(src, dst)` is true.

The validated set is configuration (`SUPERL8_P2P_VALIDATED_PAIRS`), never a runtime
discovery sweep. An empty/unset value disables all cross-GPU P2P (fail closed).
Selection is pure and deterministic (`select_route`), so chained/multi-hop
pipelines resolve each boundary independently and reproducibly.

**Ordinal contract (important):** the env values are the **torch CUDA device
ordinals exactly as seen by the serving process** (`tensor.device.index` /
`stage._device.index`), NOT physical `nvidia-smi` host indices. Physical
indices/UUIDs below are qualification provenance only. A container launched with
`--gpus '"device=5,6"'` maps physical 5/6 to ordinals 0/1, so a serving process
in that shape must set `SUPERL8_P2P_VALIDATED_PAIRS=0,1` (the qualification pair
here). An all-visible process that keeps the physical indices may set
`SUPERL8_P2P_VALIDATED_PAIRS=5,6`. Misreading the two is the exact confusion this
fix guards against.

## Live qualification evidence

### Same-switch pair: physical GPU 5 → 6 (PIX, free, CMP 100-210)

Container invocation (exposes only GPUs 5,6 → container ordinals 0,1):

```
docker run --rm -e PYTHONPATH="$PWD" -v "$PWD:$PWD" -w "$PWD" \
  --gpus '"device=5,6"' superl8-built:sm70 \
  python3 tools/qualify_peer_route.py --src 0 --dst 1 \
    --physical-src 5 --physical-dst 6 --json docs/evidence/356/gpu5-6-qualification.json
```

Result (`exit=0`):

```json
{
  "container_src_ordinal": 0, "container_dst_ordinal": 1,
  "physical_src": 5, "physical_dst": 6,
  "container_visible_uuids": ["GPU-68309a3d-8ef1-8660-8634-6953b027b34a",
                              "GPU-44c4fe09-a616-cf30-6a47-b86bba89f923"],
  "raw_can_device_access_peer": true,
  "fp16_roundtrip_bit_exact": true,
  "payload_nonzero": true,
  "verdict": "p2p-safe"
}
```

- `physical_src_uuid` = `GPU-68309a3d…` (GPU5), `physical_dst_uuid` = `GPU-44c4fe09…` (GPU6).
- Both cards reported 7 MiB used before/after the probe; GPU9 (superl8 LFM2.5) and GPU14
  (Z-Image) were never exposed or touched. `nvidia-smi` host indices 5/6 stayed free.

### Failing topology: physical GPU 6 → 14 (PHB)

Not re-probed (GPU14 hosts the live Z-Image service). Prior evidence from the
superl8-serve#354 validation is used as the immutable failing case:

- `handle.used_p2p=true` (capability false positive), destination payload `absmax=0`,
  reconstructed activation `absmax=0` for both fp16 and int8;
- `nvidia-smi topo -m` reports 6/14 as `PHB` while 5/6/7 are `PIX`.

Under the new policy this route resolves deterministically to host staging: the pair
is absent from the validated set, so `select_route(6,14,validated=…)` returns
`not-validated` regardless of the capability probe.

## Deterministic test coverage

- `tests/test_peer_routes.py` (40 tests, CPU-only): parser (empty/single/multiple/
  reversed/whitespace/malformed/same-device), pure selection (same-device,
  unvalidated-with-true-capability regression, validated+true, validated+false,
  unknown capability fails closed), chained/multi-hop (5→6→7 validated, partially
  validated chains, 3-stage cross-switch), env configuration, capability wrapper
  error collapse.
- `tests/test_qualify_peer_route.py` (10 tests, CPU-only): NaN-free deterministic
  probe pattern, forbidden-GPU guard (9/14 refused).
- `tests/test_transport.py` multi-GPU class now marks the test pair `0,1` validated
  explicitly; added `TestFailClosedUnvalidatedPair` proving a real cross-GPU send
  host-stages (and stays bit-exact) when the pair is not validated, and that
  validating the pair re-enables P2P.
- `tests/test_pipeline.py` marks its boundary pairs `0,1;1,2` validated so the
  chained pipeline keeps exercising P2P where reported.

### Focused runs

- `tests/test_peer_routes.py tests/test_qualify_peer_route.py` (CPU lane): 50 passed.
- `tests/test_transport.py tests/test_peer_routes.py` on `--gpus '"device=5,6"'`: 65 passed.
- `tests/test_pipeline.py` on `--gpus '"device=5,6"'`: 24 passed, 3 skipped (three-GPU
  tests need ≥3 GPUs; run in isolated CI, not locally).
- Ruff clean on all changed Python; `git diff --check` clean.
- Trailmark: additive only — 2159→2214 nodes, 1744→1788 functions, 240→248 classes,
  13428→13599 edges, entrypoints +1 (`tools.qualify_peer_route:main`). No removed
  nodes or entrypoints.

### Local all-GPU suite exclusion

The repo light lane uses `--gpus all`, but GPU9/GPU14 and most fleet cards are
occupied by live services, and repo AGENTS.md forbids using the pinned live-server
GPU. The local all-GPU run was therefore **not** used as evidence and the one
accidentally-started all-GPU container (exact ID `ce1cd490…2045b4`) was stopped and
verified auto-removed; GPU9/14 resident HBM unchanged (9449 / 6739 MiB). Three-GPU
and all-GPU coverage runs in the governed isolated CI runner.

## AMD0 review (bounded)

`amd0-judge.py review` returned a `fail` verdict with four findings. Overseer
verification against the exact source:

| Finding | Verdict | Disposition |
|---|---|---|
| 1 (HIGH): `_can_use_p2p` "ignores the validated set" | **False positive** | `_can_use_p2p` returns `peer_route_allowed(...)` directly; P2P is gated by validated-set membership. Capability is required but never sufficient; unknown capability and probe exceptions resolve to `no-peer-capability`, never P2P. |
| 2 (MEDIUM): forbidden-GPU guard not exercised on CPU-only hosts (device-count check precedes it) | **Real, minor** | Reordered `qualify_pair` to run the pure arg-based forbidden guard before the device-count check; probe tests now exercise it deterministically. |
| 3 (MEDIUM): pattern test doesn't verify bit-exactness | **False positive** | The offline probe (not the CPU unit test) compares the direct `x.to(dst, copy=True)` payload; live GPU5→6 evidence records `fp16_roundtrip_bit_exact: true`. Unit test only pins pattern properties, as designed. |
| 4 (LOW): no 5→6→7 chain coverage | **False positive** | `TestChainedMultiHop` covers 5→6→7 (all-validated) and partially-validated chains including the 6→14 PHB hop host-staging. |

All findings verified against source; no other issue confirmed.

## Rollback

- Code revert: revert the PR (`git revert` of the merge / `fix/356-topology-safe-p2p`
  branch deletion leaves main unchanged).
- Behavior revert: unset `SUPERL8_P2P_VALIDATED_PAIRS` (or set it to `""`) to restore
  fully host-staged transport; P2P is only ever enabled for explicitly listed pairs.
  The previous (unsafe) behavior — trusting `can_device_access_peer` alone — is not
  recoverable by config and is intentionally removed.
- No schema, weight, service, or container changes were introduced; nothing to drain.
