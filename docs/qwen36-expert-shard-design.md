# Qwen3.6-35B-A3B expert-sharded layer-queue qualification (superl8-serve#389)

## Scope

Qualify the weight-stationary / expert-sharded path (superl8#318–#324, #330/#331) on the first
real production-shaped MoE for the engine: **Qwen3.6-35B-A3B** — 256 experts, top-8/token,
40 hybrid GatedDeltaNet/full-attention layers, shared expert.

This branch adds the builder wiring so Qwen3.6 MoE layers route through the existing seams:

| config | MoE path | notes |
|---|---|---|
| (default) | `SparseMoE` | today's resident behavior, unchanged |
| `use_weight_stationary_moe` | `WeightStationaryMoE` | Phase 3: per-expert staging buffers + skip-empty |
| `expert_to_gpu` + `local_gpu` | `TransportMoELayer` | Phase 5: compressed cross-GPU expert routing |

All-local `expert_to_gpu` maps (every expert owned by this process) must stay parity-gated vs
`SparseMoE` (cos > 0.99), per the existing transport contract (`tests/test_moe_transport.py`).
The new builder tests enforce that gate through the full model build path.

## Server surface

`python3 -m superl8serve.api.server` gains:

- `--weight-stationary` — enable the weight-stationary MoE path
- `--expert-map <json>` — `{"expert_id": gpu_index, ...}`; enables `TransportMoELayer`
- `--local-gpu <n>` — owning GPU index for this process in the shard

These are serve-time topology overrides applied via `dataclasses.replace` at the API seam;
they are never persisted back into the `.superl8` meta.

## Model assets (on disk, no download)

- superl8 FP8 b8: `/path/to/weights/Qwen__Qwen3.6-35B-A3B-FP8.b8.superl8`
  (38,100,485,888 B; forge MANIFEST `done`, bits=8). NVMe copy:
  `/path/to/storage
- GGUF TQ3_4S (single-card baseline):
  `/path/to/models
- Tokenizer staged: `/path/to/storage

## Card math and topology

Five V100s (GPU4, 7, 9, 11, 14) and twelve CMP 100-210s report 16 GiB usable. FP8 b8 = 38.1 GB:
- 1 card: only the 14 GB TQ3_4S GGUF fits (llama.cpp lane, port 8017).
- 3 cards: 38.1/3 ≈ 12.7 GB weights/card, ~3 GB/card for KV + activations + graphs.
  Expert footprint is ~6.3 MB/expert int8; 256 ≈ 1.6 GB — distributes trivially.

**Topology (verified 2026-08-12)**: GPUs 8, 9, 10, 11 are **all PIX on one switch** and
all idle. GPU9 and GPU11 are V100s; GPU8 and GPU10 are CMP 100-210. That gives a
**same-switch 3-card shard with zero PHB hops**:

- **Primary shard: GPU9 (V100) + GPU8 + GPU10 (CMP)** — every leg is same-switch P2P
  (validated; the #356 zero-fill issue is specifically cross-PHB, e.g. 6→14). The V100
  member is what we need for ncu/nsight kernel profiling; the CMP members are the
  production fleet target.
- **Fallback shard: GPU9 + GPU11 (both V100) + GPU8** — also same-switch, two V100s.
- A mixed or all-V100 shard crossing switches (e.g. including GPU14) is NOT required and
  would need the #356 topology-safe host-staging fallback for its PHB leg.

So the 3-card qualification can proceed **without depending on #356** by staying on the
GPU8/9/10/11 switch.

## Qualification ladder

1. Baseline: 1-card TQ3_4S GGUF via llama.cpp (existing `docker/qwen35-tq3/`), record tok/s + TTFT.
2. Single-GPU superl8 weight-stationary on GPU9: correctness (cos/SQNR vs fp32 oracle),
   bounded batch decode, ncu/nsight profile of expert GEMMs.
3. 3-card expert shard **GPU9 (V100) + GPU8 + GPU10 (CMP, same switch)**: `--expert-map`
   split, compressed send/recv on validated same-switch P2P legs. Measure per-GPU profiles
   + wire budget. ncu/nsight runs on the GPU9 member.

## Compose

`docker/qwen36-moe/` — thin superl8-serve layer over `superl8-built:sm70` (the lfm25-b512 resident
pattern), dev/qualification only, pinned to V100 GPU9
(`GPU-7540b3cb-86d0-be61-e72f-525f7e051444`), loopback `:18089`, single-GPU
`--weight-stationary` lane. The 3-card command is documented in the compose.

## Gates

- `ruff` clean on changed files.
- `tests/test_more_models.py -k qwen3_next` (incl. the two new builder tests), plus
  `tests/test_moe_transport.py` + `tests/test_model_config.py`: 43 pass in `superl8-built:sm70`.
- All-local transport parity (cos > 0.99 vs `SparseMoE`) enforced by builder test.
- Trailmark diff + AMD0 adversarial review before merge.
