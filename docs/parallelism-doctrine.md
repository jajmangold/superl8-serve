# Parallelism doctrine for the CMP 100-210 / Volta GV100 fleet

This document formalizes the parallelism strategy for the fleet hardware used by
**superl8-serve**. It is the public companion to `superl8serve/scheduling.py` (the
server's topology-aware scheduler default) and the `superl8` transport-compression
design.  Issues tracked under epic #60 (scale-out) and #84 (pipeline parallelism).

## Hardware baseline

| Property           | Value                        |
| ------------------ | ---------------------------- |
| Silicon            | GV100 (Volta)                |
| HBM2               | 16 GB at **829 GB/s**       |
| Interconnect       | PCIe 1.0 x1 at **~250 MB/s** |
| Bandwidth ratio    | **~3,300:1** (HBM / wire)   |
| fp16 tensor cores  | firmware-limited to ~6.9 TFLOP/s |
| int8 `__dp4a`       | healthy at ~46 TOP/s         |

The 3,300:1 bandwidth gap is the dominant design constraint.  Moving a single fp16
hidden vector (say 4096 elements = 8 KB) across PCIe takes ~32 µs — roughly the
time budget for **an entire 28-layer decode step** on a single GPU.  Every
cross-GPU collective per layer is a non-starter.

## What works on this wire

| Strategy           | Viable? | Rationale |
| ------------------ | ------- | --------- |
| **Pipeline Parallelism (PP)** | Yes, shallow (2–4 way) | One hidden-state transfer per PP boundary *per token* (not per layer).  PP is O(tokens) not O(tokens × layers). |
| **Stage-local Expert Parallelism (EP)** | Yes, co-located with PP | Experts sharded within a PP stage stay on the local HBM — no expert dispatch crosses the wire. |
| **Data-parallel replication** | Yes, with independent engines | Embarrassingly parallel at the request level; zero wire budget. |

## What does NOT work on this wire

| Strategy           | Why it fails |
| ------------------ | ------------ |
| **Tensor Parallelism (TP)** | Every layer induces a collective (all-reduce / all-gather) per token. On a 28-layer model at 250 MB/s, even a single micro-step adds hundreds of ms — seconds per token. |
| **FSDP / ZeRO / full-shard** | Every layer all-gathers weights per step over the wire. Same scaling problem as TP. |
| **Deep PP (>4 way)** | Each additional PP boundary adds one hidden-state transfer per token. At 250 MB/s, deeper pipelines quickly become wire-bound, not compute-bound. |
| **Global (cross-rack) EP** | Expert dispatch over the wire costs as much as a full TP step for every MoE layer.  Keep experts local to each PP stage. |

## Recommended layout by GPU count (16 GPUs)

The principle: **replicate small PP groups first, then apply stage-local EP within each group.**  Never go above 4-way PP; never use TP.

### Dense models (no experts)

```
GPUs   Layout
────   ──────
 1     1× replicate
 2     2-way PP
 4     4-way PP
 8     2× replicate of 4-way PP   (4 GPUs each, 2 independent pipeline groups)
16     4× replicate of 4-way PP   (4 independent pipeline groups)
```

### MoE models (with experts)

```
GPUs   Layout
────   ──────
 1     1× replicate
 2     2-way PP, experts local to each stage
 4     2-way PP × 2-way stage-local EP
 8     4-way PP × 2-way stage-local EP
16     2× replicate of (4-way PP × 2-way stage-local EP)   (2 groups of 8 GPUs)
```

For 16 GPUs with MoE, the preferred layout is **two independent 8-GPU groups**,
each internally running 4-way PP with experts sharded 2 ways within each PP
stage.  This gives 2× the throughput of a single group, at zero wire cost for
the replication dimension, and only 3 PP-boundary transfers per token within
each group.

## Estimator

`superl8serve.scheduling.plan_parallelism(cfg, num_gpus)` produces a
`ParallelismStrategy` with the recommended layout.

`superl8serve.scheduling.estimate_throughput(strategy, cfg)` gives a
topology-agnostic symbolic estimate of tokens/s, p99 latency, and on-wire bytes
per token.

## Caveat: this is fleet-specific

This doctrine is tuned for **PCIe 1.0 x1 at ~250 MB/s**.  It does **not**
transfer to a real V100 box with NVLink (which can comfortably run TP=2 or even
TP=4).  On NVLink hardware (~300 GB/s bidirectional), the bandwidth ratio
drops from 3,300:1 to roughly 3:1, making the entire tradeoff space different
— TP and FSDP become competitive, and deeper PP becomes viable if it helps
balance load or fit a model in HBM.

## References

- [superl8 transport-compression.md](https://github.com/jajmangold/superl8) — the
  compressed-activation layer that makes PP transfers viable on the 250 MB/s link
- [PipeDream](https://arxiv.org/abs/1806.03377) — the original pipeline
  parallelism scheduler
- [DeepSpeed Pipeline Parallelism](https://www.deepspeed.ai/tutorials/pipeline/)
- [MegaBlocks](https://arxiv.org/abs/2211.15841) — efficient MoE dispatch
- [DeepEP](https://github.com/deepseek-ai/DeepEP) — expert-parallel
  communication library from DeepSeek
