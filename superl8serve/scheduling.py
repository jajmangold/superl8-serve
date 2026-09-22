# SPDX-License-Identifier: MIT
"""Fleet parallelism doctrine (issue #145): strategy planning, anti-pattern
warnings, and topology-agnostic throughput estimation for the CMP 100-210 fleet.

Hardware baseline: GV100 silicon, 16 GB HBM2 at 829 GB/s, PCIe 1.0 x1 at ~250 MB/s
(≈3,300x bandwidth ratio). This makes communication-minimizing strategy the
dominant design axis — kernel micro-optimization is secondary on this wire.

Doctrine (excerpt — full text at docs/parallelism-doctrine.md):

    Replicate small PP groups first (max 4-way PP), then apply stage-local EP
    within each PP group. TP and FSDP are anti-patterns on this fleet (every
    layer induces a collective over a ~250 MB/s link).

For 16 GPUs:
    - dense:  4x4-way PP (four replicas, each 4 GPUs deep)
    - MoE:    2x(4-way PP + stage-local EP)  (two replicas, each with 4 PP
              stages and experts sharded within each stage)

See also: epic #60 (scale-out), #84 (PP), and superl8 transport-compression.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .dist.transfer_model import bytes_per_element
from .models.config import ModelConfig

# ── Fleet-constant wire budget ─────────────────────────────────────────────
# HBM bandwidth (bytes/s) and PCIe-x1 bandwidth (bytes/s) — roughly 3,300:1.
_HBM_BW_BYTES = 829e9  # 829 GB/s
_PCIE_BW_BYTES = 250e6  # 250 MB/s (single lane, one direction)
_BW_RATIO = _HBM_BW_BYTES / _PCIE_BW_BYTES

# Max PP depth recommended before the wire budget dominates.
_MAX_PP_DEPTH = 4


# Per-token hidden-state bytes over one PP boundary, codec-aware.
def _activation_bytes_per_token(hidden_size: int, codec: str = "int8") -> int:
    """Bytes of one hidden vector crossing a single PP boundary under *codec*."""
    return int(hidden_size * bytes_per_element(codec))


# ── Strategy planning ──────────────────────────────────────────────────────


@dataclass
class ParallelismStrategy:
    """Recommended parallelism layout for a given model + GPU count on this fleet.

    Attributes:
        num_gpus: total GPUs consumed.
        pp_size: pipeline-parallel depth (1–4, capped per wire budget).
        ep_size: expert-parallel groups (1 for dense; >1 for MoE stage-local EP).
        strategy: short label for logging/reporting.
        pp_stages: distinct pipeline stages (= pp_size).
        ep_groups: distinct expert partitions within each PP stage (= ep_size).
        local_ep: always True — global EP is a wire anti-pattern on this fleet.
        num_replicas: how many copies of the (pp × ep) group to replicate.
    """

    num_gpus: int
    pp_size: int
    ep_size: int
    strategy: str = "replicate"
    pp_stages: int = 1
    ep_groups: int = 1
    local_ep: bool = True
    num_replicas: int = 1
    warnings: list[str] = field(default_factory=list)


def plan_parallelism(
    cfg: ModelConfig,
    num_gpus: int,
    *,
    max_pp: int = _MAX_PP_DEPTH,
) -> ParallelismStrategy:
    """Compute the fleet-recommended parallelism layout for *cfg* on *num_gpus*.

    The planner always prefers shallower PP (capped at *max_pp*, default 4) and
    stage-local EP, replicating groups when GPUs exceed the group capacity.
    TP and FSDP are never recommended.
    """
    if num_gpus <= 0:
        raise ValueError("num_gpus must be >= 1")

    n_layers = cfg.num_hidden_layers
    n_experts = cfg.num_experts if cfg.is_moe() else 0
    warnings_list: list[str] = []

    # ── single GPU ──
    if num_gpus == 1:
        return ParallelismStrategy(num_gpus=1, pp_size=1, ep_size=1, pp_stages=1, ep_groups=1)

    # ── Enumerate all (pp, ep, replicas) combos; pick the best ──
    # Scoring: (ep_bonus, pp, replicas) — lexicographic max.
    # ep_bonus = 1 when ep >= 2 AND experts-per-group >= 2 (meaningful EP),
    # else 0. This ensures stage-local EP is preferred over deeper PP only
    # when experts are abundant enough to benefit from sharding.
    best_pp, best_ep, best_replicas = 1, 1, 1
    best_score = (-1, -1, -1)

    for replicas in range(num_gpus, 0, -1):
        if num_gpus % replicas != 0:
            continue
        gpus_per_replica = num_gpus // replicas

        if n_experts == 0:
            pp = min(max_pp, n_layers, gpus_per_replica)
            if pp >= 2:
                score = (0, pp, replicas)
                if score > best_score:
                    best_score = score
                    best_pp, best_ep, best_replicas = pp, 1, replicas
        else:
            for pp in range(min(max_pp, n_layers, gpus_per_replica), 1, -1):
                for ep in range(1, min(gpus_per_replica // pp, n_experts) + 1):
                    if pp * ep != gpus_per_replica:
                        continue
                    ep_bonus = 1 if (ep >= 2 and n_experts // ep >= 2) else 0
                    score = (ep_bonus, pp, replicas)
                    if score > best_score:
                        best_score = score
                        best_pp, best_ep, best_replicas = pp, ep, replicas

    effective_pp = max(1, min(best_pp, max_pp, n_layers))
    effective_ep = max(1, min(best_ep, max(1, n_experts)))
    effective_replicas = best_replicas

    # ── label ──
    if effective_replicas == 1 and effective_pp == 1 and effective_ep == 1:
        strategy = "replicate"
    elif effective_ep > 1:
        strategy = "pp_plus_local_ep"
    elif effective_replicas > 1:
        strategy = "pp_replicate"
    else:
        strategy = "shallow_pp"

    # ── warnings ──
    layers_per_gpu = n_layers / max(effective_pp, 1)
    if layers_per_gpu < 1:
        warnings_list.append(
            f"Only {n_layers} layers across {effective_pp} PP stages gives <1 layer "
            f"per GPU; too fine-grained for the PCIe-x1 wire."
        )

    return ParallelismStrategy(
        num_gpus=num_gpus,
        pp_size=effective_pp,
        ep_size=effective_ep,
        strategy=strategy,
        pp_stages=effective_pp,
        ep_groups=effective_ep,
        num_replicas=effective_replicas,
        warnings=warnings_list,
    )


# ── Anti-pattern validation ────────────────────────────────────────────────


def validate_parallel_config(serve_cfg) -> tuple[list[str], list[str]]:
    """Check a *ServeConfig* for fleet anti-patterns.

    Returns ``(warnings, errors)``.  Errors are hard-stop violations (e.g. TP).
    Warnings are strong recommendations (e.g. deep PP) that the user is
    proceeding against advice.

    This function does NOT inspect NCCL/DCGM status or any live hardware
    diagnostic — those can confirm the link is PCIe-x1 but are not permission
    to go communication-heavy.
    """
    errors: list[str] = []
    warnings_list: list[str] = []

    tp = getattr(serve_cfg, "tensor_parallel_size", 1)
    pp = getattr(serve_cfg, "pipeline_parallel_size", 1)
    ep = getattr(serve_cfg, "expert_parallel_size", 1)

    if tp != 1:
        errors.append(
            "TP is not viable on PCIe-1.0-x1 (~250 MB/s). Use pipeline_parallel_size "
            "and/or expert_parallel_size instead — see superl8 transport-compression.md."
        )

    if pp > _MAX_PP_DEPTH:
        warnings_list.append(
            f"ServeConfig.pipeline_parallel_size={pp} exceeds the fleet-recommended "
            f"maximum of {_MAX_PP_DEPTH} for PCIe-x1 hardware (bandwidth ratio "
            f"{_BW_RATIO:.0f}:1 HBM/wire). Deep PP costs ~{pp - 1} hidden-state "
            f"transfers per token over a ~250 MB/s link. Prefer replicating "
            f"shallower PP groups (max 4-way) instead. "
            f"See docs/parallelism-doctrine.md and superl8 transport-compression.md."
        )

    if pp == 0:
        errors.append("pipeline_parallel_size must be >= 1.")

    if ep < 0:
        errors.append("expert_parallel_size must be >= 0.")

    return warnings_list, errors


# ── Throughput estimation ──────────────────────────────────────────────────


def estimate_throughput(
    strategy: ParallelismStrategy,
    cfg: ModelConfig,
    *,
    seq_len: int = 512,
    batch_size: int = 4,
    codec: str = "int8",
) -> dict:
    """Topology-agnostic throughput and latency estimate for a given strategy.

    Returns a dict with:
        tokens_per_second: decode throughput (total across replicas).
        p99_latency_ms: per-token step latency.
        on_wire_bytes_per_token: estimated bytes crossing the PCIe bus per
            generated token (PP-boundary activations only; EP is stage-local so
            expert traffic stays on-card).
        pp_boundaries: number of PP handoffs per token.
        activation_bytes: bytes of one hidden state over one PP boundary.

    The model is symbolic (no real multi-GPU execution) — it captures the
    architectural cost of the fleet wire so strategy comparisons are grounded.
    """
    hidden = cfg.hidden_size
    n_layers = cfg.num_hidden_layers
    pp = strategy.pp_size
    n_replicas = strategy.num_replicas

    act_bytes_per_boundary = _activation_bytes_per_token(hidden, codec=codec)
    pp_boundaries = max(0, pp - 1)

    on_wire_bytes = act_bytes_per_boundary * pp_boundaries

    # Model: per-GPU work ∝ n_layers / pp, peer-to-peer transfer cost ∝
    # pp_boundaries / PCIe bandwidth.  Express as a latency floor.
    base_step_ms = (n_layers / max(pp, 1)) * 1.2  # ~1.2 ms per layer (empirical)
    transfer_ms = (on_wire_bytes / _PCIE_BW_BYTES) * 1000  # ms per token
    step_ms = base_step_ms + transfer_ms

    p99_ms = step_ms * 1.3  # p99 ≈ 1.3x mean for decode steps
    tok_s = (batch_size / (step_ms / 1000)) * n_replicas

    return {
        "tokens_per_second": round(tok_s, 1),
        "p99_latency_ms": round(p99_ms, 1),
        "on_wire_bytes_per_token": on_wire_bytes,
        "pp_boundaries": pp_boundaries,
        "activation_bytes_per_boundary": act_bytes_per_boundary,
    }
