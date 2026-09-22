# SPDX-License-Identifier: MIT
"""Serving configuration."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

_MAX_PP_DEPTH = 4

_PCIE_BW_BYTES = 250e6
_HBM_BW_BYTES = 829e9


@dataclass
class ServeConfig:
    """Runtime knobs for the W8A8 server.

    weight_bits: 8 (per-row int8) or 4 (per-group int4/NF4 storage, unpacked to int8
    for dp4a). kv_cache_dtype: 'int8' (half the KV footprint + read BW) or 'fp16'.
    tensor_parallel_size stays 1 on this fleet (PCIe-1.0-x1 kills TP); multi-GPU uses
    pipeline + MoE-expert parallelism instead (see the transport docs).
    """

    model: str  # path to a .superl8 checkpoint (or HF dir to convert)
    weight_bits: int = 8  # 8 | 4
    kv_cache_dtype: str = "int8"  # 'int8' | 'fp16'
    max_seq_len: int = 8192
    block_size: int = 256  # paged-KV block (tokens)
    max_num_seqs: int = 256
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1  # keep 1 here; use PP/MoE-EP for multi-GPU
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    rotate: bool = True  # Hadamard incoherence (baked into weights)

    def __post_init__(self) -> None:
        assert self.weight_bits in (4, 8), "weight_bits must be 4 or 8"
        assert self.kv_cache_dtype in ("int8", "fp16")
        if self.tensor_parallel_size != 1:
            raise ValueError(
                "TP is not viable on PCIe-1.0-x1 (~250 MB/s). Use pipeline_parallel_size "
                "and/or expert_parallel_size instead — see superl8 transport-compression.md."
            )
        if self.pipeline_parallel_size > _MAX_PP_DEPTH:
            ratio = _HBM_BW_BYTES / _PCIE_BW_BYTES
            warnings.warn(
                f"pipeline_parallel_size={self.pipeline_parallel_size} exceeds the "
                f"fleet-recommended maximum of {_MAX_PP_DEPTH} for PCIe-x1 hardware "
                f"(bandwidth ratio {ratio:.0f}:1 HBM/wire). Deep PP costs "
                f"~{self.pipeline_parallel_size - 1} hidden-state transfers per token "
                f"over a ~250 MB/s link. Prefer replicating shallower PP groups "
                f"(max {_MAX_PP_DEPTH}-way) — see docs/parallelism-doctrine.md and "
                f"superl8 transport-compression.md.",
                stacklevel=2,
            )
