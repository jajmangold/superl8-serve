# SPDX-License-Identifier: MIT
"""Lightweight optimizer support for the superl8 training path (issue #133).

Provides ``create_optimizer`` (AdamW factory) and helpers to serialize /
deserialize optimizer state for checkpoint persistence.  Uses torch-native
AdamW which is compatible with the fp16/fp32 gradients produced by the
superl8 int8 autograd path.
"""

from __future__ import annotations

import json
from typing import Any

import torch


def create_optimizer(
    params: list[torch.Tensor] | torch.nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """AdamW factory with sensible defaults for the int8 training path."""
    return torch.optim.AdamW(
        params if isinstance(params, list) else params.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )


def serialize_optimizer_state(
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Flatten optimizer state dict into a tensor dict + JSON-safe metadata.

    Returns (flat_tensors, metadata).
    *flat_tensors* maps names like ``optimizer.state.0.exp_avg`` to tensors.
    *metadata* carries ``optimizer_type`` and ``optimizer_param_groups`` (JSON).
    """
    sd = optimizer.state_dict()
    flat: dict[str, torch.Tensor] = {}
    for pid, pstate in sd.get("state", {}).items():
        for key, val in pstate.items():
            if isinstance(val, torch.Tensor):
                flat[f"optimizer.state.{pid}.{key}"] = val
    param_groups_clean: list[dict[str, Any]] = []
    for g in sd.get("param_groups", []):
        gdict = dict(g)
        gdict["params"] = list(gdict["params"])
        param_groups_clean.append(gdict)
    meta = {
        "optimizer_type": type(optimizer).__name__,
        "optimizer_param_groups": json.dumps(param_groups_clean),
    }
    return flat, meta


def deserialize_optimizer_state(
    flat_tensors: dict[str, torch.Tensor],
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Reconstruct an optimizer state dict from serialized tensors + metadata.

    The returned dict can be passed to ``optimizer.load_state_dict()``.
    """
    state: dict[int, dict[str, torch.Tensor]] = {}
    for flat_name, t in flat_tensors.items():
        parts = flat_name.split(".")
        assert parts[:2] == ["optimizer", "state"], f"unexpected key {flat_name}"
        pid = int(parts[2])
        key = ".".join(parts[3:])
        state.setdefault(pid, {})[key] = t

    param_groups: list[dict[str, Any]] = json.loads(meta["optimizer_param_groups"])
    return {"state": state, "param_groups": param_groups}
