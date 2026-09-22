# SPDX-License-Identifier: MIT
"""LoRA adapter layer for fine-tuning on quantized int8 base (issue #129)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from superl8serve.layers.linear import LinearW8A8, _dequant_weight


def _get_weight(module: nn.Module) -> torch.Tensor | None:
    """Return the weight tensor of a linear-like module, or None."""
    if isinstance(module, LinearW8A8):
        return _dequant_weight(module.weight)
    if isinstance(module, nn.Linear):
        return module.weight.data
    return None


def _set_weight(module: nn.Module, w: torch.Tensor) -> None:
    """Set the weight tensor of a linear-like module in-place."""
    if isinstance(module, nn.Linear):
        module.weight.data.copy_(w)
    else:
        raise TypeError(f"cannot set weight on {type(module).__name__}")


def _get_bias(module: nn.Module) -> torch.Tensor | None:
    if isinstance(module, (LinearW8A8, nn.Linear)):
        return module.bias
    return None


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: Sequence[str] | None = None
    target_pattern: str | None = None
    bias: str = "none"


class LoRALayer(nn.Module):
    """Wrap a frozen linear module with a trainable low-rank adapter.

    Forward when *not* merged::

        output = base(x) + (x @ lora_A.T) @ lora_B.T * scaling

    where ``scaling = alpha / r``. The base module's parameters are frozen;
    only ``lora_A`` and ``lora_B`` receive gradients.
    """

    def __init__(
        self,
        base: nn.Module,
        r: int,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        base_weight = _get_weight(base)
        if base_weight is None:
            raise TypeError(f"LoRALayer cannot wrap {type(base).__name__}")
        out_features, in_features = base_weight.shape
        dev = base_weight.device
        dt = base_weight.dtype

        self.lora_A = nn.Parameter(torch.empty(out_features, r, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.empty(r, in_features, device=dev, dtype=dt))
        self.reset_parameters()

        self._original_weight = base_weight.detach().clone()
        self._merged = False
        self._merged_weight: torch.Tensor | None = None

        for p in self.base.parameters():
            p.requires_grad_(False)
        for p in self.base.buffers():
            p.requires_grad_(False)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._merged and self._merged_weight is not None:
            return F.linear(x, self._merged_weight, _get_bias(self.base))
        base_out = self.base(x)
        adapter_out = (x @ self.lora_B.T) @ self.lora_A.T
        return base_out + adapter_out * self.scaling

    def merge(self) -> None:
        if self._merged:
            return
        base_w = _get_weight(self.base)
        if base_w is None:
            return
        delta = (self.lora_B.T @ self.lora_A.T).T * self.scaling
        merged = (base_w.float() + delta.float()).to(base_w.dtype)
        self._merged_weight = nn.Parameter(merged, requires_grad=False)
        self._merged = True

    def unmerge(self) -> None:
        if not self._merged:
            return
        self._merged_weight = None
        self._merged = False

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, merged={self._merged}"


def _match_name(name: str, config: LoRAConfig) -> bool:
    if config.target_pattern is not None:
        return bool(re.search(config.target_pattern, name))
    if config.target_modules is not None:
        return any(name.endswith(t) for t in config.target_modules)
    return True


def inject_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Walk ``model`` and wrap matching linear modules with ``LoRALayer``.

    Modules matching *target_modules* or *target_pattern* in *config* that are
    ``LinearW8A8`` or ``nn.Linear`` get replaced in-place. Returns the model.
    """
    _valid_types = (LinearW8A8, nn.Linear)
    for name, mod in list(model.named_modules()):
        if not _match_name(name, config):
            continue
        if not isinstance(mod, _valid_types):
            continue
        parent_path = name.rpartition(".")[0]
        attr = name.rpartition(".")[2]
        parent = model.get_submodule(parent_path) if parent_path else model
        if not hasattr(parent, attr):
            continue
        lora_mod = LoRALayer(mod, r=config.r, alpha=config.alpha, dropout=config.dropout)
        setattr(parent, attr, lora_mod)
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """Merge all LoRALayer adapters into their base weights in-place."""
    for _name, mod in list(model.named_modules()):
        if isinstance(mod, LoRALayer):
            mod.merge()
    return model


def unmerge_lora(model: nn.Module) -> nn.Module:
    """Unmerge all LoRALayer adapters, restoring the original base weights."""
    for _name, mod in list(model.named_modules()):
        if isinstance(mod, LoRALayer):
            mod.unmerge()
    return model
