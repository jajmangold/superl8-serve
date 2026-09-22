# SPDX-License-Identifier: MIT
"""Gated activations on a merged gate_up projection output.

SwiGLU (`SiluAndMul`) for Qwen3; GeGLU with the tanh gelu approximation
(`GeluAndMul`) for Gemma3/4. Both split the merged `[..., 2*intermediate]` into
gate/up and gate the up branch. `get_act_and_mul(name)` maps a HF `hidden_act`.
"""
from __future__ import annotations

import superl8
import torch
import torch.nn as nn
import torch.nn.functional as F


def _fused_ok(x: torch.Tensor) -> bool:
    return x.is_cuda and x.dtype in (torch.float16, torch.bfloat16)


class SiluAndMul(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., 2*intermediate] — first half gate, second half up."""
        if _fused_ok(x):
            return superl8.act_and_mul(x, "silu")          # one fused launch
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up


class GeluAndMul(nn.Module):
    """Gemma's GeGLU. HF `hidden_act='gelu_pytorch_tanh'` -> tanh approximation."""

    def __init__(self, approximate: str = "tanh"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused kernel covers the tanh approximation; exact-erf gelu ('none') and
        # CPU/non-fp16 fall back to eager.
        if self.approximate == "tanh" and _fused_ok(x):
            return superl8.act_and_mul(x, "gelu_tanh")
        gate, up = x.chunk(2, dim=-1)
        return F.gelu(gate, approximate=self.approximate) * up


def get_act_and_mul(hidden_act: str) -> nn.Module:
    a = hidden_act.lower()
    if a == "silu":
        return SiluAndMul()
    if a in ("gelu_pytorch_tanh", "gelu_tanh"):
        return GeluAndMul("tanh")
    if a in ("gelu", "gelu_new"):
        return GeluAndMul("none")
    raise ValueError(f"unsupported gated activation {hidden_act!r}")
