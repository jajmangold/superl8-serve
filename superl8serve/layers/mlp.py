# SPDX-License-Identifier: MIT
"""Gated dense MLP: activation(gate) * up over a merged gate_up projection, then a
down projection. Covers Qwen3 (SwiGLU) and Gemma3/4 (GeGLU) by parameterizing the
activation.

Both projections run on the superl8 dp4a GEMM (`LinearW8A8`, W8A8 or W4A8 from the
`.superl8` weight). gate_proj and up_proj are stored MERGED into one `gate_up` weight
(`[2*intermediate, hidden]`) so the FFN issues two matmuls, not three — the merge
happens offline in the weight-conversion tool.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from superl8 import QTensor

from .activation import SiluAndMul, get_act_and_mul
from .linear import LinearW8A8


class GatedMLP(nn.Module):
    def __init__(
        self,
        gate_up: QTensor,
        down: QTensor,
        *,
        act: nn.Module | str = "silu",
        bias_gate_up=None,
        bias_down=None,
    ):
        super().__init__()
        self.gate_up_proj = LinearW8A8(gate_up, bias_gate_up)
        self.down_proj = LinearW8A8(down, bias_down)
        self.act_fn = get_act_and_mul(act) if isinstance(act, str) else act
        self.hidden_size = self.down_proj.out_features
        self.intermediate_size = self.gate_up_proj.out_features // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Qwen3MLP(GatedMLP):
    """Qwen3 dense MLP (SwiGLU)."""

    def __init__(self, gate_up: QTensor, down: QTensor, bias_gate_up=None, bias_down=None):
        super().__init__(gate_up, down, act=SiluAndMul(),
                         bias_gate_up=bias_gate_up, bias_down=bias_down)
