# SPDX-License-Identifier: MIT
"""RMSNorm — the only normalization Qwen3 uses.

Roles: (1) the pre-attention / pre-MLP layer norm over `hidden_size`, with an
optional fused residual add (nano-vllm style: `x = x + residual` folded in so the
decoder layer threads one tensor); (2) Qwen3/Gemma3 per-head QK-norm over
`head_dim`; (3) Gemma's pre/post feed-forward norms. Same module, different `dim`.

`add_unit_offset=True` is the **Gemma** convention: the learned weight scales as
`(1 + w)` (Gemma stores gains centered at 0), vs Qwen/Llama which use `w` directly.
Computed in fp32 (the reduction is numerically load-bearing) and cast back — never
quantized.
"""
from __future__ import annotations

import superl8
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        weight: torch.Tensor | None = None,
        *,
        add_unit_offset: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        if weight is None:
            weight = torch.zeros(dim, dtype=torch.float16) if add_unit_offset \
                else torch.ones(dim, dtype=torch.float16)
        self.weight = nn.Parameter(weight)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        w = (1.0 + self.weight.float()) if self.add_unit_offset else self.weight.float()
        return (xf * w).to(dt)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Fused single-launch path (superl8.rmsnorm handles residual add + Gemma
        # unit-offset internally); falls back to the eager chain on CPU / non-fp16.
        if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
            return superl8.rmsnorm(x, self.weight, self.eps, residual=residual,
                                unit_offset=self.add_unit_offset)
        if residual is None:
            return self._norm(x)
        # Fused add: return (normed, new_residual) so the caller keeps the residual
        # stream in the layer's native dtype without a second add.
        x = x + residual
        return self._norm(x), x
