# SPDX-License-Identifier: MIT
"""Token embedding + LM head.

Embeddings stay fp16 (a lookup, not a matmul — no dp4a benefit, and the table is
numerically sensitive). Gemma scales embeddings by sqrt(hidden_size) after lookup
(`embed_scale`); Qwen3 does not. The LM head is either tied to the embedding
(small models) or its own weight, and may be int8-quantized (a real GEMM) via a
`QTensor`, or kept fp16.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from superl8 import QTensor

from .linear import LinearW8A8


class VocabEmbedding(nn.Module):
    """Token embedding lookup.

    Accepts either an fp16 `[vocab, hidden]` weight (the default) or a
    `per_row_i8` `QTensor` — an int8 embedding table with a per-row fp32 scale.
    The int8 variant halves the (large, vocab-sized) embedding footprint; it is a
    *gather*, not a matmul, so dequant is a single per-row multiply applied only to
    the rows actually looked up (loss is negligible — see tests). This is the lever
    that fits a 27B 4-bit checkpoint (embed alone is ~2.4 GiB fp16) onto one 16 GiB
    card. Enable it at load time with `load_superl8_state_dict(..., embed_int8=True)`.
    """

    def __init__(self, weight: QTensor | torch.Tensor, embed_scale: float = 1.0,
                 out_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.embed_scale = embed_scale
        # Dtype of the residual stream this embedding seeds. bf16-native models
        # (Gemma3 etc.) must run bf16 (fp32's exponent range) or their massive
        # residual channels overflow fp16's 65504 ceiling a few layers in (#260).
        self.out_dtype = out_dtype
        if isinstance(weight, QTensor):
            if weight.scheme != "per_row_i8":
                raise ValueError(
                    f"VocabEmbedding int8 path needs a per_row_i8 QTensor, got {weight.scheme!r}"
                )
            # int8 table [vocab, hidden] + fp32 per-row scale [vocab, 1] (kept 2-D so
            # `F.embedding` gathers a broadcastable [..., 1] scale alongside the rows).
            self.register_buffer("qweight", weight.data, persistent=False)
            scale = weight.scale
            if scale.dim() == 1:
                scale = scale.unsqueeze(-1)
            self.register_buffer("qscale", scale.to(torch.float32), persistent=False)
            self.weight = None
            self._int8 = True
        else:
            self.weight = nn.Parameter(weight)  # [vocab, hidden] fp16
            self._int8 = False

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._int8:
            rows = F.embedding(input_ids, self.qweight)  # int8 [..., hidden]
            scale = F.embedding(input_ids, self.qscale)  # fp32 [..., 1]
            h = (rows.float() * scale).to(self.out_dtype)
        else:
            h = F.embedding(input_ids, self.weight).to(self.out_dtype)
        if self.embed_scale != 1.0:
            h = h * self.embed_scale
        return h


class LMHead(nn.Module):
    """Projects hidden states to vocab logits. `weight` is a QTensor (int8 GEMM) or
    an fp16 tensor `[vocab, hidden]` (tied embedding / plain matmul)."""

    def __init__(self, weight: QTensor | torch.Tensor, logit_softcap: float | None = None):
        super().__init__()
        self.logit_softcap = logit_softcap
        if isinstance(weight, QTensor):
            self.proj = LinearW8A8(weight)
            self._fp16_w = None
        else:
            self.proj = None
            self.weight = nn.Parameter(weight)
            self._fp16_w = True

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.proj(hidden) if self.proj is not None else F.linear(hidden, self.weight)
        if self.logit_softcap:  # Gemma-style final-logit soft cap
            logits = self.logit_softcap * torch.tanh(logits.float() / self.logit_softcap)
        return logits
