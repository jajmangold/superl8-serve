# SPDX-License-Identifier: MIT
"""Minimal training loop over the superl8 autograd path (issue #82).

The core superl8 autograd primitives are:

* ``superl8.autograd.attn(q, k, v, *, causal, scale)`` — differentiable int8 dp4a
  attention (head_dim ∈ {32, 64, 72, 80, 128, 256}).
* ``superl8.autograd.attn_ref(q, k, v, *, causal, scale)`` — differentiable fp reference
  attention sharing the same analytic backward, no head_dim constraint.

Non-attention layers (linear, rmsnorm, rope, activations) use torch-native
implementations since the superl8 inference kernels do not register autograd nodes.

This module provides:

* ``TrainableLM`` — a minimal decoder-only transformer (2+ layers) that wires
  ``superl8.autograd.attn`` for attention and ``nn.Linear`` (fp16 trainable) for
  projections, RNsNorm, and activations.
* ``training_step`` — forward → cross-entropy → backward → optimizer.step.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── reference RNSNorm (trainable, torch-native) ────────────────────────────


class _RMSNorm(nn.Module):
    """Trainable torch-native RNSNorm (the superl8.rnsnorm kernel has no autograd)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(dt)


# ── trainable multi-head attention (superl8.autograd.attn) ────────────────────


class _TrainableAttention(nn.Module):
    """MHA block that projects fp16 Q/K/V, then calls ``superl8.autograd.attn``."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        inner = num_heads * head_dim
        assert inner == hidden_size, f"MHA: {num_heads}*{head_dim} != {hidden_size}"
        self.q_proj = nn.Linear(hidden_size, inner, bias=False)
        self.k_proj = nn.Linear(hidden_size, inner, bias=False)
        self.v_proj = nn.Linear(hidden_size, inner, bias=False)
        self.o_proj = nn.Linear(inner, hidden_size, bias=False)
        self.scale = head_dim**-0.5

    def forward(self, x: torch.Tensor, *, use_int8_attn: bool = True) -> torch.Tensor:
        B, S, _ = x.shape
        H, D = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, S, H, D).transpose(1, 2).contiguous()
        k = self.k_proj(x).view(B, S, H, D).transpose(1, 2).contiguous()
        v = self.v_proj(x).view(B, S, H, D).transpose(1, 2).contiguous()
        import superl8

        # int8 path only works with fp16; fall back to ref for fp32/fp64.
        if use_int8_attn and q.dtype == torch.float16:
            attn_out = superl8.autograd.attn(q, k, v, causal=True, scale=self.scale)
        else:
            attn_out = superl8.autograd.attn_ref(q, k, v, causal=True, scale=self.scale)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, H * D)
        return self.o_proj(attn_out)


# ── trainable feed-forward (SwiGLU) ────────────────────────────────────────


class _TrainableFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ── decoder layer ──────────────────────────────────────────────────────────


class _TrainableDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.input_norm = _RMSNorm(hidden_size)
        self.attn = _TrainableAttention(hidden_size, num_heads, head_dim)
        self.post_norm = _RMSNorm(hidden_size)
        self.ffn = _TrainableFFN(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor, *, use_int8_attn: bool = True) -> torch.Tensor:
        x = x + self.attn(self.input_norm(x), use_int8_attn=use_int8_attn)
        x = x + self.ffn(self.post_norm(x))
        return x


# ── minimal trainable LM ───────────────────────────────────────────────────


# Valid head_dim values for superl8.autograd.attn (the int8 kernel).
_INT8_VALID_HEAD_DIMS = frozenset({32, 64, 72, 80, 128, 256})


class TrainableLM(nn.Module):
    """Minimal decoder-only transformer for training over superl8 autograd.

    Attention uses ``superl8.autograd.attn`` (int8 fwd + exact bwd) when
    *head_dim* is in {32, 64, 72, 80, 128, 256} and the inputs are fp16;
    otherwise it falls back to ``superl8.autograd.attn_ref`` (fp reference).
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_int8_attn = head_dim in _INT8_VALID_HEAD_DIMS

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                _TrainableDecoderLayer(hidden_size, num_heads, head_dim, intermediate_size)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = _RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, use_int8_attn=self.use_int8_attn)
        x = self.final_norm(x)
        return self.lm_head(x)


# ── training step ──────────────────────────────────────────────────────────


def training_step(
    model: TrainableLM,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    max_grad_norm: float | None = 1.0,
) -> float:
    """Single training step: forward → cross-entropy loss → backward → step.

    Returns the scalar loss value (detached, Python float).
    """
    model.train()
    logits = model(input_ids)  # [B, S, V]
    loss = F.cross_entropy(logits.flatten(0, -2).float(), targets.flatten(0, -1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return loss.detach().item()
