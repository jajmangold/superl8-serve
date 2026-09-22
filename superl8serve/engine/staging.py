# SPDX-License-Identifier: MIT
"""StagingBuffer — per-layer activation buffer for weight-stationary decode.

Phase 1 of the per-layer staging scheduler (issue #320). Each layer boundary
gets a StagingBuffer that holds the hidden state (and residual, for pre-norm
models) waiting to be processed by the next layer. The key invariant:

  * ``active_count`` is a **host-side** integer — the skip-empty check
    ``if staging[i].is_empty(): continue`` is a Python-level branch that
    avoids any device synchronization.
  * ``buf`` and ``residual`` are persistent device tensors whose contents are
    refreshed via ``copy_`` before each CUDA-graph replay (same contract as
    ``GraphedDecode``).

For post-norm models (e.g. LFM2) that don't carry a running residual, the
``residual`` tensor is allocated but never read — the per-layer graph simply
ignores it.
"""

from __future__ import annotations

import torch


class StagingBuffer:
    """Per-layer activation buffer. Shape: ``(max_batch, hidden_dim)`` in fp16.

    Fields:
      buf:       hidden state for this layer boundary  ``[B, H]``
      residual:  running residual (pre-norm models)    ``[B, H]``
      active_count: host-side integer — how many tokens are waiting
    """

    __slots__ = ("buf", "residual", "active_count")

    def __init__(self, max_batch: int, hidden_dim: int, device: str):
        self.buf = torch.zeros(
            max_batch, 1, hidden_dim, dtype=torch.float16, device=device
        )
        self.residual = torch.zeros(
            max_batch, 1, hidden_dim, dtype=torch.float16, device=device
        )
        self.active_count: int = 0

    # -- public API --------------------------------------------------------

    def set_active(self, tokens: torch.Tensor, count: int) -> None:
        """Copy *tokens* into ``buf`` and set ``active_count``.

        ``tokens`` may be longer than ``count``; only the first ``count``
        rows are copied.  Caller must ensure ``count <= max_batch``.

        ``tokens`` shape: ``(count, hidden_dim)`` or ``(count, 1, hidden_dim)``.
        """
        if count > 0:
            t = tokens[:count].to(self.buf.dtype)
            if t.dim() == 2:
                t = t.unsqueeze(1)
            self.buf[:count].copy_(t)
        self.active_count = count

    def set_residual(self, residual: torch.Tensor, count: int) -> None:
        """Copy *residual* into the residual buffer (pre-norm models only)."""
        if count > 0:
            r = residual[:count].to(self.residual.dtype)
            if r.dim() == 2:
                r = r.unsqueeze(1)
            self.residual[:count].copy_(r)

    def is_empty(self) -> bool:
        """Host-side check — no device sync."""
        return self.active_count == 0

    def __repr__(self) -> str:
        return (
            f"StagingBuffer(batch={self.active_count} "
            f"buf={self.buf.shape} device={self.buf.device})"
        )
