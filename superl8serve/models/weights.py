# SPDX-License-Identifier: MIT
"""Weight helpers shared by the concrete models.

A model builder receives a dict of fp16 tensors (an HF-style state dict, or the
raw tensors from a `.superl8` file) and turns linear weights into int8 `per_row_i8`
QTensors for the dp4a GEMM. QKV and gate/up projections are MERGED here (concat on
the output axis) so the runtime issues fewer, wider matmuls — the same merge the
offline `.superl8` conversion performs. Norms/embeddings stay fp16.
"""

from __future__ import annotations

import contextlib

import torch

from superl8 import QTensor
from superl8.quant.core import quantize_int8_rowwise

# When True, the q/k/v and gate/up merge helpers POP their source rows out of the
# caller's state dict as they consume them, so each un-merged projection is freed the
# moment it is merged instead of lingering alongside its merged copy for the whole
# build. On a checkpoint whose weights nearly fill the card (a 27B 4-bit on one 16 GiB
# GPU) that duplication is the difference between building and OOM-ing. It is OFF by
# default because popping MUTATES the caller's dict — many callers/tests reuse the same
# state dict to build a second model (e.g. an engine + a reference runner), and a
# destructive default would delete rows out from under them (KeyError). The real
# single-card load path owns a freshly-loaded, use-once dict, so it opts in via
# `consume_on_merge()`.
_CONSUME_ON_MERGE = False


@contextlib.contextmanager
def consume_on_merge():
    """Within this context, `qkv_weight`/`gate_up_weight` POP their source rows from the
    passed state dict (freeing each projection as it is merged) instead of indexing them
    (leaving the dict intact). Use it ONLY when the state dict is owned and consumed
    exactly once — i.e. the production `.superl8` -> `build_model` load path — never around a
    build whose dict is reused afterwards."""
    global _CONSUME_ON_MERGE
    prev = _CONSUME_ON_MERGE
    _CONSUME_ON_MERGE = True
    try:
        yield
    finally:
        _CONSUME_ON_MERGE = prev


def to_qtensor(w) -> QTensor:
    """fp16/fp32 weight [out, in] (in % 4 == 0) -> per_row_i8 QTensor. Idempotent:
    an already-quantized QTensor (from a `.superl8` load) passes through unchanged, so
    the same model builders serve both runtime-quant (fp16 in) and offline (.superl8)."""
    if isinstance(w, QTensor):
        return w
    q, s = quantize_int8_rowwise(w)
    return QTensor(q.contiguous(), s.squeeze(-1).float().contiguous(), scheme="per_row_i8")


def _row_to_fp16(r: QTensor) -> torch.Tensor:
    """Reconstruct one pre-quantized merge row to fp16 [out, in], dispatching on its
    OWN scheme. Needed for heterogeneous merged groups (Q4_K `gguf_kquant` q/k next to
    a Q6_K-derived `per_row_i8` v): each row must be dequantized by the scheme it was
    stored in before the group can be re-quantized as one `per_row_i8`."""
    if r.scheme == "gguf_kquant":
        from ..gguf_native import dequant_kquant

        return dequant_kquant(r)
    if r.scheme == "per_row_i8":
        return (r.data.float() * r.scale.unsqueeze(-1)).to(torch.float16)
    if r.scheme == "raw":
        return r.data.to(torch.float16)
    raise ValueError(f"merge_qtensor: cannot dequant merge row scheme {r.scheme!r}")


def merge_qtensor(rows: list) -> QTensor:
    """Fuse q/k/v -> qkv (and gate/up -> gate_up) on the output (row) axis. For fp16
    rows: concat then quantize as one. For pre-quantized QTensors: concat the int8/
    int4 data AND the per-row scales along the row axis (each output row keeps its
    own scale, so the fused weight is exact)."""
    if isinstance(rows[0], QTensor):
        r0 = rows[0]
        # Native GGUF k-quant rows: each row is independently k-quantized (super-blocks
        # run along the IN dim), so concatenating rows on the output axis is exact —
        # BUT only when EVERY row is the SAME native k-quant type (same codebook +
        # bytes/row). A GGUF Q4_K_M/imatrix menu MIXES types across a merged group
        # (e.g. Qwen3-8B: q,k at Q4_K but the sensitive v at Q6_K — and with the P1
        # loader keeping only some k-quant types resident as `gguf_kquant`, the other
        # rows arrive already dequant→`per_row_i8`). Those heterogeneous groups can't
        # concat natively: dequant EACH row to fp16 *per its own scheme* and merge as
        # one `per_row_i8` (the benign-requant fallback). NB: `dequant_kquant` only
        # accepts `gguf_kquant` rows, so a `per_row_i8` row must be reconstructed from
        # its int8 data+scale here, not routed through it (that was `KeyError: ''`).
        schemes = {r.scheme for r in rows}
        if "gguf_kquant" in schemes:
            widths = {r.data.shape[1] for r in rows}
            codes = {r.codebook for r in rows}
            if schemes == {"gguf_kquant"} and len(widths) == 1 and len(codes) == 1:
                return QTensor(
                    torch.cat([r.data for r in rows], dim=0).contiguous(),
                    None,
                    scheme="gguf_kquant",
                    group_size=r0.group_size,
                    codebook=r0.codebook,
                )
            return to_qtensor(torch.cat([_row_to_fp16(r) for r in rows], dim=0))
        data = torch.cat([r.data for r in rows], dim=0)
        scale = torch.cat([r.scale for r in rows], dim=0)
        return QTensor(
            data.contiguous(),
            scale.contiguous(),
            scheme=r0.scheme,
            group_size=r0.group_size,
            codebook=r0.codebook,
        )
    return to_qtensor(torch.cat(rows, dim=0))


def _take_rows(sd: dict, keys: list[str]) -> list:
    """Fetch each weight from `sd`. Under `consume_on_merge()` this POPS (removes) the
    row so the un-merged source tensor is freed as soon as the merge's local list goes
    out of scope — bounding the merge transient on a card-filling checkpoint. Otherwise
    it INDEXES (non-destructive), leaving the dict intact for callers that reuse it.
    Each q/k/v/gate/up projection is consumed exactly once per build either way."""
    if _CONSUME_ON_MERGE:
        return [sd.pop(k) for k in keys]
    return [sd[k] for k in keys]


def qkv_weight(sd: dict, prefix: str) -> QTensor:
    return merge_qtensor(
        _take_rows(
            sd, [f"{prefix}.q_proj.weight", f"{prefix}.k_proj.weight", f"{prefix}.v_proj.weight"]
        )
    )


def gate_up_weight(sd: dict, prefix: str) -> QTensor:
    return merge_qtensor(_take_rows(sd, [f"{prefix}.gate_proj.weight", f"{prefix}.up_proj.weight"]))
