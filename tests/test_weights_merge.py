# SPDX-License-Identifier: MIT
"""Merge-helper state-dict contract (`superl8serve.models.weights`).

`qkv_weight`/`gate_up_weight` fuse projections on the output axis. They may optionally
POP their source rows to bound the merge transient on a card-filling load — but that is
opt-in (`consume_on_merge()`), because the DEFAULT must leave the caller's state dict
intact: many callers build a second model from the same dict (an engine + a reference
runner, an fp16 + an int8 model), and a destructive default deletes rows out from under
them (the `KeyError: '...q_proj.weight'` regression this guards against).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from superl8 import QTensor
from superl8.quant.core import quantize_int8_rowwise
from superl8serve.models.weights import (
    consume_on_merge,
    gate_up_weight,
    merge_qtensor,
    qkv_weight,
)

_DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _attn_mlp_sd(prefix: str = "model.layers.0") -> dict:
    r = lambda a, b: torch.randn(a, b, device=_DEV, dtype=torch.float16) * 0.02  # noqa: E731
    H, hd, nh, nkv, inter = 64, 16, 4, 2, 128
    return {
        f"{prefix}.self_attn.q_proj.weight": r(nh * hd, H),
        f"{prefix}.self_attn.k_proj.weight": r(nkv * hd, H),
        f"{prefix}.self_attn.v_proj.weight": r(nkv * hd, H),
        f"{prefix}.mlp.gate_proj.weight": r(inter, H),
        f"{prefix}.mlp.up_proj.weight": r(inter, H),
    }


def test_merge_is_non_destructive_by_default():
    """Default merge INDEXES its sources — the state dict is untouched, so a second
    consumer can still read q/k/v/gate/up."""
    p = "model.layers.0"
    sd = _attn_mlp_sd(p)
    before = set(sd)

    qkv = qkv_weight(sd, f"{p}.self_attn")
    gate_up = gate_up_weight(sd, f"{p}.mlp")

    assert set(sd) == before  # nothing popped
    # Merged rows == sum of source rows on the output axis.
    assert qkv.data.shape[0] == sum(
        sd[f"{p}.self_attn.{n}_proj.weight"].shape[0] for n in ("q", "k", "v")
    )
    assert gate_up.data.shape[0] == (
        sd[f"{p}.mlp.gate_proj.weight"].shape[0] + sd[f"{p}.mlp.up_proj.weight"].shape[0]
    )

    # The exact failing scenario: build (merge) a SECOND time from the same dict.
    qkv2 = qkv_weight(sd, f"{p}.self_attn")
    assert qkv2.data.shape == qkv.data.shape


def test_consume_on_merge_pops_sources():
    """Under `consume_on_merge()` the sources are popped (freed as consumed) so the
    single-card load path bounds its transient; the merged weight is identical."""
    p = "model.layers.0"
    sd = _attn_mlp_sd(p)

    with consume_on_merge():
        qkv = qkv_weight(sd, f"{p}.self_attn")
        gate_up = gate_up_weight(sd, f"{p}.mlp")

    for n in ("q", "k", "v"):
        assert f"{p}.self_attn.{n}_proj.weight" not in sd
    for n in ("gate", "up"):
        assert f"{p}.mlp.{n}_proj.weight" not in sd
    assert qkv.data.shape[0] == (4 + 2 + 2) * 16
    assert gate_up.data.shape[0] == 2 * 128


def test_consume_flag_restored_after_context():
    """The flag is a scoped toggle — after the context, merges are non-destructive again
    (and it restores correctly even on an exception)."""
    p = "model.layers.0"
    try:
        with consume_on_merge():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    sd = _attn_mlp_sd(p)
    qkv_weight(sd, f"{p}.self_attn")
    assert f"{p}.self_attn.q_proj.weight" in sd  # not popped -> flag was restored


# ── Heterogeneous k-quant merge (P0<->P1 seam bug) ──────────────────────────
# A GGUF Q4_K_M/imatrix menu MIXES types across a merged QKV group: q,k at Q4_K
# (`gguf_kquant`) next to the sensitive v that the P1 loader kept as a dequant->
# `per_row_i8` row. merge_qtensor must dequant EACH row by its OWN scheme and fuse
# as one per_row_i8 — the earlier code routed the per_row_i8 row through
# `dequant_kquant` (gguf_kquant-only) and died with `KeyError: ''`.


def _pack_scales_min_k4(sc, m):
    """Inverse of llama.cpp get_scale_min_k4: 8x 6-bit scale + 8x 6-bit min -> 12 bytes."""
    sc, m = sc.astype(np.uint16), m.astype(np.uint16)
    q = np.zeros(sc.shape[:-1] + (12,), np.uint16)
    q[..., 0:4] = (sc[..., 0:4] & 0x3F) | (((sc[..., 4:8] >> 4) & 0x3) << 6)
    q[..., 4:8] = (m[..., 0:4] & 0x3F) | (((m[..., 4:8] >> 4) & 0x3) << 6)
    q[..., 8:12] = (sc[..., 4:8] & 0xF) | ((m[..., 4:8] & 0xF) << 4)
    return q.astype(np.uint8)


def _q4k_gguf_kquant(w: torch.Tensor) -> QTensor:
    """fp weight [out,in] (in%256==0) -> native Q4_K `gguf_kquant` QTensor. Pure-numpy
    Q4_K encoder (the gguf pkg only DEquantizes k-quants); validated byte-for-byte
    against llama.cpp's dequant loop, so `dequant_kquant` reads it back faithfully."""
    out, inf = w.shape
    nsb = inf // 256
    x = w.detach().float().cpu().numpy().reshape(out, nsb, 8, 32)
    xmax = x.max(-1)
    xmin = np.maximum(0.0, -x.min(-1))
    scale = np.where((xmax + xmin) == 0, 1.0, (xmax + xmin) / 15.0)
    q = np.clip(np.rint((x + xmin[..., None]) / scale[..., None]), 0, 15).astype(np.int64)
    d = np.where(scale.max(-1) == 0, 1.0, scale.max(-1) / 63.0)
    dm = np.where(xmin.max(-1) == 0, 1.0, xmin.max(-1) / 63.0)
    sc6 = np.clip(np.rint(scale / d[..., None]), 0, 63).astype(np.uint8)
    m6 = np.clip(np.rint(xmin / dm[..., None]), 0, 63).astype(np.uint8)
    d16, dm16 = d.astype(np.float16), dm.astype(np.float16)
    blk = np.zeros((out, nsb, 144), np.uint8)
    blk[:, :, 0:2] = np.frombuffer(np.ascontiguousarray(d16).tobytes(), np.uint8).reshape(out, nsb, 2)
    blk[:, :, 2:4] = np.frombuffer(np.ascontiguousarray(dm16).tobytes(), np.uint8).reshape(out, nsb, 2)
    blk[:, :, 4:16] = _pack_scales_min_k4(sc6, m6)
    lo = (q & 0xF).astype(np.uint8)
    for g in range(4):
        blk[:, :, 16 + g * 32:16 + (g + 1) * 32] = lo[:, :, 2 * g, :] | (lo[:, :, 2 * g + 1, :] << 4)
    raw = blk.reshape(out, nsb * 144)
    return QTensor(torch.from_numpy(raw.copy()).to(w.device), None,
                   scheme="gguf_kquant", group_size=256, codebook="q4_k")


def _per_row_i8(w: torch.Tensor) -> QTensor:
    q, s = quantize_int8_rowwise(w)
    return QTensor(q.contiguous(), s.squeeze(-1).float().contiguous(), scheme="per_row_i8")


def test_merge_heterogeneous_gguf_kquant_plus_per_row_i8():
    """The seam bug: q/k `gguf_kquant` (Q4_K) + v `per_row_i8` in one QKV group must
    merge without crashing and reconstruct correct fp16 (regression: `KeyError: ''`)."""
    gguf = pytest.importorskip("gguf")
    from superl8serve.gguf_native import dequant_kquant

    torch.manual_seed(0)
    inf = 256
    wq = torch.randn(64, inf, device=_DEV, dtype=torch.float16) * 0.05
    wk = torch.randn(32, inf, device=_DEV, dtype=torch.float16) * 0.05
    wv = torch.randn(32, inf, device=_DEV, dtype=torch.float16) * 0.05
    q, k = _q4k_gguf_kquant(wq), _q4k_gguf_kquant(wk)
    v = _per_row_i8(wv)                       # the heterogeneous row

    merged = merge_qtensor([q, k, v])         # must NOT raise KeyError

    # Mixed group -> benign requant to one per_row_i8.
    assert merged.scheme == "per_row_i8"
    assert merged.data.shape == (64 + 32 + 32, inf)
    # Reconstruct the merged weight and compare to the per-row dequant of each source.
    exp = torch.cat([dequant_kquant(q), dequant_kquant(k),
                     (v.data.float() * v.scale.unsqueeze(-1)).half()], dim=0).float()
    got = (merged.data.float() * merged.scale.unsqueeze(-1)).float()
    cos = torch.nn.functional.cosine_similarity(got.flatten(), exp.flatten(), dim=0).item()
    assert cos >= 0.999, f"merged QKV cos {cos:.5f} < 0.999"


def test_merge_homogeneous_gguf_kquant_stays_native():
    """All-Q4_K QKV stays native `gguf_kquant` (byte concat, no requant) — the fast
    path the heterogeneous fix must not regress."""
    torch.manual_seed(1)
    inf = 256
    rows = [_q4k_gguf_kquant(torch.randn(n, inf, device=_DEV, dtype=torch.float16) * 0.05)
            for n in (64, 32, 32)]
    merged = merge_qtensor(rows)
    assert merged.scheme == "gguf_kquant" and merged.codebook == "q4_k"
    assert merged.data.shape[0] == 128 and merged.scale is None
