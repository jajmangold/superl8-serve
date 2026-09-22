# SPDX-License-Identifier: MIT
"""Native TQ3_4S wiring tests (superl8-serve#408).

CPU-only: no GPU, no fused kernel.  Covers the superl8-serve side of the TQ3_4S
contract landing in superl8#271/#272:

  1. type-tag map / dispatch  — ``_KQUANT["TQ3_4S"]``, ``_native_kquant_types``
     probing, the gguf reader patch for fork type 46, ``_dequant_ggml`` routing.
  2. ``dequant_kquant`` for the ``tq3_4s`` codebook vs the ``superl8.quant.tq34s``
     reference oracle on the SAME bytes — SQNR >= 40 dB / cos >= 0.999 (the same
     correctness bar the fused kernel is gated against), plus the E3M5==0
     zero-scale case.
  3. dense-assembly tensor mapping for ``Qwen3.8-27B-MTP-TQ3_4S.gguf`` — the
     866-tensor set (506 TQ3_4S + 360 F32) maps through ``gguf_name_to_hf`` and
     the qwen3_5 hybrid remap chain to exactly the keys the builder reads.

The superl8 reference lives on the ``feat/270-tq34s-reference`` worktree until
superl8#271 merges; when testing against a stock superl8, point PYTHONPATH at it:

    PYTHONPATH=/path/to/superl8-worktree:$PWD \\
        python -m pytest tests/test_tq34s_native.py -q

Tests that need ``superl8.quant.tq34s`` skip (with a message) when the module is
absent rather than failing the whole file.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

# ── superl8 reference (tq34s oracle) present?  superl8#271 not merged → PYTHONPATH ──
try:
    from superl8.quant import tq34s as _tq34s  # noqa: F401
except Exception:  # pragma: no cover - reference not on PYTHONPATH
    _tq34s = None

requires_tq34s = pytest.mark.skipif(
    _tq34s is None,
    reason=(
        "superl8.quant.tq34s (superl8#271 reference) not importable — set PYTHONPATH to "
        "the superl8-270-tq34s-reference worktree (see module docstring)"
    ),
)

from superl8 import QTensor  # noqa: E402

from superl8serve.gguf_native import (  # noqa: E402
    _KQUANT,
    _dequant_ggml,
    _kquant_group_size,
    _native_kquant_types,
    _patch_gguf_tq34s,
    dequant_kquant,
    gguf_config,
    _open_gguf,
    _remap_hybrid_qwen35,
    _restore_hf_qwen35_weights,
)
from superl8serve.convert import _remap_qwen3_next  # noqa: E402
from superl8serve.gguf_import import gguf_name_to_hf  # noqa: E402

# Real on-disk artifact (outside the repo; skip if absent). Read-only header scan.
_TQ38_GGUF = os.path.join(
    os.environ.get("SUPERL8_WEIGHTS_DIR", "/path/to/weights"),
    os.environ.get("TQ38_GGUF", "Qwen3.8-27B-MTP-TQ3_4S.gguf"),
)
requires_gguf = pytest.mark.skipif(not os.path.exists(_TQ38_GGUF), reason=f"missing {_TQ38_GGUF}")


def _tq34s_bytes(out: int, in_features: int, seed: int = 0, *, zero_scale_at=None):
    """Random synthetic TQ3_4S bytes [out, (in/32)*16]: 4 E3M5 scales + 12 code
    bytes per 32-value block. ``zero_scale_at`` optionally zeroes one scale byte
    (the E3M5==0 → scale 0.0 case from the review gate)."""
    rng = np.random.default_rng(seed)
    nb = in_features // 32
    scales = rng.integers(1, 250, size=(out, nb, 4)).astype(np.uint8)
    if zero_scale_at is not None:
        scales.reshape(-1)[zero_scale_at] = 0
    codes = rng.integers(0, 8, size=(out, nb, 12)).astype(np.uint8)
    return np.concatenate([scales.reshape(out, -1), codes.reshape(out, -1)], axis=1).astype(
        np.uint8
    )


def _sqnr(ref, q):
    ref, q = ref.astype(np.float64), q.astype(np.float64)
    err = ref - q
    den = np.linalg.norm(ref)
    return 20.0 * np.log10(den / (np.linalg.norm(err) + 1e-12)) if den > 0 else float("inf")


def _cos(ref, q):
    ref, q = ref.astype(np.float64).ravel(), q.astype(np.float64).ravel()
    return float((ref @ q) / (np.linalg.norm(ref) * np.linalg.norm(q) + 1e-12))


# ── 1. type-tag map / dispatch ───────────────────────────────────────────────
def test_type_tag_map_has_tq34s():
    assert _KQUANT["TQ3_4S"] == ("tq3_4s", 16)
    assert _KQUANT["Q4_K"] == ("q4_k", 144)  # existing entries untouched


def test_kquant_group_size_blocks():
    assert _kquant_group_size("TQ3_4S") == 32  # QK_TQ3_0
    assert _kquant_group_size("Q4_K") == 256  # QK_K


def test_native_kquant_types_probes_linear_tq34s(monkeypatch):
    import superl8

    # Wired into the probe regardless of kernel presence...
    had = _native_kquant_types()
    assert "Q2_K" in had and "Q6_K" in had
    # ...and TQ3_4S flips with `superl8.linear_tq34s` availability (superl8#272).
    monkeypatch.setattr(superl8, "linear_tq34s", lambda *a, **k: None)
    assert "TQ3_4S" in _native_kquant_types()
    monkeypatch.delattr(superl8, "linear_tq34s", raising=False)
    assert "TQ3_4S" not in _native_kquant_types()


def test_gguf_reader_patch_registers_type46_idempotent():
    from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

    _patch_gguf_tq34s()
    _patch_gguf_tq34s()  # idempotent
    tq = GGMLQuantizationType(46)
    assert tq.name == "TQ3_4S"
    assert GGMLQuantizationType["TQ3_4S"] == tq
    assert GGML_QUANT_SIZES[tq] == (32, 16)


def test_dequant_ggml_routes_tq34s_to_reference(monkeypatch):
    import gguf

    if _tq34s is None:  # pragma: no cover
        pytest.skip("no tq34s reference on PYTHONPATH")

    calls = []
    orig = gguf.dequantize
    monkeypatch.setattr(gguf, "dequantize", lambda *a, **k: calls.append(a) or orig(*a, **k))
    raw = _tq34s_bytes(2, 64)
    out = _dequant_ggml(raw, "TQ3_4S")
    assert not calls  # TQ3_4S must NOT go through gguf.dequantize
    oracle = _tq34s.dequantize_tq34s_bytes(raw, 64)
    assert _sqnr(oracle, out) >= 40 and _cos(oracle, out) >= 0.999


# ── 2. dequant_kquant for the tq3_4s codebook ───────────────────────────────
@requires_tq34s
def test_dequant_kquant_tq34s_matches_reference():
    out, inf = 3, 128
    raw = _tq34s_bytes(out, inf, seed=7)
    oracle = _tq34s.dequantize_tq34s_bytes(raw, inf)  # fp32 [out, in]
    qt = QTensor(
        torch.from_numpy(raw.copy()),
        None,
        scheme="gguf_kquant",
        codebook="tq3_4s",
        group_size=32,
    )
    deq = dequant_kquant(qt).float().numpy()
    assert deq.shape == oracle.shape == (out, inf)
    assert _sqnr(oracle, deq) >= 40, f"SQNR {_sqnr(oracle, deq):.2f} dB < 40 dB"
    assert _cos(oracle, deq) >= 0.999, f"cos {_cos(oracle, deq):.6f} < 0.999"


@requires_tq34s
def test_dequant_kquant_tq34s_e3m5_zero_scale():
    """The E3M5==0 (scale 0.0) review-gate case: dequant must still match the
    oracle bit-for-bit (zero scale contributes exactly 0.0 to its 8-group)."""
    out, inf = 2, 96
    raw = _tq34s_bytes(out, inf, seed=11, zero_scale_at=3)
    oracle = _tq34s.dequantize_tq34s_bytes(raw, inf)
    qt = QTensor(
        torch.from_numpy(raw.copy()),
        None,
        scheme="gguf_kquant",
        codebook="tq3_4s",
        group_size=32,
    )
    deq = dequant_kquant(qt).float().numpy()
    assert _sqnr(oracle, deq) >= 40, f"SQNR {_sqnr(oracle, deq):.2f} dB < 40 dB"
    assert _cos(oracle, deq) >= 0.999, f"cos {_cos(oracle, deq):.6f} < 0.999"
    assert deq.shape == oracle.shape == (out, inf)


@requires_tq34s
def test_dequant_kquant_in_features_derivation():
    """in = (row_bytes / 16) * 32 — a 32-value block packs 16 B; an odd block
    count must still dequant to the full logical in-dim."""
    for inf in (64, 96, 160):  # 2, 3, 5 blocks
        raw = _tq34s_bytes(1, inf, seed=3)
        assert raw.shape[1] == (inf // 32) * 16
        qt = QTensor(
            torch.from_numpy(raw.copy()),
            None,
            scheme="gguf_kquant",
            codebook="tq3_4s",
            group_size=32,
        )
        assert dequant_kquant(qt).shape == (1, inf)


@requires_tq34s
def test_merge_and_linear_consume_tq34s_qtensors():
    """The dense-assembly merge path must pass tq3_4s QTensors through natively
    (same codebook + row width → row concat, group_size preserved) and the
    LinearW8A8 seam must derive the correct in_features for the codebook."""
    from superl8serve.layers.linear import LinearW8A8
    from superl8serve.models.weights import merge_qtensor

    inf, nh, nkv, hd = 128, 2, 2, 32
    q = QTensor(torch.from_numpy(_tq34s_bytes(2 * nh * hd, inf)), None,
                scheme="gguf_kquant", codebook="tq3_4s", group_size=32)
    k = QTensor(torch.from_numpy(_tq34s_bytes(nkv * hd, inf)), None,
                scheme="gguf_kquant", codebook="tq3_4s", group_size=32)
    v = QTensor(torch.from_numpy(_tq34s_bytes(nkv * hd, inf)), None,
                scheme="gguf_kquant", codebook="tq3_4s", group_size=32)
    m = merge_qtensor([q, k, v])
    assert m.scheme == "gguf_kquant" and m.codebook == "tq3_4s" and m.group_size == 32
    assert m.data.shape[0] == 2 * nh * hd + 2 * nkv * hd
    lin = LinearW8A8(m)
    assert lin.in_features == inf and lin.out_features == m.data.shape[0]


@requires_tq34s
def test_superl8_linear_dispatch_routes_tq34s():
    """superl8.linear on a tq3_4s gguf_kquant QTensor routes to the linear_tq34s
    reference (the pre-kernel correctness path, superl8#272)."""
    import superl8

    inf, out = 96, 8
    raw = _tq34s_bytes(out, inf, seed=5)
    qt = QTensor(torch.from_numpy(raw.copy()), None, scheme="gguf_kquant",
                 codebook="tq3_4s", group_size=32)
    x = torch.randn(4, inf)
    y = superl8.linear(x, qt)
    w = torch.from_numpy(_tq34s.dequantize_tq34s_bytes(raw, inf)).float()
    ref = x @ w.t()
    assert y.shape == ref.shape
    assert _sqnr(ref.numpy(), y.numpy()) >= 40


# ── 3. dense-assembly tensor mapping (real 27B GGUF, header-only) ────────────
@requires_gguf
def test_qwen38_27b_tensor_set_maps_natively():
    """All 866 tensors (506 TQ3_4S + 360 F32 norms) read through the native
    loader and map to HF names — the superl8#270 acceptance item 2 shape. Header
    only; no weight bytes are copied (read-only)."""
    import collections

    reader = _open_gguf(_TQ38_GGUF)
    tensors = list(reader.tensors)
    assert len(tensors) == 866
    hist = collections.Counter(int(t.tensor_type) for t in tensors)
    assert hist == {46: 506, 0: 360}  # 506 TQ3_4S resident + F32 norms
    assert reader.fields["general.architecture"].parts[-1].tobytes().decode() == "qwen35"

    unmapped = [t.name for t in tensors if gguf_name_to_hf(t.name, arch="qwen35") is None]
    assert not unmapped, f"unmapped tensors: {unmapped[:5]}"

    # per-layer structure: 48 DeltaNet (fused attn_qkv) + 17 full-attn q/k/v
    # (16 main + 1 MTP), 65 ffn, MTP head present.
    kinds = collections.Counter()
    for t in tensors:
        if t.name.startswith("blk."):
            kinds[t.name.split(".")[2]] += 1
    assert kinds["attn_qkv"] == 48
    assert kinds["attn_q"] == 17
    assert kinds["ffn_gate"] == 65
    assert sum(1 for t in tensors if ".nextn.eh_proj." in t.name) == 1


@requires_gguf
def test_qwen38_27b_config_hybrid_schedule():
    reader = _open_gguf(_TQ38_GGUF)
    cfg = gguf_config(_TQ38_GGUF, _reader=reader)
    assert cfg.arch == "qwen3_5"
    assert cfg.num_hidden_layers == 64
    assert cfg.hidden_size == 5120
    assert cfg.num_attention_heads == 24 and cfg.num_key_value_heads == 4
    assert cfg.max_position_embeddings == 262144  # the 262k gate
    assert cfg.num_mtp_layers == 1
    assert not cfg.is_moe()  # dense 27B — no experts
    from collections import Counter

    kinds = Counter(cfg.attention_kind(i) for i in range(cfg.num_hidden_layers))
    assert kinds == {"linear": 48, "full": 16}
    assert cfg.extra["linear_num_key_heads"] == 16
    assert cfg.extra["linear_num_value_heads"] == 48
    assert cfg.extra["linear_key_head_dim"] == 128
    assert cfg.extra["linear_value_head_dim"] == 128
    assert cfg.extra["linear_conv_kernel_dim"] == 4


@requires_gguf
def test_qwen38_27b_builder_keys_after_remap():
    """The mapped 866 keys, run through the qwen3_5 hybrid remap chain
    (_remap_hybrid_qwen35 → _remap_qwen3_next → _restore_hf_qwen35_weights),
    yield exactly the key set the dense builder reads for a DeltaNet layer, a
    full-attn layer, and the MTP/static set. TQ3_4S linears flow through as
    gguf_kquant — this PR wires that; the builder consumes them via to_qtensor
    (passthrough) / merge_qtensor (row concat)."""
    reader = _open_gguf(_TQ38_GGUF)
    cfg = gguf_config(_TQ38_GGUF, _reader=reader)
    sd = {
        gguf_name_to_hf(t.name, arch="qwen35"): torch.ones(1)
        for t in reader.tensors
        if gguf_name_to_hf(t.name, arch="qwen35") is not None
    }
    assert len(sd) == 866
    sd = _remap_hybrid_qwen35(sd, cfg)
    sd = _remap_qwen3_next(sd, cfg)
    sd, tiled = _restore_hf_qwen35_weights(sd, cfg)
    assert tiled is True  # 16 key heads != 48 value heads → tiled DeltaNet V

    delta = {
        "model.layers.0.linear_attn.qkv_proj.weight",
        "model.layers.0.linear_attn.z_proj.weight",
        "model.layers.0.linear_attn.dt_proj.weight",
        "model.layers.0.linear_attn.beta_proj.weight",
        "model.layers.0.linear_attn.out_proj.weight",
        "model.layers.0.linear_attn.conv_weight",
        "model.layers.0.linear_attn.A_log",
        "model.layers.0.linear_attn.dt_bias",
        "model.layers.0.linear_attn.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    }
    full = {
        "model.layers.3.self_attn.q_proj.weight",
        "model.layers.3.self_attn.k_proj.weight",
        "model.layers.3.self_attn.v_proj.weight",
        "model.layers.3.self_attn.o_proj.weight",
        "model.layers.3.self_attn.q_norm.weight",
        "model.layers.3.self_attn.k_norm.weight",
        "model.layers.3.mlp.gate_proj.weight",
        "model.layers.3.mlp.up_proj.weight",
        "model.layers.3.mlp.down_proj.weight",
    }
    static = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    mtp = {"mtp.fc.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.pre_fc_norm_embedding.weight",
           "mtp.norm.weight"}
    assert delta <= set(sd), delta - set(sd)
    assert full <= set(sd), full - set(sd)
    assert static <= set(sd), static - set(sd)
    assert mtp <= set(sd), mtp - set(sd)
