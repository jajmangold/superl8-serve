# SPDX-License-Identifier: MIT
"""Qwen3.5 loader compatibility: a round-tripped `.superl8` that lost the DeltaNet
weight-name remap (separate-projection HF layout) and the `extra` dims must still
build — via serve-native/HF-alias resolution and shape-derived head dims."""
import pytest

pytest.importorskip("superl8")
import torch

from superl8serve.models.qwen3_5 import _la_dims, _la_weight


def _hf_native_deltanet_sd(la="model.layers.0.linear_attn", H=1024, nv=16, kd=128, vd=128, ck=4):
    """A DeltaNet layer with the raw HF *separate*-projection names (as Qwen3.5-0.8B
    ships un-remapped), plain tensors."""
    qkv_out = 2 * nv * kd + nv * vd
    return {
        f"{la}.in_proj_qkv.weight": torch.zeros(qkv_out, H),
        f"{la}.in_proj_z.weight": torch.zeros(nv * vd, H),
        f"{la}.in_proj_b.weight": torch.zeros(nv, H),
        f"{la}.in_proj_a.weight": torch.zeros(nv, H),
        f"{la}.conv1d.weight": torch.zeros(qkv_out, 1, ck),
        f"{la}.A_log": torch.zeros(nv),
        f"{la}.dt_bias": torch.zeros(nv),
        f"{la}.norm.weight": torch.zeros(vd),
        f"{la}.out_proj.weight": torch.zeros(H, nv * vd),
    }


def test_la_weight_resolves_hf_aliases():
    sd = _hf_native_deltanet_sd()
    la = "model.layers.0.linear_attn"
    assert _la_weight(sd, la, "qkv_proj.weight", "in_proj_qkv.weight").shape[0] == 2 * 16 * 128 + 16 * 128
    assert _la_weight(sd, la, "beta_proj.weight", "in_proj_b.weight").shape[0] == 16  # b -> beta
    assert _la_weight(sd, la, "dt_proj.weight", "in_proj_a.weight").shape[0] == 16    # a -> dt
    # native name present -> returned directly, no alias needed
    assert _la_weight(sd, la, "out_proj.weight").shape == (1024, 16 * 128)
    with pytest.raises(KeyError):
        _la_weight(sd, la, "nope.weight")


def test_la_dims_derives_from_weight_shapes_when_extra_dropped():
    sd = _hf_native_deltanet_sd(nv=16, kd=128, vd=128, ck=4)
    d = _la_dims(sd, "model.layers.0.linear_attn", {})  # extra dropped by round-trip
    assert d["linear_num_value_heads"] == 16
    assert d["linear_num_key_heads"] == 16
    assert d["linear_value_head_dim"] == 128
    assert d["linear_key_head_dim"] == 128
    assert d["linear_conv_kernel_dim"] == 4


def test_la_dims_prefers_explicit_extra():
    sd = _hf_native_deltanet_sd()
    d = _la_dims(sd, "model.layers.0.linear_attn", {"linear_num_key_heads": 8})
    assert d["linear_num_key_heads"] == 8  # explicit wins over derivation
