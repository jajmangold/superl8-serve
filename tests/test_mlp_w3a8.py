# SPDX-License-Identifier: MIT
"""Uniform 3-bit MLP path (issue #181 VRAM/context/batch lever).

Converter emits `per_group_w3a8` for MLP gate/up when `mlp_gate_up_3bit=True`
(down_proj + everything else keep base precision, per the imatrix map); the
serve `LinearW8A8` routes it through `superl8.linear` -> `gemm_decode_w3a8` at decode
parity with int4. These tests are the serve-side integration gate for the hook.
"""
import pytest
import torch

from superl8 import QTensor
from superl8serve.convert import quantize_state_dict, quantize_weight_w3a8
from superl8serve.layers.linear import LinearW8A8


def _superl8_has_w3a8() -> bool:
    """The per_group_w3a8 path needs the superl8 3-bit codec + kernel wrapper (superl8 PR
    #184). The serve light-lane CI runs against a prebuilt baseline superl8 that may not
    have it yet, so these tests skip until that superl8 lands and the image rebuilds —
    a real cross-repo dependency, not a silent pass."""
    try:
        import superl8
        from superl8.quant.lowbit import pack_w3a8_bitplanes, quantize_w3a8  # noqa: F401

        return hasattr(superl8, "linear_w3a8")
    except ImportError:
        return False


_HAS_W3 = _superl8_has_w3a8()
_CUDA = torch.cuda.is_available()

# Whole module depends on the superl8 per_group_w3a8 support (superl8 PR #184).
pytestmark = pytest.mark.skipif(
    not _HAS_W3, reason="needs superl8 per_group_w3a8 support (superl8 PR #184)"
)


def test_converter_emits_w3a8_for_mlp_gate_up_only():
    """mlp_gate_up_3bit routes gate/up -> per_group_w3a8; down/attn keep base scheme."""
    sd = {
        "model.layers.0.mlp.gate_proj.weight": torch.randn(256, 512),
        "model.layers.0.mlp.up_proj.weight": torch.randn(256, 512),
        "model.layers.0.mlp.down_proj.weight": torch.randn(512, 256),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(512, 512),
        "model.norm.weight": torch.randn(512),
    }
    q = quantize_state_dict(sd, weight_bits=4, group_size=128, mlp_gate_up_3bit=True)
    assert q["model.layers.0.mlp.gate_proj.weight"].scheme == "per_group_w3a8"
    assert q["model.layers.0.mlp.up_proj.weight"].scheme == "per_group_w3a8"
    # down_proj + attention stay int4 (protected), norm stays raw.
    assert q["model.layers.0.mlp.down_proj.weight"].scheme == "per_group_i4"
    assert q["model.layers.0.self_attn.q_proj.weight"].scheme == "per_group_i4"
    assert q["model.norm.weight"].scheme == "raw"
    # bytes: w3a8 stores 3 int32 per 32 in-elems.
    gu = q["model.layers.0.mlp.gate_proj.weight"]
    assert gu.data.dtype == torch.int32 and gu.data.shape == (256, (512 // 32) * 3)


def test_linear_w8a8_in_features_for_w3a8():
    w = torch.randn(256, 512)
    qt = quantize_weight_w3a8(w, 128)
    lin = LinearW8A8(qt)
    assert lin.out_features == 256 and lin.in_features == 512


@pytest.mark.skipif(not (_CUDA and _HAS_W3), reason="needs CUDA + superl8.linear_w3a8")
def test_linear_w8a8_w3a8_forward_matches_fp():
    dev = "cuda"
    w = (torch.randn(4096, 5120) * 0.1)
    qt = quantize_weight_w3a8(w, 128)
    qt = QTensor(qt.data.to(dev), qt.scale.to(dev), scheme=qt.scheme,
                 group_size=qt.group_size, codebook=qt.codebook)
    lin = LinearW8A8(qt)
    x = torch.randn(2, 5120, device=dev, dtype=torch.float16)
    y = lin(x)  # dp4a path via superl8.linear -> gemm_decode_w3a8
    ref = x.float() @ w.float().to(dev).t()
    cos = torch.nn.functional.cosine_similarity(y.flatten().float(), ref.flatten().float(), dim=0)
    assert y.shape == (2, 4096)
    # uniform-3-bit RTN floor (~0.977 on Gaussian); the kernel adds no error beyond that.
    assert float(cos) >= 0.97, f"w3a8 serve forward cos {float(cos):.4f} below 3-bit floor"
