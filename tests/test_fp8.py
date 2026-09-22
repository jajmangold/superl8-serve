# SPDX-License-Identifier: MIT
"""FP8 dequant tests — round-trip fp32 -> fp8 -> dequant recovers values within
fp8's own precision, for static, per-channel, block, and MX/UE8M0 block scale
layouts."""
import math

import pytest
import torch

from superl8serve.fp8 import dequantize_fp8, dequantize_mxfp8, fp8_scale_name, is_fp8

fp8 = pytest.importorskip("torch").float8_e4m3fn


def _quant_static(w, scale):
    """Quantize fp32 w to fp8 with a single scalar scale (real = fp8 * scale)."""
    return (w / scale).to(fp8)


def test_static_scalar_roundtrip():
    torch.manual_seed(0)
    w = torch.randn(64, 128) * 0.1
    scale = w.abs().max() / 448.0                       # e4m3 max ~448
    q = _quant_static(w, scale)
    out = dequantize_fp8(q, scale.reshape(1))
    assert is_fp8(q)
    assert torch.cosine_similarity(out.flatten(), w.flatten(), dim=0) > 0.99


def test_per_channel_roundtrip():
    torch.manual_seed(1)
    w = torch.randn(32, 96)
    scale = w.abs().amax(dim=1, keepdim=True) / 448.0   # [O,1]
    q = (w / scale).to(fp8)
    out = dequantize_fp8(q, scale)                       # per-channel
    assert out.shape == w.shape
    assert torch.cosine_similarity(out.flatten(), w.flatten(), dim=0) > 0.99


def test_block_roundtrip():
    torch.manual_seed(2)
    O, I, b = 256, 256, 128
    w = torch.randn(O, I)
    # per-128x128-block scale
    grid = torch.zeros(O // b, I // b)
    for bi in range(O // b):
        for bj in range(I // b):
            grid[bi, bj] = w[bi*b:(bi+1)*b, bj*b:(bj+1)*b].abs().max() / 448.0
    full = grid.repeat_interleave(b, 0).repeat_interleave(b, 1)
    q = (w / full).to(fp8)
    out = dequantize_fp8(q, grid, block_size=[b, b])
    assert out.shape == w.shape
    assert torch.cosine_similarity(out.flatten(), w.flatten(), dim=0) > 0.99


def test_block_ragged_tail_clips():
    O, I, b = 130, 130, 128                              # not a multiple of 128
    w = torch.randn(O, I)
    grid = torch.ones(2, 2)                              # ceil(130/128)=2
    out = dequantize_fp8(w.to(fp8), grid, block_size=[b, b])
    assert out.shape == (O, I)                           # clipped, no shape blow-up


def test_mxfp8_power_of_two_formula():
    """Pin the UE8M0 formula independent of quantization round-off: byte 127 (the
    e8m0 bias) -> multiplier 1.0, 128 -> 2.0, 126 -> 0.5, 135 -> 256.0."""
    q = torch.tensor([[2.0]]).to(fp8)                    # e4m3 represents 2.0 exactly
    for byte, mult in ((127, 1.0), (128, 2.0), (126, 0.5), (135, 256.0)):
        scale = torch.tensor([[byte]], dtype=torch.uint8)
        out = dequantize_mxfp8(q, scale, block_size=[1, 1])
        assert torch.allclose(out, torch.tensor([[2.0 * mult]]))


def test_mxfp8_roundtrip():
    """MXFP8 (UE8M0): fp32 -> e4m3 weight + per-128x128-block power-of-2 scale ->
    dequantize_mxfp8 recovers the original values."""
    torch.manual_seed(4)
    O, I, b = 256, 256, 128
    w = torch.randn(O, I) * 0.1
    q = torch.empty(O, I, dtype=fp8)
    grid = torch.empty(O // b, I // b, dtype=torch.uint8)
    for bi in range(O // b):
        for bj in range(I // b):
            tile = w[bi*b:(bi+1)*b, bj*b:(bj+1)*b]
            amax = tile.abs().max().item()
            exp = math.ceil(math.log2(amax / 448.0))     # smallest pow-of-2 s.t. tile/2**exp fits
            grid[bi, bj] = exp + 127
            q[bi*b:(bi+1)*b, bj*b:(bj+1)*b] = (tile / (2.0 ** exp)).to(fp8)
    out = dequantize_mxfp8(q, grid, block_size=[b, b])
    assert out.shape == w.shape
    assert torch.cosine_similarity(out.flatten(), w.flatten(), dim=0) > 0.99


def test_mxfp8_ragged_tail_clips():
    O, I, b = 130, 130, 128                              # not a multiple of 128
    w = torch.randn(O, I).to(fp8)
    grid = torch.full((2, 2), 127, dtype=torch.uint8)    # multiplier 1.0 everywhere
    out = dequantize_mxfp8(w, grid, block_size=[b, b])
    assert out.shape == (O, I)                           # clipped, no shape blow-up


def test_scale_name_pairing():
    names = {"m.q_proj.weight", "m.q_proj.weight_scale", "m.o_proj.weight",
             "m.o_proj.weight_scale_inv"}
    assert fp8_scale_name("m.q_proj.weight", names) == "m.q_proj.weight_scale"
    assert fp8_scale_name("m.o_proj.weight", names) == "m.o_proj.weight_scale_inv"
    assert fp8_scale_name("m.absent.weight", names) is None


def test_scale_name_pairing_hc_naming():
    """DeepSeek-V4-Flash / MiniMax-M3-MXFP8 pair a weight with its scale under a
    non-standard name (hc_attn_scale / hc_ffn_scale / hc_head_scale / plain scale)
    instead of weight_scale_inv. fp8_scale_name tries each candidate and returns
    whichever is actually present, so it works regardless of which category a given
    weight falls into."""
    names = {
        "model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj.hc_attn_scale",
        "model.layers.0.mlp.down_proj.weight", "model.layers.0.mlp.down_proj.hc_ffn_scale",
        "lm_head.weight", "lm_head.hc_head_scale",
        "model.layers.0.self_attn.v_proj.weight", "model.layers.0.self_attn.v_proj.scale",
    }
    assert fp8_scale_name("model.layers.0.self_attn.q_proj.weight", names) \
        == "model.layers.0.self_attn.q_proj.hc_attn_scale"
    assert fp8_scale_name("model.layers.0.mlp.down_proj.weight", names) \
        == "model.layers.0.mlp.down_proj.hc_ffn_scale"
    assert fp8_scale_name("lm_head.weight", names) == "lm_head.hc_head_scale"
    assert fp8_scale_name("model.layers.0.self_attn.v_proj.weight", names) \
        == "model.layers.0.self_attn.v_proj.scale"


def test_convert_quantizes_fp8_weight_and_drops_scale():
    """quantize_state_dict: an fp8 linear + its scale -> one int8 QTensor; the scale
    tensor is consumed (not emitted), and the int8 result tracks the real values."""
    from superl8serve.convert import quantize_state_dict

    torch.manual_seed(3)
    w = torch.randn(64, 128)
    scale = w.abs().amax(dim=1, keepdim=True) / 448.0
    q = (w / scale).to(fp8)
    sd = {"model.layers.0.self_attn.q_proj.weight": q,
          "model.layers.0.self_attn.q_proj.weight_scale": scale}
    out = quantize_state_dict(sd, weight_bits=8)
    assert "model.layers.0.self_attn.q_proj.weight" in out
    assert "model.layers.0.self_attn.q_proj.weight_scale" not in out   # consumed
    qt = out["model.layers.0.self_attn.q_proj.weight"]
    assert qt.scheme == "per_row_i8"
    # int8 dequant ≈ real weights
    deq = qt.data.float() * qt.scale.reshape(-1, 1)
    assert torch.cosine_similarity(deq.flatten(), w.flatten(), dim=0) > 0.98


def test_convert_quantizes_mxfp8_weight_hc_scale():
    """quantize_state_dict(fp8_scale_fmt="ue8m0"): an e4m3 weight + its UE8M0 block
    scale under the DeepSeek-V4-Flash hc_attn_scale naming -> int8 QTensor via the
    power-of-2 MX dequant, not the fp32-multiply block path (a wrong-path selection
    would silently corrupt the reconstructed values, since a UE8M0 byte like 128 is
    nonsense as an fp32 multiplier)."""
    from superl8serve.convert import quantize_state_dict

    torch.manual_seed(5)
    O, I, b = 64, 128, 128
    w = torch.randn(O, I) * 0.1
    amax = w.abs().max().item()
    exp = math.ceil(math.log2(amax / 448.0))
    q = (w / (2.0 ** exp)).to(fp8)
    grid = torch.full((1, 1), exp + 127, dtype=torch.uint8)
    sd = {"model.layers.0.self_attn.q_proj.weight": q,
          "model.layers.0.self_attn.q_proj.hc_attn_scale": grid}
    out = quantize_state_dict(sd, weight_bits=8, fp8_block_size=[b, b], fp8_scale_fmt="ue8m0")
    assert "model.layers.0.self_attn.q_proj.hc_attn_scale" not in out      # consumed
    qt = out["model.layers.0.self_attn.q_proj.weight"]
    deq = qt.data.float() * qt.scale.reshape(-1, 1)
    assert torch.cosine_similarity(deq.flatten(), w.flatten(), dim=0) > 0.98
