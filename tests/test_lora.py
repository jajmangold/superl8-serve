# SPDX-License-Identifier: MIT
"""Tests for LoRA / QLoRA adapter machinery (issue #129)."""

import pytest

pytest.importorskip("superl8")

import torch
import torch.nn as nn

from superl8serve.layers.linear import LinearW8A8
from superl8serve.training.lora import (
    LoRAConfig,
    LoRALayer,
    inject_lora,
    merge_lora,
    unmerge_lora,
)
from superl8serve.training.qlora import dequantize_adapter_nf4, quantize_adapter_nf4


# ── helpers ────────────────────────────────────────────────────────────────


def _make_fake_qtensor(out: int, _in: int) -> object:
    """Build a fake per_row_i8 QTensor on CUDA."""
    from superl8 import QTensor
    from superl8.quant.core import quantize_int8_rowwise

    w = torch.randn(out, _in, device="cuda", dtype=torch.float16)
    q, s = quantize_int8_rowwise(w)
    return QTensor(q.contiguous(), s.squeeze(-1).float().contiguous(), scheme="per_row_i8")


def _make_linear_w8a8(out: int, _in: int) -> LinearW8A8:
    return LinearW8A8(_make_fake_qtensor(out, _in))


# ── LoRALayer forward shapes ──────────────────────────────────────────────


def test_lora_layer_shapes():
    """LoRALayer wrapping a LinearW8A8 produces correct output shape."""
    B, S, out_f, in_f = 2, 8, 32, 64
    lin = _make_linear_w8a8(out_f, in_f)
    lora = LoRALayer(lin, r=4, alpha=8.0).cuda().half()

    x = torch.randn(B, S, in_f, device="cuda", dtype=torch.float16)
    out = lora(x)
    assert out.shape == (B, S, out_f)


def test_lora_layer_nn_linear():
    """LoRALayer wrapping an nn.Linear works."""
    B, S, out_f, in_f = 2, 8, 32, 64
    lin = nn.Linear(in_f, out_f, bias=False).cuda().half()
    lora = LoRALayer(lin, r=4).cuda().half()

    x = torch.randn(B, S, in_f, device="cuda", dtype=torch.float16)
    out = lora(x)
    assert out.shape == (B, S, out_f)


# ── gradient flow ─────────────────────────────────────────────────────────


def test_lora_grad_flow():
    """Gradients flow through LoRA adapters but NOT through the base int8 weight."""
    lin = _make_linear_w8a8(32, 64)
    lora = LoRALayer(lin, r=4, alpha=8.0).cuda().half()

    x = torch.randn(2, 8, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    out = lora(x)
    loss = out.sum()
    loss.backward()

    assert lora.lora_A.grad is not None and not torch.isnan(lora.lora_A.grad).any()
    assert lora.lora_B.grad is not None and not torch.isnan(lora.lora_B.grad).any()
    assert x.grad is not None and not torch.isnan(x.grad).any()


def test_lora_base_frozen():
    """The base module's parameters should not require grad."""
    lin = nn.Linear(64, 32, bias=False).cuda().half()
    lora = LoRALayer(lin, r=4).cuda().half()

    for p in lora.base.parameters():
        assert not p.requires_grad


# ── merge / unmerge ───────────────────────────────────────────────────────


def test_merge_unmerge_roundtrip():
    """merge() then unmerge() produces the same output as before merge."""
    torch.manual_seed(42)
    lin = nn.Linear(64, 32, bias=False).cuda().half()
    lora = LoRALayer(lin, r=4, alpha=8.0).cuda().half()

    x = torch.randn(2, 8, 64, device="cuda", dtype=torch.float16)
    out_before = lora(x)

    lora.merge()
    assert lora._merged
    out_merged = lora(x)
    assert torch.allclose(out_before, out_merged, atol=1e-3, rtol=1e-3)

    lora.unmerge()
    assert not lora._merged
    out_after = lora(x)
    assert torch.allclose(out_before, out_after, atol=1e-5, rtol=1e-5)


# ── gradcheck (fp64, tiny) ────────────────────────────────────────────────


def test_lora_gradcheck_fp64():
    """LoRALayer forward-backward passes torch.autograd.gradcheck in fp64."""
    torch.manual_seed(0)
    out_f, in_f, r = 4, 8, 2
    lin = nn.Linear(in_f, out_f, bias=False).cuda().double()

    lora = LoRALayer(lin, r=r, alpha=4.0).cuda().double()
    lora.lora_A.data.normal_(0, 0.1)
    lora.lora_B.data.normal_(0, 0.1)

    x = torch.randn(1, 3, in_f, device="cuda", dtype=torch.float64, requires_grad=True)

    ok = torch.autograd.gradcheck(
        lambda x_: lora(x_),
        x,
        eps=1e-4,
        atol=1e-3,
        rtol=1e-3,
        fast_mode=True,
    )
    assert ok


# ── inject_lora ───────────────────────────────────────────────────────────


class _TinyModel(nn.Module):
    """A tiny model with LinearW8A8 layers (simulates a 1-layer Qwen3)."""

    def __init__(self):
        super().__init__()
        h, i = 32, 64
        self.self_attn = nn.ModuleDict(
            {
                "qkv_proj": _make_linear_w8a8(h * 3, h),
                "o_proj": _make_linear_w8a8(h, h * 3),
            }
        )
        self.mlp = nn.ModuleDict(
            {
                "gate_up_proj": _make_linear_w8a8(i * 2, h),
                "down_proj": _make_linear_w8a8(h, i * 2),
            }
        )

    def forward(self, x):
        x = self.self_attn["qkv_proj"](x)
        x = self.self_attn["o_proj"](x)
        x = self.mlp["gate_up_proj"](x)
        x = self.mlp["down_proj"](x)
        return x


def _build_tiny_model() -> nn.Module:
    return _TinyModel().cuda()


def test_inject_lora():
    """inject_lora wraps only the specified target modules."""
    model = _build_tiny_model()
    cfg = LoRAConfig(r=4, alpha=8.0, target_modules=["qkv_proj", "o_proj"])
    model = inject_lora(model, cfg)

    n_lora = sum(1 for _ in model.modules() if isinstance(_, LoRALayer))
    assert n_lora == 2  # qkv_proj + o_proj

    # gate_up_proj and down_proj should still be LinearW8A8
    assert isinstance(model.mlp["gate_up_proj"], LinearW8A8)
    assert isinstance(model.mlp["down_proj"], LinearW8A8)


def test_inject_lora_forward():
    """Model with injected LoRA layers produces correct shapes and gradient flow."""
    model = _build_tiny_model()
    cfg = LoRAConfig(
        r=4, alpha=8.0, target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    )
    model = inject_lora(model, cfg)

    x = torch.randn(2, 4, 32, device="cuda", dtype=torch.float16)
    out = model(x)
    assert out.shape == (2, 4, 32)

    loss = out.sum()
    loss.backward()

    for name, mod in model.named_modules():
        if isinstance(mod, LoRALayer):
            assert mod.lora_A.grad is not None, f"no grad for {name}.lora_A"
            assert mod.lora_B.grad is not None, f"no grad for {name}.lora_B"


def test_merge_unmerge_lora_full_linear():
    """merge_lora and unmerge_lora work on a full nn.Linear-based model (no int8 noise)."""

    class _FullLinearModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_a = nn.Linear(32, 64, bias=False)
            self.proj_b = nn.Linear(64, 32, bias=False)

        def forward(self, x):
            return self.proj_b(self.proj_a(x))

    model = _FullLinearModel().cuda().half()
    cfg = LoRAConfig(r=4, alpha=8.0, target_modules=["proj_a", "proj_b"])
    model = inject_lora(model, cfg)

    x = torch.randn(2, 4, 32, device="cuda", dtype=torch.float16)
    out_before = model(x)

    model = merge_lora(model)
    out_merged = model(x)
    assert torch.allclose(out_before, out_merged, atol=1e-3, rtol=1e-3)

    model = unmerge_lora(model)
    out_after = model(x)
    assert torch.allclose(out_before, out_after, atol=1e-5, rtol=1e-5)


# ── QLoRA NF4 roundtrip ───────────────────────────────────────────────────


def test_nf4_roundtrip():
    """NF4 quantize-dequantize roundtrip of a small adapter weight."""
    torch.manual_seed(1)
    w = torch.randn(16, 32, device="cuda", dtype=torch.float16)
    codes, scale = quantize_adapter_nf4(w)
    w_recovered = dequantize_adapter_nf4(codes, scale, w.shape)
    assert w_recovered.shape == w.shape
    assert w_recovered.dtype == torch.float16
    # NF4 is lossy; check that the recovered weight is reasonable
    err = (w.float() - w_recovered.float()).abs().mean()
    assert err < 0.5, f"NF4 reconstruction error too high: {err:.4f}"


def test_nf4_roundtrip_odd_size():
    """NF4 roundtrip with an odd number of elements (requires padding)."""
    torch.manual_seed(2)
    w = torch.randn(7, 19, device="cuda", dtype=torch.float16)
    codes, scale = quantize_adapter_nf4(w)
    w_recovered = dequantize_adapter_nf4(codes, scale, w.shape)
    assert w_recovered.shape == w.shape
    err = (w.float() - w_recovered.float()).abs().mean()
    assert err < 0.5
