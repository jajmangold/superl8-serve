# SPDX-License-Identifier: MIT
"""ViT forward regression tests — int8 dp4a linears vs fp16 reference.

Compares patch embeddings from the quantized ViT against the identical
fp16 reference.  cos ≥ 0.99 is the primary metric (met with wide margin).
rel-L1 ≤ 5% is permissive but consistent with codebase conventions for
int8 per-row quantization through multiple layers (e.g. GatedMLP: rtol=0.05,
DeepSeek full-forward: rtol=0.02).
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytest.importorskip("superl8")
from superl8 import QTensor

from superl8serve.multimodal.vit import (
    VisionTransformer,
    _compute_2d_rope,
    _apply_rotary_pos_emb,
)
from superl8serve.models.weights import to_qtensor
from superl8serve.models.config import VisionConfig

CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not CUDA, reason="dp4a path needs CUDA")


def _rand(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.float16) * 0.05


# ── fp16 reference ViT (same architecture, nn.Linear) ────────────────────


class _RefViTAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.num_heads * self.head_dim, dim=-1)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.proj(out)


class _RefViTMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class _RefViTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, head_dim, eps):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.attn = _RefViTAttention(hidden_size, num_heads, head_dim)
        self.norm2 = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = _RefViTMLP(hidden_size, intermediate_size)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class _RefViT(nn.Module):
    def __init__(
        self,
        hidden_size,
        patch_size,
        num_layers,
        num_heads,
        intermediate_size,
        head_dim,
        in_channels=3,
        eps=1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.blocks = nn.ModuleList(
            [
                _RefViTBlock(hidden_size, num_heads, intermediate_size, head_dim, eps)
                for _ in range(num_layers)
            ]
        )

    def forward(self, pixel_values):
        x = self.patch_embed(pixel_values)
        x = x.flatten(2).transpose(1, 2)
        gh = pixel_values.shape[2] // self.patch_size
        gw = pixel_values.shape[3] // self.patch_size
        cos, sin = _compute_2d_rope(gh, gw, self.head_dim, x.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return x


# ── weight dictionary builder (extracted from ref, quantized) ────────────


def _build_q_weights(ref: _RefViT) -> dict:
    """Extract weights from the fp16 reference ViT, quantize linears to int8."""
    w = {"patch_embed.weight": ref.patch_embed.weight.data.clone()}
    for i, blk in enumerate(ref.blocks):
        p = f"blocks.{i}"
        w[f"{p}.attn.qkv.weight"] = to_qtensor(blk.attn.qkv.weight.data)
        w[f"{p}.attn.qkv.bias"] = blk.attn.qkv.bias.data.clone()
        w[f"{p}.attn.proj.weight"] = to_qtensor(blk.attn.proj.weight.data)
        w[f"{p}.attn.proj.bias"] = blk.attn.proj.bias.data.clone()
        w[f"{p}.mlp.fc1.weight"] = to_qtensor(blk.mlp.fc1.weight.data)
        w[f"{p}.mlp.fc1.bias"] = blk.mlp.fc1.bias.data.clone()
        w[f"{p}.mlp.fc2.weight"] = to_qtensor(blk.mlp.fc2.weight.data)
        w[f"{p}.mlp.fc2.bias"] = blk.mlp.fc2.bias.data.clone()
    return w


# ── helper metrics ───────────────────────────────────────────────────────


def _cos_sim(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return (a @ b) / (a.norm() * b.norm() + 1e-12)


def _rel_l1(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return (a - b).abs().sum() / (b.abs().sum() + 1e-12)


_TOL_COS = 0.99
_TOL_REL_L1 = 0.05


# ── tests ────────────────────────────────────────────────────────────────


@cuda_only
def test_vit_forward_tiny():
    """Single-layer tiny ViT on a static 56x56 image — smoke test."""
    HIDDEN, PATCH, LAYERS, HEADS, INTER, HEAD = 128, 14, 1, 4, 256, 32
    torch.manual_seed(42)

    ref = (
        _RefViT(HIDDEN, PATCH, LAYERS, HEADS, INTER, HEAD, eps=1e-6).cuda().to(torch.float16).eval()
    )
    qw = _build_q_weights(ref)
    vit = (
        VisionTransformer(HIDDEN, PATCH, LAYERS, HEADS, INTER, HEAD, 3, 1e-6, qw)
        .cuda()
        .to(torch.float16)
        .eval()
    )

    px = _rand(1, 3, 56, 56)
    with torch.no_grad():
        ref_out = ref(px)
        q_out = vit(px)

    assert _cos_sim(ref_out, q_out) >= _TOL_COS, f"cos={_cos_sim(ref_out, q_out):.6f}"
    assert _rel_l1(ref_out, q_out) <= _TOL_REL_L1, f"rel-L1={_rel_l1(ref_out, q_out):.6e}"


@cuda_only
def test_vit_forward_small():
    """4-layer ViT on 84x84 image — medium test."""
    HIDDEN, PATCH, LAYERS, HEADS, INTER, HEAD = 128, 14, 4, 4, 256, 32
    torch.manual_seed(123)

    ref = (
        _RefViT(HIDDEN, PATCH, LAYERS, HEADS, INTER, HEAD, eps=1e-6).cuda().to(torch.float16).eval()
    )
    qw = _build_q_weights(ref)
    vit = (
        VisionTransformer(HIDDEN, PATCH, LAYERS, HEADS, INTER, HEAD, 3, 1e-6, qw)
        .cuda()
        .to(torch.float16)
        .eval()
    )

    px = _rand(1, 3, 84, 84)
    with torch.no_grad():
        ref_out = ref(px)
        q_out = vit(px)

    assert _cos_sim(ref_out, q_out) >= _TOL_COS, f"cos={_cos_sim(ref_out, q_out):.6f}"
    assert _rel_l1(ref_out, q_out) <= _TOL_REL_L1, f"rel-L1={_rel_l1(ref_out, q_out):.6e}"


@cuda_only
def test_vit_forward_deeper():
    """8-layer ViT on 112x112 image — broader coverage."""
    HIDDEN, PATCH, LAYERS, HEADS, INTER, HEAD = 128, 14, 8, 4, 256, 32
    torch.manual_seed(456)

    ref = (
        _RefViT(HIDDEN, PATCH, LAYERS, HEADS, INTER, HEAD, eps=1e-6).cuda().to(torch.float16).eval()
    )
    qw = _build_q_weights(ref)
    vit = (
        VisionTransformer(HIDDEN, PATCH, LAYERS, HEADS, INTER, HEAD, 3, 1e-6, qw)
        .cuda()
        .to(torch.float16)
        .eval()
    )

    px = _rand(1, 3, 112, 112)
    with torch.no_grad():
        ref_out = ref(px)
        q_out = vit(px)

    assert _cos_sim(ref_out, q_out) >= _TOL_COS, f"cos={_cos_sim(ref_out, q_out):.6f}"
    assert _rel_l1(ref_out, q_out) <= _TOL_REL_L1, f"rel-L1={_rel_l1(ref_out, q_out):.6e}"


@cuda_only
def test_vit_forward_aspect_ratio():
    """Non-square image (portrait) — still matches reference."""
    HIDDEN, PATCH, LAYERS, HEADS, INTER, HEAD = 128, 14, 2, 4, 256, 32
    torch.manual_seed(789)

    ref = (
        _RefViT(HIDDEN, PATCH, LAYERS, HEADS, INTER, HEAD, eps=1e-6).cuda().to(torch.float16).eval()
    )
    qw = _build_q_weights(ref)
    vit = (
        VisionTransformer(HIDDEN, PATCH, LAYERS, HEADS, INTER, HEAD, 3, 1e-6, qw)
        .cuda()
        .to(torch.float16)
        .eval()
    )

    px = _rand(1, 3, 56, 84)
    with torch.no_grad():
        ref_out = ref(px)
        q_out = vit(px)

    assert _cos_sim(ref_out, q_out) >= _TOL_COS, f"cos={_cos_sim(ref_out, q_out):.6f}"
    assert _rel_l1(ref_out, q_out) <= _TOL_REL_L1, f"rel-L1={_rel_l1(ref_out, q_out):.6e}"


@cuda_only
def test_vit_rope_2d_consistency():
    """2D M-RoPE: each patch gets pos-dependent rotation that varies per axis."""
    head_dim, gh, gw = 64, 3, 4
    cos, sin = _compute_2d_rope(gh, gw, head_dim, "cuda", torch.float16)
    assert cos.shape == (1, 1, gh * gw, head_dim)
    assert sin.shape == (1, 1, gh * gw, head_dim)
    c = cos[0, 0]
    # emb = cat([freqs_h, freqs_w, freqs_h, freqs_w])  -> [S, head_dim]
    # dims [0:16, 32:48] use h; dims [16:32, 48:64] use w
    hq = head_dim // 4
    # Same row (0,0) vs (0,1): h-based cos slices match, w-based differ
    assert torch.allclose(c[0, :hq], c[1, :hq])
    assert torch.allclose(c[0, 2 * hq : 3 * hq], c[1, 2 * hq : 3 * hq])
    assert not torch.allclose(c[0, 3 * hq :], c[1, 3 * hq :])
    # Same col (0,0) vs (1,0): w-based cos slices match, h-based differ
    assert torch.allclose(c[0, hq : 2 * hq], c[gw, hq : 2 * hq])
    assert torch.allclose(c[0, 3 * hq :], c[gw, 3 * hq :])
    assert not torch.allclose(c[0, 2 * hq : 3 * hq], c[gw, 2 * hq : 3 * hq])


@cuda_only
def test_vit_eyes_match():
    """Synthetic white image: all patches identical -> patch embeddings uniform."""
    HIDDEN, PATCH, LAYERS, HEADS, INTER, HEAD = 128, 14, 2, 4, 256, 32
    torch.manual_seed(1337)

    ref = (
        _RefViT(HIDDEN, PATCH, LAYERS, HEADS, INTER, HEAD, eps=1e-6).cuda().to(torch.float16).eval()
    )
    qw = _build_q_weights(ref)
    vit = (
        VisionTransformer(HIDDEN, PATCH, LAYERS, HEADS, INTER, HEAD, 3, 1e-6, qw)
        .cuda()
        .to(torch.float16)
        .eval()
    )

    px = torch.ones(1, 3, 56, 56, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        ref_out = ref(px)
        q_out = vit(px)

    assert _cos_sim(ref_out, q_out) >= _TOL_COS, f"cos={_cos_sim(ref_out, q_out):.6f}"
    assert _rel_l1(ref_out, q_out) <= _TOL_REL_L1, f"rel-L1={_rel_l1(ref_out, q_out):.6e}"


@cuda_only
def test_vit_vision_config_forward():
    """Build a ViT from a VisionConfig (integration happy-path)."""
    vcfg = VisionConfig(
        hidden_size=128,
        patch_size=14,
        num_layers=2,
        image_token_id=0,
        spatial_merge_size=1,
        num_attention_heads=4,
        intermediate_size=256,
        in_channels=3,
        layer_norm_eps=1e-6,
        hidden_act="gelu_pytorch_tanh",
    )
    torch.manual_seed(999)

    ref = (
        _RefViT(
            vcfg.hidden_size,
            vcfg.patch_size,
            vcfg.num_layers,
            vcfg.num_attention_heads,
            vcfg.intermediate_size,
            vcfg.head_dim,
            vcfg.in_channels,
            vcfg.layer_norm_eps,
        )
        .cuda()
        .to(torch.float16)
        .eval()
    )

    qw = _build_q_weights(ref)
    vit = (
        VisionTransformer(
            vcfg.hidden_size,
            vcfg.patch_size,
            vcfg.num_layers,
            vcfg.num_attention_heads,
            vcfg.intermediate_size,
            vcfg.head_dim,
            vcfg.in_channels,
            vcfg.layer_norm_eps,
            qw,
        )
        .cuda()
        .to(torch.float16)
        .eval()
    )

    px = _rand(1, 3, 56, 56)
    with torch.no_grad():
        ref_out = ref(px)
        q_out = vit(px)

    assert _cos_sim(ref_out, q_out) >= _TOL_COS, f"cos={_cos_sim(ref_out, q_out):.6f}"
    assert _rel_l1(ref_out, q_out) <= _TOL_REL_L1, f"rel-L1={_rel_l1(ref_out, q_out):.6e}"
