# SPDX-License-Identifier: MIT
"""Projector unit tests — ViT output → projector → merged embedding.

Compares the fp16 projector modules against their reference implementations.
Full pipeline: VisionTransformer + projector + embed_merge validated end-to-end.
"""

import pytest
import torch
import torch.nn as nn

pytest.importorskip("superl8")

from superl8serve.multimodal.projector import (
    MLPProjector,
    LinearProjector,
    build_projector,
    embed_merge,
    compute_mrope_position_ids,
)
from superl8serve.multimodal.vit import VisionTransformer, _compute_2d_rope, _apply_rotary_pos_emb
from superl8serve.models.weights import to_qtensor

CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not CUDA, reason="dp4a path needs CUDA")


def _rand(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.float16) * 0.05


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


# reference fp16 projectors


class _RefMLPProjector(nn.Module):
    def __init__(self, v, l):
        super().__init__()
        self.fc1 = nn.Linear(v, l, bias=True)
        self.fc2 = nn.Linear(l, l, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x, approximate="tanh")
        return self.fc2(x)


class _RefLinearProjector(nn.Module):
    def __init__(self, v, l):
        super().__init__()
        self.proj = nn.Linear(v, l, bias=True)

    def forward(self, x):
        return self.proj(x)


# reference ViT helpers (tiny, for the full-pipeline tests)


class _RefTinyViTBlock(nn.Module):
    def __init__(self, h, nh, ih, hd, eps):
        super().__init__()
        self.n1 = nn.LayerNorm(h, eps=eps)
        self.attn = _RefTinyViT._Attn(h, nh, hd)
        self.n2 = nn.LayerNorm(h, eps=eps)
        self.mlp = _RefTinyViT._MLP(h, ih)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.mlp(self.n2(x))


class _RefTinyViT(nn.Module):
    class _Attn(nn.Module):
        def __init__(self, h, nh, hd):
            super().__init__()
            self.nh, self.hd, self.s = nh, hd, hd**-0.5
            self.qkv = nn.Linear(h, 3 * nh * hd, bias=True)
            self.proj = nn.Linear(nh * hd, h, bias=True)

        def forward(self, x, cos, sin):
            B, S, _ = x.shape
            qkv = self.qkv(x)
            q, k, v = qkv.split(self.nh * self.hd, dim=-1)
            q = q.view(B, S, self.nh, self.hd).transpose(1, 2)
            k = k.view(B, S, self.nh, self.hd).transpose(1, 2)
            v = v.view(B, S, self.nh, self.hd).transpose(1, 2)
            q, k = _apply_rotary_pos_emb(q, k, cos, sin)
            o = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=self.s)
            return self.proj(o.transpose(1, 2).reshape(B, S, -1))

    class _MLP(nn.Module):
        def __init__(self, h, ih):
            super().__init__()
            self.fc1 = nn.Linear(h, ih, bias=True)
            self.fc2 = nn.Linear(ih, h, bias=True)

        def forward(self, x):
            return self.fc2(torch.nn.functional.gelu(self.fc1(x), approximate="tanh"))

    def __init__(self, hidden, patch, layers, heads, inter, head, channels=3, eps=1e-6):
        super().__init__()
        self.patch_size = patch
        self.head_dim = head
        self.patch_embed = nn.Conv2d(channels, hidden, kernel_size=patch, stride=patch, bias=False)
        self.blocks = nn.ModuleList(
            [_RefTinyViTBlock(hidden, heads, inter, head, eps) for _ in range(layers)]
        )

    def forward(self, px):
        x = self.patch_embed(px).flatten(2).transpose(1, 2)
        gh, gw = px.shape[2] // self.patch_size, px.shape[3] // self.patch_size
        cos, sin = _compute_2d_rope(gh, gw, self.head_dim, x.device, x.dtype)
        for b in self.blocks:
            x = b(x, cos, sin)
        return x


def _build_q_weights(ref):
    w = {"patch_embed.weight": ref.patch_embed.weight.data.clone()}
    for i, blk in enumerate(ref.blocks):
        p = f"blocks.{i}"
        a = blk.attn
        m = blk.mlp
        w[f"{p}.attn.qkv.weight"] = to_qtensor(a.qkv.weight.data)
        w[f"{p}.attn.qkv.bias"] = a.qkv.bias.data.clone()
        w[f"{p}.attn.proj.weight"] = to_qtensor(a.proj.weight.data)
        w[f"{p}.attn.proj.bias"] = a.proj.bias.data.clone()
        w[f"{p}.mlp.fc1.weight"] = to_qtensor(m.fc1.weight.data)
        w[f"{p}.mlp.fc1.bias"] = m.fc1.bias.data.clone()
        w[f"{p}.mlp.fc2.weight"] = to_qtensor(m.fc2.weight.data)
        w[f"{p}.mlp.fc2.bias"] = m.fc2.bias.data.clone()
    return w


# MLPProjector tests


@cuda_only
def test_mlp_projector_forward():
    V, L = 256, 512
    torch.manual_seed(42)
    ref = _RefMLPProjector(V, L).cuda().to(torch.float16).eval()
    proj = MLPProjector(V, L).cuda().to(torch.float16).eval()
    proj.load_state_dict(ref.state_dict())
    x = _rand(1, 64, V)
    with torch.no_grad():
        yr = ref(x)
        yp = proj(x)
    assert _cos_sim(yr, yp) >= _TOL_COS
    assert _rel_l1(yr, yp) <= _TOL_REL_L1


@cuda_only
def test_mlp_projector_shapes():
    V, L = 128, 256
    proj = MLPProjector(V, L).cuda().to(torch.float16)
    for B, P in [(1, 16), (1, 49), (1, 144)]:
        y = proj(_rand(B, P, V))
        assert y.shape == (B, P, L)
        assert torch.isfinite(y).all()


# LinearProjector tests


@cuda_only
def test_linear_projector_forward():
    V, L = 256, 512
    torch.manual_seed(42)
    ref = _RefLinearProjector(V, L).cuda().to(torch.float16).eval()
    proj = LinearProjector(V, L).cuda().to(torch.float16).eval()
    proj.load_state_dict(ref.state_dict())
    x = _rand(1, 64, V)
    with torch.no_grad():
        yr = ref(x)
        yp = proj(x)
    assert _cos_sim(yr, yp) >= _TOL_COS
    assert _rel_l1(yr, yp) <= _TOL_REL_L1


# build_projector factory tests


@cuda_only
def test_build_projector_archs():
    assert isinstance(build_projector(128, 256, "qwen2_5_vl"), MLPProjector)
    assert isinstance(build_projector(128, 256, "qwen2_vl"), MLPProjector)
    assert isinstance(build_projector(128, 256, "llava"), LinearProjector)
    assert isinstance(build_projector(128, 256, "llava_next"), LinearProjector)


@cuda_only
def test_build_projector_loads_weights():
    V, L = 128, 256
    ref = _RefMLPProjector(V, L)
    w = {k: v.clone() for k, v in ref.state_dict().items()}
    proj = build_projector(V, L, "qwen2_5_vl", weights=w)
    for k, v in w.items():
        assert torch.equal(proj.state_dict()[k], v)


# embed_merge tests


@cuda_only
def test_embed_merge_basic():
    L, S, P = 256, 16, 4
    tok = _rand(1, S, L)
    vis = _rand(1, P, L)
    ids = torch.tensor(
        [[0, 1, 2, 151654, 151654, 151654, 151654, 3, 4, 5, 6, 7, 8, 9, 10, 11]], device="cuda"
    )
    merged = embed_merge(tok, vis, image_token_id=151654, input_ids=ids)
    assert merged.shape == tok.shape
    assert torch.equal(merged[0, 3:7], vis[0])
    assert torch.equal(merged[0, 0], tok[0, 0])


@cuda_only
def test_embed_merge_no_image_tokens():
    tok = _rand(1, 10, 256)
    vis = _rand(1, 0, 256)
    ids = torch.arange(10, device="cuda").unsqueeze(0)
    merged = embed_merge(tok, vis, image_token_id=151654, input_ids=ids)
    assert torch.equal(merged, tok)


# compute_mrope_position_ids tests


@cuda_only
def test_mrope_position_ids_basic():
    S, P, gh, gw, start = 20, 12, 3, 4, 5
    pos = compute_mrope_position_ids(S, P, gh, gw, start, "cuda")
    assert pos.shape == (3, 1, S)
    for i in range(start):
        assert pos[0, 0, i].item() == i
        assert pos[1, 0, i].item() == i
        assert pos[2, 0, i].item() == i
    for i in range(P):
        h, w = i // gw, i % gw
        assert pos[0, 0, start + i].item() == 0
        assert pos[1, 0, start + i].item() == h
        assert pos[2, 0, start + i].item() == w


@cuda_only
def test_mrope_after_image():
    S, P, gh, gw, start = 16, 6, 2, 3, 4
    pos = compute_mrope_position_ids(S, P, gh, gw, start, "cuda")
    for i in range(start + P, S):
        assert pos[0, 0, i].item() == i
        assert pos[1, 0, i].item() == i
        assert pos[2, 0, i].item() == i


# full ViT -> projector -> merged embedding pipelines


@cuda_only
def test_full_pipeline_mlp():
    V, L = 128, 256
    PATCH, LAYERS, HEADS, INTER, HEAD = 14, 2, 4, 256, 32
    IMG_TOK = 151654
    torch.manual_seed(42)

    ref_vit = _RefTinyViT(V, PATCH, LAYERS, HEADS, INTER, HEAD).cuda().to(torch.float16).eval()
    qw = _build_q_weights(ref_vit)
    vit = (
        VisionTransformer(V, PATCH, LAYERS, HEADS, INTER, HEAD, 3, 1e-6, qw)
        .cuda()
        .to(torch.float16)
        .eval()
    )

    ref_proj = _RefMLPProjector(V, L).cuda().to(torch.float16).eval()
    proj = MLPProjector(V, L).cuda().to(torch.float16).eval()
    proj.load_state_dict(ref_proj.state_dict())

    px = _rand(1, 3, 56, 56)
    with torch.no_grad():
        vit_out = vit(px)
        ref_out = ref_vit(px)
    assert _cos_sim(ref_out, vit_out) >= _TOL_COS

    vit_proj = proj(vit_out)
    ref_proj_out = ref_proj(ref_out)
    assert _cos_sim(ref_proj_out, vit_proj) >= _TOL_COS
    assert _rel_l1(ref_proj_out, vit_proj) <= _TOL_REL_L1

    S = 32
    n_patches = vit_out.shape[1]
    tok = _rand(1, S, L)
    ids = torch.full((1, S), 0, device="cuda", dtype=torch.long)
    ids[0, 5 : 5 + n_patches] = IMG_TOK
    merged = embed_merge(tok, vit_proj, IMG_TOK, ids)
    assert merged.shape == (1, S, L)
    assert torch.equal(merged[0, 5 : 5 + n_patches], vit_proj[0])


@cuda_only
def test_full_pipeline_linear():
    V, L = 128, 256
    PATCH, LAYERS, HEADS, INTER, HEAD = 14, 2, 4, 256, 32
    IMG_TOK = 32000
    torch.manual_seed(42)

    ref_vit = _RefTinyViT(V, PATCH, LAYERS, HEADS, INTER, HEAD).cuda().to(torch.float16).eval()
    qw = _build_q_weights(ref_vit)
    vit = (
        VisionTransformer(V, PATCH, LAYERS, HEADS, INTER, HEAD, 3, 1e-6, qw)
        .cuda()
        .to(torch.float16)
        .eval()
    )

    ref_proj = _RefLinearProjector(V, L).cuda().to(torch.float16).eval()
    proj = LinearProjector(V, L).cuda().to(torch.float16).eval()
    proj.load_state_dict(ref_proj.state_dict())

    px = _rand(1, 3, 56, 56)
    with torch.no_grad():
        vit_out = vit(px)
        ref_out = ref_vit(px)
    assert _cos_sim(ref_out, vit_out) >= _TOL_COS

    vit_proj = proj(vit_out)
    ref_proj_out = ref_proj(ref_out)
    assert _cos_sim(ref_proj_out, vit_proj) >= _TOL_COS
    assert _rel_l1(ref_proj_out, vit_proj) <= _TOL_REL_L1

    S = 20
    n_patches = vit_out.shape[1]
    tok = _rand(1, S, L)
    ids = torch.full((1, S), 0, device="cuda", dtype=torch.long)
    ids[0, 2 : 2 + n_patches] = IMG_TOK
    merged = embed_merge(tok, vit_proj, IMG_TOK, ids)
    assert merged.shape == (1, S, L)
