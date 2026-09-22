# SPDX-License-Identifier: MIT
"""Qwen2.5-VL end-to-end integration: ViT -> projector -> embed-merge -> LLM backbone.

Builds a tiny Qwen2.5-VL model with random fp16 weights, quantises the vision tower
to int8 dp4a, runs the full ``MultimodalCausalLM`` forward (image pixels + text tokens),
and compares the hidden states against a manual fp16 reference pipeline.

The test validates **orchestration** -- that ``MultimodalCausalLM.forward`` correctly
splices vision embeddings at image-token positions and routes through the LLM backbone.
Individual-component accuracy (int8 ViT, int8 projector) is tested separately in
``test_vit.py`` and ``test_projector.py``.
"""

import pytest

pytest.importorskip("superl8")

import torch
import torch.nn as nn
import torch.nn.functional as F

from superl8serve.models.base import ForwardContext
from superl8serve.models.vlm import MultimodalCausalLM
from superl8serve.models.weights import to_qtensor
from superl8serve.multimodal.projector import MLPProjector, embed_merge
from superl8serve.multimodal.vit import VisionTransformer, _compute_2d_rope, _apply_rotary_pos_emb

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

# -- tiny dimensions ---------------------------------------------------

VIT_HIDDEN = 64
VIT_PATCH = 14
VIT_LAYERS = 1
VIT_HEADS = 2
VIT_INTER = 128
VIT_HEAD = 32

LLM_HIDDEN = 128
LLM_LAYERS = 1
LLM_HEADS = 2
LLM_KV_HEADS = 2
LLM_INTER = 256
LLM_HEAD = LLM_HIDDEN // LLM_HEADS
VOCAB = 1024

IMG_TOK = 999


# -- fp16 reference: tiny ViT matching the VisionTransformer architecture --


class _RefViTBlock(nn.Module):
    def __init__(self, h, nh, ih, hd, eps=1e-6):
        super().__init__()
        self.n1 = nn.LayerNorm(h, eps=eps)
        self.attn = _RefViT._Attn(h, nh, hd)
        self.n2 = nn.LayerNorm(h, eps=eps)
        self.mlp = _RefViT._MLP(h, ih)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.mlp(self.n2(x))


class _RefViT(nn.Module):
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
            o = F.scaled_dot_product_attention(q, k, v, scale=self.s)
            return self.proj(o.transpose(1, 2).reshape(B, S, -1))

    class _MLP(nn.Module):
        def __init__(self, h, ih):
            super().__init__()
            self.fc1 = nn.Linear(h, ih, bias=True)
            self.fc2 = nn.Linear(ih, h, bias=True)

        def forward(self, x):
            return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))

    def __init__(self, hidden, patch, layers, heads, inter, head, channels=3, eps=1e-6):
        super().__init__()
        self.patch_size = patch
        self.head_dim = head
        self.patch_embed = nn.Conv2d(channels, hidden, kernel_size=patch, stride=patch, bias=False)
        self.blocks = nn.ModuleList(
            [_RefViTBlock(hidden, heads, inter, head, eps) for _ in range(layers)]
        )

    def forward(self, px):
        x = self.patch_embed(px).flatten(2).transpose(1, 2)
        gh, gw = px.shape[2] // self.patch_size, px.shape[3] // self.patch_size
        cos, sin = _compute_2d_rope(gh, gw, self.head_dim, x.device, x.dtype)
        for b in self.blocks:
            x = b(x, cos, sin)
        return x


# -- fp16 reference: tiny Qwen3-style LLM backbone ------------------------


class _RefLLMLayer(nn.Module):
    def __init__(self, hidden, n_heads, n_kv, head_dim, intermediate, eps=1e-6):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden, eps=eps)
        self.post_attention_layernorm = nn.LayerNorm(hidden, eps=eps)
        self.self_attn = _RefLLMLayer._GQA(hidden, n_heads, n_kv, head_dim)
        self.mlp = _RefLLMLayer._MLP(hidden, intermediate)

    class _GQA(nn.Module):
        def __init__(self, hidden, n_heads, n_kv, head_dim):
            super().__init__()
            self.nh, self.nkv, self.hd = n_heads, n_kv, head_dim
            self.scale = head_dim**-0.5
            self.qkv = nn.Linear(hidden, (n_heads + 2 * n_kv) * head_dim, bias=False)
            self.proj = nn.Linear(n_heads * head_dim, hidden, bias=False)

        def forward(self, x):
            B, S, _ = x.shape
            qkv = self.qkv(x)
            q, k, v = qkv.split([self.nh * self.hd, self.nkv * self.hd, self.nkv * self.hd], dim=-1)
            q = q.view(B, S, self.nh, self.hd).transpose(1, 2)
            k = k.view(B, S, self.nkv, self.hd).transpose(1, 2)
            v = v.view(B, S, self.nkv, self.hd).transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
            return self.proj(out.transpose(1, 2).reshape(B, S, -1))

    class _MLP(nn.Module):
        def __init__(self, hidden, intermediate):
            super().__init__()
            self.gate = nn.Linear(hidden, intermediate, bias=False)
            self.up = nn.Linear(hidden, intermediate, bias=False)
            self.down = nn.Linear(intermediate, hidden, bias=False)

        def forward(self, x):
            return self.down(F.silu(self.gate(x)) * self.up(x))

    def forward(self, x, positions, ctx, residual):
        if residual is None:
            residual = x
            h = self.input_layernorm(x)
        else:
            x = x + residual
            residual = x
            h = self.input_layernorm(x)
        h = self.self_attn(h)
        h = h + residual
        residual = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        return h, residual


# -- build quantised ViT weights from a reference -------------------------


def _build_vit_weights(ref: _RefViT) -> dict:
    w = {"patch_embed.weight": ref.patch_embed.weight.data.clone()}
    for i, blk in enumerate(ref.blocks):
        p = f"blocks.{i}"
        a, m = blk.attn, blk.mlp
        w[f"{p}.attn.qkv.weight"] = to_qtensor(a.qkv.weight.data)
        w[f"{p}.attn.qkv.bias"] = a.qkv.bias.data.clone()
        w[f"{p}.attn.proj.weight"] = to_qtensor(a.proj.weight.data)
        w[f"{p}.attn.proj.bias"] = a.proj.bias.data.clone()
        w[f"{p}.mlp.fc1.weight"] = to_qtensor(m.fc1.weight.data)
        w[f"{p}.mlp.fc1.bias"] = m.fc1.bias.data.clone()
        w[f"{p}.mlp.fc2.weight"] = to_qtensor(m.fc2.weight.data)
        w[f"{p}.mlp.fc2.bias"] = m.fc2.bias.data.clone()
    return w


# -- test ----------------------------------------------------------------


@cuda_only
def test_qwen2_5_vl_e2e_orchestration():
    """MultimodalCausalLM forward: ViT (int8) -> projector -> embed-merge -> LLM backbone.

    The int8 ViT output should produce hidden states within cos >= 0.99 / rel-L1 <= 0.05
    of a fully fp16 reference pipeline.
    """
    torch.manual_seed(42)

    # -- Reference pipeline (fp16) -----------------------------------------
    ref_vit = (
        _RefViT(VIT_HIDDEN, VIT_PATCH, VIT_LAYERS, VIT_HEADS, VIT_INTER, VIT_HEAD)
        .cuda()
        .to(torch.float16)
        .eval()
    )
    ref_proj = MLPProjector(VIT_HIDDEN, LLM_HIDDEN).cuda().to(torch.float16).eval()
    ref_embed = nn.Embedding(VOCAB, LLM_HIDDEN).cuda().to(torch.float16).eval()
    ref_layers = nn.ModuleList(
        [
            _RefLLMLayer(LLM_HIDDEN, LLM_HEADS, LLM_KV_HEADS, LLM_HEAD, LLM_INTER)
            .cuda()
            .to(torch.float16)
            .eval()
        ]
    )
    ref_norm = nn.LayerNorm(LLM_HIDDEN).cuda().to(torch.float16).eval()

    # -- Quantised MultimodalCausalLM from the same weights ----------------
    vit_qw = _build_vit_weights(ref_vit)
    vit_q = (
        VisionTransformer(
            VIT_HIDDEN, VIT_PATCH, VIT_LAYERS, VIT_HEADS, VIT_INTER, VIT_HEAD, 3, 1e-6, vit_qw
        )
        .cuda()
        .to(torch.float16)
        .eval()
    )
    proj_q = MLPProjector(VIT_HIDDEN, LLM_HIDDEN).cuda().to(torch.float16).eval()
    proj_q.load_state_dict({k: v.clone() for k, v in ref_proj.state_dict().items()})

    class _FinalNorm(nn.Module):
        """Wraps nn.LayerNorm to accept the (h, residual) convention that
        MultimodalCausalLM.forward passes to lm_model.norm."""

        def __init__(self, ln):
            super().__init__()
            self.ln = ln

        def forward(self, h, residual=None):
            if residual is not None:
                h = h + residual
            return self.ln(h), h

    class _LMBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = None
            self.model = nn.Module()
            self.model.embed_tokens = ref_embed
            self.model.layers = ref_layers
            self.model.norm = _FinalNorm(ref_norm)

        def compute_logits(self, hidden):
            return hidden @ ref_embed.weight.T

    lm_q = _LMBackbone().cuda().to(torch.float16).eval()

    q_model = MultimodalCausalLM(
        vision_tower=vit_q,
        projector=proj_q,
        lm=lm_q,
        image_token_id=IMG_TOK,
        arch="qwen2_5_vl",
    ).eval()

    # -- Inputs -----------------------------------------------------------
    px = _rand(1, 3, 56, 56)
    n_patches = 16  # 56//14 * 56//14
    S = 20  # enough room for image patches + text tokens

    input_ids = torch.zeros(1, S, device="cuda", dtype=torch.long)
    img_start = 2
    input_ids[0, img_start : img_start + n_patches] = IMG_TOK
    for i in range(img_start + n_patches, S):
        input_ids[0, i] = i + 10  # arbitrary valid token ids < VOCAB
    positions = torch.arange(S, device="cuda").unsqueeze(0)

    # -- Reference forward (fp16) -----------------------------------------
    with torch.no_grad():
        vit_out = ref_vit(px)
        proj_out = ref_proj(vit_out)
        h = ref_embed(input_ids)
        h = embed_merge(h, proj_out, IMG_TOK, input_ids)
        residual = None
        for layer in ref_layers:
            h, residual = layer(h, positions, None, residual)
        h = ref_norm(h + residual)
    ref_hidden = h.clone()

    # -- Quantised forward ------------------------------------------------
    ctx = ForwardContext(is_prefill=True, pixel_values=px)
    with torch.no_grad():
        q_hidden = q_model(input_ids, positions, ctx)

    # -- Compare ----------------------------------------------------------
    cos = _cos_sim(q_hidden, ref_hidden)
    rel = _rel_l1(q_hidden, ref_hidden)
    assert cos >= _TOL_COS, f"cos={cos:.6f}"
    assert rel <= _TOL_REL_L1, f"rel-L1={rel:.6e}"

    # Also verify that image-token positions have non-zero embeddings
    # (the merge spliced vision embeddings in, not zeros from padding).
    img_mask = (input_ids == IMG_TOK).unsqueeze(-1).expand_as(q_hidden)
    img_embeds = q_hidden[img_mask]
    assert img_embeds.abs().sum().item() > 0, "image-token embeddings should not be all-zero"
