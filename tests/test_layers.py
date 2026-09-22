# SPDX-License-Identifier: MIT
"""Unit tests for the shared model layers (arch-agnostic primitives).

These cover the building blocks every model family composes: RMSNorm (Qwen + Gemma
(1+w)), RoPE, SwiGLU/GeGLU, the gated MLP on the dp4a GEMM, embedding (+ Gemma
scale), LM head, and the sampler. Run in the superl8 test container (needs torch; the
MLP/LM-head dp4a paths need CUDA)."""
import pytest
import torch

from superl8 import QTensor

from superl8serve.layers import (
    GatedMLP,
    GeluAndMul,
    LMHead,
    RMSNorm,
    RotaryEmbedding,
    Sampler,
    SiluAndMul,
    VocabEmbedding,
    get_act_and_mul,
)

CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not CUDA, reason="dp4a path needs CUDA")


def _i8(out, in_):
    w = torch.randn(out, in_, dtype=torch.float16)
    s = w.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 127
    q = torch.round(w / s).clamp_(-127, 127).to(torch.int8)
    return QTensor(q, s.squeeze(-1).float(), scheme="per_row_i8"), (q.float() * s).half()


# ---- RMSNorm ----

def test_rmsnorm_matches_reference():
    x = torch.randn(4, 8, 128, dtype=torch.float32)
    w = torch.randn(128, dtype=torch.float16)
    n = RMSNorm(128, eps=1e-6, weight=w)
    got = n(x.half()).float()
    ref = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)) * w.float()
    torch.testing.assert_close(got, ref, rtol=2e-3, atol=2e-3)


def test_rmsnorm_gemma_unit_offset():
    x = torch.randn(2, 64, dtype=torch.float16)
    w = torch.randn(64, dtype=torch.float16)
    n = RMSNorm(64, weight=w, add_unit_offset=True)
    xf = x.float()
    ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)) * (1.0 + w.float())
    torch.testing.assert_close(n(x).float(), ref, rtol=2e-3, atol=2e-3)


def test_rmsnorm_fused_residual():
    x = torch.randn(3, 32, dtype=torch.float16)
    r = torch.randn(3, 32, dtype=torch.float16)
    n = RMSNorm(32)
    out, new_res = n(x, r)
    torch.testing.assert_close(new_res, x + r)
    torch.testing.assert_close(out, n(x + r))


# ---- RoPE ----

def _hf_rope_ref(pos, q, base, head_dim):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(pos.float(), inv)
    emb = torch.cat((freqs, freqs), -1)
    cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]
    x1, x2 = q.float().chunk(2, -1)
    rot = torch.cat((-x2, x1), -1)
    return q.float() * cos + rot * sin


def test_rope_matches_hf_convention():
    S, H, D = 16, 4, 64
    pos = torch.arange(S)
    q = torch.randn(S, H, D, dtype=torch.float16)
    k = torch.randn(S, H, D, dtype=torch.float16)
    rope = RotaryEmbedding(D, max_position=32, base=1e6)
    q_r, k_r = rope(pos, q, k)
    torch.testing.assert_close(q_r.float(), _hf_rope_ref(pos, q, 1e6, D), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(k_r.float(), _hf_rope_ref(pos, k, 1e6, D), rtol=2e-3, atol=2e-3)


def test_rope_dual_theta_differs():
    """Gemma3 uses different theta for local vs global layers -> different rotation."""
    pos = torch.arange(8)
    q = torch.randn(8, 2, 64, dtype=torch.float16)
    local = RotaryEmbedding(64, 16, base=1e4)(pos, q, q)[0]
    glob = RotaryEmbedding(64, 16, base=1e6)(pos, q, q)[0]
    assert not torch.allclose(local, glob)


# ---- activations ----

def test_silu_and_mul():
    x = torch.randn(4, 256)
    g, u = x.chunk(2, -1)
    torch.testing.assert_close(SiluAndMul()(x), torch.nn.functional.silu(g) * u)


def test_gelu_and_mul_tanh():
    x = torch.randn(4, 256)
    g, u = x.chunk(2, -1)
    ref = torch.nn.functional.gelu(g, approximate="tanh") * u
    torch.testing.assert_close(GeluAndMul("tanh")(x), ref)


def test_get_act_and_mul_dispatch():
    assert isinstance(get_act_and_mul("silu"), SiluAndMul)
    assert isinstance(get_act_and_mul("gelu_pytorch_tanh"), GeluAndMul)
    with pytest.raises(ValueError):
        get_act_and_mul("relu")


# ---- gated MLP on dp4a ----

@cuda_only
@pytest.mark.parametrize("act", ["silu", "gelu_pytorch_tanh"])
def test_gated_mlp_matches_fp16(act):
    hidden, inter = 512, 1376
    x = torch.randn(8, hidden, device="cuda", dtype=torch.float16)
    gu_qt, gu_w = _i8(2 * inter, hidden)
    dn_qt, dn_w = _i8(hidden, inter)
    mlp = GatedMLP(QTensor(gu_qt.data.cuda(), gu_qt.scale.cuda(), scheme="per_row_i8"),
                   QTensor(dn_qt.data.cuda(), dn_qt.scale.cuda(), scheme="per_row_i8"),
                   act=act).cuda()
    y = mlp(x)
    assert y.shape == (8, hidden) and torch.isfinite(y).all()
    # fp16 reference through the same dequantized weights
    fn = get_act_and_mul(act)
    ref = torch.nn.functional.linear(fn(torch.nn.functional.linear(x, gu_w.cuda())), dn_w.cuda())
    cos = torch.nn.functional.cosine_similarity(y.flatten().float(), ref.flatten().float(), dim=0)
    assert cos.item() >= 0.99


# ---- embedding + LM head ----

def test_vocab_embedding_gemma_scale():
    w = torch.randn(100, 64, dtype=torch.float16)
    ids = torch.tensor([1, 5, 99])
    emb = VocabEmbedding(w, embed_scale=64 ** 0.5)
    torch.testing.assert_close(emb(ids), w[ids] * (64 ** 0.5))


@cuda_only
def test_lm_head_int8_and_softcap():
    h = torch.randn(4, 256, device="cuda", dtype=torch.float16)
    qt, w = _i8(300, 256)
    head = LMHead(QTensor(qt.data.cuda(), qt.scale.cuda(), scheme="per_row_i8"))
    assert head(h).shape == (4, 300)
    capped = LMHead(w.cuda(), logit_softcap=30.0)(h)
    assert capped.abs().max().item() <= 30.0 + 1e-3


# ---- sampler ----

def test_sampler_greedy_and_sampling():
    logits = torch.randn(5, 1000)
    s = Sampler()
    greedy = s(logits, torch.zeros(5))
    torch.testing.assert_close(greedy, logits.argmax(-1))
    # temperature>0 stays in-vocab and finite
    out = s(logits, torch.ones(5), top_p=torch.full((5,), 0.9))
    assert out.shape == (5,) and (out >= 0).all() and (out < 1000).all()


def test_sampler_top_p_restricts_support():
    # one dominant logit + top_p small -> must pick the argmax deterministically
    logits = torch.full((1, 10), -10.0)
    logits[0, 3] = 10.0
    out = Sampler()(logits, torch.ones(1), top_p=torch.full((1,), 0.5))
    assert out.item() == 3
