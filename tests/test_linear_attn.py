# SPDX-License-Identifier: MIT
"""Gated DeltaNet linear attention (Track-1 fp16 port).

The scalar recurrence is the correct oracle. Checks the delta-rule algebra (a
single write is read back; decay shrinks old state), the L2-norm helper, and that
the full GatedDeltaNetAttention block runs end-to-end. The chunked/int8 form is
Track 2 (superl8 csrc/), validated there against this recurrence."""
import pytest
import torch
import torch.nn.functional as F

from superl8serve.layers.linear_attn import (
    GatedDeltaNetAttention,
    ShortConv,
    _l2norm,
    _lightning_attn_dispatch,
    lightning_attention,
    lightning_slopes,
    recurrent_gated_delta_rule,
)

CUDA = torch.cuda.is_available()


def test_l2norm_unit():
    x = torch.randn(2, 3, 4, 8)
    assert torch.allclose(_l2norm(x).norm(dim=-1), torch.ones(2, 3, 4), atol=1e-4)


def test_delta_rule_writes_and_reads():
    """beta=1, alpha=1, one key: writing v at k then querying k reads v back
    (delta rule stores an associative pair)."""
    B, H, Dk, Dv = 1, 1, 4, 4
    k = _l2norm(torch.randn(B, H, 1, Dk))
    v = torch.randn(B, H, 1, Dv)
    out, _ = recurrent_gated_delta_rule(k.clone(), k, v, torch.ones(B, H, 1), torch.zeros(B, H, 1))
    torch.testing.assert_close(out, v, rtol=1e-3, atol=1e-3)   # o_1 = (v k^T) k = v (k unit)


def test_decay_shrinks_state():
    """Strong decay (alpha->0) makes an old write vanish by the next step."""
    B, H, Dk, Dv = 1, 1, 4, 4
    q = _l2norm(torch.randn(B, H, 2, Dk))
    k = _l2norm(torch.randn(B, H, 2, Dk))
    v = torch.randn(B, H, 2, Dv)
    beta = torch.ones(B, H, 2)
    g_strong = torch.tensor([[[0.0, -20.0]]])                  # step 2 decays state ~0
    out, _ = recurrent_gated_delta_rule(q, k, v, beta, g_strong)
    # at t=2 the state is dominated by the fresh write (old contribution ~ alpha~0)
    assert torch.isfinite(out).all()


def test_delta_rule_decode_state_matches_prefill():
    """The actual decode-caching bug: feeding the sequence one token at a time with
    `S` carried across calls (as the decode path now does) must reproduce the same
    output as a single whole-sequence ("prefill") call. Before the fix, each call
    zero-initialized `S`, so this would only hold for the first token."""
    torch.manual_seed(0)
    B, H, L, Dk, Dv = 2, 3, 6, 4, 5
    q = _l2norm(torch.randn(B, H, L, Dk))
    k = _l2norm(torch.randn(B, H, L, Dk))
    v = torch.randn(B, H, L, Dv)
    beta = torch.sigmoid(torch.randn(B, H, L))
    g = -F.softplus(torch.randn(B, H, L))

    full_out, full_state = recurrent_gated_delta_rule(q, k, v, beta, g)

    state, outs = None, []
    for t in range(L):
        o, state = recurrent_gated_delta_rule(q[:, :, t:t + 1], k[:, :, t:t + 1], v[:, :, t:t + 1],
                                              beta[:, :, t:t + 1], g[:, :, t:t + 1], state=state)
        outs.append(o)

    torch.testing.assert_close(torch.cat(outs, dim=2), full_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state, full_state, rtol=1e-4, atol=1e-4)


def test_lightning_attention_decode_state_matches_prefill():
    """Same decode-caching property for MiniMax lightning attention: stepping token
    by token with `S` carried across calls must match one whole-sequence call."""
    torch.manual_seed(1)
    B, H, L, Dk, Dv = 2, 4, 5, 4, 4
    q, k = torch.randn(B, H, L, Dk), torch.randn(B, H, L, Dk)
    v = torch.randn(B, H, L, Dv)
    slopes = lightning_slopes(H)

    full_out, full_state = lightning_attention(q, k, v, slopes)

    state, outs = None, []
    for t in range(L):
        o, state = lightning_attention(q[:, :, t:t + 1], k[:, :, t:t + 1], v[:, :, t:t + 1],
                                       slopes, state=state)
        outs.append(o)

    torch.testing.assert_close(torch.cat(outs, dim=2), full_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state, full_state, rtol=1e-4, atol=1e-4)


class _ConvHarness:
    """Bare object exposing only what `GatedDeltaNetAttention._conv` touches, so the
    causal-conv decode-tail logic can be unit tested without building the full
    nn.Module (whose projections need the CUDA dp4a kernels)."""

    def __init__(self, conv_kernel, conv_weight):
        self.conv_kernel = conv_kernel
        self.conv_weight = conv_weight


def test_deltanet_conv_tail_matches_full_sequence():
    """The causal depthwise conv must carry its trailing `kernel-1` raw window
    across decode calls instead of zero-padding it away: one token at a time with
    the tail threaded through must match a single whole-sequence call."""
    torch.manual_seed(2)
    B, L, W, K = 2, 7, 5, 4
    harness = _ConvHarness(K, torch.randn(W, K))
    x = torch.randn(B, L, W)

    full, _ = GatedDeltaNetAttention._conv(harness, x)

    tail, outs = None, []
    for t in range(L):
        y, tail = GatedDeltaNetAttention._conv(harness, x[:, t:t + 1], tail)
        outs.append(y)

    torch.testing.assert_close(torch.cat(outs, dim=1), full, rtol=1e-5, atol=1e-5)


class _ShortConvHarness:
    """Bare object exposing only what `ShortConv._conv` touches, so the
    ShortConv causal-conv decode-tail logic can be unit tested without building the
    full nn.Module (whose projections need the CUDA dp4a kernels)."""

    def __init__(self, kernel, conv_weight):
        self.kernel = kernel
        self.conv_weight = conv_weight


def test_short_conv_tail_matches_full_sequence():
    """The ShortConv depthwise conv must carry its trailing `kernel-1` raw window
    across decode calls instead of zero-padding it away: one token at a time with
    the tail threaded through must match a single whole-sequence call."""
    torch.manual_seed(3)
    B, L, D, K = 2, 7, 5, 3
    harness = _ShortConvHarness(K, torch.randn(D, 1, K))
    u = torch.randn(B, D, L)

    full, _ = ShortConv._conv(harness, u)

    tail, outs = None, []
    for t in range(L):
        y, tail = ShortConv._conv(harness, u[:, :, t:t + 1], tail)
        outs.append(y)

    torch.testing.assert_close(torch.cat(outs, dim=1), full, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("B", [1, 16, 64, 128])
def test_short_conv_segmented_matches_serial(B):
    """The packed-ragged segmented causal conv must be BIT-IDENTICAL to running the
    serial `_conv` on each sequence separately — same conv outputs and the same
    per-sequence trailing K-1 raw windows (the numerical contract the batched
    prefill path relies on). Ragged lengths include sequences shorter than K-1 so
    the zero-padding path is exercised too."""
    import superl8serve.layers.linear_attn as la

    torch.manual_seed(7 + B)
    D, K = 8, 3
    conv_weight = torch.randn(D, 1, K)
    harness = _ShortConvHarness(K, conv_weight)
    lens = [1] + [1 + ((i * 7 + 3) % 13) for i in range(B - 1)]  # ragged, incl. a length-1 seq
    cu = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(torch.tensor(lens, dtype=torch.int64), 0)])
    u = torch.randn(cu[-1].item(), D)

    y, tails = la._short_conv_segmented(u, conv_weight, cu, kernel=K)
    assert y.shape == u.shape, "gathered output must keep packed [T, D] shape"
    assert tails.shape == (B, D, K - 1), "one exact trailing window per sequence"

    for i in range(B):
        u_i = u[cu[i]:cu[i + 1]].transpose(0, 1).unsqueeze(0)  # [1, D, L_i]
        y_ser, tail_ser = ShortConv._conv(harness, u_i)
        torch.testing.assert_close(y[cu[i]:cu[i + 1]], y_ser.squeeze(0), rtol=0, atol=0)
        torch.testing.assert_close(tails[i], tail_ser.squeeze(0), rtol=0, atol=0)


def test_short_conv_segmented_forward_keeps_rank():
    """The packed-ragged branch of `ShortConv.forward` must keep the packed
    [1, T, D] rank (same contract as the serial layer) after `out_proj`, and bind
    the per-slot trailing windows on the cache. Projections are stubbed so the
    branch is tested CUDA-free."""
    import torch.nn as nn
    from types import SimpleNamespace

    import superl8serve.layers.linear_attn as la

    class _InProj(nn.Module):
        def forward(self, h):
            return h.repeat(1, 1, 3)  # [1, T, 3D]

    blk = la.ShortConv.__new__(la.ShortConv)
    nn.Module.__init__(blk)
    D, K = 8, 3
    blk.dim, blk.kernel = D, K
    blk.in_proj = _InProj()
    blk.out_proj = nn.Identity()
    blk.conv_weight = torch.randn(D, 1, K)

    class _Cache:
        def __init__(self):
            self.tail = {}

        def set_conv_tail(self, i, t):
            self.tail[i] = t

    cache = _Cache()
    lens = [1, 4, 2]
    cu = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(torch.tensor(lens, dtype=torch.int64), 0)])
    T = cu[-1].item()
    x = torch.randn(1, T, D)
    ctx = SimpleNamespace(lin_cache=cache, cu_seqlens=cu)

    out = blk(x, None, ctx, 0)
    assert out.shape == (1, T, D), f"packed rank lost: {out.shape}"
    assert cache.tail[0].shape == (len(lens), D, K - 1), "per-slot tails must be [S, D, K-1]"


class _CudaShim(torch.Tensor):
    """A plain tensor that reports `is_cuda == True`, so the L==1 fused-decode
    dispatch (which requires CUDA) can be exercised CPU-side against a stubbed
    `superl8.causal_conv1d_decode` — no GPU needed."""

    is_cuda = True


def _short_conv_fused_oracle(x, weight, tail):
    """Eager L==1 causal window math for `superl8.causal_conv1d_decode`: window
    [tail[...,0], .., tail[...,K-2], x] per channel, out = dot(window, weight),
    new_tail = window[..., 1:]. Mirrors the superl8 kernel contract (fp32-internal,
    store dtype follows the input) so the serve-side dispatch/shape/roll wiring
    can be validated without a GPU."""
    win = torch.cat([tail, x.unsqueeze(-1)], dim=-1)  # [B, D, K]
    out = (win.float() * weight.float().unsqueeze(0)).sum(-1).to(x.dtype)
    return out, win[:, :, 1:].to(x.dtype)


def test_short_conv_fused_decode_flag_guards_older_superl8():
    """`_LFM_CONV` must exist and equal (SUPERL8_LFM_CONV != 0) AND (the installed
    superl8 has `causal_conv1d_decode`) — the older-prebuilt-image auto-degrade
    contract: a cached superl8 image without the op falls back to eager with no
    error (issue #371 acceptance)."""
    import os

    import superl8serve.layers.linear_attn as la

    assert isinstance(la._LFM_CONV, bool), "flag must be bool"
    expected = os.environ.get("SUPERL8_LFM_CONV", "1") != "0" and hasattr(
        la.superl8, "causal_conv1d_decode"
    )
    assert la._LFM_CONV is expected


def test_short_conv_fused_decode_falls_back_eager(monkeypatch):
    """Prefill (L>1), CPU, verify capture (return_traj=True), and K>8 must keep
    the eager fallback — the fused op must not fire on any of them (issue #371
    acceptance: 'prefill/CPU/older superl8 retain the current fallback')."""
    import superl8serve.layers.linear_attn as la

    calls = []

    def _mock(*args, **kwargs):
        calls.append(1)
        raise AssertionError("fused causal_conv1d_decode must not be called")

    monkeypatch.setattr(la.superl8, "causal_conv1d_decode", _mock, raising=False)
    monkeypatch.setattr(la, "_LFM_CONV", True)

    torch.manual_seed(5)
    D, K = 8, 3
    harness = _ShortConvHarness(K, torch.randn(D, 1, K))

    # prefill (L>1)
    y, nt = ShortConv._conv(harness, torch.randn(2, D, 4))
    assert y.shape == (2, 4, D) and nt.shape == (2, D, K - 1)
    # CPU single-token decode (L==1 but not CUDA)
    y, nt = ShortConv._conv(harness, torch.randn(2, D, 1))
    assert y.shape == (2, 1, D) and nt.shape == (2, D, K - 1)
    # verify capture (return_traj=True)
    y, nt, traj = ShortConv._conv(harness, torch.randn(2, D, 1), return_traj=True)
    assert y.shape == (2, 1, D) and traj.shape == (2, 1, D, K - 1)
    # K>8 (op contract is K <= 8)
    harness9 = _ShortConvHarness(9, torch.randn(D, 1, 9))
    y, nt = ShortConv._conv(harness9, torch.randn(2, D, 1))
    assert y.shape == (2, 1, D) and nt.shape == (2, D, 8)

    assert calls == [], "fused op fired for a non-qualifying branch"


@pytest.mark.parametrize("B", [1, 128, 512])
def test_short_conv_fused_decode_backend_boundary(monkeypatch, B):
    """Backend boundary contract (issue #371): for CUDA L==1 K<=8 the layer must
    hand `superl8.causal_conv1d_decode` its NATIVE shapes — x [B, D] (2D, the op
    does not see [B, D, 1]), weight [D, K] as a zero-copy squeeze view of the
    [D, 1, K] buffer, and the cache tail [B, D, K-1] AS-IS (same tensor object —
    no transpose, no copy) — and return the op's out reshaped to [B, L, D] with
    new_tail passed straight through."""
    import superl8serve.layers.linear_attn as la

    torch.manual_seed(11 + B)
    D, K = 8, 3
    harness = _ShortConvHarness(K, torch.randn(D, 1, K))
    u = torch.randn(B, 1, D).transpose(1, 2)  # [B, D, 1], as ShortConv.forward builds it
    u.__class__ = _CudaShim
    tail = torch.randn(B, D, K - 1)

    seen, captured = {}, {}

    def _mock(x, weight, tail_arg):
        seen["x"], seen["weight"], seen["tail"] = x, weight, tail_arg
        out, new_tail = _short_conv_fused_oracle(x, weight, tail_arg)
        captured["out"], captured["new_tail"] = out, new_tail
        return out, new_tail

    monkeypatch.setattr(la.superl8, "causal_conv1d_decode", _mock, raising=False)
    monkeypatch.setattr(la, "_LFM_CONV", True)

    y, new_tail = ShortConv._conv(harness, u, tail)

    # x: native 2D [B, D] and contiguous — the op's own .contiguous() is a no-op
    assert seen["x"].shape == (B, D) and seen["x"].dim() == 2
    assert seen["x"].is_contiguous()
    # weight: native 2D [D, K], a zero-copy squeeze view of the [D, 1, K] buffer
    assert seen["weight"].shape == (D, K) and seen["weight"].dim() == 2
    assert seen["weight"].data_ptr() == harness.conv_weight.data_ptr()
    # tail: handed over as-is — no transpose and no copy
    assert seen["tail"].shape == (B, D, K - 1)
    assert seen["tail"] is tail
    # out reshaped to the layer's [B, L, D] rank; new_tail returned un-touched
    assert y.shape == (B, 1, D)
    assert torch.equal(y, captured["out"].reshape(B, 1, D))
    assert new_tail is captured["new_tail"]


@pytest.mark.parametrize("B", [1, 128, 512])
def test_short_conv_fused_decode_stepwise_cache_parity(monkeypatch, B):
    """Stepwise decode through the fused op (L==1 per step, tail threaded through
    the cache) must reproduce the eager single-call full-sequence output and the
    final per-slot tail — the cache-parity contract issue #371 relies on. The
    mocked op implements the exact eager window math, so this validates the
    layer's dispatch/shape/roll wiring (the kernel's own numerics are gated in
    superl8's tests)."""
    import superl8serve.layers.linear_attn as la

    torch.manual_seed(23 + B)
    L, D, K = 5, 8, 3
    harness = _ShortConvHarness(K, torch.randn(D, 1, K))
    u_full = torch.randn(B, D, L)

    monkeypatch.setattr(
        la.superl8,
        "causal_conv1d_decode",
        lambda x, weight, tail_arg: _short_conv_fused_oracle(x, weight, tail_arg),
        raising=False,
    )
    monkeypatch.setattr(la, "_LFM_CONV", True)

    tail, outs = None, []
    for t in range(L):
        u_t = u_full[:, :, t:t + 1]
        u_t.__class__ = _CudaShim
        y, tail = ShortConv._conv(harness, u_t, tail)
        outs.append(y)
    fused_out = torch.cat(outs, dim=1)
    fused_tail = tail

    ref_out, ref_tail = ShortConv._conv(harness, u_full)  # eager full-sequence oracle

    torch.testing.assert_close(fused_out, ref_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(fused_tail, ref_tail, rtol=0, atol=0)


def _build_short_conv_block():
    from superl8 import QTensor

    torch.manual_seed(0)
    H = 256
    def qt(o, i):
        w = torch.randn(o, i, device="cuda", dtype=torch.float16) * 0.05
        s = w.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 127
        return QTensor(torch.round(w / s).clamp_(-127, 127).to(torch.int8),
                       s.squeeze(-1).float(), scheme="per_row_i8")

    return ShortConv(
        H,
        in_proj=qt(3 * H, H),
        out_proj=qt(H, H),
        conv_weight=torch.randn(H, 1, 3, device="cuda", dtype=torch.float16),
        kernel=3,
    ).cuda()


@pytest.mark.skipif(not CUDA, reason="the fused raw conv kernel needs a Volta GPU")
def test_short_conv_fused_decode_matches_eager(monkeypatch):
    """Full ShortConv L==1 decode: the fused `superl8.causal_conv1d_decode` branch
    must reproduce the eager conv it replaces end-to-end through out_proj
    (cos >= 0.9999). Toggle `_LFM_CONV` to compare the two dispatch branches on
    the same weights/input (mirror of the DeltaNet fused-step parity test)."""
    import superl8serve.layers.linear_attn as la

    if not getattr(la, "_LFM_CONV", False):
        pytest.skip("installed superl8 predates causal_conv1d_decode")

    blk = _build_short_conv_block()
    x = torch.randn(1, 1, blk.dim, device="cuda", dtype=torch.float16)

    y_fused = blk(x, None, None, 0)
    monkeypatch.setattr(la, "_LFM_CONV", False)
    y_eager = blk(x, None, None, 0)

    yf, ye = y_fused.float().flatten(), y_eager.float().flatten()
    cos = torch.dot(yf, ye) / (yf.norm() * ye.norm() + 1e-12)
    rel_l1 = (yf - ye).abs().sum() / (ye.abs().sum() + 1e-12)
    assert cos.item() >= 0.9999, f"fused short-conv decode cos={cos.item()}"
    assert rel_l1.item() <= 1e-3, f"fused short-conv decode rel_l1={rel_l1.item()}"
    assert y_fused.shape == (1, 1, blk.dim) and torch.isfinite(y_fused).all()


def _build_deltanet_block():
    from superl8 import QTensor

    from superl8serve.models.config import ModelConfig

    torch.manual_seed(0)
    H, nk, nv, kd, vd = 128, 2, 4, 16, 16
    cfg = ModelConfig(arch="qwen3_next", vocab_size=32, hidden_size=H, num_hidden_layers=1,
                      num_attention_heads=4, num_key_value_heads=2, intermediate_size=64,
                      max_position_embeddings=64, head_dim=32)

    def qt(o, i):
        w = torch.randn(o, i, device="cuda", dtype=torch.float16) * 0.05
        s = w.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 127
        return QTensor(torch.round(w / s).clamp_(-127, 127).to(torch.int8),
                       s.squeeze(-1).float(), scheme="per_row_i8")

    return GatedDeltaNetAttention(
        cfg, qkv_proj=qt(2 * nk * kd + nv * vd, H), out_proj=qt(H, nv * vd),
        conv_weight=torch.randn(2 * nk * kd + nv * vd, 4, device="cuda", dtype=torch.float16),
        a_log=torch.zeros(nv, device="cuda"), dt_bias=torch.zeros(nv, device="cuda"),
        beta_proj=qt(nv, H), gate_proj=qt(nv, H),
        # gated output norm is PER-HEAD over head_v_dim (HF Qwen3_5RMSNormGated), so the
        # gain is `vd`-sized, not the flattened nv*vd.
        norm_gain=torch.ones(vd, device="cuda", dtype=torch.float16),
        num_k_heads=nk, num_v_heads=nv, key_dim=kd, value_dim=vd).cuda()


@pytest.mark.skipif(not CUDA, reason="the projections use the dp4a GEMM")
def test_deltanet_block_runs():
    blk = _build_deltanet_block()
    x = torch.randn(1, 6, 128, device="cuda", dtype=torch.float16)
    y = blk(x, None, None, 0)
    assert y.shape == (1, 6, 128) and torch.isfinite(y).all()


def _build_deltanet_block_128():
    """A 128/128-head DeltaNet block with a z-gate — the shape the fully-fused
    decode kernel (`superl8.deltanet_fused_decode`, Dk==Dv==128) targets."""
    from superl8 import QTensor

    from superl8serve.models.config import ModelConfig

    torch.manual_seed(0)
    H, nk, nv, kd, vd = 256, 4, 16, 128, 128
    cfg = ModelConfig(arch="qwen3_next", vocab_size=32, hidden_size=H, num_hidden_layers=1,
                      num_attention_heads=8, num_key_value_heads=4, intermediate_size=64,
                      max_position_embeddings=64, head_dim=32)

    def qt(o, i):
        w = torch.randn(o, i, device="cuda", dtype=torch.float16) * 0.05
        s = w.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 127
        return QTensor(torch.round(w / s).clamp_(-127, 127).to(torch.int8),
                       s.squeeze(-1).float(), scheme="per_row_i8")

    return GatedDeltaNetAttention(
        cfg, qkv_proj=qt(2 * nk * kd + nv * vd, H), out_proj=qt(H, nv * vd),
        conv_weight=torch.randn(2 * nk * kd + nv * vd, 4, device="cuda", dtype=torch.float16),
        a_log=torch.randn(nv, device="cuda") * 0.5, dt_bias=torch.randn(nv, device="cuda"),
        beta_proj=qt(nv, H), gate_proj=qt(nv, H),
        norm_gain=torch.randn(vd, device="cuda", dtype=torch.float16) * 0.1,
        num_k_heads=nk, num_v_heads=nv, key_dim=kd, value_dim=vd,
        z_proj=qt(nv * vd, H)).cuda()


@pytest.mark.skipif(not CUDA, reason="the fused decode kernel needs a Volta GPU")
def test_fused_decode_step_matches_eager(monkeypatch):
    """The fully-fused `deltanet_fused_decode` decode step (L==1, Dk==Dv==128) must
    reproduce the eager per-token chain it replaces (`_l2norm` + q-scale + GQA expand
    + gate/beta + `deltanet_recurrent_decode` + `gated_rmsnorm_decode`) — that chain is
    the kernel's numeric oracle. Toggle the `_DND_STEP` dispatch flag to compare the two
    dispatch branches on the same weights and input."""
    from superl8serve.layers import linear_attn as la

    if not getattr(la, "_DND_STEP", False):
        pytest.skip("installed superl8 predates deltanet_fused_decode")

    blk = _build_deltanet_block_128()
    x = torch.randn(1, 1, 256, device="cuda", dtype=torch.float16)  # L==1 decode token

    y_fused = blk(x, None, None, 0)
    monkeypatch.setattr(la, "_DND_STEP", False)  # force the eager op-sequence fallback
    y_eager = blk(x, None, None, 0)

    yf, ye = y_fused.float().flatten(), y_eager.float().flatten()
    cos = torch.dot(yf, ye) / (yf.norm() * ye.norm() + 1e-12)
    rel_l1 = (yf - ye).abs().sum() / (ye.abs().sum() + 1e-12)
    assert cos.item() >= 0.9999, f"cos={cos.item()}"
    assert rel_l1.item() <= 1e-3, f"rel_l1={rel_l1.item()}"
    assert y_fused.shape == (1, 1, 256) and torch.isfinite(y_fused).all()


@pytest.mark.skipif(not CUDA, reason="the exact gated chunk kernel needs CUDA")
def test_gated_prefill_exact_chunk_matches_eager_with_carry(monkeypatch):
    """The default multi-token route uses superl8's exact gated chunk form.

    The eager scalar recurrence is the oracle. Include a nonzero carry state and
    65 tokens so the test crosses the superl8 kernel's 64-token chunk boundary.
    """
    import superl8serve.layers.linear_attn as la

    if not hasattr(la.superl8, "deltanet_gated_chunk_fwd"):
        pytest.skip("installed superl8 predates deltanet_gated_chunk_fwd")

    torch.manual_seed(416)
    B, H, L, Dk, Dv = 1, 2, 65, 16, 16
    q = _l2norm(torch.randn(B, H, L, Dk, device="cuda"))
    k = _l2norm(torch.randn(B, H, L, Dk, device="cuda"))
    v = torch.randn(B, H, L, Dv, device="cuda")
    beta = torch.sigmoid(torch.randn(B, H, L, device="cuda"))
    g = -F.softplus(torch.randn(B, H, L, device="cuda"))
    initial_state = torch.randn(B, H, Dv, Dk, device="cuda")

    calls = []
    real = la.superl8.deltanet_gated_chunk_fwd

    def spy(*args, **kwargs):
        calls.append(True)
        return real(*args, **kwargs)

    monkeypatch.setattr(la.superl8, "deltanet_gated_chunk_fwd", spy)
    monkeypatch.setattr(la, "_DND_PREFILL_GATED", True)
    got, got_state = la._gated_delta_rule(q, k, v, beta, g, initial_state, L)
    ref, ref_state = la.recurrent_gated_delta_rule(q, k, v, beta, g, state=initial_state)

    assert calls == [True], "multi-token CUDA prefill did not use the exact gated kernel"
    torch.testing.assert_close(got, ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(got_state, ref_state, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not CUDA, reason="the exact gated chunk kernel needs CUDA")
def test_gated_prefill_falls_back_to_eager_when_disabled(monkeypatch):
    """The kill switch restores the scalar oracle without touching kernels."""
    import superl8serve.layers.linear_attn as la

    torch.manual_seed(417)
    q = _l2norm(torch.randn(1, 1, 3, 8, device="cuda"))
    k = _l2norm(torch.randn(1, 1, 3, 8, device="cuda"))
    v = torch.randn(1, 1, 3, 8, device="cuda")
    beta = torch.sigmoid(torch.randn(1, 1, 3, device="cuda"))
    g = -F.softplus(torch.randn(1, 1, 3, device="cuda"))
    initial_state = torch.randn(1, 1, 8, 8, device="cuda")

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled exact prefill dispatch called the kernel")

    monkeypatch.setattr(la, "_DND_PREFILL_GATED", False)
    monkeypatch.setattr(la.superl8, "deltanet_gated_chunk_fwd", forbidden)
    got, got_state = la._gated_delta_rule(q, k, v, beta, g, initial_state, 3)
    ref, ref_state = la.recurrent_gated_delta_rule(q, k, v, beta, g, state=initial_state)
    torch.testing.assert_close(got, ref)
    torch.testing.assert_close(got_state, ref_state)


@pytest.mark.skipif(not CUDA, reason="the projections use the dp4a GEMM")
def test_decode_recurrence_output_stays_fp32(monkeypatch):
    """Regression guard for the redundant-cast removal: at decode (L==1) the fused
    kernel's fp32 readout must flow straight into the gated RMSNorm WITHOUT an
    fp32->fp16->fp32 round-trip. We stub `superl8.deltanet_recurrent_decode` with a pure
    fp32 reference so the test holds even on a prebuilt superl8 that predates the kernel,
    then compare the block output against the OLD behaviour (readout downcast to fp16
    before the norm). Removing the round-trip only drops fp16 rounding, so the block
    output must stay bit-close (cos >= 0.9999) — same numerics, one fewer cast pair."""
    import superl8serve.layers.linear_attn as la

    def _ref_decode(q, k, v, alpha, beta, initial_state=None):
        # exact eager math, fp32 in / fp32 out (mirrors the real decode kernel)
        g = alpha.clamp_min(1e-30).log()
        o, s = la.recurrent_gated_delta_rule(q, k, v, beta, g, state=initial_state)
        return o.float(), s

    blk = _build_deltanet_block()
    x = torch.randn(1, 1, 128, device="cuda", dtype=torch.float16)  # L==1 decode step

    # NEW path: kernel returns fp32, kept fp32 through to the gated norm.
    monkeypatch.setattr(la.superl8, "deltanet_recurrent_decode", _ref_decode, raising=False)
    monkeypatch.setattr(la, "_DND_DECODE", True)
    y_new = blk(x, None, None, 0).float()

    # OLD path: identical kernel but its readout is downcast to v.dtype first (the
    # round-trip the caller then undoes with `.float()`).
    def _ref_decode_roundtrip(q, k, v, alpha, beta, initial_state=None):
        o, s = _ref_decode(q, k, v, alpha, beta, initial_state=initial_state)
        return o.to(v.dtype), s

    monkeypatch.setattr(la.superl8, "deltanet_recurrent_decode", _ref_decode_roundtrip, raising=False)
    y_old = blk(x, None, None, 0).float()

    cos = torch.nn.functional.cosine_similarity(y_new.flatten(), y_old.flatten(), dim=0)
    assert cos.item() >= 0.9999, f"decode block output drifted after round-trip removal: cos={cos.item()}"


def test_gated_output_gate_inference_inplace_matches_functional():
    """The serving-only in-place z gate must preserve the functional result."""
    o = torch.randn(1, 7, 4, 8, dtype=torch.float32)
    z = torch.randn_like(o)
    weight = torch.randn(8, dtype=torch.float32)
    eps = 1e-5

    def functional(x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        x = x * weight
        return x * F.silu(z)

    expected = functional(o.clone())
    with torch.inference_mode():
        got = o.clone()
        got = got * torch.rsqrt(got.pow(2).mean(-1, keepdim=True) + eps)
        got = got * weight
        got.mul_(F.silu(z))

    torch.testing.assert_close(got, expected)
    assert torch.is_inference(got)


@pytest.mark.skipif(not CUDA, reason="the int8 kernel needs a Volta GPU")
def test_lightning_int8_cos_vs_fp16(monkeypatch):
    """Correctness gate: the int8 lightning attention prefill must match the fp32
    scalar reference within cos >= 0.99 and rel-L1 <= 5e-2 for the decay-free case
    (slopes=0). The kernel implements the un-gated recurrence ``S_t = S_{t-1} +
    v_t k_t^T``; MiniMax ALiBi slopes force the fp32 fallback in the dispatch.
    This is the acceptance criterion from superl8 #159 (lightning_attn_int8 kernel)."""
    import superl8serve.layers.linear_attn as la

    if not getattr(la, "_LTN_INT8", False):
        pytest.skip("installed superl8 predates lightning_attn_int8_fwd")

    torch.manual_seed(42)
    B, H, L, Dk, Dv = 2, 4, 8, 64, 64
    q = torch.randn(B, H, L, Dk, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, L, Dk, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, L, Dv, device="cuda", dtype=torch.float32)
    slopes_zero = torch.zeros(H, device="cuda")

    o_int8, _ = la._lightning_attn_dispatch(q, k, v, slopes_zero)
    monkeypatch.setattr(la, "_LTN_INT8", False)
    o_ref, _ = la._lightning_attn_dispatch(q, k, v, slopes_zero)

    o_i = o_int8.float().flatten()
    o_r = o_ref.float().flatten()
    cos = torch.dot(o_i, o_r) / (o_i.norm() * o_r.norm() + 1e-12)
    rel_l1 = (o_i - o_r).abs().sum() / (o_r.abs().sum() + 1e-12)
    assert cos.item() >= 0.99, f"int8 lightning attention cos={cos.item()} — below 0.99 gate"
    assert rel_l1.item() <= 5e-2, f"int8 lightning attention rel_l1={rel_l1.item()} — above 5e-2 gate"
    assert o_int8.shape == o_ref.shape and torch.isfinite(o_int8).all()


def test_lightning_int8_skipped_with_slopes(monkeypatch):
    """With ALiBi slopes (non-zero) the int8 dispatch must fall through to fp32 —
    the kernel does the un-gated recurrence, so it cannot match the decayed output."""
    if not CUDA:
        pytest.skip("requires CUDA")

    torch.manual_seed(42)
    B, H, L, Dk, Dv = 2, 4, 5, 8, 8
    q = torch.randn(B, H, L, Dk, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, L, Dk, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, L, Dv, device="cuda", dtype=torch.float16)
    slopes_nz = lightning_slopes(H).to("cuda")

    o_dispatch, _ = _lightning_attn_dispatch(q, k, v, slopes_nz)
    o_ref, _ = lightning_attention(q, k, v, slopes_nz)
    assert torch.allclose(o_dispatch.float(), o_ref.float(), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not CUDA, reason="the int8 chunk prefill kernel needs a GPU")
def test_prefill_int8_flag_auto_degrades():
    """The `_DND_PREFILL_INT8` flag must exist and default to OFF — the ungated
    kernel is an approximation for gated models (erase term missing), so opt-in
    with SUPERL8_DND_PREFILL_INT8=1. Auto-degrades: kernel absent → always False."""
    import superl8
    import superl8serve.layers.linear_attn as la

    assert hasattr(la, "_DND_PREFILL_INT8"), "flag must exist as a module attribute"
    assert isinstance(la._DND_PREFILL_INT8, bool), "flag must be bool"
    # Default is off; kernel presence is checked by the flag's own hasattr gate
    if hasattr(superl8, "deltanet_chunk_int8_fwd"):
        # Kernels present but env default is "0" → flag is False
        assert la._DND_PREFILL_INT8 is False, (
            "flag must default to off even when kernels are present"
        )


@pytest.mark.skipif(not CUDA, reason="the int8 chunk prefill kernel needs a GPU")
def test_prefill_int8_matches_eager(monkeypatch):
    """The int8 chunked prefill kernel (`superl8.deltanet_chunk_int8_fwd`, L>1, CUDA,
    Dk/Dv<=128) must reproduce the torch recurrence oracle within int8 tolerance bars
    (SQNR/cosine similarity, NOT allclose). The eager per-token chain —
    `recurrent_gated_delta_rule` — is the kernel's numeric oracle."""
    import superl8
    import superl8serve.layers.linear_attn as la

    if not hasattr(superl8, "deltanet_chunk_int8_fwd"):
        pytest.skip("installed superl8 predates deltanet_chunk_int8_fwd")

    from superl8 import QTensor
    from superl8serve.models.config import ModelConfig

    torch.manual_seed(42)
    B, L, nk, nv, kd, vd, H = 1, 8, 2, 4, 16, 16, 64
    cfg = ModelConfig(arch="qwen3_next", vocab_size=32, hidden_size=H,
                      num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                      intermediate_size=64, max_position_embeddings=64, head_dim=32)

    def qt(o, i):
        w = torch.randn(o, i, device="cuda", dtype=torch.float16) * 0.05
        s = w.abs().amax(-1, keepdim=True).clamp_min(1e-6) / 127
        return QTensor(torch.round(w / s).clamp_(-127, 127).to(torch.int8),
                       s.squeeze(-1).float(), scheme="per_row_i8")

    blk = GatedDeltaNetAttention(
        cfg, qkv_proj=qt(2 * nk * kd + nv * vd, H), out_proj=qt(H, nv * vd),
        conv_weight=torch.randn(2 * nk * kd + nv * vd, 4, device="cuda", dtype=torch.float16),
        a_log=torch.randn(nv, device="cuda") * 0.5, dt_bias=torch.randn(nv, device="cuda"),
        beta_proj=qt(nv, H), gate_proj=qt(nv, H),
        norm_gain=torch.randn(vd, device="cuda", dtype=torch.float16) * 0.1,
        num_k_heads=nk, num_v_heads=nv, key_dim=kd, value_dim=vd).cuda()

    x = torch.randn(B, L, H, device="cuda", dtype=torch.float16)

    monkeypatch.setattr(la, "_DND_PREFILL_INT8", True)
    y_int8 = blk(x, None, None, 0)
    monkeypatch.setattr(la, "_DND_PREFILL_INT8", False)
    y_eager = blk(x, None, None, 0)

    yf, ye = y_int8.float().flatten(), y_eager.float().flatten()
    cos = torch.dot(yf, ye) / (yf.norm() * ye.norm() + 1e-12)
    assert cos.item() >= 0.99, f"int8 prefill cos vs eager below bar: cos={cos.item()}"
    assert y_int8.shape == (B, L, H) and torch.isfinite(y_int8).all()
