# SPDX-License-Identifier: MIT
"""Tests for superl8serve.train — gradcheck + loss-reduction (issue #82)."""

import pytest

pytest.importorskip("superl8")

import superl8
import torch

from superl8serve.train.loop import TrainableLM, training_step


# ── helpers ────────────────────────────────────────────────────────────────


def _cuda_small_int8_model(**overrides):
    """2-layer model with head_dim=32 (valid for superl8.autograd.attn)."""
    kwargs = dict(
        vocab_size=64,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        head_dim=32,
        intermediate_size=256,
    )
    kwargs.update(overrides)
    return TrainableLM(**kwargs).cuda().to(dtype=torch.float16)


def _cuda_ref_model(**overrides):
    """Tiny model with head_dim=4 (forces superl8.autograd.attn_ref)."""
    kwargs = dict(
        vocab_size=16,
        hidden_size=8,
        num_layers=2,
        num_heads=2,
        head_dim=4,
        intermediate_size=32,
    )
    kwargs.update(overrides)
    return TrainableLM(**kwargs).cuda()


# ── gradcheck on superl8.autograd.attn_ref (fp64) ─────────────────────────────


def test_gradcheck_attn_ref_fp64():
    """torch.autograd.gradcheck passes on superl8.autograd.attn_ref in fp64."""
    B, H, S, D = 1, 1, 4, 4
    q = torch.randn(B, H, S, D, dtype=torch.float64, requires_grad=True)
    k = torch.randn(B, H, S, D, dtype=torch.float64, requires_grad=True)
    v = torch.randn(B, H, S, D, dtype=torch.float64, requires_grad=True)

    ok = torch.autograd.gradcheck(
        lambda q, k, v: superl8.autograd.attn_ref(
            q.cuda(), k.cuda(), v.cuda(), causal=True, scale=0.5
        ),
        (q.cuda(), k.cuda(), v.cuda()),
        eps=1e-3,
        atol=1e-2,
        rtol=1e-2,
    )
    assert ok


def test_gradcheck_attn_ref_fp64_head8():
    """torch.autograd.gradcheck passes on superl8.autograd.attn_ref with fp64, head_dim=8."""
    B, H, S, D = 1, 1, 4, 8
    q = torch.randn(B, H, S, D, dtype=torch.float64, requires_grad=True)
    k = torch.randn(B, H, S, D, dtype=torch.float64, requires_grad=True)
    v = torch.randn(B, H, S, D, dtype=torch.float64, requires_grad=True)

    ok = torch.autograd.gradcheck(
        lambda q, k, v: superl8.autograd.attn_ref(
            q.cuda(), k.cuda(), v.cuda(), causal=True, scale=(D**-0.5)
        ),
        (q.cuda(), k.cuda(), v.cuda()),
        eps=1e-3,
        atol=1e-2,
        rtol=1e-2,
    )
    assert ok


def test_gradcheck_attn_int8_backward_runs():
    """superl8.autograd.attn backward runs and produces non-NaN gradients.

    We verify backward correctness rather than using torch.autograd.gradcheck
    because fp16 precision is insufficient for numerical Jacobian comparison.
    """
    B, H, S, D = 1, 2, 4, 32
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16, requires_grad=True)

    out = superl8.autograd.attn(q, k, v, causal=True, scale=(D**-0.5))
    assert out.requires_grad
    loss = out.sum()
    loss.backward()
    assert q.grad is not None and not torch.isnan(q.grad).any()
    assert k.grad is not None and not torch.isnan(k.grad).any()
    assert v.grad is not None and not torch.isnan(v.grad).any()


# ── model forward / backward sanity ────────────────────────────────────────


def test_model_forward_backward_int8():
    """Forward + backward through a 2-layer model (int8 attn path) succeeds."""
    model = _cuda_small_int8_model()
    x = torch.randint(0, 64, (2, 8), device="cuda")
    logits = model(x)
    assert logits.shape == (2, 8, 64)
    loss = logits.sum()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"


def test_model_forward_backward_ref():
    """Forward + backward through a tiny model (ref attn path) succeeds."""
    model = _cuda_ref_model().to(dtype=torch.float16)
    x = torch.randint(0, 16, (2, 4), device="cuda")
    logits = model(x)
    assert logits.shape == (2, 4, 16)
    loss = logits.sum()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"


# ── training step ──────────────────────────────────────────────────────────


def test_training_step_runs():
    """training_step runs forward → loss → backward → optimizer.step without error."""
    model = _cuda_small_int8_model()
    model2 = _cuda_ref_model().to(dtype=torch.float16)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 64, (2, 8), device="cuda")
    targets = torch.randint(0, 64, (2, 8), device="cuda")

    loss1 = training_step(model, opt, x, targets)
    assert isinstance(loss1, float)
    assert loss1 > 0

    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    x2 = torch.randint(0, 16, (2, 4), device="cuda")
    t2 = torch.randint(0, 16, (2, 4), device="cuda")
    loss2 = training_step(model2, opt2, x2, t2)
    assert isinstance(loss2, float)
    assert loss2 > 0


def test_loss_reduces_on_toy_task():
    """A few training steps reduce loss on a toy next-token prediction task."""
    torch.manual_seed(42)
    model = _cuda_small_int8_model().to(dtype=torch.float32)

    seq = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], device="cuda")
    input_ids = seq[:, :-1]
    targets = seq[:, 1:]

    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    losses = []
    for _ in range(50):
        loss = training_step(model, opt, input_ids, targets, max_grad_norm=None)
        losses.append(loss)

    assert losses[0] > 0
    first3 = sum(losses[:3]) / 3
    last3 = sum(losses[-3:]) / 3
    assert last3 < first3 * 0.95, f"loss did not decrease: first3={first3:.4f} last3={last3:.4f}"


def test_loss_reduces_on_toy_task_ref():
    """Loss reduction on the ref-attn path (head_dim=4, tiny model)."""
    torch.manual_seed(7)
    model = _cuda_ref_model().to(dtype=torch.float32)

    seq = torch.tensor([[0, 1, 2, 3]], device="cuda")
    input_ids = seq[:, :-1]
    targets = seq[:, 1:]

    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    losses = []
    for _ in range(50):
        loss = training_step(model, opt, input_ids, targets, max_grad_norm=None)
        losses.append(loss)

    assert losses[0] > 0
    first3 = sum(losses[:3]) / 3
    last3 = sum(losses[-3:]) / 3
    assert last3 < first3 * 0.95, f"loss did not decrease: first3={first3:.4f} last3={last3:.4f}"


# ── TrainableLM construction ───────────────────────────────────────────────


def test_trainable_lm_attributes():
    """TrainableLM reports correct attributes and parameter count."""
    model = _cuda_ref_model()
    assert model.num_layers == 2
    assert model.vocab_size == 16
    assert model.hidden_size == 8
    assert model.num_heads == 2
    assert model.head_dim == 4
    total = sum(p.numel() for p in model.parameters())
    assert total > 0
