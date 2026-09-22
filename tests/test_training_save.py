# SPDX-License-Identifier: MIT
"""Round-trip training checkpoint save/load (issue #133).

Verifies that ``save_training_checkpoint`` → ``load_training_checkpoint``
recovers model weights and optimizer state exactly (bitwise match).
"""

import pytest

pytest.importorskip("superl8")

import torch

from superl8serve.loader import load_training_checkpoint, save_training_checkpoint
from superl8serve.train.loop import TrainableLM


def _tiny_model(**kw):
    kwargs = dict(
        vocab_size=16,
        hidden_size=8,
        num_layers=2,
        num_heads=2,
        head_dim=4,
        intermediate_size=32,
    )
    kwargs.update(kw)
    return TrainableLM(**kwargs).to(dtype=torch.float32)


def test_save_training_checkpoint_roundtrip(tmp_path):
    """Model weights + optimizer state survive a save→load cycle bitwise."""
    torch.manual_seed(42)
    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # One training step to populate optimizer state (exp_avg, exp_avg_sq, step)
    x = torch.randint(0, 16, (2, 4))
    targets = torch.randint(0, 16, (2, 4))
    model.train()
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, -2).float(), targets.flatten(0, -1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    # Save weights and optimizer state before they are lost
    model_sd_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    opt_sd_before = {k: v for k, v in opt.state_dict().items()}
    # deep-copy tensor state for later comparison
    opt_tensors_before: dict = {}
    for pid, pstate in opt_sd_before.get("state", {}).items():
        opt_tensors_before[pid] = {
            k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in pstate.items()
        }

    ckpt_path = str(tmp_path / "training.superl8")
    save_training_checkpoint(ckpt_path, model.state_dict(), opt)

    # ── fresh model + optimizer ──────────────────────────────────────────
    model2 = _tiny_model()
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

    model_sd_loaded = load_training_checkpoint(ckpt_path, opt2, device="cpu")

    # Compare model weights
    for name in model_sd_before:
        assert name in model_sd_loaded, f"missing {name}"
        torch.testing.assert_close(
            model_sd_loaded[name],
            model_sd_before[name],
            rtol=0,
            atol=0,
            msg=f"model weight mismatch: {name}",
        )

    # Compare optimizer state
    opt_sd_after = opt2.state_dict()
    for pid, expected_state in opt_tensors_before.items():
        for key, expected_val in expected_state.items():
            actual_val = opt_sd_after["state"][pid][key]
            torch.testing.assert_close(
                actual_val.cpu(),
                expected_val,
                rtol=0,
                atol=0,
                msg=f"optimizer state mismatch: pid={pid} key={key}",
            )


def test_save_training_checkpoint_meta_preserved(tmp_path):
    """User-provided metadata is preserved in the checkpoint."""
    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ckpt_path = str(tmp_path / "meta.superl8")
    save_training_checkpoint(
        ckpt_path,
        model.state_dict(),
        opt,
        meta={"epoch": 5, "loss": 1.23},
    )

    from superl8serve.loader import checkpoint_info

    info = checkpoint_info(ckpt_path)
    meta = info["meta"]
    assert meta.get("epoch") == 5
    assert meta.get("loss") == 1.23
    assert meta.get("training") is True
    assert "optimizer_type" in meta
    assert "optimizer_param_groups" in meta
