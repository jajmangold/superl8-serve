# SPDX-License-Identifier: MIT
import torch

from superl8serve.models.cache import RecurrentStateCache


def test_clear_slot_zeroes_static_inference_tensor_from_normal_context():
    """Regression (GPU5 HTTP-disconnect crash): clear_slot must zero a static
    per-slot buffer even when the buffer was created under torch.inference_mode()
    (CUDA-graph capture) but the slot is cleared from a normal autograd context.
    Without the fix `t[slot].zero_()` raised 'Inplace update to inference tensor
    outside InferenceMode is not allowed', killing the engine."""
    cache = RecurrentStateCache()
    cache.enable_static_buffers(4)
    with torch.inference_mode():
        cache.bind([1])
        cache.set_state(0, torch.ones(1, 3, dtype=torch.float16))
        cache.set_conv_tail(1, torch.full((1, 3), 7.0, dtype=torch.float16))
    assert cache._state_buf[0][1].sum().item() == 3.0
    assert cache._conv_buf[1][1].sum().item() == 21.0

    cache.clear_slot(1)

    assert cache._state_buf[0][1].sum().item() == 0.0
    assert cache._conv_buf[1][1].sum().item() == 0.0
