# SPDX-License-Identifier: MIT
"""Image preprocessing tests — compares against HuggingFace AutoImageProcessor.

Rel-L1 between our ``pixel_values`` and HF's must be ≤ 1e-3 (aggressive
because the pre-processing path is pure floating-point arithmetic with no
quantisation noise).
"""

import pytest

pytest.importorskip("transformers")
pytest.importorskip("torchvision")

# Patch before any transformers import to avoid flash_attn linking issues
# inside this prebuilt container.
import transformers.modeling_flash_attention_utils as _mfa  # noqa: E402

_mfa.flash_attn_supports_top_left_mask = lambda: False  # type: ignore[method-assign]

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from transformers.models.qwen2_vl.image_processing_qwen2_vl import (  # noqa: E402
    Qwen2VLImageProcessor,
)
from transformers.models.llava.image_processing_llava import (  # noqa: E402
    LlavaImageProcessor,
)

from superl8serve.multimodal.preprocess import (  # noqa: E402
    preprocess_qwen2_5_vl,
    preprocess_llava,
    smart_resize,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rel_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return float((a - b).abs().sum() / (b.abs().sum() + 1e-12))


def _make_test_image(h: int, w: int, seed: int = 42) -> Image.Image:
    """Colour-gradient test image (deterministic)."""
    rng = __import__("numpy").random.RandomState(seed)
    arr = rng.randint(0, 256, (h, w, 3), dtype="uint8").astype("float64")
    for y in range(h):
        arr[y, :, :] = arr[y, :, :] * (h - y) // h
    return Image.fromarray(arr.astype("uint8"), "RGB")


_TOL_REL_L1 = 1e-3


# ---------------------------------------------------------------------------
# smart_resize unit
# ---------------------------------------------------------------------------


def test_smart_resize_noop():
    """Dimensions already divisible by factor and within limits → no change."""
    h, w = smart_resize(448, 448, factor=28, min_pixels=3136, max_pixels=1_000_000)
    assert (h, w) == (448, 448)


def test_smart_resize_downscale():
    """Image too large → downscaled while keeping factor alignment."""
    h, w = smart_resize(800, 600, factor=28, min_pixels=3136, max_pixels=200_000)
    assert h % 28 == 0
    assert w % 28 == 0
    assert h * w <= 200_000


def test_smart_resize_upscale():
    """Image too small → upscaled while keeping factor alignment."""
    h, w = smart_resize(56, 56, factor=28, min_pixels=100_000, max_pixels=1_000_000)
    assert h % 28 == 0
    assert w % 28 == 0
    assert h * w >= 100_000


# ---------------------------------------------------------------------------
# Qwen2.5-VL — compare against HF Qwen2VLImageProcessor
# ---------------------------------------------------------------------------


def test_preprocess_qwen2_5_vl_matches_hf():
    """pixel_values from our implementation vs HF Qwen2VLImageProcessor."""
    img = _make_test_image(336, 224)

    min_px = 56 * 56
    max_px = 28 * 28 * 1280
    patch_sz = 14
    merge_sz = 2
    temporal_patch_sz = 2

    # --- HF reference ---
    hf_processor = Qwen2VLImageProcessor(
        do_resize=True,
        do_rescale=True,
        do_normalize=True,
        do_convert_rgb=True,
        patch_size=patch_sz,
        merge_size=merge_sz,
        temporal_patch_size=temporal_patch_sz,
        min_pixels=min_px,
        max_pixels=max_px,
    )
    hf_out = hf_processor(images=img, return_tensors="pt")

    # --- our implementation ---
    our_out = preprocess_qwen2_5_vl(
        img,
        min_pixels=min_px,
        max_pixels=max_px,
        patch_size=patch_sz,
        merge_size=merge_sz,
        temporal_patch_size=temporal_patch_sz,
    )

    # --- compare shapes ---
    assert our_out["pixel_values"].shape == hf_out["pixel_values"].shape, (
        f"pixel_values shape mismatch: "
        f"ours {our_out['pixel_values'].shape} vs HF {hf_out['pixel_values'].shape}"
    )
    assert our_out["image_grid_thw"].shape == hf_out["image_grid_thw"].shape, (
        "image_grid_thw shape mismatch"
    )

    # --- compare values (tight tolerance) ---
    rel = _rel_l1(our_out["pixel_values"], hf_out["pixel_values"])
    assert rel <= _TOL_REL_L1, f"Qwen2.5-VL pixel_values rel-L1={rel:.6e} > {_TOL_REL_L1}"

    # grid/thw must match exactly
    assert torch.equal(our_out["image_grid_thw"], hf_out["image_grid_thw"]), (
        f"image_grid_thw mismatch: ours {our_out['image_grid_thw']} vs HF {hf_out['image_grid_thw']}"
    )


def test_preprocess_qwen2_5_vl_numpy_input():
    """Accepts ndarray input and produces same result as PIL."""
    img = _make_test_image(224, 224)
    import numpy as np

    arr = np.array(img)
    min_px = 56 * 56
    max_px = 28 * 28 * 1280

    out_pil = preprocess_qwen2_5_vl(img, min_pixels=min_px, max_pixels=max_px)
    out_np = preprocess_qwen2_5_vl(arr, min_pixels=min_px, max_pixels=max_px)

    rel = _rel_l1(out_pil["pixel_values"], out_np["pixel_values"])
    assert rel <= 1e-6, f"PIL vs ndarray rel-L1={rel:.6e}"


def test_preprocess_qwen2_5_vl_base64_input():
    """Accepts base64 string input."""
    import base64
    import io

    img = _make_test_image(224, 224)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    min_px = 56 * 56
    max_px = 28 * 28 * 1280

    out_pil = preprocess_qwen2_5_vl(img, min_pixels=min_px, max_pixels=max_px)
    out_b64 = preprocess_qwen2_5_vl(b64, min_pixels=min_px, max_pixels=max_px)

    rel = _rel_l1(out_pil["pixel_values"], out_b64["pixel_values"])
    assert rel <= 1e-6, f"PIL vs base64 rel-L1={rel:.6e}"


# ---------------------------------------------------------------------------
# LLaVA — compare against HF LlavaImageProcessor
# ---------------------------------------------------------------------------


def test_preprocess_llava_matches_hf():
    """pixel_values from our implementation vs HF LlavaImageProcessor."""
    img = _make_test_image(336, 336)
    image_size = 336

    # --- HF reference ---
    hf_processor = LlavaImageProcessor(
        do_resize=True,
        size={"height": image_size, "width": image_size},
        do_center_crop=False,
        do_rescale=True,
        do_normalize=True,
    )
    hf_out = hf_processor(images=img, return_tensors="pt")

    # --- our implementation ---
    our_out = preprocess_llava(img, image_size=image_size)

    # --- compare shapes ---
    assert our_out["pixel_values"].shape == hf_out["pixel_values"].shape, (
        f"pixel_values shape mismatch: "
        f"ours {our_out['pixel_values'].shape} vs HF {hf_out['pixel_values'].shape}"
    )

    # --- compare values ---
    rel = _rel_l1(our_out["pixel_values"], hf_out["pixel_values"])
    assert rel <= _TOL_REL_L1, f"LLaVA pixel_values rel-L1={rel:.6e} > {_TOL_REL_L1}"


def test_preprocess_llava_non_square():
    """Non-square input → still returns fixed 336×336 output matching HF."""
    img = _make_test_image(480, 320)
    image_size = 336

    hf_processor = LlavaImageProcessor(
        do_resize=True,
        size={"height": image_size, "width": image_size},
        do_center_crop=False,
        do_rescale=True,
        do_normalize=True,
    )
    hf_out = hf_processor(images=img, return_tensors="pt")

    our_out = preprocess_llava(img, image_size=image_size)

    assert our_out["pixel_values"].shape == hf_out["pixel_values"].shape
    rel = _rel_l1(our_out["pixel_values"], hf_out["pixel_values"])
    assert rel <= _TOL_REL_L1, f"LLaVA non-square rel-L1={rel:.6e}"
