# SPDX-License-Identifier: MIT
"""Image preprocessing: resize/patchify/normalize — HF-processor compatible.

Produces ``pixel_values`` (+ grid/thw metadata) matching the HuggingFace
``AutoImageProcessor`` for Qwen2.5-VL (dynamic resolution) and LLaVA (fixed
336).  Accepts PIL images, NumPy arrays, or decoded base64 strings.

All numerical ops use vanilla PyTorch / torchvision so the module is
lightweight and CUDA-compatible without requiring ``transformers`` at import.
"""

from __future__ import annotations

import base64
import io
import math
import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Constants -- match HuggingFace image processors
# ---------------------------------------------------------------------------

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

QWEN2_5_VL_DEFAULT_MIN_PIXELS = 256 * 28 * 28  # 200_704
QWEN2_5_VL_DEFAULT_MAX_PIXELS = 1280 * 28 * 28  # 1_003_520

LLAVA_DEFAULT_IMAGE_SIZE = 336

PATCH_SIZE = 14  # ViT patch size
MERGE_SIZE = 2  # spatial merge factor
TEMPORAL_PATCH_SIZE = 2  # temporal patch size for Qwen2-VL / Qwen2.5-VL


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------


def load_image(
    image: str | bytes | Image.Image | np.ndarray,
) -> Image.Image:
    """Normalize a PIL / ndarray / base64 input to a PIL ``Image`` (RGB)."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    if isinstance(image, str):
        image = base64.b64decode(image)  # now bytes
    if isinstance(image, bytes):
        return Image.open(io.BytesIO(image)).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def _tvf():
    """Lazy torchvision handle. Keeps ``import superl8serve`` torchvision-free for
    text-only deploys: the VLM path is eagerly imported by the model registry, so
    a top-level ``torchvision`` import would hard-require it even to serve a text
    model — and the deploy installs torchvision with ``|| true``, so a failed
    install would silently kill the server on restart. Import only when an image
    is actually preprocessed."""
    from torchvision.transforms.v2 import functional as tvF

    return tvF


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL Image -> ``[C, H, W]`` uint8 tensor."""
    return _tvf().pil_to_tensor(image)  # uint8 [C, H, W]


# ---------------------------------------------------------------------------
# Smart resize (dynamic resolution) -- matches HF ``smart_resize``
# ---------------------------------------------------------------------------


def smart_resize(
    height: int,
    width: int,
    factor: int = PATCH_SIZE * MERGE_SIZE,
    min_pixels: int = QWEN2_5_VL_DEFAULT_MIN_PIXELS,
    max_pixels: int = QWEN2_5_VL_DEFAULT_MAX_PIXELS,
) -> tuple[int, int]:
    """Rescale dimensions so both are divisible by ``factor`` and total pixels
    fall in ``[min_pixels, max_pixels]`` while preserving aspect ratio.

    This is identical to ``Qwen2VLImageProcessor.smart_resize``.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, "
            f"got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------


def resize_image(
    image: torch.Tensor,
    size: tuple[int, int],
    interpolation=None,  # torchvision InterpolationMode; resolved lazily (default BICUBIC)
) -> torch.Tensor:
    """Resize a ``[C, H, W]`` tensor to ``(height, width)``."""
    tvF = _tvf()
    if interpolation is None:
        interpolation = tvF.InterpolationMode.BICUBIC
    return tvF.resize(image, size, interpolation=interpolation, antialias=True)


# ---------------------------------------------------------------------------
# Rescale + normalize
# ---------------------------------------------------------------------------


def rescale_and_normalize(
    image: torch.Tensor,
    do_rescale: bool = True,
    rescale_factor: float = 1.0 / 255.0,
    do_normalize: bool = True,
    mean: tuple[float, float, float] = CLIP_MEAN,
    std: tuple[float, float, float] = CLIP_STD,
) -> torch.Tensor:
    """Rescale by ``rescale_factor`` then normalize with ``mean`` / ``std``.

    Matches HF ``TorchvisionBackend.rescale_and_normalize`` (fused).
    """
    x = image.float()
    if do_rescale:
        x = x * rescale_factor
    if do_normalize:
        mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype).view(3, 1, 1)
        std_t = torch.tensor(std, device=x.device, dtype=x.dtype).view(3, 1, 1)
        x = (x - mean_t) / std_t
    return x


# ---------------------------------------------------------------------------
# Patchify (Qwen2-VL / Qwen2.5-VL style)
# ---------------------------------------------------------------------------


def patchify_qwen2_vl(
    image: torch.Tensor,
    patch_size: int = PATCH_SIZE,
    merge_size: int = MERGE_SIZE,
    temporal_patch_size: int = TEMPORAL_PATCH_SIZE,
) -> torch.Tensor:
    """Convert a resized/normalized ``[1, C, H, W]`` image into patch tokens.

    Returns ``[1, num_patches, C * temporal_patch_size * patch_size ** 2]``
    matching the output of ``Qwen2VLImageProcessor._preprocess``.

    ``num_patches = grid_h * grid_w`` where ``grid_h = H // patch_size``,
    ``grid_w = W // patch_size``.
    """
    C = image.shape[1]
    H, W = image.shape[2], image.shape[3]
    grid_h = H // patch_size
    grid_w = W // patch_size

    patches = image.reshape(
        1,
        C,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    # [1, C, gh/m, m, p, gw/m, m, p]
    patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
    # [1, gh/m, gw/m, m, m, C, p, p]

    # Expand temporal dimension
    patches = patches.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, temporal_patch_size, -1, -1)
    # [1, gh/m, gw/m, m, m, C, t, p, p]

    patches = patches.reshape(
        1,
        grid_h * grid_w,
        C * temporal_patch_size * patch_size * patch_size,
    )
    # [1, gh*gw, C*t*p*p]
    return patches


# ---------------------------------------------------------------------------
# High-level public API
# ---------------------------------------------------------------------------


def preprocess_qwen2_5_vl(
    image: str | bytes | Image.Image | np.ndarray,
    min_pixels: int = QWEN2_5_VL_DEFAULT_MIN_PIXELS,
    max_pixels: int = QWEN2_5_VL_DEFAULT_MAX_PIXELS,
    patch_size: int = PATCH_SIZE,
    merge_size: int = MERGE_SIZE,
    temporal_patch_size: int = TEMPORAL_PATCH_SIZE,
) -> dict[str, torch.Tensor]:
    """Preprocess an image for Qwen2.5-VL (dynamic resolution).

    Returns a dict with:
      * ``pixel_values``: ``[num_patches, C * t * p * p]`` float tensor
        (no batch dimension — matches HuggingFace ``Qwen2VLImageProcessor``)
      * ``image_grid_thw``: ``[1, 3]`` long tensor ``[temporal, grid_h, grid_w]``
    """
    pil = load_image(image)
    tensor = _pil_to_tensor(pil)  # [C, H, W] uint8

    H, W = tensor.shape[1], tensor.shape[2]
    factor = patch_size * merge_size
    target_h, target_w = smart_resize(
        H, W, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels
    )
    resized = resize_image(tensor, (target_h, target_w))
    normalized = rescale_and_normalize(resized.unsqueeze(0))
    pixel_values = patchify_qwen2_vl(
        normalized,
        patch_size=patch_size,
        merge_size=merge_size,
        temporal_patch_size=temporal_patch_size,
    )

    grid_h = target_h // patch_size
    grid_w = target_w // patch_size
    image_grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)

    return {"pixel_values": pixel_values.squeeze(0), "image_grid_thw": image_grid_thw}


# ---------------------------------------------------------------------------
# Qwen3.5-VL preprocessing
# ---------------------------------------------------------------------------

# Qwen3.5 vision differs from Qwen2.5-VL: patch_size 16 (not 14), symmetric [0.5]
# mean/std normalization (not CLIP), and a larger pixel budget. See the shipped
# `preprocessor_config.json` (size.shortest_edge / longest_edge are pixel counts).
QWEN3_5_VL_PATCH_SIZE = 16
QWEN3_5_VL_MIN_PIXELS = 65536  # 256 * 16 * 16
QWEN3_5_VL_MAX_PIXELS = 16777216
QWEN3_5_MEAN = (0.5, 0.5, 0.5)
QWEN3_5_STD = (0.5, 0.5, 0.5)


def preprocess_qwen3_5_vl(
    image: str | bytes | Image.Image | np.ndarray,
    min_pixels: int = QWEN3_5_VL_MIN_PIXELS,
    max_pixels: int = QWEN3_5_VL_MAX_PIXELS,
    patch_size: int = QWEN3_5_VL_PATCH_SIZE,
    merge_size: int = MERGE_SIZE,
    temporal_patch_size: int = TEMPORAL_PATCH_SIZE,
) -> dict[str, torch.Tensor]:
    """Preprocess an image for Qwen3.5-VL (dynamic resolution, patch 16, [0.5] norm).

    Mirrors ``Qwen2VLImageProcessorFast`` as configured for Qwen3.5. Returns:
      * ``pixel_values``: ``[num_patches, C * t * p * p]`` float tensor
      * ``image_grid_thw``: ``[1, 3]`` long tensor ``[temporal, grid_h, grid_w]``

    ``num_patches = grid_h * grid_w``; the number of LLM image-placeholder tokens is
    ``num_patches // merge_size**2``. Feed both fields into the Qwen3.5-VL wrapper via
    ``ForwardContext`` (pixel_values + image_grid_thw).
    """
    pil = load_image(image)
    tensor = _pil_to_tensor(pil)  # [C, H, W] uint8
    H, W = tensor.shape[1], tensor.shape[2]
    factor = patch_size * merge_size
    target_h, target_w = smart_resize(
        H, W, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels
    )
    resized = resize_image(tensor, (target_h, target_w))
    normalized = rescale_and_normalize(
        resized.unsqueeze(0), mean=QWEN3_5_MEAN, std=QWEN3_5_STD
    )
    pixel_values = patchify_qwen2_vl(
        normalized, patch_size=patch_size, merge_size=merge_size,
        temporal_patch_size=temporal_patch_size,
    )
    grid_h = target_h // patch_size
    grid_w = target_w // patch_size
    image_grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)
    return {"pixel_values": pixel_values.squeeze(0), "image_grid_thw": image_grid_thw}


def preprocess_llava(
    image: str | bytes | Image.Image | np.ndarray,
    image_size: int = LLAVA_DEFAULT_IMAGE_SIZE,
) -> dict[str, torch.Tensor]:
    """Preprocess an image for LLaVA (fixed 336).

    Returns a dict with ``pixel_values``: ``[1, 3, H, W]`` float tensor.
    """
    pil = load_image(image)
    tensor = _pil_to_tensor(pil)  # [C, H, W] uint8

    resized = resize_image(tensor, (image_size, image_size))
    pixel_values = rescale_and_normalize(resized.unsqueeze(0))

    return {"pixel_values": pixel_values}
