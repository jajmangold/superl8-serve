# SPDX-License-Identifier: MIT
"""Dequantize FP8 checkpoints so they can be re-quantized to `.superl8` int8.

Big MoE releases (DeepSeek-V3/V4, Tencent Hy3) ship **FP8-only** — often the only
weights available, and always ~half the download of bf16. FP8 (`float8_e4m3fn`) is
lossy storage plus a companion scale; to reach our int8 dp4a format we first
reconstruct the real values (`fp8 * scale`), then feed fp32 to the int8 quantizer.

Three scale layouts are handled (detected from `config.json`'s `quantization_config`):

- **static / per-channel** (`weight_block_size` absent, e.g. Hy3-FP8): one
  `weight_scale` per tensor (scalar) or per output row (`[O]`/`[O,1]`).
- **block** (`weight_block_size=[bh,bw]`, `scale_fmt` absent/not `ue8m0`, e.g.
  DeepSeek-V3.1): a `weight_scale_inv` grid `[ceil(O/bh), ceil(I/bw)]`, each entry
  scaling a `bh×bw` tile as an fp32 MULTIPLIER (despite the ``_inv`` name; `real =
  fp8 * scale_inv`).
- **MX / microscaling block** (`weight_block_size=[bh,bw]`, `scale_fmt="ue8m0"`,
  e.g. DeepSeek-V4-Flash, MiniMax-M3-MXFP8): same `[ceil(O/bh), ceil(I/bw)]` grid,
  but each entry is a UE8M0 byte (unsigned biased power-of-2 exponent, no mantissa)
  rather than an fp32 multiplier: `real = fp8 * 2**(e8m0 - 127)`. These checkpoints
  also pair weights with their scale under non-standard names (`hc_attn_scale`,
  `hc_ffn_scale`, `hc_head_scale`, or plain `scale`, instead of `weight_scale_inv`)
  — `fp8_scale_name` tries all of them and returns whichever actually exists next to
  the weight, so it doesn't need to know in advance which category a given weight
  falls into.

`dequantize_fp8` / `dequantize_mxfp8` are pure/tested; `is_fp8` gates the convert
path.
"""
from __future__ import annotations

import torch

# torch exposes float8_e4m3fn (and e5m2); e4m3 is what these checkpoints use.
_FP8_DTYPES = tuple(
    d for d in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None))
    if d is not None
)


def is_fp8(t: torch.Tensor) -> bool:
    return t.dtype in _FP8_DTYPES


def dequantize_fp8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_size: list[int] | tuple[int, int] | None = None,
) -> torch.Tensor:
    """Reconstruct fp32 values from an fp8 `weight` [O, I] and its `scale`.

    `block_size=[bh,bw]` -> block layout (`scale` is the `[O/bh, I/bw]` grid);
    otherwise static/per-channel (`scale` scalar, `[O]`, or `[O,1]`)."""
    wf = weight.to(torch.float32)
    sc = scale.to(torch.float32)

    if block_size is None:
        if sc.numel() == 1:
            return wf * sc.reshape(())
        # per-output-channel: one scale per row
        return wf * sc.reshape(-1, 1)

    if wf.dim() != 2:
        raise ValueError(f"block fp8 dequant expects a 2-D weight, got {tuple(wf.shape)}")
    bh, bw = int(block_size[0]), int(block_size[1])
    o, i = wf.shape
    # Expand the [ceil(O/bh), ceil(I/bw)] grid to [O, I], clipping the ragged tail.
    full = sc.repeat_interleave(bh, dim=0).repeat_interleave(bw, dim=1)[:o, :i]
    if full.shape != wf.shape:
        raise ValueError(f"fp8 block scale {tuple(sc.shape)} x{block_size} -> "
                         f"{tuple(full.shape)} != weight {tuple(wf.shape)}")
    return wf * full


def dequantize_mxfp8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_size: list[int] | tuple[int, int] = (128, 128),
) -> torch.Tensor:
    """Reconstruct fp32 values from a microscaling (MX) fp8 `weight` [O, I]
    (`float8_e4m3fn`) and its UE8M0 block `scale` grid `[ceil(O/bh), ceil(I/bw)]`.

    UE8M0 has no mantissa or sign: the stored byte IS a biased power-of-2 exponent,
    so dequant is a pure power-of-2 multiply (`fp8.float() * 2**(e8m0 - 127)`) rather
    than the fp32-multiply `dequantize_fp8` block path uses. `scale` is read as
    plain `uint8`/int (the biased byte, per the checkpoint's `scale_fmt=ue8m0`), or
    as the dedicated `float8_e8m0fnu` dtype where the torch build has one (whose
    fp32 cast already equals `2**(e8m0 - 127)`, so the bias is not re-applied)."""
    wf = weight.to(torch.float32)
    if wf.dim() != 2:
        raise ValueError(f"MXFP8 dequant expects a 2-D weight, got {tuple(wf.shape)}")
    bh, bw = int(block_size[0]), int(block_size[1])
    o, i = wf.shape

    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is not None and scale.dtype == e8m0_dtype:
        mult = scale.to(torch.float32)                  # cast already yields 2**(e8m0-127)
    else:
        mult = torch.exp2(scale.to(torch.int32).to(torch.float32) - 127.0)

    full = mult.repeat_interleave(bh, dim=0).repeat_interleave(bw, dim=1)[:o, :i]
    if full.shape != wf.shape:
        raise ValueError(f"MXFP8 block scale {tuple(scale.shape)} x{block_size} -> "
                         f"{tuple(full.shape)} != weight {tuple(wf.shape)}")
    return wf * full


# MX/UE8M0 checkpoints (DeepSeek-V4-Flash, MiniMax-M3-MXFP8) pair a weight with its
# scale under one of these names instead of `weight_scale_inv` — see module docstring.
# Public (imported by convert.py) since these are only safe to drop-on-sight for
# checkpoints already known to be fp8-sourced: unlike `weight_scale`/`weight_scale_inv`,
# a bare `.scale` suffix could otherwise collide with an unrelated raw fp16/bf16
# tensor (e.g. a logit-scale parameter) in a non-fp8 model.
MX_SCALE_SUFFIXES = (".hc_attn_scale", ".hc_ffn_scale", ".hc_head_scale", ".scale")


def fp8_scale_name(weight_name: str, names: set[str]) -> str | None:
    """Find the scale tensor paired with an fp8 `weight_name`: `X.weight_scale_inv`
    or `X.weight_scale` for standard block/static checkpoints, or one of the MX/
    UE8M0 non-standard names (`X.hc_attn_scale`, `X.hc_ffn_scale`, `X.hc_head_scale`,
    `X.scale`) for DeepSeek-V4-Flash-style checkpoints. Each candidate is only
    returned if it's actually present in `names`, so this doesn't need to know in
    advance which of the MX names a given weight uses."""
    for suf in (".weight_scale_inv", ".weight_scale") + MX_SCALE_SUFFIXES:
        cand = weight_name[: -len(".weight")] + suf if weight_name.endswith(".weight") else None
        if cand and cand in names:
            return cand
    return None
