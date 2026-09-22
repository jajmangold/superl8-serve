# SPDX-License-Identifier: MIT
"""Model registry — the seam that makes support fully modular.

A model family registers a builder under one or more architecture keys (HF
`model_type` or `architectures[0]`). `build_model(config, weights)` looks the key
up and hands back a `CausalLM`. Adding a family = one `@register_model(...)` + deco;
no engine, runner, or kernel changes. `list_models()` powers the coverage matrix.
"""
from __future__ import annotations

from typing import Callable

import torch

from .base import CausalLM, Weights
from .config import ModelConfig

_REGISTRY: dict[str, Callable[[ModelConfig, Weights], CausalLM]] = {}
_ALIASES: dict[str, str] = {}


def register_model(*keys: str) -> Callable:
    """Register a builder `fn(config, weights) -> CausalLM` under one or more keys
    (lowercased). The first key is canonical; the rest are aliases."""
    def deco(fn: Callable[[ModelConfig, Weights], CausalLM]):
        canon = keys[0].lower()
        _REGISTRY[canon] = fn
        for k in keys:
            _ALIASES[k.lower()] = canon
        return fn
    return deco


def build_model(config: ModelConfig, weights: Weights) -> CausalLM:
    key = _resolve(config.arch)
    if key is None:
        raise KeyError(
            f"no model registered for arch {config.arch!r}. "
            f"Registered: {sorted(_REGISTRY)}"
        )
    model = _REGISTRY[key](config, weights)
    return _match_activation_dtype(model, config)


def _match_activation_dtype(model: CausalLM, config: ModelConfig) -> CausalLM:
    """Cast the model's fp16 raw params/buffers to the activation dtype (#260).

    A bf16-native checkpoint (torch_dtype bfloat16 -> ``act_dtype()`` bf16) runs its
    residual stream in bf16 so massive-activation channels stay finite. But the raw
    (non-quantized) weights the converter stored as fp16 — RMSNorm gains, tied/fp16
    LM heads, fp16 embedding tables — must MATCH that stream: the fused ``superl8.rmsnorm``
    kernel rejects a weight whose dtype differs from ``x`` ("weight dtype must match x"),
    and ``F.linear`` rejects a bf16 activation against an fp16 weight. Casting fp16 ->
    bf16 is also strictly MORE faithful than the fp16 the converter wrote, since the
    original Gemma/Qwen store these gains in bf16 to begin with. fp32 buffers (the RoPE
    cos/sin tables — numerically load-bearing) and int8 QTensor weights (not nn params)
    are deliberately left untouched. No-op for fp16-native models (act_dtype == fp16).
    """
    act = config.act_dtype()
    if act is torch.float16:
        return model
    for p in model.parameters():
        if p.dtype is torch.float16:
            p.data = p.data.to(act)
    for m in model.modules():
        for name, buf in list(m._buffers.items()):
            if buf is not None and buf.dtype is torch.float16:
                m._buffers[name] = buf.to(act)
    return model


def is_supported(arch: str) -> bool:
    return _resolve(arch) is not None


def list_models() -> list[str]:
    return sorted(_REGISTRY)


def _resolve(arch: str) -> str | None:
    a = arch.lower()
    if a in _REGISTRY:
        return a
    if a in _ALIASES:
        return _ALIASES[a]
    # HF arch class names like "Qwen3ForCausalLM" -> "qwen3". Strip the common
    # suffix and match EXACTLY — never a bare startswith (that wrongly maps
    # "qwen3_next" -> "qwen3"). Aliases (registered explicitly) handle the rest.
    stripped = a.replace("forcausallm", "").replace("forconditionalgeneration", "").rstrip("_")
    return stripped if stripped in _REGISTRY else None
