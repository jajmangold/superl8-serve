# SPDX-License-Identifier: MIT
"""Model registry + concrete architectures. Importing a model module registers it.

Adding a family = drop a `models/<family>.py` with an `@register_model(...)` builder
and import it here. The engine/runner never change.
"""

from .base import CausalLM, ForwardContext
from .cache import KVCache, MLALatentCache, RecurrentStateCache
from .config import ModelConfig
from .registry import build_model, is_supported, list_models, register_model
from .runner import ModelRunner

# Concrete architectures (import = self-register).
from . import qwen3 as _qwen3  # noqa: E402,F401
from . import gemma3 as _gemma3  # noqa: E402,F401
from . import deepseek as _deepseek  # noqa: E402,F401
from . import qwen3_next as _qwen3_next  # noqa: E402,F401
from . import qwen3_5 as _qwen3_5  # noqa: E402,F401
from . import qwen3_5_vl as _qwen3_5_vl  # noqa: E402,F401
from . import lfm2 as _lfm2  # noqa: E402,F401
from . import glm as _glm  # noqa: E402,F401
from . import hunyuan as _hunyuan  # noqa: E402,F401
from . import minimax as _minimax  # noqa: E402,F401
from . import diffusion_gemma as _diffusion_gemma  # noqa: E402,F401
from . import gemma4 as _gemma4  # noqa: E402,F401
from . import vlm as _vlm  # noqa: E402,F401

__all__ = [
    "CausalLM",
    "ForwardContext",
    "KVCache",
    "MLALatentCache",
    "RecurrentStateCache",
    "ModelConfig",
    "ModelRunner",
    "build_model",
    "register_model",
    "is_supported",
    "list_models",
]
