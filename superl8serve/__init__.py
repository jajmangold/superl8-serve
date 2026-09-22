# SPDX-License-Identifier: MIT
"""superl8-serve — W8A8 dp4a inference server for Volta / CMP 100-210, over `superl8`."""

import os


def _check_torch_build() -> None:
    """Fail loudly if torch is not the +cu129 (CUDA 12.9) sm_70 build this server
    requires. `pyproject.toml` can only pin the bare `torch==2.10.0`, so a naive
    `pip install` could resolve a +cpu / mismatched-CUDA wheel that has none of the
    dp4a kernels — this turns that into a clear ImportError at startup instead of a
    baffling CUDA error deep in a kernel launch. Set `SUPERL8_SERVE_SKIP_TORCH_CHECK=1`
    to bypass (e.g. a docs/CI job that only imports pure-Python helpers)."""
    if os.environ.get("SUPERL8_SERVE_SKIP_TORCH_CHECK"):
        return
    try:
        import torch
    except ImportError as e:  # pragma: no cover - torch is a hard dependency
        raise ImportError(
            "superl8-serve requires torch 2.10.0+cu129 (the Volta/sm_70 build); torch is "
            "not installed. Install it from the cu129 index:\n"
            "  pip install torch==2.10.0+cu129 "
            "--index-url https://download.pytorch.org/whl/cu129"
        ) from e
    cuda_ver = getattr(torch.version, "cuda", None)
    if cuda_ver != "12.9":
        raise ImportError(
            f"superl8-serve requires the torch 2.10.0+cu129 build (CUDA 12.9, sm_70), but "
            f"the installed torch is {torch.__version__!r} (torch.version.cuda="
            f"{cuda_ver!r}). This build lacks the sm_70 dp4a kernels. Reinstall from the "
            f"cu129 index:\n"
            f"  pip install torch==2.10.0+cu129 "
            f"--index-url https://download.pytorch.org/whl/cu129\n"
            f"(set SUPERL8_SERVE_SKIP_TORCH_CHECK=1 to bypass this check.)"
        )


_check_torch_build()

from .batch import BatchRequest, BatchResult, generate_batch
from .config import ServeConfig
from .convert import convert_hf_to_superl8, quantize_state_dict
from .engine import LLMEngine, SamplingParams
from .loader import checkpoint_info, load_superl8_checkpoint, load_superl8_state_dict
from .models import (
    ModelConfig,
    ModelRunner,
    build_model,
    is_supported,
    list_models,
)
from .scheduling import (
    ParallelismStrategy,
    estimate_throughput,
    plan_parallelism,
    validate_parallel_config,
)

__all__ = [
    "ServeConfig",
    "load_superl8_checkpoint",
    "load_superl8_state_dict",
    "checkpoint_info",
    "convert_hf_to_superl8",
    "quantize_state_dict",
    "LLMEngine",
    "SamplingParams",
    "ModelConfig",
    "ModelRunner",
    "build_model",
    "is_supported",
    "list_models",
    "BatchRequest",
    "BatchResult",
    "generate_batch",
    "ParallelismStrategy",
    "plan_parallelism",
    "estimate_throughput",
    "validate_parallel_config",
]
__version__ = "0.1.0"
