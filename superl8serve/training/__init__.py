# SPDX-License-Identifier: MIT
"""LoRA / QLoRA adapter machinery for fine-tuning on quantized int8 base (issue #129)."""

from superl8serve.training.lora import LoRAConfig, LoRALayer, inject_lora, merge_lora, unmerge_lora  # noqa: F401
from superl8serve.training.qlora import (  # noqa: F401
    dequantize_adapter_nf4,
    quantize_adapter_nf4,
)

__all__ = [
    "LoRAConfig",
    "LoRALayer",
    "inject_lora",
    "merge_lora",
    "unmerge_lora",
    "quantize_adapter_nf4",
    "dequantize_adapter_nf4",
]
