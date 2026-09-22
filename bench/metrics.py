# SPDX-License-Identifier: MIT
"""Shared, format-neutral generation benchmark accounting."""

from __future__ import annotations

import os
import statistics


def env_flag(name: str, *, default: bool = False) -> bool:
    """Parse an explicit boolean environment knob without truthy-string traps."""
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, yes/no, or on/off")


def summarize_generation_steps(
    step_times_s: list[float],
    *,
    warmup_decode_steps: int = 1,
    emitted_tokens_per_step: list[int] | None = None,
) -> dict[str, object]:
    """Separate prefill, graph-capture, steady decode, and end-to-end rates.

    An engine generation emits its first token during the first (prefill) step.
    The first decode step commonly performs lazy CUDA-graph capture, so it must not
    be reported as steady decode.  End-to-end throughput remains available under an
    explicit name and includes every step.
    """
    if not step_times_s:
        raise ValueError("step_times_s must contain at least the prefill step")
    if warmup_decode_steps < 0:
        raise ValueError("warmup_decode_steps must be non-negative")
    if emitted_tokens_per_step is not None and len(emitted_tokens_per_step) != len(step_times_s):
        raise ValueError("emitted_tokens_per_step must have the same length as step_times_s")

    prefill_s = step_times_s[0]
    capture = step_times_s[1 : 1 + warmup_decode_steps]
    steady = step_times_s[1 + warmup_decode_steps :]
    median_decode_s = statistics.median(steady) if steady else 0.0
    total_s = sum(step_times_s)
    if emitted_tokens_per_step is None:
        steady_tokens = len(steady)
        steady_tok_s = (1.0 / median_decode_s) if median_decode_s else 0.0
        total_tokens = len(step_times_s)
    else:
        steady_tokens = sum(emitted_tokens_per_step[1 + warmup_decode_steps :])
        steady_tok_s = steady_tokens / sum(steady) if steady else 0.0
        total_tokens = sum(emitted_tokens_per_step)
    return {
        "prefill_s": prefill_s,
        "graph_capture_s": sum(capture),
        "steady_decode_s": steady,
        "steady_decode_tokens": steady_tokens,
        "steady_decode_tok_s": steady_tok_s,
        "end_to_end_tok_s": total_tokens / total_s if total_s else 0.0,
        "total_s": total_s,
    }
