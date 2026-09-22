# SPDX-License-Identifier: MIT
"""Reasoning-aware chat output parsing shared by live and batch endpoints."""

from __future__ import annotations

from dataclasses import dataclass

_OPEN = "<think>"
_CLOSE = "</think>"


@dataclass(frozen=True)
class ReasoningText:
    reasoning_content: str | None
    content: str


def prompt_prefills_thinking(tokenizer, prompt_ids: list[int]) -> bool:
    """Return whether the rendered assistant prompt ends with ``<think>``.

    Some reasoning templates put the opening marker in the prompt. Generated text
    consequently starts with reasoning directly and only contains the closing marker.
    Token IDs are used instead of inspecting template source so tokenizer-owned and
    caller-supplied templates behave identically.
    """
    try:
        marker_ids = tokenizer.encode(_OPEN, add_special_tokens=False)
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(marker_ids) and len(prompt_ids) > len(marker_ids) and (
        prompt_ids[-len(marker_ids):] == marker_ids
    )


def _without_partial_close(text: str) -> str:
    """Hold a possible token-split closing marker until the next decode prefix."""
    for length in range(min(len(text), len(_CLOSE) - 1), 0, -1):
        if text.endswith(_CLOSE[:length]):
            return text[:-length]
    return text


def split_reasoning_content(
    text: str, *, prompt_prefilled: bool = False, final: bool = True
) -> ReasoningText:
    """Split decoded chat text into OpenAI reasoning and final-answer fields.

    ``final=False`` is used while streaming. It withholds partial delimiter text so
    token boundaries such as ``</thi`` + ``nk>`` never leak into either field.
    """
    explicit = text.startswith(_OPEN)
    if not prompt_prefilled and not explicit:
        if not final and _OPEN.startswith(text):
            return ReasoningText(None, "")
        close_at = text.find(_CLOSE)
        if close_at < 0:
            return ReasoningText(None, text)
    else:
        close_at = text.find(_CLOSE, len(_OPEN) if explicit else 0)

    reasoning_start = len(_OPEN) if explicit else 0
    if close_at >= 0:
        return ReasoningText(text[reasoning_start:close_at], text[close_at + len(_CLOSE):])

    reasoning = text[reasoning_start:]
    if not final:
        reasoning = _without_partial_close(reasoning)
    return ReasoningText(reasoning, "")
