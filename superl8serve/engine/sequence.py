# SPDX-License-Identifier: MIT
"""Sequence + SamplingParams — a request's state through the engine."""

from __future__ import annotations

import enum
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

# (this sequence's token ids so far, this step's logits row) -> logits row. Applied
# before sampling -- the seam structured outputs / grammars (#39) plug into.
LogitsProcessor = Callable[[list[int], "torch.Tensor"], "torch.Tensor"]


@dataclass
class SamplingParams:
    temperature: float = 0.0  # 0 => greedy
    top_p: float = 1.0
    top_k: int = 0  # 0 => disabled
    repetition_penalty: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    logit_processors: list[LogitsProcessor] | None = None

    def __post_init__(self) -> None:
        if self.top_k < 0:
            raise ValueError("top_k must be 0 (disabled) or a positive integer")
        if not math.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be greater than 0")


class Status(enum.Enum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    FINISHED = enum.auto()


@dataclass
class Sequence:
    seq_id: int
    prompt_ids: list[int]
    params: SamplingParams
    status: Status = Status.WAITING
    slot: int = -1
    output_ids: list[int] = field(default_factory=list)
    length: int = 0  # KV positions committed for this seq
    prefix_matched_len: int = 0  # length of shared prefix found (0 = none)
    max_len: int | None = None  # engine context window; decode stops before it (S2)
    pixel_values: torch.Tensor | None = None  # VLM: [1, 3, H, W] preprocessed image
    image_grid_thw: torch.Tensor | None = None  # VLM: [num_images, 3] patch grid (t, gh, gw)
    # -- speculative-decode pipelining (spec loop, task 1c) -------------------
    # Carried between spec steps so the base forward is skipped after a sequence's
    # first spec step: `_spec_base_tok` = token@(length+1) (the first verify token),
    # `_spec_base_hidden` = h_main@length [H] (the full-model hidden that predicted
    # it). Both come from the PRIOR step's verify (true_tokens / hidden_v at the last
    # accepted slot) — the pipelining that removes the redundant per-step base
    # weight-stream. None until the first step establishes them (or after a preempt).
    spec_base_tok: int | None = None
    spec_base_hidden: "torch.Tensor | None" = None
    # N-gram-only speculation cannot know whether prompt lookup will hit until the
    # target model produces its base token. After a miss, ordinary graphed decode is
    # used for this many steps before one inexpensive re-probe.
    spec_ngram_cooldown: int = 0
    _repetition_seen: set[int] = field(
        default_factory=set, init=False, repr=False, compare=False
    )
    _repetition_seen_count: int = field(default=0, init=False, repr=False, compare=False)

    @property
    def num_prompt(self) -> int:
        return len(self.prompt_ids)

    @property
    def last_token(self) -> int:
        return self.output_ids[-1] if self.output_ids else self.prompt_ids[-1]

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_ids + self.output_ids

    @property
    def repetition_token_ids(self) -> list[int]:
        """Unique history, incrementally maintained for repetition penalty.

        Prompt tokens are scanned once; ordinary decode then adds one token per
        step. Speculative decode is disabled while repetition penalty is active,
        so its rollback/truncation path cannot invalidate this cache.
        """
        token_count = self.num_prompt + len(self.output_ids)
        if self._repetition_seen_count > token_count:
            self._repetition_seen.clear()
            self._repetition_seen_count = 0
        if self._repetition_seen_count < self.num_prompt:
            self._repetition_seen.update(
                self.prompt_ids[self._repetition_seen_count : self.num_prompt]
            )
            self._repetition_seen_count = self.num_prompt
        if self._repetition_seen_count < token_count:
            output_start = self._repetition_seen_count - self.num_prompt
            self._repetition_seen.update(self.output_ids[output_start:])
            self._repetition_seen_count = token_count
        return list(self._repetition_seen)

    def finish_reason(self, eos_id: int | None) -> str | None:
        if len(self.output_ids) >= self.params.max_tokens:
            return "length"
        # Hard stop at the context window even if max_tokens wasn't clamped: the KV
        # cache only has `max_len` positions for this slot (S2 backstop).
        if self.max_len is not None and self.num_prompt + len(self.output_ids) >= self.max_len:
            return "length"
        if (
            not self.params.ignore_eos
            and eos_id is not None
            and self.output_ids
            and self.output_ids[-1] == eos_id
        ):
            return "stop"
        return None

    def is_finished(self, eos_id: int | None) -> bool:
        return self.finish_reason(eos_id) is not None
