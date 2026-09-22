# SPDX-License-Identifier: MIT
"""Token sampler: greedy or temperature + top-k/top-p sampling.

Operates on the last-position logits of each sequence in a batch. Temperatures are
per-sequence so a batch can mix greedy and sampled sequences in one call. Also the
per-step logit-processor hook: a per-sequence list of (input_ids, logits) -> logits
callables applied before sampling -- what structured outputs / grammars plug into.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

LogitsProcessor = Callable[[list[int], torch.Tensor], torch.Tensor]


class Sampler(nn.Module):
    @torch.inference_mode()
    def forward(
        self,
        logits: torch.Tensor,  # [num_seqs, vocab] fp16/fp32
        temperatures: torch.Tensor,  # [num_seqs] fp32 (0 -> greedy)
        top_p: torch.Tensor | None = None,  # [num_seqs] in (0,1], or None
        top_k: torch.Tensor | None = None,  # [num_seqs], 0 disables per row
        repetition_penalty: torch.Tensor | None = None,  # [num_seqs], 1 disables
        logit_processors: list[list[LogitsProcessor]] | None = None,  # per-seq processors
        input_ids: list[list[int]] | None = None,  # per-seq token history, for the hook
        *,
        all_greedy: bool | None = None,  # pure-python hint: every row is temp==0
        any_top_p: bool | None = None,  # pure-python hint: some row has top_p < 1.0
        any_top_k: bool | None = None,  # pure-python hint: some row has top_k > 0
        max_top_k: int | None = None,  # pure-python hint: largest enabled top_k
        any_repetition_penalty: bool | None = None,  # some row has penalty != 1
    ) -> torch.Tensor:
        logits = logits.float()
        if logit_processors is not None:
            logits = self._apply_logit_processors(logits, input_ids, logit_processors)
        if any_repetition_penalty is None:
            any_repetition_penalty = repetition_penalty is not None and bool(
                (repetition_penalty != 1.0).any()
            )
        if repetition_penalty is not None and any_repetition_penalty:
            if input_ids is None:
                raise ValueError("input_ids are required for repetition_penalty")
            logits = self._apply_repetition_penalty(
                logits, input_ids, repetition_penalty
            )
        greedy = logits.argmax(dim=-1)

        # Fast path (issue #183): an all-greedy batch (every temp==0) is just the
        # argmax -- skip the temp-scale + softmax + top-p sort + Gumbel-max noise,
        # which otherwise run over the full 151,936-token vocab EVERY decode step
        # (~0.9 ms). Bit-identical to the general path below: `torch.where` would
        # select `greedy` for every row anyway. The caller (EngineRunner._sample)
        # passes `all_greedy` computed from python params so this decision costs no
        # device->host sync; falling back to a tensor reduction only when unhinted.
        if all_greedy is None:
            all_greedy = bool((temperatures == 0).all())
        if all_greedy:
            return greedy

        temp = temperatures.clamp_min(1e-5)
        scaled = logits / temp.unsqueeze(-1)

        if any_top_k is None:
            any_top_k = top_k is not None and bool((top_k > 0).any())
        if top_k is not None and any_top_k:
            scaled = self._apply_top_k(scaled, top_k, max_top_k=max_top_k)

        # top_p == 1.0 keeps the whole distribution, so the nucleus filter (and its
        # full-vocab torch.sort) is a no-op -- skip it unless some row is < 1.0.
        if any_top_p is None:
            any_top_p = top_p is not None and bool((top_p < 1.0).any())
        if top_p is not None and any_top_p:
            scaled = self._apply_top_p(scaled, top_p)

        probs = torch.softmax(scaled, dim=-1)
        # Gumbel-max trick: argmax(log p + gumbel noise) ~ categorical(p), vectorized.
        noise = torch.empty_like(probs).exponential_(1.0)
        sampled = (probs / noise).argmax(dim=-1)

        return torch.where(temperatures == 0, greedy, sampled)

    @staticmethod
    def _apply_logit_processors(
        logits: torch.Tensor,
        input_ids: list[list[int]],
        logit_processors: list[list[LogitsProcessor]],
    ) -> torch.Tensor:
        rows = list(logits.unbind(0))
        for i, procs in enumerate(logit_processors):
            for proc in procs or ():
                rows[i] = proc(input_ids[i], rows[i])
        return torch.stack(rows)

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        input_ids: list[list[int]],
        repetition_penalty: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Hugging Face-compatible, once-per-token repetition penalty."""
        if not any(input_ids):
            return logits
        unique_ids = [list(dict.fromkeys(ids)) for ids in input_ids]
        lengths = torch.tensor(
            [len(ids) for ids in unique_ids], dtype=torch.long, device=logits.device
        )
        rows = torch.repeat_interleave(
            torch.arange(len(input_ids), device=logits.device), lengths
        )
        tokens = torch.tensor(
            [token for ids in unique_ids for token in ids],
            dtype=torch.long,
            device=logits.device,
        )
        scores = logits[rows, tokens]
        penalties = repetition_penalty[rows]
        adjusted = torch.where(scores < 0, scores * penalties, scores / penalties)
        result = logits.clone()
        result[rows, tokens] = adjusted
        return result

    @staticmethod
    def _apply_top_k(
        scaled: torch.Tensor,
        top_k: torch.Tensor,
        *,
        max_top_k: int | None = None,
    ) -> torch.Tensor:
        vocab_size = scaled.shape[-1]
        if max_top_k is None:
            max_top_k = int(top_k.max().item())
        max_top_k = min(max(max_top_k, 1), vocab_size)
        row_k = top_k.to(dtype=torch.long).clamp(min=1, max=vocab_size)
        top_values = torch.topk(scaled, max_top_k, dim=-1).values
        cutoffs = top_values.gather(1, (row_k - 1).unsqueeze(1)).squeeze(1)
        cutoffs = torch.where(
            top_k > 0,
            cutoffs,
            torch.full_like(cutoffs, float("-inf")),
        )
        return scaled.masked_fill(scaled < cutoffs.unsqueeze(1), float("-inf"))

    @staticmethod
    def _apply_top_p(scaled: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(scaled, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumsum = sorted_probs.cumsum(dim=-1)
        # keep the smallest prefix whose cumulative prob >= top_p (always keep top-1)
        mask = cumsum - sorted_probs > top_p.unsqueeze(-1)
        sorted_logits = scaled.gather(-1, sorted_idx).masked_fill(mask, float("-inf"))
        return sorted_logits.scatter(-1, sorted_idx, sorted_logits)
