# SPDX-License-Identifier: MIT
"""DiffusionDecodeStrategy — non-autoregressive diffusion-step decode protocol.

The engine detects `cfg.decode_strategy == "diffusion"` and routes prefill through
this strategy's generate loop, which runs masked prefill → T denoise steps (iterative
unmasking via bidirectional attention) → final discretization via `torch.multinomial`.
"""
from __future__ import annotations

import torch

from ..models.base import ForwardContext


class DiffusionDecodeStrategy:
    def __init__(self, *, num_steps: int = 8, mask_id: int = 0):
        self.num_steps = num_steps
        self.mask_id = mask_id

    def generate(
        self,
        model,
        cache,
        device,
        seq,
        ctx: ForwardContext,
    ) -> list[int]:
        prompt_len = seq.num_prompt
        max_tokens = seq.params.max_tokens
        total_len = prompt_len + max_tokens

        full_ids = torch.tensor(
            [seq.prompt_ids + [self.mask_id] * max_tokens],
            device=device, dtype=torch.long,
        )
        positions = torch.arange(total_len, device=device).unsqueeze(0)

        cache.ensure_capacity([seq.slot], [total_len])

        ctx.is_prefill = True
        ctx.slots = [seq.slot]

        masked = set(range(prompt_len, total_len))
        logits = self._forward_for_logits(model, full_ids, positions, ctx, total_len)

        for step in range(self.num_steps):
            if not masked:
                break

            masked_list = list(masked)
            masked_logits = logits[masked_list]
            confidences = masked_logits.softmax(dim=-1).max(dim=-1).values

            n_remaining = len(masked)
            steps_left = self.num_steps - step
            n_unmask = max(1, n_remaining // steps_left)
            n_unmask = min(n_unmask, n_remaining)

            _, top_idx = confidences.topk(n_unmask)
            for local_idx in top_idx:
                global_pos = masked_list[local_idx]
                best = masked_logits[local_idx].argmax().item()
                full_ids[0, global_pos] = best
                masked.remove(global_pos)

            if not masked:
                break

            logits = self._forward_for_logits(model, full_ids, positions, ctx, total_len)

        if masked:
            masked_list = list(masked)
            masked_logits = logits[masked_list]
            probs = masked_logits.softmax(dim=-1)
            sampled = torch.multinomial(probs, 1, replacement=True).squeeze(-1)
            for i, global_pos in enumerate(masked_list):
                full_ids[0, global_pos] = sampled[i].item()

        return full_ids[0, prompt_len:].tolist()

    @staticmethod
    def _forward_for_logits(model, full_ids, positions, ctx: ForwardContext, total_len: int):
        hidden = model(full_ids, positions, ctx)
        return model.compute_logits(hidden).view(total_len, -1)
