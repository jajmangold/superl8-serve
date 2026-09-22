# SPDX-License-Identifier: MIT
"""Offline batch inference (issue #41): `generate_batch` runs a list of independent
requests to completion over the SAME continuous-batching loop `LLMEngine.generate()`
drives, except each request may carry its OWN `SamplingParams` -- `generate()` only
takes one shared `SamplingParams` for the whole call. Good for eval/dataset runs
(mixed prompts, mixed sampling settings, one call) without holding a connection open
per request. `results` is always in `requests` order, regardless of which sequence
the scheduler finishes first.
"""
from __future__ import annotations

from dataclasses import dataclass

from .engine.sequence import SamplingParams, Status


@dataclass
class BatchRequest:
    prompt_ids: list[int]
    params: SamplingParams | None = None
    custom_id: str | None = None


@dataclass
class BatchResult:
    output_ids: list[int]
    finish_reason: str | None
    custom_id: str | None = None


def generate_batch(engine, requests: list[BatchRequest]) -> list[BatchResult]:
    """`engine` needs `.add_request`, `.step`, `.sequence`, `.forget`, `.eos_id` (an
    `LLMEngine`, or a test double with the same seam -- see `EngineWorker`)."""
    seq_ids = [engine.add_request(r.prompt_ids, r.params) for r in requests]
    while any(engine.sequence(sid).status is not Status.FINISHED for sid in seq_ids):
        engine.step()

    results = []
    for seq_id, req in zip(seq_ids, requests):
        seq = engine.sequence(seq_id)
        results.append(BatchResult(
            output_ids=list(seq.output_ids),
            finish_reason=seq.finish_reason(engine.eos_id) or "stop",
            custom_id=req.custom_id,
        ))
        engine.forget(seq_id)
    return results
