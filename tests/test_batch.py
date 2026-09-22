# SPDX-License-Identifier: MIT
"""Offline batch inference tests (issue #41): `generate_batch` against a fake
engine double, no CUDA/superl8/model weights needed (the engine's own continuous-
batching correctness is covered in test_engine.py / test_paged_engine.py; this
is the batch-orchestration layer on top, which is plain Python)."""
import itertools

from superl8serve.batch import BatchRequest, generate_batch
from superl8serve.engine.sequence import SamplingParams, Sequence, Status


class FakeEngine:
    """No canned reply table: each step deterministically appends
    `(prompt[0] + len(output_ids)) % 50`, so a request keeps producing tokens for
    as long as its OWN `max_tokens` allows -- needed because generate_batch's whole
    point is that requests can have different `SamplingParams` (unlike
    `LLMEngine.generate()`, which takes one shared `SamplingParams` for every
    prompt in the call)."""

    def __init__(self, eos_id=99):
        self.eos_id = eos_id
        self._seqs: dict[int, Sequence] = {}
        self._ids = itertools.count()

    def add_request(self, prompt_ids, params: SamplingParams | None = None) -> int:
        seq_id = next(self._ids)
        self._seqs[seq_id] = Sequence(seq_id, list(prompt_ids), params or SamplingParams())
        return seq_id

    def step(self) -> None:
        for seq in self._seqs.values():
            if seq.status is Status.FINISHED:
                continue
            seq.output_ids.append((seq.prompt_ids[0] + len(seq.output_ids)) % 50)
            seq.status = Status.FINISHED if seq.is_finished(self.eos_id) else Status.RUNNING

    def sequence(self, seq_id: int) -> Sequence:
        return self._seqs[seq_id]

    def forget(self, seq_id: int) -> None:
        del self._seqs[seq_id]


def test_generate_batch_preserves_order_for_mixed_prompts():
    """The core ask (issue #41): a batch of mixed prompts (different lengths,
    different max_tokens) returns correct per-request outputs, in request order --
    not completion order, and not cross-contaminated by the shared continuous-
    batching loop."""
    engine = FakeEngine()
    requests = [
        BatchRequest([1, 2, 3], SamplingParams(temperature=0.0, max_tokens=3), custom_id="a"),
        BatchRequest([4], SamplingParams(temperature=0.0, max_tokens=1), custom_id="b"),
        BatchRequest([5, 6, 7, 8, 9], SamplingParams(temperature=0.0, max_tokens=5), custom_id="c"),
    ]

    results = generate_batch(engine, requests)

    assert [r.custom_id for r in results] == ["a", "b", "c"]
    assert results[0].output_ids == [1, 2, 3]
    assert results[1].output_ids == [4]
    assert results[2].output_ids == [5, 6, 7, 8, 9]
    assert all(r.finish_reason == "length" for r in results)
    assert engine._seqs == {}   # forgotten once the batch drains


def test_generate_batch_respects_per_request_sampling_params():
    """Same prompt length, different `max_tokens` -- proves each request's own
    `SamplingParams` (not a single shared one, unlike `LLMEngine.generate()`) governs
    when it stops."""
    engine = FakeEngine()
    requests = [
        BatchRequest([10, 10], SamplingParams(max_tokens=1), custom_id="short"),
        BatchRequest([10, 10], SamplingParams(max_tokens=4), custom_id="long"),
    ]

    results = generate_batch(engine, requests)
    by_id = {r.custom_id: r for r in results}
    assert len(by_id["short"].output_ids) == 1
    assert len(by_id["long"].output_ids) == 4


def test_generate_batch_stop_reason_on_eos():
    engine = FakeEngine(eos_id=1)
    results = generate_batch(
        engine, [BatchRequest([1], SamplingParams(max_tokens=10), custom_id="x")])
    assert results[0].output_ids == [1]
    assert results[0].finish_reason == "stop"


def test_generate_batch_empty_returns_empty():
    assert generate_batch(FakeEngine(), []) == []
