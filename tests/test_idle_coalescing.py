# SPDX-License-Identifier: MIT
"""Default-off, bounded idle-start request coalescing (issue #369)."""

import itertools
import threading
import time
from types import SimpleNamespace

import pytest

from superl8serve.api.runtime import Done, EngineWorker, _CancelRequest
from superl8serve.api.server import resolve_idle_coalescing
from superl8serve.engine.sequence import SamplingParams, Sequence, Status
from superl8serve.metrics import StatsCollector


class FakeEngine:
    def __init__(self, max_num_seqs=8):
        self.eos_id = None
        self._ids = itertools.count()
        self._seqs = {}
        self.first_step = threading.Event()
        self.step_batch_sizes = []
        self.cancelled = []
        self.scheduler = SimpleNamespace(max_num_seqs=max_num_seqs, running=[], waiting=[])

    def add_request(self, prompt_ids, params):
        seq_id = next(self._ids)
        self._seqs[seq_id] = Sequence(seq_id, list(prompt_ids), params)
        self.scheduler.waiting.append(seq_id)
        return seq_id

    def sequence(self, seq_id):
        return self._seqs[seq_id]

    def step(self):
        active = [seq for seq in self._seqs.values() if seq.status is not Status.FINISHED]
        self.step_batch_sizes.append(len(active))
        self.first_step.set()
        self.scheduler.running[:] = [seq.seq_id for seq in active]
        self.scheduler.waiting.clear()
        for seq in active:
            seq.output_ids.append(7)
            seq.status = (
                Status.FINISHED if len(seq.output_ids) >= seq.params.max_tokens else Status.RUNNING
            )

    def cancel(self, seq_id):
        self.cancelled.append(seq_id)
        self._seqs.pop(seq_id, None)
        self.scheduler.waiting[:] = [sid for sid in self.scheduler.waiting if sid != seq_id]
        self.scheduler.running[:] = [sid for sid in self.scheduler.running if sid != seq_id]

    def forget(self, seq_id):
        self._seqs.pop(seq_id, None)


def _drain(handle, timeout=2.0):
    deadline = time.monotonic() + timeout
    items = []
    while time.monotonic() < deadline:
        item = handle.out_queue.get(timeout=max(0.01, deadline - time.monotonic()))
        items.append(item)
        if isinstance(item, Done):
            return items
    raise AssertionError("request did not finish")


@pytest.fixture
def make_worker():
    workers = []

    def make(engine, **kwargs):
        worker = EngineWorker(engine, **kwargs)
        workers.append(worker)
        return worker

    yield make
    for worker in workers:
        worker.close()


def test_worker_close_joins_idle_engine_thread(make_worker):
    worker = make_worker(FakeEngine())

    worker.close()

    assert not worker._thread.is_alive()
    with pytest.raises(RuntimeError, match="closed"):
        worker.submit([1], SamplingParams(max_tokens=1))


def test_default_policy_starts_first_request_immediately(make_worker):
    engine = FakeEngine()
    worker = make_worker(engine)

    handle = worker.submit([1], SamplingParams(max_tokens=1))

    assert engine.first_step.wait(0.5)
    assert engine.step_batch_sizes[0] == 1
    assert isinstance(_drain(handle)[-1], Done)


def test_enabled_policy_collects_target_before_first_step(make_worker):
    engine = FakeEngine(max_num_seqs=8)
    stats = StatsCollector()
    worker = make_worker(engine, stats=stats, idle_coalesce_ms=500.0, idle_coalesce_target=4)

    handles = [worker.submit([i], SamplingParams(max_tokens=1)) for i in range(4)]

    assert engine.first_step.wait(0.5)
    assert engine.step_batch_sizes[0] == 4
    for handle in handles:
        assert isinstance(_drain(handle)[-1], Done)
    batching = stats.snapshot()["batching"]
    assert batching["events"] == 1
    assert batching["last_effective_batch"] == 4
    assert batching["target_hits"] == 1


def test_enabled_policy_releases_at_bounded_deadline(make_worker):
    engine = FakeEngine()
    worker = make_worker(engine, idle_coalesce_ms=50.0, idle_coalesce_target=4)
    started = time.monotonic()

    handle = worker.submit([1], SamplingParams(max_tokens=1))

    assert engine.first_step.wait(0.5)
    waited = time.monotonic() - started
    assert 0.025 <= waited < 0.30
    assert engine.step_batch_sizes[0] == 1
    assert isinstance(_drain(handle)[-1], Done)


def test_deadline_expiration_and_ttft_percentiles_are_reported(make_worker):
    engine = FakeEngine()
    stats = StatsCollector()
    worker = make_worker(engine, stats=stats, idle_coalesce_ms=25.0, idle_coalesce_target=4)

    handle = worker.submit([1], SamplingParams(max_tokens=1))
    assert isinstance(_drain(handle)[-1], Done)

    snap = stats.snapshot()
    assert snap["batching"]["deadline_expirations"] == 1
    assert snap["latency"]["p50_ttft_ms"] >= 20.0
    assert snap["latency"]["p95_ttft_ms"] >= snap["latency"]["p50_ttft_ms"]


def test_coalescing_never_sleeps_inside_active_decode_loop(make_worker):
    engine = FakeEngine()
    stats = StatsCollector()
    worker = make_worker(engine, stats=stats, idle_coalesce_ms=40.0, idle_coalesce_target=4)

    handle = worker.submit([1], SamplingParams(max_tokens=3))

    assert isinstance(_drain(handle)[-1], Done)
    assert len(engine.step_batch_sizes) == 3
    assert stats.snapshot()["batching"]["events"] == 1


def test_cancellation_during_coalesce_is_fifo_and_never_steps_cancelled_request(make_worker):
    engine = FakeEngine()
    worker = make_worker(engine, idle_coalesce_ms=200.0, idle_coalesce_target=2)
    cancelled = worker.submit([1], SamplingParams(max_tokens=2))
    worker._inbox.put(_CancelRequest(cancelled))
    survivor = worker.submit([2], SamplingParams(max_tokens=1))

    assert engine.first_step.wait(0.5)
    assert cancelled.seq_id in engine.cancelled
    assert engine.step_batch_sizes[0] == 1
    assert _drain(cancelled)[-1].reason == "cancelled"
    assert isinstance(_drain(survivor)[-1], Done)


def test_target_cannot_exceed_scheduler_capacity():
    with pytest.raises(ValueError, match="max_num_seqs"):
        EngineWorker(
            FakeEngine(max_num_seqs=4),
            idle_coalesce_ms=10.0,
            idle_coalesce_target=5,
        )


def test_target_must_be_an_integer():
    with pytest.raises(ValueError, match="integer"):
        EngineWorker(
            FakeEngine(max_num_seqs=8),
            idle_coalesce_ms=10.0,
            idle_coalesce_target=4.5,
        )


def test_cli_policy_resolves_default_target_before_model_load():
    assert resolve_idle_coalescing(
        max_num_seqs=512, idle_coalesce_ms=250.0, idle_coalesce_target=None
    ) == (250.0, 512)


@pytest.mark.parametrize("max_num_seqs", [0, -1, 1.5, True])
def test_cli_policy_rejects_invalid_capacity_before_model_load(max_num_seqs):
    with pytest.raises(ValueError, match="max-num-seqs must be a positive integer"):
        resolve_idle_coalescing(
            max_num_seqs=max_num_seqs,
            idle_coalesce_ms=0.0,
            idle_coalesce_target=None,
        )


@pytest.mark.parametrize(
    ("delay", "target"),
    [(float("nan"), None), (-1.0, None), (0.0, 4), (1.0, 0), (1.0, 513)],
)
def test_cli_policy_rejects_invalid_values_before_model_load(delay, target):
    with pytest.raises(ValueError):
        resolve_idle_coalescing(
            max_num_seqs=512,
            idle_coalesce_ms=delay,
            idle_coalesce_target=target,
        )
