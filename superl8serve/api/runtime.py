# SPDX-License-Identifier: MIT
"""EngineWorker — the API layer's single owner of an `LLMEngine`.

`LLMEngine.step()` and its `Scheduler` are plain Python state (a deque + a list),
not safe to call concurrently from multiple threads. One dedicated thread drains a
request queue and drives the scheduler, so concurrent HTTP requests still share the
engine's continuous-batching loop -- exactly what `LLMEngine.generate()` does for a
list of prompts, just fed incrementally over time (as HTTP requests arrive) instead
of all at once.
"""

from __future__ import annotations

import asyncio
import math
import queue
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import torch

from ..engine.sequence import SamplingParams, Status


_STOP = object()


@dataclass
class Done:
    """Sentinel yielded (once, last) by `EngineWorker.stream` when a request finishes."""

    reason: str


@dataclass
class _EncodeRequest:
    """A pooled-embedding request routed through the worker so `engine.encode()`
    runs on the SAME thread as `engine.step()` -- the engine (scheduler + KV cache)
    is not safe to touch from the FastAPI handler thread concurrently with the
    worker's step loop. `out` carries back the embedding (or the raised exception)."""

    prompt_ids: list[int]
    out: queue.Queue = field(default_factory=queue.Queue)


@dataclass
class _CancelRequest:
    """Routed through the SAME FIFO inbox as `submit`, so a `_CancelRequest` is
    always processed after its request's registration -- the worker thread alone
    resolves it against `_pending` (a `_PendingRequest` may already be terminal or
    already removed, making the cancel a no-op)."""

    handle: _PendingRequest


@dataclass
class _PendingRequest:
    prompt_ids: list[int]
    params: SamplingParams
    pixel_values: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None
    out_queue: queue.Queue = field(default_factory=queue.Queue)
    seq_id: int | None = None
    sent: int = 0
    # Set to True by the worker thread BEFORE the terminal `Done` is put on
    # `out_queue` (finished or cancelled). `stream` consults it in `finally` to
    # decide whether a `_CancelRequest` is still owed, and `_cancel` uses it to
    # make the completion race idempotent.
    terminal: bool = False

    def get(self):
        """Backwards-compatible queue handle: block for the next streamed token
        (or the terminal `Done`) exactly as the old bare `out_queue` did."""
        return self.out_queue.get()


def _resolve_idle_policy(
    engine, idle_coalesce_ms: float, idle_coalesce_target: int | None
) -> tuple[float, int | None]:
    if not math.isfinite(idle_coalesce_ms) or idle_coalesce_ms < 0:
        raise ValueError("idle_coalesce_ms must be finite and >= 0")
    capacity = getattr(getattr(engine, "scheduler", None), "max_num_seqs", None)
    if idle_coalesce_ms > 0 and idle_coalesce_target is None:
        idle_coalesce_target = capacity or 1
    if idle_coalesce_target is not None and (
        isinstance(idle_coalesce_target, bool) or not isinstance(idle_coalesce_target, int)
    ):
        raise ValueError("idle_coalesce_target must be an integer")
    if idle_coalesce_target is not None and idle_coalesce_target < 1:
        raise ValueError("idle_coalesce_target must be >= 1")
    if capacity is not None and idle_coalesce_target is not None:
        if idle_coalesce_target > capacity:
            raise ValueError(
                f"idle_coalesce_target cannot exceed scheduler max_num_seqs ({capacity})"
            )
    return float(idle_coalesce_ms), idle_coalesce_target


class EngineWorker:
    def __init__(
        self,
        engine,
        *,
        stats=None,
        idle_coalesce_ms: float = 0.0,
        idle_coalesce_target: int | None = None,
    ) -> None:
        self.engine = engine
        # Optional superl8serve.metrics.StatsCollector: the worker owns the per-request
        # lifecycle (arrival -> first token -> finish), so it records TTFT / ITL /
        # e2e-latency and the request counters here. All host-side timestamps.
        self.stats = stats
        self.idle_coalesce_ms, self.idle_coalesce_target = _resolve_idle_policy(
            engine, idle_coalesce_ms, idle_coalesce_target
        )
        self._inbox: queue.Queue[_PendingRequest | _EncodeRequest | _CancelRequest | object] = (
            queue.Queue()
        )
        self._pending: dict[int, _PendingRequest] = {}
        if self.stats is not None and hasattr(self.stats, "configure_batching"):
            self.stats.configure_batching(
                idle_coalesce_ms=self.idle_coalesce_ms,
                idle_coalesce_target=self.idle_coalesce_target,
            )
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="superl8serve-engine")
        self._thread.start()

    def close(self, timeout: float = 2.0) -> None:
        """Stop an idle worker and join its owner thread.

        Server processes normally own the worker for their full lifetime. Tests and
        embedders can call this after all submitted handles reached ``Done`` so the
        blocking inbox owner does not leak across interpreter teardown.
        """
        if not self._closed:
            self._closed = True
            self._inbox.put(_STOP)
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("engine worker did not stop while idle")

    def submit(
        self,
        prompt_ids: list[int],
        params: SamplingParams,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
    ) -> _PendingRequest:
        """Enqueue a generation request; returns a `_PendingRequest` handle whose
        `get()` streams token ids, terminated by a single `Done`."""
        if self._closed:
            raise RuntimeError("engine worker is closed")
        req = _PendingRequest(
            prompt_ids, params, pixel_values=pixel_values, image_grid_thw=image_grid_thw
        )
        self._inbox.put(req)
        return req

    def encode(self, prompt_ids: list[int]) -> list[float]:
        """Encode a prompt and return the pooled embedding vector.

        Routed through the worker's inbox (like `submit`) so `engine.encode()` runs
        on the dedicated engine thread, serialized with `engine.step()` -- calling it
        straight from the FastAPI handler thread raced the step loop over the shared
        scheduler / KV cache (data race C1). Blocks the caller until the worker
        thread produces the embedding; the caller should run it off the event loop
        (the API handlers use `run_in_executor`)."""
        if self._closed:
            raise RuntimeError("engine worker is closed")
        req = _EncodeRequest(prompt_ids)
        self._inbox.put(req)
        result = req.out.get()
        if isinstance(result, BaseException):
            raise result
        return result

    async def stream(
        self,
        prompt_ids: list[int],
        params: SamplingParams,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
    ) -> AsyncIterator[int | Done]:
        handle = self.submit(
            prompt_ids, params, pixel_values=pixel_values, image_grid_thw=image_grid_thw
        )
        loop = asyncio.get_running_loop()
        try:
            while True:
                item = await loop.run_in_executor(None, handle.get)
                yield item
                if isinstance(item, Done):
                    handle.terminal = True
                    return
        finally:
            # The consumer abandoned the stream (e.g. the HTTP client disconnected)
            # before a terminal `Done` was observed. Owe the worker a cancellation:
            # routed through the same FIFO inbox, so it lands after registration and
            # after any already-queued `_CancelRequest`. If the worker already put a
            # terminal `Done` (`handle.terminal`), it's a no-op.
            if not handle.terminal:
                self._inbox.put(_CancelRequest(handle))

    def _run(self) -> None:
        while True:
            was_idle = not self._pending
            if was_idle:
                item = self._inbox.get()  # idle: block for the next request
                if item is _STOP:
                    return
                self._intake(item)
            while True:
                try:
                    item = self._inbox.get_nowait()
                except queue.Empty:
                    break
                if item is _STOP:
                    # ``close`` is for a drained worker. If it races the final
                    # dispatch by a few instructions, preserve the sentinel for
                    # the next idle iteration instead of abandoning live work.
                    self._inbox.put(_STOP)
                    break
                self._intake(item)
            # Optional bulk-lane policy: wait only at the idle -> active transition.
            # Once decode is running this branch is never entered, so interactive
            # traffic and the active hot loop retain their old behavior.
            if was_idle and self._pending and self.idle_coalesce_ms > 0:
                self._coalesce_idle_start()
            # Only step when a generation is in flight. An idle worker that only just
            # served an encode has nothing to decode, so skip the empty step and loop
            # back to block on the inbox.
            if self._pending:
                self.engine.step()
                self._dispatch()

    def _coalesce_idle_start(self) -> None:
        """Collect a bounded FIFO burst before the first step after an idle period.

        Encode and cancellation items are processed on the same worker thread while
        waiting. The wait ends at the generation target or absolute deadline; it is
        never called while an existing generation is already decoding.
        """
        target = self.idle_coalesce_target or 1
        started = time.perf_counter()
        deadline = started + self.idle_coalesce_ms / 1000.0
        while len(self._pending) < target:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                item = self._inbox.get(timeout=remaining)
            except queue.Empty:
                break
            if item is _STOP:
                self._inbox.put(_STOP)
                break
            self._intake(item)

        capacity = getattr(getattr(self.engine, "scheduler", None), "max_num_seqs", None)
        effective_batch = len(self._pending)
        if capacity is not None:
            effective_batch = min(effective_batch, capacity)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        target_hit = effective_batch >= target
        if self.stats is not None and hasattr(self.stats, "record_idle_coalesce"):
            self.stats.record_idle_coalesce(
                wait_ms=elapsed_ms,
                effective_batch=effective_batch,
                target_hit=target_hit,
            )

    def _intake(self, item: _PendingRequest | _EncodeRequest | _CancelRequest) -> None:
        """Dispatch one inbox item on the worker thread: an encode is answered
        inline (serialized with `step`), a generation request is registered, and a
        cancellation is applied (serialized with `step`/`_dispatch`)."""
        if isinstance(item, _EncodeRequest):
            try:
                item.out.put(self.engine.encode(item.prompt_ids))
            except BaseException as exc:  # noqa: BLE001 -- relayed to the caller thread
                item.out.put(exc)
        elif isinstance(item, _CancelRequest):
            self._cancel(item.handle)
        else:
            self._register(item)

    def _register(self, req: _PendingRequest) -> None:
        req.seq_id = self.engine.add_request(req.prompt_ids, req.params)
        seq = self.engine.sequence(req.seq_id)
        if req.pixel_values is not None:
            seq.pixel_values = req.pixel_values
            seq.image_grid_thw = req.image_grid_thw
        self._pending[req.seq_id] = req
        if self.stats is not None:
            p = req.params
            self.stats.record_request_start(
                req.seq_id,
                prompt_tokens=len(req.prompt_ids),
                sampling={
                    "temperature": p.temperature,
                    "top_p": p.top_p,
                    "top_k": p.top_k,
                    "repetition_penalty": p.repetition_penalty,
                    "max_tokens": p.max_tokens,
                },
            )

    def _cancel(self, handle: _PendingRequest) -> None:
        """Worker-thread cancellation of one `_PendingRequest` handle.

        Runs only on the worker thread, so it is serialized with `_register`,
        `engine.step()` and `_dispatch` -- no locking, no races. Idempotent: a
        handle that is already terminal (the request finished and `_dispatch` put
        its `Done` first), never registered, or already removed from `_pending`
        is a no-op. Otherwise it calls `engine.cancel(seq_id)`, marks the handle
        terminal, drops it from `_pending`, records the finish telemetry exactly
        once, and puts a single `Done("cancelled")` to release `queue.get`."""
        if handle.terminal or handle.seq_id is None:
            return
        if self._pending.get(handle.seq_id) is not handle:
            return
        seq = self.engine.sequence(handle.seq_id)
        output_tokens = len(seq.output_ids)
        self.engine.cancel(handle.seq_id)
        handle.terminal = True
        del self._pending[handle.seq_id]
        if self.stats is not None:
            # No step follows a cancel, so refresh the running/waiting gauges from
            # the scheduler before the terminal finish record (issue #363) -- the
            # depth otherwise keeps its pre-cancel values forever.
            scheduler = getattr(self.engine, "scheduler", None)
            if scheduler is not None:
                self.stats.record_queue_depth(
                    running=len(scheduler.running),
                    waiting=len(scheduler.waiting),
                )
            self.stats.record_request_finish(
                handle.seq_id, output_tokens=output_tokens, finish_reason="cancelled"
            )
        handle.out_queue.put(Done("cancelled"))

    def _dispatch(self) -> None:
        done_ids = []
        for seq_id, req in self._pending.items():
            seq = self.engine.sequence(seq_id)
            if self.stats is not None and req.sent == 0 and seq.output_ids:
                self.stats.record_first_token(seq_id)
            while req.sent < len(seq.output_ids):
                req.out_queue.put(seq.output_ids[req.sent])
                req.sent += 1
            if seq.status is Status.FINISHED:
                reason = seq.finish_reason(self.engine.eos_id) or "stop"
                # Record the finish BEFORE signalling Done so the metrics counters
                # are updated by the time the HTTP handler (woken by Done on the
                # queue) can observe them -- otherwise a fast client could GET
                # /metrics before this worker thread bumps `finished_requests`.
                if self.stats is not None:
                    self.stats.record_request_finish(
                        seq_id, output_tokens=len(seq.output_ids), finish_reason=reason
                    )
                # Mark terminal BEFORE the queue put: a `_CancelRequest` already in
                # the inbox sees `terminal` set and becomes a no-op (completion race).
                req.terminal = True
                req.out_queue.put(Done(reason))
                done_ids.append(seq_id)
        for seq_id in done_ids:
            del self._pending[seq_id]
            self.engine.forget(seq_id)
