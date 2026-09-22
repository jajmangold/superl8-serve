# SPDX-License-Identifier: MIT
"""Pre-auth DoS + encode-race hardening (C1/S1/S2/S3).

All against fake engine/tokenizer doubles (no CUDA/superl8/weights), mirroring
test_api.py / test_batches.py. Each test pins one confirmed vulnerability:

- C1: `/v1/embeddings` + `/v1/rerank` used to call `engine.encode()` straight from
  the FastAPI handler thread, concurrently with the EngineWorker thread's
  `engine.step()` (the engine is single-thread-only). encode must now run ON the
  worker thread.
- S1: `max_tokens` was never clamped to `max_len - prompt_len` -> block-table
  overflow / giant KV alloc. It must be truncated.
- S2: an over-length prompt was admitted unconditionally -> prefill OOM / KV OOB.
  It must be rejected cleanly (400 / per-line error).
- S3: `/v1/files`, `/v1/batches`, and data-URI images had no size caps.
"""

import base64
import itertools
import json
import queue
import threading

import pytest

pytest.importorskip("fastapi")

from starlette.testclient import TestClient  # noqa: E402

from superl8serve.api.app import create_app  # noqa: E402
from superl8serve.api.runtime import EngineWorker, _CancelRequest, Done  # noqa: E402
from superl8serve.engine.sequence import SamplingParams, Sequence, Status  # noqa: E402

EOS = 4


class CharTokenizer:
    """One token id per character, so a test can drive an exact prompt length."""

    eos_token_id = EOS

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=True, chat_template=None, tools=None
    ):
        return [ord(c) for c in messages[-1]["content"]]

    def encode(self, text, **kw):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        kept = [i for i in ids if not (skip_special_tokens and i == EOS)]
        return "".join(chr(i) for i in kept if 0 <= i < 0x110000)


class FakeEngine:
    """Deterministic 1-token-per-step engine that exposes `max_len` (so the API layer
    can read it for its clamp/guard) and records the thread each entrypoint runs on.

    It does NOT itself reject over-length prompts -- that lets the handler-level guard
    (S2) be tested in isolation: on unpatched code the request is admitted and decodes,
    on patched code the handler returns 400 before ever reaching the engine."""

    def __init__(self, max_len=None):
        self.eos_id = EOS
        self.max_len = max_len
        self._seqs: dict[int, Sequence] = {}
        self._ids = itertools.count()
        self.step_threads: list[int] = []
        self.encode_threads: list[int] = []

    def add_request(self, prompt_ids, params: SamplingParams | None = None) -> int:
        seq_id = next(self._ids)
        seq = Sequence(seq_id, list(prompt_ids), params or SamplingParams())
        seq.max_len = self.max_len
        self._seqs[seq_id] = seq
        return seq_id

    def step(self) -> None:
        self.step_threads.append(threading.get_ident())
        for seq in self._seqs.values():
            if seq.status is Status.FINISHED:
                continue
            seq.output_ids.append((seq.last_token + 1) % 128)
            seq.status = Status.FINISHED if seq.is_finished(self.eos_id) else Status.RUNNING

    def encode(self, prompt_ids: list[int]) -> list[float]:
        self.encode_threads.append(threading.get_ident())
        return [0.5] * 8

    def sequence(self, seq_id: int) -> Sequence:
        return self._seqs[seq_id]

    def forget(self, seq_id: int) -> None:
        self._seqs.pop(seq_id, None)


# --- C1: encode runs on the worker thread, not the caller thread ------------


def test_encode_runs_on_worker_thread_not_caller():
    engine = FakeEngine()
    worker = EngineWorker(engine)
    caller_thread = threading.get_ident()

    out = worker.encode([1, 2, 3])

    assert out == [0.5] * 8
    assert engine.encode_threads, "encode never ran"
    (encode_thread,) = set(engine.encode_threads)
    assert encode_thread != caller_thread, "encode ran on the caller thread (race)"


def test_encode_and_step_share_one_thread():
    """encode and step must serialize on the SAME (worker) thread -- the whole point
    of routing encode through the worker instead of the handler thread."""
    engine = FakeEngine()
    worker = EngineWorker(engine)
    # Drive a generation so step() runs, plus an encode, and confirm both landed on
    # the single worker thread.
    q = worker.submit([1, 2], SamplingParams(max_tokens=2))
    worker.encode([7, 8])
    # drain the generation queue
    while True:
        item = q.get()
        if item.__class__.__name__ == "Done":
            break
    assert engine.step_threads and engine.encode_threads
    assert set(engine.step_threads) | set(engine.encode_threads) == {engine.step_threads[0]}


def test_worker_telemetry_carries_native_sampling_controls():
    class CapturingStats:
        def __init__(self):
            self.sampling = None

        def record_request_start(self, _seq_id, *, prompt_tokens, sampling):
            assert prompt_tokens == 2
            self.sampling = sampling

        def record_first_token(self, _seq_id):
            pass

        def record_request_finish(self, _seq_id, **_kwargs):
            pass

    stats = CapturingStats()
    worker = EngineWorker(FakeEngine(), stats=stats)
    queue = worker.submit(
        [1, 2],
        SamplingParams(
            temperature=0.1,
            top_k=50,
            repetition_penalty=1.1,
            max_tokens=1,
        ),
    )
    while queue.get().__class__.__name__ != "Done":
        pass

    assert stats.sampling["top_k"] == 50
    assert stats.sampling["repetition_penalty"] == pytest.approx(1.1)


# --- app fixtures with a max_len-bearing engine -----------------------------


def _client(max_len=None):
    app = create_app(FakeEngine(max_len=max_len), CharTokenizer(), served_model_name="fake")
    return TestClient(app)


# --- S1: max_tokens clamped to remaining room -------------------------------


def test_completions_max_tokens_clamped_to_max_len():
    with _client(max_len=8) as c:
        # prompt is 5 tokens, max_len 8 -> at most 3 output tokens even though we ask 1000
        resp = c.post(
            "/v1/completions",
            json={
                "model": "fake",
                "prompt": "abcde",
                "max_tokens": 1000,
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["usage"]["completion_tokens"] <= 3


# --- S2: over-length prompt rejected cleanly --------------------------------


def test_completions_over_length_prompt_rejected():
    with _client(max_len=8) as c:
        resp = c.post(
            "/v1/completions",
            json={
                "model": "fake",
                "prompt": "a" * 20,
                "max_tokens": 4,
            },
        )
        assert resp.status_code == 400, resp.text


def test_embeddings_over_length_prompt_rejected():
    with _client(max_len=8) as c:
        resp = c.post("/v1/embeddings", json={"model": "fake", "input": "a" * 20})
        assert resp.status_code == 400, resp.text


def test_llm_engine_add_request_rejects_over_length_prompt():
    """The real engine-level backstop for offline callers that never touch the API.
    Exercises `LLMEngine.add_request` directly (bypassing model construction) since the
    guard fires before any Sequence/scheduler/cache work."""
    from superl8serve.engine.llm_engine import LLMEngine

    eng = object.__new__(LLMEngine)
    eng.max_len = 4
    with pytest.raises(ValueError, match="max_len"):
        eng.add_request([1, 2, 3, 4, 5])


# --- S3: request-size caps --------------------------------------------------


def _jsonl(lines):
    return ("\n".join(json.dumps(x) for x in lines) + "\n").encode("utf-8")


def test_oversized_file_upload_rejected():
    from superl8serve.api.batches import _MAX_UPLOAD_BYTES

    with _client() as c:
        big = b"x" * (_MAX_UPLOAD_BYTES + 1024)
        resp = c.post(
            "/v1/files",
            files={"file": ("big.jsonl", big, "application/jsonl")},
            data={"purpose": "batch"},
        )
        assert resp.status_code == 413, resp.text


def test_oversized_batch_line_count_rejected():
    from superl8serve.api.batches import _MAX_BATCH_LINES

    with _client() as c:
        lines = [
            {
                "custom_id": str(i),
                "url": "/v1/completions",
                "body": {"model": "fake", "prompt": "a", "max_tokens": 1},
            }
            for i in range(_MAX_BATCH_LINES + 1)
        ]
        fid = c.post(
            "/v1/files",
            files={"file": ("b.jsonl", _jsonl(lines), "application/jsonl")},
            data={"purpose": "batch"},
        ).json()["id"]
        resp = c.post("/v1/batches", json={"input_file_id": fid, "endpoint": "/v1/completions"})
        assert resp.status_code == 413, resp.text


def test_data_uri_image_size_capped():
    pytest.importorskip("PIL")
    from superl8serve.multimodal import _MAX_IMAGE_BYTES, fetch_image

    # A base64 payload whose decoded length exceeds the cap must be rejected before PIL.
    huge = base64.b64encode(b"\x00" * (_MAX_IMAGE_BYTES + 1024)).decode()
    with pytest.raises(ValueError, match="cap"):
        fetch_image("data:image/png;base64," + huge)


# --- #363: disconnect cancellation (API/thread half) -------------------------


class _FakeScheduler:
    """Minimal running/waiting depth tracker so `EngineWorker._cancel` can read
    `len(engine.scheduler.running/waiting)` the way it does against LLMEngine."""

    def __init__(self):
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []

    def add(self, seq):
        self.waiting.append(seq)

    def promote(self):
        while self.waiting:
            self.running.append(self.waiting.pop(0))

    def cancel(self, seq_id: int) -> bool:
        for container in (self.waiting, self.running):
            for i, seq in enumerate(container):
                if seq.seq_id == seq_id:
                    del container[i]
                    return True
        return False


class CancellingEngine(FakeEngine):
    """FakeEngine plus a `cancel` that behaves like LLMEngine.cancel: removes the
    sequence, marks it FINISHED, forgets it, and returns whether it was found."""

    def __init__(self, max_len=None):
        super().__init__(max_len=max_len)
        self.cancelled: list[int] = []
        self.scheduler = _FakeScheduler()

    def add_request(self, prompt_ids, params=None):
        seq_id = super().add_request(prompt_ids, params)
        self.scheduler.add(self._seqs[seq_id])
        return seq_id

    def step(self) -> None:
        self.scheduler.promote()
        super().step()

    def cancel(self, seq_id: int) -> bool:
        seq = self._seqs.pop(seq_id, None)
        if seq is None:
            return False
        seq.status = Status.FINISHED
        self.cancelled.append(seq_id)
        self.scheduler.cancel(seq_id)
        return True


class CapturingStats:
    def __init__(self):
        self.starts = []
        self.finishes = []
        self.depths = []

    def record_request_start(self, seq_id, *, prompt_tokens, sampling):
        self.starts.append(seq_id)

    def record_first_token(self, _seq_id):
        pass

    def record_queue_depth(self, *, running, waiting):
        self.depths.append((running, waiting))

    def record_request_finish(self, seq_id, *, output_tokens, finish_reason):
        self.finishes.append((seq_id, output_tokens, finish_reason))


def _bare_worker(engine, stats=None):
    """An EngineWorker WITHOUT its background thread, so the worker-side inbox
    ordering (`_PendingRequest` registration before `_CancelRequest`) and the
    completion race can be driven deterministically via direct `_intake` calls."""
    worker = object.__new__(EngineWorker)
    worker.engine = engine
    worker.stats = stats
    worker._inbox = queue.Queue()
    worker._pending = {}
    worker._closed = False
    return worker


def _wait_until(pred, timeout=2.0):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met within timeout")


def test_submit_returns_pending_handle_with_backwards_compatible_get():
    engine = FakeEngine()
    worker = EngineWorker(engine)
    handle = worker.submit([1, 2], SamplingParams(max_tokens=2))
    assert handle.__class__.__name__ == "_PendingRequest"
    # `get()` delegates to out_queue exactly like the old bare-queue return value.
    items = []
    while True:
        item = handle.get()
        items.append(item)
        if isinstance(item, Done):
            break
    assert isinstance(items[-1], Done)
    assert all(not isinstance(i, Done) for i in items[:-1])


def test_cancel_after_registration_releases_get_with_done_cancelled():
    engine = CancellingEngine()
    stats = CapturingStats()
    worker = _bare_worker(engine, stats=stats)
    handle = worker.submit([1, 2], SamplingParams(max_tokens=8))
    worker._intake(worker._inbox.get_nowait())  # register
    assert handle.seq_id is not None
    worker._inbox.put(_CancelRequest(handle))
    worker._intake(worker._inbox.get_nowait())  # cancel
    items = []
    while True:
        item = handle.get()
        items.append(item)
        if isinstance(item, Done):
            break
    assert items[-1] == Done("cancelled")
    assert len([i for i in items if isinstance(i, Done)]) == 1
    assert engine.cancelled == [handle.seq_id]
    assert handle.terminal
    assert stats.finishes == [(handle.seq_id, 0, "cancelled")]
    assert stats.depths == [(0, 0)], "gauges must reflect the drained scheduler"
    assert worker._pending == {}


def test_cancel_before_register_and_step_is_fifo():
    """A cancel enqueued right after submit lands behind the registration in the
    same FIFO inbox: registration is processed strictly before the cancel, so the
    request is never stepped and `get()` still releases with a single Done."""
    engine = CancellingEngine()
    worker = _bare_worker(engine)
    handle = worker.submit([1, 2], SamplingParams(max_tokens=8))
    worker._inbox.put(_CancelRequest(handle))  # queued after registration
    worker._intake(worker._inbox.get_nowait())  # FIFO: register first
    assert handle.seq_id is not None
    worker._intake(worker._inbox.get_nowait())  # then cancel
    assert engine.cancelled == [handle.seq_id]
    assert handle.terminal
    assert worker._pending == {}
    assert not engine.step_threads, "cancelled request must not be stepped"
    assert handle.get() == Done("cancelled")
    assert handle.out_queue.empty(), "exactly one Done, never two"


def test_cancel_race_with_completion_is_idempotent():
    """A _CancelRequest already in the inbox when the request completes: the
    normal Done path marks terminal BEFORE the queue put, so the worker's _cancel
    becomes a no-op -- the completion reason/stats win and exactly one Done and one
    finish record are emitted; engine.cancel never fires."""
    engine = CancellingEngine()
    stats = CapturingStats()
    worker = _bare_worker(engine, stats=stats)
    handle = worker.submit([1], SamplingParams(max_tokens=1))
    worker._inbox.put(_CancelRequest(handle))  # cancel queued, not yet processed
    worker._intake(worker._inbox.get_nowait())  # register
    worker.engine.step()  # the final decode step
    worker._dispatch()  # normal completion path
    assert handle.terminal
    worker._intake(worker._inbox.get_nowait())  # cancel arrives post-completion
    items = []
    while True:
        item = handle.get()
        items.append(item)
        if isinstance(item, Done):
            break
    assert items[-1].reason != "cancelled", f"completion lost the race: {items}"
    assert len([i for i in items if isinstance(i, Done)]) == 1
    assert len(stats.finishes) == 1
    assert stats.finishes[0][2] != "cancelled"
    assert engine.cancelled == [], "engine.cancel must not fire for a finished req"


def test_cancel_of_dispatched_handle_is_noop():
    """A cancel for a request the worker already dispatched (removed from `_pending`
    and terminal) must be a no-op: no second Done, no engine.cancel call."""
    engine = CancellingEngine()
    worker = _bare_worker(engine)
    handle = worker.submit([1], SamplingParams(max_tokens=1))
    worker._intake(worker._inbox.get_nowait())
    worker.engine.step()
    worker._dispatch()  # normal completion: terminal + Done + removed
    assert handle.terminal and worker._pending == {}
    worker._intake(_CancelRequest(handle))  # must be a no-op
    assert engine.cancelled == []
    items = []
    while True:
        item = handle.get()
        items.append(item)
        if isinstance(item, Done):
            break
    assert items[-1] == Done("length")  # the ONE original Done
    assert len([i for i in items if isinstance(i, Done)]) == 1
    assert handle.out_queue.empty(), "cancel must not put a second Done"


def test_stream_cancels_on_aclose_abandonment():
    """An async stream abandoned (disconnected) before Done is observed must
    enqueue exactly one _CancelRequest, which the worker thread alone resolves:
    engine.cancel fires, the request leaves `_pending`, and the finish is recorded
    as 'cancelled' exactly once."""

    async def main():
        engine = CancellingEngine()
        stats = CapturingStats()
        worker = EngineWorker(engine, stats=stats)
        # FakeEngine increments toward EOS (4) within a couple steps and would
        # finish before the aclose; ignore_eos + huge max_tokens keeps the request
        # RUNNING so the abandonment actually exercises the cancellation path.
        gen = worker.stream([1, 2], SamplingParams(max_tokens=2**20, ignore_eos=True))
        agen = gen.__aiter__()
        await agen.__anext__()  # consume at least one token
        await agen.aclose()  # consumer disconnects before Done
        _wait_until(lambda: not worker._pending)
        assert engine.cancelled, "abandoned stream must cancel the request"
        (seq_id,) = engine.cancelled
        assert len(stats.finishes) == 1
        assert stats.finishes[0][0] == seq_id
        assert stats.finishes[0][2] == "cancelled"

    import asyncio

    asyncio.run(main())


def test_stream_finally_cancel_only_when_terminal_not_observed():
    """stream's finally must put a _CancelRequest iff terminal was not observed --
    after a clean Done it must not (the worker already marked terminal)."""

    async def main():
        engine = CancellingEngine()
        worker = EngineWorker(engine)
        async for _ in worker.stream([1], SamplingParams(max_tokens=1)):
            pass
        await asyncio.sleep(0.05)
        assert engine.cancelled == [], "clean completion must not cancel"

    import asyncio

    asyncio.run(main())
