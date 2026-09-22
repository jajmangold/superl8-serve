# SPDX-License-Identifier: MIT
"""Scheduler admission / preemption unit tests (GPU-free).

Drives the real `Scheduler` with a lightweight in-memory `FakeCache` (no CUDA, no
model) so the admission + preemption *policy* can be exercised deterministically and
cheaply. Mirrors what `LLMEngine.step()` does to a batch, minus the actual forward
pass: prefill emits the first token per sequence, decode emits one token per running
sequence, and a sequence finishes once it has `max_tokens` outputs.

Regression target (issue: preemption-thrash collapse): when concurrency exceeds
`max_num_seqs`, the scheduler used to preempt a running sequence back to WAITING --
where its FULL context is recomputed -- on every step, purely because the *count*
cap was hit. With N running + M waiting that turns each step into evict + O(context)
re-prefill instead of forward decode progress, collapsing aggregate throughput
(measured: concurrency 32 -> 9.1 tok/s vs 703 tok/s at 16). The fix: excess requests
must WAIT in FIFO and be admitted only as running sequences FINISH; preemption is a
bounded last resort for genuine KV-block-pool exhaustion, never for a mere count cap.
"""

from __future__ import annotations

from superl8serve.engine.scheduler import Scheduler
from superl8serve.engine.sequence import SamplingParams, Sequence, Status


class FakeCache:
    """Minimal slot+block pool implementing the Scheduler/engine cache contract,
    with no GPU tensors. `num_blocks` defaults to the worst case (every slot at
    `max_len`) exactly like `PagedKVCache`, so `has_free_block()` tracks genuine
    block-pool pressure independently of the slot/count cap."""

    def __init__(self, num_slots, *, block_size=16, max_len=256, num_blocks=None):
        self.block_size = block_size
        self.max_blocks_per_seq = (max_len + block_size - 1) // block_size
        self.num_blocks = (
            num_blocks if num_blocks is not None else num_slots * self.max_blocks_per_seq
        )
        self._free_slots = list(range(num_slots))
        self._free_blocks = list(range(self.num_blocks))
        self._slot_blocks = {s: [] for s in range(num_slots)}

    def alloc(self):
        return self._free_slots.pop()

    def free(self, slot):
        self._free_blocks.extend(self._slot_blocks[slot])
        self._slot_blocks[slot] = []
        self._free_slots.append(slot)

    def has_free_slot(self):
        return bool(self._free_slots)

    def has_free_block(self):
        return bool(self._free_blocks)

    def ensure_capacity(self, slots, lengths):
        for slot, n in zip(slots, lengths):
            need = (n + self.block_size - 1) // self.block_size
            while len(self._slot_blocks[slot]) < need:
                self._slot_blocks[slot].append(self._free_blocks.pop())

    # Prefix cache is not under test here.
    def lookup_prefix(self, token_ids):
        return 0, []

    def share_blocks(self, slot, blocks):
        pass

    def store_prefix(self, token_ids, slot):
        pass


def _run(scheduler, seqs, max_tokens):
    """Drive the scheduler to completion, mirroring `LLMEngine.step()` without a
    model. Returns telemetry for assertions."""
    for s in seqs:
        scheduler.add(s)

    preemptions = 0
    orig_preempt = scheduler._preempt_one

    def counting_preempt():
        nonlocal preemptions
        preemptions += 1
        orig_preempt()

    scheduler._preempt_one = counting_preempt

    prefill_token_work = 0  # prompt tokens fed through "prefill" (grows on recompute)
    prefill_calls: dict[int, int] = {}
    steps = 0
    guard = 0
    while scheduler.has_work():
        guard += 1
        assert guard < 100_000, "scheduler is not making progress (livelock)"
        batch, is_prefill = scheduler.schedule()
        if not batch:
            break
        steps += 1
        if is_prefill:
            for seq in batch:
                scheduler.cache.ensure_capacity([seq.slot], [seq.num_prompt])
                seq.length = seq.num_prompt
                prefill_token_work += seq.num_prompt
                prefill_calls[seq.seq_id] = prefill_calls.get(seq.seq_id, 0) + 1
                seq.output_ids.append(7)  # emit first token
        else:
            for seq in batch:
                seq.length += 1
                scheduler.cache.ensure_capacity([seq.slot], [seq.length + 1])
                seq.output_ids.append(7)
        scheduler.postprocess(batch, is_prefill)
        # Belt-and-suspenders cap so a finished seq is FINISHED even if postprocess
        # ran before the final token pushed it over max_tokens.
        for seq in list(scheduler.running):
            if len(seq.output_ids) >= max_tokens:
                seq.status = Status.FINISHED

    return dict(
        preemptions=preemptions,
        prefill_token_work=prefill_token_work,
        prefill_calls=prefill_calls,
        steps=steps,
    )


def _mk(seqs_spec, max_tokens):
    return [
        Sequence(i, list(prompt), SamplingParams(temperature=0.0, max_tokens=max_tokens))
        for i, prompt in enumerate(seqs_spec)
    ]


def test_no_preemption_thrash_under_count_cap():
    """2x oversubscription (8 requests, max_num_seqs=4) with a worst-case block pool
    must NEVER preempt: the count cap alone is not memory pressure. Each prompt is
    prefilled EXACTLY once (no full-context recompute), and every request completes
    with the right number of tokens."""
    max_tokens = 6
    cache = FakeCache(num_slots=4)
    sched = Scheduler(cache, max_num_seqs=4, max_batch_tokens=8192, eos_id=None)
    prompts = [[1, 2, 3], [4, 5, 6, 7], [8], [9, 10], [11, 12, 13, 14, 15], [16], [17, 18], [19]]
    seqs = _mk(prompts, max_tokens)
    telem = _run(sched, seqs, max_tokens)

    assert telem["preemptions"] == 0, f"count-cap preemption thrash: {telem}"
    # Every sequence prefilled exactly once -> no O(context) recompute.
    assert all(c == 1 for c in telem["prefill_calls"].values()), telem["prefill_calls"]
    assert telem["prefill_token_work"] == sum(len(p) for p in prompts)
    for s in seqs:
        assert len(s.output_ids) == max_tokens
        assert s.status is Status.FINISHED


def test_all_requests_complete_fifo_bounded_work():
    """Heavy oversubscription (24 requests, max_num_seqs=4) still terminates with
    bounded, ~linear work -- no pathological blow-up -- and preserves FIFO order of
    completion (sequences admitted first finish first)."""
    max_tokens = 5
    cache = FakeCache(num_slots=4)
    sched = Scheduler(cache, max_num_seqs=4, max_batch_tokens=8192, eos_id=None)
    prompts = [[i + 1, i + 2, i + 3] for i in range(24)]
    seqs = _mk(prompts, max_tokens)
    telem = _run(sched, seqs, max_tokens)

    assert telem["preemptions"] == 0, telem
    assert all(len(s.output_ids) == max_tokens for s in seqs)
    # Work is linear in requests: exactly one prefill batch-worth per request plus
    # (max_tokens-1) decode tokens. Number of scheduler steps is bounded well below
    # the thrash regime (which would be thousands for 24x5).
    assert telem["steps"] <= 6 * (len(prompts) // 4 + max_tokens), telem


def _admit(sched, seq):
    """Push one waiting seq all the way to RUNNING via a prefill schedule() step."""
    sched.add(seq)
    batch, is_prefill = sched.schedule()
    assert is_prefill and seq in batch
    sched.cache.ensure_capacity([seq.slot], [seq.num_prompt])
    seq.length = seq.num_prompt
    seq.output_ids.append(7)
    sched.postprocess(batch, is_prefill)


def test_count_cap_never_preempts():
    """Policy gate, positive control: with the block pool healthy (free blocks) but
    the RUNNING set already at `max_num_seqs`, an arriving request must NOT trigger
    preemption -- `schedule()` returns the running set to decode and the waiter
    stays queued."""
    cache = FakeCache(num_slots=3, max_len=256)  # worst-case pool: always has blocks
    sched = Scheduler(cache, max_num_seqs=3, max_batch_tokens=8192, eos_id=None)
    for i, p in enumerate([[1, 2], [3, 4], [5, 6]]):
        _admit(sched, Sequence(i, p, SamplingParams(max_tokens=50)))
    assert len(sched.running) == 3 and cache.has_free_block()

    sched.add(Sequence(99, [7, 8], SamplingParams(max_tokens=50)))  # excess arrival
    batch, is_prefill = sched.schedule()
    assert is_prefill is False and len(batch) == 3  # decode the running set
    assert len(sched.running) == 3  # nobody evicted
    assert len(sched.waiting) == 1  # excess request simply waits


def test_preemption_only_on_genuine_block_exhaustion():
    """Policy gate, negative control: when a slot is free but the shared block pool
    is DRY (genuine KV-memory pressure) and requests wait, `schedule()` is allowed to
    preempt the least-progressed running sequence to reclaim blocks -- bounded, and
    only in this case."""
    cache = FakeCache(num_slots=4, max_len=256)
    sched = Scheduler(cache, max_num_seqs=4, max_batch_tokens=8192, eos_id=None)
    # Admit two sequences (uses 2 slots), give one more generated token to seq 1 so
    # the least-progressed (seq 0) is the eviction target.
    _admit(sched, Sequence(0, [1, 2, 3], SamplingParams(max_tokens=50)))
    _admit(sched, Sequence(1, [4, 5, 6], SamplingParams(max_tokens=50)))
    sched.running[1].output_ids.append(7)  # seq 1 more progressed than seq 0

    # Force genuine block-pool exhaustion (slots still free) and enqueue a waiter.
    cache._free_blocks.clear()
    assert cache.has_free_slot() and not cache.has_free_block()
    sched.add(Sequence(2, [8, 9], SamplingParams(max_tokens=50)))

    preemptions = 0
    orig = sched._preempt_one

    def counting():
        nonlocal preemptions
        preemptions += 1
        orig()

    sched._preempt_one = counting

    batch, is_prefill = sched.schedule()
    # Genuine block pressure -> the least-progressed running seq (0) IS evicted to
    # reclaim blocks (distinct from the count cap, which never preempts), and its
    # freed blocks let a waiter prefill -> forward progress, bounded.
    assert preemptions >= 1
    assert preemptions <= sched.max_num_seqs  # bounded, no livelock
    assert is_prefill and batch  # progress: a prefill batch, not a stall
    assert 0 not in {s.seq_id for s in sched.running}  # the eviction target left running


# ── Cancellation by seq id (superl8-serve#363) ─────────────────────────────────


def test_cancel_waiting_drops_without_kv_free():
    """A WAITING sequence holds no slot, so canceling it must not touch the KV pool:
    the slot count and block pool stay exactly as they were."""
    cache = FakeCache(num_slots=2, max_len=256)
    sched = Scheduler(cache, max_num_seqs=2, max_batch_tokens=8192, eos_id=None)
    seq = Sequence(7, [1, 2, 3], SamplingParams(max_tokens=50))
    sched.add(seq)

    free_slots_before = set(cache._free_slots)
    free_blocks_before = len(cache._free_blocks)
    assert sched.cancel(7) is True
    assert sched.has_work() is False
    assert seq.status is Status.FINISHED
    assert seq.slot == -1  # never admitted -> never held a slot
    assert set(cache._free_slots) == free_slots_before  # no slot churn
    assert len(cache._free_blocks) == free_blocks_before  # no KV freed


def test_cancel_running_frees_kv_slot():
    """Canceling a RUNNING sequence releases its paged KV exactly as normal
    completion does: the slot and its blocks return to the free pools."""
    cache = FakeCache(num_slots=2, max_len=256)
    sched = Scheduler(cache, max_num_seqs=2, max_batch_tokens=8192, eos_id=None)
    _admit(sched, Sequence(7, [1, 2, 3], SamplingParams(max_tokens=50)))
    seq = sched.running[0]
    slot = seq.slot
    assert slot >= 0
    free_slots_before = set(cache._free_slots)

    assert sched.cancel(7) is True
    assert sched.has_work() is False
    assert seq.status is Status.FINISHED
    assert seq.slot == -1
    assert slot in cache._free_slots  # the old slot is restored
    assert set(cache._free_slots) == free_slots_before | {slot}


def test_cancel_absent_is_idempotent_false():
    """Canceling an unknown -- or already-canceled -- seq id returns False and is a
    no-op, so repeated cancels are safe."""
    cache = FakeCache(num_slots=2, max_len=256)
    sched = Scheduler(cache, max_num_seqs=2, max_batch_tokens=8192, eos_id=None)
    _admit(sched, Sequence(7, [1, 2, 3], SamplingParams(max_tokens=50)))
    assert sched.cancel(999) is False
    assert sched.cancel(7) is True
    assert sched.cancel(7) is False  # idempotent: second cancel misses


def test_cancel_leaves_other_sequences_unchanged():
    """Canceling one sequence must not disturb the rest: the others still finish
    with the exact number of output tokens (normal completion unchanged)."""
    max_tokens = 4
    cache = FakeCache(num_slots=3, max_len=256)
    sched = Scheduler(cache, max_num_seqs=3, max_batch_tokens=8192, eos_id=None)
    seqs = _mk([[1, 2, 3], [4, 5, 6], [7, 8]], max_tokens)
    for s in seqs:
        sched.add(s)

    # Run one step so seq 0 is RUNNING, then cancel it mid-flight.
    batch, is_prefill = sched.schedule()
    for seq in batch:
        sched.cache.ensure_capacity([seq.slot], [seq.num_prompt])
        seq.length = seq.num_prompt
        seq.output_ids.append(7)
    sched.postprocess(batch, is_prefill)
    assert sched.cancel(seqs[0].seq_id) is True

    telem = _run(sched, [], max_tokens)  # survivors are already enqueued
    assert telem["preemptions"] == 0
    for s in seqs[1:]:
        assert len(s.output_ids) == max_tokens
        assert s.status is Status.FINISHED
