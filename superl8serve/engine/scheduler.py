# SPDX-License-Identifier: MIT
"""Scheduler — continuous batching over waiting/running sequences (nano-vllm shape).

Each `schedule()` returns either a batch of WAITING sequences to prefill (a slot is
allocated per sequence) or the full RUNNING set to decode one step. Prefill is
preferred while slots, KV blocks, and the batch-token budget allow, so new requests
join quickly; otherwise the running set decodes. Finished sequences free their slot.

Admission is capped at `max_num_seqs` RUNNING sequences. Requests beyond that cap —
or beyond the KV-block pool's capacity — WAIT in FIFO and are admitted only as
running sequences FINISH and free their slot/blocks. A mere count cap is NOT memory
pressure and NEVER triggers preemption: preempting a running sequence just to admit a
waiter forces a full-context recompute of the evicted sequence, and under sustained
oversubscription that degenerates into per-step evict→recompute→evict thrash that
collapses aggregate throughput (the running set never gets to decode).

Preemption is retained only as a bounded LAST RESORT for genuine KV-block-pool
exhaustion (an over-subscribed pool with fewer blocks than the running set needs):
the lowest-priority running sequence (fewest generated tokens) is evicted to reclaim
its blocks, its full context is recomputed on re-admission, and the number of
evictions per `schedule()` call is capped so it can never livelock.
"""

from __future__ import annotations

from collections import deque

from ..models.cache import MLALatentCache
from .kv_cache import PagedKVCache
from .sequence import Sequence, Status


class Scheduler:
    def __init__(
        self,
        cache: PagedKVCache | MLALatentCache,
        *,
        max_num_seqs: int,
        max_batch_tokens: int,
        eos_id: int | None,
    ):
        self.cache = cache
        self.max_num_seqs = max_num_seqs
        self.max_batch_tokens = max_batch_tokens
        self.eos_id = eos_id
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def cancel(self, seq_id: int) -> bool:
        """Remove a request by ``seq_id`` and release its resources.

        A WAITING sequence never holds a KV slot, so it is dropped with no cache
        work. A RUNNING sequence's slot (and any spec-decode carry naming positions
        in it) is released exactly as if it had finished; the caller must clear the
        runner's per-slot decode state (``lin_cache`` / ``mtp_cache``). Returns
        False when no such sequence exists (idempotent)."""
        for seq in self.waiting:
            if seq.seq_id == seq_id:
                self.waiting.remove(seq)
                seq.status = Status.FINISHED
                return True
        for seq in self.running:
            if seq.seq_id == seq_id:
                self.running.remove(seq)
                seq.status = Status.FINISHED
                self.cache.free(seq.slot)
                seq.slot = -1
                # Drop the spec-decode pipelining carry: it names positions in the
                # freed KV cache (mirror of `_preempt_one`).
                seq.spec_base_tok = None
                seq.spec_base_hidden = None
                seq.spec_ngram_cooldown = 0
                return True
        return False

    def _preempt_one(self):
        """Evict the running sequence with the fewest generated tokens and put it
        back in the waiting queue where its full context will be recomputed."""
        preempted = min(self.running, key=lambda s: len(s.output_ids))
        self.running.remove(preempted)
        self.cache.free(preempted.slot)
        if preempted.is_finished(self.eos_id):
            preempted.status = Status.FINISHED
            return
        # Extend prompt_ids so the model recomputes through the full context
        # (original prompt + previously generated tokens) on re-admission.
        # output_ids is NOT reset: it accumulates tokens across all admissions
        # so the caller sees the complete generation.
        preempted.prompt_ids = preempted.all_token_ids
        preempted.length = 0
        preempted.slot = -1
        preempted.prefix_matched_len = 0
        # Drop the spec-decode pipelining carry: it names positions in the freed KV
        # cache, which are recomputed from scratch on re-admission.
        preempted.spec_base_tok = None
        preempted.spec_base_hidden = None
        preempted.spec_ngram_cooldown = 0
        preempted.status = Status.WAITING
        self.waiting.append(preempted)

    def _has_free_block(self) -> bool:
        """Whether the KV-block pool can back another sequence. Caches that
        pre-allocate a fixed per-slot region (no shared block pool) always report
        True — for them slot availability alone bounds admission."""
        probe = getattr(self.cache, "has_free_block", None)
        return probe() if probe is not None else True

    def schedule(self) -> tuple[list[Sequence], bool]:
        """Returns (batch, is_prefill)."""
        # Prefill newly-waiting sequences while we have slots + KV blocks + token budget.
        batch, tokens = [], 0
        preempted = 0
        while True:
            while (
                self.waiting
                and len(self.running) + len(batch) < self.max_num_seqs
                and self.cache.has_free_slot()
                and self._has_free_block()
            ):
                seq = self.waiting[0]
                if batch and tokens + seq.num_prompt > self.max_batch_tokens:
                    break
                self.waiting.popleft()
                seq.slot = self.cache.alloc()
                # Prefix cache: look up shared prefix before prefill.
                matched, shared_blocks = self.cache.lookup_prefix(seq.prompt_ids)
                if matched > 0 and shared_blocks:
                    self.cache.share_blocks(seq.slot, shared_blocks)
                    seq.prefix_matched_len = matched
                seq.status = Status.RUNNING
                tokens += seq.num_prompt
                batch.append(seq)
            if batch:
                return batch, True
            # Nothing admitted. A count/slot cap is NOT memory pressure — excess
            # requests WAIT in FIFO and are admitted only as running sequences
            # finish; preempting here would thrash. Preempt ONLY under genuine
            # KV-block-pool exhaustion: a slot is free but the shared block pool is
            # dry (an over-subscribed pool), so a running sequence's blocks must be
            # reclaimed before any waiter can prefill. Bounded per call so it can
            # never livelock.
            block_pressure = (
                self.waiting
                and self.running
                and self.cache.has_free_slot()
                and not self._has_free_block()
            )
            if block_pressure and preempted < self.max_num_seqs:
                self._preempt_one()
                preempted += 1
                continue
            return list(self.running), False

    def postprocess(self, batch: list[Sequence], is_prefill: bool):
        if is_prefill:
            self.running.extend(batch)
        still = []
        for seq in self.running:
            if seq.is_finished(self.eos_id):
                seq.status = Status.FINISHED
                self.cache.free(seq.slot)
            else:
                still.append(seq)
        self.running = still
