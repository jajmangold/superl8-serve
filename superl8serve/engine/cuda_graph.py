# SPDX-License-Identifier: MIT
"""GraphedDecode -- CUDA-graph capture/replay for `EngineRunner.decode`.

Decode is dispatch-bound, not compute-bound: a 28-layer 0.6B model launches
~3,200 CUDA kernels/step (~114/layer) for ~18ms of real GPU work inside a
131ms step (measured on Qwen3-0.6B int8, batch=8, V100 idx-4 -- see issue #42).
Capturing the whole decode step (model forward + logits) into a CUDA graph and
replaying it re-issues every one of those launches as ONE `cudaGraphLaunch`,
collapsing dispatch overhead toward the real kernel time.

Requirements for capture, and how each is met here:
  * **Static shapes.** Decode batches are padded up to a batch-size bucket
    (1, 2, 4, 8, 16, ...) and a max-context-length bucket; pad rows point at a
    dedicated scratch cache slot (allocated once, never freed) and are simply
    sliced off the output before sampling.
  * **Persistent input buffers.** `ids`/`pos`/`slot_mapping`/`block_table`/
    `context_lens` are allocated once per (batch_bucket, context_bucket) and
    refreshed with `copy_` before every `replay()` -- captured kernels always
    read/write the SAME memory, so replay only picks up new values if we
    mutate that memory in place; a captured graph has no way to consume a
    freshly-built python-list-derived tensor from a later call.
  * **No mid-step sync.** `PagedKVCache.decode_attn_static` takes
    `max_context_len` as a plain python int (the bucket value) instead of
    `context_lens.max().item()` -- see kv_cache.py.
  * **Sampling stays outside the graph.** The graph captures model forward +
    `compute_logits` only; `EngineRunner._sample` (argmax/top-p + the final
    `.tolist()` host copy) runs eagerly on the sliced, unpadded logits after
    `replay()`.
  * **Paged-decode attention consumes tensors, not python lists** -- see the
    `ctx.slot_mapping is not None` branch in `GQAAttention._decode_batched`.

Only plain full-attention (dense, non-MoE, no sliding-window) decode is
capturable: MoE's per-expert token routing (`mask.nonzero()`, a data-dependent
op) and the sliding-window fallback (`PagedKVCache.read_dense` + a python loop)
cannot be represented in a CUDA graph. `_check_supported` detects this up
front from the model config; unsupported models fall back to eager decode
for every step (logged once).
"""

from __future__ import annotations

import logging
import os

import torch

from ..models.base import ForwardContext
from .sequence import Sequence

log = logging.getLogger(__name__)

DEFAULT_BATCH_BUCKETS = (1,)
DEFAULT_CONTEXT_BUCKET_SIZE = 128  # context-length bucket granularity (rounds up)
DEFAULT_MAX_GRAPHS = 32  # cap the captured-graph set (each pins its own workspace)
_WARMUP_ITERS = 3


def cuda_graph_enabled_by_env(default: bool = True) -> bool:
    """`SUPERL8SERVE_CUDA_GRAPH=0` disables graphed decode (eager stays available for
    debugging); unset or any other value keeps the default."""
    val = os.environ.get("SUPERL8SERVE_CUDA_GRAPH")
    if val is None:
        return default
    return val not in ("0", "false", "False")


def _next_bucket(buckets: tuple[int, ...], n: int) -> int | None:
    for b in buckets:
        if n <= b:
            return b
    return None


def _round_up(n: int, multiple: int) -> int:
    return ((n + multiple - 1) // multiple) * multiple


class _CapturedGraph:
    __slots__ = (
        "graph",
        "ids",
        "pos",
        "slot_mapping",
        "block_table",
        "context_lens",
        "logits",
        "batch_bucket",
        "context_bucket",
        "ids_host",
        "pos_host",
        "context_lens_host",
        "slot_idx",
        "slot_idx_host",
    )

    def __init__(
        self,
        *,
        graph,
        ids,
        pos,
        slot_mapping,
        block_table,
        context_lens,
        logits,
        batch_bucket,
        context_bucket,
        ids_host,
        pos_host,
        context_lens_host,
        slot_idx=None,
        slot_idx_host=None,
    ):
        self.graph = graph
        self.ids = ids
        self.pos = pos
        self.slot_mapping = slot_mapping
        self.block_table = block_table
        self.context_lens = context_lens
        self.logits = logits  # static output buffer -- read it before the next replay()
        self.batch_bucket = batch_bucket
        self.context_bucket = context_bucket
        # Pinned host staging (issue #183): the per-step ids/pos/context_lens refresh
        # fills these in place then non_blocking-copies into the captured device
        # buffers above, instead of rebuilding `torch.tensor(list, device=cuda)` (a
        # blocking pageable H2D) every step. The device buffers keep the SAME pointer
        # the graph captured -- we only ever mutate their contents in place.
        self.ids_host = ids_host
        self.pos_host = pos_host
        self.context_lens_host = context_lens_host
        # Recurrent decode only (None otherwise): persistent device row->slot index
        # the captured recurrent-state gather/scatter reads, plus its pinned host
        # staging. Refreshed in place each step before replay -- same pointer the
        # graph captured (models/cache.py `bind_graph`).
        self.slot_idx = slot_idx
        self.out_idx = out_idx
        self.slot_idx_host = slot_idx_host


class _CapturedVerify:
    __slots__ = (
        "graph", "ids", "pos", "verify_slot_mapping", "block_table", "context_lens",
        "hidden", "true_tokens", "batch_bucket", "context_bucket",
        "ids_host", "pos_host", "context_lens_host", "vsm_host",
        "slot_idx", "slot_idx_host", "vtraj",
    )

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


class GraphedVerify:
    """CUDA-graph capture/replay for the spec-decode VERIFY forward — the sibling of
    :class:`GraphedDecode` that makes MTP + fused-DeltaNet spec-decode compose with
    graphs (issue #259). Base decode is graphed (~1 launch/step) but the spec verify
    ran 100% eager, so on this dispatch-bound fleet every spec mode was a NET SLOWDOWN
    graphs-on (#266: cascade 0.84×, mtp 0.72×) despite high acceptance + bit-identical
    output. This captures the multi-token verify forward at a FIXED draft length so the
    per-step launch overhead collapses to one ``cudaGraphLaunch``.

    What is captured (the dominant, shape-static GPU work): the verify forward over
    ``S = spec_k + 1`` tokens (every row padded to the full ``spec_k`` drafts, so S is a
    compile-time constant), including all qkv/o/mlp GEMMs, the paged verify attention
    (``GQAAttention._verify_batched``'s static branch), the per-token DeltaNet /
    short-conv recurrence + its per-token state trajectory, ``compute_logits`` over all
    S positions, and the greedy ``argmax`` per position. Reuses the decode graph's
    contract exactly: persistent device input buffers refreshed via ``copy_`` before
    replay, a fixed ``max_context_len`` bucket int (no ``.item()`` sync), and the
    fixed-address per-slot recurrent-state buffers (``bind_graph``).

    What stays eager (truly variable / host-side, off the captured critical path):
    drafting (n-gram / MTP), the accept-longest-greedy-prefix loop (host argmax
    compare), EOS/budget truncation, the MTP prefix-KV extension, and the final
    recurrent-state COMMIT by accept length — the last reads the captured trajectory
    (fixed-address graph-pool tensors) and scatters, per row, the state after that
    row's last accepted token, exactly as the eager path does.

    Bit-identity: the captured verify uses the SAME kernels and the SAME per-position
    context lengths as the eager verify, so accepted tokens + committed K/V + committed
    recurrent state are byte-identical (guarded by tests/test_graph_verify.py)."""

    def __init__(self, graphed_decode: "GraphedDecode"):
        self.gd = graphed_decode
        self.model = graphed_decode.model
        self.cache = graphed_decode.cache
        self.lin_cache = graphed_decode.lin_cache
        self.device = graphed_decode.device
        self.has_recurrent = graphed_decode.has_recurrent
        self.batch_buckets = graphed_decode.batch_buckets
        self.context_bucket_size = graphed_decode.context_bucket_size
        self.max_graphs = graphed_decode.max_graphs
        self.supported = graphed_decode.supported
        self._graphs: dict[tuple[int, int, int], _CapturedVerify] = {}

    # -- public entry point ------------------------------------------------
    def try_run(self, batch, verify_ids, verify_pos, lengths):
        """Replay (capturing on first hit) the verify forward for ``batch``.

        ``verify_ids``/``verify_pos``: python ``[B][S]`` lists (base token + spec_k
        drafts, padded to a fixed S) and their absolute positions. ``lengths``:
        committed length per row (``seq.length``). Returns ``(hidden_v[:B] [B,S,H],
        true_tokens[:B] [B,S])`` with the recurrent verify trajectory bound onto
        ``lin_cache`` for the runner's subsequent ``commit_verify`` — or ``None`` if this
        step can't be served from a graph (unsupported model, batch over the widest
        bucket, or the graph cap hit), so the caller falls back to the eager verify."""
        if not self.supported or not batch:
            return None
        B = len(batch)
        batch_bucket = _next_bucket(self.batch_buckets, B)
        if batch_bucket is None:
            return None
        S = len(verify_ids[0])  # base token + fixed spec_k drafts (compile-time per key)
        max_real_ctx = max(lengths) + 1 + S
        cap = self.cache.max_blocks_per_seq * self.cache.block_size
        context_bucket = min(_round_up(max_real_ctx, self.context_bucket_size), cap)
        if max_real_ctx > cap:
            return None  # verify would overrun the paged capacity — let eager handle it
        key = (batch_bucket, context_bucket, S)
        g = self._graphs.get(key)
        if g is None:
            if len(self._graphs) >= self.max_graphs:
                return None
            g = self._capture(batch_bucket, context_bucket, S)
            self._graphs[key] = g
            log.info(
                "cuda-graph verify: captured bucket batch=%d max_context=%d S=%d (%d graphs)",
                batch_bucket, context_bucket, S, len(self._graphs),
            )
        self._fill_inputs(g, batch, verify_ids, verify_pos, lengths)
        g.graph.replay()
        # Bind the captured recurrent trajectory + rebind eager (python-slot) so the
        # runner's commit_verify(row_last) scatters the accepted-length state onto the
        # real slots. bind() clears the graph row-index the replay used.
        if self.has_recurrent and g.vtraj is not None:
            self.lin_cache._vtraj = g.vtraj
            self.lin_cache._vcap = True
            self.lin_cache.bind([s.slot for s in batch])
        return g.hidden[:B], g.true_tokens[:B]

    # -- capture ------------------------------------------------------------
    def _capture(self, batch_bucket: int, context_bucket: int, S: int) -> _CapturedVerify:
        cache = self.cache
        scratch = self.gd._scratch()
        cache.ensure_capacity([scratch], [S + 1])
        dev = self.device

        ids = torch.zeros(batch_bucket, S, dtype=torch.long, device=dev)
        pos = torch.zeros(batch_bucket, S, dtype=torch.long, device=dev)
        flat_slots = [scratch] * (batch_bucket * S)
        flat_pos = [t for _ in range(batch_bucket) for t in range(S)]
        verify_slot_mapping = cache.slot_mapping_for(flat_slots, flat_pos)  # [Bmax*S]
        block_table = cache.block_table([scratch] * batch_bucket)
        context_lens = torch.ones(batch_bucket, dtype=torch.int32, device=dev)

        slot_idx = slot_idx_host = None
        if self.has_recurrent and self.lin_cache is not None:
            slot_idx = torch.full((batch_bucket,), scratch, dtype=torch.long, device=dev)
            slot_idx_host = torch.empty(batch_bucket, dtype=torch.long,
                                        pin_memory=(dev != "cpu"))
            self.lin_cache.bind_graph(slot_idx)

        def _ctx():
            return ForwardContext(
                is_prefill=False, is_verify=True, kv_cache=cache,
                lin_cache=self.lin_cache if self.has_recurrent else None,
                slots=[scratch] * batch_bucket,
                slot_lengths=[1] * batch_bucket,
                verify_slot_mapping=verify_slot_mapping,
                block_tables=block_table, context_lens=context_lens,
                max_context_len=context_bucket,
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(_WARMUP_ITERS):
                with torch.inference_mode():
                    if self.has_recurrent:
                        self.lin_cache.begin_verify_capture()
                    hidden = self.model(ids, pos, _ctx())
                    self.model.compute_logits(hidden).argmax(-1)
                    if self.has_recurrent:
                        self.lin_cache.end_verify_capture()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        if self.has_recurrent:
            self.lin_cache.begin_verify_capture()
        with torch.inference_mode(), torch.cuda.graph(graph):
            hidden = self.model(ids, pos, _ctx())
            logits = self.model.compute_logits(hidden)  # [Bmax, S, vocab]
            true_tokens = logits.argmax(-1)  # [Bmax, S]
        vtraj = None
        if self.has_recurrent:
            vtraj = dict(self.lin_cache._vtraj)  # fixed-address graph-pool trajectory
            self.lin_cache._vcap = False

        pin = dev != "cpu"
        return _CapturedVerify(
            graph=graph, ids=ids, pos=pos, verify_slot_mapping=verify_slot_mapping,
            block_table=block_table, context_lens=context_lens,
            hidden=hidden, true_tokens=true_tokens,
            batch_bucket=batch_bucket, context_bucket=context_bucket,
            ids_host=torch.empty(batch_bucket, S, dtype=torch.long, pin_memory=pin),
            pos_host=torch.empty(batch_bucket, S, dtype=torch.long, pin_memory=pin),
            context_lens_host=torch.empty(batch_bucket, dtype=torch.int32, pin_memory=pin),
            vsm_host=torch.empty(batch_bucket * S, dtype=torch.int32, pin_memory=pin),
            slot_idx=slot_idx, slot_idx_host=slot_idx_host, vtraj=vtraj,
        )

    # -- per-step input refresh (eager, before replay) ---------------------
    def _fill_inputs(self, g: _CapturedVerify, batch, verify_ids, verify_pos, lengths):
        cache, S = self.cache, g.ids.shape[1]
        B, Bmax = len(batch), g.batch_bucket
        scratch = self.gd._scratch()
        pad = Bmax - B
        real_slots = [s.slot for s in batch]
        cache.ensure_capacity(real_slots, [n + 1 + S for n in lengths])

        # ids / pos: real rows carry the verify tokens/positions; pad rows are inert.
        g.ids_host[:B].copy_(torch.tensor(verify_ids, dtype=torch.long))
        g.pos_host[:B].copy_(torch.tensor(verify_pos, dtype=torch.long))
        g.context_lens_host[:B].copy_(torch.tensor([n + 1 for n in lengths], dtype=torch.int32))
        if pad:
            g.ids_host[B:].zero_()
            g.pos_host[B:].zero_()
            g.context_lens_host[B:].fill_(1)
        g.ids.copy_(g.ids_host, non_blocking=True)
        g.pos.copy_(g.pos_host, non_blocking=True)
        g.context_lens.copy_(g.context_lens_host, non_blocking=True)

        # verify_slot_mapping [Bmax*S]: (row b, verify pos t) -> paged slot for position
        # length[b]+1+t; pad rows map every position onto the scratch slot.
        flat_slots, flat_pos = [], []
        for b in range(B):
            base = lengths[b] + 1
            flat_slots.extend([real_slots[b]] * S)
            flat_pos.extend(base + t for t in range(S))
        for _ in range(pad):
            flat_slots.extend([scratch] * S)
            flat_pos.extend(range(S))
        cache.fill_slot_mapping(g.verify_slot_mapping, flat_slots, flat_pos)
        cache.fill_block_table(g.block_table, real_slots + [scratch] * pad)

        if g.slot_idx is not None:
            g.slot_idx_host.copy_(torch.tensor(real_slots + [scratch] * pad, dtype=torch.long))
            g.slot_idx.copy_(g.slot_idx_host, non_blocking=True)
            self.lin_cache.bind_graph(g.slot_idx)


class GraphedDecode:
    """Lazily captures one CUDA graph per (batch_bucket, context_bucket) bucket
    pair, replays on an exact match, and returns None (caller falls back to
    eager `EngineRunner.decode`) on a miss: unsupported model, batch larger
    than the widest bucket, or the captured-graph cap already reached."""

    def __init__(
        self,
        model,
        cache,
        *,
        device: str = "cuda",
        lin_cache=None,
        batch_buckets: tuple[int, ...] = DEFAULT_BATCH_BUCKETS,
        context_bucket_size: int = DEFAULT_CONTEXT_BUCKET_SIZE,
        max_graphs: int = DEFAULT_MAX_GRAPHS,
    ):
        self.model = model
        self.cache = cache
        self.lin_cache = lin_cache
        self.device = device
        # Never bucket past what the engine could ever schedule (`cache.num_slots`
        # == `max_num_seqs`) -- a bigger bucket would just never be hit.
        self.batch_buckets = tuple(sorted(b for b in set(batch_buckets) if b <= cache.num_slots))
        self.context_bucket_size = context_bucket_size
        self.max_graphs = max_graphs
        self._graphs: dict[tuple[int, int], _CapturedGraph] = {}
        self._scratch_slot: int | None = None
        # Host-side qualification counters.  These never touch CUDA state and make
        # the distinction between "graph object exists" and "decode actually
        # replayed" observable through the serving metrics endpoint.
        self._replay_count = 0
        self._capture_count = 0
        self._miss_counts: dict[str, int] = {}
        # Recurrent (short-conv / DeltaNet / lightning) mixers carry per-slot decode
        # state through `lin_cache`. When present AND capturable, we switch that cache
        # to fixed-address per-slot buffers so the recurrent update is static and can
        # be captured with the rest of the step (see models/cache.py).
        _modules = getattr(self.model, "modules", None)
        self.has_recurrent = callable(_modules) and any(
            getattr(m, "is_recurrent", False) for m in _modules()
        )
        self.supported, self._unsupported_reason = self._check_supported()
        if not self.supported:
            log.warning("cuda-graph decode disabled: %s", self._unsupported_reason)
        elif self.has_recurrent and self.lin_cache is not None:
            # Enable BEFORE the first prefill so prefill state lands in the same
            # fixed buffers the graphed decode replays against.
            self.lin_cache.enable_static_buffers(cache.num_slots)

    def status(self) -> dict:
        """Return cheap, host-only graph qualification state for telemetry.

        ``cuda_graph`` used to report only that this object was constructed.  A
        recurrent model can still fall back to eager decode when support checks,
        bucket admission, or the capture cap reject a step.  Keep this method
        deliberately free of CUDA queries so ``GET /metrics`` remains safe in the
        hot service and during startup diagnostics.
        """
        return {
            "supported": self.supported,
            "unsupported_reason": self._unsupported_reason or None,
            "batch_buckets": list(self.batch_buckets),
            "captured_graphs": len(self._graphs),
            "captures": self._capture_count,
            "replays": self._replay_count,
            "misses": dict(self._miss_counts),
        }

    def _record_miss(self, reason: str) -> None:
        self._miss_counts[reason] = self._miss_counts.get(reason, 0) + 1

    # -- capability check ------------------------------------------------
    def _check_supported(self) -> tuple[bool, str]:
        if self.device == "cpu" or not torch.cuda.is_available():
            return False, "no CUDA device"
        if not self.batch_buckets:
            return False, "no batch bucket <= max_num_seqs"
        cfg = self.model.config
        if cfg.is_moe():
            return False, "MoE routing is data-dependent (mask.nonzero()), not graph-capturable"
        # Latent attention (MLA / DeepSeek) decode reads a per-step-growing latent
        # slice out of MLALatentCache -- not yet expressible as a fixed-address
        # static buffer, so keep declining it (a separate optimization).
        if cfg.latent_attention:
            return False, "latent (MLA) decode reads a growing latent slice, not yet static"
        # Recurrent mixers (LFM2 short-conv, Qwen3-Next DeltaNet, MiniMax lightning)
        # ARE capturable now: their per-slot decode state lives in fixed-address
        # buffers (models/cache.py `enable_static_buffers`), gathered/scattered in
        # place through a persistent device row->slot index. So 'linear' layers are
        # accepted alongside 'full'; only 'sliding' (a data-dependent python-loop
        # fallback) and 'latent' remain uncapturable.
        for i in range(cfg.num_hidden_layers):
            kind = cfg.attention_kind(i)
            if kind not in ("full", "linear"):
                return (
                    False,
                    f"layer {i} uses the '{kind}' attention backend "
                    "(sliding-window decode is a data-dependent python-loop fallback)",
                )
        return True, ""

    def _scratch(self, min_len: int = 1) -> int:
        """A dedicated cache slot every pad row's slot/block-table entries point
        at -- allocated once and never freed, so pad rows always read/write valid
        (if inert) memory regardless of which real sequences are live."""
        if self._scratch_slot is None:
            self._scratch_slot = self.cache.alloc()
        self.cache.ensure_capacity([self._scratch_slot], [min_len])
        return self._scratch_slot

    # -- public entry point -----------------------------------------------
    def try_decode(self, batch: list[Sequence]) -> torch.Tensor | None:
        """Returns next-token logits `[len(batch), vocab]` for `batch` via a
        captured graph, or None if this step can't be served from one (the
        caller should fall back to eager decode). Never mutates `Sequence`
        state -- the caller owns `seq.length` bookkeeping either way."""
        if not self.supported:
            self._record_miss("unsupported")
            return None
        if not batch:
            self._record_miss("empty_batch")
            return None
        B = len(batch)
        batch_bucket = _next_bucket(self.batch_buckets, B)
        if batch_bucket is None:
            self._record_miss("batch_over_bucket")
            log.info(
                "cuda-graph decode miss: batch size %d exceeds largest bucket %d",
                B,
                self.batch_buckets[-1],
            )
            return None
        max_real_ctx = max(s.length for s in batch) + 1
        cap = self.cache.max_blocks_per_seq * self.cache.block_size
        context_bucket = min(_round_up(max_real_ctx, self.context_bucket_size), cap)
        key = (batch_bucket, context_bucket)

        g = self._graphs.get(key)
        if g is None:
            if len(self._graphs) >= self.max_graphs:
                log.warning(
                    "cuda-graph decode miss: bucket cap (%d graphs) reached, "
                    "not capturing batch=%d max_context=%d",
                    self.max_graphs,
                    batch_bucket,
                    context_bucket,
                )
                self._record_miss("graph_cap")
                return None
            g = self._capture(batch_bucket, context_bucket)
            self._graphs[key] = g
            self._capture_count += 1
            log.info(
                "cuda-graph decode: captured bucket batch=%d max_context=%d (%d graphs total)",
                batch_bucket,
                context_bucket,
                len(self._graphs),
            )

        self._fill_inputs(g, batch)
        g.graph.replay()
        self._replay_count += 1
        return g.logits[:B]

    # -- capture ------------------------------------------------------------
    def _capture(self, batch_bucket: int, context_bucket: int) -> _CapturedGraph:
        cache = self.cache
        scratch = self._scratch()

        ids = torch.zeros(batch_bucket, 1, dtype=torch.long, device=self.device)
        pos = torch.zeros(batch_bucket, 1, dtype=torch.long, device=self.device)
        slot_mapping = cache.slot_mapping_for([scratch] * batch_bucket, [0] * batch_bucket)
        block_table = cache.block_table([scratch] * batch_bucket)
        context_lens = torch.ones(batch_bucket, dtype=torch.int32, device=self.device)

        # Recurrent models: a persistent device row->slot index the captured
        # recurrent-state gather/scatter reads. Seed every row at the scratch slot
        # (warmup/capture must never touch a real sequence's state); _fill_inputs
        # refreshes it to the batch's real slots before each replay.
        slot_idx = slot_idx_host = None
        if self.has_recurrent and self.lin_cache is not None:
            slot_idx = torch.full((batch_bucket,), scratch, dtype=torch.long, device=self.device)
            pin = self.device != "cpu" and torch.cuda.is_available()
            slot_idx_host = torch.empty(batch_bucket, dtype=torch.long, pin_memory=pin)
            self.lin_cache.bind_graph(slot_idx)

        ctx = ForwardContext(
            is_prefill=False,
            kv_cache=cache,
            lin_cache=self.lin_cache if self.has_recurrent else None,
            slot_mapping=slot_mapping,
            block_tables=block_table,
            context_lens=context_lens,
            max_context_len=context_bucket,
        )

        # Standard two-phase capture (PyTorch CUDA-graph guidance): warm up a few
        # iterations on a side stream first so the caching allocator reaches a
        # steady state (capture fails if it has to grow the pool mid-capture),
        # THEN capture. Warmup runs against the same scratch-only inputs the
        # graph is seeded with above -- it never touches a real sequence's data.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(_WARMUP_ITERS):
                with torch.inference_mode():
                    hidden = self.model(ids, pos, ctx)
                    self.model.compute_logits(hidden[:, -1])
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            hidden = self.model(ids, pos, ctx)
            logits = self.model.compute_logits(hidden[:, -1])

        pin = self.device != "cpu" and torch.cuda.is_available()
        ids_host = torch.empty(batch_bucket, 1, dtype=torch.long, pin_memory=pin)
        pos_host = torch.empty(batch_bucket, 1, dtype=torch.long, pin_memory=pin)
        context_lens_host = torch.empty(batch_bucket, dtype=torch.int32, pin_memory=pin)

        return _CapturedGraph(
            graph=graph,
            ids=ids,
            pos=pos,
            slot_mapping=slot_mapping,
            block_table=block_table,
            context_lens=context_lens,
            logits=logits,
            batch_bucket=batch_bucket,
            context_bucket=context_bucket,
            ids_host=ids_host,
            pos_host=pos_host,
            context_lens_host=context_lens_host,
            slot_idx=slot_idx,
            slot_idx_host=slot_idx_host,
        )

    # -- per-step input refresh (eager -- runs before replay(), not captured) ----
    def _fill_inputs(self, g: _CapturedGraph, batch: list[Sequence]):
        cache = self.cache
        B, Bmax = len(batch), g.batch_bucket
        scratch = self._scratch()
        pad = Bmax - B

        real_slots = [s.slot for s in batch]
        real_lengths = [s.length for s in batch]
        cache.ensure_capacity(real_slots, [n + 1 for n in real_lengths])

        slots = real_slots + [scratch] * pad
        lengths = real_lengths + [0] * pad

        # Fill the pinned host staging in place (CPU-only work, no GPU sync): real
        # rows carry the sequence's next token / position / context length, pad rows
        # are inert (id 0, pos 0, context_len 1 -- always point at the scratch slot).
        pin = g.ids_host.is_pinned() if hasattr(g.ids_host, "is_pinned") else False
        g.ids_host[:B, 0].copy_(torch.tensor([s.last_token for s in batch], dtype=torch.long))
        g.pos_host[:B, 0].copy_(torch.tensor([s.length for s in batch], dtype=torch.long))
        g.context_lens_host[:B].copy_(
            torch.tensor([s.length + 1 for s in batch], dtype=torch.int32)
        )
        if pad:
            g.ids_host[B:, 0].zero_()
            g.pos_host[B:, 0].zero_()
            g.context_lens_host[B:].fill_(1)

        # One non_blocking H2D per buffer, into the SAME device tensor the graph
        # captured (contents mutated in place -- pointer preserved for replay).
        g.ids.copy_(g.ids_host, non_blocking=pin)
        g.pos.copy_(g.pos_host, non_blocking=pin)
        g.context_lens.copy_(g.context_lens_host, non_blocking=pin)
        cache.fill_slot_mapping(g.slot_mapping, slots, lengths)
        cache.fill_block_table(g.block_table, slots)

        # Recurrent models: refresh the persistent row->slot index in place (real
        # rows -> their slot, pad rows -> scratch) so the captured recurrent-state
        # gather/scatter reads/writes each live sequence's own fixed buffer row.
        # Re-bind every step: an intervening eager step (prefill / fallback) clears
        # the graph index, so the graphed path must reinstate its own before replay.
        if g.slot_idx is not None:
            g.slot_idx_host.copy_(torch.tensor(slots, dtype=torch.long))
            g.slot_idx.copy_(g.slot_idx_host, non_blocking=pin)
            self.lin_cache.bind_graph(g.slot_idx)


# ---------------------------------------------------------------------------
# Per-layer graph capture (Phase 1 — issue #320)
# ---------------------------------------------------------------------------

def layer_graph_enabled_by_env(default: bool = True) -> bool:
    """``SUPERL8SERVE_LAYER_GRAPH=0`` disables per-layer graphs (falls back to
    :class:`GraphedDecode`); unset or any other value keeps the default."""
    val = os.environ.get("SUPERL8SERVE_LAYER_GRAPH")
    if val is None:
        return default
    return val not in ("0", "false", "False")


def _layer_uses_residual(layer) -> bool:
    """Detect whether a decoder layer's forward takes a ``residual`` parameter
    (pre-norm style, e.g. Qwen3 / Gemma3) vs a ``layer_idx`` parameter
    (LFM2 post-norm style).  Returns ``True`` for the residual pattern."""
    import inspect

    sig = inspect.signature(layer.forward)
    params = list(sig.parameters.keys())
    if not params:
        return False
    return params[-1] in ("residual",)


def _model_inner(model):
    """Return the inner ``nn.Module`` that holds ``.layers``, ``.embed_tokens``,
    ``.norm``.  Most CausalLM wrappers have ``self.model = InnerModel``; flat
    models (LFM2) carry them directly."""
    inner = getattr(model, "model", model)
    if hasattr(inner, "layers"):
        return inner
    return model


class GraphedDecodeLayer:
    """Captured CUDA graph for a single decoder layer.

    The graph reads from ``staging_bufs[idx]`` (persistent device tensors)
    and writes into ``staging_bufs[idx + 1]``.  Persistent context tensors
    (``ids``, ``pos``, ``slot_mapping``, etc.) are refreshed by the caller
    *before* the first replay — the same contract as :class:`GraphedDecode`.

    ``staging_bufs`` is the full list (length ``num_layers + 1``); the layer
    only touches ``[idx]`` and ``[idx + 1]``.
    """

    __slots__ = ("graph", "idx", "staging_bufs", "uses_residual")

    def __init__(self, graph, idx, staging_bufs, uses_residual: bool):
        self.graph = graph
        self.idx = idx
        self.staging_bufs = staging_bufs
        self.uses_residual = uses_residual

    def replay(self, stream=None):
        if stream is not None:
            # Replay the graph on the specified stream by setting it as current
            with torch.cuda.stream(stream):
                self.graph.replay()
        else:
            self.graph.replay()


# ---------------------------------------------------------------------------
# Phase 2: inter-layer pipelining helpers (issue #321)
# ---------------------------------------------------------------------------

def _check_l2_persistent_support() -> bool:
    """Check if the GPU supports L2 persistence hints (sm_70+ / Volta+)."""
    try:
        if not torch.cuda.is_available():
            return False
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return props.major >= 7
    except Exception:
        return False


class _L2PersistHelper:
    """Best-effort L2 persistence hints via CUDA driver API.

    Pins a tensor's device memory in L2 cache using ``cudaAccessPolicyWindow``
    so that when the NEXT layer's graph replays, its weight reads hit L2 instead
    of going to HBM.  Falls back silently if the driver API is unavailable or
    the call fails (non-fatal optimization).

    Only the CUDA driver ``libcuda.so`` path is attempted; no recompilation,
    no custom kernels.
    """

    _lib = None
    _available: bool | None = None  # None = not probed yet

    @classmethod
    def available(cls) -> bool:
        if cls._available is not None:
            return cls._available
        try:
            import ctypes

            try:
                lib = ctypes.CDLL("libcuda.so.1")
            except OSError:
                lib = ctypes.CDLL("libcuda.so")
            cls._lib = lib
            ret = lib.cuInit(0)
            cls._available = ret == 0
        except Exception:
            cls._available = False
        return cls._available

    @classmethod
    def set_persistent(cls, tensor: torch.Tensor) -> bool:
        """Pin *tensor* in L2 as a persisting window.  Returns True on success."""
        if not cls.available():
            return False
        try:
            import ctypes

            lib = cls._lib

            device = ctypes.c_int()
            ret = lib.cuCtxGetDevice(ctypes.byref(device))
            if ret != 0:
                return False

            CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 35
            l2_size = ctypes.c_int()
            ret = lib.cuDeviceGetAttribute(
                ctypes.byref(l2_size), CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device
            )
            if ret != 0 or l2_size.value <= 0:
                return False

            ptr = tensor.data_ptr()
            size = tensor.nelement() * tensor.element_size()
            if size == 0:
                return False

            class CUaccessPolicyWindow(ctypes.Structure):
                _fields_ = [
                    ("base_ptr", ctypes.c_void_p),
                    ("size", ctypes.c_size_t),
                    ("offset", ctypes.c_size_t),
                    ("hitProp", ctypes.c_uint),
                    ("missProp", ctypes.c_uint),
                ]

            CU_ACCESS_PROPERTY_PERSISTING = 3
            CU_ACCESS_PROPERTY_STREAMING = 0

            policy = CUaccessPolicyWindow()
            policy.base_ptr = ptr
            policy.size = size
            policy.offset = 0
            policy.hitProp = CU_ACCESS_PROPERTY_PERSISTING
            policy.missProp = CU_ACCESS_PROPERTY_STREAMING

            stream = torch.cuda.current_stream().cuda_stream
            ret = lib.cuStreamSetAccessPolicy(stream, ctypes.byref(policy), 1)
            return ret == 0
        except Exception:
            return False

    @classmethod
    def clear_persistent(cls, tensor: torch.Tensor) -> bool:
        """Remove L2 persistence hint (back to streaming)."""
        if not cls.available():
            return False
        try:
            import ctypes

            lib = cls._lib

            ptr = tensor.data_ptr()
            size = tensor.nelement() * tensor.element_size()
            if size == 0:
                return False

            class CUaccessPolicyWindow(ctypes.Structure):
                _fields_ = [
                    ("base_ptr", ctypes.c_void_p),
                    ("size", ctypes.c_size_t),
                    ("offset", ctypes.c_size_t),
                    ("hitProp", ctypes.c_uint),
                    ("missProp", ctypes.c_uint),
                ]

            CU_ACCESS_PROPERTY_STREAMING = 0

            policy = CUaccessPolicyWindow()
            policy.base_ptr = ptr
            policy.size = size
            policy.offset = 0
            policy.hitProp = CU_ACCESS_PROPERTY_STREAMING
            policy.missProp = CU_ACCESS_PROPERTY_STREAMING

            stream = torch.cuda.current_stream().cuda_stream
            ret = lib.cuStreamSetAccessPolicy(stream, ctypes.byref(policy), 1)
            return ret == 0
        except Exception:
            return False


class GraphedDecodeLayers:
    """Per-layer CUDA graph capture and replay for weight-stationary decode.

    Captures N separate graphs (one per model layer) plus a final-norm+logits
    graph.  Each layer graph reads from ``staging_bufs[i]`` and writes to
    ``staging_bufs[i+1]``.  Persistent context tensors are refreshed via
    ``copy_`` before replay (same contract as :class:`GraphedDecode`).

    Falls back to ``None`` from ``capture_layer_graphs`` for unsupported
    models (MoE, sliding-window, latent attention).
    """

    def __init__(
        self,
        model,
        cache,
        *,
        device: str = "cuda",
        lin_cache=None,
        forward_cache=None,
        batch_buckets: tuple[int, ...] = DEFAULT_BATCH_BUCKETS,
        context_bucket_size: int = DEFAULT_CONTEXT_BUCKET_SIZE,
        max_graphs: int = DEFAULT_MAX_GRAPHS,
        enable_stream_overlap: bool = True,
        layers_only: bool = False,
        input_residual: bool = False,
    ):
        self.model = model
        self.cache = cache
        self.forward_cache = forward_cache or cache
        self.lin_cache = lin_cache
        self.device = device
        self.batch_buckets = tuple(
            sorted(b for b in set(batch_buckets) if b <= cache.num_slots)
        )
        self.context_bucket_size = context_bucket_size
        self.max_graphs = max_graphs
        self.layers_only = layers_only
        self.input_residual = input_residual
        self.supported = True
        self._unsupported_reason = ""

        # Check model support BEFORE accessing inner.layers (FakeModel may lack them)
        cfg = model.config
        if cfg.is_moe():
            self.supported = False
            self._unsupported_reason = "MoE routing is data-dependent, not graph-capturable"
        elif cfg.latent_attention:
            self.supported = False
            self._unsupported_reason = "latent (MLA) decode not yet static"
        else:
            for i in range(cfg.num_hidden_layers):
                kind = cfg.attention_kind(i)
                if kind not in ("full", "linear"):
                    self.supported = False
                    self._unsupported_reason = (
                        f"layer {i} uses '{kind}' attention backend"
                    )
                    break

        if not self.supported:
            log.warning("per-layer graph decode disabled: %s", self._unsupported_reason)
            return

        inner = _model_inner(model)
        self._inner = inner
        self._num_layers = len(inner.layers)
        self._hidden_dim = model.config.hidden_size
        self._uses_residual = _layer_uses_residual(inner.layers[0])

        # _captures[bucket_key] = (layer_graphs, norm_logits_graph,
        #   ids, pos, slot_mapping, block_table, context_lens,
        #   scratch_slot, slot_idx, slot_idx_host, has_recurrent, ctx)
        self._captures: dict[tuple[int, int], tuple] = {}

        # Phase 2: double-buffered CUDA streams for inter-layer pipelining
        self._enable_stream_overlap = (
            enable_stream_overlap
            and device != "cpu"
            and torch.cuda.is_available()
        )
        self._streams: list[torch.cuda.Stream] = []
        if self._enable_stream_overlap:
            self._streams = [
                torch.cuda.Stream(device=self.device) for _ in range(2)
            ]

        # Recurrent detection
        _modules = getattr(model, "modules", None)
        self._has_recurrent = callable(_modules) and any(
            getattr(m, "is_recurrent", False) for m in _modules()
        )
        if self._has_recurrent and lin_cache is not None:
            lin_cache.enable_static_buffers(cache.num_slots)

    def _scratch(self) -> int:
        if not hasattr(self, "_scratch_slot") or self._scratch_slot is None:
            self._scratch_slot = self.cache.alloc()
            self.cache.ensure_capacity([self._scratch_slot], [1])
        return self._scratch_slot

    def _make_staging_bufs(self, B: int):
        """Create fresh staging buffers for a bucket of size B."""
        from .staging import StagingBuffer

        return [
            StagingBuffer(B, self._hidden_dim, self.device)
            for _ in range(self._num_layers + 1)
        ]

    def _ensure_capture(
        self, batch_bucket: int, context_bucket: int
    ) -> tuple | None:
        with torch.cuda.device(self.device):
            return self._ensure_capture_on_device(batch_bucket, context_bucket)

    def _ensure_capture_on_device(
        self, batch_bucket: int, context_bucket: int
    ) -> tuple | None:
        key = (batch_bucket, context_bucket)
        if key in self._captures:
            return self._captures[key]
        if len(self._captures) >= self.max_graphs:
            return None

        B = batch_bucket
        staging_bufs = self._make_staging_bufs(B)
        scratch = self._scratch()
        cache = self.cache

        ids = torch.zeros(B, 1, dtype=torch.long, device=self.device)
        pos = torch.zeros(B, 1, dtype=torch.long, device=self.device)
        slot_mapping = cache.slot_mapping_for(
            [scratch] * B, [0] * B
        )
        block_table = cache.block_table([scratch] * B)
        context_lens = torch.ones(B, dtype=torch.int32, device=self.device)

        slot_idx = slot_idx_host = None
        if self._has_recurrent and self.lin_cache is not None:
            slot_idx = torch.full(
                (B,), scratch, dtype=torch.long, device=self.device
            )
            pin = self.device != "cpu" and torch.cuda.is_available()
            slot_idx_host = torch.empty(B, dtype=torch.long, pin_memory=pin)
            self.lin_cache.bind_graph(slot_idx)

        ctx = ForwardContext(
            is_prefill=False,
            kv_cache=self.forward_cache,
            lin_cache=self.lin_cache if self._has_recurrent else None,
            slot_mapping=slot_mapping,
            block_tables=block_table,
            context_lens=context_lens,
            max_context_len=context_bucket,
        )

        inner = self._inner
        layers = inner.layers
        uses_residual = self._uses_residual

        # Warmup
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(_WARMUP_ITERS):
                with torch.inference_mode():
                    if self.layers_only:
                        h = staging_bufs[0].buf[:B]
                        residual = (
                            staging_bufs[0].residual[:B]
                            if uses_residual and self.input_residual
                            else None
                        )
                    else:
                        h = inner.embed_tokens(ids)
                        residual = None
                    for layer in layers:
                        if uses_residual:
                            h, residual = layer(h, pos, ctx, residual)
                        else:
                            out = layer(h, pos, ctx, 0)
                            h = out[0] if isinstance(out, tuple) else out
                    if not self.layers_only:
                        if uses_residual:
                            h, _ = inner.norm(h, residual)
                        else:
                            h = inner.norm(h)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        # Capture the strictly ordered layer chain into one private allocator
        # pool.  Without this hint PyTorch reserves a separate graph pool for
        # every layer (64 for Qwen3.8), which pushes the otherwise-fitting model
        # over the physical memory envelope. Replay follows this same layer order and the
        # stream hand-off serializes adjacent graphs, so their temporary storage
        # is never live concurrently.
        shared_pool = torch.cuda.graph_pool_handle()
        layer_graphs: list[GraphedDecodeLayer] = []
        for layer_idx, layer in enumerate(layers):
            graph = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(
                graph, pool=shared_pool, stream=stream
            ):
                staging_in = staging_bufs[layer_idx]
                staging_out = staging_bufs[layer_idx + 1]
                if uses_residual:
                    if layer_idx == 0 and not self.input_residual:
                        h_out, res_out = layer(
                            staging_in.buf[:B], pos, ctx, None
                        )
                    else:
                        h_out, res_out = layer(
                            staging_in.buf[:B], pos, ctx,
                            staging_in.residual[:B],
                        )
                    staging_out.buf[:B].copy_(h_out)
                    staging_out.residual[:B].copy_(res_out)
                else:
                    out = layer(staging_in.buf[:B], pos, ctx, layer_idx)
                    h_out = out[0] if isinstance(out, tuple) else out
                    staging_out.buf[:B].copy_(h_out)
            layer_graphs.append(
                GraphedDecodeLayer(
                    graph, layer_idx, staging_bufs, uses_residual
                )
            )

        norm_logits_graph = None
        logits_out = None
        if not self.layers_only:
            # Ordinary model decode owns the final norm and vocabulary projection.
            # Pipeline stages only capture their local layer range; the pipeline
            # applies the final stage's norm/head after all boundary transfers.
            last_buf = staging_bufs[self._num_layers]
            norm_logits_graph = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(
                norm_logits_graph, pool=shared_pool, stream=stream
            ):
                if uses_residual:
                    h_norm, _ = inner.norm(
                        last_buf.buf[:B], last_buf.residual[:B]
                    )
                else:
                    h_norm = inner.norm(last_buf.buf[:B])
                logits_out = self.model.compute_logits(h_norm)

        pin = self.device != "cpu" and torch.cuda.is_available()
        ids_host = torch.empty(B, 1, dtype=torch.long, pin_memory=pin)
        pos_host = torch.empty(B, 1, dtype=torch.long, pin_memory=pin)
        context_lens_host = torch.empty(B, dtype=torch.int32, pin_memory=pin)

        result = (
            layer_graphs,
            norm_logits_graph,
            ids,
            pos,
            slot_mapping,
            block_table,
            context_lens,
            scratch,
            slot_idx,
            slot_idx_host,
            self._has_recurrent,
            ctx,
            self._num_layers,
            uses_residual,
            staging_bufs,
            logits_out,
            ids_host,
            pos_host,
            context_lens_host,
        )
        self._captures[key] = result
        log.info(
            "per-layer graph: captured batch=%d max_context=%d "
            "(%d layers, %d captures)",
            batch_bucket,
            context_bucket,
            self._num_layers,
            len(self._captures),
        )
        return result

    def try_decode(self, batch: list[Sequence]) -> torch.Tensor | None:
        """Per-layer graph decode.  Returns logits ``[B, vocab]`` or ``None``
        (caller falls back to :class:`GraphedDecode` or eager)."""
        if self.layers_only or not self.supported or not batch:
            return None
        B = len(batch)
        batch_bucket = _next_bucket(self.batch_buckets, B)
        if batch_bucket is None:
            return None
        max_real_ctx = max(s.length for s in batch) + 1
        cap = self.cache.max_blocks_per_seq * self.cache.block_size
        context_bucket = min(
            _round_up(max_real_ctx, self.context_bucket_size), cap
        )
        if max_real_ctx > cap:
            return None

        cap_data = self._ensure_capture(batch_bucket, context_bucket)
        if cap_data is None:
            return None

        (
            layer_graphs,
            norm_logits_graph,
            ids,
            pos,
            slot_mapping,
            block_table,
            context_lens,
            scratch,
            slot_idx,
            slot_idx_host,
            has_recurrent,
            ctx,
            num_layers,
            uses_residual,
            staging_bufs,
            logits_out,
            ids_host,
            pos_host,
            context_lens_host,
        ) = cap_data

        # -- fill persistent inputs (same as GraphedDecode._fill_inputs) -----
        cache = self.cache
        Bmax = batch_bucket
        pad = Bmax - B
        real_slots = [s.slot for s in batch]
        real_lengths = [s.length for s in batch]
        cache.ensure_capacity(real_slots, [n + 1 for n in real_lengths])
        slots = real_slots + [scratch] * pad
        lengths = real_lengths + [0] * pad

        pin = ids_host.is_pinned()
        # Populate the persistent pinned staging tensors in place.  Constructing
        # temporary CPU tensors here would reintroduce per-token allocations in
        # the replay path that this class exists to remove.
        for i, seq in enumerate(batch):
            ids_host[i, 0] = seq.last_token
            pos_host[i, 0] = seq.length
            context_lens_host[i] = seq.length + 1
        if pad:
            ids_host[B:, 0].zero_()
            pos_host[B:, 0].zero_()
            context_lens_host[B:].fill_(1)
        ids.copy_(ids_host, non_blocking=pin)
        pos.copy_(pos_host, non_blocking=pin)
        context_lens.copy_(context_lens_host, non_blocking=pin)
        cache.fill_slot_mapping(slot_mapping, slots, lengths)
        cache.fill_block_table(block_table, slots)

        if slot_idx is not None:
            for i, slot in enumerate(slots):
                slot_idx_host[i] = slot
            slot_idx.copy_(slot_idx_host, non_blocking=pin)
            self.lin_cache.bind_graph(slot_idx)

        # -- run embedding eagerly into staging_bufs[0] --------------------
        h_emb = self._inner.embed_tokens(ids[:B])
        staging_bufs[0].buf[:B].copy_(h_emb)
        staging_bufs[0].active_count = B

        # -- replay per-layer graphs (Phase 2: stream overlap) ----------------
        self._replay_layer_graphs(layer_graphs, staging_bufs, B)

        # -- final norm + logits (eager) -----------------------------------
        if staging_bufs[num_layers].is_empty():
            return None

        if uses_residual:
            h_norm, _ = self._inner.norm(
                staging_bufs[num_layers].buf[:B],
                staging_bufs[num_layers].residual[:B],
            )
        else:
            h_norm = self._inner.norm(staging_bufs[num_layers].buf[:B])
        # h_norm is [B, 1, H]; take last position to get [B, H] for logits.
        return self.model.compute_logits(h_norm[:, -1])

    def _replay_layer_graphs(self, layer_graphs, staging_bufs, B):
        """Replay per-layer graphs with stream overlap (shared by try_decode and
        replay_layers_from_staging)."""
        if self._enable_stream_overlap and self._streams and len(layer_graphs) > 1:
            streams = self._streams
            active_stream = streams[0]
            for i, lg in enumerate(layer_graphs):
                if staging_bufs[lg.idx].is_empty():
                    staging_bufs[lg.idx + 1].active_count = 0
                    continue
                stream_idx = i % 2
                current_stream = streams[stream_idx]
                other_stream = streams[1 - stream_idx]
                current_stream.wait_stream(other_stream)
                if _L2PersistHelper.available():
                    _L2PersistHelper.set_persistent(staging_bufs[lg.idx].buf)
                lg.replay(current_stream)
                active_stream = current_stream
                staging_bufs[lg.idx + 1].active_count = B
            torch.cuda.current_stream().wait_stream(active_stream)
        else:
            for lg in layer_graphs:
                if staging_bufs[lg.idx].is_empty():
                    staging_bufs[lg.idx + 1].active_count = 0
                    continue
                lg.replay()
                staging_bufs[lg.idx + 1].active_count = B

    def replay_layers_from_staging(
        self,
        staging_bufs: list,
        B: int,
        batch_bucket: int,
        context_bucket: int,
    ) -> torch.Tensor | None:
        """Replay layer graphs starting from a pre-filled staging_bufs[0].

        Used by downstream GPU in pipeline parallelism: receives activations into
        staging_bufs[0], replays layers[0:] -> writes to staging_bufs[num_layers].

        Returns logits [B, vocab] or None (if boundary is empty).
        """
        cap_data = self._ensure_capture(batch_bucket, context_bucket)
        if cap_data is None:
            return None

        (
            layer_graphs,
            norm_logits_graph,
            ids,
            pos,
            slot_mapping,
            block_table,
            context_lens,
            scratch,
            slot_idx,
            slot_idx_host,
            has_recurrent,
            ctx,
            num_layers,
            uses_residual,
            _own_staging_bufs,
            logits_out,
            ids_host,
            pos_host,
            context_lens_host,
        ) = cap_data

        self._replay_layer_graphs(layer_graphs, staging_bufs, B)

        if staging_bufs[num_layers].is_empty():
            return None

        if uses_residual:
            h_norm, _ = self._inner.norm(
                staging_bufs[num_layers].buf[:B],
                staging_bufs[num_layers].residual[:B],
            )
        else:
            h_norm = self._inner.norm(staging_bufs[num_layers].buf[:B])
        return self.model.compute_logits(h_norm[:, -1])

    def get_boundary_output(
        self, staging_bufs: list, B: int
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        """Return (hidden, residual, active_count) from the last staging buffer.

        Used by upstream GPU in PP to get the activations to send downstream.
        Returns (buf_tensor [B, 1, H], residual_tensor [B, 1, H] | None, count).
        """
        num_layers = self._num_layers
        last_buf = staging_bufs[num_layers]
        hidden = last_buf.buf[:B].clone()
        residual = last_buf.residual[:B].clone() if self._uses_residual else None
        return hidden, residual, last_buf.active_count

# ---------------------------------------------------------------------------
# GraphedPrefill -- CUDA-graph capture/replay for single-sequence prefill
# ---------------------------------------------------------------------------
# Prefill is dispatch-bound just like decode (81,502 kernel launches for 256
# tokens, 797ms GPU but 6.2s CPU). Capturing the full prefill forward pass
# into a CUDA graph eliminates all Python dispatch overhead: replay re-issues
# every kernel as ONE cudaGraphLaunch.
#
# Unlike decode (batch=1..128, variable context), prefill is always batch=1
# with variable prompt length S. We bucket by S and pad to the next bucket
# size; padding tokens write to scratch KV slots and are sliced off before
# sampling.
#
# See also: GraphedDecode (decode capture), issue #459.
# ---------------------------------------------------------------------------

# Finer-grained near the low/mid end (where relative padding waste hurts
# most -- a 147-token prompt against the old 4 low-range buckets
# (32,64,128,256) padded up to 256, wasting 43% of the forward pass on
# padding; against these 9 low-range buckets it resolves to 160, ~8%
# waste), coarser at the high end where each extra bucket costs
# proportionally more VRAM for proportionally less padding-waste benefit.
# Measured 2026-09-14 on a V100, Qwen3.5-9B W8A8, 147-token prompt: 245 ->
# 367 tok/s pure prefill (superl8-serve#459).
DEFAULT_PREFILL_BUCKETS = (
    32, 48, 64, 96, 128, 160, 192, 224, 256, 384, 512, 768, 1024, 1536, 2048,
)


class _CapturedPrefillGraph:
    """A single captured CUDA graph for one prefill bucket size."""

    __slots__ = (
        "graph", "ids", "pos", "slot_mapping", "context_lens",
        "logits", "seq_len_bucket",
        "ids_host", "pos_host", "slot_idx", "out_idx",
    )

    def __init__(self, *, graph, ids, pos, slot_mapping, context_lens,
                 logits, seq_len_bucket, ids_host, pos_host, slot_idx=None, out_idx=None):
        self.graph = graph
        self.ids = ids
        self.pos = pos
        self.slot_mapping = slot_mapping
        self.context_lens = context_lens
        self.logits = logits
        self.seq_len_bucket = seq_len_bucket
        self.ids_host = ids_host
        self.pos_host = pos_host
        self.slot_idx = slot_idx
        self.out_idx = out_idx


class GraphedPrefill:
    """Lazily captures one CUDA graph per prompt-length bucket, replays on
    an exact match, and returns None (caller falls back to eager prefill)
    on a miss.

    Expected speedup: prefill goes from ~50 tok/s (CPU dispatch bound) to
    ~300 tok/s (GPU compute bound) for 256-token prompts on V100.
    """

    def __init__(
        self,
        model,
        cache,
        *,
        device: str = "cuda",
        lin_cache=None,
        seq_buckets: tuple[int, ...] = DEFAULT_PREFILL_BUCKETS,
        max_graphs: int = 16,  # >= len(DEFAULT_PREFILL_BUCKETS), was 8
    ):
        self.model = model
        self.cache = cache
        self.lin_cache = lin_cache
        self.device = device
        self.seq_buckets = tuple(sorted(seq_buckets))
        self.max_graphs = max_graphs
        self._graphs: dict[int, _CapturedPrefillGraph] = {}
        self._scratch_slot: int | None = None
        self._replay_count = 0
        self._capture_count = 0
        self._miss_counts: dict[str, int] = {}

        # Recurrent layers need static buffers for graph capture
        _modules = getattr(self.model, "modules", None)
        self.has_recurrent = callable(_modules) and any(
            getattr(m, "is_recurrent", False) for m in _modules()
        )
        self.supported = self._check_supported()
        if self.supported and self.has_recurrent and self.lin_cache is not None:
            self.lin_cache.enable_static_buffers(cache.num_slots)

    def _check_supported(self) -> bool:
        if self.device == "cpu" or not torch.cuda.is_available():
            return False
        if not self.seq_buckets:
            return False
        cfg = self.model.config
        if cfg.is_moe():
            return False  # MoE routing is data-dependent
        if cfg.latent_attention:
            return False  # MLA has growing latent state
        for i in range(cfg.num_hidden_layers):
            kind = cfg.attention_kind(i)
            if kind not in ("full", "linear"):
                return False
        return True

    def _scratch(self, min_len: int = 1) -> int:
        if self._scratch_slot is None:
            self._scratch_slot = self.cache.alloc()
            self.cache.ensure_capacity([self._scratch_slot], [min_len])
        return self._scratch_slot

    def _next_bucket(self, n: int) -> int | None:
        for b in self.seq_buckets:
            if n <= b:
                return b
        return None

    def try_prefill(self, seq) -> int | None:
        """Try to serve prefill via a captured CUDA graph. Returns sampled
        token id, or None if the caller should fall back to eager."""
        from .sequence import Sequence

        if not self.supported:
            self._record_miss("unsupported")
            return None

        n = seq.num_prompt
        bucket = self._next_bucket(n)
        if bucket is None:
            self._record_miss("over_bucket")
            return None

        key = bucket
        g = self._graphs.get(key)
        if g is None:
            if len(self._graphs) >= self.max_graphs:
                self._record_miss("graph_cap")
                return None
            g = self._capture(bucket)
            if g is None:
                self._record_miss("capture_failed")
                return None
            self._graphs[key] = g
            self._capture_count += 1
            log.info("cuda-graph prefill captured: S=%d", bucket)

        # Refresh input buffers with real data
        self._fill_inputs(g, seq, n)
        g.graph.replay()
        self._replay_count += 1

        # Sample from the graph's output logits (last real token position)
        logits = g.logits[:1]  # [1, vocab] -- always row 0 (batch=1)
        return logits.argmax(dim=-1).item()

    def _record_miss(self, reason: str) -> None:
        self._miss_counts[reason] = self._miss_counts.get(reason, 0) + 1

    def status(self) -> dict:
        return {
            "supported": self.supported,
            "seq_buckets": list(self.seq_buckets),
            "captured_graphs": len(self._graphs),
            "captures": self._capture_count,
            "replays": self._replay_count,
            "misses": dict(self._miss_counts),
        }

    def _fill_inputs(self, g: _CapturedPrefillGraph, seq, n: int) -> None:
        """Copy real prompt data into the captured graph's static buffers."""
        bucket = g.seq_len_bucket
        scratch = self._scratch()

        # Slot mapping: real tokens -> seq.slot, padding -> scratch
        slot = seq.slot
        slots_real = self.cache.flat_slot_mapping(slot, n)
        slots_pad = [scratch] * (bucket - n)
        all_slots = slots_real + slots_pad
        g.slot_mapping.copy_(
            torch.tensor(all_slots, dtype=torch.int32, device=self.device)
        )

        # Position: 0..n-1 for real tokens, 0 for padding
        pos_real = list(range(n))
        pos_pad = [0] * (bucket - n)
        g.pos_host[:bucket] = torch.tensor(pos_real + pos_pad, dtype=torch.long)
        g.pos.copy_(g.pos_host[:bucket])

        # Token ids: real prompt + padding zeros
        ids_real = seq.prompt_ids
        ids_pad = [0] * (bucket - n)
        g.ids_host[:bucket] = torch.tensor(ids_real + ids_pad, dtype=torch.long)
        g.ids.copy_(g.ids_host[:bucket])

        # Context lengths for attention
        g.context_lens.copy_(
            torch.tensor([n], dtype=torch.int32, device=self.device)
        )

        # Update the gather index so the graph extracts hidden[n-1]
        # (the last REAL token) instead of hidden[S-1] (padding).
        if g.out_idx is not None:
            g.out_idx.copy_(
                torch.tensor([n - 1], dtype=torch.long, device=self.device)
            )

        # Refresh static write_prefill mapping for the real slot
        if hasattr(self.cache, 'bind_graph_write_prefill'):
            self.cache.bind_graph_write_prefill(slot, n)

        # Recurrent state: bind graph to real slot for recurrent layers
        if g.slot_idx is not None and self.has_recurrent:
            g.slot_idx.copy_(
                torch.tensor([slot], dtype=torch.long, device=self.device)
            )
            self.lin_cache.bind_graph(g.slot_idx)

    def _capture(self, bucket: int) -> _CapturedPrefillGraph | None:
        """Warm up + capture a CUDA graph for one prompt-length bucket."""
        S = bucket
        scratch = self._scratch(S)

        # Allocate static buffers
        ids = torch.zeros(1, S, dtype=torch.long, device=self.device)
        pos = torch.zeros(1, S, dtype=torch.long, device=self.device)
        pin = torch.cuda.is_available()
        ids_host = torch.empty(S, dtype=torch.long, pin_memory=pin)
        pos_host = torch.empty(S, dtype=torch.long, pin_memory=pin)

        # Slot mapping: all point to scratch during capture
        slot_mapping = torch.full(
            (S,), scratch, dtype=torch.int32, device=self.device
        )
        context_lens = torch.tensor([S], dtype=torch.int32, device=self.device)

        # Ensure cache capacity for scratch
        self.cache.ensure_capacity([scratch], [S])

        # ForwardContext for capture
        ctx = ForwardContext(
            is_prefill=True,
            kv_cache=self.cache,
            lin_cache=self.lin_cache,
            slots=[scratch],
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            max_context_len=S,
        )

        # Bind lin_cache to scratch for recurrent layers (static index for graph capture)
        slot_idx = slot_idx_host = None
        if self.has_recurrent and self.lin_cache is not None:
            self.lin_cache.enable_static_buffers(self.cache.num_slots)
            slot_idx = torch.full((1,), scratch, dtype=torch.long, device=self.device)
            self.lin_cache.bind_graph(slot_idx)

        # Enable static write buffers for graph-captured prefill
        if hasattr(self.cache, 'enable_graph_prefill'):
            self.cache.enable_graph_prefill(bucket)
            self.cache.bind_graph_write_prefill(scratch, bucket)

        # Static index for extracting the last REAL token's hidden state at
        # replay time.  During capture it points to S-1 (== last position,
        # which is real during capture).  _fill_inputs updates it to n-1.
        # torch.gather requires the index tensor to have the same number of
        # dims as the input (hidden[0] is [S, hidden_size], 2D) -- expand the
        # scalar row index across hidden_size so gather(0, out_idx) is valid.
        hidden_size = self.model.config.hidden_size
        out_idx = torch.tensor([S - 1], dtype=torch.long, device=self.device)
        # .contiguous(): out_idx is mutated in-place via .copy_() every replay
        # (_fill_inputs) -- an .expand()'d view aliases one real element across
        # all hidden_size columns via stride 0, which in-place writes should not
        # target. Materialize real storage instead.
        out_idx = out_idx.unsqueeze(-1).expand(-1, hidden_size).contiguous()

        # Warmup (3 iters, standard CUDA graph practice)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                with torch.inference_mode():
                    hidden = self.model(ids, pos, ctx)
                    self.model.compute_logits(hidden[0].gather(0, out_idx))
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        # Capture
        logits_buf = None
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            hidden = self.model(ids, pos, ctx)
            logits_buf = self.model.compute_logits(
                hidden[0].gather(0, out_idx)
            ).clone()

        return _CapturedPrefillGraph(
            graph=graph,
            ids=ids,
            pos=pos,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            logits=logits_buf,
            seq_len_bucket=bucket,
            ids_host=ids_host,
            pos_host=pos_host,
            slot_idx=slot_idx,
            out_idx=out_idx,
        )
