# SPDX-License-Identifier: MIT
"""Contiguous KV cache (v0) — one fp16 [L, B, Hkv, max_len, D] store per model.

This is the simple, correct cache used by the standalone `ModelRunner`: prefill
writes the whole sequence, decode appends one token. The engine uses
`superl8serve.engine.kv_cache.PagedKVCache` instead (int8, block-table paged); the
model code is unchanged either way because it only touches `write_prefill` /
`append_decode`.

`length` is the number of valid tokens; the runner calls `advance(n)` once per step
after all layers have written (all layers share one length).
"""

from __future__ import annotations

import torch


class KVCache:
    def __init__(
        self, num_layers, batch, num_kv_heads, max_len, head_dim, *, device, dtype=torch.float16
    ):
        shape = (num_layers, batch, num_kv_heads, max_len, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.length = 0

    def reset(self):
        self.length = 0

    def write_prefill(
        self, layer: int, k: torch.Tensor, v: torch.Tensor, *, slot=None, start: int = 0, **kwargs
    ):
        """k, v: [B, Hkv, S, D] at positions [0, S). `slot` is ignored (the simple
        runner cache is single-batch); the engine's PagedKVCache uses it."""
        s = k.shape[2]
        self.k[layer, :, :, start:s] = k[:, :, start:, :]
        self.v[layer, :, :, start:s] = v[:, :, start:, :]

    def append_decode(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        """k, v: [B, Hkv, 1, D] at position `length`. Returns the [B,Hkv,length+1,D]
        cache slice to attend against."""
        p = self.length
        self.k[layer, :, :, p : p + 1] = k
        self.v[layer, :, :, p : p + 1] = v
        return self.k[layer, :, :, : p + 1], self.v[layer, :, :, : p + 1]

    def advance(self, n: int = 1):
        self.length += n


class RecurrentStateCache:
    """Per-**slot** decode-time state for linear-attention layers (Gated DeltaNet,
    lightning attention, LFM2 short-conv): the recurrent state `S` and, for the
    causal depthwise conv, the trailing `kernel-1` raw window feeding it. Only
    linear/conv/lightning layers touch this (full/sliding/latent layers use
    `KVCache` instead). Prefill starts each layer from scratch (`get_*` returns
    None); decode carries the previous call's state back in so the recurrence
    doesn't reset every step.

    **Continuous batching**: a single global (layer-keyed) state cannot serve more
    than one concurrent sequence — the engine batches every running sequence into
    one `[B, 1, H]` decode call, so a batch-1 state cached from one sequence would
    either collide with or (for the conv tail) shape-mismatch the batch-B decode
    input. State is therefore keyed by ``(slot, layer)``: the runner calls
    ``bind(slots)`` with the ordered slot of each batch row before a forward, and
    ``get_*`` gathers those slots' rows into a ``[B, ...]`` tensor while ``set_*``
    scatters the updated ``[B, ...]`` back per slot. The single-batch
    ``ModelRunner`` never binds, so it transparently uses one implicit slot 0."""

    def __init__(self):
        self._state: dict[tuple[int, int], torch.Tensor] = {}
        self._conv_tail: dict[tuple[int, int], torch.Tensor] = {}
        self._slots: list[int] | None = None
        # -- static (CUDA-graph-capturable) mode --------------------------------
        # When enabled, per-slot state lives in FIXED-ADDRESS pre-allocated buffers
        # `[num_slots, *row_shape]` (one per layer, lazily sized from the first
        # write) instead of a python dict of freshly-allocated tensors. get/set
        # gather/scatter into those buffers IN PLACE, so the recurrent decode step
        # reads/writes the SAME memory every step and can be captured into a CUDA
        # graph (engine/cuda_graph.py). Eager callers (prefill, eager-decode
        # fallback) index the buffers by the python slot list; the graphed decode
        # replay indexes them by a persistent device tensor (`bind_graph`) that maps
        # each batch row to its slot -- refreshed in place before every replay, the
        # way the paged-KV path refreshes its slot_mapping. Both share ONE store, so
        # prefill state is visible to graphed decode with no cross-buffer copy.
        self._static = False
        self._num_slots = 0
        self._state_buf: dict[int, torch.Tensor] = {}
        self._conv_buf: dict[int, torch.Tensor] = {}
        self._row_idx: torch.Tensor | None = None  # device Long[B] row->slot, or None
        # -- spec-decode verify-trajectory capture (task d) ---------------------
        # During a verify forward the recurrence runs L=k+1 tokens in ONE call, but
        # only the accepted prefix (a per-row count) may commit. Instead of the old
        # snapshot→verify→restore→replay-accepted dance (which re-streamed the model
        # weights to re-advance the state), the DeltaNet layer records the PER-TOKEN
        # state (and conv tail) trajectory here; after acceptance the runner commits,
        # per row, exactly the state after that row's last accepted token — no
        # re-decode. `_vcap` gates recording; `_vtraj[layer]` holds (state[B,L,...],
        # conv[B,L,...]) for the bound slots.
        self._vcap = False
        self._vtraj: dict[int, tuple[torch.Tensor, torch.Tensor | None]] = {}

    def enable_static_buffers(self, num_slots: int) -> None:
        """Switch to fixed-address per-slot buffers (graph-capturable). Must be
        called BEFORE the first prefill so prefill and graphed decode share one
        store. Idempotent."""
        self._static = True
        self._num_slots = num_slots

    def bind(self, slots: list[int] | None) -> None:
        """Set the ordered slot list for the next (eager) forward — row ``i`` of
        every get/set maps to ``slots[i]``. ``None`` (the default) selects the
        single-batch path (one implicit slot 0). Clears any graph row-index so the
        eager python-list path is used."""
        self._slots = list(slots) if slots is not None else None
        self._row_idx = None

    def bind_graph(self, row_idx: torch.Tensor) -> None:
        """Static mode only: bind the persistent device row->slot index tensor the
        captured graph gathers/scatters through. `row_idx[i]` is the cache slot of
        batch row `i` (pad rows point at the scratch slot). The SAME tensor object
        must be reused across replays (its contents are refreshed in place)."""
        self._row_idx = row_idx

    def _active(self) -> list[int]:
        return self._slots if self._slots is not None else [0]

    def _sget(self, buf: dict[int, torch.Tensor], layer_idx: int) -> torch.Tensor | None:
        t = buf.get(layer_idx)
        if t is None:
            return None  # not yet written this run -> recurrence starts at zero
        if self._row_idx is not None:
            return t.index_select(0, self._row_idx)  # graph: gather by device index
        return t[self._active()]  # eager: gather by python slot list (a copy)

    def _sset(self, buf: dict[int, torch.Tensor], layer_idx: int, value: torch.Tensor) -> None:
        t = buf.get(layer_idx)
        if t is None:
            # Lazy alloc, sized from the per-row shape. Only ever happens EAGERLY
            # (prefill / capture warmup), never inside a captured graph region.
            t = value.new_zeros((self._num_slots, *value.shape[1:]))
            buf[layer_idx] = t
        if self._row_idx is not None:
            t.index_copy_(0, self._row_idx, value)  # graph: scatter in place
        else:
            t[self._active()] = value  # eager: scatter in place

    @torch.inference_mode()
    def clear_slot(self, slot: int) -> None:
        """Drop every layer's state/tail for one slot — called when a slot is
        (re)allocated to a fresh sequence so leftover state can never leak in.

        Static buffers are created inside inference mode (CUDA-graph capture), so
        their tensors are inference tensors; the in-place ``zero_`` below would
        otherwise raise 'Inplace update to inference tensor outside InferenceMode
        is not allowed' when the slot is cleared from a normal autograd context.
        """
        if self._static:
            for buf in (self._state_buf, self._conv_buf):
                for t in buf.values():
                    t[slot].zero_()
            return
        for store in (self._state, self._conv_tail):
            for key in [k for k in store if k[0] == slot]:
                del store[key]

    def _gather(self, store, layer_idx):
        rows = [store.get((s, layer_idx)) for s in self._active()]
        present = next((r for r in rows if r is not None), None)
        if present is None:
            return None  # fresh (prefill / first decode) — recurrence starts at zero
        rows = [r if r is not None else torch.zeros_like(present) for r in rows]
        return torch.cat(rows, dim=0)

    def _scatter(self, store, layer_idx, value):
        slots = self._active()
        for i, s in enumerate(slots):
            store[(s, layer_idx)] = value[i : i + 1]

    def get_state(self, layer_idx: int) -> torch.Tensor | None:
        if self._static:
            return self._sget(self._state_buf, layer_idx)
        return self._gather(self._state, layer_idx)

    def set_state(self, layer_idx: int, state: torch.Tensor):
        if self._static:
            return self._sset(self._state_buf, layer_idx, state)
        self._scatter(self._state, layer_idx, state)

    def get_conv_tail(self, layer_idx: int) -> torch.Tensor | None:
        if self._static:
            return self._sget(self._conv_buf, layer_idx)
        return self._gather(self._conv_tail, layer_idx)

    def set_conv_tail(self, layer_idx: int, tail: torch.Tensor):
        if self._static:
            return self._sset(self._conv_buf, layer_idx, tail)
        self._scatter(self._conv_tail, layer_idx, tail)

    def verify_trajectory_nbytes(self, batch_rows: int, tokens: int) -> int:
        """Estimate storage for a recurrent verify trajectory.

        Verification records every token's state and convolution tail for every
        active row. Static cache buffers include all scheduler slots, so count one
        row from each layer buffer rather than the full preallocated tensor.
        """
        if batch_rows < 1 or tokens < 1:
            return 0

        def row_nbytes(t: torch.Tensor) -> int:
            return t[0].numel() * t.element_size()

        if self._static:
            per_row = sum(row_nbytes(t) for t in self._state_buf.values())
            per_row += sum(row_nbytes(t) for t in self._conv_buf.values())
        else:
            # Dict values are stored one row per (slot, layer). Count each layer
            # once, then scale by the prospective verify batch below.
            state_rows = {layer: t for (_, layer), t in self._state.items()}
            conv_rows = {layer: t for (_, layer), t in self._conv_tail.items()}
            per_row = sum(row_nbytes(t) for t in state_rows.values())
            per_row += sum(row_nbytes(t) for t in conv_rows.values())
        return per_row * batch_rows * tokens

    def snapshot(self) -> dict:
        """Deep-copy every layer's recurrent state + conv tail for the CURRENTLY
        BOUND slots (``bind(slots)`` must have been called). Returns an opaque token
        for :meth:`restore`.

        This is the recurrent analogue of the paged-KV accepted-token canonicalizer
        (#235): speculative decode's verify forward advances DeltaNet / lightning /
        short-conv state by EVERY drafted token, but only the accepted prefix may
        commit. The runner snapshots the post-base-forward state, lets verify run,
        then restores this snapshot and replays only the accepted tokens (through the
        normal decode path) to reach the correct committed state. The clone is
        essential — ``get_state`` may return a view into the live buffer/dict, so a
        bare reference would alias the very state verify is about to overwrite."""
        snap: dict = {"state": {}, "conv": {}}
        if self._static:
            slots = self._active()
            for lidx, t in self._state_buf.items():
                snap["state"][lidx] = t[slots].clone()
            for lidx, t in self._conv_buf.items():
                snap["conv"][lidx] = t[slots].clone()
        else:
            for lidx in {k[1] for k in self._state}:
                g = self._gather(self._state, lidx)
                if g is not None:
                    snap["state"][lidx] = g.clone()
            for lidx in {k[1] for k in self._conv_tail}:
                g = self._gather(self._conv_tail, lidx)
                if g is not None:
                    snap["conv"][lidx] = g.clone()
        return snap

    def restore(self, snap: dict) -> None:
        """Scatter a :meth:`snapshot` back onto the currently-bound slots, discarding
        any state mutation made since it was taken (the verify forward's)."""
        for lidx, v in snap["state"].items():
            self.set_state(lidx, v)
        for lidx, v in snap["conv"].items():
            self.set_conv_tail(lidx, v)

    # -- spec-decode verify-trajectory capture (task d) -----------------------

    def begin_verify_capture(self) -> None:
        """Arm per-token trajectory recording for the next verify forward. The
        DeltaNet / lightning layers check :attr:`capturing_verify` and record their
        full state (and conv tail) trajectory via :meth:`record_verify_traj`."""
        self._vcap = True
        self._vtraj = {}

    @property
    def capturing_verify(self) -> bool:
        return self._vcap

    def record_verify_traj(
        self, layer_idx: int, state_traj: torch.Tensor | None, conv_traj: torch.Tensor | None
    ) -> None:
        """Store one recurrent layer's per-token trajectory over the verify tokens.
        ``state_traj`` is [B, L, ...] = the recurrent state AFTER each of the L verify
        tokens (row-aligned to the bound slots), or None for a state-free layer;
        ``conv_traj`` is the matching conv tail [B, L, ...] (or None if the layer has
        no conv). At least one is non-None."""
        self._vtraj[layer_idx] = (state_traj, conv_traj)

    @staticmethod
    def _pick(traj: torch.Tensor, row_last: list[int]) -> torch.Tensor:
        """Gather ``traj[b, row_last[b]]`` -> [B, ...] on ``traj``'s device, for the
        first ``len(row_last)`` rows. When the verify forward is CUDA-graph-captured
        the trajectory is recorded over the padded batch bucket (``traj.shape[0]`` ==
        Bmax), but only the ``B`` REAL rows are committed — index the leading B rows so
        a padded bucket commits exactly its real sequences (Bmax==B in the eager path,
        so this is a no-op there)."""
        n = len(row_last)
        rows = torch.arange(n, device=traj.device)
        idx = torch.tensor(row_last, device=traj.device)
        return traj[rows, idx]

    def commit_verify(self, row_last: list[int]) -> None:
        """Commit, per bound row ``b``, the recurrent state (and/or conv tail) AFTER
        that row's last accepted verify token ``row_last[b]`` (a 0-based index into the
        trajectory). Scatters into the per-slot committed store exactly as a normal
        decode step's ``set_state`` / ``set_conv_tail`` would — reaching the correct
        committed state with no re-decode. Clears the capture afterward."""
        for lidx, (state_traj, conv_traj) in self._vtraj.items():
            if state_traj is not None:
                self.set_state(lidx, self._pick(state_traj, row_last))
            if conv_traj is not None:
                self.set_conv_tail(lidx, self._pick(conv_traj, row_last))
        self.end_verify_capture()

    def verify_captured_layers(self) -> int:
        """Number of recurrent layers that recorded a trajectory this verify."""
        return len(self._vtraj)

    def end_verify_capture(self) -> None:
        self._vcap = False
        self._vtraj = {}

    def reset(self):
        self._state.clear()
        self._conv_tail.clear()
        self._slots = None
        self._row_idx = None
        self._vcap = False
        self._vtraj = {}
        if self._static:
            for buf in (self._state_buf, self._conv_buf):
                for t in buf.values():
                    t.zero_()


class MLALatentCache:
    """Latent-KV cache for MLA (DeepSeek): stores only the compressed c_KV + k_pe
    per token (`latent_dim = kv_lora_rank + qk_rope_head_dim`), one shared MQA-style
    "head", instead of full per-head K/V. `MLAAttention` up-projects the read-back
    latent through kv_b_proj to get per-head K/V at decode time.

    Engine-compatible: provides ``alloc``/``free`` slot management (simple free-list
    over the fixed ``batch`` dimension), plus no-op stubs for the scheduler's
    prefix-cache and capacity hooks that only ``PagedKVCache`` implements.
    """

    def __init__(self, num_layers, batch, latent_dim, max_len, *, device, dtype=torch.float16):
        self.kv = torch.zeros((num_layers, batch, max_len, latent_dim), device=device, dtype=dtype)
        self.length = 0
        self.num_slots = batch
        self.max_len = max_len
        self._free_slots = list(range(batch))

    # -- slot management (engine / scheduler interface) -----------------------

    def alloc(self) -> int:
        if not self._free_slots:
            raise RuntimeError("MLALatentCache: no free slots")
        return self._free_slots.pop()

    def free(self, slot: int):
        if slot not in self._free_slots:
            self._free_slots.append(slot)

    def has_free_slot(self) -> bool:
        return len(self._free_slots) > 0

    def has_free_block(self) -> bool:
        # Pre-allocated fixed per-slot region — no shared block pool, so slot
        # availability alone bounds admission; blocks are never the scarce resource.
        return True

    def ensure_capacity(self, slots: list[int], lengths: list[int]):
        pass  # MLALatentCache is pre-allocated; capacity is fixed.

    def store_prefix(self, token_ids: list[int], slot: int):
        pass  # Prefix cache not applicable to latent storage.

    def lookup_prefix(self, token_ids: list[int]) -> tuple[int, list[int]]:
        return 0, []

    def share_blocks(self, slot: int, blocks: list[int]):
        pass  # No block sharing for latent cache.

    # -- latent-KV read/write ------------------------------------------------

    def reset(self):
        self.length = 0

    def write_prefill(self, layer: int, latent: torch.Tensor, *, slot: int | None = None, **kwargs):
        """latent: [B, S, D] at positions [0, S). When *slot* is given, B must be 1
        and only that slot row is written (engine path — one seq at a time)."""
        s = latent.shape[1]
        if slot is not None:
            self.kv[layer, slot, :s] = latent[0]
        else:
            self.kv[layer, : latent.shape[0], :s] = latent

    def append_decode(
        self,
        layer: int,
        latent: torch.Tensor,
        *,
        slot: int | None = None,
        slots: list[int] | None = None,
    ) -> torch.Tensor:
        """latent: [B, 1, D] at position ``length``.

        *slot*  (int, engine prefill): B=1, write/return that slot row only.
        *slots* (list[int], engine decode): write each batch row to its slot
                 and return the per-slot cached history.
        Neither (ModelRunner path): write/return rows 0..B-1.
        """
        p = self.length
        if slots is not None:
            B = latent.shape[0]
            for i in range(B):
                self.kv[layer, slots[i], p : p + 1] = latent[i : i + 1]
            return torch.stack([self.kv[layer, s, : p + 1] for s in slots], dim=0)
        if slot is not None:
            self.kv[layer, slot, p : p + 1] = latent[0:1]
            return self.kv[layer, slot : slot + 1, : p + 1]
        B = latent.shape[0]
        self.kv[layer, :B, p : p + 1] = latent
        return self.kv[layer, :B, : p + 1]

    def advance(self, n: int = 1):
        self.length += n


class MTPKVCache:
    """Per-slot fp16 KV store for the MTP speculative-draft head's OWN attention.

    The MTP draft head is a separate transformer layer from the backbone, so it needs
    its OWN K/V over the committed context — the backbone's paged cache holds the
    backbone layers' K/V, never the head's. Mirrors qengine's inline nextn head, which
    keeps a dedicated ``kv_k``/``kv_v`` (Haru-neo/qengine, Apache-2.0). Kept in fp16
    (not the backbone's int8 paged store) because the draft is a small dense
    single-query attention and the int8 paged-decode kernel is head-dim {32,64,128}
    only (this family is 256); fp16 also keeps draft quality (accept rate) up.

    Stored per slot as ``k``/``v`` [L, nkv, hd] where MTP-position ``i`` (0-based) holds
    the K/V projected from ``(h_main[i], emb(token[i+1]))`` at RoPE position ``i``. The
    store is ALWAYS authoritative (only committed tokens are ever
    appended — the base token is always accepted and drafts are re-projected from the
    verified hidden states), so no speculative rollback is needed.
    """

    def __init__(self):
        self._k: dict[int, torch.Tensor] = {}
        self._v: dict[int, torch.Tensor] = {}

    def clear_slot(self, slot: int) -> None:
        self._k.pop(slot, None)
        self._v.pop(slot, None)

    def length(self, slot: int) -> int:
        t = self._k.get(slot)
        return 0 if t is None else t.shape[0]

    def reset_slot(self, slot: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Replace a slot's whole prefix (prefill priming). k, v: [L, nkv, hd]."""
        self._k[slot] = k.contiguous()
        self._v[slot] = v.contiguous()

    def append(self, slot: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append ``m`` positions. k, v: [m, nkv, hd]."""
        if slot not in self._k:
            self._k[slot] = k.contiguous()
            self._v[slot] = v.contiguous()
        else:
            self._k[slot] = torch.cat([self._k[slot], k], dim=0)
            self._v[slot] = torch.cat([self._v[slot], v], dim=0)

    def truncate(self, slot: int, length: int) -> None:
        """Keep only the first ``length`` positions (discard speculative tail)."""
        if slot in self._k and self._k[slot].shape[0] > length:
            self._k[slot] = self._k[slot][:length].contiguous()
            self._v[slot] = self._v[slot][:length].contiguous()

    def read(self, slot: int):
        """Return (k, v) as [1, L, nkv, hd] ready to concat with the current token's
        K/V, or (None, None) if the slot has no primed prefix."""
        t = self._k.get(slot)
        if t is None:
            return None, None
        return t.unsqueeze(0), self._v[slot].unsqueeze(0)
