# SPDX-License-Identifier: MIT
"""PagedKVCache — block-table paged, int8 quantize-on-write KV store for
continuous batching.

Physical storage is one pool of `block_size`-token int8 blocks per layer, shared
across every sequence; each sequence gets a "slot" (a block-table row) grown one
block at a time as its length crosses a block boundary, so a short sequence only
ever pins the few blocks it actually uses instead of reserving a `max_len`-sized
region up front (the old `BatchedKVCache` this replaces). A finished sequence's
blocks return to the free pool immediately (`free`) for the next request to
reuse.

Every new token is quantized straight to int8 on write (`superl8.quantize_kv_write_paged`,
per-token RTN scale) — no fp16 K/V is ever kept resident. Decode reads the WHOLE
ragged running batch in one `superl8.attn_paged_decode_cached` launch (one block table
+ one context-lens tensor for every row), replacing the old per-slot Python loop
over `attn_int8_decode`.

Sliding-window layers (Gemma3) are the one gap `attn_paged_decode_cached` doesn't
cover (no window parameter yet): `read_dense` reconstructs a small dequantized fp16
window slice for the existing `attn_int8_decode` fallback in that case — see
`GQAAttention._decode_batched`.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Callable

import torch

import superl8
from superl8.quant.rotation import rotate_last

logger = logging.getLogger(__name__)


# ── Eviction configuration ────────────────────────────────────────────────────


@dataclass
class EvictionConfig:
    """Configuration for KV-cache eviction policies (SnapKV + Ada-KV + DuoAttention).

    When *enabled*, eviction runs prompt-time (once after prefill, before decode)
    to free HBM by keeping only the *kv_budget* most important token positions per
    sequence.  Gated behind *recall_threshold*: if the cosine similarity between the
    full-KV and evicted-KV decode logits falls below this value, eviction is skipped
    for that step (the "recall gate").

    Policy-specific knobs:
      *snapkv* — attention-pooled selection.  ``kv_budget`` tokens are kept across
      all layers; which positions depends on the attention score of the last query
      token against every cached K position.

      *adakv* — per-layer / per-head budget following a pyramid schedule: early
      layers get ``adakv_min_budget`` tokens, late layers get ``adakv_max_budget``,
      with a linear ramp in between.  ``kv_budget`` becomes the total budget summed
      across layers.

      *duoattn* — a fraction of heads (``duoattn_streaming_heads``) keep only a
      fixed-size streaming window of recent tokens; the rest use SnapKV with the
      per-head budget.  ``duoattn_window`` sets the streaming window size.
    """

    enabled: bool = False
    policy: str = "snapkv"  # "snapkv" | "adakv" | "duoattn"
    kv_budget: int = 512
    recall_threshold: float = 0.99
    measure_hbm: bool = True
    adakv_min_budget: int = 128
    adakv_max_budget: int = 1024
    duoattn_streaming_heads: float = 0.25
    duoattn_window: int = 128


# ── KV eviction strategy engine ────────────────────────────────────────────────


class KVEviction:
    """Prompt-time KV-cache eviction: SnapKV + Ada-KV budgets + DuoAttention.

    Usage::

        # After prefill completes for a slot:
        keep = KVEviction.snapkv_selection(q_last, k_full, cfg.kv_budget, ...)
        metrics = KVEviction.compact(cache, slot, keep)

    All methods are static/classmethods usable without instantiation.
    Compact operates on the PagedKVCache's internal block store.
    """

    # -- HBM accounting --------------------------------------------------------

    @staticmethod
    def block_bytes(
        num_kv_heads: int,
        block_size: int,
        head_dim: int,
        v_quant: str = "int8",
        lloyd_block: int = 128,
    ) -> int:
        """HBM bytes consumed by one block (K+V + scales) per layer.

        ``v_quant="lloydmax3"`` is the K8V3 layout: K stays int8 (+ fp32 per-token
        scale), V is 3-bit Lloyd-Max (packed codes + per-``lloyd_block`` fp32 norm
        per token). Everything else is the int8 layout (K+V int8 + fp32 per-token
        scales)."""
        n = num_kv_heads * block_size
        if v_quant == "lloydmax3":
            k = n * (head_dim + 4)  # int8 + fp32 per-token scale
            nb = head_dim // lloyd_block
            v = n * (head_dim * 3 // 8 + nb * 4)  # packed 3-bit codes + fp32 norms
            return k + v
        kv = 2 * n * head_dim  # int8
        scales = 2 * n * 4  # fp32
        return kv + scales

    # -- SnapKV: attention-pooled selection -----------------------------------

    @staticmethod
    def snapkv_selection(
        q_last: torch.Tensor,
        k_full: torch.Tensor,
        budget: int,
        num_heads: int,
        num_kv_heads: int,
    ) -> list[int]:
        """Select top-*budget* KV positions by attention score of the last query
        against every cached K position.

        Args:
            q_last: last query token  ``[num_heads, head_dim]`` or ``[1, num_heads, 1, head_dim]``.
            k_full: all K states ``[num_kv_heads, seq_len, head_dim]`` or ``[1, num_kv_heads, seq_len, head_dim]``.
            budget: max tokens to keep.
            num_heads, num_kv_heads: GQA dimensions.

        Returns:
            Sorted list of position indices to keep.
        """
        seq_len = k_full.shape[-2]
        if seq_len <= budget:
            return list(range(seq_len))

        # Normalise shapes
        q = q_last.squeeze().reshape(num_heads, -1)  # [num_heads, head_dim]
        k = k_full.squeeze(0) if k_full.dim() == 4 else k_full  # [num_kv_heads, seq_len, hd]
        hd = q.shape[-1]

        # GQA repeat: [num_kv_heads, seq_len, hd] -> [num_heads, seq_len, hd]
        ng = num_heads // num_kv_heads
        if ng > 1:
            k = k.unsqueeze(1).expand(-1, ng, -1, -1).reshape(num_heads, seq_len, hd)

        # Attention scores: Q @ K^T / sqrt(d)  -> softmax -> average over heads
        scores = torch.einsum("hd,hsd->hs", q, k)  # [num_heads, seq_len]
        scores = (scores / math.sqrt(hd)).softmax(dim=-1).mean(dim=0)  # [seq_len]

        _, top_idx = torch.topk(scores, min(budget, seq_len))
        return sorted(top_idx.tolist())

    # -- Ada-KV / PyramidKV: per-layer budgets --------------------------------

    @staticmethod
    def adakv_budgets(
        num_layers: int,
        total_budget: int,
        min_budget: int = 128,
        max_budget: int = 1024,
    ) -> list[int]:
        """Pyramid schedule: early layers get few tokens, later layers many.

        Returns a list of ``num_layers`` per-layer token budgets that sum to
        *total_budget* (approximately).
        """
        if num_layers <= 1:
            return [total_budget]

        raw = [int(min_budget + i / (num_layers - 1) * (max_budget - min_budget)) for i in range(num_layers)]
        scale = total_budget / sum(raw)
        return [max(1, int(b * scale)) for b in raw]

    # -- DuoAttention: streaming-head split -----------------------------------

    @staticmethod
    def duoattn_split(
        num_heads: int,
        streaming_fraction: float = 0.25,
    ) -> tuple[list[int], list[int]]:
        """Return ``(streaming_head_indices, snap_head_indices)``."""
        n = max(1, int(num_heads * streaming_fraction))
        return list(range(n)), list(range(n, num_heads))

    # -- Block compaction -----------------------------------------------------

    @classmethod
    def compact(
        cls,
        cache: "PagedKVCache",
        slot: int,
        keep_positions: list[int],
    ) -> dict:
        """Compact a slot's KV blocks to keep only *keep_positions*.

        Dequantizes each kept token from its old block, re-quantizes into the
        minimum number of new blocks, frees the old blocks, and updates the
        slot's block table.

        Returns a metrics dict:

            bytes_freed     total HBM freed across all layers
            blocks_freed    number of physical blocks freed
            blocks_after    number of physical blocks after compaction
            tokens_kept     number of token positions preserved
        """
        old_blocks = list(cache._slot_blocks[slot])
        if not old_blocks or not keep_positions:
            return {"bytes_freed": 0, "blocks_freed": 0, "blocks_after": len(old_blocks), "tokens_kept": 0}
        if getattr(cache, "v_quant", "int8") != "int8":
            raise NotImplementedError(
                "KV eviction compaction is not implemented for the K8V3 (lloydmax3 V) "
                "cache format — the packed 3-bit V block store has no requant path yet."
            )

        num_layers = cache.k_cache.shape[0]
        num_kept = len(keep_positions)
        needed_blocks = (num_kept + cache.block_size - 1) // cache.block_size

        if needed_blocks >= len(old_blocks):
            return {
                "bytes_freed": 0,
                "blocks_freed": 0,
                "blocks_after": len(old_blocks),
                "tokens_kept": num_kept,
            }

        # Allocate new blocks from the shared free pool
        new_blocks: list[int] = []
        for _ in range(needed_blocks):
            if not cache._free_blocks:
                raise RuntimeError("PagedKVCache: no free blocks for eviction compaction")
            new_blocks.append(cache._free_blocks.pop())

        blocks_freed = len(old_blocks) - needed_blocks

        # Per-layer: dequantize kept tokens from old blocks and requantize into new
        for layer in range(num_layers):
            for new_blk_idx, new_blk in enumerate(new_blocks):
                start = new_blk_idx * cache.block_size
                end = min(start + cache.block_size, num_kept)
                positions_in_this_block = keep_positions[start:end]
                n_local = len(positions_in_this_block)
                if n_local == 0:
                    continue

                k_chunks, v_chunks, mappings = [], [], []
                for local_off, global_pos in enumerate(positions_in_this_block):
                    old_blk = old_blocks[global_pos // cache.block_size]
                    old_off = global_pos % cache.block_size

                    # Dequantize: int8 * fp32 scale → fp16
                    k_fp16 = (
                        cache.k_cache[layer, old_blk, :, old_off, :].float()
                        * cache.k_scale[layer, old_blk, :, old_off].unsqueeze(-1)
                    ).to(torch.float16)
                    v_fp16 = (
                        cache.v_cache[layer, old_blk, :, old_off, :].float()
                        * cache.v_scale[layer, old_blk, :, old_off].unsqueeze(-1)
                    ).to(torch.float16)

                    # Undo the write-time Hadamard rotation (involution) so that
                    # quantize_kv_write_paged below re-applies it — without this
                    # step the stored K would be double-rotated after a compact
                    # cycle and read_dense would produce wrong output.
                    k_fp16 = rotate_last(k_fp16)

                    k_chunks.append(k_fp16)
                    v_chunks.append(v_fp16)
                    mappings.append(new_blk * cache.block_size + local_off)

                k_batch = torch.stack(k_chunks, dim=0)  # [T, Hkv, D]
                v_batch = torch.stack(v_chunks, dim=0)  # [T, Hkv, D]
                mapping_tensor = torch.tensor(mappings, dtype=torch.int32, device=cache.device)

                superl8.quantize_kv_write_paged(
                    k_batch.contiguous(),
                    v_batch.contiguous(),
                    cache.k_cache[layer],
                    cache.k_scale[layer],
                    cache.v_cache[layer],
                    cache.v_scale[layer],
                    mapping_tensor,
                )

        # Free old blocks
        for blk in old_blocks:
            cache._block_refcount.pop(blk, None)
            cache._free_blocks.append(blk)

        cache._slot_blocks[slot] = new_blocks

        bb = cls.block_bytes(cache.k_cache.shape[2], cache.block_size, cache.k_cache.shape[-1])
        bytes_freed = blocks_freed * num_layers * bb

        return {
            "bytes_freed": bytes_freed,
            "blocks_freed": blocks_freed,
            "blocks_after": len(new_blocks),
            "tokens_kept": num_kept,
        }

    # -- Recall gate ----------------------------------------------------------

    @classmethod
    def recall_check(
        cls,
        logits_full: torch.Tensor,
        logits_evicted: torch.Tensor,
        threshold: float = 0.99,
    ) -> dict:
        """Cos-sim recall gate: if similarity < threshold, return failed."""
        a, b = logits_full.float().flatten(), logits_evicted.float().flatten()
        cos = (a @ b / (a.norm() * b.norm() + 1e-12)).item()
        return {"cos_sim": cos, "passed": cos >= threshold, "threshold": threshold}

    # -- Top-level eviction driver --------------------------------------------

    @classmethod
    def evict(
        cls,
        cache: "PagedKVCache",
        slot: int,
        seq_len: int,
        *,
        eviction_config: EvictionConfig,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_last: torch.Tensor | None = None,
        k_full: torch.Tensor | None = None,
        keep_mask: list[int] | None = None,
        logits_full: torch.Tensor | None = None,
        logits_fn: Callable[[], torch.Tensor] | None = None,
    ) -> dict:
        """Run eviction for *slot* after prefill.  Returns a dict with all metrics.

        Parameters:
            cache: the PagedKVCache instance.
            slot, seq_len: the slot to evict and its prefill length.
            eviction_config: policy knobs.
            num_heads, num_kv_heads, head_dim: model attention dimensions.
            q_last: ``[num_heads, head_dim]`` last query token for SnapKV.
            k_full: ``[num_kv_heads, seq_len, head_dim]`` full K cache for SnapKV.
            keep_mask: explicit list of positions to keep (bypasses selection).
            logits_full: pre-eviction logits for the recall gate.
            logits_fn: callable that returns post-eviction logits for the recall
                       gate (lazy — only called if recall is enabled).

        Returns:
            metrics dict with at minimum:
                evicted     bool
                strategy    str
                metrics     dict from compact()
                recall      dict from recall_check() or None
        """
        if not eviction_config.enabled:
            return {"evicted": False, "strategy": None, "metrics": None, "recall": None}

        # 1. Determine which positions to keep
        if keep_mask is not None:
            positions = keep_mask
        elif eviction_config.policy == "snapkv" and q_last is not None and k_full is not None:
            positions = cls.snapkv_selection(q_last, k_full, eviction_config.kv_budget, num_heads, num_kv_heads)
        elif eviction_config.policy == "adakv":
            if seq_len <= eviction_config.kv_budget:
                positions = list(range(seq_len))
            else:
                # For Ada-KV, each layer gets its own budget
                budgets = cls.adakv_budgets(
                    cache.k_cache.shape[0],
                    eviction_config.kv_budget,
                    eviction_config.adakv_min_budget,
                    eviction_config.adakv_max_budget,
                )
                # Layer 0 budget determines positions
                per_layer_budget = budgets[0]
                if q_last is not None and k_full is not None:
                    positions = cls.snapkv_selection(q_last, k_full, per_layer_budget, num_heads, num_kv_heads)
                else:
                    positions = list(range(min(per_layer_budget, seq_len)))
        elif eviction_config.policy == "duoattn":
            streaming_heads, _ = cls.duoattn_split(num_heads, eviction_config.duoattn_streaming_heads)
            window = eviction_config.duoattn_window
            streaming_positions = list(range(max(0, seq_len - window), seq_len))
            if q_last is not None and k_full is not None:
                snap_budget = eviction_config.kv_budget
                snap_positions = cls.snapkv_selection(q_last, k_full, snap_budget, num_heads, num_kv_heads)
            else:
                snap_positions = list(range(min(eviction_config.kv_budget, seq_len)))
            positions = sorted(set(streaming_positions + snap_positions))
        else:
            # Fallback: keep first kv_budget tokens (no attention info available)
            positions = list(range(min(eviction_config.kv_budget, seq_len)))

        if len(positions) >= seq_len:
            return {"evicted": False, "strategy": eviction_config.policy, "metrics": None, "recall": None}

        # 2. Recall gate: check quality BEFORE eviction
        recall = None
        if eviction_config.recall_threshold < 1.0 and logits_full is not None and logits_fn is not None:
            logits_after = logits_fn()
            recall = cls.recall_check(logits_full, logits_after, eviction_config.recall_threshold)
            if not recall["passed"]:
                logger.warning("Recall gate BLOCKED eviction (cos=%.4f < %.4f)", recall["cos_sim"], recall["threshold"])
                return {"evicted": False, "strategy": eviction_config.policy, "metrics": None, "recall": recall}

        # 3. Compact
        metrics = cls.compact(cache, slot, positions)
        metrics["budget"] = eviction_config.kv_budget

        return {
            "evicted": True,
            "strategy": eviction_config.policy,
            "metrics": metrics,
            "recall": recall,
        }


class PagedKVCache:
    # Bound the int64 bit-expansion and fp32 centroid-lookup temporaries used by
    # the Python K8V3 fallback.  The final dense K/V pair is still required by
    # the current prefill attention operator; only one small token tile is
    # reconstructed at a time into those final output tensors.
    _DENSE_DEQUANT_CHUNK_TOKENS = 1024

    def __init__(
        self,
        num_layers,
        num_slots,
        num_kv_heads,
        max_len,
        head_dim,
        *,
        device,
        block_size: int = 16,
        num_blocks: int | None = None,
        max_prefix_entries: int | None = None,
        max_prefix_blocks: int | None = None,
        max_prefix_block_fraction: float = 0.5,
        kv_layers: list[int] | None = None,
        v_quant: str = "int8",
        lloyd_block: int = 128,
    ):
        self.block_size = block_size
        # `kv_layers` lists the REAL model layer indices that bear a KV cache (for a
        # qwen3_5 hybrid: the full-attention layers only — the DeltaNet layers carry
        # recurrent state, never KV). Storage is sized for exactly those layers, so a
        # 64-layer hybrid allocates a 16-row cache instead of a 4x-wasteful 64-row
        # one; `None` = identity (every layer bears KV). Every layer-indexed access
        # goes through `_layer_row`.
        self._kv_map = {layer: i for i, layer in enumerate(kv_layers)} if kv_layers is not None else None
        self.num_layers = num_layers if self._kv_map is None else len(kv_layers)
        # V storage layout: "int8" (K+V int8 + fp32 scales, the default) or
        # "lloydmax3" (K8V3 — K int8 + fp32 scales, V 3-bit Lloyd-Max codes +
        # per-`lloyd_block` fp32 norms + per-layer codebook). The lloydmax3 layout is
        # the K8V3 production cache for the 16 long-range full-attention layers.
        if v_quant not in ("int8", "lloydmax3"):
            raise ValueError(f"v_quant must be 'int8' or 'lloydmax3', got {v_quant!r}")
        self.v_quant = v_quant
        self.lloyd_block = lloyd_block
        self._k8v3_fused_enabled = os.environ.get("SUPERL8SERVE_K8V3_FUSED", "1") != "0"
        self._k8v3_fused_failed = False
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_blocks_per_seq = (max_len + block_size - 1) // block_size
        # Worst case (every slot at max_len) by default -- never starves the
        # scheduler's slot-based admission; pass num_blocks to over-subscribe
        # the pool for higher throughput once average length < max_len.
        self.num_blocks = num_blocks or num_slots * self.max_blocks_per_seq
        self.device = device
        shape = (self.num_layers, self.num_blocks, num_kv_heads, block_size, head_dim)
        self.k_cache = torch.zeros(shape, dtype=torch.int8, device=device)
        scale_shape = (self.num_layers, self.num_blocks, num_kv_heads, block_size)
        self.k_scale = torch.ones(scale_shape, dtype=torch.float32, device=device)
        if v_quant == "lloydmax3":
            # 3-bit packed V codes: `head_dim*3//8` bytes per token -> int32 words.
            packed_words = head_dim * 3 // 32
            num_blocks_hd = head_dim // lloyd_block
            self.v_packed = torch.zeros(
                (self.num_layers, self.num_blocks, num_kv_heads, block_size, packed_words),
                dtype=torch.int32,
                device=device,
            )
            self.v_norm = torch.zeros(
                (self.num_layers, self.num_blocks, num_kv_heads, block_size, num_blocks_hd),
                dtype=torch.float32,
                device=device,
            )
            from superl8.quant.lloydmax import gaussian_codebook

            cb = gaussian_codebook(3, lloyd_block, device)
            # Per-layer codebook rows (identical fixed Gaussian table for now; a
            # fitted per-layer codebook slots in here without a storage-format change).
            self.v_codebook = cb.repeat(self.num_layers, 1)
            self.v_cache = None
            self.v_scale = None
        else:
            self.v_cache = torch.zeros(shape, dtype=torch.int8, device=device)
            self.v_scale = torch.ones(scale_shape, dtype=torch.float32, device=device)
        self.num_slots = num_slots
        self.max_len = max_len
        self._free_blocks = list(range(self.num_blocks))
        self._free_slots = list(range(num_slots))
        self._slot_blocks: list[list[int]] = [[] for _ in range(num_slots)]
        # Static buffer for graph-captured prefill slot mapping (issue #459)
        self._prefill_mapping_static: torch.Tensor | None = None
        self._prefill_positions_static: torch.Tensor | None = None
        self._prefill_blocks_static: torch.Tensor | None = None
        self._graph_prefill_mode = False
        self._block_refcount: dict[int, int] = {}
        # RadixAttention trie: token-by-token nested dict with block-aligned entries.
        # Each node is a dict with optional key "_entry" holding prefix block data.
        # "_lru" is a monotonic counter bumped on every store/access for LRU eviction.
        self._prefix_trie: dict = {"_lru": 0}
        self._max_prefix_entries = max_prefix_entries
        self._prefix_entry_count = 0
        # HBM budget on the prefix cache (issue: unbounded prefix-cache retention).
        # A long-running server serving mostly-unique prompts would otherwise pin KV
        # blocks in the radix trie forever — completed requests free their OWN slot
        # refs, but the trie's refs persist until an LRU eviction that (with
        # `max_prefix_entries=None`) never triggered, so the pool could be exhausted.
        # We bound the retention by the number of PHYSICAL BLOCKS the prefix cache is
        # allowed to pin (block bytes + trie metadata scale with block count), NOT
        # just an entry count. Default: half the pool, computed from `num_blocks` so
        # it is always non-None. `max_prefix_blocks` overrides the fraction.
        # `_prefix_block_refs[blk]` counts how many prefix ENTRIES reference `blk`
        # (distinct pinned blocks = `len(self._prefix_block_refs)`); a block also held
        # by a live in-flight slot keeps a positive `_block_refcount` and is never
        # freed by prefix eviction — eviction only drops the completed-but-cached
        # trie entry's reference.
        if max_prefix_blocks is not None:
            self._max_prefix_blocks: int | None = max_prefix_blocks
        elif max_prefix_block_fraction is not None:
            self._max_prefix_blocks = max(1, int(self.num_blocks * max_prefix_block_fraction))
        else:
            self._max_prefix_blocks = None
        self._prefix_block_refs: dict[int, int] = {}
        # Pinned host staging for the CUDA-graph decode hot path (issue #183): the
        # per-step slot-mapping / block-table used to be rebuilt with
        # `torch.tensor(list, device=cuda)` (a blocking pageable H2D) then D2D-copied
        # into the captured graph buffers. `fill_slot_mapping` / `fill_block_table`
        # fill these pinned buffers in place and issue ONE non_blocking copy into the
        # caller's persistent device tensor. Allocated lazily / grown on demand.
        self._pin = device != "cpu" and torch.cuda.is_available()
        self._slot_map_host: torch.Tensor | None = None
        self._block_table_host: torch.Tensor | None = None

    # -- layer mapping + K8V3 codec helpers -----------------------------------

    def _layer_row(self, layer: int) -> int:
        """Storage row for a real model layer index. Identity unless the cache was
        built with `kv_layers` (then only those layers bear KV storage)."""
        if self._kv_map is None:
            return layer
        return self._kv_map[layer]

    def _flat_rows(self, slot_mapping: torch.Tensor) -> torch.Tensor:
        """[B] flat slot (block*block_size + offset) -> [B*Hkv] flat ROW indices into
        the cache's [.., nkv*block_size, ..] per-layer storage, matching the
        `kv_write_paged` kernel's `row = (block * h_kv + h) * block_size + offset`
        layout (head-major within a block)."""
        bs = self.block_size
        nkv = self.num_kv_heads
        b = slot_mapping.shape[0]
        blk = slot_mapping // bs
        off = slot_mapping % bs
        heads = torch.arange(nkv, device=slot_mapping.device)
        return (
            blk.repeat_interleave(nkv) * (nkv * bs)
            + heads.repeat(b) * bs
            + off.repeat_interleave(nkv)
        ).long()

    def _write_k_int8(
        self, layer: int, k_new: torch.Tensor, slot_mapping: torch.Tensor
    ) -> None:
        """Quantize-on-write of K (fp16 [B, Hkv, D]) into the paged int8 store,
        byte-identical to the K side of the fused `quantize_kv_write_paged` kernel:
        Hadamard rotation (involution) then per-row symmetric-RTN int8 with an fp32
        per-token scale, scattered at the caller-computed flat slots."""
        from superl8.quant.rotation import rotate_last

        row = self._layer_row(layer)
        k_rot = rotate_last(k_new)  # [B, Hkv, D] fp16
        # The fused rowwise quantizer is [M, K] (the per-row RTN recipe of the
        # `kv_write_paged` K side); flatten the (token, head) rows like the paged
        # kernel does, then scatter by flat row.
        k_i8, k_sc = superl8.quantize_i8_rowwise(k_rot.reshape(-1, self.head_dim))  # [B*Hkv,D] int8
        rows = self._flat_rows(slot_mapping)
        self.k_cache[row].reshape(-1, self.head_dim)[rows] = k_i8
        self.k_scale[row].reshape(-1)[rows] = k_sc

    def _write_v_lloydmax3(
        self, layer: int, v_new: torch.Tensor, slot_mapping: torch.Tensor
    ) -> None:
        """Quantize-on-write of V (fp16 [B, Hkv, D]) into the packed 3-bit Lloyd-Max
        store: block-L2-normalize, bucketize onto the per-layer codebook, bit-pack the
        codes and scatter them plus the fp32 block norms at the flat slots. Reuses
        `superl8.quant.lloydmax` verbatim (no codec reimplementation)."""
        from superl8.quant.lloydmax import pack_indices_lowbit, quantize_lloydmax

        row = self._layer_row(layer)
        codes, norms, _ = quantize_lloydmax(
            v_new,
            bits=3,
            block_size=self.lloyd_block,
            dim=-1,
            codebook=self.v_codebook[row],
        )
        packed = pack_indices_lowbit(codes, 3)  # [B, Hkv, hd*3//32] int32
        rows = self._flat_rows(slot_mapping)
        self.v_packed[row].reshape(-1, packed.shape[-1])[rows] = packed.reshape(-1, packed.shape[-1])
        self.v_norm[row].reshape(-1, self.v_norm.shape[-1])[rows] = norms.reshape(-1, self.v_norm.shape[-1])

    def _read_sequence_fp(self, layer: int, blocks: list[int], length: int, *, window: int | None = None):
        """Dequantize one sequence's cached K/V back to fp16 [1, Hkv, N, D] in the
        lloydmax3 layout: K from the int8 store (rotation undone), V from the packed
        3-bit Lloyd-Max store. `window` caps N to the last `window` positions.
        lloydmax3-only (the int8 format has the fused `attn_paged_decode_cached`
        kernel and the existing `read_dense`)."""
        start = max(0, length - window) if window is not None else 0
        positions = range(start, length)
        blk = torch.tensor(
            [blocks[p // self.block_size] for p in positions],
            device=self.device,
            dtype=torch.long,
        )
        off = torch.arange(start, length, device=self.device, dtype=torch.long) % self.block_size
        row = self._layer_row(layer)
        from superl8.quant.lloydmax import dequantize_lloydmax, unpack_indices_lowbit

        n = length - start
        k_out = torch.empty(
            (1, self.num_kv_heads, n, self.head_dim),
            dtype=torch.float16,
            device=self.device,
        )
        v_out = torch.empty_like(k_out)
        for lo in range(0, n, self._DENSE_DEQUANT_CHUNK_TOKENS):
            hi = min(lo + self._DENSE_DEQUANT_CHUNK_TOKENS, n)
            b, o = blk[lo:hi], off[lo:hi]
            k = (
                self.k_cache[row, b, :, o, :].float()
                * self.k_scale[row, b, :, o].unsqueeze(-1)
            ).half()
            k_out[0, :, lo:hi, :].copy_(rotate_last(k).permute(1, 0, 2))

            packed = self.v_packed[row, b, :, o, :]
            codes = unpack_indices_lowbit(packed, 3, self.head_dim)
            v = dequantize_lloydmax(
                codes,
                self.v_norm[row, b, :, o, :],
                self.v_codebook[row],
                block_size=self.lloyd_block,
                dim=-1,
            ).half()
            v_out[0, :, lo:hi, :].copy_(v.permute(1, 0, 2))
        return k_out, v_out

    def _decode_lloydmax3(self, layer: int, q: torch.Tensor, block_table, context_lens, *, scale, max_context_len: int | None = None):
        """K8V3 decode fallback: dequantize each sequence's K/V to fp16 (int8 K +
        3-bit Lloyd-Max V) and run the fp16-input `attn_int8_decode` per slot — the
        accuracy-gated fp fallback from the K8V3 wiring plan until a fused
        lowbit-V-over-int8-K paged decode kernel lands (superl8#272-style)."""
        outs = []
        B = q.shape[0]
        for b in range(B):
            # Use precomputed max_context_len if available to avoid host sync during CUDA graph capture
            if max_context_len is not None:
                n = int(max_context_len)
            else:
                n = int(context_lens[b].item())
            blocks = [int(x) for x in block_table[b][: (n + self.block_size - 1) // self.block_size]]
            kb, vb = self._read_sequence_fp(layer, blocks, n)
            outs.append(superl8.attn_int8_decode(q[b : b + 1], kb, vb, scale=scale))
        return torch.cat(outs, dim=0)

    def _can_fused_k8v3(self, q: torch.Tensor) -> bool:
        """Whether this cache/query can use superl8#295's additive fused ABI."""
        return (
            self.v_quant == "lloydmax3"
            and self._k8v3_fused_enabled
            and not self._k8v3_fused_failed
            and q.is_cuda
            and self.head_dim in (128, 256)
            and self.lloyd_block == 128
            and callable(getattr(superl8, "attn_paged_decode_k8v3", None))
        )

    def alloc(self) -> int:
        if not self._free_slots:
            raise RuntimeError("PagedKVCache: no free slots")
        return self._free_slots.pop()

    def free(self, slot: int):
        """Recycle a finished sequence's blocks back into the shared pool.
        Reference-counted: blocks shared via prefix cache are only freed when
        the last reference is gone."""
        for blk in self._slot_blocks[slot]:
            c = self._block_refcount.get(blk, 1) - 1
            if c <= 0:
                self._block_refcount.pop(blk, None)
                self._free_blocks.append(blk)
            else:
                self._block_refcount[blk] = c
        self._slot_blocks[slot] = []
        self._free_slots.append(slot)

    def has_free_slot(self) -> bool:
        return bool(self._free_slots)

    def has_free_block(self) -> bool:
        """Whether the shared block pool has at least one free physical block. The
        scheduler uses this to distinguish genuine KV-memory pressure (dry pool)
        from a mere count/slot cap: only the former justifies preemption."""
        return bool(self._free_blocks)

    @property
    def used_blocks(self) -> int:
        """Physical blocks currently pinned (allocated) out of `num_blocks` --
        drives the KV-cache usage % in the telemetry heartbeat / TUI. Host-side
        int arithmetic only; no GPU sync."""
        return self.num_blocks - len(self._free_blocks)

    def ensure_capacity(self, slots: list[int], lengths: list[int]):
        """Grow each slot's block table so it can hold `lengths[i]` tokens,
        pulling new physical blocks from the shared pool one at a time."""
        for slot, n in zip(slots, lengths):
            blocks = self._slot_blocks[slot]
            need = (n + self.block_size - 1) // self.block_size
            while len(blocks) < need:
                if not self._free_blocks:
                    raise RuntimeError("PagedKVCache: no free blocks")
                blk = self._free_blocks.pop()
                blocks.append(blk)
                self._block_refcount[blk] = 1

    def share_blocks(self, slot: int, blocks: list[int]):
        """Point *slot* at existing physical *blocks* and increment their refcounts."""
        self._slot_blocks[slot] = list(blocks)
        for blk in blocks:
            self._block_refcount[blk] = self._block_refcount.get(blk, 1) + 1

    def _tick_lru(self) -> int:
        """Bump and return the global LRU counter."""
        self._prefix_trie["_lru"] = self._prefix_trie.get("_lru", 0) + 1
        return self._prefix_trie["_lru"]

    def _evict_one_prefix(self) -> bool:
        """Find the prefix entry with the oldest LRU timestamp and evict it.
        Releases block refcounts and cleans up orphaned trie nodes. Returns True if
        an entry was evicted, False if the trie held no evictable entry."""
        oldest = None
        oldest_lru = None

        def _walk(node: dict, path: list):
            nonlocal oldest, oldest_lru
            for k, v in node.items():
                if isinstance(k, str) and k.startswith("_"):
                    continue
                if isinstance(v, dict):
                    if "_entry" in v:
                        lru = v["_entry"]["lru"]
                        if oldest is None or lru < oldest_lru:
                            oldest = (path + [k], v)
                            oldest_lru = lru
                    _walk(v, path + [k])

        _walk(self._prefix_trie, [])
        if oldest is None:
            return False
        path, node = oldest
        entry = node.pop("_entry", None)
        if entry is not None:
            for blk in entry["blocks"]:
                # Drop this entry's prefix-cache reference on the block (budget
                # accounting). A block still referenced by a live in-flight slot or
                # another prefix entry keeps a positive `_block_refcount` below and
                # is NOT returned to the free pool.
                pc = self._prefix_block_refs.get(blk, 0) - 1
                if pc <= 0:
                    self._prefix_block_refs.pop(blk, None)
                else:
                    self._prefix_block_refs[blk] = pc
                c = self._block_refcount.get(blk, 1) - 1
                if c <= 0:
                    self._block_refcount.pop(blk, None)
                    self._free_blocks.append(blk)
                else:
                    self._block_refcount[blk] = c
            self._prefix_entry_count -= 1
        # Clean up orphaned nodes (leaf nodes with no children or entries)
        if len(node) == 0:
            parent = self._prefix_trie
            for p in path[:-1]:
                parent = parent.get(p, {})
            parent.pop(path[-1], None)
        return True

    def _enforce_prefix_budget(self) -> None:
        """Evict oldest (LRU) prefix entries until the pinned-block budget holds.
        Bounds HBM pinned by the completed-prefix cache so a stream of unique prompts
        cannot exhaust the block pool. Live in-flight blocks are never freed here —
        eviction only releases the trie's own references (see `_evict_one_prefix`)."""
        if self._max_prefix_blocks is None:
            return
        # Each iteration removes exactly one trie entry, so the loop always makes
        # progress and terminates once the trie is empty (blocks -> 0) even if some
        # nested inner entries share blocks with an outer entry not yet evicted.
        while len(self._prefix_block_refs) > self._max_prefix_blocks:
            if not self._evict_one_prefix():
                break

    @property
    def pinned_prefix_blocks(self) -> int:
        """Distinct physical blocks currently pinned by the prefix (radix) cache."""
        return len(self._prefix_block_refs)

    def prefix_block_bytes(self) -> int:
        """HBM bytes pinned by the prefix cache across all layers (K+V + scales in
        the active cache format). Drives the block budget and telemetry."""
        bb = KVEviction.block_bytes(
            self.num_kv_heads, self.block_size, self.head_dim, v_quant=self.v_quant
        )
        return len(self._prefix_block_refs) * self.num_layers * bb

    def store_prefix(self, token_ids: list[int], slot: int):
        """Store a completed prefix in the radix trie for future lookups.
        Stores entries at every block-aligned boundary with LRU tracking.
        Evicts the oldest entry when max_prefix_entries is exceeded."""
        n = len(token_ids)
        num_shared = n // self.block_size
        if num_shared == 0:
            return
        full_blocks = list(self._slot_blocks[slot])
        lru_now = self._tick_lru()
        node = self._prefix_trie
        for i, tid in enumerate(token_ids):
            node = node.setdefault(tid, {})
            pos = i + 1
            if pos % self.block_size == 0 and pos // self.block_size <= num_shared:
                nb = pos // self.block_size
                # Block-aligned boundary — store an entry if not already present
                if "_entry" not in node:
                    if (
                        self._max_prefix_entries is not None
                        and self._prefix_entry_count >= self._max_prefix_entries
                    ):
                        self._evict_one_prefix()
                    node["_entry"] = {
                        "blocks": full_blocks[:nb],
                        "num_tokens": pos,
                        "lru": lru_now,
                    }
                    for blk in full_blocks[:nb]:
                        self._block_refcount[blk] = self._block_refcount.get(blk, 1) + 1
                        self._prefix_block_refs[blk] = self._prefix_block_refs.get(blk, 0) + 1
                    self._prefix_entry_count += 1
                else:
                    # Entry already exists — bump its LRU timestamp
                    node["_entry"]["lru"] = lru_now
        # Bound HBM pinned by the completed-prefix cache (block-BYTES budget, not
        # only entry count). Runs after the whole prefix is stored so the entries we
        # just added (highest LRU) are evicted last.
        self._enforce_prefix_budget()

    def lookup_prefix(self, token_ids: list[int]) -> tuple[int, list[int]]:
        """Find the longest block-aligned matching prefix.
        Returns (matched_len, shared_blocks). Bumps LRU on accessed entries."""
        node = self._prefix_trie
        best_len = 0
        best_blocks: list[int] = []
        lru_now = self._tick_lru()
        for i, tid in enumerate(token_ids):
            if tid not in node:
                break
            node = node[tid]
            if "_entry" in node:
                entry = node["_entry"]
                stored = entry["num_tokens"]
                if i + 1 >= stored:
                    best_len = stored
                    best_blocks = list(entry["blocks"])
                    entry["lru"] = lru_now  # bump LRU on access
        return best_len, best_blocks

    def _slot_mapping(self, slots: list[int], positions: list[int]) -> torch.Tensor:
        flat = [
            self._slot_blocks[s][p // self.block_size] * self.block_size + p % self.block_size
            for s, p in zip(slots, positions)
        ]
        return torch.tensor(flat, dtype=torch.int32, device=self.device)

    def flat_slot_mapping(self, slot: int, length: int) -> list[int]:
        """Host-only per-token slot-mapping ints for positions [0, length) of one
        slot: pure Python integer arithmetic, no device tensor and no per-token
        ``.item()`` sync. BYTE-IDENTICAL to what ``_slot_mapping([slot], range)``
        computes (same ``blocks[p // block_size] * block_size + p % block_size``
        formula) — used by the varlen-prefill packer to build its slot_mapping
        tensor in one shot."""
        bs = self.block_size
        blocks = self._slot_blocks[slot]
        return [blocks[p // bs] * bs + p % bs for p in range(length)]

    def enable_graph_prefill(self, max_len: int) -> None:
        """Pre-allocate static buffers for graph-captured prefill writes.

        ``_prefill_blocks_static`` is indexed by ``block_idx = pos // block_size``
        and must therefore be at least ``num_blocks`` wide (not ``num_slots`` which
        is typically 2).  ``_prefill_mapping_static`` stores the flat physical-slot
        mapping for ``max_len`` tokens."""
        self._prefill_mapping_static = torch.zeros(
            max_len, dtype=torch.int32, device=self.device
        )
        # Sized to num_blocks so block_idx (pos // block_size) never OOBs.
        self._prefill_blocks_static = torch.zeros(
            self.num_blocks, dtype=torch.int64, device=self.device
        )
        self._graph_prefill_mode = True

    def bind_graph_write_prefill(self, slot: int, length: int) -> None:
        """Fill static slot-mapping buffer for a captured graph prefill.

        Populates ``_prefill_mapping_static`` (flat physical-slot mapping) and
        ``_prefill_blocks_static`` (physical block-id table, indexed by
        ``block_idx = pos // block_size``) so the graph-captured ``write_prefill``
        can compute ``mapping = blocks[block_idx] * block_size + remainder`` without
        any per-call tensor allocation."""
        if self._prefill_mapping_static is None:
            return
        mapping = self.flat_slot_mapping(slot, length)
        self._prefill_mapping_static[:len(mapping)] = torch.tensor(
            mapping, dtype=torch.int32, device=self.device
        )
        # Fill the full block table for this slot so block_idx lookups work.
        # _prefill_blocks_static is indexed by block_idx (not slot), so we write
        # the entire _slot_blocks[slot] list into it.
        blocks = self._slot_blocks[slot]
        n = len(blocks)
        if n > 0:
            self._prefill_blocks_static[:n] = torch.tensor(
                blocks, dtype=torch.int64, device=self.device
            )

    # Public alias -- `engine/cuda_graph.py` builds this tensor itself (once per
    # step, reused across every layer) instead of going through `write_decode`.
    def slot_mapping_for(self, slots: list[int], positions: list[int]) -> torch.Tensor:
        return self._slot_mapping(slots, positions)

    def _ensure_decode_staging(self, batch_size: int):
        if self._slot_map_host is None or self._slot_map_host.numel() < batch_size:
            self._slot_map_host = torch.empty(batch_size, dtype=torch.int32, pin_memory=self._pin)
        if self._block_table_host is None or self._block_table_host.shape[0] < batch_size:
            self._block_table_host = torch.zeros(
                batch_size, self.max_blocks_per_seq, dtype=torch.int32, pin_memory=self._pin
            )

    def fill_slot_mapping(self, dst: torch.Tensor, slots: list[int], positions: list[int]):
        """Refresh the CUDA-graph decode slot-mapping buffer `dst` in place (same
        device pointer the graph captured) via a pinned host staging buffer + one
        non_blocking copy -- no per-step device allocation, no blocking H2D."""
        n = len(slots)
        self._ensure_decode_staging(n)
        bs = self.block_size
        flat = [self._slot_blocks[s][p // bs] * bs + p % bs for s, p in zip(slots, positions)]
        self._slot_map_host[:n].copy_(torch.tensor(flat, dtype=torch.int32))
        dst.copy_(self._slot_map_host[:n], non_blocking=self._pin)

    def fill_block_table(self, dst: torch.Tensor, slots: list[int]):
        """Refresh the CUDA-graph decode block-table buffer `dst` in place via a
        pinned host staging buffer + one non_blocking copy (companion to
        `fill_slot_mapping`). Rows shorter than the widest sequence are zero-padded
        (never read: the kernel gates every read by `context_lens`)."""
        n = len(slots)
        self._ensure_decode_staging(n)
        host = self._block_table_host[:n]
        host.zero_()
        for i, s in enumerate(slots):
            blocks = self._slot_blocks[s]
            if blocks:
                host[i, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
        dst.copy_(host, non_blocking=self._pin)

    def write_prefill(
        self, layer: int, k: torch.Tensor, v: torch.Tensor, *, slot: int, start: int = 0,
        positions: torch.Tensor | None = None,
    ):
        """k, v: [1, Hkv, S, D] fp16 -> one batched quantize-on-write into
        this sequence's blocks.

        Like vLLM's reshape-and-cache path, construct a vector physical-slot
        mapping and submit all tokens together.  In particular, never turn a CUDA
        position into a Python integer: that old loop imposed one device sync and
        one tiny cache-write launch per token per KV-bearing layer.

        *start* skips the first *start* positions (shared-prefix reuse).
        *positions*, if given, overrides the implicit [0, S) sequence positions with
        the actual positions in the sequence (needed for chunked prefill where the
        K/V tensor covers a subset of the prompt)."""
        s = k.shape[2]
        if s == 0:
            return
        k_tokens = k.permute(0, 2, 1, 3).reshape(s, self.num_kv_heads, self.head_dim)
        v_tokens = v.permute(0, 2, 1, 3).reshape(s, self.num_kv_heads, self.head_dim)

        if positions is None:
            first = min(max(start, 0), s)
            if first == s:
                return
            # Graph-captured path: use pre-allocated static mapping buffer
            if self._graph_prefill_mode and self._prefill_mapping_static is not None:
                mapping = self._prefill_mapping_static[first:s]
            else:
                mapping = torch.tensor(
                    self.flat_slot_mapping(slot, s)[first:],
                    dtype=torch.int32,
                    device=self.device,
                )
            k_tokens = k_tokens[first:]
            v_tokens = v_tokens[first:]
        else:
            pos = positions.reshape(-1).to(device=self.device, dtype=torch.int64)
            if start > 0:
                keep = pos >= start
                # Prefix reuse is not the ordinary chunked-prefill hot path.  One
                # bounded sync here preserves the old all-prefix no-op contract;
                # the common start=0 path has no host synchronization at all.
                if not bool(keep.any()):
                    return
                pos = pos[keep]
                k_tokens = k_tokens[keep]
                v_tokens = v_tokens[keep]
            if self._graph_prefill_mode and self._prefill_blocks_static is not None:
                blocks = self._prefill_blocks_static
            else:
                blocks = torch.tensor(
                    self._slot_blocks[slot], dtype=torch.int64, device=self.device
                )
            block_idx = torch.div(pos, self.block_size, rounding_mode="floor")
            mapping = (
                blocks[block_idx] * self.block_size + torch.remainder(pos, self.block_size)
            ).to(torch.int32)

        self.write_prefill_varlen(layer, mapping, k_tokens, v_tokens)

    def write_prefill_varlen(
        self, layer: int, slot_mapping: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ):
        """k, v: [total_tokens, Hkv, D] fp16. slot_mapping: [total_tokens] int32.
        Write every token's K/V to its correct page in ONE call — the varlen
        batched prefill path. Every token's slot_mapping entry points to the
        correct physical page + offset for its sequence and position, including
        tokens that were already filled by a shared prefix (overwrite is cheap
        and avoids per-sequence branching)."""
        if self.v_quant == "lloydmax3":
            self._write_k_int8(layer, k, slot_mapping)
            self._write_v_lloydmax3(layer, v, slot_mapping)
            return
        row = self._layer_row(layer)
        superl8.quantize_kv_write_paged(
            k.contiguous(),
            v.contiguous(),
            self.k_cache[row],
            self.k_scale[row],
            self.v_cache[row],
            self.v_scale[row],
            slot_mapping,
        )

    def write_decode(
        self,
        layer: int,
        slots: list[int],
        positions: list[int],
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ):
        """k_new, v_new: [B, Hkv, D] fp16 -- the newest token for every sequence in
        the batch, committed with ONE `quantize_kv_write_paged` call."""
        self.write_decode_static(layer, self._slot_mapping(slots, positions), k_new, v_new)

    def write_decode_static(
        self, layer: int, slot_mapping: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor
    ):
        """Same as `write_decode`, but takes an already-built `slot_mapping` device
        tensor instead of python `slots`/`positions` lists -- the CUDA-graph decode
        path (engine/cuda_graph.py) calls this with a persistent buffer it refreshes
        via `copy_` before each replay, since a captured graph can only re-execute
        kernels against fixed memory, not rebuild tensors from python lists."""
        if self.v_quant == "lloydmax3":
            self._write_k_int8(layer, k_new, slot_mapping)
            self._write_v_lloydmax3(layer, v_new, slot_mapping)
            return
        row = self._layer_row(layer)
        superl8.quantize_kv_write_paged(
            k_new.contiguous(),
            v_new.contiguous(),
            self.k_cache[row],
            self.k_scale[row],
            self.v_cache[row],
            self.v_scale[row],
            slot_mapping,
        )

    def block_table(self, slots: list[int]) -> torch.Tensor:
        """[B, max_blocks_per_seq] int32 physical block ids for this batch -- rows
        shorter than the widest sequence are zero-padded (never read: the kernel
        gates every read by `context_lens`)."""
        bt = torch.zeros(len(slots), self.max_blocks_per_seq, dtype=torch.int32, device=self.device)
        for i, s in enumerate(slots):
            blocks = self._slot_blocks[s]
            if blocks:
                bt[i, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32, device=self.device)
        return bt

    def decode_attn(
        self, layer: int, q: torch.Tensor, slots: list[int], lengths: list[int], *, scale: float
    ) -> torch.Tensor:
        """ONE batched paged-decode launch across the whole ragged running batch --
        `lengths[i]` is the write position of the token just committed by
        `write_decode`, so the valid context per row is `lengths[i] + 1`.

        K8V3 uses superl8#295's fused packed-V kernel when supported and otherwise
        retains the exact dense reconstruction fallback."""
        context_lens = torch.tensor([n + 1 for n in lengths], dtype=torch.int32, device=self.device)
        return self.decode_attn_static(
            layer,
            q,
            self.block_table(slots),
            context_lens,
            int(context_lens.max().item()),
            scale=scale,
        )

    def decode_attn_static(
        self,
        layer: int,
        q: torch.Tensor,
        block_table: torch.Tensor,
        context_lens: torch.Tensor,
        max_context_len: int,
        *,
        scale: float,
    ) -> torch.Tensor:
        """Same as `decode_attn`, but takes precomputed `block_table`/`context_lens`
        device tensors and `max_context_len` as a plain python int instead of calling
        `context_lens.max().item()` -- that `.item()` is a device->host sync, which
        CUDA graph capture cannot contain. The CUDA-graph decode path passes a
        per-bucket compile-time upper bound here (safe: `attn_paged_decode_cached`
        only requires `max_context_len >= max(context_lens)`, since it just
        upper-bounds kernel split-sizing and every row is still gated by its own
        `context_lens` entry)."""
        row = self._layer_row(layer)
        if self.v_quant == "lloydmax3":
            if self._can_fused_k8v3(q):
                try:
                    return superl8.attn_paged_decode_k8v3(
                        q,
                        self.k_cache[row],
                        self.k_scale[row],
                        self.v_packed[row],
                        self.v_norm[row],
                        self.v_codebook[row],
                        block_table,
                        context_lens,
                        self.block_size,
                        max_context_len=max_context_len,
                        scale=scale,
                    )
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    # An older/mismatched extension must not impose one failed launch
                    # on every generated token. Disable this cache instance and keep
                    # the proven dense path as the exact rollback.
                    self._k8v3_fused_failed = True
                    logger.warning("disabling fused K8V3 decode after operator failure: %s", exc)
            return self._decode_lloydmax3(layer, q, block_table, context_lens, scale=scale, max_context_len=max_context_len)
        return superl8.attn_paged_decode_cached(
            q,
            self.k_cache[row],
            self.k_scale[row],
            self.v_cache[row],
            self.v_scale[row],
            block_table,
            context_lens,
            self.block_size,
            max_context_len=max_context_len,
            scale=scale,
        )

    def build_verify_cache(
        self,
        layer: int,
        slots: list[int],
        lengths: list[int],
        k_draft: torch.Tensor,
        v_draft: torch.Tensor,
    ):
        """Build a contiguous int8 KV cache for the verify kernel: prefix K/V
        (read from the paged store and dequantised) + draft K/V concatenated.

        *k_draft*, *v_draft*: fp16 [B, Hkv, k, D]. Returns
        ``(k_i8, k_scale, v_i8, v_scale)`` shaped ``[B, Hkv, N, D]`` /
        ``[B, Hkv, N]`` float32 as required by :func:`superl8.attn_int8_verify`."""
        B = len(slots)
        max_prefix = max(lengths) if lengths else 0
        k_parts, v_parts = [], []
        for b in range(B):
            kp, vp = self.read_dense(layer, slots[b], lengths[b])
            if lengths[b] < max_prefix:
                pad_sz = max_prefix - lengths[b]
                pad = kp.new_zeros(1, kp.shape[1], pad_sz, kp.shape[3])
                kp = torch.cat([kp, pad], dim=2)
                vp = torch.cat([vp, pad], dim=2)
            k_parts.append(torch.cat([kp, k_draft[b : b + 1]], dim=2))
            v_parts.append(torch.cat([vp, v_draft[b : b + 1]], dim=2))
        full_k = torch.cat(k_parts, dim=0)  # [B, Hkv, N, D]
        full_v = torch.cat(v_parts, dim=0)
        return superl8.quantize_kv_cache(full_k, full_v)

    def read_dense(
        self,
        layer: int,
        slot: int,
        length: int,
        *,
        window: int | None = None,
        dtype: torch.dtype = torch.float16,
    ):
        """Dequantize one sequence's cached K/V back to fp16 [1,Hkv,N,D] -- the
        sliding-window fallback (`attn_paged_decode_cached` has no window parameter
        yet). `window` caps N to the last `window` positions. In lloydmax3 (K8V3)
        mode V is dequantized from the packed 3-bit store instead of int8."""
        start = max(0, length - window) if window is not None else 0
        blocks = self._slot_blocks[slot]
        positions = range(start, length)
        blk = torch.tensor(
            [blocks[p // self.block_size] for p in positions],
            device=self.device,
            dtype=torch.long,
        )
        off = torch.arange(start, length, device=self.device, dtype=torch.long) % self.block_size
        row = self._layer_row(layer)
        n = length - start
        k_out = torch.empty(
            (1, self.num_kv_heads, n, self.head_dim), dtype=dtype, device=self.device
        )
        v_out = torch.empty_like(k_out)
        chunk = self._DENSE_DEQUANT_CHUNK_TOKENS
        if self.v_quant == "lloydmax3":
            from superl8.quant.lloydmax import dequantize_lloydmax, unpack_indices_lowbit

            for lo in range(0, n, chunk):
                hi = min(lo + chunk, n)
                b, o = blk[lo:hi], off[lo:hi]
                k = (
                    self.k_cache[row, b, :, o, :].float()
                    * self.k_scale[row, b, :, o].unsqueeze(-1)
                ).to(dtype)
                k = rotate_last(k)
                k_out[0, :, lo:hi, :].copy_(k.permute(1, 0, 2))

                packed = self.v_packed[row, b, :, o, :]
                codes = unpack_indices_lowbit(packed, 3, self.head_dim)
                v = dequantize_lloydmax(
                    codes,
                    self.v_norm[row, b, :, o, :],
                    self.v_codebook[row],
                    block_size=self.lloyd_block,
                    dim=-1,
                ).to(dtype)
                v_out[0, :, lo:hi, :].copy_(v.permute(1, 0, 2))
        else:
            for lo in range(0, n, chunk):
                hi = min(lo + chunk, n)
                b, o = blk[lo:hi], off[lo:hi]
                k = (
                    self.k_cache[row, b, :, o, :].float()
                    * self.k_scale[row, b, :, o].unsqueeze(-1)
                ).to(dtype)
                v = (
                    self.v_cache[row, b, :, o, :].float()
                    * self.v_scale[row, b, :, o].unsqueeze(-1)
                ).to(dtype)
                k_out[0, :, lo:hi, :].copy_(rotate_last(k).permute(1, 0, 2))
                v_out[0, :, lo:hi, :].copy_(v.permute(1, 0, 2))
        return k_out, v_out

    # ── Eviction convenience API ──────────────────────────────────────────────

    def evict_after_prefill(
        self,
        slot: int,
        seq_len: int,
        *,
        eviction_config: EvictionConfig,
        num_heads: int,
        keep_mask: list[int] | None = None,
        q_last: torch.Tensor | None = None,
        k_full: torch.Tensor | None = None,
        logits_full: torch.Tensor | None = None,
        logits_fn: Callable[[], torch.Tensor] | None = None,
    ) -> dict:
        """Run eviction for *slot* after prefill.

        Thin wrapper over ``KVEviction.evict`` with the cache's own dimensions
        filled in.  See ``KVEviction.evict`` for parameter documentation.

        Returns the metrics dict from ``KVEviction.evict``.
        """
        return KVEviction.evict(
            self,
            slot,
            seq_len,
            eviction_config=eviction_config,
            num_heads=num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            q_last=q_last,
            k_full=k_full,
            keep_mask=keep_mask,
            logits_full=logits_full,
            logits_fn=logits_fn,
        )
