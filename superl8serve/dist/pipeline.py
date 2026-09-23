# SPDX-License-Identifier: MIT
"""Pipeline-parallel LLM inference across 2 GPUs (issue #84).

Splits a model's layer stack into 2 stages: stage 0 owns the embedding + first
half of layers; stage 1 owns the second half + final norm + lm_head. The hidden
state (and the pre-norm residual stream) crosses the PP boundary through the
existing transport ``send``/``recv`` seam (issue #81/D9). The wire codec is
selected per-boundary by :func:`select_wire_scheme`: int4 when its fidelity
clears the accuracy-gate bar, int8 otherwise (issue #186).

Micro-batched decode overlaps the transfer with compute: while stage 0 forwards
micro-batch k+1, stage 1 receives and forwards micro-batch k, keeping both GPUs
busy.  2-stage PP only (MoE-EP is the next rung — see issue #60 / D10).
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
import torch.nn as nn
from superl8 import QTensor

from ..engine.kv_cache import PagedKVCache
from ..engine.sequence import SamplingParams, Sequence
from ..models.base import ForwardContext
from ..models.cache import RecurrentStateCache
from ..models.config import ModelConfig
from ..models.registry import build_model
from . import recv, send
from .codec_quality import select_wire_scheme

# Monkey-patch fni8's hadamard_matrix so its LRU cache key includes the concrete
# GPU device index, not just the device type.  Without this, the matrix created
# on cuda:0 gets reused on cuda:1 and causes a device-mismatch crash.
try:
    import superl8.quant.rotation as _fni8_rot_rel

    _orig_hadamard = _fni8_rot_rel.hadamard_matrix

    def _patched_hadamard(d: int, device: str = "cpu", dtype=torch.float32):
        if device == "cuda" and torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
        return _orig_hadamard(d, device=device, dtype=dtype)

    _fni8_rot_rel.hadamard_matrix = _patched_hadamard
    _orig_rotate = _fni8_rot_rel.rotate_last

    def _patched_rotate(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        dev = x.device
        m = _patched_hadamard(d, device=str(dev), dtype=torch.float32)
        return (x.float() @ m).to(x.dtype)

    _fni8_rot_rel.rotate_last = _patched_rotate
except Exception:
    pass


def _clear_hadamard_cache():
    "Clear fni8's LRU hadamard-matrix cache (legacy; the monkey-patch above should suffice)."
    try:
        from superl8.quant.rotation import hadamard_matrix

        hadamard_matrix.cache_clear()
    except Exception:
        pass


class LayerRemappedCache:
    """Wraps a KV cache so a stage's layers (original indices i..j) map to the
    stage-local cache layout (0..j-i). Each access subtracts *offset*."""

    def __init__(self, cache: PagedKVCache, offset: int):
        self._cache = cache
        self._offset = offset

    @property
    def block_size(self):
        return self._cache.block_size

    @property
    def max_blocks_per_seq(self):
        return self._cache.max_blocks_per_seq

    @property
    def num_blocks(self):
        return self._cache.num_blocks

    @property
    def device(self):
        return self._cache.device

    @property
    def num_slots(self):
        return self._cache.num_slots

    @property
    def max_len(self):
        return self._cache.max_len

    def alloc(self) -> int:
        return self._cache.alloc()

    def free(self, slot: int):
        return self._cache.free(slot)

    def has_free_slot(self) -> bool:
        return self._cache.has_free_slot()

    def ensure_capacity(self, slots: list[int], lengths: list[int]):
        return self._cache.ensure_capacity(slots, lengths)

    def share_blocks(self, slot: int, blocks: list[int]):
        return self._cache.share_blocks(slot, blocks)

    def store_prefix(self, token_ids: list[int], slot: int):
        return self._cache.store_prefix(token_ids, slot)

    def lookup_prefix(self, token_ids: list[int]):
        return self._cache.lookup_prefix(token_ids)

    def _slot_mapping(self, slots: list[int], positions: list[int]):
        return self._cache._slot_mapping(slots, positions)

    def slot_mapping_for(self, slots: list[int], positions: list[int]):
        return self._cache.slot_mapping_for(slots, positions)

    def block_table(self, slots: list[int]):
        return self._cache.block_table(slots)

    def read_dense(self, layer: int, slot: int, length: int, **kwargs):
        return self._cache.read_dense(layer - self._offset, slot, length, **kwargs)

    def write_prefill(self, layer: int, k, v, *, slot: int, start: int = 0, **kwargs):
        return self._cache.write_prefill(
            layer - self._offset, k, v, slot=slot, start=start, **kwargs
        )

    def write_prefill_varlen(self, layer: int, slot_mapping, k, v):
        return self._cache.write_prefill_varlen(layer - self._offset, slot_mapping, k, v)

    def write_decode(self, layer: int, slots, positions, k_new, v_new):
        return self._cache.write_decode(layer - self._offset, slots, positions, k_new, v_new)

    def write_decode_static(self, layer: int, slot_mapping, k_new, v_new):
        return self._cache.write_decode_static(layer - self._offset, slot_mapping, k_new, v_new)

    def decode_attn(self, layer: int, q, slots, lengths, *, scale: float):
        return self._cache.decode_attn(layer - self._offset, q, slots, lengths, scale=scale)

    def decode_attn_static(
        self, layer: int, q, block_table, context_lens, max_context_len: int, *, scale: float
    ):
        return self._cache.decode_attn_static(
            layer - self._offset, q, block_table, context_lens, max_context_len, scale=scale
        )


def _build_stage_cache(
    num_layers: int,
    num_slots: int,
    num_kv_heads: int,
    max_len: int,
    head_dim: int,
    *,
    device,
    block_size: int = 16,
    num_blocks: int | None = None,
):
    return PagedKVCache(
        num_layers,
        num_slots,
        num_kv_heads,
        max_len,
        head_dim,
        device=device,
        block_size=block_size,
        num_blocks=num_blocks,
    )


class _StageInnerModel:
    """Minimal inner model wrapper for GraphedDecodeLayers.

    Provides `.layers`, `.embed_tokens`, `.norm` that GraphedDecodeLayers
    needs to capture per-layer graphs. Wraps a PipelineStage's modules.
    """

    def __init__(self, stage: PipelineStage):
        self.layers = stage.layers
        self.embed_tokens = stage.embed
        self.norm = stage.norm


class _StageModel:
    """Minimal model wrapper for GraphedDecodeLayers.

    Provides `.config`, `.model` (the inner), `.compute_logits()`, `.modules()`.
    """

    def __init__(self, inner: _StageInnerModel, stage: PipelineStage, cfg):
        self.model = inner
        self.config = cfg
        self._stage = stage

    def compute_logits(self, hidden):
        if self._stage.lm_head is None:
            raise RuntimeError("compute_logits only on the last stage")
        return self._stage.lm_head(hidden)

    def modules(self, recurse=True):
        for layer in self._stage.layers:
            yield layer
        if self._stage.norm is not None:
            yield self._stage.norm
        if self._stage.lm_head is not None:
            yield self._stage.lm_head


def _remap_ctx(ctx: ForwardContext, cache, lin_cache=None) -> ForwardContext:
    """Return a new ForwardContext with *cache* swapped in (layer-remapped)."""
    return ForwardContext(
        is_prefill=ctx.is_prefill,
        kv_cache=cache,
        lin_cache=lin_cache or ctx.lin_cache,
        cu_seqlens=ctx.cu_seqlens,
        seq_lens=ctx.seq_lens,
        slot_mapping=ctx.slot_mapping,
        block_tables=ctx.block_tables,
        context_lens=ctx.context_lens,
        max_context_len=ctx.max_context_len,
        attn_mask=ctx.attn_mask,
        slots=ctx.slots,
        slot_lengths=ctx.slot_lengths,
        prefill_start=ctx.prefill_start,
        pixel_values=ctx.pixel_values,
        image_grid_thw=ctx.image_grid_thw,
    )


def _pack_boundary(hidden: torch.Tensor, residual: torch.Tensor | None) -> torch.Tensor:
    """Pack (hidden, residual) into one tensor for transport. residual may be None."""
    if residual is None:
        return hidden
    return torch.cat([hidden, residual], dim=-1)


def _sample_logits(sampler, logits, batch: list[Sequence], device):
    """Build per-row controls for pipeline sampling, including mixed batches."""
    temperatures = torch.tensor(
        [s.params.temperature for s in batch], device=device, dtype=torch.float32
    )
    top_p = torch.tensor(
        [s.params.top_p for s in batch], device=device, dtype=torch.float32
    )
    top_k_values = [s.params.top_k for s in batch]
    repetition_values = [s.params.repetition_penalty for s in batch]
    any_top_k = any(k > 0 for k in top_k_values)
    any_repetition_penalty = any(p != 1.0 for p in repetition_values)
    top_k = (
        torch.tensor(top_k_values, device=device, dtype=torch.int32)
        if any_top_k
        else None
    )
    repetition_penalty = (
        torch.tensor(repetition_values, device=device, dtype=torch.float32)
        if any_repetition_penalty
        else None
    )
    processors = [s.params.logit_processors for s in batch]
    has_processors = any(processors)
    return sampler(
        logits,
        temperatures,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        logit_processors=processors if has_processors else None,
        input_ids=(
            [s.all_token_ids for s in batch]
            if has_processors
            else (
                [s.repetition_token_ids for s in batch]
                if any_repetition_penalty
                else None
            )
        ),
        all_greedy=all(s.params.temperature == 0.0 for s in batch),
        any_top_p=any(s.params.top_p < 1.0 for s in batch),
        any_top_k=any_top_k,
        max_top_k=max(top_k_values) if any_top_k else None,
        any_repetition_penalty=any_repetition_penalty,
    )


def _unpack_boundary(packed: torch.Tensor, hidden_size: int):
    """Unpack -> (hidden, residual)."""
    if packed.shape[-1] == hidden_size:
        return packed, None
    return packed[..., :hidden_size], packed[..., hidden_size:]


def _detach_shared_stage_modules(stage_modules: list[list[nn.Module]]):
    """Give every stage one copy of buffer-bearing modules shared across roots.

    Model builders intentionally share one RoPE table across every layer. Moving the
    layer slices independently would otherwise move that same table to the final GPU
    and leave earlier stages with cross-device references.
    """
    owner: dict[int, int] = {}
    for stage_id, roots in enumerate(stage_modules):
        replacements: dict[int, nn.Module] = {}
        for root in roots:
            for parent in root.modules():
                for name, child in list(parent.named_children()):
                    child_id = id(child)
                    first_stage = owner.setdefault(child_id, stage_id)
                    if first_stage != stage_id:
                        replacement = replacements.setdefault(child_id, copy.deepcopy(child))
                        setattr(parent, name, replacement)


def _move_stage_module(module: nn.Module, device: str) -> nn.Module:
    """Move parameters, buffers, and fni8's dataclass QTensors to one stage GPU."""
    module.to(device)
    for child in module.modules():
        for name, value in list(vars(child).items()):
            if isinstance(value, QTensor):
                setattr(
                    child,
                    name,
                    dataclasses.replace(
                        value,
                        data=value.data.to(device),
                        scale=value.scale.to(device) if value.scale is not None else None,
                    ),
                )
            elif isinstance(value, torch.Tensor) and name not in child._parameters:
                if name not in child._buffers:
                    setattr(child, name, value.to(device))
    return module


def _move_stage_groups(module_groups: list[list[nn.Module]], devices: tuple[int, ...]) -> None:
    """Move disjoint pipeline stages concurrently, one worker per GPU."""

    def move_group(group: list[nn.Module], device: int) -> None:
        for module in group:
            _move_stage_module(module, f"cuda:{device}")
            module.eval()

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        list(pool.map(move_group, module_groups, devices))


class PipelineStage:
    """One contiguous slice of model layers on a single GPU.

    Attributes:
        stage_id: 0-indexed stage number.
        embed: VocabEmbedding (stage 0 only).
        layers: list of DecoderLayer modules.
        norm: final RMSNorm (last stage only).
        lm_head: LMHead (last stage only).
        kv_cache: this stage's PagedKVCache or MLALatentCache.
        lin_cache: this stage's RecurrentStateCache.
        layer_offset: original layer index of the first layer in this stage.
        _graphed_layers: GraphedDecodeLayers for per-layer graph decode (Phase 5).
    """

    def __init__(
        self,
        stage_id: int,
        *,
        embed=None,
        layers: list | None = None,
        norm=None,
        lm_head=None,
        kv_cache: PagedKVCache,
        lin_cache=None,
        layer_offset: int = 0,
        num_stages: int = 2,
    ):
        self.stage_id = stage_id
        self.embed = embed
        self.layers = nn.ModuleList(layers or [])
        self.norm = norm
        self.lm_head = lm_head
        self.kv_cache: PagedKVCache = kv_cache
        self.lin_cache = lin_cache or RecurrentStateCache()
        self.layer_offset = layer_offset
        self.num_stages = num_stages
        self._remapped_cache: LayerRemappedCache | None = None
        self._graphed_layers = None  # Phase 5: GraphedDecodeLayers for this stage

    @property
    def is_first(self) -> bool:
        return self.stage_id == 0

    @property
    def is_last(self) -> bool:
        return self.stage_id == self.num_stages - 1

    @property
    def _device(self):
        p = (
            next(self.layers.parameters())
            if self.layers
            else (next(self.embed.parameters()) if self.embed is not None else torch.device("cpu"))
        )
        return p.device

    def _remap(self, cache: PagedKVCache):
        if self._remapped_cache is not None:
            return self._remapped_cache
        self._remapped_cache = LayerRemappedCache(cache, self.layer_offset)
        return self._remapped_cache

    def compute_logits(self, hidden):
        if self.lm_head is None:
            raise RuntimeError("compute_logits only available on the last stage")
        return self.lm_head(hidden)


def make_pipeline(
    cfg: ModelConfig,
    weights: dict,
    *,
    devices: tuple[int, ...] = (0, 1),
    max_num_seqs: int = 16,
    max_len: int = 2048,
    block_size: int = 16,
) -> tuple[PipelineStage, ...]:
    """Build contiguous stage-local pipeline slices on two or more GPUs.

    The weights dict must contain tensors on CPU (or the same device for each stage)
    so `build_model` can place them on the correct GPU via post-build .to().

    Returns one :class:`PipelineStage` per device.
    """
    n_layers = cfg.num_hidden_layers
    n_stages = len(devices)
    if n_stages < 2:
        raise ValueError("pipeline parallelism needs at least two devices")
    if n_stages > n_layers:
        raise ValueError("pipeline stage count cannot exceed the model layer count")
    bounds = [i * n_layers // n_stages for i in range(n_stages + 1)]

    # Assemble once on CPU, split the module graph, then move only each stage's
    # parameters/buffers/QTensor payloads. The previous implementation built the
    # complete model on BOTH GPUs, so a model larger than one card could never enter
    # the supposedly sharded path and quantized QTensor attrs did not move at all.
    model: Any = build_model(cfg, weights)
    # Most families expose a `.model` backbone; Qwen3.5 is itself the backbone.
    backbone: Any = getattr(model, "model", model)
    module_groups: list[list[nn.Module]] = []
    for stage_id in range(n_stages):
        start, end = bounds[stage_id : stage_id + 2]
        group = list(backbone.layers[start:end])
        if stage_id == 0:
            group.insert(0, backbone.embed_tokens)
        if stage_id == n_stages - 1:
            group.extend([backbone.norm, model.lm_head])
        module_groups.append(group)
    _detach_shared_stage_modules(module_groups)

    _move_stage_groups(module_groups, devices)

    stages = []
    for stage_id, (device, modules) in enumerate(zip(devices, module_groups)):
        start, end = bounds[stage_id : stage_id + 2]
        stages.append(
            PipelineStage(
                stage_id,
                embed=backbone.embed_tokens if stage_id == 0 else None,
                layers=list(backbone.layers[start:end]),
                norm=backbone.norm if stage_id == n_stages - 1 else None,
                lm_head=model.lm_head if stage_id == n_stages - 1 else None,
                layer_offset=start,
                num_stages=n_stages,
                kv_cache=_build_stage_cache(
                    end - start,
                    max_num_seqs,
                    cfg.num_key_value_heads,
                    max_len,
                    cfg.resolved_head_dim(),
                    device=device,
                    block_size=block_size,
                ),
            )
        )
    return tuple(stages)


class PipelineEngine:
    """Pipeline-parallel inference across two or more contiguous GPU stages."""

    def __init__(
        self,
        stage0: PipelineStage,
        stage1: PipelineStage,
        cfg: ModelConfig,
        *,
        max_num_seqs: int = 16,
        max_len: int = 2048,
        max_batch_tokens: int = 8192,
        eos_id: int | None = None,
        wire_scheme: str | None = None,
    ):
        self.stage0 = stage0
        self.stage1 = stage1
        self.stages = (stage0, stage1)
        self.cfg = cfg
        self.eos_id = eos_id
        self.hidden_size = cfg.hidden_size
        self._ids = itertools.count()
        self._out: dict[int, Sequence] = {}
        self._wire_scheme = wire_scheme
        self._graphed_stages_initialized = False

        from ..engine.scheduler import Scheduler

        self.scheduler = Scheduler(
            stage0.kv_cache,
            max_num_seqs=max_num_seqs,
            max_batch_tokens=max_batch_tokens,
            eos_id=eos_id,
        )

    @classmethod
    def from_stages(cls, stages: tuple[PipelineStage, ...], cfg: ModelConfig, **kwargs):
        if len(stages) < 2:
            raise ValueError("PipelineEngine needs at least two stages")
        engine = cls(stages[0], stages[-1], cfg, **kwargs)
        engine.stages = tuple(stages)
        return engine

    def _init_graphed_stages(self):
        """Initialize GraphedDecodeLayers for each stage (Phase 5).

        Each stage gets a GraphedDecodeLayers instance covering its layer range.
        For stage 0, this provides embedding + layer replay. For downstream stages,
        this provides layer replay from a pre-filled staging buffer.
        """
        if self._graphed_stages_initialized:
            return

        from ..engine.cuda_graph import GraphedDecodeLayers

        for stage in self.stages:
            stage_model = self._build_stage_model(stage)
            if stage_model is not None:
                with torch.cuda.device(stage._device):
                    stage._graphed_layers = GraphedDecodeLayers(
                        stage_model,
                        stage.kv_cache,
                        device=str(stage._device),
                        lin_cache=stage.lin_cache,
                        forward_cache=stage._remap(stage.kv_cache),
                        layers_only=True,
                        input_residual=not stage.is_first,
                    )
                if not stage._graphed_layers.supported:
                    stage._graphed_layers = None

        self._graphed_stages_initialized = True

    def _build_stage_model(self, stage: PipelineStage):
        """Build a minimal model-like object for GraphedDecodeLayers.

        GraphedDecodeLayers needs a model with `.config`, `.model.layers`,
        `.model.embed_tokens`, `.model.norm`, `.compute_logits()`, and `.modules()`.
        """
        if not stage.layers:
            return None

        cfg = self.cfg
        inner = _StageInnerModel(stage)
        model = _StageModel(inner, stage, cfg)
        return model

    @property
    def devices(self):
        return tuple(stage._device.index or 0 for stage in self.stages)

    def add_request(self, prompt_ids: list[int], params: SamplingParams | None = None) -> int:
        seq = Sequence(next(self._ids), list(prompt_ids), params or SamplingParams())
        self.scheduler.add(seq)
        self._out[seq.seq_id] = seq
        return seq.seq_id

    def sequence(self, seq_id: int) -> Sequence:
        return self._out[seq_id]

    def forget(self, seq_id: int) -> None:
        self._out.pop(seq_id, None)

    def _sync_caches(self, slot_op: str, slots: Any = None):
        """Mirror slot-level cache ops across every stage."""
        caches = [stage.kv_cache for stage in self.stages]
        if slot_op == "alloc":
            allocated = [cache.alloc() for cache in caches]
            if len(set(allocated)) != 1:
                raise RuntimeError(f"pipeline cache slots diverged: {allocated}")
            return allocated[0]
        elif slot_op == "free":
            for s in slots:
                for cache in caches:
                    cache.free(s)
        elif slot_op == "ensure_capacity":
            for cache in caches:
                cache.ensure_capacity(slots[0], slots[1])
        return None

    def step(self):
        batch, is_prefill = self.scheduler.schedule()
        if not batch:
            return

        if is_prefill:
            self._prefill(batch)
        else:
            if self._graphed_stages_initialized and all(
                s._graphed_layers is not None for s in self.stages
            ):
                self._decode_async(batch)
            else:
                self._decode_microbatched(batch)

        self.scheduler.postprocess(batch, is_prefill)

    def _init_boundary_staging(self, batch_bucket: int):
        """Initialize boundary staging buffers for async PP (Phase 5).

        Creates per-stage staging buffers used for cross-GPU activation transfer.
        """
        if not hasattr(self, '_boundary_bufs'):
            from ..engine.staging import StagingBuffer
            self._boundary_bufs = [
                StagingBuffer(batch_bucket, self.hidden_size, f"cuda:{self.devices[i]}")
                for i in range(len(self.stages))
            ]

    def _replay_stage(
        self,
        stage: PipelineStage,
        batch,
        boundary_hidden: torch.Tensor | None = None,
        boundary_residual: torch.Tensor | None = None,
        boundary_count: int = 0,
    ):
        """Replay a stage's layer graphs.

        For stage 0: embed tokens + replay layers.
        For downstream stages: fill staging buffer from boundary + replay layers.

        Returns (hidden, residual, active_count) from the stage's last staging buffer.
        """
        graphed = stage._graphed_layers
        if graphed is None or not graphed.supported:
            return None, None, 0

        B = len(batch)
        # Use the same bucket logic as GraphedDecodeLayers
        from ..engine.cuda_graph import _next_bucket, _round_up

        batch_bucket = _next_bucket(graphed.batch_buckets, B)
        if batch_bucket is None:
            return None, None, 0
        max_real_ctx = max(s.length for s in batch) + 1
        cap = graphed.cache.max_blocks_per_seq * graphed.cache.block_size
        context_bucket = min(
            _round_up(max_real_ctx, graphed.context_bucket_size), cap
        )
        if max_real_ctx > cap:
            return None, None, 0

        cap_data = graphed._ensure_capture(batch_bucket, context_bucket)
        if cap_data is None:
            return None, None, 0

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
            _ids_host,
            _pos_host,
            _context_lens_host,
        ) = cap_data

        # Every stage owns a distinct KV cache, so every captured stage needs
        # the current decode positions and cache addressing metadata.  Only
        # token embedding is exclusive to stage 0.
        cache = graphed.cache
        Bmax = batch_bucket
        pad = Bmax - B
        real_slots = [s.slot for s in batch]
        real_lengths = [s.length for s in batch]
        cache.ensure_capacity(real_slots, [n + 1 for n in real_lengths])
        slots = real_slots + [scratch] * pad
        lengths = real_lengths + [0] * pad

        pos[:B, 0].copy_(
            torch.tensor([s.length for s in batch], dtype=torch.long)
        )
        context_lens[:B].copy_(
            torch.tensor([s.length + 1 for s in batch], dtype=torch.int32)
        )
        if pad:
            pos[B:, 0].zero_()
            context_lens[B:].fill_(1)
        cache.fill_slot_mapping(slot_mapping, slots, lengths)
        cache.fill_block_table(block_table, slots)

        if slot_idx is not None:
            slot_idx_host.copy_(torch.tensor(slots, dtype=torch.long))
            slot_idx.copy_(slot_idx_host, non_blocking=False)
            stage.lin_cache.bind_graph(slot_idx)

        if stage.is_first:
            # Stage 0 embeds tokens into staging_bufs[0].
            ids[:B, 0].copy_(
                torch.tensor([s.last_token for s in batch], dtype=torch.long)
            )
            if pad:
                ids[B:, 0].zero_()

            h_emb = graphed._inner.embed_tokens(ids[:B])
            staging_bufs[0].buf[:B].copy_(h_emb)
            staging_bufs[0].active_count = B
        else:
            # Downstream stage: fill staging_bufs[0] from boundary transfer
            if boundary_hidden is not None and boundary_count > 0:
                staging_bufs[0].buf[:boundary_count].copy_(boundary_hidden[:boundary_count])
                if boundary_residual is not None and uses_residual:
                    staging_bufs[0].residual[:boundary_count].copy_(
                        boundary_residual[:boundary_count]
                    )
                staging_bufs[0].active_count = boundary_count
            else:
                staging_bufs[0].active_count = 0

        # Replay layer graphs
        graphed._replay_layer_graphs(layer_graphs, staging_bufs, B)

        # Get boundary output
        last_buf = staging_bufs[num_layers]
        hidden_out = last_buf.buf[:B].clone() if not last_buf.is_empty() else None
        residual_out = (
            last_buf.residual[:B].clone()
            if not last_buf.is_empty() and uses_residual
            else None
        )
        return hidden_out, residual_out, last_buf.active_count

    def _replay_stage_logits(
        self,
        stage: PipelineStage,
        staging_bufs: list,
        B: int,
        batch_bucket: int,
        context_bucket: int,
    ):
        """Replay final norm + logits for the last stage."""
        graphed = stage._graphed_layers
        if graphed is None:
            return None

        cap_data = graphed._ensure_capture(batch_bucket, context_bucket)
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
            _ids_host,
            _pos_host,
            _context_lens_host,
        ) = cap_data

        if staging_bufs[num_layers].is_empty():
            return None

        if uses_residual:
            h_norm, _ = graphed._inner.norm(
                staging_bufs[num_layers].buf[:B],
                staging_bufs[num_layers].residual[:B],
            )
        else:
            h_norm = graphed._inner.norm(staging_bufs[num_layers].buf[:B])
        return graphed.model.compute_logits(h_norm[:, -1])

    def _decode_async(self, batch: list[Sequence]):
        """Phase 5: Async decode with per-layer graph staging across GPUs.

        Each GPU runs its own GraphedDecodeLayers for its layer range.
        Activations cross the PP boundary via compressed send/recv.
        Skip-empty: if GPU 0's boundary is empty, GPU 1 skips entirely.
        """
        from ..engine.cuda_graph import _next_bucket

        if not batch:
            return

        from ..layers.sampler import Sampler

        sampler = Sampler()
        B = len(batch)

        # Phase 5: use graphed stages if initialized
        self._init_graphed_stages()
        use_graphed = all(s._graphed_layers is not None for s in self.stages)

        if not use_graphed:
            self._decode_microbatched(batch)
            return

        # Determine batch bucket for staging buffers
        graphed0 = self.stages[0]._graphed_layers
        batch_bucket = _next_bucket(graphed0.batch_buckets, B)
        if batch_bucket is None:
            self._decode_microbatched(batch)
            return
        # Stage 0: embed + run layers
        with torch.cuda.device(self.stages[0]._device):
            hidden, residual, count = self._replay_stage(
                self.stages[0], batch
            )

        # Chain through downstream stages
        for stage_idx in range(1, len(self.stages)):
            if count == 0:
                # Skip-empty: downstream stage skips entirely
                break

            # Transfer boundary activations to next stage (compressed)
            packed = _pack_boundary(
                hidden[:count].contiguous(),
                residual[:count] if residual is not None else None,
            )
            scheme = self._wire_scheme or select_wire_scheme(packed)
            handle = send(packed, dst=self.devices[stage_idx], scheme=scheme)
            received = recv(handle)
            recv_hidden, recv_residual = _unpack_boundary(received, self.hidden_size)

            # Run downstream stage from staging buffer
            with torch.cuda.device(self.stages[stage_idx]._device):
                hidden, residual, count = self._replay_stage(
                    self.stages[stage_idx],
                    batch,
                    boundary_hidden=recv_hidden,
                    boundary_residual=recv_residual,
                    boundary_count=count,
                )

        # Final norm + logits on last stage
        if count > 0 and hidden is not None and self.stages[-1].norm is not None:
            last = self.stages[-1]
            final_device = f"cuda:{self.devices[-1]}"

            # _replay_stage already returns the active output of the final stage.
            # Consume it directly rather than re-deriving another stage's capture
            # key and indexing into the private capture tuple.
            with torch.cuda.device(last._device):
                if residual is not None:
                    h_norm, _ = last.norm(hidden, residual)
                else:
                    h_norm = last.norm(hidden)
                logits = last.compute_logits(h_norm[:, -1])

            toks = _sample_logits(sampler, logits, batch, final_device)
            for seq, tok in zip(batch, toks.tolist()):
                seq.output_ids.append(tok)
                seq.length += 1

    def _prefill(self, batch: list[Sequence]):
        """Prefill every contiguous stage, transferring one compressed boundary."""
        from ..layers.sampler import Sampler

        sampler = Sampler()
        all_ids: list[int] = []
        all_positions: list[int] = []
        cu_seqlens: list[int] = [0]
        slots: list[int] = []
        slot_mappings: list[list[int]] = [[] for _ in self.stages]

        for seq in batch:
            for stage in self.stages:
                stage.kv_cache.ensure_capacity([seq.slot], [seq.num_prompt])
            n = seq.num_prompt
            all_ids.extend(seq.prompt_ids)
            all_positions.extend(range(n))
            cu_seqlens.append(cu_seqlens[-1] + n)
            slots.append(seq.slot)
            for t in range(n):
                for stage_id, stage in enumerate(self.stages):
                    slot_mappings[stage_id].append(
                        stage.kv_cache._slot_mapping([seq.slot], [t]).item()
                    )

        hidden = None
        residual = None
        for stage_id, stage in enumerate(self.stages):
            dev = self.devices[stage_id]
            device = f"cuda:{dev}"
            with torch.cuda.device(dev):
                _clear_hadamard_cache()
                stage.lin_cache.reset()
                stage.lin_cache.bind(slots)
                positions = torch.tensor([all_positions], device=device)
                cu = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
                mapping = torch.tensor(slot_mappings[stage_id], dtype=torch.int32, device=device)
                ctx = ForwardContext(
                    is_prefill=True,
                    kv_cache=stage.kv_cache,
                    lin_cache=stage.lin_cache,
                    slots=slots,
                    cu_seqlens=cu,
                    slot_mapping=mapping,
                )
                if stage.is_first:
                    assert stage.embed is not None
                    ids = torch.tensor([all_ids], device=device)
                    hidden = stage.embed(ids)
                assert hidden is not None
                stage_ctx = _remap_ctx(ctx, stage._remap(stage.kv_cache), stage.lin_cache)
                for layer in stage.layers:
                    hidden, residual = layer(hidden, positions, stage_ctx, residual)
                if not stage.is_last:
                    packed = _pack_boundary(hidden.contiguous(), residual)
                    scheme = self._wire_scheme or select_wire_scheme(packed)
                    handle = send(packed, dst=self.devices[stage_id + 1], scheme=scheme)
                    hidden, residual = _unpack_boundary(recv(handle), self.hidden_size)

        last = self.stages[-1]
        final_device = f"cuda:{self.devices[-1]}"
        assert last.norm is not None and hidden is not None
        hidden, _ = last.norm(hidden, residual)
        last_indices = torch.tensor(
            [cu_seqlens[i] - 1 for i in range(1, len(cu_seqlens))],
            device=final_device,
            dtype=torch.long,
        )
        logits = last.compute_logits(hidden[:, last_indices]).squeeze(0)
        toks = _sample_logits(sampler, logits, batch, final_device)
        for seq, tok in zip(batch, toks.tolist()):
            seq.output_ids.append(tok)

        for seq in batch:
            seq.length = seq.num_prompt
            for stage in self.stages:
                stage.kv_cache.store_prefix(seq.prompt_ids, seq.slot)

    def _decode_microbatched(self, batch: list[Sequence]):
        """Decode micro-batches through every stage and compressed boundary."""
        if not batch:
            return

        from ..layers.sampler import Sampler

        sampler = Sampler()
        microbatch_size = 2
        micros = [batch[i : i + microbatch_size] for i in range(0, len(batch), microbatch_size)]

        for mb in micros:
            slots = [s.slot for s in mb]
            lengths = [s.length for s in mb]
            hidden = None
            residual = None
            for stage_id, stage in enumerate(self.stages):
                dev = self.devices[stage_id]
                device = f"cuda:{dev}"
                with torch.cuda.device(dev):
                    _clear_hadamard_cache()
                    stage.lin_cache.bind(slots)
                    stage.kv_cache.ensure_capacity(slots, [n + 1 for n in lengths])
                    positions = torch.tensor([[s.length] for s in mb], device=device)
                    ctx = ForwardContext(
                        is_prefill=False,
                        kv_cache=stage.kv_cache,
                        lin_cache=stage.lin_cache,
                        slots=slots,
                        slot_lengths=lengths,
                    )
                    if stage.is_first:
                        assert stage.embed is not None
                        ids = torch.tensor([[s.last_token] for s in mb], device=device)
                        hidden = stage.embed(ids)
                    assert hidden is not None
                    stage_ctx = _remap_ctx(ctx, stage._remap(stage.kv_cache), stage.lin_cache)
                    for layer in stage.layers:
                        hidden, residual = layer(hidden, positions, stage_ctx, residual)
                    if not stage.is_last:
                        packed = _pack_boundary(hidden.contiguous(), residual)
                        scheme = self._wire_scheme or select_wire_scheme(packed)
                        handle = send(packed, dst=self.devices[stage_id + 1], scheme=scheme)
                        hidden, residual = _unpack_boundary(recv(handle), self.hidden_size)

            last = self.stages[-1]
            final_device = f"cuda:{self.devices[-1]}"
            assert last.norm is not None and hidden is not None
            hidden, _ = last.norm(hidden, residual)
            logits_mb = last.compute_logits(hidden[:, -1])
            mb_toks = _sample_logits(sampler, logits_mb, mb, final_device)

            for seq, tok in zip(mb, mb_toks.tolist()):
                seq.output_ids.append(tok)
                seq.length += 1

    def generate(
        self, prompts: list[list[int]], params: SamplingParams | None = None
    ) -> list[list[int]]:
        ids = [self.add_request(p, params) for p in prompts]
        while self.scheduler.has_work():
            self.step()
        return [self._out[i].output_ids for i in ids]
