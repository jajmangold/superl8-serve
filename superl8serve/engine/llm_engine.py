# SPDX-License-Identifier: MIT
"""LLMEngine — request queue + scheduler + runner, with an offline generate() API.

Arch-agnostic: it drives any registered CausalLM. Build it from a ModelConfig + a
weights dict (the `.superl8` loader output, or an HF state dict). Multi-GPU (PP +
MoE-EP, never TP) is a later layer; KV storage is `PagedKVCache` -- int8
quantize-on-write, block-table addressed, one batched decode launch per step.
"""

from __future__ import annotations

import itertools
import time

import torch.nn as nn

from ..models.base import ForwardContext
from ..models.cache import MLALatentCache, RecurrentStateCache
from ..models.config import ModelConfig
from ..models.registry import build_model
from ..models.weights import consume_on_merge
from .cuda_graph import cuda_graph_enabled_by_env, layer_graph_enabled_by_env
from .decode_strategy import DiffusionDecodeStrategy
from .kv_cache import PagedKVCache
from .model_runner import EngineRunner
from .scheduler import Scheduler
from .sequence import SamplingParams, Sequence, Status


class LLMEngine:
    def __init__(
        self,
        cfg: ModelConfig,
        weights: dict,
        *,
        device="cuda",
        max_num_seqs: int = 16,
        max_len: int = 2048,
        max_batch_tokens: int = 8192,
        eos_id: int | None = None,
        enable_cuda_graph: bool | None = None,
        num_diffusion_steps: int = 8,
        chunked_prefill_size: int = 0,
        spec_decode: bool | None = None,
        cuda_graph_batch_buckets: tuple[int, ...] | None = None,
        consume_weights: bool = False,
        cache_format: str = "int8",
    ):
        self.cfg = cfg
        self.device = device
        self.eos_id = eos_id
        # Context window: the KV cache is sized for exactly `max_len` positions per
        # slot, so a prompt longer than this would write KV out of range and a decode
        # past it overflows the block table. `add_request` rejects over-length prompts
        # up front (S2) and every Sequence carries it so decode stops at the boundary.
        self.max_len = max_len
        # `consume_weights=True` lets the qkv/gate_up merges POP their source rows out of
        # `weights` as they build, bounding the merge transient so a card-filling 27B
        # fits on one 16 GiB GPU. It MUTATES `weights`, so only pass it when the caller
        # owns the dict and will not reuse it (the `.superl8` load path). Default off keeps
        # the dict intact for callers that build a second model from it.
        if consume_weights:
            with consume_on_merge():
                self.model = self._build(cfg, weights, device)
        else:
            self.model = self._build(cfg, weights, device)
        self._diffusion_strategy = (
            DiffusionDecodeStrategy(num_steps=num_diffusion_steps)
            if cfg.decode_strategy == "diffusion"
            else None
        )
        graph_wanted = (
            cuda_graph_enabled_by_env() if enable_cuda_graph is None else enable_cuda_graph
        )
        # Every graphed-execution object pins its own extra, never-freed cache
        # slot for padding rows (each has an independent `_scratch()` that calls
        # `cache.alloc()` once and holds it forever -- see kv_cache.py `alloc`
        # and cuda_graph.py's three `_scratch` implementations). `EngineRunner`
        # can construct up to THREE such objects at once: `GraphedPrefill`
        # (prefill path), `GraphedDecodeLayers` (per-layer decode graphs, the
        # default), and `GraphedDecode` (whole-step decode graph, always built
        # alongside layer graphs as its fallback -- see `decode()`'s `try_decode`
        # waterfall). All three gate on the same `enable_cuda_graph` flag, and
        # layer graphs are on by default too, so the worst case is all three
        # live simultaneously. Reserving only one spare slot (the original fix)
        # starves whichever mechanism allocates its scratch slot second once
        # `max_num_seqs` real sequences are also running, crashing with
        # `PagedKVCache: no free slots` right at peak concurrency -- the exact
        # condition production traffic is guaranteed to hit. Reserve one spare
        # slot per mechanism that can independently want one; the scheduler
        # itself still never admits more than `max_num_seqs` running sequences,
        # so real capacity is never reduced.
        _extra_graph_slots = 3 if (graph_wanted and layer_graph_enabled_by_env()) else (
            1 if graph_wanted else 0
        )
        num_slots = max_num_seqs + _extra_graph_slots
        if cfg.latent_attention:
            self.cache = MLALatentCache(
                cfg.num_hidden_layers,
                num_slots,
                cfg.mla_cache_dim(),
                max_len,
                device=device,
            )
        else:
            cache_kwargs = {}
            if cache_format in ("k8v3", "k8v8"):
                # K8V3 = int8 K + 3-bit Lloyd-Max V; K8V8 = int8 K + int8 V (the
                # fallback). Both are applied to the KV-bearing layers only. For a
                # qwen3_5 hybrid that is the full-attention subset (the DeltaNet
                # layers carry recurrent state, never KV) — sizing storage for
                # exactly those layers is what makes 262k multi-slot feasible.
                kv_layers = [
                    i for i in range(cfg.num_hidden_layers) if cfg.attention_kind(i) == "full"
                ]
                if not kv_layers:
                    raise ValueError(
                        f"cache_format={cache_format!r} but the config has no full-attention layers"
                    )
                cache_kwargs = dict(
                    kv_layers=kv_layers, v_quant="lloydmax3" if cache_format == "k8v3" else "int8"
                )
            elif cache_format != "int8":
                raise ValueError(f"cache_format must be 'int8', 'k8v8' or 'k8v3', got {cache_format!r}")
            self.cache = PagedKVCache(
                cfg.num_hidden_layers,
                num_slots,
                cfg.num_key_value_heads,
                max_len,
                cfg.resolved_head_dim(),
                device=device,
                **cache_kwargs,
            )
        self.scheduler = Scheduler(
            self.cache, max_num_seqs=max_num_seqs, max_batch_tokens=max_batch_tokens, eos_id=eos_id
        )
        self.lin_cache = RecurrentStateCache()
        self.runner = EngineRunner(
            self.model,
            self.cache,
            device=device,
            enable_cuda_graph=enable_cuda_graph,
            lin_cache=self.lin_cache,
            chunked_prefill_size=chunked_prefill_size,
            spec_decode=spec_decode,
            eos_id=eos_id,
            cuda_graph_batch_buckets=cuda_graph_batch_buckets,
        )
        self._ids = itertools.count()
        self._out: dict[int, Sequence] = {}
        # Optional telemetry sink (superl8serve.metrics.StatsCollector). The API layer
        # sets this; offline `generate()` leaves it None. `step()` records only
        # host-side ints + a perf_counter span into it -- never a GPU sync.
        self.stats = None

    @staticmethod
    def _build(cfg, weights, device) -> nn.Module:
        """Build the model, placing it on `device`.

        With a multi-GPU expert shard (`cfg.expert_to_gpu`), the expert weights
        were already placed on their owning GPUs at LOAD time (gguf_state_dict
        `expert_device`), and the shared (non-expert) tensors live on `device` —
        so the blanket `.to(device)` is skipped to avoid collapsing the shard.
        The model is still built CPU-first, and `build_model` never places
        tensors (only casts dtype), so any CPU-resident buffers/params (RoPE
        cos/sin tables, default-constructed RMSNorm gains) must be moved to
        `device` explicitly — feeding CPU pointers to CUDA kernels is an async
        illegal-access that surfaces at the next host sync. Only CPU-resident
        tensors move; the cuda:1 experts are left untouched.
        """
        model = build_model(cfg, weights).eval()
        if getattr(cfg, "expert_to_gpu", None):
            for m in model.modules():
                for n, b in list(m._buffers.items()):
                    if b is not None and b.device.type == "cpu":
                        m._buffers[n] = b.to(device)
                for n, p in list(m._parameters.items()):
                    if p is not None and p.device.type == "cpu":
                        p.data = p.data.to(device)
            return model
        return model.to(device)

    def add_request(self, prompt_ids: list[int], params: SamplingParams | None = None) -> int:
        # Reject an over-length prompt at admission (S2) rather than letting the
        # scheduler admit a single over-budget prefill (its `if batch` short-circuit
        # admits the first waiter unconditionally) into an OOM / out-of-range KV write.
        if self.max_len is not None and len(prompt_ids) > self.max_len:
            raise ValueError(f"prompt length {len(prompt_ids)} exceeds max_len {self.max_len}")
        seq = Sequence(next(self._ids), list(prompt_ids), params or SamplingParams())
        seq.max_len = self.max_len
        self.scheduler.add(seq)
        self._out[seq.seq_id] = seq
        return seq.seq_id

    def sequence(self, seq_id: int) -> Sequence:
        return self._out[seq_id]

    def forget(self, seq_id: int) -> None:
        """Drop bookkeeping for a finished request. Needed by long-lived callers (the
        API server) that generate() never returns to -- without this, `_out` would
        retain every request's Sequence for the life of the process."""
        self._out.pop(seq_id, None)

    def cancel(self, seq_id: int) -> bool:
        """Cancel a request by ``seq_id`` and release its resources.

        A WAITING request holds no KV slot, so it is dropped with no cache work. A
        RUNNING request is removed from the scheduler (freeing its paged KV) and its
        runner per-slot decode state -- the recurrent ``lin_cache`` row and the MTP
        draft head's ``mtp_cache`` slot -- is cleared so a later request that reuses
        the slot never inherits stale state. Returns False when the request is
        unknown (idempotent). The Sequence is forgotten from `_out`."""
        seq = self._out.get(seq_id)
        if seq is None:
            return False
        slot = seq.slot
        if not self.scheduler.cancel(seq_id):
            return False
        if slot >= 0:
            self.lin_cache.clear_slot(slot)
            self.runner.mtp_cache.clear_slot(slot)
        self.forget(seq_id)
        return True

    def step(self):
        batch, is_prefill = self.scheduler.schedule()
        if not batch:
            return
        # Token count for this step from host-side ints we already hold: prefill
        # processes every prompt token, decode emits exactly one token per row.
        # (No `.item()`/`.cpu()` -- the existing `int(tok)` below already
        # materialises decode tokens on host, so the perf_counter span honestly
        # reflects real GPU time without any *added* sync.)
        step_t0 = time.perf_counter() if self.stats is not None else 0.0
        num_tokens = sum(seq.num_prompt for seq in batch) if is_prefill else len(batch)
        if self._diffusion_strategy is not None and is_prefill:
            for seq in batch:
                total_len = seq.num_prompt + seq.params.max_tokens
                self.cache.ensure_capacity([seq.slot], [total_len])
                seq.length = total_len
                ctx = ForwardContext(
                    is_prefill=True, kv_cache=self.cache, lin_cache=self.lin_cache, slots=[seq.slot]
                )
                out_tokens = self._diffusion_strategy.generate(
                    self.model, self.cache, self.device, seq, ctx
                )
                seq.output_ids.extend(out_tokens)
                self.cache.store_prefix(seq.prompt_ids, seq.slot)
        else:
            toks = self.runner.prefill(batch) if is_prefill else self.runner.decode(batch)
            if toks is not None:
                for seq, tok in zip(batch, toks):
                    seq.output_ids.append(int(tok))
            if is_prefill:
                for seq in batch:
                    self.cache.store_prefix(seq.prompt_ids, seq.slot)
        self.scheduler.postprocess(batch, is_prefill)
        if self.stats is not None:
            self.stats.record_step(
                is_prefill=is_prefill,
                num_tokens=num_tokens,
                running=len(self.scheduler.running),
                waiting=len(self.scheduler.waiting),
                dt=time.perf_counter() - step_t0,
            )

    def encode(self, prompt_ids: list[int]) -> list[float]:
        seq_id = self.add_request(prompt_ids)
        batch, is_prefill = self.scheduler.schedule()
        if not batch:
            self.forget(seq_id)
            return []
        pooled = self.runner.encode(batch)
        for seq in batch:
            self.cache.store_prefix(seq.prompt_ids, seq.slot)
            seq.status = Status.FINISHED
        self.scheduler.postprocess(batch, is_prefill)
        self.forget(seq_id)
        return pooled[0].cpu().tolist()

    def generate(
        self, prompts: list[list[int]], params: SamplingParams | None = None
    ) -> list[list[int]]:
        """Offline batched generation: returns the output token ids per prompt.

        Auto-forgets each collected Sequence from `_out` before returning: offline
        callers loop on `generate()` and never call `forget()` themselves, so without
        this every completed request's Sequence (prompt + output ids) would be pinned
        in `_out` for the life of the process — unbounded host-memory growth. The
        long-lived API path (`add_request` + explicit `forget`) is unaffected: it
        does not go through `generate()`."""
        ids = [self.add_request(p, params) for p in prompts]
        while self.scheduler.has_work():
            self.step()
        # Materialise the outputs, then drop the Sequences from `_out`. The returned
        # lists stay alive via these references after their Sequence is forgotten.
        outputs = [self._out[i].output_ids for i in ids]
        for i in ids:
            self.forget(i)
        return outputs
