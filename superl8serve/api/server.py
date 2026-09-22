# SPDX-License-Identifier: MIT
"""Runnable entrypoint: serve a `.superl8` checkpoint behind the OpenAI-compatible API.

    python -m superl8serve.api.server --model qwen3-8b.superl8 --tokenizer Qwen/Qwen3-8B

The `.superl8` file carries weights + architecture config, not tokenizer files, so
`--tokenizer` points at an HF repo id (or local dir) to load the tokenizer /
chat template from -- the base model repo, or the superl8-quant repo if it mirrors one.
`--chat-template` optionally overrides the tokenizer's own embedded Jinja template
with one loaded from a file (matches vLLM's `--chat-template` flag); the override is
rendered through the same sandboxed Jinja environment `apply_chat_template` always
uses, so this grants a custom template no more than the model's own gets.
Needs the `serve` extra: `pip install -e ".[serve]"`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time

from ..engine import LLMEngine
from ..loader import checkpoint_info, load_superl8_state_dict
from ..models import ModelConfig
from .app import create_app


def resolve_idle_coalescing(
    *, max_num_seqs: int, idle_coalesce_ms: float, idle_coalesce_target: int | None
) -> tuple[float, int | None]:
    """Validate CLI policy before the expensive model load and resolve its target."""
    if isinstance(max_num_seqs, bool) or not isinstance(max_num_seqs, int) or max_num_seqs < 1:
        raise ValueError("max-num-seqs must be a positive integer")
    if not math.isfinite(idle_coalesce_ms) or idle_coalesce_ms < 0:
        raise ValueError("idle-coalesce-ms must be finite and >= 0")
    if idle_coalesce_ms == 0:
        if idle_coalesce_target is not None:
            raise ValueError("idle-coalesce-target requires idle-coalesce-ms > 0")
        return 0.0, None
    target = max_num_seqs if idle_coalesce_target is None else idle_coalesce_target
    if target < 1 or target > max_num_seqs:
        raise ValueError("idle-coalesce-target must be between 1 and max-num-seqs")
    return idle_coalesce_ms, target


def parse_batch_buckets(value: str) -> tuple[int, ...]:
    """Parse the ``--cuda-graph-batch-buckets`` spec: a comma-separated list of
    strictly positive integers. Returns the sorted, de-duplicated ascending
    tuple. Raises ValueError for an empty, malformed, or nonpositive spec --
    the CLI boundary (``_batch_buckets_arg``) turns that into an argparse error."""
    buckets = []
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            raise ValueError(f"empty batch bucket in {value!r}")
        try:
            b = int(token)
        except ValueError:
            raise ValueError(
                f"invalid batch bucket {token!r} in {value!r} "
                "(expected a positive integer)"
            ) from None
        if b <= 0:
            raise ValueError(f"batch bucket must be positive, got {b} in {value!r}")
        buckets.append(b)
    return tuple(sorted(set(buckets)))


def _batch_buckets_arg(value: str) -> tuple[int, ...]:
    """argparse ``type=`` for ``--cuda-graph-batch-buckets``. The
    ``> max_num_seqs`` bound can't be checked here (that's a sibling flag), so
    ``main`` enforces it after both values are known."""
    try:
        return parse_batch_buckets(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def _batch_buckets_over_max(buckets, max_num_seqs) -> list[int]:
    """Requested buckets the scheduler can never admit (``> max_num_seqs``), or
    [] when ``buckets`` is None / every value is in bounds."""
    if buckets is None:
        return []
    return [b for b in buckets if b > max_num_seqs]


def load_engine(model_path: str, *, device: str = "cuda",
                max_num_seqs: int = 16, max_len: int = 2048, eos_id: int | None = None,
                spec_decode: bool | None = None,
                cuda_graph_batch_buckets: tuple[int, ...] | None = None,
                gguf_force_w8: bool = False,
                weight_stationary: bool = False,
                expert_map: dict[int, int] | None = None,
                local_gpu: int | None = None,
                cache_format: str = "int8") -> LLMEngine:
    """Architecture is auto-detected from the checkpoint; see the README Quickstart.

    ``spec_decode`` (None → `SUPERL8SERVE_MTP_SPEC` env, default off) enables the engine's
    speculative-decode path (n-gram cascade + MTP-head fallback) for BOTH formats.

    ``cuda_graph_batch_buckets`` (None → the built-in 1,2,4,...,128 ladder) requests
    the CUDA-graph capture batch buckets; the graph objects filter the list against
    ``max_num_seqs`` and the CLI rejects any over-limit value up front.

    ``cache_format`` selects the KV cache layout: ``"int8"``/``"k8v8"`` (both K and
    V at int8, the default) or ``"k8v3"`` (K stays int8, V drops to Lloyd-Max 3-bit
    codes on the model's full-attention layers -- see ``PagedKVCache``'s
    ``v_quant="lloydmax3"``). Was previously only reachable on the .gguf path via
    ``load_gguf_engine`` directly; wired through here for both formats
    (superl8-serve#459 follow-up)."""
    # Suffix branch: a `.gguf` loads NATIVELY (GGUF-KV → ModelConfig + resident
    # k-quant weights, no `.superl8`); anything else takes the legacy `.superl8` path. Both
    # formats coexist through the migration — no `.superl8` code is removed here (P5).
    if model_path.endswith(".gguf"):
        from ..gguf_native import load_gguf_engine

        return load_gguf_engine(
            model_path, device=device, max_num_seqs=max_num_seqs, max_len=max_len,
            eos_id=eos_id, spec_decode=spec_decode,
            cuda_graph_batch_buckets=cuda_graph_batch_buckets,
            force_w8=gguf_force_w8,
            weight_stationary=weight_stationary,
            expert_map=expert_map, local_gpu=local_gpu,
            cache_format=cache_format,
        )
    if gguf_force_w8:
        raise ValueError("gguf_force_w8 is only valid for .gguf models")
    info = checkpoint_info(model_path)
    meta_cfg = info["meta"]["config"]
    # Honor the arch stored at conversion time. The meta config is a ModelConfig dump
    # (carries `arch`, e.g. "qwen3_5_text") but often lacks HF `model_type`/`architectures`,
    # so from_hf's derivation alone would fall through to "unknown". Prefer the stored arch.
    cfg = ModelConfig.from_hf(meta_cfg, arch=meta_cfg.get("arch") or None)
    weights = load_superl8_state_dict(model_path, device=device)
    # Trust the weights over the recorded config: some converted Qwen3-family checkpoints
    # recorded qk_norm=False yet DO carry per-head q_norm/k_norm weights. Qwen3's QK-norm
    # is load-bearing (skipping it feeds un-normalized Q/K into RoPE -> garbage output).
    if not getattr(cfg, "qk_norm", False) and any(".q_norm.weight" in n for n in weights):
        try:
            cfg.qk_norm = True
        except Exception:  # frozen dataclass
            import dataclasses
            cfg = dataclasses.replace(cfg, qk_norm=True)
    # Serve-time MoE topology overrides (weight-stationary / expert shard). These
    # are runtime concerns, not checkpoint properties: they are applied here, at
    # the API seam, and never persisted back into the .superl8 meta.
    if weight_stationary or expert_map:
        import dataclasses
        # Normalize to int keys so downstream int lookups (expert_device,
        # WeightStationaryMoE buffer selection) never silently miss a
        # JSON-string-keyed map and drop every expert onto local_gpu.
        if expert_map:
            expert_map = {int(k): int(v) for k, v in expert_map.items()}
        cfg = dataclasses.replace(
            cfg,
            use_weight_stationary_moe=weight_stationary or bool(expert_map),
            expert_to_gpu=expert_map,
            local_gpu=local_gpu,
        )
    # `weights` was just loaded from the `.superl8` file and is used only to build this one
    # engine, so let the qkv/gate_up merges consume (pop) their source rows as they go --
    # bounding the merge transient so a single-card 27B fits in 16 GiB.
    return LLMEngine(cfg, weights, device=device, max_num_seqs=max_num_seqs,
                     max_len=max_len, eos_id=eos_id, consume_weights=True,
                     spec_decode=spec_decode,
                     cuda_graph_batch_buckets=cuda_graph_batch_buckets,
                     cache_format=cache_format)


def _human_count(n: int) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return str(n)


def build_banner(engine, tokenizer, *, served_model_name, chat_template, load_time_s, max_len,
                 idle_coalesce_ms=0.0, idle_coalesce_target=None):
    """Assemble the one-time startup banner facts from a live engine: model dims,
    GPU, a weights/KV/free VRAM memory breakdown, and the serving config (incl.
    CUDA-graph state + captured batch buckets). Read once at startup -- the one
    place a couple of `.numel()`/`mem_get_info` calls are fine (not the hot loop)."""
    from .. import __version__
    from ..metrics import gpu_stats

    cfg = engine.cfg
    model = engine.model
    tensors = list(itertools.chain(model.parameters(), model.buffers()))
    params = sum(t.numel() for t in tensors)
    weights_bytes = sum(t.numel() * t.element_size() for t in tensors)

    cache = getattr(engine, "cache", None)
    kv_blocks = getattr(cache, "num_blocks", 0)
    kv_bytes = 0
    for attr in ("k_cache", "v_cache", "k_scale", "v_scale"):
        t = getattr(cache, attr, None)
        if t is not None:
            kv_bytes += t.numel() * t.element_size()

    free_gib = 0.0
    try:
        import torch

        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info(0)
            free_gib = free / 1024**3
    except Exception:
        pass

    graphed = getattr(engine.runner, "graphed", None)
    cuda_graph = graphed is not None
    if cuda_graph and hasattr(graphed, "status"):
        graph_status = graphed.status()
    elif cuda_graph:
        # Keep banner compatibility with lightweight/fake graph objects used by
        # callers and tests that predate the qualification status method.
        graph_status = {
            "batch_buckets": list(getattr(graphed, "batch_buckets", ()) or ())
        }
    else:
        graph_status = {}
    captured = graph_status.get("batch_buckets", []) if cuda_graph else []
    layer_graphs = getattr(engine.runner, "layer_graphs", None)
    layer_status = (
        layer_graphs.status()
        if layer_graphs is not None and hasattr(layer_graphs, "status")
        else None
    )

    return {
        "version": __version__,
        "model_name": served_model_name,
        "arch": cfg.arch,
        "quant": f"int8 W8A8 dp4a (weight_bits={cfg.weight_bits})",
        "compute": "sm_70",
        "gpu": gpu_stats(),
        "load_time_s": load_time_s,
        "dims": {
            "params": params,
            "params_str": _human_count(params),
            "layers": cfg.num_hidden_layers,
            "hidden": cfg.hidden_size,
            "num_heads": cfg.num_attention_heads,
            "num_kv_heads": cfg.num_key_value_heads,
            "head_dim": cfg.resolved_head_dim(),
            "vocab": cfg.vocab_size,
            "max_len": max_len,
        },
        "memory": {
            "weights_gib": weights_bytes / 1024**3,
            "kv_gib": kv_bytes / 1024**3,
            "kv_blocks": kv_blocks,
            "free_gib": free_gib,
        },
        "config": {
            "max_num_seqs": engine.scheduler.max_num_seqs,
            "max_len": max_len,
            "cuda_graph": cuda_graph,
            "captured_batch_sizes": captured,
            "cuda_graph_supported": graph_status.get("supported") if cuda_graph else False,
            "cuda_graph_unsupported_reason": (
                graph_status.get("unsupported_reason") if cuda_graph else None
            ),
            "cuda_graph_captured_graphs": graph_status.get("captured_graphs", 0),
            "cuda_graph_captures": graph_status.get("captures", 0),
            "cuda_graph_replays": graph_status.get("replays", 0),
            "cuda_graph_misses": graph_status.get("misses", {}),
            "layer_graph": layer_status,
            "tokenizer_id": getattr(tokenizer, "name_or_path", None),
            "chat_template": bool(chat_template) or bool(getattr(tokenizer, "chat_template", None)),
            "weight_runtime": getattr(engine, "weight_runtime", "checkpoint-default"),
            "idle_coalesce_ms": idle_coalesce_ms,
            "idle_coalesce_target": idle_coalesce_target,
        },
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Serve a .superl8 checkpoint over an OpenAI-compatible API")
    ap.add_argument("--model", required=True, help="path to a .superl8 checkpoint")
    ap.add_argument("--tokenizer", default=None,
                    help="HF repo id or local dir with the tokenizer (default: --model)")
    ap.add_argument("--chat-template", default=None,
                    help="path to a Jinja file overriding the tokenizer's own chat template")
    ap.add_argument("--tool-parser", default="hermes",
                    help="superl8serve.tool_calls parser for `tool_choice=auto` tool-call "
                         "extraction (default: hermes, Qwen's native tool-calling format)")
    ap.add_argument("--served-model-name", default=None, help="default: --model")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument(
        "--weight-stationary",
        action="store_true",
        help="use the Phase-3 weight-stationary MoE path (per-expert staging "
             "buffers + skip-empty) for MoE models",
    )
    ap.add_argument(
        "--expert-map",
        default=None,
        help="expert shard map as JSON `{expert_id: gpu_index, ...}`; when set, MoE "
             "layers route remote experts over the compressed transport seam",
    )
    ap.add_argument(
        "--local-gpu",
        type=int,
        default=None,
        help="owning GPU index for this process in an expert shard (default: 0)",
    )
    ap.add_argument(
        "--gguf-force-w8",
        action="store_true",
        help="transcode GGUF k-quant linear weights once at load to per-row W8 for throughput",
    )
    ap.add_argument(
        "--idle-coalesce-ms",
        type=float,
        default=0.0,
        help="default-off idle-start batching deadline in milliseconds",
    )
    ap.add_argument(
        "--idle-coalesce-target",
        type=int,
        default=None,
        help="generation target for idle-start batching (default: max-num-seqs when enabled)",
    )
    ap.add_argument("--spec-decode", action="store_true", default=None,
                    help="enable speculative decode (n-gram cascade + MTP-head "
                         "fallback); default off (or set SUPERL8SERVE_MTP_SPEC=1)")
    ap.add_argument("--cuda-graph-batch-buckets", type=_batch_buckets_arg,
                    default=None, metavar="N1,N2,...",
                    help="comma-separated positive CUDA-graph capture batch "
                         "buckets (default: 1,2,4,...,128); every value must be "
                         "<= --max-num-seqs")
    ap.add_argument("--cache-format", choices=["int8", "k8v8", "k8v3"],
                    default="int8",
                    help="KV cache layout: int8/k8v8 (both K and V at int8, "
                         "default) or k8v3 (K int8, V Lloyd-Max 3-bit on "
                         "full-attention layers -- smaller footprint, more "
                         "headroom for concurrent sequences / longer context)")
    return ap


def main() -> None:
    ap = _build_arg_parser()
    args = ap.parse_args()
    try:
        args.idle_coalesce_ms, args.idle_coalesce_target = resolve_idle_coalescing(
            max_num_seqs=args.max_num_seqs,
            idle_coalesce_ms=args.idle_coalesce_ms,
            idle_coalesce_target=args.idle_coalesce_target,
        )
    except ValueError as exc:
        ap.error(str(exc))
    if args.cuda_graph_batch_buckets is not None:
        over = _batch_buckets_over_max(args.cuda_graph_batch_buckets, args.max_num_seqs)
        if over:
            ap.error(
                f"argument --cuda-graph-batch-buckets: bucket(s) {over} exceed "
                f"--max-num-seqs {args.max_num_seqs}; raise --max-num-seqs or "
                "drop those buckets"
            )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    chat_template = None
    if args.chat_template is not None:
        with open(args.chat_template, encoding="utf-8") as f:
            chat_template = f.read()

    _t0 = time.perf_counter()
    expert_map = json.loads(args.expert_map) if args.expert_map else None
    engine = load_engine(args.model, device=args.device, max_num_seqs=args.max_num_seqs,
                         max_len=args.max_len, eos_id=tokenizer.eos_token_id,
                         spec_decode=args.spec_decode,
                         cuda_graph_batch_buckets=args.cuda_graph_batch_buckets,
                         gguf_force_w8=args.gguf_force_w8,
                         weight_stationary=args.weight_stationary,
                         expert_map=expert_map, local_gpu=args.local_gpu,
                         cache_format=args.cache_format)
    load_time_s = time.perf_counter() - _t0

    from ..metrics import StatsCollector, render_banner, start_heartbeat
    from rich.console import Console

    served = args.served_model_name or args.model
    banner = build_banner(engine, tokenizer, served_model_name=served,
                          chat_template=chat_template, load_time_s=load_time_s,
                          max_len=args.max_len,
                          idle_coalesce_ms=args.idle_coalesce_ms,
                          idle_coalesce_target=args.idle_coalesce_target)
    stats = StatsCollector()
    stats.set_banner(banner)

    console = Console()
    console.print(render_banner(banner))
    start_heartbeat(stats, console=console, interval=5.0)

    app = create_app(engine, tokenizer, served_model_name=served,
                     chat_template=chat_template, tool_parser=args.tool_parser, stats=stats,
                     idle_coalesce_ms=args.idle_coalesce_ms,
                     idle_coalesce_target=args.idle_coalesce_target)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
