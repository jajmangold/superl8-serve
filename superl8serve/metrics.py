# SPDX-License-Identifier: MIT
"""StatsCollector — cheap, sync-free serving telemetry for superl8-serve.

The design contract (issue #182): **zero per-step GPU syncs**. Every hot-loop
record here counts host-side ints the engine already has (token counts, sequence
counts) and times wall-clock with `time.perf_counter()`. GPU utilisation / VRAM /
power / temperature are read via NVML (or `nvidia-smi`) **only** on the heartbeat /
snapshot path, never inside `LLMEngine.step()`, and even there they are throttled so
a 2 Hz TUI poll can't hammer the driver.

Three record surfaces:
  * `record_step(is_prefill, num_tokens, running, waiting, dt)` — one engine step.
    Feeds the rolling prefill/decode throughput windows and the lifetime token
    counters. Called from `LLMEngine.step()`.
  * `record_request_start / record_first_token / record_request_finish` — the
    per-request lifecycle, called from the API worker (`api/runtime.py`). Feeds
    TTFT, inter-token-latency (ITL), end-to-end latency percentiles and the
    request/finish-reason counters.
  * `snapshot()` — a plain-dict view for `GET /metrics`, the heartbeat log line and
    the `superl8serve-top` TUI. Dependency-light and fast.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank percentile (q in [0,1]) over an already-sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = int(round(q * (len(sorted_vals) - 1)))
    return sorted_vals[max(0, min(idx, len(sorted_vals) - 1))]


class ThroughputWindow:
    """Rolling tokens/second over a fixed wall-clock window.

    Each `add(tokens, dt)` records `tokens` produced across a step that took `dt`
    seconds, stamped at ingest. `rate()` sums tokens over the trailing `window_s`
    and divides by the summed step-time in that window — instantaneous throughput
    that decays cleanly to 0 once the engine goes idle (old samples age out)."""

    __slots__ = ("window_s", "_samples")

    def __init__(self, window_s: float = 5.0):
        self.window_s = window_s
        # (ingest_time, tokens, dt)
        self._samples: deque[tuple[float, int, float]] = deque()

    def add(self, tokens: int, dt: float, *, now: float | None = None) -> None:
        now = time.perf_counter() if now is None else now
        self._samples.append((now, tokens, dt))
        self._evict(now)

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def rate(self, *, now: float | None = None) -> float:
        now = time.perf_counter() if now is None else now
        self._evict(now)
        tok = sum(s[1] for s in self._samples)
        dt = sum(s[2] for s in self._samples)
        return tok / dt if dt > 0 else 0.0


@dataclass
class _ReqState:
    start: float
    prompt_tokens: int
    first_token_at: float | None = None
    sampling: dict = field(default_factory=dict)


class StatsCollector:
    """Thread-safe (one lock) collector. Cheap enough to call every engine step."""

    def __init__(self, *, window_s: float = 5.0, latency_window: int = 512):
        self._lock = threading.Lock()
        self._t0 = time.perf_counter()
        self._wall0 = time.time()

        # lifetime counters
        self.total_requests = 0
        self.finished_requests = 0
        self.prompt_tokens_total = 0
        self.output_tokens_total = 0
        self.prefill_tokens_total = 0
        self.finish_reasons: Counter[str] = Counter()

        # rolling throughput
        self.prefill_win = ThroughputWindow(window_s)
        self.decode_win = ThroughputWindow(window_s)

        # per-request latency samples (ms), bounded
        self.ttft_ms: deque[float] = deque(maxlen=latency_window)
        self.itl_ms: deque[float] = deque(maxlen=latency_window)
        self.e2e_ms: deque[float] = deque(maxlen=latency_window)
        self._inflight: dict[int, _ReqState] = {}

        # live gauges (updated each step, host-side ints — no sync)
        self.running = 0
        self.waiting = 0

        # Idle-start batching telemetry (issue #369). Updated once per transition,
        # never in the active decode hot loop.
        self.batching_idle_coalesce_ms = 0.0
        self.batching_target: int | None = None
        self.batching_events = 0
        self.batching_wait_ms_total = 0.0
        self.batching_effective_total = 0
        self.batching_last_wait_ms = 0.0
        self.batching_last_effective = 0
        self.batching_target_hits = 0
        self.batching_deadline_expirations = 0

        # attached engine/cache for KV% and the banner (set once, at wiring time)
        self._cache = None
        self.banner: dict = {}

        # throttled GPU snapshot cache
        self._gpu_cache: dict | None = None
        self._gpu_cache_at: float = 0.0
        self._gpu_ttl = 0.9

    # ---- wiring -------------------------------------------------------------
    def attach_engine(self, engine) -> None:
        """Grab the KV cache off an LLMEngine (if present) for usage %."""
        self._cache = getattr(engine, "cache", None)

    def set_banner(self, banner: dict) -> None:
        with self._lock:
            self.banner = dict(banner)

    def configure_batching(
        self, *, idle_coalesce_ms: float, idle_coalesce_target: int | None
    ) -> None:
        with self._lock:
            self.batching_idle_coalesce_ms = idle_coalesce_ms
            self.batching_target = idle_coalesce_target

    def record_idle_coalesce(
        self, *, wait_ms: float, effective_batch: int, target_hit: bool
    ) -> None:
        with self._lock:
            self.batching_events += 1
            self.batching_wait_ms_total += wait_ms
            self.batching_effective_total += effective_batch
            self.batching_last_wait_ms = wait_ms
            self.batching_last_effective = effective_batch
            if target_hit:
                self.batching_target_hits += 1
            else:
                self.batching_deadline_expirations += 1

    # ---- hot-loop records (host-side ints only) -----------------------------
    def record_step(
        self, *, is_prefill: bool, num_tokens: int, running: int, waiting: int, dt: float
    ) -> None:
        now = time.perf_counter()
        with self._lock:
            self.running = running
            self.waiting = waiting
            if is_prefill:
                self.prefill_tokens_total += num_tokens
                self.prefill_win.add(num_tokens, dt, now=now)
            else:
                self.output_tokens_total += num_tokens
                self.decode_win.add(num_tokens, dt, now=now)

    def record_queue_depth(self, *, running: int, waiting: int) -> None:
        """Update the running/waiting gauges outside an engine step (e.g. after a
        cancel drained the scheduler). Same host-side ints `record_step` sets."""
        with self._lock:
            self.running = running
            self.waiting = waiting

    def record_request_start(self, req_id, *, prompt_tokens: int, sampling: dict | None = None):
        with self._lock:
            self.total_requests += 1
            self.prompt_tokens_total += prompt_tokens
            self._inflight[req_id] = _ReqState(
                start=time.perf_counter(),
                prompt_tokens=prompt_tokens,
                sampling=sampling or {},
            )

    def record_first_token(self, req_id) -> None:
        now = time.perf_counter()
        with self._lock:
            st = self._inflight.get(req_id)
            if st is not None and st.first_token_at is None:
                st.first_token_at = now
                self.ttft_ms.append((now - st.start) * 1e3)

    def record_request_finish(self, req_id, *, output_tokens: int, finish_reason: str) -> None:
        now = time.perf_counter()
        with self._lock:
            st = self._inflight.pop(req_id, None)
            self.finished_requests += 1
            self.finish_reasons[finish_reason] += 1
            if st is None:
                return
            e2e = (now - st.start) * 1e3
            self.e2e_ms.append(e2e)
            # inter-token latency: decode span / decode steps (excludes the TTFT gap)
            if st.first_token_at is not None and output_tokens > 1:
                decode_span = (now - st.first_token_at) * 1e3
                self.itl_ms.append(decode_span / (output_tokens - 1))

    # ---- snapshot -----------------------------------------------------------
    def _kv(self) -> dict:
        c = self._cache
        used = getattr(c, "used_blocks", None)
        total = getattr(c, "num_blocks", None)
        if callable(used):
            used = used()
        if used is None or not total:
            return {"used_blocks": 0, "total_blocks": 0, "usage_pct": 0.0}
        return {
            "used_blocks": int(used),
            "total_blocks": int(total),
            "usage_pct": 100.0 * used / total if total else 0.0,
        }

    def _gpu(self) -> dict | None:
        now = time.perf_counter()
        if self._gpu_cache is not None and (now - self._gpu_cache_at) < self._gpu_ttl:
            return self._gpu_cache
        stats = gpu_stats()
        self._gpu_cache = stats
        self._gpu_cache_at = now
        return stats

    def snapshot(self) -> dict:
        now = time.perf_counter()
        with self._lock:
            ttft = sorted(self.ttft_ms)
            itl = sorted(self.itl_ms)
            e2e = sorted(self.e2e_ms)
            snap = {
                "uptime_s": now - self._t0,
                "started_at": self._wall0,
                "running": self.running,
                "waiting": self.waiting,
                "throughput": {
                    "prefill_tok_s": self.prefill_win.rate(now=now),
                    "decode_tok_s": self.decode_win.rate(now=now),
                },
                "latency": {
                    "avg_ttft_ms": (sum(ttft) / len(ttft)) if ttft else 0.0,
                    "p50_ttft_ms": _percentile(ttft, 0.50),
                    "p95_ttft_ms": _percentile(ttft, 0.95),
                    "avg_itl_ms": (sum(itl) / len(itl)) if itl else 0.0,
                    "p50_ms": _percentile(e2e, 0.50),
                    "p99_ms": _percentile(e2e, 0.99),
                },
                "batching": {
                    "enabled": self.batching_idle_coalesce_ms > 0,
                    "idle_coalesce_ms": self.batching_idle_coalesce_ms,
                    "target_batch": self.batching_target,
                    "events": self.batching_events,
                    "avg_wait_ms": (
                        self.batching_wait_ms_total / self.batching_events
                        if self.batching_events
                        else 0.0
                    ),
                    "last_wait_ms": self.batching_last_wait_ms,
                    "avg_effective_batch": (
                        self.batching_effective_total / self.batching_events
                        if self.batching_events
                        else 0.0
                    ),
                    "last_effective_batch": self.batching_last_effective,
                    "target_hits": self.batching_target_hits,
                    "deadline_expirations": self.batching_deadline_expirations,
                },
                "counters": {
                    "total_requests": self.total_requests,
                    "finished_requests": self.finished_requests,
                    "inflight_requests": len(self._inflight),
                    "prompt_tokens": self.prompt_tokens_total,
                    "output_tokens": self.output_tokens_total,
                    "prefill_tokens": self.prefill_tokens_total,
                    "finish_reasons": dict(self.finish_reasons),
                },
                "model": dict(self.banner),
            }
        # KV + GPU read their own throttled paths, outside the counter lock.
        snap["kv_cache"] = self._kv()
        snap["gpu"] = self._gpu()
        return snap


# ---------------------------------------------------------------------------
# GPU telemetry — heartbeat / snapshot path ONLY. Never call from the hot loop.
# ---------------------------------------------------------------------------
_NVML_READY: bool | None = None


def _nvml():
    global _NVML_READY
    try:
        import pynvml
    except Exception:
        _NVML_READY = False
        return None
    if _NVML_READY is None:
        try:
            pynvml.nvmlInit()
            _NVML_READY = True
        except Exception:
            _NVML_READY = False
    return pynvml if _NVML_READY else None


def _gpu_via_nvml() -> dict | None:
    pynvml = _nvml()
    if pynvml is None:
        return None
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode()
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        except Exception:
            power = None
        try:
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            temp = None
        return {
            "name": name,
            "vram_used_mb": mem.used / 1024**2,
            "vram_total_mb": mem.total / 1024**2,
            "util_pct": float(util),
            "power_w": power,
            "temp_c": temp,
        }
    except Exception:
        return None


def _gpu_via_smi() -> dict | None:
    import shutil
    import subprocess

    if shutil.which("nvidia-smi") is None:
        return None
    q = "name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    row = out.stdout.strip().splitlines()[0].split(",")

    def _f(x):
        x = x.strip()
        try:
            return float(x)
        except ValueError:
            return None

    return {
        "name": row[0].strip(),
        "vram_used_mb": _f(row[1]),
        "vram_total_mb": _f(row[2]),
        "util_pct": _f(row[3]),
        "power_w": _f(row[4]) if len(row) > 4 else None,
        "temp_c": _f(row[5]) if len(row) > 5 else None,
    }


def gpu_stats() -> dict | None:
    """Best-effort single-GPU telemetry: NVML first, then `nvidia-smi`, else None.
    Falls back to `torch.cuda` for name/VRAM if neither exposes utilisation."""
    stats = _gpu_via_nvml() or _gpu_via_smi()
    if stats is not None:
        return stats
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            return {
                "name": torch.cuda.get_device_name(0),
                "vram_used_mb": (total - free) / 1024**2,
                "vram_total_mb": total / 1024**2,
                "util_pct": None,
                "power_w": None,
                "temp_c": None,
            }
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Presentation helpers (rich). Imported lazily so `metrics` stays dependency-light
# for the `GET /metrics` path and the unit tests — only the banner / heartbeat /
# TUI paths need `rich` (declared in the `serve` extra).
# ---------------------------------------------------------------------------
def _threshold_color(pct: float, *, warn: float = 70.0, crit: float = 90.0) -> str:
    """green under `warn`, yellow up to `crit`, red above — for KV% / util bars."""
    if pct >= crit:
        return "red"
    if pct >= warn:
        return "yellow"
    return "green"


def bar(pct: float, width: int = 20, *, warn: float = 70.0, crit: float = 90.0) -> str:
    """A rich-markup unicode meter, colour-graded by threshold."""
    pct = max(0.0, min(100.0, float(pct)))
    filled = int(round(pct / 100.0 * width))
    color = _threshold_color(pct, warn=warn, crit=crit)
    return f"[{color}]{'█' * filled}[/]{'░' * (width - filled)}"


def _fmt_gpu_kind(name: str) -> str:
    n = (name or "").lower()
    if "cmp" in n:
        return "CMP 100-210 (GV100, tensor-cores firmware-gimped)"
    if "v100" in n:
        return "Tesla V100 (GV100)"
    return name or "unknown GPU"


def render_banner(banner: dict):
    """Startup banner: a rich Panel with a two-column table of model / GPU / memory /
    config facts. `banner` is the dict assembled in `api/server.py`."""
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    m = banner
    t = Table.grid(padding=(0, 2))
    t.add_column(justify="right", style="bold cyan", no_wrap=True)
    t.add_column(style="white")

    def row(k, v):
        if v is not None and v != "":
            t.add_row(k, str(v))

    row("model", m.get("model_name"))
    row("arch", m.get("arch"))
    row("quant", m.get("quant"))
    gpu = m.get("gpu") or {}
    if gpu:
        vram = gpu.get("vram_total_mb")
        vram_s = f"{vram / 1024:.1f} GiB" if vram else "?"
        row(
            "gpu",
            f"{_fmt_gpu_kind(gpu.get('name', ''))}  •  {vram_s}  •  {m.get('compute', 'sm_70')}",
        )
    dims = m.get("dims") or {}
    if dims:
        row(
            "dims",
            f"{dims.get('params_str', '?')} params  •  {dims.get('layers')} layers  •  "
            f"hidden {dims.get('hidden')}  •  {dims.get('num_heads')}Q/"
            f"{dims.get('num_kv_heads')}KV heads  •  head_dim {dims.get('head_dim')}",
        )
        row("vocab / max_len", f"{dims.get('vocab')} / {dims.get('max_len')}")
    mem = m.get("memory") or {}
    if mem:
        row(
            "memory",
            f"weights {mem.get('weights_gib', 0):.2f} GiB  •  "
            f"KV {mem.get('kv_gib', 0):.2f} GiB ({mem.get('kv_blocks', 0)} blocks)  •  "
            f"free {mem.get('free_gib', 0):.2f} GiB",
        )
    cfg = m.get("config") or {}
    if cfg:
        cg = cfg.get("cuda_graph")
        cg_s = "off"
        if cg:
            caps = cfg.get("captured_batch_sizes") or []
            cg_s = f"on (batches {caps})" if caps else "on"
        row("max_num_seqs", cfg.get("max_num_seqs"))
        row("cuda-graph", cg_s)
        row("tokenizer", cfg.get("tokenizer_id"))
        row("chat template", "present" if cfg.get("chat_template") else "none")
    row("model load", f"{m.get('load_time_s', 0):.1f}s" if m.get("load_time_s") else None)

    title = Text("  superl8-serve", style="bold magenta")
    title.append(f"  v{m.get('version', '?')}", style="dim")
    title.append("   W8A8 int8 dp4a  •  Volta sm_70", style="dim cyan")
    return Panel(t, title=title, border_style="magenta", padding=(1, 2))


def render_heartbeat(snap: dict):
    """One compact, control-code-free line-block for `docker logs` (Console.print,
    NOT Live). Returns a rich renderable."""
    from rich.table import Table
    from rich.text import Text

    tp = snap["throughput"]
    kv = snap["kv_cache"]
    lat = snap["latency"]
    c = snap["counters"]
    gpu = snap.get("gpu") or {}

    line = Text()
    line.append("stats ", style="bold dim")
    line.append(f"run={snap['running']} ", style="cyan")
    line.append(f"wait={snap['waiting']}  ", style="cyan")
    line.append(f"decode={tp['decode_tok_s']:.0f} tok/s ", style="bold green")
    line.append(f"prefill={tp['prefill_tok_s']:.0f} tok/s  ", style="green")
    kvc = _threshold_color(kv["usage_pct"])
    line.append(f"kv={kv['usage_pct']:.0f}% ", style=kvc)
    line.append(f"({kv['used_blocks']}/{kv['total_blocks']})  ", style="dim")
    line.append(f"ttft={lat['avg_ttft_ms']:.0f}ms ", style="yellow")
    line.append(f"itl={lat['avg_itl_ms']:.1f}ms ", style="yellow")
    line.append(f"p99={lat['p99_ms']:.0f}ms  ", style="yellow")
    if gpu:
        u = gpu.get("util_pct")
        line.append(f"gpu={u:.0f}% " if u is not None else "gpu=?% ", style="magenta")
        vu, vt = gpu.get("vram_used_mb"), gpu.get("vram_total_mb")
        if vu and vt:
            line.append(f"vram={vu / 1024:.1f}/{vt / 1024:.1f}G ", style="magenta")
        if gpu.get("temp_c") is not None:
            line.append(f"{gpu['temp_c']:.0f}°C ", style="magenta")
        if gpu.get("power_w") is not None:
            line.append(f"{gpu['power_w']:.0f}W ", style="magenta")
    line.append(f" | reqs={c['total_requests']} tok={c['output_tokens']} ", style="dim")
    line.append(f"up={_fmt_uptime(snap['uptime_s'])}", style="dim")

    grid = Table.grid()
    grid.add_column()
    grid.add_row(line)
    return grid


def _fmt_uptime(s: float) -> str:
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def start_heartbeat(stats: StatsCollector, *, console=None, interval: float = 5.0):
    """Spawn a daemon thread that prints one `render_heartbeat` block every
    `interval` seconds. Returns the thread. `docker logs`-safe: plain Console.print,
    no full-screen control codes."""
    from rich.console import Console

    con = console or Console()

    def _loop():
        while True:
            time.sleep(interval)
            try:
                con.print(render_heartbeat(stats.snapshot()))
            except Exception:  # never let telemetry crash the server
                pass

    th = threading.Thread(target=_loop, name="superl8serve-heartbeat", daemon=True)
    th.start()
    return th
