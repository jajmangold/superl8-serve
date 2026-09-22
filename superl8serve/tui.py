# SPDX-License-Identifier: MIT
"""superl8serve-top — a live, pretty terminal dashboard for a running superl8-serve.

Polls `GET /metrics` (~2 Hz) and renders a Rich `Live` dashboard: a GPU panel
(name, VRAM bar, util/temp/power), a throughput panel (decode/prefill tok/s with a
sparkline), a KV-cache usage meter, the queue (running/waiting), latency
(TTFT/ITL/p50/p99), and lifetime counters (requests, tokens, uptime). Colour
thresholds (green/yellow/red) flag KV pressure and GPU utilisation at a glance.

    superl8serve-top                       # http://localhost:8000
    superl8serve-top --url http://gpu-4:8000 --interval 0.5

Dependency-light: stdlib `urllib` for the poll (no httpx), `rich` for the render.
Reconnects gracefully -- an unreachable / restarting server shows a WAITING panel
instead of crashing.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections import deque

from rich.align import Align
from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .metrics import _fmt_gpu_kind, _fmt_uptime, _threshold_color, bar

_SPARK = "▁▂▃▄▅▆▇█"


def sparkline(vals, width: int = 24) -> str:
    """Unicode sparkline over the trailing `width` samples."""
    data = list(vals)[-width:]
    if not data:
        return " " * width
    lo, hi = min(data), max(data)
    span = (hi - lo) or 1.0
    out = "".join(_SPARK[min(len(_SPARK) - 1, int((v - lo) / span * (len(_SPARK) - 1)))] for v in data)
    return out.rjust(width)


def _fetch(url: str, timeout: float = 2.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None


def _kv_row(label, value, *, label_style="bold cyan", value_style="white"):
    return Text.assemble((f"{label}  ", label_style), (str(value), value_style))


def _gpu_panel(snap) -> Panel:
    gpu = snap.get("gpu") or {}
    if not gpu:
        return Panel(Align.center(Text("no GPU telemetry", style="dim")),
                     title="GPU", border_style="grey37")
    body = Table.grid(padding=(0, 1))
    body.add_column(justify="right", style="bold cyan", no_wrap=True)
    body.add_column()
    body.add_row("device", Text(_fmt_gpu_kind(gpu.get("name", "")), style="white"))
    vu, vt = gpu.get("vram_used_mb") or 0, gpu.get("vram_total_mb") or 0
    vpct = 100.0 * vu / vt if vt else 0.0
    body.add_row("vram", Text.from_markup(
        f"{bar(vpct, 22)}  [white]{vu / 1024:.1f}/{vt / 1024:.1f} GiB[/] [dim]({vpct:.0f}%)[/]"))
    util = gpu.get("util_pct")
    if util is not None:
        body.add_row("util", Text.from_markup(f"{bar(util, 22)}  [white]{util:.0f}%[/]"))
    else:
        body.add_row("util", Text("n/a (mining-card counters locked)", style="dim"))
    extra = []
    if gpu.get("temp_c") is not None:
        extra.append(f"[magenta]{gpu['temp_c']:.0f}°C[/]")
    if gpu.get("power_w") is not None:
        extra.append(f"[magenta]{gpu['power_w']:.0f} W[/]")
    if extra:
        body.add_row("", Text.from_markup("   ".join(extra)))
    return Panel(body, title="[bold]GPU[/]", border_style="magenta")


def _throughput_panel(snap, decode_hist, prefill_hist) -> Panel:
    tp = snap["throughput"]
    body = Table.grid(padding=(0, 1))
    body.add_column(justify="right", style="bold cyan", no_wrap=True)
    body.add_column()
    body.add_row("decode", Text.from_markup(
        f"[bold green]{tp['decode_tok_s']:8.1f}[/] tok/s  [green]{sparkline(decode_hist)}[/]"))
    body.add_row("prefill", Text.from_markup(
        f"[bold green]{tp['prefill_tok_s']:8.1f}[/] tok/s  [cyan]{sparkline(prefill_hist)}[/]"))
    return Panel(body, title="[bold]Throughput[/]", border_style="green")


def _queue_kv_panel(snap) -> Panel:
    kv = snap["kv_cache"]
    body = Table.grid(padding=(0, 1))
    body.add_column(justify="right", style="bold cyan", no_wrap=True)
    body.add_column()
    body.add_row("running", Text(str(snap["running"]), style="bold white"))
    body.add_row("waiting", Text(str(snap["waiting"]),
                                 style="bold yellow" if snap["waiting"] else "white"))
    kvc = _threshold_color(kv["usage_pct"])
    body.add_row("kv-cache", Text.from_markup(
        f"{bar(kv['usage_pct'], 22)}  [{kvc}]{kv['usage_pct']:.0f}%[/] "
        f"[dim]({kv['used_blocks']}/{kv['total_blocks']} blk)[/]"))
    return Panel(body, title="[bold]Queue / KV[/]", border_style="cyan")


def _latency_panel(snap) -> Panel:
    lat = snap["latency"]
    body = Table.grid(padding=(0, 2))
    body.add_column(justify="right", style="bold cyan", no_wrap=True)
    body.add_column(justify="right", style="yellow")
    body.add_row("TTFT (avg)", f"{lat['avg_ttft_ms']:.0f} ms")
    body.add_row("ITL (avg)", f"{lat['avg_itl_ms']:.1f} ms")
    body.add_row("e2e p50", f"{lat['p50_ms']:.0f} ms")
    body.add_row("e2e p99", f"{lat['p99_ms']:.0f} ms")
    return Panel(body, title="[bold]Latency[/]", border_style="yellow")


def _counters_panel(snap) -> Panel:
    c = snap["counters"]
    body = Table.grid(padding=(0, 2))
    body.add_column(justify="right", style="bold cyan", no_wrap=True)
    body.add_column(justify="right", style="white")
    body.add_row("requests", f"{c['total_requests']}  ({c['finished_requests']} done)")
    body.add_row("prompt tok", f"{c['prompt_tokens']:,}")
    body.add_row("output tok", f"{c['output_tokens']:,}")
    body.add_row("uptime", _fmt_uptime(snap["uptime_s"]))
    fr = c.get("finish_reasons") or {}
    if fr:
        body.add_row("finish", "  ".join(f"{k}:{v}" for k, v in fr.items()))
    return Panel(body, title="[bold]Lifetime[/]", border_style="blue")


def _header(snap, url: str) -> Panel:
    m = snap.get("model") or {}
    t = Text()
    t.append("superl8serve-top", style="bold magenta")
    name = m.get("model_name")
    if name:
        t.append(f"  {name}", style="bold white")
    if m.get("arch"):
        t.append(f"  [{m['arch']}]", style="cyan")
    if m.get("quant"):
        t.append(f"  {m['quant']}", style="dim")
    t.append(f"   {url}", style="dim")
    return Panel(Align.center(t), border_style="magenta")


def _waiting_panel(url: str) -> Panel:
    return Panel(
        Align.center(Group(
            Text("superl8serve-top", style="bold magenta"),
            Text(""),
            Text(f"waiting for server at {url} …", style="yellow"),
            Text("(is it up? try --url)", style="dim"),
        ), vertical="middle"),
        border_style="yellow", title="[bold]disconnected[/]")


def render(snap, url, decode_hist, prefill_hist) -> Layout:
    root = Layout()
    root.split_column(
        Layout(_header(snap, url), size=3, name="head"),
        Layout(name="body"),
        Layout(name="lower"),
    )
    root["body"].split_row(
        Layout(_gpu_panel(snap), name="gpu"),
        Layout(_throughput_panel(snap, decode_hist, prefill_hist), name="tp"),
    )
    root["lower"].split_row(
        Layout(_queue_kv_panel(snap), name="qkv"),
        Layout(_latency_panel(snap), name="lat"),
        Layout(_counters_panel(snap), name="cnt"),
    )
    return root


def main() -> None:
    ap = argparse.ArgumentParser(description="Live TUI dashboard for a running superl8-serve")
    ap.add_argument("--url", default="http://localhost:8000",
                    help="superl8-serve base URL (default: http://localhost:8000)")
    ap.add_argument("--interval", type=float, default=0.5,
                    help="poll interval in seconds (default: 0.5 = 2 Hz)")
    args = ap.parse_args()

    metrics_url = args.url.rstrip("/") + "/metrics"
    decode_hist: deque[float] = deque(maxlen=60)
    prefill_hist: deque[float] = deque(maxlen=60)

    from rich.console import Console

    console = Console()
    try:
        with Live(console=console, screen=True, refresh_per_second=8, transient=True) as live:
            while True:
                snap = _fetch(metrics_url)
                if snap is None:
                    live.update(_waiting_panel(args.url))
                else:
                    decode_hist.append(snap["throughput"]["decode_tok_s"])
                    prefill_hist.append(snap["throughput"]["prefill_tok_s"])
                    live.update(render(snap, args.url, decode_hist, prefill_hist))
                time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
