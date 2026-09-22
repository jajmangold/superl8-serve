# SPDX-License-Identifier: MIT
"""cert_loadgen — end-to-end (HTTP, OpenAI /v1/completions) throughput sweep and
sustained-load soak driver for a running superl8-serve instance. Stdlib only, runs on
the HOST (no GPU footprint) so the measured server is the only thing on the card.

Modes:
  sweep  — for each concurrency in --concurrency, fire --reqs-per-conc requests
           (fixed prompt + max_tokens), report aggregate + per-stream tok/s and
           TTFT/latency. This is the END-TO-END number (prefill+decode+queue+HTTP),
           distinct from the decode-only tools/bench_llm.py figure.
  soak   — run --duration seconds with --concurrency workers each sending a
           continuous stream of MIXED prompt/gen-length requests; print a periodic
           heartbeat (elapsed, completed, rolling e2e tok/s, errors) so an external
           VRAM/RAM sampler can be correlated against it.

Example:
  python3 bench/cert_loadgen.py sweep --url http://localhost:8001 --model Qwen3.5-9B \
      --concurrency 1 2 4 8 --reqs-per-conc 8 --max-tokens 128
  python3 bench/cert_loadgen.py soak  --url http://localhost:8001 --model Qwen3.5-9B \
      --concurrency 4 --duration 1800
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_PROMPT = (
    "In a detailed and well organized manner, explain the following topic to a curious "
    "reader, giving concrete examples and covering both the history and the practical "
    "implications. Topic: "
)
TOPICS = [
    "the transformer neural network architecture and self-attention",
    "how photosynthesis converts sunlight into chemical energy",
    "the causes and consequences of the industrial revolution",
    "integer quantization of large language models for efficient inference",
    "the water cycle and how rain forms in the atmosphere",
    "the theory of plate tectonics and continental drift",
    "how a binary search algorithm works and why it is efficient",
    "the economic principles of supply and demand in a market",
]


def _post(url: str, body: dict, timeout: float = 300.0) -> tuple[dict, float, float | None]:
    """Return (json, wall_s, ttft_s|None). Non-streaming, so ttft is None."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(url + "/v1/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read())
    return payload, time.perf_counter() - t0, None


def _mk_prompt(topic_idx: int, repeat: int) -> str:
    return (BASE_PROMPT + TOPICS[topic_idx % len(TOPICS)] + " ") * repeat


def sweep(a):
    print(f"# e2e sweep against {a.url}  model={a.model}  max_tokens={a.max_tokens}")
    rows = []
    for c in a.concurrency:
        n = a.reqs_per_conc
        bodies = [{
            "model": a.model,
            "prompt": _mk_prompt(i, a.prompt_repeat),
            "max_tokens": a.max_tokens,
            "temperature": 0.0,
        } for i in range(n)]
        # warm one request so weights/graphs are hot and queues drained
        _post(a.url, {**bodies[0], "max_tokens": 8})
        lat, comp_toks, errs = [], 0, 0
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=c) as ex:
            futs = [ex.submit(_post, a.url, b) for b in bodies]
            for f in as_completed(futs):
                try:
                    payload, wall, _ = f.result()
                    comp_toks += payload["usage"]["completion_tokens"]
                    lat.append(wall)
                except Exception as e:  # noqa: BLE001
                    errs += 1
                    print("  ERR", repr(e)[:120])
        wall = time.perf_counter() - t0
        agg = comp_toks / wall if wall else 0
        row = {
            "concurrency": c, "requests": n, "errors": errs,
            "wall_s": round(wall, 2), "completion_tokens": comp_toks,
            "agg_tok_s": round(agg, 1), "per_stream_tok_s": round(agg / c, 1),
            "lat_p50_s": round(statistics.median(lat), 2) if lat else None,
            "lat_max_s": round(max(lat), 2) if lat else None,
        }
        rows.append(row)
        print(json.dumps(row))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=2)
    return rows


def soak(a):
    stop_at = time.perf_counter() + a.duration
    lock = threading.Lock()
    state = {"done": 0, "toks": 0, "errs": 0, "recent_toks": 0, "recent_t": time.perf_counter()}
    rng = random.Random(1234)

    def worker(wid: int):
        r = random.Random(1000 + wid)
        while time.perf_counter() < stop_at:
            repeat = r.choice([1, 2, 3, 5])           # mixed prompt length
            max_tok = r.choice([32, 64, 128, 256])    # mixed gen length
            body = {"model": a.model, "prompt": _mk_prompt(r.randrange(len(TOPICS)), repeat),
                    "max_tokens": max_tok, "temperature": 0.0}
            try:
                payload, wall, _ = _post(a.url, body)
                ct = payload["usage"]["completion_tokens"]
                with lock:
                    state["done"] += 1
                    state["toks"] += ct
                    state["recent_toks"] += ct
            except Exception as e:  # noqa: BLE001
                with lock:
                    state["errs"] += 1
                if state["errs"] <= 20:
                    print(f"  [{int(time.perf_counter())}] ERR w{wid}: {repr(e)[:140]}", flush=True)

    t0 = time.perf_counter()
    print(f"# soak {a.duration}s  concurrency={a.concurrency[0]}  url={a.url}", flush=True)
    threads = [threading.Thread(target=worker, args=(i,), daemon=True)
               for i in range(a.concurrency[0])]
    for t in threads:
        t.start()
    # heartbeat
    while time.perf_counter() < stop_at:
        time.sleep(a.heartbeat)
        with lock:
            now = time.perf_counter()
            dt = now - state["recent_t"]
            rate = state["recent_toks"] / dt if dt else 0
            state["recent_toks"] = 0
            state["recent_t"] = now
            el = now - t0
            print(f"[soak t={el:6.0f}s] done={state['done']:5d} "
                  f"errs={state['errs']:3d} rolling_tok_s={rate:6.1f} "
                  f"cum_tok={state['toks']}", flush=True)
    for t in threads:
        t.join(timeout=305)
    el = time.perf_counter() - t0
    summary = {"duration_s": round(el, 1), "completed": state["done"], "errors": state["errs"],
               "total_completion_tokens": state["toks"],
               "mean_tok_s": round(state["toks"] / el, 1) if el else 0}
    print("SOAK_SUMMARY " + json.dumps(summary), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("sweep", "soak"):
        p = sub.add_parser(name)
        p.add_argument("--url", default="http://localhost:8001")
        p.add_argument("--model", default="Qwen3.5-9B")
        p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
        p.add_argument("--max-tokens", type=int, default=128)
        p.add_argument("--prompt-repeat", type=int, default=1)
        p.add_argument("--out", default=None)
        if name == "sweep":
            p.add_argument("--reqs-per-conc", type=int, default=8)
        else:
            p.add_argument("--duration", type=int, default=1800)
            p.add_argument("--heartbeat", type=int, default=30)
    a = ap.parse_args()
    if a.mode == "sweep":
        sweep(a)
    else:
        soak(a)


if __name__ == "__main__":
    main()
