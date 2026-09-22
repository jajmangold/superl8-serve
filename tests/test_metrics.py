# SPDX-License-Identifier: MIT
"""Unit tests for the serving telemetry layer (issue #182).

Covers StatsCollector aggregation (rolling throughput windows, latency
percentiles, KV-cache usage %), the `GET /metrics` route schema via the same
fake-engine harness `test_api.py` uses (no CUDA / weights), and a smoke-import of
the `superl8serve-top` TUI. No live GPU required -- GPU telemetry degrades to None."""

import itertools

import pytest

from superl8serve.metrics import (
    StatsCollector,
    ThroughputWindow,
    _percentile,
    bar,
)


# --------------------------------------------------------------------------- #
# ThroughputWindow                                                            #
# --------------------------------------------------------------------------- #
def test_throughput_window_rate():
    w = ThroughputWindow(window_s=5.0)
    # 100 tokens over 1s => 100 tok/s
    w.add(100, 1.0, now=0.0)
    assert w.rate(now=0.5) == pytest.approx(100.0)
    # add another 100 tokens over 1s => 200 tok / 2s = 100 tok/s
    w.add(100, 1.0, now=1.0)
    assert w.rate(now=1.5) == pytest.approx(100.0)


def test_throughput_window_evicts_old_samples():
    w = ThroughputWindow(window_s=5.0)
    w.add(1000, 1.0, now=0.0)
    # far in the future: the old sample has aged out => 0
    assert w.rate(now=100.0) == 0.0


def test_throughput_window_empty_is_zero():
    assert ThroughputWindow().rate(now=0.0) == 0.0


# --------------------------------------------------------------------------- #
# percentiles                                                                 #
# --------------------------------------------------------------------------- #
def test_percentile_basic():
    vals = list(range(1, 101))  # 1..100
    assert _percentile(vals, 0.50) == pytest.approx(50, abs=1)
    assert _percentile(vals, 0.99) == pytest.approx(100, abs=1)
    assert _percentile(vals, 0.0) == 1


def test_percentile_edge_cases():
    assert _percentile([], 0.5) == 0.0
    assert _percentile([42.0], 0.99) == 42.0


# --------------------------------------------------------------------------- #
# StatsCollector: step throughput + counters                                  #
# --------------------------------------------------------------------------- #
def test_record_step_counters_and_throughput():
    s = StatsCollector()
    s.record_step(is_prefill=True, num_tokens=128, running=1, waiting=0, dt=0.128)
    s.record_step(is_prefill=False, num_tokens=4, running=4, waiting=2, dt=0.004)
    snap = s.snapshot()
    assert snap["counters"]["prefill_tokens"] == 128
    assert snap["counters"]["output_tokens"] == 4
    assert snap["running"] == 4
    assert snap["waiting"] == 2
    assert snap["throughput"]["prefill_tok_s"] > 0
    assert snap["throughput"]["decode_tok_s"] > 0


def test_record_queue_depth_refreshes_gauges_without_step():
    """After a cancel there is no following engine step, so the running/waiting
    gauges must be refreshable on their own (issue #363)."""
    s = StatsCollector()
    s.record_step(is_prefill=False, num_tokens=1, running=3, waiting=2, dt=0.001)
    s.record_queue_depth(running=0, waiting=0)
    snap = s.snapshot()
    assert snap["running"] == 0
    assert snap["waiting"] == 0


# --------------------------------------------------------------------------- #
# StatsCollector: per-request lifecycle -> TTFT / ITL / e2e / counters         #
# --------------------------------------------------------------------------- #
def test_request_lifecycle_latencies():
    s = StatsCollector()
    ids = itertools.count()
    for _ in range(5):
        rid = next(ids)
        s.record_request_start(rid, prompt_tokens=10, sampling={"temperature": 0.0})
        s.record_first_token(rid)
        s.record_request_finish(rid, output_tokens=8, finish_reason="stop")
    snap = s.snapshot()
    c = snap["counters"]
    assert c["total_requests"] == 5
    assert c["finished_requests"] == 5
    assert c["prompt_tokens"] == 50
    assert c["inflight_requests"] == 0
    assert c["finish_reasons"] == {"stop": 5}
    # latency fields are populated and non-negative
    lat = snap["latency"]
    assert lat["avg_ttft_ms"] >= 0.0
    assert lat["p50_ms"] >= 0.0
    assert lat["p99_ms"] >= lat["p50_ms"]


def test_inflight_request_counted_until_finish():
    s = StatsCollector()
    s.record_request_start(1, prompt_tokens=3)
    assert s.snapshot()["counters"]["inflight_requests"] == 1
    s.record_request_finish(1, output_tokens=2, finish_reason="length")
    snap = s.snapshot()
    assert snap["counters"]["inflight_requests"] == 0
    assert snap["counters"]["finish_reasons"] == {"length": 1}


# --------------------------------------------------------------------------- #
# StatsCollector: KV-cache usage %                                            #
# --------------------------------------------------------------------------- #
class _FakeCache:
    num_blocks = 200

    def __init__(self, used):
        self._used = used

    @property
    def used_blocks(self):
        return self._used


class _FakeEngine:
    def __init__(self, cache):
        self.cache = cache


def test_kv_usage_percent():
    s = StatsCollector()
    s.attach_engine(_FakeEngine(_FakeCache(50)))
    kv = s.snapshot()["kv_cache"]
    assert kv["used_blocks"] == 50
    assert kv["total_blocks"] == 200
    assert kv["usage_pct"] == pytest.approx(25.0)


def test_kv_usage_no_cache_is_zero():
    s = StatsCollector()
    kv = s.snapshot()["kv_cache"]
    assert kv == {"used_blocks": 0, "total_blocks": 0, "usage_pct": 0.0}


# --------------------------------------------------------------------------- #
# bar rendering helper                                                        #
# --------------------------------------------------------------------------- #
def test_bar_color_thresholds():
    assert "green" in bar(10.0)
    assert "yellow" in bar(75.0)
    assert "red" in bar(95.0)
    # width honoured (filled + empty cells == width)
    b = bar(50.0, width=10)
    assert b.count("█") + b.count("░") == 10


def test_snapshot_has_full_schema():
    s = StatsCollector()
    s.set_banner({"model_name": "test", "arch": "qwen3"})
    snap = s.snapshot()
    for key in (
        "uptime_s",
        "running",
        "waiting",
        "throughput",
        "kv_cache",
        "latency",
        "counters",
        "gpu",
        "model",
    ):
        assert key in snap
    assert snap["model"]["arch"] == "qwen3"


# --------------------------------------------------------------------------- #
# GET /metrics route via the fake-engine harness (mirrors test_api.py)        #
# --------------------------------------------------------------------------- #
pytest.importorskip("fastapi")


def _build_client():
    from starlette.testclient import TestClient

    from superl8serve.api.app import create_app
    from superl8serve.engine.sequence import SamplingParams, Sequence, Status

    EOS = 4

    class FakeTokenizer:
        eos_token_id = EOS

        def apply_chat_template(
            self,
            messages,
            tokenize=True,
            add_generation_prompt=True,
            chat_template=None,
            tools=None,
        ):
            return [10, 11, 12]

        def encode(self, text, **kw):
            return [10, 11, 12]

        def decode(self, ids, skip_special_tokens=True):
            return "".join(str(t) for t in ids if not (skip_special_tokens and t == EOS))

    class FakeEngine:
        def __init__(self):
            self.eos_id = EOS
            self.reply = [0, 1, 2, 3, EOS]
            self._seqs = {}
            self._ids = itertools.count()

        def add_request(self, prompt_ids, params=None):
            sid = next(self._ids)
            self._seqs[sid] = Sequence(sid, list(prompt_ids), params or SamplingParams())
            return sid

        def step(self):
            for seq in self._seqs.values():
                if seq.status is Status.FINISHED:
                    continue
                seq.output_ids.append(self.reply[len(seq.output_ids)])
                seq.status = Status.FINISHED if seq.is_finished(self.eos_id) else Status.RUNNING

        def sequence(self, sid):
            return self._seqs[sid]

        def forget(self, sid):
            self._seqs.pop(sid, None)

    app = create_app(FakeEngine(), FakeTokenizer(), served_model_name="fake")
    return TestClient(app)


def test_metrics_route_schema():
    with _build_client() as client:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        body = resp.json()
        for key in (
            "uptime_s",
            "running",
            "waiting",
            "throughput",
            "kv_cache",
            "latency",
            "counters",
            "gpu",
            "model",
        ):
            assert key in body
        assert "decode_tok_s" in body["throughput"]
        assert "usage_pct" in body["kv_cache"]
        assert "p99_ms" in body["latency"]


def test_metrics_counts_requests_after_generation():
    with _build_client() as client:
        r = client.post("/v1/completions", json={"model": "fake", "prompt": "hi"})
        assert r.status_code == 200
        body = client.get("/metrics").json()
        # the worker recorded the request lifecycle through the stats collector
        # (output_tokens is accounted in the engine step hook, which the fake
        # engine bypasses -- so we assert only the worker-driven counters here).
        assert body["counters"]["total_requests"] >= 1
        assert body["counters"]["finished_requests"] >= 1
        assert body["counters"]["prompt_tokens"] >= 1


# --------------------------------------------------------------------------- #
# TUI smoke import                                                            #
# --------------------------------------------------------------------------- #
def test_tui_imports_and_sparkline():
    pytest.importorskip("rich")
    from superl8serve import tui

    assert callable(tui.main)
    sp = tui.sparkline([1, 2, 3, 4, 5], width=5)
    assert len(sp) == 5
    # empty history renders blanks, never crashes
    assert tui.sparkline([], width=8) == " " * 8
