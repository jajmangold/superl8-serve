# SPDX-License-Identifier: MIT
"""CUDA-graph batch-bucket config seam (issue #368). CPU-only, no CUDA."""

import types

import pytest

pytest.importorskip("superl8")
pytest.importorskip("fastapi")

from superl8serve.api.server import (  # noqa: E402
    _batch_buckets_arg,
    _batch_buckets_over_max,
    _build_arg_parser,
    build_banner,
    parse_batch_buckets,
)
from superl8serve.engine.cuda_graph import (  # noqa: E402
    GraphedDecode,
    GraphedDecodeLayers,
)
from superl8serve.models import ModelConfig  # noqa: E402


def _cfg():
    return ModelConfig(
        arch="qwen3", vocab_size=256, hidden_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=256,
        max_position_embeddings=512, head_dim=32, qk_norm=True, tie_word_embeddings=True)


@pytest.mark.parametrize("spec,expected", [
    ("1,2,4,8,16,32,64,128,256,512", (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)),
    ("512", (512,)),
    ("1,1,2,2", (1, 2)),
    (" 4 , 2 , 1 ", (1, 2, 4)),
])
def test_parse_returns_positive_unique_ascending(spec, expected):
    assert parse_batch_buckets(spec) == expected


@pytest.mark.parametrize("spec", [
    "", "   ", ",", "1,,2",  # empty
    "0,4", "1,-2,8", "-1", "0",  # nonpositive
    "1,abc,8", "1;2",  # malformed
])
def test_parse_rejects_bad_spec(spec):
    with pytest.raises(ValueError):
        parse_batch_buckets(spec)


def test_arg_type_accepts_valid():
    assert _batch_buckets_arg("1,2,512") == (1, 2, 512)


@pytest.mark.parametrize("extra,expected", [
    ([], None),
    (["--cuda-graph-batch-buckets", "1,2,4,8,16,32,64,128,256,512"],
     (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)),
])
def test_cli_bucket_arg_default_and_valid(extra, expected):
    args = _build_arg_parser().parse_args(["--model", "m.superl8", *extra])
    assert args.cuda_graph_batch_buckets == expected


@pytest.mark.parametrize("spec", ["", "0,4", "1,abc,8", "-1,2"])
def test_cli_rejects_invalid_spec(spec):
    with pytest.raises(SystemExit) as e:
        _build_arg_parser().parse_args(
            ["--model", "m.superl8", "--cuda-graph-batch-buckets", spec])
    assert e.value.code == 2


@pytest.mark.parametrize("buckets,max_seq,expected", [
    (None, 16, []),
    ((1, 2, 16), 16, []),
    ((1, 2, 32), 16, [32]),
    ((512,), 16, [512]),
])
def test_over_max_helper(buckets, max_seq, expected):
    assert _batch_buckets_over_max(buckets, max_seq) == expected


class _FakeModel:
    def __init__(self):
        self._mods = []
    def modules(self):
        return iter(self._mods)
    def to(self, device):
        return self
    def eval(self):
        return self
    def parameters(self):
        return iter(())
    def buffers(self):
        return iter(())


class _FakeCache:
    def __init__(self, num_slots=512):
        self.num_slots = num_slots


def _runner_seen(monkeypatch, buckets):
    import superl8serve.engine.model_runner as mr
    seen = {}
    class FakeGraphed:
        supported = False
        def __init__(self, model, cache, **kw):
            seen["graphed"] = kw
    class FakeLayers:
        supported = False
        _unsupported_reason = "fake"
        def __init__(self, model, cache, **kw):
            seen["layers"] = kw
    monkeypatch.setattr(mr, "GraphedDecode", FakeGraphed)
    monkeypatch.setattr(mr, "GraphedDecodeLayers", FakeLayers)
    monkeypatch.setenv("SUPERL8SERVE_LAYER_GRAPH", "1")
    mr.EngineRunner(_FakeModel(), _FakeCache(), device="cpu",
                    enable_cuda_graph=True, cuda_graph_batch_buckets=buckets)
    return seen


@pytest.mark.parametrize("buckets,expected", [
    ((1, 2, 4, 512), (1, 2, 4, 512)),
    (None, (1, 2, 4, 8, 16, 32, 64, 128)),
])
def test_runner_forwards_buckets_to_both_graphs(monkeypatch, buckets, expected):
    seen = _runner_seen(monkeypatch, buckets)
    assert seen["graphed"]["batch_buckets"] == expected
    assert seen["layers"]["batch_buckets"] == expected


def _llm_kwargs(monkeypatch, buckets):
    import superl8serve.engine.llm_engine as le
    captured = {}
    class FakeRunner:
        def __init__(self, model, cache, **kw):
            captured["kw"] = kw
    monkeypatch.setattr(le, "EngineRunner", FakeRunner)
    monkeypatch.setattr(le, "build_model", lambda cfg, weights: _FakeModel())
    le.LLMEngine(_cfg(), {}, device="cpu", max_num_seqs=16,
                 enable_cuda_graph=True, cuda_graph_batch_buckets=buckets)
    return captured["kw"]


@pytest.mark.parametrize("buckets,expected", [
    (None, None),
    ((1, 2, 512), (1, 2, 512)),
])
def test_llm_engine_forwards_buckets(monkeypatch, buckets, expected):
    assert _llm_kwargs(monkeypatch, buckets)["cuda_graph_batch_buckets"] == expected


def test_load_engine_forwards_buckets(monkeypatch):
    import superl8serve.api.server as server
    import superl8serve.gguf_native as gguf_native
    superl8_captured = {}
    class FakeLLMEngine:
        def __init__(self, cfg, weights, **kw):
            superl8_captured["kw"] = kw
    gguf_captured = {}
    def fake_gguf(path, **kw):
        gguf_captured["kw"] = kw
    monkeypatch.setattr(server, "LLMEngine", FakeLLMEngine)
    monkeypatch.setattr(server, "checkpoint_info",
                        lambda p: {"meta": {"config": {"arch": "qwen3"}}})
    monkeypatch.setattr(server, "load_superl8_state_dict", lambda p, device: {})
    monkeypatch.setattr(server.ModelConfig, "from_hf",
                        staticmethod(lambda hf, arch=None: _cfg()))
    monkeypatch.setattr(gguf_native, "load_gguf_engine", fake_gguf)
    server.load_engine("m.superl8", device="cpu", max_num_seqs=16,
                       cuda_graph_batch_buckets=(1, 2, 512))
    server.load_engine("m.gguf", device="cpu", max_num_seqs=16,
                       cuda_graph_batch_buckets=(1, 2, 512))
    assert superl8_captured["kw"]["cuda_graph_batch_buckets"] == (1, 2, 512)
    assert gguf_captured["kw"]["cuda_graph_batch_buckets"] == (1, 2, 512)


class _FakeLayer:
    def forward(self, h, pos, ctx, residual):
        return h, h


class _FakeGraphModel(_FakeModel):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.model = types.SimpleNamespace(layers=[_FakeLayer()], embed_tokens=None,
                                           norm=None)


@pytest.mark.parametrize("cls", [GraphedDecode, GraphedDecodeLayers])
def test_graph_objects_filter_buckets_over_slots(cls):
    obj = cls(_FakeGraphModel(_cfg()), _FakeCache(16), device="cpu",
              batch_buckets=(1, 2, 16, 64, 512))
    assert obj.batch_buckets == (1, 2, 16)


def _banner(graphed):
    engine = types.SimpleNamespace(
        cfg=_cfg(), model=_FakeModel(), cache=_FakeCache(),
        scheduler=types.SimpleNamespace(max_num_seqs=16),
        runner=types.SimpleNamespace(graphed=graphed))
    return build_banner(engine, types.SimpleNamespace(name_or_path="tok",
                                                      chat_template=None),
                        served_model_name="m", chat_template=None,
                        load_time_s=1.0, max_len=2048)


@pytest.mark.parametrize("buckets,expected", [
    ((1, 2, 4, 8, 16), [1, 2, 4, 8, 16]),
    (None, []),
])
def test_banner_reports_effective_buckets(buckets, expected):
    graphed = None if buckets is None else types.SimpleNamespace(
        batch_buckets=tuple(buckets))
    banner = _banner(graphed)
    assert banner["config"]["cuda_graph"] is (buckets is not None)
    assert banner["config"]["captured_batch_sizes"] == expected


def test_banner_reports_graph_replay_status():
    graphed = types.SimpleNamespace(
        batch_buckets=(1, 2),
        status=lambda: {
            "supported": True,
            "unsupported_reason": None,
            "batch_buckets": [1, 2],
            "captured_graphs": 1,
            "captures": 1,
            "replays": 7,
            "misses": {"graph_cap": 2},
        },
    )
    banner = _banner(graphed)
    cfg = banner["config"]
    assert cfg["cuda_graph_supported"] is True
    assert cfg["cuda_graph_captured_graphs"] == 1
    assert cfg["cuda_graph_captures"] == 1
    assert cfg["cuda_graph_replays"] == 7
    assert cfg["cuda_graph_misses"] == {"graph_cap": 2}
