# SPDX-License-Identifier: MIT
"""Native-GGUF loader tests (P1 of the GGUF-native migration).

Header-only, no GPU: read a REAL GGUF's KV metadata via `gguf.GGUFReader` and
assert `gguf_native.gguf_config` reproduces the same `ModelConfig` that
`ModelConfig.from_hf` builds from the model's HF `config.json` — dims, rope,
heads, the hybrid `full_attention_interval`, and the divergent SSM / MoE fields.

The reference numbers below are the AUTHORITATIVE HF `config.json` values for the
two on-disk GGUFs (verified against the HF configs at test-authoring time):

  * Qwen3.5-9B (hybrid gated-DeltaNet + full attn, DENSE) — the rich case that
    exercises the SSM/hybrid/partial-rope mapping.
  * Qwen3-8B (standard dense GQA) — the clean e2e target used by PR-3.

If the GGUFs are not present the model tests skip (they live outside the repo).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from superl8serve.gguf_native import gguf_config, gguf_dit_config

# ── Real on-disk GGUFs (outside the repo; skip if absent) ────────────────────
_WEIGHTS_DIR = os.environ.get("SUPERL8_WEIGHTS_DIR", "/path/to/weights")
_QWEN35_9B = os.path.join(_WEIGHTS_DIR, os.environ.get("QWEN35_9B_GGUF", "Qwen3.5-9B-UD-Q6_K_XL.gguf"))
_QWEN3_8B = os.path.join(
    _WEIGHTS_DIR, os.environ.get(
        "QWEN3_8B_GGUF",
        "text_encoders/Qwen 3/Qwen3-8B-Q4_K_M.gguf",
    )
)
_LTX_DIT = os.path.join(_WEIGHTS_DIR, os.environ.get("LTX_DIT_GGUF", "ltx-2.3-22b-distilled-Q4_K_M_light.gguf"))
_LFM25_26B = os.environ.get(
    "SUPERL8_TEST_LFM25_GGUF",
    os.path.join(_WEIGHTS_DIR, "LiquidAI_LFM2.5-2.6B-Q6_K.gguf"),
)

_HAVE_GGUF = pytest.importorskip("gguf", reason="gguf lib required for native loader")

requires_qwen35 = pytest.mark.skipif(not os.path.exists(_QWEN35_9B), reason=f"missing {_QWEN35_9B}")
requires_qwen3 = pytest.mark.skipif(not os.path.exists(_QWEN3_8B), reason=f"missing {_QWEN3_8B}")
requires_ltx = pytest.mark.skipif(not os.path.exists(_LTX_DIT), reason=f"missing {_LTX_DIT}")
requires_lfm25 = pytest.mark.skipif(
    not os.path.exists(_LFM25_26B), reason=f"missing {_LFM25_26B}"
)


class _FakeField:
    def __init__(self, value, *, data=None):
        self._value = value
        self.data = [] if data is None else data

    def contents(self):
        return self._value


def _lfm25_reader():
    """Header-only shape of Bartowski LFM2.5-2.6B Q6_K at 0446ca3a34ae."""
    kv_heads = [
        0, 0, 8, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, 0,
        0, 0, 8, 0, 0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 0,
    ]
    values = {
        "general.architecture": "lfm2",
        "lfm2.embedding_length": 2048,
        "lfm2.block_count": 30,
        "lfm2.attention.head_count": 32,
        "lfm2.attention.head_count_kv": kv_heads,
        "lfm2.feed_forward_length": 10752,
        "lfm2.context_length": 128000,
        "lfm2.attention.layer_norm_rms_epsilon": 1e-5,
        "lfm2.rope.freq_base": 1e7,
        "lfm2.shortconv.l_cache": 3,
    }
    fields = {key: _FakeField(value) for key, value in values.items()}
    fields["tokenizer.ggml.tokens"] = _FakeField(None, data=range(128000))
    tensors = [SimpleNamespace(name="token_embd.weight", shape=(2048, 128000))]
    return SimpleNamespace(fields=fields, tensors=tensors)


@pytest.mark.correctness
def test_lfm25_config_recovers_per_layer_attention_schedule():
    """LFM2 GGUF stores KV heads per layer; zero means ShortConv, nonzero means GQA."""
    cfg = gguf_config("lfm25.gguf", _reader=_lfm25_reader())

    assert cfg.arch == "lfm2"
    assert cfg.vocab_size == 128000
    assert cfg.hidden_size == 2048
    assert cfg.num_hidden_layers == 30
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 8
    assert cfg.intermediate_size == 10752
    assert cfg.resolved_head_dim() == 64
    assert cfg.max_position_embeddings == 128000
    assert cfg.rms_norm_eps == pytest.approx(1e-5)
    assert cfg.rope_theta == pytest.approx(1e7)
    assert cfg.tie_word_embeddings is True
    assert cfg.extra["full_attn_idxs"] == [2, 5, 9, 13, 17, 21, 24, 27]
    assert cfg.extra["layer_types"] == [
        "full_attention" if i in {2, 5, 9, 13, 17, 21, 24, 27} else "conv"
        for i in range(30)
    ]
    assert cfg.extra["conv_L_cache"] == 3


@pytest.mark.correctness
def test_lfm2_gguf_names_map_to_existing_builder_without_changing_llama():
    from superl8serve.gguf_import import gguf_name_to_hf

    expected = {
        "token_embd_norm.weight": "model.embedding_norm.weight",
        "blk.0.attn_norm.weight": "model.layers.0.operator_norm.weight",
        "blk.0.ffn_norm.weight": "model.layers.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight": "model.layers.0.feed_forward.w1.weight",
        "blk.0.ffn_up.weight": "model.layers.0.feed_forward.w3.weight",
        "blk.0.ffn_down.weight": "model.layers.0.feed_forward.w2.weight",
        "blk.2.attn_output.weight": "model.layers.2.self_attn.out_proj.weight",
        "blk.2.attn_q_norm.weight": "model.layers.2.self_attn.q_layernorm.weight",
        "blk.2.attn_k_norm.weight": "model.layers.2.self_attn.k_layernorm.weight",
        "blk.0.shortconv.in_proj.weight": "model.layers.0.conv.in_proj.weight",
        "blk.0.shortconv.out_proj.weight": "model.layers.0.conv.out_proj.weight",
        "blk.0.shortconv.conv.weight": "model.layers.0.conv.conv.weight",
    }
    for gguf_name, hf_name in expected.items():
        assert gguf_name_to_hf(gguf_name, arch="lfm2") == hf_name

    assert gguf_name_to_hf("blk.0.attn_norm.weight") == "model.layers.0.input_layernorm.weight"
    assert gguf_name_to_hf("blk.0.ffn_gate.weight") == "model.layers.0.mlp.gate_proj.weight"


@pytest.mark.correctness
def test_lfm2_shortconv_gguf_filter_restores_conv1d_rank():
    from superl8serve.gguf_native import _reshape_family_tensor

    flat = torch.arange(2048 * 3, dtype=torch.float32).reshape(2048, 3)
    restored = _reshape_family_tensor(
        "model.layers.0.conv.conv.weight", flat, arch="lfm2", conv_kernel=3
    )
    assert restored.shape == (2048, 1, 3)
    assert torch.equal(restored[:, 0, :], flat)


@pytest.mark.correctness
@requires_lfm25
def test_real_lfm25_header_and_every_tensor_name_are_supported():
    from superl8serve.gguf_import import gguf_name_to_hf
    from superl8serve.gguf_native import _field_value, _open_gguf, _reshape_family_tensor

    reader = _open_gguf(_LFM25_26B)
    cfg = gguf_config(_LFM25_26B, _reader=reader)
    raw_kv_heads = _field_value(reader.fields["lfm2.attention.head_count_kv"])
    metadata_schedule = [i for i, count in enumerate(raw_kv_heads) if int(count) > 0]
    assert cfg.extra["full_attn_idxs"] == metadata_schedule
    assert cfg.extra["full_attn_idxs"] == [2, 5, 9, 13, 17, 21, 24, 27]
    unmapped = [
        t.name for t in reader.tensors if gguf_name_to_hf(t.name, arch="lfm2") is None
    ]
    assert len(reader.tensors) == 266
    assert unmapped == []

    shortconv = next(t for t in reader.tensors if t.name == "blk.0.shortconv.conv.weight")
    restored = _reshape_family_tensor(
        "model.layers.0.conv.conv.weight",
        torch.from_numpy(shortconv.data.copy()),
        arch="lfm2",
        conv_kernel=cfg.extra["conv_L_cache"],
    )
    assert restored.shape == (2048, 1, 3)


@pytest.mark.perf
@requires_qwen35
def test_native_reader_skips_materializing_token_string_parts():
    """Tokenizer vocabulary metadata must not allocate one Python entry per token."""
    from superl8serve.gguf_native import _open_gguf

    reader = _open_gguf(_QWEN35_9B)
    tokens = reader.fields["tokenizer.ggml.tokens"]

    assert isinstance(tokens.data, range)
    assert len(tokens.data) == 248320
    assert len(reader.tensors) > 400


def test_gguf_engine_consumes_owned_weights(monkeypatch):
    """The one-shot GGUF load must free source rows while merging a card-filling model."""
    from types import SimpleNamespace

    import superl8serve.engine
    import superl8serve.gguf_native as native

    cfg = SimpleNamespace(
        arch="qwen3",
        hidden_size=4,
        linear_attention=False,
        qk_norm=False,
        num_mtp_layers=0,
    )
    weights = {"model.embed_tokens.weight": torch.zeros(8, 4)}
    seen = {}

    class FakeEngine:
        def __init__(self, passed_cfg, passed_weights, **kwargs):
            seen.update(cfg=passed_cfg, weights=passed_weights, kwargs=kwargs)

    monkeypatch.setattr(native, "_open_gguf", lambda _path: object())
    monkeypatch.setattr(native, "gguf_config", lambda _path, **_kwargs: cfg)
    monkeypatch.setattr(native, "gguf_state_dict", lambda *_args, **_kwargs: weights)
    monkeypatch.setattr(native, "_native_kquant_types", lambda: set())
    monkeypatch.setattr(native, "_gguf_eos_id", lambda _path, **_kwargs: 1)
    monkeypatch.setattr(superl8serve.engine, "LLMEngine", FakeEngine)

    native.load_gguf_engine("model.gguf", device="cpu", chunked_prefill_size=256)
    assert seen["kwargs"]["consume_weights"] is True
    assert seen["kwargs"]["chunked_prefill_size"] == 256


def test_gguf_engine_unsloth_probe_accepts_quantized_embedding(monkeypatch):
    """superl8-serve#418: the layout probe must handle the compact embedding QTensor."""
    from types import SimpleNamespace

    import superl8serve.engine
    import superl8serve.gguf_native as native
    from superl8 import QTensor

    cfg = SimpleNamespace(
        arch="qwen3", hidden_size=4, linear_attention=False, qk_norm=False, num_mtp_layers=0
    )
    # Deliberately use the Unsloth [in, out] shape so the transpose probe runs.
    weights = {
        "model.embed_tokens.weight": QTensor(
            torch.ones(4, 8, dtype=torch.int8),
            torch.ones(4, dtype=torch.float32),
            scheme="per_row_i8",
        )
    }
    monkeypatch.setattr(native, "_open_gguf", lambda _path: object())
    monkeypatch.setattr(native, "gguf_config", lambda _path, **_kwargs: cfg)
    monkeypatch.setattr(native, "gguf_state_dict", lambda *_args, **_kwargs: weights)
    monkeypatch.setattr(native, "_native_kquant_types", lambda: ())
    monkeypatch.setattr(native, "_gguf_eos_id", lambda _path, **_kwargs: 1)
    monkeypatch.setattr(superl8serve.engine, "LLMEngine", lambda *args, **kwargs: SimpleNamespace())

    native.load_gguf_engine("model.gguf", device="cpu")


def test_gguf_engine_reuses_one_reader(monkeypatch):
    """A cold load must not parse the multi-GB GGUF header three separate times."""
    from types import SimpleNamespace

    import superl8serve.engine
    import superl8serve.gguf_native as native

    reader = object()
    opened = []
    observed = []
    cfg = SimpleNamespace(
        arch="qwen3",
        hidden_size=4,
        linear_attention=False,
        qk_norm=False,
        num_mtp_layers=0,
    )
    monkeypatch.setattr(native, "_open_gguf", lambda path: opened.append(path) or reader)
    monkeypatch.setattr(
        native,
        "gguf_config",
        lambda path, *, _reader=None: observed.append(("config", _reader)) or cfg,
    )
    monkeypatch.setattr(
        native,
        "gguf_state_dict",
        lambda path, *, device, native_types, _reader=None: (
            observed.append(("weights", _reader))
            or {"model.embed_tokens.weight": torch.zeros(8, 4)}
        ),
    )
    monkeypatch.setattr(
        native,
        "_gguf_eos_id",
        lambda path, *, _reader=None: observed.append(("eos", _reader)) or 1,
    )
    monkeypatch.setattr(native, "_native_kquant_types", lambda: ())
    monkeypatch.setattr(superl8serve.engine, "LLMEngine", lambda *args, **kwargs: SimpleNamespace())

    native.load_gguf_engine("model.gguf", device="cpu")

    assert opened == ["model.gguf"]
    assert observed == [("config", reader), ("weights", reader), ("eos", reader)]


@pytest.mark.parametrize(
    ("force_w8", "expected_types", "expected_runtime"),
    [
        (False, ("Q4_K", "Q6_K"), "checkpoint-default"),
        (True, (), "forced-per-row-w8"),
    ],
)
def test_gguf_engine_explicit_w8_policy_controls_native_types(
    monkeypatch, force_w8, expected_types, expected_runtime
):
    """Default loads preserve supported native k-quants; the explicit throughput
    policy passes an empty native set so every k-quant linear is transcoded once."""
    from types import SimpleNamespace

    import superl8serve.engine
    import superl8serve.gguf_native as native

    cfg = SimpleNamespace(
        arch="qwen3", hidden_size=4, linear_attention=False, qk_norm=False, num_mtp_layers=0
    )
    seen = {}
    monkeypatch.setattr(native, "_open_gguf", lambda _path: object())
    monkeypatch.setattr(native, "gguf_config", lambda _path, **_kwargs: cfg)
    monkeypatch.setattr(native, "_native_kquant_types", lambda: ("Q4_K", "Q6_K"))
    monkeypatch.setattr(
        native,
        "gguf_state_dict",
        lambda *_args, **kwargs: seen.update(native_types=kwargs["native_types"])
        or {"model.embed_tokens.weight": torch.zeros(8, 4)},
    )
    monkeypatch.setattr(native, "_gguf_eos_id", lambda _path, **_kwargs: 1)
    engine = SimpleNamespace()
    monkeypatch.setattr(superl8serve.engine, "LLMEngine", lambda *args, **kwargs: engine)

    result = native.load_gguf_engine("model.gguf", device="cpu", force_w8=force_w8)

    assert seen["native_types"] == expected_types
    assert result is engine
    assert result.weight_runtime == expected_runtime


@pytest.mark.correctness
def test_qwen35_gguf_transforms_are_restored_to_hf_semantics():
    """llama.cpp rewrites Qwen3.5 tensors; the HF-layout builder needs the inverse."""
    from types import SimpleNamespace

    from superl8serve.gguf_native import _restore_hf_qwen35_weights

    cfg = SimpleNamespace(
        linear_attention=True,
        extra={"linear_num_key_heads": 2, "linear_num_value_heads": 4},
    )
    sd = {
        "model.layers.0.input_layernorm.weight": torch.tensor([1.25]),
        "model.layers.0.self_attn.q_norm.weight": torch.tensor([0.75]),
        "model.layers.0.linear_attn.norm.weight": torch.tensor([0.875]),
        "model.layers.0.linear_attn.A_log": torch.tensor([-torch.exp(torch.tensor(2.0))]),
    }

    restored, tiled = _restore_hf_qwen35_weights(sd, cfg)

    assert torch.equal(restored["model.layers.0.input_layernorm.weight"], torch.tensor([0.25]))
    assert torch.equal(restored["model.layers.0.self_attn.q_norm.weight"], torch.tensor([-0.25]))
    # The gated DeltaNet output norm is not zero-centered and is not shifted by the converter.
    assert torch.equal(restored["model.layers.0.linear_attn.norm.weight"], torch.tensor([0.875]))
    assert torch.allclose(restored["model.layers.0.linear_attn.A_log"], torch.tensor([2.0]))
    assert tiled is True


@pytest.mark.correctness
def test_native_kquant_capabilities_cover_every_installed_kernel(monkeypatch):
    """The loader must not silently transcode a format whose fused kernel exists."""
    import superl8

    from superl8serve.gguf_native import _native_kquant_types

    expected = {
        gguf_type
        for gguf_type, op in {
            "Q2_K": "linear_q2k",
            "Q3_K": "linear_q3k",
            "Q4_K": "linear_q4k",
            "Q5_K": "linear_q5k",
            "Q6_K": "linear_q6k",
            "TQ3_4S": "linear_tq34s",
        }.items()
        if callable(getattr(superl8, op, None))
    }
    assert set(_native_kquant_types()) == expected

    monkeypatch.setattr(superl8, "linear_q6k", None, raising=False)
    assert "Q6_K" not in _native_kquant_types()


@pytest.mark.correctness
@requires_qwen35
def test_qwen35_9b_config_matches_hf():
    """Qwen3.5-9B hybrid: every dim/rope/head/hybrid field maps from GGUF-KV, and
    the divergent SSM params round-trip into `extra` for the qwen3_next builder."""
    cfg = gguf_config(_QWEN35_9B)

    # arch: GGUF `qwen35` must map onto our registered Qwen3.5 builder key
    # (registry._resolve does NOT know `qwen35`; the GGUF-arch alias table does).
    from superl8serve.models.registry import _resolve

    assert _resolve(cfg.arch) == "qwen3_5", f"arch {cfg.arch!r} did not resolve to a builder"

    # ── dims (qwen35.* KV → ModelConfig) ──
    assert cfg.vocab_size == 248320  # len(tokenizer.ggml.tokens)
    assert cfg.hidden_size == 4096  # qwen35.embedding_length
    assert cfg.num_hidden_layers == 32  # qwen35.block_count
    assert cfg.num_attention_heads == 16  # qwen35.attention.head_count
    assert cfg.num_key_value_heads == 4  # qwen35.attention.head_count_kv
    assert cfg.intermediate_size == 12288  # qwen35.feed_forward_length
    assert cfg.resolved_head_dim() == 256  # qwen35.attention.key_length
    assert cfg.max_position_embeddings == 262144  # qwen35.context_length
    assert cfg.tie_word_embeddings is False  # output.weight present → untied

    # ── rope (qwen35.rope.* KV) ──
    assert cfg.rope_theta == 10000000.0  # qwen35.rope.freq_base
    # partial_rotary = rope.dimension_count / head_dim = 64 / 256 = 0.25
    assert abs(cfg.partial_rotary_factor - 0.25) < 1e-9
    assert cfg.rotary_dim() == 64
    assert abs(cfg.rms_norm_eps - 1e-6) < 1e-8  # qwen35.attention.layer_norm_rms_epsilon

    # ── hybrid layer schedule (qwen35.full_attention_interval + ssm.* present) ──
    assert cfg.linear_attention is True
    assert cfg.full_attention_interval == 4
    # every 4th layer full, rest linear (matches HF layer_types)
    assert cfg.attention_kind(0) == "linear"
    assert cfg.attention_kind(3) == "full"
    assert cfg.attention_kind(7) == "full"
    assert cfg.attention_kind(31) == "full"

    # ── divergent SSM / gated-DeltaNet params (qwen35.ssm.* → extra) ──
    # Sources cross-checked vs research/llamacpp-hybrid-linattn.md §1.5 (HF→GGUF):
    #   linear_key_head_dim   = ssm.state_size       (128)
    #   linear_num_key_heads  = ssm.group_count      (16)
    #   linear_num_value_heads= ssm.time_step_rank   (32)
    #   linear_conv_kernel_dim= ssm.conv_kernel      (4)
    #   linear_value_head_dim = ssm.inner_size / ssm.time_step_rank = 4096/32 = 128
    x = cfg.extra
    assert x["linear_key_head_dim"] == 128
    assert x["linear_num_key_heads"] == 16
    assert x["linear_num_value_heads"] == 32
    assert x["linear_conv_kernel_dim"] == 4
    assert x["linear_value_head_dim"] == 128

    # this GGUF dropped the MTP/nextn head (no qwen35.nextn_predict_layers) → 0,
    # a documented divergence from the HF config (§5c of MIGRATION).
    assert cfg.num_mtp_layers == 0


@pytest.mark.correctness
@requires_qwen3
def test_qwen3_8b_config_matches_hf():
    """Standard dense Qwen3 (the PR-3 e2e model): plain GQA, full rotary, no SSM."""
    cfg = gguf_config(_QWEN3_8B)
    from superl8serve.models.registry import _resolve

    assert _resolve(cfg.arch) == "qwen3"
    assert cfg.hidden_size == 4096  # qwen3.embedding_length
    assert cfg.num_hidden_layers == 36  # qwen3.block_count
    assert cfg.num_attention_heads == 32  # qwen3.attention.head_count
    assert cfg.num_key_value_heads == 8  # qwen3.attention.head_count_kv
    assert cfg.intermediate_size == 12288  # qwen3.feed_forward_length
    assert cfg.resolved_head_dim() == 128  # qwen3.attention.key_length
    assert cfg.rope_theta == 1000000.0  # qwen3.rope.freq_base
    # no qwen3.rope.dimension_count → full rotary
    assert abs(cfg.partial_rotary_factor - 1.0) < 1e-9
    assert cfg.linear_attention is False  # dense, no hybrid
    assert cfg.num_experts == 0  # dense, no MoE
    assert cfg.vocab_size > 150000  # from tokenizer.ggml.tokens


@pytest.mark.correctness
@requires_qwen35
def test_gguf_config_moe_gating_asserted_not_defaulted():
    """A silent softmax default for a sigmoid group-routing MoE = wrong experts =
    garbage output (MIGRATION §5b). Assert the resolver NEVER leaves a MoE config
    with an unresolved gate. (Qwen3.5-9B is dense → num_experts==0 → vacuously ok;
    this guards the invariant so a future MoE GGUF cannot slip through defaulted.)"""
    cfg = gguf_config(_QWEN35_9B)
    if cfg.num_experts > 0:
        # if experts exist, the gate func MUST be explicitly resolved into extra
        assert "expert_gating_func" in cfg.extra, "MoE config left gating func defaulted"


@pytest.mark.correctness
@requires_qwen3
def test_gguf_state_dict_schemes_native():
    """PR-2: k-quant weights land RESIDENT as native `gguf_kquant` (raw bytes, no
    transcode); norms stay fp16 while quantized embeddings use the compact
    rowwise-int8 lookup contract."""
    from superl8 import QTensor

    from superl8serve.gguf_native import gguf_state_dict

    sd = gguf_state_dict(_QWEN3_8B, device="cpu")
    # a Q4_K linear (o_proj) → native gguf_kquant raw bytes
    o = sd["model.layers.0.self_attn.o_proj.weight"]
    assert isinstance(o, QTensor) and o.scheme == "gguf_kquant" and o.codebook == "q4_k"
    assert o.scale is None and o.data.dtype == torch.uint8
    # QTensor invariant: [out, n_superblocks*144] (Q4_K type_size=144), no paired scale
    assert o.data.dim() == 2 and o.data.shape[1] % 144 == 0
    # a Q6_K linear (ffn_down) → native gguf_kquant q6_k
    d = sd["model.layers.0.mlp.down_proj.weight"]
    assert isinstance(d, QTensor) and d.scheme == "gguf_kquant" and d.codebook == "q6_k"
    # norms stay fp16; the quantized embedding is compact and gatherable.
    assert not isinstance(sd["model.layers.0.input_layernorm.weight"], QTensor)
    emb = sd["model.embed_tokens.weight"]
    assert isinstance(emb, QTensor) and emb.scheme == "per_row_i8"
    assert emb.data.dtype == torch.int8 and emb.scale.dtype == torch.float32


@pytest.mark.perf
@requires_qwen3
def test_linear_gguf_kquant_matches_dequant():
    """PR-2 proof: LinearW8A8 over a native `gguf_kquant` weight matches a full-
    precision dequant reference at cos≥0.99 — the Q4_K fused dp4a path AND the
    Q5_K/Q6_K dequant fallback. Needs a CUDA device."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from superl8 import QTensor

    from superl8serve.gguf_native import dequant_kquant, gguf_state_dict
    from superl8serve.layers.linear import LinearW8A8

    # load on CPU (the full 8B doesn't fit alongside other jobs), move only the two
    # tested weights to CUDA — keeps this a targeted per-linear correctness probe.
    sd = gguf_state_dict(_QWEN3_8B, device="cpu")
    torch.manual_seed(0)
    for name in ("model.layers.0.self_attn.o_proj.weight", "model.layers.0.mlp.down_proj.weight"):
        src = sd[name]
        qt = QTensor(
            src.data.cuda(), None, scheme="gguf_kquant", codebook=src.codebook, group_size=256
        )
        lin = LinearW8A8(qt).cuda()
        x = torch.randn(4, lin.in_features, device="cuda", dtype=torch.float16)
        y = lin(x)
        ref = torch.nn.functional.linear(x, dequant_kquant(qt).to(x.dtype))
        cos = torch.nn.functional.cosine_similarity(
            y.float().flatten(), ref.float().flatten(), dim=0
        ).item()
        assert cos >= 0.99, f"{name} ({qt.codebook}) cos={cos:.4f} < 0.99"


@pytest.mark.perf
@requires_qwen3
def test_gguf_e2e_greedy_decode_zero_superl8():
    """PR-3 / the P1 milestone: greedily decode from a REAL GGUF LLM through the full
    superl8-serve engine (prefill + paged-KV decode) with ZERO `.superl8` — no .superl8 file,
    no .superl8 loader touched. Assert it runs end-to-end, the output is deterministic,
    and every emitted id is in-vocab. Needs a CUDA device with room for an 8B."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    import time

    from superl8serve.engine.sequence import SamplingParams
    from superl8serve.gguf_native import load_gguf_engine

    eng = load_gguf_engine(_QWEN3_8B, device="cuda", max_num_seqs=2, max_len=256)
    assert eng.cfg.vocab_size > 150000
    prompt = [3838, 374, 279, 6722, 315, 9625, 30]  # arbitrary in-vocab Qwen3 ids
    n = 16
    params = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)  # greedy

    t0 = time.perf_counter()
    out1 = eng.generate([prompt], params)[0]
    dt = time.perf_counter() - t0
    out2 = eng.generate([prompt], params)[0]

    assert len(out1) == n, f"expected {n} decoded tokens, got {len(out1)}"
    assert out1 == out2, "greedy decode must be deterministic across runs"
    assert all(0 <= t < eng.cfg.vocab_size for t in out1), "decoded ids out of vocab range"
    print(f"\n[e2e] Qwen3-8B GGUF greedy decode: {out1}  ({n / dt:.1f} tok/s incl. prefill)")


@pytest.mark.correctness
def test_dequant_chunked_to_fp16_device(monkeypatch):
    """superl8-serve#413: chunked host-side dequant produces the same fp16 result as
    calling _dequant_ggml per-chunk and concatenating — verifies row-chunked
    iteration and preallocated output. CPU-only — no GPU or real GGUF needed."""
    import numpy as np

    from superl8serve.gguf_native import _dequant_chunked_to_fp16_device

    _IN_FEATURES = 64  # (32 bytes_per_row // 16) * 32
    _OUT_ROWS = 640    # 3 chunks with default chunk_rows=256
    _CHUNK = 256

    def _fake_dequant(np_bytes, gtype):
        """Deterministic dequant: values derived from the raw bytes themselves."""
        rows = int(np_bytes.shape[0])
        # Use the first byte of each row as a seed so chunks produce distinct values.
        seeds = np_bytes[:, 0:1].astype(np.float32)
        return np.broadcast_to(seeds, (rows, _IN_FEATURES)).copy()

    monkeypatch.setattr("superl8serve.gguf_native._dequant_ggml", _fake_dequant)

    # Seed each row with a unique value so chunk boundaries are visible.
    raw = np.zeros((_OUT_ROWS, 32), dtype=np.uint8)
    raw[:, 0] = np.arange(_OUT_ROWS, dtype=np.uint8)

    result = _dequant_chunked_to_fp16_device(raw, "TQ3_4S", "cpu")

    # Build expected: fake_dequant each chunk, stack, convert to fp16.
    chunks = []
    for s in range(0, _OUT_ROWS, _CHUNK):
        e = min(s + _CHUNK, _OUT_ROWS)
        chunks.append(_fake_dequant(raw[s:e], "TQ3_4S"))
    expected = np.concatenate(chunks, axis=0).astype(np.float16)

    assert result.shape == (_OUT_ROWS, _IN_FEATURES)
    assert result.dtype == torch.float16
    np.testing.assert_array_equal(result.numpy(), expected)


@pytest.mark.correctness
def test_dequant_chunked_single_chunk(monkeypatch):
    """Edge case: tensor smaller than one chunk — should still work correctly."""
    import numpy as np

    from superl8serve.gguf_native import _dequant_chunked_to_fp16_device

    _IN_FEATURES = 64

    def _fake_dequant(np_bytes, gtype):
        rows = int(np_bytes.shape[0])
        return np.arange(rows * _IN_FEATURES, dtype=np.float32).reshape(rows, _IN_FEATURES)

    monkeypatch.setattr("superl8serve.gguf_native._dequant_ggml", _fake_dequant)

    raw = np.zeros((128, 32), dtype=np.uint8)  # < 256 rows → single chunk
    result = _dequant_chunked_to_fp16_device(raw, "TQ3_4S", "cpu")

    expected = np.arange(128 * _IN_FEATURES, dtype=np.float32).reshape(
        128, _IN_FEATURES
    ).astype(np.float16)

    assert result.shape == (128, _IN_FEATURES)
    assert result.dtype == torch.float16
    np.testing.assert_array_equal(result.numpy(), expected)


@pytest.mark.correctness
def test_dequant_chunked_embedding_to_per_row_i8(monkeypatch):
    """superl8-serve#418: embedding conversion stays row-chunked and compact."""
    import numpy as np

    from superl8serve.gguf_native import _dequant_chunked_to_per_row_i8

    calls = []

    def _fake_dequant(np_bytes, gtype):
        calls.append(int(np_bytes.shape[0]))
        rows = int(np_bytes.shape[0])
        starts = np_bytes[:, :1].astype(np.float32) * 4.0
        return starts + np.arange(4, dtype=np.float32)[None, :] + 1.0

    monkeypatch.setattr("superl8serve.gguf_native._dequant_ggml", _fake_dequant)
    raw = np.zeros((8, 32), dtype=np.uint8)
    raw[:, 0] = np.arange(8, dtype=np.uint8)
    result = _dequant_chunked_to_per_row_i8(raw, "TQ3_4S", "cpu", _chunk_rows=3)

    assert calls == [3, 3, 2]
    assert result.scheme == "per_row_i8"
    assert result.data.shape == (8, 4)
    assert result.data.dtype == torch.int8
    assert result.scale.shape == (8,)
    assert result.scale.dtype == torch.float32
    reconstructed = result.data.float() * result.scale.unsqueeze(-1)
    expected = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4) + 1.0
    # Rowwise RTN is lossy but bounded by half a scale per element.
    assert torch.all((reconstructed - expected).abs() <= result.scale.unsqueeze(-1))


@pytest.mark.correctness
def test_quantized_embedding_contract_supports_tied_lm_head():
    """superl8-serve#418: one compact QTensor can back lookup and tied logits."""
    from superl8 import QTensor
    from superl8serve.layers.embedding import LMHead, VocabEmbedding

    weight = QTensor(
        torch.ones(8, 4, dtype=torch.int8),
        torch.ones(8, dtype=torch.float32),
        scheme="per_row_i8",
    )
    embed = VocabEmbedding(weight)
    head = LMHead(weight)
    hidden = embed(torch.tensor([[0, 3]], dtype=torch.long))
    logits = head(hidden)

    assert hidden.shape == (1, 2, 4)
    assert logits.shape == (1, 2, 8)
    assert torch.isfinite(logits).all()


@pytest.mark.correctness
@requires_ltx
def test_dit_config_blob_parses():
    """DiT GGUFs (ComfyUI-GGUF convention) carry the diffusers config as one opaque
    JSON blob, not <arch>.* KV (MIGRATION §2c). `gguf_dit_config` must json-parse it."""
    blob = gguf_dit_config(_LTX_DIT)
    assert isinstance(blob, dict)
    # LTX-2 diffusers config nests the transformer knobs
    assert "transformer" in blob or "_class_name" in blob
    # and gguf_config must route DiT arches to that helper, not misparse as an LLM
    with pytest.raises(Exception):
        gguf_config(_LTX_DIT)  # LLM path can't build a DiT; must point to gguf_dit_config
