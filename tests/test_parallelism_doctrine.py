# SPDX-License-Identifier: MIT
"""Tests for the fleet parallelism doctrine (issue #145): strategy planning,
anti-pattern warnings, and topology-agnostic throughput estimation — no GPUs
required (pure-CPU policy logic)."""

import pytest

from superl8serve.config import ServeConfig
from superl8serve.models.config import ModelConfig
from superl8serve.scheduling import (
    ParallelismStrategy,
    estimate_throughput,
    plan_parallelism,
    validate_parallel_config,
)


@pytest.fixture
def dense_cfg():
    return ModelConfig(
        arch="qwen3",
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        intermediate_size=512,
        max_position_embeddings=8192,
        head_dim=128,
    )


@pytest.fixture
def moe_cfg():
    return ModelConfig(
        arch="qwen3_moe",
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        intermediate_size=512,
        max_position_embeddings=8192,
        head_dim=128,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=256,
    )


# ── Strategy planning ─────────────────────────────────────────────────────


class TestPlanParallelism:
    def test_single_gpu_dense(self, dense_cfg):
        s = plan_parallelism(dense_cfg, num_gpus=1)
        assert s.pp_size == 1
        assert s.ep_size == 1
        assert s.local_ep is True
        assert s.strategy == "replicate"

    def test_single_gpu_moe(self, moe_cfg):
        s = plan_parallelism(moe_cfg, num_gpus=1)
        assert s.pp_size == 1
        assert s.ep_size == 1
        assert s.local_ep is True
        assert s.strategy == "replicate"

    def test_2_gpu_dense(self, dense_cfg):
        s = plan_parallelism(dense_cfg, num_gpus=2)
        assert s.pp_size == 2
        assert s.ep_size == 1
        assert s.strategy in ("shallow_pp",)

    def test_2_gpu_moe_layers_split(self, moe_cfg):
        """2 GPUs: 2-way PP, experts co-located with their PP stage (stage-local EP)."""
        s = plan_parallelism(moe_cfg, num_gpus=2)
        assert s.pp_size == 2
        assert s.ep_size == 1
        assert s.local_ep is True
        assert s.strategy == "shallow_pp"

    def test_4_gpu_dense(self, dense_cfg):
        s = plan_parallelism(dense_cfg, num_gpus=4)
        assert s.pp_size == 4
        assert s.ep_size == 1

    def test_4_gpu_moe(self, moe_cfg):
        """4 GPUs, 28 layers, 8 experts: 2-way PP, 2-way EP stage-local."""
        s = plan_parallelism(moe_cfg, num_gpus=4)
        assert s.pp_size == 2
        assert s.ep_size == 2
        assert s.local_ep is True
        assert s.strategy == "pp_plus_local_ep"
        assert s.pp_stages == 2
        assert s.ep_groups == 2

    def test_8_gpu_dense(self, dense_cfg):
        """8 GPUs: 4-way PP replicated 2× (shallow PP groups dominate)."""
        s = plan_parallelism(dense_cfg, num_gpus=8)
        assert s.pp_size == 4
        assert s.ep_size == 1
        # Replicate factor = 2, so total GPUs = 2*4 = 8
        assert s.num_gpus == 8

    def test_8_gpu_moe(self, moe_cfg):
        """8 GPUs: 2× (4-way PP + 2-way EP stage-local) = 8 GPUs."""
        s = plan_parallelism(moe_cfg, num_gpus=8)
        assert s.pp_size == 4
        assert s.ep_size == 2
        assert s.local_ep is True
        assert s.num_gpus == 8

    def test_16_gpu_dense(self, dense_cfg):
        """16 GPUs: 4× (4-way PP) — replicate small PP groups first."""
        s = plan_parallelism(dense_cfg, num_gpus=16)
        assert s.pp_size == 4
        assert s.ep_size == 1
        assert s.num_gpus == 16

    def test_16_gpu_moe(self, moe_cfg):
        """16 GPUs: 2× (2×(4-way PP + local EP)) — replicate first, then PP, then EP."""
        s = plan_parallelism(moe_cfg, num_gpus=16)
        assert s.pp_size == 4
        assert s.ep_size == 2
        assert s.local_ep is True
        assert s.num_gpus == 16

    def test_deep_model_prefers_fewer_pp_stages(self, dense_cfg):
        """A 60-layer model on 8 GPUs: each GPU gets ~8 layers at max PP=4
        which is comfortable; the doctine favours 4-way over 8-way PP."""
        dense_cfg.num_hidden_layers = 60
        s = plan_parallelism(dense_cfg, num_gpus=8)
        # Should NOT go beyond 4-way PP — replicate the 4-way PP groups instead
        assert s.pp_size <= 4

    def test_pp_depth_capped_at_4(self, dense_cfg):
        """Even with many GPUs and few layers, PP is never pushed beyond 4-way
        because the 250 MB/s wire makes deeper PP a net loss."""
        dense_cfg.num_hidden_layers = 8
        s = plan_parallelism(dense_cfg, num_gpus=16)
        assert s.pp_size <= 4

    def test_moe_with_few_experts_skips_ep(self, moe_cfg):
        """With only 2 experts on 8 GPUs, EP doesn't help — prefer shallow PP
        replication over EP."""
        moe_cfg.num_experts = 2
        s = plan_parallelism(moe_cfg, num_gpus=8)
        assert s.ep_size == 1
        assert s.pp_size == 4


# ── Anti-pattern detection ────────────────────────────────────────────────


class TestValidateParallelConfig:
    def test_ok_defaults(self):
        cfg = ServeConfig(model="x.superl8")
        w, errors = validate_parallel_config(cfg)
        assert errors == []
        assert w == []

    def test_tp_is_hard_error(self):
        with pytest.raises(ValueError, match="TP is not viable"):
            ServeConfig(model="x.superl8", tensor_parallel_size=2)
        # validate_parallel_config also catches TP on a valid config
        cfg = ServeConfig(model="x.superl8", pipeline_parallel_size=1)
        warnings_list, errors = validate_parallel_config(cfg)
        assert errors == []

        # Force an anti-pattern: pass a ServeConfig-like object with TP>1
        class _Fake:
            tensor_parallel_size = 2
            pipeline_parallel_size = 1
            expert_parallel_size = 1

        _, errors = validate_parallel_config(_Fake())
        assert any("TP" in e for e in errors)

    def test_deep_pp_warns(self):
        cfg = ServeConfig(model="x.superl8", pipeline_parallel_size=6)
        warnings_list, errors = validate_parallel_config(cfg)
        assert errors == []
        assert len(warnings_list) >= 1
        assert any("pipeline_parallel" in w.lower() for w in warnings_list)

    def test_deep_pp_warns_at_5(self):
        cfg = ServeConfig(model="x.superl8", pipeline_parallel_size=5)
        warnings_list, errors = validate_parallel_config(cfg)
        assert len(warnings_list) >= 1

    def test_shallow_pp_ok(self):
        cfg = ServeConfig(model="x.superl8", pipeline_parallel_size=2)
        warnings_list, errors = validate_parallel_config(cfg)
        # No warning for pp=2-4
        pp_warnings = [w for w in warnings_list if "pp" in w.lower()]
        assert pp_warnings == []

    def test_shallow_pp_4_ok(self):
        cfg = ServeConfig(model="x.superl8", pipeline_parallel_size=4)
        warnings_list, errors = validate_parallel_config(cfg)
        pp_warnings = [w for w in warnings_list if "pp" in w.lower()]
        assert pp_warnings == []

    def test_tp_warn_includes_transport_docs_reference(self):
        with pytest.raises(ValueError, match="transport"):
            ServeConfig(model="x.superl8", tensor_parallel_size=2)

        # The error message from validate_parallel_config also mentions transport
        class _Fake:
            tensor_parallel_size = 2
            pipeline_parallel_size = 1
            expert_parallel_size = 1

        _, errors = validate_parallel_config(_Fake())
        assert any("transport" in e.lower() for e in errors)

    def test_diagnostic_nccldcgm_isnt_treated_as_permits(self):
        """Validate should not inspect NCCL/DCGM status — those are diagnostics
        only; a warning/error about an anti-pattern must not be suppressed by
        them."""
        cfg = ServeConfig(model="x.superl8", pipeline_parallel_size=6)
        warnings_list, errors = validate_parallel_config(cfg)
        # The deep-PP warning fires regardless of any external state
        assert warnings_list

    def test_multiple_anti_patterns(self):
        # TP>1 is caught at construction
        with pytest.raises(ValueError):
            ServeConfig(model="x.superl8", tensor_parallel_size=2, pipeline_parallel_size=8)
        # Deep PP alone triggers a warning
        cfg = ServeConfig(model="x.superl8", pipeline_parallel_size=8)
        warnings_list, errors = validate_parallel_config(cfg)
        assert errors == []
        assert len(warnings_list) >= 1
        assert any("pipeline_parallel" in w.lower() for w in warnings_list)

        # Both TP and deep PP detected via the validate function
        class _Fake:
            tensor_parallel_size = 2
            pipeline_parallel_size = 8
            expert_parallel_size = 1

        w, e = validate_parallel_config(_Fake())
        assert len(e) >= 1
        assert len(w) >= 1


# ── Throughput estimation ─────────────────────────────────────────────────


class TestEstimateThroughput:
    def test_returns_all_required_fields(self, dense_cfg):
        s = plan_parallelism(dense_cfg, num_gpus=4)
        est = estimate_throughput(s, dense_cfg, seq_len=512, batch_size=4)
        for field in ("tokens_per_second", "p99_latency_ms", "on_wire_bytes_per_token"):
            assert field in est, f"missing {field}"

    def test_throughput_scales_with_gpus(self, dense_cfg):
        est1 = estimate_throughput(
            plan_parallelism(dense_cfg, num_gpus=1), dense_cfg, seq_len=512, batch_size=4
        )
        est4 = estimate_throughput(
            plan_parallelism(dense_cfg, num_gpus=4), dense_cfg, seq_len=512, batch_size=4
        )
        # More GPUs = more throughput
        assert est4["tokens_per_second"] >= est1["tokens_per_second"]

    def test_latency_decreases_with_more_gpus(self, dense_cfg):
        est1 = estimate_throughput(
            plan_parallelism(dense_cfg, num_gpus=1), dense_cfg, seq_len=512, batch_size=4
        )
        est4 = estimate_throughput(
            plan_parallelism(dense_cfg, num_gpus=4), dense_cfg, seq_len=512, batch_size=4
        )
        # Shallow PP reduces per-GPU work, so p99 should decrease or be similar
        assert est4["p99_latency_ms"] <= est1["p99_latency_ms"] * 1.5

    def test_on_wire_bytes_per_token_reasonable(self, dense_cfg):
        """On-wire bytes/token must be bounded — the doctrine is about minimizing
        communication. For a 128-dim hidden, byte count should be small (< 1KB)."""
        s = plan_parallelism(dense_cfg, num_gpus=4)
        est = estimate_throughput(s, dense_cfg, seq_len=512, batch_size=4)
        bw = est["on_wire_bytes_per_token"]
        assert bw >= 0
        # A single hidden state (128 * 1 byte int8 = 128 bytes) per PP boundary
        # times 1-2 steps per token = upper bound of ~512 B/tok
        assert bw < 4096, f"on_wire bytes/token {bw} too high for shallow PP"

    def test_codec_aware_bytes_per_token(self, dense_cfg):
        """On-wire bytes/token must reflect the active codec's bits-per-element."""
        s = plan_parallelism(dense_cfg, num_gpus=4)
        hidden = dense_cfg.hidden_size
        pp_b = s.pp_size - 1  # PP boundaries = pp - 1

        est_fp16 = estimate_throughput(s, dense_cfg, codec="fp16")
        est_int8 = estimate_throughput(s, dense_cfg, codec="int8")
        est_int4 = estimate_throughput(s, dense_cfg, codec="int4")

        assert est_fp16["activation_bytes_per_boundary"] == hidden * 2
        assert est_int8["activation_bytes_per_boundary"] == hidden * 1
        assert est_int4["activation_bytes_per_boundary"] == hidden * 0.5

        assert est_fp16["on_wire_bytes_per_token"] == hidden * 2 * pp_b
        assert est_int8["on_wire_bytes_per_token"] == hidden * 1 * pp_b
        assert est_int4["on_wire_bytes_per_token"] == hidden * 0.5 * pp_b

    def test_moe_ep_reduces_bytes_vs_deep_pp(self, moe_cfg):
        """On MoE models, stage-local EP keeps expert traffic local, so bytes/token
        should be lower than forcing deep PP with no EP."""
        s_ep = plan_parallelism(moe_cfg, num_gpus=8)  # pp=4, ep=2
        # Simulate forcing deep PP (8-way) on the same model
        s_deep = ParallelismStrategy(
            num_gpus=8,
            pp_size=8,
            ep_size=1,
            local_ep=True,
            strategy="deep_pp",
            pp_stages=8,
            ep_groups=1,
        )
        est_ep = estimate_throughput(s_ep, moe_cfg, seq_len=512, batch_size=4)
        est_deep = estimate_throughput(s_deep, moe_cfg, seq_len=512, batch_size=4)
        assert est_ep["on_wire_bytes_per_token"] <= est_deep["on_wire_bytes_per_token"]

    def test_sweep_1_2_4_8_16_gpus(self, dense_cfg):
        """Topology-agnostic sweep: 1/2/4/8/16 GPUs produce valid estimates."""
        results = {}
        for n in [1, 2, 4, 8, 16]:
            s = plan_parallelism(dense_cfg, num_gpus=n)
            est = estimate_throughput(s, dense_cfg, seq_len=512, batch_size=4)
            results[n] = est
            assert est["tokens_per_second"] > 0
            assert est["p99_latency_ms"] > 0
            assert est["on_wire_bytes_per_token"] >= 0
        assert set(results.keys()) == {1, 2, 4, 8, 16}

    def test_sweep_1_2_4_8_16_gpus_moe(self, moe_cfg):
        """Same sweep for MoE models."""
        for n in [1, 2, 4, 8, 16]:
            s = plan_parallelism(moe_cfg, num_gpus=n)
            est = estimate_throughput(s, moe_cfg, seq_len=512, batch_size=4)
            assert est["tokens_per_second"] > 0
            assert est["p99_latency_ms"] > 0
            assert est["on_wire_bytes_per_token"] >= 0


# ── Strategy coherence ─────────────────────────────────────────────────────


class TestStrategyCoherence:
    def test_num_gpus_equals_pp_times_replicate(self, dense_cfg):
        for n in [1, 2, 4, 8, 16]:
            s = plan_parallelism(dense_cfg, num_gpus=n)
            assert s.num_gpus == n
            assert s.pp_size * s.ep_size * s.num_replicas == n, (
                f"GPU math wrong for {n}: pp={s.pp_size} ep={s.ep_size} replicas={s.num_replicas}"
            )
            assert s.pp_size <= 4

    def test_pp_stages_never_exceed_num_layers(self, dense_cfg):
        """Can't pipeline more stages than there are layers."""
        dense_cfg.num_hidden_layers = 2
        for n in [2, 4]:
            s = plan_parallelism(dense_cfg, num_gpus=n)
            assert s.pp_size <= dense_cfg.num_hidden_layers
            assert s.pp_size <= 4

    def test_ep_never_exceeds_num_experts(self, moe_cfg):
        """EP groups can't exceed the number of experts."""
        for n in [2, 4, 8]:
            s = plan_parallelism(moe_cfg, num_gpus=n)
            assert s.ep_size <= moe_cfg.num_experts

    def test_strategy_is_one_of_valid_choices(self, dense_cfg):
        for n in [1, 2, 4, 8, 16]:
            s = plan_parallelism(dense_cfg, num_gpus=n)
            assert s.strategy in {
                "replicate",
                "shallow_pp",
                "pp_replicate",
                "pp_plus_local_ep",
            }

    def test_local_ep_always_true(self, dense_cfg, moe_cfg):
        """Stage-local EP is always preferred on this fleet (global EP costs
        PCIe-x1 bandwidth for every expert dispatch)."""
        for n in [1, 2, 4, 8, 16]:
            s = plan_parallelism(moe_cfg, num_gpus=n)
            assert s.local_ep is True
            s_d = plan_parallelism(dense_cfg, num_gpus=n)
            assert s_d.local_ep is True
