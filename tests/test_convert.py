# SPDX-License-Identifier: MIT
"""HF -> .superl8 conversion: quantize a state dict, save, reload, build, and confirm
the offline path is numerically identical to runtime quantization (per-row/per-group
int8 scale depends only on its own row, so merge-then-quant == quant-then-merge).
Needs CUDA for the forward."""

import json
import os

import pytest
import torch

pytest.importorskip("superl8")

from superl8serve.convert import (
    _remap_qwen3_next,
    convert_hf_to_superl8,
    is_quantizable_linear,
    quantize_state_dict,
)
from superl8serve.loader import checkpoint_info, load_superl8_state_dict
from superl8serve.models import ModelConfig, ModelRunner, build_model

CUDA = torch.cuda.is_available()


def _cfg():
    return ModelConfig(
        arch="qwen3",
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        max_position_embeddings=256,
        head_dim=32,
        qk_norm=True,
        tie_word_embeddings=False,
    )


def _sd(cfg):
    def r(*s):
        return torch.randn(*s, dtype=torch.float16) * 0.1

    hd, nh, nkv, H = (
        cfg.resolved_head_dim(),
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.hidden_size,
    )
    sd = {
        "model.embed_tokens.weight": r(cfg.vocab_size, H),
        "model.norm.weight": r(H),
        "lm_head.weight": r(cfg.vocab_size, H),
    }
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = r(H)
        sd[f"{p}.post_attention_layernorm.weight"] = r(H)
        sd[f"{p}.self_attn.q_proj.weight"] = r(nh * hd, H)
        sd[f"{p}.self_attn.k_proj.weight"] = r(nkv * hd, H)
        sd[f"{p}.self_attn.v_proj.weight"] = r(nkv * hd, H)
        sd[f"{p}.self_attn.o_proj.weight"] = r(H, nh * hd)
        sd[f"{p}.self_attn.q_norm.weight"] = r(hd)
        sd[f"{p}.self_attn.k_norm.weight"] = r(hd)
        sd[f"{p}.mlp.gate_proj.weight"] = r(cfg.intermediate_size, H)
        sd[f"{p}.mlp.up_proj.weight"] = r(cfg.intermediate_size, H)
        sd[f"{p}.mlp.down_proj.weight"] = r(H, cfg.intermediate_size)
    return sd


def test_is_quantizable_linear_excludes_router_and_norms():
    assert is_quantizable_linear("model.layers.0.self_attn.q_proj.weight")
    assert is_quantizable_linear("model.layers.0.mlp.experts.3.down_proj.weight")
    assert is_quantizable_linear("lm_head.weight")
    assert not is_quantizable_linear("model.layers.0.mlp.gate.weight")  # MoE router
    assert not is_quantizable_linear("model.layers.0.input_layernorm.weight")
    assert not is_quantizable_linear("model.embed_tokens.weight")
    # Non-standard MLP names (LFM2 feed_forward.w1/2/3, Mistral-style) must be
    # quantized too -- the old name-allowlist silently left them fp16.
    assert is_quantizable_linear("model.layers.0.feed_forward.w1.weight")
    assert is_quantizable_linear("model.layers.0.feed_forward.w2.weight")
    assert is_quantizable_linear("model.layers.0.feed_forward.w3.weight")
    assert is_quantizable_linear("model.layers.0.self_attn.out_proj.weight")
    # ...but a router by any common name still stays fp (routing is sensitive).
    assert not is_quantizable_linear("model.layers.0.block_sparse_moe.gate.weight")
    assert not is_quantizable_linear("model.layers.0.mlp.router.weight")


def test_quantize_state_dict_schemes():
    cfg = _cfg()
    q8 = quantize_state_dict(_sd(cfg), weight_bits=8)
    assert q8["model.layers.0.self_attn.q_proj.weight"].scheme == "per_row_i8"
    assert q8["model.norm.weight"].scheme == "raw"
    assert q8["model.layers.0.mlp.gate_proj.weight"].scheme == "per_row_i8"
    q4 = quantize_state_dict(_sd(cfg), weight_bits=4, group_size=32)
    qt = q4["model.layers.0.mlp.down_proj.weight"]
    assert qt.scheme == "per_group_i4" and qt.codebook == "int4" and qt.group_size == 32


def _to_cuda(qsd):
    from superl8 import QTensor

    out = {}
    for k, v in qsd.items():
        if isinstance(v, QTensor):
            out[k] = QTensor(
                v.data.cuda(),
                v.scale.cuda() if v.scale is not None else None,
                scheme=v.scheme,
                group_size=v.group_size,
                codebook=v.codebook,
            )
        else:
            out[k] = v.cuda()
    return out


@pytest.mark.skipif(not CUDA, reason="forward needs CUDA")
def test_superl8_roundtrip_is_exact(tmp_path):
    """save_superl8 -> load must be byte-exact: the offline-loaded model equals one built
    directly from the same quantized dict, bitwise (isolates the container round-trip
    from any CPU/GPU quant ULP)."""
    from superl8 import save_superl8

    cfg = _cfg()
    qsd = quantize_state_dict(_sd(cfg), weight_bits=8)  # CPU QTensors

    path = str(tmp_path / "m.superl8")
    save_superl8(path, qsd)
    # unwrap raw QTensors -> tensors for the direct build (mirrors load_superl8_state_dict)
    direct = {k: (v.data if v.scheme == "raw" else v) for k, v in qsd.items()}
    model_direct = build_model(cfg, _to_cuda(direct)).cuda().eval()
    model_off = build_model(cfg, load_superl8_state_dict(path, device="cuda")).cuda().eval()

    prompt = torch.randint(0, cfg.vocab_size, (1, 5), device="cuda")
    r_d = ModelRunner(model_direct, cfg, max_batch=1, max_len=32, device="cuda")
    r_o = ModelRunner(model_off, cfg, max_batch=1, max_len=32, device="cuda")
    torch.testing.assert_close(r_o.prefill(prompt), r_d.prefill(prompt), rtol=0, atol=0)


@pytest.mark.skipif(not CUDA, reason="forward needs CUDA")
def test_superl8_model_close_to_runtime_quant(tmp_path):
    """Offline (.superl8, CPU-quant) vs runtime (GPU-quant) differ only by quant ULPs."""
    from superl8 import save_superl8

    cfg = _cfg()
    sd = _sd(cfg)
    path = str(tmp_path / "m.superl8")
    save_superl8(path, quantize_state_dict(sd, weight_bits=8))
    model_off = build_model(cfg, load_superl8_state_dict(path, device="cuda")).cuda().eval()
    model_rt = build_model(cfg, {k: v.cuda() for k, v in sd.items()}).cuda().eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 5), device="cuda")
    a = ModelRunner(model_off, cfg, max_batch=1, max_len=32, device="cuda").prefill(prompt)
    b = ModelRunner(model_rt, cfg, max_batch=1, max_len=32, device="cuda").prefill(prompt)
    cos = torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0)
    assert cos.item() >= 0.9999


def _hf_config_dict(cfg):
    return {
        "model_type": cfg.arch,
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "intermediate_size": cfg.intermediate_size,
        "max_position_embeddings": cfg.max_position_embeddings,
        "head_dim": cfg.head_dim,
    }


def test_convert_persists_extra_for_divergent_families(tmp_path):
    """A `.superl8` must carry the `extra` dict — every divergent family's hybrid layer
    metadata (LFM2 `layer_types`, MiniMax `attn_type_list`, Qwen3-Next linear head
    dims, DeepSeek `kv_lora_rank`) lives there and the model builders read it. The
    old meta dump dropped every dict field, so a real divergent checkpoint converted
    fine but could not be rebuilt (arch silently lost). Both load paths must recover
    it: `ModelConfig.from_hf(meta_cfg)` (api server) and `ModelConfig(**meta_cfg)`
    (bench). Uses a plain qwen3 checkpoint with extra HF keys injected — the metadata
    round-trip is arch-independent."""
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    import superl8serve.convert as convert_mod

    cfg = _cfg()
    sd = _sd(cfg)
    save_file(sd, str(tmp_path / "model.safetensors"))
    hf = _hf_config_dict(cfg)
    # Divergent-family-style keys that from_hf funnels into cfg.extra:
    hf["layer_types"] = ["conv", "full_attention"]
    hf["conv_L_cache"] = 3
    hf["linear_num_key_heads"] = 2
    (tmp_path / "config.json").write_text(json.dumps(hf))

    out = tmp_path / "m.superl8"
    convert_mod.convert_hf_to_superl8(str(tmp_path), str(out), weight_bits=8)

    meta_cfg = checkpoint_info(str(out))["meta"]["config"]
    assert "extra" in meta_cfg and meta_cfg["extra"].get("conv_L_cache") == 3

    from superl8serve.models.config import ModelConfig

    via_from_hf = ModelConfig.from_hf(meta_cfg, arch=meta_cfg.get("arch"))
    assert via_from_hf.extra.get("layer_types") == ["conv", "full_attention"]
    assert via_from_hf.extra.get("linear_num_key_heads") == 2

    via_ctor = ModelConfig(**meta_cfg)
    assert via_ctor.extra.get("conv_L_cache") == 3


def test_convert_hf_to_superl8_streams_shards_one_at_a_time(tmp_path, monkeypatch):
    """Regression for the 220GB-in-RAM OOM (GLM-4.5-Air): convert_hf_to_superl8 must
    quantize each safetensors shard as it's loaded rather than merging the whole
    checkpoint into one dict first. A synthetic 2-shard checkpoint stands in for the
    real multi-hundred-shard ones — the behavior under test (one quantize call per
    shard, never the full state dict at once) is shard-count-independent."""
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    import superl8serve.convert as convert_mod

    cfg = _cfg()
    sd = _sd(cfg)
    keys = list(sd)
    mid = len(keys) // 2
    save_file({k: sd[k] for k in keys[:mid]}, str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file({k: sd[k] for k in keys[mid:]}, str(tmp_path / "model-00002-of-00002.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(_hf_config_dict(cfg)))

    seen_shard_sizes = []
    orig_quantize = convert_mod.quantize_state_dict

    def spy(sd_arg, **kw):
        seen_shard_sizes.append(len(sd_arg))
        return orig_quantize(sd_arg, **kw)

    monkeypatch.setattr(convert_mod, "quantize_state_dict", spy)

    out = tmp_path / "m.superl8"
    convert_mod.convert_hf_to_superl8(str(tmp_path), str(out), weight_bits=8)

    assert seen_shard_sizes == [mid, len(keys) - mid]  # one call per shard...
    assert all(n < len(keys) for n in seen_shard_sizes)  # ...never the full state dict
    assert checkpoint_info(str(out))["num_tensors"] == len(keys)


def test_convert_hf_to_superl8_deletes_source_shards_as_processed(tmp_path, monkeypatch):
    """Regression for the 850GB+ disk-capacity failure (MiniMax-M3 / issue #69): with
    `delete_source=True`, convert_hf_to_superl8 deletes each source safetensors shard
    immediately after its tensors are quantized and written, so peak disk is
    max(source, consumed+output) rather than source+output. (This is opt-in — the
    default preserves the input; see the preserve-by-default test below for #200.)"""
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    import superl8serve.convert as convert_mod

    cfg = _cfg()
    sd = _sd(cfg)
    keys = list(sd)
    mid = len(keys) // 2
    save_file({k: sd[k] for k in keys[:mid]}, str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file({k: sd[k] for k in keys[mid:]}, str(tmp_path / "model-00002-of-00002.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(_hf_config_dict(cfg)))

    # Track which safetensors files get deleted
    removed_safetensors = []
    orig_remove = os.remove

    def spy_remove(path):
        if path.endswith(".safetensors"):
            removed_safetensors.append(path)
        return orig_remove(path)

    monkeypatch.setattr(os, "remove", spy_remove)

    # Shards exist before conversion
    assert (tmp_path / "model-00001-of-00002.safetensors").exists()
    assert (tmp_path / "model-00002-of-00002.safetensors").exists()
    shard_count_before = len(list(tmp_path.glob("*.safetensors")))

    out = tmp_path / "m.superl8"
    convert_mod.convert_hf_to_superl8(
        str(tmp_path), str(out), weight_bits=8, delete_source=True
    )

    # Every source shard was deleted via os.remove during conversion
    assert len(removed_safetensors) == 2
    assert shard_count_before == 2
    assert not (tmp_path / "model-00001-of-00002.safetensors").exists()
    assert not (tmp_path / "model-00002-of-00002.safetensors").exists()
    # config.json and the output file must survive
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "m.superl8").exists()
    assert checkpoint_info(str(out))["num_tensors"] == len(keys)


def test_convert_hf_to_superl8_preserves_source_by_default(tmp_path):
    """Regression for issue #200 (data-loss footgun): with no delete_source flag,
    convert_hf_to_superl8 must NOT touch the input checkpoint. The documented CLI points
    at a user's own model dir (`python -m superl8serve.convert /path/to/Qwen3-8B out.superl8`)
    — a converter that silently deletes its input is unacceptable. Destruction is
    strictly opt-in (delete_source=True / --free-source-shards), used only by the
    disposable-staging forge pipeline for 850GB+ models (#69)."""
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    cfg = _cfg()
    sd = _sd(cfg)
    keys = list(sd)
    mid = len(keys) // 2
    s1 = tmp_path / "model-00001-of-00002.safetensors"
    s2 = tmp_path / "model-00002-of-00002.safetensors"
    save_file({k: sd[k] for k in keys[:mid]}, str(s1))
    save_file({k: sd[k] for k in keys[mid:]}, str(s2))
    (tmp_path / "config.json").write_text(json.dumps(_hf_config_dict(cfg)))

    out = tmp_path / "m.superl8"
    convert_hf_to_superl8(str(tmp_path), str(out), weight_bits=8)  # default: no delete

    # Input checkpoint is fully intact...
    assert s1.exists() and s2.exists()
    assert len(list(tmp_path.glob("*.safetensors"))) == 2
    # ...and the conversion still produced a complete output.
    assert out.exists()
    assert checkpoint_info(str(out))["num_tensors"] == len(keys)


def test_convert_hf_to_superl8_multi_shard_output_matches_single_shard(tmp_path):
    """However the checkpoint is split into shards, the quantized output must be
    identical (per-row/per-group scales depend only on a tensor's own values)."""
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    cfg = _cfg()
    sd = _sd(cfg)
    cfg_json = json.dumps(_hf_config_dict(cfg))

    one_shard_dir = tmp_path / "one"
    one_shard_dir.mkdir()
    save_file(sd, str(one_shard_dir / "model.safetensors"))
    (one_shard_dir / "config.json").write_text(cfg_json)
    out_one = one_shard_dir / "m.superl8"
    convert_hf_to_superl8(str(one_shard_dir), str(out_one), weight_bits=8)

    many_shard_dir = tmp_path / "many"
    many_shard_dir.mkdir()
    for i, k in enumerate(sd):
        save_file({k: sd[k]}, str(many_shard_dir / f"model-{i:05d}.safetensors"))
    (many_shard_dir / "config.json").write_text(cfg_json)
    out_many = many_shard_dir / "m.superl8"
    convert_hf_to_superl8(str(many_shard_dir), str(out_many), weight_bits=8)

    info_one, info_many = checkpoint_info(str(out_one)), checkpoint_info(str(out_many))
    assert info_one["num_tensors"] == info_many["num_tensors"] == len(sd)
    # header "arch" is the hardware target (sm70); the model arch lives in meta.
    assert info_one["arch"] == info_many["arch"]
    assert info_one["meta"]["arch"] == info_many["meta"]["arch"] == cfg.arch


def _qwen3_next_cfg():
    return ModelConfig(
        arch="qwen3_next",
        vocab_size=64,
        hidden_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        head_dim=32,
        qk_norm=True,
        tie_word_embeddings=True,
        num_experts=2,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        linear_attention=True,
        full_attention_interval=1,
        extra=dict(
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
        ),
    )


def _hf_fused_sd(cfg):
    """Qwen3-Next state dict using the fused HF tensor names."""
    H, nk, nv, kd, vd = (
        cfg.hidden_size,
        cfg.extra["linear_num_key_heads"],
        cfg.extra["linear_num_value_heads"],
        cfg.extra["linear_key_head_dim"],
        cfg.extra["linear_value_head_dim"],
    )
    qk = nk * kd
    v_dim = nv * vd
    sd = {
        "model.embed_tokens.weight": torch.randn(cfg.vocab_size, H, dtype=torch.float16),
        "model.norm.weight": torch.randn(H, dtype=torch.float16),
    }
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = torch.randn(H, dtype=torch.float16)
        sd[f"{p}.post_attention_layernorm.weight"] = torch.randn(H, dtype=torch.float16)
        la = f"{p}.linear_attn"
        # Fused HF names
        sd[f"{la}.in_proj_qkvz.weight"] = torch.randn(
            qk + qk + v_dim + v_dim, H, dtype=torch.float16
        )
        sd[f"{la}.in_proj_ba.weight"] = torch.randn(2 * nv, H, dtype=torch.float16)
        sd[f"{la}.conv1d.weight"] = torch.randn(qk + qk + v_dim, 4, dtype=torch.float16)
        sd[f"{la}.out_proj.weight"] = torch.randn(H, v_dim, dtype=torch.float16)
        sd[f"{la}.A_log"] = torch.randn(nv).float()
        sd[f"{la}.dt_bias"] = torch.randn(nv).float()
        sd[f"{la}.norm.weight"] = torch.randn(v_dim, dtype=torch.float16)
    return sd


def test_qwen3_next_hf_fused_names_remap():
    """_remap_qwen3_next must split fused HF names into the per-name tensors the
    builder expects (qkv_proj, z_proj, beta_proj, dt_proj, conv_weight)."""
    cfg = _qwen3_next_cfg()
    hf_sd = _hf_fused_sd(cfg)
    mapped = _remap_qwen3_next(hf_sd, cfg)

    H = cfg.hidden_size
    nk = cfg.extra["linear_num_key_heads"]
    nv = cfg.extra["linear_num_value_heads"]
    kd = cfg.extra["linear_key_head_dim"]
    vd = cfg.extra["linear_value_head_dim"]
    qk = nk * kd
    v_dim = nv * vd

    # Fused names must be gone
    assert not any("in_proj_qkvz" in k for k in mapped)
    assert not any("in_proj_ba" in k for k in mapped)
    assert not any("conv1d" in k for k in mapped)

    # Builder-expected names must exist
    la = "model.layers.0.linear_attn"
    assert f"{la}.qkv_proj.weight" in mapped
    assert f"{la}.z_proj.weight" in mapped
    assert f"{la}.beta_proj.weight" in mapped
    assert f"{la}.dt_proj.weight" in mapped
    assert f"{la}.conv_weight" in mapped

    # Shape checks
    assert mapped[f"{la}.qkv_proj.weight"].shape == (qk + qk + v_dim, H)
    assert mapped[f"{la}.z_proj.weight"].shape == (v_dim, H)
    assert mapped[f"{la}.beta_proj.weight"].shape == (nv, H)
    assert mapped[f"{la}.dt_proj.weight"].shape == (nv, H)
    assert mapped[f"{la}.conv_weight"].shape == (qk + qk + v_dim, 4)

    # Non-linear-attn tensors pass through unchanged
    assert "model.embed_tokens.weight" in mapped
    assert "model.norm.weight" in mapped
    assert "model.layers.0.input_layernorm.weight" in mapped


def test_qwen3_next_conv1d_trim_4part():
    """conv1d.weight with 4 parts (including Z rows) is trimmed to 3 parts."""
    cfg = _qwen3_next_cfg()
    hf_sd = _hf_fused_sd(cfg)
    H = cfg.hidden_size
    nk = cfg.extra["linear_num_key_heads"]
    nv = cfg.extra["linear_num_value_heads"]
    kd = cfg.extra["linear_key_head_dim"]
    vd = cfg.extra["linear_value_head_dim"]
    qk = nk * kd
    v_dim = nv * vd
    qkv_dim = qk + qk + v_dim
    # Replace with 4-part conv weight
    hf_sd["model.layers.0.linear_attn.conv1d.weight"] = torch.randn(
        qkv_dim + v_dim, 4, dtype=torch.float16
    )
    mapped = _remap_qwen3_next(hf_sd, cfg)
    conv_w = mapped["model.layers.0.linear_attn.conv_weight"]
    assert conv_w.shape == (qkv_dim, 4), f"expected ({qkv_dim}, 4) got {conv_w.shape}"


def test_qwen3_5_vl_remap_prefix():
    """_remap_qwen3_next strips model.language_model.* and model.visual.*
    prefixes for Qwen3.5-VL checkpoints so the qwen3_next text builder binds."""
    H = 128
    sd = {
        "model.language_model.embed_tokens.weight": torch.randn(64, H, dtype=torch.float16),
        "model.language_model.norm.weight": torch.randn(H, dtype=torch.float16),
        "lm_head.weight": torch.randn(64, H, dtype=torch.float16),
        "model.language_model.layers.0.input_layernorm.weight": torch.randn(H, dtype=torch.float16),
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(128, H, dtype=torch.float16),
        "model.visual.blocks.0.attn.qkv.weight": torch.randn(128, 128, dtype=torch.float16),
        "model.visual.class_embedding": torch.randn(128, dtype=torch.float16),
        "mtp.0.pred_model.weight": torch.randn(64, H, dtype=torch.float16),
    }
    cfg = ModelConfig(
        arch="qwen3_5_vl", vocab_size=64, hidden_size=H,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=64, head_dim=32,
    )
    mapped = _remap_qwen3_next(sd, cfg)

    # Text prefixes stripped: model.language_model.* -> model.*
    assert "model.embed_tokens.weight" in mapped
    assert "model.norm.weight" in mapped
    assert "lm_head.weight" in mapped
    assert "model.layers.0.input_layernorm.weight" in mapped
    assert "model.layers.0.self_attn.q_proj.weight" in mapped
    # Visual prefixes stripped: model.visual.* -> visual.*
    assert "visual.blocks.0.attn.qkv.weight" in mapped
    assert "visual.class_embedding" in mapped
    # MTP passes through unchanged
    assert "mtp.0.pred_model.weight" in mapped
    # Original prefixed names must be gone
    assert not any("model.language_model" in k for k in mapped)
    assert not any("model.visual." in k for k in mapped)


def test_qwen3_5_vl_remap_prefix_also_splits_fused_linear_attn_names():
    """For Qwen3.5-VL, prefix stripping runs BEFORE the fused-name split, so
    model.language_model.layers.0.linear_attn.in_proj_qkvz.weight is first
    renamed to layers.0.linear_attn.in_proj_qkvz.weight, then split into
    qkv_proj + z_proj."""
    H = 128
    nk, nv, kd, vd = 2, 4, 16, 16
    qk = nk * kd
    v_dim = nv * vd
    sd = {
        "model.language_model.embed_tokens.weight": torch.randn(64, H, dtype=torch.float16),
        "model.language_model.norm.weight": torch.randn(H, dtype=torch.float16),
        "lm_head.weight": torch.randn(64, H, dtype=torch.float16),
    }
    p = "model.language_model.layers.0"
    la = f"{p}.linear_attn"
    sd[f"{la}.in_proj_qkvz.weight"] = torch.randn(qk + qk + v_dim + v_dim, H, dtype=torch.float16)
    sd[f"{la}.in_proj_ba.weight"] = torch.randn(2 * nv, H, dtype=torch.float16)
    sd[f"{la}.conv1d.weight"] = torch.randn(qk + qk + v_dim, 4, dtype=torch.float16)
    sd[f"{la}.out_proj.weight"] = torch.randn(H, v_dim, dtype=torch.float16)
    sd[f"{la}.A_log"] = torch.randn(nv).float()
    sd[f"{la}.dt_bias"] = torch.randn(nv).float()
    sd[f"{la}.norm.weight"] = torch.randn(v_dim, dtype=torch.float16)
    sd["model.language_model.layers.0.post_attention_layernorm.weight"] = torch.randn(H, dtype=torch.float16)
    sd["model.language_model.layers.0.mlp.gate_proj.weight"] = torch.randn(64, H, dtype=torch.float16)
    sd["model.language_model.layers.0.mlp.up_proj.weight"] = torch.randn(64, H, dtype=torch.float16)
    sd["model.language_model.layers.0.mlp.down_proj.weight"] = torch.randn(H, 64, dtype=torch.float16)

    cfg = ModelConfig(
        arch="qwen3_5_vl", vocab_size=64, hidden_size=H,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=64, head_dim=32,
        extra=dict(
            linear_num_key_heads=nk, linear_num_value_heads=nv,
            linear_key_head_dim=kd, linear_value_head_dim=vd,
            linear_conv_kernel_dim=4,
        ),
    )
    mapped = _remap_qwen3_next(sd, cfg)

    # Fused names must be gone
    assert not any("in_proj_qkvz" in k for k in mapped)
    assert not any("in_proj_ba" in k for k in mapped)
    assert not any("conv1d" in k for k in mapped)

    # Builder-expected names must exist with correct prefix (model.layers.0.*)
    la_out = "model.layers.0.linear_attn"
    assert f"{la_out}.qkv_proj.weight" in mapped
    assert f"{la_out}.z_proj.weight" in mapped
    assert f"{la_out}.beta_proj.weight" in mapped
    assert f"{la_out}.dt_proj.weight" in mapped
    assert f"{la_out}.conv_weight" in mapped
    assert f"{la_out}.out_proj.weight" in mapped
    # Standard layers pass through
    assert "model.layers.0.post_attention_layernorm.weight" in mapped
    assert "model.layers.0.mlp.gate_proj.weight" in mapped


def test_qwen3_next_remap_skips_other_archs():
    """_remap_qwen3_next is a no-op for non-qwen3_next architectures."""
    cfg = _cfg()
    sd = _sd(cfg)
    mapped = _remap_qwen3_next(sd, cfg)
    # same keys, same tensors
    assert set(mapped.keys()) == set(sd.keys())
    for k in sd:
        assert mapped[k] is sd[k]


def test_convert_detects_qk_norm_from_weights(tmp_path):
    """A Qwen3 checkpoint WITH q_norm/k_norm weights must round-trip qk_norm=True
    in the .superl8 metadata, even when the HF config.json lacks an explicit qk_norm
    field (issue #171)."""
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    import superl8serve.convert as convert_mod

    cfg = _cfg()
    cfg.qk_norm = False  # simulate the bug: converter must detect from weights
    sd = _sd(cfg)  # includes q_norm/k_norm weights
    save_file(sd, str(tmp_path / "model.safetensors"))
    hf = _hf_config_dict(cfg)  # no qk_norm key
    (tmp_path / "config.json").write_text(json.dumps(hf))

    out = tmp_path / "m.superl8"
    convert_mod.convert_hf_to_superl8(str(tmp_path), str(out), weight_bits=8)

    meta_cfg = checkpoint_info(str(out))["meta"]["config"]
    assert meta_cfg["qk_norm"] is True, (
        "Qwen3 checkpoint with q_norm/k_norm weights must record qk_norm=True"
    )
    assert checkpoint_info(str(out))["num_tensors"] == len(sd)


def test_convert_qk_norm_stays_false_without_norm_weights(tmp_path):
    """A checkpoint WITHOUT q_norm/k_norm weights must keep qk_norm=False."""
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    cfg = _cfg()
    cfg.qk_norm = False
    sd = _sd(cfg)
    # Strip q_norm/k_norm weights
    for k in list(sd):
        if ".q_norm." in k or ".k_norm." in k:
            del sd[k]
    save_file(sd, str(tmp_path / "model.safetensors"))
    hf = _hf_config_dict(cfg)
    (tmp_path / "config.json").write_text(json.dumps(hf))

    out = tmp_path / "m.superl8"
    convert_hf_to_superl8(str(tmp_path), str(out), weight_bits=8)

    meta_cfg = checkpoint_info(str(out))["meta"]["config"]
    assert meta_cfg["qk_norm"] is False, (
        "Checkpoint without q_norm/k_norm weights must keep qk_norm=False"
    )
    assert checkpoint_info(str(out))["num_tensors"] == len(sd)
