# SPDX-License-Identifier: MIT
"""Qwen3.5-9B (real `Qwen/Qwen3.5-9B`) — DENSE hybrid (Gated-DeltaNet linear + GATED
full attention).

Covers the three deliverable stages:
  * config: `from_hf` derives the right hybrid layer pattern / head dims / gate flag
    from the REAL Qwen3.5-9B config (verifies #204 for this checkpoint);
  * convert: the fused linear-attn remap fires for the `qwen3_5` arch;
  * build + prefill + full autoregressive decode through the standalone ModelRunner,
    where decode must match a single teacher-forced forward.

The full-attention layers exercise the `attn_output_gate` path (query|gate split,
`sigmoid(gate)` before o_proj). head_dim is 256 (the real value) to prove the int8
prefill/decode kernels handle it end-to-end.
"""

import pytest
import torch

pytest.importorskip("superl8")

from superl8serve.convert import _remap_qwen3_next
from superl8serve.models import ModelRunner, build_model, is_supported
from superl8serve.models.base import ForwardContext
from superl8serve.models.cache import KVCache
from superl8serve.models.config import ModelConfig as _MC

CUDA = torch.cuda.is_available()


# The real Qwen/Qwen3.5-9B config (text_config trimmed to the load-bearing axes).
_REAL_HF = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "image_token_id": 248056,
    "model_type": "qwen3_5",
    "tie_word_embeddings": False,
    "text_config": {
        "attention_bias": False,
        "attn_output_gate": True,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 8,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "max_position_embeddings": 262144,
        "mlp_only_layers": [],
        "model_type": "qwen3_5_text",
        "mtp_num_hidden_layers": 1,
        "num_attention_heads": 16,
        "num_hidden_layers": 32,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-06,
        "vocab_size": 248320,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        },
    },
    "vision_config": {"depth": 27, "hidden_size": 1152},
}


def test_qwen3_5_registered():
    assert is_supported("qwen3_5")


def test_qwen36_moe_legacy_superl8_meta_recovers_family_semantics():
    """The fleet's 35B-A3B artifact predates nested Qwen3.6 config preservation."""
    from superl8serve.models.registry import _resolve

    legacy = {
        "arch": "qwen3_5_moe",
        "vocab_size": 248320,
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "intermediate_size": 0,
        "max_position_embeddings": 262144,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1e6,
        "partial_rotary_factor": 1.0,
        "linear_attention": False,
        "full_attention_interval": 0,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
    }
    cfg = _MC.from_hf(legacy, arch=legacy["arch"])

    assert _resolve(cfg.arch) == "qwen3_5"
    assert cfg.linear_attention is True
    assert cfg.full_attention_interval == 4
    assert cfg.rope_theta == 1e7
    assert cfg.partial_rotary_factor == 0.25
    assert cfg.torch_dtype == "bfloat16"
    assert cfg.attention_kind(0) == "linear"
    assert cfg.attention_kind(3) == "full"
    assert is_supported("Qwen3_5ForConditionalGeneration")
    assert is_supported("Qwen3_5ForCausalLM")


def test_qwen3_5_from_hf_real_config():
    """The real Qwen3.5-9B config must derive: hybrid 3:1 linear/full pattern,
    dense MLP (no experts), partial-rotary 0.25 over head_dim 256, rope_theta 1e7,
    the output-gate flag, and the linear head dims (#204 + this PR's mtp key)."""
    c = _MC.from_hf(_REAL_HF)
    # The real Qwen3.5 config carries a vision_config, so it IS a VLM: from_hf routes
    # it to the multimodal `qwen3_5_vl` builder (which composes THIS text backbone with
    # the vision tower). The text-backbone axes asserted below are derived identically.
    assert c.arch == "qwen3_5_vl"
    assert c.is_multimodal and c.vision_config is not None
    assert c.linear_attention and c.full_attention_interval == 4
    assert [c.attention_kind(i) for i in range(4)] == ["linear", "linear", "linear", "full"]
    assert c.resolved_head_dim() == 256 and c.rotary_dim() == 64
    assert c.partial_rotary_factor == 0.25 and c.rope_theta == 10000000
    assert c.num_experts == 0 and not c.is_moe()  # DENSE — unlike Qwen3-Next MoE
    assert c.intermediate_size == 12288
    assert not c.qkv_bias
    assert c.extra["attn_output_gate"] is True
    assert c.extra["linear_num_key_heads"] == 16 and c.extra["linear_num_value_heads"] == 32
    assert c.extra["linear_key_head_dim"] == 128 and c.extra["linear_value_head_dim"] == 128
    assert c.num_mtp_layers == 1  # read from mtp_num_hidden_layers


def test_qwen3_5_convert_remaps_fused_linear_attn():
    """The converter's fused in_proj remap must fire for the qwen3_5 arch, splitting
    HF's in_proj_qkvz -> qkv_proj|z_proj and in_proj_ba -> beta_proj|dt_proj."""
    nk, nv, kd, vd = 2, 4, 16, 16
    cfg = _MC(
        arch="qwen3_5",
        vocab_size=32,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        linear_attention=True,
        full_attention_interval=2,
        extra=dict(
            linear_num_key_heads=nk,
            linear_num_value_heads=nv,
            linear_key_head_dim=kd,
            linear_value_head_dim=vd,
        ),
    )
    H = 64
    qk, vdim = nk * kd, nv * vd
    qkvz = qk + qk + vdim
    sd = {
        "model.layers.0.linear_attn.in_proj_qkvz.weight": torch.randn(qkvz + vdim, H),
        "model.layers.0.linear_attn.in_proj_ba.weight": torch.randn(2 * nv, H),
        "model.layers.0.linear_attn.conv1d.weight": torch.randn(qkvz + vdim, 4),
    }
    out = _remap_qwen3_next(sd, cfg)
    assert out["model.layers.0.linear_attn.qkv_proj.weight"].shape[0] == qkvz
    assert out["model.layers.0.linear_attn.z_proj.weight"].shape[0] == vdim
    assert out["model.layers.0.linear_attn.beta_proj.weight"].shape[0] == nv
    assert out["model.layers.0.linear_attn.dt_proj.weight"].shape[0] == nv
    assert out["model.layers.0.linear_attn.conv_weight"].shape[0] == qkvz


# ── synthetic DENSE hybrid checkpoint (gated full attn + DeltaNet linear) ──
_H, _NH, _NKV, _HD = 128, 2, 1, 256  # head_dim 256 == real Qwen3.5-9B
_NK, _NV, _KD, _VD = 1, 2, 16, 16  # linear-attn head dims (kept tiny)
_INTER = 64


def _cfg():
    return _MC(
        arch="qwen3_5",
        vocab_size=64,
        hidden_size=_H,
        num_hidden_layers=2,
        num_attention_heads=_NH,
        num_key_value_heads=_NKV,
        intermediate_size=_INTER,
        max_position_embeddings=64,
        head_dim=_HD,
        qk_norm=True,
        tie_word_embeddings=False,
        rope_theta=1e7,
        partial_rotary_factor=0.25,
        linear_attention=True,
        full_attention_interval=2,
        extra=dict(
            linear_num_key_heads=_NK,
            linear_num_value_heads=_NV,
            linear_key_head_dim=_KD,
            linear_value_head_dim=_VD,
            linear_conv_kernel_dim=4,
            attn_output_gate=True,
        ),
    )


def _sd(cfg, device="cpu"):
    def r(*s):
        return torch.randn(*s, device=device, dtype=torch.float16) * 0.05

    qkv_lin = 2 * _NK * _KD + _NV * _VD
    sd = {
        "model.embed_tokens.weight": r(cfg.vocab_size, _H),
        "model.norm.weight": r(_H),
        "lm_head.weight": r(cfg.vocab_size, _H),
    }
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = r(_H)
        sd[f"{p}.post_attention_layernorm.weight"] = r(_H)
        if cfg.attention_kind(i) == "full":
            # q_proj emits query|gate -> 2*nh*hd rows (attn_output_gate)
            sd[f"{p}.self_attn.q_proj.weight"] = r(2 * _NH * _HD, _H)
            sd[f"{p}.self_attn.k_proj.weight"] = r(_NKV * _HD, _H)
            sd[f"{p}.self_attn.v_proj.weight"] = r(_NKV * _HD, _H)
            sd[f"{p}.self_attn.o_proj.weight"] = r(_H, _NH * _HD)
            sd[f"{p}.self_attn.q_norm.weight"] = r(_HD)
            sd[f"{p}.self_attn.k_norm.weight"] = r(_HD)
        else:
            la = f"{p}.linear_attn"
            sd[f"{la}.qkv_proj.weight"] = r(qkv_lin, _H)
            sd[f"{la}.z_proj.weight"] = r(_NV * _VD, _H)
            sd[f"{la}.out_proj.weight"] = r(_H, _NV * _VD)
            sd[f"{la}.conv_weight"] = r(qkv_lin, 4)
            sd[f"{la}.A_log"] = r(_NV).float()
            sd[f"{la}.dt_bias"] = r(_NV).float()
            sd[f"{la}.beta_proj.weight"] = r(_NV, _H)
            sd[f"{la}.dt_proj.weight"] = r(_NV, _H)
            sd[f"{la}.norm.weight"] = r(_VD)  # gated DeltaNet norm is PER-HEAD (head_v_dim)
        # dense SwiGLU MLP (no experts)
        sd[f"{p}.mlp.gate_proj.weight"] = r(_INTER, _H)
        sd[f"{p}.mlp.up_proj.weight"] = r(_INTER, _H)
        sd[f"{p}.mlp.down_proj.weight"] = r(_H, _INTER)
    return sd


def test_qwen3_5_merges_deltanet_scalar_projections():
    cfg = _cfg()
    model = build_model(cfg, _sd(cfg))
    attn = model.layers[0].attn
    assert hasattr(attn, "gate_beta_proj")
    assert attn.gate_beta_proj.out_features == 2 * _NV
    assert not hasattr(attn, "gate_proj")
    assert not hasattr(attn, "beta_proj")


@pytest.mark.skipif(not CUDA, reason="prefill needs the dp4a kernels")
def test_qwen3_5_hybrid_prefill():
    cfg = _cfg()
    model = build_model(cfg, _sd(cfg, "cuda")).cuda().eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 6), device="cuda")
    pos = torch.arange(6, device="cuda").unsqueeze(0)
    cache = KVCache(
        cfg.num_hidden_layers,
        1,
        cfg.num_key_value_heads,
        16,
        cfg.resolved_head_dim(),
        device="cuda",
    )
    from superl8serve.models.cache import RecurrentStateCache

    ctx = ForwardContext(is_prefill=True, kv_cache=cache, lin_cache=RecurrentStateCache())
    h = model(ids, pos, ctx)
    logits = model.compute_logits(h[:, -1])
    assert logits.shape == (1, cfg.vocab_size) and torch.isfinite(logits).all()


@pytest.mark.skipif(not CUDA, reason="forward needs CUDA")
def test_qwen3_5_hybrid_decode_matches_teacher_forced():
    """Full generate (prefill + N decode steps) must match a single teacher-forced
    forward over the whole (prompt + generated) sequence — the property the recurrent
    state (linear layers) and per-layer KV cache (gated full layers) preserve."""
    cfg = _cfg()
    model = build_model(cfg, _sd(cfg, "cuda")).cuda().eval()
    runner = ModelRunner(model, cfg, max_batch=1, max_len=32, device="cuda")
    prompt = torch.randint(0, cfg.vocab_size, (1, 5), device="cuda")

    # Decode path: capture the per-step logits (not just tokens), stepping greedily.
    step_logits = [runner.prefill(prompt)[0]]
    tok = step_logits[-1].argmax(-1, keepdim=True).unsqueeze(0)
    gen = [tok]
    for _ in range(3):
        step_logits.append(runner.decode(tok)[0])
        tok = step_logits[-1].argmax(-1, keepdim=True).unsqueeze(0)
        gen.append(tok)
    gen = torch.cat(gen, dim=1)

    full_ids = torch.cat([prompt, gen], dim=1)
    pos = torch.arange(full_ids.shape[1], device="cuda").unsqueeze(0)
    from superl8serve.models.cache import RecurrentStateCache

    ref_cache = KVCache(
        cfg.num_hidden_layers, 1, cfg.num_key_value_heads, 32, cfg.resolved_head_dim(),
        device="cuda",
    )
    ctx = ForwardContext(is_prefill=True, kv_cache=ref_cache, lin_cache=RecurrentStateCache())
    hidden = model(full_ids, pos, ctx)
    tf_logits = model.compute_logits(hidden)[0, prompt.shape[1] - 1 : -1]  # [4, vocab]

    # The recurrent state + KV cache must make each decode step reproduce the
    # teacher-forced logits. int8 prefill vs decode kernels round slightly differently,
    # so compare with a cosine bar (AGENTS.md: int8 paths use cosine-sim, not argmax
    # equality) rather than exact token match, which is brittle at near-tie logits.
    for i in range(4):
        cos = torch.nn.functional.cosine_similarity(
            step_logits[i].float(), tf_logits[i].float(), dim=0
        )
        assert cos > 0.999, f"decode step {i} diverges from teacher-forced: cos={cos.item()}"


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


@pytest.mark.skipif(not CUDA, reason="engine paged decode needs the CUDA superl8 kernels")
@torch.inference_mode()
def test_qwen3_5_engine_paged_decode_gate_matches_standalone():
    """Qwen3.5-9B's gated full-attention layers must decode through the ENGINE's
    paged/continuous-batch path (slot_mapping/slot_lengths), not just the standalone
    ModelRunner. `GatedGQAAttention` now threads the sigmoid output gate through
    `GQAAttention._decode_batched` (the shared paged-decode kernels), so the whole
    dense-hybrid model generates under `LLMEngine`.

    Two properties, one per the two int8 error regimes:
      * ENGINE greedy decode == standalone ModelRunner greedy decode (the same
        weights + prompt), the end-to-end "it decodes under the batching engine" bar;
      * teacher-forced (SAME token fed to both) engine paged-decode logits match the
        standalone contiguous-cache logits within the int8 paged bar the shared
        `test_paged_engine` uses (cos > 0.995) — isolates the gated paged kernel
        from any autoregressive argmax drift.
    The engine forces per-sequence NON-varlen prefill for recurrent (hybrid) models,
    so this validates decode without touching the separate varlen-256 prefill gap.
    """
    from superl8serve.engine import LLMEngine, PagedKVCache, SamplingParams

    cfg = _cfg()
    # Identical weights for both models even if build_model consumes the dict.
    torch.manual_seed(0)
    sd_std = _sd(cfg, "cuda")
    torch.manual_seed(0)
    sd_eng = _sd(cfg, "cuda")

    model = build_model(cfg, sd_std).cuda().eval()
    runner = ModelRunner(model, cfg, max_batch=1, max_len=32, device="cuda")
    prompt = [3, 1, 4, 1, 5]
    gen_std = runner.generate_greedy(
        torch.tensor([prompt], device="cuda"), max_new_tokens=4
    )[0]

    eng = LLMEngine(
        cfg, sd_eng, device="cuda", max_num_seqs=1, max_len=32, enable_cuda_graph=False
    )
    gen_eng = eng.generate([prompt], SamplingParams(temperature=0.0, max_tokens=4))[0]

    assert gen_eng == gen_std.tolist(), (
        f"engine paged-decode greedy {gen_eng} != standalone {gen_std.tolist()}"
    )

    # Teacher-forced: same next token fed to a fresh standalone decode and a fresh
    # engine paged decode; the gated paged kernel path must match within int8 bar.
    next_tok = 7
    ref_runner = ModelRunner(model, cfg, max_batch=1, max_len=32, device="cuda")
    ref_runner.prefill(torch.tensor([prompt], device="cuda"))
    ref_decode = ref_runner.decode(torch.tensor([[next_tok]], device="cuda"))

    from superl8serve.models.cache import RecurrentStateCache

    cache = PagedKVCache(
        cfg.num_hidden_layers, 1, cfg.num_key_value_heads, 64,
        cfg.resolved_head_dim(), device="cuda",
    )
    lin_cache = RecurrentStateCache()
    slot = cache.alloc()
    cache.ensure_capacity([slot], [len(prompt)])
    lin_cache.clear_slot(slot)  # fresh recurrent state for this slot (EngineRunner order)
    lin_cache.bind([slot])
    ids = torch.tensor([prompt], device="cuda")
    pos = torch.arange(len(prompt), device="cuda").unsqueeze(0)
    ctx = ForwardContext(
        is_prefill=True, kv_cache=cache, lin_cache=lin_cache, slots=[slot]
    )
    model(ids, pos, ctx)

    cache.ensure_capacity([slot], [len(prompt) + 1])
    lin_cache.bind([slot])
    ctx = ForwardContext(
        is_prefill=False, kv_cache=cache, lin_cache=lin_cache,
        slots=[slot], slot_lengths=[len(prompt)],
    )
    hidden = model(torch.tensor([[next_tok]], device="cuda"),
                   torch.tensor([[len(prompt)]], device="cuda"), ctx)
    eng_decode = model.compute_logits(hidden[:, -1])

    assert _cos(ref_decode, eng_decode) > 0.995


@pytest.mark.skipif(not CUDA, reason="forward needs CUDA")
def test_qwen3_5_offline_superl8_roundtrip(tmp_path):
    """Convert path: quantize -> .superl8 -> load -> build must byte-match a model built
    directly from the same quantized dict (proves the dense-hybrid .superl8 round-trip)."""
    from superl8 import QTensor, save_superl8

    from superl8serve.convert import quantize_state_dict
    from superl8serve.loader import load_superl8_state_dict

    cfg = _cfg()
    qsd = quantize_state_dict(_sd(cfg, "cpu"), weight_bits=8)
    path = str(tmp_path / "q35.superl8")
    save_superl8(path, qsd)

    def _to_cuda(d):
        out = {}
        for k, v in d.items():
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

    direct = {k: (v.data if v.scheme == "raw" else v) for k, v in qsd.items()}
    m_direct = build_model(cfg, _to_cuda(direct)).cuda().eval()
    m_off = build_model(cfg, load_superl8_state_dict(path, device="cuda")).cuda().eval()

    prompt = torch.randint(0, cfg.vocab_size, (1, 5), device="cuda")
    r_d = ModelRunner(m_direct, cfg, max_batch=1, max_len=32, device="cuda")
    r_o = ModelRunner(m_off, cfg, max_batch=1, max_len=32, device="cuda")
    torch.testing.assert_close(r_o.prefill(prompt), r_d.prefill(prompt), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# HF-equivalence guards (transformers >= Qwen3.5). These are the tests that would
# have caught the real-checkpoint bring-up bugs the earlier self-consistency tests
# missed (they matched OUR own reference, not HF): the DeltaNet q-scale, the per-head
# gated output norm, and the zero-centered (Gemma-style) Qwen3_5RMSNorm. Pure DeltaNet
# + norm, so no attention kernel is needed.
# ---------------------------------------------------------------------------

def _hf_qwen35_cfg():
    cfg_mod = pytest.importorskip("transformers.models.qwen3_5.configuration_qwen3_5")
    return cfg_mod.Qwen3_5Config(
        vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=1,
        num_attention_heads=8, num_key_value_heads=8, head_dim=64, hidden_act="silu",
        linear_num_key_heads=4, linear_num_value_heads=4,
        linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4,
        full_attention_interval=999, layer_types=["linear_attention"],
        partial_rotary_factor=0.25, rope_theta=1e6, rms_norm_eps=1e-6,
        max_position_embeddings=64, pad_token_id=0, tie_word_embeddings=True,
    )


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_qwen3_5_rmsnorm_is_zero_centered():
    """Qwen3_5RMSNorm is Gemma-style zero-centered (gain = 1 + weight, weight init 0);
    q_norm/k_norm and all decoder norms use it. Our RMSNorm(add_unit_offset=True) must
    reproduce it — the plain-weight variant (the bring-up bug) must NOT."""
    import torch.nn.functional as F
    hf_mod = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from superl8serve.layers.norm import RMSNorm
    torch.manual_seed(0)
    x = torch.randn(2, 5, 64, device="cuda", dtype=torch.float16)
    w = (torch.randn(64, device="cuda") * 0.3)
    hf = hf_mod.Qwen3_5RMSNorm(64).cuda()
    with torch.no_grad():
        hf.weight.copy_(w)
    ref = hf(x.float())
    ours = RMSNorm(64, 1e-6, w.half(), add_unit_offset=True).cuda()(x).float()
    bug = RMSNorm(64, 1e-6, w.half(), add_unit_offset=False).cuda()(x).float()
    assert F.cosine_similarity(ref.reshape(-1), ours.reshape(-1), dim=0) > 0.999
    assert F.cosine_similarity(ref.reshape(-1), bug.reshape(-1), dim=0) < 0.99


def test_tiled_gguf_delta_head_broadcast_matches_llama_layout():
    """GGUF stores V heads tiled, so Q/K broadcast must tile instead of group."""
    from superl8serve.layers.linear_attn import _expand_delta_k_heads

    x = torch.tensor([[[[0.0]], [[1.0]]]])
    grouped = _expand_delta_k_heads(x, 4, tiled=False)
    tiled = _expand_delta_k_heads(x, 4, tiled=True)
    assert grouped.flatten().tolist() == [0.0, 0.0, 1.0, 1.0]
    assert tiled.flatten().tolist() == [0.0, 1.0, 0.0, 1.0]


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_qwen3_5_gated_deltanet_matches_hf():
    """Our GatedDeltaNetAttention vs HF Qwen3_5GatedDeltaNet on shared fp weights: guards
    the q-scale (1/sqrt head_k_dim) readout and the PER-HEAD gated output norm. Either
    fix regressing drops the cosine well below the bar."""
    import torch.nn.functional as F
    hf_mod = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from superl8 import QTensor

    from superl8serve.layers.linear_attn import GatedDeltaNetAttention
    c = _hf_qwen35_cfg()
    torch.manual_seed(0)
    hfdn = hf_mod.Qwen3_5GatedDeltaNet(c, 0).cuda().float().eval()
    W = {n: p.detach() for n, p in hfdn.named_parameters()}
    nk, nv = c.linear_num_key_heads, c.linear_num_value_heads
    kd, vd = c.linear_key_head_dim, c.linear_value_head_dim

    def qt(w):
        w = w.cuda().half()
        s = w.abs().amax(-1, keepdim=True).clamp_min(1e-8).float() / 127
        return QTensor(torch.round(w / s).clamp_(-127, 127).to(torch.int8),
                       s.squeeze(-1), scheme="per_row_i8")

    ocfg = _MC.from_hf(c.to_dict(), arch="qwen3_5")
    blk = GatedDeltaNetAttention(
        ocfg, qkv_proj=qt(W["in_proj_qkv.weight"]), out_proj=qt(W["out_proj.weight"]),
        conv_weight=W["conv1d.weight"].squeeze(1).cuda().half(),
        a_log=W["A_log"].cuda().float(), dt_bias=W["dt_bias"].cuda().float(),
        beta_proj=qt(W["in_proj_b.weight"]), gate_proj=qt(W["in_proj_a.weight"]),
        z_proj=qt(W["in_proj_z.weight"]), norm_gain=W["norm.weight"].cuda().half(),
        num_k_heads=nk, num_v_heads=nv, key_dim=kd, value_dim=vd,
        conv_kernel=c.linear_conv_kernel_dim,
    ).cuda()
    x = torch.randn(1, 6, c.hidden_size, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        ref = hfdn(x.float())
        ref = ref[0] if isinstance(ref, tuple) else ref
        ours = blk(x, None, None, 0).float()
    cos = F.cosine_similarity(ref.reshape(-1), ours.reshape(-1), dim=0).item()
    assert cos > 0.98, f"DeltaNet diverges from HF: cos={cos}"
