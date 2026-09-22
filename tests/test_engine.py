# SPDX-License-Identifier: MIT
"""Engine tests: continuous batching over concurrent requests of DIFFERENT prompt
lengths through a small random Qwen3, plus determinism vs the single-sequence
runner. Proves the scheduler + slot cache + ragged decode produce correct, stable
output. Needs CUDA."""

import pytest
import torch

pytest.importorskip("superl8")

from superl8serve.engine import LLMEngine, SamplingParams
from superl8serve.models import ModelConfig, ModelRunner, build_model

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="engine needs the CUDA superl8 kernels")


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
        tie_word_embeddings=True,
    )


def _sd(cfg):
    def r(*s):
        return torch.randn(*s, device="cuda", dtype=torch.float16) * 0.05

    hd, nh, nkv, H = (
        cfg.resolved_head_dim(),
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.hidden_size,
    )
    sd = {"model.embed_tokens.weight": r(cfg.vocab_size, H), "model.norm.weight": r(H)}
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


def test_engine_generates_for_concurrent_requests():
    torch.manual_seed(0)
    cfg = _cfg()
    eng = LLMEngine(cfg, _sd(cfg), device="cuda", max_num_seqs=8, max_len=64)
    prompts = [[1, 2, 3], [4, 5, 6, 7, 8], [9], [10, 11, 12, 13]]  # different lengths
    outs = eng.generate(prompts, SamplingParams(temperature=0.0, max_tokens=6))
    assert len(outs) == 4
    for o in outs:
        assert len(o) == 6 and all(0 <= t < cfg.vocab_size for t in o)


def test_engine_matches_single_sequence_runner():
    """Greedy engine output for one request must closely track the standalone
    runner. Not bit-exact any more: the engine's decode KV cache is now paged
    int8 (quantize-on-write), while the standalone runner stays fp16
    contiguous, so a token can occasionally flip on a near-tied logit -- see
    `tests/test_paged_engine.py` for the direct cosine-similarity check of the
    two paths' logits."""
    torch.manual_seed(1)
    cfg = _cfg()
    sd = _sd(cfg)
    prompt = [3, 1, 4, 1, 5]

    eng = LLMEngine(cfg, sd, device="cuda", max_num_seqs=4, max_len=64)
    eng_out = eng.generate([prompt], SamplingParams(temperature=0.0, max_tokens=5))[0]

    model = build_model(cfg, sd).cuda().eval()
    runner = ModelRunner(model, cfg, max_batch=1, max_len=64, device="cuda")
    ref = runner.generate_greedy(torch.tensor([prompt], device="cuda"), max_new_tokens=5)[
        0
    ].tolist()

    assert eng_out[0] == ref[0]
    agree = sum(1 for a, b in zip(eng_out, ref) if a == b) / len(ref)
    assert agree >= 0.8, f"paged vs contiguous greedy diverged too much: {eng_out} vs {ref}"


def test_engine_respects_eos():
    torch.manual_seed(2)
    cfg = _cfg()
    eng = LLMEngine(cfg, _sd(cfg), device="cuda", max_num_seqs=4, max_len=64, eos_id=None)
    out = eng.generate([[1, 2, 3]], SamplingParams(temperature=0.0, max_tokens=4))[0]
    assert len(out) == 4


def test_engine_applies_logit_processors():
    """A per-request logit processor can force a specific token every step, proving
    the hook (issue #38) threads from SamplingParams through the engine's sampler."""
    torch.manual_seed(3)
    cfg = _cfg()
    eng = LLMEngine(cfg, _sd(cfg), device="cuda", max_num_seqs=4, max_len=64)

    def force_token_5(input_ids, logits):
        logits = logits.clone()
        logits[:] = float("-inf")
        logits[5] = 0.0
        return logits

    params = SamplingParams(temperature=0.0, max_tokens=4, logit_processors=[force_token_5])
    out = eng.generate([[1, 2, 3]], params)[0]
    assert out == [5, 5, 5, 5]


def test_varlen_prefill_matches_individual():
    """Multiple sequences of DIFFERENT lengths packed into one varlen forward pass
    must produce identical first-token output (per sequence) as individual
    single-sequence prefills. Proves the packed attention kernel, cumulative
    sequence lengths, and paged KV slot_mapping all thread correctly."""
    torch.manual_seed(42)
    cfg = _cfg()
    sd = _sd(cfg)
    prompts = [
        [1, 2, 3, 4],
        [5, 6],
        [7, 8, 9, 10, 11, 12, 13],
    ]
    params = SamplingParams(temperature=0.0, max_tokens=1)
    kv_cfg = dict(device="cuda", max_num_seqs=8, max_len=64, max_batch_tokens=8192)

    eng = LLMEngine(cfg, sd, **kv_cfg)
    outs_varlen = eng.generate(prompts, params)

    outs_ref: list[list[int]] = []
    for p in prompts:
        eng2 = LLMEngine(cfg, sd, **kv_cfg)
        outs_ref.append(eng2.generate([p], params)[0])

    for i, (v, ref) in enumerate(zip(outs_varlen, outs_ref)):
        assert v == ref, f"seq {i} (len={len(prompts[i])}): varlen {v} != individual {ref}"


def test_varlen_prefill_same_length():
    """Same-length sequences also hit the varlen path when batch > 1 and should
    still match individual prefills."""
    torch.manual_seed(42)
    cfg = _cfg()
    sd = _sd(cfg)
    prompts = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    params = SamplingParams(temperature=0.0, max_tokens=1)
    kv_cfg = dict(device="cuda", max_num_seqs=8, max_len=64, max_batch_tokens=8192)

    eng = LLMEngine(cfg, sd, **kv_cfg)
    outs_varlen = eng.generate(prompts, params)

    outs_ref: list[list[int]] = []
    for p in prompts:
        eng2 = LLMEngine(cfg, sd, **kv_cfg)
        outs_ref.append(eng2.generate([p], params)[0])

    for i, (v, ref) in enumerate(zip(outs_varlen, outs_ref)):
        assert v == ref, f"seq {i}: varlen {v} != individual {ref}"


def test_varlen_prefill_and_decode():
    """Varlen prefill followed by decode must produce the same full multi-token
    output as individual prefills + decodes."""
    torch.manual_seed(42)
    cfg = _cfg()
    sd = _sd(cfg)
    prompts = [
        [1, 2, 3, 4],
        [5, 6, 7],
        [8, 9, 10, 11, 12],
    ]
    params = SamplingParams(temperature=0.0, max_tokens=4)
    kv_cfg = dict(device="cuda", max_num_seqs=8, max_len=64, max_batch_tokens=8192)

    eng = LLMEngine(cfg, sd, **kv_cfg)
    outs_varlen = eng.generate(prompts, params)

    outs_ref: list[list[int]] = []
    for p in prompts:
        eng2 = LLMEngine(cfg, sd, **kv_cfg)
        outs_ref.append(eng2.generate([p], params)[0])

    for i, (v, ref) in enumerate(zip(outs_varlen, outs_ref)):
        assert v == ref, f"seq {i} (prompt len={len(prompts[i])}): varlen {v} != individual {ref}"


def test_preempt_and_resume():
    """When load exceeds max_num_seqs, excess requests WAIT in FIFO and are admitted
    as running sequences finish (no count-cap preemption / recompute thrash); every
    request still completes with the correct number of output tokens."""
    torch.manual_seed(0)
    cfg = _cfg()
    eng = LLMEngine(cfg, _sd(cfg), device="cuda", max_num_seqs=2, max_len=64)
    # 5 prompts with max_num_seqs=2 -> slot pressure; the extra 3 queue and drain
    prompts = [[1, 2, 3], [4, 5, 6, 7, 8], [9], [10, 11], [12, 13, 14]]
    outs = eng.generate(prompts, SamplingParams(temperature=0.0, max_tokens=8))
    assert len(outs) == 5
    for o in outs:
        assert len(o) == 8 and all(0 <= t < cfg.vocab_size for t in o)


def _qwen3_next_hybrid_cfg_sd():
    x = dict(
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
    )
    cfg = ModelConfig(
        arch="qwen3_next",
        vocab_size=64,
        hidden_size=128,
        num_hidden_layers=2,
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
        full_attention_interval=2,
        shared_expert_intermediate_size=64,
        extra=x,
    )
    H, nh, nkv, hd = 128, 4, 2, 32
    nk, nv, kd, vd = 2, 4, 16, 16
    qkv_lin = 2 * nk * kd + nv * vd

    def r(*s):
        return torch.randn(*s, device="cuda", dtype=torch.float16) * 0.05

    sd = {"model.embed_tokens.weight": r(cfg.vocab_size, H), "model.norm.weight": r(H)}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = r(H)
        sd[f"{p}.post_attention_layernorm.weight"] = r(H)
        if (i + 1) % 2 == 0:  # full attn
            sd[f"{p}.self_attn.q_proj.weight"] = r(nh * hd, H)
            sd[f"{p}.self_attn.k_proj.weight"] = r(nkv * hd, H)
            sd[f"{p}.self_attn.v_proj.weight"] = r(nkv * hd, H)
            sd[f"{p}.self_attn.o_proj.weight"] = r(H, nh * hd)
            sd[f"{p}.self_attn.q_norm.weight"] = r(hd)
            sd[f"{p}.self_attn.k_norm.weight"] = r(hd)
        else:  # linear (DeltaNet)
            la = f"{p}.linear_attn"
            sd[f"{la}.qkv_proj.weight"] = r(qkv_lin, H)
            sd[f"{la}.out_proj.weight"] = r(H, nv * vd)
            sd[f"{la}.conv_weight"] = r(qkv_lin, 4)
            sd[f"{la}.A_log"] = r(nv).float()
            sd[f"{la}.dt_bias"] = r(nv).float()
            sd[f"{la}.beta_proj.weight"] = r(nv, H)
            sd[f"{la}.dt_proj.weight"] = r(nv, H)
            sd[f"{la}.z_proj.weight"] = r(nv * vd, H)
            sd[f"{la}.norm.weight"] = r(vd)  # gated DeltaNet norm is PER-HEAD (head_v_dim)
        for w, d in (("gate", cfg.num_experts),):
            sd[f"{p}.mlp.gate.weight"] = r(d, H)
        for e in range(cfg.num_experts):
            sd[f"{p}.mlp.experts.{e}.gate_proj.weight"] = r(cfg.moe_intermediate_size, H)
            sd[f"{p}.mlp.experts.{e}.up_proj.weight"] = r(cfg.moe_intermediate_size, H)
            sd[f"{p}.mlp.experts.{e}.down_proj.weight"] = r(H, cfg.moe_intermediate_size)
        sd[f"{p}.mlp.shared_expert.gate_proj.weight"] = r(cfg.moe_intermediate_size, H)
        sd[f"{p}.mlp.shared_expert.up_proj.weight"] = r(cfg.moe_intermediate_size, H)
        sd[f"{p}.mlp.shared_expert.down_proj.weight"] = r(H, cfg.moe_intermediate_size)
    return cfg, sd


def test_engine_qwen3_next_hybrid_matches_runner():
    """A hybrid (DeltaNet + full-attn) model decoded through the engine with
    recurrent-state carry must match the standalone ModelRunner output at the same
    token-agreement threshold as the pure-attention path."""
    torch.manual_seed(42)
    cfg, sd = _qwen3_next_hybrid_cfg_sd()
    prompt = [3, 1, 4, 1, 5]

    eng = LLMEngine(cfg, sd, device="cuda", max_num_seqs=4, max_len=64)
    eng_out = eng.generate([prompt], SamplingParams(temperature=0.0, max_tokens=5))[0]

    model = build_model(cfg, sd).cuda().eval()
    runner = ModelRunner(model, cfg, max_batch=1, max_len=64, device="cuda")
    ref = runner.generate_greedy(torch.tensor([prompt], device="cuda"), max_new_tokens=5)[
        0
    ].tolist()

    assert eng_out[0] == ref[0], f"first token mismatch: {eng_out} vs {ref}"
    agree = sum(1 for a, b in zip(eng_out, ref) if a == b) / len(ref)
    assert agree >= 0.8, f"engine vs runner hybrid decode diverged too much: {eng_out} vs {ref}"


def test_engine_qwen3_next_concurrent_recurrent_decode():
    """Continuous batching for a recurrent (hybrid DeltaNet) family: two sequences
    decoded CONCURRENTLY through one engine must produce exactly what each produces
    when run alone. Before per-slot recurrent state this crashed outright — the
    conv-tail cached from the last prefilled sequence (batch 1) was `torch.cat`-ed
    against a batch-2 decode input — and, absent the crash, the two sequences shared
    one global recurrent state and corrupted each other. This is the property that
    makes divergent families actually SERVE (not just single-request generate)."""
    torch.manual_seed(42)
    cfg, sd = _qwen3_next_hybrid_cfg_sd()
    pA = [3, 1, 4, 1, 5, 9, 2, 6]
    pB = [7, 2, 7, 1, 8, 2, 8]  # different length on purpose (ragged batch)
    params = SamplingParams(temperature=0.0, max_tokens=6)

    eng = LLMEngine(cfg, sd, device="cuda", max_num_seqs=4, max_len=64)
    both = eng.generate([pA, pB], params)

    solo_a = LLMEngine(cfg, sd, device="cuda", max_num_seqs=4, max_len=64).generate([pA], params)[0]
    solo_b = LLMEngine(cfg, sd, device="cuda", max_num_seqs=4, max_len=64).generate([pB], params)[0]

    assert both[0] == solo_a, f"seq A corrupted by concurrent decode: {both[0]} vs {solo_a}"
    assert both[1] == solo_b, f"seq B corrupted by concurrent decode: {both[1]} vs {solo_b}"


# ── MTP speculative-decode tests ──────────────────────────────────────────


def _cfg_mtp():
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
        tie_word_embeddings=True,
        num_mtp_layers=1,  # single MTP depth
    )


def _sd_mtp(cfg):
    def r(*s):
        # Larger weight scale than the other engine tests on purpose: it makes the
        # tiny random model's greedy decode genuinely CONTEXT-sensitive (a varied,
        # non-cyclic token stream) instead of collapsing to a 2-token limit cycle.
        # That sensitivity is what lets the spec-decode bit-identical test below
        # actually catch a corrupted / missing accepted-token KV commit — with a
        # degenerate cyclic model, a KV hole leaves the output unchanged.
        return torch.randn(*s, device="cuda", dtype=torch.float16) * 0.5

    hd, nh, nkv, H = (
        cfg.resolved_head_dim(),
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.hidden_size,
    )
    sd = {"model.embed_tokens.weight": r(cfg.vocab_size, H), "model.norm.weight": r(H)}
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
    # MTP depth 1
    mp = "model.mtp.1"
    sd[f"{mp}.fc.weight"] = r(H, 2 * H)
    sd[f"{mp}.pre_fc_norm_hidden.weight"] = r(H)
    sd[f"{mp}.pre_fc_norm_embedding.weight"] = r(H)
    sd[f"{mp}.self_attn.q_proj.weight"] = r(nh * hd, H)
    sd[f"{mp}.self_attn.k_proj.weight"] = r(nkv * hd, H)
    sd[f"{mp}.self_attn.v_proj.weight"] = r(nkv * hd, H)
    sd[f"{mp}.self_attn.o_proj.weight"] = r(H, nh * hd)
    sd[f"{mp}.self_attn.q_norm.weight"] = r(hd)
    sd[f"{mp}.self_attn.k_norm.weight"] = r(hd)
    sd[f"{mp}.input_layernorm.weight"] = r(H)
    sd[f"{mp}.post_attention_layernorm.weight"] = r(H)
    sd[f"{mp}.mlp.gate_proj.weight"] = r(cfg.intermediate_size, H)
    sd[f"{mp}.mlp.up_proj.weight"] = r(cfg.intermediate_size, H)
    sd[f"{mp}.mlp.down_proj.weight"] = r(H, cfg.intermediate_size)
    return sd


def test_mtp_spec_decode_runs():
    """MTP spec-decode completes without error (single seq, greedy)."""
    torch.manual_seed(0)
    cfg = _cfg_mtp()
    sd = _sd_mtp(cfg)
    eng = LLMEngine(cfg, sd, device="cuda", max_num_seqs=2, max_len=64)
    prompt = [1, 2, 3, 4]
    out = eng.generate([prompt], SamplingParams(temperature=0.0, max_tokens=8))
    assert len(out) == 1
    assert len(out[0]) == 8
    assert all(0 <= t < cfg.vocab_size for t in out[0])


def _no_mtp_cfg(cfg):
    """The same ModelConfig with the MTP head stripped — the non-spec greedy
    reference engine (plain autoregressive decode, one token per step)."""
    return ModelConfig(
        **{k: v for k, v in vars(cfg).items() if k not in ("num_mtp_layers", "extra")},
        extra=cfg.extra,
    )


def _force_full_acceptance(eng_spec, ref_by_prompt):
    """Wire a spec-decode engine so its MTP draft head ALWAYS proposes the true
    greedy continuation, forcing every draft to be accepted (n_acc >= 2).

    Random-init MTP heads have ~0 draft acceptance, so the accept/commit path
    (the part with the corrupting bugs) would never run — every step would fall
    back to n_acc == 1, which is trivially equal to non-spec greedy and hides
    both the double-o_proj verify bug and the missing accepted-token KV commit.
    We instead monkeypatch ``draft_greedy`` to emit, per sequence, the very token
    the target model will greedily verify next (looked up from a precomputed
    non-spec reference), so acceptance is guaranteed and the real commit path is
    exercised. Returns a list that accumulates the per-step n_acc actually taken.

    ``ref_by_prompt`` maps ``tuple(prompt_ids) -> (prompt_len, reference_output)``.
    """
    holder: dict = {}
    naccs: list[int] = []

    runner = eng_spec.runner
    runner._spec_enabled = True  # spec-decode is opt-in/off by default; force it on here

    def forced_compute_drafts(mtp, base_hidden, base_tok, lengths, slots, batch, views=None):
        # Force each row's drafts to the true greedy continuation (up to _spec_k
        # tokens) so every draft is accepted and the real verify/commit + multi-token
        # accept path is exercised. Patches the single draft seam ``_compute_drafts``
        # (new signature: per-row draft LISTS) so it forces the restructured loop too.
        out = []
        for b in range(len(batch)):
            P, ref = ref_by_prompt[tuple(batch[b].prompt_ids)]
            # length L = lengths[b]; base_tok is token@(L+1) = ref[L+1-P]; the drafts
            # are the continuation token@(L+2+t) = ref[L+2-P+t].
            start = int(lengths[b]) + 2 - P
            dl = [int(ref[start + t]) for t in range(runner._spec_k) if 0 <= start + t < len(ref)]
            out.append(dl if dl else [0])
        return out

    runner._compute_drafts = forced_compute_drafts

    orig = runner._spec_decode_eager

    def wrapped(batch, mtp):
        holder["batch"] = batch  # so forced_draft can map row -> sequence
        before = [len(s.output_ids) for s in batch]
        result = orig(batch, mtp)
        naccs.extend(len(s.output_ids) - b for s, b in zip(batch, before))
        return result

    runner._spec_decode_eager = wrapped
    return naccs


def test_mtp_spec_decode_bit_identical_greedy():
    """Greedy spec-decode MUST be bit-identical to plain greedy decode.

    This is the load-bearing correctness contract: with temperature 0 the
    accept-longest-greedy-prefix rule can only ever emit tokens the target model
    would have emitted anyway, so the token stream must match non-spec greedy
    EXACTLY (not "mostly" — the old >= 0.8 agreement bar passed even with both
    silent-corruption bugs present). We force full draft acceptance so several
    tokens are committed per step, which:
      * exercises the accepted-token KV commit — if the intermediate accepted
        tokens' K/V are not written to the paged cache, the next step reads a
        hole and the output diverges (caught by the bit-identical assert);
      * exercises the verify o_proj path — if ``_verify_batched``'s output is
        o_proj'd twice, the verify logits are garbage, no forced draft is ever
        accepted, and n_acc collapses to 1 (caught by the n_acc >= 2 assert).
    """
    torch.manual_seed(0)
    cfg = _cfg_mtp()
    sd = _sd_mtp(cfg)
    prompt = [3, 1, 4, 1, 5, 9, 2, 6]
    params = SamplingParams(temperature=0.0, max_tokens=24)

    # Non-spec greedy reference.
    eng_ref = LLMEngine(
        _no_mtp_cfg(cfg), sd, device="cuda", max_num_seqs=2, max_len=64, enable_cuda_graph=False
    )
    out_ref = eng_ref.generate([prompt], params)[0]

    # Spec-decode engine with forced full acceptance.
    eng_spec = LLMEngine(
        cfg, sd, device="cuda", max_num_seqs=2, max_len=64, enable_cuda_graph=False
    )
    naccs = _force_full_acceptance(eng_spec, {tuple(prompt): (len(prompt), out_ref)})
    out_spec = eng_spec.generate([prompt], params)[0]

    assert max(naccs) >= 2, (
        f"forced draft was never accepted (n_acc stayed 1: {naccs}); verify logits "
        f"are wrong — check for double o_proj in _verify_batched"
    )
    assert out_spec == out_ref, (
        f"spec-decode NOT bit-identical to non-spec greedy:\n  spec={out_spec}\n   ref={out_ref}\n"
        f"  (accepted-token KV likely not committed to the paged cache)"
    )


def test_mtp_spec_decode_bit_identical_greedy_concurrent():
    """Same bit-identical contract as above, but with THREE sequences of
    different lengths decoded concurrently through one engine — the accepted-token
    KV commit and verify o_proj must be correct per row in a ragged batch, not
    just for a single sequence."""
    torch.manual_seed(0)
    cfg = _cfg_mtp()
    sd = _sd_mtp(cfg)
    prompts = [[3, 1, 4, 1, 5, 9, 2, 6], [7, 2, 7, 1, 8], [1, 6, 1, 8, 0, 3]]
    params = SamplingParams(temperature=0.0, max_tokens=20)

    eng_ref = LLMEngine(
        _no_mtp_cfg(cfg), sd, device="cuda", max_num_seqs=4, max_len=64, enable_cuda_graph=False
    )
    refs = eng_ref.generate(prompts, params)

    eng_spec = LLMEngine(
        cfg, sd, device="cuda", max_num_seqs=4, max_len=64, enable_cuda_graph=False
    )
    ref_by_prompt = {tuple(p): (len(p), r) for p, r in zip(prompts, refs)}
    naccs = _force_full_acceptance(eng_spec, ref_by_prompt)
    outs = eng_spec.generate(prompts, params)

    assert max(naccs) >= 2, f"forced draft never accepted in the concurrent batch: {naccs}"
    for i, (spec, ref) in enumerate(zip(outs, refs)):
        assert spec == ref, (
            f"seq {i}: spec-decode NOT bit-identical to non-spec greedy:\n"
            f"  spec={spec}\n   ref={ref}"
        )


# ── Engine-side cancellation by seq id (superl8-serve#363) ────────────────────


def test_engine_cancel_waiting_request():
    """Canceling a request that has not yet been scheduled (still WAITING) removes
    it with no KV churn: the engine reports True once, then False on the repeat."""
    torch.manual_seed(0)
    cfg = _cfg()
    eng = LLMEngine(
        cfg, _sd(cfg), device="cuda", max_num_seqs=4, max_len=64, enable_cuda_graph=False
    )
    sid = eng.add_request([1, 2, 3])
    assert eng.scheduler.has_work()
    assert eng.cancel(sid) is True
    assert eng.scheduler.has_work() is False
    assert eng.cancel(sid) is False  # idempotent
    assert eng.cancel(999) is False  # unknown id
    with pytest.raises(KeyError):
        eng.sequence(sid)  # forgotten from _out


def test_engine_cancel_running_frees_kv_lin_mtp():
    """Canceling a RUNNING request frees its paged KV slot AND scrubs the runner's
    per-slot decode state (recurrent `lin_cache` row + MTP `mtp_cache` slot) so a
    later request reusing the slot never inherits stale state."""
    torch.manual_seed(0)
    cfg = _cfg()
    eng = LLMEngine(
        cfg, _sd(cfg), device="cuda", max_num_seqs=4, max_len=64, enable_cuda_graph=False
    )
    sid = eng.add_request([1, 2, 3, 4])
    eng.step()  # prefill -> RUNNING, holds a KV slot
    seq = eng.sequence(sid)
    slot = seq.slot
    assert slot >= 0
    # Seed stale per-slot decode state that cancel MUST scrub.
    eng.lin_cache.bind([slot])
    eng.lin_cache.set_state(0, torch.zeros(1, 4, device="cuda"))
    eng.runner.mtp_cache.reset_slot(
        slot,
        torch.zeros(2, 1, 8, device="cuda"),
        torch.zeros(2, 1, 8, device="cuda"),
    )

    assert eng.cancel(sid) is True
    assert eng.scheduler.has_work() is False
    assert eng.cache.has_free_slot()  # paged KV slot recycled
    assert slot in eng.cache._free_slots
    assert (slot, 0) not in eng.lin_cache._state  # lin_cache row cleared
    assert slot not in eng.runner.mtp_cache._k  # mtp_cache slot cleared
    with pytest.raises(KeyError):
        eng.sequence(sid)


def test_engine_cancel_mid_generation_other_completes():
    """Canceling one of two concurrently-running requests must leave the other's
    normal completion unchanged (correct output token count)."""
    torch.manual_seed(0)
    cfg = _cfg()
    eng = LLMEngine(
        cfg, _sd(cfg), device="cuda", max_num_seqs=4, max_len=64, enable_cuda_graph=False
    )
    params = SamplingParams(temperature=0.0, max_tokens=6)
    a = eng.add_request([1, 2, 3], params)
    b = eng.add_request([4, 5, 6, 7, 8], params)
    eng.step()  # prefill both -> both RUNNING
    assert eng.cancel(a) is True
    assert len(eng.scheduler.running) == 1  # only b still runs
    guard = 0
    while eng.scheduler.has_work():
        guard += 1
        assert guard < 1000, "engine did not drain after cancel"
        eng.step()
    out = eng.sequence(b).output_ids
    assert len(out) == 6  # normal completion unchanged
    eng.forget(b)


# the DeltaNet recurrent state by every drafted token, but only the accepted prefix
# may commit. The runner snapshots the post-base recurrent state, runs verify from
# it, restores the snapshot, then replays only the accepted tokens (through the
# normal decode path) to reach the correct committed state — the recurrent analogue
# of the paged-KV accepted-token canonicalizer. These tests lock BOTH the gated-attn
# verify wiring (GatedGQAAttention.is_verify branch) and the recurrent rollback.


def _cfg_mtp_hybrid():
    """A Qwen3.5-shaped hybrid: gated full attention (attn_output_gate) + Gated
    DeltaNet linear layers + a dense SwiGLU MLP + a single-depth MTP head. head_dim
    256 mirrors the real 0.8B AND routes the gated verify through the fp16-PV
    `attn_int8_fwd` fallback (the int8 verify kernel is head-dim {32,64,128} only)."""
    x = dict(
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
    )
    return ModelConfig(
        arch="qwen3_5",
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=256,
        max_position_embeddings=256,
        head_dim=256,
        qk_norm=True,
        tie_word_embeddings=True,
        num_mtp_layers=1,
        linear_attention=True,
        full_attention_interval=2,
        extra=x,
    )


def _sd_mtp_hybrid(cfg):
    def r(*s):
        return torch.randn(*s, device="cuda", dtype=torch.float16) * 0.5

    H = cfg.hidden_size
    hd, nh, nkv = cfg.resolved_head_dim(), cfg.num_attention_heads, cfg.num_key_value_heads
    x = cfg.extra
    nk, nv = x["linear_num_key_heads"], x["linear_num_value_heads"]
    kd, vd, ck = x["linear_key_head_dim"], x["linear_value_head_dim"], x["linear_conv_kernel_dim"]
    qkv_lin = 2 * nk * kd + nv * vd

    def gated_full(p):
        # gated full-attn layer: fused query|gate q_proj is [2*nh*hd, H].
        return {
            f"{p}.self_attn.q_proj.weight": r(2 * nh * hd, H),
            f"{p}.self_attn.k_proj.weight": r(nkv * hd, H),
            f"{p}.self_attn.v_proj.weight": r(nkv * hd, H),
            f"{p}.self_attn.o_proj.weight": r(H, nh * hd),
            f"{p}.self_attn.q_norm.weight": r(hd),
            f"{p}.self_attn.k_norm.weight": r(hd),
            f"{p}.input_layernorm.weight": r(H),
            f"{p}.post_attention_layernorm.weight": r(H),
            f"{p}.mlp.gate_proj.weight": r(cfg.intermediate_size, H),
            f"{p}.mlp.up_proj.weight": r(cfg.intermediate_size, H),
            f"{p}.mlp.down_proj.weight": r(H, cfg.intermediate_size),
        }

    sd = {"model.embed_tokens.weight": r(cfg.vocab_size, H), "model.norm.weight": r(H)}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        if cfg.attention_kind(i) == "full":
            sd.update(gated_full(p))
        else:  # linear (DeltaNet)
            la = f"{p}.linear_attn"
            sd[f"{la}.qkv_proj.weight"] = r(qkv_lin, H)
            sd[f"{la}.out_proj.weight"] = r(H, nv * vd)
            sd[f"{la}.conv_weight"] = r(qkv_lin, ck)
            sd[f"{la}.A_log"] = r(nv).float()
            sd[f"{la}.dt_bias"] = r(nv).float()
            sd[f"{la}.beta_proj.weight"] = r(nv, H)
            sd[f"{la}.dt_proj.weight"] = r(nv, H)
            sd[f"{la}.z_proj.weight"] = r(nv * vd, H)
            sd[f"{la}.norm.weight"] = r(vd)
            sd[f"{p}.input_layernorm.weight"] = r(H)
            sd[f"{p}.post_attention_layernorm.weight"] = r(H)
            sd[f"{p}.mlp.gate_proj.weight"] = r(cfg.intermediate_size, H)
            sd[f"{p}.mlp.up_proj.weight"] = r(cfg.intermediate_size, H)
            sd[f"{p}.mlp.down_proj.weight"] = r(H, cfg.intermediate_size)
    # Single-depth MTP head (root `mtp.*`, the shipped 0.8B layout): shared fc +
    # two pre-fc norms + a gated-full-attn decoder block + its own final norm.
    sd["mtp.fc.weight"] = r(H, 2 * H)
    sd["mtp.pre_fc_norm_hidden.weight"] = r(H)
    sd["mtp.pre_fc_norm_embedding.weight"] = r(H)
    sd["mtp.norm.weight"] = r(H)
    sd.update(gated_full("mtp.layers.0"))
    return sd


def _drop_mtp(sd):
    """Non-spec reference sd: strip the MTP head so the engine can't build one
    (the qwen3_5 builder recovers depth from the WEIGHTS, so cfg alone can't
    disable it)."""
    return {k: v for k, v in sd.items() if not k.startswith("mtp.")}


def test_mtp_spec_decode_hybrid_bit_identical_greedy():
    """The load-bearing gate for the DeltaNet hybrid: forced-full-acceptance greedy
    spec-decode MUST be bit-identical to plain greedy. Exercises the gated-attn
    verify path AND the recurrent-state snapshot/restore/replay across several
    committed tokens per step (n_acc >= 2)."""
    torch.manual_seed(0)
    cfg = _cfg_mtp_hybrid()
    sd = _sd_mtp_hybrid(cfg)
    prompt = [3, 1, 4, 1, 5, 9, 2, 6]
    params = SamplingParams(temperature=0.0, max_tokens=20)

    eng_ref = LLMEngine(
        cfg, _drop_mtp(sd), device="cuda", max_num_seqs=2, max_len=64, enable_cuda_graph=False
    )
    assert eng_ref.model.mtp is None
    out_ref = eng_ref.generate([prompt], params)[0]

    eng_spec = LLMEngine(
        cfg, sd, device="cuda", max_num_seqs=2, max_len=64, enable_cuda_graph=False
    )
    assert eng_spec.model.mtp is not None
    naccs = _force_full_acceptance(eng_spec, {tuple(prompt): (len(prompt), out_ref)})
    out_spec = eng_spec.generate([prompt], params)[0]

    assert max(naccs) >= 2, (
        f"forced draft never accepted on the hybrid (n_acc stayed 1: {naccs}) — the "
        f"gated verify logits are wrong (double o_proj / bad gate / recurrent-state leak)"
    )
    assert out_spec == out_ref, (
        f"hybrid spec-decode NOT bit-identical to non-spec greedy:\n  spec={out_spec}\n"
        f"   ref={out_ref}\n  (DeltaNet recurrent state not rolled back / replayed "
        f"correctly, or accepted-token K/V not committed)"
    )


def test_mtp_spec_decode_hybrid_bit_identical_concurrent():
    """Same hybrid bit-identity contract with THREE ragged sequences decoded
    concurrently — the recurrent snapshot/restore/replay must be correct per slot,
    not just for a single sequence."""
    torch.manual_seed(0)
    cfg = _cfg_mtp_hybrid()
    sd = _sd_mtp_hybrid(cfg)
    prompts = [[3, 1, 4, 1, 5, 9, 2, 6], [7, 2, 7, 1, 8], [1, 6, 1, 8, 0, 3]]
    params = SamplingParams(temperature=0.0, max_tokens=16)

    eng_ref = LLMEngine(
        cfg, _drop_mtp(sd), device="cuda", max_num_seqs=4, max_len=64, enable_cuda_graph=False
    )
    refs = eng_ref.generate(prompts, params)

    eng_spec = LLMEngine(
        cfg, sd, device="cuda", max_num_seqs=4, max_len=64, enable_cuda_graph=False
    )
    ref_by_prompt = {tuple(p): (len(p), r) for p, r in zip(prompts, refs)}
    naccs = _force_full_acceptance(eng_spec, ref_by_prompt)
    outs = eng_spec.generate(prompts, params)

    assert max(naccs) >= 2, f"forced draft never accepted in the hybrid concurrent batch: {naccs}"
    for i, (spec, ref) in enumerate(zip(outs, refs)):
        assert spec == ref, (
            f"hybrid seq {i}: spec-decode NOT bit-identical to non-spec greedy:\n"
            f"  spec={spec}\n   ref={ref}"
        )
