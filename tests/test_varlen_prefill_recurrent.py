# SPDX-License-Identifier: MIT
"""EngineRunner varlen-prefill eligibility for RECURRENT models (issue #351).

LFM2's short-conv mixers are `varlen_prefill_safe`: ShortConv.forward switches to a
segmented causal conv keyed on ctx.cu_seqlens, so a whole ragged batch can be packed
into ONE prefill forward without running the conv scan across sequence boundaries.
Gated DeltaNet / lightning mixers are NOT safe and must stay one-at-a-time serial.
The runner only engages `_prefill_varlen` when every recurrent layer advertises the
marker, clears only the incoming slots (never the global `lin_cache`), and binds the
batch's slots so each layer scatters its per-slot state to the right rows.
"""
import pytest
import torch

pytest.importorskip("superl8")

from superl8serve.engine.kv_cache import PagedKVCache
from superl8serve.engine.model_runner import EngineRunner
from superl8serve.engine.sequence import SamplingParams, Sequence
from superl8serve.models import ModelConfig, build_model
from superl8serve.models.cache import RecurrentStateCache


def _cfg(**o):
    base = dict(arch="lfm2", vocab_size=32, hidden_size=64, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=1, intermediate_size=128,
                max_position_embeddings=256, head_dim=16, qk_norm=True,
                tie_word_embeddings=True, rms_norm_eps=1e-5,
                extra=dict(full_attn_idxs=[], conv_L_cache=3))
    base.update(o)
    return ModelConfig(**base)


def _sd(cfg):
    """CPU fp16 weights for a pure-ShortConv LFM2 (every layer a conv mixer)."""
    torch.manual_seed(0)

    def r(*s):
        return torch.randn(*s, dtype=torch.float16) * 0.05

    H = cfg.hidden_size
    sd = {"model.embed_tokens.weight": r(cfg.vocab_size, H), "model.norm.weight": r(H)}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.operator_norm.weight"] = r(H)
        sd[f"{p}.ffn_norm.weight"] = r(H)
        for w, d in (("w1", cfg.intermediate_size), ("w3", cfg.intermediate_size), ("w2", H)):
            sd[f"{p}.feed_forward.{w}.weight"] = r(d, H if w != "w2" else cfg.intermediate_size)
        c = f"{p}.conv"
        sd[f"{c}.in_proj.weight"] = r(3 * H, H)
        sd[f"{c}.out_proj.weight"] = r(H, H)
        sd[f"{c}.conv.weight"] = r(H, 1, 3)
    return sd


def _runner(cfg, *, sabotage=None, num_slots=8):
    model = build_model(cfg, _sd(cfg)).eval()
    if sabotage is not None:
        sabotage(model)
    cache = PagedKVCache(cfg.num_hidden_layers, num_slots, cfg.num_key_value_heads,
                         cfg.max_position_embeddings, cfg.resolved_head_dim(), device="cpu")
    return EngineRunner(model, cache, device="cpu", enable_cuda_graph=False,
                        spec_decode=False, lin_cache=RecurrentStateCache())


def _seq(i, prompt, slot):
    s = Sequence(i, list(prompt), SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True))
    s.slot = slot
    return s


def _ragged_prompts(B):
    torch.manual_seed(5)
    return [[1 + (i * 7 + t * 3) % 31 for t in range(1 + (i % 7))] for i in range(B)]


def test_lfm2_varlen_prefill_safe_uses_varlen_path(monkeypatch):
    """A pure-ShortConv LFM2 recurrent model is `varlen_prefill_safe` and a
    multi-sequence prefill actually routes through `_prefill_varlen` (and runs)."""
    cfg = _cfg()
    runner = _runner(cfg)
    assert runner.has_recurrent is True
    assert runner._varlen_prefill_safe is True

    calls = []
    orig = EngineRunner._prefill_varlen

    def spy(self, batch):
        calls.append(len(batch))
        return orig(self, batch)

    monkeypatch.setattr(EngineRunner, "_prefill_varlen", spy)
    seqs = [_seq(i, p, runner.cache.alloc()) for i, p in enumerate(_ragged_prompts(2))]
    toks = runner.prefill(seqs)
    assert calls == [2], "recurrent-safe model must take the varlen prefill path"
    assert len(toks) == 2 and all(0 <= t < cfg.vocab_size for t in toks)


def test_unsafe_recurrent_layer_stays_serial(monkeypatch):
    """If ANY recurrent layer is not `varlen_prefill_safe` (DeltaNet / lightning
    are), the whole model falls back to one-at-a-time serial prefill."""
    cfg = _cfg()
    runner = _runner(cfg, sabotage=lambda m: setattr(m.layers[0].mixer, "varlen_prefill_safe", False))
    assert runner.has_recurrent is True
    assert runner._varlen_prefill_safe is False

    calls = []
    orig = EngineRunner._prefill_varlen

    def spy(self, batch):
        calls.append(len(batch))
        return orig(self, batch)

    monkeypatch.setattr(EngineRunner, "_prefill_varlen", spy)
    seqs = [_seq(i, p, runner.cache.alloc()) for i, p in enumerate(_ragged_prompts(2))]
    toks = runner.prefill(seqs)
    assert calls == [], "an unsafe recurrent layer must force serial prefill"
    assert len(toks) == 2


@pytest.mark.parametrize("B", [1, 16, 64, 128])
def test_lfm2_varlen_ragged_matches_serial(B):
    """Ragged varlen prefill (the segmented ShortConv path) must reproduce the
    serial one-sequence-at-a-time greedy outputs — including a following decode
    step that carries each slot's conv tail — with per-slot state isolated."""
    cfg = _cfg()
    runner = _runner(cfg, num_slots=2 * B + 2)
    prompts = _ragged_prompts(B)

    serial_seqs = [_seq(i, p, runner.cache.alloc()) for i, p in enumerate(prompts)]
    serial_toks = [runner.prefill([s])[0] for s in serial_seqs]

    varlen_seqs = [_seq(i, p, runner.cache.alloc()) for i, p in enumerate(prompts)]
    varlen_toks = runner.prefill(varlen_seqs)
    assert varlen_toks == serial_toks, "varlen greedy prefill tokens must match serial"

    # Per-slot conv-tail isolation: each varlen slot's trailing window must be
    # BIT-IDENTICAL to the serial slot's for the same sequence.
    for s, v in zip(serial_seqs, varlen_seqs):
        runner.lin_cache.bind([s.slot])
        tail_s = runner.lin_cache.get_conv_tail(0)
        runner.lin_cache.bind([v.slot])
        tail_v = runner.lin_cache.get_conv_tail(0)
        assert tail_s is not None and tail_v is not None
        torch.testing.assert_close(tail_v, tail_s, rtol=0, atol=0)

    # A decode step from the varlen-prefilled state must match the serial decode.
    if B > 1:
        dec_v = runner.decode(varlen_seqs)
        dec_s = runner.decode(serial_seqs)
        assert dec_v == dec_s, "decode from varlen-prefilled slots must match serial"


def test_varlen_preserves_unrelated_slot_state():
    """`_prefill_varlen` must clear only the incoming slots — never the global
    `lin_cache` — so a running sequence's per-slot state survives a varlen prefill
    of other sequences (the old `lin_cache.reset()` wiped every slot)."""
    cfg = _cfg()
    runner = _runner(cfg)
    s0 = runner.cache.alloc()
    runner.prefill([_seq(0, [1, 2, 3, 4, 5], s0)])  # serial; leaves slot s0 conv tail
    runner.lin_cache.bind([s0])
    tail_before = runner.lin_cache.get_conv_tail(0)
    assert tail_before is not None, "prefill must leave a conv tail on its own slot"

    seqs = [_seq(i, p, runner.cache.alloc()) for i, p in enumerate(_ragged_prompts(2))]
    runner.prefill(seqs)  # varlen into OTHER slots

    runner.lin_cache.bind([s0])
    tail_after = runner.lin_cache.get_conv_tail(0)
    assert tail_after is not None, "unrelated slot state was cleared by varlen prefill"
    torch.testing.assert_close(tail_after, tail_before, rtol=0, atol=0)


def test_varlen_flat_slot_mapping_matches_reference():
    """`PagedKVCache.flat_slot_mapping` must be BYTE-IDENTICAL to the per-token
    `_slot_mapping` reference (host-only integer arithmetic — no device tensor, no
    per-token `.item()` sync) across block boundaries and ragged lengths, and slots
    must not share physical blocks."""
    cache = PagedKVCache(1, 8, 1, 128, 16, device="cpu", block_size=16)
    slots = [cache.alloc() for _ in range(3)]
    lengths = [1, 16, 33]  # boundary: single token, exactly one block, spans 3 blocks
    cache.ensure_capacity(slots, lengths)
    for s, n in zip(slots, lengths):
        flat = cache.flat_slot_mapping(s, n)
        ref = [cache._slot_mapping([s], [t]).item() for t in range(n)]
        assert flat == ref, f"slot {s} len {n}: flat != reference"
        assert len(set(flat)) == n, "positions within a slot must map to distinct slots"

    owned = [set(cache._slot_blocks[s]) for s in slots]
    assert sum(len(o) for o in owned) == len(set().union(*owned)), "slots share blocks"


def test_varlen_mixed_lifecycle_running_decode_plus_varlen_prefill():
    """A RUNNING sequence keeps decoding while a NEW varlen batch prefills and joins
    the shared decode batch. Every token (first decode of the running seq, varlen
    prefill of the newcomers, combined decode) must match the all-serial reference —
    the running sequence's per-slot conv tail must survive the varlen prefill."""
    cfg = _cfg()
    runner_v = _runner(cfg, num_slots=8)
    runner_s = _runner(cfg, num_slots=8,
                       sabotage=lambda m: setattr(m.layers[0].mixer, "varlen_prefill_safe", False))
    A = [1, 2, 3, 4, 5]
    Bs = _ragged_prompts(3)

    def run_mixed(runner):
        seq_a = _seq(0, A, runner.cache.alloc())
        runner.prefill([seq_a])                      # B1 serial prefill (running)
        d1 = runner.decode([seq_a])[0]               # decode the running seq once
        bs = [_seq(i + 1, q, runner.cache.alloc()) for i, q in enumerate(Bs)]
        pv = runner.prefill(bs)                      # varlen (or serial) prefill while A runs
        dc = runner.decode([seq_a] + bs)             # combined decode
        return d1, pv, dc

    r_v = run_mixed(runner_v)
    r_s = run_mixed(runner_s)
    assert r_v == r_s, "mixed lifecycle (running decode + varlen prefill) must match serial"


def test_varlen_slot_reuse_across_batches():
    """Slots freed after a completed batch are REALLOCATED cleanly: a second varlen
    batch reusing the same slot numbers must match the serial reference with no stale
    per-slot conv tails leaking in."""
    cfg = _cfg()
    runner = _runner(cfg, num_slots=8)
    prompts = _ragged_prompts(4)

    seqs1 = [_seq(i, q, runner.cache.alloc()) for i, q in enumerate(prompts)]
    runner.prefill(seqs1)                            # varlen batch 1 (leaves conv tails)
    for s in seqs1:
        runner.cache.free(s.slot)                    # batch 1 completes -> slots freed

    ser = [_seq(100 + i, q, runner.cache.alloc()) for i, q in enumerate(prompts)]
    ser_toks = [runner.prefill([s])[0] for s in ser]  # serial reference on reused slots
    # Snapshot the serial conv tails BY PROMPT before the slots are recycled (LIFO
    # free/alloc permutes slot->prompt, so reading `ser[i].slot` later would not be
    # a serial reference to prompt i).
    ser_tails = {}
    for i, s in enumerate(ser):
        runner.lin_cache.bind([s.slot])
        t = runner.lin_cache.get_conv_tail(0)
        assert t is not None, f"serial prefill left no tail for prompt {i}"
        ser_tails[i] = t.clone()
    for s in ser:
        runner.cache.free(s.slot)

    seqs2 = [_seq(200 + i, q, runner.cache.alloc()) for i, q in enumerate(prompts)]
    toks2 = runner.prefill(seqs2)                    # varlen batch 2 on reused slots
    assert toks2 == ser_toks, "reused varlen slots must not leak stale state"

    for i, v in enumerate(seqs2):
        runner.lin_cache.bind([v.slot])
        tail_v = runner.lin_cache.get_conv_tail(0)
        assert tail_v is not None, f"varlen prefill left no tail for prompt {i}"
        torch.testing.assert_close(tail_v, ser_tails[i], rtol=0, atol=0)
