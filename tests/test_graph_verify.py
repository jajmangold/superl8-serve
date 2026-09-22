# SPDX-License-Identifier: MIT
"""CUDA-graph spec-decode VERIFY capture (issue #259 / task #100).

Base decode is CUDA-graphed but the spec-decode verify forward ran 100% eager, so on
this dispatch-bound fleet every spec mode was a NET SLOWDOWN graphs-on despite high
acceptance + bit-identical output (#266). `GraphedVerify` captures the fixed-draft-
length verify forward the same way `GraphedDecode` captures base decode.

The one thing that MUST hold: the CAPTURED verify forward produces BYTE-IDENTICAL
greedy tokens (and, for recurrent hybrids, the same committed recurrent state) to the
EAGER verify forward from the same committed prefix — spec-decode is lossless only if
the graphed verify commits exactly what the eager verify would. These tests drive
`GraphedVerify.try_run` directly against a hand-built eager verify over dense
(full-attention) and recurrent (LFM2 short-conv) synthetic models, so they run in CI
without a real checkpoint. The end-to-end bit-identity on the real Qwen3.5 DeltaNet
hybrid is `bench/spec_decode_bench.py` (its `identical` column) + the netwin gate.
"""
import pytest
import torch

pytest.importorskip("superl8")

from superl8serve.engine.kv_cache import PagedKVCache
from superl8serve.engine.model_runner import EngineRunner
from superl8serve.engine.sequence import SamplingParams, Sequence
from superl8serve.models import ModelConfig, build_model
from superl8serve.models.base import ForwardContext
from superl8serve.models.cache import RecurrentStateCache

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="cuda-graph verify needs a real CUDA device")


# -- synthetic models (mirror tests/test_cuda_graph.py) -------------------------
def _dense_cfg(**o):
    base = dict(arch="qwen3", vocab_size=256, hidden_size=128, num_hidden_layers=3,
                num_attention_heads=4, num_key_value_heads=2, intermediate_size=256,
                max_position_embeddings=512, head_dim=32, qk_norm=True,
                tie_word_embeddings=True)
    base.update(o)
    return ModelConfig(**base)


def _dense_sd(cfg):
    def r(*s):
        return torch.randn(*s, device="cuda", dtype=torch.float16) * 0.05
    hd, nh, nkv, H = (cfg.resolved_head_dim(), cfg.num_attention_heads,
                      cfg.num_key_value_heads, cfg.hidden_size)
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


def _lfm2_cfg(**o):
    base = dict(arch="lfm2", vocab_size=256, hidden_size=128, num_hidden_layers=3,
                num_attention_heads=4, num_key_value_heads=2, intermediate_size=256,
                max_position_embeddings=512, head_dim=32, qk_norm=True,
                tie_word_embeddings=True, rms_norm_eps=1e-5,
                extra=dict(full_attn_idxs=[1], conv_L_cache=3))
    base.update(o)
    return ModelConfig(**base)


def _lfm2_sd(cfg):
    def r(*s):
        return torch.randn(*s, device="cuda", dtype=torch.float16) * 0.05
    H = cfg.hidden_size
    sd = {"model.embed_tokens.weight": r(cfg.vocab_size, H), "model.norm.weight": r(H)}
    hd, nh, nkv = cfg.resolved_head_dim(), cfg.num_attention_heads, cfg.num_key_value_heads
    attn = set(cfg.extra["full_attn_idxs"])
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.operator_norm.weight"] = r(H)
        sd[f"{p}.ffn_norm.weight"] = r(H)
        for w, d in (("w1", cfg.intermediate_size), ("w3", cfg.intermediate_size), ("w2", H)):
            sd[f"{p}.feed_forward.{w}.weight"] = r(d, H if w != "w2" else cfg.intermediate_size)
        if i in attn:
            a = f"{p}.self_attn"
            sd[f"{a}.q_proj.weight"] = r(nh * hd, H)
            sd[f"{a}.k_proj.weight"] = r(nkv * hd, H)
            sd[f"{a}.v_proj.weight"] = r(nkv * hd, H)
            sd[f"{a}.out_proj.weight"] = r(H, nh * hd)
            sd[f"{a}.q_layernorm.weight"] = r(hd)
            sd[f"{a}.k_layernorm.weight"] = r(hd)
        else:
            c = f"{p}.conv"
            sd[f"{c}.in_proj.weight"] = r(3 * H, H)
            sd[f"{c}.out_proj.weight"] = r(H, H)
            sd[f"{c}.conv.weight"] = r(H, 1, 3)
    return sd


def _runner(cfg, sd, *, num_slots=8, max_len=128):
    model = build_model(cfg, sd).cuda().eval()
    cache = PagedKVCache(cfg.num_hidden_layers, num_slots + 1, cfg.num_key_value_heads,
                         max_len, cfg.resolved_head_dim(), device="cuda")
    lin = RecurrentStateCache()
    return EngineRunner(model, cache, device="cuda", enable_cuda_graph=True,
                        spec_decode=True, lin_cache=lin)


def _prefill(runner, prompts):
    seqs = []
    for i, p in enumerate(prompts):
        s = Sequence(i, list(p), SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True))
        s.slot = runner.cache.alloc()
        seqs.append(s)
    runner.prefill(seqs)
    return seqs


def _eager_verify(runner, batch, verify_ids_l, verify_pos_l, lengths):
    """The runner's eager verify forward (no graph), returning true greedy tokens."""
    dev = runner.device
    slots = [s.slot for s in batch]
    S = len(verify_ids_l[0])
    verify_ids = torch.tensor(verify_ids_l, device=dev)
    verify_pos = torch.tensor(verify_pos_l, device=dev)
    runner.cache.ensure_capacity(slots, [n + 1 + S for n in lengths])
    flat_slots, flat_pos = [], []
    for b in range(len(batch)):
        base = lengths[b] + 1
        for t in range(S):
            flat_slots.append(slots[b]); flat_pos.append(base + t)
    vsm = runner.cache.slot_mapping_for(flat_slots, flat_pos)
    ctx = ForwardContext(is_prefill=False, is_verify=True, kv_cache=runner.cache,
                         lin_cache=runner.lin_cache, slots=slots,
                         slot_lengths=[n + 1 for n in lengths], verify_slot_mapping=vsm)
    if runner.has_recurrent:
        runner.lin_cache.bind(slots)
        runner.lin_cache.begin_verify_capture()
    hidden = runner.model(verify_ids, verify_pos, ctx)
    true = runner.model.compute_logits(hidden).argmax(-1)
    # NB: leave the verify capture ARMED (do not end_verify_capture) — the recurrent
    # trajectory must survive for the caller's commit_verify, exactly as the runner
    # leaves it (commit_verify clears it). The dense path has no trajectory to leak.
    return true


def _verify_inputs(batch, lengths, spec_k, vocab):
    """Deterministic pseudo-drafts (fixed S = spec_k + 1) for each row."""
    ids, pos = [], []
    for b, seq in enumerate(batch):
        base = (seq.last_token * 7 + 3) % vocab
        drafts = [((seq.last_token + t * 13 + 5) % vocab) for t in range(spec_k)]
        ids.append([base] + drafts)
        pos.append([lengths[b] + 1 + t for t in range(spec_k + 1)])
    return ids, pos


@pytest.mark.parametrize("num_seqs", [1, 3])
def test_graphed_verify_matches_eager_dense(num_seqs):
    """Graphed verify true-tokens == eager verify true-tokens (full attention)."""
    torch.manual_seed(0)
    cfg = _dense_cfg()
    runner = _runner(cfg, _dense_sd(cfg))
    assert runner.graphed_verify is not None and runner.graphed_verify.supported
    prompts = [[(i * 7 + j) % 200 + 1 for j in range(4 + i)] for i in range(num_seqs)]
    batch = _prefill(runner, prompts)
    lengths = [s.length for s in batch]
    vids, vpos = _verify_inputs(batch, lengths, spec_k=4, vocab=cfg.vocab_size)

    res = runner.graphed_verify.try_run(batch, vids, vpos, lengths)
    assert res is not None, "expected a capturable verify bucket"
    _, true_g = res
    true_g = true_g.clone()
    assert len(runner.graphed_verify._graphs) == 1

    true_e = _eager_verify(runner, batch, vids, vpos, lengths)
    assert torch.equal(true_g, true_e), (true_g, true_e)


def test_graphed_verify_deterministic_dense():
    torch.manual_seed(1)
    cfg = _dense_cfg()
    runner = _runner(cfg, _dense_sd(cfg))
    batch = _prefill(runner, [[3, 1, 4, 1, 5], [9, 2, 6, 5]])
    lengths = [s.length for s in batch]
    vids, vpos = _verify_inputs(batch, lengths, spec_k=5, vocab=cfg.vocab_size)
    outs = []
    for _ in range(3):
        _, t = runner.graphed_verify.try_run(batch, vids, vpos, lengths)
        outs.append(t.clone())
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[1], outs[2])


@pytest.mark.parametrize("num_seqs", [1, 3])
def test_graphed_verify_matches_eager_recurrent_lfm2(num_seqs):
    """Recurrent hybrid: graphed verify must match eager on BOTH the greedy tokens and
    the committed recurrent (conv-tail) state after an accepted prefix length."""
    torch.manual_seed(2)
    cfg = _lfm2_cfg()
    runner = _runner(cfg, _lfm2_sd(cfg))
    assert runner.has_recurrent and runner.graphed_verify is not None
    prompts = [[(i * 5 + j) % 180 + 1 for j in range(5 + i)] for i in range(num_seqs)]
    batch = _prefill(runner, prompts)
    slots = [s.slot for s in batch]
    lengths = [s.length for s in batch]
    spec_k = 4
    vids, vpos = _verify_inputs(batch, lengths, spec_k=spec_k, vocab=cfg.vocab_size)
    row_last = [min(b + 1, spec_k) for b in range(num_seqs)]  # accepted prefix per row

    # The runner drives spec-decode under @torch.inference_mode(); mirror that so the
    # commit's in-place state scatter is allowed on the inference-tensor buffers.
    with torch.inference_mode():
        # Snapshot the committed recurrent state so eager and graphed start identically.
        runner.lin_cache.bind(slots)
        snap = runner.lin_cache.snapshot()

        # Graphed verify → commit accepted-length state.
        _, true_g = runner.graphed_verify.try_run(batch, vids, vpos, lengths)
        true_g = true_g.clone()
        runner.lin_cache.commit_verify(row_last)
        runner.lin_cache.bind(slots)
        state_g = runner.lin_cache.snapshot()

        # Reset to the pre-verify committed state, run eager verify + commit.
        runner.lin_cache.bind(slots)
        runner.lin_cache.restore(snap)
        true_e = _eager_verify(runner, batch, vids, vpos, lengths)
        runner.lin_cache.commit_verify(row_last)
        runner.lin_cache.bind(slots)
        state_e = runner.lin_cache.snapshot()

    assert torch.equal(true_g, true_e), (true_g, true_e)
    for lidx in state_e["conv"]:
        assert torch.equal(state_g["conv"][lidx], state_e["conv"][lidx]), f"conv tail layer {lidx}"
