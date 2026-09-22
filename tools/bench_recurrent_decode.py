# SPDX-License-Identifier: MIT
"""Single-stream LFM2 decode: eager vs CUDA-graph tok/s (synthetic 2.6B-shaped weights).

Random weights -> tokens are meaningless, but the kernel-launch pattern and per-step
GPU work are real, so the eager-vs-graphed tok/s RATIO is the honest measure of what
CUDA-graph capture buys the recurrent (short-conv) decode path on this fleet.
"""
import time

import torch

from superl8serve.engine import LLMEngine, SamplingParams
from superl8serve.models import ModelConfig

torch.manual_seed(0)

# LFM2-2.6B-ish shape (vocab shrunk to keep embed/lm_head off the critical path & RAM):
H = 2048
LAYERS = 32
ATTN = {2, 5, 8, 11, 14, 17, 20, 23, 26, 29}  # ~1/3 full-attention, rest short-conv
cfg = ModelConfig(
    arch="lfm2", vocab_size=32000, hidden_size=H, num_hidden_layers=LAYERS,
    num_attention_heads=32, num_key_value_heads=8, intermediate_size=8192,
    max_position_embeddings=4096, head_dim=64, qk_norm=True, tie_word_embeddings=True,
    rms_norm_eps=1e-5, extra=dict(full_attn_idxs=sorted(ATTN), conv_L_cache=3),
)


def r(*s):
    return (torch.randn(*s, device="cuda", dtype=torch.float16) * 0.02)


def build_sd():
    hd, nh, nkv = cfg.resolved_head_dim(), cfg.num_attention_heads, cfg.num_key_value_heads
    sd = {"model.embed_tokens.weight": r(cfg.vocab_size, H), "model.norm.weight": r(H)}
    for i in range(LAYERS):
        p = f"model.layers.{i}"
        sd[f"{p}.operator_norm.weight"] = r(H)
        sd[f"{p}.ffn_norm.weight"] = r(H)
        for w, d in (("w1", cfg.intermediate_size), ("w3", cfg.intermediate_size), ("w2", H)):
            sd[f"{p}.feed_forward.{w}.weight"] = r(d, H if w != "w2" else cfg.intermediate_size)
        if i in ATTN:
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


def measure(enable_graph, sd, warmup=8, iters=64):
    eng = LLMEngine(cfg, sd, device="cuda", max_num_seqs=1, max_len=512,
                    enable_cuda_graph=enable_graph)
    prompt = [1, 2, 3, 4, 5, 6, 7, 8]
    sp = SamplingParams(temperature=0.0, max_tokens=warmup + iters, ignore_eos=True)
    sid = eng.add_request(prompt, sp)
    eng.step()  # prefill
    for _ in range(warmup):
        eng.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        eng.step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    g = eng.runner.graphed
    info = "eager (graph disabled)"
    if g is not None:
        info = f"supported={g.supported} captured_graphs={len(g._graphs)}"
    eng.forget(sid)
    del eng
    torch.cuda.empty_cache()
    return iters / dt, info


sd = build_sd()
print(f"config: hidden={H} layers={LAYERS} ({len(ATTN)} full-attn, {LAYERS-len(ATTN)} short-conv)")
eager_tps, eager_info = measure(False, sd)
print(f"EAGER   : {eager_tps:6.1f} tok/s   [{eager_info}]")
graph_tps, graph_info = measure(True, sd)
print(f"GRAPHED : {graph_tps:6.1f} tok/s   [{graph_info}]")
print(f"SPEEDUP : {graph_tps / eager_tps:.2f}x")
