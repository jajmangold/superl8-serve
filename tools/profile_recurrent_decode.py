# SPDX-License-Identifier: MIT
"""Profile ONE LFM2 decode step (eager): per-op CUDA time + kernel launch counts,
to locate where 71 ms/step actually goes (launch-bound vs recurrence einsum vs
structural O(N)). Synthetic 2.6B-shaped weights; the launch pattern is real."""
import torch
from torch.profiler import ProfilerActivity, profile

from superl8serve.engine import LLMEngine, SamplingParams
from superl8serve.models import ModelConfig

torch.manual_seed(0)
H, LAYERS = 2048, 32
ATTN = {2, 5, 8, 11, 14, 17, 20, 23, 26, 29}
cfg = ModelConfig(
    arch="lfm2", vocab_size=32000, hidden_size=H, num_hidden_layers=LAYERS,
    num_attention_heads=32, num_key_value_heads=8, intermediate_size=8192,
    max_position_embeddings=4096, head_dim=64, qk_norm=True, tie_word_embeddings=True,
    rms_norm_eps=1e-5, extra=dict(full_attn_idxs=sorted(ATTN), conv_L_cache=3),
)


def r(*s):
    return torch.randn(*s, device="cuda", dtype=torch.float16) * 0.02


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


sd = build_sd()
eng = LLMEngine(cfg, sd, device="cuda", max_num_seqs=1, max_len=512, enable_cuda_graph=False)
eng.add_request([1, 2, 3, 4, 5, 6, 7, 8], SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True))
eng.step()  # prefill
for _ in range(8):
    eng.step()
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
    for _ in range(10):
        eng.step()
    torch.cuda.synchronize()

evts = prof.key_averages()
tot_cuda = sum(e.self_device_time_total for e in evts)
tot_launches = sum(e.count for e in evts if e.self_device_time_total > 0)
print(f"=== EAGER decode: 10 steps, total CUDA self-time {tot_cuda/1000:.2f} ms "
      f"({tot_cuda/1000/10:.2f} ms/step), ~{tot_launches//10} device ops/step ===\n")
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
