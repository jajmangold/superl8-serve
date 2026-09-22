# SPDX-License-Identifier: MIT
"""A/B throughput sweep for the scheduler preemption-thrash fix.

Runs a controlled `LLMEngine` (tiny random Qwen3, so the numbers isolate the
*scheduler policy* rather than kernel speed) at increasing concurrency with a fixed
`max_num_seqs`, under two schedulers on identical hardware in one process:

  * OLD: the count-cap preemption policy (evict+full-context-recompute whenever the
    slot pool is full and anything waits) -- reproduces the collapse.
  * NEW: admit-up-to-capacity, queue-the-rest-FIFO, preempt only on genuine
    KV-block-pool exhaustion.

Reports aggregate decode throughput (tok/s) per concurrency for both. The absolute
tok/s is tiny-model-specific; what matters is the SHAPE: OLD cliffs once concurrency
exceeds max_num_seqs, NEW holds the ~max_num_seqs plateau.
"""

from __future__ import annotations

import time

import torch

from superl8serve.engine import LLMEngine, SamplingParams
from superl8serve.engine.scheduler import Scheduler
from superl8serve.engine.sequence import Status

MAX_NUM_SEQS = 16
PROMPT_LEN = 32
MAX_TOKENS = 32
CONCURRENCIES = [1, 8, 16, 24, 32, 48]


def _cfg():
    from superl8serve.models import ModelConfig

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


def _old_schedule(self):
    """The original (buggy) count-cap preemption schedule(), verbatim, for A/B."""
    batch, tokens = [], 0
    while True:
        while (
            self.waiting
            and len(self.running) + len(batch) < self.max_num_seqs
            and self.cache.has_free_slot()
        ):
            seq = self.waiting[0]
            if batch and tokens + seq.num_prompt > self.max_batch_tokens:
                break
            self.waiting.popleft()
            seq.slot = self.cache.alloc()
            matched, shared_blocks = self.cache.lookup_prefix(seq.prompt_ids)
            if matched > 0 and shared_blocks:
                self.cache.share_blocks(seq.slot, shared_blocks)
                seq.prefix_matched_len = matched
            seq.status = Status.RUNNING
            tokens += seq.num_prompt
            batch.append(seq)
        if batch:
            return batch, True
        if self.waiting and (
            len(self.running) >= self.max_num_seqs or not self.cache.has_free_slot()
        ):
            self._preempt_one()
            continue
        return list(self.running), False


def run_sweep(label, old=False):
    cfg = _cfg()
    torch.manual_seed(0)
    sd = _sd(cfg)
    orig = Scheduler.schedule
    if old:
        Scheduler.schedule = _old_schedule
    print(
        f"\n=== {label} (max_num_seqs={MAX_NUM_SEQS}, prompt={PROMPT_LEN}, "
        f"max_tokens={MAX_TOKENS}) ==="
    )
    print(f"{'concurrency':>11} | {'wall_s':>8} | {'tok/s':>9} | {'preempts':>8}")
    results = {}
    try:
        for c in CONCURRENCIES:
            eng = LLMEngine(
                cfg,
                sd,
                device="cuda",
                max_num_seqs=MAX_NUM_SEQS,
                max_len=128,
                enable_cuda_graph=False,
            )
            preempts = {"n": 0}
            base = eng.scheduler._preempt_one

            def counting(_base=base, _p=preempts):
                _p["n"] += 1
                _base()

            eng.scheduler._preempt_one = counting
            prompts = [[(i * 7 + j) % 200 + 1 for j in range(PROMPT_LEN)] for i in range(c)]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                outs = eng.generate(prompts, SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS))
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                total_tok = sum(len(o) for o in outs)
                toks = total_tok / dt
                results[c] = (dt, toks, preempts["n"])
                print(f"{c:>11} | {dt:>8.3f} | {toks:>9.1f} | {preempts['n']:>8}")
            except Exception as e:  # noqa: BLE001 -- collapse can surface as a crash
                dt = time.perf_counter() - t0
                results[c] = (dt, 0.0, preempts["n"])
                print(
                    f"{c:>11} | {dt:>8.3f} | {'CRASH':>9} | {preempts['n']:>8}  "
                    f"({type(e).__name__}: {e})"
                )
            del eng
            torch.cuda.empty_cache()
    finally:
        Scheduler.schedule = orig
    return results


if __name__ == "__main__":
    old = run_sweep("OLD (count-cap preemption)", old=True)
    new = run_sweep("NEW (queue-FIFO, preempt only on block exhaustion)", old=False)
    print("\n=== SUMMARY: aggregate tok/s ===")
    print(f"{'concurrency':>11} | {'OLD tok/s':>10} | {'NEW tok/s':>10} | {'speedup':>8}")
    for c in CONCURRENCIES:
        o = old[c][1]
        n = new[c][1]
        o_s = f"{o:>10.1f}" if o > 0 else f"{'CRASH':>10}"
        speed = f"{n / o:>7.1f}x" if o > 0 else f"{'inf':>8}"
        print(f"{c:>11} | {o_s} | {n:>10.1f} | {speed}")
