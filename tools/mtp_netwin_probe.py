# SPDX-License-Identifier: MIT
"""MTP net-decode-speedup probe for the 4-bit Qwen3.5-9B.

Measures decode tok/s MTP-off vs MTP-on, accept rate, greedy bit-identity, and
(via cheap monkeypatched CUDA-event timers) attributes the on-path cost across
base fwd / draft / verify fwd / canonicalize / snapshot+restore. Run in the
superl8-serve test image, GPU 8 pinned via CUDA_VISIBLE_DEVICES=8 (-> cuda:0 inside).
"""
from __future__ import annotations

import os
import time

import torch

SUPERL8 = os.environ.get("QWEN35_9B_SUPERL8", os.path.join(os.environ.get("SUPERL8_WEIGHTS_DIR", "/path/to/weights"), "Qwen__Qwen3.5-9B.b4.superl8"))
TOK = os.environ.get("QWEN35_9B_TOK", os.path.join(os.environ.get("SUPERL8_WEIGHTS_DIR", "/path/to/weights"), "tok"))
MAXTOK = int(os.environ.get("MTP_PROBE_MAXTOK", "96"))

PROMPTS = [
    "Count from one to ten in words.",
    "Explain what a prime number is in one sentence.",
    "What is the capital of France, and why is it famous?",
]


def _mk_engine():
    from superl8serve.engine.llm_engine import LLMEngine
    from superl8serve.loader import checkpoint_info, load_superl8_state_dict
    from superl8serve.models.config import ModelConfig

    meta_cfg = dict(checkpoint_info(SUPERL8)["meta"]["config"])
    cfg = ModelConfig.from_hf(meta_cfg, arch="qwen3_5")
    weights = load_superl8_state_dict(SUPERL8, device="cuda")
    eng = LLMEngine(
        cfg, weights, device="cuda", max_num_seqs=2, max_len=1024, enable_cuda_graph=False
    )
    return eng


def _encode(tok, text):
    ids = tok.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True
    )
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if isinstance(ids, list):
        return ids[0] if ids and isinstance(ids[0], list) else ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
        return ids[0] if ids and isinstance(ids[0], list) else ids
    return list(ids)


def _run(eng, prompts, spec, params):
    eng.runner._spec_enabled = spec
    eng.runner.spec_stats = {"steps": 0, "drafts": 0, "accepts": 0}
    # fresh sequences each call
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = eng.generate([list(p) for p in prompts], params)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return outs, dt, dict(eng.runner.spec_stats)


class PhaseTimer:
    """Accumulate GPU time of selected runner phases via CUDA events (monkeypatch)."""

    def __init__(self, runner, model):
        self.runner = runner
        self.model = model
        self.acc = {}
        self._orig = {}

    def _wrap(self, obj, name, key):
        orig = getattr(obj, name)
        self._orig[(obj, name)] = orig

        def wrapped(*a, **kw):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            r = orig(*a, **kw)
            e.record()
            e.synchronize()
            self.acc[key] = self.acc.get(key, 0.0) + s.elapsed_time(e)
            return r

        setattr(obj, name, wrapped)

    def install(self):
        self._wrap(self.runner, "_compute_drafts", "draft")
        self._wrap(self.runner, "_canonicalize_accepted_kv", "canon")
        # Classify each model forward by its ForwardContext (verify vs base/other) via
        # a thin proxy that delegates every attribute to the real model and only times
        # __call__. The runner reaches the model through `self.runner.model`.
        real = self.runner.model
        acc = self.acc

        class _Proxy:
            def __call__(self, ids, positions, ctx, *a, **kw):
                key = "verify_fwd" if getattr(ctx, "is_verify", False) else "base_fwd"
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                r = real(ids, positions, ctx, *a, **kw)
                e.record()
                e.synchronize()
                acc[key] = acc.get(key, 0.0) + s.elapsed_time(e)
                return r

            def __getattr__(self, name):
                return getattr(real, name)

        self._orig[(self.runner, "model")] = real
        self.runner.model = _Proxy()
        return self

    def restore(self):
        for (obj, name), orig in self._orig.items():
            setattr(obj, name, orig)


def main():
    from transformers import AutoTokenizer
    from superl8serve.engine.sequence import SamplingParams

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(TOK)
    prompts = [_encode(tok, t) for t in PROMPTS]
    params = SamplingParams(temperature=0.0, max_tokens=MAXTOK)

    eng = _mk_engine()
    from superl8serve.layers.gqa_attention import _int8_verify_supports
    print(f"[probe] model loaded; int8-verify D=256 supported = {_int8_verify_supports(256)}")

    # -- MTP OFF (plain greedy) --
    outs_off, dt_off, _ = _run(eng, prompts, False, params)
    n_off = sum(len(o) for o in outs_off)
    print(f"[OFF ] {n_off} toks in {dt_off:.3f}s = {n_off/dt_off:.2f} tok/s")

    # -- MTP ON (full path) with phase timing --
    pt = PhaseTimer(eng.runner, eng.model).install()
    outs_on, dt_on, st = _run(eng, prompts, True, params)
    pt.restore()
    n_on = sum(len(o) for o in outs_on)
    acc = st["accepts"] / st["drafts"] if st["drafts"] else 0.0
    print(f"[ON  ] {n_on} toks in {dt_on:.3f}s = {n_on/dt_on:.2f} tok/s | "
          f"accept={acc:.3f} ({st['accepts']}/{st['drafts']}) steps={st['steps']}")
    print(f"       phase GPU-ms: base_fwd={pt.acc.get('base_fwd',0):.0f} "
          f"verify_fwd={pt.acc.get('verify_fwd',0):.0f} "
          f"draft={pt.acc.get('draft',0):.0f} canon={pt.acc.get('canon',0):.0f}")

    # -- bit-identity --
    ident = all(outs_on[i] == outs_off[i] for i in range(len(prompts)))
    agree = []
    for a, b in zip(outs_on, outs_off):
        n = min(len(a), len(b))
        agree.append(sum(1 for j in range(n) if a[j] == b[j]) / max(1, n))
    print(f"[IDENT] bit-identical={ident}  per-prompt-agree={[f'{x:.2%}' for x in agree]}")

    # -- ON with canon disabled (isolate canon cost / correctness contribution) --
    noop = lambda *a, **kw: None  # noqa: E731
    eng.runner._canonicalize_accepted_kv = noop
    outs_nc, dt_nc, st_nc = _run(eng, prompts, True, params)
    n_nc = sum(len(o) for o in outs_nc)
    ident_nc = all(outs_nc[i] == outs_off[i] for i in range(len(prompts)))
    print(f"[ON-noCanon] {n_nc/dt_nc:.2f} tok/s  bit-identical={ident_nc}")
    # restore real method
    del eng.runner._canonicalize_accepted_kv

    # -- ON with snapshot/restore disabled (isolate recurrent snapshot cost) --
    dt_ns = None
    if eng.runner.has_recurrent:
        orig_snap = eng.lin_cache.snapshot
        orig_rest = eng.lin_cache.restore
        eng.lin_cache.snapshot = lambda *a, **kw: None
        eng.lin_cache.restore = lambda *a, **kw: None
        outs_ns, dt_ns, _ = _run(eng, prompts, True, params)
        eng.lin_cache.snapshot = orig_snap
        eng.lin_cache.restore = orig_rest
        n_ns = sum(len(o) for o in outs_ns)
        print(f"[ON-noSnap ] {n_ns/dt_ns:.2f} tok/s (snapshot/restore stubbed; correctness N/A)")

    print("\n=== SUMMARY ===")
    print(f"off      : {n_off/dt_off:.2f} tok/s")
    print(f"on(full) : {n_on/dt_on:.2f} tok/s  ({(n_on/dt_on)/(n_off/dt_off):.2f}x)  identical={ident}")
    print(f"on(noCan): {n_nc/dt_nc:.2f} tok/s  ({(n_nc/dt_nc)/(n_off/dt_off):.2f}x)  identical={ident_nc}")
    print(f"accept   : {acc:.3f}")


if __name__ == "__main__":
    main()
