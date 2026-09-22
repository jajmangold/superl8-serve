#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""K8V3 needle-in-haystack eval for Qwen3.8-27B-MTP-TQ3_4S (superl8-serve#410).

The acceptance gate for the K8V3 cache: 3-bit V sits on the 16 long-range
full-attention layers, so a retrieval/needle eval — not just perplexity — is
what proves (or disproves) that V quality. Runs the SAME needle at increasing
context lengths, comparing the K8V3 format against the int8-V (K8V8) fallback.

Modes:
  full model (default):   the entire 64-layer qwen3_5 hybrid — fits ~64-96k ctx
                          on one 16 GiB card (12.73 GiB resident weights).
  --layer-scope:          the 16 full-attention layers in isolation (embed +
                          full-attn blocks + MLP + LM head, no DeltaNet). This is
                          exactly the layer set the K8V3 cache serves; at ~4 GiB
                          of weights it leaves room for the full 262k K8V3 cache
                          (5.69 GiB) on one card — the 262k gate.

Usage:
  python3 bench/needle_k8v3.py --ctx 32768 --format k8v3
  python3 bench/needle_k8v3.py --ctx 65536 --format k8v8
  python3 bench/needle_k8v3.py --ctx 262144 --format k8v3 --layer-scope
  python3 bench/needle_k8v3.py --ctx 262144 --format k8v8 --layer-scope
  python3 bench/needle_k8v3.py --probe              # memory calibration only

Pinned to a free GPU via CUDA_VISIBLE_DEVICES (e.g. =8). Model path and
tokenizer are absolute on-disk artifacts (mounted into the test container).
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import time

import torch
import torch.nn as nn

from superl8 import QTensor

from superl8serve.engine.kv_cache import PagedKVCache
from superl8serve.models.base import ForwardContext
from superl8serve.models.cache import RecurrentStateCache
from superl8serve.models.config import ModelConfig
from superl8serve.models.qwen3_5 import _build_mlp, _gated_full_attn
from superl8serve.layers.embedding import LMHead, VocabEmbedding
from superl8serve.layers.norm import RMSNorm
from superl8serve.layers.rotary import RotaryEmbedding

GGUF = "/qwen38-gguf/Qwen3.8-27B-MTP-TQ3_4S.gguf"
TOKENIZER_DIR = "/qwen38-model-src"

HD = 256
NKV = 4
BS = 16
LLOYD_BLOCK = 128
KV_LAYERS = [i for i in range(64) if (i + 1) % 4 == 0]  # full-attention layers

HAYSTACK = "The quick brown fox jumps over the lazy dog. "
NEEDLE = "The secret passphrase is K8V3NEEDLE7. "
QUESTION = (
    "\n\nQuestion: What is the exact secret passphrase mentioned in the document? "
    "Answer with only the passphrase.\nAnswer: The passphrase is "
)
ANSWER = "K8V3NEEDLE7"


# ── layer-scope (reduced) model ──────────────────────────────────────────────


class FullAttnScopeDecoderLayer(nn.Module):
    """One full-attention layer of the qwen3_5 hybrid, in isolation (the production
    `Qwen3_5DecoderLayer` logic for a full-attn layer, without the DeltaNet branch)."""

    def __init__(self, cfg: ModelConfig, i: int, sd: dict, rope: RotaryEmbedding):
        super().__init__()
        self.layer_idx = i
        p = f"model.layers.{i}"
        self.input_layernorm = RMSNorm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            sd[f"{p}.input_layernorm.weight"],
            add_unit_offset=True,
        )
        self.attn = _gated_full_attn(cfg, sd, p, rope)
        self.post_attention_layernorm = RMSNorm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            sd[f"{p}.post_attention_layernorm.weight"],
            add_unit_offset=True,
        )
        self.mlp = _build_mlp(cfg, sd, p)

    def forward(self, x, positions, ctx, residual):
        if residual is None:
            residual, h = x, self.input_layernorm(x)
        else:
            h, residual = self.input_layernorm(x, residual)
        h = self.attn(h, positions, ctx, self.layer_idx)
        h, residual = self.post_attention_layernorm(h, residual)
        return self.mlp(h), residual


class FullAttnScopeModel(nn.Module):
    """The 16 full-attention layers of Qwen3.8-27B in isolation — the exact layer
    set the K8V3 cache serves. Reuses the production full-attn block + MLP builders
    and mirrors `Qwen3_5ForCausalLM`'s forward; DeltaNet layers are omitted (they
    carry recurrent state, never KV). ~4 GiB of weights -> a 262k K8V3 cache fits
    one 16 GiB card."""

    def __init__(self, cfg: ModelConfig, sd: dict, kv_layers: list[int]):
        super().__init__()
        self.config = cfg
        self.embed_tokens = VocabEmbedding(
            sd["model.embed_tokens.weight"], out_dtype=cfg.act_dtype()
        )
        rope = RotaryEmbedding(
            cfg.resolved_head_dim(),
            cfg.max_position_embeddings,
            base=cfg.rope_theta,
            rotary_dim=cfg.rotary_dim(),
        )
        self.layers = nn.ModuleList(
            [FullAttnScopeDecoderLayer(cfg, i, sd, rope) for i in kv_layers]
        )
        self.norm = RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, sd["model.norm.weight"], add_unit_offset=True
        )
        lm_w = (
            sd["model.embed_tokens.weight"]
            if cfg.tie_word_embeddings
            else sd["lm_head.weight"]
        )
        from superl8serve.models.weights import to_qtensor

        self.lm_head = LMHead(to_qtensor(lm_w))

    def forward(self, input_ids, positions, ctx: ForwardContext):
        h = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            h, residual = layer(h, positions, ctx, residual)
        h, _ = self.norm(h, residual)
        return h

    def compute_logits(self, hidden):
        return self.lm_head(hidden)


def _move_to_device(model: nn.Module, device: str):
    """Move every parameter/buffer/QTensor-held weight to `device` (the standard
    path loads weights directly on-device; the reduced model builds CPU-first)."""
    for m in model.modules():
        for n, b in m._buffers.items():
            if b is not None and b.device.type != device:
                m._buffers[n] = b.to(device)
        for n, p in m._parameters.items():
            if p is not None and p.device.type != device:
                p.data = p.data.to(device)
        w = getattr(m, "weight", None)
        if isinstance(w, QTensor) and w.data.device.type != device:
            m.weight = QTensor(
                w.data.to(device),
                w.scale.to(device) if w.scale is not None else None,
                scheme=w.scheme,
                rotated=w.rotated,
                hadamard_dim=w.hadamard_dim,
                smoothed=w.smoothed,
                group_size=w.group_size,
                codebook=w.codebook,
            )
        if getattr(m, "bias", None) is not None and m.bias.device.type != device:
            m.bias = m.bias.to(device)
    return model


# ── load + build ─────────────────────────────────────────────────────────────


def _weights_to_device(weights: dict, device: str) -> dict:
    for k, v in weights.items():
        if isinstance(v, QTensor) and v.data.device.type != device:
            weights[k] = QTensor(
                v.data.to(device),
                v.scale.to(device) if v.scale is not None else None,
                scheme=v.scheme, rotated=v.rotated, hadamard_dim=v.hadamard_dim,
                smoothed=v.smoothed, group_size=v.group_size, codebook=v.codebook,
            )
        elif isinstance(v, torch.Tensor) and v.device.type != device:
            weights[k] = v.to(device)
    return weights


def _embedding_needs_transpose(embed, hidden_size: int) -> bool:
    """Dense Unsloth embeddings may be [hidden,vocab]; native QTensor is final."""
    return (
        isinstance(embed, torch.Tensor)
        and embed.dim() == 2
        and embed.shape[0] == hidden_size
    )


def _has_deep_mtp_head(weights: dict) -> bool:
    return any(k.startswith("model.mtp.") and k.endswith(".fc.weight") for k in weights)


def cache_capacity(ctx_tokens: int, max_tokens: int) -> int:
    if ctx_tokens < 1 or max_tokens < 1:
        raise ValueError("ctx_tokens and max_tokens must be greater than zero")
    return ctx_tokens + max_tokens


def _load_and_remap(cfg: ModelConfig, device: str):
    """GGUF state dict + the qwen3_5 hybrid remap chain (the `load_gguf_engine`
    prelude). Built CPU-first: the embedding is kept RESIDENT as int8 instead of
    the loader's fp16 dequant (an int8 embed table is 1.19 GiB vs 2.37 GiB fp16
    for a 248k-vocab 27B — the difference between fitting a 16 GiB card and not),
    and only the needed weights are moved to `device` afterward."""
    from superl8serve.gguf_native import _native_kquant_types, gguf_state_dict

    weights = gguf_state_dict(GGUF, device="cpu", native_types=_native_kquant_types())
    from superl8serve.gguf_native import (
        _remap_hybrid_qwen35,
        _restore_hf_qwen35_weights,
    )
    from superl8serve.convert import _remap_qwen3_next

    weights = _remap_hybrid_qwen35(weights, cfg)
    weights = _remap_qwen3_next(weights, cfg)
    weights, tiled = _restore_hf_qwen35_weights(weights, cfg)
    if tiled:
        cfg = dataclasses.replace(cfg, extra={**cfg.extra, "gguf_tiled_linear_attention": True})
    # qk_norm is detect-from-tensors (no GGUF KV for it) — same guard the loader
    # applies, or un-normalized Q/K goes into RoPE and attention is garbage.
    if not getattr(cfg, "qk_norm", False) and any(".q_norm.weight" in n for n in weights):
        cfg = dataclasses.replace(cfg, qk_norm=True)
    # Match load_gguf_engine: Qwen3.8 declares one nextn layer but this artifact
    # carries only the shallow shared head, not a buildable MTP decoder block.
    if cfg.num_mtp_layers > 0 and not _has_deep_mtp_head(weights):
        cfg = dataclasses.replace(cfg, num_mtp_layers=0)
    # Unsloth GGUF stores 2-D weights [in, out] instead of [out, in] — the same
    # loader heuristic: token_embd [hidden, vocab] (wrong) means transpose 2-D.
    _embed = weights.get("model.embed_tokens.weight")
    if _embedding_needs_transpose(_embed, cfg.hidden_size):
        for k, v in list(weights.items()):
            if isinstance(v, torch.Tensor) and v.dim() == 2 and ".mlp.experts." not in k:
                weights[k] = v.T.contiguous()
    if "model.embed_tokens.weight" in weights and not isinstance(
        weights["model.embed_tokens.weight"], QTensor
    ):
        from superl8.quant.core import quantize_int8_rowwise

        w = weights["model.embed_tokens.weight"].float()  # [vocab, hidden]
        q, s = quantize_int8_rowwise(w)
        weights["model.embed_tokens.weight"] = QTensor(
            q.contiguous(), s.squeeze(-1).float().contiguous(), scheme="per_row_i8"
        )
    if device != "cpu":
        _weights_to_device(weights, device)
    return cfg, weights


def build_full(cfg: ModelConfig, device: str, cache_format: str, max_len: int):
    """Build the full model through the production loader, not a copied prelude."""
    from superl8serve.gguf_native import load_gguf_engine

    return load_gguf_engine(
        GGUF,
        device=device,
        max_num_seqs=1,
        max_len=max_len,
        eos_id=None,
        spec_decode=False,
        cache_format=cache_format,
        chunked_prefill_size=0,
    )


def build_scope(cfg: ModelConfig, device: str):
    """Reduced 16-full-attn-layer model built CPU-first, only its weights moved."""
    cfg, weights = _load_and_remap(cfg, "cpu")
    model = FullAttnScopeModel(cfg, weights, KV_LAYERS)
    _move_to_device(model, device)
    del weights
    gc.collect()
    torch.cuda.empty_cache()
    return model, None


# ── needle prompt construction ───────────────────────────────────────────────


def build_prompt(tok, ctx_tokens: int) -> list[int]:
    """Haystack repeated to `ctx_tokens` (minus needle + question) with the needle
    planted at 50% depth. Returns the prompt token ids."""
    hay = tok.encode(HAYSTACK, add_special_tokens=False)
    needle = tok.encode(NEEDLE, add_special_tokens=False)
    q = tok.encode(QUESTION, add_special_tokens=False)
    budget = ctx_tokens - len(needle) - len(q)
    if budget <= 0:
        raise ValueError(f"ctx_tokens {ctx_tokens} too small")
    # Ceiling division is required: floor division silently undershoots the
    # advertised context by up to one whole haystack span.
    reps = max(1, (budget + len(hay) - 1) // len(hay))
    hay_tokens = (hay * reps)[:budget]
    half = len(hay_tokens) // 2
    return hay_tokens[:half] + needle + hay_tokens[half:] + q


# ── prefill + decode ─────────────────────────────────────────────────────────


def chunked_prefill(model, cache, lin_cache, slot, prompt_ids, device, chunk_size):
    """Bound prompt activations while preserving production prefill semantics.

    This mirrors :meth:`EngineRunner._prefill_chunked`: absolute positions and
    ``prefill_length`` advance monotonically, the paged cache remains the sole
    accumulated K/V store, and the recurrent cache stays bound across chunks.
    Only the final token's hidden state is retained for the caller.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    n = len(prompt_ids)
    cache.ensure_capacity([slot], [n])
    lin_cache.clear_slot(slot)
    lin_cache.bind([slot])
    hidden = None
    with torch.inference_mode():
        for chunk_start in range(0, n, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n)
            ids = torch.tensor([prompt_ids[chunk_start:chunk_end]], device=device)
            pos = torch.arange(chunk_start, chunk_end, device=device).unsqueeze(0)
            ctx = ForwardContext(
                is_prefill=True,
                kv_cache=cache,
                lin_cache=lin_cache,
                slots=[slot],
                prefill_start=0,
                prefill_length=chunk_end,
            )
            hidden = model(ids, pos, ctx)
    return hidden[:, -1:]


def decode_step(model, cache, lin_cache, slot, length, last_token, device):
    """One eager decode step: write the token's K/V (vectorized batch of 1) and
    attend over the whole context via the K8V3 dequant fallback / K8V8 fused kernel."""
    cache.ensure_capacity([slot], [length + 1])
    lin_cache.bind([slot])
    ids = torch.tensor([[last_token]], device=device)
    pos = torch.tensor([[length]], device=device)
    ctx = ForwardContext(
        is_prefill=False,
        kv_cache=cache,
        lin_cache=lin_cache,
        slots=[slot],
        slot_lengths=[length],
    )
    with torch.inference_mode():
        hidden = model(ids, pos, ctx)
        logits = model.compute_logits(hidden[:, -1])
    return int(logits.argmax().item())


def greedy_from_prefill(model, hidden) -> int:
    """Sample token@N from the prompt's final hidden, as EngineRunner.prefill does."""
    with torch.inference_mode():
        logits = model.compute_logits(hidden[:, -1])
    return int(logits.argmax().item())


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--format", choices=["k8v3", "k8v8"], default="k8v3")
    ap.add_argument("--layer-scope", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if args.chunk_size <= 0:
        ap.error("--chunk-size must be greater than zero")
    if args.ctx <= 0 or args.max_tokens <= 0:
        ap.error("--ctx and --max-tokens must be greater than zero")

    from transformers import AutoTokenizer

    device = "cuda"
    print(f"[needle] format={args.format} ctx={args.ctx} layer_scope={args.layer_scope} "
          f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    from superl8serve.gguf_native import gguf_config

    cfg = gguf_config(GGUF)
    if args.probe:
        if args.layer_scope:
            model, _ = build_scope(cfg, device)
        else:
            engine = build_full(cfg, device, "k8v3", 8192)
            model = engine.model
        torch.cuda.synchronize()
        print(json.dumps({
            "mode": "scope" if args.layer_scope else "full",
            "gpu_resident_gib": round(torch.cuda.memory_reserved() / 2**30, 2),
            "free_gib": round(torch.cuda.mem_get_info()[0] / 2**30, 2),
            "cfg": {
                "hidden": cfg.hidden_size, "layers": cfg.num_hidden_layers,
                "nkv": cfg.num_key_value_heads, "hd": cfg.resolved_head_dim(),
                "ctx": cfg.max_position_embeddings,
            },
        }))
        return

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    max_len = cache_capacity(args.ctx, args.max_tokens)
    if args.layer_scope:
        model, _ = build_scope(cfg, device)
        cache = PagedKVCache(
            64,
            1,
            NKV,
            max_len,
            HD,
            device=device,
            block_size=BS,
            kv_layers=KV_LAYERS,
            v_quant="lloydmax3" if args.format == "k8v3" else "int8",
        )
        lin_cache = RecurrentStateCache()
    else:
        engine = build_full(cfg, device, args.format, max_len)
        model, cache, lin_cache = engine.model, engine.cache, engine.lin_cache
    torch.cuda.synchronize()
    print(f"[needle] model built in {time.perf_counter() - t0:.1f}s; "
          f"resident={torch.cuda.memory_reserved() / 2**30:.2f} GiB", flush=True)

    prompt = build_prompt(tok, args.ctx)
    print(f"[needle] prompt tokens: {len(prompt)} (target {args.ctx})", flush=True)

    slot = cache.alloc()
    t0 = time.perf_counter()
    hidden = chunked_prefill(model, cache, lin_cache, slot, prompt, device, args.chunk_size)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0
    print(f"[needle] prefill done in {prefill_s:.1f}s; "
          f"resident={torch.cuda.memory_reserved() / 2**36 * 4:.2f} GiB "
          f"(alloc {torch.cuda.memory_allocated() / 2**30:.2f} GiB)", flush=True)

    t0 = time.perf_counter()
    first = greedy_from_prefill(model, hidden)
    out_toks = [first]
    length = len(prompt)
    last = first
    for _ in range(max(0, args.max_tokens - 1)):
        if last == tok.eos_token_id:
            break
        last = decode_step(model, cache, lin_cache, slot, length, last, device)
        length += 1
        out_toks.append(last)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    text = tok.decode(out_toks, skip_special_tokens=True)
    recall = ANSWER in text
    print(f"[needle] decode {len(out_toks)} tok in {elapsed:.1f}s "
          f"({len(out_toks) / max(elapsed, 1e-9):.2f} tok/s)")
    print(f"[needle] OUTPUT: {text!r}")
    print(f"[needle] RECALL: {recall}")

    result = {
        "ctx": args.ctx,
        "format": args.format,
        "layer_scope": args.layer_scope,
        "prompt_tokens": len(prompt),
        "output": text,
        "recall": recall,
        "decoded_tokens": len(out_toks),
        "prefill_s": round(prefill_s, 2),
        "gpu_alloc_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
        "gpu_reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2),
    }
    if args.json:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
