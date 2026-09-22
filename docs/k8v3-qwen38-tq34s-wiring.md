# K8V3 wiring plan — Qwen3.8-27B TQ3_4S (superl8-serve#408, work item 6)

Status: **PLAN** (cache implementation is a follow-up that needs the fused kernel,
superl8#272). The native TQ3_4S weight path ships in this PR; the K8V3 cache ships as
a separate PR once the kernel contract lands.

## Why K8V3, and where

`Qwen3.8-27B-MTP-TQ3_4S.gguf` is a 64-layer hybrid: **48 Gated-DeltaNet layers**
(no KV — recurrent state only, `deltanet_*` kernels) + **16 full-attention layers**
(the ones at `full_attention_interval == 4`, plus the MTP block which uses a
dense fp16 draft attention and never touches the main KV). Only the 16 full-attn
layers carry a K/V cache, so their cache is the only memory/traffic lever.

At the 262k native context (`qwen35.context_length = 262144`) a full fp16 K/V
would not fit a 16 GiB card next to the 13.68 GB resident model. **K8V3** (int8 K +
3-bit Lloyd-Max V) is the chosen layout:

| layout | K bytes | V bytes | 262k KV (16 layers) |
|---|---|---|---|
| fp16 K/V | 2 | 2 | ~11.8 GB |
| int8 K + int8 V | 1 | 1 | 5.91 GB |
| **K8V3** (int8 K + 3-bit V) | 1 | 0.375 | **~4.2 GB** |

The 5.91 GB int8 line is the reference design-doc number (superl8-tq34s-fusion.md
§TP); K8V3 cuts the V side further so a TP=2 pair holds 2× 262k slots comfortably
(2.95 → ~2.1 GB/card V+K at the 16-layer scope). V is the dominant decode read
(the K is reused across query heads via attention weights), so 3-bit V also cuts
decode KV bandwidth ~35% on top of the memory win.

## Codec (exists, reference)

`superl8.quant.lloydmax` (per-block norm + fitted 8-entry codebook + 3-bit packed
indices) and `superl8.quant.lowbit` (bit-plane pack/unpack) ship the pure-torch
codec. The reference pack/unpack pair is `superl8.quantize_kv_cache_3bit` /
`dequantize_kv_cache_3bit` in `superl8/ops.py`. The per-channel V-quant primitive
(`quantize_v_perchannel`) lives in `superl8.quant.core`.

K8V3 differs from that reference only in the K side: K stays **int8 per-token**
(the proven `quantize_kv_write_paged` path) while **V goes 3-bit Lloyd-Max** —
so the fused decode kernel has to mix two dequant paths (int8 K with fp32
per-block scale; 3-bit V with per-block fp32 norm + codebook). The current
`attn_int8_decode_kv8` (int8 K + per-channel int8 V) is the closest existing
kernel; the K8V3 kernel extends its V side to the lowbit codec.

## Wiring seam (superl8-serve, future PR — kernel-gated)

1. `superl8serve/engine/kv_cache.py` — `PagedKVCache` keeps the int8 K/V block pool
   (KV, k_scale, v_scale). The V8V3 variant swaps `v_cache` for a packed 3-bit
   V buffer `(num_layers, n_blocks, nkv, block_size, (head_dim*3)//8)` plus the
   per-block fp32 V norms and the (per-layer) fitted codebook. Select it per
   model via a `cache_format="k8v3"` knob on the engine/model-runner, applied
   ONLY to the 16 full-attention layers; DeltaNet layers are untouched (recurrent
   state, no KV).
2. `superl8serve/layers/gqa_attention.py` (the full-attn decode path) — route the
   decode through the new `superl8.attn_decode_k8v3` op once the kernel lands, with
   an fp32-dequant fallback (accuracy gate) until then. fp16/int8 stays for
   short-context/local layers if the gate demands it.
3. `superl8serve/engine/staging.py` — V write path: quantize V per-channel
   (`quantize_v_perchannel` for int8 baseline; Lloyd-Max fit + pack for 3-bit)
   on store, mirroring the existing K/V write.
4. Memory accounting in `PagedKVCache.block_bytes` / the scheduler's
   admission logic must key off the active cache format so slot admission,
   prefix cache, and the TUI memory gauge stay honest.

## Quality gate (mandatory)

3-bit V sits on the 16 **long-range** full-attention layers, so perplexity alone
does not catch V-coverage failures. The K8V3 PR must pass:

- **262k retrieval / needle-in-haystack eval** (the operator gate, mirroring the
  design doc acceptance item 5), not just PPL — the 16 full-attn layers carry the
  long-range recall.
- Per-layer KV SQNR vs the fp32 V oracle (`cos >= 0.999` / SQNR >= 40 dB style,
  same convention as the TQ3_4S weight gate), on the exact `Qwen3.8-27B` split.
- A quiet-GPU decode bench with the checkpoint, precision, graph mode, batch,
  device, and fleet caveat (hard rule), reporting the 44–47 tok/s envelope.

## Dependencies / sequencing

- superl8#271 (format/loader contract + `superl8.quant.tq34s`) — required for the
  weight path; merged first.
- superl8#272 (fused `gemm_tq34s_dp4a` + `gemm_decode_tq34s`) — required for
  single-card residency (13.68 GB native); until it merges, TQ3_4S loads
  dp4a-correct at 1 B/wt (too big for one card) — the explicit non-goal for now.
- K8V3 cache: a superl8 kernel PR (lowbit-V decode over int8-K) + a superl8-serve
  wiring PR (steps 1–4). Opened when the weight path is green end-to-end.
