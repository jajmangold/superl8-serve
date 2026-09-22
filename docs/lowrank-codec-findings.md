# Low-rank ("VAE-ish") wire codec — findings

**Verdict: a tunable compression↔fidelity tradeoff codec, NOT rejected.** It loses to
`int8` at normal fidelity and wins only at extreme compression with degraded fidelity.
Two earlier verdicts were both wrong and are corrected here.

## What was wrong before

1. **"int8 strictly dominates" (iso-ratio framing)** — wrong axis. On a bandwidth-bound
   PCIe-1.0-x1 link the metric is *max compression at an acceptable fidelity*, not
   fidelity at a fixed ratio.
2. **"rejected / E2E-toxic" (earlier revision)** — tested a **strawman**: an inline
   pure-PCA projection with **zero raw channels**, not the production `LowRankCodec`
   (which keeps the top-magnitude channels raw). It also rode a **per-tensor latent-quant
   bug** (below). Both made the codec look far worse than it is.

## Bug fixed: per-channel latent quantization

The r-dim projection latent was int8-quantized with a **single per-tensor scale**. One
outlier latent dim dominated that scale and crushed all others, capping reconstruction
**SQNR flat at ~16 dB regardless of rank** — i.e. the low-rank projection bought nothing.
Fixed to **per-channel** int8 scales (one per latent dim, like per-row weight quant):

| rank | SQNR (per-tensor → per-channel) |
|---|---|
| 128 | 15.76 → **28.40 dB** |
| 400 | 15.76 → **42.28 dB** |
| 800 | 15.76 → **50.96 dB** |

Regression-guarded by `tests/test_lowrank_latent.py`.

## Measured LLM tradeoff (Qwen3-1.7B, single PP boundary, layer 14/28)

int8 baseline: **top-1 0.970 @ 2.0× (codec)**. Low-rank at r=400, per-channel latent:

| raw_fraction | codec-only | e2e top-1 |
|---|---|---|
| 0.5% | 9.75× | 0.614 |
| 5%   | 6.78× | 0.696 |
| 10%  | 5.07× | 0.788 |
| 20%  | 3.36× | 0.842 |
| 40%  | 2.01× | 0.885 |

Low-rank never reaches int8's 0.97 at iso-2×; it trades fidelity for compression.

## The full stack: codec × rANS (the decisive part)

The transport can stack a lossless rANS entropy stage on the quantized codes. **rANS
strongly favors int8, not low-rank** — int8 codes of real activations are low-entropy
(peaked/sparse) while the per-channel low-rank latent is high-entropy by design:

| codec | codec-only | +rANS | rANS gain |
|---|---|---|---|
| int8 | 2.0× | **4.27×** | 2.13× |
| low-rank raw=0.5% | 9.72× | 11.99× | 1.23× |
| low-rank raw=20% | 3.36× | 3.59× | 1.07× |

At the *stacked* wire ratio, **int8+rANS (4.27× @ top-1 0.97)** beats low-rank on both
axes until the extreme-compression corner (11.99× @ top-1 0.61).

## rANS is TOO SLOW as implemented — do not enable by default

The rANS stage is pure Python: **~8.5 s to encode+decode an 8.4 MB payload**, vs the
**~5 ms** of wire time it saves at 250 MB/s — ~1700× more cost than benefit. So the
"4.27×" above is theoretical: **`entropy=True` is a latency footgun; keep it OFF** until
rANS is reimplemented in C/CUDA. Practical fast defaults remain **int8 (2×)** and
**int4/int4-had (4×)**.

## Where low-rank could still win

Only when you need >4.3× AND can tolerate degraded fidelity. Whether a **DiT/image**
boundary tolerates it better than an LLM readout (whose next-token signal sits in the
low-variance tail low-rank discards) is a separate open test.
