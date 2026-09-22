# Prototype, Weight-Sharing, and Codebook Spike Evidence

**Status:** Historical evidence index
**Governing issue:** [superl8-serve #335](https://git.python-bull.ts.net/superl8/superl8-serve/issues/335)
**Migration issue:** [superl8-serve #342](https://git.python-bull.ts.net/superl8/superl8-serve/issues/342)
**Current architecture:** [Weight-Stationary Runtime Architecture](weight-stationary-architecture.md)

This page preserves the dated experiment chain that retired shared prototypes, naive weight
sharing, and product-codebook representations from the production inference path. Historical
documents retain their original hypotheses so the correction is auditable; they are not current
implementation recommendations.

## Current Decision

The production direction is **W8A8 with per-row scales and a weight-stationary execution engine**.
The codebook/prototype inference direction is a measured **NO-GO**:

- scalar PQ: PPL **25.52 → 300.55**, KL **1.6061**, token accuracy **60%**;
- PQ index entropy: **7.0 bits/index**, correcting the initial, invalid 0.05-bit result;
- shared scale/codebook: PPL **21.22 → 8,094,770**;
- naive sharing: baseline PPL about **12** increased to **28–114 million**;
- cross-model prototypes: residuals remained full-sized, dense, and not low-rank.

## Dated Evidence Chain

| Date | Record | Immutable source | Disposition |
|------|--------|------------------|-------------|
| 2026-07-24 | [Shared prototype feasibility](shared-prototype-spike.md) | [control PR #169](https://git.python-bull.ts.net/content-factory/content-factory/pulls/169), [`ea54e1ec`](https://git.python-bull.ts.net/content-factory/content-factory/commit/ea54e1ecfdb362cca20d97d4a5c1e486f6c58229) | Cross-model prototype residuals were not sparse or low-rank and consumed the same storage as the prototype. Historical conditional language applies only to an untested base-plus-fine-tune-delta idea. |
| 2026-07-24 | [Initial PQ codebook spike](pq-codebook-spike.md) | [control PR #171](https://git.python-bull.ts.net/content-factory/content-factory/pulls/171), [`a73a45cd`](https://git.python-bull.ts.net/content-factory/content-factory/commit/a73a45cde76583c68d94910b461eba67d1e8f97f) | **Superseded and corrected.** It used cosine as a quality proxy, did not run inference, and reported an invalid 0.05 bits/index. Do not use its CONDITIONAL GO. |
| 2026-07-25 | [Rigorous PQ validation](pq-rigorous-spike.md) | [control PR #172](https://git.python-bull.ts.net/content-factory/content-factory/pulls/172), [`935a051a`](https://git.python-bull.ts.net/content-factory/content-factory/commit/935a051a0308ce67c20c028fd5a0d879b9528c8d) | **Canonical correction / NO-GO.** Exact entropy was 7.0 bits/index; inference quality failed at PPL 300.55, KL 1.6061, and 60% token accuracy. |
| 2026-07-24 | [Naive weight-sharing compression](weight-sharing-compression-spike.md) | [control PR #173](https://git.python-bull.ts.net/content-factory/content-factory/pulls/173), [`a979a2e4`](https://git.python-bull.ts.net/content-factory/content-factory/commit/a979a2e41aa7d8f25bf470cb86da7745718040ac) | **NO-GO.** Head averaging and four-layer sharing without retraining raised PPL from about 12 to millions. |

The table keeps the initial PQ claim adjacent to its correction; the independent #173 record has
the earlier stated date shown in its row. The commits were later applied to their source branches
on 2026-07-25; the immutable commit links above are the provenance authority.

## Independent Confirmation

The [per-row scale + shared codebook spike](scale-codebook-spike.md) independently tested a
256-centroid, eight-dimensional codebook across the full Qwen3-0.6B model. It measured PPL
**8,094,770** versus **21.22** for FP16, while raw per-row INT8 measured **21.19**. This result
reinforces the NO-GO and the current W8A8 direction.

The earlier [compression spike](compression-spike.md) remains evidence for its measured NF4
result. Its historical proposal to build a codebook phase is explicitly retired by the later
quality measurements above.

## Interpretation Rules

1. Later inference-quality measurements override earlier reconstruction proxies.
2. The #171 0.05-bit entropy claim is retained only as a documented error; **7.0 bits/index** is
   canonical.
3. “Conditional GO,” “next step,” and “what would need to change” sections in historical reports
   are not backlog items. New experiments require a new issue, current community/official
   research, and production quality gates.
4. The source control-plane PRs remain immutable provenance. This migration does not merge or
   close them.
