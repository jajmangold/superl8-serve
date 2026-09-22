# K8V3 Decode for Qwen3.5-9B — Agent Findings

**Date:** 2026-09-15  
**Issue:** superl8-serve#468 (follows up on #464)  
**Branch:** `agent/k8v3-decode-kernel-9b`

---

## 1. Kernel Location and Status

### Findings

The fused K8V3 paged decode kernel **already exists** in the superl8 repository:

- **File:** `/path/to/storage
- **Function:** `launch_paged_decode_k8v3()` / `attn_paged_decode_k8v3()`
- **Line:** ~237 in `attn_paged_decode.cu`
- **API:** `at::Tensor attn_paged_decode_k8v3(...)` (see `/path/to/storage

The kernel is **NOT YET COMPILED** into the superl8 PyTorch extension. The implementation:
- Accepts int8 K cache + 3-bit Lloyd-Max packed V
- Consumes V norms and codebook in-kernel (no fp16 V materialization)
- Handles mixed-length sequences via `max_context_len` parameter (no host sync required)
- Supports head dims 128 and 256 (required for Qwen3.5-9B)

### What's Missing

1. **Kernel compilation** — The superl8 extension must be recompiled to include `attn_paged_decode_k8v3`
2. **Python wrapper validation** — The wrapper in `superl8/ops.py` needs the fix to accept `max_context_len`

---

## 2. Code Analysis

### Problem: Host Sync Breaking CUDA Graph Capture

Both the fused kernel path and the fallback path have `.item()` calls that trigger device→host sync:

#### In `superl8/ops.py` (fused kernel path)
```python
# Line 872 (original)
if max_context_len is None:
    max_context_len = int(context_lens.max().item())  # ❌ HOST SYNC
```

#### In `superl8-serve/superl8serve/engine/kv_cache.py` (fallback path)
```python
# Line 679 (original)
n = int(context_lens[b].item())  # ❌ HOST SYNC
```

### Solution

#### 1. Fused Kernel Wrapper Fix (`superl8/ops.py`)
Made `max_context_len` **required** for the fused kernel path:
```python
if max_context_len is None:
    raise ValueError("attn_paged_decode_k8v3 requires max_context_len to be passed to avoid host sync")
```

**Rationale:** The kernel already accepts `max_context_len` as a parameter and uses it as an upper bound for split sizing. Callers must provide a safe upper bound (e.g., from prefill bucket tracking).

#### 2. Fallback Path Fix (`kv_cache.py`)
Added optional `max_context_len` parameter to `_decode_lloydmax3()`:
```python
def _decode_lloydmax3(self, layer: int, q: torch.Tensor, block_table, context_lens, 
                       *, scale, max_context_len: int | None = None):
    if max_context_len is not None:
        n = int(max_context_len)  # ✅ NO SYNC
    else:
        n = int(context_lens[b].item())  # Fallback only
```

#### 3. Call Site Updates
`decode_attn_static()` now passes `max_context_len` to both:
- The fused `attn_paged_decode_k8v3()` (line 1179)
- The fallback `_decode_lloydmax3()` (line 1189)

---

## 3. Python-Side Wiring Changes

### Changes Made

| File | Change |
|------|--------|
| `superl8/ops.py` | Made `max_context_len` required, added error for missing param |
| `superl8serve/engine/kv_cache.py` | Added `max_context_len` param to `_decode_lloydmax3()`, pass it from call site |
| `bench/k8v3_decode_verify.sh` | New verification script |

### Wiring Contract

When the fused kernel is available:
1. `PagedKVCache.decode_attn()` calls `decode_attn_static()` with `max_context_len`
2. `_can_fused_k8v3()` checks: `v_quant == "lloydmax3"` + kernel callable + head dim ∈ {128, 256}
3. Fused kernel is called with `max_context_len` (upper bound for split sizing)
4. If kernel fails, fallback to `_decode_lloydmax3()` with the same `max_context_len`

---

## 4. Verification Script

**Location:** `bench/k8v3_decode_verify.sh`

**Usage:**
```bash
./bench/k8v3_decode_verify.sh \
    --model /path/to/qwen35-9b.superl8 \
    --tokenizer Qwen/Qwen3.5-9B \
    --cache-format k8v3 \
    --buckets 1,2,4,8,16
```

**What it does:**
1. Loads Qwen3.5-9B with `cache_format='k8v3'`
2. Runs prefill + decode under CUDA graph capture
3. Tests multiple bucket sizes (1, 2, 4, 8, ...)
4. Runs 3+ reps per bucket
5. Confirms no crash (crash would indicate graph capture incompatibility)

**Expected output per bucket:**
```
[bucket=4 rep=0] wall=0.12s  out_tokens=100  decode_tok/s=500.0  CAPTURE OK
[bucket=4 rep=1] wall=0.11s  out_tokens=100  decode_tok/s=550.0  CAPTURE OK
[bucket=4 rep=2] wall=0.11s  out_tokens=100  decode_tok/s=545.0  CAPTURE OK
```

---

## 5. What's Left for Human/GPU Pass

### To Complete This PR

1. **Compile superl8 extension** with the existing `attn_paged_decode_k8v3` kernel:
   ```bash
   cd /path/to/storage
   docker compose run --rm build
   ```

2. **Build superl8-serve** to pick up the compiled extension:
   ```bash
   cd /root/work/repo/superl8-serve
   pip install -e ".[kernels]"
   ```

3. **Run verification:**
   ```bash
   ./bench/k8v3_decode_verify.sh \
       --model /path/to/qwen35-9b.superl8 \
       --tokenizer Qwen/Qwen3.5-9B \
       --buckets 1,2,4,8
   ```

4. **Quality gate** (per k8v3-qwen38-tq34s-wiring.md):
   - 262k retrieval / needle-in-haystack eval
   - Per-layer KV SQNR vs fp32 V oracle (cos ≥ 0.999, SQNR ≥ 40 dB)
   - Quiet-GPU decode bench (report 44–47 tok/s on CMP 100-210)

5. **Create PR** with:
   - This Python wiring PR (no kernel changes needed)
   - A follow-up PR in superl8 to add correctness tests for `attn_paged_decode_k8v3`
   - Performance benchmarks vs. fallback path

### Notes

- The kernel is **already implemented** and **just needs compilation**
- The Python changes are **minimal** and **graph-safe**
- Qwen3.5-9B head dim is **128**, which the kernel supports
- The fallback path still requires `.item()` for edge cases (acceptable for non-graph mode)

---

## 6. Summary

**Status:** Python-side wiring complete; kernel compilation remaining.

**Blocks for merge:** superl8 extension must be recompiled with the existing `attn_paged_decode_k8v3` kernel.

**Next step:** Compile superl8, run verification, file PR.
