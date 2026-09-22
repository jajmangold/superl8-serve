# SPDX-License-Identifier: MIT
"""K8V3 KV-cache wiring tests (superl8-serve#410) — CPU-only.

Covers the superl8-serve side of the K8V3 contract: the int8-K + 3-bit Lloyd-Max V
cache layout, its codec round-trip quality, per-layer/per-slot shape + memory
accounting, and the KV-bearing-layer mapping for the qwen3_5 hybrid (48 DeltaNet
+ 16 full-attention layers — only the 16 bear KV).

Correctness bars (do not weaken): int8 K round-trips at the int8 codec gate
(SQNR >= 35 dB / cos >= 0.999); 3-bit V round-trips at the documented 3-bit
Lloyd-Max quality (the 8-entry codebook is theoretically capped ~14-15 dB on
unit-normal — the attention/retrieval gate is the needle eval, not a looser
codec bar). The `attn_int8_decode` decode path needs CUDA; the needle eval on a
pinned free GPU is the runtime gate.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("superl8")

import superl8  # noqa: E402

from superl8.quant.lloydmax import (  # noqa: E402
    dequantize_lloydmax,
    pack_indices_lowbit,
    quantize_lloydmax,
    unpack_indices_lowbit,
)
from superl8serve.engine.kv_cache import KVEviction, PagedKVCache  # noqa: E402

# Qwen3.8-27B geometry (the 16 full-attention layers of the qwen3_5 hybrid).
HD = 256
NKV = 4
BLOCK = 16
LLOYD_BLOCK = 128
# The 16 full-attn layers at full_attention_interval == 4 in a 64-layer stack.
KV_LAYERS = [i for i in range(64) if (i + 1) % 4 == 0]


def _sqnr(ref, q):
    ref, q = ref.double().flatten(), q.double().flatten()
    err = ref - q
    den = ref.norm()
    return float(20.0 * torch.log10(den / (err.norm() + 1e-12))) if den > 0 else float("inf")


def _cos(ref, q):
    ref, q = ref.double().flatten(), q.double().flatten()
    return float((ref @ q) / (ref.norm() * q.norm() + 1e-12))


def _v_gauss(b, h, s, d, *, outlier=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(b, h, s, d, generator=g, dtype=torch.float16)
    if outlier:
        chan = torch.zeros(d, dtype=torch.float16)
        chan[:: d // 8] = 12.0
        v = v + chan
    return v


# ── 1. codec round-trip (reference `superl8.quant.lloydmax`, reused verbatim) ────


class TestK8V3Codec:
    def test_v3_roundtrip_shape_and_dtype(self):
        v = _v_gauss(2, NKV, 8, HD)
        codes, norm, cb = quantize_lloydmax(v, bits=3, block_size=LLOYD_BLOCK, dim=-1)
        assert codes.dtype == torch.uint8 and codes.shape == v.shape
        assert norm.shape == (*v.shape[:2], v.shape[2], HD // LLOYD_BLOCK)
        assert cb.shape == (8,)
        packed = pack_indices_lowbit(codes, 3)
        assert packed.shape[-1] == HD * 3 // 32
        codes2 = unpack_indices_lowbit(packed, 3, HD)
        assert codes2.shape == v.shape
        deq = dequantize_lloydmax(codes2, norm, cb, block_size=LLOYD_BLOCK, dim=-1)
        assert deq.shape == v.shape and deq.dtype == torch.float32

    def test_v3_pack_is_lossless(self):
        v = _v_gauss(2, NKV, 8, HD, seed=3)
        codes, _, cb = quantize_lloydmax(v, bits=3, block_size=LLOYD_BLOCK, dim=-1)
        codes2 = unpack_indices_lowbit(pack_indices_lowbit(codes, 3), 3, HD)
        assert torch.equal(codes, codes2)

    def test_v3_roundtrip_quality_on_gaussian(self):
        """3-bit Lloyd-Max on Gaussian V: the 8-entry codebook's theoretical ceiling
        is ~14-15 dB SQNR; assert the honest measured codec quality (cos >= 0.97,
        SQNR >= 12 dB) — the retrieval gate is the needle eval, not this bar."""
        v = _v_gauss(2, NKV, 16, HD, seed=5)
        codes, norm, cb = quantize_lloydmax(v, bits=3, block_size=LLOYD_BLOCK, dim=-1)
        deq = dequantize_lloydmax(codes, norm, cb, block_size=LLOYD_BLOCK, dim=-1)
        assert _cos(v, deq) >= 0.97, f"V cos {_cos(v, deq):.5f} < 0.97"
        assert _sqnr(v, deq) >= 12.0, f"V SQNR {_sqnr(v, deq):.1f} dB < 12 dB"

    def test_v3_zero_block(self):
        """A zero-norm block stays at the codebook's center entry scaled by norm 1 —
        the reference codec's exact round-trip (no NaN, bounded error, matches
        `fake_quant_lloydmax` on the same input)."""
        from superl8.quant.lloydmax import fake_quant_lloydmax

        v = torch.zeros(1, NKV, 4, HD, dtype=torch.float16)
        v[:, :, 1] = 1.0
        codes, norm, cb = quantize_lloydmax(v, bits=3, block_size=LLOYD_BLOCK, dim=-1)
        deq = dequantize_lloydmax(codes, norm, cb, block_size=LLOYD_BLOCK, dim=-1)
        ref = fake_quant_lloydmax(v, bits=3, block_size=LLOYD_BLOCK)
        # The reference casts back to fp16 (v's dtype), so allow fp16 rounding.
        assert torch.allclose(deq.float(), ref.float(), atol=2e-3)
        assert torch.isfinite(deq).all()

    def test_k8_roundtrip_quality(self):
        """int8 K (the K side of K8V3) round-trips at the int8 gate."""
        k = _v_gauss(2, NKV, 8, HD, seed=7)
        k_rot = superl8.quant.rotation.rotate_last(k)
        k_i8, k_sc = superl8.quantize_i8_rowwise(k_rot)
        deq = superl8.quant.dequantize_int8_rowwise(k_i8, k_sc.unsqueeze(-1))
        assert _cos(k_rot, deq) >= 0.999, f"K cos {_cos(k_rot, deq):.6f} < 0.999"
        assert _sqnr(k_rot, deq) >= 35.0, f"K SQNR {_sqnr(k_rot, deq):.1f} dB < 35 dB"


# ── 2. K8V3 paged-cache wiring (CPU) ─────────────────────────────────────────


class TestK8V3Cache:
    def _cache(self, v_quant="lloydmax3", max_len=4096, num_slots=2, kv_layers=None):
        return PagedKVCache(
            64,
            num_slots,
            NKV,
            max_len,
            HD,
            device="cpu",
            block_size=BLOCK,
            kv_layers=kv_layers,
            v_quant=v_quant,
        )

    def test_layer_map_and_storage_rows(self):
        c = self._cache(kv_layers=KV_LAYERS)
        assert c.num_layers == 16  # 64-layer hybrid → 16 KV-bearing storage rows
        assert c.k_cache.shape[0] == c.v_packed.shape[0] == 16
        assert c._layer_row(3) == 0 and c._layer_row(63) == 15
        c_identity = self._cache(kv_layers=None)
        assert c_identity.num_layers == 64 and c_identity._layer_row(63) == 63

    def test_v_storage_shapes(self):
        c = self._cache(kv_layers=KV_LAYERS, max_len=BLOCK * 2, num_slots=1)
        assert c.v_packed.shape == (16, 2, NKV, BLOCK, HD * 3 // 32)
        assert c.v_packed.dtype == torch.int32
        assert c.v_norm.shape == (16, 2, NKV, BLOCK, HD // LLOYD_BLOCK)
        assert c.v_codebook.shape == (16, 8)
        assert c.v_cache is None and c.v_scale is None
        c_i8 = self._cache(v_quant="int8", kv_layers=KV_LAYERS)
        assert c_i8.v_cache is not None and c_i8.v_scale is not None

    def test_fused_decode_static_is_one_batched_call(self, monkeypatch):
        c = self._cache(kv_layers=KV_LAYERS, max_len=64, num_slots=2)
        q = torch.randn(2, 8, 1, HD, dtype=torch.float16)
        block_table = torch.tensor([[3, 2, 0, 0], [1, 0, 0, 0]], dtype=torch.int32)
        context_lens = torch.tensor([33, 5], dtype=torch.int32)
        expected = torch.randn_like(q)
        calls = []

        monkeypatch.setattr(c, "_can_fused_k8v3", lambda _q: True)

        def fused(*args, **kwargs):
            calls.append((args, kwargs))
            return expected

        monkeypatch.setattr(superl8, "attn_paged_decode_k8v3", fused, raising=False)
        out = c.decode_attn_static(
            63, q, block_table, context_lens, 33, scale=0.125
        )

        assert out is expected
        assert len(calls) == 1
        args, kwargs = calls[0]
        assert args[0] is q
        assert args[1].data_ptr() == c.k_cache[15].data_ptr()
        assert args[3].data_ptr() == c.v_packed[15].data_ptr()
        assert args[4].data_ptr() == c.v_norm[15].data_ptr()
        assert args[5].data_ptr() == c.v_codebook[15].data_ptr()
        assert args[6] is block_table and args[7] is context_lens
        assert kwargs == {"max_context_len": 33, "scale": 0.125}

    def test_fused_decode_error_disables_retry_and_uses_dense(self, monkeypatch):
        c = self._cache(max_len=64, num_slots=1)
        q = torch.randn(1, 4, 1, HD, dtype=torch.float16)
        block_table = torch.zeros(1, 4, dtype=torch.int32)
        context_lens = torch.tensor([5], dtype=torch.int32)
        expected = torch.randn_like(q)
        calls = {"fused": 0, "dense": 0}

        monkeypatch.setattr(c, "_can_fused_k8v3", lambda _q: not c._k8v3_fused_failed)

        def fused(*_args, **_kwargs):
            calls["fused"] += 1
            raise RuntimeError("older superl8 ABI")

        def dense(*_args, **_kwargs):
            calls["dense"] += 1
            return expected

        monkeypatch.setattr(superl8, "attn_paged_decode_k8v3", fused, raising=False)
        monkeypatch.setattr(c, "_decode_lloydmax3", dense)
        for _ in range(2):
            assert c.decode_attn_static(
                0, q, block_table, context_lens, 5, scale=0.125
            ) is expected
        assert calls == {"fused": 1, "dense": 2}

    def test_fused_decode_env_rollback_and_cpu_fallback(self, monkeypatch):
        monkeypatch.setenv("SUPERL8SERVE_K8V3_FUSED", "0")
        disabled = self._cache(max_len=64, num_slots=1)
        assert not disabled._can_fused_k8v3(torch.empty(1, 4, 1, HD))

        monkeypatch.setenv("SUPERL8SERVE_K8V3_FUSED", "1")
        cpu = self._cache(max_len=64, num_slots=1)
        monkeypatch.setattr(superl8, "attn_paged_decode_k8v3", lambda: None, raising=False)
        assert not cpu._can_fused_k8v3(torch.empty(1, 4, 1, HD))

        class FakeCudaQuery:
            is_cuda = True

        monkeypatch.delattr(superl8, "attn_paged_decode_k8v3", raising=False)
        assert not cpu._can_fused_k8v3(FakeCudaQuery())

        monkeypatch.setattr(superl8, "attn_paged_decode_k8v3", lambda: None, raising=False)
        cpu.head_dim = 64
        assert not cpu._can_fused_k8v3(FakeCudaQuery())
        cpu.head_dim = HD
        cpu.lloyd_block = 64
        assert not cpu._can_fused_k8v3(FakeCudaQuery())

    @pytest.mark.parametrize("head_dim", [128, 256])
    @pytest.mark.skipif(
        not torch.cuda.is_available()
        or not callable(getattr(superl8, "attn_paged_decode_k8v3", None)),
        reason="requires the superl8#295 CUDA extension",
    )
    def test_fused_decode_cuda_matches_dense_fallback(self, head_dim):
        device = "cuda"
        c = PagedKVCache(
            1, 2, NKV, 64, head_dim, device=device, block_size=BLOCK,
            v_quant="lloydmax3",
        )
        slots = [c.alloc(), c.alloc()]
        lengths = [33, 64]
        for seed, (slot, length) in enumerate(zip(slots, lengths)):
            c.ensure_capacity([slot], [length])
            torch.manual_seed(seed)
            k = torch.randn(1, NKV, length, head_dim, dtype=torch.float16, device=device)
            v = torch.randn_like(k)
            c.write_prefill(0, k, v, slot=slot)

        q = torch.randn(2, 8, 1, head_dim, dtype=torch.float16, device=device)
        block_table = c.block_table(slots)
        context_lens = torch.tensor(lengths, dtype=torch.int32, device=device)
        scale = head_dim**-0.5
        fused = c.decode_attn_static(
            0, q, block_table, context_lens, max(lengths), scale=scale
        )
        assert not c._k8v3_fused_failed
        c._k8v3_fused_enabled = False
        dense = c.decode_attn_static(
            0, q, block_table, context_lens, max(lengths), scale=scale
        )
        assert _cos(fused, dense) >= 0.99

    def test_prefill_roundtrip_vs_oracle(self):
        """write_prefill + read_dense reconstructs fp16 K/V at the codec quality
        bars: K at the int8 gate, V at the 3-bit Lloyd-Max gate."""
        c = self._cache(kv_layers=KV_LAYERS, max_len=64, num_slots=1)
        slot = c.alloc()
        c.ensure_capacity([slot], [16])
        k = torch.randn(1, NKV, 16, HD, dtype=torch.float16)
        v = _v_gauss(1, NKV, 16, HD, seed=9)
        layer = 63  # real layer index → storage row 15
        c.write_prefill(layer, k, v, slot=slot)
        kd, vd = c.read_dense(layer, slot, 16)
        assert kd.shape == vd.shape == (1, NKV, 16, HD)
        assert _cos(k, kd) >= 0.999, f"K cos {_cos(k, kd):.6f}"
        assert _sqnr(k, kd) >= 35.0, f"K SQNR {_sqnr(k, kd):.1f} dB"
        assert _cos(v, vd) >= 0.97, f"V cos {_cos(v, vd):.5f}"
        assert _sqnr(v, vd) >= 12.0, f"V SQNR {_sqnr(v, vd):.1f} dB"
        kd_bf16, vd_bf16 = c.read_dense(layer, slot, 16, dtype=torch.bfloat16)
        assert kd_bf16.dtype == vd_bf16.dtype == torch.bfloat16

    def test_read_dense_bounds_lloydmax_unpack_by_token_chunk(self, monkeypatch):
        """Long-prefix reconstruction must never expand every packed 3-bit V
        code through the int64 bit-unpack temporary at once."""
        import superl8.quant.lloydmax as lloydmax

        seq_len = BLOCK * 65
        c = self._cache(kv_layers=KV_LAYERS, max_len=seq_len, num_slots=1)
        slot = c.alloc()
        c.ensure_capacity([slot], [seq_len])
        k = torch.randn(1, NKV, seq_len, HD, dtype=torch.float16)
        v = _v_gauss(1, NKV, seq_len, HD, seed=31)
        layer = 63
        c.write_prefill(layer, k, v, slot=slot)

        unpack_sizes = []
        unpack = lloydmax.unpack_indices_lowbit

        def capture(packed, bits, d):
            unpack_sizes.append(packed.shape[0])
            return unpack(packed, bits, d)

        monkeypatch.setattr(lloydmax, "unpack_indices_lowbit", capture)
        kd, vd = c.read_dense(layer, slot, seq_len)

        assert unpack_sizes == [1024, 16]
        assert kd.shape == vd.shape == (1, NKV, seq_len, HD)
        assert _cos(k, kd) >= 0.999
        assert _sqnr(v, vd) >= 12.0

        unpack_sizes.clear()
        kw, vw = c.read_dense(layer, slot, seq_len, window=1025)
        assert unpack_sizes == [1024, 1]
        assert torch.equal(kw, kd[:, :, -1025:, :])
        assert torch.equal(vw, vd[:, :, -1025:, :])

    def test_decode_fallback_bounds_private_lloydmax_unpack(self, monkeypatch):
        """The env/operator rollback must not bypass the bounded dense reader."""
        import superl8.quant.lloydmax as lloydmax

        seq_len = BLOCK * 65
        c = self._cache(kv_layers=KV_LAYERS, max_len=seq_len, num_slots=1)
        slot = c.alloc()
        c.ensure_capacity([slot], [seq_len])
        k = torch.randn(1, NKV, seq_len, HD, dtype=torch.float16)
        v = _v_gauss(1, NKV, seq_len, HD, seed=32)
        c.write_prefill(63, k, v, slot=slot)
        expected_k, expected_v = c.read_dense(63, slot, seq_len)

        unpack_sizes = []
        unpack = lloydmax.unpack_indices_lowbit
        seen = {}

        def capture(packed, bits, d):
            unpack_sizes.append(packed.shape[0])
            return unpack(packed, bits, d)

        q = torch.randn(1, 8, 1, HD, dtype=torch.float16)
        monkeypatch.setattr(lloydmax, "unpack_indices_lowbit", capture)

        def attention(query, cached_k, cached_v, **_kwargs):
            seen["k"], seen["v"] = cached_k, cached_v
            return query

        monkeypatch.setattr(superl8, "attn_int8_decode", attention)
        out = c._decode_lloydmax3(
            63,
            q,
            c.block_table([slot]),
            torch.tensor([seq_len], dtype=torch.int32),
            scale=HD**-0.5,
        )

        assert torch.equal(out, q)
        assert unpack_sizes == [1024, 16]
        assert torch.equal(seen["k"], expected_k)
        assert torch.equal(seen["v"], expected_v)

    def test_read_dense_bounds_int8_kv_by_token_chunk(self):
        seq_len = BLOCK * 65
        c = self._cache(v_quant="int8", max_len=seq_len, num_slots=1)
        slot = c.alloc()
        c.ensure_capacity([slot], [seq_len])
        k = torch.randn(1, NKV, seq_len, HD, dtype=torch.float16)
        v = torch.randn_like(k)
        mapping = torch.tensor(c.flat_slot_mapping(slot, seq_len), dtype=torch.int32)
        k_tokens = k.permute(0, 2, 1, 3).reshape(seq_len, NKV, HD)
        v_tokens = v.permute(0, 2, 1, 3).reshape(seq_len, NKV, HD)
        c._write_k_int8(3, k_tokens, mapping)
        v_i8, v_sc = superl8.quantize_i8_rowwise(v_tokens.reshape(-1, HD))
        rows = c._flat_rows(mapping)
        c.v_cache[3].reshape(-1, HD)[rows] = v_i8
        c.v_scale[3].reshape(-1)[rows] = v_sc

        kd, vd = c.read_dense(3, slot, seq_len)

        assert kd.shape == vd.shape == (1, NKV, seq_len, HD)
        assert _cos(k, kd) >= 0.999
        assert _cos(v, vd) >= 0.999

    @pytest.mark.parametrize("position_shape", ["flat", "batched"])
    def test_prefill_batches_nonzero_positions_across_blocks(self, monkeypatch, position_shape):
        """Chunked prefill must build one vector slot mapping without one write/sync
        per token.  The selected range crosses a physical block boundary and skips
        positions below ``start`` (shared-prefix reuse)."""
        c = self._cache(v_quant="int8", max_len=64, num_slots=1)
        slot = c.alloc()
        c.ensure_capacity([slot], [20])
        positions = torch.arange(12, 20)
        if position_shape == "batched":
            positions = positions.unsqueeze(0)
        k = torch.arange(8 * NKV * HD, dtype=torch.float16).view(1, NKV, 8, HD)
        v = -k
        calls = []

        def capture(layer, slot_mapping, k_new, v_new):
            calls.append((layer, slot_mapping.clone(), k_new.clone(), v_new.clone()))

        monkeypatch.setattr(c, "write_prefill_varlen", capture)
        c.write_prefill(3, k, v, slot=slot, start=14, positions=positions)

        assert len(calls) == 1
        layer, mapping, k_new, v_new = calls[0]
        assert layer == 3
        assert mapping.tolist() == c.flat_slot_mapping(slot, 20)[14:20]
        expected_k = k[:, :, 2:, :].permute(0, 2, 1, 3).reshape(6, NKV, HD)
        expected_v = v[:, :, 2:, :].permute(0, 2, 1, 3).reshape(6, NKV, HD)
        assert torch.equal(k_new, expected_k)
        assert torch.equal(v_new, expected_v)

    def test_prefill_batches_implicit_positions_after_prefix(self, monkeypatch):
        c = self._cache(v_quant="int8", max_len=64, num_slots=1)
        slot = c.alloc()
        c.ensure_capacity([slot], [20])
        k = torch.randn(1, NKV, 20, HD, dtype=torch.float16)
        v = torch.randn_like(k)
        calls = []
        monkeypatch.setattr(
            c,
            "write_prefill_varlen",
            lambda layer, mapping, k_new, v_new: calls.append((mapping, k_new, v_new)),
        )

        c.write_prefill(3, k, v, slot=slot, start=14)

        assert len(calls) == 1
        mapping, k_new, v_new = calls[0]
        assert mapping.tolist() == c.flat_slot_mapping(slot, 20)[14:]
        assert k_new.shape == v_new.shape == (6, NKV, HD)

    def test_prefill_explicit_start_zero_batches_once(self, monkeypatch):
        c = self._cache(v_quant="int8", max_len=64, num_slots=1)
        slot = c.alloc()
        c.ensure_capacity([slot], [8])
        k = torch.randn(1, NKV, 8, HD, dtype=torch.float16)
        v = torch.randn_like(k)
        calls = []
        monkeypatch.setattr(
            c,
            "write_prefill_varlen",
            lambda layer, mapping, k_new, v_new: calls.append(mapping.clone()),
        )

        c.write_prefill(3, k, v, slot=slot, positions=torch.arange(8).unsqueeze(0))

        assert len(calls) == 1
        assert calls[0].tolist() == c.flat_slot_mapping(slot, 8)

    @pytest.mark.parametrize("explicit_positions", [False, True])
    def test_prefill_all_prefix_is_noop(self, monkeypatch, explicit_positions):
        c = self._cache(v_quant="int8", max_len=64, num_slots=1)
        slot = c.alloc()
        c.ensure_capacity([slot], [8])
        k = torch.randn(1, NKV, 8, HD, dtype=torch.float16)
        v = torch.randn_like(k)
        monkeypatch.setattr(
            c,
            "write_prefill_varlen",
            lambda *args, **kwargs: pytest.fail("all-prefix prefill must not write"),
        )
        positions = torch.arange(8) if explicit_positions else None

        c.write_prefill(3, k, v, slot=slot, start=8, positions=positions)

    def test_prefill_empty_is_noop(self, monkeypatch):
        c = self._cache(v_quant="int8", max_len=64, num_slots=1)
        slot = c.alloc()
        k = torch.empty(1, NKV, 0, HD, dtype=torch.float16)
        v = torch.empty_like(k)
        monkeypatch.setattr(
            c,
            "write_prefill_varlen",
            lambda *args, **kwargs: pytest.fail("empty prefill must not write"),
        )

        c.write_prefill(3, k, v, slot=slot, positions=torch.empty(0, dtype=torch.long))

    def test_k_is_format_independent(self):
        """The K8V3 K write is byte-identical to the fused kernel's K path (the
        `kv_write_paged` recipe: Hadamard rotation + per-row RTN int8 with fp32
        scale) — the needle A/B isolates the V format. The CUDA-only kernel can't
        run on CPU, so the oracle is the same rotate+quantize the kernel applies,
        checked against the storage block the slot actually received (blocks are
        allocated LIFO from the shared pool)."""
        c = self._cache(v_quant="lloydmax3", kv_layers=KV_LAYERS, max_len=64, num_slots=1)
        c.alloc()
        c.ensure_capacity([0], [16])
        k = torch.randn(1, NKV, 16, HD, dtype=torch.float16)
        v = _v_gauss(1, NKV, 16, HD, seed=11)
        c.write_prefill(3, k, v, slot=0)
        blk = c._slot_blocks[0][0]  # the block this slot actually got
        assert blk == c.num_blocks - 1  # LIFO pool: first slot gets the last block
        k_rot = superl8.quant.rotation.rotate_last(k)
        k_i8, _ = superl8.quantize_i8_rowwise(k_rot)
        assert torch.equal(c.k_cache[0, blk], k_i8[0].to(torch.int8))
        kd, _ = c.read_dense(3, 0, 16)
        assert _cos(k, kd) >= 0.999  # K read back at the int8 gate

    def test_write_decode_roundtrip(self):
        c = self._cache(kv_layers=KV_LAYERS, max_len=64, num_slots=1)
        slot = c.alloc()
        c.ensure_capacity([slot], [8])
        layer = 7
        for t in range(8):
            k = torch.randn(1, NKV, HD, dtype=torch.float16)
            v = _v_gauss(1, NKV, 1, HD, seed=t).squeeze(2)
            c.write_decode(layer, [slot], [t], k, v)
        kd, vd = c.read_dense(layer, slot, 8)
        assert kd.shape == vd.shape == (1, NKV, 8, HD)

    def test_varlen_prefill_roundtrip(self):
        c = self._cache(kv_layers=KV_LAYERS, max_len=64, num_slots=1)
        slot = c.alloc()
        c.ensure_capacity([slot], [8])
        sm = torch.tensor(c.flat_slot_mapping(slot, 8), dtype=torch.int32)
        k = torch.randn(8, NKV, HD, dtype=torch.float16)
        v = _v_gauss(1, NKV, 8, HD, seed=13).squeeze(0).permute(1, 0, 2).contiguous()
        c.write_prefill_varlen(3, sm, k, v)
        kd, vd = c.read_dense(3, slot, 8)
        assert vd.squeeze(0).permute(1, 0, 2).shape == v.shape
        assert _cos(v, vd.squeeze(0).permute(1, 0, 2)) >= 0.97, f"varlen V cos {_cos(v, vd.squeeze(0).permute(1, 0, 2)):.5f}"

    def test_prefix_bytes_honors_v_quant(self):
        c = self._cache(v_quant="lloydmax3", kv_layers=KV_LAYERS, max_len=64, num_slots=1)
        c.alloc()
        c.ensure_capacity([0], [16])
        c.store_prefix([1] * 16, 0)
        i8 = self._cache(v_quant="int8", kv_layers=KV_LAYERS, max_len=64, num_slots=1)
        i8.alloc()
        i8.ensure_capacity([0], [16])
        i8.store_prefix([1] * 16, 0)
        assert c.pinned_prefix_blocks >= 1
        assert c.prefix_block_bytes() < i8.prefix_block_bytes()

    def test_eviction_guarded_on_k8v3(self):
        c = self._cache(kv_layers=KV_LAYERS, max_len=64, num_slots=1)
        slot = c.alloc()
        c.ensure_capacity([slot], [32])
        k = torch.randn(1, NKV, 32, HD, dtype=torch.float16)
        v = _v_gauss(1, NKV, 32, HD, seed=17)
        c.write_prefill(3, k, v, slot=slot)
        with pytest.raises(NotImplementedError, match="lloydmax3"):
            KVEviction.compact(c, slot, [0, 4, 8, 12, 16, 20, 24, 28])


# ── 3. per-slot memory accounting (the 262k envelope) ────────────────────────


class TestK8V3Memory:
    def test_block_bytes_k8v3_vs_int8(self):
        bb_v3 = KVEviction.block_bytes(NKV, BLOCK, HD, v_quant="lloydmax3")
        bb_i8 = KVEviction.block_bytes(NKV, BLOCK, HD, v_quant="int8")
        # K8V3 block: 4 heads * 16 tok * (256 K + 4 K-scale + 96 V codes + 8 V norms)
        assert bb_v3 == NKV * BLOCK * (HD + 4 + HD * 3 // 8 + (HD // LLOYD_BLOCK) * 4)
        assert bb_v3 < bb_i8  # 3-bit V is strictly smaller than int8 V

    def test_262k_single_slot_envelope(self):
        """K8V3 at the 262k native context over the 16 full-attn layers:
        ~6.1 GB K+V (K ~4.36 GB int8+scales, V ~1.74 GB 3-bit+norms)."""
        tokens = 262144
        bb = KVEviction.block_bytes(NKV, BLOCK, HD, v_quant="lloydmax3")
        total = 16 * (tokens // BLOCK) * bb  # 16 KV-bearing layers
        per_slot = total / 2**30
        assert 5.5 <= per_slot <= 6.5, f"262k K8V3 per-slot {per_slot:.2f} GiB out of envelope"

    def test_two_slots_under_tp2_card_budget(self):
        """TP=2 pair at 262k, 2 slots: per card = half the model (13.68 GB native ->
        ~6.84 GB) + the slot's KV for ITS half of the heads. The K8V3 16-layer
        envelope must keep 2 slots comfortably inside a 16 GiB card."""
        tokens = 262144
        bb = KVEviction.block_bytes(NKV, BLOCK, HD, v_quant="lloydmax3")
        total_kv = 16 * (tokens // BLOCK) * bb / 2**30  # 6.1 GiB both slots both cards
        per_card_slots = total_kv  # TP=2: each card holds half the KV of 2 slots
        model_half = 13.68 / 2  # half the TQ3_4S resident weights per card
        budget = model_half + per_card_slots
        assert budget <= 16.0, f"TP=2 2-slot 262k per-card {budget:.2f} GB > 16 GB"

    def test_storage_bytes_match_accounting(self):
        c = PagedKVCache(
            64,
            1,
            NKV,
            BLOCK,
            HD,
            device="cpu",
            block_size=BLOCK,
            kv_layers=KV_LAYERS,
            v_quant="lloydmax3",
        )
        bb = KVEviction.block_bytes(NKV, BLOCK, HD, v_quant="lloydmax3")
        allocated = c.k_cache.numel() + c.k_scale.numel() * 4
        allocated += c.v_packed.numel() * 4 + c.v_norm.numel() * 4
        assert allocated == c.num_layers * bb
