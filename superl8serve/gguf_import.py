# SPDX-License-Identifier: MIT
"""GGUF (k-quant) -> `.superl8` importer.

**Why.** llama.cpp GGUF k-quants (Q3_K / Q4_K / Q5_K / Q6_K, and the Unsloth /
bartowski "Dynamic" mixes) ship imatrix-CALIBRATED weights for free — someone
already ran importance-matrix calibration and picked a per-tensor precision menu
(early `ffn_gate/up` at Q3_K, sensitive `ffn_down` + late layers at Q4_K/Q6_K).
This importer HARVESTS that menu instead of re-deriving imatrix: it dequantizes
each GGUF block to fp and **repacks into the resident `.superl8` dp4a layout**
(`per_group_w3a8` 3-bit bit-planes / `per_group_i4` / `per_row_i8`), mapping each
tensor's target precision from its SOURCE k-quant type. `.superl8` stays the runtime
format — GGUF's block layout does not feed dp4a, so we never run GGUF natively.

**Weight-only.** GGUF carries no activation scales; superl8 activations are quantized
dynamically per-row at runtime (`quantize_i8_rowwise`), so the W8A8 activation side
needs nothing stored here — the weights are the only calibrated artifact to harvest.

Prototype: standard Llama/Qwen tensor naming (dense + MoE experts). Requires the
`gguf` pip package (llama.cpp's own reader + block dequant — reliable vs
hand-rolling super-block layouts).
"""

from __future__ import annotations

import logging
import re

import numpy as np
import torch

from superl8 import QTensor
from superl8.format import save_superl8

from .convert import (
    is_quantizable_linear,
    quantize_weight_i4,
    quantize_weight_i8,
    quantize_weight_w3a8,
)

# ── GGUF ggml quant type -> target .superl8 precision (the harvested menu) ───────
# A source type's bit budget picks our scheme: 2/3-bit k-quants -> uniform 3-bit
# dp4a (per_group_w3a8); 4/5-bit -> int4; 6/8-bit -> int8; float -> raw/int8.
_W3_SRC = {"Q2_K", "Q3_K", "IQ3_S", "IQ3_XXS", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ1_S"}
_I4_SRC = {"Q4_K", "Q5_K", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "IQ4_NL", "IQ4_XS"}
_I8_SRC = {"Q6_K", "Q8_0", "Q8_1", "Q8_K"}
_FLOAT_SRC = {"F32", "F16", "BF16"}


def _target_scheme(src_type: str) -> str:
    if src_type in _W3_SRC:
        return "w3a8"
    if src_type in _I4_SRC:
        return "i4"
    if src_type in _I8_SRC:
        return "i8"
    if src_type in _FLOAT_SRC:
        return "float"
    return "i8"  # unknown / exotic -> safe int8


# ── GGUF tensor name -> HF name (standard llama.cpp block layout) ────────────
_STATIC = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}

_LFM2_STATIC = {
    **_STATIC,
    "token_embd_norm.weight": "model.embedding_norm.weight",
}

_LFM2_BLK = {
    "attn_norm": "operator_norm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.out_proj",
    "attn_q_norm": "self_attn.q_layernorm",
    "attn_k_norm": "self_attn.k_layernorm",
    "ffn_norm": "ffn_norm",
    "ffn_gate": "feed_forward.w1",
    "ffn_up": "feed_forward.w3",
    "ffn_down": "feed_forward.w2",
}
_BLK = {
    "attn_norm": "input_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
    "ffn_norm": "post_attention_layernorm",
    "post_attention_norm": "post_attention_layernorm",  # Qwen3.5 GGUF variant
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate_inp": "mlp.gate",  # MoE router (stays raw)
    "ffn_gate_exps": "mlp.experts.gate_proj",
    "ffn_up_exps": "mlp.experts.up_proj",
    "ffn_down_exps": "mlp.experts.down_proj",
    # shared-expert (DeepSeek/Qwen-MoE) — research/llamacpp-moe.md §2.2, constants.py:1189
    "ffn_gate_shexp": "mlp.shared_expert.gate_proj",
    "ffn_up_shexp": "mlp.shared_expert.up_proj",
    "ffn_down_shexp": "mlp.shared_expert.down_proj",
    "ffn_gate_inp_shexp": "mlp.shared_expert_gate",  # per-token sigmoid gate on shexp out
    "exp_probs_b": "mlp.gate.e_score_correction_bias",  # router bias (§2.3)
    # ── Qwen3.5 / hybrid DeltaNet SSM tensors (llama.cpp qwen35 layout) ──
    # DeltaNet linear-attention projections.  The GGUF converter uses the raw
    # ssm_* names from llama.cpp's qwen35 arch; the HF model stores them under
    # `linear_attn.in_proj_*`.  The `_la_weight` helper in the qwen3_5 builder
    # tries BOTH naming conventions (native + HF alias), so either mapping works.
    "ssm_alpha": "linear_attn.in_proj_a",  # dt gate (decay projection)
    "ssm_beta": "linear_attn.in_proj_b",  # beta (write gate)
    "ssm_out": "linear_attn.out_proj",  # output projection
    "ssm_conv1d": "linear_attn.conv1d",  # causal depthwise conv1d
    "ssm_norm": "linear_attn.norm",  # output gated RMSNorm gain
    # attn_qkv for DeltaNet layers → fused Q|K|V (NOT for full-attn layers,
    # which use separate attn_q/attn_k/attn_v — those are already in _BLK).
    "attn_qkv": "linear_attn.in_proj_qkv",  # fused Q|K|V projection
    # attn_gate: kept as-is (self_attn.attn_gate) — gguf_native._remap_hybrid_qwen35
    # handles the layer-type-dependent routing: DeltaNet → linear_attn.in_proj_a,
    # full-attn → fuse into self_attn.q_proj.
    "attn_gate": "self_attn.attn_gate",
}

# Bare parameters (no .weight/.bias suffix) from the SSM block.
_BARE_PARAMS = {
    "ssm_a": "linear_attn.A_log",  # log-decay parameter
    "ssm_dt": "linear_attn.dt_bias",  # dt bias
}

# MTP tensor prefixes (blk.32.nextn.*) — the llama.cpp Qwen3.5 MTP head layout.
# These carry a simplified MTP head (eh_proj + norms) without a full decoder block;
# the qwen3_5 builder's `build_qwen3_5_mtp` can use eh_proj as fc and the norms
# as pre_fc_norm_{hidden,embedding}.  The decoder-block weights (q/k/v/o/mlp) are
# absent in this GGUF — the builder gracefully returns None when they're missing,
# falling back to n-gram spec-decode.
_MTP = {
    "eh_proj": "mtp.fc",  # fc weight [2H, H]
    "enorm": "mtp.pre_fc_norm_hidden",  # hidden-state norm
    "hnorm": "mtp.pre_fc_norm_embedding",  # embedding norm
    "shared_head_norm": "mtp.norm",  # final norm before LM head
}


def _lfm2_name_to_hf(name: str) -> str | None:
    """Map llama.cpp's LFM2 layout onto the existing HF-layout LFM2 builder."""
    if name in _LFM2_STATIC:
        return _LFM2_STATIC[name]
    m = re.match(r"blk\.(\d+)\.shortconv\.(conv|in_proj|out_proj)\.(weight|bias)$", name)
    if m:
        layer, part, suffix = m.groups()
        return f"model.layers.{layer}.conv.{part}.{suffix}"
    m = re.match(r"blk\.(\d+)\.([a-z0-9_]+)\.(weight|bias)$", name)
    if m:
        layer, part, suffix = m.groups()
        if part in _LFM2_BLK:
            return f"model.layers.{layer}.{_LFM2_BLK[part]}.{suffix}"
    return None


def gguf_name_to_hf(name: str, *, arch: str | None = None) -> str | None:
    """Map a GGUF tensor name to its HF equivalent, or None if unrecognized."""
    if arch and arch.lower() == "lfm2":
        return _lfm2_name_to_hf(name)
    if name in _STATIC:
        return _STATIC[name]
    # MTP tensors: blk.{L}.nextn.{name}.{suffix}
    m = re.match(r"blk\.(\d+)\.nextn\.([a-z_]+)\.(weight|bias)$", name)
    if m:
        part = m.group(2)
        suffix = m.group(3)
        if part in _MTP:
            return f"{_MTP[part]}.{suffix}"
        return None
    # Bare parameters (no .weight/.bias): blk.{L}.{name}
    m = re.match(r"blk\.(\d+)\.(ssm_a)$", name)
    if m:
        layer, part = m.group(1), m.group(2)
        if part in _BARE_PARAMS:
            return f"model.layers.{layer}.{_BARE_PARAMS[part]}"
    # Parameters with .bias suffix that map to bare HF names (no suffix).
    # ssm_dt.bias → linear_attn.dt_bias (the HF model stores this as a bare
    # nn.Parameter, not as a `.bias` attribute).
    m = re.match(r"blk\.(\d+)\.(ssm_dt)\.bias$", name)
    if m:
        return f"model.layers.{m.group(1)}.linear_attn.dt_bias"
    # Standard block tensors: blk.{L}.{part}.{suffix}
    m = re.match(r"blk\.(\d+)\.([a-z0-9_]+)\.(weight|bias)$", name)
    if m:
        layer, part, suffix = m.group(1), m.group(2), m.group(3)
        if part in _BLK:
            return f"model.layers.{layer}.{_BLK[part]}.{suffix}"
    return None


def read_gguf_weights(path: str):
    """Yield (hf_name, weight fp32 torch.Tensor [out,in], src_qtype_name) for every
    mappable tensor. Dequantizes each block via llama.cpp's own dequant."""
    from gguf import GGUFReader, dequantize
    from gguf.constants import GGMLQuantizationType as T

    reader = GGUFReader(path)
    arch_field = reader.fields.get("general.architecture")
    arch = arch_field.contents() if arch_field is not None else None
    for t in reader.tensors:
        hf = gguf_name_to_hf(t.name, arch=arch)
        if hf is None:
            continue
        qtype = T(t.tensor_type)
        deq = dequantize(t.data, qtype).astype(np.float32)  # numpy [.. , in]
        w = torch.from_numpy(np.ascontiguousarray(deq))
        # GGUF stores 2-D weights as [out, in] already (row-major, same as HF).
        yield hf, w, qtype.name


def gguf_state_to_qtensors(
    path: str, *, group_size: int = 128, force_mlp_w3a8: bool = False
) -> tuple[dict[str, QTensor], dict[str, str]]:
    """Import a GGUF into a dict[name, QTensor] using the harvested per-tensor menu.

    Returns (qtensors, source_type_map). `force_mlp_w3a8` overrides the source menu
    for MLP gate/up to 3-bit (to build the issue #181 VRAM-lever variant even from a
    uniformly-int4 GGUF). `ffn_down` and non-MLP tensors keep their harvested scheme.
    """
    out: dict[str, QTensor] = {}
    src_map: dict[str, str] = {}
    for name, w, src_type in read_gguf_weights(path):
        src_map[name] = src_type
        quantizable = is_quantizable_linear(name) and w.dim() == 2 and w.shape[-1] % 4 == 0
        if not quantizable:
            out[name] = QTensor(w.half(), None, scheme="raw")
            continue
        target = _target_scheme(src_type)
        grouped = w.shape[-1] % group_size == 0
        is_gate_up = name.endswith((".gate_proj.weight", ".up_proj.weight"))
        if force_mlp_w3a8 and is_gate_up and grouped:
            target = "w3a8"
        if target == "w3a8" and grouped:
            out[name] = quantize_weight_w3a8(w, group_size)
        elif target == "i4" and grouped:
            out[name] = quantize_weight_i4(w, group_size)
        else:
            out[name] = quantize_weight_i8(w)
    return out, src_map


def convert_gguf_to_superl8(
    gguf_path: str,
    out_path: str,
    *,
    group_size: int = 128,
    force_mlp_w3a8: bool = False,
    meta: dict | None = None,
) -> dict[str, str]:
    """Read a GGUF k-quant checkpoint and write a resident `.superl8`. Returns the
    per-tensor source-type map (for provenance / audit)."""
    qtensors, src_map = gguf_state_to_qtensors(
        gguf_path, group_size=group_size, force_mlp_w3a8=force_mlp_w3a8
    )
    save_superl8(
        out_path,
        qtensors,
        meta={"source": "gguf", "gguf_path": gguf_path, "source_types": src_map, **(meta or {})},
    )
    return src_map


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Import a GGUF k-quant checkpoint to .superl8")
    ap.add_argument("gguf_path")
    ap.add_argument("out_path")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument(
        "--force-mlp-w3a8",
        action="store_true",
        help="override the source menu: MLP gate/up -> uniform 3-bit dp4a "
        "(the issue #181 VRAM lever), down/others keep harvested scheme",
    )
    args = ap.parse_args()
    src_map = convert_gguf_to_superl8(
        args.gguf_path,
        args.out_path,
        group_size=args.group,
        force_mlp_w3a8=args.force_mlp_w3a8,
    )
    from collections import Counter

    logging.info("wrote %s", args.out_path)
    logging.info("source k-quant types harvested: %s", dict(Counter(src_map.values())))


if __name__ == "__main__":
    main()
