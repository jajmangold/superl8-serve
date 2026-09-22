# SPDX-License-Identifier: MIT
"""HF checkpoint -> `.superl8` conversion.

Quantizes the linear weights (attention + MLP + experts + untied LM head) to int8
`per_row_i8` or 4-bit `per_group_i4`, and keeps the numerically load-bearing
tensors (all norms, embeddings, and the MoE router gate) in fp16 `raw`. Weights are
stored under their HF names; the model builders merge q/k/v and gate/up at load
(cheap concat, valid per-row). The on-disk bytes are the resident dp4a layout, so
serving loads with no dequant/repack (`superl8.format`).

`quantize_state_dict` is the file-free core (tested directly). `convert_hf_to_superl8`
wraps it with config.json + safetensors IO.
"""

from __future__ import annotations

import json
import os

import torch

from superl8 import QTensor
from superl8.quant.core import quantize_int8_rowwise
from superl8.quant.lowbit import quantize_lowbit

from .models.config import ModelConfig

# A weight is a quantizable linear iff its name ends with one of these. The MoE
# router (`mlp.gate.weight`) ends with `.gate.weight` (NOT `.gate_proj.weight`), so
# it correctly stays fp16.
# DENYLIST, not an allowlist: quantize EVERY 2-D matmul weight (the caller also
# checks `w.dim()==2 and w.shape[-1]%4==0`) EXCEPT the numerically-sensitive /
# non-matmul ones below. A name allowlist silently skipped any model with
# non-standard naming (LFM2 `feed_forward.w1/2/3`, other MoE/hybrid layouts),
# leaving the bulk of the model fp16 -> "4-bit" files nearly fp16-sized. For the
# standard Llama/Qwen names this denylist quantizes the identical set (q/k/v/o,
# gate/up/down, lm_head); it just no longer misses everything else.
#   - `embed`/`wte`/`wpe`: token embeddings are an index LOOKUP, not a matmul
#     (the serve embedding layer reads them as fp) -> keep raw. (Tied lm_head is
#     quantized separately on the serve side.)
#   - `norm`: layernorm/rmsnorm gains (tiny, load-bearing).
#   - MoE `router`/`.gate.weight`/`.wg.weight`: routing logits are sensitive and
#     tiny (NOT the MLP `gate_proj`, which stays quantizable).
_QUANT_DENY_SUBSTR = (
    "norm",
    "embed",
    "wte",
    "wpe",
    "rotary",
    "router",
    "relative_attention",
    "conv_weight",
)
_QUANT_DENY_SUFFIX = (".gate.weight", ".router.weight", ".wg.weight")


def is_quantizable_linear(name: str) -> bool:
    n = name.lower()
    if any(s in n for s in _QUANT_DENY_SUBSTR):
        return False
    return not n.endswith(_QUANT_DENY_SUFFIX)


def quantize_weight_i8(w: torch.Tensor) -> QTensor:
    q, s = quantize_int8_rowwise(w)
    return QTensor(q.contiguous(), s.squeeze(-1).float().contiguous(), scheme="per_row_i8")


def quantize_weight_i4(w: torch.Tensor, group_size: int) -> QTensor:
    codes, scale = quantize_lowbit(w, 4, dim=-1, group_size=group_size)  # [O,I], [O,I//g]
    c = codes.to(torch.int64)
    packed = ((c[:, 0::2] & 0xF) | ((c[:, 1::2] & 0xF) << 4)).to(torch.uint8)
    return QTensor(
        packed.contiguous(),
        scale.float().contiguous(),
        scheme="per_group_i4",
        group_size=group_size,
        codebook="int4",
    )


def quantize_weight_w3a8(w: torch.Tensor, group_size: int) -> QTensor:
    """Uniform 3-bit dp4a WEIGHTS (`per_group_w3a8`, issue #181): full asymmetric
    [-4,3] range, per-group absmax/4 scale, Q3_K-style bit-planes -> int32
    [O,(I//32)*3] (3.0 bit/wt). The `gemm_decode_w3a8` kernel unpacks each 32-value
    group -> int8 and dp4a's it at decode PARITY with int4 (not faster) while storing
    0.75x the bytes — a VRAM / context / batch lever (frees KV context / batch slots
    before OOM). `group_size` must be a multiple of 32. Lossier than int4 by design,
    so it is opt-in and only for the MLP gate/up; down_proj keeps its base precision."""
    from superl8.quant.lowbit import pack_w3a8_bitplanes, quantize_w3a8

    codes, scale = quantize_w3a8(w, group_size=group_size)          # [O,I] int8, [O,I//g]
    return QTensor(
        pack_w3a8_bitplanes(codes).contiguous(),                   # int32 [O,(I//32)*3]
        scale.float().contiguous(),
        scheme="per_group_w3a8",
        group_size=group_size,
        codebook="w3a8",
    )


# MLP gate/up projections — the bulk of the streamed decode weights. The 3-bit MLP
# lever quantizes ONLY these (early layers); down_proj + late layers keep the base
# precision (int4/int8) for quality, per the imatrix map (utils/docs).
_MLP_GATE_UP_SUFFIX = (".gate_proj.weight", ".up_proj.weight",
                       ".linear_fc1.weight", ".w1.weight", ".w3.weight")


def _is_mlp_gate_up(name: str) -> bool:
    return name.endswith(_MLP_GATE_UP_SUFFIX) and is_quantizable_linear(name)


_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_inv", ".input_scale")


def quantize_state_dict(
    sd: dict,
    *,
    weight_bits: int = 8,
    group_size: int = 128,
    mlp_gate_up_3bit: bool = False,
    fp8_scales: dict | None = None,
    fp8_block_size: list | None = None,
    fp8_scale_fmt: str | None = None,
) -> dict:
    """HF state dict (fp16/bf16/**fp8**) -> dict[name, QTensor]. Linears quantized
    from full precision; everything else stored raw fp16 (fp32 only if fp16 would
    overflow — see _raw_dtype).

    FP8 checkpoints (DeepSeek/Hy3): each `float8` weight is first reconstructed to
    fp32 via its companion scale (`fp8_scales`, pre-collected so it works even when
    the scale lives in another shard), then quantized as usual. Scale tensors
    themselves are consumed, not emitted. `fp8_scale_fmt="ue8m0"` (microscaling /
    MXFP8 checkpoints, e.g. DeepSeek-V4-Flash, MiniMax-M3-MXFP8) selects the
    power-of-2 MX dequant (`dequantize_mxfp8`) instead of the fp32-multiply block
    dequant, and additionally drops the MX-specific scale-tensor names (`.scale`,
    `.hc_attn_scale`, `.hc_ffn_scale`, `.hc_head_scale`) — gated on `fp8_scale_fmt`
    being set (i.e. only for a checkpoint already known to be fp8-sourced) so a
    plain `.scale`-suffixed tensor in an unrelated fp16/bf16 model is never
    silently dropped."""
    from .fp8 import MX_SCALE_SUFFIXES, dequantize_fp8, dequantize_mxfp8, fp8_scale_name, is_fp8

    fp8_scales = fp8_scales or {}
    out: dict[str, QTensor] = {}
    for name, w in sd.items():
        if name.endswith(_SCALE_SUFFIXES):
            continue  # consumed by its weight / activation scale
        if fp8_scale_fmt is not None and name.endswith(MX_SCALE_SUFFIXES):
            continue  # consumed MX/UE8M0 companion scale
        w = w.detach().cpu()  # keep NATIVE dtype (don't truncate bf16)
        if is_fp8(w):  # reconstruct fp32 from fp8 * scale
            sname = fp8_scale_name(name, set(fp8_scales) | set(sd))
            sc = fp8_scales.get(sname) if sname else (sd.get(sname) if sname else None)
            if sc is None:
                w = w.float()  # no scale found -> best-effort raw upcast
            elif fp8_scale_fmt == "ue8m0":
                w = dequantize_mxfp8(w, sc.detach().cpu(), block_size=fp8_block_size or (128, 128))
            else:
                w = dequantize_fp8(w, sc.detach().cpu(), block_size=fp8_block_size)
        if is_quantizable_linear(name) and w.dim() == 2 and w.shape[-1] % 4 == 0:
            wf = w.float()  # quantize from full precision
            grouped = w.shape[-1] % group_size == 0
            # Opt-in issue #181 VRAM lever: MLP gate/up -> uniform 3-bit dp4a
            # (decode-parity, 0.75x int4 bytes). down_proj + everything else keep
            # the base precision for quality (per the imatrix mixed-precision map).
            if mlp_gate_up_3bit and grouped and _is_mlp_gate_up(name):
                out[name] = quantize_weight_w3a8(wf, group_size)
            elif weight_bits == 4 and grouped:
                out[name] = quantize_weight_i4(wf, group_size)
            else:
                out[name] = quantize_weight_i8(wf)
        else:
            out[name] = QTensor(_raw_dtype(w), None, scheme="raw")
    return out


def _raw_dtype(w: torch.Tensor) -> torch.Tensor:
    """Store a raw passthrough tensor (norm/embedding/router) as fp16 — which has MORE
    mantissa than bf16, only less range — and upcast to fp32 ONLY if fp16 would
    overflow a finite value. bf16-native models (Gemma, most DiTs) can carry
    out-of-fp16-range tensors that otherwise become inf -> NaN / black output."""
    w16 = w.to(torch.float16)
    if torch.isinf(w16).any() and not torch.isinf(w.float()).any():
        return w.float()
    return w16


_QWEN3_NEXT_ARCHS = frozenset(
    {
        "qwen3_next",
        "qwen3next",
        "Qwen3NextForCausalLM",
        "qwen3_5_vl",
        "Qwen3_5VLForConditionalGeneration",
        # Qwen3.5-9B (dense hybrid) ships as a VLM wrapper; text backbone reuses the
        # same fused linear-attn tensor layout, so it needs the same in_proj remap.
        "qwen3_5",
        "qwen3.5",
        "Qwen3_5ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
    }
)


def _remap_qwen3_next(sd: dict, cfg: ModelConfig) -> dict:
    """Rename HF fused tensor names to the per-name tensors the builder expects
    for the Qwen3-Next / Qwen3.5-VL hybrid linear-attention layers.

    For VLM checkpoints (Qwen3.5-VL) the text weights live under
    ``model.language_model.*`` and the vision tower under ``model.visual.*``;
    those prefixes are stripped so the qwen3_next text builder can bind.

    HF stores (linear-attention layers):
      * ``in_proj_qkvz.weight``  — fused [qk+qk+nv*vd+nv*vd, H]
      * ``in_proj_ba.weight``     — fused [nv+nv, H]
      * ``conv1d.weight``         — [Wc, K]

    Builder expects:
      * ``qkv_proj.weight``  — first 3/4 of in_proj_qkvz  (QKV fused)
      * ``z_proj.weight``    — last 1/4 of in_proj_qkvz   (output gate)
      * ``beta_proj.weight`` — first half of in_proj_ba
      * ``dt_proj.weight``   — second half of in_proj_ba
      * ``conv_weight``      — conv1d.weight unchanged
    """
    # ── VLM prefix stripping (Qwen3.5-VL nests text under model.language_model.*,
    #    visual under model.visual.*).  Detect by key presence so it works
    #    regardless of whether the user passes --arch or auto-detects. ──
    if any(k.startswith("model.language_model.") for k in sd):
        out = {}
        for name, w in sd.items():
            if name.startswith("model.language_model."):
                out["model." + name[len("model.language_model.") :]] = w
            elif name.startswith("model.visual."):
                out[name[len("model.") :]] = w
            else:
                out[name] = w
        sd = out

    if cfg.arch not in _QWEN3_NEXT_ARCHS:
        return sd

    x = cfg.extra
    nk = x.get("linear_num_key_heads", 0)
    nv = x.get("linear_num_value_heads", 0)
    kd = x.get("linear_key_head_dim", 0)
    vd = x.get("linear_value_head_dim", 0)
    if not (nk and nv and kd and vd):
        return sd

    qk = nk * kd
    v_dim = nv * vd
    out = {}
    for name, w in sd.items():
        if ".linear_attn.in_proj_qkvz.weight" in name:
            prefix = name.replace(".linear_attn.in_proj_qkvz.weight", "")
            # Weight layout: [q, k, v, z] stacked on row (output) dim
            qkv_w = w[: qk + qk + v_dim]
            z_w = w[qk + qk + v_dim :]
            out[f"{prefix}.linear_attn.qkv_proj.weight"] = qkv_w
            out[f"{prefix}.linear_attn.z_proj.weight"] = z_w
        elif ".linear_attn.in_proj_ba.weight" in name:
            prefix = name.replace(".linear_attn.in_proj_ba.weight", "")
            out[f"{prefix}.linear_attn.beta_proj.weight"] = w[:nv]
            out[f"{prefix}.linear_attn.dt_proj.weight"] = w[nv:]
        elif ".linear_attn.conv1d.weight" in name:
            prefix = name.replace(".linear_attn.conv1d.weight", "")
            # Trim to match qkv output dim (builder conv applies only to QKV, not Z),
            # then squeeze the depthwise in-channel dim: HF stores Conv1d weight as
            # [Wc, 1, K]; the builder buffer is [Wc, K] (it re-adds the 1 via unsqueeze).
            qkv_dim = qk + qk + v_dim
            cw = w[:qkv_dim]
            out[f"{prefix}.linear_attn.conv_weight"] = cw.squeeze(1) if cw.dim() == 3 else cw
        # Qwen3.5 stores the DeltaNet projections already-SEPARATE (not fused like
        # Qwen3-Next's in_proj_qkvz / in_proj_ba) — a rename, not a split. Mapping
        # matches the fused convention above: b -> beta_proj, a -> dt_proj.
        elif ".linear_attn.in_proj_qkv.weight" in name:
            out[name.replace(".linear_attn.in_proj_qkv.weight", ".linear_attn.qkv_proj.weight")] = w
        elif ".linear_attn.in_proj_z.weight" in name:
            out[name.replace(".linear_attn.in_proj_z.weight", ".linear_attn.z_proj.weight")] = w
        elif ".linear_attn.in_proj_b.weight" in name:
            out[name.replace(".linear_attn.in_proj_b.weight", ".linear_attn.beta_proj.weight")] = w
        elif ".linear_attn.in_proj_a.weight" in name:
            out[name.replace(".linear_attn.in_proj_a.weight", ".linear_attn.dt_proj.weight")] = w
        elif name.endswith(".linear_attn.conv_weight"):
            # Qwen3.5 stores the depthwise conv as [Wc, 1, K]; buffer wants [Wc, K].
            out[name] = w.squeeze(1) if w.dim() == 3 else w
        else:
            out[name] = w
    return out


def convert_hf_to_superl8(
    hf_dir: str,
    out_path: str,
    *,
    weight_bits: int = 8,
    group_size: int = 128,
    arch: str | None = None,
    delete_source: bool = False,
) -> ModelConfig:
    """Read an HF model dir (config.json + *.safetensors) and write a `.superl8`.
    Returns the ModelConfig (also embedded in the file meta).

    Shards are streamed one at a time — load, quantize, free the raw shard's RAM,
    next — so peak RAM is the quantized-so-far dict plus a single raw shard, never
    the full fp16/bf16 model (the 220GB-model-in-RAM OOM that killed GLM-4.5-Air
    mid-batch). Each HF tensor lives wholly in one shard, and q/k/v & gate/up stay
    unmerged on disk (the model builders merge them at load time, see
    superl8serve/models/weights.py), so per-shard quantization needs no cross-shard state.

    `delete_source` (default False) — when True, each source `.safetensors` shard is
    `os.remove`d from `hf_dir` the moment its tensors are quantized and written, so
    peak *disk* is max(source, consumed+output) rather than source+output. Required to
    fit 850GB+ models (MiniMax-M3, issue #69) on a disk that can't hold source+output
    at once, so the automated pipeline (`tools/forge.py`, which downloads to a
    disposable staging dir) passes it. It defaults to False because it DESTROYS the
    input checkpoint: the documented CLI points at a user's own model dir (issue #200),
    and a converter must never delete its input unless explicitly told to
    (`--free-source-shards`)."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    with open(os.path.join(hf_dir, "config.json")) as f:
        hf_cfg = json.load(f)
    cfg = ModelConfig.from_hf(hf_cfg, arch=arch)
    cfg.weight_bits = weight_bits

    shards = sorted(f for f in os.listdir(hf_dir) if f.endswith(".safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors in {hf_dir}")

    # FP8 source? Pre-collect the (tiny) scale tensors across ALL shards first, so a
    # weight can be dequantized even when its scale lives in a different shard.
    qcfg = hf_cfg.get("quantization_config") or {}
    is_fp8_src = str(qcfg.get("quant_method", "")).lower() == "fp8"
    fp8_block_size = qcfg.get("weight_block_size") if is_fp8_src else None
    # "ue8m0" (UE8M0 block scales) marks a microscaling (MX) FP8 checkpoint, e.g.
    # DeepSeek-V4-Flash / MiniMax-M3-MXFP8 — see fp8.dequantize_mxfp8.
    fp8_scale_fmt = str(qcfg.get("scale_fmt", "")).lower() if is_fp8_src else None
    fp8_scales: dict = {}
    if is_fp8_src:
        from .fp8 import MX_SCALE_SUFFIXES

        scale_suffixes = _SCALE_SUFFIXES + MX_SCALE_SUFFIXES
        for shard in shards:
            with safe_open(os.path.join(hf_dir, shard), framework="pt") as f:
                for k in f.keys():
                    if k.endswith(scale_suffixes):
                        fp8_scales[k] = f.get_tensor(k)

    # Stream to disk one shard at a time: quantize a shard, write its tensors, free
    # them, next shard. Peak RAM is a single shard (~a few GB), NOT the whole
    # quantized model -- so 160GB+ models (DeepSeek-V4-Flash, MiniMax-M3) convert
    # without OOM. (Accumulating the full `qsd` then save_superl8-ing it needed the
    # entire model resident, which SIGKILL'd the converter on the 256GB flavor.)
    from superl8.format import FQWriter

    with FQWriter(out_path) as w:
        for shard in shards:
            shard_path = os.path.join(hf_dir, shard)
            try:
                sd = load_file(shard_path)
                sd = _remap_qwen3_next(sd, cfg)
                # Detect q_norm/k_norm weights in the state dict and promote
                # qk_norm — some HF configs (e.g. Qwen3 family) don't carry an
                # explicit qk_norm field, but the checkpoint DOES have per-head
                # QK-norm weights. Without this, the persisted ModelConfig
                # metadata silently records qk_norm=False (issue #171).
                if not cfg.qk_norm and any(
                    ".q_norm.weight" in k or ".k_norm.weight" in k for k in sd
                ):
                    cfg.qk_norm = True
                q = quantize_state_dict(
                    sd,
                    weight_bits=weight_bits,
                    group_size=group_size,
                    fp8_scales=fp8_scales,
                    fp8_block_size=fp8_block_size,
                    fp8_scale_fmt=fp8_scale_fmt,
                )
                for name, qt in q.items():
                    w.add(name, qt)
                del sd, q
            finally:
                if delete_source:
                    os.remove(shard_path)

        # Build meta AFTER the shard loop so any fields promoted from weights
        # (e.g. qk_norm detected via q_norm/k_norm tensors, issue #171) are
        # reflected in the persisted cfg_dump.
        cfg_dump = {k: v for k, v in vars(cfg).items() if not isinstance(v, dict)}
        cfg_dump["extra"] = dict(cfg.extra)
        meta = {
            "arch": cfg.arch,
            "weight_bits": weight_bits,
            "fp8_source": is_fp8_src,
            "config": cfg_dump,
        }
        w.finalize(meta=meta)
    return cfg


def main():
    """CLI: python -m superl8serve.convert <hf_dir> <out.superl8> [--bits 8|4] [--group 128]"""
    import argparse

    ap = argparse.ArgumentParser(description="Convert an HF checkpoint to .superl8")
    ap.add_argument("hf_dir")
    ap.add_argument("out_path")
    ap.add_argument("--bits", type=int, default=8, choices=(4, 8))
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--arch", default=None)
    ap.add_argument(
        "--free-source-shards",
        action="store_true",
        help="DESTRUCTIVE: delete each source .safetensors shard from hf_dir as it is "
        "converted, so peak disk is max(source, output) not source+output. Needed to "
        "fit 850GB+ models that can't hold source+output at once. Off by default — the "
        "input checkpoint is preserved unless you pass this.",
    )
    a = ap.parse_args()
    cfg = convert_hf_to_superl8(
        a.hf_dir,
        a.out_path,
        weight_bits=a.bits,
        group_size=a.group,
        arch=a.arch,
        delete_source=a.free_source_shards,
    )
    print(f"wrote {a.out_path}  arch={cfg.arch}  bits={a.bits}  layers={cfg.num_hidden_layers}")


if __name__ == "__main__":
    main()
