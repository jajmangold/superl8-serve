# SPDX-License-Identifier: MIT
"""Zero-transform weight loader — reads the `.superl8` container from `superl8`.

The on-disk bytes ARE the resident dp4a layout, so loading is mmap + copy with no
dequant / repack / transpose. Supports rank-local PARTIAL loads via the shard index
(a PP stage or MoE expert loads only its tensors — the mmap faults in only those
pages), which is what makes weight loading tractable on the 250 MB/s fleet.
"""

from __future__ import annotations

from typing import Any

import torch

from superl8 import FQReader, QTensor, save_superl8  # the format lives in superl8


def load_superl8_checkpoint(
    path: str,
    *,
    device: str = "cuda",
    names: list[str] | None = None,
    shard: str | None = None,
) -> dict[str, QTensor]:
    """Load (part of) a `.superl8` checkpoint as {name: QTensor} on `device`.

    names : explicit tensor list, or None for all.
    shard : a key into the checkpoint's shard index, e.g. a PP-stage or expert id;
            loads only that shard's tensors. Mutually exclusive with `names`.
    """
    with FQReader(path) as r:
        if shard is not None:
            idx = r.shards
            if shard in idx.get("experts", {}):
                names = idx["experts"][shard]
            elif shard.startswith("pp:"):
                names = idx["pp_stages"][int(shard[3:])]
            else:
                raise KeyError(f"unknown shard {shard!r}; have {list(idx)}")
        names = names if names is not None else r.names
        return r.load_many(names, device)


EMBED_TOKENS_NAME = "model.embed_tokens.weight"


def _quantize_embed_int8(fp16_weight: torch.Tensor) -> QTensor:
    """Symmetric per-row int8 for an embedding table [vocab, hidden]. Done on the
    tensor's current device (CPU during load, to avoid ever putting the ~2.4 GiB
    fp16 table on the GPU). Returns a `per_row_i8` QTensor consumed by
    `VocabEmbedding` (dequant-on-gather)."""
    from superl8.quant.core import quantize_int8_rowwise

    q, scale = quantize_int8_rowwise(fp16_weight)  # int8 [V, H], fp32 [V, 1]
    return QTensor(q.contiguous(), scale.squeeze(-1).contiguous(), scheme="per_row_i8")


def load_superl8_state_dict(
    path: str,
    *,
    device: str = "cuda",
    names: list[str] | None = None,
    shard: str | None = None,
    embed_int8: bool = False,
) -> dict:
    """Build-ready state dict: quantized tensors stay QTensor, `raw` tensors (norms,
    embeddings, router gate) are unwrapped to plain fp16 Tensors — exactly what the
    model builders expect (LinearW8A8 takes a QTensor; RMSNorm/Embedding take a
    Tensor). Feed straight into `build_model(cfg, state_dict)`.

    `embed_int8=True` downcasts `model.embed_tokens.weight` (fp16) to a `per_row_i8`
    QTensor as it loads — halving the vocab-sized embedding table (~2.4 GiB → ~1.2 GiB
    for a 27B). The fp16 table is quantized on the *host* and only the int8 result is
    moved to `device`, so the fp16 copy never occupies GPU memory (the difference
    between fitting and OOM-ing a 27B 4-bit checkpoint on a single 16 GiB card).
    `VocabEmbedding` dequantizes per-row on gather; loss is negligible (a lookup, not
    a matmul). Only applies to a full load (no `names`/`shard` subset)."""
    if embed_int8 and names is None and shard is None:
        with FQReader(path) as r:
            all_names = list(r.names)
        if EMBED_TOKENS_NAME in all_names:
            others = [n for n in all_names if n != EMBED_TOKENS_NAME]
            loaded = load_superl8_checkpoint(path, device=device, names=others)
            # Load the embedding table to the HOST, quantize there, ship int8 to device.
            emb_host = load_superl8_checkpoint(path, device="cpu", names=[EMBED_TOKENS_NAME])
            qt = _quantize_embed_int8(emb_host[EMBED_TOKENS_NAME].data)
            loaded[EMBED_TOKENS_NAME] = QTensor(
                qt.data.to(device), qt.scale.to(device), scheme="per_row_i8"
            )
        else:
            loaded = load_superl8_checkpoint(path, device=device)
    else:
        loaded = load_superl8_checkpoint(path, device=device, names=names, shard=shard)
    out: dict = {}
    for name, qt in loaded.items():
        out[name] = qt.data if getattr(qt, "scheme", None) == "raw" else qt
    return out


def checkpoint_info(path: str) -> dict:
    """Header summary (arch, quant, tensor count, shard index) without loading data."""
    with FQReader(path) as r:
        return {
            "arch": r.header["arch"],
            "quant": r.header["quant"],
            "version": r.header["version"],
            "num_tensors": len(r.names),
            "shards": {k: len(v) for k, v in r.shards.items()},
            "meta": r.header.get("__meta__", {}),
        }


# ── training checkpoint save / load (issue #133) ──────────────────────────


def save_training_checkpoint(
    path: str,
    model_state_dict: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    meta: dict[str, Any] | None = None,
) -> None:
    """Save model weights + optimizer state to a ``.superl8`` checkpoint.

    Each model tensor is stored as a ``raw`` QTensor (fp16).  Optimizer
    buffers are stored under flat names prefixed ``optimizer.state.*``.
    Training metadata (optimizer type, param groups) is embedded in the
    file header's ``__meta__`` dict.
    """
    from superl8serve.training.optimizer import serialize_optimizer_state

    opt_tensors, opt_meta = serialize_optimizer_state(optimizer)

    all_tensors: dict[str, QTensor] = {}
    for name, t in model_state_dict.items():
        all_tensors[name] = QTensor(t.contiguous().cpu(), None, scheme="raw")
    for name, t in opt_tensors.items():
        all_tensors[name] = QTensor(t.contiguous().cpu(), None, scheme="raw")

    full_meta: dict[str, Any] = dict(meta or {})
    full_meta["training"] = True
    full_meta.update(opt_meta)

    save_superl8(path, all_tensors, meta=full_meta)


def load_training_checkpoint(
    path: str,
    optimizer: torch.optim.Optimizer,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    """Load model weights from a training ``.superl8`` checkpoint.

    Restores the model state dict (unwrapped to plain Tensors, matching the
    format of ``nn.Module.state_dict()``) and updates *optimizer* in place.
    Returns the model state dict.
    """
    from superl8serve.training.optimizer import deserialize_optimizer_state

    loaded = load_superl8_checkpoint(path, device=device)
    info = checkpoint_info(path)
    meta = info.get("meta", {})

    model_sd: dict[str, Any] = {}
    opt_flat: dict[str, torch.Tensor] = {}
    for name, qt in loaded.items():
        if name.startswith("optimizer.state."):
            opt_flat[name] = qt.data
        else:
            model_sd[name] = qt.data

    opt_state_dict = deserialize_optimizer_state(opt_flat, meta)
    optimizer.load_state_dict(opt_state_dict)

    return model_sd
