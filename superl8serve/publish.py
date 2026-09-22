# SPDX-License-Identifier: MIT
"""Publish `.superl8` quants to the Hugging Face Hub as **linked** quantizations.

The point of the linkage: a repo whose model-card frontmatter declares
``base_model: <parent>`` + ``base_model_relation: quantized`` shows up under the
parent model's **"Quantizations"** section on HF (the other relations are
``finetune``/``adapter``/``merge``). So every upload carries a generated model card,
not just the raw ``.superl8`` bytes.

License hygiene: we do NOT guess licenses — `publish_one` reads the *parent's* own
license + gated flag from the Hub and (a) refuses to publicly re-host anything
non-commercial or gated, flagging it for a human decision, and (b) inherits the
parent's license tag onto our card otherwise.

`model_card` / `is_restricted` / `repo_name` are pure and unit-tested; `publish_one`
wraps them with the Hub IO.
"""
from __future__ import annotations

from pathlib import Path

# License tags / substrings that forbid (or gate) public re-hosting of a derivative.
# Matched case-insensitively as substrings of the parent's `license` tag.
NONCOMMERCIAL_MARKERS = (
    "-nc", "-nc-", "noncommercial", "non-commercial", "cc-by-nc", "cc-nc",
    "flux-1-dev-non-commercial", "research", "research-only", "rail",  # *RAIL carry use-restrictions
)


def is_restricted(license_tag: str | None, gated) -> bool:
    """True if the parent forbids/gates public redistribution, so we must NOT create
    a public quant repo for it (hold for a human decision).

    `gated` is HF's flag: False / None (open) vs 'auto' / 'manual' / True (gated).
    A missing license is treated as restricted — never publish something whose terms
    we can't read."""
    if gated not in (False, None, "", "false"):
        return True
    if not license_tag:
        return True
    lt = license_tag.strip().lower()
    if lt in ("unknown", "other"):          # `other` = a custom license we can't vet
        return True
    return any(m in lt for m in NONCOMMERCIAL_MARKERS)


def repo_name(parent_repo: str, hf_user: str) -> str:
    """Our repo id for a parent's quant: ``<hf_user>/<parent-basename>-superl8``.
    e.g. ``black-forest-labs/FLUX.1-dev`` -> ``jajmangold/FLUX.1-dev-superl8``.
    Bit-width is NOT in the repo name — both int8 and int4 files live in one repo."""
    base = parent_repo.split("/")[-1]
    return f"{hf_user}/{base}-superl8"


def parse_superl8_name(filename: str) -> dict | None:
    """Recover (parent_repo, kind, bits) from a forge output filename. This is the
    inverse of forge's naming (`repo.replace('/','__')` + `.dit`? + `.b{bits}.superl8`),
    so publishing can group every bit-width of a model into its one repo by scanning
    the weights dir — no dependence on the single-status manifest.

    ``Qwen__Qwen3-0.6B.b8.superl8``            -> Qwen/Qwen3-0.6B      llm  8
    ``Qwen__Qwen-Image.dit.b8.superl8``        -> Qwen/Qwen-Image      dit  8
    Returns None if the name doesn't match."""
    if not filename.endswith(".superl8"):
        return None
    stem = filename[: -len(".superl8")]                     # drop .superl8
    parts = stem.split(".")
    if not parts[-1].startswith("b") or not parts[-1][1:].isdigit():
        return None
    bits = int(parts[-1][1:])
    rest = parts[:-1]
    kind = "llm"
    if rest and rest[-1] == "dit":
        kind = "dit"
        rest = rest[:-1]
    name = ".".join(rest)                                # base names may contain dots
    parent_repo = name.replace("__", "/")
    if "/" not in parent_repo:
        return None
    return {"parent_repo": parent_repo, "kind": kind, "bits": bits}


def _superl8_tags(kind: str) -> list[str]:
    tags = ["superl8", "int8", "w8a8", "dp4a", "volta", "sm_70", "quantized"]
    if kind == "dit":
        tags += ["diffusion", "text-to-image", "comfyui"]
    return tags


def _scheme(bits: int) -> str:
    return "int4 per-group W4A8 (int8 activations)" if bits == 4 else "int8 per-row W8A8"


def model_card(
    parent_repo: str,
    kind: str,
    quants: list[dict],
    *,
    license_tag: str | None,
    pipeline_tag: str | None = None,
    native_dtype: str | None = None,
) -> str:
    """Render README.md for a repo that may hold SEVERAL quant files (e.g. int8 +
    int4). `quants` is ``[{"bits": int, "filename": str, "gb": float}, ...]``.

    The frontmatter's ``base_model`` + ``base_model_relation: quantized`` file the
    repo under the parent's Quantizations on the Hub. A **Files** table + per-bits
    tags make the precision of each file unmistakable (you can tell 4- vs 8-bit at a
    glance)."""
    bits_present = sorted({q["bits"] for q in quants})
    fm: list[str] = ["---"]
    fm.append(f"base_model: {parent_repo}")
    fm.append("base_model_relation: quantized")
    if license_tag:
        fm.append(f"license: {license_tag}")
    if pipeline_tag:
        fm.append(f"pipeline_tag: {pipeline_tag}")
    fm.append("tags:")
    tags = list(_superl8_tags(kind)) + [f"{b}-bit" for b in bits_present]
    for t in tags:
        fm.append(f"  - {t}")
    fm.append("---")

    parent_name = parent_repo.split("/")[-1]
    label = " + ".join(f"int{b}" for b in bits_present)
    dt = (f"\n- Source dtype: `{native_dtype}`. Non-quantized tensors are stored fp16, "
          f"upcast to fp32 only where fp16 would overflow.") if native_dtype else ""
    rows = "\n".join(
        f"| `{q['filename']}` | int{q['bits']} | {_scheme(q['bits'])} | {q['gb']:.2f} GB |"
        for q in sorted(quants, key=lambda q: q["bits"])
    )
    if kind == "dit":
        use = ("Load with [ComfyUI-superl8](https://github.com/jajmangold/ComfyUI-superl8), "
               "which runs the diffusion transformer through the dp4a kernels inside "
               "ComfyUI (`UnetLoaderSUPERL8`). The text encoder and VAE are unchanged.")
    else:
        use = ("Serve with [superl8-serve](https://github.com/jajmangold/superl8-serve): "
               "`load_superl8_state_dict(<file>)` into an `LLMEngine`. The architecture is "
               "read from the file; no separate config is needed.")

    body = f"""
# {parent_name} — superl8 ({label})

An int8/int4 quantization of [`{parent_repo}`](https://huggingface.co/{parent_repo})
to the `.superl8` format, for the [superl8](https://github.com/jajmangold/superl8) DP4A kernels
on NVIDIA Volta (sm_70) GPUs (Tesla V100 and CMP 100-210). It is a derivative of the
parent model; its license and acceptable uses follow the parent, linked above.

## Files

| file | precision | scheme | size |
|------|-----------|--------|------|
{rows}

`.b8.` files are int8, `.b4.` files are int4. Download the one you want. The bytes are
the resident dp4a VRAM layout, so loading is a memory-map and copy with no dequantize
or repack step.

## How to use

{use}

## What was quantized

- Linear and attention weights go to int8 (per-row) or int4 (per-group), with fp32
  scales. Norms, embeddings, and the MoE router are kept in fp16.{dt}
- Target hardware is sm_70, where the fp16 tensor cores are firmware-limited, so the
  integer `__dp4a` path is used for the matmuls.

## Intended use and scope

- For inference with the superl8 runtimes above, on Volta (sm_70) GPUs.
- Out of scope: other GPU architectures (the kernels require sm_70), and anything the
  parent model's license does not permit. It is a derivative, not a new model.

## Limitations

- Quantization is lossy. int8 and especially int4 outputs differ from the fp16/bf16
  parent, and the difference varies by model and task.
- This repository does not include per-model accuracy or benchmark measurements.
  Evaluate on your own task before relying on it.
- Any capabilities, biases, and risks of the parent model carry over. See the parent
  model card for those.

## License

Follows the parent model{f' (`{license_tag}`)' if license_tag else ''}. This is a
derivative quantization, not a relicense.

---

Part of the superl8 stack: [kernels](https://github.com/jajmangold/superl8) ·
[LLM serving](https://github.com/jajmangold/superl8-serve) ·
[ComfyUI DiTs](https://github.com/jajmangold/ComfyUI-superl8).
"""
    return "\n".join(fm) + "\n" + body


def parent_meta(parent_repo: str, token: str | None) -> dict:
    """Fetch the parent's license tag, gated flag, and pipeline_tag from the Hub."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(parent_repo, token=token)
    cd = info.card_data or {}
    lic = cd.get("license") if hasattr(cd, "get") else getattr(cd, "license", None)
    return {
        "license": lic,
        "gated": getattr(info, "gated", False),
        "pipeline_tag": getattr(info, "pipeline_tag", None),
    }


def publish_repo(
    parent_repo: str,
    quant_files: list,
    *,
    hf_user: str,
    token: str | None,
    kind: str = "llm",
    native_dtype: str | None = None,
    private: bool = False,
    skip_restricted: bool = False,
) -> dict:
    """Publish ALL quant files of one parent into its single `<name>-superl8` repo, then
    write ONE model card that enumerates them (so 4- vs 8-bit is unmistakable).

    `quant_files` is ``[(bits, path), ...]``. Creates the repo, uploads every file,
    and writes the card last. Publishes every model by default; the parent's license
    is inherited. `skip_restricted=True` holds non-commercial/gated parents."""
    from huggingface_hub import HfApi

    meta = parent_meta(parent_repo, token)
    if skip_restricted and is_restricted(meta["license"], meta["gated"]):
        return {"status": "held_restricted", "parent": parent_repo,
                "license": meta["license"], "gated": meta["gated"]}

    rid = repo_name(parent_repo, hf_user)
    api = HfApi()
    api.create_repo(rid, token=token, private=private, exist_ok=True, repo_type="model")

    quants = []
    for bits, path in sorted(quant_files):
        fname = Path(path).name
        api.upload_file(path_or_fileobj=str(path), path_in_repo=fname, repo_id=rid,
                        token=token, commit_message=f"upload {fname}")
        quants.append({"bits": bits, "filename": fname, "gb": Path(path).stat().st_size / 1e9})

    card = model_card(parent_repo, kind, quants, license_tag=meta["license"],
                      pipeline_tag=meta["pipeline_tag"], native_dtype=native_dtype)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=rid, token=token, commit_message="model card (linked quant)")
    return {"status": "published", "repo": rid, "license": meta["license"],
            "bits": [q["bits"] for q in quants], "url": f"https://huggingface.co/{rid}"}
