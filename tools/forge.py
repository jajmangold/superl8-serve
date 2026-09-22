# SPDX-License-Identifier: MIT
"""forge — download HF models, quantize them to the resident `.superl8` format on the
archive disk, and purge the transient download so the drive never fills.

Per model: check free space -> snapshot_download to staging/ -> convert to
weights/<name>.superl8 -> delete staging/ -> record in MANIFEST.json. A crash still
purges its staging dir (atexit + per-model try/finally). LLMs go through
`superl8serve.convert`; diffusion DiTs quantize the transformer weights inline with the
same skip-list `ComfyUI-superl8` uses (norms/modulation/embedders stay fp16).

Runs inside the superl8 container (needs the built `superl8` + torch + safetensors +
huggingface_hub). See tools/forge.sh for the container wrapper.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make superl8serve importable

BASE = Path(os.environ.get("FORGE_BASE", "/mnt/24tb/superl8-forge"))
STAGING, WEIGHTS, LOGS = BASE / "staging", BASE / "weights", BASE / "logs"
MANIFEST = BASE / "MANIFEST.json"
MIN_FREE_GB = float(os.environ.get("FORGE_MIN_FREE_GB", "300"))
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def free_gb(p: Path) -> float:
    s = os.statvfs(p)
    return s.f_bavail * s.f_frsize / 1e9


def _name(repo: str) -> str:
    return repo.replace("/", "__")


def _purge_stale_staging():
    """Remove any leftover staging/<name> dirs from a prior crash. An OOM SIGKILL
    (e.g. GLM-4.5-Air) kills the process outright, skipping forge_one's per-model
    `finally` cleanup, so a stale staging dir can otherwise sit on the archive disk
    (up to a full model's worth of shards) until someone notices."""
    if not STAGING.exists():
        return
    for d in STAGING.iterdir():
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            print(f"[purge] removed stale staging dir {d.name}")


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"models": {}}


def _save(m: dict):
    MANIFEST.write_text(json.dumps(m, indent=2) + "\n")


def _set(repo: str, **kw):
    m = _manifest()
    m["models"].setdefault(repo, {})
    m["models"][repo].update(kw)
    _save(m)


def _repo_size_gb(repo: str) -> float:
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, token=HF_TOKEN, files_metadata=True)
    total = sum((s.size or 0) for s in (info.siblings or [])
                if s.rfilename.endswith((".safetensors", ".bin")))
    return total / 1e9


def _download(repo: str, dest: Path, allow: list[str], subfolder: str | None = None):
    from huggingface_hub import snapshot_download
    patterns = allow if not subfolder else [f"{subfolder}/{p}" for p in allow] + ["*.json"]
    snapshot_download(repo_id=repo, local_dir=str(dest), allow_patterns=patterns, token=HF_TOKEN)


def quant_llm(repo: str, bits: int, group: int) -> Path:
    from superl8serve.convert import convert_hf_to_superl8
    name = _name(repo)
    stage, out = STAGING / name, WEIGHTS / f"{name}.b{bits}.superl8"
    _download(repo, stage, ["*.safetensors", "config.json", "*.model", "tokenizer*", "*.txt"])
    # `stage` is a disposable download we own; free each shard as it's consumed so
    # peak disk stays ~max(source, output) — required to forge 850GB+ models (#69).
    convert_hf_to_superl8(
        str(stage), str(out), weight_bits=bits, group_size=group, delete_source=True
    )
    return out


# ---- DiT quant (inline; mirrors ComfyUI-superl8's skip-list) ----
_DIT_SKIP = ("norm", "modulation", "adaln", "ada_ln", "pos_embed", "patch_embed",
             "_embed", "embedder", "time_in", "txt_in", "img_in", "vector_in",
             "guidance_in", "context_embedder", "final_layer", "proj_out")


def quant_dit(repo: str, subfolder: str, bits: int, weight_file: str | None = None) -> Path:
    import torch
    from safetensors.torch import load_file

    from superl8 import QTensor, save_superl8
    from superl8.quant.core import quantize_int8_rowwise
    from superl8.quant.lowbit import quantize_lowbit

    name = _name(repo)
    stage, out = STAGING / name, WEIGHTS / f"{name}.dit.b{bits}.superl8"
    if weight_file:
        # Flat repos (e.g. Lightricks/LTX-2.3 ships many variants) — grab ONE file.
        _download(repo, stage, [weight_file, "config.json"])
        files = [stage / weight_file]
    else:
        _download(repo, stage, ["*.safetensors", "config.json"], subfolder=subfolder)
        tdir = stage / subfolder if (stage / subfolder).exists() else stage
        files = sorted(tdir.glob("*.safetensors"))
    sd: dict = {}
    for f in files:
        sd.update(load_file(str(f)))
    src_bf16 = any(w.dtype == torch.bfloat16 for w in sd.values())
    qsd: dict = {}
    for k, w in sd.items():
        w = w.detach().cpu()                        # keep NATIVE dtype (don't truncate bf16!)
        kl = k.lower()
        if w.dim() == 2 and w.shape[-1] % 4 == 0 and not any(s in kl for s in _DIT_SKIP):
            wf = w.float()                          # quantize from full precision
            if bits == 4 and w.shape[-1] % 128 == 0:
                codes, sc = quantize_lowbit(wf, 4, dim=-1, group_size=128)
                c = codes.to(torch.int64)
                packed = ((c[:, 0::2] & 0xF) | ((c[:, 1::2] & 0xF) << 4)).to(torch.uint8)
                qsd[k] = QTensor(packed.contiguous(), sc.float().contiguous(),
                                 scheme="per_group_i4", group_size=128, codebook="int4")
            else:
                q, sc = quantize_int8_rowwise(wf)
                qsd[k] = QTensor(q.contiguous(), sc.squeeze(-1).float().contiguous(),
                                 scheme="per_row_i8")
        else:
            # Raw passthrough (norms/modulation/embedders): fp16 unless it overflows
            # (bf16 DiTs can have out-of-fp16-range tensors -> inf -> black images).
            w16 = w.to(torch.float16)
            raw = w.float() if (torch.isinf(w16).any() and not torch.isinf(w.float()).any()) else w16
            qsd[k] = QTensor(raw, None, scheme="raw")
    save_superl8(str(out), qsd, meta={"kind": "dit", "repo": repo, "bits": bits,
                                   "native_dtype": "bfloat16" if src_bf16 else "float16"})
    return out


def _expected_out(repo: str, kind: str, bits: int) -> Path:
    """The .superl8 path forge writes for (repo, kind, bits) — used for per-bits
    idempotency so an int4 run doesn't skip an already-int8-done model."""
    name = _name(repo)
    return WEIGHTS / (f"{name}.dit.b{bits}.superl8" if kind == "dit" else f"{name}.b{bits}.superl8")


def forge_one(repo: str, *, kind: str, bits: int, group: int, subfolder: str,
              weight_file: str | None = None) -> bool:
    if _expected_out(repo, kind, bits).exists():
        print(f"[skip] {repo} b{bits} already done"); return True
    stage = STAGING / _name(repo)
    log = (LOGS / f"{_name(repo)}.log").open("a")

    def say(msg):
        line = f"{time.strftime('%H:%M:%S')} {repo}: {msg}"
        print(line); log.write(line + "\n"); log.flush()

    try:
        avail = free_gb(BASE)
        if avail < MIN_FREE_GB:
            say(f"SKIP — archive low ({avail:.0f}GB < {MIN_FREE_GB:.0f}GB floor)")
            _set(repo, status="skipped_disk"); return False
        try:
            need = _repo_size_gb(repo)
            say(f"~{need:.1f}GB download; {avail:.0f}GB free")
            if avail - need < MIN_FREE_GB:
                say("SKIP — would drop below floor"); _set(repo, status="skipped_disk"); return False
        except Exception as e:
            say(f"size probe failed ({e}); proceeding")
        _set(repo, status="downloading", started=time.time())
        say(f"quant {kind} bits={bits} ...")
        out = (quant_dit(repo, subfolder, bits, weight_file) if kind == "dit"
               else quant_llm(repo, bits, group))
        size = out.stat().st_size / 1e9
        # Record per-bits under `outputs` so an int4 run doesn't clobber the int8 record.
        m = _manifest()
        outputs = m["models"].get(repo, {}).get("outputs", {})
        outputs[f"b{bits}"] = {"out": str(out), "out_gb": round(size, 3), "finished": time.time()}
        _set(repo, status="done", kind=kind, bits=bits, out=str(out),
             out_gb=round(size, 3), finished=time.time(), outputs=outputs)
        say(f"DONE -> {out.name} ({size:.2f}GB)")
        return True
    except Exception as e:
        say(f"FAILED: {e}\n{traceback.format_exc()}")
        _set(repo, status="failed", error=str(e)); return False
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True); say("purged staging")
        log.close()


def _native_dtype_of(out_path: str) -> str | None:
    """Read the `native_dtype` recorded in a `.superl8` file's meta (DiTs), if present."""
    try:
        from superl8.format import FQReader
        r = FQReader(out_path)
        try:
            return (r.header.get("__meta__") or {}).get("native_dtype")
        finally:
            r.close()
    except Exception:
        return None


def publish_all(hf_user: str, *, only: str | None = None, private: bool = False):
    """Upload every `.superl8` on disk to the Hub as a linked quantization
    (base_model_relation=quantized), inheriting the parent's license. Scans the
    weights dir (not the single-status manifest) so BOTH int8 and int4 of a model
    land in its one `<name>-superl8` repo. Publishes ALL models — the parent's real
    license tag is carried onto our card either way."""
    from superl8serve.publish import parse_superl8_name, publish_repo

    # Group every weight file by parent repo so BOTH int8 and int4 land in one repo.
    groups: dict[str, dict] = {}
    for f in sorted(WEIGHTS.glob("*.superl8")):
        info = parse_superl8_name(f.name)
        if not info:
            print(f"  [skip] unparseable {f.name}"); continue
        if only and only not in info["parent_repo"]:
            continue
        g = groups.setdefault(info["parent_repo"], {"kind": info["kind"], "files": []})
        g["files"].append((info["bits"], f))

    print(f"publishing {sum(len(g['files']) for g in groups.values())} file(s) across "
          f"{len(groups)} model(s) as {hf_user}/*-superl8 ...")
    if not groups:
        # Fail loudly: a quant ran but nothing matched the filter (or no .superl8 on disk).
        # Silent 0-published was how HF Jobs "completed" without publishing anything.
        print(f"[error] publish matched 0 files in {WEIGHTS}"
              f"{f' for filter {only!r}' if only else ''}")
        sys.exit(2)
    published = 0
    for parent in sorted(groups):
        g = groups[parent]
        native = _native_dtype_of(str(g["files"][0][1]))
        try:
            res = publish_repo(parent, g["files"], hf_user=hf_user, token=HF_TOKEN,
                               kind=g["kind"], native_dtype=native, private=private)
            bits = ",".join(f"int{b}" for b in res.get("bits", []))
            print(f"  {res['status']:16} {parent} [{bits}] -> "
                  f"{res.get('url', res.get('license', ''))}")
            if res.get("status") == "published":
                published += 1
        except Exception as e:
            print(f"  ERROR            {parent}: {e}")
    if published == 0:
        print("[error] published 0 repos"); sys.exit(2)


def main():
    ap = argparse.ArgumentParser(description="forge: HF -> .superl8 with disk cleanup")
    ap.add_argument("action", choices=["one", "batch", "status", "publish"])
    ap.add_argument("target", nargs="?", help="repo id (one) / models.txt (batch) / "
                    "substring filter (publish)")
    ap.add_argument("--kind", choices=["llm", "dit"], default="llm")
    ap.add_argument("--bits", type=int, default=8, choices=(4, 8))
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--subfolder", default="transformer")
    ap.add_argument("--weight-file", default=None,
                    help="DiT: quant ONE specific safetensors file from a flat repo "
                         "(e.g. ltx-2.3-22b-distilled-1.1.safetensors)")
    ap.add_argument("--hf-user", default="jajmangold", help="HF account for *-superl8 repos")
    ap.add_argument("--private", action="store_true", help="create private repos")
    a = ap.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    for d in (STAGING, WEIGHTS, LOGS):
        d.mkdir(exist_ok=True)

    if a.action == "status":
        m = _manifest()
        for repo, v in sorted(m["models"].items()):
            print(f"  {v.get('status','?'):14} {repo}  {v.get('out_gb','')}")
        print(f"archive free: {free_gb(BASE):.0f} GB")
        return

    if a.action == "publish":
        publish_all(a.hf_user, only=a.target, private=a.private)
        return

    if a.action == "one":
        ok = forge_one(a.target, kind=a.kind, bits=a.bits, group=a.group,
                       subfolder=a.subfolder, weight_file=a.weight_file)
        sys.exit(0 if ok else 1)

    # batch: lines "repo[,kind[,bits]]", '#' comments
    _purge_stale_staging()
    for raw in Path(a.target).read_text().splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        repo = parts[0]
        kind = parts[1] if len(parts) > 1 else "llm"
        bits = int(parts[2]) if len(parts) > 2 else a.bits
        if free_gb(BASE) < MIN_FREE_GB:
            print(f"[halt] archive below floor; stopping batch before {repo}"); break
        forge_one(repo, kind=kind, bits=bits, group=a.group, subfolder=a.subfolder)


if __name__ == "__main__":
    main()
