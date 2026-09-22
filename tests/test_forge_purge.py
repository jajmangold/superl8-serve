# SPDX-License-Identifier: MIT
"""tools/forge.py orphan safety: a prior OOM SIGKILL skips forge_one's per-model
`finally` cleanup, so a leftover staging/<name> dir (up to a full model's worth of
shards) can sit on the archive disk. `_purge_stale_staging` clears them at batch
start. No `superl8`/torch/safetensors needed — forge.py only imports those lazily
inside the functions that use them."""
import importlib.util
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"


def _load_forge(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_BASE", str(tmp_path))
    spec = importlib.util.spec_from_file_location("forge_under_test", TOOLS_DIR / "forge.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_purge_stale_staging_removes_leftover_model_dirs(tmp_path, monkeypatch):
    forge = _load_forge(tmp_path, monkeypatch)
    orphan = forge.STAGING / "org__giant-model"
    orphan.mkdir(parents=True)
    (orphan / "shard-00001.safetensors").write_bytes(b"leftover bytes from a killed run")

    forge._purge_stale_staging()

    assert not orphan.exists()
    assert forge.STAGING.exists()             # the staging dir itself is kept


def test_purge_stale_staging_is_noop_when_clean(tmp_path, monkeypatch):
    forge = _load_forge(tmp_path, monkeypatch)
    forge.STAGING.mkdir(parents=True, exist_ok=True)

    forge._purge_stale_staging()              # must not raise

    assert list(forge.STAGING.iterdir()) == []


def test_purge_stale_staging_handles_missing_staging_dir(tmp_path, monkeypatch):
    forge = _load_forge(tmp_path, monkeypatch)
    assert not forge.STAGING.exists()

    forge._purge_stale_staging()               # must not raise even if never created
