# SPDX-License-Identifier: MIT
"""Regression tests for the reference-oracle scorer (tools/reference_oracle).

Pure-Python: no GGUF, no torch, no built dumper needed — these pin the metric
math and the first-divergence localization logic that the harness relies on to
catch a wrong forward (the flint8 #105 lesson: a real oracle must *catch* a bug).
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
_RECON = os.path.join(_HERE, "..", "tools", "reference_oracle", "reconcile.py")


def _load_reconcile():
    spec = importlib.util.spec_from_file_location("reconcile", _RECON)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load_reconcile()


def _write_dump(d, tensors):
    """Write a {name: ndarray[...,ne0]} mapping in the on-disk dump format."""
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "manifest.tsv"), "w") as f:
        f.write("name\tne0\tne1\tne2\tne3\top\tfile\n")
        for name, arr in tensors.items():
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            # logical shape [ne3,ne2,ne1,ne0]; pad to 4 dims
            shp = (1,) * (4 - arr.ndim) + arr.shape
            ne3, ne2, ne1, ne0 = shp
            fn = name.replace("/", "_") + ".f32"
            arr.tofile(os.path.join(d, fn))
            f.write(f"{name}\t{ne0}\t{ne1}\t{ne2}\t{ne3}\tOP\t{fn}\n")


def test_metrics_identical():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert R.cos(a, a) == pytest.approx(1.0)
    assert R.rel_l1(a, a) == pytest.approx(0.0)


def test_metrics_perturbed():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((7, 16)).astype(np.float32)
    b = a + 0.1 * rng.standard_normal(a.shape).astype(np.float32)
    assert 0.9 < R.cos(a, b) < 1.0
    assert R.rel_l1(a, b) > 0.0


def test_layer_parse():
    assert R._layer_of("__fattn__-15") == 15
    assert R._layer_of("l_out-3") == 3
    assert R._layer_of("result_output") == -1


def test_compare_no_divergence(tmp_path, capsys):
    ref = str(tmp_path / "ref")
    rng = np.random.default_rng(1)
    tensors = {f"l_out-{i}": rng.standard_normal((7, 8)).astype(np.float32) for i in range(4)}
    _write_dump(ref, tensors)
    np.savez(str(tmp_path / "cand.npz"), **tensors)

    args = _Args(ref_dir=ref, candidate=str(tmp_path / "cand.npz"),
                 map=None, cos=0.999, rell1=0.02)
    rc = R.cmd_compare(args)
    assert rc == 0
    assert "no divergence" in capsys.readouterr().out


def test_compare_first_divergence(tmp_path, capsys):
    ref = str(tmp_path / "ref")
    rng = np.random.default_rng(2)
    tensors = {f"__fattn__-{i}": rng.standard_normal((7, 8)).astype(np.float32)
               for i in (3, 7, 11, 15)}
    _write_dump(ref, tensors)
    cand = dict(tensors)
    cand["__fattn__-11"] = cand["__fattn__-11"] + 5.0  # break layer 11
    np.savez(str(tmp_path / "cand.npz"), **cand)

    args = _Args(ref_dir=ref, candidate=str(tmp_path / "cand.npz"),
                 map=None, cos=0.999, rell1=0.02)
    rc = R.cmd_compare(args)
    out = capsys.readouterr().out
    assert rc == 1
    assert "FIRST DIVERGENCE: layer 11" in out


def test_compare_name_map(tmp_path):
    ref = str(tmp_path / "ref")
    a = np.arange(8, dtype=np.float32)
    _write_dump(ref, {"__fattn__-0": a})
    np.savez(str(tmp_path / "cand.npz"), my_attn_0=a)
    mp = str(tmp_path / "map.json")
    with open(mp, "w") as f:
        f.write('{"my_attn_0": "__fattn__-0"}')
    args = _Args(ref_dir=ref, candidate=str(tmp_path / "cand.npz"),
                 map=mp, cos=0.999, rell1=0.02)
    assert R.cmd_compare(args) == 0


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)
