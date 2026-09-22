# SPDX-License-Identifier: MIT
"""Explicit GGUF W8 throughput policy (issue #384)."""

import pytest

from superl8serve.api import server


def test_cli_w8_policy_is_default_off_and_explicit():
    parser = server._build_arg_parser()
    assert parser.parse_args(["--model", "m.gguf"]).gguf_force_w8 is False
    assert parser.parse_args(["--model", "m.gguf", "--gguf-force-w8"]).gguf_force_w8 is True


def test_load_engine_forwards_explicit_policy_only_to_gguf(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "superl8serve.gguf_native.load_gguf_engine",
        lambda path, **kwargs: captured.update(path=path, kwargs=kwargs) or object(),
    )

    server.load_engine("m.gguf", device="cpu", gguf_force_w8=True)

    assert captured["kwargs"]["force_w8"] is True
    with pytest.raises(ValueError, match="only valid for .gguf"):
        server.load_engine("m.superl8", device="cpu", gguf_force_w8=True)
