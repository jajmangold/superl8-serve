# SPDX-License-Identifier: MIT
"""OpenAI-compatible HTTP surface over `LLMEngine`. See `superl8serve.api.app.create_app`
and the `python -m superl8serve.api.server` entrypoint."""
from .app import create_app

__all__ = ["create_app"]
