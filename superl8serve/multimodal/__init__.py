# SPDX-License-Identifier: MIT
"""Multimodal pipeline: vision tower (ViT) with int8 dp4a linears + fp16 attention."""

from __future__ import annotations

import base64
import io
import ipaddress
import re
import socket
import urllib.parse
import urllib.request

from PIL import Image

from .preprocess import preprocess_qwen2_5_vl, preprocess_qwen3_5_vl  # noqa: F401
from .projector import build_projector, embed_merge  # noqa: F401
from .vit import VisionTransformer  # noqa: F401

# SSRF guard for user-supplied image_url fetches (public OpenAI-compatible API).
_ALLOWED_SCHEMES = ("http", "https")
_MAX_IMAGE_BYTES = 32 * 1024 * 1024      # 32 MB cap — reject oversized bodies (DoS)
_FETCH_TIMEOUT_S = 10


def _reject_internal_host(host: str) -> None:
    """Resolve `host` and raise if ANY resolved address is non-public — blocks SSRF to
    loopback/private/link-local (incl. 169.254.169.254 cloud metadata)/reserved."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise ValueError(f"cannot resolve image host {host!r}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            raise ValueError(f"refusing to fetch image from non-public address ({ip})")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects — a 30x could point at an internal address after the host check."""

    def redirect_request(self, *a, **k):  # noqa: D401
        raise ValueError("image_url redirects are not followed (SSRF guard)")


def _guarded_urlopen(url: str):
    """urlopen with a scheme allowlist + private-IP block + no-redirect + timeout. NOTE:
    a residual DNS-rebind TOCTOU remains (resolve happens twice) — pinning the resolved
    IP is the follow-up hardening; this already blocks file://, private/metadata IPs, and
    redirect-to-internal, which are the exploitable cases today."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"image_url scheme {parts.scheme!r} not allowed (http/https only)")
    if not parts.hostname:
        raise ValueError("image_url has no host")
    _reject_internal_host(parts.hostname)
    return urllib.request.build_opener(_NoRedirect).open(url, timeout=_FETCH_TIMEOUT_S)


def fetch_image(url: str) -> Image.Image:
    """Fetch an image from an HTTP(S) URL or a base64 data URI and return PIL ``Image`` (RGB).

    Args:
        url: HTTP(S) URL or ``data:image/...;base64,...`` data URI.

    Returns:
        PIL ``Image.Image`` in RGB mode.
    """
    if url.startswith("data:"):
        # data:[<mediatype>][;base64],<data>
        match = re.match(r"^data:[^;]*(?:;base64)?,(.+)$", url)
        if not match:
            raise ValueError(f"Unsupported data URI format: {url[:80]}")
        payload = match.group(1)
        # Cap BEFORE decoding (4 base64 chars -> 3 bytes) so an oversized data URI is
        # rejected without allocating the decoded body or handing it to PIL (S3 DoS) --
        # same 32 MB cap the HTTP fetch path enforces below.
        if len(payload) > (_MAX_IMAGE_BYTES // 3 + 1) * 4:
            raise ValueError(f"data URI image exceeds the {_MAX_IMAGE_BYTES}-byte cap")
        raw = base64.b64decode(payload)
        if len(raw) > _MAX_IMAGE_BYTES:
            raise ValueError(f"data URI image exceeds the {_MAX_IMAGE_BYTES}-byte cap")
        return Image.open(io.BytesIO(raw)).convert("RGB")
    # HTTP(S) URL — SSRF-guarded (scheme allowlist, private/metadata-IP block, no
    # redirects, timeout) with a size cap read.
    with _guarded_urlopen(url) as resp:
        raw = resp.read(_MAX_IMAGE_BYTES + 1)
    if len(raw) > _MAX_IMAGE_BYTES:
        raise ValueError(f"image exceeds the {_MAX_IMAGE_BYTES}-byte cap")
    return Image.open(io.BytesIO(raw)).convert("RGB")
