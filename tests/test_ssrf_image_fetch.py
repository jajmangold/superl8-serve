# SPDX-License-Identifier: MIT
"""SSRF guard on the user-supplied image_url fetch (superl8serve.multimodal.fetch_image).

All negative cases reject BEFORE any HTTP request (scheme check + local DNS resolve of
private/literal IPs), so these run offline."""
import base64

import pytest

pytest.importorskip("PIL")
from superl8serve.multimodal import _reject_internal_host, fetch_image


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",           # local file read via urllib
    "ftp://example.com/x",
    "gopher://127.0.0.1:6379/",
])
def test_non_http_scheme_rejected(url):
    with pytest.raises(ValueError, match="scheme"):
        fetch_image(url)


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/x",
    "http://localhost/x",
    "http://169.254.169.254/latest/meta-data/",   # cloud metadata (link-local)
    "http://10.0.0.5/x",
    "http://192.168.1.1/x",
    "http://[::1]/x",
])
def test_internal_address_rejected(url):
    with pytest.raises(ValueError, match="non-public|resolve"):
        fetch_image(url)


def test_reject_internal_host_helper():
    with pytest.raises(ValueError, match="non-public"):
        _reject_internal_host("127.0.0.1")
    with pytest.raises(ValueError, match="non-public"):
        _reject_internal_host("169.254.169.254")


def test_data_uri_still_works():
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4"
        "2mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")  # 1x1 PNG
    img = fetch_image("data:image/png;base64," + base64.b64encode(png).decode())
    assert img.size == (1, 1)
