import http.client
import importlib.util
import json
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


ROOT = Path(__file__).resolve().parents[1]
ROUTER_PATH = ROOT / "docker/qwen35-tq3/model_router.py"
SPEC = importlib.util.spec_from_file_location(
    "qwen_model_router", ROUTER_PATH
)
router = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(router)


def load_router():
    spec = importlib.util.spec_from_file_location(
        "qwen_model_router_test", ROUTER_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_exact_model_routes_are_distinct():
    assert router.backend_for("coder", "session-a")[0] == "qwen"
    assert router.backend_for(router.LFM_MODEL, "session-a") == (
        "lfm25", router.LFM_BACKEND
    )
    assert router.backend_for(router.LFM_FULL_CONTEXT_MODEL, "session-a") == (
        "lfm25", router.LFM_FULL_CONTEXT_BACKEND
    )
    assert router.backend_for("unknown", "session-a") is None
    assert router.backend_for(None, "session-a") is None


def test_qwen_session_affinity_is_stable():
    first = router.backend_for("coder", "stable-session")
    assert first == router.backend_for("coder", "stable-session")
    assert first[1] in router.QWEN_BACKENDS


def test_both_lfm_ids_never_fall_through_to_qwen():
    for model in (router.LFM_MODEL, router.LFM_FULL_CONTEXT_MODEL, "unknown", None):
        selected = router.backend_for(model, "session-a")
        assert selected is None or selected[0] == "lfm25"


def test_both_lfm_ids_can_map_to_one_active_backend(monkeypatch):
    monkeypatch.setattr(router, "LFM_BACKEND", "lane-a:8000")
    monkeypatch.setattr(router, "LFM_FULL_CONTEXT_BACKEND", "lane-a:8000")
    monkeypatch.setattr(router, "LFM_LANES", {
        router.LFM_MODEL: "lane-a:8000",
        router.LFM_FULL_CONTEXT_MODEL: "lane-a:8000",
    })
    assert router.backend_for(router.LFM_MODEL, "session-a") == ("lfm25", "lane-a:8000")
    assert router.backend_for(router.LFM_FULL_CONTEXT_MODEL, "session-a") == (
        "lfm25", "lane-a:8000"
    )


def test_lane_backend_env_defaults_and_overrides(monkeypatch):
    fresh = load_router()
    assert fresh.LFM_BACKEND == "content-factory-lfm25-b512:8000"
    assert fresh.LFM_FULL_CONTEXT_BACKEND == "content-factory-lfm25-full-context:8000"
    monkeypatch.setenv("LFM_FULL_CONTEXT_BACKEND", "custom-fc:8000")
    fresh2 = load_router()
    assert fresh2.LFM_LANES[fresh2.LFM_FULL_CONTEXT_MODEL] == "custom-fc:8000"
    assert fresh2.LFM_LANES[fresh2.LFM_MODEL] == "content-factory-lfm25-b512:8000"


def test_lane_available_fails_closed_for_dead_backend(monkeypatch):
    monkeypatch.setattr(router, "LFM_HEALTH_CHECK", True)
    monkeypatch.setattr(router, "LFM_LANES", {router.LFM_MODEL: "127.0.0.1:1"})
    router._lfm_health_cache.clear()
    assert router.lane_available(router.LFM_MODEL) is False


def test_configured_only_policy_treats_all_lanes_available(monkeypatch):
    monkeypatch.setenv("LFM_HEALTH_CHECK", "0")
    fresh = load_router()
    assert fresh.LFM_HEALTH_CHECK is False
    monkeypatch.setattr(fresh, "LFM_LANES", {
        fresh.LFM_MODEL: "127.0.0.1:1",
        fresh.LFM_FULL_CONTEXT_MODEL: "127.0.0.1:1",
    })
    assert fresh.lane_available(fresh.LFM_MODEL) is True
    assert fresh.lane_available(fresh.LFM_FULL_CONTEXT_MODEL) is True
    assert fresh.lane_available("unmapped") is False


class _MockOpenAIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):  # noqa: N802
        if self.path == "/v1/models":
            body = json.dumps({
                "object": "list",
                "data": [
                    {"id": mid, "object": "model", "owned_by": "superl8-serve"}
                    for mid in self.server.served_ids
                ],
            }).encode()
            self._respond(200, {"Content-Type": "application/json"}, body)
        elif self.path in self.server.get_responses:
            status, headers, body = self.server.get_responses[self.path]
            self._respond(status, headers, body)
        else:
            self._respond(404, {"Content-Type": "application/json"}, b'{"error":"not found"}')

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        self.server.last_body = body
        self.server.last_headers = {k.lower(): v for k, v in self.headers.items()}
        status, headers, resp_body = self.server.post_response
        self._respond(status, headers, resp_body)

    def _respond(self, status, headers, body):
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        if any(k.lower() == "connection" and v.lower() == "close" for k, v in headers.items()):
            self.close_connection = True
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()


def _start_backend(served_ids, post_response=(200, {"Content-Type": "application/json"}, b"{}")):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockOpenAIHandler)
    server.served_ids = list(served_ids)
    server.post_response = post_response
    server.get_responses = {}
    server.last_body = None
    server.last_headers = None
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _addr(server):
    return f"127.0.0.1:{server.server_address[1]}"


def _stop(server):
    server.shutdown()
    server.server_close()


def _start_router():
    server = ThreadingHTTPServer(("127.0.0.1", 0), router.RouterHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _request(server, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection(
        server.server_address[0], server.server_address[1], timeout=15
    )
    conn.request(method, path, body=body, headers=headers or {})
    resp = conn.getresponse()
    data = resp.read()
    out_headers = {k.lower(): v for k, v in resp.getheaders()}
    conn.close()
    return resp.status, out_headers, data


SSE_BODY = b'data: {"id":"chatcmpl-x","object":"chat.completion.chunk"}\n\ndata: [DONE]\n\n'


def test_models_endpoint_advertises_only_available_lanes(monkeypatch):
    backend = _start_backend(served_ids=[router.LFM_MODEL])
    monkeypatch.setattr(router, "LFM_HEALTH_CHECK", True)
    monkeypatch.setattr(router, "LFM_LANES", {
        router.LFM_MODEL: _addr(backend),
        router.LFM_FULL_CONTEXT_MODEL: "127.0.0.1:1",
    })
    router._lfm_health_cache.clear()
    rtr = _start_router()
    try:
        status, _, body = _request(rtr, "GET", "/v1/models")
        assert status == 200
        ids = [m["id"] for m in json.loads(body)["data"]]
        assert ids == [router.QWEN_MODEL, router.LFM_MODEL]
        assert router.LFM_FULL_CONTEXT_MODEL not in ids
    finally:
        _stop(rtr)
        _stop(backend)


def test_models_endpoint_both_ids_when_one_backend_serves(monkeypatch):
    backend = _start_backend(served_ids=[router.LFM_MODEL])
    monkeypatch.setattr(router, "LFM_HEALTH_CHECK", True)
    monkeypatch.setattr(router, "LFM_LANES", {
        router.LFM_MODEL: _addr(backend),
        router.LFM_FULL_CONTEXT_MODEL: _addr(backend),
    })
    router._lfm_health_cache.clear()
    rtr = _start_router()
    try:
        status, _, body = _request(rtr, "GET", "/v1/models")
        assert status == 200
        ids = [m["id"] for m in json.loads(body)["data"]]
        assert router.LFM_MODEL in ids
        assert router.LFM_FULL_CONTEXT_MODEL in ids
    finally:
        _stop(rtr)
        _stop(backend)


def test_post_streaming_and_model_backend_label(monkeypatch):
    backend = _start_backend(
        served_ids=[router.LFM_MODEL],
        post_response=(
            200,
            {"Content-Type": "text/event-stream", "Connection": "close", "X-Backend-Probe": "mock"},
            SSE_BODY,
        ),
    )
    monkeypatch.setattr(router, "LFM_HEALTH_CHECK", True)
    monkeypatch.setattr(router, "LFM_LANES", {
        router.LFM_MODEL: _addr(backend),
        router.LFM_FULL_CONTEXT_MODEL: _addr(backend),
    })
    router._lfm_health_cache.clear()
    rtr = _start_router()
    try:
        status, headers, body = _request(
            rtr, "POST", "/v1/chat/completions",
            body=json.dumps({
                "model": router.LFM_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
            }).encode(),
            headers={"Content-Type": "application/json", "X-Code-Session": "s1"},
        )
        assert status == 200
        assert headers.get("content-type") == "text/event-stream"
        assert body == SSE_BODY
        assert headers.get("x-model-backend") == "lfm25"
        assert headers.get("x-backend-probe") == "mock"
        assert json.loads(backend.last_body)["model"] == router.LFM_MODEL
    finally:
        _stop(rtr)
        _stop(backend)


def test_post_full_context_id_routes_to_full_context_lane(monkeypatch):
    b512 = _start_backend(
        served_ids=[router.LFM_MODEL],
        post_response=(200, {"Content-Type": "text/event-stream", "Connection": "close"}, SSE_BODY),
    )
    fc = _start_backend(
        served_ids=[router.LFM_FULL_CONTEXT_MODEL],
        post_response=(
            200,
            {"Content-Type": "text/event-stream", "Connection": "close"},
            b'data: {"fc":true}\n\n',
        ),
    )
    monkeypatch.setattr(router, "LFM_HEALTH_CHECK", True)
    monkeypatch.setattr(router, "LFM_LANES", {
        router.LFM_MODEL: _addr(b512),
        router.LFM_FULL_CONTEXT_MODEL: _addr(fc),
    })
    router._lfm_health_cache.clear()
    rtr = _start_router()
    try:
        status, headers, body = _request(
            rtr, "POST", "/v1/chat/completions",
            body=json.dumps({"model": router.LFM_FULL_CONTEXT_MODEL}).encode(),
            headers={"Content-Type": "application/json"},
        )
        assert status == 200
        assert body == b'data: {"fc":true}\n\n'
        assert headers.get("x-model-backend") == "lfm25"
        assert json.loads(fc.last_body)["model"] == router.LFM_FULL_CONTEXT_MODEL
        assert b512.last_body is None
    finally:
        _stop(rtr)
        _stop(b512)
        _stop(fc)


def test_post_unknown_model_is_404_without_qwen_fallthrough(monkeypatch):
    monkeypatch.setattr(router, "LFM_LANES", {})
    rtr = _start_router()
    try:
        status, _, body = _request(
            rtr, "POST", "/v1/chat/completions",
            body=json.dumps({"model": "not-a-model"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        assert status == 404
        assert json.loads(body)["error"]["type"] == "model_not_found"
    finally:
        _stop(rtr)


def test_post_missing_or_non_string_model_is_404(monkeypatch):
    monkeypatch.setattr(router, "LFM_LANES", {})
    rtr = _start_router()
    try:
        status, _, body = _request(
            rtr, "POST", "/v1/chat/completions",
            body=json.dumps({"messages": []}).encode(),
            headers={"Content-Type": "application/json"},
        )
        assert status == 404
        status, _, body = _request(
            rtr, "POST", "/v1/chat/completions",
            body=json.dumps({"model": {"id": 1}}).encode(),
            headers={"Content-Type": "application/json"},
        )
        assert status == 404
    finally:
        _stop(rtr)


def test_post_unavailable_lane_is_404(monkeypatch):
    monkeypatch.setattr(router, "LFM_HEALTH_CHECK", True)
    monkeypatch.setattr(router, "LFM_LANES", {
        router.LFM_MODEL: "127.0.0.1:1",
        router.LFM_FULL_CONTEXT_MODEL: "127.0.0.1:1",
    })
    router._lfm_health_cache.clear()
    rtr = _start_router()
    try:
        status, _, body = _request(
            rtr, "POST", "/v1/chat/completions",
            body=json.dumps({"model": router.LFM_MODEL}).encode(),
            headers={"Content-Type": "application/json"},
        )
        assert status == 404
        assert json.loads(body)["error"]["type"] == "lane_unavailable"
    finally:
        _stop(rtr)


def test_post_backend_status_and_body_passthrough(monkeypatch):
    err_body = b'{"error":{"message":"nope","type":"invalid_request_error"}}'
    backend = _start_backend(
        served_ids=[router.LFM_MODEL],
        post_response=(400, {"Content-Type": "application/json"}, err_body),
    )
    monkeypatch.setattr(router, "LFM_HEALTH_CHECK", True)
    monkeypatch.setattr(router, "LFM_LANES", {
        router.LFM_MODEL: _addr(backend),
        router.LFM_FULL_CONTEXT_MODEL: _addr(backend),
    })
    router._lfm_health_cache.clear()
    rtr = _start_router()
    try:
        status, headers, body = _request(
            rtr, "POST", "/v1/chat/completions",
            body=json.dumps({"model": router.LFM_MODEL}).encode(),
            headers={"Content-Type": "application/json"},
        )
        assert status == 400
        assert body == err_body
        assert headers.get("x-model-backend") == "lfm25"
    finally:
        _stop(rtr)
        _stop(backend)


def test_post_malformed_json_is_400(monkeypatch):
    monkeypatch.setattr(router, "LFM_LANES", {})
    rtr = _start_router()
    try:
        status, _, _ = _request(
            rtr, "POST", "/v1/chat/completions",
            body=b"{not json",
            headers={"Content-Type": "application/json"},
        )
        assert status == 400
    finally:
        _stop(rtr)
