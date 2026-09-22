# SPDX-License-Identifier: MIT
"""`/v1/files` + `/v1/batches` tests (issue #41): OpenAI Batch API shape (JSONL
file in, JSONL file out) layered over the SAME `EngineWorker` continuous-batching
loop the streaming endpoints use -- against a fake engine + fake tokenizer, no
CUDA/superl8/model weights needed (see test_api.py's own fakes for the streaming-
endpoint version of this same rationale). Also covers a batch line carrying
`tools`/`tool_choice` (issue #40), proving batch requests go through the exact
same tool-calling code path a live HTTP request does."""
import itertools
import json

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from starlette.testclient import TestClient  # noqa: E402  (sync client for the ASGI app)

from superl8serve.api.app import create_app  # noqa: E402
from superl8serve.engine.sequence import SamplingParams, Sequence, Status  # noqa: E402

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}


class FakeTokenizer:
    """`encode`/`apply_chat_template` map every prompt to its own char codes (rather
    than a fixed token stream like test_api.py's fake) so a batch's per-request
    outputs are individually checkable, not just present."""
    eos_token_id = 999

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True,
                            chat_template=None, tools=None):
        assert messages[-1]["role"] == "user"
        return [ord(c) for c in messages[-1]["content"]]

    def encode(self, text, **kw):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        kept = [i for i in ids if not (skip_special_tokens and i == self.eos_token_id)]
        return "".join(chr(i) for i in kept)


class FixedTextTokenizer(FakeTokenizer):
    """Decodes any non-empty token id sequence to a fixed string, so a tool-call
    batch line can drive an exact model output (a `<tool_call>...</tool_call>` tag)
    without modeling a real subword vocabulary -- mirrors test_api.py's fake of the
    same name."""

    def __init__(self, text: str):
        self.text = text

    def decode(self, ids, skip_special_tokens=True):
        return self.text if ids else ""


class ThinkingTextTokenizer(FixedTextTokenizer):
    think_token_id = 98

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True,
                            chat_template=None, tools=None):
        prompt_ids = super().apply_chat_template(
            messages, tokenize, add_generation_prompt, chat_template, tools
        )
        return [*prompt_ids, self.think_token_id]

    def encode(self, text, **kw):
        if text == "<think>":
            return [self.think_token_id]
        return super().encode(text, **kw)


class FakeEngine:
    """Continuous-batching double: each step appends `last_token + 1`, so a
    request's completion is derived from (and checkable against) its own prompt --
    proving per-request outputs don't cross-contaminate when multiple requests are
    driven through one shared continuous-batching loop."""

    def __init__(self, eos_id=FakeTokenizer.eos_token_id, reply=None):
        self.eos_id = eos_id
        self.reply = reply
        self._seqs: dict[int, Sequence] = {}
        self._ids = itertools.count()
        self.seen_params: list[SamplingParams] = []

    def add_request(self, prompt_ids, params: SamplingParams | None = None) -> int:
        seq_id = next(self._ids)
        params = params or SamplingParams()
        self.seen_params.append(params)
        self._seqs[seq_id] = Sequence(seq_id, list(prompt_ids), params)
        return seq_id

    def step(self) -> None:
        for seq in self._seqs.values():
            if seq.status is Status.FINISHED:
                continue
            if self.reply is not None:
                seq.output_ids.append(self.reply[len(seq.output_ids)])
            else:
                seq.output_ids.append((seq.last_token + 1) % 128)
            seq.status = Status.FINISHED if seq.is_finished(self.eos_id) else Status.RUNNING

    def sequence(self, seq_id: int) -> Sequence:
        return self._seqs[seq_id]

    def forget(self, seq_id: int) -> None:
        del self._seqs[seq_id]


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def http_client(engine):
    app = create_app(engine, FakeTokenizer(), served_model_name="fake-qwen3")
    with TestClient(app) as c:
        yield c


def _jsonl(lines: list[dict]) -> bytes:
    return ("\n".join(json.dumps(line) for line in lines) + "\n").encode("utf-8")


def _upload(http_client, lines: list[dict]) -> str:
    resp = http_client.post(
        "/v1/files",
        files={"file": ("batch.jsonl", _jsonl(lines), "application/jsonl")},
        data={"purpose": "batch"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


def test_batch_threads_sampling_extensions_for_both_endpoints(http_client, engine):
    input_file_id = _upload(
        http_client,
        [
            {
                "custom_id": "chat",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "fake-qwen3",
                    "messages": [{"role": "user", "content": "a"}],
                    "top_k": 50,
                    "repetition_penalty": 1.1,
                    "max_tokens": 1,
                },
            },
            {
                "custom_id": "completion",
                "method": "POST",
                "url": "/v1/completions",
                "body": {
                    "model": "fake-qwen3",
                    "prompt": "b",
                    "top_k": 7,
                    "repetition_penalty": 1.2,
                    "max_tokens": 1,
                },
            },
        ],
    )
    response = http_client.post(
        "/v1/batches",
        json={"input_file_id": input_file_id, "endpoint": "/v1/chat/completions"},
    )
    assert response.status_code == 200, response.text
    assert [(p.top_k, p.repetition_penalty) for p in engine.seen_params] == [
        (50, pytest.approx(1.1)),
        (7, pytest.approx(1.2)),
    ]


def test_batch_of_mixed_completions_prompts_preserves_order(http_client):
    """The core ask (issue #41): a JSONL of mixed prompts in -> completions out, in
    the SAME order, each request's own output."""
    input_file_id = _upload(http_client, [
        {"custom_id": "req-1", "method": "POST", "url": "/v1/completions",
         "body": {"model": "fake-qwen3", "prompt": "ab", "max_tokens": 3}},
        {"custom_id": "req-2", "method": "POST", "url": "/v1/completions",
         "body": {"model": "fake-qwen3", "prompt": "X", "max_tokens": 1}},
        {"custom_id": "req-3", "method": "POST", "url": "/v1/completions",
         "body": {"model": "fake-qwen3", "prompt": "mno", "max_tokens": 2}},
    ])

    batch_resp = http_client.post("/v1/batches", json={
        "input_file_id": input_file_id, "endpoint": "/v1/completions",
    })
    assert batch_resp.status_code == 200, batch_resp.text
    batch = batch_resp.json()
    assert batch["status"] == "completed"
    assert batch["request_counts"] == {"total": 3, "completed": 3, "failed": 0}

    retrieved = http_client.get(f"/v1/batches/{batch['id']}")
    assert retrieved.status_code == 200
    assert retrieved.json()["output_file_id"] == batch["output_file_id"]

    content_resp = http_client.get(f"/v1/files/{batch['output_file_id']}/content")
    assert content_resp.status_code == 200
    out_lines = [json.loads(line) for line in content_resp.text.splitlines()]

    assert [line["custom_id"] for line in out_lines] == ["req-1", "req-2", "req-3"]
    assert [line["error"] for line in out_lines] == [None, None, None]
    texts = [line["response"]["body"]["choices"][0]["text"] for line in out_lines]
    assert texts == ["cde", "Y", "pq"]


def test_batch_chat_completions_endpoint(http_client):
    input_file_id = _upload(http_client, [
        {"custom_id": "c1", "url": "/v1/chat/completions",
         "body": {"model": "fake-qwen3", "messages": [{"role": "user", "content": "hi"}],
                  "max_tokens": 2}},
    ])
    batch = http_client.post("/v1/batches", json={
        "input_file_id": input_file_id, "endpoint": "/v1/chat/completions",
    }).json()

    out = http_client.get(f"/v1/files/{batch['output_file_id']}/content").text
    line = json.loads(out.strip())
    assert line["custom_id"] == "c1"
    assert line["response"]["body"]["choices"][0]["message"]["content"] == "jk"


def test_batch_chat_separates_prefilled_reasoning_from_content():
    tokenizer = ThinkingTextTokenizer("plan the rhyme</think>Finished lyric")
    engine = FakeEngine(eos_id=999, reply=[0, 999])
    app = create_app(engine, tokenizer, served_model_name="fake-lfm")
    with TestClient(app) as client:
        input_file_id = _upload(client, [
            {"custom_id": "reasoning-1", "url": "/v1/chat/completions",
             "body": {"model": "fake-lfm",
                      "messages": [{"role": "user", "content": "write a lyric"}]}},
        ])
        batch = client.post("/v1/batches", json={
            "input_file_id": input_file_id, "endpoint": "/v1/chat/completions",
        }).json()
        out = client.get(f"/v1/files/{batch['output_file_id']}/content").text

    message = json.loads(out)["response"]["body"]["choices"][0]["message"]
    assert message["reasoning_content"] == "plan the rhyme"
    assert message["content"] == "Finished lyric"


def test_batch_line_error_does_not_fail_whole_batch(http_client):
    input_file_id = _upload(http_client, [
        {"custom_id": "ok", "url": "/v1/completions",
         "body": {"model": "fake-qwen3", "prompt": "A", "max_tokens": 1}},
        {"custom_id": "bad", "url": "/v1/embeddings",
         "body": {"model": "fake-qwen3", "input": "nope"}},
    ])
    batch = http_client.post("/v1/batches", json={
        "input_file_id": input_file_id, "endpoint": "/v1/completions",
    }).json()

    assert batch["request_counts"] == {"total": 2, "completed": 1, "failed": 1}
    out_lines = [json.loads(line) for line in
                http_client.get(f"/v1/files/{batch['output_file_id']}/content").text.splitlines()]
    by_id = {line["custom_id"]: line for line in out_lines}
    assert by_id["ok"]["error"] is None
    assert by_id["ok"]["response"]["body"]["choices"][0]["text"] == "B"
    assert by_id["bad"]["response"] is None
    assert by_id["bad"]["error"] is not None


def test_create_batch_missing_input_file_404(http_client):
    resp = http_client.post("/v1/batches", json={
        "input_file_id": "file-does-not-exist", "endpoint": "/v1/completions",
    })
    assert resp.status_code == 404


def test_retrieve_unknown_batch_404(http_client):
    assert http_client.get("/v1/batches/does-not-exist").status_code == 404


def test_retrieve_file_metadata(http_client):
    input_file_id = _upload(http_client, [
        {"custom_id": "x", "url": "/v1/completions",
         "body": {"model": "fake-qwen3", "prompt": "A", "max_tokens": 1}},
    ])
    meta = http_client.get(f"/v1/files/{input_file_id}").json()
    assert meta["id"] == input_file_id
    assert meta["purpose"] == "batch"


def test_batch_chat_completions_line_with_tools_round_trips_tool_call():
    """The semantic conflict this branch had to resolve: a `/v1/chat/completions`
    batch line carrying `tools` (issue #40) must extract `tool_calls` in the batch
    output exactly like a live HTTP request would (`test_api.py`'s
    `test_tool_call_round_trip_via_hermes_parser`, run through `/v1/batches`
    instead of a direct POST)."""
    tool_call_text = (
        '<tool_call>\n'
        '{"name": "get_weather", "arguments": {"location": "San Francisco"}}\n'
        '</tool_call>'
    )
    tokenizer = FixedTextTokenizer(tool_call_text)
    engine = FakeEngine(eos_id=999, reply=[0, 999])
    app = create_app(engine, tokenizer, served_model_name="fake-qwen3")
    with TestClient(app) as client:
        input_file_id = _upload(client, [
            {"custom_id": "tool-1", "url": "/v1/chat/completions",
             "body": {"model": "fake-qwen3",
                      "messages": [{"role": "user", "content": "weather in SF?"}],
                      "tools": [WEATHER_TOOL]}},
        ])
        batch = client.post("/v1/batches", json={
            "input_file_id": input_file_id, "endpoint": "/v1/chat/completions",
        }).json()
        out = client.get(f"/v1/files/{batch['output_file_id']}/content").text

    line = json.loads(out.strip())
    assert line["custom_id"] == "tool-1"
    message = line["response"]["body"]["choices"][0]["message"]
    assert line["response"]["body"]["choices"][0]["finish_reason"] == "tool_calls"
    call = message["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"location": "San Francisco"}


def test_batch_lfm2_native_tool_call_uses_selected_parser():
    tool_call_text = (
        "<|tool_call_start|>"
        "[get_weather(location='Chicago')]"
        "<|tool_call_end|>"
    )
    tokenizer = FixedTextTokenizer(tool_call_text)
    engine = FakeEngine(eos_id=999, reply=[0, 999])
    app = create_app(
        engine, tokenizer, served_model_name="fake-lfm", tool_parser="lfm2"
    )
    with TestClient(app) as client:
        input_file_id = _upload(
            client,
            [
                {
                    "custom_id": "lfm-tool-1",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "fake-lfm",
                        "messages": [{"role": "user", "content": "weather in Chicago?"}],
                        "tools": [WEATHER_TOOL],
                    },
                }
            ],
        )
        batch = client.post(
            "/v1/batches",
            json={
                "input_file_id": input_file_id,
                "endpoint": "/v1/chat/completions",
            },
        ).json()
        output = client.get(f"/v1/files/{batch['output_file_id']}/content").text

    choice = json.loads(output)["response"]["body"]["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"location": "Chicago"}


def test_batch_mixes_tool_and_plain_chat_lines_without_cross_contamination():
    """A batch of mixed lines -- one plain chat line, one `tools`-forced line --
    must keep each line's tool-handling independent (the `forced`/`wants_tools`
    flags are per-job, not shared app-level state)."""
    tokenizer = FakeTokenizer()
    engine = FakeEngine()
    app = create_app(engine, tokenizer, served_model_name="fake-qwen3")
    with TestClient(app) as client:
        input_file_id = _upload(client, [
            {"custom_id": "plain", "url": "/v1/chat/completions",
             "body": {"model": "fake-qwen3", "messages": [{"role": "user", "content": "hi"}],
                      "max_tokens": 2}},
            {"custom_id": "with-tools", "url": "/v1/chat/completions",
             "body": {"model": "fake-qwen3",
                      "messages": [{"role": "user", "content": "hi"}],
                      "tools": [WEATHER_TOOL], "tool_choice": "none", "max_tokens": 2}},
        ])
        batch = client.post("/v1/batches", json={
            "input_file_id": input_file_id, "endpoint": "/v1/chat/completions",
        }).json()
        out_lines = [json.loads(line) for line in
                    client.get(f"/v1/files/{batch['output_file_id']}/content").text.splitlines()]

    by_id = {line["custom_id"]: line for line in out_lines}
    assert by_id["plain"]["response"]["body"]["choices"][0]["message"]["content"] == "jk"
    assert by_id["with-tools"]["response"]["body"]["choices"][0]["message"]["content"] == "jk"
    assert by_id["with-tools"]["response"]["body"]["choices"][0]["message"].get("tool_calls") \
        is None
