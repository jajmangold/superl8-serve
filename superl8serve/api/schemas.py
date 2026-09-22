# SPDX-License-Identifier: MIT
"""Pydantic request/response models mirroring the OpenAI REST API shapes closely
enough that the `openai` Python client works unmodified against `base_url`."""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


class FunctionDefinition(BaseModel):
    name: str
    description: str | None = None
    parameters: dict = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    """An OpenAI `tools[]` entry. Threaded straight into the tokenizer's
    `apply_chat_template(tools=...)` (issue #40) -- HF's own tools-aware templates
    (Qwen, etc.) already expect exactly this `{"type": "function", "function": {...}}`
    shape, so no reshaping is needed between the request and the template."""

    type: Literal["function"] = "function"
    function: FunctionDefinition


class FunctionCall(BaseModel):
    name: str
    arguments: str  # JSON-encoded, per the OpenAI wire shape


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: _id("call"))
    type: Literal["function"] = "function"
    function: FunctionCall


class NamedToolChoice(BaseModel):
    """`tool_choice={"type": "function", "function": {"name": ...}}` -- forces
    that one function via the structured-output backend rather than free-form
    generation (issue #40)."""

    type: Literal["function"] = "function"
    function: dict


class ImageURL(BaseModel):
    """OpenAI `image_url` content part (issue #151) — either an HTTP(S) URL or a
    base64 data URI (``data:image/...;base64,...``)."""

    url: str
    detail: str | None = None       # OpenAI compat ("auto"/"low"/"high"); accepted, unused


class ContentPart(BaseModel):
    """OpenAI content part (issue #151): ``{"type": "text", "text": "..."}`` or
    ``{"type": "image_url", "image_url": {"url": "..."}}``."""

    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageURL | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentPart] | None = None
    reasoning_content: str | None = Field(default=None, exclude_if=lambda value: value is None)
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # role="tool": which call this is the result of


class JSONSchemaSpec(BaseModel):
    name: str | None = None
    schema_: dict | None = Field(default=None, alias="schema")
    strict: bool | None = None

    model_config = {"populate_by_name": True}


class ResponseFormat(BaseModel):
    """OpenAI `response_format` (issue #39): `json_schema` is compiled by XGrammar
    into a token-mask matcher and applied via the sampler's logit-processor hook
    (`superl8serve.structured`). `json_object` uses XGrammar's builtin JSON grammar
    (any valid JSON, unconstrained by a schema)."""

    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: JSONSchemaSpec | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = Field(default=None, ge=1)
    repetition_penalty: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    max_tokens: int | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    response_format: ResponseFormat | None = None
    # superl8-serve extension: a raw grammar (GBNF/EBNF), compiled by XGrammar the same
    # way as `response_format={"type": "json_schema"}`. Takes precedence if both are set.
    grammar: str | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: Literal["none", "auto", "required"] | NamedToolChoice | None = None


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = Field(default=None, ge=1)
    repetition_penalty: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    max_tokens: int = 16
    stream: bool = False
    stop: str | list[str] | None = None
    response_format: ResponseFormat | None = None
    grammar: str | None = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponseChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = None  # "tool_calls" when `message.tool_calls` is set


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: _id("chatcmpl"))
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo


class FunctionCallDelta(BaseModel):
    name: str | None = None
    arguments: str | None = None


class ToolCallDelta(BaseModel):
    index: int
    id: str | None = None
    type: Literal["function"] | None = None
    function: FunctionCallDelta | None = None


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = Field(default=None, exclude_if=lambda value: value is None)
    tool_calls: list[ToolCallDelta] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: _id("chatcmpl"))
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChunkChoice]


class CompletionResponseChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str | None = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: _id("cmpl"))
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionResponseChoice]
    usage: UsageInfo


class CompletionChunkChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str | None = None


class CompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: _id("cmpl"))
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChunkChoice]


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "superl8-serve"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class FileObject(BaseModel):
    """OpenAI `/v1/files` shape (issue #41): the input/output JSONL of a batch job."""

    id: str = Field(default_factory=lambda: _id("file"))
    object: Literal["file"] = "file"
    bytes: int
    created_at: int = Field(default_factory=lambda: int(time.time()))
    filename: str
    purpose: str


class BatchRequestCounts(BaseModel):
    total: int = 0
    completed: int = 0
    failed: int = 0


class CreateBatchRequest(BaseModel):
    input_file_id: str
    endpoint: Literal["/v1/chat/completions", "/v1/completions"]
    completion_window: str = "24h"
    metadata: dict | None = None


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: Literal["float", "base64"] = "float"


class EmbeddingData(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: list[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    usage: UsageInfo


class RerankDocument(BaseModel):
    text: str


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: RerankDocument | None = None


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: list[str]
    top_n: int | None = None


class RerankResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[RerankResult]
    model: str
    usage: UsageInfo


class Batch(BaseModel):
    """OpenAI `/v1/batches` shape (issue #41). This server has no background job
    queue, so `POST /v1/batches` runs the file to completion before returning --
    `status` is always `"completed"` by the time a `Batch` is handed back."""

    id: str = Field(default_factory=lambda: _id("batch"))
    object: Literal["batch"] = "batch"
    endpoint: str
    input_file_id: str
    completion_window: str
    status: Literal["completed", "failed"] = "completed"
    output_file_id: str | None = None
    error_file_id: str | None = None
    created_at: int = Field(default_factory=lambda: int(time.time()))
    in_progress_at: int | None = None
    completed_at: int | None = None
    failed_at: int | None = None
    request_counts: BatchRequestCounts = Field(default_factory=BatchRequestCounts)
    metadata: dict | None = None
