# SPDX-License-Identifier: MIT
"""OpenAI-compatible `/v1/files` + `/v1/batches` (issue #41): offline batch
inference for eval/dataset runs, without holding one HTTP connection open per
request. A batch's input is a JSONL file of individual `/v1/chat/completions` or
`/v1/completions` request bodies -- OpenAI's own Batch API shape
(`client.files.create(purpose="batch")` -> `client.batches.create(...)` -> poll
`client.batches.retrieve(...)` -> `client.files.content(output_file_id)`). Each
line is run through the SAME `EngineWorker` the streaming endpoints share, so
batch traffic and live request traffic interleave through one continuous-
batching loop instead of contending for the engine. A `/v1/chat/completions` line
gets the exact same `tools`/`tool_choice` handling (issue #40) as a live request --
`request_helpers` is the shared seam both go through.

This server has no background job queue, so unlike OpenAI's real (up-to-24h)
completion window, `POST /v1/batches` runs the whole file to completion before
returning -- fine for the eval/dataset-run use case this is for, not for
million-line jobs. `status` is therefore always `"completed"` by the time the
call returns; a malformed or unsupported line becomes an error entry on its own
output line (matching OpenAI's per-line error semantics) rather than failing
the whole batch.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from ..tool_calls import forced_tool_schema
from .request_helpers import (
    build_logit_processors,
    chat_prompt_ids,
    forced_tool_message,
    parsed_tool_message,
    sampling_params,
    tool_choice_forced,
)
from .reasoning import prompt_prefills_thinking, split_reasoning_content
from .runtime import Done, EngineWorker
from .schemas import (
    Batch,
    BatchRequestCounts,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CreateBatchRequest,
    FileObject,
    NamedToolChoice,
    UsageInfo,
    _id,
)

# Pre-auth request-size caps (S3). 32 MB matches the image-fetch cap in
# `superl8serve.multimodal` (kept as a local literal so this module stays import-light --
# importing `multimodal` would pull torch/PIL/torchvision into the batch path).
_MAX_UPLOAD_BYTES = 32 * 1024 * 1024      # per /v1/files upload
_MAX_BATCH_LINES = 50_000                 # lines admitted per /v1/batches submit
_MAX_STORED_FILES = 256                   # bound the in-memory _FileStore (FIFO evict)


class _FileStore:
    """In-memory `/v1/files`: content plus the metadata `FileObject` echoes back.

    Bounded to `max_files` entries with FIFO eviction so a long-lived server can't be
    grown without limit by repeated uploads (the store never expired anything before)."""

    def __init__(self, max_files: int = _MAX_STORED_FILES) -> None:
        self._content: dict[str, bytes] = {}
        self._meta: dict[str, FileObject] = {}
        self._max_files = max_files

    def put(self, content: bytes, *, filename: str, purpose: str) -> FileObject:
        obj = FileObject(bytes=len(content), filename=filename, purpose=purpose)
        self._content[obj.id] = content
        self._meta[obj.id] = obj
        while len(self._content) > self._max_files:
            oldest = next(iter(self._content))
            self._content.pop(oldest, None)
            self._meta.pop(oldest, None)
        return obj

    def content(self, file_id: str) -> bytes:
        if file_id not in self._content:
            raise HTTPException(status_code=404, detail=f"No such file: {file_id!r}")
        return self._content[file_id]

    def meta(self, file_id: str) -> FileObject:
        if file_id not in self._meta:
            raise HTTPException(status_code=404, detail=f"No such file: {file_id!r}")
        return self._meta[file_id]


@dataclass
class _Job:
    """One JSONL input line, submitted to the engine (or a parse/build error that
    keeps its place in line without stopping the rest of the batch)."""
    custom_id: str | None
    url: str | None = None
    model: str | None = None
    prompt_ids: list[int] | None = None
    queue: object | None = None
    forced: bool = False
    wants_tools: bool = False
    thinking_prefilled: bool = False
    error: str | None = None


def _guard_prompt_len(prompt_ids: list[int], max_len: int | None) -> None:
    if max_len is not None and len(prompt_ids) > max_len:
        raise ValueError(f"prompt length {len(prompt_ids)} exceeds max_len {max_len}")


def _build_request(url: str, body: dict, tokenizer, grammars, chat_template, tool_parser: str,
                   max_len: int | None = None):
    """Parses one batch line's `body` exactly the way the live HTTP endpoint for
    `url` would -- same schemas, same prompt building, same tools/structured-output
    handling."""
    if url == "/v1/chat/completions":
        req = ChatCompletionRequest.model_validate(body)
        wants_tools = bool(req.tools) and req.tool_choice != "none"
        tools_wire = [t.model_dump() for t in req.tools] if wants_tools else None

        logit_processors = build_logit_processors(
            grammars, tokenizer, req.response_format, req.grammar)
        forced = wants_tools and tool_choice_forced(req.tool_choice)
        if forced:
            tool_choice = (req.tool_choice.model_dump()
                          if isinstance(req.tool_choice, NamedToolChoice) else req.tool_choice)
            _, schema = forced_tool_schema(tools_wire, tool_choice)
            logit_processors = [grammars.for_json_schema(tokenizer, schema)]

        prompt_ids = chat_prompt_ids(tokenizer, req.messages, chat_template, tools=tools_wire)
        thinking_prefilled = (
            not forced
            and req.response_format is None
            and req.grammar is None
            and prompt_prefills_thinking(tokenizer, prompt_ids)
        )
        _guard_prompt_len(prompt_ids, max_len)
        params = sampling_params(req.temperature, req.top_p, req.max_tokens, logit_processors,
                                 top_k=req.top_k, repetition_penalty=req.repetition_penalty,
                                 max_len=max_len, prompt_len=len(prompt_ids))
        return req, prompt_ids, params, forced, wants_tools, thinking_prefilled
    if url == "/v1/completions":
        req = CompletionRequest.model_validate(body)
        prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
        prompt_ids = tokenizer.encode(prompt)
        _guard_prompt_len(prompt_ids, max_len)
        logit_processors = build_logit_processors(
            grammars, tokenizer, req.response_format, req.grammar)
        params = sampling_params(req.temperature, req.top_p, req.max_tokens, logit_processors,
                                 top_k=req.top_k, repetition_penalty=req.repetition_penalty,
                                 max_len=max_len, prompt_len=len(prompt_ids))
        return req, prompt_ids, params, False, False, False
    raise ValueError(f"unsupported batch endpoint: {url!r}")


def _response_body(url: str, model: str, text: str, prompt_ids: list[int],
                   output_ids: list[int], reason: str, *, forced: bool, wants_tools: bool,
                   thinking_prefilled: bool, tool_parser: str) -> dict:
    usage = UsageInfo(prompt_tokens=len(prompt_ids), completion_tokens=len(output_ids),
                      total_tokens=len(prompt_ids) + len(output_ids))
    if url == "/v1/chat/completions":
        split = split_reasoning_content(text, prompt_prefilled=thinking_prefilled)
        if forced:
            message, reason = forced_tool_message(split.content)
        elif wants_tools:
            message, reason = parsed_tool_message(split.content, reason, tool_parser)
        else:
            message = ChatMessage(role="assistant", content=split.content)
        message.reasoning_content = split.reasoning_content
        return ChatCompletionResponse(
            model=model,
            choices=[ChatCompletionResponseChoice(message=message, finish_reason=reason)],
            usage=usage,
        ).model_dump(mode="json")
    return CompletionResponse(
        model=model,
        choices=[CompletionResponseChoice(text=text, finish_reason=reason)],
        usage=usage,
    ).model_dump(mode="json")


def _submit_jobs(worker: EngineWorker, tokenizer, grammars, lines: list[str],
                 endpoint: str, chat_template, tool_parser: str,
                 max_len: int | None = None) -> list[_Job]:
    jobs = []
    for raw in lines:
        custom_id = None
        try:
            line = json.loads(raw)
            custom_id = line.get("custom_id")
            url = line.get("url") or endpoint
            req, prompt_ids, params, forced, wants_tools, thinking_prefilled = _build_request(
                url, line["body"], tokenizer, grammars, chat_template, tool_parser, max_len)
            jobs.append(_Job(custom_id, url=url, model=req.model, prompt_ids=prompt_ids,
                             queue=worker.submit(prompt_ids, params),
                             forced=forced, wants_tools=wants_tools,
                             thinking_prefilled=thinking_prefilled))
        except Exception as exc:   # noqa: BLE001 -- one bad line must not sink the batch
            jobs.append(_Job(custom_id, error=str(exc)))
    return jobs


async def _drain_jobs(jobs: list[_Job], tokenizer, tool_parser: str) -> tuple[list[str], int]:
    loop = asyncio.get_running_loop()
    out_lines, failed = [], 0
    for job in jobs:
        if job.error is not None:
            failed += 1
            out_lines.append(json.dumps({
                "id": _id("batch_req"), "custom_id": job.custom_id,
                "response": None, "error": {"message": job.error},
            }))
            continue

        ids: list[int] = []
        reason = "stop"
        while True:
            item = await loop.run_in_executor(None, job.queue.get)
            if isinstance(item, Done):
                reason = item.reason
                break
            ids.append(item)

        text = tokenizer.decode(ids, skip_special_tokens=True)
        body = _response_body(job.url, job.model, text, job.prompt_ids, ids, reason,
                              forced=job.forced, wants_tools=job.wants_tools,
                              thinking_prefilled=job.thinking_prefilled,
                              tool_parser=tool_parser)
        out_lines.append(json.dumps({
            "id": _id("batch_req"), "custom_id": job.custom_id,
            "response": {"status_code": 200, "body": body}, "error": None,
        }))
    return out_lines, failed


def register_batch_routes(app: FastAPI, worker: EngineWorker, tokenizer, grammars, *,
                          chat_template: str | None = None, tool_parser: str = "hermes",
                          max_len: int | None = None) -> None:
    """Mounts `/v1/files` + `/v1/batches` onto `app`, sharing `worker` (and so the
    underlying engine) with the streaming endpoints `create_app` already registered."""
    files = _FileStore()
    batches: dict[str, Batch] = {}

    @app.post("/v1/files")
    async def create_file(file: UploadFile = File(...), purpose: str = Form(...)) -> FileObject:
        # Read one byte past the cap so an oversized upload is rejected without ever
        # materialising the whole body in memory (S3).
        content = await file.read(_MAX_UPLOAD_BYTES + 1)
        if len(content) > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"file exceeds the {_MAX_UPLOAD_BYTES}-byte upload cap")
        return files.put(content, filename=file.filename or "upload.jsonl", purpose=purpose)

    @app.get("/v1/files/{file_id}")
    async def retrieve_file(file_id: str) -> FileObject:
        return files.meta(file_id)

    @app.get("/v1/files/{file_id}/content")
    async def file_content(file_id: str) -> Response:
        return Response(content=files.content(file_id), media_type="application/octet-stream")

    @app.get("/v1/batches/{batch_id}")
    async def retrieve_batch(batch_id: str) -> Batch:
        if batch_id not in batches:
            raise HTTPException(status_code=404, detail=f"No such batch: {batch_id!r}")
        return batches[batch_id]

    @app.post("/v1/batches")
    async def create_batch(req: CreateBatchRequest) -> Batch:
        input_bytes = files.content(req.input_file_id)
        lines = [ln for ln in input_bytes.decode("utf-8").splitlines() if ln.strip()]
        # This server runs the whole file to completion inline (no background queue),
        # so an unbounded line count is a pre-auth compute/memory DoS -- cap it (S3).
        if len(lines) > _MAX_BATCH_LINES:
            raise HTTPException(
                status_code=413,
                detail=f"batch has {len(lines)} lines, exceeds the {_MAX_BATCH_LINES} cap")

        jobs = _submit_jobs(worker, tokenizer, grammars, lines, req.endpoint, chat_template,
                            tool_parser, max_len)
        out_lines, failed = await _drain_jobs(jobs, tokenizer, tool_parser)
        output_file = files.put(
            ("\n".join(out_lines) + "\n").encode("utf-8") if out_lines else b"",
            filename="batch_output.jsonl", purpose="batch_output")

        now = int(time.time())
        batch = Batch(
            endpoint=req.endpoint, input_file_id=req.input_file_id,
            completion_window=req.completion_window, status="completed",
            output_file_id=output_file.id, created_at=now, in_progress_at=now,
            completed_at=now,
            request_counts=BatchRequestCounts(
                total=len(lines), completed=len(lines) - failed, failed=failed),
            metadata=req.metadata,
        )
        batches[batch.id] = batch
        return batch
