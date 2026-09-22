# SPDX-License-Identifier: MIT
"""FastAPI app exposing an OpenAI-compatible surface over an `LLMEngine`.

`/v1/chat/completions` and `/v1/completions` support both streaming (SSE, the
`text/event-stream` framing the `openai` client expects) and non-streaming
responses. Chat formatting goes through the tokenizer's own `apply_chat_template`
(Jinja2, sandboxed by `transformers` itself via
`jinja2.sandbox.ImmutableSandboxedEnvironment`), not a hardcoded template, so
tool/template formatting comes free per model. `chat_template` optionally overrides
the model's embedded template (a Jinja source string) with a caller-supplied one --
same mechanism vLLM's `--chat-template` uses, and the same sandboxed environment, so
a custom template gets no more code-execution power than the model's own.

`/v1/files` + `/v1/batches` (issue #41) mount alongside these on the same
`EngineWorker`, so offline batch traffic and live HTTP traffic share one
continuous-batching loop instead of contending for the engine.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from ..structured import GrammarCompilerCache
from ..tool_calls import forced_tool_schema
from .batches import register_batch_routes
from .request_helpers import (
    build_logit_processors,
    chat_prompt_ids,
    extract_images_from_messages,
    forced_tool_message,
    parsed_tool_message,
    sampling_params,
    tool_choice_forced,
)
from .reasoning import prompt_prefills_thinking, split_reasoning_content
from .runtime import Done, EngineWorker
from .schemas import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    CompletionChunk,
    CompletionChunkChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    DeltaMessage,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    FunctionCallDelta,
    ModelCard,
    ModelList,
    NamedToolChoice,
    RerankDocument,
    RerankRequest,
    RerankResponse,
    RerankResult,
    ToolCallDelta,
    UsageInfo,
)


def create_app(
    engine,
    tokenizer,
    *,
    served_model_name: str,
    chat_template: str | None = None,
    tool_parser: str = "hermes",
    stats=None,
    idle_coalesce_ms: float = 0.0,
    idle_coalesce_target: int | None = None,
) -> FastAPI:
    """`engine` needs `.add_request`, `.step`, `.sequence`, `.forget`, `.eos_id`
    (an `LLMEngine`, or a test double with the same seam). `tokenizer` needs
    `.apply_chat_template`, `.encode`, `.decode` (an HF `PreTrainedTokenizerBase`).
    `chat_template`, if given, overrides the tokenizer's own embedded chat template
    (e.g. loaded from a file by the `server.py` CLI's `--chat-template` flag).
    `tool_parser` (issue #40) names the `superl8serve.tool_calls` parser used to
    extract `tool_calls` from free-form generation when `tool_choice` is `"auto"`
    (the default once `tools` is set) -- irrelevant for a forced `tool_choice`,
    which is schema-constrained instead (see `request_helpers.forced_tool_message`)."""
    from .. import __version__

    app = FastAPI(title="superl8-serve", version=__version__)
    # Telemetry (issue #182). Create a collector if the caller didn't supply one so
    # `GET /metrics` always works; wire it to the engine (KV cache) and the worker
    # (per-request TTFT / latency). Hot-loop recording is sync-free -- see metrics.py.
    if stats is None:
        from ..metrics import StatsCollector

        stats = StatsCollector()
    stats.attach_engine(engine)
    engine.stats = stats
    worker = EngineWorker(
        engine,
        stats=stats,
        idle_coalesce_ms=idle_coalesce_ms,
        idle_coalesce_target=idle_coalesce_target,
    )
    grammars = GrammarCompilerCache()
    # Context-window budget used to clamp `max_tokens` and reject over-length prompts
    # (pre-auth DoS hardening S1/S2). `None` on a test double without the attribute.
    max_len = getattr(engine, "max_len", None)

    def _guard_prompt_len(prompt_ids: list[int]) -> None:
        if max_len is not None and len(prompt_ids) > max_len:
            raise HTTPException(
                status_code=400,
                detail=f"prompt is {len(prompt_ids)} tokens, exceeds max_len {max_len}",
            )

    @app.get("/metrics")
    async def metrics() -> dict:
        """Live serving telemetry as JSON: running/waiting, prefill/decode tok/s,
        KV-cache usage %, VRAM/GPU, TTFT/ITL/p50/p99 latency, lifetime counters,
        uptime, and the startup model/config banner. Consumed by `superl8serve-top`."""
        return stats.snapshot()

    def _logit_processors(response_format, grammar):
        return build_logit_processors(grammars, tokenizer, response_format, grammar)

    @app.get("/v1/models")
    async def list_models() -> ModelList:
        return ModelList(data=[ModelCard(id=served_model_name)])

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        wants_tools = bool(req.tools) and req.tool_choice != "none"
        tools_wire = [t.model_dump() for t in req.tools] if wants_tools else None

        logit_processors = _logit_processors(req.response_format, req.grammar)
        forced = wants_tools and tool_choice_forced(req.tool_choice)
        if forced:
            tool_choice = (
                req.tool_choice.model_dump()
                if isinstance(req.tool_choice, NamedToolChoice)
                else req.tool_choice
            )
            _, schema = forced_tool_schema(tools_wire, tool_choice)
            logit_processors = [grammars.for_json_schema(tokenizer, schema)]

        prompt_ids = chat_prompt_ids(tokenizer, req.messages, chat_template, tools=tools_wire)
        thinking_prefilled = (
            not forced
            and req.response_format is None
            and req.grammar is None
            and prompt_prefills_thinking(tokenizer, prompt_ids)
        )
        _guard_prompt_len(prompt_ids)
        params = sampling_params(
            req.temperature, req.top_p, req.max_tokens, logit_processors,
            top_k=req.top_k, repetition_penalty=req.repetition_penalty,
            max_len=max_len, prompt_len=len(prompt_ids),
        )

        # Issue #151: extract and preprocess image_url content parts
        image_results = extract_images_from_messages(req.messages)
        pixel_values = image_results[0]["pixel_values"] if image_results else None
        image_grid_thw = (
            image_results[0].get("image_grid_thw") if image_results else None
        )

        if req.stream:
            return StreamingResponse(
                _chat_stream(
                    worker, tokenizer, prompt_ids, params, req.model,
                    thinking_prefilled=thinking_prefilled,
                    forced=forced,
                    wants_tools=wants_tools,
                    tool_parser=tool_parser,
                    pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                ),
                media_type="text/event-stream",
            )

        output_ids, reason = await _generate(
            worker, prompt_ids, params,
            pixel_values=pixel_values, image_grid_thw=image_grid_thw,
        )
        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        split = split_reasoning_content(text, prompt_prefilled=thinking_prefilled)

        if forced:
            message, reason = forced_tool_message(split.content)
        elif wants_tools:
            message, reason = parsed_tool_message(split.content, reason, tool_parser)
        else:
            message = ChatMessage(role="assistant", content=split.content)
        message.reasoning_content = split.reasoning_content

        return ChatCompletionResponse(
            model=req.model,
            choices=[ChatCompletionResponseChoice(message=message, finish_reason=reason)],
            usage=UsageInfo(
                prompt_tokens=len(prompt_ids),
                completion_tokens=len(output_ids),
                total_tokens=len(prompt_ids) + len(output_ids),
            ),
        )

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
        prompt_ids = tokenizer.encode(prompt)
        _guard_prompt_len(prompt_ids)
        params = sampling_params(
            req.temperature,
            req.top_p,
            req.max_tokens,
            _logit_processors(req.response_format, req.grammar),
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            max_len=max_len,
            prompt_len=len(prompt_ids),
        )

        if req.stream:
            return StreamingResponse(
                _completion_stream(worker, tokenizer, prompt_ids, params, req.model),
                media_type="text/event-stream",
            )

        output_ids, reason = await _generate(worker, prompt_ids, params)
        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        return CompletionResponse(
            model=req.model,
            choices=[CompletionResponseChoice(text=text, finish_reason=reason)],
            usage=UsageInfo(
                prompt_tokens=len(prompt_ids),
                completion_tokens=len(output_ids),
                total_tokens=len(prompt_ids) + len(output_ids),
            ),
        )

    @app.post("/v1/embeddings")
    async def embeddings(req: EmbeddingRequest):
        inputs = [req.input] if isinstance(req.input, str) else req.input
        all_data: list[EmbeddingData] = []
        total_tokens = 0
        loop = asyncio.get_running_loop()
        for i, text in enumerate(inputs):
            prompt_ids = tokenizer.encode(text)
            _guard_prompt_len(prompt_ids)
            # encode blocks on the worker thread -- keep it off the event loop.
            embedding = await loop.run_in_executor(None, worker.encode, prompt_ids)
            all_data.append(EmbeddingData(embedding=embedding, index=i))
            total_tokens += len(prompt_ids)
        if req.encoding_format != "float":
            raise NotImplementedError(
                f"encoding_format={req.encoding_format!r} is not supported; only 'float' is"
            )
        return EmbeddingResponse(
            model=req.model,
            data=all_data,
            usage=UsageInfo(
                prompt_tokens=total_tokens, completion_tokens=0, total_tokens=total_tokens
            ),
        )

    @app.post("/v1/rerank")
    async def rerank(req: RerankRequest):
        results: list[RerankResult] = []
        total_tokens = 0
        loop = asyncio.get_running_loop()
        for i, doc in enumerate(req.documents):
            text = f"{req.query}\n{doc}"
            prompt_ids = tokenizer.encode(text)
            _guard_prompt_len(prompt_ids)
            embedding = await loop.run_in_executor(None, worker.encode, prompt_ids)
            score = 1.0 / (1.0 + math.exp(-embedding[0]))
            results.append(
                RerankResult(index=i, relevance_score=score, document=RerankDocument(text=doc))
            )
            total_tokens += len(prompt_ids)
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        if req.top_n is not None:
            results = results[: req.top_n]
        return RerankResponse(
            model=req.model,
            data=results,
            usage=UsageInfo(
                prompt_tokens=total_tokens, completion_tokens=0, total_tokens=total_tokens
            ),
        )

    register_batch_routes(
        app, worker, tokenizer, grammars, chat_template=chat_template,
        tool_parser=tool_parser, max_len=max_len,
    )

    return app


async def _generate(
    worker: EngineWorker,
    prompt_ids: list[int],
    params,
    pixel_values: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
) -> tuple[list[int], str]:
    ids: list[int] = []
    reason = "stop"
    async for item in worker.stream(
        prompt_ids, params, pixel_values=pixel_values, image_grid_thw=image_grid_thw
    ):
        if isinstance(item, Done):
            reason = item.reason
        else:
            ids.append(item)
    return ids, reason


async def _chat_stream(
    worker, tokenizer, prompt_ids, params, model,
    thinking_prefilled: bool = False,
    forced: bool = False,
    wants_tools: bool = False,
    tool_parser: str = "hermes",
    pixel_values: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
) -> AsyncIterator[str]:
    if wants_tools:
        async for chunk in _tool_chat_stream(
            worker, tokenizer, prompt_ids, params, model,
            thinking_prefilled=thinking_prefilled,
            forced=forced,
            tool_parser=tool_parser,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        ):
            yield chunk
        return

    first, prev_reasoning, prev_content, ids = True, "", "", []

    def chunks_for(text: str, *, final: bool = False) -> list[ChatCompletionChunk]:
        nonlocal first, prev_reasoning, prev_content
        split = split_reasoning_content(
            text, prompt_prefilled=thinking_prefilled, final=final
        )
        chunks = []
        reasoning = split.reasoning_content or ""
        reasoning_delta = reasoning[len(prev_reasoning):]
        content_delta = split.content[len(prev_content):]
        prev_reasoning, prev_content = reasoning, split.content
        for field, delta in (
            ("reasoning_content", reasoning_delta),
            ("content", content_delta),
        ):
            if not delta:
                continue
            kwargs = {field: delta, "role": "assistant" if first else None}
            chunks.append(
                ChatCompletionChunk(
                    model=model,
                    choices=[ChatCompletionChunkChoice(delta=DeltaMessage(**kwargs))],
                )
            )
            first = False
        return chunks

    async for item in worker.stream(
        prompt_ids, params, pixel_values=pixel_values, image_grid_thw=image_grid_thw
    ):
        if isinstance(item, Done):
            text = tokenizer.decode(ids, skip_special_tokens=True)
            for pending in chunks_for(text, final=True):
                yield f"data: {pending.model_dump_json()}\n\n"
            choice = ChatCompletionChunkChoice(delta=DeltaMessage(), finish_reason=item.reason)
            chunk = ChatCompletionChunk(model=model, choices=[choice])
            yield f"data: {chunk.model_dump_json()}\n\n"
            break
        ids.append(item)
        text = tokenizer.decode(ids, skip_special_tokens=True)
        for chunk in chunks_for(text):
            yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


async def _tool_chat_stream(
    worker, tokenizer, prompt_ids, params, model,
    *,
    thinking_prefilled: bool,
    forced: bool,
    tool_parser: str,
    pixel_values: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
) -> AsyncIterator[str]:
    """Buffer tool-bearing final content so native calls never leak as text.

    Reasoning remains incremental. Final content is held until generation ends,
    when it can be classified as an ordinary answer or converted into OpenAI
    tool-call deltas. The no-tools stream keeps its existing incremental path.
    """
    first, prev_reasoning, ids = True, "", []

    def chunk_for_delta(**kwargs) -> ChatCompletionChunk:
        nonlocal first
        if first:
            kwargs["role"] = "assistant"
        chunk = ChatCompletionChunk(
            model=model,
            choices=[ChatCompletionChunkChoice(delta=DeltaMessage(**kwargs))],
        )
        first = False
        return chunk

    def reasoning_chunk(text: str, *, final: bool) -> ChatCompletionChunk | None:
        nonlocal prev_reasoning
        split = split_reasoning_content(
            text, prompt_prefilled=thinking_prefilled, final=final
        )
        reasoning = split.reasoning_content or ""
        delta = reasoning[len(prev_reasoning):]
        prev_reasoning = reasoning
        return chunk_for_delta(reasoning_content=delta) if delta else None

    async for item in worker.stream(
        prompt_ids, params, pixel_values=pixel_values, image_grid_thw=image_grid_thw
    ):
        if not isinstance(item, Done):
            ids.append(item)
            text = tokenizer.decode(ids, skip_special_tokens=True)
            chunk = reasoning_chunk(text, final=False)
            if chunk is not None:
                yield f"data: {chunk.model_dump_json()}\n\n"
            continue

        text = tokenizer.decode(ids, skip_special_tokens=True)
        chunk = reasoning_chunk(text, final=True)
        if chunk is not None:
            yield f"data: {chunk.model_dump_json()}\n\n"
        split = split_reasoning_content(
            text, prompt_prefilled=thinking_prefilled, final=True
        )
        if forced:
            message, reason = forced_tool_message(split.content)
        else:
            message, reason = parsed_tool_message(split.content, item.reason, tool_parser)

        if message.content:
            content_chunk = chunk_for_delta(content=message.content)
            yield f"data: {content_chunk.model_dump_json()}\n\n"
        if message.tool_calls:
            tool_deltas = [
                ToolCallDelta(
                    index=index,
                    id=call.id,
                    type=call.type,
                    function=FunctionCallDelta(
                        name=call.function.name,
                        arguments=call.function.arguments,
                    ),
                )
                for index, call in enumerate(message.tool_calls)
            ]
            tool_chunk = chunk_for_delta(tool_calls=tool_deltas)
            yield f"data: {tool_chunk.model_dump_json()}\n\n"

        choice = ChatCompletionChunkChoice(delta=DeltaMessage(), finish_reason=reason)
        terminal = ChatCompletionChunk(model=model, choices=[choice])
        yield f"data: {terminal.model_dump_json()}\n\n"
        break
    yield "data: [DONE]\n\n"


async def _completion_stream(worker, tokenizer, prompt_ids, params, model) -> AsyncIterator[str]:
    prev_text, ids = "", []
    async for item in worker.stream(prompt_ids, params):
        if isinstance(item, Done):
            chunk = CompletionChunk(
                model=model, choices=[CompletionChunkChoice(text="", finish_reason=item.reason)]
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
            break
        ids.append(item)
        text = tokenizer.decode(ids, skip_special_tokens=True)
        delta = text[len(prev_text) :]
        prev_text = text
        if delta:
            chunk = CompletionChunk(model=model, choices=[CompletionChunkChoice(text=delta)])
            yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"
