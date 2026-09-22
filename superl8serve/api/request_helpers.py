# SPDX-License-Identifier: MIT
"""Request-building + tool-call helpers shared by the live streaming endpoints
(`app.py`) and the offline `/v1/batches` endpoint (`batches.py`), so a batch line
builds its prompt / sampling-params / tool-call handling exactly the same way a
live HTTP request does (issue #40's tools-aware chat formatting included)."""

from __future__ import annotations

import json

import torch

from ..engine.sequence import SamplingParams
from ..structured import GrammarCompilerCache
from ..tool_calls import get_tool_call_parser, parse_forced_tool_call
from .schemas import ChatMessage, FunctionCall, NamedToolChoice, ResponseFormat, ToolCall


def message_dict(m: ChatMessage) -> dict:
    """Like `m.model_dump(exclude_none=True)`, except a tool-calling assistant
    message's `tool_calls[].function.arguments` is parsed back from the OpenAI wire
    shape (a JSON string) into a mapping -- chat templates (e.g. Qwen's) render
    `tool_call.function.arguments` as a mapping, not a string."""
    d = m.model_dump(exclude_none=True, exclude={"tool_calls"})
    if m.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                },
            }
            for tc in m.tool_calls
        ]
    return d


def extract_images_from_messages(
    messages: list[ChatMessage],
) -> list[dict[str, torch.Tensor]]:
    """Extract and preprocess ``image_url`` content parts from chat messages.

    For each message whose ``content`` is a list of ``ContentPart``, any part with
    ``type == "image_url"`` has its URL fetched (HTTP or base64 data URI) and
    preprocessed through ``preprocess_qwen2_5_vl``.

    The multimodal backend (which needs torchvision) is imported LAZILY, and only when
    an image is actually present — so text-only requests, and this whole module, import
    cleanly even in a build without torchvision. (Salvaged from the superseded #151
    vision attempt; the merged #154 imported it at module top, which broke
    ``request_helpers`` import entirely when torchvision was absent.)

    Returns:
        A list of preprocessing result dicts (``pixel_values``, ``image_grid_thw``),
        one per image in message order.
    """
    has_images = any(
        isinstance(msg.content, list)
        and any(p.type == "image_url" and p.image_url is not None for p in msg.content)
        for msg in messages
    )
    if not has_images:
        return []
    from ..multimodal import fetch_image, preprocess_qwen2_5_vl

    results: list[dict[str, torch.Tensor]] = []
    for msg in messages:
        if isinstance(msg.content, list):
            for part in msg.content:
                if part.type == "image_url" and part.image_url is not None:
                    pil = fetch_image(part.image_url.url)
                    results.append(preprocess_qwen2_5_vl(pil))
    return results


def chat_prompt_ids(
    tokenizer,
    messages: list[ChatMessage],
    chat_template: str | None = None,
    tools: list[dict] | None = None,
) -> list[int]:
    kwargs = {"tokenize": True, "add_generation_prompt": True}
    if chat_template is not None:
        kwargs["chat_template"] = chat_template
    if tools:
        kwargs["tools"] = tools
    out = tokenizer.apply_chat_template([message_dict(m) for m in messages], **kwargs)
    # transformers >=5 returns a BatchEncoding (dict with "input_ids") from
    # apply_chat_template(tokenize=True); older versions returned a flat list[int].
    # Normalize to a flat list[int] either way.
    if hasattr(out, "input_ids") or isinstance(out, dict):
        out = out["input_ids"]
    if out and isinstance(out[0], list):  # some versions nest per-conversation
        out = out[0]
    return out


def build_logit_processors(
    grammars: GrammarCompilerCache,
    tokenizer,
    response_format: ResponseFormat | None,
    grammar: str | None,
) -> list | None:
    """Builds a request's structured-output `LogitsProcessor` (issue #39), if any --
    `grammar` wins over `response_format` when both are given."""
    if grammar is not None:
        return [grammars.for_grammar(tokenizer, grammar)]
    if response_format is None or response_format.type == "text":
        return None
    if response_format.type == "json_object":
        return [grammars.for_json_object(tokenizer)]
    schema = response_format.json_schema.schema_ if response_format.json_schema else None
    return [grammars.for_json_schema(tokenizer, schema or {})]


def sampling_params(
    temperature: float,
    top_p: float,
    max_tokens: int | None,
    logit_processors: list | None = None,
    *,
    top_k: int | None = None,
    repetition_penalty: float = 1.0,
    max_len: int | None = None,
    prompt_len: int = 0,
) -> SamplingParams:
    resolved = max_tokens or 16
    # Clamp to the KV room this prompt leaves (S1): an unclamped max_tokens decodes
    # past the slot's block range (block-table overflow / "no free blocks"), and the
    # diffusion path pre-allocates `num_prompt + max_tokens` KV up front -- a huge
    # max_tokens is an instant multi-TB alloc. At least 1 token so a full-context
    # prompt still produces output.
    if max_len is not None:
        resolved = min(resolved, max(1, max_len - prompt_len))
    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k or 0,
        repetition_penalty=repetition_penalty,
        max_tokens=resolved,
        logit_processors=logit_processors,
    )


def tool_choice_forced(tool_choice: str | NamedToolChoice | None) -> bool:
    return tool_choice == "required" or isinstance(tool_choice, NamedToolChoice)


def forced_tool_message(text: str) -> tuple[ChatMessage, str]:
    """`tool_choice="required"` / a named tool_choice: `text` is already
    schema-constrained JSON (issue #39's XGrammar backend, via `forced_tool_schema`),
    so no parser is needed -- just `json.loads`."""
    parsed = parse_forced_tool_call(text)
    function = FunctionCall(name=parsed.name, arguments=json.dumps(parsed.arguments))
    return ChatMessage(role="assistant", tool_calls=[ToolCall(function=function)]), "tool_calls"


def parsed_tool_message(text: str, reason: str, tool_parser: str) -> tuple[ChatMessage, str]:
    """`tool_choice="auto"` (the default once `tools` is set): `text` is free-form
    generation, so extract any `tool_calls` with the per-model parser named by
    `tool_parser`. Falls back to a plain content message if the model didn't end up
    calling a tool."""
    extracted = get_tool_call_parser(tool_parser).extract_tool_calls(text)
    if not extracted.tool_calls:
        return ChatMessage(role="assistant", content=text), reason
    tool_calls = [
        ToolCall(function=FunctionCall(name=c.name, arguments=json.dumps(c.arguments)))
        for c in extracted.tool_calls
    ]
    message = ChatMessage(role="assistant", content=extracted.content, tool_calls=tool_calls)
    return message, "tool_calls"
