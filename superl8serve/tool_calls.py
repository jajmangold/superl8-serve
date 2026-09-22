# SPDX-License-Identifier: MIT
"""OpenAI `tools`/`tool_choice` support (issue #40).

Two paths, matching the request's `tool_choice`:

- `"auto"` (the default once `tools` is set) or unset: `tools` is formatted into
  the prompt via the tokenizer's own `apply_chat_template(tools=...)` (HF's
  tools-aware templates, e.g. Qwen's, already render these; nothing
  superl8-serve-specific is needed there). The model free-generates, and a
  per-model `ToolCallParser` (registered in `TOOL_CALL_PARSERS`) extracts any
  `tool_calls` from the raw text afterwards.
- `"required"` or a named tool_choice: no parser needed, because generation
  itself is constrained. `forced_tool_schema` builds a JSON schema of
  `{"name": ..., "arguments": {...}}` (narrowed to the eligible tool(s)) and the
  caller compiles it through the existing XGrammar structured-output backend
  (`superl8serve.structured`, issue #39) the same way `response_format=
  {"type": "json_schema"}` does. The resulting text is guaranteed schema-valid
  JSON, so `parse_forced_tool_call` just does `json.loads`.

`HermesToolCallParser` implements the `<tool_call>{"name": ..., "arguments":
{...}}</tool_call>` convention Qwen 2.5/3 (and most Hermes-templated instruct
models) use for native tool calling. Its shape was studied from vLLM's
Apache-2.0-licensed `hermes_tool_parser.py` (see NOTICE) -- this is an
independent reimplementation of the same tag convention, not a port of that
file's code. `LFM2ToolCallParser` implements Liquid's native Pythonic call-list
envelope with an AST whitelist and no evaluation of model-generated code.
`GemmaToolCallParser` handles Gemma4's bare, untagged `name{key: value, ...}`
shape the same way (issue #457) -- Gemma has no native function-calling
tokens and doesn't reliably follow the Hermes tag convention even when a
tools-aware chat template instructs it to.
"""
from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field


@dataclass
class ParsedToolCall:
    name: str
    arguments: dict


@dataclass
class ExtractedToolCalls:
    content: str | None
    tool_calls: list[ParsedToolCall] = field(default_factory=list)


class ToolCallParser:
    """Extracts OpenAI-shape tool calls from a model's raw text completion."""

    def extract_tool_calls(self, text: str) -> ExtractedToolCalls:
        raise NotImplementedError


class HermesToolCallParser(ToolCallParser):
    """`<tool_call>\\n{"name": ..., "arguments": {...}}\\n</tool_call>` -- one tag
    per call. Text before the first tag is kept as `content` (e.g. a model's
    preamble before it decides to call a tool). A call that isn't a JSON object,
    or lacks a `name` / has non-dict `arguments`, is dropped rather than raising
    -- one malformed call shouldn't sink an otherwise-valid response."""

    _CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

    def extract_tool_calls(self, text: str) -> ExtractedToolCalls:
        matches = list(self._CALL_RE.finditer(text))
        if not matches:
            return ExtractedToolCalls(content=text or None)

        calls = []
        for m in matches:
            try:
                obj = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or "name" not in obj:
                continue
            arguments = obj.get("arguments", {})
            if not isinstance(arguments, dict):
                continue
            calls.append(ParsedToolCall(name=obj["name"], arguments=arguments))

        content = text[:matches[0].start()].strip() or None
        return ExtractedToolCalls(content=content, tool_calls=calls)


def _json_safe_literal(node: ast.expr):
    """Convert a Python literal AST to JSON-compatible values, or reject it.

    ``ast.literal_eval`` also admits bytes, sets, tuples, complex numbers, and
    non-finite floats -- values that cannot round-trip through the OpenAI JSON
    ``function.arguments`` field. Keep this deliberately smaller and recursive.
    """
    if isinstance(node, ast.Constant):
        value = node.value
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        raise ValueError("argument is not a JSON-safe scalar")
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_json_safe_literal(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        result = {}
        for key_node, value_node in zip(node.keys, node.values):
            if key_node is None:
                raise ValueError("dictionary unpacking is not allowed")
            key = _json_safe_literal(key_node)
            if not isinstance(key, str):
                raise ValueError("object keys must be strings")
            result[key] = _json_safe_literal(value_node)
        return result
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
        and not isinstance(node.operand.value, bool)
    ):
        value = +node.operand.value if isinstance(node.op, ast.UAdd) else -node.operand.value
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite numbers are not allowed")
        return value
    raise ValueError("argument is not a JSON-safe literal")


class LFM2ToolCallParser(ToolCallParser):
    """Liquid LFM2.5's native Pythonic function-call envelope.

    The model emits a Python list of bare calls between
    ``<|tool_call_start|>`` / ``<|tool_call_end|>``. Parsing uses the Python AST
    only as syntax -- no code is evaluated. Calls must use a bare function name,
    keyword-only arguments, and JSON-safe literal values. Invalid calls are
    dropped, matching the Hermes parser's fail-soft response behavior.
    """

    _CALL_RE = re.compile(
        r"<\|tool_call_start\|>\s*(.*?)\s*<\|tool_call_end\|>", re.DOTALL
    )

    @staticmethod
    def _parse_call(node: ast.expr) -> ParsedToolCall:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise ValueError("tool call must use a bare function name")
        if node.args:
            raise ValueError("positional tool arguments are not allowed")
        arguments = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                raise ValueError("expanded keyword arguments are not allowed")
            if keyword.arg in arguments:
                raise ValueError("duplicate keyword argument")
            arguments[keyword.arg] = _json_safe_literal(keyword.value)
        return ParsedToolCall(name=node.func.id, arguments=arguments)

    def extract_tool_calls(self, text: str) -> ExtractedToolCalls:
        matches = list(self._CALL_RE.finditer(text))
        if not matches:
            return ExtractedToolCalls(content=text or None)

        calls = []
        for match in matches:
            try:
                expression = ast.parse(match.group(1), mode="eval").body
                nodes = expression.elts if isinstance(expression, (ast.List, ast.Tuple)) else []
            except SyntaxError:
                continue
            for node in nodes:
                try:
                    calls.append(self._parse_call(node))
                except (ValueError, TypeError):
                    continue

        outside = []
        cursor = 0
        for match in matches:
            segment = text[cursor:match.start()].strip()
            if segment:
                outside.append(segment)
            cursor = match.end()
        trailing = text[cursor:].strip()
        if trailing:
            outside.append(trailing)
        return ExtractedToolCalls(content="\n\n".join(outside) or None, tool_calls=calls)


class GemmaToolCallParser(ToolCallParser):
    """Gemma4's bare, untagged native call shape (issue #457).

    Gemma has no native function-calling tokens (unlike LFM2.5's
    ``<|tool_call_start|>`` envelope or Hermes's ``<tool_call>`` tags), and
    doesn't reliably follow the Hermes tag convention even when a
    tools-aware chat template instructs it to. Confirmed live: instead of
    ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``, it
    sometimes emits a bare identifier immediately followed by a brace
    block whose keys are unquoted identifiers rather than JSON strings,
    e.g. ``set_reaction{reaction: 'thinking', strength: 0.5}`` -- with no
    enclosing tag at all. Parsing uses the Python AST only as syntax --
    no code is evaluated -- matching ``LFM2ToolCallParser``'s approach for
    the same class of problem.

    Unlike Hermes/LFM2's unique envelope tokens, ``name{...}`` has no
    special delimiter of its own and could in principle collide with
    ordinary text (e.g. a reply that quotes example code containing a
    brace). Two things keep that narrow: a match is only accepted if the
    brace block parses as a *non-empty* dict literal (incidental text
    essentially never does), and this parser is opt-in per deployment via
    ``--tool-parser gemma`` -- it never runs unless explicitly selected.

    Also unlike Hermes/LFM2 (where the call is typically the last thing
    the model emits), Gemma's leaked calls have been observed with the
    *real* reply text following the call, not preceding it -- so text
    before, between, and after every matched call is preserved as
    ``content``, not just the prefix before the first one.
    """

    _CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(\{[^{}]*\})")

    @staticmethod
    def _dict_key(node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        raise ValueError("dict key must be a bare identifier or a string")

    @classmethod
    def _parse_call(cls, name: str, brace_text: str) -> ParsedToolCall:
        node = ast.parse(brace_text, mode="eval").body
        if not isinstance(node, ast.Dict) or not node.keys:
            raise ValueError("brace block is not a non-empty dict literal")
        arguments: dict = {}
        for key_node, value_node in zip(node.keys, node.values):
            if key_node is None:
                raise ValueError("dictionary unpacking is not allowed")
            key = cls._dict_key(key_node)
            if key in arguments:
                raise ValueError("duplicate keyword argument")
            arguments[key] = _json_safe_literal(value_node)
        return ParsedToolCall(name=name, arguments=arguments)

    def extract_tool_calls(self, text: str) -> ExtractedToolCalls:
        calls: list[ParsedToolCall] = []
        outside: list[str] = []
        cursor = 0
        for match in self._CALL_RE.finditer(text):
            try:
                call = self._parse_call(match.group(1), match.group(2))
            except (SyntaxError, ValueError, TypeError):
                # Not a real call (e.g. incidental text, or a brace block
                # that isn't a non-empty dict literal) -- leave it as
                # ordinary content rather than guessing at it. Cursor is
                # deliberately not advanced here: the next real match (or
                # the trailing flush below) will fold this span back into
                # `outside` naturally.
                continue
            segment = text[cursor:match.start()].strip()
            if segment:
                outside.append(segment)
            cursor = match.end()
            calls.append(call)

        if not calls:
            return ExtractedToolCalls(content=text or None)
        trailing = text[cursor:].strip()
        if trailing:
            outside.append(trailing)
        return ExtractedToolCalls(content="\n\n".join(outside) or None, tool_calls=calls)


TOOL_CALL_PARSERS: dict[str, type[ToolCallParser]] = {
    "hermes": HermesToolCallParser,
    "lfm2": LFM2ToolCallParser,
    "gemma": GemmaToolCallParser,
}


def get_tool_call_parser(name: str) -> ToolCallParser:
    try:
        return TOOL_CALL_PARSERS[name]()
    except KeyError:
        raise ValueError(
            f"unknown tool_parser {name!r}; available: {sorted(TOOL_CALL_PARSERS)}"
        ) from None


def forced_tool_schema(tools: list[dict], tool_choice: str | dict) -> tuple[list[dict], dict]:
    """Builds the JSON schema that constrains generation to a single
    `{"name": ..., "arguments": {...}}` object, for `tool_choice="required"`
    (any of `tools`) or a named `tool_choice` (exactly that one). `tools` and
    `tool_choice` are plain dicts in the OpenAI wire shape (`ChatCompletionRequest`
    already validates that shape; this stays decoupled from the pydantic models).
    Returns the narrowed candidate tool list alongside the schema -- the caller
    doesn't need it to interpret the result (the generated JSON already names the
    tool), but it's the natural place to raise on an unknown `tool_choice` name.
    """
    if isinstance(tool_choice, dict):
        name = tool_choice.get("function", {}).get("name")
        candidates = [t for t in tools if t["function"]["name"] == name]
        if not candidates:
            raise ValueError(f"tool_choice names unknown function {name!r}")
    else:
        candidates = tools

    schema = {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "name": {"const": t["function"]["name"]},
                    "arguments": t["function"].get("parameters") or {"type": "object"},
                },
                "required": ["name", "arguments"],
            }
            for t in candidates
        ]
    }
    return candidates, schema


def parse_forced_tool_call(text: str) -> ParsedToolCall:
    obj = json.loads(text)
    return ParsedToolCall(name=obj["name"], arguments=obj.get("arguments", {}))
