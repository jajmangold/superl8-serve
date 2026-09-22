# SPDX-License-Identifier: MIT
"""`superl8serve.tool_calls` unit tests (issue #40): the Hermes-style
`<tool_call>...</tool_call>` text parser used for `tool_choice="auto"`, and the
JSON-schema builder that drives `tool_choice="required"` / a named tool_choice
through the existing structured-output backend (issue #39). The API-layer
round trip (through `/v1/chat/completions`) lives in `tests/test_api.py`; these
tests exercise the parsing/schema logic directly, without an app or engine."""
from __future__ import annotations

import json

import pytest

from superl8serve.tool_calls import (
    GemmaToolCallParser,
    HermesToolCallParser,
    LFM2ToolCallParser,
    ParsedToolCall,
    forced_tool_schema,
    get_tool_call_parser,
    parse_forced_tool_call,
)

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

TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "parameters": {
            "type": "object",
            "properties": {"timezone": {"type": "string"}},
            "required": ["timezone"],
        },
    },
}


def test_hermes_parser_extracts_a_single_call():
    text = ('<tool_call>\n'
           '{"name": "get_weather", "arguments": {"location": "San Francisco"}}\n'
           '</tool_call>')
    extracted = HermesToolCallParser().extract_tool_calls(text)
    assert extracted.content is None
    assert extracted.tool_calls == [
        ParsedToolCall(name="get_weather", arguments={"location": "San Francisco"})]


def test_hermes_parser_extracts_multiple_calls_in_order():
    text = (
        '<tool_call>{"name": "get_weather", "arguments": {"location": "SF"}}</tool_call>\n'
        '<tool_call>{"name": "get_time", "arguments": {"timezone": "PT"}}</tool_call>'
    )
    extracted = HermesToolCallParser().extract_tool_calls(text)
    assert [c.name for c in extracted.tool_calls] == ["get_weather", "get_time"]
    assert extracted.tool_calls[1].arguments == {"timezone": "PT"}


def test_hermes_parser_keeps_leading_text_as_content():
    text = ('Let me check that for you.\n'
           '<tool_call>{"name": "get_weather", "arguments": {"location": "SF"}}</tool_call>')
    extracted = HermesToolCallParser().extract_tool_calls(text)
    assert extracted.content == "Let me check that for you."
    assert extracted.tool_calls[0].name == "get_weather"


def test_hermes_parser_with_no_tags_returns_content_only():
    extracted = HermesToolCallParser().extract_tool_calls("just a plain reply")
    assert extracted.content == "just a plain reply"
    assert extracted.tool_calls == []


def test_hermes_parser_empty_text_has_no_content():
    extracted = HermesToolCallParser().extract_tool_calls("")
    assert extracted.content is None
    assert extracted.tool_calls == []


def test_hermes_parser_drops_malformed_json_but_keeps_valid_calls():
    text = (
        '<tool_call>{not valid json}</tool_call>'
        '<tool_call>{"name": "get_weather", "arguments": {"location": "SF"}}</tool_call>'
    )
    extracted = HermesToolCallParser().extract_tool_calls(text)
    assert len(extracted.tool_calls) == 1
    assert extracted.tool_calls[0].name == "get_weather"


def test_hermes_parser_drops_call_missing_a_name():
    text = '<tool_call>{"arguments": {"location": "SF"}}</tool_call>'
    extracted = HermesToolCallParser().extract_tool_calls(text)
    assert extracted.tool_calls == []


def test_get_tool_call_parser_returns_hermes_by_default():
    assert isinstance(get_tool_call_parser("hermes"), HermesToolCallParser)


def test_lfm2_parser_extracts_native_pythonic_calls_in_order():
    text = (
        "Checking both.\n"
        "<|tool_call_start|>[get_weather(location='Chicago'), "
        "get_time(timezone='America/Chicago')]<|tool_call_end|>\n"
        "I will use those results."
    )
    extracted = LFM2ToolCallParser().extract_tool_calls(text)
    assert extracted.content == "Checking both.\n\nI will use those results."
    assert extracted.tool_calls == [
        ParsedToolCall(name="get_weather", arguments={"location": "Chicago"}),
        ParsedToolCall(name="get_time", arguments={"timezone": "America/Chicago"}),
    ]


def test_lfm2_parser_accepts_only_json_safe_literal_keyword_arguments():
    text = (
        "<|tool_call_start|>[configure("
        "enabled=True, count=-2, ratio=0.5, labels=['a', 'b'], "
        "options={'mode': None})]<|tool_call_end|>"
    )
    extracted = LFM2ToolCallParser().extract_tool_calls(text)
    assert extracted.tool_calls == [
        ParsedToolCall(
            name="configure",
            arguments={
                "enabled": True,
                "count": -2,
                "ratio": 0.5,
                "labels": ["a", "b"],
                "options": {"mode": None},
            },
        )
    ]


def test_lfm2_parser_drops_only_the_invalid_call_in_a_mixed_list():
    text = (
        "<|tool_call_start|>[get_weather(location='Chicago'), "
        "obj.unsafe(value='drop me'), get_time(timezone='UTC')]<|tool_call_end|>"
    )
    extracted = LFM2ToolCallParser().extract_tool_calls(text)
    assert extracted.tool_calls == [
        ParsedToolCall(name="get_weather", arguments={"location": "Chicago"}),
        ParsedToolCall(name="get_time", arguments={"timezone": "UTC"}),
    ]


@pytest.mark.parametrize(
    "call",
    [
        "obj.weather(city='Chicago')",
        "weather('Chicago')",
        "weather(**{'city': 'Chicago'})",
        "weather(city=get_city())",
        "weather(city={1, 2})",
        "weather(city=b'Chicago')",
    ],
)
def test_lfm2_parser_rejects_unsafe_or_non_json_call_shapes(call):
    text = f"<|tool_call_start|>[{call}]<|tool_call_end|>"
    extracted = LFM2ToolCallParser().extract_tool_calls(text)
    assert extracted.tool_calls == []
    assert extracted.content is None


def test_lfm2_parser_malformed_envelope_falls_back_to_original_content():
    text = "<|tool_call_start|>[weather(city='Chicago')<|tool_call_end|>"
    extracted = LFM2ToolCallParser().extract_tool_calls(text)
    assert extracted.tool_calls == []
    assert extracted.content is None


def test_get_tool_call_parser_returns_lfm2():
    assert isinstance(get_tool_call_parser("lfm2"), LFM2ToolCallParser)


def test_gemma_parser_extracts_the_reported_leak():
    # The exact text observed live (josh/musetalk-volta#45, issue #457):
    # gemma4-26b's real spoken reply followed the leaked call, rather than
    # preceding it as Hermes/LFM2 calls typically do.
    text = (
        "set_reaction{reaction: 'thinking', strength: 0.5} Not much. "
        "I'm just here and ready to chat. Is there something on your mind?"
    )
    extracted = GemmaToolCallParser().extract_tool_calls(text)
    assert extracted.tool_calls == [
        ParsedToolCall(name="set_reaction", arguments={"reaction": "thinking", "strength": 0.5})
    ]
    assert extracted.content == (
        "Not much. I'm just here and ready to chat. Is there something on your mind?"
    )


def test_gemma_parser_keeps_text_before_and_after_the_call():
    text = "Sure thing. set_reaction{reaction: 'amusement', strength: 0.7} Here's a joke for you."
    extracted = GemmaToolCallParser().extract_tool_calls(text)
    assert extracted.content == "Sure thing.\n\nHere's a joke for you."
    assert extracted.tool_calls[0].arguments["reaction"] == "amusement"


def test_gemma_parser_extracts_multiple_calls_in_order():
    text = (
        "set_reaction{reaction: 'interest', strength: 0.4} First. "
        "set_reaction{reaction: 'amusement', strength: 0.6} Second."
    )
    extracted = GemmaToolCallParser().extract_tool_calls(text)
    assert [c.arguments["reaction"] for c in extracted.tool_calls] == ["interest", "amusement"]
    assert extracted.content == "First.\n\nSecond."


def test_gemma_parser_accepts_double_quoted_keys_too():
    text = 'set_reaction{"reaction": "concern", "strength": 0.3}'
    extracted = GemmaToolCallParser().extract_tool_calls(text)
    assert extracted.tool_calls == [
        ParsedToolCall(name="set_reaction", arguments={"reaction": "concern", "strength": 0.3})
    ]


def test_gemma_parser_with_no_call_returns_content_only():
    text = "That's a classic sentence. It's got every letter in the alphabet."
    extracted = GemmaToolCallParser().extract_tool_calls(text)
    assert extracted.content == text
    assert extracted.tool_calls == []


def test_gemma_parser_empty_text_has_no_content():
    extracted = GemmaToolCallParser().extract_tool_calls("")
    assert extracted.content is None
    assert extracted.tool_calls == []


@pytest.mark.parametrize(
    "text",
    [
        "check the config{}",  # empty braces -- not a call with zero arguments
        "math example set{1, 2, 3}",  # a set literal, not a dict
        "here's a dict comprehension idea: total{x for x in range(3)}",
    ],
)
def test_gemma_parser_ignores_non_dict_or_empty_brace_blocks(text):
    # Ordinary text that happens to contain name{...} but isn't a real call
    # must never be misparsed as one -- left alone as plain content.
    extracted = GemmaToolCallParser().extract_tool_calls(text)
    assert extracted.tool_calls == []
    assert extracted.content == text


def test_gemma_parser_rejects_duplicate_keyword_argument():
    text = "set_reaction{reaction: 'thinking', reaction: 'amusement'}"
    extracted = GemmaToolCallParser().extract_tool_calls(text)
    assert extracted.tool_calls == []
    assert extracted.content == text


def test_get_tool_call_parser_returns_gemma():
    assert isinstance(get_tool_call_parser("gemma"), GemmaToolCallParser)


def test_get_tool_call_parser_raises_on_unknown_name():
    with pytest.raises(ValueError, match="unknown tool_parser"):
        get_tool_call_parser("not-a-real-parser")


def test_forced_tool_schema_required_covers_every_tool():
    candidates, schema = forced_tool_schema([WEATHER_TOOL, TIME_TOOL], "required")
    assert candidates == [WEATHER_TOOL, TIME_TOOL]
    names = {branch["properties"]["name"]["const"] for branch in schema["oneOf"]}
    assert names == {"get_weather", "get_time"}


def test_forced_tool_schema_named_choice_narrows_to_one():
    tool_choice = {"type": "function", "function": {"name": "get_time"}}
    candidates, schema = forced_tool_schema([WEATHER_TOOL, TIME_TOOL], tool_choice)
    assert candidates == [TIME_TOOL]
    assert len(schema["oneOf"]) == 1
    assert schema["oneOf"][0]["properties"]["name"]["const"] == "get_time"
    assert schema["oneOf"][0]["properties"]["arguments"] == TIME_TOOL["function"]["parameters"]


def test_forced_tool_schema_unknown_named_choice_raises():
    tool_choice = {"type": "function", "function": {"name": "not_a_tool"}}
    with pytest.raises(ValueError, match="not_a_tool"):
        forced_tool_schema([WEATHER_TOOL], tool_choice)


def test_forced_tool_schema_validates_a_matching_call():
    jsonschema = pytest.importorskip("jsonschema")
    _, schema = forced_tool_schema([WEATHER_TOOL, TIME_TOOL], "required")
    jsonschema.validate({"name": "get_weather", "arguments": {"location": "SF"}}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"name": "get_weather", "arguments": {"location": 5}}, schema)


def test_parse_forced_tool_call():
    parsed = parse_forced_tool_call(json.dumps({"name": "get_weather",
                                                "arguments": {"location": "SF"}}))
    assert parsed == ParsedToolCall(name="get_weather", arguments={"location": "SF"})
