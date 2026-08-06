# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LangChain and Switchyard mapper boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import pytest
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages import BaseMessage, convert_to_openai_messages
from switchyard_langchain import (
    LangChainInvocation,
    SwitchyardRequestMapper,
    SwitchyardResponseMapper,
)

TOOL = {
    "type": "function",
    "function": {
        "name": "run_check",
        "description": "Run a test path.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        "strict": True,
    },
}


def map_request(
    messages: Sequence[BaseMessage] | None = None,
    *,
    tools: Sequence[object] = (),
    tool_choice: object | None = None,
    model_settings: Mapping[str, object] | None = None,
    stop: list[str] | None = None,
) -> dict[str, Any]:
    """Map a request with concise defaults for focused tests."""
    return cast(
        dict[str, Any],
        SwitchyardRequestMapper.to_switchyard(
            messages or [HumanMessage("hello")],
            tools=tools,
            tool_choice=tool_choice,
            model_settings=model_settings or {},
            stop=stop,
        ),
    )


def switchyard_request(**updates: object) -> dict[str, object]:
    """Build the smallest valid Switchyard request used by reverse-mapping tests."""
    request: dict[str, object] = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    }
    request.update(updates)
    return request


def test_request_preserves_messages_tools_and_model_options() -> None:
    messages = [
        SystemMessage("Be concise."),
        HumanMessage(content=["Run the check.", "Return only the result."]),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "run_check",
                    "args": {"path": "tests"},
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            "fatal runtime error: out of memory",
            tool_call_id="call-1",
            name="run_check",
            status="error",
        ),
    ]

    result = map_request(
        messages,
        tools=[TOOL],
        tool_choice={"type": "function", "function": {"name": "run_check"}},
        model_settings={
            "temperature": 0.2,
            "top_p": 0.8,
            "max_tokens": 64,
            "reasoning": {"effort": "low"},
        },
        stop=["DONE"],
    )

    assert result == {
        "model": "auto",
        "instructions": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Be concise."}],
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Run the check."},
                    {"type": "text", "text": "Return only the result."},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "call-1",
                        "name": "run_check",
                        "arguments": {"path": "tests"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "call-1",
                        "content": [
                            {
                                "type": "text",
                                "text": "fatal runtime error: out of memory",
                            }
                        ],
                        "is_error": True,
                    }
                ],
            },
        ],
        "tools": [
            {
                "name": "run_check",
                "description": "Run a test path.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                "strict": True,
            }
        ],
        "tool_choice": {"type": "tool", "data": {"name": "run_check"}},
        "sampling": {"temperature": 0.2, "top_p": 0.8},
        "output": {"max_output_tokens": 64},
        "reasoning": {"effort": "low"},
        "stream": False,
        "extensions": {"fields": {"stop": ["DONE"]}},
    }


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ("auto", {"type": "auto"}),
        ("required", {"type": "required"}),
        ("any", {"type": "required"}),
        ("none", {"type": "none"}),
        ("run_check", {"type": "tool", "data": {"name": "run_check"}}),
    ],
)
def test_tool_choice_strings_are_normalized(choice: str, expected: dict[str, object]) -> None:
    assert map_request(tools=[TOOL], tool_choice=choice)["tool_choice"] == expected


def test_tool_choice_requires_tools_in_both_directions() -> None:
    with pytest.raises(ValueError, match="tool_choice requires at least one tool"):
        map_request(tool_choice="required")

    with pytest.raises(ValueError, match="tool_choice requires at least one tool"):
        SwitchyardRequestMapper.from_switchyard(
            switchyard_request(tool_choice={"type": "required"})
        )


def test_request_rejects_instruction_after_conversation_turn() -> None:
    with pytest.raises(ValueError, match="instruction after the conversation has started"):
        map_request([HumanMessage("hello"), SystemMessage("Change the rules.")])


def test_request_rejects_unsupported_content() -> None:
    with pytest.raises(ValueError, match=r"messages\[0\]\.content\[0\].*not supported"):
        map_request([HumanMessage(content=[{"type": "image", "url": "https://example.test/x"}])])


def test_request_preserves_assistant_content_block_order() -> None:
    message = AIMessage(
        content_blocks=[
            {"type": "text", "text": "before"},
            {
                "type": "tool_call",
                "id": "call-ordered",
                "name": "lookup",
                "args": {"query": "switchyard"},
            },
            {"type": "text", "text": "after"},
        ]
    )

    result = map_request([HumanMessage("hello"), message])

    assert result["messages"][1]["content"] == [
        {"type": "text", "text": "before"},
        {
            "type": "tool_call",
            "id": "call-ordered",
            "name": "lookup",
            "arguments": {"query": "switchyard"},
        },
        {"type": "text", "text": "after"},
    ]


def test_request_rejects_invalid_tool_calls() -> None:
    message = AIMessage(
        "",
        invalid_tool_calls=[
            {
                "type": "invalid_tool_call",
                "id": "call-invalid",
                "name": "lookup",
                "args": "{not json",
                "error": "invalid JSON",
            }
        ],
    )

    with pytest.raises(ValueError, match="invalid_tool_calls is not supported"):
        map_request([HumanMessage("hello"), message])


def test_switchyard_request_becomes_target_invocation() -> None:
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "decision", "schema": {"type": "object"}},
    }
    request = {
        "instructions": [
            {
                "role": "developer",
                "content": [
                    {"type": "text", "text": "Follow the rubric."},
                    {"type": "text", "text": "Return JSON."},
                ],
            }
        ],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "call-2",
                        "name": "lookup",
                        "arguments": {"query": "rust"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "call-2",
                        "content": [{"type": "text", "text": "result"}],
                        "is_error": False,
                    }
                ],
            },
        ],
        "tools": [
            {
                "name": "lookup",
                "description": "Look up text.",
                "parameters": {"type": "object", "properties": {}},
                "strict": False,
            }
        ],
        "tool_choice": {"type": "tool", "data": {"name": "lookup"}},
        "sampling": {"temperature": 0.1, "top_p": 0.9},
        "output": {"max_output_tokens": 32, "response_format": response_format},
        "reasoning": {"effort": "medium"},
        "extensions": {"fields": {"stop": ["STOP"]}},
    }

    invocation = SwitchyardRequestMapper.from_switchyard(request)

    assert isinstance(invocation, LangChainInvocation)
    assert convert_to_openai_messages(
        invocation.messages[0],
        text_format="block",
        pass_through_unknown_blocks=False,
    ) == {
        "role": "developer",
        "content": [
            {"type": "text", "text": "Follow the rubric."},
            {"type": "text", "text": "Return JSON."},
        ],
    }
    assert isinstance(invocation.messages[1], HumanMessage)
    assert isinstance(invocation.messages[2], AIMessage)
    assert invocation.messages[2].tool_calls[0]["args"] == {"query": "rust"}
    assert isinstance(invocation.messages[3], ToolMessage)
    assert invocation.messages[3].content == [{"type": "text", "text": "result"}]
    assert invocation.tools == [{"type": "function", "function": request["tools"][0]}]
    assert invocation.tool_choice == "lookup"
    assert invocation.options == {
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 32,
        "response_format": response_format,
        "reasoning": {"effort": "medium"},
    }
    assert invocation.stop == ["STOP"]


@pytest.mark.parametrize(
    ("settings", "section", "expected"),
    [
        ({"max_completion_tokens": 64}, "output", {"max_output_tokens": 64}),
        (
            {"max_tokens": 64, "max_completion_tokens": None},
            "output",
            {"max_output_tokens": 64},
        ),
        ({"reasoning_effort": "high"}, "reasoning", {"effort": "high"}),
        (
            {"reasoning": {}, "reasoning_effort": "high"},
            "reasoning",
            {"effort": "high"},
        ),
    ],
)
def test_request_maps_model_setting_aliases(
    settings: dict[str, object],
    section: str,
    expected: dict[str, object],
) -> None:
    assert map_request(model_settings=settings)[section] == expected


@pytest.mark.parametrize(
    "settings",
    [
        {"max_tokens": 64, "max_completion_tokens": 32},
        {"reasoning": {"effort": "low"}, "reasoning_effort": "high"},
    ],
)
def test_request_rejects_conflicting_model_setting_aliases(
    settings: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="cannot both be set"):
        map_request(model_settings=settings)


@pytest.mark.parametrize(
    ("settings", "error"),
    [
        ({"seed": 7}, "model setting 'seed' has no Switchyard representation"),
        (
            {"reasoning": {"effort": "low", "summary": "auto"}},
            r"model setting 'reasoning\.summary' has no Switchyard representation",
        ),
    ],
)
def test_request_rejects_unsupported_model_settings(
    settings: dict[str, object],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        map_request(model_settings=settings)


def test_switchyard_request_rejects_invalid_stop_sequences() -> None:
    with pytest.raises(ValueError, match="extensions.fields.stop must be a list of strings"):
        SwitchyardRequestMapper.from_switchyard(
            switchyard_request(extensions={"fields": {"stop": "DONE"}})
        )


@pytest.mark.parametrize(
    ("section", "error"),
    [
        ({"sampling": {"top_k": 40}}, "sampling.top_k has no supported"),
        ({"reasoning": {"raw": {"budget_tokens": 512}}}, "reasoning.raw has no supported"),
    ],
)
def test_switchyard_request_rejects_options_without_langchain_equivalents(
    section: dict[str, object],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        SwitchyardRequestMapper.from_switchyard(switchyard_request(**section))


def test_ai_message_becomes_switchyard_response_with_usage() -> None:
    message = AIMessage(
        content="",
        id="response-1",
        tool_calls=[
            {
                "id": "call-3",
                "name": "search",
                "args": {"query": "switchyard"},
                "type": "tool_call",
            }
        ],
        response_metadata={"finish_reason": "tool_calls"},
        usage_metadata={
            "input_tokens": 12,
            "output_tokens": 4,
            "total_tokens": 16,
            "input_token_details": {"cache_read": 2, "cache_creation": 3},
            "output_token_details": {"reasoning": 2},
        },
    )

    assert SwitchyardResponseMapper.to_switchyard(message, model_name="efficient") == {
        "id": "response-1",
        "model": "efficient",
        "outputs": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "call-3",
                        "name": "search",
                        "arguments": {"query": "switchyard"},
                    }
                ],
                "stop_reason": "tool_use",
            }
        ],
        "usage": {
            "input_tokens": 7,
            "output_tokens": 4,
            "total_tokens": 16,
            "cached_input_tokens": 2,
            "cache_creation_input_tokens": 3,
            "reasoning_tokens": 2,
        },
    }


def test_switchyard_response_becomes_ai_message_with_trace_and_usage() -> None:
    decisions = [
        {"selected_model": "judge", "reasoning": "score request"},
        {"selected_model": "capable", "reasoning": "score exceeded threshold"},
    ]
    message = SwitchyardResponseMapper.from_switchyard(
        {
            "id": "response-2",
            "model": "provider-model",
            "outputs": [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "done"}],
                    "stop_reason": "end_turn",
                }
            ],
            "usage": {
                "input_tokens": 7,
                "cached_input_tokens": 2,
                "cache_creation_input_tokens": 3,
                "output_tokens": 4,
                "total_tokens": 16,
                "reasoning_tokens": 2,
            },
        },
        decisions=decisions,
    )

    assert message.content == [{"type": "text", "text": "done"}]
    assert message.id == "response-2"
    assert message.response_metadata == {
        "model_name": "provider-model",
        "finish_reason": "stop",
        "switchyard": {"decisions": decisions, "selected_model": "capable"},
    }
    assert message.usage_metadata == {
        "input_tokens": 12,
        "output_tokens": 4,
        "total_tokens": 16,
        "input_token_details": {"cache_read": 2, "cache_creation": 3},
        "output_token_details": {"reasoning": 2},
    }


def test_switchyard_response_preserves_assistant_content_block_order() -> None:
    response = {
        "outputs": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "before"},
                    {
                        "type": "tool_call",
                        "id": "call-ordered",
                        "name": "lookup",
                        "arguments": {"query": "switchyard"},
                    },
                    {"type": "text", "text": "after"},
                ],
                "stop_reason": "tool_use",
            }
        ]
    }

    message = SwitchyardResponseMapper.from_switchyard(response, decisions=[])
    round_trip = cast(
        dict[str, Any],
        SwitchyardResponseMapper.to_switchyard(message, model_name="provider-model"),
    )

    assert message.content_blocks == [
        {"type": "text", "text": "before"},
        {
            "type": "tool_call",
            "id": "call-ordered",
            "name": "lookup",
            "args": {"query": "switchyard"},
        },
        {"type": "text", "text": "after"},
    ]
    assert round_trip["outputs"][0]["content"] == response["outputs"][0]["content"]


def test_response_infers_tool_use_from_normalized_provider_content() -> None:
    message = AIMessage(
        content=[
            {
                "type": "tool_use",
                "id": "call-anthropic",
                "name": "lookup",
                "input": {"query": "switchyard"},
            }
        ],
        response_metadata={"model_provider": "anthropic"},
    )
    assert message.tool_calls == []

    response = cast(
        dict[str, Any],
        SwitchyardResponseMapper.to_switchyard(message, model_name="provider-model"),
    )

    assert response["outputs"][0]["stop_reason"] == "tool_use"
    assert response["outputs"][0]["content"] == [
        {
            "type": "tool_call",
            "id": "call-anthropic",
            "name": "lookup",
            "arguments": {"query": "switchyard"},
        }
    ]


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [("stop", "end_turn"), ("length", "max_tokens"), ("content_filter", "content_filter")],
)
def test_response_normalizes_finish_reasons(finish_reason: str, expected: str) -> None:
    response = cast(
        dict[str, Any],
        SwitchyardResponseMapper.to_switchyard(
            AIMessage("done", response_metadata={"finish_reason": finish_reason}),
            model_name="provider-model",
        ),
    )
    assert response["outputs"][0]["stop_reason"] == expected


def test_switchyard_response_rejects_multiple_outputs() -> None:
    output = {"role": "assistant", "content": [{"type": "text", "text": "done"}]}
    with pytest.raises(ValueError, match="must contain exactly one output"):
        SwitchyardResponseMapper.from_switchyard(
            {"outputs": [output, output]},
            decisions=[],
        )


def test_switchyard_response_requires_assistant_content() -> None:
    with pytest.raises(ValueError, match="no assistant text or tool calls"):
        SwitchyardResponseMapper.from_switchyard(
            {"outputs": [{"role": "assistant", "content": []}]},
            decisions=[],
        )
