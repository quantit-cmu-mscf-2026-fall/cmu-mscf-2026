"""Tests for capstone.agent.actions.

Inputs are the raw text Claude would return; nothing calls Claude or runs a tool.
"""

from __future__ import annotations

import json

import pytest

from capstone.agent.actions import (
    ActionParseError,
    FinalAnswerAction,
    ToolCallAction,
    parse_action,
)


class TestValidActions:
    def test_tool_call(self):
        raw = json.dumps(
            {
                "type": "tool_call",
                "tool": "get_crsp_returns",
                "arguments": {"start_date": "2020-01-01", "permnos": [1, 2]},
            }
        )
        action = parse_action(raw)

        assert isinstance(action, ToolCallAction)
        assert action.tool == "get_crsp_returns"
        assert action.arguments == {"start_date": "2020-01-01", "permnos": [1, 2]}

    def test_tool_call_with_empty_arguments(self):
        action = parse_action('{"type": "tool_call", "tool": "ping", "arguments": {}}')
        assert action == ToolCallAction(type="tool_call", tool="ping", arguments={})

    def test_final_answer(self):
        action = parse_action('{"type": "final", "content": "rank(close / open)"}')

        assert isinstance(action, FinalAnswerAction)
        assert action.content == "rank(close / open)"

    def test_surrounding_whitespace_is_allowed(self):
        action = parse_action('\n  {"type": "final", "content": "done"}  \n')
        assert isinstance(action, FinalAnswerAction)


class TestMalformedJSON:
    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "not json",
            '{"type": "final", "content": "unterminated}',
            '{"type": "final", "content": "x",}',
            "{'type': 'final', 'content': 'single quotes'}",
            '```json\n{"type": "final", "content": "fenced"}\n```',
            'Sure! {"type": "final", "content": "prose first"}',
            '{"type": "final", "content": "a"}\n{"type": "final", "content": "b"}',
        ],
    )
    def test_is_rejected_not_repaired(self, raw):
        with pytest.raises(ActionParseError, match="not valid JSON") as info:
            parse_action(raw)
        assert info.value.raw == raw

    def test_nan_is_rejected(self):
        raw = '{"type": "tool_call", "tool": "t", "arguments": {"x": NaN}}'
        with pytest.raises(ActionParseError, match="NaN"):
            parse_action(raw)

    def test_duplicate_keys_are_rejected(self):
        raw = '{"type": "final", "type": "tool_call", "content": "x"}'
        with pytest.raises(ActionParseError, match="duplicate key 'type'"):
            parse_action(raw)

    @pytest.mark.parametrize("raw", ["[]", '"final"', "null", "42"])
    def test_non_object_is_rejected(self, raw):
        with pytest.raises(ActionParseError, match="must be a JSON object"):
            parse_action(raw)


class TestUnsupportedType:
    @pytest.mark.parametrize("action_type", ["search", "FINAL", "tool", "", 1])
    def test_unknown_action_type(self, action_type):
        raw = json.dumps({"type": action_type, "content": "x"})
        with pytest.raises(ActionParseError, match="unsupported agent action type"):
            parse_action(raw)

    def test_missing_type(self):
        with pytest.raises(ActionParseError, match="no 'type' field"):
            parse_action('{"content": "x"}')


class TestSchemaViolations:
    @pytest.mark.parametrize(
        "payload",
        [
            {"type": "tool_call", "arguments": {}},
            {"type": "tool_call", "tool": "t"},
            {"type": "tool_call", "tool": "", "arguments": {}},
            {"type": "tool_call", "tool": 3, "arguments": {}},
            {"type": "tool_call", "tool": "t", "arguments": []},
            {"type": "tool_call", "tool": "t", "arguments": "{}"},
            {"type": "tool_call", "tool": "t", "arguments": {}, "extra": 1},
            {"type": "tool_call", "content": "wrong shape for this type"},
        ],
    )
    def test_invalid_tool_call(self, payload):
        with pytest.raises(ActionParseError, match="invalid 'tool_call' action"):
            parse_action(json.dumps(payload))

    @pytest.mark.parametrize(
        "payload",
        [
            {"type": "final"},
            {"type": "final", "content": 42},
            {"type": "final", "content": None},
            {"type": "final", "content": "x", "tool": "t"},
        ],
    )
    def test_invalid_final_answer(self, payload):
        with pytest.raises(ActionParseError, match="invalid 'final' action"):
            parse_action(json.dumps(payload))

    def test_non_string_input_is_a_caller_bug(self):
        with pytest.raises(TypeError):
            parse_action(b'{"type": "final", "content": "x"}')
