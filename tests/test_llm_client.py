"""Tests for llm_client.py — JSON parsing robustness."""

import pytest

from acadgraph.llm_client import LLMClient


class TestParseJson:
    """Verify the LLM response JSON extractor handles edge cases."""

    def test_direct_json(self):
        result = LLMClient._parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_in_code_fence(self):
        text = '```json\n{"claims": [1, 2]}\n```'
        result = LLMClient._parse_json(text)
        assert result == {"claims": [1, 2]}

    def test_json_in_generic_fence(self):
        text = '```\n{"a": 1}\n```'
        result = LLMClient._parse_json(text)
        assert result == {"a": 1}

    def test_json_embedded_in_text(self):
        text = 'Here is the result: {"status": "ok"} and some trailing text.'
        result = LLMClient._parse_json(text)
        assert result == {"status": "ok"}

    def test_empty_string_returns_empty_dict(self):
        result = LLMClient._parse_json("")
        assert result == {}

    def test_non_json_returns_empty_dict(self):
        result = LLMClient._parse_json("This is not JSON at all")
        assert result == {}

    def test_nested_json(self):
        text = '{"outer": {"inner": [1, 2, 3]}}'
        result = LLMClient._parse_json(text)
        assert result == {"outer": {"inner": [1, 2, 3]}}

    def test_json_with_unicode(self):
        text = '{"描述": "测试中文", "name": "方法A"}'
        result = LLMClient._parse_json(text)
        assert result["描述"] == "测试中文"


class TestLLMClientContextManager:
    """Verify async context manager protocol."""

    def test_has_context_manager_methods(self):
        assert hasattr(LLMClient, "__aenter__")
        assert hasattr(LLMClient, "__aexit__")
