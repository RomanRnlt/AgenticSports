"""Tests for tool result truncation with LLM-based compression.

Covers:
- Token estimation
- Fast-path passthrough (under budget)
- LLM compression pipeline (over budget)
- Fallback to naive truncation on LLM failure
- Per-tool budget map
- execute_with_budget integration
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agent.tools.truncation import (
    PER_TOOL_BUDGET,
    _compress_with_llm,
    _estimate_tokens,
    execute_with_budget,
)
from src.agent.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry_with_tool(name: str, result: dict) -> ToolRegistry:
    """Create a registry with a single tool returning a fixed result."""
    from src.agent.tools.registry import Tool

    registry = ToolRegistry()
    registry.register(Tool(
        name=name,
        description=f"Test tool {name}",
        handler=lambda **_kw: result,
        parameters={"type": "object", "properties": {}},
    ))
    return registry


def _make_large_result(n_items: int = 100) -> dict:
    """Generate a result dict that exceeds typical budgets."""
    return {
        "activities": [
            {
                "id": f"act_{i}",
                "sport": "running",
                "distance_km": 10.5 + i * 0.1,
                "duration_minutes": 55 + i,
                "avg_hr": 145 + i % 20,
                "date": f"2026-02-{(i % 28) + 1:02d}",
                "notes": f"Morning run #{i} with good conditions and steady pace",
            }
            for i in range(n_items)
        ],
        "total": n_items,
    }


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_string(self) -> None:
        assert _estimate_tokens("") == 0

    def test_short_string(self) -> None:
        assert _estimate_tokens("hello") == 1  # 5 // 4 = 1

    def test_known_length(self) -> None:
        text = "a" * 400
        assert _estimate_tokens(text) == 100  # 400 // 4


# ---------------------------------------------------------------------------
# LLM compression
# ---------------------------------------------------------------------------


class TestCompressWithLlm:
    @patch("src.agent.llm.chat_completion")
    def test_successful_compression(self, mock_chat) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"summary": "compressed"}'
        mock_chat.return_value = mock_response

        result = _compress_with_llm('{"large": "data"}', budget_tokens=500)
        assert result == '{"summary": "compressed"}'
        mock_chat.assert_called_once()

    @patch("src.agent.llm.chat_completion")
    def test_empty_response_returns_none(self, mock_chat) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_chat.return_value = mock_response

        result = _compress_with_llm('{"large": "data"}', budget_tokens=500)
        assert result is None

    @patch("src.agent.llm.chat_completion", side_effect=Exception("LLM down"))
    def test_exception_returns_none(self, mock_chat) -> None:
        result = _compress_with_llm('{"large": "data"}', budget_tokens=500)
        assert result is None

    @patch("src.agent.llm.chat_completion")
    def test_none_content_returns_none(self, mock_chat) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_chat.return_value = mock_response

        result = _compress_with_llm('{"data": 1}', budget_tokens=500)
        assert result is None


# ---------------------------------------------------------------------------
# execute_with_budget
# ---------------------------------------------------------------------------


class TestExecuteWithBudget:
    def test_under_budget_passthrough(self) -> None:
        """Small results are returned as-is (fast path)."""
        small = {"status": "ok", "count": 3}
        registry = _make_registry_with_tool("get_config", small)

        result = execute_with_budget(registry, "get_config", {})

        assert result == small
        assert "_compressed" not in result
        assert "_truncated" not in result

    @patch("src.agent.tools.truncation._compress_with_llm")
    def test_over_budget_triggers_compression(self, mock_compress) -> None:
        """Large results trigger LLM compression."""
        mock_compress.return_value = '{"summary": "compressed", "total": 100}'

        large = _make_large_result(100)
        registry = _make_registry_with_tool("get_activities", large)

        result = execute_with_budget(registry, "get_activities", {"days": 90})

        assert result["_compressed"] is True
        assert result["summary"] == "compressed"
        assert result["total"] == 100
        assert result["_original_tokens"] > PER_TOOL_BUDGET["get_activities"]
        mock_compress.assert_called_once()

    @patch("src.agent.tools.truncation._compress_with_llm", return_value=None)
    def test_llm_failure_falls_back_to_truncation(self, mock_compress) -> None:
        """When LLM compression fails, naive truncation is used."""
        large = _make_large_result(100)
        registry = _make_registry_with_tool("get_activities", large)

        result = execute_with_budget(registry, "get_activities", {"days": 90})

        assert result["_truncated"] is True
        assert "truncated from" in result["_note"]
        assert isinstance(result["result"], str)

    def test_per_tool_budget_used(self) -> None:
        """Verifies that per-tool budgets override the default."""
        assert PER_TOOL_BUDGET["get_activities"] == 1500
        assert PER_TOOL_BUDGET["analyze_training_load"] == 800

    @patch("src.agent.tools.truncation._compress_with_llm")
    def test_default_budget_for_unknown_tool(self, mock_compress) -> None:
        """Tools not in PER_TOOL_BUDGET use the default budget."""
        mock_compress.return_value = '{"compressed": true}'

        large = _make_large_result(200)
        registry = _make_registry_with_tool("some_custom_tool", large)

        result = execute_with_budget(
            registry, "some_custom_tool", {}, budget_tokens=500,
        )

        assert result["_compressed"] is True

    @patch("src.agent.tools.truncation._compress_with_llm")
    def test_non_json_compression_result(self, mock_compress) -> None:
        """LLM returns non-JSON text — wrapped as raw string."""
        mock_compress.return_value = "Summary: 100 activities, avg HR 150"

        large = _make_large_result(100)
        registry = _make_registry_with_tool("get_activities", large)

        result = execute_with_budget(registry, "get_activities", {})

        assert result["_compressed"] is True
        assert "Summary" in result["result"]

    def test_tool_error_not_compressed(self) -> None:
        """Error results are small and should pass through uncompressed."""
        error_result = {"error": "Unknown tool: foo"}
        registry = _make_registry_with_tool("foo", error_result)

        result = execute_with_budget(registry, "foo", {})
        assert result == error_result

    def test_unknown_tool_returns_error(self) -> None:
        """Calling an unregistered tool returns an error dict."""
        registry = ToolRegistry()
        result = execute_with_budget(registry, "nonexistent", {})
        assert "error" in result
