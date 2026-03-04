"""Tests for session auto-summarizer.

Covers:
- Message formatting for LLM prompt
- Summary generation (LLM call, JSON parsing)
- Markdown fence stripping
- Error handling (LLM failure, empty responses)
- summarize_session() integration
- summarize_previous_session() flow
- Minimum message threshold
- get_unsummarized_sessions() DB function
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.session_summarizer import (
    _MAX_MESSAGES_FOR_SUMMARY,
    _MIN_MESSAGES_FOR_SUMMARY,
    _format_messages_for_prompt,
    _generate_summary,
    summarize_previous_session,
    summarize_session,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_messages(n: int = 10) -> list[dict]:
    """Generate a list of realistic session messages."""
    messages = []
    for i in range(n):
        if i % 3 == 0:
            messages.append({
                "role": "user",
                "content": f"User message #{i}: How is my training going?",
            })
        elif i % 3 == 1:
            messages.append({
                "role": "model",
                "content": f"Coach response #{i}: Your training looks great!",
            })
        else:
            messages.append({
                "role": "tool_call",
                "content": '{"status": "ok"}',
                "meta": {"tool": "get_activities"},
            })
    return messages


def _mock_llm_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


# ---------------------------------------------------------------------------
# _format_messages_for_prompt
# ---------------------------------------------------------------------------


class TestFormatMessages:
    def test_user_and_model_messages(self) -> None:
        messages = [
            {"role": "user", "content": "Hello coach"},
            {"role": "model", "content": "Hello! How are you?"},
        ]
        result = _format_messages_for_prompt(messages)
        assert "User: Hello coach" in result
        assert "Coach: Hello!" in result

    def test_tool_call_messages(self) -> None:
        messages = [
            {"role": "tool_call", "content": "{}", "meta": {"tool": "get_activities"}},
        ]
        result = _format_messages_for_prompt(messages)
        assert "[Tool: get_activities]" in result

    def test_truncates_long_content(self) -> None:
        messages = [{"role": "user", "content": "x" * 1000}]
        result = _format_messages_for_prompt(messages)
        # Should be truncated to 500 chars
        assert len(result.split(": ", 1)[1]) <= 500

    def test_max_messages_limit(self) -> None:
        messages = _make_messages(_MAX_MESSAGES_FOR_SUMMARY + 20)
        result = _format_messages_for_prompt(messages)
        lines = [l for l in result.strip().split("\n") if l]
        assert len(lines) <= _MAX_MESSAGES_FOR_SUMMARY

    def test_empty_messages(self) -> None:
        result = _format_messages_for_prompt([])
        assert result == ""

    def test_missing_meta_tool(self) -> None:
        messages = [{"role": "tool_call", "content": "{}"}]
        result = _format_messages_for_prompt(messages)
        assert "[Tool: tool]" in result


# ---------------------------------------------------------------------------
# _generate_summary
# ---------------------------------------------------------------------------


class TestGenerateSummary:
    @patch("src.agent.llm.chat_completion")
    def test_successful_summary(self, mock_chat) -> None:
        mock_chat.return_value = _mock_llm_response(
            '{"summary": "Good training week.", "tags": ["running", "progress"]}'
        )

        messages = _make_messages(10)
        result = asyncio.run(_generate_summary(messages))

        assert result["summary"] == "Good training week."
        assert "running" in result["tags"]
        mock_chat.assert_called_once()

    @patch("src.agent.llm.chat_completion")
    def test_strips_markdown_fences(self, mock_chat) -> None:
        mock_chat.return_value = _mock_llm_response(
            '```json\n{"summary": "Good week.", "tags": ["test"]}\n```'
        )

        result = asyncio.run(_generate_summary(_make_messages(5)))
        assert result["summary"] == "Good week."

    @patch("src.agent.llm.chat_completion")
    def test_empty_response(self, mock_chat) -> None:
        mock_chat.return_value = _mock_llm_response("")

        result = asyncio.run(_generate_summary(_make_messages(5)))
        assert result["summary"] == ""
        assert result["tags"] == []

    @patch("src.agent.llm.chat_completion", side_effect=Exception("LLM error"))
    def test_llm_failure(self, mock_chat) -> None:
        result = asyncio.run(_generate_summary(_make_messages(5)))
        assert result["summary"] == ""
        assert result["tags"] == []

    @patch("src.agent.llm.chat_completion")
    def test_invalid_json_response(self, mock_chat) -> None:
        mock_chat.return_value = _mock_llm_response("Not valid JSON at all")

        result = asyncio.run(_generate_summary(_make_messages(5)))
        assert result["summary"] == ""


# ---------------------------------------------------------------------------
# summarize_session
# ---------------------------------------------------------------------------


class TestSummarizeSession:
    @patch("src.db.session_store_db.update_session_summary")
    @patch("src.services.session_summarizer._generate_summary", new_callable=AsyncMock)
    @patch("src.db.session_store_db.load_session_messages")
    def test_full_summarize_flow(self, mock_load, mock_gen, mock_update) -> None:
        mock_load.return_value = _make_messages(10)
        mock_gen.return_value = {
            "summary": "Great session with plan updates.",
            "tags": ["plan", "marathon"],
        }

        result = asyncio.run(summarize_session("session-123"))

        assert result["summary"] == "Great session with plan updates."
        assert "plan" in result["tags"]
        mock_update.assert_called_once()

    @patch("src.db.session_store_db.load_session_messages")
    def test_too_few_messages_skipped(self, mock_load) -> None:
        mock_load.return_value = _make_messages(2)

        result = asyncio.run(summarize_session("session-short"))
        assert result is None

    @patch("src.services.session_summarizer._generate_summary", new_callable=AsyncMock)
    @patch("src.db.session_store_db.load_session_messages")
    def test_empty_summary_not_persisted(self, mock_load, mock_gen) -> None:
        mock_load.return_value = _make_messages(10)
        mock_gen.return_value = {"summary": "", "tags": []}

        result = asyncio.run(summarize_session("session-empty"))
        assert result is None


# ---------------------------------------------------------------------------
# summarize_previous_session
# ---------------------------------------------------------------------------


class TestSummarizePreviousSession:
    @patch("src.services.session_summarizer.summarize_session", new_callable=AsyncMock)
    @patch("src.db.session_store_db.get_unsummarized_sessions")
    def test_summarizes_latest_unsummarized(self, mock_get, mock_summarize) -> None:
        mock_get.return_value = [{"id": "prev-session-1"}]
        mock_summarize.return_value = {"summary": "Done.", "tags": []}

        result = asyncio.run(summarize_previous_session("user-1"))

        assert result["summary"] == "Done."
        mock_summarize.assert_called_once_with("prev-session-1")

    @patch("src.db.session_store_db.get_unsummarized_sessions")
    def test_no_unsummarized_sessions(self, mock_get) -> None:
        mock_get.return_value = []
        result = asyncio.run(summarize_previous_session("user-1"))
        assert result is None

    @patch(
        "src.db.session_store_db.get_unsummarized_sessions",
        side_effect=Exception("DB error"),
    )
    def test_db_error_returns_none(self, mock_get) -> None:
        result = asyncio.run(summarize_previous_session("user-1"))
        assert result is None
