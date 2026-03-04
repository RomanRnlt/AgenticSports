"""Tests for active context compression (Visionplan 8.12 E).

Covers:
- System prompt contains context window management hint
- _consecutive_tool_calls counter initialized to zero
- Counter resets when model produces a text response (no tool calls)
- Counter increments on each tool-call round
- System summary message injected after 8 consecutive tool-call rounds
- Counter resets to zero after injection
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agent.system_prompt import STATIC_SYSTEM_PROMPT
from src.agent.agent_loop import AgentLoop, TOOL_CALL_SUMMARY_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_user_model() -> MagicMock:
    """Create a minimal mock user model for AgentLoop."""
    model = MagicMock()
    model.project_profile.return_value = {
        "name": "Test",
        "sports": ["running"],
        "goal": {"event": "5K"},
        "constraints": {
            "training_days_per_week": 3,
            "max_session_minutes": 60,
        },
    }
    model.get_active_beliefs.return_value = []
    model.get_active_plan_summary.return_value = None
    model.meta = {}
    model.user_id = "test-user"
    return model


def _make_llm_response(content: str | None = None, tool_calls: list | None = None):
    """Build a mock LLM response (LiteLLM/OpenAI format)."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message = message
    return response


def _make_tool_call(name: str = "get_activities", args: dict | None = None):
    """Build a single mock tool call object."""
    tc = MagicMock()
    tc.id = f"call_{name}"
    tc.function.name = name
    tc.function.arguments = json.dumps(args or {})
    return tc


# ---------------------------------------------------------------------------
# System prompt hint
# ---------------------------------------------------------------------------


class TestSystemPromptHint:
    def test_contains_context_window_management_section(self) -> None:
        assert "## Context Window Management" in STATIC_SYSTEM_PROMPT

    def test_contains_8_tool_calls_guidance(self) -> None:
        assert "8+ consecutive tool calls" in STATIC_SYSTEM_PROMPT

    def test_contains_summarize_instruction(self) -> None:
        assert "Summarize your findings" in STATIC_SYSTEM_PROMPT

    def test_section_is_after_self_correction(self) -> None:
        sc_pos = STATIC_SYSTEM_PROMPT.index("## Self-Correction")
        cwm_pos = STATIC_SYSTEM_PROMPT.index("## Context Window Management")
        assert cwm_pos > sc_pos

    def test_section_is_before_error_handling(self) -> None:
        cwm_pos = STATIC_SYSTEM_PROMPT.index("## Context Window Management")
        eh_pos = STATIC_SYSTEM_PROMPT.index("## Error Handling Rule")
        assert cwm_pos < eh_pos


# ---------------------------------------------------------------------------
# Counter initialization
# ---------------------------------------------------------------------------


class TestCounterInitialization:
    @patch("src.agent.tools.registry.get_default_tools")
    @patch("src.agent.agent_loop.get_settings")
    def test_counter_initialized_to_zero(self, mock_settings, mock_tools) -> None:
        mock_settings.return_value = MagicMock(
            use_supabase=False, agenticsports_user_id="test",
        )
        mock_tools.return_value = MagicMock()
        loop = AgentLoop(user_model=_make_user_model())
        assert loop._consecutive_tool_calls == 0


# ---------------------------------------------------------------------------
# Counter resets on text response
# ---------------------------------------------------------------------------


class TestCounterResetsOnTextResponse:
    @patch("src.agent.agent_loop.execute_with_budget")
    @patch("src.agent.agent_loop.chat_completion")
    @patch("src.agent.tools.registry.get_default_tools")
    @patch("src.agent.agent_loop.get_settings")
    def test_counter_resets_on_text_response(
        self, mock_settings, mock_tools, mock_chat, mock_exec,
    ) -> None:
        mock_settings.return_value = MagicMock(
            use_supabase=False, agenticsports_user_id="test",
        )
        registry = MagicMock()
        registry.get_openai_tools.return_value = [{"type": "function", "function": {"name": "get_activities"}}]
        mock_tools.return_value = registry

        # First call: tool call (increments counter)
        # Second call: text response (resets counter)
        mock_chat.side_effect = [
            _make_llm_response(tool_calls=[_make_tool_call()]),
            _make_llm_response(content="Here is your summary."),
        ]
        mock_exec.return_value = {"activities": []}

        loop = AgentLoop(user_model=_make_user_model())
        loop._consecutive_tool_calls = 5  # simulate prior tool calls

        result = loop.process_message("How was my training?")

        assert loop._consecutive_tool_calls == 0
        assert result.response_text == "Here is your summary."


# ---------------------------------------------------------------------------
# Counter increments on tool calls
# ---------------------------------------------------------------------------


class TestCounterIncrementsOnToolCalls:
    @patch("src.agent.agent_loop.execute_with_budget")
    @patch("src.agent.agent_loop.chat_completion")
    @patch("src.agent.tools.registry.get_default_tools")
    @patch("src.agent.agent_loop.get_settings")
    def test_counter_increments_each_tool_round(
        self, mock_settings, mock_tools, mock_chat, mock_exec,
    ) -> None:
        mock_settings.return_value = MagicMock(
            use_supabase=False, agenticsports_user_id="test",
        )
        registry = MagicMock()
        registry.get_openai_tools.return_value = [{"type": "function", "function": {"name": "get_activities"}}]
        mock_tools.return_value = registry

        # 3 rounds of tool calls, then text response
        mock_chat.side_effect = [
            _make_llm_response(tool_calls=[_make_tool_call("get_activities")]),
            _make_llm_response(tool_calls=[_make_tool_call("analyze_training_load")]),
            _make_llm_response(tool_calls=[_make_tool_call("get_health_data")]),
            _make_llm_response(content="Analysis complete."),
        ]
        mock_exec.return_value = {"data": "ok"}

        loop = AgentLoop(user_model=_make_user_model())
        assert loop._consecutive_tool_calls == 0

        result = loop.process_message("Analyze my week")

        # After text response, counter is reset to 0
        assert loop._consecutive_tool_calls == 0
        assert result.tool_calls_made == 3


# ---------------------------------------------------------------------------
# Summary injected after 8 tool rounds
# ---------------------------------------------------------------------------


class TestSummaryInjectionAfter8Rounds:
    @patch("src.agent.agent_loop.execute_with_budget")
    @patch("src.agent.agent_loop.chat_completion")
    @patch("src.agent.tools.registry.get_default_tools")
    @patch("src.agent.agent_loop.get_settings")
    def test_summary_injected_after_8_tool_rounds(
        self, mock_settings, mock_tools, mock_chat, mock_exec,
    ) -> None:
        mock_settings.return_value = MagicMock(
            use_supabase=False, agenticsports_user_id="test",
        )
        registry = MagicMock()
        registry.get_openai_tools.return_value = [{"type": "function", "function": {"name": "t"}}]
        mock_tools.return_value = registry

        # 8 rounds of tool calls, then after injection the model responds with text
        responses = [
            _make_llm_response(tool_calls=[_make_tool_call(f"tool_{i}")])
            for i in range(TOOL_CALL_SUMMARY_THRESHOLD)
        ]
        responses.append(_make_llm_response(content="Summary: done."))
        mock_chat.side_effect = responses
        mock_exec.return_value = {"ok": True}

        loop = AgentLoop(user_model=_make_user_model())
        result = loop.process_message("Do everything")

        # Verify the injection message was appended to messages
        injection_messages = [
            m for m in loop._messages
            if m.get("role") == "user"
            and "[System:" in (m.get("content") or "")
            and "8+ consecutive tool calls" in (m.get("content") or "")
        ]
        assert len(injection_messages) == 1
        assert result.tool_calls_made == TOOL_CALL_SUMMARY_THRESHOLD

    @patch("src.agent.agent_loop.execute_with_budget")
    @patch("src.agent.agent_loop.chat_completion")
    @patch("src.agent.tools.registry.get_default_tools")
    @patch("src.agent.agent_loop.get_settings")
    def test_injection_message_has_correct_format(
        self, mock_settings, mock_tools, mock_chat, mock_exec,
    ) -> None:
        mock_settings.return_value = MagicMock(
            use_supabase=False, agenticsports_user_id="test",
        )
        registry = MagicMock()
        registry.get_openai_tools.return_value = []
        mock_tools.return_value = registry

        responses = [
            _make_llm_response(tool_calls=[_make_tool_call(f"t{i}")])
            for i in range(TOOL_CALL_SUMMARY_THRESHOLD)
        ]
        responses.append(_make_llm_response(content="Done."))
        mock_chat.side_effect = responses
        mock_exec.return_value = {"ok": True}

        loop = AgentLoop(user_model=_make_user_model())
        loop.process_message("Go")

        injection = next(
            m for m in loop._messages
            if "[System:" in (m.get("content") or "")
        )
        assert injection["role"] == "user"
        assert "Summarize your findings" in injection["content"]


# ---------------------------------------------------------------------------
# Counter resets after injection
# ---------------------------------------------------------------------------


class TestCounterResetsAfterInjection:
    @patch("src.agent.agent_loop.execute_with_budget")
    @patch("src.agent.agent_loop.chat_completion")
    @patch("src.agent.tools.registry.get_default_tools")
    @patch("src.agent.agent_loop.get_settings")
    def test_counter_resets_after_injection(
        self, mock_settings, mock_tools, mock_chat, mock_exec,
    ) -> None:
        mock_settings.return_value = MagicMock(
            use_supabase=False, agenticsports_user_id="test",
        )
        registry = MagicMock()
        registry.get_openai_tools.return_value = []
        mock_tools.return_value = registry

        # 8 tool rounds -> injection (counter reset) -> 2 more tool rounds -> text
        responses = [
            _make_llm_response(tool_calls=[_make_tool_call(f"t{i}")])
            for i in range(TOOL_CALL_SUMMARY_THRESHOLD)
        ]
        # After injection, 2 more tool rounds
        responses.append(_make_llm_response(tool_calls=[_make_tool_call("extra_1")]))
        responses.append(_make_llm_response(tool_calls=[_make_tool_call("extra_2")]))
        responses.append(_make_llm_response(content="Final answer."))
        mock_chat.side_effect = responses
        mock_exec.return_value = {"ok": True}

        loop = AgentLoop(user_model=_make_user_model())
        result = loop.process_message("Complex task")

        # Counter should be 0 because text response resets it
        assert loop._consecutive_tool_calls == 0
        assert result.response_text == "Final answer."
        # Total tool calls: 8 (before injection) + 2 (after) = 10
        assert result.tool_calls_made == TOOL_CALL_SUMMARY_THRESHOLD + 2


# ---------------------------------------------------------------------------
# Threshold constant
# ---------------------------------------------------------------------------


class TestThresholdConstant:
    def test_threshold_is_8(self) -> None:
        assert TOOL_CALL_SUMMARY_THRESHOLD == 8
