"""Tests for recovery-aware plan generation.

Covers:
- _build_recovery_planning_context: converts health summary into planning hints
- Plan prompt injection: recovery context appears in create_training_plan prompt
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_summary(
    sleep_score=85,
    hrv=62.0,
    avg_hrv=58.0,
    stress=20,
    body_battery=90,
    recovery_score=88,
):
    return {
        "latest": {
            "sleep_score": sleep_score,
            "hrv": hrv,
            "stress": stress,
            "body_battery_high": body_battery,
            "recovery_score": recovery_score,
            "sleep_minutes": 450,
        },
        "averages_7d": {
            "sleep_score": 80.0,
            "hrv": avg_hrv,
            "resting_hr": 52.0,
            "stress": 22.0,
        },
        "data_available": True,
        "days_with_data": 7,
    }


# ---------------------------------------------------------------------------
# TestBuildRecoveryPlanningContext
# ---------------------------------------------------------------------------


class TestBuildRecoveryPlanningContext:
    """Test the _build_recovery_planning_context helper."""

    def test_returns_none_when_no_user_id(self) -> None:
        from src.agent.tools.planning_tools import _build_recovery_planning_context

        assert _build_recovery_planning_context(None) is None

    @patch("src.services.health_context.build_health_summary", return_value=None)
    def test_returns_none_when_no_data(self, mock_bhs) -> None:
        from src.agent.tools.planning_tools import _build_recovery_planning_context

        assert _build_recovery_planning_context("user-1") is None

    @patch("src.services.health_context.build_health_summary")
    def test_returns_none_when_data_not_available(self, mock_bhs) -> None:
        from src.agent.tools.planning_tools import _build_recovery_planning_context

        mock_bhs.return_value = {
            "latest": {},
            "averages_7d": {},
            "data_available": False,
            "days_with_data": 0,
        }
        assert _build_recovery_planning_context("user-1") is None

    @patch("src.services.health_context.build_health_summary")
    def test_poor_sleep_generates_hint(self, mock_bhs) -> None:
        from src.agent.tools.planning_tools import _build_recovery_planning_context

        mock_bhs.return_value = _make_summary(sleep_score=55)
        result = _build_recovery_planning_context("user-1")

        assert result is not None
        assert "Reduce high-intensity" in result
        assert "55" in result

    @patch("src.services.health_context.build_health_summary")
    def test_low_hrv_generates_hint(self, mock_bhs) -> None:
        from src.agent.tools.planning_tools import _build_recovery_planning_context

        mock_bhs.return_value = _make_summary(hrv=40, avg_hrv=60.0)
        result = _build_recovery_planning_context("user-1")

        assert result is not None
        assert "accumulated fatigue" in result
        assert "40" in result

    @patch("src.services.health_context.build_health_summary")
    def test_high_stress_generates_hint(self, mock_bhs) -> None:
        from src.agent.tools.planning_tools import _build_recovery_planning_context

        mock_bhs.return_value = _make_summary(stress=65)
        result = _build_recovery_planning_context("user-1")

        assert result is not None
        assert "lighter training load" in result
        assert "65" in result

    @patch("src.services.health_context.build_health_summary")
    def test_low_battery_generates_hint(self, mock_bhs) -> None:
        from src.agent.tools.planning_tools import _build_recovery_planning_context

        mock_bhs.return_value = _make_summary(body_battery=20)
        result = _build_recovery_planning_context("user-1")

        assert result is not None
        assert "Rest day recommended" in result
        assert "20" in result

    @patch("src.services.health_context.build_health_summary")
    def test_good_recovery_generates_positive_hint(self, mock_bhs) -> None:
        from src.agent.tools.planning_tools import _build_recovery_planning_context

        mock_bhs.return_value = _make_summary(sleep_score=85, hrv=62.0)
        result = _build_recovery_planning_context("user-1")

        assert result is not None
        assert "Recovery looks good" in result
        assert "Sleep 85" in result
        assert "HRV 62" in result

    @patch("src.services.health_context.build_health_summary")
    def test_combined_poor_metrics(self, mock_bhs) -> None:
        from src.agent.tools.planning_tools import _build_recovery_planning_context

        mock_bhs.return_value = _make_summary(
            sleep_score=50, stress=70, body_battery=20,
        )
        result = _build_recovery_planning_context("user-1")

        assert result is not None
        lines = result.split("\n")
        # Should have at least 3 hint lines (sleep + stress + battery)
        assert len(lines) >= 3
        assert "Reduce high-intensity" in result
        assert "lighter training load" in result
        assert "Rest day recommended" in result

    @patch("src.services.health_context.build_health_summary")
    def test_missing_fields_handled_gracefully(self, mock_bhs) -> None:
        from src.agent.tools.planning_tools import _build_recovery_planning_context

        # Summary with only partial data — no sleep_score, no stress, no battery
        mock_bhs.return_value = {
            "latest": {"hrv": 60.0, "sleep_score": None, "stress": None, "body_battery_high": None},
            "averages_7d": {"hrv": 58.0},
            "data_available": True,
            "days_with_data": 3,
        }
        result = _build_recovery_planning_context("user-1")

        # Should not crash; HRV is fine so we get positive hint
        assert result is not None
        assert "Recovery looks good" in result

    @patch("src.services.health_context.build_health_summary", side_effect=Exception("DB down"))
    def test_exception_returns_none(self, mock_bhs) -> None:
        from src.agent.tools.planning_tools import _build_recovery_planning_context

        assert _build_recovery_planning_context("user-1") is None


# ---------------------------------------------------------------------------
# TestPlanPromptRecoveryInjection
# ---------------------------------------------------------------------------


class TestPlanPromptRecoveryInjection:
    """Test that recovery context is injected into the plan prompt."""

    @patch("src.services.health_context.build_health_summary")
    @patch("src.agent.tools.planning_tools.chat_completion")
    @patch("src.agent.tools.planning_tools.extract_json", return_value={"weeks": []})
    def test_plan_prompt_includes_recovery_context(
        self, mock_json, mock_llm, mock_bhs,
    ) -> None:
        from src.agent.tools.registry import ToolRegistry
        from src.agent.tools.planning_tools import register_planning_tools

        mock_bhs.return_value = _make_summary(sleep_score=50, body_battery=20)

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"weeks": []}'
        mock_llm.return_value = mock_response

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.use_supabase = True
        mock_settings.agenticsports_user_id = "user-recovery-1"

        # Mock user model
        mock_user_model = MagicMock()
        mock_user_model.project_profile.return_value = {"sports": ["running"], "goal": {"event": "5K"}}
        mock_user_model.get_active_beliefs.return_value = []

        registry = ToolRegistry()

        with (
            patch("src.agent.tools.planning_tools.get_settings", return_value=mock_settings),
            patch("src.db.list_activities", return_value=[]),
            patch("src.db.activity_store_db.list_activities", return_value=[]),
            patch("src.db.list_episodes", return_value=[]),
            patch("src.db.episodes_db.list_episodes", return_value=[]),
            patch("src.memory.episodes.retrieve_relevant_episodes", return_value=[]),
            patch("src.agent.prompts.build_plan_prompt", return_value="BASE PROMPT"),
        ):
            register_planning_tools(registry, mock_user_model)
            registry.execute("create_training_plan", {})

        # Verify the prompt sent to LLM includes recovery context
        call_args = mock_llm.call_args
        prompt_sent = call_args[1]["messages"][0]["content"] if "messages" in call_args[1] else call_args[0][0][0]["content"]
        assert "CURRENT RECOVERY STATUS" in prompt_sent
        assert "Rest day recommended" in prompt_sent

    @patch("src.services.health_context.build_health_summary", return_value=None)
    @patch("src.agent.tools.planning_tools.chat_completion")
    @patch("src.agent.tools.planning_tools.extract_json", return_value={"weeks": []})
    def test_plan_prompt_excludes_recovery_when_no_data(
        self, mock_json, mock_llm, mock_bhs,
    ) -> None:
        from src.agent.tools.registry import ToolRegistry
        from src.agent.tools.planning_tools import register_planning_tools

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"weeks": []}'
        mock_llm.return_value = mock_response

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.use_supabase = True
        mock_settings.agenticsports_user_id = "user-no-health"

        # Mock user model
        mock_user_model = MagicMock()
        mock_user_model.project_profile.return_value = {"sports": ["cycling"], "goal": {}}
        mock_user_model.get_active_beliefs.return_value = []

        registry = ToolRegistry()

        with (
            patch("src.agent.tools.planning_tools.get_settings", return_value=mock_settings),
            patch("src.db.list_activities", return_value=[]),
            patch("src.db.activity_store_db.list_activities", return_value=[]),
            patch("src.db.list_episodes", return_value=[]),
            patch("src.db.episodes_db.list_episodes", return_value=[]),
            patch("src.memory.episodes.retrieve_relevant_episodes", return_value=[]),
            patch("src.agent.prompts.build_plan_prompt", return_value="BASE PROMPT"),
        ):
            register_planning_tools(registry, mock_user_model)
            registry.execute("create_training_plan", {})

        # Verify the prompt does NOT include recovery context
        call_args = mock_llm.call_args
        prompt_sent = call_args[1]["messages"][0]["content"] if "messages" in call_args[1] else call_args[0][0][0]["content"]
        assert "CURRENT RECOVERY STATUS" not in prompt_sent
