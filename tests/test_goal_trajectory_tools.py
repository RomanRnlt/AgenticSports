"""Unit tests for goal trajectory service and tool.

Covers:
- TrajectoryResult dataclass
- analyze_trajectory: LLM-based analysis with mocked responses
- assess_goal_trajectory tool: handler flow, registration, edge cases

All LLM, DB, and service dependencies are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent.tools.registry import ToolRegistry
from src.services.goal_trajectory import TrajectoryResult, analyze_trajectory


# ---------------------------------------------------------------------------
# Fixtures / Helpers
# ---------------------------------------------------------------------------

USER_ID = "user-trajectory-test"

_SAMPLE_GOAL = {
    "event": "Berlin Marathon",
    "target_date": "2026-09-27",
    "target_time": "3:30:00",
}

_SAMPLE_PROFILE = {
    "name": "Athlete",
    "sports": ["running"],
    "goal": _SAMPLE_GOAL,
    "fitness": {"weekly_volume_km": 45, "estimated_vo2max": 50},
    "constraints": {"training_days_per_week": 5},
}

_LLM_RESPONSE_ON_TRACK = {
    "trajectory_status": "on_track",
    "confidence": 0.85,
    "projected_outcome": "Likely to finish around 3:25-3:35",
    "analysis": "Training volume and consistency are solid. Recovery metrics are good.",
    "key_factors": ["Consistent weekly mileage", "Good sleep quality"],
    "risk_factors": ["Heat stress in summer months"],
    "recommendations": ["Add one tempo run per week", "Increase long run by 2km"],
}

_LLM_RESPONSE_BEHIND = {
    "trajectory_status": "behind",
    "confidence": 0.7,
    "projected_outcome": "Current pace suggests 3:50+ finish",
    "analysis": "Volume is below target. Missing key quality sessions.",
    "key_factors": ["Low weekly volume"],
    "risk_factors": ["Injury risk from ramping too fast", "Declining HRV"],
    "recommendations": ["Increase weekly volume gradually", "Add interval training"],
}


def _mock_llm_response(content: str) -> MagicMock:
    """Build a mock LLM response with the given content."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def _mock_settings(user_id: str = USER_ID) -> MagicMock:
    s = MagicMock()
    s.agenticsports_user_id = user_id
    return s


def _mock_user_model(
    profile: dict | None = None,
    beliefs: list[dict] | None = None,
) -> MagicMock:
    model = MagicMock()
    model.project_profile.return_value = profile or _SAMPLE_PROFILE
    model.get_active_beliefs.return_value = beliefs or []
    return model


def _build_registry(
    settings: MagicMock | None = None,
    user_model: MagicMock | None = None,
) -> ToolRegistry:
    from src.agent.tools.goal_trajectory_tools import register_goal_trajectory_tools

    registry = ToolRegistry()
    with patch(
        "src.agent.tools.goal_trajectory_tools.get_settings",
        return_value=settings or _mock_settings(),
    ):
        register_goal_trajectory_tools(registry, user_model or _mock_user_model())
    return registry


# ---------------------------------------------------------------------------
# TestAnalyzeTrajectory — service function tests
# ---------------------------------------------------------------------------


class TestAnalyzeTrajectory:
    """Tests for the analyze_trajectory service function."""

    @patch("src.services.goal_trajectory.extract_json")
    @patch("src.services.goal_trajectory.chat_completion")
    def test_on_track_result(self, mock_llm, mock_json) -> None:
        mock_llm.return_value = _mock_llm_response('{"trajectory_status": "on_track"}')
        mock_json.return_value = _LLM_RESPONSE_ON_TRACK

        result = analyze_trajectory(goal=_SAMPLE_GOAL, profile=_SAMPLE_PROFILE)

        assert isinstance(result, TrajectoryResult)
        assert result.trajectory_status == "on_track"
        assert result.confidence == 0.85
        assert "3:25" in result.projected_outcome
        assert len(result.key_factors) == 2
        assert len(result.risk_factors) == 1
        assert len(result.recommendations) == 2

    @patch("src.services.goal_trajectory.extract_json")
    @patch("src.services.goal_trajectory.chat_completion")
    def test_behind_result(self, mock_llm, mock_json) -> None:
        mock_llm.return_value = _mock_llm_response('{"trajectory_status": "behind"}')
        mock_json.return_value = _LLM_RESPONSE_BEHIND

        result = analyze_trajectory(goal=_SAMPLE_GOAL, profile=_SAMPLE_PROFILE)

        assert result.trajectory_status == "behind"
        assert result.confidence == 0.7
        assert "3:50" in result.projected_outcome

    @patch("src.services.goal_trajectory.chat_completion")
    def test_llm_error_returns_insufficient_data(self, mock_llm) -> None:
        mock_llm.side_effect = Exception("LLM timeout")

        result = analyze_trajectory(goal=_SAMPLE_GOAL, profile=_SAMPLE_PROFILE)

        assert result.trajectory_status == "insufficient_data"
        assert result.confidence == 0.0
        assert len(result.recommendations) > 0

    @patch("src.services.goal_trajectory.extract_json")
    @patch("src.services.goal_trajectory.chat_completion")
    def test_empty_llm_response_returns_insufficient_data(self, mock_llm, mock_json) -> None:
        mock_llm.return_value = _mock_llm_response("")

        result = analyze_trajectory(goal=_SAMPLE_GOAL, profile=_SAMPLE_PROFILE)

        assert result.trajectory_status == "insufficient_data"
        assert result.confidence == 0.0
        mock_json.assert_not_called()

    def test_empty_goal_returns_insufficient_data(self) -> None:
        result = analyze_trajectory(goal={}, profile=_SAMPLE_PROFILE)

        assert result.trajectory_status == "insufficient_data"
        assert result.confidence == 0.0

    def test_none_goal_returns_insufficient_data(self) -> None:
        result = analyze_trajectory(goal=None, profile=_SAMPLE_PROFILE)

        assert result.trajectory_status == "insufficient_data"

    @patch("src.services.goal_trajectory.extract_json")
    @patch("src.services.goal_trajectory.chat_completion")
    def test_with_all_optional_context(self, mock_llm, mock_json) -> None:
        mock_llm.return_value = _mock_llm_response('{"trajectory_status": "ahead"}')
        mock_json.return_value = {
            "trajectory_status": "ahead",
            "confidence": 0.9,
            "projected_outcome": "Ahead of schedule",
            "analysis": "Strong progress across all metrics.",
            "key_factors": ["High volume"],
            "risk_factors": [],
            "recommendations": ["Maintain current approach"],
        }

        result = analyze_trajectory(
            goal=_SAMPLE_GOAL,
            profile=_SAMPLE_PROFILE,
            training_summary={"total_sessions": 20, "total_duration_minutes": 1200},
            health_trends={"latest": {"hrv": 55}, "averages_7d": {"hrv": 52}},
            periodization_phase="build",
            beliefs=[{"text": "Responds well to intervals", "confidence": 0.8}],
            previous_trajectory={"trajectory_status": "on_track", "analysis": "Was on track"},
        )

        assert result.trajectory_status == "ahead"
        assert result.confidence == 0.9
        # Verify the prompt included optional sections
        call_args = mock_llm.call_args
        prompt_content = call_args[1]["messages"][0]["content"] if "messages" in call_args[1] else call_args[0][0][0]["content"]
        assert "TRAINING SUMMARY" in prompt_content
        assert "HEALTH TRENDS" in prompt_content
        assert "PERIODIZATION PHASE" in prompt_content
        assert "KNOWN ATHLETE PATTERNS" in prompt_content
        assert "PREVIOUS TRAJECTORY" in prompt_content

    @patch("src.services.goal_trajectory.extract_json")
    @patch("src.services.goal_trajectory.chat_completion")
    def test_json_parse_error_returns_insufficient_data(self, mock_llm, mock_json) -> None:
        mock_llm.return_value = _mock_llm_response("not valid json at all")
        mock_json.side_effect = ValueError("Could not extract JSON")

        result = analyze_trajectory(goal=_SAMPLE_GOAL, profile=_SAMPLE_PROFILE)

        assert result.trajectory_status == "insufficient_data"
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# TestTrajectoryResult — dataclass tests
# ---------------------------------------------------------------------------


class TestTrajectoryResult:
    def test_frozen_immutability(self) -> None:
        result = TrajectoryResult(
            trajectory_status="on_track",
            confidence=0.85,
            projected_outcome="Good",
            analysis="Solid",
            key_factors=["A"],
            risk_factors=["B"],
            recommendations=["C"],
        )
        with pytest.raises(AttributeError):
            result.trajectory_status = "behind"  # type: ignore[misc]

    def test_all_fields_accessible(self) -> None:
        result = TrajectoryResult(
            trajectory_status="ahead",
            confidence=0.95,
            projected_outcome="Very strong",
            analysis="Excellent progress",
            key_factors=["High volume", "Good recovery"],
            risk_factors=[],
            recommendations=["Stay the course"],
        )
        assert result.trajectory_status == "ahead"
        assert result.confidence == 0.95
        assert len(result.key_factors) == 2
        assert len(result.risk_factors) == 0
        assert len(result.recommendations) == 1


# ---------------------------------------------------------------------------
# TestAssessGoalTrajectoryTool — tool handler tests
# ---------------------------------------------------------------------------


class TestAssessGoalTrajectoryTool:
    """Tests for the assess_goal_trajectory tool handler."""

    def test_tool_registered(self) -> None:
        registry = _build_registry()
        names = [t["name"] for t in registry.list_tools()]
        assert "assess_goal_trajectory" in names

    def test_tool_category_is_analysis(self) -> None:
        registry = _build_registry()
        tools = {t["name"]: t for t in registry.list_tools()}
        assert tools["assess_goal_trajectory"]["category"] == "analysis"

    @patch("src.agent.tools.goal_trajectory_tools._save_snapshot")
    @patch("src.agent.tools.goal_trajectory_tools._get_previous_trajectory", return_value=None)
    @patch("src.agent.tools.goal_trajectory_tools._get_beliefs", return_value=None)
    @patch("src.agent.tools.goal_trajectory_tools._get_current_phase", return_value="build")
    @patch("src.agent.tools.goal_trajectory_tools._build_health_trends", return_value=None)
    @patch("src.agent.tools.goal_trajectory_tools._build_training_summary", return_value={"total_sessions": 10})
    @patch("src.services.goal_trajectory.extract_json")
    @patch("src.services.goal_trajectory.chat_completion")
    def test_handler_calls_analyze_and_saves(
        self, mock_llm, mock_json, mock_training, mock_health,
        mock_phase, mock_beliefs, mock_prev, mock_save,
    ) -> None:
        mock_llm.return_value = _mock_llm_response('{"ok": true}')
        mock_json.return_value = _LLM_RESPONSE_ON_TRACK

        registry = _build_registry()
        result = registry.execute("assess_goal_trajectory", {"save_snapshot": True})

        assert result["trajectory_status"] == "on_track"
        assert result["confidence"] == 0.85
        mock_save.assert_called_once()

    @patch("src.agent.tools.goal_trajectory_tools._save_snapshot")
    @patch("src.agent.tools.goal_trajectory_tools._get_previous_trajectory", return_value=None)
    @patch("src.agent.tools.goal_trajectory_tools._get_beliefs", return_value=None)
    @patch("src.agent.tools.goal_trajectory_tools._get_current_phase", return_value=None)
    @patch("src.agent.tools.goal_trajectory_tools._build_health_trends", return_value=None)
    @patch("src.agent.tools.goal_trajectory_tools._build_training_summary", return_value=None)
    @patch("src.services.goal_trajectory.extract_json")
    @patch("src.services.goal_trajectory.chat_completion")
    def test_handler_skips_save_when_false(
        self, mock_llm, mock_json, mock_training, mock_health,
        mock_phase, mock_beliefs, mock_prev, mock_save,
    ) -> None:
        mock_llm.return_value = _mock_llm_response('{"ok": true}')
        mock_json.return_value = _LLM_RESPONSE_ON_TRACK

        registry = _build_registry()
        result = registry.execute("assess_goal_trajectory", {"save_snapshot": False})

        assert result["trajectory_status"] == "on_track"
        mock_save.assert_not_called()

    def test_handler_missing_goal_returns_error(self) -> None:
        profile_no_goal = {**_SAMPLE_PROFILE, "goal": {}}
        user_model = _mock_user_model(profile=profile_no_goal)
        registry = _build_registry(user_model=user_model)

        result = registry.execute("assess_goal_trajectory", {})

        assert "error" in result
        assert result["trajectory_status"] == "insufficient_data"

    def test_handler_missing_user_id_returns_error(self) -> None:
        settings = _mock_settings(user_id="")
        registry = _build_registry(settings=settings)

        result = registry.execute("assess_goal_trajectory", {})

        assert "error" in result
        assert result["trajectory_status"] == "insufficient_data"

    def test_handler_no_user_model_returns_error(self) -> None:
        from src.agent.tools.goal_trajectory_tools import register_goal_trajectory_tools

        registry = ToolRegistry()
        with patch(
            "src.agent.tools.goal_trajectory_tools.get_settings",
            return_value=_mock_settings(),
        ):
            register_goal_trajectory_tools(registry, user_model=None)

        result = registry.execute("assess_goal_trajectory", {})

        assert "error" in result
        assert result["trajectory_status"] == "insufficient_data"

    @patch("src.agent.tools.goal_trajectory_tools._save_snapshot")
    @patch("src.agent.tools.goal_trajectory_tools._get_previous_trajectory", return_value=None)
    @patch("src.agent.tools.goal_trajectory_tools._get_beliefs", return_value=None)
    @patch("src.agent.tools.goal_trajectory_tools._get_current_phase", return_value=None)
    @patch("src.agent.tools.goal_trajectory_tools._build_health_trends", return_value=None)
    @patch("src.agent.tools.goal_trajectory_tools._build_training_summary", return_value=None)
    @patch("src.services.goal_trajectory.chat_completion")
    def test_handler_insufficient_data_skips_save(
        self, mock_llm, mock_training, mock_health,
        mock_phase, mock_beliefs, mock_prev, mock_save,
    ) -> None:
        """When LLM returns insufficient_data, snapshot should not be saved."""
        mock_llm.side_effect = Exception("LLM down")

        registry = _build_registry()
        result = registry.execute("assess_goal_trajectory", {"save_snapshot": True})

        assert result["trajectory_status"] == "insufficient_data"
        mock_save.assert_not_called()
