"""Unit tests for macrocycle planning tools.

Covers:
- build_macrocycle_prompt: prompt builder with various inputs
- create_macrocycle_plan: LLM sub-agent call (mock LLM + extract_json)
- get_macrocycle: active macrocycle retrieval (mock DB)
- save_macrocycle: macrocycle persistence (mock DB)
- create_training_plan with macrocycle_week: macrocycle-weekly bridge

All DB, LLM, and external dependencies are mocked.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agent.tools.registry import ToolRegistry


USER_ID = "user-macrocycle-test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_settings(use_supabase: bool = True) -> MagicMock:
    s = MagicMock()
    s.agenticsports_user_id = USER_ID
    s.use_supabase = use_supabase
    return s


def _mock_user_model(
    profile: dict | None = None,
    beliefs: list[dict] | None = None,
) -> MagicMock:
    um = MagicMock()
    um.project_profile.return_value = profile or {
        "goal": {"event": "Marathon", "target_date": "2026-09-15"},
        "constraints": {"training_days_per_week": 5, "max_session_minutes": 90},
        "fitness": {"estimated_vo2max": 52, "weekly_volume_km": 40},
        "sports": ["running"],
    }
    um.get_active_beliefs.return_value = beliefs or []
    return um


def _sample_weeks(count: int = 3) -> list[dict]:
    return [
        {
            "week_number": i + 1,
            "phase": "Base" if i < 2 else "Build",
            "focus": "Aerobic endurance" if i < 2 else "Threshold work",
            "volume_target": {"total_minutes": 250 + i * 25, "total_sessions": 5},
            "intensity_distribution": {"low": 80, "moderate": 15, "high": 5},
            "key_sessions": ["Long run 60min", "Tempo 30min"],
            "notes": f"Week {i + 1}",
        }
        for i in range(count)
    ]


def _build_macrocycle_registry(
    settings: MagicMock | None = None,
    user_model: MagicMock | None = None,
) -> ToolRegistry:
    from src.agent.tools.macrocycle_tools import register_macrocycle_tools

    registry = ToolRegistry()
    with patch(
        "src.agent.tools.macrocycle_tools.get_settings",
        return_value=settings or _mock_settings(),
    ):
        register_macrocycle_tools(registry, user_model or _mock_user_model())
    return registry


def _mock_llm_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


# ---------------------------------------------------------------------------
# TestBuildMacrocyclePrompt
# ---------------------------------------------------------------------------


class TestBuildMacrocyclePrompt:
    def test_basic_prompt_contains_weeks(self) -> None:
        from src.agent.prompts import build_macrocycle_prompt

        prompt = build_macrocycle_prompt(
            profile={"goal": {"event": "10K"}, "sports": ["running"],
                     "fitness": {}, "constraints": {}},
            total_weeks=8,
        )
        assert "8-week" in prompt
        assert "10K" in prompt

    def test_prompt_includes_fitness_data(self) -> None:
        from src.agent.prompts import build_macrocycle_prompt

        prompt = build_macrocycle_prompt(
            profile={
                "goal": {},
                "sports": ["cycling"],
                "fitness": {"estimated_vo2max": 55, "threshold_pace_min_km": "4:30"},
                "constraints": {"training_days_per_week": 4},
            },
            total_weeks=12,
        )
        assert "VO2max: 55" in prompt
        assert "4:30" in prompt
        assert "cycling" in prompt

    def test_prompt_includes_beliefs(self) -> None:
        from src.agent.prompts import build_macrocycle_prompt

        beliefs = [{"text": "Prefers morning runs", "category": "scheduling", "confidence": 0.8}]
        prompt = build_macrocycle_prompt(
            profile={"goal": {}, "sports": [], "fitness": {}, "constraints": {}},
            total_weeks=6,
            beliefs=beliefs,
        )
        assert "Prefers morning runs" in prompt

    def test_prompt_includes_periodization_model(self) -> None:
        from src.agent.prompts import build_macrocycle_prompt

        model = {
            "name": "Classic Linear",
            "phases": [
                {"name": "Base", "weeks": 4, "focus": "Aerobic foundation"},
                {"name": "Build", "weeks": 4, "focus": "Threshold development"},
            ],
        }
        prompt = build_macrocycle_prompt(
            profile={"goal": {}, "sports": ["running"], "fitness": {}, "constraints": {}},
            total_weeks=8,
            periodization_model=model,
        )
        assert "Classic Linear" in prompt
        assert "Aerobic foundation" in prompt
        assert "Threshold development" in prompt

    def test_prompt_includes_health_summary(self) -> None:
        from src.agent.prompts import build_macrocycle_prompt

        health = {"data_available": True, "latest": {"sleep_score": 82, "hrv": 45}}
        prompt = build_macrocycle_prompt(
            profile={"goal": {}, "sports": [], "fitness": {}, "constraints": {}},
            total_weeks=4,
            health_summary=health,
        )
        assert "Sleep score: 82" in prompt
        assert "HRV: 45" in prompt

    def test_prompt_includes_activities(self) -> None:
        from src.agent.prompts import build_macrocycle_prompt

        activities = [{"sport": "running", "duration_minutes": 60, "date": "2026-03-01"}]
        prompt = build_macrocycle_prompt(
            profile={"goal": {}, "sports": [], "fitness": {}, "constraints": {}},
            total_weeks=4,
            activities=activities,
        )
        assert "running: 60 min" in prompt

    def test_prompt_no_fitness_data(self) -> None:
        from src.agent.prompts import build_macrocycle_prompt

        prompt = build_macrocycle_prompt(
            profile={"goal": {}, "sports": [], "fitness": {}, "constraints": {}},
            total_weeks=4,
        )
        assert "beginner/unknown" in prompt


# ---------------------------------------------------------------------------
# TestCreateMacrocyclePlan
# ---------------------------------------------------------------------------


class TestCreateMacrocyclePlan:
    def test_creates_plan_with_llm(self) -> None:
        weeks_json = json.dumps({"weeks": _sample_weeks(3)})
        registry = _build_macrocycle_registry()

        with (
            patch(
                "src.agent.tools.macrocycle_tools.chat_completion",
                return_value=_mock_llm_response(weeks_json),
            ),
            patch(
                "src.agent.tools.macrocycle_tools.extract_json",
                return_value={"weeks": _sample_weeks(3)},
            ),
        ):
            result = registry.execute("create_macrocycle_plan", {
                "name": "Marathon Build 2026",
                "weeks": 12,
            })

        assert result["name"] == "Marathon Build 2026"
        assert result["total_weeks"] == 12
        assert result["_status"] == "draft"
        assert len(result["weeks"]) == 3

    def test_clamps_weeks_minimum(self) -> None:
        registry = _build_macrocycle_registry()

        with (
            patch(
                "src.agent.tools.macrocycle_tools.chat_completion",
                return_value=_mock_llm_response("{}"),
            ),
            patch(
                "src.agent.tools.macrocycle_tools.extract_json",
                return_value={"weeks": []},
            ),
        ):
            result = registry.execute("create_macrocycle_plan", {
                "name": "Short",
                "weeks": 2,
            })

        assert result["total_weeks"] == 4  # Clamped to minimum

    def test_clamps_weeks_maximum(self) -> None:
        registry = _build_macrocycle_registry()

        with (
            patch(
                "src.agent.tools.macrocycle_tools.chat_completion",
                return_value=_mock_llm_response("{}"),
            ),
            patch(
                "src.agent.tools.macrocycle_tools.extract_json",
                return_value={"weeks": []},
            ),
        ):
            result = registry.execute("create_macrocycle_plan", {
                "name": "Long",
                "weeks": 100,
            })

        assert result["total_weeks"] == 52  # Clamped to maximum

    def test_loads_periodization_model(self) -> None:
        registry = _build_macrocycle_registry()
        model_data = {"name": "Linear", "phases": [{"name": "Base", "weeks": 4, "focus": "Aerobic"}]}

        with (
            patch(
                "src.agent.tools.macrocycle_tools.chat_completion",
                return_value=_mock_llm_response("{}"),
            ),
            patch(
                "src.agent.tools.macrocycle_tools.extract_json",
                return_value={"weeks": _sample_weeks(1)},
            ),
            patch(
                "src.db.agent_config_db.get_periodization_model",
                return_value=model_data,
            ),
        ):
            result = registry.execute("create_macrocycle_plan", {
                "name": "With Model",
                "periodization_model": "Linear",
            })

        assert result["periodization_model_name"] == "Linear"

    def test_uses_custom_start_date(self) -> None:
        registry = _build_macrocycle_registry()

        with (
            patch(
                "src.agent.tools.macrocycle_tools.chat_completion",
                return_value=_mock_llm_response("{}"),
            ),
            patch(
                "src.agent.tools.macrocycle_tools.extract_json",
                return_value={"weeks": []},
            ),
        ):
            result = registry.execute("create_macrocycle_plan", {
                "name": "Custom Start",
                "start_date": "2026-04-01",
            })

        assert result["start_date"] == "2026-04-01"


# ---------------------------------------------------------------------------
# TestGetMacrocycle
# ---------------------------------------------------------------------------


class TestGetMacrocycle:
    def test_returns_active_macrocycle(self) -> None:
        registry = _build_macrocycle_registry()
        mock_macro = {"name": "Test Plan", "weeks": _sample_weeks(2), "status": "active"}

        with patch(
            "src.db.macrocycle_db.get_active_macrocycle",
            return_value=mock_macro,
        ):
            result = registry.execute("get_macrocycle", {})

        assert result["name"] == "Test Plan"
        assert result["status"] == "active"

    def test_returns_error_when_none(self) -> None:
        registry = _build_macrocycle_registry()

        with patch(
            "src.db.macrocycle_db.get_active_macrocycle",
            return_value=None,
        ):
            result = registry.execute("get_macrocycle", {})

        assert "error" in result
        assert "No active macrocycle" in result["error"]

    def test_returns_error_when_supabase_disabled(self) -> None:
        settings = _mock_settings(use_supabase=False)
        registry = _build_macrocycle_registry(settings=settings)

        result = registry.execute("get_macrocycle", {})

        assert "error" in result
        assert "Supabase not configured" in result["error"]


# ---------------------------------------------------------------------------
# TestSaveMacrocycle
# ---------------------------------------------------------------------------


class TestSaveMacrocycle:
    def test_saves_macrocycle(self) -> None:
        registry = _build_macrocycle_registry()
        saved_row = {"id": "macro-001", "name": "Saved Plan", "status": "active"}

        with patch(
            "src.db.macrocycle_db.store_macrocycle",
            return_value=saved_row,
        ):
            result = registry.execute("save_macrocycle", {
                "macrocycle": {
                    "name": "Saved Plan",
                    "total_weeks": 8,
                    "start_date": "2026-04-01",
                    "weeks": _sample_weeks(2),
                },
            })

        assert result["saved"] is True
        assert result["id"] == "macro-001"
        assert result["name"] == "Saved Plan"

    def test_error_without_name(self) -> None:
        registry = _build_macrocycle_registry()

        result = registry.execute("save_macrocycle", {
            "macrocycle": {"weeks": _sample_weeks(1)},
        })

        assert "error" in result
        assert "name" in result["error"]

    def test_error_without_weeks(self) -> None:
        registry = _build_macrocycle_registry()

        result = registry.execute("save_macrocycle", {
            "macrocycle": {"name": "Empty Plan", "weeks": []},
        })

        assert "error" in result
        assert "weeks" in result["error"]

    def test_error_when_supabase_disabled(self) -> None:
        settings = _mock_settings(use_supabase=False)
        registry = _build_macrocycle_registry(settings=settings)

        result = registry.execute("save_macrocycle", {
            "macrocycle": {"name": "Plan", "weeks": _sample_weeks(1)},
        })

        assert "error" in result
        assert "Supabase not configured" in result["error"]


# ---------------------------------------------------------------------------
# TestCreateTrainingPlanWithMacrocycle
# ---------------------------------------------------------------------------


class TestCreateTrainingPlanWithMacrocycle:
    """Test the macrocycle_week integration in create_training_plan."""

    def _build_planning_registry(self) -> ToolRegistry:
        from src.agent.tools.planning_tools import register_planning_tools

        registry = ToolRegistry()
        user_model = _mock_user_model()

        with patch("src.agent.tools.planning_tools.get_settings", return_value=_mock_settings()):
            register_planning_tools(registry, user_model)
        return registry

    def test_macrocycle_week_injected_into_prompt(self) -> None:
        macro_data = {
            "name": "Marathon Build",
            "total_weeks": 12,
            "weeks": _sample_weeks(3),
        }

        plan_result = {
            "week_start": "2026-03-09",
            "week_number": 1,
            "sessions": [],
            "weekly_summary": {"total_sessions": 5, "total_duration_minutes": 300, "focus": "Base"},
        }

        mock_llm = MagicMock(return_value=_mock_llm_response("{}"))

        with (
            patch("src.agent.tools.planning_tools.get_settings", return_value=_mock_settings()),
            patch("src.db.macrocycle_db.get_active_macrocycle", return_value=macro_data),
            patch("src.agent.tools.planning_tools.chat_completion", mock_llm),
            patch("src.agent.tools.planning_tools.extract_json", return_value=plan_result),
            patch("src.db.list_activities", return_value=[]),
            patch("src.db.list_episodes", return_value=[]),
            patch("src.memory.episodes.retrieve_relevant_episodes", return_value=[]),
        ):
            registry = self._build_planning_registry()
            result = registry.execute("create_training_plan", {"macrocycle_week": 1})

        # Verify macrocycle context was injected into the prompt
        assert mock_llm.called
        kwargs = mock_llm.call_args.kwargs
        prompt = kwargs["messages"][0]["content"]
        assert "MACROCYCLE CONTEXT" in prompt
        assert "Week 1" in prompt
        assert "Base" in prompt

    def test_macrocycle_week_stored_in_plan(self) -> None:
        macro_data = {
            "name": "Test",
            "total_weeks": 4,
            "weeks": _sample_weeks(2),
        }

        plan_result = {
            "week_start": "2026-03-09",
            "sessions": [],
            "weekly_summary": {"total_sessions": 5, "total_duration_minutes": 300, "focus": "Base"},
        }

        with (
            patch("src.agent.tools.planning_tools.get_settings", return_value=_mock_settings()),
            patch("src.db.macrocycle_db.get_active_macrocycle", return_value=macro_data),
            patch("src.agent.tools.planning_tools.chat_completion", return_value=_mock_llm_response("{}")),
            patch("src.agent.tools.planning_tools.extract_json", return_value=plan_result),
            patch("src.db.list_activities", return_value=[]),
            patch("src.db.list_episodes", return_value=[]),
            patch("src.memory.episodes.retrieve_relevant_episodes", return_value=[]),
        ):
            registry = self._build_planning_registry()
            result = registry.execute("create_training_plan", {"macrocycle_week": 2})

        assert result.get("_macrocycle_week") == 2

    def test_no_macrocycle_no_context(self) -> None:
        """When no active macrocycle exists, plan is still created without error."""
        plan_result = {
            "week_start": "2026-03-09",
            "sessions": [],
            "weekly_summary": {"total_sessions": 5, "total_duration_minutes": 300, "focus": "General"},
        }

        mock_llm = MagicMock(return_value=_mock_llm_response("{}"))

        with (
            patch("src.agent.tools.planning_tools.get_settings", return_value=_mock_settings()),
            patch("src.db.macrocycle_db.get_active_macrocycle", return_value=None),
            patch("src.agent.tools.planning_tools.chat_completion", mock_llm),
            patch("src.agent.tools.planning_tools.extract_json", return_value=plan_result),
            patch("src.db.list_activities", return_value=[]),
            patch("src.db.list_episodes", return_value=[]),
            patch("src.memory.episodes.retrieve_relevant_episodes", return_value=[]),
        ):
            registry = self._build_planning_registry()
            result = registry.execute("create_training_plan", {"macrocycle_week": 5})

        # Plan should still be created (macrocycle context is optional)
        assert mock_llm.called
        kwargs = mock_llm.call_args.kwargs
        prompt = kwargs["messages"][0]["content"]
        assert "MACROCYCLE CONTEXT" not in prompt


# ---------------------------------------------------------------------------
# Tool Registration
# ---------------------------------------------------------------------------


class TestMacrocycleToolRegistration:
    def test_all_tools_registered(self) -> None:
        registry = _build_macrocycle_registry()
        names = {t["name"] for t in registry.list_tools()}
        assert "create_macrocycle_plan" in names
        assert "get_macrocycle" in names
        assert "save_macrocycle" in names

    def test_tools_in_planning_category(self) -> None:
        registry = _build_macrocycle_registry()
        tools = {t["name"]: t for t in registry.list_tools()}
        assert tools["create_macrocycle_plan"]["category"] == "planning"
        assert tools["get_macrocycle"]["category"] == "planning"
        assert tools["save_macrocycle"]["category"] == "planning"
