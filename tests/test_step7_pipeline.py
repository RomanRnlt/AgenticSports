"""Tests for Step 7 Phase D: Pipeline Integration.

Tests that beliefs and conversation context flow through:
prompts.py, assessment.py, planner.py, state_machine.py, episodes.py
"""

import pytest
from unittest.mock import patch, MagicMock

from src.agent.prompts import build_plan_prompt, _format_beliefs_section
from src.agent.assessment import (
    assess_training,
    _build_assessment_prompt,
    _format_conversation_context,
)
from src.agent.planner import generate_adjusted_plan, _build_adjusted_plan_prompt
from src.agent.state_machine import AgentCore, AgentState
from src.memory.episodes import generate_reflection, _build_reflection_prompt
from src.memory.user_model import UserModel


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def sample_profile():
    return {
        "name": "Test Runner",
        "sports": ["running"],
        "goal": {"event": "Half Marathon", "target_date": "2026-10-15", "target_time": "1:45:00"},
        "constraints": {"training_days_per_week": 5, "max_session_minutes": 90},
        "fitness": {},
    }


@pytest.fixture
def sample_beliefs():
    return [
        {"text": "Prefers morning training before 7am", "category": "scheduling", "confidence": 0.8},
        {"text": "Has a recurring left knee issue", "category": "physical", "confidence": 0.9},
        {"text": "Motivated by race-specific workouts", "category": "motivation", "confidence": 0.7},
        {"text": "Runs easy sessions too fast (Zone 3 instead of Zone 2)", "category": "fitness", "confidence": 0.85},
    ]


@pytest.fixture
def sample_plan():
    return {
        "week_start": "2026-02-09",
        "week_number": 6,
        "sessions": [
            {"day": "Monday", "sport": "running", "type": "Easy Run", "duration_minutes": 45},
            {"day": "Wednesday", "sport": "running", "type": "Intervals", "duration_minutes": 60},
            {"day": "Friday", "sport": "running", "type": "Tempo", "duration_minutes": 50},
        ],
    }


@pytest.fixture
def sample_activities():
    return [
        {
            "sport": "running",
            "start_time": "2026-02-09T06:30:00",
            "duration_seconds": 2700,
            "distance_meters": 8000,
            "heart_rate": {"avg": 155},
            "pace": {"avg_min_per_km": "5:24"},
            "trimp": 85,
        },
    ]


@pytest.fixture
def sample_assessment():
    return {
        "assessment": {
            "compliance": 0.67,
            "observations": ["Easy runs executed at too high intensity"],
            "fitness_trend": "stable",
            "fatigue_level": "moderate",
            "injury_risk": "low",
        },
        "recommended_adjustments": [
            {"type": "pace_target", "description": "Lower easy run pace", "impact": "low", "autonomous": True},
        ],
    }


# ── _format_beliefs_section ──────────────────────────────────────


class TestFormatBeliefsSection:
    def test_none_returns_empty(self):
        assert _format_beliefs_section(None) == ""

    def test_empty_list_returns_empty(self):
        assert _format_beliefs_section([]) == ""

    def test_beliefs_grouped_by_category(self, sample_beliefs):
        result = _format_beliefs_section(sample_beliefs)
        assert "COACH'S NOTES ON THIS ATHLETE" in result
        assert "[SCHEDULING]" in result
        assert "[PHYSICAL]" in result
        assert "[MOTIVATION]" in result
        assert "[FITNESS]" in result

    def test_beliefs_include_text_and_confidence(self, sample_beliefs):
        result = _format_beliefs_section(sample_beliefs)
        assert "Prefers morning training before 7am" in result
        assert "0.8" in result
        assert "left knee issue" in result

    def test_categories_sorted_alphabetically(self, sample_beliefs):
        result = _format_beliefs_section(sample_beliefs)
        fitness_pos = result.index("[FITNESS]")
        motivation_pos = result.index("[MOTIVATION]")
        physical_pos = result.index("[PHYSICAL]")
        scheduling_pos = result.index("[SCHEDULING]")
        assert fitness_pos < motivation_pos < physical_pos < scheduling_pos


# ── build_plan_prompt ────────────────────────────────────────────


class TestBuildPlanPrompt:
    def test_without_beliefs_backward_compat(self, sample_profile):
        prompt = build_plan_prompt(sample_profile)
        assert "Half Marathon" in prompt
        assert "COACH'S NOTES" not in prompt

    def test_with_beliefs_injects_notes(self, sample_profile, sample_beliefs):
        prompt = build_plan_prompt(sample_profile, beliefs=sample_beliefs)
        assert "COACH'S NOTES ON THIS ATHLETE" in prompt
        assert "morning training" in prompt
        assert "left knee issue" in prompt

    def test_with_empty_beliefs(self, sample_profile):
        prompt = build_plan_prompt(sample_profile, beliefs=[])
        assert "COACH'S NOTES" not in prompt

    def test_none_beliefs_same_as_no_beliefs(self, sample_profile):
        prompt_none = build_plan_prompt(sample_profile, beliefs=None)
        prompt_no = build_plan_prompt(sample_profile)
        assert prompt_none == prompt_no


# ── _build_assessment_prompt ─────────────────────────────────────


class TestBuildAssessmentPrompt:
    def test_without_context_backward_compat(self, sample_profile, sample_plan, sample_activities):
        prompt = _build_assessment_prompt(sample_profile, sample_plan, sample_activities)
        assert "ATHLETE GOAL" in prompt
        assert "ATHLETE'S RECENT SELF-REPORTED CONTEXT" not in prompt

    def test_with_conversation_context(self, sample_profile, sample_plan, sample_activities):
        prompt = _build_assessment_prompt(
            sample_profile, sample_plan, sample_activities,
            conversation_context="I've been sleeping badly this week, very stressed at work.",
        )
        assert "ATHLETE'S RECENT SELF-REPORTED CONTEXT" in prompt
        assert "sleeping badly" in prompt

    def test_with_beliefs(self, sample_profile, sample_plan, sample_activities, sample_beliefs):
        prompt = _build_assessment_prompt(
            sample_profile, sample_plan, sample_activities,
            beliefs=sample_beliefs,
        )
        assert "COACH'S NOTES" in prompt
        assert "left knee issue" in prompt

    def test_with_both_context_and_beliefs(
        self, sample_profile, sample_plan, sample_activities, sample_beliefs
    ):
        prompt = _build_assessment_prompt(
            sample_profile, sample_plan, sample_activities,
            conversation_context="Felt tired all week.",
            beliefs=sample_beliefs,
        )
        assert "Felt tired all week" in prompt
        assert "COACH'S NOTES" in prompt


# ── _format_conversation_context ─────────────────────────────────


class TestFormatConversationContext:
    def test_both_none_returns_empty(self):
        assert _format_conversation_context(None, None) == ""

    def test_context_only(self):
        result = _format_conversation_context("I feel tired", None)
        assert "SELF-REPORTED CONTEXT" in result
        assert "I feel tired" in result

    def test_beliefs_only(self, sample_beliefs):
        result = _format_conversation_context(None, sample_beliefs)
        assert "COACH'S NOTES" in result

    def test_both(self, sample_beliefs):
        result = _format_conversation_context("Sore legs", sample_beliefs)
        assert "Sore legs" in result
        assert "COACH'S NOTES" in result


# ── _build_adjusted_plan_prompt ──────────────────────────────────


class TestBuildAdjustedPlanPrompt:
    def test_without_beliefs_backward_compat(
        self, sample_profile, sample_plan, sample_assessment
    ):
        prompt = _build_adjusted_plan_prompt(
            sample_profile, sample_plan, sample_assessment
        )
        assert "ADJUSTED" in prompt
        assert "COACH'S NOTES" not in prompt

    def test_with_beliefs(self, sample_profile, sample_plan, sample_assessment, sample_beliefs):
        prompt = _build_adjusted_plan_prompt(
            sample_profile, sample_plan, sample_assessment, beliefs=sample_beliefs
        )
        assert "COACH'S NOTES" in prompt
        assert "morning training" in prompt


# ── _build_reflection_prompt ─────────────────────────────────────


class TestBuildReflectionPrompt:
    def test_without_context_backward_compat(
        self, sample_profile, sample_plan, sample_activities, sample_assessment
    ):
        prompt = _build_reflection_prompt(
            sample_plan, sample_activities, sample_assessment, sample_profile
        )
        assert "Reflect on this completed training block" in prompt
        assert "SELF-REPORTED CONTEXT" not in prompt

    def test_with_conversation_context(
        self, sample_profile, sample_plan, sample_activities, sample_assessment
    ):
        prompt = _build_reflection_prompt(
            sample_plan, sample_activities, sample_assessment, sample_profile,
            conversation_context="I felt great this week, energy was high.",
        )
        assert "SELF-REPORTED CONTEXT" in prompt
        assert "felt great" in prompt

    def test_with_beliefs(
        self, sample_profile, sample_plan, sample_activities, sample_assessment, sample_beliefs
    ):
        prompt = _build_reflection_prompt(
            sample_plan, sample_activities, sample_assessment, sample_profile,
            beliefs=sample_beliefs,
        )
        assert "COACH'S NOTES" in prompt


# ── AgentCore.run_cycle ──────────────────────────────────────────


class TestRunCycleWithUserModel:
    @patch("src.agent.autonomy.classify_and_apply")
    @patch("src.agent.planner.generate_adjusted_plan")
    @patch("src.agent.assessment.assess_training")
    def test_run_cycle_without_user_model_backward_compat(
        self, mock_assess, mock_plan, mock_autonomy, sample_profile, sample_plan, sample_activities
    ):
        """Existing callers without user_model should still work."""
        mock_assess.return_value = {
            "assessment": {"compliance": 0.9},
            "recommended_adjustments": [],
        }
        mock_plan.return_value = {"sessions": []}
        mock_autonomy.return_value = {"auto_applied": [], "proposals": []}

        agent = AgentCore()
        result = agent.run_cycle(sample_profile, sample_plan, sample_activities)

        mock_assess.assert_called_once_with(
            sample_profile, sample_plan, sample_activities,
            conversation_context=None, beliefs=None,
        )
        mock_plan.assert_called_once_with(
            sample_profile, sample_plan, mock_assess.return_value, beliefs=None,
        )
        assert "assessment" in result

    @patch("src.agent.autonomy.classify_and_apply")
    @patch("src.agent.planner.generate_adjusted_plan")
    @patch("src.agent.assessment.assess_training")
    def test_run_cycle_with_user_model_passes_beliefs(
        self, mock_assess, mock_plan, mock_autonomy,
        sample_profile, sample_plan, sample_activities, sample_beliefs, tmp_path,
    ):
        """When user_model is provided, beliefs are extracted and passed through."""
        model_dir = tmp_path / "user_model"
        model_dir.mkdir()
        model = UserModel(data_dir=model_dir)
        for b in sample_beliefs:
            model.add_belief(b["text"], b["category"], confidence=b["confidence"])

        mock_assess.return_value = {
            "assessment": {"compliance": 0.8},
            "recommended_adjustments": [],
        }
        mock_plan.return_value = {"sessions": []}
        mock_autonomy.return_value = {"auto_applied": [], "proposals": []}

        agent = AgentCore()
        result = agent.run_cycle(
            sample_profile, sample_plan, sample_activities, user_model=model,
        )

        # Verify beliefs were passed to assess_training
        call_kwargs = mock_assess.call_args
        passed_beliefs = call_kwargs.kwargs.get("beliefs") or call_kwargs[1].get("beliefs")
        assert passed_beliefs is not None
        assert len(passed_beliefs) == 4

        # Verify beliefs were passed to generate_adjusted_plan
        plan_kwargs = mock_plan.call_args
        plan_beliefs = plan_kwargs.kwargs.get("beliefs") or plan_kwargs[1].get("beliefs")
        assert plan_beliefs is not None

    @patch("src.agent.autonomy.classify_and_apply")
    @patch("src.agent.planner.generate_adjusted_plan")
    @patch("src.agent.assessment.assess_training")
    def test_run_cycle_with_conversation_context(
        self, mock_assess, mock_plan, mock_autonomy,
        sample_profile, sample_plan, sample_activities,
    ):
        """Conversation context should flow to assess_training."""
        mock_assess.return_value = {
            "assessment": {"compliance": 0.9},
            "recommended_adjustments": [],
        }
        mock_plan.return_value = {"sessions": []}
        mock_autonomy.return_value = {"auto_applied": [], "proposals": []}

        agent = AgentCore()
        result = agent.run_cycle(
            sample_profile, sample_plan, sample_activities,
            conversation_context="I've been sleeping badly this week.",
        )

        call_kwargs = mock_assess.call_args
        passed_context = call_kwargs.kwargs.get("conversation_context") or call_kwargs[1].get("conversation_context")
        assert passed_context == "I've been sleeping badly this week."
        assert "conversation_context" in result["cycle_context"]

    @patch("src.agent.autonomy.classify_and_apply")
    @patch("src.agent.planner.generate_adjusted_plan")
    @patch("src.agent.assessment.assess_training")
    def test_run_cycle_filters_low_confidence_beliefs(
        self, mock_assess, mock_plan, mock_autonomy,
        sample_profile, sample_plan, sample_activities, tmp_path,
    ):
        """Only beliefs >= 0.6 confidence should be passed through."""
        model_dir = tmp_path / "user_model"
        model_dir.mkdir()
        model = UserModel(data_dir=model_dir)
        model.add_belief("High confidence belief", "fitness", confidence=0.9)
        model.add_belief("Low confidence belief", "fitness", confidence=0.3)

        mock_assess.return_value = {
            "assessment": {"compliance": 0.9},
            "recommended_adjustments": [],
        }
        mock_plan.return_value = {"sessions": []}
        mock_autonomy.return_value = {"auto_applied": [], "proposals": []}

        agent = AgentCore()
        agent.run_cycle(sample_profile, sample_plan, sample_activities, user_model=model)

        call_kwargs = mock_assess.call_args
        passed_beliefs = call_kwargs.kwargs.get("beliefs") or call_kwargs[1].get("beliefs")
        assert len(passed_beliefs) == 1
        assert passed_beliefs[0]["text"] == "High confidence belief"


# ── Regression: existing signatures still work ───────────────────


class TestBackwardCompatibility:
    def test_assess_training_positional_args(self):
        """Verify assess_training can be called with only positional args."""
        import inspect
        sig = inspect.signature(assess_training)
        params = list(sig.parameters.keys())
        assert params[:3] == ["profile", "plan", "activities"]
        # New params must have defaults
        assert sig.parameters["conversation_context"].default is None
        assert sig.parameters["beliefs"].default is None

    def test_generate_adjusted_plan_positional_args(self):
        """Verify generate_adjusted_plan can be called with only positional args."""
        import inspect
        sig = inspect.signature(generate_adjusted_plan)
        params = list(sig.parameters.keys())
        assert params[:3] == ["profile", "previous_plan", "assessment"]
        assert sig.parameters["beliefs"].default is None

    def test_generate_reflection_positional_args(self):
        """Verify generate_reflection can be called with original positional args."""
        import inspect
        sig = inspect.signature(generate_reflection)
        params = list(sig.parameters.keys())
        assert params[:4] == ["plan", "activities", "assessment", "athlete_profile"]
        assert sig.parameters["conversation_context"].default is None
        assert sig.parameters["beliefs"].default is None

    def test_build_plan_prompt_positional_args(self):
        """Verify build_plan_prompt can be called with only profile."""
        import inspect
        sig = inspect.signature(build_plan_prompt)
        params = list(sig.parameters.keys())
        assert params[0] == "profile"
        assert sig.parameters["beliefs"].default is None

    def test_run_cycle_positional_args(self):
        """Verify run_cycle can be called with original positional args."""
        import inspect
        sig = inspect.signature(AgentCore.run_cycle)
        # First param is self
        params = list(sig.parameters.keys())
        assert params[1:4] == ["profile", "plan", "activities"]
        assert sig.parameters["user_model"].default is None
        assert sig.parameters["conversation_context"].default is None
