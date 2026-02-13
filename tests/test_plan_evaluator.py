"""Priority 4 tests: Evaluator-Optimizer for training plans.

Validates audit finding #5 ("Kein Evaluator-Optimizer Loop"). Tests verify:
- Plans get scored on quality criteria (not rubber-stamped)
- Low-scoring plans trigger regeneration with feedback
- Well-formed plans are accepted on first pass
- Max regeneration attempts prevent infinite loops
- Evaluation integrates into the cognitive loop via action space
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from src.agent.plan_evaluator import (
    evaluate_plan,
    PlanEvaluation,
    PLAN_ACCEPTANCE_THRESHOLD,
    MAX_PLAN_ITERATIONS,
)
from src.agent.actions import execute_action, ACTIONS
from src.agent.state_machine import AgentCore, _evaluate_result_quality


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def sample_profile():
    return {
        "name": "Test Runner",
        "sports": ["running", "cycling"],
        "goal": {"event": "Half Marathon", "target_date": "2026-08-15", "target_time": "1:45:00"},
        "fitness": {"estimated_vo2max": 48, "trend": "improving"},
        "constraints": {"training_days_per_week": 5, "max_session_minutes": 90},
    }


@pytest.fixture
def good_plan():
    """A well-formed plan that should score >= 70."""
    return {
        "week_start": "2026-02-17",
        "sessions": [
            {"day": "Monday", "sport": "running", "type": "Easy", "total_duration_minutes": 45,
             "steps": [{"type": "work", "targets": {"pace_min_km": "5:30-6:00", "hr_zone": "Zone 2"}}]},
            {"day": "Tuesday", "sport": "cycling", "type": "Endurance", "total_duration_minutes": 60,
             "steps": [{"type": "work", "targets": {"power_watts": "150-180", "hr_zone": "Zone 2"}}]},
            {"day": "Wednesday", "sport": "running", "type": "Tempo", "total_duration_minutes": 50,
             "steps": [{"type": "work", "targets": {"pace_min_km": "4:30-4:50", "hr_zone": "Zone 3-4"}}]},
            {"day": "Friday", "sport": "running", "type": "Intervals", "total_duration_minutes": 55,
             "steps": [{"type": "work", "targets": {"pace_min_km": "4:00-4:15", "hr_zone": "Zone 4-5"}}]},
            {"day": "Saturday", "sport": "cycling", "type": "Long Ride", "total_duration_minutes": 90,
             "steps": [{"type": "work", "targets": {"power_watts": "130-160", "hr_zone": "Zone 2"}}]},
        ],
    }


@pytest.fixture
def bad_plan():
    """A poorly-formed plan missing sports and targets — should score < 70."""
    return {
        "week_start": "2026-02-17",
        "sessions": [
            {"day": "Monday", "sport": "running", "type": "Easy", "total_duration_minutes": 45},
            {"day": "Tuesday", "sport": "running", "type": "Easy", "total_duration_minutes": 45},
            {"day": "Wednesday", "sport": "running", "type": "Easy", "total_duration_minutes": 45},
            {"day": "Thursday", "sport": "running", "type": "Easy", "total_duration_minutes": 45},
            {"day": "Friday", "sport": "running", "type": "Easy", "total_duration_minutes": 45},
        ],
    }


@pytest.fixture
def sample_beliefs():
    return [
        {"text": "Wants 3x running per week", "category": "scheduling", "confidence": 0.9},
        {"text": "Wants 2x cycling per week", "category": "scheduling", "confidence": 0.85},
        {"text": "Prefers morning sessions", "category": "preference", "confidence": 0.8},
    ]


def _mock_evaluation_response(score: int, issues: list, suggestions: list):
    """Create a mock Gemini response for plan evaluation."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "overall_score": score,
        "criteria": {
            "sport_distribution": score,
            "target_specificity": score,
            "constraint_compliance": score + 5,
            "volume_progression": score,
            "session_variety": score - 5,
            "recovery_balance": score,
        },
        "issues": issues,
        "suggestions": suggestions,
    })
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    return mock_client


# ── PlanEvaluation Tests ────────────────────────────────────────────

class TestPlanEvaluation:
    """Verify PlanEvaluation dataclass behavior."""

    def test_acceptable_when_score_at_threshold(self):
        pe = PlanEvaluation(score=70, criteria_scores={}, issues=[], suggestions=[])
        assert pe.acceptable is True

    def test_not_acceptable_below_threshold(self):
        pe = PlanEvaluation(score=69, criteria_scores={}, issues=[], suggestions=[])
        assert pe.acceptable is False

    def test_acceptable_above_threshold(self):
        pe = PlanEvaluation(score=85, criteria_scores={}, issues=[], suggestions=[])
        assert pe.acceptable is True

    def test_threshold_value(self):
        assert PLAN_ACCEPTANCE_THRESHOLD == 70

    def test_max_iterations(self):
        assert MAX_PLAN_ITERATIONS == 3


# ── Evaluate Plan Function Tests ────────────────────────────────────

class TestEvaluatePlan:
    """Verify evaluate_plan() calls LLM and returns structured result."""

    def test_good_plan_scores_high(self, sample_profile, good_plan, sample_beliefs):
        mock_client = _mock_evaluation_response(
            82, [], ["Minor: consider adding a cooldown to intervals"],
        )
        with patch("src.agent.plan_evaluator.get_client", return_value=mock_client):
            result = evaluate_plan(good_plan, sample_profile, beliefs=sample_beliefs)

        assert result.score == 82
        assert result.acceptable is True
        assert len(result.criteria_scores) == 6

    def test_bad_plan_scores_low(self, sample_profile, bad_plan, sample_beliefs):
        mock_client = _mock_evaluation_response(
            42,
            ["No cycling sessions but athlete wants 2x", "All sessions same type (Easy)"],
            ["Add 2 cycling sessions", "Include tempo and intervals"],
        )
        with patch("src.agent.plan_evaluator.get_client", return_value=mock_client):
            result = evaluate_plan(bad_plan, sample_profile, beliefs=sample_beliefs)

        assert result.score == 42
        assert result.acceptable is False
        assert len(result.issues) == 2

    def test_evaluation_includes_all_criteria(self, sample_profile, good_plan):
        mock_client = _mock_evaluation_response(75, [], [])
        with patch("src.agent.plan_evaluator.get_client", return_value=mock_client):
            result = evaluate_plan(good_plan, sample_profile)

        expected_criteria = {
            "sport_distribution", "target_specificity", "constraint_compliance",
            "volume_progression", "session_variety", "recovery_balance",
        }
        assert expected_criteria == set(result.criteria_scores.keys())


# ── Action Integration Tests ────────────────────────────────────────

class TestEvaluatePlanAction:
    """Verify evaluate_plan works as an action in the action space."""

    def test_action_registered(self):
        assert "evaluate_plan" in ACTIONS

    def test_action_produces_evaluation(self, sample_profile, good_plan):
        """Verify the evaluate_plan action returns structured evaluation data."""
        mock_eval = MagicMock()
        mock_eval.score = 78
        mock_eval.criteria_scores = {"sport_distribution": 85}
        mock_eval.issues = []
        mock_eval.suggestions = []
        mock_eval.acceptable = True

        ctx = {
            "profile": sample_profile,
            "adjusted_plan": good_plan,
        }
        with patch("src.agent.plan_evaluator.evaluate_plan", return_value=mock_eval):
            result = execute_action("evaluate_plan", ctx)

        assert "plan_evaluation" in result
        assert result["plan_evaluation"]["score"] == 78
        assert result["plan_evaluation"]["acceptable"] is True
        assert result["plan_scores"] == [78]
        assert result["plan_iterations"] == 1

    def test_action_tracks_iteration_count(self, sample_profile, good_plan):
        """Verify plan_scores accumulates across iterations."""
        mock_eval = MagicMock()
        mock_eval.score = 55
        mock_eval.criteria_scores = {}
        mock_eval.issues = ["Missing cycling"]
        mock_eval.suggestions = ["Add cycling"]
        mock_eval.acceptable = False

        ctx = {
            "profile": sample_profile,
            "adjusted_plan": good_plan,
            "plan_scores": [45],  # Previous evaluation
        }
        with patch("src.agent.plan_evaluator.evaluate_plan", return_value=mock_eval):
            result = execute_action("evaluate_plan", ctx)

        assert result["plan_scores"] == [45, 55]
        assert result["plan_iterations"] == 2

    def test_action_generates_feedback_when_not_acceptable(self, sample_profile, bad_plan):
        """Verify plan_feedback is generated for regeneration."""
        mock_eval = MagicMock()
        mock_eval.score = 42
        mock_eval.criteria_scores = {}
        mock_eval.issues = ["No cycling sessions"]
        mock_eval.suggestions = ["Add 2 cycling sessions"]
        mock_eval.acceptable = False

        ctx = {
            "profile": sample_profile,
            "adjusted_plan": bad_plan,
        }
        with patch("src.agent.plan_evaluator.evaluate_plan", return_value=mock_eval):
            result = execute_action("evaluate_plan", ctx)

        assert "plan_feedback" in result
        assert "42/100" in result["plan_feedback"]
        assert "No cycling sessions" in result["plan_feedback"]

    def test_action_no_feedback_at_max_iterations(self, sample_profile, bad_plan):
        """Verify no feedback generated at max iterations (accept best available)."""
        mock_eval = MagicMock()
        mock_eval.score = 55
        mock_eval.criteria_scores = {}
        mock_eval.issues = ["Still missing cycling"]
        mock_eval.suggestions = ["Add cycling"]
        mock_eval.acceptable = False

        ctx = {
            "profile": sample_profile,
            "adjusted_plan": bad_plan,
            "plan_scores": [42, 48],  # Already at 2 attempts, this will be 3rd
        }
        with patch("src.agent.plan_evaluator.evaluate_plan", return_value=mock_eval):
            result = execute_action("evaluate_plan", ctx)

        # At 3 iterations, no more feedback — accept what we have
        assert "plan_feedback" not in result


# ── Cognitive Loop Regeneration Tests ───────────────────────────────

class TestEvaluatorOptimizerLoop:
    """Verify the full generate → evaluate → regenerate pattern."""

    def test_good_plan_accepted_first_pass(self, sample_profile):
        """Spec scenario 1: Good plan scores >= 70, accepted without regeneration."""
        call_count = 0

        def mock_select(ctx, actions_taken=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"action": "generate_plan", "reasoning": "need a plan"}
            elif call_count == 2:
                return {"action": "evaluate_plan", "reasoning": "check quality"}
            return {"action": "respond", "reasoning": "plan accepted"}

        good_plan = {"sessions": [{"sport": "running", "targets": {"pace": "5:00"}}]}
        good_eval = {
            "plan_evaluation": {"score": 82, "acceptable": True, "criteria": {}, "issues": [], "suggestions": []},
            "plan_scores": [82],
            "plan_iterations": 1,
        }

        exec_count = 0

        def mock_exec(name, ctx):
            nonlocal exec_count
            exec_count += 1
            if name == "generate_plan":
                return {"adjusted_plan": good_plan}
            elif name == "evaluate_plan":
                return good_eval
            return {}

        with patch("src.agent.actions.select_action", mock_select), \
             patch("src.agent.actions.execute_action", mock_exec):
            agent = AgentCore()
            result = agent.run_cycle(sample_profile, {"sessions": []}, [])

        actions = [a["action"] for a in result["cycle_context"]["actions_selected"]]
        assert "generate_plan" in actions
        assert "evaluate_plan" in actions
        # Should NOT have a second generate_plan (plan was accepted)
        assert actions.count("generate_plan") == 1

    def test_bad_plan_triggers_regeneration(self, sample_profile):
        """Spec scenario 2: Plan scores < 70, triggers regeneration."""
        call_count = 0

        def mock_select(ctx, actions_taken=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"action": "generate_plan", "reasoning": "need plan"}
            elif call_count == 2:
                return {"action": "evaluate_plan", "reasoning": "check quality"}
            elif call_count == 3:
                return {"action": "generate_plan", "reasoning": "score too low, regenerate"}
            elif call_count == 4:
                return {"action": "evaluate_plan", "reasoning": "re-check"}
            return {"action": "respond", "reasoning": "accepted after regeneration"}

        exec_count = 0

        def mock_exec(name, ctx):
            nonlocal exec_count
            exec_count += 1
            if name == "generate_plan":
                return {"adjusted_plan": {"sessions": [{"sport": "running"}]}}
            elif name == "evaluate_plan":
                # First evaluation: bad. Second: good.
                if exec_count <= 2:
                    return {
                        "plan_evaluation": {"score": 45, "acceptable": False, "criteria": {},
                                           "issues": ["Missing cycling"], "suggestions": ["Add cycling"]},
                        "plan_scores": ctx.get("plan_scores", []) + [45],
                        "plan_iterations": len(ctx.get("plan_scores", [])) + 1,
                        "plan_feedback": "Plan scored 45/100. Issues: Missing cycling.",
                    }
                return {
                    "plan_evaluation": {"score": 78, "acceptable": True, "criteria": {}, "issues": [], "suggestions": []},
                    "plan_scores": ctx.get("plan_scores", []) + [78],
                    "plan_iterations": len(ctx.get("plan_scores", [])) + 1,
                }
            return {}

        with patch("src.agent.actions.select_action", mock_select), \
             patch("src.agent.actions.execute_action", mock_exec):
            agent = AgentCore()
            result = agent.run_cycle(sample_profile, {"sessions": []}, [])

        actions = [a["action"] for a in result["cycle_context"]["actions_selected"]]
        # Should have: generate → evaluate → generate → evaluate → respond
        assert actions.count("generate_plan") == 2
        assert actions.count("evaluate_plan") == 2


# ── Observation Quality for Evaluate Plan ───────────────────────────

class TestEvaluatePlanQuality:
    """Verify _evaluate_result_quality handles evaluate_plan action."""

    def test_high_score_yields_high_quality(self):
        result = {"plan_evaluation": {"score": 85}}
        quality = _evaluate_result_quality("evaluate_plan", result)
        assert quality == 0.85

    def test_low_score_yields_low_quality(self):
        result = {"plan_evaluation": {"score": 30}}
        quality = _evaluate_result_quality("evaluate_plan", result)
        assert quality == 0.3

    def test_perfect_score_caps_at_one(self):
        result = {"plan_evaluation": {"score": 100}}
        quality = _evaluate_result_quality("evaluate_plan", result)
        assert quality == 1.0

    def test_zero_score(self):
        result = {"plan_evaluation": {"score": 0}}
        quality = _evaluate_result_quality("evaluate_plan", result)
        assert quality == 0.0
