"""Step 3 tests: State machine, assessment, planner, autonomy."""

import json
from pathlib import Path

import pytest

from src.agent.state_machine import AgentState, AgentCore
from src.agent.autonomy import ImpactLevel, classify_impact, classify_and_apply
from src.tools.fit_parser import parse_fit_file
from src.tools.metrics import calculate_trimp, classify_hr_zone
from src.tools.activity_store import store_activity, list_activities, get_weekly_summary

FIXTURES = Path(__file__).parent / "fixtures"


# ── Helper: build a minimal test profile and plan ─────────────────────

def _test_profile():
    return {
        "name": "Test Athlete",
        "sports": ["running"],
        "goal": {
            "event": "Half Marathon",
            "target_date": "2026-08-15",
            "target_time": "1:45:00",
        },
        "fitness": {
            "estimated_vo2max": None,
            "threshold_pace_min_km": None,
            "weekly_volume_km": None,
            "trend": "unknown",
        },
        "constraints": {
            "training_days_per_week": 5,
            "max_session_minutes": 90,
            "available_sports": ["running"],
        },
    }


def _test_plan():
    return {
        "week_start": "2026-02-02",
        "week_number": 1,
        "sessions": [
            {"day": "Monday", "sport": "running", "type": "Easy Run", "duration_minutes": 45,
             "target_pace_min_km": "5:30-6:00", "target_hr_zone": "Zone 2"},
            {"day": "Tuesday", "sport": "running", "type": "Tempo Run", "duration_minutes": 40,
             "target_pace_min_km": "5:00-5:15", "target_hr_zone": "Zone 3"},
            {"day": "Thursday", "sport": "running", "type": "Intervals", "duration_minutes": 50,
             "target_pace_min_km": "4:00-4:30", "target_hr_zone": "Zone 4-5"},
            {"day": "Saturday", "sport": "running", "type": "Long Run", "duration_minutes": 70,
             "target_pace_min_km": "5:45-6:15", "target_hr_zone": "Zone 2"},
            {"day": "Sunday", "sport": "running", "type": "Recovery Run", "duration_minutes": 30,
             "target_pace_min_km": "6:00-6:30", "target_hr_zone": "Zone 1"},
        ],
        "weekly_summary": {
            "total_sessions": 5,
            "total_duration_minutes": 235,
            "focus": "Base building",
        },
    }


def _load_mock_activities():
    """Load all 4 mock fixtures and add TRIMP/zone data."""
    activities = []
    for name in ["easy_run.json", "interval_run.json", "long_bike.json", "gym_session.json"]:
        act = parse_fit_file(str(FIXTURES / name))
        hr = act.get("heart_rate")
        if hr and hr.get("avg"):
            dur = act["duration_seconds"] / 60
            act["trimp"] = calculate_trimp(dur, hr["avg"])
            act["hr_zone"] = classify_hr_zone(hr["avg"])
        activities.append(act)
    return activities


# ── State Machine ─────────────────────────────────────────────────────

class TestStateMachine:
    def test_initial_state_is_idle(self):
        agent = AgentCore()
        assert agent.state == AgentState.IDLE

    def test_transition_records_history(self):
        agent = AgentCore()
        agent.context = {"state_history": []}
        agent.transition(AgentState.PERCEIVING)
        assert agent.state == AgentState.PERCEIVING
        assert len(agent.context["state_history"]) == 1
        assert agent.context["state_history"][0]["from"] == "idle"
        assert agent.context["state_history"][0]["to"] == "perceiving"

    def test_all_states_exist(self):
        states = [s.value for s in AgentState]
        expected = ["idle", "perceiving", "reasoning", "planning", "proposing", "executing", "reflecting"]
        assert states == expected


# ── Autonomy Classification ──────────────────────────────────────────

class TestAutonomy:
    def test_classify_low_impact(self):
        adj = {"impact": "low", "description": "Adjust easy run pace target"}
        assert classify_impact(adj) == ImpactLevel.LOW

    def test_classify_medium_impact(self):
        adj = {"impact": "medium", "description": "Reduce volume by 20%"}
        assert classify_impact(adj) == ImpactLevel.MEDIUM

    def test_classify_high_impact(self):
        adj = {"impact": "high", "description": "Restructure periodization"}
        assert classify_impact(adj) == ImpactLevel.HIGH

    def test_classify_infers_from_description_volume(self):
        adj = {"description": "reduce weekly volume by 15%"}
        assert classify_impact(adj) == ImpactLevel.MEDIUM

    def test_classify_infers_from_description_injury(self):
        adj = {"description": "possible injury risk detected, stop intervals"}
        assert classify_impact(adj) == ImpactLevel.HIGH

    def test_classify_defaults_to_low(self):
        adj = {"description": "tweak warm-up routine"}
        assert classify_impact(adj) == ImpactLevel.LOW

    def test_classify_and_apply_splits_correctly(self):
        adjustments = [
            {"impact": "low", "description": "Lower easy pace"},
            {"impact": "medium", "description": "Add rest day"},
            {"impact": "high", "description": "Restructure plan"},
            {"impact": "low", "description": "Adjust HR target"},
        ]
        result = classify_and_apply(adjustments)
        assert len(result["auto_applied"]) == 2
        assert len(result["proposals"]) == 2
        assert result["counts"]["low"] == 2
        assert result["counts"]["medium"] == 1
        assert result["counts"]["high"] == 1


# ── Integration Tests (calls Gemini) ─────────────────────────────────

class TestStep3Integration:
    @pytest.mark.integration
    def test_assessment_returns_valid_structure(self):
        """Assessment should return compliance, observations, adjustments."""
        from src.agent.assessment import assess_training

        profile = _test_profile()
        plan = _test_plan()
        activities = _load_mock_activities()

        result = assess_training(profile, plan, activities)

        assert "assessment" in result
        assessment = result["assessment"]
        assert "compliance" in assessment
        assert isinstance(assessment["compliance"], (int, float))
        assert 0 <= assessment["compliance"] <= 1
        assert "observations" in assessment
        assert isinstance(assessment["observations"], list)
        assert len(assessment["observations"]) > 0
        assert "fitness_trend" in assessment
        assert assessment["fitness_trend"] in ("improving", "stable", "declining")

        assert "recommended_adjustments" in result
        adjustments = result["recommended_adjustments"]
        assert isinstance(adjustments, list)

    @pytest.mark.integration
    def test_adjusted_plan_returns_valid_structure(self):
        """Adjusted plan should have sessions and adjustments_applied."""
        from src.agent.assessment import assess_training
        from src.agent.planner import generate_adjusted_plan

        profile = _test_profile()
        plan = _test_plan()
        activities = _load_mock_activities()

        assessment = assess_training(profile, plan, activities)
        adjusted = generate_adjusted_plan(profile, plan, assessment)

        assert "sessions" in adjusted
        assert isinstance(adjusted["sessions"], list)
        assert len(adjusted["sessions"]) >= 3  # at least 3 training sessions

    @pytest.mark.integration
    def test_full_agent_cycle(self, tmp_path):
        """Full cycle: profile + plan + activities -> assessment + adjusted plan + autonomy."""
        profile = _test_profile()
        plan = _test_plan()

        # Store mock activities
        activities = _load_mock_activities()
        for act in activities:
            store_activity(act, storage_dir=tmp_path)
        stored = list_activities(storage_dir=tmp_path)
        assert len(stored) == 4

        # Run agent cycle
        agent = AgentCore()
        result = agent.run_cycle(profile, plan, stored)

        # Verify assessment
        assert "assessment" in result
        assessment = result["assessment"]
        assert "assessment" in assessment
        assert "recommended_adjustments" in assessment

        # Verify adjusted plan
        assert "adjusted_plan" in result
        assert "sessions" in result["adjusted_plan"]

        # Verify autonomy
        assert "autonomy_result" in result
        autonomy = result["autonomy_result"]
        assert "auto_applied" in autonomy
        assert "proposals" in autonomy

        # Verify state machine ended in IDLE
        assert agent.state == AgentState.IDLE

    @pytest.mark.integration
    def test_assessment_detects_deviations(self):
        """Assessment should notice activities don't match the running-only plan."""
        from src.agent.assessment import assess_training

        profile = _test_profile()
        plan = _test_plan()  # 5 running sessions
        activities = _load_mock_activities()  # 2 runs, 1 bike, 1 gym

        result = assess_training(profile, plan, activities)
        assessment = result["assessment"]

        # Should detect low compliance (4 activities but plan had 5 running sessions,
        # and 2 of the activities aren't even running)
        assert assessment["compliance"] < 1.0
        # Should have observations about the deviations
        assert len(assessment["observations"]) > 0
