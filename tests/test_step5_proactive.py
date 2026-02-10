"""Step 5 tests: Trajectory assessment, confidence scoring, proactive communication."""

import json
from pathlib import Path

import pytest

from src.agent.trajectory import assess_trajectory, calculate_confidence
from src.agent.proactive import (
    check_proactive_triggers,
    format_proactive_message,
)
from src.tools.fit_parser import parse_fit_file
from src.tools.metrics import calculate_trimp, classify_hr_zone

FIXTURES = Path(__file__).parent / "fixtures"


# ── Helpers ───────────────────────────────────────────────────────────

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


def _load_fixtures(names: list[str]) -> list[dict]:
    activities = []
    for name in names:
        act = parse_fit_file(str(FIXTURES / name))
        hr = act.get("heart_rate")
        if hr and hr.get("avg"):
            dur = act["duration_seconds"] / 60
            act["trimp"] = calculate_trimp(dur, hr["avg"])
            act["hr_zone"] = classify_hr_zone(hr["avg"])
        activities.append(act)
    return activities


def _load_plan(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _mock_episodes():
    """Create mock episodes for testing."""
    return [
        {
            "id": "ep_2026-02-02",
            "block": "2026-W05",
            "compliance_rate": 0.4,
            "key_observations": ["Only 2/5 sessions completed"],
            "lessons": ["Need better schedule management for Thursday sessions"],
            "patterns_detected": ["Thursday sessions frequently skipped"],
            "fitness_delta": {"estimated_vo2max_change": "stable", "weekly_volume_trend": "stable"},
            "confidence": 0.5,
        },
        {
            "id": "ep_2026-02-09",
            "block": "2026-W06",
            "compliance_rate": 0.8,
            "key_observations": ["HR improving on easy runs", "Missed Thursday"],
            "lessons": ["Thursday is unreliable for hard sessions", "Easy pace can be updated"],
            "patterns_detected": ["Consistent Thursday skipping", "HR trend improving"],
            "fitness_delta": {"estimated_vo2max_change": "+0.5", "weekly_volume_trend": "increasing"},
            "confidence": 0.6,
        },
        {
            "id": "ep_2026-02-16",
            "block": "2026-W07",
            "compliance_rate": 1.0,
            "key_observations": ["All sessions completed", "Long run improved", "Sunday showed fatigue"],
            "lessons": ["Long run distance can increase", "Watch for fatigue after volume increases"],
            "patterns_detected": ["Steady aerobic improvement"],
            "fitness_delta": {"estimated_vo2max_change": "+0.8", "weekly_volume_trend": "increasing"},
            "confidence": 0.7,
        },
    ]


# ── Confidence Scoring (unit tests, no API) ──────────────────────────

class TestConfidenceScoring:
    def test_less_than_4_weeks_capped_at_05(self):
        conf = calculate_confidence(data_points=10, consistency=0.9, weeks_of_data=3)
        assert conf <= 0.5

    def test_less_than_8_weeks_capped_at_075(self):
        conf = calculate_confidence(data_points=25, consistency=0.9, weeks_of_data=6)
        assert conf <= 0.75

    def test_12_plus_weeks_high_confidence(self):
        conf = calculate_confidence(data_points=60, consistency=0.9, weeks_of_data=12)
        assert conf >= 0.75

    def test_inconsistent_training_reduces_confidence(self):
        high = calculate_confidence(data_points=20, consistency=0.9, weeks_of_data=6)
        low = calculate_confidence(data_points=20, consistency=0.5, weeks_of_data=6)
        assert low < high

    def test_more_data_points_increase_confidence(self):
        few = calculate_confidence(data_points=5, consistency=0.9, weeks_of_data=8)
        many = calculate_confidence(data_points=40, consistency=0.9, weeks_of_data=8)
        assert many >= few

    def test_zero_weeks_very_low(self):
        conf = calculate_confidence(data_points=0, consistency=0.0, weeks_of_data=0)
        assert conf <= 0.15


# ── Proactive Triggers (unit tests, no API) ──────────────────────────

class TestProactiveTriggers:
    def test_on_track_trigger(self):
        trajectory = {
            "trajectory": {"on_track": True, "predicted_race_time": "1:43-1:48"},
            "confidence": 0.65,
            "goal": {"target_time": "1:45:00"},
        }
        triggers = check_proactive_triggers(
            _test_profile(), [], _mock_episodes(), trajectory
        )
        types = [t["type"] for t in triggers]
        assert "on_track" in types

    def test_goal_at_risk_trigger(self):
        trajectory = {
            "trajectory": {"on_track": False, "predicted_race_time": "1:55-2:05"},
            "confidence": 0.6,
            "goal": {"target_time": "1:45:00"},
        }
        triggers = check_proactive_triggers(
            _test_profile(), [], _mock_episodes(), trajectory
        )
        types = [t["type"] for t in triggers]
        assert "goal_at_risk" in types

    def test_missed_session_pattern_trigger(self):
        episodes = _mock_episodes()  # contain "Thursday" skip patterns
        trajectory = {
            "trajectory": {"on_track": True},
            "confidence": 0.5,
        }
        triggers = check_proactive_triggers(
            _test_profile(), [], episodes, trajectory
        )
        types = [t["type"] for t in triggers]
        assert "missed_session_pattern" in types

    def test_fitness_improving_trigger(self):
        episodes = _mock_episodes()  # contain "increasing" volume trend
        trajectory = {
            "trajectory": {"on_track": True},
            "confidence": 0.5,
        }
        triggers = check_proactive_triggers(
            _test_profile(), [], episodes, trajectory
        )
        types = [t["type"] for t in triggers]
        assert "fitness_improving" in types


class TestProactiveMessages:
    def test_on_track_message(self):
        trigger = {"type": "on_track", "data": {"predicted_time": "1:43-1:48", "confidence": 0.65}}
        msg = format_proactive_message(trigger, _test_profile())
        assert "1:43-1:48" in msg
        assert "65%" in msg

    def test_goal_at_risk_message(self):
        trigger = {"type": "goal_at_risk", "data": {"predicted_time": "1:55-2:05", "target_time": "1:45:00"}}
        msg = format_proactive_message(trigger, _test_profile())
        assert "1:55-2:05" in msg
        assert "1:45:00" in msg

    def test_missed_session_message(self):
        trigger = {"type": "missed_session_pattern", "data": {"day": "Thursday", "missed_count": 3}}
        msg = format_proactive_message(trigger, _test_profile())
        assert "Thursday" in msg

    def test_fatigue_warning_message(self):
        trigger = {"type": "fatigue_warning", "data": {"message": "fatigue detected"}}
        msg = format_proactive_message(trigger, _test_profile())
        assert "fatigue" in msg.lower()


# ── Integration Tests (calls Gemini) ─────────────────────────────────

class TestStep5Integration:
    @pytest.mark.integration
    def test_trajectory_assessment(self):
        """Full trajectory assessment with 3 weeks of data."""
        profile = _test_profile()
        plan = _load_plan("week3_plan.json")

        # Load all activities across 3 weeks
        all_activity_files = [
            "easy_run.json", "interval_run.json",
            "week2_mon_easy.json", "week2_wed_tempo.json", "week2_sat_long.json", "week2_sun_recovery.json",
            "week3_mon_easy.json", "week3_wed_tempo.json", "week3_thu_intervals.json",
            "week3_sat_long.json", "week3_sun_fatigued.json",
        ]
        activities = _load_fixtures(all_activity_files)
        episodes = _mock_episodes()

        traj = assess_trajectory(profile, activities, episodes, plan)

        # Verify structure
        assert "goal" in traj
        assert traj["goal"]["event"] == "Half Marathon"
        assert traj["goal"]["weeks_remaining"] is not None
        assert traj["goal"]["weeks_remaining"] > 0

        assert "confidence" in traj
        # 3 weeks of data -> capped at 0.5
        assert traj["confidence"] <= 0.5

        assert "confidence_explanation" in traj
        assert isinstance(traj["confidence_explanation"], str)

        assert "trajectory" in traj
        assert "recommendations" in traj
        assert len(traj["recommendations"]) > 0

    @pytest.mark.integration
    def test_full_cycle_trajectory_proactive(self):
        """End-to-end: activities -> trajectory -> proactive messages."""
        profile = _test_profile()
        plan = _load_plan("week3_plan.json")
        activities = _load_fixtures([
            "easy_run.json", "interval_run.json",
            "week2_mon_easy.json", "week2_wed_tempo.json",
            "week3_mon_easy.json", "week3_wed_tempo.json", "week3_sat_long.json",
        ])
        episodes = _mock_episodes()

        # Get trajectory
        traj = assess_trajectory(profile, activities, episodes, plan)
        assert "trajectory" in traj

        # Get proactive triggers
        triggers = check_proactive_triggers(profile, activities, episodes, traj)
        assert len(triggers) > 0

        # Format messages
        for trigger in triggers:
            msg = format_proactive_message(trigger, profile)
            assert len(msg) > 10  # meaningful message, not empty
