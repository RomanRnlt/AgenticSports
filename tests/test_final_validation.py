"""Final validation: end-to-end smoke test of the complete agent lifecycle."""

import json
from pathlib import Path

import pytest

from src.memory.profile import create_profile, save_profile, load_profile
from src.agent.coach import generate_plan, save_plan
from src.tools.fit_parser import parse_fit_file
from src.tools.metrics import calculate_trimp, classify_hr_zone
from src.tools.activity_store import store_activity, list_activities, get_weekly_summary
from src.agent.assessment import assess_training
from src.agent.planner import generate_adjusted_plan
from src.agent.autonomy import classify_and_apply
from src.memory.episodes import generate_reflection, store_episode, list_episodes, retrieve_relevant_episodes
from src.agent.trajectory import assess_trajectory, calculate_confidence
from src.agent.proactive import check_proactive_triggers, format_proactive_message
from src.agent.state_machine import AgentCore, AgentState

FIXTURES = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    act = parse_fit_file(str(FIXTURES / name))
    hr = act.get("heart_rate")
    if hr and hr.get("avg"):
        dur = act["duration_seconds"] / 60
        act["trimp"] = calculate_trimp(dur, hr["avg"])
        act["hr_zone"] = classify_hr_zone(hr["avg"])
    return act


class TestFinalValidation:
    @pytest.mark.integration
    def test_end_to_end_lifecycle(self, tmp_path):
        """Complete agent lifecycle: profile -> plan -> import -> assess -> reflect -> trajectory -> proactive.

        This is the definitive end-to-end test that exercises every component.
        """
        # ── Step 1: Create profile ──────────────────────────────────
        profile_path = tmp_path / "athlete"
        profile_path.mkdir()

        profile = create_profile(
            sports=["running"],
            event="Half Marathon",
            target_date="2026-08-15",
            target_time="1:45:00",
            training_days_per_week=5,
            max_session_minutes=90,
        )
        assert profile["goal"]["event"] == "Half Marathon"
        assert profile["constraints"]["training_days_per_week"] == 5

        # ── Step 1: Generate plan ───────────────────────────────────
        plan = generate_plan(profile)
        assert "sessions" in plan
        assert len(plan["sessions"]) >= 3
        plan_path = tmp_path / "plans"
        plan_path.mkdir()
        save_path = save_plan(plan)
        assert save_path.exists()

        # ── Step 2: Import activities ───────────────────────────────
        activities_dir = tmp_path / "activities"

        fixture_names = [
            "easy_run.json", "interval_run.json",
            "week2_mon_easy.json", "week2_wed_tempo.json",
            "week2_sat_long.json", "week2_sun_recovery.json",
            "week3_mon_easy.json", "week3_wed_tempo.json",
            "week3_thu_intervals.json", "week3_sat_long.json",
        ]

        for name in fixture_names:
            act = _load_fixture(name)
            store_activity(act, storage_dir=activities_dir)

        activities = list_activities(storage_dir=activities_dir)
        assert len(activities) == 10

        # Weekly summary
        summary = get_weekly_summary(activities)
        assert summary["total_sessions"] == 10
        assert summary["total_distance_km"] > 0
        assert summary["avg_hr"] is not None

        # ── Step 3: Assess training ─────────────────────────────────
        assessment = assess_training(profile, plan, activities)

        assert "assessment" in assessment
        assert "recommended_adjustments" in assessment
        assess_data = assessment["assessment"]
        assert "compliance" in assess_data
        assert "observations" in assess_data
        assert len(assess_data["observations"]) > 0

        # Graduated autonomy
        adjustments = assessment.get("recommended_adjustments", [])
        autonomy_result = classify_and_apply(adjustments)
        assert "auto_applied" in autonomy_result
        assert "proposals" in autonomy_result

        # Generate adjusted plan
        adjusted_plan = generate_adjusted_plan(profile, plan, assessment)
        assert "sessions" in adjusted_plan
        assert len(adjusted_plan["sessions"]) >= 3

        # ── Step 4: Generate reflections ────────────────────────────
        episodes_dir = tmp_path / "episodes"

        # Week 1 reflection
        w1_activities = activities[:2]
        w1_ep = generate_reflection(
            plan, w1_activities, assessment, profile
        )
        assert "lessons" in w1_ep
        assert len(w1_ep["lessons"]) > 0
        store_episode(w1_ep, storage_dir=episodes_dir)

        # Week 2 reflection (reuse assessment for simplicity)
        w2_activities = activities[2:6]
        w2_ep = generate_reflection(
            plan, w2_activities, assessment, profile
        )
        store_episode(w2_ep, storage_dir=episodes_dir)

        # Week 3 reflection
        w3_activities = activities[6:10]
        w3_ep = generate_reflection(
            plan, w3_activities, assessment, profile
        )
        store_episode(w3_ep, storage_dir=episodes_dir)

        episodes = list_episodes(storage_dir=episodes_dir)
        assert len(episodes) == 3

        # Retrieve relevant episodes
        relevant = retrieve_relevant_episodes(profile, episodes, max_results=3)
        assert len(relevant) >= 2

        # Plan with episodes
        plan_with_episodes = generate_adjusted_plan(
            profile, adjusted_plan, assessment, relevant_episodes=relevant
        )
        assert "sessions" in plan_with_episodes

        # ── Step 5: Trajectory assessment ───────────────────────────
        current_plan_fixture = json.loads((FIXTURES / "week3_plan.json").read_text())
        trajectory = assess_trajectory(profile, activities, episodes, current_plan_fixture)

        assert "goal" in trajectory
        assert trajectory["goal"]["event"] == "Half Marathon"
        assert "confidence" in trajectory
        assert 0.0 <= trajectory["confidence"] <= 1.0
        # 3 weeks of data -> capped at 0.5
        assert trajectory["confidence"] <= 0.5
        assert "trajectory" in trajectory
        assert "recommendations" in trajectory
        assert len(trajectory["recommendations"]) > 0

        # ── Step 5: Proactive communication ─────────────────────────
        triggers = check_proactive_triggers(profile, activities, episodes, trajectory)
        assert len(triggers) > 0

        messages = []
        for trigger in triggers:
            msg = format_proactive_message(trigger, profile)
            assert len(msg) > 10
            messages.append(msg)

        assert len(messages) > 0

        # ── Verify state machine ────────────────────────────────────
        agent = AgentCore()
        assert agent.state == AgentState.IDLE

        # Run a full cycle
        result = agent.run_cycle(profile, current_plan_fixture, activities)
        assert agent.state == AgentState.IDLE
        assert "assessment" in result
        assert "adjusted_plan" in result
        assert "autonomy_result" in result

    def test_all_fixtures_valid_json(self):
        """Verify all fixture files are valid JSON."""
        for path in FIXTURES.glob("*.json"):
            data = json.loads(path.read_text())
            assert isinstance(data, dict), f"{path.name} is not a JSON object"

    def test_confidence_scoring_rules(self):
        """Verify confidence calculation follows spec rules."""
        # <4 weeks -> capped at 0.5
        assert calculate_confidence(20, 0.9, 3) <= 0.5
        assert calculate_confidence(20, 0.9, 2) <= 0.5
        assert calculate_confidence(20, 0.9, 1) <= 0.5

        # <8 weeks -> capped at 0.75
        assert calculate_confidence(30, 0.9, 5) <= 0.75
        assert calculate_confidence(30, 0.9, 7) <= 0.75

        # Inconsistent (<70%) reduces by 0.2
        high_consistency = calculate_confidence(20, 0.9, 6)
        low_consistency = calculate_confidence(20, 0.5, 6)
        assert high_consistency - low_consistency >= 0.15  # approximately 0.2 difference
