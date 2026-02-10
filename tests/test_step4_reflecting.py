"""Step 4 tests: Episodic memory - reflection generation, storage, retrieval, and plan enhancement."""

import json
from pathlib import Path

import pytest

from src.tools.fit_parser import parse_fit_file
from src.tools.metrics import calculate_trimp, classify_hr_zone
from src.memory.episodes import (
    generate_reflection,
    store_episode,
    list_episodes,
    retrieve_relevant_episodes,
)

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
    """Load fixture files and compute TRIMP."""
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


def _make_assessment(compliance: float, observations: list[str], trend: str = "stable") -> dict:
    """Create a mock assessment dict."""
    return {
        "assessment": {
            "compliance": compliance,
            "observations": observations,
            "fitness_trend": trend,
            "fatigue_level": "low",
            "injury_risk": "low",
        },
        "recommended_adjustments": [],
    }


# ── Episode Storage (unit tests, no API) ─────────────────────────────

class TestEpisodeStorage:
    def test_store_and_list(self, tmp_path):
        episode = {
            "id": "ep_2026-02-02",
            "block": "2026-W05",
            "lessons": ["Test lesson"],
            "key_observations": ["Test observation"],
            "patterns_detected": [],
            "confidence": 0.7,
        }
        store_episode(episode, storage_dir=tmp_path)

        episodes = list_episodes(storage_dir=tmp_path)
        assert len(episodes) == 1
        assert episodes[0]["id"] == "ep_2026-02-02"
        assert episodes[0]["lessons"] == ["Test lesson"]

    def test_list_episodes_most_recent_first(self, tmp_path):
        for i, date in enumerate(["2026-02-02", "2026-02-09", "2026-02-16"]):
            ep = {"id": f"ep_{date}", "block": f"2026-W{5+i:02d}", "lessons": [f"Lesson {i}"]}
            store_episode(ep, storage_dir=tmp_path)

        episodes = list_episodes(storage_dir=tmp_path)
        assert len(episodes) == 3
        # Most recent first (reverse alphabetical by filename)
        assert episodes[0]["id"] == "ep_2026-02-16"
        assert episodes[2]["id"] == "ep_2026-02-02"

    def test_list_episodes_with_limit(self, tmp_path):
        for date in ["2026-02-02", "2026-02-09", "2026-02-16"]:
            store_episode({"id": f"ep_{date}", "lessons": []}, storage_dir=tmp_path)

        episodes = list_episodes(storage_dir=tmp_path, limit=2)
        assert len(episodes) == 2

    def test_list_empty_directory(self, tmp_path):
        episodes = list_episodes(storage_dir=tmp_path)
        assert episodes == []


class TestEpisodeRetrieval:
    def test_retrieve_returns_relevant_episodes(self):
        episodes = [
            {
                "id": "ep_1",
                "key_observations": ["Easy run HR was 145, too high for zone 2"],
                "lessons": ["Need to slow down easy runs to keep HR in zone 2"],
                "patterns_detected": ["Consistently running easy runs too fast"],
            },
            {
                "id": "ep_2",
                "key_observations": ["Swimming technique improved"],
                "lessons": ["Focus on bilateral breathing"],
                "patterns_detected": ["Swim times improving"],
            },
            {
                "id": "ep_3",
                "key_observations": ["Long run pace improved"],
                "lessons": ["Increasing volume working well"],
                "patterns_detected": ["Steady aerobic improvement"],
            },
        ]

        context = {
            "sports": ["running"],
            "goal": {"event": "Half Marathon"},
            "fitness": {"trend": "improving"},
        }

        result = retrieve_relevant_episodes(context, episodes, max_results=2)
        assert len(result) == 2
        # Running-related episodes should score higher than swimming
        ids = [r["id"] for r in result]
        assert "ep_2" not in ids  # swimming episode should be filtered out

    def test_retrieve_empty_episodes(self):
        result = retrieve_relevant_episodes({}, [], max_results=5)
        assert result == []

    def test_retrieve_respects_max_results(self):
        episodes = [
            {"id": f"ep_{i}", "key_observations": ["pace improved"], "lessons": ["run more"], "patterns_detected": []}
            for i in range(10)
        ]
        result = retrieve_relevant_episodes({"sports": ["running"]}, episodes, max_results=3)
        assert len(result) == 3


# ── Integration Tests (calls Gemini) ─────────────────────────────────

class TestStep4Integration:
    @pytest.mark.integration
    def test_generate_reflection_week1(self):
        """Generate a reflection for Week 1 and verify structure."""
        profile = _test_profile()
        plan = _load_plan("week1_plan.json")
        # Week 1 activities: easy run + interval (from original fixtures)
        activities = _load_fixtures(["easy_run.json", "interval_run.json"])
        assessment = _make_assessment(
            compliance=0.4,
            observations=[
                "Only 2 of 5 prescribed sessions completed",
                "Easy run HR 135 appropriate for zone 2",
            ],
            trend="stable",
        )

        episode = generate_reflection(plan, activities, assessment, profile)

        assert "id" in episode
        assert "key_observations" in episode
        assert len(episode["key_observations"]) > 0
        assert "lessons" in episode
        assert len(episode["lessons"]) > 0
        assert "patterns_detected" in episode
        assert "fitness_delta" in episode
        assert "confidence" in episode
        assert 0 <= episode["confidence"] <= 1

    @pytest.mark.integration
    def test_three_week_reflection_cycle(self, tmp_path):
        """Generate reflections for 3 weeks, store them, verify lessons accumulate."""
        profile = _test_profile()

        # Week 1
        w1_plan = _load_plan("week1_plan.json")
        w1_acts = _load_fixtures(["easy_run.json", "interval_run.json"])
        w1_assess = _make_assessment(0.4, ["Only 2/5 sessions completed", "Paces slightly fast"])
        w1_ep = generate_reflection(w1_plan, w1_acts, w1_assess, profile)
        store_episode(w1_ep, storage_dir=tmp_path)

        # Week 2 (missed Thursday)
        w2_plan = _load_plan("week2_plan.json")
        w2_acts = _load_fixtures([
            "week2_mon_easy.json", "week2_wed_tempo.json",
            "week2_sat_long.json", "week2_sun_recovery.json",
        ])
        w2_assess = _make_assessment(0.8, ["Missed Thursday intervals", "HR improving on easy runs"], "improving")
        w2_ep = generate_reflection(w2_plan, w2_acts, w2_assess, profile)
        store_episode(w2_ep, storage_dir=tmp_path)

        # Week 3 (all completed, one fatigued session)
        w3_plan = _load_plan("week3_plan.json")
        w3_acts = _load_fixtures([
            "week3_mon_easy.json", "week3_wed_tempo.json", "week3_thu_intervals.json",
            "week3_sat_long.json", "week3_sun_fatigued.json",
        ])
        w3_assess = _make_assessment(1.0, [
            "All sessions completed",
            "Long run pace improved to 5:52/km at HR 135",
            "Sunday recovery run showed elevated HR 142 suggesting fatigue",
        ], "improving")
        w3_ep = generate_reflection(w3_plan, w3_acts, w3_assess, profile)
        store_episode(w3_ep, storage_dir=tmp_path)

        # Verify all 3 stored
        episodes = list_episodes(storage_dir=tmp_path)
        assert len(episodes) == 3

        # All episodes should have lessons
        for ep in episodes:
            assert len(ep.get("lessons", [])) > 0

        # Retrieve relevant episodes for Week 4 planning
        context = _test_profile()
        relevant = retrieve_relevant_episodes(context, episodes, max_results=3)
        assert len(relevant) >= 2  # Should find at least 2 relevant episodes

    @pytest.mark.integration
    def test_plan_with_episodes(self):
        """Generate a plan that incorporates episode lessons."""
        from src.agent.planner import generate_adjusted_plan

        profile = _test_profile()
        plan = _load_plan("week3_plan.json")
        assessment = _make_assessment(0.8, ["Improving fitness", "Some fatigue on Sunday"], "improving")

        episodes = [
            {
                "id": "ep_2026-02-02",
                "block": "2026-W05",
                "lessons": [
                    "Thursday is unreliable for hard sessions -- consider moving to Wednesday",
                    "Easy run pace can be updated to 5:20/km based on HR improvement",
                ],
                "patterns_detected": [
                    "Consistent Thursday session skipping (2 out of 3 weeks)",
                ],
                "key_observations": [],
            },
            {
                "id": "ep_2026-02-09",
                "block": "2026-W06",
                "lessons": [
                    "Long run distance can increase to 15km based on steady aerobic development",
                ],
                "patterns_detected": [],
                "key_observations": [],
            },
        ]

        adjusted = generate_adjusted_plan(profile, plan, assessment, relevant_episodes=episodes)
        assert "sessions" in adjusted
        assert len(adjusted["sessions"]) >= 3
