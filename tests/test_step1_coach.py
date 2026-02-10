"""Step 1: Tests for the Dumb Coach — profile, prompts, plan generation."""

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parent.parent


class TestProfile:
    """Unit tests for athlete profile creation and persistence."""

    def test_create_profile_structure(self):
        from src.memory.profile import create_profile

        profile = create_profile(
            sports=["running"],
            event="Half Marathon",
            target_date="2026-08-15",
            target_time="1:45:00",
            training_days_per_week=5,
            max_session_minutes=90,
        )
        assert profile["sports"] == ["running"]
        assert profile["goal"]["event"] == "Half Marathon"
        assert profile["goal"]["target_date"] == "2026-08-15"
        assert profile["constraints"]["training_days_per_week"] == 5
        assert profile["constraints"]["max_session_minutes"] == 90
        assert profile["fitness"]["trend"] == "unknown"
        assert profile["created_at"]

    def test_save_and_load_profile(self, tmp_path, monkeypatch):
        from src.memory import profile as profile_mod

        monkeypatch.setattr(profile_mod, "PROFILE_PATH", tmp_path / "profile.json")

        p = profile_mod.create_profile(
            sports=["running", "cycling"],
            event="Marathon",
            target_date="2026-10-01",
            target_time="3:30:00",
            training_days_per_week=4,
            max_session_minutes=120,
        )
        path = profile_mod.save_profile(p)
        assert path.exists()

        loaded = profile_mod.load_profile()
        assert loaded["sports"] == ["running", "cycling"]
        assert loaded["goal"]["event"] == "Marathon"


class TestPrompts:
    """Unit tests for prompt construction."""

    def test_build_plan_prompt_contains_key_info(self):
        from src.agent.prompts import build_plan_prompt

        profile = {
            "sports": ["running"],
            "goal": {"event": "10K", "target_date": "2026-06-01", "target_time": "0:50:00"},
            "fitness": {
                "estimated_vo2max": None,
                "threshold_pace_min_km": None,
                "weekly_volume_km": None,
                "trend": "unknown",
            },
            "constraints": {
                "training_days_per_week": 3,
                "max_session_minutes": 60,
                "available_sports": ["running"],
            },
        }
        prompt = build_plan_prompt(profile)
        assert "10K" in prompt
        assert "3" in prompt
        assert "60" in prompt
        assert "beginner" in prompt.lower() or "unknown" in prompt.lower()

    def test_build_plan_prompt_with_fitness_data(self):
        from src.agent.prompts import build_plan_prompt

        profile = {
            "sports": ["running"],
            "goal": {"event": "Marathon", "target_date": "2026-10-01", "target_time": "3:30:00"},
            "fitness": {
                "estimated_vo2max": 52,
                "threshold_pace_min_km": "4:30",
                "weekly_volume_km": 45,
                "trend": "improving",
            },
            "constraints": {
                "training_days_per_week": 5,
                "max_session_minutes": 120,
                "available_sports": ["running"],
            },
        }
        prompt = build_plan_prompt(profile)
        assert "52" in prompt
        assert "4:30" in prompt
        assert "45" in prompt


class TestPlanGeneration:
    """Integration test: actually calls Gemini to generate a plan."""

    @pytest.mark.integration
    def test_generate_plan_returns_valid_json(self):
        """Generate a real plan via Gemini and validate its structure."""
        import src  # noqa: F401 — load .env
        from src.agent.coach import generate_plan

        profile = {
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

        plan = generate_plan(profile)

        # Must have sessions
        assert "sessions" in plan, f"Plan missing 'sessions': {json.dumps(plan, indent=2)}"
        sessions = plan["sessions"]
        assert len(sessions) >= 3, f"Expected at least 3 sessions, got {len(sessions)}"
        assert len(sessions) <= 7, f"Expected at most 7 sessions, got {len(sessions)}"

        # Each session must have required fields
        required_fields = ["day", "sport", "type", "duration_minutes"]
        for i, session in enumerate(sessions):
            for field in required_fields:
                assert field in session, f"Session {i} missing '{field}': {session}"
            assert isinstance(session["duration_minutes"], (int, float)), (
                f"Session {i} duration_minutes should be a number: {session['duration_minutes']}"
            )
            assert session["duration_minutes"] <= 90, (
                f"Session {i} exceeds max 90min: {session['duration_minutes']}"
            )

        # Must have weekly summary
        assert "weekly_summary" in plan, "Plan missing 'weekly_summary'"

    @pytest.mark.integration
    def test_save_plan_creates_file(self, tmp_path, monkeypatch):
        from src.agent import coach

        monkeypatch.setattr(coach, "PLANS_DIR", tmp_path)

        sample_plan = {"sessions": [], "weekly_summary": {}}
        path = coach.save_plan(sample_plan)
        assert path.exists()
        assert path.name.startswith("plan_")
        assert path.suffix == ".json"

        loaded = json.loads(path.read_text())
        assert loaded == sample_plan
