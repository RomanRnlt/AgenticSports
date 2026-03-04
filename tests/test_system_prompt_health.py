"""Tests for health recovery context injection in system prompt.

Verifies that:
- Recovery section is present when health data exists
- Recovery section is absent when no health data
- Recovery section is absent when exception occurs (graceful skip)
"""

from unittest.mock import MagicMock, patch

from src.agent.system_prompt import build_runtime_context


def _make_mock_user_model() -> MagicMock:
    """Create a mock user_model with a minimal profile."""
    profile = {
        "name": "TestAthlete",
        "sports": ["running"],
        "goal": {"event": "Marathon", "target_date": None},
        "constraints": {
            "training_days_per_week": 4,
            "max_session_minutes": 60,
        },
        "fitness": {},
    }
    mock = MagicMock()
    mock.project_profile.return_value = profile
    mock.get_active_beliefs.return_value = []
    mock.get_active_plan_summary.return_value = None
    return mock


def _mock_settings(use_supabase: bool = True, user_id: str = "test-user") -> MagicMock:
    s = MagicMock()
    s.use_supabase = use_supabase
    s.agenticsports_user_id = user_id
    return s


class TestRecoveryContextInjection:
    """Test that health recovery context is injected into runtime context."""

    def test_recovery_section_present_when_data_available(self) -> None:
        summary = {
            "latest": {
                "sleep_minutes": 472,
                "sleep_score": 85,
                "hrv": 62.0,
                "stress": 20,
                "body_battery_high": 90,
                "recovery_score": 88,
            },
            "averages_7d": {"sleep_score": 80.0, "hrv": 58.0, "resting_hr": 52.0},
            "data_available": True,
            "days_with_data": 7,
        }

        with (
            patch(
                "src.config.get_settings",
                return_value=_mock_settings(),
            ),
            patch(
                "src.db.health_data_db.get_merged_daily_metrics",
                return_value=[],
            ),
            patch(
                "src.services.health_context.build_health_summary",
                return_value=summary,
            ),
            patch(
                "src.db.health_data_db.get_cross_source_load_summary",
                return_value={"total_sessions": 0, "sports_seen": [], "sessions_by_source": {}},
            ),
        ):
            result = build_runtime_context(_make_mock_user_model())

        assert "Current Recovery Status" in result

    def test_recovery_section_absent_when_no_data(self) -> None:
        with (
            patch(
                "src.config.get_settings",
                return_value=_mock_settings(),
            ),
            patch(
                "src.services.health_context.build_health_summary",
                return_value=None,
            ),
            patch(
                "src.db.health_data_db.get_cross_source_load_summary",
                return_value={"total_sessions": 0, "sports_seen": [], "sessions_by_source": {}},
            ),
        ):
            result = build_runtime_context(_make_mock_user_model())

        assert "Current Recovery Status" not in result

    def test_recovery_section_absent_when_supabase_disabled(self) -> None:
        with patch(
            "src.config.get_settings",
            return_value=_mock_settings(use_supabase=False),
        ):
            result = build_runtime_context(_make_mock_user_model())

        assert "Current Recovery Status" not in result

    def test_recovery_section_exception_safe(self) -> None:
        """Exception in health_context must not crash context building."""
        with (
            patch(
                "src.config.get_settings",
                return_value=_mock_settings(),
            ),
            patch(
                "src.services.health_context.build_health_summary",
                side_effect=RuntimeError("DB unavailable"),
            ),
            patch(
                "src.db.health_data_db.get_cross_source_load_summary",
                return_value={"total_sessions": 0, "sports_seen": [], "sessions_by_source": {}},
            ),
        ):
            result = build_runtime_context(_make_mock_user_model())

        # No crash, basic sections still present
        assert "TestAthlete" in result
        assert "Current Recovery Status" not in result
