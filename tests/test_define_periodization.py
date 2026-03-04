"""Tests for define_periodization tool.

Covers:
- Validation: empty name, empty phases, missing phase fields
- Success: upsert + roundtrip via get_config
- Similarity warning: near-duplicate names logged
- Update: update_config roundtrip
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent.tools.registry import ToolRegistry


USER_ID = "test-user-periodization"

VALID_PHASES = [
    {"name": "Base", "weeks": 4, "focus": "aerobic endurance",
     "intensity_distribution": {"low": 80, "moderate": 15, "high": 5}},
    {"name": "Build", "weeks": 3, "focus": "threshold work",
     "intensity_distribution": {"low": 60, "moderate": 25, "high": 15}},
    {"name": "Peak", "weeks": 2, "focus": "race-specific",
     "intensity_distribution": {"low": 50, "moderate": 20, "high": 30}},
    {"name": "Taper", "weeks": 1, "focus": "recovery",
     "intensity_distribution": {"low": 90, "moderate": 8, "high": 2}},
]


def _make_registry(use_supabase: bool = True) -> ToolRegistry:
    """Create a ToolRegistry with config tools registered."""
    registry = ToolRegistry()
    user_model = MagicMock()
    user_model.user_id = USER_ID

    settings = MagicMock()
    settings.use_supabase = use_supabase
    settings.agenticsports_user_id = USER_ID

    with patch("src.agent.tools.config_tools.get_settings", return_value=settings):
        from src.agent.tools.config_tools import register_config_tools
        register_config_tools(registry, user_model)

    return registry


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestDefinePeriodiationValidation:
    def test_empty_name(self) -> None:
        registry = _make_registry()
        result = registry.execute("define_periodization", {
            "name": "", "phases": VALID_PHASES,
        })
        assert result["status"] == "error"
        assert "name" in result["error"].lower()

    def test_empty_phases(self) -> None:
        registry = _make_registry()
        result = registry.execute("define_periodization", {
            "name": "test_model", "phases": [],
        })
        assert result["status"] == "error"
        assert "non-empty" in result["error"]

    def test_phases_not_list(self) -> None:
        registry = _make_registry()
        result = registry.execute("define_periodization", {
            "name": "test_model", "phases": "not_a_list",
        })
        assert result["status"] == "error"
        assert "non-empty list" in result["error"]

    def test_phase_missing_name(self) -> None:
        registry = _make_registry()
        result = registry.execute("define_periodization", {
            "name": "test_model",
            "phases": [{"weeks": 4}],
        })
        assert result["status"] == "error"
        assert "name" in result["error"]

    def test_phase_missing_weeks(self) -> None:
        registry = _make_registry()
        result = registry.execute("define_periodization", {
            "name": "test_model",
            "phases": [{"name": "Base"}],
        })
        assert result["status"] == "error"
        assert "weeks" in result["error"]

    def test_phase_not_dict(self) -> None:
        registry = _make_registry()
        result = registry.execute("define_periodization", {
            "name": "test_model",
            "phases": ["not_a_dict"],
        })
        assert result["status"] == "error"
        assert "object" in result["error"]

    def test_supabase_not_configured(self) -> None:
        registry = _make_registry(use_supabase=False)
        result = registry.execute("define_periodization", {
            "name": "test_model", "phases": VALID_PHASES,
        })
        assert result["status"] == "error"
        assert "Supabase" in result["error"]


# ---------------------------------------------------------------------------
# Success tests
# ---------------------------------------------------------------------------


class TestDefinePeriodiationSuccess:
    @patch("src.db.agent_config_db.get_periodization_models", return_value=[])
    @patch("src.db.agent_config_db.upsert_periodization_model")
    def test_create_model(self, mock_upsert, mock_get) -> None:
        mock_upsert.return_value = {
            "user_id": USER_ID, "name": "marathon_16w",
            "phases": VALID_PHASES,
        }
        registry = _make_registry()
        result = registry.execute("define_periodization", {
            "name": "marathon_16w", "phases": VALID_PHASES,
        })
        assert result["status"] == "success"
        assert result["periodization_model"]["name"] == "marathon_16w"
        mock_upsert.assert_called_once_with(
            user_id=USER_ID, name="marathon_16w", phases=VALID_PHASES,
        )

    @patch("src.db.agent_config_db.get_periodization_models", return_value=[])
    @patch("src.db.agent_config_db.upsert_periodization_model")
    def test_create_with_description(self, mock_upsert, mock_get) -> None:
        mock_upsert.return_value = {
            "user_id": USER_ID, "name": "half_marathon_12w",
            "phases": VALID_PHASES[:2],
        }
        registry = _make_registry()
        result = registry.execute("define_periodization", {
            "name": "half_marathon_12w",
            "phases": VALID_PHASES[:2],
            "description": "12-week half marathon prep",
        })
        assert result["status"] == "success"

    @patch("src.db.agent_config_db.get_periodization_models", return_value=[])
    @patch("src.db.agent_config_db.upsert_periodization_model")
    def test_minimal_phase(self, mock_upsert, mock_get) -> None:
        """Phases only need name + weeks."""
        minimal = [{"name": "Build", "weeks": 6}]
        mock_upsert.return_value = {
            "user_id": USER_ID, "name": "simple", "phases": minimal,
        }
        registry = _make_registry()
        result = registry.execute("define_periodization", {
            "name": "simple", "phases": minimal,
        })
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Similarity warning tests
# ---------------------------------------------------------------------------


class TestPeriodiationSimilarityWarning:
    @patch("src.db.agent_config_db.upsert_periodization_model")
    @patch("src.db.agent_config_db.get_periodization_models")
    def test_similar_name_logs_warning(self, mock_get, mock_upsert, caplog) -> None:
        """Near-duplicate name should produce a log warning (not block)."""
        mock_get.return_value = [
            {"name": "marathon_16_week", "phases": VALID_PHASES},
        ]
        mock_upsert.return_value = {
            "user_id": USER_ID, "name": "marathon_16_weeks",
            "phases": VALID_PHASES,
        }
        registry = _make_registry()
        import logging
        with caplog.at_level(logging.WARNING):
            result = registry.execute("define_periodization", {
                "name": "marathon_16_weeks", "phases": VALID_PHASES,
            })
        assert result["status"] == "success"
        # Warning should be logged for similar name
        assert any("similar" in r.message.lower() for r in caplog.records)

    @patch("src.db.agent_config_db.upsert_periodization_model")
    @patch("src.db.agent_config_db.get_periodization_models")
    def test_same_name_no_warning(self, mock_get, mock_upsert) -> None:
        """Same name = update, should not trigger similarity warning."""
        mock_get.return_value = [
            {"name": "marathon_16w", "phases": VALID_PHASES},
        ]
        mock_upsert.return_value = {
            "user_id": USER_ID, "name": "marathon_16w", "phases": VALID_PHASES,
        }
        registry = _make_registry()
        result = registry.execute("define_periodization", {
            "name": "marathon_16w", "phases": VALID_PHASES,
        })
        assert result["status"] == "success"

    @patch("src.db.agent_config_db.upsert_periodization_model")
    @patch("src.db.agent_config_db.get_periodization_models")
    def test_db_error_skips_similarity(self, mock_get, mock_upsert) -> None:
        """DB error during similarity check should be silently skipped."""
        mock_get.side_effect = Exception("DB connection error")
        mock_upsert.return_value = {
            "user_id": USER_ID, "name": "test", "phases": VALID_PHASES,
        }
        registry = _make_registry()
        result = registry.execute("define_periodization", {
            "name": "test", "phases": VALID_PHASES,
        })
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# get_config roundtrip
# ---------------------------------------------------------------------------


class TestPeriodiationGetConfig:
    @patch("src.db.agent_config_db.get_periodization_models")
    def test_get_config_roundtrip(self, mock_get) -> None:
        mock_get.return_value = [
            {"name": "marathon_16w", "phases": VALID_PHASES},
        ]
        registry = _make_registry()
        result = registry.execute("get_config", {"config_type": "periodization_models"})
        assert result["status"] == "success"
        assert result["count"] == 1
        assert result["items"][0]["name"] == "marathon_16w"


# ---------------------------------------------------------------------------
# update_config roundtrip
# ---------------------------------------------------------------------------


class TestPeriodiationUpdateConfig:
    @patch("src.db.agent_config_db.update_periodization_model")
    def test_update_config_roundtrip(self, mock_update) -> None:
        new_phases = [{"name": "Base", "weeks": 6, "focus": "endurance"}]
        mock_update.return_value = {
            "name": "marathon_16w", "phases": new_phases,
        }
        registry = _make_registry()
        result = registry.execute("update_config", {
            "config_type": "periodization_models",
            "name": "marathon_16w",
            "updates": {"phases": new_phases},
        })
        assert result["status"] == "success"
        assert result["updated"]["phases"] == new_phases


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class TestPeriodiationRegistration:
    def test_tool_registered(self) -> None:
        registry = _make_registry()
        tools = registry.list_tools()
        tool_names = [t["name"] for t in tools]
        assert "define_periodization" in tool_names

    def test_tool_category(self) -> None:
        registry = _make_registry()
        tools = registry.list_tools()
        tool = next(t for t in tools if t["name"] == "define_periodization")
        assert tool["category"] == "config"
