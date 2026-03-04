"""Tests for define_trigger_rule tool.

Covers:
- Validation: empty name/condition/action, invalid CalcEngine formula
- Success: upsert + get_config roundtrip
- Cooldown: default and custom values
- Similarity warning: near-duplicate names logged
- CalcEngine integration: valid and invalid formulas
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent.tools.registry import ToolRegistry


USER_ID = "test-user-trigger-rule"


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


class TestTriggerRuleValidation:
    def test_empty_name(self) -> None:
        registry = _make_registry()
        result = registry.execute("define_trigger_rule", {
            "name": "", "condition": "x > 1", "action": "alert",
        })
        assert result["status"] == "error"
        assert "name" in result["error"].lower()

    def test_empty_condition(self) -> None:
        registry = _make_registry()
        result = registry.execute("define_trigger_rule", {
            "name": "test_rule", "condition": "", "action": "alert",
        })
        assert result["status"] == "error"
        assert "condition" in result["error"].lower()

    def test_empty_action(self) -> None:
        registry = _make_registry()
        result = registry.execute("define_trigger_rule", {
            "name": "test_rule", "condition": "x > 1", "action": "",
        })
        assert result["status"] == "error"
        assert "action" in result["error"].lower()

    def test_invalid_formula(self) -> None:
        registry = _make_registry()
        result = registry.execute("define_trigger_rule", {
            "name": "test_rule",
            "condition": "import os; os.system('rm -rf /')",
            "action": "alert",
        })
        assert result["status"] == "error"
        assert "Invalid condition formula" in result["error"]

    def test_invalid_formula_syntax_error(self) -> None:
        registry = _make_registry()
        result = registry.execute("define_trigger_rule", {
            "name": "test_rule",
            "condition": "x >>> 1 !!",
            "action": "alert",
        })
        assert result["status"] == "error"
        assert "Invalid condition formula" in result["error"]

    def test_supabase_not_configured(self) -> None:
        registry = _make_registry(use_supabase=False)
        result = registry.execute("define_trigger_rule", {
            "name": "test_rule", "condition": "x > 1", "action": "alert",
        })
        assert result["status"] == "error"
        assert "Supabase" in result["error"]


# ---------------------------------------------------------------------------
# Success tests
# ---------------------------------------------------------------------------


class TestTriggerRuleSuccess:
    @patch("src.db.agent_config_db.get_proactive_trigger_rules", return_value=[])
    @patch("src.db.agent_config_db.upsert_proactive_trigger_rule")
    def test_create_rule(self, mock_upsert, mock_get) -> None:
        mock_upsert.return_value = {
            "user_id": USER_ID,
            "name": "high_fatigue",
            "condition": "total_trimp_7d > 500",
            "action": "Reduce training load",
            "cooldown_hours": 24,
        }
        registry = _make_registry()
        result = registry.execute("define_trigger_rule", {
            "name": "high_fatigue",
            "condition": "total_trimp_7d > 500",
            "action": "Reduce training load",
        })
        assert result["status"] == "success"
        assert result["trigger_rule"]["name"] == "high_fatigue"
        mock_upsert.assert_called_once_with(
            user_id=USER_ID,
            name="high_fatigue",
            condition="total_trimp_7d > 500",
            action="Reduce training load",
            cooldown_hours=24,
        )

    @patch("src.db.agent_config_db.get_proactive_trigger_rules", return_value=[])
    @patch("src.db.agent_config_db.upsert_proactive_trigger_rule")
    def test_custom_cooldown(self, mock_upsert, mock_get) -> None:
        mock_upsert.return_value = {
            "user_id": USER_ID,
            "name": "missed_sessions",
            "condition": "days_since_last_session > 3",
            "action": "Check in on athlete",
            "cooldown_hours": 48,
        }
        registry = _make_registry()
        result = registry.execute("define_trigger_rule", {
            "name": "missed_sessions",
            "condition": "days_since_last_session > 3",
            "action": "Check in on athlete",
            "cooldown_hours": 48,
        })
        assert result["status"] == "success"
        assert result["trigger_rule"]["cooldown_hours"] == 48
        mock_upsert.assert_called_once_with(
            user_id=USER_ID,
            name="missed_sessions",
            condition="days_since_last_session > 3",
            action="Check in on athlete",
            cooldown_hours=48,
        )

    @patch("src.db.agent_config_db.get_proactive_trigger_rules", return_value=[])
    @patch("src.db.agent_config_db.upsert_proactive_trigger_rule")
    def test_compound_condition(self, mock_upsert, mock_get) -> None:
        """CalcEngine supports 'and' / 'or' in conditions."""
        condition = "total_trimp_7d > 500 and avg_hrv_7d < 40"
        mock_upsert.return_value = {
            "user_id": USER_ID,
            "name": "overtraining",
            "condition": condition,
            "action": "Warn about overtraining risk",
            "cooldown_hours": 24,
        }
        registry = _make_registry()
        result = registry.execute("define_trigger_rule", {
            "name": "overtraining",
            "condition": condition,
            "action": "Warn about overtraining risk",
        })
        assert result["status"] == "success"

    @patch("src.db.agent_config_db.get_proactive_trigger_rules", return_value=[])
    @patch("src.db.agent_config_db.upsert_proactive_trigger_rule")
    def test_math_condition(self, mock_upsert, mock_get) -> None:
        """CalcEngine supports math expressions."""
        condition = "total_sessions_7d * 2 + 1 > 10"
        mock_upsert.return_value = {
            "user_id": USER_ID,
            "name": "complex_rule",
            "condition": condition,
            "action": "Do something",
            "cooldown_hours": 24,
        }
        registry = _make_registry()
        result = registry.execute("define_trigger_rule", {
            "name": "complex_rule",
            "condition": condition,
            "action": "Do something",
        })
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Similarity warning tests
# ---------------------------------------------------------------------------


class TestTriggerRuleSimilarityWarning:
    @patch("src.db.agent_config_db.upsert_proactive_trigger_rule")
    @patch("src.db.agent_config_db.get_proactive_trigger_rules")
    def test_similar_name_logs_warning(self, mock_get, mock_upsert, caplog) -> None:
        mock_get.return_value = [
            {"name": "high_fatigue_alert", "condition": "x > 1", "action": "a"},
        ]
        mock_upsert.return_value = {
            "user_id": USER_ID,
            "name": "high_fatigue_alerts",
            "condition": "x > 2",
            "action": "b",
            "cooldown_hours": 24,
        }
        registry = _make_registry()
        import logging
        with caplog.at_level(logging.WARNING):
            result = registry.execute("define_trigger_rule", {
                "name": "high_fatigue_alerts",
                "condition": "x > 2",
                "action": "b",
            })
        assert result["status"] == "success"
        assert any("similar" in r.message.lower() for r in caplog.records)

    @patch("src.db.agent_config_db.upsert_proactive_trigger_rule")
    @patch("src.db.agent_config_db.get_proactive_trigger_rules")
    def test_same_name_no_warning(self, mock_get, mock_upsert, caplog) -> None:
        """Same name = update, should not trigger similarity warning."""
        mock_get.return_value = [
            {"name": "test_rule", "condition": "x > 1", "action": "a"},
        ]
        mock_upsert.return_value = {
            "user_id": USER_ID,
            "name": "test_rule",
            "condition": "x > 2",
            "action": "b",
            "cooldown_hours": 24,
        }
        registry = _make_registry()
        import logging
        with caplog.at_level(logging.WARNING):
            result = registry.execute("define_trigger_rule", {
                "name": "test_rule",
                "condition": "x > 2",
                "action": "b",
            })
        assert result["status"] == "success"
        assert not any("similar" in r.message.lower() for r in caplog.records
                       if "trigger rule" in r.message.lower())

    @patch("src.db.agent_config_db.upsert_proactive_trigger_rule")
    @patch("src.db.agent_config_db.get_proactive_trigger_rules")
    def test_db_error_skips_similarity(self, mock_get, mock_upsert) -> None:
        mock_get.side_effect = Exception("DB error")
        mock_upsert.return_value = {
            "user_id": USER_ID,
            "name": "test",
            "condition": "x > 1",
            "action": "a",
            "cooldown_hours": 24,
        }
        registry = _make_registry()
        result = registry.execute("define_trigger_rule", {
            "name": "test", "condition": "x > 1", "action": "a",
        })
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# get_config roundtrip
# ---------------------------------------------------------------------------


class TestTriggerRuleGetConfig:
    @patch("src.db.agent_config_db.get_proactive_trigger_rules")
    def test_get_config_roundtrip(self, mock_get) -> None:
        mock_get.return_value = [
            {"name": "high_fatigue", "condition": "x > 500",
             "action": "Reduce load", "cooldown_hours": 24},
        ]
        registry = _make_registry()
        result = registry.execute("get_config", {
            "config_type": "proactive_trigger_rules",
        })
        assert result["status"] == "success"
        assert result["count"] == 1
        assert result["items"][0]["name"] == "high_fatigue"


# ---------------------------------------------------------------------------
# update_config roundtrip
# ---------------------------------------------------------------------------


class TestTriggerRuleUpdateConfig:
    @patch("src.db.agent_config_db.update_proactive_trigger_rule")
    def test_update_config_roundtrip(self, mock_update) -> None:
        mock_update.return_value = {
            "name": "high_fatigue",
            "condition": "x > 600",
            "action": "Reduce load significantly",
            "cooldown_hours": 48,
        }
        registry = _make_registry()
        result = registry.execute("update_config", {
            "config_type": "proactive_trigger_rules",
            "name": "high_fatigue",
            "updates": {"condition": "x > 600", "cooldown_hours": 48},
        })
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class TestTriggerRuleRegistration:
    def test_tool_registered(self) -> None:
        registry = _make_registry()
        tools = registry.list_tools()
        tool_names = [t["name"] for t in tools]
        assert "define_trigger_rule" in tool_names

    def test_tool_category(self) -> None:
        registry = _make_registry()
        tools = registry.list_tools()
        tool = next(t for t in tools if t["name"] == "define_trigger_rule")
        assert tool["category"] == "config"
