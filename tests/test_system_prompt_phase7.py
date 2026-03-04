"""Tests for Phase 7 system prompt additions and registry wiring.

Verifies:
- Macrocycle Planning section in system prompt
- Goal Trajectory section in system prompt
- Updated planning sequence (macrocycle_week reference)
- Updated onboarding sequence (macrocycle creation step)
- Registry: goal trajectory tools in default + restricted registries
- Registry: macrocycle tools in default registry
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# System Prompt Content Tests
# ---------------------------------------------------------------------------


class TestSystemPromptMacrocycleSection:
    def test_macrocycle_planning_section_exists(self) -> None:
        from src.agent.system_prompt import STATIC_SYSTEM_PROMPT

        assert "## Macrocycle Planning" in STATIC_SYSTEM_PROMPT

    def test_macrocycle_creation_sequence(self) -> None:
        from src.agent.system_prompt import STATIC_SYSTEM_PROMPT

        assert "create_macrocycle_plan" in STATIC_SYSTEM_PROMPT
        assert "save_macrocycle" in STATIC_SYSTEM_PROMPT
        assert "get_macrocycle" in STATIC_SYSTEM_PROMPT

    def test_planning_sequence_includes_macrocycle_week(self) -> None:
        from src.agent.system_prompt import STATIC_SYSTEM_PROMPT

        assert "macrocycle_week" in STATIC_SYSTEM_PROMPT

    def test_onboarding_sequence_includes_macrocycle(self) -> None:
        from src.agent.system_prompt import ONBOARDING_MODE_INSTRUCTIONS

        assert "create_macrocycle_plan" in ONBOARDING_MODE_INSTRUCTIONS
        assert "save_macrocycle" in ONBOARDING_MODE_INSTRUCTIONS


class TestSystemPromptGoalTrajectorySection:
    def test_goal_trajectory_section_exists(self) -> None:
        from src.agent.system_prompt import STATIC_SYSTEM_PROMPT

        assert "## Goal Trajectory" in STATIC_SYSTEM_PROMPT

    def test_trajectory_statuses_documented(self) -> None:
        from src.agent.system_prompt import STATIC_SYSTEM_PROMPT

        assert "on_track" in STATIC_SYSTEM_PROMPT
        assert "behind" in STATIC_SYSTEM_PROMPT
        assert "at_risk" in STATIC_SYSTEM_PROMPT
        assert "insufficient_data" in STATIC_SYSTEM_PROMPT

    def test_assess_goal_trajectory_tool_referenced(self) -> None:
        from src.agent.system_prompt import STATIC_SYSTEM_PROMPT

        assert "assess_goal_trajectory" in STATIC_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Registry Wiring Tests
# ---------------------------------------------------------------------------


def _mock_settings(**overrides):
    """Create a mock settings object."""
    settings = MagicMock()
    settings.use_supabase = overrides.get("use_supabase", True)
    settings.agenticsports_user_id = overrides.get("user_id", "test-user")
    settings.debug = False
    return settings


def _mock_user_model():
    """Create a mock user model."""
    model = MagicMock()
    model.project_profile.return_value = {
        "name": "Test", "sports": ["running"],
        "goal": {"event": "marathon"}, "constraints": {},
    }
    model.get_active_beliefs.return_value = []
    model.user_id = "test-user"
    return model


class TestRegistryGoalTrajectory:
    @patch("src.agent.mcp.client.load_mcp_tools", return_value=[])
    def test_goal_trajectory_in_default_tools(self, _mcp) -> None:
        from src.agent.tools.registry import get_default_tools

        model = _mock_user_model()
        with patch("src.config.get_settings", return_value=_mock_settings()):
            registry = get_default_tools(model)

        tool_names = [t["name"] for t in registry.list_tools()]
        assert "assess_goal_trajectory" in tool_names

    def test_goal_trajectory_in_restricted_tools(self) -> None:
        from src.agent.tools.registry import get_restricted_tools

        model = _mock_user_model()
        with patch("src.config.get_settings", return_value=_mock_settings()):
            registry = get_restricted_tools(model)

        tool_names = [t["name"] for t in registry.list_tools()]
        assert "assess_goal_trajectory" in tool_names


class TestRegistryMacrocycle:
    @patch("src.agent.mcp.client.load_mcp_tools", return_value=[])
    def test_macrocycle_tools_in_default(self, _mcp) -> None:
        from src.agent.tools.registry import get_default_tools

        model = _mock_user_model()
        with patch("src.config.get_settings", return_value=_mock_settings()):
            registry = get_default_tools(model)

        tool_names = [t["name"] for t in registry.list_tools()]
        assert "create_macrocycle_plan" in tool_names
        assert "get_macrocycle" in tool_names
        assert "save_macrocycle" in tool_names
