"""Unit tests for src.agent.tools.health_inventory_tools.

Covers:
- Tool registration (name in registry, category)
- get_health_inventory: providers, metrics, gaps, empty user
- Tool in data category

All DB and config dependencies are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_ID = "user-inventory-tools-test"

# Reusable empty-metrics dict (all False).
_ALL_FALSE_METRICS = {
    "sleep": False, "hrv": False, "stress": False,
    "body_battery": False, "recovery": False, "resting_hr": False,
    "sleep_score": False, "steps": False,
    "intensity_minutes": False, "floors_climbed": False,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_settings(user_id: str = USER_ID) -> MagicMock:
    """Return a mock Settings with agenticsports_user_id set."""
    s = MagicMock()
    s.agenticsports_user_id = user_id
    return s


def _build_registry(settings: MagicMock | None = None) -> ToolRegistry:
    """Register health inventory tools on a fresh registry."""
    from src.agent.tools.health_inventory_tools import register_health_inventory_tools

    registry = ToolRegistry()
    with patch("src.agent.tools.health_inventory_tools.get_settings", return_value=settings or _mock_settings()):
        register_health_inventory_tools(registry)
    return registry


def _execute_with_mocks(
    registry: ToolRegistry,
    providers: list[dict],
    available_metrics: dict[str, bool],
    activity_sports: list[dict],
    merged: list[dict],
) -> dict:
    """Execute get_health_inventory with all DB calls mocked."""
    with (
        patch("src.db.health_inventory_db.get_connected_providers", return_value=providers),
        patch("src.db.health_inventory_db.get_available_metric_types", return_value=available_metrics),
        patch("src.db.health_inventory_db.get_activity_sport_summary", return_value=activity_sports),
        patch("src.db.health_data_db.get_merged_daily_metrics", return_value=merged),
    ):
        return registry.execute("get_health_inventory", {})


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    """Verify tool registration in the ToolRegistry."""

    def test_get_health_inventory_registered(self) -> None:
        registry = _build_registry()
        tool_names = [t["name"] for t in registry.list_tools()]
        assert "get_health_inventory" in tool_names

    def test_tool_category_is_data(self) -> None:
        registry = _build_registry()
        tools = {t["name"]: t for t in registry.list_tools()}
        assert tools["get_health_inventory"]["category"] == "data"

    def test_tool_source_is_native(self) -> None:
        registry = _build_registry()
        tools = {t["name"]: t for t in registry.list_tools()}
        assert tools["get_health_inventory"]["source"] == "native"


# ---------------------------------------------------------------------------
# get_health_inventory tool
# ---------------------------------------------------------------------------


class TestGetHealthInventory:
    """Tests for the get_health_inventory tool handler."""

    def test_returns_all_sections(self) -> None:
        providers = [{"id": "p1", "provider_type": "garmin", "status": "active", "last_sync_at": "2026-03-04T10:00:00", "created_at": "2026-01-01"}]
        available_metrics = {
            "sleep": True, "hrv": True, "stress": False, "body_battery": False,
            "recovery": False, "resting_hr": True, "sleep_score": True, "steps": True,
            "intensity_minutes": False, "floors_climbed": False,
        }
        activity_sports = [{"sport": "running", "count": 5, "sources": ["garmin"]}]
        merged = [
            {"date": "2026-03-04", "sleep_minutes": 430, "hrv": 55, "source": "garmin"},
            {"date": "2026-03-03", "sleep_minutes": 420, "hrv": 52, "source": "garmin"},
        ]

        registry = _build_registry()
        result = _execute_with_mocks(registry, providers, available_metrics, activity_sports, merged)

        assert "providers" in result
        assert "available_metrics" in result
        assert "activity_sports" in result
        assert "data_coverage_days" in result
        assert "gaps" in result
        assert result["data_coverage_days"] == 2
        assert len(result["providers"]) == 1
        assert result["activity_sports"][0]["sport"] == "running"

    def test_gaps_detects_missing_metrics(self) -> None:
        available_metrics = {
            "sleep": True, "hrv": False, "stress": False, "body_battery": False,
            "recovery": False, "resting_hr": False, "sleep_score": True, "steps": False,
            "intensity_minutes": False, "floors_climbed": False,
        }
        merged = [{"date": "2026-03-04", "sleep_minutes": 430, "hrv": None, "source": "garmin"}]

        registry = _build_registry()
        result = _execute_with_mocks(registry, [], available_metrics, [], merged)

        gaps = result["gaps"]
        gap_texts = " ".join(gaps)
        assert "hrv" in gap_texts
        assert "stress" in gap_texts

    def test_gaps_detects_stale_provider(self) -> None:
        providers = [{"id": "p1", "provider_type": "garmin", "status": "active", "last_sync_at": "2026-01-01T00:00:00", "created_at": "2025-12-01"}]
        all_true = {k: True for k in _ALL_FALSE_METRICS}
        merged = [{"date": "2026-03-04", "sleep_minutes": 430, "hrv": 55, "source": "garmin"}]
        activity_sports = [{"sport": "running", "count": 3, "sources": ["garmin"]}]

        registry = _build_registry()
        result = _execute_with_mocks(registry, providers, all_true, activity_sports, merged)

        gaps = result["gaps"]
        assert any("garmin" in g and "synced" in g for g in gaps)

    def test_empty_user_returns_error(self) -> None:
        settings = _mock_settings(user_id="")
        registry = _build_registry(settings)

        result = registry.execute("get_health_inventory", {})
        assert "error" in result
        assert result["providers"] == []

    def test_no_providers_no_data(self) -> None:
        registry = _build_registry()
        result = _execute_with_mocks(registry, [], _ALL_FALSE_METRICS, [], [])

        assert result["data_coverage_days"] == 0
        assert result["providers"] == []
        assert result["activity_sports"] == []

    def test_gaps_detects_never_synced_provider(self) -> None:
        providers = [{"id": "p1", "provider_type": "apple_health", "status": "active", "last_sync_at": None, "created_at": "2026-01-01"}]

        registry = _build_registry()
        result = _execute_with_mocks(registry, providers, _ALL_FALSE_METRICS, [], [])

        gaps = result["gaps"]
        assert any("never synced" in g for g in gaps)

    def test_gaps_detects_stale_metrics(self) -> None:
        """If newest metric is >7 days old, it should appear in gaps."""
        all_true = {k: True for k in _ALL_FALSE_METRICS}
        merged = [{"date": "2026-02-01", "sleep_minutes": 430, "hrv": 55, "source": "garmin"}]

        registry = _build_registry()
        result = _execute_with_mocks(registry, [], all_true, [], merged)

        gaps = result["gaps"]
        assert any("2026-02-01" in g for g in gaps)

    def test_gaps_no_metrics_with_providers(self) -> None:
        """If providers exist but no metric data, a specific gap should be flagged."""
        providers = [{"id": "p1", "provider_type": "garmin", "status": "active", "last_sync_at": "2026-03-04T10:00:00", "created_at": "2026-01-01"}]

        registry = _build_registry()
        result = _execute_with_mocks(registry, providers, _ALL_FALSE_METRICS, [], [])

        gaps = result["gaps"]
        assert any("No daily metrics" in g for g in gaps)
