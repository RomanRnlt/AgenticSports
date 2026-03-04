"""Unit tests for src.db.health_inventory_db.

Covers:
- get_connected_providers: returns providers, empty, result is new list
- get_available_metric_types: has data, no data, returns dict
- get_activity_sport_summary: multi-source, empty, dedup sports

All Supabase I/O is mocked via patch("src.db.health_inventory_db.get_supabase").
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.db.health_inventory_db import (
    get_activity_sport_summary,
    get_available_metric_types,
    get_connected_providers,
)

USER_ID = "user-inventory-test-uuid"


# ---------------------------------------------------------------------------
# Mock builder (same pattern as test_health_data_db.py)
# ---------------------------------------------------------------------------


def _make_chain(rows: list[dict]) -> MagicMock:
    """Return a chainable query mock whose .execute().data equals *rows*."""
    chain = MagicMock()
    result = MagicMock()
    result.data = rows
    for method in ["select", "eq", "gte", "lt", "order", "limit", "neq", "in_"]:
        getattr(chain, method).return_value = chain
    chain.execute.return_value = result
    return chain


def _mock_supabase(table_data: dict[str, list[dict]]) -> MagicMock:
    """Return a mock Supabase client whose .table(name) resolves *table_data*."""
    client = MagicMock()

    def _table(name: str) -> MagicMock:
        rows = table_data.get(name, [])
        return _make_chain(rows)

    client.table.side_effect = _table
    return client


# ---------------------------------------------------------------------------
# get_connected_providers
# ---------------------------------------------------------------------------


class TestGetConnectedProviders:
    """Tests for get_connected_providers()."""

    def test_returns_providers(self) -> None:
        rows = [
            {"id": "p1", "provider_type": "garmin", "status": "active", "last_sync_at": "2026-03-04T10:00:00", "created_at": "2026-01-01"},
            {"id": "p2", "provider_type": "apple_health", "status": "active", "last_sync_at": "2026-03-03T08:00:00", "created_at": "2026-02-01"},
        ]
        client = _mock_supabase({"health_providers": rows})
        with patch("src.db.health_inventory_db.get_supabase", return_value=client):
            result = get_connected_providers(USER_ID)
        assert len(result) == 2
        assert result[0]["provider_type"] == "garmin"

    def test_returns_empty_list_when_no_providers(self) -> None:
        client = _mock_supabase({"health_providers": []})
        with patch("src.db.health_inventory_db.get_supabase", return_value=client):
            result = get_connected_providers(USER_ID)
        assert result == []

    def test_result_is_new_list(self) -> None:
        rows = [{"id": "p3", "provider_type": "garmin", "status": "active", "last_sync_at": None, "created_at": "2026-01-01"}]
        client = _mock_supabase({"health_providers": rows})
        with patch("src.db.health_inventory_db.get_supabase", return_value=client):
            result = get_connected_providers(USER_ID)
        assert result is not rows

    def test_returns_empty_on_error(self) -> None:
        client = MagicMock()
        client.table.side_effect = RuntimeError("DB error")
        with patch("src.db.health_inventory_db.get_supabase", return_value=client):
            result = get_connected_providers(USER_ID)
        assert result == []


# ---------------------------------------------------------------------------
# get_available_metric_types
# ---------------------------------------------------------------------------


class TestGetAvailableMetricTypes:
    """Tests for get_available_metric_types()."""

    def test_has_data(self) -> None:
        health_rows = [{"sleep_duration_minutes": 480, "sleep_score": 85, "hrv_avg": 62, "resting_heart_rate": None, "stress_avg": None, "body_battery_high": None, "body_battery_low": None, "recovery_score": None, "steps": None}]
        garmin_rows = [{"sleep_duration_minutes": None, "sleep_score": None, "hrv_weekly_avg": None, "resting_heart_rate": 52, "stress_avg": 28, "body_battery_high": 90, "body_battery_low": 40, "steps": 8500, "intensity_minutes": 45, "floors_climbed": 12}]
        client = _mock_supabase({"health_daily_metrics": health_rows, "garmin_daily_stats": garmin_rows})
        with patch("src.db.health_inventory_db.get_supabase", return_value=client):
            result = get_available_metric_types(USER_ID)
        assert result["sleep"] is True
        assert result["hrv"] is True
        assert result["resting_hr"] is True
        assert result["stress"] is True
        assert result["body_battery"] is True
        assert result["steps"] is True
        assert result["intensity_minutes"] is True
        assert result["floors_climbed"] is True

    def test_no_data(self) -> None:
        client = _mock_supabase({"health_daily_metrics": [], "garmin_daily_stats": []})
        with patch("src.db.health_inventory_db.get_supabase", return_value=client):
            result = get_available_metric_types(USER_ID)
        assert isinstance(result, dict)
        # All metrics should be False
        for value in result.values():
            assert value is False

    def test_returns_dict(self) -> None:
        client = _mock_supabase({"health_daily_metrics": [], "garmin_daily_stats": []})
        with patch("src.db.health_inventory_db.get_supabase", return_value=client):
            result = get_available_metric_types(USER_ID)
        assert isinstance(result, dict)
        # Must contain the known unified metric names
        assert "sleep" in result
        assert "hrv" in result
        assert "recovery" in result

    def test_partial_data(self) -> None:
        """Only some metrics have data."""
        health_rows = [{"sleep_duration_minutes": 420, "sleep_score": None, "hrv_avg": None, "resting_heart_rate": None, "stress_avg": None, "body_battery_high": None, "body_battery_low": None, "recovery_score": 80, "steps": None}]
        client = _mock_supabase({"health_daily_metrics": health_rows, "garmin_daily_stats": []})
        with patch("src.db.health_inventory_db.get_supabase", return_value=client):
            result = get_available_metric_types(USER_ID)
        assert result["sleep"] is True
        assert result["recovery"] is True
        assert result["hrv"] is False
        assert result["stress"] is False


# ---------------------------------------------------------------------------
# get_activity_sport_summary
# ---------------------------------------------------------------------------


class TestGetActivitySportSummary:
    """Tests for get_activity_sport_summary()."""

    def test_multi_source(self) -> None:
        health_rows = [
            {"activity_type": "running"},
            {"activity_type": "running"},
            {"activity_type": "cycling"},
        ]
        garmin_rows = [
            {"type": "running"},
            {"type": "swimming"},
        ]
        client = _mock_supabase({"health_activities": health_rows, "garmin_activities": garmin_rows})
        with patch("src.db.health_inventory_db.get_supabase", return_value=client):
            result = get_activity_sport_summary(USER_ID)

        sports = {r["sport"]: r for r in result}
        assert "running" in sports
        assert sports["running"]["count"] == 3
        assert set(sports["running"]["sources"]) == {"health", "garmin"}
        assert "cycling" in sports
        assert sports["cycling"]["count"] == 1
        assert sports["swimming"]["count"] == 1

    def test_empty(self) -> None:
        client = _mock_supabase({"health_activities": [], "garmin_activities": []})
        with patch("src.db.health_inventory_db.get_supabase", return_value=client):
            result = get_activity_sport_summary(USER_ID)
        assert result == []

    def test_dedup_sports_across_sources(self) -> None:
        """Same sport from both sources is merged, not duplicated."""
        health_rows = [{"activity_type": "running"}]
        garmin_rows = [{"type": "running"}]
        client = _mock_supabase({"health_activities": health_rows, "garmin_activities": garmin_rows})
        with patch("src.db.health_inventory_db.get_supabase", return_value=client):
            result = get_activity_sport_summary(USER_ID)
        assert len(result) == 1
        assert result[0]["sport"] == "running"
        assert result[0]["count"] == 2
        assert set(result[0]["sources"]) == {"health", "garmin"}

    def test_sorted_by_count_desc(self) -> None:
        health_rows = [
            {"activity_type": "yoga"},
            {"activity_type": "running"},
            {"activity_type": "running"},
            {"activity_type": "running"},
        ]
        client = _mock_supabase({"health_activities": health_rows, "garmin_activities": []})
        with patch("src.db.health_inventory_db.get_supabase", return_value=client):
            result = get_activity_sport_summary(USER_ID)
        assert result[0]["sport"] == "running"
        assert result[0]["count"] == 3

    def test_unknown_type_handling(self) -> None:
        """Rows with None activity_type should be bucketed as 'unknown'."""
        health_rows = [{"activity_type": None}]
        garmin_rows = [{"type": None}]
        client = _mock_supabase({"health_activities": health_rows, "garmin_activities": garmin_rows})
        with patch("src.db.health_inventory_db.get_supabase", return_value=client):
            result = get_activity_sport_summary(USER_ID)
        assert len(result) == 1
        assert result[0]["sport"] == "unknown"
        assert result[0]["count"] == 2
