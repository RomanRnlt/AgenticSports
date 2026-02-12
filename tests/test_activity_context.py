"""Unit tests for activity context builder formatting and aggregation functions."""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from src.tools.activity_context import (
    build_activity_context,
    compute_weekly_trends,
    format_pace,
    format_zone_distribution,
)


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def sample_running_activity():
    """A realistic running activity matching the real JSON schema."""
    return {
        "source_file": "test_running.fit",
        "sport": "running",
        "sub_sport": "generic",
        "start_time": "2026-02-08T11:00:45+00:00",
        "duration_seconds": 4139,
        "distance_meters": 11117,
        "heart_rate": {"avg": 148, "max": 177, "min": 94},
        "pace": {"avg_min_per_km": 5.98, "best_min_per_km": 4.36},
        "speed": None,
        "power": {"avg_watts": 289, "max_watts": 411, "normalized_watts": None},
        "elevation": {"gain_meters": 93, "loss_meters": 95},
        "calories": 775,
        "zone_distribution": {
            "zone_1_seconds": 27.886,
            "zone_2_seconds": 186.5,
            "zone_3_seconds": 3518.872,
            "zone_4_seconds": 241.121,
            "zone_5_seconds": 8.004,
        },
        "zone_distribution_source": "device",
        "trimp": 109.6,
        "hr_zone": 2,
    }


@pytest.fixture
def sample_cycling_activity():
    """A realistic cycling activity matching the real JSON schema."""
    return {
        "source_file": "test_cycling.fit",
        "sport": "cycling",
        "sub_sport": "virtual_activity",
        "start_time": "2026-02-09T16:30:16+00:00",
        "duration_seconds": 3267,
        "distance_meters": 25516,
        "heart_rate": {"avg": 156, "max": 183, "min": 82},
        "pace": None,
        "speed": {"avg_km_h": 29.2, "max_km_h": 53.81},
        "power": {"avg_watts": 164, "max_watts": 251, "normalized_watts": None},
        "elevation": {"gain_meters": 252, "loss_meters": 263},
        "calories": 481,
        "zone_distribution": None,
        "zone_distribution_source": None,
        "trimp": 106.2,
        "hr_zone": 3,
    }


@pytest.fixture
def sample_strength_activity():
    """A realistic strength activity matching the real JSON schema."""
    return {
        "source_file": "test_strength.fit",
        "sport": "strength",
        "sub_sport": "strength_training",
        "start_time": "2026-02-08T08:39:11+00:00",
        "duration_seconds": 3946,
        "distance_meters": None,
        "heart_rate": {"avg": 105, "max": 142, "min": 78},
        "pace": None,
        "speed": None,
        "power": None,
        "elevation": None,
        "calories": 345,
        "zone_distribution": {
            "zone_1_seconds": 3243.325,
            "zone_2_seconds": 676.999,
            "zone_3_seconds": 26.001,
            "zone_4_seconds": 0.0,
            "zone_5_seconds": 0.0,
        },
        "zone_distribution_source": "device",
        "trimp": 28.3,
        "hr_zone": 1,
    }


# ── format_pace tests ────────────────────────────────────────────


class TestFormatPace:
    def test_format_pace_running(self):
        """Standard running paces convert to MM:SS/km."""
        assert format_pace(5.98) == "5:59/km"
        assert format_pace(4.0) == "4:00/km"
        assert format_pace(6.50) == "6:30/km"

    def test_format_pace_swimming(self):
        """Swimming pace uses /100m unit."""
        assert format_pace(1.73, "100m") == "1:44/100m"

    def test_format_pace_edge_seconds_60(self):
        """When fractional part rounds to 60 seconds, roll to next minute."""
        # 4.999 -> 4 min + 0.999*60 = 59.94 -> rounds to 60 -> should be 5:00
        result = format_pace(4.999)
        assert result == "5:00/km", f"Expected '5:00/km' not '{result}'"


# ── format_zone_distribution tests ───────────────────────────────


class TestFormatZoneDistribution:
    def test_format_zone_distribution_normal(self):
        """Real zone data produces correct percentages."""
        zones = {
            "zone_1_seconds": 27.886,
            "zone_2_seconds": 186.5,
            "zone_3_seconds": 3518.872,
            "zone_4_seconds": 241.121,
            "zone_5_seconds": 8.004,
        }
        # Total: ~3982.383 seconds
        total_dur = 3982.383
        result = format_zone_distribution(zones, total_dur)
        assert "Z3 88%" in result
        assert "Z1" in result
        assert "Z5 <1%" in result

    def test_format_zone_distribution_none(self):
        """None zones or 0 duration returns empty string."""
        assert format_zone_distribution(None, 3600) == ""
        assert format_zone_distribution({"zone_1_seconds": 100}, 0) == ""

    def test_format_zone_distribution_all_zone1(self):
        """All time in zone 1 returns Z1 100% and others 0%."""
        zones = {
            "zone_1_seconds": 3600.0,
            "zone_2_seconds": 0.0,
            "zone_3_seconds": 0.0,
            "zone_4_seconds": 0.0,
            "zone_5_seconds": 0.0,
        }
        result = format_zone_distribution(zones, 3600)
        assert result == "Z1 100% | Z2 0% | Z3 0% | Z4 0% | Z5 0%"


# ── compute_weekly_trends tests ──────────────────────────────────


class TestComputeWeeklyTrends:
    def test_groups_by_week(self):
        """Activities across 2 ISO weeks produce 2 week groups."""
        # Monday Jan 19 and Monday Jan 26 are different ISO weeks
        activities = [
            {"start_time": "2026-01-19T10:00:00+00:00", "duration_seconds": 3600,
             "distance_meters": 10000, "trimp": 80},
            {"start_time": "2026-01-21T10:00:00+00:00", "duration_seconds": 3600,
             "distance_meters": 10000, "trimp": 90},
            {"start_time": "2026-01-26T10:00:00+00:00", "duration_seconds": 3600,
             "distance_meters": 10000, "trimp": 100},
        ]
        result = compute_weekly_trends(activities)
        assert len(result) == 2
        assert result[0]["week_start"] < result[1]["week_start"]

    def test_empty(self):
        """Empty list returns empty list."""
        assert compute_weekly_trends([]) == []

    def test_aggregates(self):
        """Sessions, duration, distance, trimp are correctly summed per week."""
        activities = [
            {"start_time": "2026-01-19T10:00:00+00:00", "duration_seconds": 3600,
             "distance_meters": 10000, "trimp": 80},
            {"start_time": "2026-01-21T10:00:00+00:00", "duration_seconds": 1800,
             "distance_meters": 5000, "trimp": 40},
        ]
        result = compute_weekly_trends(activities)
        assert len(result) == 1
        week = result[0]
        assert week["sessions"] == 2
        assert week["duration_min"] == 90  # (3600 + 1800) / 60
        assert week["distance_km"] == 15.0  # (10000 + 5000) / 1000
        assert week["trimp"] == 120  # 80 + 40


# ── build_activity_context tests ─────────────────────────────────


class TestBuildActivityContext:
    def test_empty(self):
        """Mocked empty list_activities returns graceful message."""
        with patch("src.tools.activity_context.list_activities", return_value=[]):
            ctx = build_activity_context()
            assert ctx == "No training data available."

    def test_with_data(
        self,
        sample_running_activity,
        sample_cycling_activity,
        sample_strength_activity,
    ):
        """Mocked activities produce context with expected sections."""
        mock_activities = [
            sample_strength_activity,  # earliest
            sample_running_activity,   # middle
            sample_cycling_activity,   # most recent
        ]
        with patch("src.tools.activity_context.list_activities", return_value=mock_activities):
            ctx = build_activity_context()
            assert "LAST SESSION" in ctx
            assert "THIS WEEK" in ctx
            # Verify sport-specific formatting
            assert "Cycling" in ctx
            assert "Running" in ctx or "running" in ctx
            assert len(ctx) > 100
