"""Tests for src.services.health_context.

Covers:
- build_health_summary: garmin-only, health-only, merged, no data, partial data
- format_recovery_context_block: expected format, missing fields
- Immutability checks
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.services.health_context import (
    build_health_summary,
    format_recovery_context_block,
)


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

USER_ID = "test-user-health-ctx-001"


def _make_metric(
    date: str,
    sleep_minutes: int | None = 472,
    sleep_score: int | None = 85,
    hrv: float | None = 62.0,
    resting_hr: int | None = 52,
    stress: int | None = 20,
    body_battery_high: int | None = 90,
    body_battery_low: int | None = 35,
    recovery_score: int | None = 88,
    steps: int | None = 9200,
    source: str = "garmin",
) -> dict:
    return {
        "date": date,
        "sleep_minutes": sleep_minutes,
        "sleep_score": sleep_score,
        "hrv": hrv,
        "resting_hr": resting_hr,
        "stress": stress,
        "body_battery_high": body_battery_high,
        "body_battery_low": body_battery_low,
        "recovery_score": recovery_score,
        "steps": steps,
        "source": source,
    }


# ---------------------------------------------------------------------------
# build_health_summary
# ---------------------------------------------------------------------------


class TestBuildHealthSummaryNoData:
    def test_returns_none_when_no_data(self) -> None:
        with patch(
            "src.db.health_data_db.get_merged_daily_metrics",
            return_value=[],
        ):
            result = build_health_summary(USER_ID)
        assert result is None


class TestBuildHealthSummaryGarminOnly:
    def test_garmin_only_returns_summary(self) -> None:
        metrics = [
            _make_metric("2026-03-04", source="garmin", recovery_score=None),
            _make_metric("2026-03-03", source="garmin", recovery_score=None),
        ]
        with patch(
            "src.db.health_data_db.get_merged_daily_metrics",
            return_value=metrics,
        ):
            result = build_health_summary(USER_ID, days=7)

        assert result is not None
        assert result["data_available"] is True
        assert result["days_with_data"] == 2
        assert result["latest"]["date"] == "2026-03-04"
        assert result["latest"]["hrv"] == 62.0


class TestBuildHealthSummaryHealthOnly:
    def test_health_only_returns_summary(self) -> None:
        metrics = [
            _make_metric("2026-03-04", source="health", recovery_score=88),
        ]
        with patch(
            "src.db.health_data_db.get_merged_daily_metrics",
            return_value=metrics,
        ):
            result = build_health_summary(USER_ID)

        assert result["data_available"] is True
        assert result["latest"]["recovery_score"] == 88


class TestBuildHealthSummaryMerged:
    def test_merged_data_computes_averages(self) -> None:
        metrics = [
            _make_metric("2026-03-04", hrv=60.0, sleep_score=80),
            _make_metric("2026-03-03", hrv=55.0, sleep_score=75),
            _make_metric("2026-03-02", hrv=65.0, sleep_score=90),
        ]
        with patch(
            "src.db.health_data_db.get_merged_daily_metrics",
            return_value=metrics,
        ):
            result = build_health_summary(USER_ID, days=7)

        assert result["averages_7d"]["hrv"] == 60.0
        assert result["averages_7d"]["sleep_score"] == pytest.approx(81.7, abs=0.1)


class TestBuildHealthSummaryPartialData:
    def test_partial_data_handles_none_fields(self) -> None:
        metrics = [
            _make_metric("2026-03-04", hrv=None, stress=None, sleep_score=80),
            _make_metric("2026-03-03", hrv=55.0, stress=None, sleep_score=None),
        ]
        with patch(
            "src.db.health_data_db.get_merged_daily_metrics",
            return_value=metrics,
        ):
            result = build_health_summary(USER_ID)

        assert result["data_available"] is True
        assert result["averages_7d"]["hrv"] == 55.0
        assert result["averages_7d"]["sleep_score"] == 80.0
        assert "stress" not in result["averages_7d"]


class TestBuildHealthSummaryImmutability:
    def test_returned_dict_is_independent(self) -> None:
        metrics = [_make_metric("2026-03-04")]
        with patch(
            "src.db.health_data_db.get_merged_daily_metrics",
            return_value=metrics,
        ):
            a = build_health_summary(USER_ID)
            b = build_health_summary(USER_ID)
        assert a is not b
        assert a["latest"] is not b["latest"]


# ---------------------------------------------------------------------------
# format_recovery_context_block
# ---------------------------------------------------------------------------


class TestFormatRecoveryContextBlock:
    def test_full_data_produces_expected_format(self) -> None:
        summary = {
            "latest": {
                "date": "2026-03-04",
                "sleep_minutes": 472,
                "sleep_score": 85,
                "hrv": 62.0,
                "stress": 20,
                "body_battery_high": 90,
                "recovery_score": 88,
                "resting_hr": 52,
                "steps": 9200,
            },
            "averages_7d": {
                "sleep_score": 80.0,
                "hrv": 58.0,
                "resting_hr": 52.0,
                "stress": 22.0,
            },
            "data_available": True,
        }
        block = format_recovery_context_block(summary)

        assert block.startswith("# Current Recovery Status")
        assert "Sleep 7h52" in block
        assert "score 85" in block
        assert "HRV 62" in block
        assert "Stress 20" in block
        assert "Body Battery 90" in block
        assert "Recovery 88" in block
        assert "7d avg:" in block
        assert "RHR 52" in block

    def test_missing_sleep_score_omitted_from_latest(self) -> None:
        summary = {
            "latest": {
                "sleep_minutes": 420,
                "sleep_score": None,
                "hrv": 55.0,
                "stress": None,
                "body_battery_high": None,
                "recovery_score": None,
            },
            "averages_7d": {"hrv": 55.0},
            "data_available": True,
        }
        block = format_recovery_context_block(summary)
        assert "Sleep 7h00" in block
        assert "score" not in block
        assert "Stress" not in block
        assert "Body Battery" not in block

    def test_empty_latest_still_produces_header(self) -> None:
        summary = {
            "latest": {},
            "averages_7d": {},
            "data_available": True,
        }
        block = format_recovery_context_block(summary)
        assert block == "# Current Recovery Status"

    def test_no_averages_skips_avg_line(self) -> None:
        summary = {
            "latest": {"hrv": 60.0},
            "averages_7d": {},
            "data_available": True,
        }
        block = format_recovery_context_block(summary)
        assert "7d avg:" not in block
        assert "HRV 60" in block
