"""Tests for health trend analysis tools -- analyze_health_trends().

Covers:
- Tool registration and category
- Empty data returns data_available=False
- All metrics analyzed when metric="all"
- Single metric filtered correctly
- Unknown metric returns error
- Trend direction: improving, declining, stable
- Inverted metrics (stress, resting_hr)
- Insufficient data handling
- Immutability of returned dicts
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_ID = "test-user-trend-001"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(user_id: str = USER_ID) -> MagicMock:
    s = MagicMock()
    s.agenticsports_user_id = user_id
    s.use_supabase = True
    return s


def _make_metrics(
    count: int,
    base_hrv: float = 60.0,
    hrv_delta: float = 0.0,
    base_sleep: int = 80,
    sleep_delta: int = 0,
    base_stress: int = 25,
    stress_delta: int = 0,
    base_resting_hr: int = 52,
    resting_hr_delta: int = 0,
    base_steps: int = 9000,
    steps_delta: int = 0,
    base_body_battery: int = 85,
    body_battery_delta: int = 0,
    base_recovery: int = 80,
    recovery_delta: int = 0,
) -> list[dict]:
    """Generate mock merged daily metrics for trend testing.

    Days 0-6 (recent window) get base + delta.
    Days 7+ (prior window) get base only.
    """
    metrics = []
    for i in range(count):
        is_recent = i < 7
        metrics.append({
            "date": f"2026-03-{15 - i:02d}",
            "sleep_minutes": 450,
            "sleep_score": base_sleep + (sleep_delta if is_recent else 0),
            "hrv": base_hrv + (hrv_delta if is_recent else 0),
            "resting_hr": base_resting_hr + (resting_hr_delta if is_recent else 0),
            "stress": base_stress + (stress_delta if is_recent else 0),
            "body_battery_high": base_body_battery + (body_battery_delta if is_recent else 0),
            "body_battery_low": 30,
            "recovery_score": base_recovery + (recovery_delta if is_recent else 0),
            "steps": base_steps + (steps_delta if is_recent else 0),
            "source": "garmin",
        })
    return metrics


def _execute_trend(
    mock_metrics: list[dict],
    args: dict | None = None,
) -> dict:
    """Register health trend tools with mocked DB and execute analyze_health_trends."""
    registry = ToolRegistry()

    with (
        patch(
            "src.agent.tools.health_trend_tools.get_settings",
            return_value=_make_settings(),
        ),
        patch(
            "src.db.health_data_db.get_merged_daily_metrics",
            return_value=mock_metrics,
        ),
    ):
        from src.agent.tools.health_trend_tools import register_health_trend_tools

        register_health_trend_tools(registry)
        return registry.execute("analyze_health_trends", args or {})


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestHealthTrendToolRegistration:
    """Verify the tool is registered with the expected name and category."""

    def test_analyze_health_trends_registered(self) -> None:
        registry = ToolRegistry()
        with patch(
            "src.agent.tools.health_trend_tools.get_settings",
            return_value=_make_settings(),
        ):
            from src.agent.tools.health_trend_tools import register_health_trend_tools

            register_health_trend_tools(registry)

        names = [t["function"]["name"] for t in registry.get_openai_tools()]
        assert "analyze_health_trends" in names

    def test_tool_in_data_category(self) -> None:
        registry = ToolRegistry()
        with patch(
            "src.agent.tools.health_trend_tools.get_settings",
            return_value=_make_settings(),
        ):
            from src.agent.tools.health_trend_tools import register_health_trend_tools

            register_health_trend_tools(registry)

        listed = registry.list_tools()
        trend_tools = [t for t in listed if t["name"] == "analyze_health_trends"]
        assert len(trend_tools) == 1
        assert trend_tools[0]["category"] == "data"


# ---------------------------------------------------------------------------
# Empty data
# ---------------------------------------------------------------------------


class TestAnalyzeHealthTrendsEmpty:
    """Empty data returns data_available=False."""

    def test_empty_data_returns_false(self) -> None:
        result = _execute_trend([])
        assert result["data_available"] is False
        assert result["trends"] == {}


# ---------------------------------------------------------------------------
# All metrics
# ---------------------------------------------------------------------------


class TestAnalyzeHealthTrendsAllMetrics:
    """All metrics analyzed when metric='all'."""

    def test_all_metrics_analyzed(self) -> None:
        metrics = _make_metrics(14)
        result = _execute_trend(metrics, {"metric": "all"})

        assert result["data_available"] is True
        assert result["days_analyzed"] == 14

        expected_keys = {
            "sleep_score", "hrv", "resting_hr", "stress",
            "body_battery_high", "recovery_score", "steps",
        }
        assert set(result["trends"].keys()) == expected_keys

    def test_default_metric_is_all(self) -> None:
        """Calling without metric arg should analyze all metrics."""
        metrics = _make_metrics(14)
        result = _execute_trend(metrics, {})

        assert len(result["trends"]) == 7


# ---------------------------------------------------------------------------
# Specific metric
# ---------------------------------------------------------------------------


class TestAnalyzeHealthTrendsSpecificMetric:
    """Single metric filtered correctly."""

    def test_single_metric_returned(self) -> None:
        metrics = _make_metrics(14)
        result = _execute_trend(metrics, {"metric": "hrv"})

        assert "hrv" in result["trends"]
        assert len(result["trends"]) == 1
        assert result["data_available"] is True

    def test_sleep_score_specific(self) -> None:
        metrics = _make_metrics(14)
        result = _execute_trend(metrics, {"metric": "sleep_score"})

        assert "sleep_score" in result["trends"]
        assert len(result["trends"]) == 1


# ---------------------------------------------------------------------------
# Unknown metric
# ---------------------------------------------------------------------------


class TestAnalyzeHealthTrendsUnknownMetric:
    """Unknown metric returns error."""

    def test_unknown_metric_returns_error(self) -> None:
        metrics = _make_metrics(14)
        result = _execute_trend(metrics, {"metric": "vo2max"})

        assert "error" in result
        assert "vo2max" in result["error"]
        assert "available_metrics" in result

    def test_error_lists_available_metrics(self) -> None:
        metrics = _make_metrics(14)
        result = _execute_trend(metrics, {"metric": "bogus"})

        assert "hrv" in result["available_metrics"]
        assert "sleep_score" in result["available_metrics"]


# ---------------------------------------------------------------------------
# Trend direction: improving
# ---------------------------------------------------------------------------


class TestTrendDirectionImproving:
    """Values going up -> 'improving' for normal metrics."""

    def test_hrv_increasing_is_improving(self) -> None:
        # Recent HRV = 60 + 10 = 70, Prior HRV = 60. Change = +16.7%
        metrics = _make_metrics(14, base_hrv=60.0, hrv_delta=10.0)
        result = _execute_trend(metrics, {"metric": "hrv"})

        trend = result["trends"]["hrv"]
        assert trend["trend_direction"] == "improving"
        assert trend["trend_pct"] > 5

    def test_sleep_score_increasing_is_improving(self) -> None:
        # Recent sleep = 80 + 10 = 90, Prior = 80. Change = +12.5%
        metrics = _make_metrics(14, base_sleep=80, sleep_delta=10)
        result = _execute_trend(metrics, {"metric": "sleep_score"})

        trend = result["trends"]["sleep_score"]
        assert trend["trend_direction"] == "improving"


# ---------------------------------------------------------------------------
# Trend direction: declining
# ---------------------------------------------------------------------------


class TestTrendDirectionDeclining:
    """Values going down -> 'declining' for normal metrics."""

    def test_hrv_decreasing_is_declining(self) -> None:
        # Recent HRV = 60 - 10 = 50, Prior HRV = 60. Change = -16.7%
        metrics = _make_metrics(14, base_hrv=60.0, hrv_delta=-10.0)
        result = _execute_trend(metrics, {"metric": "hrv"})

        trend = result["trends"]["hrv"]
        assert trend["trend_direction"] == "declining"
        assert trend["trend_pct"] < -5

    def test_steps_decreasing_is_declining(self) -> None:
        # Recent steps = 9000 - 2000 = 7000, Prior = 9000. Change = -22.2%
        metrics = _make_metrics(14, base_steps=9000, steps_delta=-2000)
        result = _execute_trend(metrics, {"metric": "steps"})

        trend = result["trends"]["steps"]
        assert trend["trend_direction"] == "declining"


# ---------------------------------------------------------------------------
# Trend direction: stable
# ---------------------------------------------------------------------------


class TestTrendDirectionStable:
    """<5% change -> 'stable'."""

    def test_small_change_is_stable(self) -> None:
        # Recent HRV = 60 + 2 = 62, Prior HRV = 60. Change = +3.3% (< 5%)
        metrics = _make_metrics(14, base_hrv=60.0, hrv_delta=2.0)
        result = _execute_trend(metrics, {"metric": "hrv"})

        trend = result["trends"]["hrv"]
        assert trend["trend_direction"] == "stable"
        assert abs(trend["trend_pct"]) <= 5

    def test_zero_change_is_stable(self) -> None:
        metrics = _make_metrics(14, base_hrv=60.0, hrv_delta=0.0)
        result = _execute_trend(metrics, {"metric": "hrv"})

        trend = result["trends"]["hrv"]
        assert trend["trend_direction"] == "stable"
        assert trend["trend_pct"] == 0.0


# ---------------------------------------------------------------------------
# Inverted metrics (stress, resting_hr)
# ---------------------------------------------------------------------------


class TestTrendDirectionInverted:
    """Stress decreasing -> 'improving', stress increasing -> 'declining'."""

    def test_stress_decreasing_is_improving(self) -> None:
        # Recent stress = 25 - 5 = 20, Prior = 25. Change = -20% -> improving (inverted)
        metrics = _make_metrics(14, base_stress=25, stress_delta=-5)
        result = _execute_trend(metrics, {"metric": "stress"})

        trend = result["trends"]["stress"]
        assert trend["trend_direction"] == "improving"

    def test_stress_increasing_is_declining(self) -> None:
        # Recent stress = 25 + 10 = 35, Prior = 25. Change = +40% -> declining (inverted)
        metrics = _make_metrics(14, base_stress=25, stress_delta=10)
        result = _execute_trend(metrics, {"metric": "stress"})

        trend = result["trends"]["stress"]
        assert trend["trend_direction"] == "declining"

    def test_resting_hr_decreasing_is_improving(self) -> None:
        # Recent resting_hr = 52 - 5 = 47, Prior = 52. Change = -9.6% -> improving (inverted)
        metrics = _make_metrics(14, base_resting_hr=52, resting_hr_delta=-5)
        result = _execute_trend(metrics, {"metric": "resting_hr"})

        trend = result["trends"]["resting_hr"]
        assert trend["trend_direction"] == "improving"

    def test_resting_hr_increasing_is_declining(self) -> None:
        # Recent resting_hr = 52 + 5 = 57, Prior = 52. Change = +9.6% -> declining (inverted)
        metrics = _make_metrics(14, base_resting_hr=52, resting_hr_delta=5)
        result = _execute_trend(metrics, {"metric": "resting_hr"})

        trend = result["trends"]["resting_hr"]
        assert trend["trend_direction"] == "declining"


# ---------------------------------------------------------------------------
# Insufficient data
# ---------------------------------------------------------------------------


class TestTrendInsufficientData:
    """<3 data points per window -> 'insufficient_data'."""

    def test_too_few_days_returns_insufficient(self) -> None:
        # Only 5 days total: 5 in recent, 0 in prior -> insufficient
        metrics = _make_metrics(5)
        result = _execute_trend(metrics, {"metric": "hrv"})

        trend = result["trends"]["hrv"]
        assert trend["trend_direction"] == "insufficient_data"
        assert trend["trend_pct"] is None

    def test_only_recent_window_returns_insufficient(self) -> None:
        # 7 days total: 7 in recent, 0 in prior -> insufficient
        metrics = _make_metrics(7)
        result = _execute_trend(metrics, {"metric": "hrv"})

        trend = result["trends"]["hrv"]
        assert trend["trend_direction"] == "insufficient_data"

    def test_nine_days_has_two_in_prior_insufficient(self) -> None:
        # 9 days: 7 recent + 2 prior -> prior < 3 -> insufficient
        metrics = _make_metrics(9)
        result = _execute_trend(metrics, {"metric": "hrv"})

        trend = result["trends"]["hrv"]
        assert trend["trend_direction"] == "insufficient_data"

    def test_ten_days_has_three_in_prior_sufficient(self) -> None:
        # 10 days: 7 recent + 3 prior -> both >= 3 -> direction computed
        metrics = _make_metrics(10, base_hrv=60.0, hrv_delta=10.0)
        result = _execute_trend(metrics, {"metric": "hrv"})

        trend = result["trends"]["hrv"]
        assert trend["trend_direction"] != "insufficient_data"


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestTrendImmutability:
    """Returned dict is a new object, not a reference to input data."""

    def test_returned_dict_is_new(self) -> None:
        metrics = _make_metrics(14)
        original_first = dict(metrics[0])  # snapshot before call

        result = _execute_trend(metrics, {"metric": "hrv"})

        # Verify input was not mutated
        assert metrics[0] == original_first

        # Verify result is a fresh dict
        assert isinstance(result, dict)
        assert "trends" in result

    def test_trend_values_are_new_dicts(self) -> None:
        metrics = _make_metrics(14)
        result = _execute_trend(metrics, {"metric": "hrv"})

        trend = result["trends"]["hrv"]
        # Trend dict must not be any of the input metric dicts
        for m in metrics:
            assert trend is not m
