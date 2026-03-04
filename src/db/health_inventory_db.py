"""DB layer for health data inventory — connected providers, available metrics, sport summary.

Provides a read-only view of what health data is available for a given user,
enabling the agent to understand data coverage before making recommendations.

Tables: health_providers, health_daily_metrics, garmin_daily_stats,
health_activities, garmin_activities.

Usage::

    from src.db.health_inventory_db import get_connected_providers, get_available_metric_types

    providers = get_connected_providers(user_id)
    metrics = get_available_metric_types(user_id)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from src.db.client import get_supabase

logger = logging.getLogger(__name__)

# Metric columns to check for data availability in health_daily_metrics.
_HEALTH_METRIC_COLUMNS = (
    "sleep_duration_minutes",
    "sleep_score",
    "hrv_avg",
    "resting_heart_rate",
    "stress_avg",
    "body_battery_high",
    "body_battery_low",
    "recovery_score",
    "steps",
)

# Metric columns in garmin_daily_stats that map to unified names.
_GARMIN_METRIC_COLUMNS = (
    "sleep_duration_minutes",
    "sleep_score",
    "hrv_weekly_avg",
    "resting_heart_rate",
    "stress_avg",
    "body_battery_high",
    "body_battery_low",
    "steps",
    "intensity_minutes",
    "floors_climbed",
)

# Mapping from raw column names to unified metric names.
_UNIFIED_NAMES: dict[str, str] = {
    "sleep_duration_minutes": "sleep",
    "sleep_score": "sleep_score",
    "hrv_avg": "hrv",
    "hrv_weekly_avg": "hrv",
    "resting_heart_rate": "resting_hr",
    "stress_avg": "stress",
    "body_battery_high": "body_battery",
    "body_battery_low": "body_battery",
    "recovery_score": "recovery",
    "steps": "steps",
    "intensity_minutes": "intensity_minutes",
    "floors_climbed": "floors_climbed",
}


def get_connected_providers(user_id: str) -> list[dict]:
    """Query the health_providers table for connected providers.

    Returns a list of dicts with keys: id, provider_type, status,
    last_sync_at, created_at. Returns an empty list on error or no data.
    """
    try:
        result = (
            get_supabase()
            .table("health_providers")
            .select("id,provider_type,status,last_sync_at,created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
        return list(result.data or [])
    except Exception as exc:
        logger.error("Failed to fetch connected providers for %s: %s", user_id, exc)
        return []


def get_available_metric_types(user_id: str) -> dict[str, bool]:
    """Check which metric types have non-null data for the user.

    Scans the last 30 days of health_daily_metrics and garmin_daily_stats
    to determine which metrics have at least one non-null value.

    Returns a dict mapping unified metric names to booleans, e.g.
    ``{"sleep": True, "hrv": True, "stress": False, ...}``.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).date().isoformat()
    available: dict[str, bool] = {}

    # Check health_daily_metrics
    try:
        health_result = (
            get_supabase()
            .table("health_daily_metrics")
            .select(",".join(_HEALTH_METRIC_COLUMNS))
            .eq("user_id", user_id)
            .gte("date", cutoff)
            .limit(30)
            .execute()
        )
        for row in (health_result.data or []):
            for col in _HEALTH_METRIC_COLUMNS:
                if row.get(col) is not None:
                    unified = _UNIFIED_NAMES.get(col, col)
                    available[unified] = True
    except Exception as exc:
        logger.debug("health_daily_metrics scan skipped: %s", exc)

    # Check garmin_daily_stats
    try:
        garmin_result = (
            get_supabase()
            .table("garmin_daily_stats")
            .select(",".join(_GARMIN_METRIC_COLUMNS))
            .eq("user_id", user_id)
            .gte("date", cutoff)
            .limit(30)
            .execute()
        )
        for row in (garmin_result.data or []):
            for col in _GARMIN_METRIC_COLUMNS:
                if row.get(col) is not None:
                    unified = _UNIFIED_NAMES.get(col, col)
                    available[unified] = True
    except Exception as exc:
        logger.debug("garmin_daily_stats scan skipped: %s", exc)

    # Fill in False for known metrics not found
    all_unified = sorted(set(_UNIFIED_NAMES.values()))
    return {name: available.get(name, False) for name in all_unified}


def get_activity_sport_summary(user_id: str) -> list[dict]:
    """Aggregate sport types across health_activities and garmin_activities.

    Returns a list of dicts:
    ``[{"sport": "running", "count": 12, "sources": ["garmin", "health"]}, ...]``

    Sports are discovered dynamically — never hardcoded.
    """
    sport_data: dict[str, dict] = {}  # sport -> {"count": int, "sources": set}

    # health_activities
    try:
        health_result = (
            get_supabase()
            .table("health_activities")
            .select("activity_type")
            .eq("user_id", user_id)
            .limit(1000)
            .execute()
        )
        for row in (health_result.data or []):
            sport = (row.get("activity_type") or "unknown").lower()
            entry = sport_data.get(sport, {"count": 0, "sources": set()})
            sport_data[sport] = {
                "count": entry["count"] + 1,
                "sources": entry["sources"] | {"health"},
            }
    except Exception as exc:
        logger.debug("health_activities sport scan skipped: %s", exc)

    # garmin_activities
    try:
        garmin_result = (
            get_supabase()
            .table("garmin_activities")
            .select("type")
            .eq("user_id", user_id)
            .limit(1000)
            .execute()
        )
        for row in (garmin_result.data or []):
            sport = (row.get("type") or "unknown").lower()
            entry = sport_data.get(sport, {"count": 0, "sources": set()})
            sport_data[sport] = {
                "count": entry["count"] + 1,
                "sources": entry["sources"] | {"garmin"},
            }
    except Exception as exc:
        logger.debug("garmin_activities sport scan skipped: %s", exc)

    return sorted(
        [
            {"sport": sport, "count": info["count"], "sources": sorted(info["sources"])}
            for sport, info in sport_data.items()
        ],
        key=lambda x: x["count"],
        reverse=True,
    )
