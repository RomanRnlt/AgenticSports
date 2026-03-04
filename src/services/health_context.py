"""Health context service — builds recovery summaries for runtime injection.

Provides ``build_health_summary()`` to compute latest metrics and 7-day
averages, and ``format_recovery_context_block()`` to render a compact
Markdown block for the agent's runtime context.

Usage::

    from src.services.health_context import build_health_summary, format_recovery_context_block

    summary = build_health_summary(user_id, days=7)
    if summary and summary["data_available"]:
        block = format_recovery_context_block(summary)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build_health_summary(user_id: str, days: int = 7) -> dict | None:
    """Compute a health recovery summary for the given user.

    Returns a dict with keys:
    - ``latest``: most recent day's metrics (dict)
    - ``averages_7d``: 7-day averages for key metrics (dict)
    - ``data_available``: True if any metrics exist

    Returns ``None`` when no data is available.
    """
    from src.db.health_data_db import get_merged_daily_metrics

    metrics = get_merged_daily_metrics(user_id, days=days)
    if not metrics:
        return None

    latest = metrics[0]  # newest first

    # Compute 7-day averages for key fields
    window = metrics[:7]
    averages = _compute_averages(window)

    return {
        "latest": dict(latest),
        "averages_7d": averages,
        "data_available": True,
        "days_with_data": len(metrics),
    }


def format_recovery_context_block(summary: dict) -> str:
    """Format a health summary into a compact Markdown block for runtime context.

    Returns a multi-line string suitable for injection into the system prompt.
    """
    latest = summary.get("latest", {})
    avgs = summary.get("averages_7d", {})

    lines = ["# Current Recovery Status"]

    # Latest values line
    latest_parts: list[str] = []
    sleep_min = latest.get("sleep_minutes")
    if sleep_min is not None:
        hours = int(sleep_min) // 60
        mins = int(sleep_min) % 60
        sleep_str = f"Sleep {hours}h{mins:02d}"
        score = latest.get("sleep_score")
        if score is not None:
            sleep_str += f" (score {score})"
        latest_parts.append(sleep_str)

    hrv = latest.get("hrv")
    if hrv is not None:
        latest_parts.append(f"HRV {hrv}")

    stress = latest.get("stress")
    if stress is not None:
        latest_parts.append(f"Stress {stress}")

    bb_high = latest.get("body_battery_high")
    if bb_high is not None:
        latest_parts.append(f"Body Battery {bb_high}")

    recovery = latest.get("recovery_score")
    if recovery is not None:
        latest_parts.append(f"Recovery {recovery}")

    if latest_parts:
        lines.append(f"Latest: {', '.join(latest_parts)}")

    # 7-day averages line
    avg_parts: list[str] = []
    for key, label in [
        ("sleep_score", "Sleep"),
        ("hrv", "HRV"),
        ("resting_hr", "RHR"),
        ("stress", "Stress"),
    ]:
        val = avgs.get(key)
        if val is not None:
            avg_parts.append(f"{label} {val}")

    if avg_parts:
        lines.append(f"7d avg: {', '.join(avg_parts)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_averages(metrics: list[dict]) -> dict:
    """Compute averages for key health fields across a list of metric dicts.

    Only includes fields that have at least one non-None value.
    Returns rounded values.
    """
    fields = ["sleep_score", "hrv", "resting_hr", "stress", "body_battery_high"]
    result: dict = {}

    for field in fields:
        values = [m[field] for m in metrics if m.get(field) is not None]
        if values:
            avg = sum(values) / len(values)
            result[field] = round(avg, 1)

    return result
