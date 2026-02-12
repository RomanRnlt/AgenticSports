"""Startup coaching pipeline: goal type inference and assessment computation.

Computes deterministic assessment data for the startup greeting. All arithmetic
is done here so the LLM never computes numbers -- it only interprets and coaches.

Public API:
    infer_goal_type(goal) -> str
    compute_startup_assessment(plan, activities, goal) -> dict
"""

from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# Goal type classification
# ---------------------------------------------------------------------------

GOAL_TYPES = {"race_target", "performance_target", "routine", "general"}


def infer_goal_type(goal: dict) -> str:
    """Infer goal type from structured_core.goal fields.

    Rules (applied in order):
    1. Has target_time AND target_date AND event -> "race_target"
    2. Event name implies measurable performance -> "performance_target"
    3. Event text mentions frequency/consistency patterns -> "routine"
    4. Everything else -> "general"

    All string matching is case-insensitive.
    """
    event = (goal.get("event") or "").lower()
    has_target_time = bool(goal.get("target_time"))
    has_target_date = bool(goal.get("target_date"))

    # Race target: specific event + time + date
    if has_target_time and has_target_date and event:
        return "race_target"

    # Performance target: event name implies measurable performance
    performance_words = {
        "marathon", "half", "10k", "5k", "triathlon", "ironman",
        "century", "gran fondo", "time trial", "ftp", "improve",
    }
    if event and any(w in event for w in performance_words):
        return "performance_target"

    # Routine: frequency/consistency language
    routine_words = {
        "per week", "x run", "x bike", "x swim", "times a week",
        "consistency", "routine", "regular", "maintain",
    }
    if event and any(w in event for w in routine_words):
        return "routine"

    return "general"


# ---------------------------------------------------------------------------
# Startup assessment computation
# ---------------------------------------------------------------------------


def compute_startup_assessment(
    plan: dict | None,
    activities: list[dict],
    goal: dict,
) -> dict:
    """Compute the full startup assessment for the greeting pipeline.

    Aggregates session matching, weekly compliance, intensity summary, trends,
    goal type, goal summary, and triggers into one dict. All arithmetic is done
    here so the LLM never computes numbers.

    Args:
        plan: Current training plan dict (may be None).
        activities: Full list of activity dicts from activity store.
        goal: structured_core.goal dict from user model.

    Returns:
        Assessment dict matching the StartupAssessment structure.
    """
    now = datetime.now(timezone.utc)
    cutoff_14 = (now - timedelta(days=14)).isoformat()
    recent_activities = [
        a for a in activities
        if a.get("start_time", "") >= cutoff_14
    ]

    # Goal type
    goal_type = goal.get("goal_type") or infer_goal_type(goal)

    # Goal summary
    goal_summary = _build_goal_summary(goal, goal_type)

    # Session matching and weekly compliance
    session_matching = None
    weekly_compliance = None
    intensity_summary = None

    if plan and plan.get("sessions"):
        # Lazy import following established pattern from 02-02
        from src.tools.activity_context import match_plan_sessions

        match_result = match_plan_sessions(plan, recent_activities)
        session_matching = match_result

        # Weekly compliance
        volume_actual_min = round(
            sum(a.get("duration_seconds", 0) for a in recent_activities) / 60
        )
        volume_planned_min = round(
            sum(s.get("duration_minutes", 0) for s in plan.get("sessions", []))
        )
        volume_delta_pct = (
            round(((volume_actual_min - volume_planned_min) / volume_planned_min) * 100, 1)
            if volume_planned_min > 0
            else 0
        )

        weekly_compliance = {
            "compliance_rate": match_result["compliance_rate"],
            "matched_count": match_result["matched_count"],
            "planned_count": match_result["planned_count"],
            "volume_actual_min": volume_actual_min,
            "volume_planned_min": volume_planned_min,
            "volume_delta_pct": volume_delta_pct,
        }

        # Intensity summary from matched sessions
        intensity_summary = _compute_intensity_summary(match_result["matched"])

    # Trends
    trends = _compute_trends(recent_activities)

    # Triggers
    triggers = _detect_triggers(
        weekly_compliance=weekly_compliance,
        trends=trends,
        recent_activities=recent_activities,
    )

    return {
        "goal_type": goal_type,
        "goal_summary": goal_summary,
        "session_matching": session_matching,
        "weekly_compliance": weekly_compliance,
        "intensity_summary": intensity_summary,
        "trends": trends,
        "triggers": triggers,
    }


def _build_goal_summary(goal: dict, goal_type: str) -> str:
    """Build a human-readable one-line goal summary."""
    event = goal.get("event") or "General fitness"
    target_date = goal.get("target_date")
    target_time = goal.get("target_time")

    parts = [event]

    if target_time:
        parts.append(f"under {target_time}")

    if target_date:
        parts.append(f"by {target_date}")
        try:
            target_dt = datetime.fromisoformat(target_date)
            now = datetime.now()
            # Handle date-only strings (no timezone)
            weeks_remaining = max(0, (target_dt - now).days // 7)
            parts.append(f"({weeks_remaining} weeks remaining)")
        except ValueError:
            pass

    if not target_time and not target_date:
        parts.append(f"({goal_type} goal)")

    return " ".join(parts)


def _compute_intensity_summary(matched: list[dict]) -> dict:
    """Compute intensity match summary from matched session pairs."""
    on_target = 0
    lower = 0
    higher = 0

    for m in matched:
        match_val = m.get("intensity_match", "unknown")
        if match_val == "on_target":
            on_target += 1
        elif match_val == "lower_than_planned":
            lower += 1
        elif match_val == "higher_than_planned":
            higher += 1

    total_known = on_target + lower + higher

    if total_known == 0:
        overall = "unknown"
    elif on_target >= total_known / 2:
        overall = "on_target"
    elif higher > lower:
        overall = "slightly_harder_than_planned"
    else:
        overall = "slightly_easier_than_planned"

    return {
        "on_target": on_target,
        "lower_than_planned": lower,
        "higher_than_planned": higher,
        "match": overall,
    }


def _compute_trends(recent_activities: list[dict]) -> dict:
    """Compute volume and TRIMP trends from recent activities."""
    # Lazy import following established pattern
    from src.tools.activity_context import compute_weekly_trends

    weeks = compute_weekly_trends(recent_activities)

    if not weeks:
        return {
            "volume_direction": "no data",
            "trimp_direction": "no data",
            "sessions_per_week": [],
        }

    sessions_per_week = [w["sessions"] for w in weeks]

    # Compute trend direction using first-half vs second-half comparison
    if len(weeks) >= 2:
        half = len(weeks) // 2
        first_half = weeks[:half] if half > 0 else weeks[:1]
        second_half = weeks[half:] if half > 0 else weeks[1:]

        def avg_metric(wks: list[dict], key: str) -> float:
            vals = [w[key] for w in wks]
            return sum(vals) / len(vals) if vals else 0

        vol_first = avg_metric(first_half, "duration_min")
        vol_second = avg_metric(second_half, "duration_min")
        trimp_first = avg_metric(first_half, "trimp")
        trimp_second = avg_metric(second_half, "trimp")

        def trend_word(first: float, second: float) -> str:
            if first == 0:
                return "increasing" if second > 0 else "stable"
            change_pct = ((second - first) / first) * 100
            if change_pct >= 15:
                return f"increasing (+{change_pct:.0f}%)"
            elif change_pct <= -15:
                return f"decreasing ({change_pct:.0f}%)"
            else:
                return "stable"

        volume_direction = trend_word(vol_first, vol_second)
        trimp_direction = trend_word(trimp_first, trimp_second)
    else:
        volume_direction = "stable"
        trimp_direction = "stable"

    return {
        "volume_direction": volume_direction,
        "trimp_direction": trimp_direction,
        "sessions_per_week": sessions_per_week,
    }


def _detect_triggers(
    weekly_compliance: dict | None,
    trends: dict,
    recent_activities: list[dict],
) -> list[dict]:
    """Detect data-driven triggers from computed assessment data.

    Triggers are simple, data-driven, no LLM call. Each trigger is:
    {"type": str, "priority": "high"|"medium"|"low", "data": {...}}
    """
    triggers = []

    # Fatigue warning: TRIMP trend increasing >30%
    trimp_dir = trends.get("trimp_direction", "")
    if "increasing" in trimp_dir:
        # Extract percentage from string like "increasing (+35%)"
        try:
            pct_str = trimp_dir.split("(")[1].rstrip("%)").lstrip("+")
            pct_val = float(pct_str)
            if pct_val > 30:
                triggers.append({
                    "type": "fatigue_warning",
                    "priority": "high",
                    "data": {"trimp_change": trimp_dir},
                })
        except (IndexError, ValueError):
            pass

    # Fitness improving: volume trend increasing and sessions consistent
    vol_dir = trends.get("volume_direction", "")
    sessions = trends.get("sessions_per_week", [])
    if "increasing" in vol_dir and len(sessions) >= 2:
        # Check consistency: no week with 0 sessions
        if all(s > 0 for s in sessions):
            triggers.append({
                "type": "fitness_improving",
                "priority": "low",
                "data": {"volume_trend": vol_dir, "sessions_per_week": sessions},
            })

    # Compliance-based triggers
    if weekly_compliance:
        rate = weekly_compliance.get("compliance_rate", 0)
        if rate < 0.6:
            triggers.append({
                "type": "compliance_low",
                "priority": "medium",
                "data": {
                    "compliance_rate": rate,
                    "matched": weekly_compliance.get("matched_count", 0),
                    "planned": weekly_compliance.get("planned_count", 0),
                },
            })
        elif rate >= 0.9:
            triggers.append({
                "type": "great_consistency",
                "priority": "low",
                "data": {
                    "compliance_rate": rate,
                    "matched": weekly_compliance.get("matched_count", 0),
                    "planned": weekly_compliance.get("planned_count", 0),
                },
            })

    return triggers
