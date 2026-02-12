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
