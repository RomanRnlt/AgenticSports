"""Supabase CRUD for macrocycle_plans table.

Macrocycles are multi-week training structures (4-52 weeks) that define
phases (base, build, peak, taper), weekly volume targets, and intensity
distribution. Each user has at most one active macrocycle at a time.

Each macrocycle includes:
- name, total_weeks, start_date
- weeks: JSONB array of weekly plans (phase, focus, volume, key sessions)
- periodization_model_name: optional link to agent-defined periodization
- status: active | completed | archived
- evaluation_score: optional quality score from plan evaluator
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.db.client import get_supabase

logger = logging.getLogger(__name__)

TABLE = "macrocycle_plans"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def store_macrocycle(
    user_id: str,
    name: str,
    total_weeks: int,
    start_date: str,
    weeks: list[dict],
    periodization_model_name: str | None = None,
    evaluation_score: int | None = None,
) -> dict:
    """Insert a new macrocycle plan, deactivating any previous active one.

    Returns the inserted row.
    """
    # Deactivate current active macrocycle (at most one active per user)
    _deactivate_all_active(user_id)

    now = _now_iso()
    row = {
        "user_id": user_id,
        "name": name,
        "total_weeks": total_weeks,
        "start_date": start_date,
        "weeks": weeks,
        "periodization_model_name": periodization_model_name,
        "evaluation_score": evaluation_score,
        "status": "active",
        "created_at": now,
        "updated_at": now,
    }

    result = get_supabase().table(TABLE).insert(row).execute()
    return result.data[0] if result.data else row


def get_active_macrocycle(user_id: str) -> dict | None:
    """Return the active macrocycle for a user, or None."""
    result = (
        get_supabase()
        .table(TABLE)
        .select("*")
        .eq("user_id", user_id)
        .eq("status", "active")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


def get_macrocycle(user_id: str, name: str) -> dict | None:
    """Return a specific macrocycle by name, or None."""
    result = (
        get_supabase()
        .table(TABLE)
        .select("*")
        .eq("user_id", user_id)
        .eq("name", name)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


def list_macrocycles(user_id: str, limit: int = 10) -> list[dict]:
    """Return the most recent macrocycles for a user."""
    result = (
        get_supabase()
        .table(TABLE)
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data


def update_macrocycle(user_id: str, name: str, updates: dict) -> dict | None:
    """Partial update of a macrocycle by name.

    Allowed fields: weeks, evaluation_score, status, periodization_model_name.
    Returns the updated row or None if not found.
    """
    allowed = {"weeks", "evaluation_score", "status", "periodization_model_name"}
    filtered = {k: v for k, v in updates.items() if k in allowed}
    if not filtered:
        return None

    filtered["updated_at"] = _now_iso()

    result = (
        get_supabase()
        .table(TABLE)
        .update(filtered)
        .eq("user_id", user_id)
        .eq("name", name)
        .execute()
    )
    return result.data[0] if result.data else None


def deactivate_macrocycle(user_id: str, name: str) -> dict | None:
    """Archive a macrocycle by name. Returns the updated row."""
    return update_macrocycle(user_id, name, {"status": "archived"})


def _deactivate_all_active(user_id: str) -> None:
    """Set all active macrocycles for a user to archived."""
    (
        get_supabase()
        .table(TABLE)
        .update({"status": "archived", "updated_at": _now_iso()})
        .eq("user_id", user_id)
        .eq("status", "active")
        .execute()
    )
