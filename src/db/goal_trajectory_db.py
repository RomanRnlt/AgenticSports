"""Supabase CRUD for goal_trajectory_snapshots table.

Goal trajectory snapshots are append-only records that track an athlete's
progress toward their goals over time. Each snapshot includes:
- trajectory_status: on_track, ahead, behind, at_risk, insufficient_data
- confidence: 0.0-1.0 (how confident the analysis is)
- projected_outcome, analysis, recommendations, risk_factors
- context_snapshot: JSONB with the data used for the analysis

The LLM decides what "on track" means for each sport/goal combination.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.db.client import get_supabase

logger = logging.getLogger(__name__)

TABLE = "goal_trajectory_snapshots"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def save_trajectory_snapshot(
    user_id: str,
    goal_name: str,
    trajectory_status: str,
    confidence: float,
    projected_outcome: str = "",
    analysis: str = "",
    recommendations: list[str] | None = None,
    risk_factors: list[str] | None = None,
    context_snapshot: dict | None = None,
) -> dict:
    """Append a new trajectory snapshot (append-only).

    Returns the inserted row.
    """
    row = {
        "user_id": user_id,
        "goal_name": goal_name,
        "trajectory_status": trajectory_status,
        "confidence": confidence,
        "projected_outcome": projected_outcome,
        "analysis": analysis,
        "recommendations": recommendations or [],
        "risk_factors": risk_factors or [],
        "context_snapshot": context_snapshot or {},
        "created_at": _now_iso(),
    }

    result = get_supabase().table(TABLE).insert(row).execute()
    return result.data[0] if result.data else row


def get_latest_trajectory(user_id: str, goal_name: str) -> dict | None:
    """Return the most recent trajectory snapshot for a specific goal."""
    result = (
        get_supabase()
        .table(TABLE)
        .select("*")
        .eq("user_id", user_id)
        .eq("goal_name", goal_name)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


def list_trajectory_snapshots(
    user_id: str,
    goal_name: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """Return trajectory snapshot history, optionally filtered by goal.

    Returns most recent first.
    """
    query = (
        get_supabase()
        .table(TABLE)
        .select("*")
        .eq("user_id", user_id)
    )

    if goal_name is not None:
        query = query.eq("goal_name", goal_name)

    result = query.order("created_at", desc=True).limit(limit).execute()
    return result.data
