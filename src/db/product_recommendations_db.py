"""Supabase CRUD for product_recommendations table.

The agent recommends products (gear, nutrition, recovery tools) based on the
athlete's training context. This module persists those recommendations so
the app can display them as a horizontal product bar.

Each recommendation includes:
- Agent-provided fields: product_name, reason, category, sport
- Search-enriched fields: image_url, price, product_url (from PA-API)
- Monetisation fields: affiliate_url, affiliate_provider
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.db.client import get_supabase

logger = logging.getLogger(__name__)

TABLE = "product_recommendations"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def save_recommendations(user_id: str, recommendations: list[dict]) -> list[dict]:
    """Bulk-insert product recommendations.

    Each dict in *recommendations* must contain at least ``product_name`` and
    ``reason``.  All other fields are optional.

    Returns the inserted rows.
    """
    if not recommendations:
        return []

    rows = []
    now = _now_iso()
    for rec in recommendations:
        row = {
            "user_id": user_id,
            "product_name": rec["product_name"],
            "reason": rec["reason"],
            "product_description": rec.get("product_description"),
            "image_url": rec.get("image_url"),
            "price": rec.get("price"),
            "currency": rec.get("currency", "EUR"),
            "product_url": rec.get("product_url"),
            "affiliate_url": rec.get("affiliate_url"),
            "affiliate_provider": rec.get("affiliate_provider"),
            "category": rec.get("category"),
            "sport": rec.get("sport"),
            "search_query": rec.get("search_query"),
            "source": rec.get("source", "llm"),
            "session_id": rec.get("session_id"),
            "plan_id": rec.get("plan_id"),
            "created_at": now,
            "updated_at": now,
        }
        rows.append(row)

    result = get_supabase().table(TABLE).insert(rows).execute()
    return result.data


def get_recommendations_for_session(user_id: str, session_id: str) -> list[dict]:
    """Return all recommendations linked to a specific training session."""
    result = (
        get_supabase()
        .table(TABLE)
        .select("*")
        .eq("user_id", user_id)
        .eq("session_id", session_id)
        .order("created_at", desc=False)
        .execute()
    )
    return result.data


def get_recommendations_for_plan(user_id: str, plan_id: str) -> list[dict]:
    """Return all recommendations linked to a specific training plan."""
    result = (
        get_supabase()
        .table(TABLE)
        .select("*")
        .eq("user_id", user_id)
        .eq("plan_id", plan_id)
        .order("created_at", desc=False)
        .execute()
    )
    return result.data


def get_recent_recommendations(user_id: str, limit: int = 20) -> list[dict]:
    """Return the most recent *limit* recommendations for a user."""
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


def mark_clicked(recommendation_id: str) -> None:
    """Mark a recommendation as clicked (for analytics)."""
    (
        get_supabase()
        .table(TABLE)
        .update({"clicked": True, "updated_at": _now_iso()})
        .eq("id", recommendation_id)
        .execute()
    )
