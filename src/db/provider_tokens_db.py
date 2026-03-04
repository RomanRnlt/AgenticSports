"""Supabase CRUD for provider_tokens table.

Stores OAuth / session tokens for external data providers (Garmin, Apple Health,
Strava, etc.).  Each (user_id, provider) pair is unique — operations upsert on
that composite key so reconnecting a provider simply overwrites the old row.

All write operations return new dicts / values rather than mutating inputs
(immutability pattern).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.db.client import get_supabase

logger = logging.getLogger(__name__)

TABLE = "provider_tokens"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def store_token(
    user_id: str,
    provider: str,
    token_data: dict,
    provider_user_id: str | None = None,
) -> dict:
    """Store or update provider tokens (upsert on user_id+provider).

    Args:
        user_id: Supabase user UUID.
        provider: Provider name (e.g. "garmin", "strava").
        token_data: Provider-specific token payload (opaque JSONB).
        provider_user_id: Display name / ID within the provider system.

    Returns:
        The upserted row dict.
    """
    client = get_supabase()
    row = {
        "user_id": user_id,
        "provider": provider,
        "token_data": token_data,
        "provider_user_id": provider_user_id,
        "status": "active",
        "updated_at": _now_iso(),
    }
    result = client.table(TABLE).upsert(
        row, on_conflict="user_id,provider"
    ).execute()
    return result.data[0] if result.data else row


def get_token(user_id: str, provider: str) -> dict | None:
    """Return the stored token row for a user+provider, or None if not found."""
    client = get_supabase()
    result = (
        client.table(TABLE)
        .select("*")
        .eq("user_id", user_id)
        .eq("provider", provider)
        .execute()
    )
    return result.data[0] if result.data else None


def update_last_sync(user_id: str, provider: str) -> None:
    """Update the last_sync_at timestamp to now."""
    client = get_supabase()
    client.table(TABLE).update({
        "last_sync_at": _now_iso(),
    }).eq("user_id", user_id).eq("provider", provider).execute()


def update_token_status(user_id: str, provider: str, status: str) -> None:
    """Update token status to one of: active, expired, revoked.

    Args:
        user_id: Supabase user UUID.
        provider: Provider name.
        status: New status value ("active" | "expired" | "revoked").
    """
    client = get_supabase()
    client.table(TABLE).update({
        "status": status,
        "updated_at": _now_iso(),
    }).eq("user_id", user_id).eq("provider", provider).execute()


def delete_token(user_id: str, provider: str) -> None:
    """Delete the stored token for a user+provider combo."""
    client = get_supabase()
    client.table(TABLE).delete().eq(
        "user_id", user_id
    ).eq("provider", provider).execute()


def list_connected_providers(user_id: str) -> list[str]:
    """Return provider names with active tokens for the given user."""
    client = get_supabase()
    result = (
        client.table(TABLE)
        .select("provider")
        .eq("user_id", user_id)
        .eq("status", "active")
        .execute()
    )
    return [row["provider"] for row in result.data] if result.data else []
