"""Phase 8d: Integration tests for multi-user readiness, cost monitoring, and Garmin sync."""

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. Budget check blocks chat at exceeded limit
# ---------------------------------------------------------------------------


def test_budget_check_blocks_chat_when_exceeded():
    """POST /chat should return 429 when daily budget is exceeded."""
    with patch("src.services.usage_tracker.check_budget", return_value=False):
        from fastapi.testclient import TestClient
        from src.api.main import create_app

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        # Mock JWT auth to return a valid user
        with patch("src.api.auth.verify_jwt", return_value={
            "sub": "user-123", "email": "t@t.com", "role": "authenticated",
        }):
            resp = client.post(
                "/chat",
                json={"message": "hello"},
                headers={"Authorization": "Bearer fake-token"},
            )
            assert resp.status_code == 429
            assert "budget" in resp.json().get("detail", "").lower()


# ---------------------------------------------------------------------------
# 2. Garmin router is registered and endpoints are reachable
# ---------------------------------------------------------------------------


def test_garmin_router_registered():
    """The /garmin prefix should be registered in the FastAPI app."""
    from src.api.main import create_app

    app = create_app()
    routes = [r.path for r in app.routes]
    garmin_paths = [r for r in routes if "/garmin" in r]
    assert len(garmin_paths) >= 4, f"Expected >=4 garmin routes, got {garmin_paths}"


def test_garmin_endpoints_require_auth():
    """All /garmin endpoints should return 401 without auth."""
    from fastapi.testclient import TestClient
    from src.api.main import create_app

    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)

    for method, path in [
        ("GET", "/garmin/status"),
        ("POST", "/garmin/connect"),
        ("POST", "/garmin/sync"),
        ("DELETE", "/garmin/disconnect"),
    ]:
        resp = client.request(method, path)
        assert resp.status_code in (401, 403, 422), (
            f"{method} {path} returned {resp.status_code}, expected auth error"
        )


# ---------------------------------------------------------------------------
# 3. sync_garmin_data tool is in the default registry
# ---------------------------------------------------------------------------


def test_sync_garmin_data_in_default_registry():
    """sync_garmin_data should be registered in get_default_tools."""
    mock_user_model = MagicMock()
    mock_user_model.user_id = "user-123"

    with patch("src.agent.mcp.client.load_mcp_tools", return_value=[]):
        from src.agent.tools.registry import get_default_tools

        registry = get_default_tools(mock_user_model, context="coach")
        tool_names = [t["name"] for t in registry.list_tools()]
        assert "sync_garmin_data" in tool_names


# ---------------------------------------------------------------------------
# 4. Service-role policies are consistent across all migrations
# ---------------------------------------------------------------------------


MIGRATIONS_DIR = Path(__file__).parent.parent / "supabase" / "migrations"


def _read_all_sql() -> str:
    parts = []
    for sql_file in sorted(MIGRATIONS_DIR.glob("*.sql")):
        parts.append(sql_file.read_text())
    return "\n".join(parts)


ALL_SQL = _read_all_sql()

# Every table with RLS should have a service_role policy
TABLES_WITH_RLS = [
    "profiles", "activities", "beliefs", "sessions", "session_messages",
    "plans", "episodes", "daily_usage", "import_manifest",
    "metric_definitions", "eval_criteria", "session_schemas",
    "periodization_models", "proactive_trigger_rules", "proactive_queue",
    "pending_actions", "product_recommendations", "macrocycle_plans",
    "goal_trajectory_snapshots", "provider_tokens",
]


def test_all_rls_tables_have_service_role_policy():
    """Every RLS-enabled table must have a service_role bypass policy."""
    for table in TABLES_WITH_RLS:
        pattern = rf"CREATE\s+POLICY\s+.*ON\s+(?:public\.)?{table}\s+.*service_role"
        assert re.search(pattern, ALL_SQL, re.IGNORECASE | re.DOTALL), (
            f"Table '{table}' is missing a service_role policy"
        )


# ---------------------------------------------------------------------------
# 5. Provider tokens migration exists
# ---------------------------------------------------------------------------


def test_provider_tokens_migration_exists():
    """provider_tokens migration should exist and create the table."""
    migration_file = MIGRATIONS_DIR / "20260311000001_provider_tokens.sql"
    assert migration_file.exists(), "provider_tokens migration file missing"
    content = migration_file.read_text()
    assert "provider_tokens" in content
    assert "ENABLE ROW LEVEL SECURITY" in content
