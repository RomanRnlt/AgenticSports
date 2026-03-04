"""Phase 8a: RLS Audit Tests.

Verifies:
- All tables are covered in migrations with service_role policies
- get_user_id dependency extracts sub correctly
- get_user_id raises 401 for missing tokens
"""

import re
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from src.api.auth import get_current_user, get_user_id

MIGRATIONS_DIR = Path(__file__).parent.parent / "supabase" / "migrations"


# ---------------------------------------------------------------------------
# Migration coverage tests
# ---------------------------------------------------------------------------

def _read_all_migrations() -> str:
    """Concatenate all migration SQL files into a single string."""
    parts = []
    for sql_file in sorted(MIGRATIONS_DIR.glob("*.sql")):
        parts.append(sql_file.read_text())
    return "\n".join(parts)


ALL_SQL = _read_all_migrations()

# Tables managed by the AgenticSports backend (each has RLS enabled).
EXPECTED_TABLES = [
    "profiles",
    "activities",
    "beliefs",
    "sessions",
    "session_messages",
    "plans",
    "episodes",
    "daily_usage",
    "import_manifest",
    "metric_definitions",
    "eval_criteria",
    "session_schemas",
    "periodization_models",
    "proactive_trigger_rules",
    "proactive_queue",
    "pending_actions",
    "product_recommendations",
    "macrocycle_plans",
    "goal_trajectory_snapshots",
]


def test_all_tables_have_rls_enabled():
    """Every expected table must have ALTER TABLE ... ENABLE ROW LEVEL SECURITY."""
    for table in EXPECTED_TABLES:
        pattern = rf"ALTER\s+TABLE\s+(?:public\.)?{table}\s+ENABLE\s+ROW\s+LEVEL\s+SECURITY"
        assert re.search(pattern, ALL_SQL, re.IGNORECASE), (
            f"Table '{table}' is missing ENABLE ROW LEVEL SECURITY in migrations"
        )


def test_all_tables_have_service_role_policy():
    """Every expected table must have a service_role bypass policy."""
    for table in EXPECTED_TABLES:
        # Match CREATE POLICY ... ON table ... service_role ...
        pattern = rf"CREATE\s+POLICY\s+.*ON\s+(?:public\.)?{table}\s+.*service_role"
        assert re.search(pattern, ALL_SQL, re.IGNORECASE | re.DOTALL), (
            f"Table '{table}' is missing a service_role policy in migrations"
        )


def test_service_role_policy_pattern_is_for_all():
    """Service role policies should be FOR ALL (not just SELECT)."""
    for table in EXPECTED_TABLES:
        # Find all service_role policies for this table
        pattern = (
            rf"CREATE\s+POLICY\s+.*ON\s+(?:public\.)?{table}\s+"
            rf"FOR\s+ALL\s+.*service_role"
        )
        assert re.search(pattern, ALL_SQL, re.IGNORECASE | re.DOTALL), (
            f"Table '{table}' service_role policy should be FOR ALL"
        )


# ---------------------------------------------------------------------------
# get_user_id dependency tests
# ---------------------------------------------------------------------------


def test_get_user_id_extracts_sub():
    """get_user_id should return the 'sub' claim from the JWT payload."""
    fake_user = {"sub": "user-123-uuid", "email": "test@example.com", "role": "authenticated"}
    result = get_user_id(current_user=fake_user)
    assert result == "user-123-uuid"


def test_get_user_id_raises_on_missing_token():
    """get_current_user raises 401 when no credentials are provided."""
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(credentials=None)
    assert exc_info.value.status_code == 401


def test_get_user_id_raises_on_invalid_token():
    """get_current_user raises 401 for an invalid JWT."""
    from unittest.mock import MagicMock
    creds = MagicMock()
    creds.credentials = "invalid.jwt.token"
    with patch("src.api.auth.get_settings") as mock_settings:
        mock_settings.return_value.supabase_jwt_secret = "test-secret"
        with pytest.raises(HTTPException) as exc_info:
            get_current_user(credentials=creds)
        assert exc_info.value.status_code == 401
