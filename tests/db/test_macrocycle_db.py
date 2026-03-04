"""Unit tests for src.db.macrocycle_db.

All Supabase calls are mocked. Tests verify:
- store_macrocycle (insert, deactivation of previous active)
- get_active_macrocycle
- get_macrocycle (by name)
- list_macrocycles
- update_macrocycle (partial, allowed fields only)
- deactivate_macrocycle
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


USER_ID = "user-macrocycle-test"
MACRO_NAME = "Marathon Prep 2026"


def _mock_supabase() -> MagicMock:
    """Return a mock Supabase client with chainable table/select/insert/update."""
    return MagicMock()


def _chain(client: MagicMock, data: list[dict] | None = None) -> MagicMock:
    """Set up chainable mock: table().select/insert/update().eq().order().limit().execute()."""
    result = MagicMock()
    result.data = data or []

    chain_mock = MagicMock()
    chain_mock.execute.return_value = result
    chain_mock.eq.return_value = chain_mock
    chain_mock.order.return_value = chain_mock
    chain_mock.limit.return_value = chain_mock

    table_mock = MagicMock()
    table_mock.insert.return_value = chain_mock
    table_mock.select.return_value = chain_mock
    table_mock.update.return_value = chain_mock

    client.table.return_value = table_mock
    return chain_mock


# ---------------------------------------------------------------------------
# store_macrocycle
# ---------------------------------------------------------------------------


class TestStoreMacrocycle:
    def test_inserts_and_returns_row(self) -> None:
        from src.db.macrocycle_db import store_macrocycle

        client = _mock_supabase()
        inserted = [{"id": "m1", "name": MACRO_NAME, "status": "active"}]
        _chain(client, inserted)

        with patch("src.db.macrocycle_db.get_supabase", return_value=client):
            result = store_macrocycle(
                USER_ID, MACRO_NAME, total_weeks=12,
                start_date="2026-04-01", weeks=[{"week_number": 1}],
            )

        assert result["name"] == MACRO_NAME
        assert result["status"] == "active"

    def test_deactivates_previous_active(self) -> None:
        from src.db.macrocycle_db import store_macrocycle

        client = _mock_supabase()
        _chain(client, [{"id": "m2", "name": "New Plan"}])

        with patch("src.db.macrocycle_db.get_supabase", return_value=client):
            store_macrocycle(
                USER_ID, "New Plan", total_weeks=8,
                start_date="2026-05-01", weeks=[],
            )

        # Should call update (deactivate) then insert
        update_mock = client.table.return_value.update
        assert update_mock.called

    def test_optional_fields(self) -> None:
        from src.db.macrocycle_db import store_macrocycle

        client = _mock_supabase()
        _chain(client, [{"id": "m3"}])

        with patch("src.db.macrocycle_db.get_supabase", return_value=client):
            store_macrocycle(
                USER_ID, "With Model", total_weeks=16,
                start_date="2026-06-01", weeks=[],
                periodization_model_name="linear",
                evaluation_score=85,
            )

        insert_args = client.table.return_value.insert.call_args[0][0]
        assert insert_args["periodization_model_name"] == "linear"
        assert insert_args["evaluation_score"] == 85


# ---------------------------------------------------------------------------
# get_active_macrocycle
# ---------------------------------------------------------------------------


class TestGetActiveMacrocycle:
    def test_returns_active(self) -> None:
        from src.db.macrocycle_db import get_active_macrocycle

        client = _mock_supabase()
        _chain(client, [{"id": "m1", "status": "active", "name": MACRO_NAME}])

        with patch("src.db.macrocycle_db.get_supabase", return_value=client):
            result = get_active_macrocycle(USER_ID)

        assert result is not None
        assert result["name"] == MACRO_NAME

    def test_returns_none_when_no_active(self) -> None:
        from src.db.macrocycle_db import get_active_macrocycle

        client = _mock_supabase()
        _chain(client, [])

        with patch("src.db.macrocycle_db.get_supabase", return_value=client):
            result = get_active_macrocycle(USER_ID)

        assert result is None


# ---------------------------------------------------------------------------
# get_macrocycle (by name)
# ---------------------------------------------------------------------------


class TestGetMacrocycle:
    def test_returns_by_name(self) -> None:
        from src.db.macrocycle_db import get_macrocycle

        client = _mock_supabase()
        _chain(client, [{"id": "m1", "name": MACRO_NAME}])

        with patch("src.db.macrocycle_db.get_supabase", return_value=client):
            result = get_macrocycle(USER_ID, MACRO_NAME)

        assert result is not None
        assert result["name"] == MACRO_NAME


# ---------------------------------------------------------------------------
# list_macrocycles
# ---------------------------------------------------------------------------


class TestListMacrocycles:
    def test_returns_list(self) -> None:
        from src.db.macrocycle_db import list_macrocycles

        client = _mock_supabase()
        data = [{"id": "m1"}, {"id": "m2"}]
        _chain(client, data)

        with patch("src.db.macrocycle_db.get_supabase", return_value=client):
            result = list_macrocycles(USER_ID)

        assert len(result) == 2

    def test_respects_limit(self) -> None:
        from src.db.macrocycle_db import list_macrocycles

        client = _mock_supabase()
        chain = _chain(client, [])

        with patch("src.db.macrocycle_db.get_supabase", return_value=client):
            list_macrocycles(USER_ID, limit=5)

        chain.limit.assert_called_with(5)


# ---------------------------------------------------------------------------
# update_macrocycle
# ---------------------------------------------------------------------------


class TestUpdateMacrocycle:
    def test_updates_allowed_fields(self) -> None:
        from src.db.macrocycle_db import update_macrocycle

        client = _mock_supabase()
        _chain(client, [{"id": "m1", "evaluation_score": 90}])

        with patch("src.db.macrocycle_db.get_supabase", return_value=client):
            result = update_macrocycle(USER_ID, MACRO_NAME, {"evaluation_score": 90})

        assert result is not None
        update_call = client.table.return_value.update
        args = update_call.call_args[0][0]
        assert args["evaluation_score"] == 90
        assert "updated_at" in args

    def test_filters_disallowed_fields(self) -> None:
        from src.db.macrocycle_db import update_macrocycle

        client = _mock_supabase()
        _chain(client, [])

        with patch("src.db.macrocycle_db.get_supabase", return_value=client):
            result = update_macrocycle(USER_ID, MACRO_NAME, {"user_id": "hacker", "name": "bad"})

        assert result is None


# ---------------------------------------------------------------------------
# deactivate_macrocycle
# ---------------------------------------------------------------------------


class TestDeactivateMacrocycle:
    def test_sets_status_archived(self) -> None:
        from src.db.macrocycle_db import deactivate_macrocycle

        client = _mock_supabase()
        _chain(client, [{"id": "m1", "status": "archived"}])

        with patch("src.db.macrocycle_db.get_supabase", return_value=client):
            result = deactivate_macrocycle(USER_ID, MACRO_NAME)

        assert result is not None
        update_call = client.table.return_value.update
        args = update_call.call_args[0][0]
        assert args["status"] == "archived"
