"""Unit tests for src.db.product_recommendations_db.

All Supabase calls are mocked. Tests verify:
- save_recommendations (bulk insert, required fields, optional fields)
- get_recommendations_for_session
- get_recommendations_for_plan
- get_recent_recommendations (ordering, limit)
- mark_clicked
- Edge cases (empty list, missing optional fields)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


USER_ID = "user-prod-rec-test"
SESSION_ID = "session-001"
PLAN_ID = "plan-001"
REC_ID = "rec-001"


def _mock_supabase() -> MagicMock:
    """Return a mock Supabase client with chainable table/select/insert/update."""
    client = MagicMock()
    return client


def _chain(client: MagicMock, data: list[dict] | None = None) -> MagicMock:
    """Set up chainable mock: table().select/insert/update().eq().order().limit().execute()."""
    result = MagicMock()
    result.data = data or []

    # Make every chained call return itself, then execute returns result
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
# save_recommendations
# ---------------------------------------------------------------------------


class TestSaveRecommendations:
    def test_bulk_insert_returns_data(self) -> None:
        from src.db.product_recommendations_db import save_recommendations

        client = _mock_supabase()
        inserted = [
            {"id": "r1", "product_name": "Nike Pegasus", "reason": "Good daily trainer"},
            {"id": "r2", "product_name": "Garmin 265", "reason": "GPS watch"},
        ]
        _chain(client, inserted)

        with patch("src.db.product_recommendations_db.get_supabase", return_value=client):
            result = save_recommendations(USER_ID, [
                {"product_name": "Nike Pegasus", "reason": "Good daily trainer", "category": "shoes"},
                {"product_name": "Garmin 265", "reason": "GPS watch", "category": "watch"},
            ])

        assert len(result) == 2
        assert result[0]["product_name"] == "Nike Pegasus"
        client.table.assert_called_with("product_recommendations")

    def test_empty_list_returns_empty(self) -> None:
        from src.db.product_recommendations_db import save_recommendations

        result = save_recommendations(USER_ID, [])
        assert result == []

    def test_minimal_fields_accepted(self) -> None:
        from src.db.product_recommendations_db import save_recommendations

        client = _mock_supabase()
        _chain(client, [{"id": "r1", "product_name": "Foam Roller", "reason": "Recovery"}])

        with patch("src.db.product_recommendations_db.get_supabase", return_value=client):
            result = save_recommendations(USER_ID, [
                {"product_name": "Foam Roller", "reason": "Recovery"},
            ])

        assert len(result) == 1
        # Verify the inserted row has defaults
        insert_call = client.table.return_value.insert
        rows = insert_call.call_args[0][0]
        assert rows[0]["currency"] == "EUR"
        assert rows[0]["source"] == "llm"

    def test_optional_fields_passed_through(self) -> None:
        from src.db.product_recommendations_db import save_recommendations

        client = _mock_supabase()
        _chain(client, [{"id": "r1"}])

        rec = {
            "product_name": "Theragun",
            "reason": "Recovery tool",
            "image_url": "https://example.com/img.jpg",
            "price": 299.99,
            "currency": "USD",
            "product_url": "https://example.com/product",
            "affiliate_url": "https://example.com/aff",
            "affiliate_provider": "amazon_associates",
            "category": "recovery",
            "sport": "running",
            "search_query": "Theragun Mini",
            "source": "amazon_api",
            "session_id": SESSION_ID,
            "plan_id": PLAN_ID,
        }
        with patch("src.db.product_recommendations_db.get_supabase", return_value=client):
            save_recommendations(USER_ID, [rec])

        rows = client.table.return_value.insert.call_args[0][0]
        row = rows[0]
        assert row["image_url"] == "https://example.com/img.jpg"
        assert row["price"] == 299.99
        assert row["affiliate_provider"] == "amazon_associates"
        assert row["session_id"] == SESSION_ID
        assert row["plan_id"] == PLAN_ID

    def test_user_id_set_on_each_row(self) -> None:
        from src.db.product_recommendations_db import save_recommendations

        client = _mock_supabase()
        _chain(client, [{"id": "r1"}, {"id": "r2"}])

        with patch("src.db.product_recommendations_db.get_supabase", return_value=client):
            save_recommendations(USER_ID, [
                {"product_name": "A", "reason": "R1"},
                {"product_name": "B", "reason": "R2"},
            ])

        rows = client.table.return_value.insert.call_args[0][0]
        for row in rows:
            assert row["user_id"] == USER_ID


# ---------------------------------------------------------------------------
# get_recommendations_for_session
# ---------------------------------------------------------------------------


class TestGetRecommendationsForSession:
    def test_returns_filtered_data(self) -> None:
        from src.db.product_recommendations_db import get_recommendations_for_session

        client = _mock_supabase()
        data = [{"id": "r1", "session_id": SESSION_ID, "product_name": "Shoes"}]
        _chain(client, data)

        with patch("src.db.product_recommendations_db.get_supabase", return_value=client):
            result = get_recommendations_for_session(USER_ID, SESSION_ID)

        assert len(result) == 1
        assert result[0]["product_name"] == "Shoes"

    def test_empty_session_returns_empty(self) -> None:
        from src.db.product_recommendations_db import get_recommendations_for_session

        client = _mock_supabase()
        _chain(client, [])

        with patch("src.db.product_recommendations_db.get_supabase", return_value=client):
            result = get_recommendations_for_session(USER_ID, "nonexistent")

        assert result == []


# ---------------------------------------------------------------------------
# get_recommendations_for_plan
# ---------------------------------------------------------------------------


class TestGetRecommendationsForPlan:
    def test_returns_plan_recs(self) -> None:
        from src.db.product_recommendations_db import get_recommendations_for_plan

        client = _mock_supabase()
        data = [{"id": "r1", "plan_id": PLAN_ID, "product_name": "Watch"}]
        _chain(client, data)

        with patch("src.db.product_recommendations_db.get_supabase", return_value=client):
            result = get_recommendations_for_plan(USER_ID, PLAN_ID)

        assert len(result) == 1


# ---------------------------------------------------------------------------
# get_recent_recommendations
# ---------------------------------------------------------------------------


class TestGetRecentRecommendations:
    def test_returns_recent_data(self) -> None:
        from src.db.product_recommendations_db import get_recent_recommendations

        client = _mock_supabase()
        data = [
            {"id": "r3", "product_name": "C"},
            {"id": "r2", "product_name": "B"},
        ]
        _chain(client, data)

        with patch("src.db.product_recommendations_db.get_supabase", return_value=client):
            result = get_recent_recommendations(USER_ID, limit=5)

        assert len(result) == 2

    def test_default_limit(self) -> None:
        from src.db.product_recommendations_db import get_recent_recommendations

        client = _mock_supabase()
        chain = _chain(client, [])

        with patch("src.db.product_recommendations_db.get_supabase", return_value=client):
            get_recent_recommendations(USER_ID)

        chain.limit.assert_called_with(20)


# ---------------------------------------------------------------------------
# mark_clicked
# ---------------------------------------------------------------------------


class TestMarkClicked:
    def test_updates_clicked_flag(self) -> None:
        from src.db.product_recommendations_db import mark_clicked

        client = _mock_supabase()
        _chain(client, [])

        with patch("src.db.product_recommendations_db.get_supabase", return_value=client):
            mark_clicked(REC_ID)

        update_call = client.table.return_value.update
        update_call.assert_called_once()
        args = update_call.call_args[0][0]
        assert args["clicked"] is True
        assert "updated_at" in args
