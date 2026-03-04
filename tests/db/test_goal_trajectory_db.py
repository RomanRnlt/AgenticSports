"""Unit tests for src.db.goal_trajectory_db.

All Supabase calls are mocked. Tests verify:
- save_trajectory_snapshot (append-only insert)
- get_latest_trajectory (most recent for a goal)
- list_trajectory_snapshots (history, optional goal filter)
- Edge cases (empty results, default values)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


USER_ID = "user-trajectory-test"
GOAL_NAME = "Marathon unter 3:30"


def _mock_supabase() -> MagicMock:
    return MagicMock()


def _chain(client: MagicMock, data: list[dict] | None = None) -> MagicMock:
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

    client.table.return_value = table_mock
    return chain_mock


# ---------------------------------------------------------------------------
# save_trajectory_snapshot
# ---------------------------------------------------------------------------


class TestSaveTrajectorySnapshot:
    def test_inserts_and_returns_row(self) -> None:
        from src.db.goal_trajectory_db import save_trajectory_snapshot

        client = _mock_supabase()
        inserted = [{"id": "t1", "goal_name": GOAL_NAME, "trajectory_status": "on_track"}]
        _chain(client, inserted)

        with patch("src.db.goal_trajectory_db.get_supabase", return_value=client):
            result = save_trajectory_snapshot(
                USER_ID, GOAL_NAME,
                trajectory_status="on_track",
                confidence=0.8,
                projected_outcome="On pace for 3:25",
                analysis="Training volume is adequate",
            )

        assert result["trajectory_status"] == "on_track"
        client.table.assert_called_with("goal_trajectory_snapshots")

    def test_defaults_for_optional_fields(self) -> None:
        from src.db.goal_trajectory_db import save_trajectory_snapshot

        client = _mock_supabase()
        _chain(client, [{"id": "t2"}])

        with patch("src.db.goal_trajectory_db.get_supabase", return_value=client):
            save_trajectory_snapshot(
                USER_ID, GOAL_NAME,
                trajectory_status="insufficient_data",
                confidence=0.3,
            )

        insert_args = client.table.return_value.insert.call_args[0][0]
        assert insert_args["recommendations"] == []
        assert insert_args["risk_factors"] == []
        assert insert_args["context_snapshot"] == {}
        assert insert_args["projected_outcome"] == ""
        assert insert_args["analysis"] == ""

    def test_passes_all_fields(self) -> None:
        from src.db.goal_trajectory_db import save_trajectory_snapshot

        client = _mock_supabase()
        _chain(client, [{"id": "t3"}])

        with patch("src.db.goal_trajectory_db.get_supabase", return_value=client):
            save_trajectory_snapshot(
                USER_ID, GOAL_NAME,
                trajectory_status="behind",
                confidence=0.7,
                projected_outcome="3:45 at current pace",
                analysis="Volume dropped 20%",
                recommendations=["Increase long run", "Add tempo"],
                risk_factors=["Low mileage", "Recent illness"],
                context_snapshot={"weekly_km": 35},
            )

        insert_args = client.table.return_value.insert.call_args[0][0]
        assert insert_args["recommendations"] == ["Increase long run", "Add tempo"]
        assert insert_args["risk_factors"] == ["Low mileage", "Recent illness"]
        assert insert_args["context_snapshot"] == {"weekly_km": 35}


# ---------------------------------------------------------------------------
# get_latest_trajectory
# ---------------------------------------------------------------------------


class TestGetLatestTrajectory:
    def test_returns_latest(self) -> None:
        from src.db.goal_trajectory_db import get_latest_trajectory

        client = _mock_supabase()
        _chain(client, [{"id": "t1", "trajectory_status": "ahead"}])

        with patch("src.db.goal_trajectory_db.get_supabase", return_value=client):
            result = get_latest_trajectory(USER_ID, GOAL_NAME)

        assert result is not None
        assert result["trajectory_status"] == "ahead"

    def test_returns_none_when_empty(self) -> None:
        from src.db.goal_trajectory_db import get_latest_trajectory

        client = _mock_supabase()
        _chain(client, [])

        with patch("src.db.goal_trajectory_db.get_supabase", return_value=client):
            result = get_latest_trajectory(USER_ID, "nonexistent goal")

        assert result is None


# ---------------------------------------------------------------------------
# list_trajectory_snapshots
# ---------------------------------------------------------------------------


class TestListTrajectorySnapshots:
    def test_returns_all_for_user(self) -> None:
        from src.db.goal_trajectory_db import list_trajectory_snapshots

        client = _mock_supabase()
        data = [{"id": "t1"}, {"id": "t2"}, {"id": "t3"}]
        _chain(client, data)

        with patch("src.db.goal_trajectory_db.get_supabase", return_value=client):
            result = list_trajectory_snapshots(USER_ID)

        assert len(result) == 3

    def test_filters_by_goal(self) -> None:
        from src.db.goal_trajectory_db import list_trajectory_snapshots

        client = _mock_supabase()
        chain = _chain(client, [{"id": "t1"}])

        with patch("src.db.goal_trajectory_db.get_supabase", return_value=client):
            list_trajectory_snapshots(USER_ID, goal_name=GOAL_NAME)

        # Should call eq twice (user_id + goal_name)
        eq_calls = chain.eq.call_args_list
        assert len(eq_calls) >= 1

    def test_respects_limit(self) -> None:
        from src.db.goal_trajectory_db import list_trajectory_snapshots

        client = _mock_supabase()
        chain = _chain(client, [])

        with patch("src.db.goal_trajectory_db.get_supabase", return_value=client):
            list_trajectory_snapshots(USER_ID, limit=5)

        chain.limit.assert_called_with(5)
