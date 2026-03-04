"""Tests for config store versioning and garbage collection.

Covers:
- _archive_stale_configs: archiving old configs
- _count_active_configs: counting active (non-archived) entries
- _check_duplicates: detecting name-based duplicates
- run_config_gc: full GC flow + cap warning
- define_metric dedup check
- Migration column expectations
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.services.config_gc import (
    _MAX_ACTIVE_CAP,
    _STALE_DAYS,
    _archive_stale_configs,
    _check_duplicates,
    _count_active_configs,
    run_config_gc,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

USER_ID = "test-user-gc"


def _make_mock_supabase(table_data: dict[str, list[dict]] | None = None):
    """Build a mock Supabase client with per-table responses.

    table_data: {"metric_definitions": [...], "eval_criteria": [...], ...}
    """
    table_data = table_data or {}
    client = MagicMock()

    def _table(name):
        mock_table = MagicMock()
        rows = table_data.get(name, [])

        # .select().eq().is_().lt().execute()
        chain = MagicMock()
        result = MagicMock()
        result.data = rows
        result.count = len(rows)
        chain.execute.return_value = result
        chain.eq.return_value = chain
        chain.is_.return_value = chain
        chain.lt.return_value = chain
        chain.gte.return_value = chain
        chain.order.return_value = chain
        chain.limit.return_value = chain

        mock_table.select.return_value = chain

        # .update().eq().execute()
        update_chain = MagicMock()
        update_chain.eq.return_value = update_chain
        update_chain.execute.return_value = MagicMock(data=[])
        mock_table.update.return_value = update_chain

        return mock_table

    client.table.side_effect = _table
    return client


# ---------------------------------------------------------------------------
# _count_active_configs
# ---------------------------------------------------------------------------


class TestCountActiveConfigs:
    @patch("src.services.config_gc.get_supabase")
    def test_counts_across_tables(self, mock_get_sb) -> None:
        mock_get_sb.return_value = _make_mock_supabase({
            "metric_definitions": [{"id": "1"}, {"id": "2"}],
            "eval_criteria": [{"id": "3"}],
        })

        count = _count_active_configs(USER_ID)
        # Only metric_definitions (2) and eval_criteria (1) have data
        # Other tables return 0 via count attribute
        assert count >= 3

    @patch("src.services.config_gc.get_supabase", side_effect=Exception("DB down"))
    def test_db_error_returns_zero(self, mock_get_sb) -> None:
        count = _count_active_configs(USER_ID)
        assert count == 0


# ---------------------------------------------------------------------------
# _check_duplicates
# ---------------------------------------------------------------------------


class TestCheckDuplicates:
    @patch("src.services.config_gc.get_supabase")
    def test_finds_duplicates(self, mock_get_sb) -> None:
        mock_get_sb.return_value = _make_mock_supabase({
            "metric_definitions": [
                {"name": "trimp"},
                {"name": "trimp"},
                {"name": "unique_metric"},
            ],
        })

        dupes = _check_duplicates(USER_ID)
        trimp_dupes = [d for d in dupes if d["name"] == "trimp"]
        assert len(trimp_dupes) == 1
        assert trimp_dupes[0]["count"] == 2

    @patch("src.services.config_gc.get_supabase")
    def test_no_duplicates(self, mock_get_sb) -> None:
        mock_get_sb.return_value = _make_mock_supabase({
            "metric_definitions": [
                {"name": "trimp"},
                {"name": "hrv_score"},
            ],
        })

        dupes = _check_duplicates(USER_ID)
        assert dupes == []

    @patch("src.services.config_gc.get_supabase")
    def test_session_schemas_use_sport_column(self, mock_get_sb) -> None:
        mock_get_sb.return_value = _make_mock_supabase({
            "session_schemas": [
                {"sport": "running"},
                {"sport": "running"},
            ],
        })

        dupes = _check_duplicates(USER_ID)
        running_dupes = [d for d in dupes if d["table"] == "session_schemas"]
        assert len(running_dupes) == 1

    @patch("src.services.config_gc.get_supabase", side_effect=Exception("DB down"))
    def test_db_error_returns_empty(self, mock_get_sb) -> None:
        dupes = _check_duplicates(USER_ID)
        assert dupes == []


# ---------------------------------------------------------------------------
# _archive_stale_configs
# ---------------------------------------------------------------------------


class TestArchiveStaleConfigs:
    @patch("src.services.config_gc.get_supabase")
    def test_archives_old_configs(self, mock_get_sb) -> None:
        old_date = (datetime.now(timezone.utc) - timedelta(days=_STALE_DAYS * 3 + 1)).isoformat()
        mock_get_sb.return_value = _make_mock_supabase({
            "metric_definitions": [
                {"id": "old-1", "updated_at": old_date},
            ],
        })

        archived = _archive_stale_configs(USER_ID)
        assert archived >= 1

    @patch("src.services.config_gc.get_supabase")
    def test_no_stale_configs(self, mock_get_sb) -> None:
        mock_get_sb.return_value = _make_mock_supabase()

        archived = _archive_stale_configs(USER_ID)
        assert archived == 0

    @patch("src.services.config_gc.get_supabase", side_effect=Exception("DB down"))
    def test_db_error_returns_zero(self, mock_get_sb) -> None:
        archived = _archive_stale_configs(USER_ID)
        assert archived == 0


# ---------------------------------------------------------------------------
# run_config_gc
# ---------------------------------------------------------------------------


class TestRunConfigGc:
    @patch("src.services.config_gc._check_duplicates", return_value=[])
    @patch("src.services.config_gc._count_active_configs", return_value=5)
    @patch("src.services.config_gc._archive_stale_configs", return_value=0)
    def test_clean_gc_run(self, mock_archive, mock_count, mock_dupes) -> None:
        result = run_config_gc(USER_ID)
        assert result["archived"] == 0
        assert result["active_count"] == 5
        assert result["duplicates"] == []
        assert "warning" not in result

    @patch("src.services.config_gc._check_duplicates", return_value=[])
    @patch("src.services.config_gc._count_active_configs", return_value=75)
    @patch("src.services.config_gc._archive_stale_configs", return_value=0)
    def test_cap_warning(self, mock_archive, mock_count, mock_dupes) -> None:
        result = run_config_gc(USER_ID)
        assert "warning" in result
        assert str(_MAX_ACTIVE_CAP) in result["warning"]

    @patch("src.services.config_gc._check_duplicates", return_value=[
        {"table": "metric_definitions", "name": "trimp", "count": 2},
    ])
    @patch("src.services.config_gc._count_active_configs", return_value=10)
    @patch("src.services.config_gc._archive_stale_configs", return_value=3)
    def test_gc_with_archives_and_dupes(self, mock_archive, mock_count, mock_dupes) -> None:
        result = run_config_gc(USER_ID)
        assert result["archived"] == 3
        assert len(result["duplicates"]) == 1

    @patch("src.services.config_gc._archive_stale_configs", side_effect=Exception("Boom"))
    def test_gc_failure_returns_empty_result(self, mock_archive) -> None:
        result = run_config_gc(USER_ID)
        assert result["archived"] == 0
        assert result["active_count"] == 0


# ---------------------------------------------------------------------------
# define_metric dedup
# ---------------------------------------------------------------------------


class TestDefineMetricDedup:
    @patch("src.db.agent_config_db.get_metric_definitions")
    @patch("src.db.agent_config_db.upsert_metric_definition")
    @patch("src.agent.tools.config_tools.get_settings")
    def test_detects_formula_duplicate(self, mock_settings, mock_upsert, mock_get) -> None:
        """If the same formula exists under a different name, return duplicate status."""
        settings = MagicMock()
        settings.use_supabase = True
        settings.agenticsports_user_id = USER_ID
        mock_settings.return_value = settings

        mock_get.return_value = [
            {"name": "existing_trimp", "formula": "hr * duration * 0.5"},
        ]

        from src.agent.tools.registry import ToolRegistry
        from src.agent.tools.config_tools import register_config_tools

        registry = ToolRegistry()
        user_model = MagicMock()
        user_model.user_id = USER_ID
        register_config_tools(registry, user_model)

        result = registry.execute("define_metric", {
            "name": "new_trimp",
            "formula": "hr * duration * 0.5",
        })

        assert result["status"] == "duplicate"
        assert result["existing_name"] == "existing_trimp"
        mock_upsert.assert_not_called()

    @patch("src.db.agent_config_db.get_metric_definitions")
    @patch("src.db.agent_config_db.upsert_metric_definition")
    @patch("src.agent.tools.config_tools.get_settings")
    def test_allows_same_name_update(self, mock_settings, mock_upsert, mock_get) -> None:
        """Updating an existing metric (same name) should NOT be flagged as duplicate."""
        settings = MagicMock()
        settings.use_supabase = True
        settings.agenticsports_user_id = USER_ID
        mock_settings.return_value = settings

        mock_get.return_value = [
            {"name": "trimp", "formula": "hr * duration * 0.5"},
        ]
        mock_upsert.return_value = {"id": "123", "name": "trimp"}

        from src.agent.tools.registry import ToolRegistry
        from src.agent.tools.config_tools import register_config_tools

        registry = ToolRegistry()
        user_model = MagicMock()
        user_model.user_id = USER_ID
        register_config_tools(registry, user_model)

        result = registry.execute("define_metric", {
            "name": "trimp",
            "formula": "hr * duration * 0.6",
        })

        assert result["status"] == "success"
        mock_upsert.assert_called_once()

    @patch("src.db.agent_config_db.get_metric_definitions")
    @patch("src.db.agent_config_db.upsert_metric_definition")
    @patch("src.agent.tools.config_tools.get_settings")
    def test_allows_unique_formula(self, mock_settings, mock_upsert, mock_get) -> None:
        settings = MagicMock()
        settings.use_supabase = True
        settings.agenticsports_user_id = USER_ID
        mock_settings.return_value = settings

        mock_get.return_value = [
            {"name": "trimp", "formula": "hr * duration * 0.5"},
        ]
        mock_upsert.return_value = {"id": "456", "name": "hrv_score"}

        from src.agent.tools.registry import ToolRegistry
        from src.agent.tools.config_tools import register_config_tools

        registry = ToolRegistry()
        user_model = MagicMock()
        user_model.user_id = USER_ID
        register_config_tools(registry, user_model)

        result = registry.execute("define_metric", {
            "name": "hrv_score",
            "formula": "rmssd * log(rmssd)",
        })

        assert result["status"] == "success"

    @patch("src.db.agent_config_db.get_metric_definitions", side_effect=Exception("DB"))
    @patch("src.db.agent_config_db.upsert_metric_definition")
    @patch("src.agent.tools.config_tools.get_settings")
    def test_dedup_failure_proceeds(self, mock_settings, mock_upsert, mock_get) -> None:
        """If dedup check fails, metric should still be created."""
        settings = MagicMock()
        settings.use_supabase = True
        settings.agenticsports_user_id = USER_ID
        mock_settings.return_value = settings
        mock_upsert.return_value = {"id": "789", "name": "test_metric"}

        from src.agent.tools.registry import ToolRegistry
        from src.agent.tools.config_tools import register_config_tools

        registry = ToolRegistry()
        user_model = MagicMock()
        user_model.user_id = USER_ID
        register_config_tools(registry, user_model)

        result = registry.execute("define_metric", {
            "name": "test_metric",
            "formula": "a + b",
        })

        assert result["status"] == "success"
