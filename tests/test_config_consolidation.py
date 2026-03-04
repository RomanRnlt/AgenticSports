"""Tests for LLM-based config consolidation (Visionplan 8.12 D).

Covers:
- _consolidate_configs: LLM consolidation pass triggered above cap
- _fetch_active_metrics: metric fetching for consolidation
- _build_metrics_text: prompt formatting
- _parse_merge_groups: LLM response parsing
- _archive_config_by_name: per-config archival
- Integration with run_config_gc: consolidation triggered/not triggered
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.services.config_gc import (
    _MAX_ACTIVE_CAP,
    _build_metrics_text,
    _consolidate_configs,
    _fetch_active_metrics,
    _parse_merge_groups,
    run_config_gc,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

USER_ID = "test-user-consolidation"


def _mock_llm_response(content: str) -> MagicMock:
    """Build a mock LiteLLM response with the given content."""
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    response.choices = [choice]
    return response


def _make_mock_supabase_for_consolidation(
    metric_rows: list[dict] | None = None,
    archive_success: bool = True,
):
    """Build a mock Supabase client for consolidation tests.

    Supports the chained query patterns used by:
    - _fetch_active_metrics: select().eq().is_().execute()
    - _archive_config_by_name: update().eq().eq().is_().execute()
    - _count_active_configs: select(count=exact).eq().is_().execute()
    """
    metric_rows = metric_rows or []
    client = MagicMock()

    def _table(name):
        mock_table = MagicMock()

        # select chain (used by _fetch_active_metrics and _count_active_configs)
        select_chain = MagicMock()
        select_result = MagicMock()
        select_result.data = metric_rows if name == "metric_definitions" else []
        select_result.count = len(metric_rows) if name == "metric_definitions" else 0
        select_chain.execute.return_value = select_result
        select_chain.eq.return_value = select_chain
        select_chain.is_.return_value = select_chain
        select_chain.lt.return_value = select_chain
        select_chain.order.return_value = select_chain
        mock_table.select.return_value = select_chain

        # update chain (used by _archive_config_by_name)
        update_chain = MagicMock()
        update_result = MagicMock()
        update_result.data = [{"id": "archived"}] if archive_success else []
        update_chain.execute.return_value = update_result
        update_chain.eq.return_value = update_chain
        update_chain.is_.return_value = update_chain
        mock_table.update.return_value = update_chain

        return mock_table

    client.table.side_effect = _table
    return client


# ---------------------------------------------------------------------------
# _build_metrics_text
# ---------------------------------------------------------------------------


class TestBuildMetricsText:
    def test_formats_metrics_as_numbered_list(self) -> None:
        metrics = [
            {"name": "trimp", "formula": "hr * duration", "description": "Training impulse"},
            {"name": "hrv", "formula": "rmssd", "description": "Heart rate variability"},
        ]
        text = _build_metrics_text(metrics)
        assert "1. trimp:" in text
        assert "2. hrv:" in text
        assert "formula='hr * duration'" in text
        assert "description='Training impulse'" in text

    def test_handles_empty_list(self) -> None:
        assert _build_metrics_text([]) == ""

    def test_handles_missing_fields(self) -> None:
        metrics = [{"name": "x"}]
        text = _build_metrics_text(metrics)
        assert "1. x:" in text
        assert "formula=''" in text


# ---------------------------------------------------------------------------
# _parse_merge_groups
# ---------------------------------------------------------------------------


class TestParseMergeGroups:
    def test_parses_valid_response(self) -> None:
        content = json.dumps({
            "merge_groups": [
                {"keep": "trimp", "archive": ["training_impulse", "trimp_v2"], "reason": "Same concept"},
            ],
        })
        groups = _parse_merge_groups(content)
        assert len(groups) == 1
        assert groups[0]["keep"] == "trimp"
        assert groups[0]["archive"] == ["training_impulse", "trimp_v2"]

    def test_returns_empty_for_invalid_json(self) -> None:
        assert _parse_merge_groups("not json at all") == []

    def test_returns_empty_for_missing_merge_groups_key(self) -> None:
        assert _parse_merge_groups('{"other": "data"}') == []

    def test_returns_empty_for_none(self) -> None:
        assert _parse_merge_groups(None) == []

    def test_filters_invalid_groups(self) -> None:
        content = json.dumps({
            "merge_groups": [
                {"keep": "trimp", "archive": ["old_trimp"], "reason": "ok"},
                {"keep": 123, "archive": ["x"]},  # invalid: keep is not str
                {"archive": ["x"]},  # missing keep
                {"keep": "y", "archive": "not_a_list"},  # archive is not list
            ],
        })
        groups = _parse_merge_groups(content)
        assert len(groups) == 1
        assert groups[0]["keep"] == "trimp"

    def test_empty_merge_groups(self) -> None:
        content = json.dumps({"merge_groups": []})
        groups = _parse_merge_groups(content)
        assert groups == []


# ---------------------------------------------------------------------------
# _consolidate_configs
# ---------------------------------------------------------------------------


class TestConsolidateConfigs:
    @patch("src.agent.llm.chat_completion")
    @patch("src.services.config_gc.get_supabase")
    def test_consolidation_merges_configs(self, mock_get_sb, mock_llm) -> None:
        """LLM identifies merge groups and correct configs are archived."""
        metrics = [
            {"name": "trimp", "formula": "hr * duration * 0.5", "description": "TRIMP"},
            {"name": "training_impulse", "formula": "hr * dur * 0.5", "description": "Training impulse"},
            {"name": "hrv_score", "formula": "rmssd * log(rmssd)", "description": "HRV"},
        ]
        mock_get_sb.return_value = _make_mock_supabase_for_consolidation(metrics)

        llm_response = json.dumps({
            "merge_groups": [
                {
                    "keep": "trimp",
                    "archive": ["training_impulse"],
                    "reason": "Same metric, different name",
                },
            ],
        })
        mock_llm.return_value = _mock_llm_response(llm_response)

        result = _consolidate_configs(USER_ID)
        assert result == 1
        mock_llm.assert_called_once()

    @patch("src.agent.llm.chat_completion")
    @patch("src.services.config_gc.get_supabase")
    def test_consolidation_llm_failure_graceful(self, mock_get_sb, mock_llm) -> None:
        """LLM call raises an exception -- consolidation returns 0, no crash."""
        metrics = [
            {"name": "a", "formula": "x", "description": ""},
            {"name": "b", "formula": "y", "description": ""},
        ]
        mock_get_sb.return_value = _make_mock_supabase_for_consolidation(metrics)
        mock_llm.side_effect = Exception("LLM unavailable")

        result = _consolidate_configs(USER_ID)
        assert result == 0

    @patch("src.agent.llm.chat_completion")
    @patch("src.services.config_gc.get_supabase")
    def test_consolidation_empty_response(self, mock_get_sb, mock_llm) -> None:
        """LLM returns empty merge_groups -- 0 consolidated."""
        metrics = [
            {"name": "a", "formula": "x", "description": ""},
            {"name": "b", "formula": "y", "description": ""},
        ]
        mock_get_sb.return_value = _make_mock_supabase_for_consolidation(metrics)
        mock_llm.return_value = _mock_llm_response(json.dumps({"merge_groups": []}))

        result = _consolidate_configs(USER_ID)
        assert result == 0

    @patch("src.agent.llm.chat_completion")
    @patch("src.services.config_gc.get_supabase")
    def test_consolidation_invalid_json(self, mock_get_sb, mock_llm) -> None:
        """LLM returns non-JSON -- 0 consolidated, no crash."""
        metrics = [
            {"name": "a", "formula": "x", "description": ""},
            {"name": "b", "formula": "y", "description": ""},
        ]
        mock_get_sb.return_value = _make_mock_supabase_for_consolidation(metrics)
        mock_llm.return_value = _mock_llm_response("I cannot parse this as JSON")

        result = _consolidate_configs(USER_ID)
        assert result == 0

    @patch("src.agent.llm.chat_completion")
    @patch("src.services.config_gc.get_supabase")
    def test_consolidation_skips_unknown_metrics(self, mock_get_sb, mock_llm) -> None:
        """LLM suggests archiving a metric that doesn't exist -- safely skipped."""
        metrics = [
            {"name": "trimp", "formula": "hr * dur", "description": ""},
            {"name": "hrv", "formula": "rmssd", "description": ""},
        ]
        mock_get_sb.return_value = _make_mock_supabase_for_consolidation(metrics)

        llm_response = json.dumps({
            "merge_groups": [
                {"keep": "trimp", "archive": ["nonexistent_metric"], "reason": "test"},
            ],
        })
        mock_llm.return_value = _mock_llm_response(llm_response)

        result = _consolidate_configs(USER_ID)
        assert result == 0

    @patch("src.agent.llm.chat_completion")
    @patch("src.services.config_gc.get_supabase")
    def test_consolidation_never_archives_keep_metric(self, mock_get_sb, mock_llm) -> None:
        """If LLM accidentally lists keep in archive, it's skipped."""
        metrics = [
            {"name": "trimp", "formula": "hr * dur", "description": ""},
            {"name": "hrv", "formula": "rmssd", "description": ""},
        ]
        mock_get_sb.return_value = _make_mock_supabase_for_consolidation(metrics)

        llm_response = json.dumps({
            "merge_groups": [
                {"keep": "trimp", "archive": ["trimp"], "reason": "bug in LLM"},
            ],
        })
        mock_llm.return_value = _mock_llm_response(llm_response)

        result = _consolidate_configs(USER_ID)
        assert result == 0

    @patch("src.services.config_gc.get_supabase")
    def test_consolidation_with_single_metric_skips(self, mock_get_sb) -> None:
        """Only 1 metric -- nothing to consolidate, LLM not called."""
        metrics = [{"name": "only_one", "formula": "x", "description": ""}]
        mock_get_sb.return_value = _make_mock_supabase_for_consolidation(metrics)

        result = _consolidate_configs(USER_ID)
        assert result == 0

    @patch("src.agent.llm.chat_completion")
    @patch("src.services.config_gc.get_supabase")
    def test_consolidation_archives_correct_configs(self, mock_get_sb, mock_llm) -> None:
        """Verify that only the 'archive' names get valid_until set, not the 'keep' one."""
        metrics = [
            {"name": "trimp", "formula": "hr * duration * 0.5", "description": "TRIMP"},
            {"name": "training_impulse", "formula": "hr * dur * 0.5", "description": ""},
            {"name": "trimp_v2", "formula": "hr * duration * 0.5", "description": ""},
            {"name": "hrv_score", "formula": "rmssd", "description": "HRV"},
        ]

        # Track which names get archived via update calls
        archived_names = []
        client = MagicMock()

        def _table(name):
            mock_table = MagicMock()

            # select chain
            select_chain = MagicMock()
            select_result = MagicMock()
            select_result.data = metrics if name == "metric_definitions" else []
            select_result.count = len(metrics) if name == "metric_definitions" else 0
            select_chain.execute.return_value = select_result
            select_chain.eq.return_value = select_chain
            select_chain.is_.return_value = select_chain
            select_chain.lt.return_value = select_chain
            select_chain.order.return_value = select_chain
            mock_table.select.return_value = select_chain

            # update chain -- capture which name is being archived
            def _make_update_chain(update_data):
                uc = MagicMock()
                captured = {"user_id": None, "name": None}

                def _eq(field, value):
                    if field == "name":
                        captured["name"] = value
                    return uc

                uc.eq.side_effect = _eq
                uc.is_.return_value = uc

                def _execute():
                    if captured["name"]:
                        archived_names.append(captured["name"])
                    result = MagicMock()
                    result.data = [{"id": "ok"}]
                    return result

                uc.execute.side_effect = _execute
                return uc

            mock_table.update.side_effect = _make_update_chain
            return mock_table

        client.table.side_effect = _table
        mock_get_sb.return_value = client

        llm_response = json.dumps({
            "merge_groups": [
                {
                    "keep": "trimp",
                    "archive": ["training_impulse", "trimp_v2"],
                    "reason": "Same TRIMP metric",
                },
            ],
        })
        mock_llm.return_value = _mock_llm_response(llm_response)

        result = _consolidate_configs(USER_ID)
        assert result == 2
        assert "training_impulse" in archived_names
        assert "trimp_v2" in archived_names
        assert "trimp" not in archived_names
        assert "hrv_score" not in archived_names


# ---------------------------------------------------------------------------
# run_config_gc integration with consolidation
# ---------------------------------------------------------------------------


class TestRunConfigGcConsolidation:
    @patch("src.services.config_gc._consolidate_configs")
    @patch("src.services.config_gc._check_duplicates", return_value=[])
    @patch("src.services.config_gc._count_active_configs", return_value=75)
    @patch("src.services.config_gc._archive_stale_configs", return_value=0)
    def test_consolidation_triggered_above_cap(
        self, mock_archive, mock_count, mock_dupes, mock_consolidate,
    ) -> None:
        """When active_count > 60, consolidation is triggered."""
        mock_consolidate.return_value = 3
        result = run_config_gc(USER_ID)

        mock_consolidate.assert_called_once_with(USER_ID)
        assert result["consolidated"] == 3
        assert "warning" in result
        assert "Consolidated 3" in result["warning"]

    @patch("src.services.config_gc._consolidate_configs")
    @patch("src.services.config_gc._check_duplicates", return_value=[])
    @patch("src.services.config_gc._count_active_configs", return_value=50)
    @patch("src.services.config_gc._archive_stale_configs", return_value=0)
    def test_consolidation_not_triggered_below_cap(
        self, mock_archive, mock_count, mock_dupes, mock_consolidate,
    ) -> None:
        """When active_count <= 60, consolidation is NOT triggered."""
        result = run_config_gc(USER_ID)

        mock_consolidate.assert_not_called()
        assert "consolidated" not in result
        assert "warning" not in result

    @patch("src.services.config_gc._consolidate_configs", side_effect=Exception("LLM down"))
    @patch("src.services.config_gc._check_duplicates", return_value=[])
    @patch("src.services.config_gc._count_active_configs", return_value=75)
    @patch("src.services.config_gc._archive_stale_configs", return_value=0)
    def test_consolidation_failure_doesnt_break_gc(
        self, mock_archive, mock_count, mock_dupes, mock_consolidate,
    ) -> None:
        """Even if consolidation throws, GC returns a safe fallback result."""
        result = run_config_gc(USER_ID)
        # The outer try/except catches the exception
        assert result["archived"] == 0
        assert result["active_count"] == 0

    @patch("src.services.config_gc._consolidate_configs", return_value=0)
    @patch("src.services.config_gc._check_duplicates", return_value=[])
    @patch("src.services.config_gc._count_active_configs", return_value=75)
    @patch("src.services.config_gc._archive_stale_configs", return_value=0)
    def test_consolidation_zero_still_reports_warning(
        self, mock_archive, mock_count, mock_dupes, mock_consolidate,
    ) -> None:
        """Even if consolidation finds nothing, the warning is still reported."""
        result = run_config_gc(USER_ID)
        assert result["consolidated"] == 0
        assert "warning" in result
        assert "Consolidated 0" in result["warning"]
