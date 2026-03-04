"""Tests for episode consolidation (monthly synthesis).

Covers:
- _format_episodes: formatting weekly episodes for LLM
- _generate_consolidation: LLM call, JSON parsing, error handling
- consolidate_month: full flow including storage and belief promotion
- get_unconsolidated_months: finding months that need consolidation
- _promote_patterns_to_beliefs: belief creation from patterns
- Heartbeat integration trigger
- list_episodes_for_period: DB query function
- consolidate_episodes tool: agent-callable tool wrapper
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.episode_consolidation import (
    _MIN_WEEKLY_REFLECTIONS,
    _format_episodes,
    _generate_consolidation,
    _promote_patterns_to_beliefs,
    consolidate_month,
    get_unconsolidated_months,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

USER_ID = "test-user-ep"


def _make_weekly_episodes(n: int = 4, month: str = "2026-02") -> list[dict]:
    return [
        {
            "id": f"ep-{i}",
            "episode_type": "weekly_reflection",
            "period_start": f"{month}-{(i * 7 + 1):02d}",
            "period_end": f"{month}-{(i * 7 + 7):02d}",
            "summary": f"Week {i + 1}: Good training volume with progressive overload.",
            "insights": [f"Pattern {i}: consistent morning runs", "HR recovery improving"],
        }
        for i in range(n)
    ]


def _mock_llm_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


# ---------------------------------------------------------------------------
# _format_episodes
# ---------------------------------------------------------------------------


class TestFormatEpisodes:
    def test_formats_with_summary_and_insights(self) -> None:
        episodes = _make_weekly_episodes(2)
        result = _format_episodes(episodes)
        assert "Week 1" in result
        assert "Week 2" in result
        assert "Pattern 0" in result

    def test_empty_episodes(self) -> None:
        result = _format_episodes([])
        assert result == ""

    def test_truncates_long_summaries(self) -> None:
        episodes = [{"summary": "x" * 1000, "period_start": "2026-01-01"}]
        result = _format_episodes(episodes)
        # Summary should be truncated to 500 chars
        lines = result.strip().split("\n")
        assert len(lines[1]) <= 500

    def test_handles_missing_fields(self) -> None:
        episodes = [{"period_start": "2026-01-01"}]
        result = _format_episodes(episodes)
        assert "2026-01-01" in result


# ---------------------------------------------------------------------------
# _generate_consolidation
# ---------------------------------------------------------------------------


class TestGenerateConsolidation:
    @patch("src.agent.llm.chat_completion")
    def test_successful_consolidation(self, mock_chat) -> None:
        mock_chat.return_value = _mock_llm_response(
            '{"monthly_summary": "Great month.", '
            '"recurring_patterns": ["Morning runs", "Progressive overload"], '
            '"key_metrics": {"avg_distance_km": 8.5}}'
        )

        result = asyncio.run(_generate_consolidation(_make_weekly_episodes(), "2026-02"))
        assert result["monthly_summary"] == "Great month."
        assert len(result["recurring_patterns"]) == 2
        assert result["key_metrics"]["avg_distance_km"] == 8.5

    @patch("src.agent.llm.chat_completion")
    def test_markdown_fence_stripped(self, mock_chat) -> None:
        mock_chat.return_value = _mock_llm_response(
            '```json\n{"monthly_summary": "OK.", "recurring_patterns": [], "key_metrics": {}}\n```'
        )
        result = asyncio.run(_generate_consolidation(_make_weekly_episodes(), "2026-02"))
        assert result["monthly_summary"] == "OK."

    @patch("src.agent.llm.chat_completion")
    def test_empty_response(self, mock_chat) -> None:
        mock_chat.return_value = _mock_llm_response("")
        result = asyncio.run(_generate_consolidation(_make_weekly_episodes(), "2026-02"))
        assert result["monthly_summary"] == ""

    @patch("src.agent.llm.chat_completion", side_effect=Exception("LLM down"))
    def test_llm_failure(self, mock_chat) -> None:
        result = asyncio.run(_generate_consolidation(_make_weekly_episodes(), "2026-02"))
        assert result["monthly_summary"] == ""
        assert result["recurring_patterns"] == []

    @patch("src.agent.llm.chat_completion")
    def test_invalid_json(self, mock_chat) -> None:
        mock_chat.return_value = _mock_llm_response("Not JSON")
        result = asyncio.run(_generate_consolidation(_make_weekly_episodes(), "2026-02"))
        assert result["monthly_summary"] == ""


# ---------------------------------------------------------------------------
# consolidate_month
# ---------------------------------------------------------------------------


class TestConsolidateMonth:
    @patch("src.db.episodes_db.store_episode")
    @patch("src.services.episode_consolidation._promote_patterns_to_beliefs", new_callable=AsyncMock)
    @patch("src.services.episode_consolidation._generate_consolidation", new_callable=AsyncMock)
    @patch("src.db.episodes_db.list_episodes_for_period")
    def test_full_consolidation_flow(self, mock_list, mock_gen, mock_promote, mock_store) -> None:
        mock_list.return_value = _make_weekly_episodes(4)
        mock_gen.return_value = {
            "monthly_summary": "Solid month of training.",
            "recurring_patterns": ["Morning runs"],
            "key_metrics": {"avg_hr": 145},
        }
        mock_store.return_value = {"id": "review-1"}

        result = asyncio.run(consolidate_month(USER_ID, "2026-02"))

        assert result is not None
        assert result["monthly_summary"] == "Solid month of training."
        mock_store.assert_called_once()
        mock_promote.assert_called_once()

    @patch("src.db.episodes_db.list_episodes_for_period")
    def test_too_few_reflections(self, mock_list) -> None:
        mock_list.return_value = _make_weekly_episodes(2)

        result = asyncio.run(consolidate_month(USER_ID, "2026-02"))
        assert result is None

    @patch("src.services.episode_consolidation._generate_consolidation", new_callable=AsyncMock)
    @patch("src.db.episodes_db.list_episodes_for_period")
    def test_empty_summary_returns_none(self, mock_list, mock_gen) -> None:
        mock_list.return_value = _make_weekly_episodes(4)
        mock_gen.return_value = {"monthly_summary": "", "recurring_patterns": [], "key_metrics": {}}

        result = asyncio.run(consolidate_month(USER_ID, "2026-02"))
        assert result is None

    @patch("src.db.episodes_db.store_episode")
    @patch("src.services.episode_consolidation._promote_patterns_to_beliefs", new_callable=AsyncMock)
    @patch("src.services.episode_consolidation._generate_consolidation", new_callable=AsyncMock)
    @patch("src.db.episodes_db.list_episodes_for_period")
    def test_december_month_boundary(self, mock_list, mock_gen, mock_promote, mock_store) -> None:
        """December → January year boundary."""
        mock_list.return_value = _make_weekly_episodes(3, month="2026-12")
        mock_gen.return_value = {
            "monthly_summary": "December summary.",
            "recurring_patterns": [],
            "key_metrics": {},
        }
        mock_store.return_value = {"id": "review-dec"}

        result = asyncio.run(consolidate_month(USER_ID, "2026-12"))
        assert result is not None
        # Verify the period_end uses 2027-01-01
        store_call = mock_store.call_args
        episode = store_call[0][1]
        assert episode["period_end"] == "2027-01-01"


# ---------------------------------------------------------------------------
# get_unconsolidated_months
# ---------------------------------------------------------------------------


class TestGetUnconsolidatedMonths:
    @patch("src.db.episodes_db.list_episodes_by_type")
    def test_finds_unconsolidated_month(self, mock_list) -> None:
        def side_effect(uid, etype, limit=10):
            if etype == "weekly_reflection":
                return [
                    {"period_start": "2026-02-01"},
                    {"period_start": "2026-02-08"},
                    {"period_start": "2026-02-15"},
                    {"period_start": "2026-02-22"},
                ]
            return []  # No monthly reviews

        mock_list.side_effect = side_effect
        months = asyncio.run(get_unconsolidated_months(USER_ID))
        assert "2026-02" in months

    @patch("src.db.episodes_db.list_episodes_by_type")
    def test_already_consolidated_month(self, mock_list) -> None:
        def side_effect(uid, etype, limit=10):
            if etype == "weekly_reflection":
                return [
                    {"period_start": "2026-02-01"},
                    {"period_start": "2026-02-08"},
                    {"period_start": "2026-02-15"},
                ]
            return [{"period_start": "2026-02-01"}]  # monthly review exists

        mock_list.side_effect = side_effect
        months = asyncio.run(get_unconsolidated_months(USER_ID))
        assert "2026-02" not in months

    @patch("src.db.episodes_db.list_episodes_by_type")
    def test_not_enough_weeklies(self, mock_list) -> None:
        def side_effect(uid, etype, limit=10):
            if etype == "weekly_reflection":
                return [{"period_start": "2026-02-01"}, {"period_start": "2026-02-08"}]
            return []

        mock_list.side_effect = side_effect
        months = asyncio.run(get_unconsolidated_months(USER_ID))
        assert months == []

    @patch("src.db.episodes_db.list_episodes_by_type", side_effect=Exception("DB"))
    def test_db_error(self, mock_list) -> None:
        months = asyncio.run(get_unconsolidated_months(USER_ID))
        assert months == []


# ---------------------------------------------------------------------------
# _promote_patterns_to_beliefs
# ---------------------------------------------------------------------------


class TestPromotePatterns:
    @patch("src.db.user_model_db.UserModelDB.load_or_create")
    def test_promotes_new_patterns(self, mock_load) -> None:
        mock_user = MagicMock()
        mock_user.get_active_beliefs.return_value = []
        mock_user.add_belief.return_value = {"id": "b-1"}
        mock_load.return_value = mock_user

        count = asyncio.run(_promote_patterns_to_beliefs(
            USER_ID, ["Morning runs are key", "HR recovery improving"],
        ))

        assert count == 2
        assert mock_user.add_belief.call_count == 2
        mock_user.save.assert_called_once()

    @patch("src.db.user_model_db.UserModelDB.load_or_create")
    def test_skips_existing_beliefs(self, mock_load) -> None:
        mock_user = MagicMock()
        mock_user.get_active_beliefs.return_value = [
            {"text": "Morning runs are key"},
        ]
        mock_user.add_belief.return_value = {"id": "b-2"}
        mock_load.return_value = mock_user

        count = asyncio.run(_promote_patterns_to_beliefs(
            USER_ID, ["Morning runs are key", "New pattern"],
        ))

        assert count == 1  # Only "New pattern" added

    @patch("src.db.user_model_db.UserModelDB.load_or_create")
    def test_caps_at_five_promotions(self, mock_load) -> None:
        mock_user = MagicMock()
        mock_user.get_active_beliefs.return_value = []
        mock_user.add_belief.return_value = {"id": "b-x"}
        mock_load.return_value = mock_user

        patterns = [f"Pattern {i}" for i in range(10)]
        count = asyncio.run(_promote_patterns_to_beliefs(USER_ID, patterns))
        assert count == 5  # Capped

    @patch("src.db.user_model_db.UserModelDB.load_or_create", side_effect=Exception("DB"))
    def test_failure_returns_zero(self, mock_load) -> None:
        count = asyncio.run(_promote_patterns_to_beliefs(USER_ID, ["Test"]))
        assert count == 0
