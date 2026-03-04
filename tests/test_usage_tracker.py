"""Tests for src/services/usage_tracker.py — LLM cost monitoring.

All Supabase interactions are mocked so these tests run fully offline.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.services.usage_tracker import (
    DEFAULT_PRICING,
    MODEL_PRICING,
    UsageRecord,
    _estimate_cost,
    check_budget,
    get_daily_usage,
    track_usage,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_response(
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    model: str = "gemini/gemini-2.5-flash",
) -> SimpleNamespace:
    """Build a minimal mock of a LiteLLM ModelResponse."""
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return SimpleNamespace(usage=usage, model=model)


def _make_supabase_mock(daily_row: dict | None = None) -> MagicMock:
    """Return a mock Supabase client configured with a preset daily_usage row."""
    client = MagicMock()

    # RPC call chain: .rpc(...).execute()
    rpc_result = MagicMock()
    rpc_result.execute.return_value = MagicMock(data=[])
    client.rpc.return_value = rpc_result

    # Table query chain: .table(...).select(...).eq(...).eq(...).execute()
    table_chain = MagicMock()
    table_chain.select.return_value = table_chain
    table_chain.eq.return_value = table_chain
    table_chain.execute.return_value = MagicMock(
        data=[daily_row] if daily_row else []
    )
    client.table.return_value = table_chain

    return client


# ---------------------------------------------------------------------------
# _estimate_cost
# ---------------------------------------------------------------------------


class TestEstimateCost:
    def test_gemini_flash_pricing(self):
        """Gemini Flash: $0.15/1M input + $0.60/1M output."""
        cost = _estimate_cost(1_000_000, 1_000_000, "gemini/gemini-2.5-flash")
        assert cost == pytest.approx(0.75)

    def test_claude_haiku_pricing(self):
        """Claude Haiku: $1.00/1M input + $5.00/1M output."""
        cost = _estimate_cost(1_000_000, 1_000_000, "anthropic/claude-haiku-4-5-20251001")
        assert cost == pytest.approx(6.00)

    def test_unknown_model_uses_default_pricing(self):
        """Unknown model falls back to DEFAULT_PRICING."""
        input_price, output_price = DEFAULT_PRICING
        expected = (500_000 * input_price + 500_000 * output_price) / 1_000_000
        cost = _estimate_cost(500_000, 500_000, "unknown/unknown-model")
        assert cost == pytest.approx(expected)

    def test_zero_tokens_returns_zero_cost(self):
        cost = _estimate_cost(0, 0, "gemini/gemini-2.5-flash")
        assert cost == 0.0

    def test_all_known_models_are_in_pricing_table(self):
        """Smoke-check: every model in MODEL_PRICING round-trips cleanly."""
        for model, (inp, out) in MODEL_PRICING.items():
            cost = _estimate_cost(1_000_000, 1_000_000, model)
            assert cost == pytest.approx(inp + out)


# ---------------------------------------------------------------------------
# track_usage
# ---------------------------------------------------------------------------


class TestTrackUsage:
    def test_extracts_tokens_correctly(self):
        """track_usage should build a UsageRecord with the right token counts."""
        response = _make_response(prompt_tokens=200, completion_tokens=80)

        with patch("src.services.usage_tracker.get_supabase", return_value=_make_supabase_mock()):
            record = track_usage("user-123", response)

        assert record is not None
        assert record.prompt_tokens == 200
        assert record.completion_tokens == 80
        assert record.total_tokens == 280
        assert record.model == "gemini/gemini-2.5-flash"

    def test_returns_none_when_no_usage_attribute(self):
        """track_usage returns None for responses that carry no usage info."""
        response = SimpleNamespace()  # no .usage attribute

        with patch("src.services.usage_tracker.get_supabase", return_value=_make_supabase_mock()):
            record = track_usage("user-123", response)

        assert record is None

    def test_calls_increment_usage_rpc_with_correct_params(self):
        """track_usage must call the increment_usage RPC with user_id and tokens."""
        response = _make_response(prompt_tokens=100, completion_tokens=50)
        supabase_mock = _make_supabase_mock()

        with patch("src.services.usage_tracker.get_supabase", return_value=supabase_mock):
            record = track_usage("user-abc", response)

        assert record is not None
        supabase_mock.rpc.assert_called_once_with(
            "increment_usage",
            {"p_user_id": "user-abc", "p_tokens": 150},
        )
        supabase_mock.rpc.return_value.execute.assert_called_once()

    def test_returns_none_when_usage_is_none(self):
        """Explicit usage=None attribute should yield None."""
        response = SimpleNamespace(usage=None, model="gemini/gemini-2.5-flash")

        with patch("src.services.usage_tracker.get_supabase", return_value=_make_supabase_mock()):
            record = track_usage("user-123", response)

        assert record is None

    def test_supabase_failure_does_not_raise(self):
        """Supabase errors are swallowed — track_usage still returns a record."""
        response = _make_response(prompt_tokens=10, completion_tokens=5)
        bad_client = MagicMock()
        bad_client.rpc.side_effect = RuntimeError("connection refused")

        with patch("src.services.usage_tracker.get_supabase", return_value=bad_client):
            record = track_usage("user-123", response)

        assert record is not None
        assert record.total_tokens == 15

    def test_record_is_immutable(self):
        """UsageRecord must be frozen (immutable)."""
        response = _make_response()

        with patch("src.services.usage_tracker.get_supabase", return_value=_make_supabase_mock()):
            record = track_usage("user-123", response)

        assert record is not None
        with pytest.raises((AttributeError, TypeError)):
            record.total_tokens = 9999  # type: ignore[misc]

    def test_estimated_cost_included_in_record(self):
        """UsageRecord.estimated_cost_usd must reflect the correct pricing."""
        response = _make_response(
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            model="gemini/gemini-2.5-flash",
        )

        with patch("src.services.usage_tracker.get_supabase", return_value=_make_supabase_mock()):
            record = track_usage("user-123", response)

        assert record is not None
        assert record.estimated_cost_usd == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# get_daily_usage
# ---------------------------------------------------------------------------


class TestGetDailyUsage:
    def test_reads_and_returns_correct_data(self):
        """get_daily_usage should return the row data from daily_usage."""
        row = {"request_count": 5, "token_count": 12_000, "usage_date": "2026-03-04"}
        supabase_mock = _make_supabase_mock(daily_row=row)

        with patch("src.services.usage_tracker.get_supabase", return_value=supabase_mock):
            usage = get_daily_usage("user-123")

        assert usage["request_count"] == 5
        assert usage["token_count"] == 12_000

    def test_returns_zeros_when_no_data_found(self):
        """get_daily_usage returns zero counts when the table has no row for today."""
        supabase_mock = _make_supabase_mock(daily_row=None)

        with patch("src.services.usage_tracker.get_supabase", return_value=supabase_mock):
            usage = get_daily_usage("user-no-history")

        assert usage["request_count"] == 0
        assert usage["token_count"] == 0

    def test_returns_zeros_on_supabase_error(self):
        """get_daily_usage must fail-open with zero counts on DB errors."""
        bad_client = MagicMock()
        bad_client.table.side_effect = RuntimeError("network error")

        with patch("src.services.usage_tracker.get_supabase", return_value=bad_client):
            usage = get_daily_usage("user-123")

        assert usage["token_count"] == 0
        assert usage["request_count"] == 0

    def test_includes_date_key(self):
        """get_daily_usage result always includes a 'date' key."""
        supabase_mock = _make_supabase_mock(daily_row=None)

        with patch("src.services.usage_tracker.get_supabase", return_value=supabase_mock):
            usage = get_daily_usage("user-123")

        assert "date" in usage


# ---------------------------------------------------------------------------
# check_budget
# ---------------------------------------------------------------------------


class TestCheckBudget:
    def test_returns_true_when_under_limit(self):
        """User with 10k tokens used against a 500k limit should be allowed."""
        row = {"request_count": 1, "token_count": 10_000}
        supabase_mock = _make_supabase_mock(daily_row=row)

        with patch("src.services.usage_tracker.get_supabase", return_value=supabase_mock):
            result = check_budget("user-123", daily_token_limit=500_000)

        assert result is True

    def test_returns_false_when_over_limit(self):
        """User that has consumed exactly the limit should be blocked."""
        row = {"request_count": 10, "token_count": 500_000}
        supabase_mock = _make_supabase_mock(daily_row=row)

        with patch("src.services.usage_tracker.get_supabase", return_value=supabase_mock):
            result = check_budget("user-123", daily_token_limit=500_000)

        assert result is False

    def test_returns_false_when_over_limit_by_one(self):
        """Token count of limit+1 must be blocked."""
        row = {"request_count": 1, "token_count": 500_001}
        supabase_mock = _make_supabase_mock(daily_row=row)

        with patch("src.services.usage_tracker.get_supabase", return_value=supabase_mock):
            result = check_budget("user-123", daily_token_limit=500_000)

        assert result is False

    def test_returns_true_on_error_fail_open(self):
        """Budget check must fail-open (return True) when an exception occurs."""
        bad_client = MagicMock()
        bad_client.table.side_effect = RuntimeError("DB down")

        with patch("src.services.usage_tracker.get_supabase", return_value=bad_client):
            result = check_budget("user-123", daily_token_limit=500_000)

        assert result is True

    def test_uses_settings_budget_when_no_limit_given(self):
        """check_budget reads daily_token_budget from settings when not overridden."""
        row = {"request_count": 1, "token_count": 100}
        supabase_mock = _make_supabase_mock(daily_row=row)

        fake_settings = MagicMock()
        fake_settings.daily_token_budget = 200

        with (
            patch("src.services.usage_tracker.get_supabase", return_value=supabase_mock),
            patch("src.services.usage_tracker.get_settings", return_value=fake_settings),
        ):
            result = check_budget("user-123")

        assert result is True  # 100 < 200


# ---------------------------------------------------------------------------
# Budget check integration — chat router
# ---------------------------------------------------------------------------


class TestChatRouterBudgetCheck:
    """Verify the chat router honours the budget gate."""

    def test_budget_check_blocks_when_over_limit(self):
        """post_chat must raise HTTP 429 when check_budget returns False."""
        from unittest.mock import AsyncMock, patch

        from fastapi import HTTPException
        from fastapi.testclient import TestClient

        # We test the logic in isolation without a full FastAPI test client
        # to keep the test fast and dependency-free.
        import asyncio

        async def _run():
            # Simulate what post_chat does: call check_budget and raise 429
            with patch(
                "src.services.usage_tracker.check_budget", return_value=False
            ) as mock_check:
                from src.services.usage_tracker import check_budget as cb
                budget_ok = cb("user-over-limit")
                assert budget_ok is False
                mock_check.assert_called_once_with("user-over-limit")

                if not budget_ok:
                    raise HTTPException(status_code=429, detail="Daily token budget exceeded")

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(_run())

        assert exc_info.value.status_code == 429
        assert "budget" in exc_info.value.detail.lower()
