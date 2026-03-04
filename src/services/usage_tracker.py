"""LLM usage tracking and cost monitoring for AgenticSports.

Tracks token consumption per user per day and enforces a configurable
daily budget. All tracking is non-blocking and fail-open: tracking
failures never block the agent loop or end-user requests.

Usage::

    from src.services.usage_tracker import track_usage, check_budget, get_daily_usage

    # After an LLM call:
    record = track_usage(user_id, llm_response)

    # Before accepting a chat request:
    if not check_budget(user_id):
        raise HTTPException(status_code=429, detail="Daily token budget exceeded")
"""

import logging
from dataclasses import dataclass

from src.config import get_settings
from src.db.client import get_supabase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model pricing table — input and output price per 1M tokens (USD)
# ---------------------------------------------------------------------------

MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gemini/gemini-2.5-flash": (0.15, 0.60),
    "gemini/gemini-2.0-flash": (0.10, 0.40),
    "anthropic/claude-haiku-4-5-20251001": (1.00, 5.00),
    "anthropic/claude-sonnet-4-5-20250514": (3.00, 15.00),
}

# Conservative fallback when model is unknown or not in the table
DEFAULT_PRICING: tuple[float, float] = (0.50, 2.00)


# ---------------------------------------------------------------------------
# Immutable result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UsageRecord:
    """Immutable snapshot of a single LLM call's token consumption."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    estimated_cost_usd: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _estimate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """Calculate estimated USD cost for one LLM call.

    Args:
        prompt_tokens: Number of input tokens consumed.
        completion_tokens: Number of output tokens generated.
        model: LiteLLM model identifier (e.g. "gemini/gemini-2.5-flash").

    Returns:
        Estimated cost in USD.
    """
    input_price, output_price = MODEL_PRICING.get(model, DEFAULT_PRICING)
    return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def track_usage(user_id: str, response) -> UsageRecord | None:
    """Extract token usage from an LLM response and persist it to daily_usage.

    Calls the ``increment_usage`` Supabase RPC, which does an atomic
    INSERT ... ON CONFLICT DO UPDATE so concurrent requests are safe.

    Args:
        user_id: The authenticated user's UUID.
        response: A LiteLLM ModelResponse (or any object with a ``.usage``
                  attribute containing ``prompt_tokens`` / ``completion_tokens``).

    Returns:
        A :class:`UsageRecord` on success, or ``None`` when the response
        carries no usage information.

    Note:
        This function is intentionally fail-open: any exception is logged at
        DEBUG level and ``None`` is returned.  The agent loop must never be
        blocked by usage tracking failures.
    """
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None

        prompt_tokens: int = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens: int = getattr(usage, "completion_tokens", 0) or 0
        total_tokens: int = prompt_tokens + completion_tokens
        model: str = getattr(response, "model", "") or ""
        cost: float = _estimate_cost(prompt_tokens, completion_tokens, model)

        record = UsageRecord(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=model,
            estimated_cost_usd=cost,
        )

        # Persist via Supabase RPC (fail-safe: log and continue on error)
        try:
            client = get_supabase()
            client.rpc(
                "increment_usage",
                {"p_user_id": user_id, "p_tokens": total_tokens},
            ).execute()
        except Exception:
            logger.debug("Failed to persist usage to Supabase", exc_info=True)

        return record

    except Exception:
        logger.debug("Usage tracking failed", exc_info=True)
        return None


def get_daily_usage(user_id: str) -> dict:
    """Read today's token usage for a user from the daily_usage table.

    Args:
        user_id: The authenticated user's UUID.

    Returns:
        A dict with keys ``request_count``, ``token_count``, and ``date``.
        Returns zero counts when no row exists yet for today.
    """
    from datetime import date

    today = date.today().isoformat()
    try:
        client = get_supabase()
        result = (
            client.table("daily_usage")
            .select("*")
            .eq("user_id", user_id)
            .eq("usage_date", today)
            .execute()
        )
        if result.data:
            row = result.data[0]
            return {
                "request_count": row.get("request_count", 0),
                "token_count": row.get("token_count", 0),
                "date": today,
            }
        return {"request_count": 0, "token_count": 0, "date": today}
    except Exception:
        logger.debug("Failed to read daily usage", exc_info=True)
        return {"request_count": 0, "token_count": 0, "date": today}


def check_budget(user_id: str, daily_token_limit: int | None = None) -> bool:
    """Check whether a user is within their daily token budget.

    Args:
        user_id: The authenticated user's UUID.
        daily_token_limit: Override the budget from settings (useful in tests).
            When ``None``, reads ``settings.daily_token_budget``.

    Returns:
        ``True`` when the user is under the limit (request should proceed).
        ``False`` when the daily budget has been exhausted.

    Note:
        Fail-open policy: any exception returns ``True`` so that tracking
        failures never block legitimate user requests.
    """
    try:
        if daily_token_limit is None:
            daily_token_limit = get_settings().daily_token_budget
        usage = get_daily_usage(user_id)
        return usage["token_count"] < daily_token_limit
    except Exception:
        logger.debug("Budget check failed — fail-open", exc_info=True)
        return True
