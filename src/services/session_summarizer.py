"""Session auto-summarizer — LLM-based session summarization.

When a user starts a new session, the previous session (if unsummarized)
is compressed into a 3-sentence summary + keyword tags.  This runs
asynchronously so it never blocks the chat flow.

Usage::

    from src.services.session_summarizer import summarize_previous_session

    # Fire-and-forget from the chat endpoint:
    asyncio.create_task(summarize_previous_session(user_id))
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

# Maximum messages fed to the summarizer (prevents excessive LLM input).
_MAX_MESSAGES_FOR_SUMMARY = 50

# Minimum message count to justify a summary.
_MIN_MESSAGES_FOR_SUMMARY = 4

# LLM model for summarization — fast and cheap.
_SUMMARY_MODEL = "gemini/gemini-2.5-flash"

_SUMMARY_PROMPT = (
    "Summarize the following coaching conversation in EXACTLY this JSON format:\n"
    '{{"summary": "<3 sentences max>", "tags": ["tag1", "tag2", ...]}}\n\n'
    "RULES:\n"
    "- Summary: 3 sentences maximum, focus on key decisions and outcomes.\n"
    "- Tags: 3-5 keyword tags (e.g., 'marathon', 'injury', 'plan-change').\n"
    "- Output valid JSON only — no markdown, no explanation.\n\n"
    "CONVERSATION:\n{messages}"
)


def _format_messages_for_prompt(messages: list[dict]) -> str:
    """Format session messages into a readable conversation transcript."""
    lines = []
    for msg in messages[-_MAX_MESSAGES_FOR_SUMMARY:]:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if role == "tool_call":
            tool_name = msg.get("meta", {}).get("tool", "tool")
            lines.append(f"[Tool: {tool_name}]")
        elif role in ("user", "model"):
            label = "User" if role == "user" else "Coach"
            # Truncate very long messages
            text = content[:500] if content else ""
            lines.append(f"{label}: {text}")
    return "\n".join(lines)


async def _generate_summary(messages: list[dict]) -> dict:
    """Call LLM to generate a session summary.

    Returns:
        Dict with ``summary`` (str) and ``tags`` (list[str]) keys.
        On failure, returns a minimal fallback dict.
    """
    import json

    transcript = _format_messages_for_prompt(messages)
    prompt = _SUMMARY_PROMPT.format(messages=transcript)

    try:
        from src.agent.llm import chat_completion

        response = await asyncio.to_thread(
            chat_completion,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            model=_SUMMARY_MODEL,
        )

        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            return {"summary": "", "tags": []}

        # Strip markdown fences if the LLM added them
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        parsed = json.loads(raw)
        return {
            "summary": str(parsed.get("summary", "")),
            "tags": list(parsed.get("tags", [])),
        }
    except Exception:
        logger.warning("Session summary generation failed", exc_info=True)
        return {"summary": "", "tags": []}


async def summarize_session(session_id: str) -> dict | None:
    """Summarize a single session by its ID.

    Loads messages, generates an LLM summary, and persists it.

    Args:
        session_id: UUID of the session to summarize.

    Returns:
        The summary dict, or None if the session was too short or failed.
    """
    from src.db.session_store_db import (
        load_session_messages,
        update_session_summary,
    )

    messages = await asyncio.to_thread(load_session_messages, session_id)

    if len(messages) < _MIN_MESSAGES_FOR_SUMMARY:
        logger.debug(
            "Session %s too short (%d messages) — skipping summary",
            session_id, len(messages),
        )
        return None

    summary_data = await _generate_summary(messages)

    if not summary_data["summary"]:
        return None

    # Count user turns and tool calls for metadata
    user_turns = sum(1 for m in messages if m.get("role") == "user")
    tool_calls = sum(1 for m in messages if m.get("role") == "tool_call")

    # Build compressed_summary with tags
    compressed = summary_data["summary"]
    if summary_data["tags"]:
        compressed += "\n\nTags: " + ", ".join(summary_data["tags"])

    await asyncio.to_thread(
        update_session_summary,
        session_id,
        compressed,
        user_turns,
        tool_calls,
    )

    logger.info(
        "Session %s summarized: %d chars, %d tags",
        session_id, len(compressed), len(summary_data["tags"]),
    )
    return summary_data


async def summarize_previous_session(user_id: str) -> dict | None:
    """Find and summarize the user's most recent unsummarized session.

    Called on new session creation to summarize the PREVIOUS session.
    Non-blocking: returns None on any failure.

    Args:
        user_id: UUID of the user.

    Returns:
        The summary dict, or None.
    """
    try:
        from src.db.session_store_db import get_unsummarized_sessions

        sessions = await asyncio.to_thread(
            get_unsummarized_sessions, user_id, limit=1,
        )

        if not sessions:
            return None

        session = sessions[0]
        return await summarize_session(session["id"])
    except Exception:
        logger.warning(
            "summarize_previous_session failed for user %s", user_id,
            exc_info=True,
        )
        return None
