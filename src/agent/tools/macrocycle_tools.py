"""Macrocycle planning tools -- create, view, and save multi-week training plans.

A macrocycle defines the high-level training structure across weeks/months,
including phases (base, build, peak, taper), weekly volume targets, and
intensity distribution. Weekly training plans are then derived from
the active macrocycle.

Tools:
- create_macrocycle_plan: Generate a macrocycle via LLM sub-agent
- get_macrocycle: Retrieve the active macrocycle from DB
- save_macrocycle: Persist a reviewed macrocycle to DB
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

from src.agent.llm import chat_completion
from src.agent.json_utils import extract_json
from src.agent.tools.registry import Tool, ToolRegistry
from src.config import get_settings

logger = logging.getLogger(__name__)


def register_macrocycle_tools(registry: ToolRegistry, user_model) -> None:
    """Register macrocycle planning tools on the given *registry*."""
    _settings = get_settings()

    def create_macrocycle_plan(
        name: str,
        weeks: int = 12,
        periodization_model: str | None = None,
        start_date: str | None = None,
    ) -> dict:
        """Generate a multi-week macrocycle plan using LLM sub-agent.

        Args:
            name: Descriptive name for the macrocycle.
            weeks: Total weeks (4-52, default 12).
            periodization_model: Optional name of agent-defined periodization model.
            start_date: Optional start date (YYYY-MM-DD). Defaults to today.

        Returns:
            Dict with name, weeks plan, start_date -- ready for athlete review.
        """
        from src.agent.prompts import (
            MACROCYCLE_SYSTEM_PROMPT,
            build_macrocycle_prompt,
        )

        # Validate weeks range
        clamped_weeks = max(4, min(52, weeks))

        # Get athlete context
        profile = user_model.project_profile()
        beliefs = user_model.get_active_beliefs(min_confidence=0.6)

        # Load periodization model if specified
        model_data = None
        if periodization_model and _settings.use_supabase:
            try:
                from src.db.agent_config_db import get_periodization_model
                model_data = get_periodization_model(
                    _settings.agenticsports_user_id,
                    periodization_model,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load periodization model '%s': %s",
                    periodization_model,
                    exc,
                )

        # Load recent activities (28 days)
        activities = None
        if _settings.use_supabase:
            try:
                from src.db import list_activities as db_list_activities
                cutoff = (date.today() - timedelta(days=28)).isoformat()
                activities = db_list_activities(
                    _settings.agenticsports_user_id,
                    limit=50,
                    after=cutoff,
                )
            except Exception as exc:
                logger.warning("Failed to load activities: %s", exc)

        # Load health summary
        health_summary = None
        if _settings.use_supabase:
            try:
                from src.services.health_context import build_health_summary
                health_summary = build_health_summary(
                    _settings.agenticsports_user_id,
                    days=7,
                )
            except Exception as exc:
                logger.warning("Failed to load health summary: %s", exc)

        # Build prompt and call LLM
        user_prompt = build_macrocycle_prompt(
            profile=profile,
            total_weeks=clamped_weeks,
            beliefs=beliefs,
            activities=activities,
            health_summary=health_summary,
            periodization_model=model_data,
        )

        response = chat_completion(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=MACROCYCLE_SYSTEM_PROMPT,
            temperature=0.7,
        )

        plan = extract_json(response.choices[0].message.content.strip())

        resolved_start = start_date or date.today().isoformat()

        return {
            "name": name,
            "total_weeks": clamped_weeks,
            "start_date": resolved_start,
            "weeks": plan.get("weeks", []),
            "periodization_model_name": periodization_model,
            "_generated_at": datetime.now().isoformat(),
            "_status": "draft",
        }

    registry.register(Tool(
        name="create_macrocycle_plan",
        description=(
            "Generate a multi-week macrocycle training plan (4-52 weeks) with "
            "training phases, weekly volume targets, and intensity distribution. "
            "The plan is returned as a draft for athlete review — use "
            "save_macrocycle to persist after approval."
        ),
        handler=create_macrocycle_plan,
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Descriptive name (e.g., 'Marathon Build 2026')",
                },
                "weeks": {
                    "type": "integer",
                    "description": "Total weeks (4-52, default 12)",
                },
                "periodization_model": {
                    "type": "string",
                    "description": "Name of agent-defined periodization model to follow",
                    "nullable": True,
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format (default: today)",
                    "nullable": True,
                },
            },
            "required": ["name"],
        },
        category="planning",
    ))

    def get_macrocycle() -> dict:
        """Return the active macrocycle from the database.

        Returns:
            The active macrocycle dict, or an error dict if none exists.
        """
        if not _settings.use_supabase:
            return {"error": "Supabase not configured — macrocycle storage unavailable"}

        try:
            from src.db.macrocycle_db import get_active_macrocycle
            macrocycle = get_active_macrocycle(_settings.agenticsports_user_id)
            if not macrocycle:
                return {"error": "No active macrocycle found. Use create_macrocycle_plan to create one."}
            return macrocycle
        except Exception as exc:
            logger.warning("Failed to load active macrocycle: %s", exc)
            return {"error": f"Failed to load macrocycle: {exc}"}

    registry.register(Tool(
        name="get_macrocycle",
        description=(
            "Retrieve the currently active macrocycle plan from the database. "
            "Returns the full multi-week structure with phases, volume targets, "
            "and key sessions."
        ),
        handler=get_macrocycle,
        parameters={"type": "object", "properties": {}},
        category="planning",
    ))

    def save_macrocycle(macrocycle: dict) -> dict:
        """Persist a macrocycle plan to the database.

        Args:
            macrocycle: Dict with name, total_weeks, start_date, weeks array,
                        and optional periodization_model_name.

        Returns:
            Confirmation dict with saved status and macrocycle ID.
        """
        if not _settings.use_supabase:
            return {"error": "Supabase not configured — macrocycle storage unavailable"}

        name = macrocycle.get("name")
        if not name:
            return {"error": "Macrocycle must have a 'name' field"}

        weeks_data = macrocycle.get("weeks", [])
        if not weeks_data:
            return {"error": "Macrocycle must have a non-empty 'weeks' array"}

        total_weeks = macrocycle.get("total_weeks", len(weeks_data))
        start_date = macrocycle.get("start_date", date.today().isoformat())
        period_model = macrocycle.get("periodization_model_name")

        try:
            from src.db.macrocycle_db import store_macrocycle
            row = store_macrocycle(
                user_id=_settings.agenticsports_user_id,
                name=name,
                total_weeks=total_weeks,
                start_date=start_date,
                weeks=weeks_data,
                periodization_model_name=period_model,
            )
            return {
                "saved": True,
                "id": row.get("id"),
                "name": name,
                "total_weeks": total_weeks,
                "status": "active",
            }
        except Exception as exc:
            logger.warning("Failed to save macrocycle: %s", exc)
            return {"error": f"Failed to save macrocycle: {exc}"}

    registry.register(Tool(
        name="save_macrocycle",
        description=(
            "Save a reviewed macrocycle plan to the database. Any previously "
            "active macrocycle is automatically archived. Use this after the "
            "athlete approves the plan from create_macrocycle_plan."
        ),
        handler=save_macrocycle,
        parameters={
            "type": "object",
            "properties": {
                "macrocycle": {
                    "type": "object",
                    "description": (
                        "The macrocycle to save, with keys: name, total_weeks, "
                        "start_date, weeks (array), periodization_model_name (optional)"
                    ),
                },
            },
            "required": ["macrocycle"],
        },
        category="planning",
    ))
