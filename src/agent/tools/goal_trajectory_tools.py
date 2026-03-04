"""Goal trajectory assessment tool — assess progress toward athlete goals.

The agent calls ``assess_goal_trajectory`` to get an LLM-powered analysis
of whether the athlete is on track, ahead, behind, or at risk for their
stated goal. The analysis considers training history, health trends,
periodization phase, and known athlete patterns (beliefs).

Results are optionally persisted as trajectory snapshots for trend tracking.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from src.agent.tools.registry import Tool, ToolRegistry
from src.config import get_settings

logger = logging.getLogger(__name__)


def register_goal_trajectory_tools(
    registry: ToolRegistry,
    user_model=None,
) -> None:
    """Register goal trajectory tools on the given *registry*.

    Args:
        registry: Tool registry to register tools on.
        user_model: User model instance. May be None for restricted registries.
    """
    settings = get_settings()

    def assess_goal_trajectory(save_snapshot: bool = True) -> dict:
        """Assess athlete's trajectory toward their stated goal.

        Gathers training data, health trends, periodization context, and
        beliefs, then calls the LLM-based trajectory analyzer.

        Args:
            save_snapshot: Whether to persist the snapshot to DB (default True).

        Returns:
            Dict with trajectory_status, confidence, analysis, recommendations, etc.
        """
        # 1. Get profile from user_model
        if user_model is None:
            return {"error": "No user model available", "trajectory_status": "insufficient_data"}

        try:
            profile = user_model.project_profile()
        except Exception:
            logger.warning("Failed to get user profile for trajectory", exc_info=True)
            return {"error": "Failed to load profile", "trajectory_status": "insufficient_data"}

        # 2. Extract goal from profile
        goal = profile.get("goal", {})
        if not goal or not goal.get("event"):
            return {
                "error": "No goal defined — set a goal first",
                "trajectory_status": "insufficient_data",
            }

        # 3. Get user_id
        user_id = settings.agenticsports_user_id
        if not user_id:
            return {"error": "No user configured", "trajectory_status": "insufficient_data"}

        # 4. Load training activities (last 28 days)
        training_summary = _build_training_summary(user_id)

        # 5. Load health trends
        health_trends = _build_health_trends(user_id)

        # 6. Get active macrocycle for periodization phase
        periodization_phase = _get_current_phase(user_id)

        # 7. Get beliefs from user_model
        beliefs = _get_beliefs(user_model)

        # 8. Get previous trajectory
        previous_trajectory = _get_previous_trajectory(user_id, goal.get("event", ""))

        # 9. Call analyze_trajectory
        from src.services.goal_trajectory import analyze_trajectory

        result = analyze_trajectory(
            goal=goal,
            profile=profile,
            training_summary=training_summary,
            health_trends=health_trends,
            periodization_phase=periodization_phase,
            beliefs=beliefs,
            previous_trajectory=previous_trajectory,
        )

        # 10. Optionally save snapshot
        if save_snapshot and result.trajectory_status != "insufficient_data":
            _save_snapshot(user_id, goal.get("event", "unknown"), result, {
                "training_summary": training_summary,
                "health_trends": health_trends,
                "periodization_phase": periodization_phase,
            })

        return {
            "trajectory_status": result.trajectory_status,
            "confidence": result.confidence,
            "projected_outcome": result.projected_outcome,
            "analysis": result.analysis,
            "key_factors": result.key_factors,
            "risk_factors": result.risk_factors,
            "recommendations": result.recommendations,
        }

    registry.register(Tool(
        name="assess_goal_trajectory",
        description=(
            "Assess the athlete's progress trajectory toward their stated goal. "
            "Analyzes training data (last 28 days), health trends, periodization "
            "phase, and known patterns to determine if the athlete is on_track, "
            "ahead, behind, or at_risk. Use during check-ins, after plan reviews, "
            "or when the athlete asks about their goal progress."
        ),
        handler=assess_goal_trajectory,
        parameters={
            "type": "object",
            "properties": {
                "save_snapshot": {
                    "type": "boolean",
                    "description": (
                        "Whether to save a trajectory snapshot for trend tracking "
                        "(default true)."
                    ),
                },
            },
        },
        category="analysis",
    ))


# ---------------------------------------------------------------------------
# Private helpers — gather context for trajectory analysis
# ---------------------------------------------------------------------------


def _build_training_summary(user_id: str) -> dict | None:
    """Load last 28 days of activities and summarize."""
    try:
        from src.db.activity_store_db import list_activities

        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=28)).isoformat()
        activities = list_activities(user_id, limit=100, after=cutoff)

        if not activities:
            return None

        total_sessions = len(activities)
        sports: dict[str, int] = {}
        total_duration_min = 0

        for act in activities:
            sport = act.get("sport", "unknown")
            sports[sport] = sports.get(sport, 0) + 1
            total_duration_min += act.get("duration_minutes", 0)

        return {
            "total_sessions": total_sessions,
            "total_duration_minutes": total_duration_min,
            "sports_breakdown": sports,
            "days_covered": 28,
        }
    except Exception:
        logger.warning("Failed to build training summary", exc_info=True)
        return None


def _build_health_trends(user_id: str) -> dict | None:
    """Build health summary from recent metrics."""
    try:
        from src.services.health_context import build_health_summary

        summary = build_health_summary(user_id, days=7)
        if summary and summary.get("data_available"):
            return {
                "latest": summary.get("latest", {}),
                "averages_7d": summary.get("averages_7d", {}),
            }
        return None
    except Exception:
        logger.warning("Failed to build health trends", exc_info=True)
        return None


def _get_current_phase(user_id: str) -> str | None:
    """Get the current periodization phase from the active macrocycle."""
    try:
        from src.db.macrocycle_db import get_active_macrocycle

        macro = get_active_macrocycle(user_id)
        if not macro:
            return None

        weeks = macro.get("weeks", [])
        if not weeks:
            return None

        # Find current week based on start_date
        start_date_str = macro.get("start_date", "")
        if not start_date_str:
            return weeks[0].get("phase") if weeks else None

        try:
            start_date = datetime.fromisoformat(start_date_str.replace("Z", "+00:00"))
            now = datetime.now(tz=timezone.utc)
            weeks_elapsed = max(0, (now - start_date).days // 7)
            week_index = min(weeks_elapsed, len(weeks) - 1)
            return weeks[week_index].get("phase")
        except (ValueError, TypeError):
            return weeks[0].get("phase") if weeks else None

    except Exception:
        logger.warning("Failed to get current periodization phase", exc_info=True)
        return None


def _get_beliefs(user_model) -> list[dict] | None:
    """Get active beliefs from user model."""
    try:
        beliefs = user_model.get_active_beliefs()
        return beliefs if beliefs else None
    except Exception:
        logger.warning("Failed to get beliefs", exc_info=True)
        return None


def _get_previous_trajectory(user_id: str, goal_name: str) -> dict | None:
    """Load the most recent trajectory snapshot for this goal."""
    try:
        from src.db.goal_trajectory_db import get_latest_trajectory

        return get_latest_trajectory(user_id, goal_name)
    except Exception:
        logger.warning("Failed to get previous trajectory", exc_info=True)
        return None


def _save_snapshot(
    user_id: str,
    goal_name: str,
    result,
    context: dict,
) -> None:
    """Persist a trajectory snapshot to the database."""
    try:
        from src.db.goal_trajectory_db import save_trajectory_snapshot

        save_trajectory_snapshot(
            user_id=user_id,
            goal_name=goal_name,
            trajectory_status=result.trajectory_status,
            confidence=result.confidence,
            projected_outcome=result.projected_outcome,
            analysis=result.analysis,
            recommendations=result.recommendations,
            risk_factors=result.risk_factors,
            context_snapshot=context,
        )
    except Exception:
        logger.warning("Failed to save trajectory snapshot", exc_info=True)
