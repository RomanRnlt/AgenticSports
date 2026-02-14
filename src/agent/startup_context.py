"""Startup context builder -- pre-compute context for instant greeting (Gap 5).

This runs BEFORE the agent loop starts. It gathers all the data the agent
would need to call via tools, and packages it as a single text block that
gets injected into the system prompt.

The agent can then greet the athlete WITHOUT making any tool calls.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path


def build_startup_context(user_model, imported: list | None = None) -> str:
    """Build pre-computed context for the agent's first turn.

    Args:
        user_model: The UserModel instance.
        imported: Result of run_import() -- list of imported activity dicts.

    Returns:
        A text block that gets injected into the system prompt.
        The agent reads this instead of making startup tool calls.
    """
    profile = user_model.project_profile()
    parts = []

    # -- Athlete summary --
    name = profile.get("name", "Unknown")
    sports = profile.get("sports", [])
    goal = profile.get("goal", {})
    constraints = profile.get("constraints", {})

    parts.append(f"Athlete: {name}")
    if sports:
        parts.append(f"Sports: {', '.join(sports)}")
    if goal.get("event"):
        goal_str = goal["event"]
        if goal.get("target_date"):
            goal_str += f" (target: {goal['target_date']})"
        if goal.get("target_time"):
            goal_str += f" in {goal['target_time']}"
        parts.append(f"Goal: {goal_str}")
    if constraints.get("training_days_per_week"):
        parts.append(f"Training days: {constraints['training_days_per_week']}/week")
    if constraints.get("max_session_minutes"):
        parts.append(f"Max session: {constraints['max_session_minutes']} min")

    # -- Last session info --
    from src.tools.activity_store import list_activities
    activities = list_activities()
    if activities:
        sorted_acts = sorted(activities, key=lambda a: a.get("start_time", ""), reverse=True)
        last = sorted_acts[0]
        last_date = last.get("start_time", "")[:10]
        try:
            last_dt = datetime.fromisoformat(last.get("start_time", datetime.now().isoformat()))
            days_ago = (datetime.now() - last_dt.replace(tzinfo=None)).days
        except (ValueError, TypeError):
            days_ago = 0
        parts.append(f"Last session: {last.get('sport', 'unknown')} on {last_date} ({days_ago} days ago)")

        # Week summary
        week_ago = datetime.now() - timedelta(days=7)
        week_acts = []
        for a in sorted_acts:
            try:
                act_dt = datetime.fromisoformat(a.get("start_time", "2000-01-01"))
                if act_dt.replace(tzinfo=None) > week_ago:
                    week_acts.append(a)
            except (ValueError, TypeError):
                continue
        if week_acts:
            total_min = sum(a.get("duration_seconds", 0) / 60 for a in week_acts)
            sports_this_week = list(set(a.get("sport", "unknown") for a in week_acts))
            parts.append(
                f"This week: {len(week_acts)} sessions, "
                f"{round(total_min)} min total ({', '.join(sports_this_week)})"
            )

    # -- Import summary (adapted: run_import returns list[dict]) --
    if imported and len(imported) > 0:
        parts.append(f"New imports: {len(imported)} activities just imported from FIT files")

    # -- Plan compliance --
    plans_dir = Path("data/plans")
    if plans_dir.exists():
        plan_files = sorted(plans_dir.glob("plan_*.json"), reverse=True)
        if plan_files:
            try:
                latest_plan = json.loads(plan_files[0].read_text())
                sessions_planned = len(latest_plan.get("sessions", []))
                phase = latest_plan.get("training_phase", "unknown")
                parts.append(f"Active plan: {sessions_planned} sessions/week, phase: {phase}")
            except (json.JSONDecodeError, OSError):
                pass

    # -- Belief count --
    beliefs = user_model.get_active_beliefs()
    if beliefs:
        parts.append(f"Known beliefs: {len(beliefs)}")

    return "\n".join(parts)
