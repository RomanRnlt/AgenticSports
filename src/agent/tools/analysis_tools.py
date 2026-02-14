"""Analysis tools -- compute insights from training data.

These are the equivalent of Claude Code using Bash to run analysis commands.
The agent calls these when it needs computed insights, not raw data.

FIX vs Blueprint: build_compliance_summary() does not exist in the codebase.
We use match_plan_sessions() from activity_context instead.
"""

from src.agent.tools.registry import Tool, ToolRegistry


def register_analysis_tools(registry: ToolRegistry):
    """Register all analysis tools."""

    def analyze_training_load(period_days: int = 28) -> dict:
        """Analyze training load, trends, and recovery status."""
        from src.tools.activity_store import list_activities
        from src.tools.activity_context import build_planning_context
        from datetime import datetime, timedelta

        activities = list_activities()
        if not activities:
            return {
                "status": "no_data",
                "message": "No training data available. Cannot analyze load.",
                "recommendation": "Start conservative -- gather baseline data first.",
            }

        # Use existing activity_context engine for aggregation
        context_text = build_planning_context(activities)

        # Compute weekly summaries for the requested period
        cutoff = datetime.now() - timedelta(days=period_days)
        recent = []
        for a in activities:
            try:
                dt = datetime.fromisoformat(a.get("start_time", "2000-01-01"))
                if dt.replace(tzinfo=None) > cutoff:
                    recent.append(a)
            except (ValueError, TypeError):
                continue

        total_sessions = len(recent)
        total_minutes = sum(a.get("duration_seconds", 0) / 60 for a in recent)
        total_trimp = sum(a.get("trimp", 0) for a in recent)
        sports_seen = list(set(a.get("sport", "unknown") for a in recent))
        weeks = max(1, period_days / 7)

        return {
            "period_days": period_days,
            "total_sessions": total_sessions,
            "sessions_per_week": round(total_sessions / weeks, 1),
            "total_minutes": round(total_minutes),
            "minutes_per_week": round(total_minutes / weeks),
            "total_trimp": round(total_trimp),
            "trimp_per_week": round(total_trimp / weeks),
            "sports": sports_seen,
            "detailed_context": context_text,
        }

    registry.register(Tool(
        name="analyze_training_load",
        description=(
            "Analyze training load over a period: total sessions, weekly averages, "
            "TRIMP, sport breakdown, and detailed trends. Use this before creating "
            "a plan or when the athlete asks about their training. "
            "Returns 'no_data' status if no activities exist."
        ),
        handler=analyze_training_load,
        parameters={
            "type": "object",
            "properties": {
                "period_days": {
                    "type": "integer",
                    "description": "Analysis period in days (default 28)",
                },
            },
        },
        category="analysis",
    ))

    def compare_plan_vs_actual() -> dict:
        """Compare planned vs actual training this week."""
        from src.tools.activity_context import match_plan_sessions
        from src.tools.activity_store import list_activities
        from pathlib import Path
        import json

        plans_dir = Path("data/plans")
        if not plans_dir.exists():
            return {"status": "no_plan", "message": "No active plan to compare against."}

        plan_files = sorted(plans_dir.glob("plan_*.json"), reverse=True)
        if not plan_files:
            return {"status": "no_plan", "message": "No active plan to compare against."}

        plan = json.loads(plan_files[0].read_text())
        activities = list_activities()

        if not activities:
            return {"status": "no_activities", "message": "No activities to compare."}

        # Use match_plan_sessions for compliance analysis
        compliance = match_plan_sessions(plan, activities)
        return compliance

    registry.register(Tool(
        name="compare_plan_vs_actual",
        description=(
            "Compare this week's planned training against actual activities. "
            "Shows which sessions were completed, missed, and overall compliance rate. "
            "Use this when the athlete asks about plan adherence or when assessing progress."
        ),
        handler=compare_plan_vs_actual,
        parameters={},
        category="analysis",
    ))
