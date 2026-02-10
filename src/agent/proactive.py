"""Proactive communication: triggers and messages for the athlete."""

from datetime import datetime, date


def check_proactive_triggers(
    athlete_profile: dict,
    activities: list[dict],
    episodes: list[dict],
    trajectory: dict,
) -> list[dict]:
    """Check if any proactive notifications should be sent.

    Returns a list of trigger dicts with type, priority, and data.
    """
    triggers = []

    # 1. Trajectory status check
    traj = trajectory.get("trajectory", {})
    confidence = trajectory.get("confidence", 0)

    if traj.get("on_track") is False:
        triggers.append({
            "type": "goal_at_risk",
            "priority": "high",
            "data": {
                "predicted_time": traj.get("predicted_race_time", "unknown"),
                "target_time": trajectory.get("goal", {}).get("target_time", "unknown"),
            },
        })
    elif traj.get("on_track") is True and confidence >= 0.5:
        triggers.append({
            "type": "on_track",
            "priority": "low",
            "data": {
                "predicted_time": traj.get("predicted_race_time", "unknown"),
                "confidence": confidence,
            },
        })

    # 2. Missed session patterns
    missed_patterns = _detect_missed_patterns(episodes)
    for pattern in missed_patterns:
        triggers.append({
            "type": "missed_session_pattern",
            "priority": "medium",
            "data": pattern,
        })

    # 3. Fitness improvement milestone
    if _detect_fitness_improvement(episodes):
        triggers.append({
            "type": "fitness_improving",
            "priority": "low",
            "data": {"trend": "improving"},
        })

    # 4. Upcoming milestone
    milestones = traj.get("key_milestones", [])
    upcoming = [m for m in milestones if m.get("status") in ("on_track", "at_risk")]
    if upcoming:
        triggers.append({
            "type": "milestone_approaching",
            "priority": "medium",
            "data": {"milestone": upcoming[0]},
        })

    # 5. High fatigue warning
    if _detect_high_fatigue(activities, episodes):
        triggers.append({
            "type": "fatigue_warning",
            "priority": "high",
            "data": {"message": "Recent sessions show signs of accumulated fatigue"},
        })

    return triggers


def format_proactive_message(trigger: dict, context: dict) -> str:
    """Format a proactive message for the user.

    Messages should be conversational, specific, and reference actual data.
    """
    msg_type = trigger.get("type", "")
    data = trigger.get("data", {})
    goal = context.get("goal", {})

    if msg_type == "goal_at_risk":
        return (
            f"Heads up: based on current training data, your predicted finish time "
            f"is {data.get('predicted_time', 'unclear')}, which may not meet your "
            f"target of {data.get('target_time', 'N/A')}. "
            f"Let's look at adjustments to get back on track."
        )

    if msg_type == "on_track":
        confidence_pct = int(data.get("confidence", 0) * 100)
        return (
            f"Looking good! Your current trajectory points to a finish time of "
            f"{data.get('predicted_time', 'TBD')} for your "
            f"{goal.get('event', 'race')} "
            f"(confidence: {confidence_pct}%). Keep up the consistent training."
        )

    if msg_type == "missed_session_pattern":
        day = data.get("day", "a certain day")
        count = data.get("missed_count", "multiple")
        return (
            f"I've noticed you've missed {count} sessions on {day} recently. "
            f"Want me to move that workout to a different day?"
        )

    if msg_type == "fitness_improving":
        return (
            "Great news: your fitness metrics are trending upward. "
            "Heart rate is improving at similar paces, indicating stronger aerobic capacity."
        )

    if msg_type == "milestone_approaching":
        ms = data.get("milestone", {})
        return (
            f"Milestone coming up: \"{ms.get('milestone', 'next goal')}\" "
            f"by {ms.get('date', 'soon')} — status: {ms.get('status', 'unknown')}."
        )

    if msg_type == "fatigue_warning":
        return (
            "Watch out: your recent training data shows signs of accumulated fatigue. "
            "Consider taking an extra rest day or reducing intensity this week."
        )

    return f"[{msg_type}] {data}"


def _detect_missed_patterns(episodes: list[dict]) -> list[dict]:
    """Detect recurring missed session patterns from episodes."""
    patterns = []

    # Look for "Thursday" or specific day mentions in patterns
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    all_patterns_text = " ".join(
        " ".join(ep.get("patterns_detected", []) + ep.get("lessons", []))
        for ep in episodes
    ).lower()

    for day in day_names:
        if day.lower() in all_patterns_text and ("miss" in all_patterns_text or "skip" in all_patterns_text):
            patterns.append({
                "day": day,
                "missed_count": "multiple",
                "source": "episode_patterns",
            })
            break  # One pattern per day check is enough

    return patterns


def _detect_fitness_improvement(episodes: list[dict]) -> bool:
    """Check if episodes indicate fitness improvement."""
    if not episodes:
        return False

    for ep in episodes:
        delta = ep.get("fitness_delta", {})
        trend = delta.get("weekly_volume_trend", "")
        vo2_change = delta.get("estimated_vo2max_change", "")
        if trend == "increasing" or (isinstance(vo2_change, str) and vo2_change.startswith("+")):
            return True

    return False


def _detect_high_fatigue(activities: list[dict], episodes: list[dict]) -> bool:
    """Detect signs of accumulated fatigue in recent data."""
    # Check if recent activities have elevated HR for easy efforts
    easy_activities = [
        a for a in activities[-5:] if a.get("hr_zone", 0) >= 3
        and a.get("heart_rate", {}).get("avg", 0) > 140
    ]

    # Multiple easy-effort activities with high HR suggests fatigue
    if len(easy_activities) >= 2:
        return True

    return False
