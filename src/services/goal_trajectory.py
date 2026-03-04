"""Goal trajectory analysis — LLM-based progress assessment toward goals.

Uses the LLM to analyze whether an athlete is on track, ahead, behind,
or at risk for reaching their stated goal, given their training history,
health trends, periodization phase, and beliefs.

The LLM decides what "on track" means per sport — no hardcoded rules.

Usage::

    from src.services.goal_trajectory import analyze_trajectory

    result = analyze_trajectory(
        goal={"event": "Berlin Marathon", "target_date": "2026-09-27"},
        profile={"sports": ["running"], "fitness": {"weekly_volume_km": 45}},
        training_summary={"total_sessions": 12, "total_distance_km": 150},
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.agent.json_utils import extract_json
from src.agent.llm import chat_completion

logger = logging.getLogger(__name__)

# LLM model for trajectory analysis (same as episode_consolidation).
_TRAJECTORY_MODEL = "gemini/gemini-2.5-flash"

_TRAJECTORY_PROMPT = (
    "You are a sports coaching AI analyzing an athlete's goal trajectory.\n\n"
    "Given the athlete's goal, profile, training data, and context, assess whether "
    "they are on track to achieve their goal.\n\n"
    "GOAL:\n{goal}\n\n"
    "ATHLETE PROFILE:\n{profile}\n\n"
    "{optional_sections}"
    "Analyze the trajectory and return valid JSON only:\n"
    '{{\n'
    '  "trajectory_status": "on_track|ahead|behind|at_risk|insufficient_data",\n'
    '  "confidence": 0.0-1.0,\n'
    '  "projected_outcome": "Brief projection of likely outcome",\n'
    '  "analysis": "2-4 sentence analysis of current trajectory",\n'
    '  "key_factors": ["factor1", "factor2"],\n'
    '  "risk_factors": ["risk1", "risk2"],\n'
    '  "recommendations": ["recommendation1", "recommendation2"]\n'
    '}}'
)


@dataclass(frozen=True)
class TrajectoryResult:
    """Immutable result of a goal trajectory analysis."""

    trajectory_status: str      # on_track, ahead, behind, at_risk, insufficient_data
    confidence: float           # 0.0-1.0
    projected_outcome: str
    analysis: str
    key_factors: list[str]
    risk_factors: list[str]
    recommendations: list[str]


def _insufficient_data_result() -> TrajectoryResult:
    """Return a safe fallback result when analysis cannot be performed."""
    return TrajectoryResult(
        trajectory_status="insufficient_data",
        confidence=0.0,
        projected_outcome="Unable to project — insufficient data.",
        analysis="Not enough information available to assess trajectory.",
        key_factors=[],
        risk_factors=[],
        recommendations=["Continue training and logging activities."],
    )


def _build_optional_sections(
    training_summary: dict | None,
    health_trends: dict | None,
    periodization_phase: str | None,
    beliefs: list[dict] | None,
    previous_trajectory: dict | None,
) -> str:
    """Build optional prompt sections from available context."""
    sections: list[str] = []

    if training_summary:
        sections.append(f"TRAINING SUMMARY (last 28 days):\n{_format_dict(training_summary)}\n")

    if health_trends:
        sections.append(f"HEALTH TRENDS:\n{_format_dict(health_trends)}\n")

    if periodization_phase:
        sections.append(f"CURRENT PERIODIZATION PHASE: {periodization_phase}\n")

    if beliefs:
        belief_texts = [b.get("text", "") for b in beliefs[:10] if b.get("text")]
        if belief_texts:
            sections.append(f"KNOWN ATHLETE PATTERNS:\n" + "\n".join(f"- {t}" for t in belief_texts) + "\n")

    if previous_trajectory:
        prev_status = previous_trajectory.get("trajectory_status", "unknown")
        prev_analysis = previous_trajectory.get("analysis", "")
        sections.append(
            f"PREVIOUS TRAJECTORY ASSESSMENT:\n"
            f"Status: {prev_status}\n"
            f"Analysis: {prev_analysis}\n"
        )

    return "\n".join(sections) + "\n" if sections else ""


def _format_dict(d: dict) -> str:
    """Format a dict as indented key-value lines for the prompt."""
    lines = []
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"  {key}:")
            for k2, v2 in value.items():
                lines.append(f"    {k2}: {v2}")
        elif isinstance(value, list):
            lines.append(f"  {key}: {value}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def analyze_trajectory(
    goal: dict,
    profile: dict,
    training_summary: dict | None = None,
    health_trends: dict | None = None,
    periodization_phase: str | None = None,
    beliefs: list[dict] | None = None,
    previous_trajectory: dict | None = None,
) -> TrajectoryResult:
    """Analyze goal trajectory using the LLM.

    This is a synchronous function (not async). It calls chat_completion
    directly and parses the JSON response.

    Args:
        goal: Athlete's goal dict (event, target_date, target_time, etc.).
        profile: Athlete profile dict (sports, fitness, constraints).
        training_summary: Summary of recent training (sessions, volume, etc.).
        health_trends: Health trend data (sleep, HRV, stress trends).
        periodization_phase: Current phase name (base, build, peak, taper).
        beliefs: List of belief dicts from user model.
        previous_trajectory: Previous trajectory snapshot for comparison.

    Returns:
        TrajectoryResult with the LLM's assessment.
    """
    if not goal:
        return _insufficient_data_result()

    optional_sections = _build_optional_sections(
        training_summary, health_trends, periodization_phase,
        beliefs, previous_trajectory,
    )

    prompt = _TRAJECTORY_PROMPT.format(
        goal=_format_dict(goal),
        profile=_format_dict(profile),
        optional_sections=optional_sections,
    )

    try:
        response = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            model=_TRAJECTORY_MODEL,
        )

        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            return _insufficient_data_result()

        parsed = extract_json(raw)

        return TrajectoryResult(
            trajectory_status=str(parsed.get("trajectory_status", "insufficient_data")),
            confidence=float(parsed.get("confidence", 0.0)),
            projected_outcome=str(parsed.get("projected_outcome", "")),
            analysis=str(parsed.get("analysis", "")),
            key_factors=list(parsed.get("key_factors", [])),
            risk_factors=list(parsed.get("risk_factors", [])),
            recommendations=list(parsed.get("recommendations", [])),
        )

    except Exception:
        logger.warning("Goal trajectory analysis failed", exc_info=True)
        return _insufficient_data_result()
