"""Episodic memory: generate, store, and retrieve training reflections."""

import json
import re
from datetime import datetime
from pathlib import Path

from google import genai

from src.agent.json_utils import extract_json
from src.agent.llm import MODEL, get_client

DATA_DIR = Path(__file__).parent.parent.parent / "data"
EPISODES_DIR = DATA_DIR / "episodes"

REFLECTION_SYSTEM_PROMPT = """\
You are an expert endurance sports coach reflecting on a completed training block.

Analyze the prescribed plan, actual activities, and assessment to generate a structured reflection.
Focus on actionable lessons, patterns, and specific observations grounded in data.

You MUST respond with ONLY a valid JSON object. No markdown, no explanation, no code fences.

The JSON must follow this exact structure:
{
    "key_observations": [
        "Specific data-driven observation (reference actual HR, pace, duration numbers)"
    ],
    "lessons": [
        "Actionable lesson that should inform future planning"
    ],
    "patterns_detected": [
        "Recurring pattern observed across sessions or weeks"
    ],
    "fitness_delta": {
        "estimated_vo2max_change": "+0.5 or -0.3 or stable",
        "threshold_pace_change": "-0:05/km or +0:03/km or stable",
        "weekly_volume_trend": "increasing|stable|decreasing"
    },
    "confidence": 0.8
}

Rules:
- key_observations must reference actual numbers from the data (HR values, pace, distances)
- lessons must be actionable (e.g., "move Thursday interval to Wednesday" not "train smarter")
- patterns_detected should identify recurring behaviors across the training block
- confidence is 0.0-1.0 based on how much data was available
- Be specific, not generic. Reference actual session data.
"""


def generate_reflection(
    plan: dict,
    activities: list[dict],
    assessment: dict,
    athlete_profile: dict,
) -> dict:
    """Generate a structured reflection for a completed training block.

    Sends plan + activities + assessment to Gemini.
    Returns a complete episode dict ready for storage.
    """
    client = get_client()
    prompt = _build_reflection_prompt(plan, activities, assessment, athlete_profile)

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            genai.types.Content(
                role="user",
                parts=[genai.types.Part(text=prompt)],
            ),
        ],
        config=genai.types.GenerateContentConfig(
            system_instruction=REFLECTION_SYSTEM_PROMPT,
            temperature=0.4,
        ),
    )

    text = response.text.strip()
    reflection = extract_json(text)

    # Build complete episode
    week_start = plan.get("week_start", "unknown")
    week_num = plan.get("week_number", "?")

    total_prescribed_km = 0
    for s in plan.get("sessions", []):
        # Rough estimate: duration * typical pace
        dur = s.get("duration_minutes", 0)
        total_prescribed_km += dur / 6  # ~6 min/km average

    total_actual_km = sum(
        (a.get("distance_meters", 0) or 0) / 1000 for a in activities
    )

    episode = {
        "id": f"ep_{week_start}",
        "block": f"2026-W{week_num:02d}" if isinstance(week_num, int) else f"week-{week_num}",
        "period": week_start,
        "prescribed_sessions": len(plan.get("sessions", [])),
        "actual_sessions": len(activities),
        "prescribed_volume_km": round(total_prescribed_km, 1),
        "actual_volume_km": round(total_actual_km, 1),
        "compliance_rate": assessment.get("assessment", {}).get("compliance", 0),
        "key_observations": reflection.get("key_observations", []),
        "lessons": reflection.get("lessons", []),
        "patterns_detected": reflection.get("patterns_detected", []),
        "fitness_delta": reflection.get("fitness_delta", {}),
        "confidence": reflection.get("confidence", 0.5),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    return episode


def store_episode(episode: dict, storage_dir: str | Path | None = None) -> Path:
    """Store a reflection episode as JSON. Returns the file path."""
    dest = Path(storage_dir) if storage_dir else EPISODES_DIR
    dest.mkdir(parents=True, exist_ok=True)

    ep_id = episode.get("id", f"ep_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    filename = f"{ep_id}.json"
    path = dest / filename
    path.write_text(json.dumps(episode, indent=2))
    return path


def list_episodes(
    storage_dir: str | Path | None = None,
    limit: int = 10,
) -> list[dict]:
    """List stored episodes, most recent first."""
    src = Path(storage_dir) if storage_dir else EPISODES_DIR
    if not src.exists():
        return []

    episodes = []
    for path in sorted(src.glob("ep_*.json"), reverse=True):
        episodes.append(json.loads(path.read_text()))
        if len(episodes) >= limit:
            break

    return episodes


def retrieve_relevant_episodes(
    current_context: dict,
    episodes: list[dict],
    max_results: int = 5,
) -> list[dict]:
    """Retrieve episodes relevant to current planning context.

    Uses keyword matching + recency bias for MVP.
    Scores each episode based on keyword overlap with current context.
    """
    if not episodes:
        return []

    # Extract keywords from context
    context_keywords = _extract_keywords(current_context)

    scored = []
    for i, ep in enumerate(episodes):
        score = 0.0

        # Keyword matching across observations, lessons, patterns
        ep_text = " ".join(
            ep.get("key_observations", [])
            + ep.get("lessons", [])
            + ep.get("patterns_detected", [])
        ).lower()

        for kw in context_keywords:
            if kw in ep_text:
                score += 1.0

        # Recency bias: more recent episodes score higher
        # episodes are already sorted most-recent-first
        recency_bonus = max(0, (len(episodes) - i)) / len(episodes)
        score += recency_bonus * 2.0

        scored.append((score, ep))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    return [ep for _, ep in scored[:max_results]]


def _extract_keywords(context: dict) -> list[str]:
    """Extract search keywords from a planning context."""
    keywords = []

    # From goal
    goal = context.get("goal", {})
    if goal.get("event"):
        keywords.extend(goal["event"].lower().split())

    # From sports
    for sport in context.get("sports", []):
        keywords.append(sport.lower())

    # From fitness
    fitness = context.get("fitness", {})
    if fitness.get("trend"):
        keywords.append(fitness["trend"].lower())

    # Common training keywords
    keywords.extend(["pace", "hr", "fatigue", "volume", "easy", "interval", "long run", "threshold"])

    return keywords


def _build_reflection_prompt(
    plan: dict, activities: list[dict], assessment: dict, profile: dict
) -> str:
    """Build the prompt for reflection generation."""
    goal = profile.get("goal", {})
    assess = assessment.get("assessment", {})

    # Prescribed sessions
    prescribed = plan.get("sessions", [])
    prescribed_text = []
    for s in prescribed:
        prescribed_text.append(
            f"  - {s.get('day', '?')}: {s.get('type', '?')} "
            f"({s.get('duration_minutes', '?')}min, {s.get('target_hr_zone', '?')})"
        )

    # Actual activities
    actual_text = []
    for a in activities:
        hr = a.get("heart_rate", {})
        dur_min = round(a["duration_seconds"] / 60) if a.get("duration_seconds") else "?"
        dist_km = round(a["distance_meters"] / 1000, 1) if a.get("distance_meters") else "N/A"
        avg_hr = hr.get("avg", "N/A") if hr else "N/A"

        actual_text.append(
            f"  - {a.get('start_time', '?')}: {a.get('sport', '?')} | "
            f"{dur_min}min | {dist_km}km | HR {avg_hr}"
        )

    # Assessment summary
    obs = assess.get("observations", [])
    obs_text = "\n".join(f"  - {o}" for o in obs) if obs else "  None"

    return f"""\
Reflect on this completed training block:

GOAL: {goal.get('event', 'General')} by {goal.get('target_date', 'N/A')}

PRESCRIBED PLAN ({len(prescribed)} sessions):
{chr(10).join(prescribed_text)}

ACTUAL TRAINING ({len(activities)} activities):
{chr(10).join(actual_text)}

ASSESSMENT:
- Compliance: {assess.get('compliance', 'N/A')}
- Fitness trend: {assess.get('fitness_trend', 'N/A')}
- Fatigue: {assess.get('fatigue_level', 'N/A')}
- Observations:
{obs_text}

Generate specific, data-driven observations, actionable lessons, and patterns detected.
"""
