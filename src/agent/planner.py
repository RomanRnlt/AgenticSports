"""Adjusted plan generation: creates a new weekly plan informed by assessment."""

import json
import re

from google import genai

from src.agent.json_utils import extract_json
from src.agent.llm import MODEL, get_client
from src.agent.prompts import COACH_SYSTEM_PROMPT

ADJUSTED_PLAN_SYSTEM_PROMPT = COACH_SYSTEM_PROMPT + """

ADDITIONAL CONTEXT: You are generating an ADJUSTED plan based on an assessment of the athlete's recent training.
Incorporate the assessment's observations and recommended adjustments into the new plan.
If the assessment notes the athlete is running too fast on easy days, lower the target pace.
If sessions were missed, consider whether to compensate or simply continue the progression.
If fatigue is high, include more recovery.

In the JSON output, add a "adjustments_applied" field listing what was changed from the previous plan and why.
"""


def generate_adjusted_plan(
    profile: dict,
    previous_plan: dict,
    assessment: dict,
    relevant_episodes: list[dict] | None = None,
) -> dict:
    """Generate an adjusted 1-week plan based on assessment results.

    Args:
        profile: Athlete profile
        previous_plan: The plan that was assessed
        assessment: Assessment result from assess_training()
        relevant_episodes: Past episode reflections to inform planning

    Returns:
        New plan dict with adjustments_applied field
    """
    client = get_client()
    prompt = _build_adjusted_plan_prompt(profile, previous_plan, assessment, relevant_episodes)

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            genai.types.Content(
                role="user",
                parts=[genai.types.Part(text=prompt)],
            ),
        ],
        config=genai.types.GenerateContentConfig(
            system_instruction=ADJUSTED_PLAN_SYSTEM_PROMPT,
            temperature=0.7,
        ),
    )

    text = response.text.strip()
    return extract_json(text)


def _build_adjusted_plan_prompt(
    profile: dict,
    previous_plan: dict,
    assessment: dict,
    relevant_episodes: list[dict] | None = None,
) -> str:
    """Build prompt for adjusted plan generation."""
    goal = profile.get("goal", {})
    constraints = profile.get("constraints", {})
    fitness = profile.get("fitness", {})
    sports = profile.get("sports", [])

    # Assessment summary
    assess = assessment.get("assessment", {})
    adjustments = assessment.get("recommended_adjustments", [])

    observations = assess.get("observations", [])
    obs_text = "\n".join(f"  - {o}" for o in observations) if observations else "  None"

    adj_text = ""
    for a in adjustments:
        adj_text += f"  - [{a.get('impact', '?')}] {a.get('description', '?')}\n"
    if not adj_text:
        adj_text = "  None\n"

    fitness_info = ""
    if fitness.get("estimated_vo2max"):
        fitness_info += f"- Estimated VO2max: {fitness['estimated_vo2max']}\n"
    if fitness.get("threshold_pace_min_km"):
        fitness_info += f"- Threshold pace: {fitness['threshold_pace_min_km']} min/km\n"
    if fitness.get("weekly_volume_km"):
        fitness_info += f"- Current weekly volume: {fitness['weekly_volume_km']} km\n"
    if not fitness_info:
        fitness_info = "- No fitness data available (assume beginner/unknown level)\n"

    return f"""\
Create an ADJUSTED 1-week training plan based on the following assessment:

ATHLETE:
Sports: {', '.join(sports)}
Goal: {goal.get('event', 'General fitness')} by {goal.get('target_date', 'Not set')}
Target time: {goal.get('target_time', 'Not set')}

Fitness:
{fitness_info}
Constraints:
- Training days: {constraints.get('training_days_per_week', 5)}/week
- Max session: {constraints.get('max_session_minutes', 90)} minutes

ASSESSMENT OF LAST WEEK:
- Compliance: {assess.get('compliance', 'N/A')}
- Fitness trend: {assess.get('fitness_trend', 'N/A')}
- Fatigue level: {assess.get('fatigue_level', 'N/A')}
- Injury risk: {assess.get('injury_risk', 'N/A')}
- Observations:
{obs_text}

RECOMMENDED ADJUSTMENTS:
{adj_text}
{_format_episodes_section(relevant_episodes)}
Incorporate these adjustments into the new plan. Include an "adjustments_applied" array in the output listing what you changed and why.
"""


def _format_episodes_section(episodes: list[dict] | None) -> str:
    """Format past episodes as a lessons section for the prompt."""
    if not episodes:
        return ""

    lines = ["LESSONS FROM PAST TRAINING BLOCKS:"]
    for ep in episodes:
        block = ep.get("block", "?")
        lines.append(f"\n  [{block}]")
        for lesson in ep.get("lessons", []):
            lines.append(f"  - {lesson}")
        for pattern in ep.get("patterns_detected", []):
            lines.append(f"  - Pattern: {pattern}")

    return "\n".join(lines) + "\n"
