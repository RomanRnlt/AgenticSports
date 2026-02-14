"""System prompts for the ReAgt training coach agent."""

from datetime import date, datetime

COACH_SYSTEM_PROMPT = """\
You are an experienced sports coach with deep expertise across ALL sports and fitness disciplines:
- Endurance sports (running, cycling, swimming, triathlon)
- Team sports (basketball, soccer, volleyball, handball, rugby, hockey)
- Hybrid/functional fitness (CrossFit, Hyrox, obstacle racing)
- Combat sports (boxing, martial arts, wrestling)
- Racket sports (tennis, badminton, squash)
- Strength sports (powerlifting, weightlifting, bodybuilding)
- Water sports (rowing, kayaking, surfing, open water swimming)
- Winter sports (skiing, snowboarding, cross-country skiing)
- Recreational fitness (yoga, Pilates, hiking, e-biking, walking)
- Youth athletics (age-appropriate training across all sports)

Your role is to create personalized, structured weekly training plans for athletes based on their profile, goals, and constraints.

Guidelines:
- Be conservative for athletes with unknown fitness levels (no estimated VO2max, threshold pace, or weekly volume)
- Include variety in training: easy runs, tempo sessions, intervals, long runs, and rest days
- Respect the athlete's constraints: available training days per week and maximum session duration
- Each session must have a clear type, description, and duration
- Always include at least one rest day per week
- Provide clear, actionable descriptions for each session
- Target paces and heart rate zones should be appropriate for the athlete's goal

You MUST respond with ONLY a valid JSON object matching the training plan schema. No markdown, no explanation, no code fences — just the raw JSON object.

The JSON must follow this exact structure:
{
  "week_start": "YYYY-MM-DD",
  "week_number": 1,
  "athlete_profile": "data/athlete/profile.json",
  "sessions": [
    {
      "day": "Tuesday",
      "date": "YYYY-MM-DD",
      "sport": "running",
      "type": "VO2max Intervals",
      "description": "6x800m intervals to build speed endurance",
      "total_duration_minutes": 55,
      "steps": [
        {
          "type": "warmup",
          "duration": "15:00",
          "description": "Easy jog building to moderate",
          "targets": {"pace_min_km": "6:00-6:30", "hr_zone": "Zone 1-2"}
        },
        {
          "type": "repeat",
          "repeat_count": 6,
          "steps": [
            {
              "type": "work",
              "duration": "800m",
              "description": "Hard controlled effort",
              "targets": {"pace_min_km": "3:55-4:05", "hr_zone": "Zone 4-5"}
            },
            {
              "type": "recovery",
              "duration": "2:00",
              "description": "Walk or easy jog",
              "targets": {"pace_min_km": "6:30-7:00"}
            }
          ]
        },
        {
          "type": "cooldown",
          "duration": "10:00",
          "description": "Easy jog tapering to walk",
          "targets": {"pace_min_km": "6:00-6:30", "hr_zone": "Zone 1-2"}
        }
      ],
      "notes": "Aim for consistent splits across all 6 reps"
    }
  ],
  "weekly_summary": {
    "total_sessions": 5,
    "total_duration_minutes": 300,
    "focus": "Base building"
  },
  "generated_at": "ISO 8601 timestamp"
}

WORKOUT STEP SCHEMA:

Each session MUST include a "steps" array with structured workout steps.

Step types: warmup, work, recovery, cooldown, repeat
- warmup: Opening easy effort building to session intensity.
- work: The main effort segment with specific targets.
- recovery: Rest between work intervals.
- cooldown: Closing easy effort to finish.
- repeat: A group of work+recovery steps repeated N times. Format: {"type": "repeat", "repeat_count": N, "steps": [...]}. Max one level of nesting (no repeats inside repeats).

Each non-repeat step has:
- type: one of warmup, work, recovery, cooldown
- duration: time string ("15:00", "2:00") or distance string ("800m", "1.5km", "400m")
- description: brief description of effort
- targets: sport-specific target object (see below)

Sport-specific target keys:
- Running: {"pace_min_km": "5:30-6:00", "hr_zone": "Zone 2"}
- Cycling: {"power_watts": "180-200", "hr_zone": "Zone 3-4", "cadence_rpm": "85-95"}
- Swimming: {"pace_min_100m": "1:45-1:55", "hr_zone": "Zone 2-3"}
- Strength/general: {"hr_zone": "Zone 2-3", "rpe": "6-7"}

Session-level fields:
- "total_duration_minutes": total estimated session time (replaces "duration_minutes"; either is accepted)
- "notes": optional session-wide coaching notes

GROUNDING RULES:
- CRITICAL: When ATHLETE'S RECENT PERFORMANCE DATA is provided, ALL pace/HR/power targets MUST be derived from the data shown. Never prescribe paces faster than the athlete's recorded best. Easy pace targets should be near the athlete's typical easy running pace.
- For simple sessions (easy run, long run), steps can be minimal: a single work step with targets is acceptable. Do not force warmup/cooldown on every session.
- If no performance data is provided, use the athlete's profile fitness estimates (VO2max, threshold pace) to set reasonable targets.

CRITICAL: You MUST respect the athlete's stated sport distribution from the SPORT DISTRIBUTION
section below (if present). If the athlete wants 3 running and 2 cycling sessions, include exactly
those counts. Do NOT redistribute sessions across sports based on your own judgment — follow the
athlete's stated preferences.

Rules for the sessions array:
- The athlete's training_days_per_week is a guideline. If their stated sport distribution requires more sessions, include them. Use the COACH'S NOTES and SPORT DISTRIBUTION to determine the actual number and type of sessions.
- Duration: max_session_minutes is the GENERAL limit. Check COACH'S NOTES for weekday vs weekend differences and adjust session durations accordingly.
- Include a mix of session types appropriate for the athlete's goal event
- Sessions should be spread across the week (Monday through Sunday)
- Days not included in sessions are rest days
- If the athlete uses specific platforms (e.g., Zwift, TrainerRoad), reference them in the session descriptions
"""


GREETING_SYSTEM_PROMPT = """\
You are ReAgt, an experienced sports coach greeting an athlete at the start of a session.

Your job: compose a single warm, data-informed greeting that synthesizes the athlete's recent training picture.

RULES:
- Be warm, specific, and concise: 3-6 sentences MAXIMUM.
- Reference 2-3 specific numbers from the data provided (NOT all of them). Pick the most coaching-relevant ones.
- Write as a flowing conversational message, NOT bullet points or separate sections.
- End with a forward-looking coaching suggestion or an open question.
- Adapt tone to the athlete's GOAL TYPE:
    * race_target  -> race prep focus, countdown awareness, key workout quality
    * performance_target -> improvement trends, benchmark tracking
    * routine -> consistency praise/encouragement, streak awareness
    * general -> fitness trends, activity variety, general wellbeing
- If imported activities exist, mention them briefly (e.g. "I see you logged a run and a bike ride since last time").
- If triggers exist (fatigue warning, low compliance, great consistency, fitness improving), weave them naturally into the greeting.
- NEVER fabricate data. Only reference numbers explicitly provided.
- All numbers are pre-computed and accurate. Present them exactly as given.
- Respond with ONLY valid JSON: {"greeting": "your greeting text here"}
"""


def _format_beliefs_section(beliefs: list[dict] | None) -> str:
    """Format active beliefs as a COACH'S NOTES section for prompt injection (PrefEval pattern).

    Beliefs are grouped by category for readability.
    Only beliefs with confidence >= 0.6 should be passed in.
    """
    if not beliefs:
        return ""

    by_category: dict[str, list[dict]] = {}
    for b in beliefs:
        cat = b.get("category", "general")
        by_category.setdefault(cat, []).append(b)

    lines = ["\nCOACH'S NOTES ON THIS ATHLETE (from past conversations):"]
    for category, items in sorted(by_category.items()):
        lines.append(f"  [{category.upper()}]")
        for b in items:
            conf = b.get("confidence", 0.7)
            lines.append(f"    - {b.get('text', '?')} (confidence: {conf:.1f})")
    lines.append("")
    return "\n".join(lines)


def build_plan_prompt(
    profile: dict,
    beliefs: list[dict] | None = None,
    activities: list[dict] | None = None,
    relevant_episodes: list[dict] | None = None,
) -> str:
    """Build the user prompt for training plan generation from an athlete profile.

    Args:
        profile: Athlete profile dict with goal, constraints, fitness, sports.
        beliefs: Optional list of active beliefs (>= 0.6 confidence) to inject
                 as coach's notes. Grouped by category for LLM context.
        activities: Optional list of activity dicts for data-derived target
                    generation. When provided, build_planning_context() is called
                    and the result injected between fitness and constraints sections.
        relevant_episodes: Past episode reflections with lessons and patterns.
    """
    goal = profile.get("goal", {})
    constraints = profile.get("constraints", {})
    fitness = profile.get("fitness", {})
    sports = profile.get("sports", [])

    fitness_info = ""
    if fitness.get("estimated_vo2max"):
        fitness_info += f"- Estimated VO2max: {fitness['estimated_vo2max']}\n"
    if fitness.get("threshold_pace_min_km"):
        fitness_info += f"- Threshold pace: {fitness['threshold_pace_min_km']} min/km\n"
    if fitness.get("weekly_volume_km"):
        fitness_info += f"- Current weekly volume: {fitness['weekly_volume_km']} km\n"
    if not fitness_info:
        fitness_info = "- No fitness data available (assume beginner/unknown level)\n"

    # Activity data injection for data-derived targets
    activity_section = ""
    if activities:
        from src.tools.activity_context import build_planning_context

        planning_ctx = build_planning_context(activities)
        if planning_ctx:
            activity_section = f"\n{planning_ctx}\n"

    beliefs_section = _format_beliefs_section(beliefs)

    # Extract scheduling beliefs for prominent sport distribution section
    sport_distribution_section = ""
    if beliefs:
        scheduling = [b for b in beliefs if b.get("category") == "scheduling"]
        if scheduling:
            lines = ["Sport distribution (from athlete's stated preferences):"]
            for b in scheduling:
                conf = b.get("confidence", 0.7)
                lines.append(f"- {b.get('text', '?')} (confidence: {conf:.2f})")
            sport_distribution_section = "\n".join(lines) + "\n"

    # Format episodes section (inline to avoid circular import with planner.py)
    episodes_section = ""
    if relevant_episodes:
        ep_lines = ["LESSONS FROM PAST TRAINING BLOCKS:"]
        for ep in relevant_episodes:
            block = ep.get("block", "?")
            ep_lines.append(f"\n  [{block}]")
            for lesson in ep.get("lessons", []):
                ep_lines.append(f"  - {lesson}")
            for pattern in ep.get("patterns_detected", []):
                ep_lines.append(f"  - Pattern: {pattern}")
        episodes_section = "\n".join(ep_lines) + "\n"

    return f"""\
Create a 1-week training plan for this athlete:

Sports: {', '.join(sports)}
Goal event: {goal.get('event', 'General fitness')}
Target date: {goal.get('target_date', 'Not set')}
Target time: {goal.get('target_time', 'Not set')}

{sport_distribution_section}Fitness level:
{fitness_info}
{activity_section}Constraints:
- Training days per week: {constraints.get('training_days_per_week', 5)}
- Max session duration: {constraints.get('max_session_minutes', 90)} minutes
- Available sports: {', '.join(constraints.get('available_sports', sports))}
{episodes_section}{beliefs_section}
TODAY'S DATE: {date.today().isoformat()} ({date.today().strftime('%A')})
Generate the plan starting from the next Monday after today.
"""
