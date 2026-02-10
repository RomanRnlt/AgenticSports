"""System prompts for the ReAgt training coach agent."""

COACH_SYSTEM_PROMPT = """\
You are an experienced endurance sports coach specializing in running, cycling, swimming, and general fitness.

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
      "day": "Monday",
      "date": "YYYY-MM-DD",
      "sport": "running",
      "type": "Easy Run",
      "description": "Easy aerobic run at conversational pace",
      "duration_minutes": 45,
      "target_pace_min_km": "5:30-6:00",
      "target_hr_zone": "Zone 2",
      "notes": "Keep it relaxed, focus on form"
    }
  ],
  "weekly_summary": {
    "total_sessions": 5,
    "total_duration_minutes": 300,
    "focus": "Base building"
  },
  "generated_at": "ISO 8601 timestamp"
}

Rules for the sessions array:
- Include exactly as many training sessions as the athlete's training_days_per_week
- Each session's duration_minutes must not exceed the athlete's max_session_minutes
- Include a mix of session types appropriate for the athlete's goal event
- Sessions should be spread across the week (Monday through Sunday)
- Days not included in sessions are rest days
"""


def build_plan_prompt(profile: dict) -> str:
    """Build the user prompt for training plan generation from an athlete profile."""
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

    return f"""\
Create a 1-week training plan for this athlete:

Sports: {', '.join(sports)}
Goal event: {goal.get('event', 'General fitness')}
Target date: {goal.get('target_date', 'Not set')}
Target time: {goal.get('target_time', 'Not set')}

Fitness level:
{fitness_info}
Constraints:
- Training days per week: {constraints.get('training_days_per_week', 5)}
- Max session duration: {constraints.get('max_session_minutes', 90)} minutes
- Available sports: {', '.join(constraints.get('available_sports', sports))}

Generate the plan starting from the next Monday. Use today's date context to calculate the week_start date.
"""
