"""Acceptance criteria for ReAgt coaching agent.

These criteria define what "excellent" looks like. Each criterion is a function
that takes test results and returns (passed: bool, detail: str).
"""

from datetime import date, datetime


def plan_respects_sport_distribution(plan: dict, beliefs: list[dict]) -> tuple[bool, str]:
    """Plan must include sessions matching the user's stated sport distribution.

    If user said "3x running, 1x swimming, 2-3x cycling, gym",
    the plan must reflect this, not a generic split.
    """
    sessions = plan.get("sessions", [])
    sports_in_plan = {}
    for s in sessions:
        sport = s.get("sport", "unknown").lower()
        sports_in_plan[sport] = sports_in_plan.get(sport, 0) + 1

    # Check if beliefs mention specific session counts
    distribution_beliefs = [
        b for b in beliefs
        if any(kw in b.get("text", "").lower() for kw in ["session", "mal", "times", "per week"])
    ]

    issues = []

    # Must have at least 2 different sports IF user trains multi-sport (check beliefs)
    sport_count = len([s for s in sports_in_plan if s not in ("rest", "recovery", "general fitness")])
    multi_sport_indicators = [
        b for b in beliefs
        if any(kw in b.get("text", "").lower() for kw in ["cycling", "swimming", "swim", "rad", "bike", "triathlon"])
    ]
    if multi_sport_indicators and sport_count < 2:
        issues.append(f"Only {sport_count} sport types in plan, expected multi-sport")

    # If user mentioned cycling/ZWIFT, must have cycling sessions
    # Count all cycling variants (cycling, indoor_cycling, road_cycling, zwift, etc.)
    cycling_count = sum(v for k, v in sports_in_plan.items() if "cycling" in k or "bike" in k or "zwift" in k or "rad" in k)
    cycling_beliefs = [b for b in beliefs if "cycling" in b.get("text", "").lower() or "zwift" in b.get("text", "").lower()]
    if cycling_beliefs and cycling_count < 2:
        issues.append(f"Only {cycling_count} cycling sessions but user wants 2-3")

    # If user mentioned swimming, must have swimming
    swim_beliefs = [b for b in beliefs if "swim" in b.get("text", "").lower()]
    if swim_beliefs and sports_in_plan.get("swimming", 0) < 1:
        issues.append(f"No swimming sessions but user swims weekly")

    # If user mentioned gym, must have gym/strength sessions
    # Count all gym/strength variants (gym, strength, strength_training, weight_training, etc.)
    gym_count = sum(v for k, v in sports_in_plan.items() if "gym" in k or "strength" in k or "weight" in k or "krafttraining" in k)
    gym_beliefs = [b for b in beliefs if "gym" in b.get("text", "").lower() or "strength" in b.get("text", "").lower() or "muscle" in b.get("text", "").lower()]
    if gym_beliefs and gym_count < 1:
        issues.append(f"No gym/strength sessions but user goes to gym")

    passed = len(issues) == 0
    detail = f"Sports: {sports_in_plan}" + (f" | Issues: {'; '.join(issues)}" if issues else " | OK")
    return passed, detail


def plan_dates_are_valid(plan: dict) -> tuple[bool, str]:
    """Plan dates must be in the future and use ISO format."""
    issues = []
    today = date.today()

    week_start = plan.get("week_start", "")
    try:
        ws = date.fromisoformat(week_start)
        if ws.year < 2026:
            issues.append(f"week_start {week_start} is in the past (year < 2026)")
    except ValueError:
        issues.append(f"week_start '{week_start}' is not valid ISO date")

    for s in plan.get("sessions", []):
        d = s.get("date", "")
        try:
            sd = date.fromisoformat(d)
            if sd.year < 2026:
                issues.append(f"Session date {d} has wrong year")
        except ValueError:
            issues.append(f"Session date '{d}' is not valid ISO")

    passed = len(issues) == 0
    return passed, "; ".join(issues) if issues else "All dates valid"


def target_date_is_iso(structured_core: dict) -> tuple[bool, str]:
    """Goal target_date must be stored in ISO format (YYYY-MM-DD), not 'August'."""
    td = structured_core.get("goal", {}).get("target_date", "")
    if not td:
        return False, "target_date is empty"

    try:
        date.fromisoformat(td)
        return True, f"target_date={td} (valid ISO)"
    except (ValueError, TypeError):
        return False, f"target_date='{td}' is NOT ISO format"


def fitness_fields_populated(structured_core: dict, beliefs: list[dict] | None = None) -> tuple[bool, str]:
    """After providing performance data, fitness fields should be estimated.

    VO2max should always be derived when any performance data is given.
    Threshold pace is only expected if the athlete has running performance data.
    """
    beliefs = beliefs or []
    fitness = structured_core.get("fitness", {})
    sports = structured_core.get("sports", [])
    vo2 = fitness.get("estimated_vo2max")
    threshold = fitness.get("threshold_pace_min_km")

    issues = []
    if vo2 is None:
        issues.append("estimated_vo2max is None")

    # Only require threshold pace if athlete has running performance data in beliefs
    # (Don't require it for cyclists who also run but haven't provided running times)
    # Be careful not to match "Radmarathon" as running data
    if threshold is None and beliefs:
        running_perf_keywords = ["10km", "5km", "10k ", "5k ", "half marathon", "halbmarathon", "laufzeit", "run time", "bestzeit", "lauf bestzeit"]
        has_running_performance = any(
            any(kw in b.get("text", "").lower() for kw in running_perf_keywords)
            and "rad" not in b.get("text", "").lower()  # exclude cycling "Radmarathon"
            for b in beliefs
        )
        if has_running_performance:
            issues.append("threshold_pace_min_km is None (has running performance data)")

    passed = len(issues) == 0
    detail = f"VO2max={vo2}, threshold={threshold}"
    if issues:
        detail += f" | Missing: {'; '.join(issues)}"
    return passed, detail


def ongoing_session_answers_questions(response: str, question_keywords: list[str]) -> tuple[bool, str]:
    """Agent must actually answer the user's question, not deflect to asking for more info.

    If user asks about HR zones, the response should contain HR zone information.
    If user asks about ZWIFT workouts, the response should discuss ZWIFT.
    """
    response_lower = response.lower()

    addressed = []
    missed = []
    for kw in question_keywords:
        if kw.lower() in response_lower:
            addressed.append(kw)
        else:
            missed.append(kw)

    # Check for deflection patterns
    deflection_phrases = [
        "sobald ich das habe",
        "brauche ich noch",
        "fehlt mir noch",
        "when i have",
        "once you provide",
        "i still need",
    ]
    is_deflecting = any(p in response_lower for p in deflection_phrases)

    passed = len(missed) <= len(question_keywords) // 2 and not is_deflecting
    detail = f"Addressed: {addressed}, Missed: {missed}, Deflecting: {is_deflecting}"
    return passed, detail


def beliefs_have_correct_categories(beliefs: list[dict]) -> tuple[bool, str]:
    """Beliefs should use appropriate categories, not all 'preference'."""
    categories = [b.get("category", "unknown") for b in beliefs]
    unique = set(categories)

    # Should have at least 3 different categories for a rich user profile
    preference_ratio = categories.count("preference") / len(categories) if categories else 0

    issues = []
    if len(unique) < 3:
        issues.append(f"Only {len(unique)} unique categories: {unique}")
    if preference_ratio > 0.7:
        issues.append(f"{preference_ratio:.0%} beliefs are 'preference' (too uniform)")

    passed = len(issues) == 0
    detail = f"Categories: {dict((c, categories.count(c)) for c in unique)}"
    if issues:
        detail += f" | {'; '.join(issues)}"
    return passed, detail


def structured_core_constraints_realistic(structured_core: dict) -> tuple[bool, str]:
    """Constraints should capture the full picture, including weekend differences."""
    constraints = structured_core.get("constraints", {})

    issues = []
    max_min = constraints.get("max_session_minutes")
    if max_min and max_min <= 90:
        # If user mentioned longer weekend sessions, this should be reflected
        issues.append(f"max_session_minutes={max_min} may ignore longer weekend sessions")

    passed = len(issues) == 0
    return passed, f"Constraints: {constraints}" + (f" | {'; '.join(issues)}" if issues else "")


def plan_has_specific_targets(plan: dict) -> tuple[bool, str]:
    """Each session should have meaningful targets (pace, HR zone, or description).

    Not just "Easy Run" but specific guidance.
    """
    sessions = plan.get("sessions", [])
    vague_sessions = []

    for s in sessions:
        has_pace = bool(s.get("target_pace_min_km"))
        has_hr = bool(s.get("target_hr_zone"))
        desc_len = len(s.get("description", ""))

        if not has_pace and not has_hr and desc_len < 50:
            vague_sessions.append(s.get("day", "?"))

    passed = len(vague_sessions) <= 1  # Allow 1 rest/recovery day to be vague
    detail = f"{len(sessions)} sessions, {len(vague_sessions)} vague"
    if vague_sessions:
        detail += f": {vague_sessions}"
    return passed, detail


ALL_CRITERIA = {
    "plan_sport_distribution": plan_respects_sport_distribution,
    "plan_dates_valid": plan_dates_are_valid,
    "target_date_iso": target_date_is_iso,
    "fitness_populated": fitness_fields_populated,
    "ongoing_answers_questions": ongoing_session_answers_questions,
    "belief_categories": beliefs_have_correct_categories,
    "constraints_realistic": structured_core_constraints_realistic,
    "plan_specific_targets": plan_has_specific_targets,
}
