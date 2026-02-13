"""Acceptance criteria for ReAgt coaching agent.

These criteria define what "excellent" looks like. Each criterion is a function
that takes test results and returns (passed: bool, detail: str).
"""

import re
from datetime import date, datetime


def plan_respects_sport_distribution(plan: dict, beliefs: list[dict]) -> tuple[bool, str]:
    """Plan must include sessions matching the user's stated sport distribution.

    Generically parses scheduling beliefs for patterns like "Wants Nx <sport> per week"
    and compares against actual plan session counts. No hardcoded sport names.
    """
    sessions = plan.get("sessions", [])
    sports_in_plan: dict[str, int] = {}
    for s in sessions:
        sport = s.get("sport", "unknown").lower().replace(" ", "_")
        sports_in_plan[sport] = sports_in_plan.get(sport, 0) + 1

    issues = []

    # Extract expected sport counts from scheduling beliefs
    # Matches patterns like "Wants 3x running per week", "Wants 2-3x cycling per week"
    expected_counts: dict[str, tuple[int, int]] = {}  # sport -> (min_count, max_count)
    scheduling_beliefs = [b for b in beliefs if b.get("category") == "scheduling"]
    for b in scheduling_beliefs:
        text = b.get("text", "")
        # Match "Wants Nx sport" or "Wants N-Mx sport"
        m = re.search(r"[Ww]ants?\s+(\d+)(?:\s*-\s*(\d+))?x?\s+(.+?)(?:\s+per\s+week|\s+sessions?|\s*$)", text)
        if m:
            min_count = int(m.group(1))
            max_count = int(m.group(2)) if m.group(2) else min_count
            sport_name = m.group(3).strip().lower().replace(" ", "_")
            expected_counts[sport_name] = (min_count, max_count)

    # Compare expected vs actual for each expected sport
    for sport, (min_c, max_c) in expected_counts.items():
        # Find matching sport in plan (flexible matching: "run" matches "running", etc.)
        actual = 0
        for plan_sport, count in sports_in_plan.items():
            if sport in plan_sport or plan_sport in sport or plan_sport.startswith(sport[:3]):
                actual += count
        if actual < min_c:
            issues.append(f"Expected {min_c}-{max_c}x {sport} but got {actual}")

    # If no scheduling beliefs parsed, fall back to checking multi-sport coverage
    if not expected_counts:
        sport_mentions: set[str] = set()
        for b in beliefs:
            text = b.get("text", "").lower()
            # Look for any belief mentioning a sport with session counts
            if any(kw in text for kw in ["session", "mal", "times", "per week"]):
                sport_mentions.add("multi")
        real_sports = [s for s in sports_in_plan if s not in ("rest", "recovery", "general_fitness")]
        if sport_mentions and len(real_sports) < 2:
            issues.append(f"Only {len(real_sports)} sport types in plan, expected multi-sport")

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

    Checks both session-level fields (legacy flat format) and inside
    steps[].targets / steps[].steps[].targets (structured workout format).
    """
    sessions = plan.get("sessions", [])
    vague_sessions = []

    for s in sessions:
        has_pace = bool(s.get("target_pace_min_km"))
        has_hr = bool(s.get("target_hr_zone"))
        # Also check inside steps for targets
        for step in s.get("steps", []):
            targets = step.get("targets", {})
            if targets.get("pace_min_km") or targets.get("power_watts") or targets.get("pace_min_100m"):
                has_pace = True
            if targets.get("hr_zone"):
                has_hr = True
            # Check inside repeat steps too
            for sub in step.get("steps", []):
                sub_targets = sub.get("targets", {})
                if sub_targets.get("pace_min_km") or sub_targets.get("power_watts") or sub_targets.get("pace_min_100m"):
                    has_pace = True
                if sub_targets.get("hr_zone"):
                    has_hr = True
        desc_len = len(s.get("description", "")) + len(s.get("notes", ""))

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
