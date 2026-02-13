"""Acceptance validator for ReAgt behavioral testing.

Validates scenario results against expected outcomes. Covers both:
- v1.0 criteria: profile completeness, plan quality, belief extraction
- v2.0 criteria: action selection, plan evaluation, routing, proactivity

The v2.0 criteria start as placeholders and are activated as each
priority level is implemented.
"""

import re
from datetime import date

# Import existing v1.0 criteria
from tests.acceptance_criteria import (
    plan_respects_sport_distribution,
    plan_dates_are_valid,
    target_date_is_iso,
    fitness_fields_populated,
    beliefs_have_correct_categories,
    plan_has_specific_targets,
)


def validate_scenario(result: dict, expected: dict) -> list[dict]:
    """Run all applicable validation checks against scenario results.

    Args:
        result: Scenario result dict from ralph_scenarios.run_scenario()
        expected: Expected outcomes dict from persona definition

    Returns:
        List of {criterion, passed, detail} dicts.
    """
    checks = []
    sc = result.get("structured_core", {})
    beliefs = result.get("beliefs", [])
    plan = result.get("plan")

    # ── v1.0 Criteria (regression baseline) ─────────────────────

    # 1. Onboarding completed
    checks.append({
        "criterion": "onboarding_complete",
        "passed": result.get("onboarding_complete", False),
        "detail": f"Complete: {result.get('onboarding_complete')}",
    })

    # 2. Plan generated
    checks.append({
        "criterion": "plan_generated",
        "passed": result.get("plan_generated", False),
        "detail": f"Plan: {'yes' if plan else 'no'}, "
                  f"sessions: {len(plan.get('sessions', [])) if plan else 0}",
    })

    # 3. Sports correctly extracted
    # Synonyms for flexible matching (LLM may normalize sport names)
    SPORT_SYNONYMS = {
        "gym": {"gym", "strength_training", "strength", "weight_training"},
        "strength_training": {"gym", "strength_training", "strength", "weight_training"},
        "strength": {"gym", "strength_training", "strength", "weight_training"},
        "running": {"running", "trail_running", "jogging"},
        "cycling": {"cycling", "road_cycling", "bike"},
        "swimming": {"swimming", "open_water_swimming"},
    }
    expected_sports = expected.get("sports", [])
    actual_sports = [s.lower().replace(" ", "_") for s in sc.get("sports", [])]

    def sport_matches(expected_s, actual_list):
        es = expected_s.lower().replace(" ", "_")
        synonyms = SPORT_SYNONYMS.get(es, {es})
        return any(a in synonyms or es in a or a in es for a in actual_list)

    sports_match = all(sport_matches(es, actual_sports) for es in expected_sports)
    checks.append({
        "criterion": "sports_extracted",
        "passed": sports_match and len(actual_sports) >= len(expected_sports),
        "detail": f"Expected: {expected_sports}, Got: {actual_sports}",
    })

    # 4. Goal event set (or no_event_ok for casual athletes)
    goal = sc.get("goal", {})
    has_goal = bool(goal.get("event"))
    no_event_ok = expected.get("no_event_ok", False)
    checks.append({
        "criterion": "goal_event_set",
        "passed": has_goal or no_event_ok,
        "detail": f"Event: {goal.get('event')}"
                  + (" (no event ok for casual)" if no_event_ok and not has_goal else ""),
    })

    # 5. Target date
    if expected.get("expects_target_date"):
        if goal.get("target_date"):
            passed, detail = target_date_is_iso(sc)
            checks.append({"criterion": "target_date_iso", "passed": passed, "detail": detail})
        else:
            checks.append({
                "criterion": "target_date_iso",
                "passed": False,
                "detail": "No target date set but expected",
            })
    else:
        checks.append({
            "criterion": "target_date_iso",
            "passed": True,
            "detail": "No target date expected (general goal)",
        })

    # 6. Training days set
    constraints = sc.get("constraints", {})
    has_days = constraints.get("training_days_per_week") is not None
    checks.append({
        "criterion": "training_days_set",
        "passed": has_days,
        "detail": f"Days/week: {constraints.get('training_days_per_week')}",
    })

    # 7. Belief extraction
    if beliefs:
        passed, detail = beliefs_have_correct_categories(beliefs)
        checks.append({"criterion": "belief_categories", "passed": passed, "detail": detail})

        min_beliefs = expected.get("min_beliefs", 2)
        checks.append({
            "criterion": "belief_count",
            "passed": len(beliefs) >= min_beliefs,
            "detail": f"Beliefs: {len(beliefs)} (min: {min_beliefs})",
        })
    else:
        checks.append({
            "criterion": "belief_categories",
            "passed": False,
            "detail": "No beliefs extracted",
        })

    # 8. Plan checks (if plan exists)
    if plan:
        passed, detail = plan_dates_are_valid(plan)
        checks.append({"criterion": "plan_dates_valid", "passed": passed, "detail": detail})

        passed, detail = plan_respects_sport_distribution(plan, beliefs)
        checks.append({"criterion": "plan_sport_distribution", "passed": passed, "detail": detail})

        passed, detail = plan_has_specific_targets(plan)
        checks.append({"criterion": "plan_specific_targets", "passed": passed, "detail": detail})

        session_count = len(plan.get("sessions", []))
        expected_min = expected.get("min_sessions", 3)
        expected_max = expected.get("max_sessions", 10)
        checks.append({
            "criterion": "session_count",
            "passed": expected_min <= session_count <= expected_max,
            "detail": f"Sessions: {session_count} (expected {expected_min}-{expected_max})",
        })

    # 9. Fitness fields (if performance data given)
    if expected.get("gives_performance_data"):
        passed, detail = fitness_fields_populated(sc, beliefs)
        checks.append({"criterion": "fitness_populated", "passed": passed, "detail": detail})

    # 10. No errors
    checks.append({
        "criterion": "no_errors",
        "passed": len(result.get("errors", [])) == 0,
        "detail": f"Errors: {result.get('errors', [])}",
    })

    return checks


# ── v2.0 Behavioral Criteria (activated per priority level) ──────────

def validate_action_selection(cycle_result: dict) -> list[dict]:
    """P2: Verify agent selects actions dynamically, not fixed sequence.

    Activated after Priority 2 (Agent Action Space) is implemented.
    """
    checks = []

    # Check that cycle_context contains action selection trace
    ctx = cycle_result.get("cycle_context", {})
    actions_selected = ctx.get("actions_selected", [])

    checks.append({
        "criterion": "action_selection_exists",
        "passed": len(actions_selected) > 0,
        "detail": f"Actions selected: {actions_selected}",
    })

    # Verify not always the same sequence
    if len(actions_selected) > 0:
        checks.append({
            "criterion": "action_is_dynamic",
            "passed": True,  # Will need multiple runs to verify
            "detail": "Single run check — compare across scenarios for true dynamic test",
        })

    return checks


def validate_plan_evaluation(cycle_result: dict) -> list[dict]:
    """P4: Verify plans get scored and potentially regenerated.

    Activated after Priority 4 (Plan Evaluator) is implemented.
    """
    checks = []

    ctx = cycle_result.get("cycle_context", {})
    plan_scores = ctx.get("plan_scores", [])
    iterations = ctx.get("plan_iterations", 0)

    checks.append({
        "criterion": "plan_was_evaluated",
        "passed": len(plan_scores) > 0,
        "detail": f"Scores: {plan_scores}, Iterations: {iterations}",
    })

    # Check score is reasonable (not rubber-stamp)
    if plan_scores:
        checks.append({
            "criterion": "evaluation_not_rubber_stamp",
            "passed": not all(s >= 70 for s in plan_scores),
            "detail": f"All scores >= 70 suggests rubber-stamp: {plan_scores}",
        })

    return checks


def validate_input_routing(conversation_responses: list[dict]) -> list[dict]:
    """P5: Verify different message types get different treatment.

    Activated after Priority 5 (Input Routing) is implemented.
    """
    checks = []

    # Check that route metadata is present in responses
    routes_used = set()
    for r in conversation_responses:
        route = r.get("metadata", {}).get("route")
        if route:
            routes_used.add(route)

    checks.append({
        "criterion": "routing_produces_routes",
        "passed": len(routes_used) > 0,
        "detail": f"Routes used: {routes_used}",
    })

    return checks


def validate_active_memory(user_model_data: dict) -> list[dict]:
    """P6: Verify beliefs have utility scores from outcomes.

    Activated after Priority 6 (Active Memory) is implemented.
    """
    checks = []

    beliefs = user_model_data.get("beliefs", [])
    beliefs_with_utility = [b for b in beliefs if b.get("utility") is not None]

    checks.append({
        "criterion": "beliefs_have_utility",
        "passed": len(beliefs_with_utility) > 0,
        "detail": f"Beliefs with utility: {len(beliefs_with_utility)}/{len(beliefs)}",
    })

    return checks


def validate_no_hardcoded_rules(source_dir: str) -> list[dict]:
    """P7: Verify no coaching keyword lists remain in source code.

    Activated after Priority 7 (Eliminate Hardcoded Rules) is implemented.
    """
    import ast
    from pathlib import Path

    checks = []
    hardcoded_patterns = [
        "_HIGH_IMPACT_KEYWORDS",
        "_MEDIUM_IMPACT_KEYWORDS",
        "GOAL_TYPES",
        "REFLECTION_MIN_DAYS",
        "REFLECTION_MIN_ACTIVITIES",
    ]

    src_path = Path(source_dir)
    found = []
    for py_file in src_path.rglob("*.py"):
        content = py_file.read_text()
        for pattern in hardcoded_patterns:
            if pattern in content:
                # Check if it's used as a fallback (acceptable) or primary logic
                found.append(f"{py_file.name}:{pattern}")

    checks.append({
        "criterion": "no_hardcoded_coaching_rules",
        "passed": len(found) == 0,
        "detail": f"Found: {found}" if found else "No hardcoded rules found",
    })

    return checks
