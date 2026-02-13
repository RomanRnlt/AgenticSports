#!/usr/bin/env python3
"""End-to-end scenario tests for ReAgt coaching agent.

Runs 5 diverse athlete scenarios through the onboarding + plan generation
pipeline, inspects user model / beliefs / plans, and validates against
acceptance criteria. Each scenario resets state before running.

Usage:
    uv run python tests/scenario_test.py [scenario_number]
    uv run python tests/scenario_test.py          # run all
    uv run python tests/scenario_test.py 1         # run scenario 1 only
"""

import json
import sys
import time
import traceback
from datetime import date
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = DATA_DIR / "user_model" / "model.json"
PLANS_DIR = DATA_DIR / "plans"
SESSIONS_DIR = DATA_DIR / "sessions"
EPISODES_DIR = DATA_DIR / "episodes"

# ── Acceptance criteria imports ──
from tests.acceptance_criteria import (
    plan_respects_sport_distribution,
    plan_dates_are_valid,
    target_date_is_iso,
    fitness_fields_populated,
    beliefs_have_correct_categories,
    structured_core_constraints_realistic,
    plan_has_specific_targets,
)


def reset_state():
    """Reset all user data (equivalent to ./start.sh --reset)."""
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
    for d in [SESSIONS_DIR, PLANS_DIR, EPISODES_DIR]:
        if d.exists():
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()
    # Small delay for file system
    time.sleep(0.2)


def run_onboarding(messages: list[str], scenario_name: str) -> dict:
    """Run a full onboarding conversation and return results.

    Returns dict with: user_model, plan, beliefs, structured_core, responses, errors
    """
    from src.agent.onboarding import OnboardingEngine
    from src.memory.user_model import UserModel
    from src.agent.coach import generate_plan, save_plan
    from src.tools.activity_store import list_activities
    from src.memory.episodes import list_episodes, retrieve_relevant_episodes

    results = {
        "scenario": scenario_name,
        "responses": [],
        "errors": [],
        "user_model": None,
        "plan": None,
        "beliefs": [],
        "structured_core": {},
        "onboarding_complete": False,
        "plan_generated": False,
        "turn_count": 0,
    }

    try:
        user_model = UserModel.load_or_create()
        engine = OnboardingEngine(user_model=user_model)

        greeting = engine.start()
        results["responses"].append({"role": "agent", "content": greeting})
        print(f"  Coach: {greeting[:120]}...")

        for i, msg in enumerate(messages):
            print(f"  User [{i+1}/{len(messages)}]: {msg[:100]}...")
            try:
                response = engine.process_message(msg)
                results["responses"].append({"role": "user", "content": msg})
                results["responses"].append({"role": "agent", "content": response})
                results["turn_count"] += 1
                print(f"  Coach: {response[:150]}...")

                if engine.is_onboarding_complete():
                    results["onboarding_complete"] = True
                    print(f"  [Onboarding complete after turn {i+1}]")

                    # Generate plan
                    profile = user_model.project_profile()
                    beliefs = user_model.get_active_beliefs(min_confidence=0.6)
                    activities = list_activities()
                    episodes = list_episodes(limit=10)
                    relevant_eps = retrieve_relevant_episodes(
                        {"goal": profile.get("goal", {}), "sports": profile.get("sports", [])},
                        episodes,
                        max_results=5,
                    )

                    try:
                        plan = generate_plan(
                            profile, beliefs=beliefs, activities=activities,
                            relevant_episodes=relevant_eps,
                        )
                        save_plan(plan)
                        results["plan"] = plan
                        results["plan_generated"] = True
                        print(f"  [Plan generated: {len(plan.get('sessions', []))} sessions]")
                    except Exception as e:
                        results["errors"].append(f"Plan generation failed: {e}")
                        print(f"  [Plan generation FAILED: {e}]")

                    break
            except Exception as e:
                results["errors"].append(f"Turn {i+1} failed: {e}")
                print(f"  [ERROR in turn {i+1}: {e}]")
                traceback.print_exc()

        # Save and collect final state
        engine.end_session()
        user_model.save()

        results["user_model"] = {
            "structured_core": user_model.structured_core,
            "beliefs": [
                {"id": b["id"], "text": b["text"], "category": b["category"],
                 "confidence": b["confidence"], "active": b["active"]}
                for b in user_model.beliefs
            ],
            "meta": user_model.meta,
        }
        results["beliefs"] = user_model.get_active_beliefs(min_confidence=0.6)
        results["structured_core"] = user_model.structured_core

    except Exception as e:
        results["errors"].append(f"Scenario failed: {e}")
        traceback.print_exc()

    return results


def validate_results(results: dict, expected: dict) -> list[dict]:
    """Run acceptance criteria and custom checks against scenario results.

    Returns list of {criterion, passed, detail} dicts.
    """
    checks = []
    sc = results.get("structured_core", {})
    beliefs = results.get("beliefs", [])
    plan = results.get("plan")

    # 1. Onboarding completed
    checks.append({
        "criterion": "onboarding_complete",
        "passed": results.get("onboarding_complete", False),
        "detail": f"Complete: {results.get('onboarding_complete')}, turns: {results.get('turn_count')}",
    })

    # 2. Plan generated
    checks.append({
        "criterion": "plan_generated",
        "passed": results.get("plan_generated", False),
        "detail": f"Plan: {'yes' if plan else 'no'}, sessions: {len(plan.get('sessions', [])) if plan else 0}",
    })

    # 3. Sports correctly extracted
    expected_sports = expected.get("sports", [])
    actual_sports = [s.lower() for s in sc.get("sports", [])]
    sports_match = all(
        any(es.lower() in a or a in es.lower() for a in actual_sports)
        for es in expected_sports
    )
    checks.append({
        "criterion": "sports_extracted",
        "passed": sports_match and len(actual_sports) >= len(expected_sports),
        "detail": f"Expected: {expected_sports}, Got: {actual_sports}",
    })

    # 4. Goal event set
    goal = sc.get("goal", {})
    has_goal = bool(goal.get("event"))
    checks.append({
        "criterion": "goal_event_set",
        "passed": has_goal,
        "detail": f"Event: {goal.get('event')}, Type: {goal.get('goal_type')}",
    })

    # 5. Target date is ISO format
    if goal.get("target_date"):
        passed, detail = target_date_is_iso(sc)
        checks.append({"criterion": "target_date_iso", "passed": passed, "detail": detail})
    else:
        # No target date expected (e.g. general fitness goal)
        expects_date = expected.get("expects_target_date", True)
        checks.append({
            "criterion": "target_date_iso",
            "passed": not expects_date,
            "detail": f"No target date set (expected: {expects_date})",
        })

    # 6. Training days set
    constraints = sc.get("constraints", {})
    has_days = constraints.get("training_days_per_week") is not None
    checks.append({
        "criterion": "training_days_set",
        "passed": has_days,
        "detail": f"Days/week: {constraints.get('training_days_per_week')}",
    })

    # 7. Belief categories diverse
    if beliefs:
        passed, detail = beliefs_have_correct_categories(beliefs)
        checks.append({"criterion": "belief_categories", "passed": passed, "detail": detail})

    # 8. Plan checks (if plan exists)
    if plan:
        # Plan dates valid
        passed, detail = plan_dates_are_valid(plan)
        checks.append({"criterion": "plan_dates_valid", "passed": passed, "detail": detail})

        # Plan sport distribution
        passed, detail = plan_respects_sport_distribution(plan, beliefs)
        checks.append({"criterion": "plan_sport_distribution", "passed": passed, "detail": detail})

        # Plan has specific targets
        passed, detail = plan_has_specific_targets(plan)
        checks.append({"criterion": "plan_specific_targets", "passed": passed, "detail": detail})

        # Session count reasonable
        session_count = len(plan.get("sessions", []))
        expected_min = expected.get("min_sessions", 3)
        expected_max = expected.get("max_sessions", 10)
        checks.append({
            "criterion": "session_count",
            "passed": expected_min <= session_count <= expected_max,
            "detail": f"Sessions: {session_count} (expected {expected_min}-{expected_max})",
        })

        # Multi-sport plan has multiple sports
        if len(expected_sports) > 1:
            plan_sports = set()
            for s in plan.get("sessions", []):
                plan_sports.add(s.get("sport", "").lower())
            plan_sports.discard("rest")
            plan_sports.discard("recovery")
            checks.append({
                "criterion": "multi_sport_plan",
                "passed": len(plan_sports) >= 2,
                "detail": f"Sports in plan: {plan_sports}",
            })

    # 9. Fitness fields (if performance data was given)
    if expected.get("gives_performance_data"):
        passed, detail = fitness_fields_populated(sc, beliefs)
        checks.append({"criterion": "fitness_populated", "passed": passed, "detail": detail})

    # 10. No errors
    checks.append({
        "criterion": "no_errors",
        "passed": len(results.get("errors", [])) == 0,
        "detail": f"Errors: {results.get('errors', [])}",
    })

    return checks


# ── SCENARIO DEFINITIONS ──────────────────────────────────────────────

SCENARIOS = {
    1: {
        "name": "Multi-Sport Triathlete (HM + Tri)",
        "messages": [
            "Hi, ich laufe gerne, fahre Rennrad und schwimme einmal in der Woche. "
            "Im August möchte ich gerne einen Halbmarathon unter 1:30 laufen, das ist mein Hauptziel. "
            "Ich strebe auch an einen Triathlon zu machen, vielleicht Q3 oder Q4 2026, aber nur Kurzdistanz. "
            "Ich gehe auch noch ins Gym und möchte Muskeln aufbauen (lean). "
            "Ich dachte daran 3 mal die Woche laufen zu gehen, 1 mal schwimmen und Rennrad 2-3 mal. "
            "Ich habe übrigens auch eine Rolle fürs Rennrad und fahre dort online in Zwift.",
            "Meine 10km Bestzeit ist 42:30, die bin ich letztes Jahr gelaufen. "
            "Unter der Woche habe ich maximal 90 Minuten pro Einheit, am Wochenende auch mal 2-3 Stunden. "
            "Ach ja, ich heiße Roman.",
        ],
        "expected": {
            "sports": ["running", "cycling", "swimming"],
            "expects_target_date": True,
            "gives_performance_data": True,
            "min_sessions": 5,
            "max_sessions": 10,
        },
    },
    2: {
        "name": "Pure Road Cyclist (Gran Fondo)",
        "messages": [
            "Hey! I'm a road cyclist, that's basically all I do. "
            "I'm training for a Gran Fondo in September 2026, about 160km with 2000m climbing. "
            "My FTP is around 260W at 75kg. I can ride 5-6 days a week, "
            "with longer rides on weekends up to 4-5 hours. "
            "During the week I mostly do 60-90 min on the trainer (Zwift). "
            "My name is Klaus.",
        ],
        "expected": {
            "sports": ["cycling"],
            "expects_target_date": True,
            "gives_performance_data": True,
            "min_sessions": 4,
            "max_sessions": 7,
        },
    },
    3: {
        "name": "Casual Fitness (Gym + Running, No Race)",
        "messages": [
            "Hi! Ich bin Lisa und ich möchte einfach fit bleiben und ein bisschen abnehmen. "
            "Ich gehe 3 mal die Woche ins Gym für Krafttraining und jogge gerne 2 mal. "
            "Kein spezielles Rennen oder so, einfach gesund bleiben und stärker werden. "
            "Unter der Woche habe ich eine Stunde Zeit pro Training.",
            "Ich habe keine Bestzeiten oder so, laufe meistens so 30-40 Minuten easy. "
            "Am Wochenende habe ich mehr Zeit, da kann ich auch mal 90 Minuten machen. "
            "Mein Ziel ist eigentlich einfach regelmäßig Sport zu machen und lean muscle aufzubauen.",
        ],
        "expected": {
            "sports": ["gym", "running"],
            "expects_target_date": False,
            "gives_performance_data": False,
            "min_sessions": 4,
            "max_sessions": 7,
        },
    },
    4: {
        "name": "Competitive Open Water Swimmer",
        "messages": [
            "Hello, I'm Marco and I'm an open water swimmer. I'm training for a 5km open water race "
            "in Lake Zurich in July 2026. My best 1500m pool time is 22:30 from last month. "
            "I swim 4 times a week in the pool and once in open water when the weather allows. "
            "I also run twice a week for general conditioning. "
            "Each pool session is about 60-75 minutes, weekends I do longer sets up to 2 hours.",
        ],
        "expected": {
            "sports": ["swimming"],
            "expects_target_date": True,
            "gives_performance_data": True,
            "min_sessions": 4,
            "max_sessions": 8,
        },
    },
    5: {
        "name": "Trail / Ultra Runner",
        "messages": [
            "Hey, ich bin Tom und mache Trail Running. Mein Ziel ist der Eiger Ultra Trail E51 "
            "(51km, 3000hm+) im Juli 2026. Mein letzter Halbmarathon war in 1:38 auf Straße. "
            "Ich trainiere 5 mal die Woche: 3x Laufen (davon 1x Trail am Wochenende), "
            "1x Krafttraining und 1x lockeres Radfahren zur Erholung. "
            "Unter der Woche maximal 75 Minuten, am Wochenende der Long Run kann 3-4 Stunden sein.",
        ],
        "expected": {
            "sports": ["running"],
            "expects_target_date": True,
            "gives_performance_data": True,
            "min_sessions": 4,
            "max_sessions": 7,
        },
    },
}


def run_scenario(scenario_id: int) -> dict:
    """Run a single test scenario end-to-end."""
    scenario = SCENARIOS[scenario_id]
    name = scenario["name"]

    print(f"\n{'='*70}")
    print(f"SCENARIO {scenario_id}: {name}")
    print(f"{'='*70}")

    # Reset
    print("  [Resetting state...]")
    reset_state()
    time.sleep(0.5)

    # Run onboarding
    print("  [Starting onboarding...]")
    results = run_onboarding(scenario["messages"], name)

    # Validate
    print(f"\n  [Validating...]")
    checks = validate_results(results, scenario["expected"])

    # Print results
    passed = sum(1 for c in checks if c["passed"])
    total = len(checks)
    print(f"\n  Results: {passed}/{total} checks passed")
    for c in checks:
        icon = "PASS" if c["passed"] else "FAIL"
        print(f"    [{icon}] {c['criterion']}: {c['detail']}")

    return {
        "scenario_id": scenario_id,
        "scenario_name": name,
        "checks": checks,
        "passed": passed,
        "total": total,
        "all_passed": passed == total,
        "structured_core": results.get("structured_core"),
        "belief_count": len(results.get("beliefs", [])),
        "plan_session_count": len(results.get("plan", {}).get("sessions", [])) if results.get("plan") else 0,
        "errors": results.get("errors", []),
        "plan": results.get("plan"),
    }


def main():
    """Run all (or selected) scenarios and print summary report."""
    if len(sys.argv) > 1:
        ids = [int(x) for x in sys.argv[1:]]
    else:
        ids = sorted(SCENARIOS.keys())

    all_results = []
    for sid in ids:
        result = run_scenario(sid)
        all_results.append(result)

    # ── SUMMARY REPORT ──
    print(f"\n\n{'='*70}")
    print("SUMMARY REPORT")
    print(f"{'='*70}")

    total_checks = 0
    total_passed = 0

    for r in all_results:
        status = "PASS" if r["all_passed"] else "FAIL"
        print(f"\n  [{status}] Scenario {r['scenario_id']}: {r['scenario_name']}")
        print(f"         Checks: {r['passed']}/{r['total']}")
        print(f"         Beliefs: {r['belief_count']}, Plan sessions: {r['plan_session_count']}")

        core = r.get("structured_core", {})
        print(f"         Sports: {core.get('sports', [])}")
        print(f"         Goal: {core.get('goal', {}).get('event')}")
        print(f"         Goal type: {core.get('goal', {}).get('goal_type')}")
        print(f"         Days/week: {core.get('constraints', {}).get('training_days_per_week')}")
        print(f"         VO2max: {core.get('fitness', {}).get('estimated_vo2max')}")
        print(f"         Threshold: {core.get('fitness', {}).get('threshold_pace_min_km')}")

        if r["errors"]:
            print(f"         ERRORS: {r['errors']}")

        # Failed checks
        failed = [c for c in r["checks"] if not c["passed"]]
        if failed:
            print(f"         FAILED CHECKS:")
            for c in failed:
                print(f"           - {c['criterion']}: {c['detail']}")

        total_checks += r["total"]
        total_passed += r["passed"]

    print(f"\n  TOTAL: {total_passed}/{total_checks} checks passed across {len(all_results)} scenarios")

    # Save detailed report
    report_path = PROJECT_ROOT / "tests" / "scenario_report.json"
    report_data = []
    for r in all_results:
        report_entry = {k: v for k, v in r.items() if k != "plan"}
        if r.get("plan"):
            report_entry["plan_summary"] = {
                "week_start": r["plan"].get("week_start"),
                "sessions": [
                    {
                        "day": s.get("day"),
                        "sport": s.get("sport"),
                        "type": s.get("type"),
                        "duration": s.get("total_duration_minutes", s.get("duration_minutes")),
                        "has_steps": bool(s.get("steps")),
                        "step_count": len(s.get("steps", [])),
                    }
                    for s in r["plan"].get("sessions", [])
                ],
                "weekly_summary": r["plan"].get("weekly_summary"),
            }
        report_data.append(report_entry)

    report_path.write_text(json.dumps(report_data, indent=2, default=str))
    print(f"\n  Detailed report saved to: {report_path}")


if __name__ == "__main__":
    main()
