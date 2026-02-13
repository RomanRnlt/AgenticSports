"""Persona definitions and scenario runner for ReAgt behavioral testing.

5 diverse athlete personas with scripted onboarding conversations,
follow-up questions, and expected outcomes. Compatible with both
the existing scenario_test.py approach and the new ralph_harness.py.

Usage:
    python tests/ralph_scenarios.py [scenario_id]   # run one or all
"""

import json
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.ralph_harness import ReAgtHarness
from tests.ralph_validator import validate_scenario


# ── PERSONA DEFINITIONS ──────────────────────────────────────────────

PERSONAS = {
    1: {
        "name": "Multi-Sport Triathlete (Roman)",
        "language": "de",
        "onboarding_messages": [
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
        "followup_messages": [
            "Wie sieht mein Plan für diese Woche aus?",
            "Ich hatte gestern leichte Knieschmerzen beim Laufen.",
        ],
        "expected": {
            "sports": ["running", "cycling", "swimming"],
            "expects_target_date": True,
            "gives_performance_data": True,
            "min_sessions": 5,
            "max_sessions": 10,
            "min_beliefs": 3,
            "goal_event_keywords": ["marathon", "halbmarathon", "half"],
        },
    },
    2: {
        "name": "Pure Road Cyclist (Klaus)",
        "language": "en",
        "onboarding_messages": [
            "Hey! I'm a road cyclist, that's basically all I do. "
            "I'm training for a Gran Fondo in September 2026, about 160km with 2000m climbing. "
            "My FTP is around 260W at 75kg. I can ride 5-6 days a week, "
            "with longer rides on weekends up to 4-5 hours. "
            "During the week I mostly do 60-90 min on the trainer (Zwift). "
            "My name is Klaus.",
        ],
        "followup_messages": [
            "What should my threshold intervals look like this week?",
        ],
        "expected": {
            "sports": ["cycling"],
            "expects_target_date": True,
            "gives_performance_data": True,
            "min_sessions": 4,
            "max_sessions": 7,
            "min_beliefs": 2,
            "goal_event_keywords": ["fondo", "gran", "160km"],
        },
    },
    3: {
        "name": "Casual Fitness (Lisa)",
        "language": "de",
        "onboarding_messages": [
            "Hi! Ich bin Lisa und ich möchte einfach fit bleiben und ein bisschen abnehmen. "
            "Ich gehe 3 mal die Woche ins Gym für Krafttraining und jogge gerne 2 mal. "
            "Kein spezielles Rennen oder so, einfach gesund bleiben und stärker werden. "
            "Unter der Woche habe ich eine Stunde Zeit pro Training.",
            "Ich habe keine Bestzeiten oder so, laufe meistens so 30-40 Minuten easy. "
            "Am Wochenende habe ich mehr Zeit, da kann ich auch mal 90 Minuten machen. "
            "Mein Ziel ist eigentlich einfach regelmäßig Sport zu machen und lean muscle aufzubauen.",
        ],
        "followup_messages": [],
        "expected": {
            "sports": ["strength_training", "running"],
            "expects_target_date": False,
            "gives_performance_data": False,
            "min_sessions": 4,
            "max_sessions": 7,
            "min_beliefs": 2,
            "goal_event_keywords": ["fitness", "fit", "gesund", "lean"],
            "no_event_ok": True,  # casual athlete, no race target
        },
    },
    4: {
        "name": "Competitive Swimmer (Marco)",
        "language": "en",
        "onboarding_messages": [
            "Hello, I'm Marco and I'm an open water swimmer. I'm training for a 5km open water race "
            "in Lake Zurich in July 2026. My best 1500m pool time is 22:30 from last month. "
            "I swim 4 times a week in the pool and once in open water when the weather allows. "
            "I also run twice a week for general conditioning. "
            "Each pool session is about 60-75 minutes, weekends I do longer sets up to 2 hours.",
        ],
        "followup_messages": [
            "How should I structure my pool sessions for open water prep?",
        ],
        "expected": {
            "sports": ["swimming"],
            "expects_target_date": True,
            "gives_performance_data": True,
            "min_sessions": 4,
            "max_sessions": 8,
            "min_beliefs": 2,
            "goal_event_keywords": ["swim", "5km", "open water"],
        },
    },
    5: {
        "name": "Trail / Ultra Runner (Tom)",
        "language": "de",
        "onboarding_messages": [
            "Hey, ich bin Tom und mache Trail Running. Mein Ziel ist der Eiger Ultra Trail E51 "
            "(51km, 3000hm+) im Juli 2026. Mein letzter Halbmarathon war in 1:38 auf Straße. "
            "Ich trainiere 5 mal die Woche: 3x Laufen (davon 1x Trail am Wochenende), "
            "1x Krafttraining und 1x lockeres Radfahren zur Erholung. "
            "Unter der Woche maximal 75 Minuten, am Wochenende der Long Run kann 3-4 Stunden sein.",
        ],
        "followup_messages": [
            "Wie viel Höhenmeter sollte ich pro Woche trainieren?",
        ],
        "expected": {
            "sports": ["running"],
            "expects_target_date": True,
            "gives_performance_data": True,
            "min_sessions": 4,
            "max_sessions": 7,
            "min_beliefs": 2,
            "goal_event_keywords": ["trail", "ultra", "eiger", "E51"],
        },
    },
}


# ── SCENARIO RUNNER ──────────────────────────────────────────────────

def run_scenario(persona_id: int) -> dict:
    """Run a single persona through onboarding + plan generation.

    Returns a result dict with all data needed for validation.
    """
    persona = PERSONAS[persona_id]
    name = persona["name"]

    print(f"\n{'='*70}")
    print(f"SCENARIO {persona_id}: {name}")
    print(f"{'='*70}")

    harness = ReAgtHarness()
    harness.reset()
    time.sleep(0.3)

    result = {
        "persona_id": persona_id,
        "persona_name": name,
        "onboarding_complete": False,
        "plan_generated": False,
        "plan": None,
        "structured_core": {},
        "beliefs": [],
        "responses": [],
        "errors": [],
        "followup_responses": [],
    }

    try:
        # Start onboarding
        print("  [Starting onboarding...]")
        greeting = harness.start_onboarding()
        print(f"  Coach: {greeting[:120]}...")

        # Send onboarding messages
        for i, msg in enumerate(persona["onboarding_messages"]):
            print(f"  User [{i+1}/{len(persona['onboarding_messages'])}]: {msg[:100]}...")
            response = harness.send(msg)
            print(f"  Coach: {response[:150]}...")

            if harness.onboarding_complete:
                result["onboarding_complete"] = True
                print(f"  [Onboarding complete after message {i+1}]")
                break

        # Check if onboarding completed even without explicit trigger
        if not result["onboarding_complete"] and harness.onboarding_complete:
            result["onboarding_complete"] = True
            print("  [Onboarding complete (detected from structured_core)]")

        # Generate plan if onboarding complete
        if result["onboarding_complete"]:
            print("  [Generating plan...]")
            plan = harness.generate_plan()
            if plan:
                result["plan"] = plan
                result["plan_generated"] = True
                print(f"  [Plan generated: {len(plan.get('sessions', []))} sessions]")
            else:
                print("  [Plan generation FAILED]")

        # End session and collect state
        harness.end_session()

        result["structured_core"] = harness.structured_core
        result["beliefs"] = harness.beliefs
        result["responses"] = harness.responses
        result["errors"] = harness.errors

    except Exception as e:
        result["errors"].append(f"Scenario failed: {e}")
        traceback.print_exc()

    # Validate
    print(f"\n  [Validating...]")
    checks = validate_scenario(result, persona["expected"])

    passed = sum(1 for c in checks if c["passed"])
    total = len(checks)
    print(f"\n  Results: {passed}/{total} checks passed")
    for c in checks:
        icon = "PASS" if c["passed"] else "FAIL"
        print(f"    [{icon}] {c['criterion']}: {c['detail']}")

    result["checks"] = checks
    result["passed"] = passed
    result["total"] = total
    result["all_passed"] = passed == total

    return result


def main():
    """Run all (or selected) scenarios and print summary."""
    if len(sys.argv) > 1:
        ids = [int(x) for x in sys.argv[1:]]
    else:
        ids = sorted(PERSONAS.keys())

    all_results = []
    for sid in ids:
        result = run_scenario(sid)
        all_results.append(result)

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY REPORT")
    print(f"{'='*70}")

    total_checks = 0
    total_passed = 0

    for r in all_results:
        status = "PASS" if r["all_passed"] else "FAIL"
        print(f"\n  [{status}] Scenario {r['persona_id']}: {r['persona_name']}")
        print(f"         Checks: {r['passed']}/{r['total']}")
        print(f"         Beliefs: {len(r.get('beliefs', []))}")

        sc = r.get("structured_core", {})
        print(f"         Sports: {sc.get('sports', [])}")
        print(f"         Goal: {sc.get('goal', {}).get('event')}")
        print(f"         Days/week: {sc.get('constraints', {}).get('training_days_per_week')}")

        if r["errors"]:
            print(f"         ERRORS: {r['errors']}")

        failed = [c for c in r.get("checks", []) if not c["passed"]]
        if failed:
            for c in failed:
                print(f"         FAIL: {c['criterion']}: {c['detail']}")

        total_checks += r.get("total", 0)
        total_passed += r.get("passed", 0)

    pct = (total_passed / total_checks * 100) if total_checks > 0 else 0
    print(f"\n  TOTAL: {total_passed}/{total_checks} ({pct:.1f}%) across {len(all_results)} scenarios")

    # Save report
    report_path = PROJECT_ROOT / "tests" / "scenario_report_v2.json"
    report_data = []
    for r in all_results:
        entry = {k: v for k, v in r.items() if k != "plan"}
        if r.get("plan"):
            entry["plan_summary"] = {
                "week_start": r["plan"].get("week_start"),
                "session_count": len(r["plan"].get("sessions", [])),
                "weekly_summary": r["plan"].get("weekly_summary"),
            }
        report_data.append(entry)

    report_path.write_text(json.dumps(report_data, indent=2, default=str))
    print(f"\n  Report saved to: {report_path}")


if __name__ == "__main__":
    main()
