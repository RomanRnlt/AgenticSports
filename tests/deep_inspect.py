#!/usr/bin/env python3
"""Deep inspection of a single scenario: run onboarding, save plan, then test ongoing conversation.

Runs scenario 1 (Multi-Sport Triathlete), saves full plan, then simulates
returning user with follow-up questions to test ongoing coaching mode.
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = DATA_DIR / "user_model" / "model.json"
PLANS_DIR = DATA_DIR / "plans"
SESSIONS_DIR = DATA_DIR / "sessions"
EPISODES_DIR = DATA_DIR / "episodes"


def reset_state():
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
    for d in [SESSIONS_DIR, PLANS_DIR, EPISODES_DIR]:
        if d.exists():
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()
    time.sleep(0.2)


def print_plan_detail(plan: dict):
    """Print full plan with step details."""
    print(f"\n  Plan week_start: {plan.get('week_start')}")
    print(f"  Generated at: {plan.get('generated_at')}")
    for s in plan.get("sessions", []):
        print(f"\n  --- {s.get('day')} ({s.get('date')}) ---")
        print(f"  Sport: {s.get('sport')}, Type: {s.get('type')}")
        dur = s.get('total_duration_minutes', s.get('duration_minutes', '?'))
        print(f"  Duration: {dur} min")
        if s.get("steps"):
            for step in s["steps"]:
                if step.get("type") == "repeat":
                    count = step.get("repeat_count", 1)
                    print(f"    {count}x:")
                    for sub in step.get("steps", []):
                        targets = sub.get("targets", {})
                        t_str = ", ".join(f"{k}={v}" for k, v in targets.items()) if targets else "no targets"
                        print(f"      {sub.get('type','?')} {sub.get('duration','')}: {t_str}")
                else:
                    targets = step.get("targets", {})
                    t_str = ", ".join(f"{k}={v}" for k, v in targets.items()) if targets else "no targets"
                    print(f"    {step.get('type','?')} {step.get('duration','')}: {t_str}")
        if s.get("notes"):
            print(f"    Notes: {s['notes']}")

    summary = plan.get("weekly_summary", {})
    if summary:
        print(f"\n  Weekly Summary: {summary.get('total_sessions')} sessions, "
              f"{summary.get('total_duration_minutes')} min, "
              f"focus: {summary.get('focus')}")


def test_ongoing_conversation(questions: list[str]):
    """Test ongoing conversation mode with follow-up questions."""
    from src.agent.conversation import ConversationEngine
    from src.memory.user_model import UserModel

    user_model = UserModel.load_or_create()
    engine = ConversationEngine(user_model=user_model)
    engine.start_session()

    print("\n  === ONGOING CONVERSATION TEST ===")
    results = []

    for q in questions:
        print(f"\n  User: {q}")
        try:
            response = engine.process_message(q)
            print(f"  Coach: {response[:300]}...")
            results.append({"question": q, "response": response, "error": None})
        except Exception as e:
            print(f"  [ERROR: {e}]")
            results.append({"question": q, "response": None, "error": str(e)})

    engine.end_session()
    user_model.save()
    return results


def main():
    print("=" * 70)
    print("DEEP INSPECTION: Scenario 1 (Multi-Sport Triathlete)")
    print("=" * 70)

    # Reset
    reset_state()
    time.sleep(0.5)

    # Phase 1: Onboarding
    from src.agent.onboarding import OnboardingEngine
    from src.memory.user_model import UserModel
    from src.agent.coach import generate_plan, save_plan
    from src.tools.activity_store import list_activities
    from src.memory.episodes import list_episodes, retrieve_relevant_episodes

    user_model = UserModel.load_or_create()
    engine = OnboardingEngine(user_model=user_model)

    greeting = engine.start()
    print(f"\n  Coach: {greeting}")

    messages = [
        "Hi, ich laufe gerne, fahre Rennrad und schwimme einmal in der Woche. "
        "Im August möchte ich gerne einen Halbmarathon unter 1:30 laufen, das ist mein Hauptziel. "
        "Ich strebe auch an einen Triathlon zu machen, vielleicht Q3 oder Q4 2026, aber nur Kurzdistanz. "
        "Ich gehe auch noch ins Gym und möchte Muskeln aufbauen (lean). "
        "Ich dachte daran 3 mal die Woche laufen zu gehen, 1 mal schwimmen und Rennrad 2-3 mal. "
        "Ich habe übrigens auch eine Rolle fürs Rennrad und fahre dort online in Zwift.",
        "Meine 10km Bestzeit ist 42:30, die bin ich letztes Jahr gelaufen. "
        "Unter der Woche habe ich maximal 90 Minuten pro Einheit, am Wochenende auch mal 2-3 Stunden. "
        "Ach ja, ich heiße Roman.",
    ]

    for i, msg in enumerate(messages):
        print(f"\n  User: {msg[:120]}...")
        response = engine.process_message(msg)
        print(f"  Coach: {response[:200]}...")

        if engine.is_onboarding_complete():
            print(f"  [Onboarding complete after message {i+1}]")

            profile = user_model.project_profile()
            beliefs = user_model.get_active_beliefs(min_confidence=0.6)
            activities = list_activities()
            episodes = list_episodes(limit=10)
            relevant_eps = retrieve_relevant_episodes(
                {"goal": profile.get("goal", {}), "sports": profile.get("sports", [])},
                episodes, max_results=5,
            )

            plan = generate_plan(
                profile, beliefs=beliefs, activities=activities,
                relevant_episodes=relevant_eps,
            )
            save_plan(plan)
            break

    engine.end_session()
    user_model.save()

    # Phase 2: Inspect model
    print("\n\n  === USER MODEL INSPECTION ===")
    core = user_model.structured_core
    print(f"  Name: {core.get('name')}")
    print(f"  Sports: {core.get('sports')}")
    print(f"  Goal: {core.get('goal')}")
    print(f"  Fitness: {core.get('fitness')}")
    print(f"  Constraints: {core.get('constraints')}")

    print(f"\n  Active beliefs ({len(user_model.get_active_beliefs())}):")
    for b in user_model.get_active_beliefs():
        print(f"    [{b['category']}] {b['text']} (conf: {b['confidence']})")

    # Phase 3: Inspect plan
    print("\n\n  === PLAN INSPECTION ===")
    print_plan_detail(plan)

    # Phase 4: Plan quality checks
    print("\n\n  === PLAN QUALITY CHECKS ===")

    # Check: Does the plan have Zwift references?
    plan_text = json.dumps(plan).lower()
    has_zwift = "zwift" in plan_text
    print(f"  Zwift mentioned in plan: {has_zwift}")

    # Check: Does the plan distinguish weekday/weekend durations?
    weekday_durs = []
    weekend_durs = []
    for s in plan.get("sessions", []):
        day = s.get("day", "").lower()
        dur = s.get("total_duration_minutes", s.get("duration_minutes", 0))
        if day in ("saturday", "sunday"):
            weekend_durs.append(dur)
        else:
            weekday_durs.append(dur)
    print(f"  Weekday durations: {weekday_durs}")
    print(f"  Weekend durations: {weekend_durs}")
    if weekday_durs and weekend_durs:
        avg_weekday = sum(weekday_durs) / len(weekday_durs)
        avg_weekend = sum(weekend_durs) / len(weekend_durs)
        print(f"  Avg weekday: {avg_weekday:.0f}min, Avg weekend: {avg_weekend:.0f}min")
        print(f"  Weekend longer: {avg_weekend > avg_weekday}")

    # Check: Are running paces aligned with 10km PB of 42:30 (4:15/km)?
    print(f"\n  Running paces in plan (10km PB = 42:30, ~4:15/km):")
    for s in plan.get("sessions", []):
        if s.get("sport", "").lower() in ("running", "laufen"):
            for step in s.get("steps", []):
                targets = step.get("targets", {})
                if targets.get("pace_min_km"):
                    print(f"    {s['day']} {s['type']} - {step.get('type')}: {targets['pace_min_km']}/km")
                for sub in step.get("steps", []):
                    sub_targets = sub.get("targets", {})
                    if sub_targets.get("pace_min_km"):
                        print(f"    {s['day']} {s['type']} - {sub.get('type')}: {sub_targets['pace_min_km']}/km")

    # Phase 5: Test ongoing conversation
    print("\n\n  === ONGOING CONVERSATION TEST ===")
    conv_results = test_ongoing_conversation([
        "Kannst du mir mein Intervalltraining für diese Woche genauer erklären?",
        "Was wäre ein gutes Zwift Workout für mich diese Woche?",
        "Wie soll ich mich auf den Triathlon vorbereiten wenn mein Hauptfokus der HM ist?",
    ])

    # Check if responses are substantive
    print("\n\n  === CONVERSATION QUALITY CHECKS ===")
    for r in conv_results:
        resp = r.get("response", "")
        if resp:
            word_count = len(resp.split())
            has_deflection = any(p in resp.lower() for p in [
                "sobald ich", "brauche ich noch", "fehlt mir noch",
                "once you provide", "i still need",
            ])
            print(f"  Q: {r['question'][:60]}...")
            print(f"    Length: {word_count} words, Deflecting: {has_deflection}")
        else:
            print(f"  Q: {r['question'][:60]}... -> ERROR: {r['error']}")


if __name__ == "__main__":
    main()
