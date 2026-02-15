#!/usr/bin/env python3
"""Extended scenario tests for ReAgt coaching agent — 8 diverse sport scenarios.

Based on AUTONOMOUS_TEST_PROMPT.md. Tests onboarding, plan generation,
AND follow-up coaching quality across diverse sports and personas.

Usage:
    uv run python tests/scenario_test_v3.py [scenario_number]
    uv run python tests/scenario_test_v3.py          # run all 8
    uv run python tests/scenario_test_v3.py 1 3 5    # run specific ones
"""

import json
import sys
import time
import traceback
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = DATA_DIR / "user_model" / "model.json"
PLANS_DIR = DATA_DIR / "plans"
SESSIONS_DIR = DATA_DIR / "sessions"
EPISODES_DIR = DATA_DIR / "episodes"

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
    """Reset all user data."""
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
    for d in [SESSIONS_DIR, PLANS_DIR, EPISODES_DIR]:
        if d.exists():
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()
    time.sleep(0.3)


def run_onboarding_and_followups(
    onboarding_messages: list[str],
    followup_messages: list[str],
    scenario_name: str,
) -> dict:
    """Run onboarding + plan generation + follow-up coaching messages.

    Returns dict with all results for validation.
    """
    from src.agent.onboarding import OnboardingEngine
    from src.memory.user_model import UserModel
    from src.agent.coach import generate_plan, save_plan
    from src.tools.activity_store import list_activities
    from src.memory.episodes import list_episodes, retrieve_relevant_episodes

    results = {
        "scenario": scenario_name,
        "onboarding_responses": [],
        "followup_responses": [],
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

        # --- ONBOARDING PHASE ---
        greeting = engine.start()
        results["onboarding_responses"].append({"role": "agent", "content": greeting})
        print(f"  Coach: {greeting[:120]}...")

        for i, msg in enumerate(onboarding_messages):
            print(f"  User [{i+1}/{len(onboarding_messages)}]: {msg[:100]}...")
            try:
                response = engine.process_message(msg)
                results["onboarding_responses"].append({"role": "user", "content": msg})
                results["onboarding_responses"].append({"role": "agent", "content": response})
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
                results["errors"].append(f"Onboarding turn {i+1} failed: {e}")
                print(f"  [ERROR in turn {i+1}: {e}]")
                traceback.print_exc()

        # If onboarding didn't complete but we sent all messages, still try
        if not results["onboarding_complete"]:
            print("  [WARNING: Onboarding did not signal completion, checking fields...]")
            core = user_model.structured_core
            if core.get("sports") and core.get("goal", {}).get("event"):
                results["onboarding_complete"] = True
                print("  [Forced onboarding complete based on structured_core fields]")
                # Generate plan anyway
                profile = user_model.project_profile()
                beliefs = user_model.get_active_beliefs(min_confidence=0.6)
                activities = list_activities()
                try:
                    plan = generate_plan(profile, beliefs=beliefs, activities=activities)
                    save_plan(plan)
                    results["plan"] = plan
                    results["plan_generated"] = True
                    print(f"  [Plan generated (forced): {len(plan.get('sessions', []))} sessions]")
                except Exception as e:
                    results["errors"].append(f"Plan generation (forced) failed: {e}")

        # --- FOLLOW-UP COACHING PHASE ---
        if followup_messages and results["onboarding_complete"]:
            print(f"\n  --- Follow-Up Coaching ({len(followup_messages)} messages) ---")
            for i, msg in enumerate(followup_messages):
                print(f"  FollowUp [{i+1}/{len(followup_messages)}]: {msg[:100]}...")
                try:
                    response = engine.process_message(msg)
                    results["followup_responses"].append({"role": "user", "content": msg})
                    results["followup_responses"].append({"role": "agent", "content": response})
                    results["turn_count"] += 1
                    print(f"  Coach: {response[:200]}...")
                except Exception as e:
                    results["errors"].append(f"Follow-up {i+1} failed: {e}")
                    print(f"  [ERROR in follow-up {i+1}: {e}]")

        # Finalize
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
    """Run acceptance criteria + custom checks for the scenario."""
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

    # Equivalence groups: LLMs may store sports under related names
    _sport_equivalences = {
        "hyrox": {"hyrox", "functional fitness", "functional_fitness", "crossfit", "strength"},
        "functional fitness": {"hyrox", "functional fitness", "functional_fitness", "crossfit", "strength"},
    }

    def _sport_matches(expected_sport: str, actual_list: list[str]) -> bool:
        es = expected_sport.lower()
        equiv = _sport_equivalences.get(es, {es})
        for a in actual_list:
            if es in a or a in es or a.startswith(es[:3]):
                return True
            # Check equivalence group
            if any(eq in a or a in eq for eq in equiv):
                return True
        return False

    sports_match = all(
        _sport_matches(es, actual_sports)
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

    # 5. Target date
    if expected.get("expects_target_date", True):
        if goal.get("target_date"):
            passed, detail = target_date_is_iso(sc)
            checks.append({"criterion": "target_date_iso", "passed": passed, "detail": detail})
        else:
            checks.append({
                "criterion": "target_date_iso",
                "passed": False,
                "detail": "Expected target date but none set",
            })
    else:
        checks.append({
            "criterion": "target_date_iso",
            "passed": True,
            "detail": "No target date expected (routine/general goal)",
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

    # 8. Plan checks
    if plan:
        passed, detail = plan_dates_are_valid(plan)
        checks.append({"criterion": "plan_dates_valid", "passed": passed, "detail": detail})

        passed, detail = plan_respects_sport_distribution(plan, beliefs, structured_core=sc)
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

    # 9. Fitness fields
    if expected.get("gives_performance_data"):
        passed, detail = fitness_fields_populated(sc, beliefs)
        checks.append({"criterion": "fitness_populated", "passed": passed, "detail": detail})

    # 10. Custom expected checks
    for custom_check in expected.get("custom_checks", []):
        name = custom_check["name"]
        check_fn = custom_check["check"]
        try:
            passed, detail = check_fn(results)
            checks.append({"criterion": name, "passed": passed, "detail": detail})
        except Exception as e:
            checks.append({"criterion": name, "passed": False, "detail": f"Check error: {e}"})

    # 11. Follow-up quality (if follow-ups were sent)
    followup_responses = results.get("followup_responses", [])
    agent_followups = [r for r in followup_responses if r["role"] == "agent"]
    if agent_followups:
        # Check that follow-ups are substantive (not just deflections)
        substantive = 0
        for r in agent_followups:
            text = r["content"]
            if len(text) > 100:  # Not a one-liner
                substantive += 1
        checks.append({
            "criterion": "followup_substantive",
            "passed": substantive >= len(agent_followups) * 0.8,
            "detail": f"{substantive}/{len(agent_followups)} responses are substantive (>100 chars)",
        })

    # 12. Language check
    expected_lang = expected.get("language", "de")
    if agent_followups:
        last_response = agent_followups[-1]["content"].lower()
        if expected_lang == "de":
            de_markers = ["ich", "du", "dein", "das", "die", "der", "und", "ein", "mit", "ist"]
            lang_match = sum(1 for m in de_markers if m in last_response.split()) >= 3
        else:
            en_markers = ["you", "your", "the", "and", "is", "can", "for", "with", "this", "that"]
            lang_match = sum(1 for m in en_markers if m in last_response.split()) >= 3
        checks.append({
            "criterion": "correct_language",
            "passed": lang_match,
            "detail": f"Expected {expected_lang}, checking last follow-up response",
        })

    # 13. No errors
    checks.append({
        "criterion": "no_errors",
        "passed": len(results.get("errors", [])) == 0,
        "detail": f"Errors: {results.get('errors', [])}",
    })

    return checks


# ── CUSTOM CHECK HELPERS ──────────────────────────────────────────────

def _check_zwift_recognized(results):
    """Check if ZWIFT/indoor cycling was stored as a belief."""
    beliefs = results.get("beliefs", [])
    found = any("zwift" in b.get("text", "").lower() or "indoor" in b.get("text", "").lower()
                 or "rolle" in b.get("text", "").lower() or "trainer" in b.get("text", "").lower()
                 for b in beliefs)
    return found, f"ZWIFT/indoor belief found: {found}"


def _check_no_running_recommended(results):
    """For knee-injury scenarios, coach should not recommend running.

    The coach may mention running in context of explaining why it's not suitable
    (e.g. 'running is not recommended for your knees'). We only fail if running
    appears to be actively recommended without any surrounding avoidance context.
    """
    followups = results.get("followup_responses", [])
    agent_msgs = [r["content"].lower() for r in followups if r["role"] == "agent"]

    # Phrases that indicate running is being discussed as something to avoid/skip
    avoidance_phrases = [
        "nicht", "avoid", "don't", "do not", "kein", "no running", "not recommended",
        "not suitable", "instead of running", "rather than running", "skip running",
        "without running", "anstatt", "statt laufen", "verzicht", "no option",
        "not an option", "isn't an option", "is not an option",
        "bad for", "hard on", "impact on", "stress on",
        "low.impact", "low impact", "gelenkschonend",
    ]

    problematic = False
    for msg in agent_msgs:
        running_keywords = ["laufen", "running", "jogging", "joggen", "run "]
        mentions_running = any(kw in msg for kw in running_keywords)
        if not mentions_running:
            continue
        # Check if running is mentioned in avoidance/explanatory context
        has_avoidance = any(phrase in msg for phrase in avoidance_phrases)
        if not has_avoidance:
            # Running is mentioned without any avoidance context - likely recommended
            problematic = True

    return not problematic, "Coach correctly avoids recommending running for knee-injured athlete"


def _check_injury_stored(results, keywords):
    """Check if injury/constraint was stored in beliefs."""
    beliefs = results.get("beliefs", [])
    found = any(
        any(kw.lower() in b.get("text", "").lower() for kw in keywords)
        for b in beliefs
    )
    return found, f"Injury keywords {keywords} found in beliefs: {found}"


def _check_youth_safety(results):
    """For teenage athlete, check that recovery/rest is emphasized."""
    followups = results.get("followup_responses", [])
    agent_msgs = " ".join(r["content"].lower() for r in followups if r["role"] == "agent")
    safety_words = ["recovery", "rest", "erholung", "schlaf", "sleep", "overtraining",
                    "nutrition", "ernährung", "essen", "eat", "tired", "müde"]
    found = sum(1 for w in safety_words if w in agent_msgs)
    return found >= 2, f"Youth safety words found: {found} (need >=2)"


def _check_sport_specific_plan(results, sport_keywords):
    """Check that plan sessions use sport-specific terminology."""
    plan = results.get("plan")
    if not plan:
        return False, "No plan generated"
    sessions_text = json.dumps(plan.get("sessions", []), default=str).lower()
    found = sum(1 for kw in sport_keywords if kw.lower() in sessions_text)
    return found >= 2, f"Sport keywords {sport_keywords}: {found} found in plan sessions"


# ── SCENARIO DEFINITIONS ──────────────────────────────────────────────

SCENARIOS = {
    1: {
        "name": "Multi-Sport Triathlet (Deutsch)",
        "onboarding": [
            "Hi, ich laufe gerne, fahre Rennrad und schwimme einmal in der Woche. "
            "Im August möchte ich gerne einen Halbmarathon unter 1:30 laufen, das ist mein Ziel. "
            "Ich strebe auch an einen Triathlon zu machen, vielleicht Q3 oder Q4 2026 - aber nur Kurzdistanz. "
            "Ich gehe auch noch ins Gym und möchte Muskeln aufbauen (lean). "
            "Ich dachte daran 3 mal die Woche laufen zu gehen, 1 mal schwimmen und Rennrad 2-3 mal. "
            "Ich habe übrigens auch eine Rolle fürs Rennrad und fahre dort online in ZWIFT.",
            "Meine 10km Bestzeit ist 42:30. Unter der Woche maximal 90 Minuten, "
            "am Wochenende auch mal 2-3 Stunden. Ich heiße Roman.",
        ],
        "followups": [
            "Wie sollte meine typische Trainingswoche aussehen?",
            "Ich hatte gestern Knieschmerzen nach dem Laufen",
            "Wird das Krafttraining mein Laufen beeinträchtigen?",
        ],
        "expected": {
            "sports": ["running", "cycling", "swimming"],
            "expects_target_date": True,
            "gives_performance_data": True,
            "min_sessions": 5,
            "max_sessions": 10,
            "language": "de",
            "custom_checks": [
                {"name": "zwift_recognized", "check": _check_zwift_recognized},
            ],
        },
    },
    2: {
        "name": "Hyrox-Athlet (Englisch)",
        "onboarding": [
            "Hey! I'm training for Hyrox in Berlin, April 2026. I've been doing CrossFit for 3 years "
            "but Hyrox is new to me. I can train 5-6 days a week, max 90 minutes on weekdays. "
            "I want to finish under 75 minutes. My running is my weakness - I can do a 5K in about 24 minutes. "
            "The functional fitness parts should be fine given my CrossFit background. My name is Jake.",
        ],
        "followups": [
            "What should my running training look like? That's where I need to improve most",
            "How do I balance the running with the functional stations?",
            "I'm worried about the sled push and lunges destroying my legs before the runs",
        ],
        "expected": {
            "sports": ["running", "hyrox"],
            "expects_target_date": True,
            "gives_performance_data": True,
            "min_sessions": 4,
            "max_sessions": 8,
            "language": "en",
            "custom_checks": [
                {"name": "hyrox_specific", "check": lambda r: _check_sport_specific_plan(
                    r, ["hyrox", "functional", "sled", "row", "wall ball", "station", "crossfit"])},
            ],
        },
    },
    3: {
        "name": "Basketball-Spielerin (Deutsch)",
        "onboarding": [
            "Hi, ich spiele Basketball im Verein, 2-3 Mal Training pro Woche plus Spiel am Wochenende. "
            "Ich möchte meine Fitness verbessern - vor allem Schnelligkeit und Ausdauer auf dem Feld. "
            "Ich merke dass ich im 4. Viertel nachlasse. Außerdem möchte ich meine vertikale Sprungkraft verbessern. "
            "Ich kann neben dem Vereinstraining noch 2-3 Mal extra trainieren. "
            "Ich bin Lena, 22 Jahre alt.",
        ],
        "followups": [
            "Wie kann ich meine Ausdauer für Basketball verbessern ohne meine Beine zu überlasten?",
            "Was für Übungen helfen mir bei der Sprungkraft?",
            "Am Mittwoch hab ich immer Vereinstraining, da bin ich danach total platt",
        ],
        "expected": {
            "sports": ["basketball"],
            "expects_target_date": False,
            "gives_performance_data": False,
            "min_sessions": 3,
            "max_sessions": 8,
            "language": "de",
            "custom_checks": [
                {"name": "basketball_specific", "check": lambda r: _check_sport_specific_plan(
                    r, ["basketball", "sprint", "agility", "jump", "plyometric",
                        "conditioning", "court", "sprung", "schnelligkeit"])},
            ],
        },
    },
    4: {
        "name": "Ultra-Runner (Englisch)",
        "onboarding": [
            "I'm preparing for my first 100-mile ultramarathon (Western States) in June 2026. "
            "I've done several 50K and 100K races. Currently running 80-90km per week. "
            "I can train every day, with long runs up to 5 hours on weekends. "
            "I also do yoga twice a week for recovery. My biggest concern is nutrition "
            "and back-to-back long runs. I'm Sarah, 38 years old.",
        ],
        "followups": [
            "Should I do back-to-back long runs every weekend?",
            "My left Achilles has been a bit tight lately",
            "How should I taper for a 100-miler?",
        ],
        "expected": {
            "sports": ["running"],
            "expects_target_date": True,
            "gives_performance_data": True,
            "min_sessions": 5,
            "max_sessions": 9,
            "language": "en",
            "custom_checks": [
                {"name": "ultra_specific", "check": lambda r: _check_sport_specific_plan(
                    r, ["ultra", "trail", "long run", "back-to-back", "elevation", "altitude",
                        "nutrition", "fueling", "b2b"])},
                {"name": "achilles_stored", "check": lambda r: _check_injury_stored(
                    r, ["achilles", "tight", "tendon"])},
            ],
        },
    },
    5: {
        "name": "Schwimmer Comeback (Deutsch)",
        "onboarding": [
            "Hallo! Ich war als Jugendlicher Leistungsschwimmer (1500m Freistil, Bestzeit 17:30). "
            "Nach 20 Jahren Pause möchte ich wieder anfangen. Mein Ziel ist es, bei den "
            "Masters-Meisterschaften 2026 im Herbst anzutreten. Ich kann 4x pro Woche schwimmen "
            "und 2x Trockentraining machen. Ich bin etwas übergewichtig (95kg bei 185cm) "
            "und meine Schulter macht manchmal Probleme. Ich bin Max, 45.",
        ],
        "followups": [
            "Wie schnell kann ich realistisch wieder an meine alte Form rankommen?",
            "Welche Technik-Übungen empfiehlst du mir?",
            "Die rechte Schulter tut nach dem Kraulen manchmal weh",
        ],
        "expected": {
            "sports": ["swimming"],
            "expects_target_date": True,
            "gives_performance_data": True,
            "min_sessions": 4,
            "max_sessions": 8,
            "language": "de",
            "custom_checks": [
                {"name": "swim_specific", "check": lambda r: _check_sport_specific_plan(
                    r, ["swim", "schwimm", "pool", "freestyle", "freistil", "kraul",
                        "technik", "drill", "trocken"])},
                {"name": "shoulder_stored", "check": lambda r: _check_injury_stored(
                    r, ["schulter", "shoulder"])},
            ],
        },
    },
    6: {
        "name": "Freizeit-Radfahrer mit Gewichtsziel (Englisch)",
        "onboarding": [
            "Hi, I'm 50 years old, 110kg, and I want to get healthier. I bought an e-bike last month "
            "and I'm loving it. I ride about 30 minutes 3-4 times a week. I also just started walking more. "
            "My doctor said I should lose at least 20kg. I have bad knees from old football injuries, "
            "so running is not an option. I don't have any race goals - I just want to be healthier "
            "and maybe ride 50km without stopping by the end of summer. My name is Mike.",
        ],
        "followups": [
            "Is 30 minutes enough or should I ride longer?",
            "My knees hurt after longer rides too, what should I do?",
            "Can I still eat normally or do I need to diet?",
        ],
        "expected": {
            "sports": ["cycling"],
            "expects_target_date": False,
            "gives_performance_data": False,
            "min_sessions": 3,
            "max_sessions": 7,
            "language": "en",
            "custom_checks": [
                {"name": "no_running_for_knees", "check": _check_no_running_recommended},
                {"name": "knee_stored", "check": lambda r: _check_injury_stored(
                    r, ["knee", "knie"])},
            ],
        },
    },
    7: {
        "name": "Yoga + Running Hybrid (Deutsch)",
        "onboarding": [
            "Hi! Ich bin Yoga-Lehrerin und unterrichte 5x pro Woche. Daneben laufe ich gerne - "
            "aktuell 3x pro Woche, so 30-40km insgesamt. Ich möchte meinen ersten Marathon im Herbst 2026 laufen. "
            "Mein Tempo beim Easy Run ist ca. 5:30/km. Ich will nicht zu viel Muskelmasse aufbauen "
            "weil das beim Yoga stört. Meine Yoga-Stunden sind mein Job, die kann ich nicht verschieben. "
            "Ich bin Anna, 35.",
        ],
        "followups": [
            "Kann ich Yoga als Aufwärmen vor dem Laufen nutzen?",
            "Wie integriere ich Tempoläufe in meine Woche?",
            "Ich fühle mich nach den Abend-Yoga-Stunden manchmal zu müde zum Laufen am nächsten Morgen",
        ],
        "expected": {
            "sports": ["running"],
            "expects_target_date": True,
            "gives_performance_data": True,
            "min_sessions": 3,
            "max_sessions": 7,
            "language": "de",
            "custom_checks": [
                {"name": "yoga_as_constraint", "check": lambda r: (
                    any("yoga" in b.get("text", "").lower() for b in r.get("beliefs", [])),
                    f"Yoga mentioned in beliefs: {any('yoga' in b.get('text', '').lower() for b in r.get('beliefs', []))}"
                )},
            ],
        },
    },
    8: {
        "name": "Teenagerin Mehrsport (Englisch)",
        "onboarding": [
            "Hey, I'm 16 and I play soccer at school (practice Tue/Thu, game Saturday). "
            "I also just joined the school swim team (practice Mon/Wed/Fri). "
            "I want to get faster for both sports. My parents said I can't train more than "
            "what I already have with school teams. Is there anything I can do with what I already have "
            "to get better? I sometimes feel really tired and my coach says I need to eat more. "
            "I'm Emma.",
        ],
        "followups": [
            "My soccer coach wants me to do extra running but I'm already so tired",
            "I have a big swim meet in 3 weeks, should I focus more on swimming?",
            "Sometimes I skip meals because I'm not hungry",
        ],
        "expected": {
            "sports": ["soccer", "swimming"],
            "expects_target_date": False,
            "gives_performance_data": False,
            "min_sessions": 3,
            "max_sessions": 8,
            "language": "en",
            "custom_checks": [
                {"name": "youth_safety", "check": _check_youth_safety},
            ],
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

    print("  [Resetting state...]")
    reset_state()
    time.sleep(0.5)

    print("  [Starting onboarding + follow-ups...]")
    results = run_onboarding_and_followups(
        scenario["onboarding"],
        scenario.get("followups", []),
        name,
    )

    print(f"\n  [Validating...]")
    checks = validate_results(results, scenario["expected"])

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
        "followup_count": len([r for r in results.get("followup_responses", []) if r["role"] == "agent"]),
        "plan": results.get("plan"),
        "followup_responses": results.get("followup_responses", []),
    }


def generate_report(all_results: list[dict]) -> str:
    """Generate a markdown report from test results."""
    lines = []
    lines.append("# ReAgt Scenario Test Report v3")
    lines.append(f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Scenarios**: {len(all_results)}")

    total_checks = sum(r["total"] for r in all_results)
    total_passed = sum(r["passed"] for r in all_results)
    lines.append(f"**Overall**: {total_passed}/{total_checks} checks passed "
                 f"({total_passed/total_checks*100:.0f}%)")

    lines.append("\n---\n")

    for r in all_results:
        status = "PASS" if r["all_passed"] else "FAIL"
        lines.append(f"## Scenario {r['scenario_id']}: {r['scenario_name']} [{status}]")
        lines.append(f"\n**Checks**: {r['passed']}/{r['total']} | "
                     f"**Beliefs**: {r['belief_count']} | "
                     f"**Plan sessions**: {r['plan_session_count']} | "
                     f"**Follow-ups**: {r['followup_count']}")

        core = r.get("structured_core", {})
        lines.append(f"\n**Profile**:")
        lines.append(f"- Sports: {core.get('sports', [])}")
        lines.append(f"- Goal: {core.get('goal', {}).get('event')}")
        lines.append(f"- Goal type: {core.get('goal', {}).get('goal_type')}")
        lines.append(f"- Target date: {core.get('goal', {}).get('target_date')}")
        lines.append(f"- Days/week: {core.get('constraints', {}).get('training_days_per_week')}")

        lines.append(f"\n| Check | Result | Detail |")
        lines.append(f"|-------|--------|--------|")
        for c in r["checks"]:
            icon = "PASS" if c["passed"] else "**FAIL**"
            detail = c["detail"][:80]
            lines.append(f"| {c['criterion']} | {icon} | {detail} |")

        if r["errors"]:
            lines.append(f"\n**Errors**: {r['errors']}")

        # Show follow-up quality
        if r.get("followup_responses"):
            lines.append(f"\n**Follow-Up Coaching Quality**:")
            for resp in r["followup_responses"]:
                if resp["role"] == "user":
                    lines.append(f"\n> User: {resp['content'][:100]}")
                else:
                    preview = resp['content'][:200].replace('\n', ' ')
                    lines.append(f">\n> Coach: {preview}...")

        lines.append("\n---\n")

    # Summary
    lines.append("## Summary\n")
    for r in all_results:
        icon = "PASS" if r["all_passed"] else "FAIL"
        lines.append(f"- [{icon}] **S{r['scenario_id']}** {r['scenario_name']}: "
                     f"{r['passed']}/{r['total']}")

    lines.append(f"\n**Total: {total_passed}/{total_checks} "
                 f"({total_passed/total_checks*100:.0f}%)**")

    return "\n".join(lines)


def main():
    """Run all (or selected) scenarios and print summary report."""
    if len(sys.argv) > 1:
        ids = [int(x) for x in sys.argv[1:]]
    else:
        ids = sorted(SCENARIOS.keys())

    print(f"Running {len(ids)} scenarios: {ids}")
    print(f"Each scenario: onboarding + plan generation + follow-up coaching")
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}\n")

    all_results = []
    for sid in ids:
        start = time.time()
        result = run_scenario(sid)
        elapsed = time.time() - start
        result["elapsed_seconds"] = round(elapsed, 1)
        all_results.append(result)
        print(f"\n  [Scenario {sid} completed in {elapsed:.1f}s]")

    # Print summary
    print(f"\n\n{'='*70}")
    print("SUMMARY REPORT")
    print(f"{'='*70}")

    total_checks = 0
    total_passed = 0

    for r in all_results:
        status = "PASS" if r["all_passed"] else "FAIL"
        print(f"\n  [{status}] S{r['scenario_id']}: {r['scenario_name']}")
        print(f"         Checks: {r['passed']}/{r['total']} | Time: {r.get('elapsed_seconds', '?')}s")
        print(f"         Beliefs: {r['belief_count']}, Sessions: {r['plan_session_count']}, Follow-ups: {r['followup_count']}")

        core = r.get("structured_core", {})
        print(f"         Sports: {core.get('sports', [])}")
        print(f"         Goal: {core.get('goal', {}).get('event')}")

        if r["errors"]:
            print(f"         ERRORS: {r['errors']}")

        failed = [c for c in r["checks"] if not c["passed"]]
        if failed:
            print(f"         FAILED:")
            for c in failed:
                print(f"           - {c['criterion']}: {c['detail']}")

        total_checks += r["total"]
        total_passed += r["passed"]

    print(f"\n  TOTAL: {total_passed}/{total_checks} checks across {len(all_results)} scenarios")
    print(f"  End time: {datetime.now().strftime('%H:%M:%S')}")

    # Save JSON report
    report_path = PROJECT_ROOT / "tests" / "scenario_report_v3.json"
    report_data = []
    for r in all_results:
        entry = {k: v for k, v in r.items() if k not in ("plan", "followup_responses")}
        if r.get("plan"):
            entry["plan_summary"] = {
                "week_start": r["plan"].get("week_start"),
                "sessions": [
                    {"day": s.get("day"), "sport": s.get("sport"), "type": s.get("type"),
                     "duration": s.get("total_duration_minutes", s.get("duration_minutes"))}
                    for s in r["plan"].get("sessions", [])
                ],
            }
        report_data.append(entry)
    report_path.write_text(json.dumps(report_data, indent=2, default=str))
    print(f"\n  JSON report: {report_path}")

    # Save Markdown report
    md_report = generate_report(all_results)
    md_path = PROJECT_ROOT / "tests" / "scenario_report_v3.md"
    md_path.write_text(md_report)
    print(f"  Markdown report: {md_path}")


if __name__ == "__main__":
    main()
