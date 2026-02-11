"""Autonomous test harness for ReAgt coaching agent.

Runs the agent programmatically through onboarding + ongoing sessions,
evaluates against acceptance criteria, and produces structured reports.

Each run uses a fresh user model and a different test scenario.
"""

import json
import os
import sys
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env if present
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

from src.agent.onboarding import OnboardingEngine
from src.agent.conversation import ConversationEngine
from src.agent.coach import generate_plan, save_plan
from src.memory.user_model import UserModel
from tests.acceptance_criteria import (
    plan_respects_sport_distribution,
    plan_dates_are_valid,
    target_date_is_iso,
    fitness_fields_populated,
    ongoing_session_answers_questions,
    beliefs_have_correct_categories,
    structured_core_constraints_realistic,
    plan_has_specific_targets,
)


# ── Test Scenarios ────────────────────────────────────────────────

SCENARIOS = [
    {
        "name": "roman_triathlete",
        "description": "Multi-sport athlete: running focus + cycling + swimming + gym",
        "onboarding_messages": [
            (
                "Hi, ich Laufe gerne, fahre Rennrad und Schwimme einmal in der Woche. "
                "Im August möchte ich gerne einen Halbmarathon unter 1:30 laufen, das ist mein Ziel. "
                "Ich strebe auch an einen Triathlon zu machen vlt Q3 oder Q4 2026 - aber nur Kurzdistanz. "
                "Ich gehe auch noch ins Gym und möchte Muskeln aufbauen (lean). "
                "Ich dachte daran 3 mal die Woche laufen zu gehen, 1 mal schwimmen und Rennrad 2-3 mal. "
                "Ich habe übrigens auch eine Rolle fürs Rennrad und fahre dort online in ZWIFT."
            ),
        ],
        "follow_up_messages": [
            "Unter der Woche kann ich maximal 90 Minuten trainieren, am Wochenende 2-3 Stunden.",
            "Meine aktuelle 10km Bestzeit ist 42:30. Was wären meine Herzfrequenzzonen für das Training?",
        ],
        "ongoing_messages": [
            "Kannst du mir sagen was meine HR Zonen sein sollten und welche ZWIFT Workouts gut wären?",
        ],
        "ongoing_keywords": ["zone", "zwift", "herzfrequenz"],
    },
    {
        "name": "marathon_runner",
        "description": "Dedicated marathon runner, minimal cross-training",
        "onboarding_messages": [
            (
                "Hey! I'm training for the Berlin Marathon in September 2026. "
                "My goal is sub 3:30. I currently run about 40km per week, "
                "my recent half marathon was 1:42. I can train 5 days a week, "
                "max 2 hours on weekdays and 3 hours on weekends. "
                "I also do yoga twice a week for flexibility."
            ),
        ],
        "follow_up_messages": [
            "My resting HR is 52 and my max HR is around 185. I usually run easy at around 5:15-5:30/km.",
        ],
        "ongoing_messages": [
            "What should my long run pace be and how should I structure my weekly mileage buildup?",
        ],
        "ongoing_keywords": ["long run", "pace", "mileage", "km"],
    },
    {
        "name": "cycling_focused",
        "description": "Cyclist wanting to add running",
        "onboarding_messages": [
            (
                "Ich bin hauptsächlich Rennradfahrer, FTP liegt bei 260W bei 75kg. "
                "Ich möchte im Oktober 2026 an einem Radmarathon teilnehmen (Ötztaler Radmarathon). "
                "Nebenbei möchte ich auch mit Laufen anfangen, 2x pro Woche. "
                "Ich kann jeden Tag trainieren, unter der Woche 1-1.5h, am Wochenende bis zu 5 Stunden auf dem Rad. "
                "Mein Ruhepuls ist 48, max HR 190."
            ),
        ],
        "follow_up_messages": [
            "Ich habe ein Wahoo KICKR und nutze TrainerRoad für strukturierte Intervalle.",
        ],
        "ongoing_messages": [
            "Wie sollte ich meine Trainingswochen periodisieren mit dem Mix aus Rad und Laufen?",
        ],
        "ongoing_keywords": ["periodisierung", "rad", "laufen", "woche"],
    },
]


# ── Test Runner ──────────────────────────────────────────────────

class TestRunner:
    """Runs a single test scenario and evaluates results."""

    def __init__(self, scenario: dict, work_dir: Path):
        self.scenario = scenario
        self.work_dir = work_dir
        self.results: dict = {
            "scenario": scenario["name"],
            "description": scenario["description"],
            "timestamp": datetime.now().isoformat(),
            "criteria": {},
            "beliefs": [],
            "structured_core": {},
            "plan": {},
            "ongoing_response": "",
            "errors": [],
        }

    def run(self) -> dict:
        """Execute full test: onboarding → plan → ongoing → evaluate."""
        print(f"\n{'='*60}")
        print(f"SCENARIO: {self.scenario['name']}")
        print(f"  {self.scenario['description']}")
        print(f"{'='*60}")

        try:
            # 1. Fresh user model in temp directory
            model_dir = self.work_dir / "user_model"
            model_dir.mkdir(parents=True, exist_ok=True)
            sessions_dir = self.work_dir / "sessions"
            sessions_dir.mkdir(parents=True, exist_ok=True)
            plans_dir = self.work_dir / "plans"
            plans_dir.mkdir(parents=True, exist_ok=True)

            user_model = UserModel(data_dir=model_dir)

            # 2. Onboarding
            print("\n--- ONBOARDING ---")
            engine = OnboardingEngine(
                user_model=user_model,
                sessions_dir=sessions_dir,
                data_dir=self.work_dir,
            )
            greeting = engine.start()
            print(f"  Coach: {greeting[:80]}...")

            for msg in self.scenario["onboarding_messages"]:
                print(f"  User: {msg[:80]}...")
                response = engine.process_message(msg)
                print(f"  Coach: {response[:120]}...")

            # Follow-up messages (still onboarding)
            for msg in self.scenario.get("follow_up_messages", []):
                print(f"  User: {msg[:80]}...")
                response = engine.process_message(msg)
                print(f"  Coach: {response[:120]}...")

            # End onboarding session
            engine.end_session()
            user_model.save()

            # 3. Capture state after onboarding
            beliefs = user_model.get_active_beliefs()
            structured_core = user_model.structured_core
            self.results["beliefs"] = [
                {"text": b["text"], "category": b["category"], "confidence": b["confidence"]}
                for b in beliefs
            ]
            self.results["structured_core"] = structured_core

            print(f"\n  Beliefs extracted: {len(beliefs)}")
            print(f"  Structured core: sports={structured_core.get('sports')}, "
                  f"goal={structured_core.get('goal', {}).get('event')}, "
                  f"target_date={structured_core.get('goal', {}).get('target_date')}")

            # 4. Generate plan
            print("\n--- PLAN GENERATION ---")
            profile = user_model.project_profile()
            plan = generate_plan(profile, beliefs=beliefs)
            self.results["plan"] = plan

            # Save plan
            plan_path = plans_dir / f"plan_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json"
            plan_path.write_text(json.dumps(plan, indent=2))

            sessions = plan.get("sessions", [])
            sports_in_plan = {}
            for s in sessions:
                sport = s.get("sport", "unknown").lower()
                sports_in_plan[sport] = sports_in_plan.get(sport, 0) + 1
            print(f"  Sessions: {len(sessions)}")
            print(f"  Sports: {sports_in_plan}")
            print(f"  Week start: {plan.get('week_start', 'MISSING')}")

            # 5. Ongoing session
            print("\n--- ONGOING SESSION ---")
            conv_engine = ConversationEngine(
                user_model=user_model,
                sessions_dir=sessions_dir,
                data_dir=self.work_dir,
            )
            conv_engine.start_session()

            ongoing_response = ""
            for msg in self.scenario.get("ongoing_messages", []):
                print(f"  User: {msg[:80]}...")
                ongoing_response = conv_engine.process_message(msg)
                print(f"  Coach: {ongoing_response[:150]}...")

            self.results["ongoing_response"] = ongoing_response
            conv_engine.end_session()

            # 6. Evaluate
            print("\n--- EVALUATION ---")
            self._evaluate(beliefs, structured_core, plan, ongoing_response)

        except Exception as e:
            self.results["errors"].append(f"Runtime error: {type(e).__name__}: {e}")
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()

        return self.results

    def _evaluate(self, beliefs, structured_core, plan, ongoing_response):
        """Run all acceptance criteria."""
        criteria = {}

        # 1. Plan sport distribution
        passed, detail = plan_respects_sport_distribution(plan, beliefs)
        criteria["plan_sport_distribution"] = {"passed": passed, "detail": detail}
        self._print_result("Plan sport distribution", passed, detail)

        # 2. Plan dates valid
        passed, detail = plan_dates_are_valid(plan)
        criteria["plan_dates_valid"] = {"passed": passed, "detail": detail}
        self._print_result("Plan dates valid", passed, detail)

        # 3. Target date ISO
        passed, detail = target_date_is_iso(structured_core)
        criteria["target_date_iso"] = {"passed": passed, "detail": detail}
        self._print_result("Target date ISO", passed, detail)

        # 4. Fitness populated
        belief_dicts_for_fitness = [
            {"text": b["text"], "category": b["category"], "confidence": b["confidence"]}
            for b in beliefs
        ]
        passed, detail = fitness_fields_populated(structured_core, beliefs=belief_dicts_for_fitness)
        criteria["fitness_populated"] = {"passed": passed, "detail": detail}
        self._print_result("Fitness populated", passed, detail)

        # 5. Ongoing answers questions
        keywords = self.scenario.get("ongoing_keywords", [])
        if keywords and ongoing_response:
            passed, detail = ongoing_session_answers_questions(ongoing_response, keywords)
        else:
            passed, detail = True, "No ongoing keywords to check"
        criteria["ongoing_answers_questions"] = {"passed": passed, "detail": detail}
        self._print_result("Ongoing answers questions", passed, detail)

        # 6. Belief categories
        belief_dicts = [
            {"text": b["text"], "category": b["category"], "confidence": b["confidence"]}
            for b in beliefs
        ]
        passed, detail = beliefs_have_correct_categories(belief_dicts)
        criteria["belief_categories"] = {"passed": passed, "detail": detail}
        self._print_result("Belief categories", passed, detail)

        # 7. Constraints realistic
        passed, detail = structured_core_constraints_realistic(structured_core)
        criteria["constraints_realistic"] = {"passed": passed, "detail": detail}
        self._print_result("Constraints realistic", passed, detail)

        # 8. Plan specific targets
        passed, detail = plan_has_specific_targets(plan)
        criteria["plan_specific_targets"] = {"passed": passed, "detail": detail}
        self._print_result("Plan specific targets", passed, detail)

        self.results["criteria"] = criteria

        # Summary
        passed_count = sum(1 for c in criteria.values() if c["passed"])
        total = len(criteria)
        print(f"\n  RESULT: {passed_count}/{total} criteria passed")

    def _print_result(self, name, passed, detail):
        status = "PASS" if passed else "FAIL"
        symbol = "+" if passed else "x"
        print(f"  [{symbol}] {name}: {status} - {detail[:100]}")


def run_scenario(scenario_name: str | None = None, scenario_index: int = 0) -> dict:
    """Run a specific scenario or default to first."""
    if scenario_name:
        scenario = next((s for s in SCENARIOS if s["name"] == scenario_name), None)
        if not scenario:
            print(f"Unknown scenario: {scenario_name}")
            print(f"Available: {[s['name'] for s in SCENARIOS]}")
            sys.exit(1)
    else:
        scenario = SCENARIOS[scenario_index % len(SCENARIOS)]

    with tempfile.TemporaryDirectory(prefix="reagt_test_") as tmp:
        work_dir = Path(tmp)
        runner = TestRunner(scenario, work_dir)
        return runner.run()


def run_all_scenarios() -> list[dict]:
    """Run all scenarios and return results."""
    results = []
    for scenario in SCENARIOS:
        with tempfile.TemporaryDirectory(prefix="reagt_test_") as tmp:
            work_dir = Path(tmp)
            runner = TestRunner(scenario, work_dir)
            results.append(runner.run())
    return results


def print_summary(results: list[dict]):
    """Print overall summary across all scenarios."""
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")

    all_criteria = set()
    for r in results:
        all_criteria.update(r.get("criteria", {}).keys())

    for criterion in sorted(all_criteria):
        passes = sum(
            1 for r in results
            if r.get("criteria", {}).get(criterion, {}).get("passed", False)
        )
        total = len(results)
        status = "ALL PASS" if passes == total else f"{passes}/{total}"
        print(f"  {criterion}: {status}")

    total_passed = sum(
        sum(1 for c in r.get("criteria", {}).values() if c.get("passed", False))
        for r in results
    )
    total_criteria = sum(len(r.get("criteria", {})) for r in results)
    print(f"\n  TOTAL: {total_passed}/{total_criteria} criteria passed across {len(results)} scenarios")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ReAgt Test Harness")
    parser.add_argument("--scenario", type=str, help="Scenario name to run")
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument("--index", type=int, default=0, help="Scenario index")
    args = parser.parse_args()

    if args.all:
        results = run_all_scenarios()
        print_summary(results)
    else:
        result = run_scenario(scenario_name=args.scenario, scenario_index=args.index)
        print_summary([result])
