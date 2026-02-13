"""Engine-level test harness for ReAgt.

Wraps ConversationEngine, OnboardingEngine, and AgentCore for scripted
behavioral testing WITHOUT subprocess overhead. Tests the engine layer
directly — all agentic behavior lives here, not in the CLI display layer.

Usage:
    harness = ReAgtHarness(data_dir=tmp_path)
    harness.reset()
    harness.start_onboarding()
    harness.send("I'm a runner training for a half marathon")
    assert harness.onboarding_complete
    plan = harness.generate_plan()
    assert len(plan["sessions"]) >= 3
"""

import json
import shutil
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


class ReAgtHarness:
    """Engine-level test harness for scripted ReAgt interactions."""

    def __init__(self, data_dir: Path | None = None):
        self._data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self._model_dir = self._data_dir / "user_model"
        self._sessions_dir = self._data_dir / "sessions"
        self._plans_dir = self._data_dir / "plans"
        self._episodes_dir = self._data_dir / "episodes"

        self._user_model = None
        self._onboarding_engine = None
        self._conversation_engine = None
        self._responses: list[dict] = []
        self._errors: list[str] = []
        self._mode: str = "idle"  # idle, onboarding, conversation

    # ── Setup / Teardown ────────────────────────────────────────

    def reset(self) -> None:
        """Reset all user data for a clean test run."""
        model_path = self._model_dir / "model.json"
        if model_path.exists():
            model_path.unlink()

        for d in [self._sessions_dir, self._plans_dir, self._episodes_dir]:
            if d.exists():
                for f in d.iterdir():
                    if f.is_file():
                        f.unlink()

        self._user_model = None
        self._onboarding_engine = None
        self._conversation_engine = None
        self._responses = []
        self._errors = []
        self._mode = "idle"
        time.sleep(0.1)  # filesystem settle

    # ── Onboarding Flow ─────────────────────────────────────────

    def start_onboarding(self) -> str:
        """Start a fresh onboarding session. Returns the greeting."""
        from src.agent.onboarding import OnboardingEngine
        from src.memory.user_model import UserModel

        self._user_model = UserModel.load_or_create(data_dir=self._model_dir)
        self._onboarding_engine = OnboardingEngine(
            user_model=self._user_model,
            sessions_dir=self._sessions_dir,
            data_dir=self._data_dir,
        )
        self._mode = "onboarding"

        greeting = self._onboarding_engine.start()
        self._responses.append({"role": "agent", "content": greeting, "type": "greeting"})
        return greeting

    def start_conversation(self) -> str:
        """Start a conversation session for an existing user. Returns session ID."""
        from src.agent.conversation import ConversationEngine
        from src.memory.user_model import UserModel

        self._user_model = UserModel.load_or_create(data_dir=self._model_dir)
        self._conversation_engine = ConversationEngine(
            user_model=self._user_model,
            sessions_dir=self._sessions_dir,
            data_dir=self._data_dir,
        )
        self._mode = "conversation"

        session_id = self._conversation_engine.start_session()
        return session_id

    def send(self, message: str) -> str:
        """Send a message and return the response.

        Works in both onboarding and conversation mode.
        """
        self._responses.append({"role": "user", "content": message})

        try:
            if self._mode == "onboarding" and self._onboarding_engine:
                response = self._onboarding_engine.process_message(message)
            elif self._mode == "conversation" and self._conversation_engine:
                response = self._conversation_engine.process_message(message)
            else:
                raise RuntimeError(f"Harness not started (mode={self._mode})")
        except Exception as e:
            self._errors.append(f"send() failed: {e}")
            response = f"[ERROR: {e}]"

        self._responses.append({"role": "agent", "content": response, "type": "response"})
        return response

    def end_session(self) -> dict | None:
        """End the current session with consolidation."""
        summary = None
        if self._mode == "onboarding" and self._onboarding_engine:
            summary = self._onboarding_engine.end_session()
        elif self._mode == "conversation" and self._conversation_engine:
            summary = self._conversation_engine.end_session()

        if self._user_model:
            self._user_model.save()

        self._mode = "idle"
        return summary

    # ── Plan Generation ─────────────────────────────────────────

    def generate_plan(self) -> dict | None:
        """Generate a training plan from current user model state.

        Returns the plan dict, or None on failure.
        """
        if not self._user_model:
            self._errors.append("Cannot generate plan: no user model")
            return None

        from src.agent.coach import generate_plan, save_plan
        from src.tools.activity_store import list_activities
        from src.memory.episodes import list_episodes, retrieve_relevant_episodes

        profile = self._user_model.project_profile()
        beliefs = self._user_model.get_active_beliefs(min_confidence=0.6)
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
            return plan
        except Exception as e:
            self._errors.append(f"Plan generation failed: {e}")
            return None

    # ── Agent Cycle ─────────────────────────────────────────────

    def run_agent_cycle(self, activities: list[dict] | None = None) -> dict | None:
        """Run the agent cognitive cycle (PERCEIVE→REASON→PLAN→...).

        Returns the cycle result dict, or None on failure.
        """
        if not self._user_model:
            self._errors.append("Cannot run cycle: no user model")
            return None

        from src.agent.state_machine import AgentCore
        from src.tools.activity_store import list_activities

        if activities is None:
            activities = list_activities()

        profile = self._user_model.project_profile()
        plan = self._load_latest_plan() or {"sessions": []}

        agent = AgentCore()
        try:
            result = agent.run_cycle(
                profile, plan, activities,
                user_model=self._user_model,
            )
            return result
        except Exception as e:
            self._errors.append(f"Agent cycle failed: {e}")
            return None

    # ── Inspection ──────────────────────────────────────────────

    @property
    def onboarding_complete(self) -> bool:
        """Check if onboarding has gathered enough information."""
        if self._onboarding_engine:
            return self._onboarding_engine.is_onboarding_complete()
        return False

    @property
    def cycle_triggered(self) -> bool:
        """Check if the conversation engine triggered a cycle."""
        if self._onboarding_engine:
            return self._onboarding_engine.conversation.cycle_triggered
        if self._conversation_engine:
            return self._conversation_engine.cycle_triggered
        return False

    @property
    def structured_core(self) -> dict:
        """Return the current structured_core from user model."""
        if self._user_model:
            return self._user_model.structured_core
        return {}

    @property
    def beliefs(self) -> list[dict]:
        """Return active beliefs from user model."""
        if self._user_model:
            return self._user_model.get_active_beliefs(min_confidence=0.5)
        return []

    @property
    def responses(self) -> list[dict]:
        """Return all conversation turns."""
        return self._responses

    @property
    def errors(self) -> list[str]:
        """Return any errors that occurred during testing."""
        return self._errors

    @property
    def user_model(self) -> Any:
        """Return the UserModel instance."""
        return self._user_model

    # ── Data File Access ────────────────────────────────────────

    def get_latest_plan(self) -> dict | None:
        """Load the most recent plan from disk."""
        return self._load_latest_plan()

    def get_data_files(self) -> dict:
        """Return all data files as dicts for inspection."""
        result = {}

        model_path = self._model_dir / "model.json"
        if model_path.exists():
            result["model"] = json.loads(model_path.read_text())

        plan = self._load_latest_plan()
        if plan:
            result["plan"] = plan

        # Session files
        sessions = []
        if self._sessions_dir.exists():
            for p in self._sessions_dir.glob("session_*.jsonl"):
                turns = []
                for line in p.read_text().splitlines():
                    if line.strip():
                        try:
                            turns.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                sessions.append({"file": p.name, "turns": turns})
        result["sessions"] = sessions

        return result

    def _load_latest_plan(self) -> dict | None:
        """Load the most recent training plan."""
        if not self._plans_dir.exists():
            return None
        plan_files = sorted(self._plans_dir.glob("plan_*.json"))
        if not plan_files:
            return None
        try:
            return json.loads(plan_files[-1].read_text())
        except (json.JSONDecodeError, OSError):
            return None
