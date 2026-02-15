"""Conversational onboarding engine.

Wraps ConversationEngine with onboarding-specific behavior:
- Provides opening message for new athletes
- Detects when enough information is gathered for plan generation
- Auto-triggers first plan when onboarding is complete
"""

from pathlib import Path

from src.agent.conversation import ConversationEngine
from src.memory.user_model import UserModel


# Opening message for new athletes
ONBOARDING_GREETING = (
    "Welcome to ReAgt! I'm your adaptive training coach. "
    "Tell me about yourself -- what sport do you train, "
    "what are you training for, and how does your typical training week look?"
)


class OnboardingEngine:
    """Wraps ConversationEngine for the onboarding flow."""

    def __init__(
        self,
        user_model: UserModel | None = None,
        sessions_dir: Path | None = None,
        data_dir: Path | None = None,
    ):
        self.user_model = user_model or UserModel.load_or_create()
        self.conversation = ConversationEngine(
            user_model=self.user_model,
            sessions_dir=sessions_dir,
            data_dir=data_dir,
        )
        self._plan_triggered = False

    def start(self) -> str:
        """Start onboarding session and return the greeting message."""
        self.conversation.start_session()
        # Log the greeting as the first agent message
        self.conversation._append_to_session("agent", ONBOARDING_GREETING)
        self.conversation._turn_count += 1
        return ONBOARDING_GREETING

    def process_message(self, user_message: str) -> str:
        """Process a user message during onboarding.

        Returns the agent's response. The ConversationEngine handles
        belief extraction, structured_core updates, and cycle triggers.
        """
        response = self.conversation.process_message(user_message)

        # Check if the LLM triggered a cycle (meaning onboarding is complete)
        if self.conversation.cycle_triggered:
            self._plan_triggered = True

        return response

    def is_onboarding_complete(self) -> bool:
        """Check if structured_core has minimum fields for plan generation.

        The LLM decides this via trigger_cycle/onboarding_complete in its
        structured response. We also check the minimum required fields
        as a safety net for plan generation.

        Athletes with a specific race event satisfy the goal requirement
        via has_event. Athletes with routine/general fitness goals (no
        specific event) satisfy it via has_goal_type instead.
        """
        if self._plan_triggered:
            return True

        core = self.user_model.structured_core
        has_sports = bool(core.get("sports"))
        has_event = bool(core.get("goal", {}).get("event"))
        has_goal_type = bool(core.get("goal", {}).get("goal_type"))
        has_days = bool(core.get("constraints", {}).get("training_days_per_week"))

        has_goal = has_event or has_goal_type
        return has_sports and has_goal and has_days

    def get_onboarding_prompt(self) -> str:
        """Return the opening message for onboarding."""
        return ONBOARDING_GREETING

    def end_session(self) -> dict | None:
        """End the onboarding session with consolidation."""
        return self.conversation.end_session()
