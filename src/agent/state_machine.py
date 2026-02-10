"""Agent state machine: the core cognitive loop for ReAgt."""

from enum import Enum
from datetime import datetime


class AgentState(Enum):
    IDLE = "idle"
    PERCEIVING = "perceiving"
    REASONING = "reasoning"
    PLANNING = "planning"
    PROPOSING = "proposing"
    EXECUTING = "executing"
    REFLECTING = "reflecting"


class AgentCore:
    """Orchestrates the agent cognitive cycle: perceive -> reason -> plan -> propose -> execute."""

    def __init__(self):
        self.state = AgentState.IDLE
        self.context: dict = {}  # working memory for current cycle

    def transition(self, new_state: AgentState) -> None:
        """Transition to a new state, recording the change in context."""
        self.context.setdefault("state_history", []).append({
            "from": self.state.value,
            "to": new_state.value,
            "at": datetime.now().isoformat(timespec="seconds"),
        })
        self.state = new_state

    def run_cycle(
        self,
        profile: dict,
        plan: dict,
        activities: list[dict],
        user_model: "object | None" = None,
        conversation_context: str | None = None,
    ) -> dict:
        """Execute one full cognitive cycle.

        Args:
            profile: Athlete profile dict
            plan: Current training plan dict
            activities: List of activity dicts (actual training data)
            user_model: Optional UserModel instance. When provided, active beliefs
                        are injected into assessment and planning prompts.
            conversation_context: Optional recent conversation text for subjective
                                  context (e.g. "I've been sleeping badly").

        Returns:
            dict with assessment, adjustments, and updated plan
        """
        from src.agent.assessment import assess_training
        from src.agent.planner import generate_adjusted_plan
        from src.agent.autonomy import classify_and_apply

        self.context = {
            "cycle_started": datetime.now().isoformat(timespec="seconds"),
            "state_history": [],
        }

        # Extract beliefs from user model if available
        beliefs = None
        if user_model is not None:
            beliefs = user_model.get_active_beliefs(min_confidence=0.6)

        # PERCEIVING: activities already parsed and passed in
        self.transition(AgentState.PERCEIVING)
        self.context["profile"] = profile
        self.context["plan"] = plan
        self.context["activities"] = activities
        if conversation_context:
            self.context["conversation_context"] = conversation_context

        # REASONING: assess current training vs plan
        self.transition(AgentState.REASONING)
        assessment = assess_training(
            profile, plan, activities,
            conversation_context=conversation_context,
            beliefs=beliefs,
        )
        self.context["assessment"] = assessment

        # PLANNING: generate adjusted plan
        self.transition(AgentState.PLANNING)
        adjustments = assessment.get("recommended_adjustments", [])
        adjusted_plan = generate_adjusted_plan(
            profile, plan, assessment, beliefs=beliefs,
        )
        self.context["adjusted_plan"] = adjusted_plan

        # PROPOSING: classify adjustments by impact
        self.transition(AgentState.PROPOSING)
        autonomy_result = classify_and_apply(adjustments)
        self.context["autonomy_result"] = autonomy_result

        # EXECUTING: save the plan (caller handles persistence)
        self.transition(AgentState.EXECUTING)

        self.transition(AgentState.IDLE)

        return {
            "assessment": assessment,
            "adjusted_plan": adjusted_plan,
            "autonomy_result": autonomy_result,
            "cycle_context": self.context,
        }
