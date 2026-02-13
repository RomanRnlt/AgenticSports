"""Agent state machine: the core cognitive loop for ReAgt.

v2.0 Architecture (Priority 2+3):
    Replaced the v1.0 fixed pipeline (PERCEIVE→REASON→PLAN→PROPOSE→EXECUTE→REFLECT)
    with dynamic action selection. The agent now decides what to do at each step
    using LLM-based action selection from the Action Space (src/agent/actions.py).

    The cycle runs as: PERCEIVE → (SELECT → EXECUTE → OBSERVE)* → IDLE
    where * means the inner loop repeats until the agent selects 'respond'
    or hits the max iteration limit.

    This resolves audit findings:
    - #1 (linear pipeline → cognitive loop with decisions)
    - #2 (no action space → dynamic tool selection)
"""

from enum import Enum
from datetime import datetime


class AgentState(Enum):
    IDLE = "idle"
    PERCEIVING = "perceiving"
    SELECTING = "selecting"      # NEW: agent choosing next action
    EXECUTING = "executing"
    OBSERVING = "observing"      # NEW: evaluating action result
    # Legacy states kept for backwards compatibility in tests
    REASONING = "reasoning"
    PLANNING = "planning"
    PROPOSING = "proposing"
    REFLECTING = "reflecting"


# Maximum actions per cycle to prevent runaway loops
MAX_ACTIONS_PER_CYCLE = 6


def _evaluate_result_quality(action_name: str, result: dict) -> float:
    """Score the quality of an action result (0.0-1.0).

    Used in the OBSERVE phase to inform subsequent action selection.
    A low quality score signals the cognitive loop may need to retry
    or take corrective action.
    """
    # Error results have zero quality
    if result.get("error"):
        return 0.0

    if action_name == "assess_activities":
        assessment = result.get("assessment", {}).get("assessment", {})
        # Quality is high if assessment has compliance and observations
        has_compliance = assessment.get("compliance") is not None
        has_observations = len(assessment.get("observations", [])) > 0
        has_trend = assessment.get("fitness_trend") is not None
        score = sum([has_compliance, has_observations, has_trend]) / 3.0
        return round(score, 2)

    if action_name == "generate_plan":
        plan = result.get("adjusted_plan", {})
        sessions = plan.get("sessions", [])
        if not sessions:
            return 0.0
        # Check sessions have required fields
        has_targets = sum(1 for s in sessions if s.get("targets") or s.get("steps")) / len(sessions)
        has_sport = sum(1 for s in sessions if s.get("sport")) / len(sessions)
        return round((has_targets + has_sport) / 2.0, 2)

    if action_name == "evaluate_trajectory":
        trajectory = result.get("trajectory", {}).get("trajectory", {})
        return 1.0 if trajectory.get("on_track") is not None else 0.0

    if action_name == "query_episodes":
        episodes = result.get("relevant_episodes", [])
        return min(1.0, len(episodes) / 3.0)

    if action_name == "evaluate_plan":
        pe = result.get("plan_evaluation", {})
        score = pe.get("score", 0)
        # Quality is the normalized evaluation score
        return round(min(1.0, score / 100.0), 2)

    if action_name == "classify_adjustments":
        ar = result.get("autonomy_result", {})
        return 1.0 if ar else 0.0

    if action_name == "update_beliefs":
        return 1.0 if result.get("beliefs_updated") else 0.0

    # respond or unknown — pass-through quality
    return 1.0


class AgentCore:
    """Orchestrates the agent cognitive cycle with dynamic action selection.

    v2.0: The agent DECIDES what to do at each step instead of following
    a fixed sequence. Uses LLM-based action selection from the Action Space.
    """

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
        """Execute one cognitive cycle with dynamic action selection.

        The agent perceives the context, then iteratively selects and executes
        actions until it decides no more actions are needed (selects 'respond')
        or hits the iteration limit.

        Args:
            profile: Athlete profile dict
            plan: Current training plan dict
            activities: List of activity dicts (actual training data)
            user_model: Optional UserModel instance for belief injection
            conversation_context: Optional recent conversation text

        Returns:
            dict with assessment, adjusted_plan, autonomy_result, cycle_context
            (backwards-compatible with v1.0 callers)
        """
        from src.agent.actions import select_action, execute_action

        self.context = {
            "cycle_started": datetime.now().isoformat(timespec="seconds"),
            "state_history": [],
            "actions_selected": [],
            "action_results": [],
        }

        # Extract beliefs from user model if available
        beliefs = None
        if user_model is not None:
            beliefs = user_model.get_active_beliefs(min_confidence=0.6)

        # ── PERCEIVE: load all available context ──────────────────
        self.transition(AgentState.PERCEIVING)
        action_ctx = {
            "profile": profile,
            "plan": plan,
            "activities": activities,
            "beliefs": beliefs,
            "user_model": user_model,
        }
        if conversation_context:
            action_ctx["conversation_context"] = conversation_context

        self.context["profile"] = profile
        self.context["plan"] = plan
        self.context["activities"] = activities

        # ── ACTION LOOP: select → execute → observe ───────────────
        actions_taken = []

        for iteration in range(MAX_ACTIONS_PER_CYCLE):
            # SELECT: ask the LLM what to do next
            self.transition(AgentState.SELECTING)
            selection = select_action(action_ctx, actions_taken=actions_taken)
            action_name = selection.get("action", "respond")
            reasoning = selection.get("reasoning", "")

            self.context["actions_selected"].append({
                "iteration": iteration,
                "action": action_name,
                "reasoning": reasoning,
            })
            actions_taken.append(action_name)

            # Stop if agent decides no more actions needed
            if action_name == "respond":
                break

            # EXECUTE: run the selected action
            self.transition(AgentState.EXECUTING)
            try:
                result = execute_action(action_name, action_ctx)
            except Exception as e:
                result = {"error": str(e)}

            # OBSERVE: evaluate result quality and merge into context
            self.transition(AgentState.OBSERVING)
            quality = _evaluate_result_quality(action_name, result)

            self.context["action_results"].append({
                "iteration": iteration,
                "action": action_name,
                "result_keys": list(result.keys()),
                "error": result.get("error"),
                "quality": quality,
            })

            action_ctx.update(result)
            self.context.update(result)

        # ── FINALIZE: return backwards-compatible result ──────────
        self.transition(AgentState.IDLE)

        return {
            "assessment": action_ctx.get("assessment"),
            "adjusted_plan": action_ctx.get("adjusted_plan"),
            "autonomy_result": action_ctx.get("autonomy_result"),
            "cycle_context": self.context,
        }
