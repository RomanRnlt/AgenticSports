"""Agent Action Space: dynamic tool selection for the cognitive loop.

Design rationale (Priority 2 — Audit findings #1 and #2):
    The v1.0 state_machine.py runs a fixed pipeline (PERCEIVE → REASON → PLAN →
    PROPOSE → EXECUTE → REFLECT). The agent never decides *what* to do — it always
    does everything in the same order.

    This module introduces an Action Space: a registry of discrete capabilities
    the agent can invoke. An LLM-based selector decides which action to take
    given the current context, replacing the fixed sequence.

    Each action wraps an existing capability (assessment, planning, trajectory,
    etc.) as a callable with a standardized interface: takes a context dict,
    returns a result dict.

Actions:
    assess_activities — Analyze recent activities vs current plan (assessment.py)
    generate_plan — Create or regenerate a training plan (coach.py / planner.py)
    evaluate_trajectory — Assess long-term goal trajectory (trajectory.py)
    update_beliefs — Update user model beliefs from new evidence (user_model.py)
    query_episodes — Retrieve relevant past episodes (episodes.py)
    respond — Signal that no further action is needed (pass-through)

Selection:
    select_action() sends the current context summary to the LLM and asks it
    to choose the single most valuable action. The LLM must return a structured
    JSON response with the action name and reasoning.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from google import genai

from src.agent.json_utils import extract_json
from src.agent.llm import MODEL, get_client


# ── Action Definition ────────────────────────────────────────────────

@dataclass
class Action:
    """A discrete capability the agent can invoke."""

    name: str
    description: str
    handler: Callable[[dict], dict]
    # What context keys this action needs to be useful
    requires: list[str] = field(default_factory=list)
    # What context keys this action produces
    produces: list[str] = field(default_factory=list)


# ── Action Handlers ──────────────────────────────────────────────────

def _handle_assess_activities(ctx: dict) -> dict:
    """Analyze recent activities against the current plan."""
    from src.agent.assessment import assess_training

    profile = ctx["profile"]
    plan = ctx.get("plan", {"sessions": []})
    activities = ctx.get("activities", [])
    conversation_context = ctx.get("conversation_context")
    beliefs = ctx.get("beliefs")

    assessment = assess_training(
        profile, plan, activities,
        conversation_context=conversation_context,
        beliefs=beliefs,
    )
    return {"assessment": assessment}


def _handle_generate_plan(ctx: dict) -> dict:
    """Generate or regenerate a training plan.

    If plan_feedback exists in context (from evaluate_plan), it's available
    for the planner to address the evaluator's critique on regeneration.
    """
    assessment = ctx.get("assessment")

    if assessment:
        # Adjusted plan based on assessment
        from src.agent.planner import generate_adjusted_plan

        plan = generate_adjusted_plan(
            ctx["profile"],
            ctx.get("plan", {"sessions": []}),
            assessment,
            relevant_episodes=ctx.get("relevant_episodes"),
            beliefs=ctx.get("beliefs"),
            activities=ctx.get("activities"),
        )
    else:
        # Fresh plan generation
        from src.agent.coach import generate_plan

        plan = generate_plan(
            ctx["profile"],
            beliefs=ctx.get("beliefs"),
            activities=ctx.get("activities"),
            relevant_episodes=ctx.get("relevant_episodes"),
        )

    return {"adjusted_plan": plan}


def _handle_evaluate_trajectory(ctx: dict) -> dict:
    """Assess long-term goal trajectory."""
    from src.agent.trajectory import assess_trajectory

    trajectory = assess_trajectory(
        ctx["profile"],
        ctx.get("activities", []),
        ctx.get("episodes", []),
        ctx.get("plan", {"sessions": []}),
    )
    return {"trajectory": trajectory}


def _handle_update_beliefs(ctx: dict) -> dict:
    """Update beliefs from assessment evidence and record outcomes.

    P6 enhancement: compares assessment data against existing beliefs to
    record confirmed/contradicted outcomes, updating confidence and utility.
    """
    user_model = ctx.get("user_model")
    assessment = ctx.get("assessment")

    if not user_model or not assessment:
        return {"beliefs_updated": False, "reason": "missing user_model or assessment"}

    assess_data = assessment.get("assessment", {})
    observations = assess_data.get("observations", [])
    compliance = assess_data.get("compliance")

    outcomes_recorded = 0

    # Record outcomes for active beliefs based on assessment evidence
    active_beliefs = user_model.get_active_beliefs()
    if compliance is not None and active_beliefs:
        obs_text = " ".join(observations).lower()

        for belief in active_beliefs:
            belief_text = belief["text"].lower()

            # Check if any observation references this belief's topic
            # Simple heuristic: check for keyword overlap
            belief_words = set(belief_text.split()) - {"the", "a", "an", "is", "to", "and", "or", "in", "on", "for"}
            overlap = sum(1 for w in belief_words if w in obs_text)

            if overlap >= 2:
                # This belief is relevant to the current assessment
                if compliance >= 0.7:
                    user_model.record_outcome(
                        belief["id"], "confirmed",
                        detail=f"Assessment compliance {compliance:.0%}",
                    )
                else:
                    user_model.record_outcome(
                        belief["id"], "contradicted",
                        detail=f"Assessment compliance {compliance:.0%}",
                    )
                outcomes_recorded += 1

    if observations or outcomes_recorded > 0:
        user_model.save()

    return {
        "beliefs_updated": True,
        "observations_count": len(observations),
        "outcomes_recorded": outcomes_recorded,
    }


def _handle_query_episodes(ctx: dict) -> dict:
    """Retrieve relevant past episodes for context."""
    from src.memory.episodes import list_episodes, retrieve_relevant_episodes

    episodes = list_episodes(limit=10)
    profile = ctx["profile"]

    relevant = retrieve_relevant_episodes(
        {"goal": profile.get("goal", {}), "sports": profile.get("sports", [])},
        episodes,
        max_results=5,
    )
    return {"episodes": episodes, "relevant_episodes": relevant}


def _handle_classify_adjustments(ctx: dict) -> dict:
    """Classify adjustment impact levels for graduated autonomy."""
    from src.agent.autonomy import classify_and_apply

    assessment = ctx.get("assessment", {})
    adjustments = assessment.get("recommended_adjustments", [])
    result = classify_and_apply(adjustments)
    return {"autonomy_result": result}


def _handle_evaluate_plan(ctx: dict) -> dict:
    """Evaluate a generated plan's quality and decide if regeneration is needed."""
    from src.agent.plan_evaluator import evaluate_plan, PLAN_ACCEPTANCE_THRESHOLD

    plan = ctx.get("adjusted_plan") or ctx.get("plan", {})
    profile = ctx["profile"]
    beliefs = ctx.get("beliefs")
    assessment = ctx.get("assessment")

    evaluation = evaluate_plan(plan, profile, beliefs=beliefs, assessment=assessment)

    # Track evaluation history for regeneration decisions
    plan_scores = ctx.get("plan_scores", [])
    plan_scores.append(evaluation.score)

    result = {
        "plan_evaluation": {
            "score": evaluation.score,
            "criteria": evaluation.criteria_scores,
            "issues": evaluation.issues,
            "suggestions": evaluation.suggestions,
            "acceptable": evaluation.acceptable,
        },
        "plan_scores": plan_scores,
        "plan_iterations": len(plan_scores),
    }

    # If not acceptable and under iteration limit, inject feedback for regeneration
    if not evaluation.acceptable and len(plan_scores) < 3:
        result["plan_feedback"] = (
            f"Previous plan scored {evaluation.score}/100. "
            f"Issues: {'; '.join(evaluation.issues[:3])}. "
            f"Suggestions: {'; '.join(evaluation.suggestions[:3])}. "
            f"Address these in the regenerated plan."
        )

    return result


def _handle_respond(ctx: dict) -> dict:
    """No further action needed — signal to respond to user."""
    return {"action": "respond", "reason": "no further action needed"}


# ── Action Registry ──────────────────────────────────────────────────

ACTIONS: dict[str, Action] = {
    "assess_activities": Action(
        name="assess_activities",
        description="Analyze recent training activities against the current plan. "
                    "Use when new activity data is available or compliance needs checking.",
        handler=_handle_assess_activities,
        requires=["profile", "activities"],
        produces=["assessment"],
    ),
    "generate_plan": Action(
        name="generate_plan",
        description="Generate or regenerate a training plan. Use after assessment reveals "
                    "low compliance, or when the athlete needs a new plan. If assessment "
                    "exists, generates an adjusted plan; otherwise generates fresh.",
        handler=_handle_generate_plan,
        requires=["profile"],
        produces=["adjusted_plan"],
    ),
    "evaluate_trajectory": Action(
        name="evaluate_trajectory",
        description="Assess whether the athlete is on track for their goal. Use when "
                    "enough data exists (4+ weeks) to make meaningful predictions.",
        handler=_handle_evaluate_trajectory,
        requires=["profile", "activities"],
        produces=["trajectory"],
    ),
    "update_beliefs": Action(
        name="update_beliefs",
        description="Update the user model with new evidence from assessment observations. "
                    "Use after assessment to persist learned insights.",
        handler=_handle_update_beliefs,
        requires=["user_model", "assessment"],
        produces=["beliefs_updated"],
    ),
    "query_episodes": Action(
        name="query_episodes",
        description="Retrieve relevant past training episodes for context. Use before "
                    "planning to inform the plan with historical lessons.",
        handler=_handle_query_episodes,
        requires=["profile"],
        produces=["episodes", "relevant_episodes"],
    ),
    "evaluate_plan": Action(
        name="evaluate_plan",
        description="Score a generated plan's quality (0-100) and provide feedback. "
                    "Use AFTER generate_plan to check sport distribution, target "
                    "specificity, and constraint compliance. If score < 70, the plan "
                    "should be regenerated with the feedback.",
        handler=_handle_evaluate_plan,
        requires=["profile", "adjusted_plan"],
        produces=["plan_evaluation", "plan_scores", "plan_iterations"],
    ),
    "classify_adjustments": Action(
        name="classify_adjustments",
        description="Classify proposed adjustments by impact level (low/medium/high) "
                    "for graduated autonomy. Use after assessment to decide what to "
                    "auto-apply vs propose to the athlete.",
        handler=_handle_classify_adjustments,
        requires=["assessment"],
        produces=["autonomy_result"],
    ),
    "respond": Action(
        name="respond",
        description="No further analysis needed — just respond to the user. Use for "
                    "simple chat, questions that don't need data analysis, or when "
                    "all needed actions have already been taken.",
        handler=_handle_respond,
        requires=[],
        produces=[],
    ),
}


# ── Action Selection ─────────────────────────────────────────────────

ACTION_SELECTION_PROMPT = """\
You are an AI coaching agent deciding what action to take next.

Given the current context about an athlete, decide which SINGLE action is most valuable right now.

Available actions:
{actions_description}

Current context:
{context_summary}

Actions already taken this cycle: {actions_taken}

You MUST respond with ONLY a valid JSON object:
{{
    "action": "<action_name>",
    "reasoning": "<1-2 sentences explaining why this action>"
}}

Rules:
- Choose the action that provides the MOST value given what you know and don't know
- If activities exist but haven't been assessed, assess_activities is usually first
- After assessment, decide if a plan update is needed based on compliance
- query_episodes is valuable before planning if episodes haven't been loaded
- AFTER generate_plan, ALWAYS use evaluate_plan to check quality
- If plan_evaluation shows score < 70 and plan_iterations < 3, use generate_plan again (regenerate with feedback)
- If plan_evaluation shows score >= 70, the plan is accepted — proceed to classify_adjustments or respond
- classify_adjustments is needed after assessment before responding
- respond means "I'm done, no more actions needed"
- Do NOT repeat actions already taken this cycle UNLESS it is generate_plan after a low evaluation score
- If assessment shows good compliance (>= 0.8) and no high-impact adjustments, you may skip generate_plan
"""


def _build_context_summary(ctx: dict) -> str:
    """Build a concise context summary for the action selector."""
    parts = []

    profile = ctx.get("profile", {})
    if profile:
        goal = profile.get("goal", {})
        parts.append(f"Athlete: {profile.get('name', 'unknown')}")
        parts.append(f"Sports: {profile.get('sports', [])}")
        parts.append(f"Goal: {goal.get('event', 'general')} by {goal.get('target_date', 'N/A')}")

    activities = ctx.get("activities", [])
    parts.append(f"Activities available: {len(activities)}")

    plan = ctx.get("plan", {})
    parts.append(f"Current plan sessions: {len(plan.get('sessions', []))}")

    if ctx.get("assessment"):
        assess = ctx["assessment"].get("assessment", {})
        parts.append(f"Assessment done: compliance={assess.get('compliance', '?')}, "
                     f"fatigue={assess.get('fatigue_level', '?')}, "
                     f"adjustments={len(ctx['assessment'].get('recommended_adjustments', []))}")

    if ctx.get("adjusted_plan"):
        parts.append(f"New plan generated: {len(ctx['adjusted_plan'].get('sessions', []))} sessions")

    if ctx.get("relevant_episodes"):
        parts.append(f"Episodes loaded: {len(ctx['relevant_episodes'])}")

    if ctx.get("plan_evaluation"):
        pe = ctx["plan_evaluation"]
        parts.append(f"Plan evaluation: score={pe.get('score', '?')}/100, "
                     f"acceptable={pe.get('acceptable', '?')}, "
                     f"iterations={ctx.get('plan_iterations', 0)}")
        if pe.get("issues"):
            parts.append(f"Plan issues: {'; '.join(pe['issues'][:2])}")

    if ctx.get("autonomy_result"):
        ar = ctx["autonomy_result"]
        parts.append(f"Adjustments classified: auto={len(ar.get('auto_applied', []))}, "
                     f"proposals={len(ar.get('proposals', []))}")

    if ctx.get("conversation_context"):
        parts.append(f"Conversation context: {ctx['conversation_context'][:100]}")

    return "\n".join(f"- {p}" for p in parts)


def _build_actions_description() -> str:
    """Build the actions list for the selection prompt."""
    lines = []
    for name, action in ACTIONS.items():
        lines.append(f"- {name}: {action.description}")
    return "\n".join(lines)


def select_action(ctx: dict, actions_taken: list[str] | None = None) -> dict:
    """Use LLM to select the next action based on current context.

    Args:
        ctx: Current cycle context dict
        actions_taken: List of action names already taken this cycle

    Returns:
        dict with 'action' (str) and 'reasoning' (str)
    """
    if actions_taken is None:
        actions_taken = []

    prompt = ACTION_SELECTION_PROMPT.format(
        actions_description=_build_actions_description(),
        context_summary=_build_context_summary(ctx),
        actions_taken=actions_taken if actions_taken else "none",
    )

    client = get_client()
    response = client.models.generate_content(
        model=MODEL,
        contents=[
            genai.types.Content(
                role="user",
                parts=[genai.types.Part(text=prompt)],
            ),
        ],
        config=genai.types.GenerateContentConfig(
            temperature=0.2,  # Low temperature for deterministic selection
        ),
    )

    text = response.text.strip()
    result = extract_json(text)

    # Validate action name
    action_name = result.get("action", "respond")
    if action_name not in ACTIONS:
        action_name = "respond"
        result["action"] = action_name
        result["reasoning"] = f"Unknown action '{result.get('action')}', defaulting to respond"

    return result


def execute_action(action_name: str, ctx: dict) -> dict:
    """Execute a named action and return its result.

    Args:
        action_name: Name of the action to execute
        ctx: Current cycle context dict

    Returns:
        Action result dict (merged into ctx by the caller)
    """
    action = ACTIONS.get(action_name)
    if not action:
        return {"error": f"Unknown action: {action_name}"}

    return action.handler(ctx)
