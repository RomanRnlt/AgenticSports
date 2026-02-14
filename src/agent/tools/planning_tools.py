"""Planning tools -- create and evaluate training plans.

These are the equivalent of Claude Code using Write/Edit tools to create code.
Plan creation uses a specialized LLM call (like a sub-agent) with a
coaching-specific system prompt.
"""

import json
from datetime import datetime
from pathlib import Path

from google import genai

from src.agent.llm import MODEL, get_client
from src.agent.json_utils import extract_json
from src.agent.tools.registry import Tool, ToolRegistry


def register_planning_tools(registry: ToolRegistry, user_model):
    """Register all planning tools."""

    def create_training_plan(
        focus: str = None,
        feedback: str = None,
        sport_distribution: dict = None,
    ) -> dict:
        """Generate a training plan using the coach persona."""
        from src.agent.prompts import COACH_SYSTEM_PROMPT, build_plan_prompt
        from src.tools.activity_store import list_activities
        from src.memory.episodes import list_episodes, retrieve_relevant_episodes

        profile = user_model.project_profile()
        beliefs = user_model.get_active_beliefs(min_confidence=0.6)
        activities = list_activities()
        episodes = list_episodes(limit=10)
        relevant_eps = retrieve_relevant_episodes(
            {"goal": profile.get("goal", {}), "sports": profile.get("sports", [])},
            episodes,
            max_results=5,
        )

        base_prompt = build_plan_prompt(
            profile, beliefs=beliefs, activities=activities,
            relevant_episodes=relevant_eps,
        )

        if focus:
            base_prompt += f"\n\nFOCUS FOR THIS PLAN: {focus}"
        if feedback:
            base_prompt += f"\n\nPREVIOUS PLAN FEEDBACK (address these issues): {feedback}"
        if sport_distribution:
            base_prompt += f"\n\nREQUESTED SPORT DISTRIBUTION: {json.dumps(sport_distribution)}"

        client = get_client()
        response = client.models.generate_content(
            model=MODEL,
            contents=[genai.types.Content(
                role="user",
                parts=[genai.types.Part(text=base_prompt)],
            )],
            config=genai.types.GenerateContentConfig(
                system_instruction=COACH_SYSTEM_PROMPT,
                temperature=0.7,
            ),
        )

        plan = extract_json(response.text.strip())
        plan["_generated_at"] = datetime.now().isoformat()
        return plan

    registry.register(Tool(
        name="create_training_plan",
        description=(
            "Generate a structured weekly training plan based on athlete profile, "
            "goals, and recent training data. The plan includes specific sessions "
            "with sport, type, duration, and intensity targets. "
            "Optionally pass focus (e.g., 'base building') or feedback from "
            "a previous evaluation to improve the plan. "
            "IMPORTANT: After creating a plan, ALWAYS use evaluate_plan to check quality."
        ),
        handler=create_training_plan,
        parameters={
            "type": "object",
            "properties": {
                "focus": {
                    "type": "string",
                    "description": "Training focus or emphasis (e.g., 'base building', 'speed work', 'recovery week')",
                    "nullable": True,
                },
                "feedback": {
                    "type": "string",
                    "description": "Feedback from a previous plan evaluation -- issues to fix",
                    "nullable": True,
                },
                "sport_distribution": {
                    "type": "object",
                    "description": "Requested session counts per sport (e.g., {\"running\": 3, \"cycling\": 2})",
                    "nullable": True,
                },
            },
        },
        category="planning",
    ))

    def evaluate_plan(plan: dict) -> dict:
        """Evaluate plan quality with an independent reviewer persona."""
        from src.agent.plan_evaluator import evaluate_plan as _evaluate

        profile = user_model.project_profile()
        beliefs = user_model.get_active_beliefs(min_confidence=0.6)
        evaluation = _evaluate(plan, profile, beliefs=beliefs)

        return {
            "score": evaluation.score,
            "acceptable": evaluation.acceptable,
            "criteria": evaluation.criteria_scores,
            "issues": evaluation.issues,
            "suggestions": evaluation.suggestions,
        }

    registry.register(Tool(
        name="evaluate_plan",
        description=(
            "Have a training plan independently reviewed by a sports science persona. "
            "Returns a quality score (0-100), specific issues found, and improvement "
            "suggestions. Use this AFTER create_training_plan. If score < 70, "
            "regenerate the plan with the feedback."
        ),
        handler=evaluate_plan,
        parameters={
            "type": "object",
            "properties": {
                "plan": {
                    "type": "object",
                    "description": "The training plan to evaluate",
                },
            },
            "required": ["plan"],
        },
        category="planning",
    ))

    def save_plan(plan: dict) -> dict:
        """Save a training plan to disk."""
        plans_dir = Path("data/plans")
        plans_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        path = plans_dir / f"plan_{timestamp}.json"
        path.write_text(json.dumps(plan, indent=2))
        return {"saved": True, "path": str(path)}

    registry.register(Tool(
        name="save_plan",
        description=(
            "Save a finalized training plan to disk. Only use this AFTER the plan "
            "has passed evaluation (score >= 70). Returns the file path."
        ),
        handler=save_plan,
        parameters={
            "type": "object",
            "properties": {
                "plan": {
                    "type": "object",
                    "description": "The finalized training plan to save",
                },
            },
            "required": ["plan"],
        },
        category="planning",
    ))
