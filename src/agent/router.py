"""Input routing: classify user messages to optimize context assembly.

Priority 5 — Audit finding #8 ("Kein dynamisches Routing"):
    v1.0 runs every message through the same pipeline with identical context
    assembly. Whether the user asks about their last run, reports an injury,
    or says "thanks", the same 5-level context + LLM call happens.

    The router classifies messages into route types and returns context
    budget overrides. This allows the conversation engine to load heavy
    activity data only when the question is about activities, or skip
    activity context entirely for simple chat.

Route types:
    activity_question — Questions about training data, workouts, performance
    plan_question — Questions about the training plan, upcoming sessions
    constraint_update — New constraints, injuries, availability changes
    goal_discussion — Goal progress, target changes, trajectory questions
    motivation — Emotional support, encouragement, frustration
    general_chat — Greetings, thanks, knowledge questions, off-topic
"""

import json

from google import genai

from src.agent.json_utils import extract_json
from src.agent.llm import MODEL, get_client


ROUTE_TYPES = [
    "activity_question",
    "plan_question",
    "constraint_update",
    "goal_discussion",
    "motivation",
    "general_chat",
]

# Context budget multipliers per route type.
# Values are multipliers on the base budgets from TOKEN_BUDGETS.
# 1.0 = use full budget, 0.0 = skip entirely, 1.5 = boost.
ROUTE_CONTEXT_WEIGHTS = {
    "activity_question": {
        "model": 0.8, "activity": 1.5, "cross": 0.5, "rolling": 0.8, "recent": 1.0,
    },
    "plan_question": {
        "model": 1.0, "activity": 0.5, "cross": 0.5, "rolling": 1.0, "recent": 1.0,
    },
    "constraint_update": {
        "model": 1.2, "activity": 0.3, "cross": 0.5, "rolling": 1.0, "recent": 1.0,
    },
    "goal_discussion": {
        "model": 1.0, "activity": 0.8, "cross": 1.0, "rolling": 1.0, "recent": 1.0,
    },
    "motivation": {
        "model": 0.8, "activity": 0.3, "cross": 0.8, "rolling": 1.0, "recent": 1.2,
    },
    "general_chat": {
        "model": 0.5, "activity": 0.0, "cross": 0.3, "rolling": 0.5, "recent": 1.0,
    },
}

CLASSIFICATION_PROMPT = """\
Classify this athlete's message into exactly ONE category.

Categories:
- activity_question: asks about a specific workout, training data, pace, HR, or performance ("How was my run?", "What did I do this week?")
- plan_question: asks about the training plan, upcoming sessions, or schedule ("What's my plan?", "What should I do tomorrow?")
- constraint_update: reports a new constraint, injury, or availability change ("I have knee pain", "I can only train 4 days now")
- goal_discussion: discusses goals, race targets, or long-term progress ("Am I on track?", "I want to change my race goal")
- motivation: expresses emotion about training — frustration, excitement, fatigue ("I don't feel like training", "That was amazing!")
- general_chat: greetings, thanks, knowledge questions, or off-topic ("Hi", "Thanks!", "What is zone 2?")

Message: "{message}"

You MUST respond with ONLY a valid JSON object:
{{"route": "<category_name>"}}
"""


def classify_message(message: str) -> str:
    """Classify a user message into a route type using LLM.

    Returns one of ROUTE_TYPES. Falls back to 'general_chat' on failure.
    """
    try:
        client = get_client()
        prompt = CLASSIFICATION_PROMPT.format(message=message[:500])

        response = client.models.generate_content(
            model=MODEL,
            contents=[
                genai.types.Content(
                    role="user",
                    parts=[genai.types.Part(text=prompt)],
                ),
            ],
            config=genai.types.GenerateContentConfig(
                temperature=0.1,  # Very low for consistent classification
            ),
        )

        result = extract_json(response.text.strip())
        route = result.get("route", "general_chat")
        if route in ROUTE_TYPES:
            return route
    except Exception:
        pass

    return "general_chat"


def get_budget_overrides(route: str, base_budgets: dict) -> dict:
    """Apply route-specific context weight multipliers to base budgets.

    Args:
        route: One of ROUTE_TYPES
        base_budgets: The base TOKEN_BUDGETS dict for the current phase

    Returns:
        New budget dict with adjusted values (system budget unchanged).
    """
    weights = ROUTE_CONTEXT_WEIGHTS.get(route, {})
    overrides = dict(base_budgets)

    for key, multiplier in weights.items():
        if key in overrides:
            overrides[key] = int(overrides[key] * multiplier)

    return overrides
