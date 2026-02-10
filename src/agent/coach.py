"""Training coach agent: generates weekly plans via Gemini."""

import json
from datetime import datetime
from pathlib import Path

from google import genai

from src.agent.json_utils import extract_json
from src.agent.llm import MODEL, get_client
from src.agent.prompts import COACH_SYSTEM_PROMPT, build_plan_prompt

PLANS_DIR = Path(__file__).parent.parent.parent / "data" / "plans"


def generate_plan(profile: dict) -> dict:
    """Send athlete profile to Gemini and return a structured training plan.

    Raises ValueError if the response is not valid JSON.
    Raises google.genai errors on API failure.
    """
    client = get_client()
    user_prompt = build_plan_prompt(profile)

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            genai.types.Content(
                role="user",
                parts=[genai.types.Part(text=user_prompt)],
            ),
        ],
        config=genai.types.GenerateContentConfig(
            system_instruction=COACH_SYSTEM_PROMPT,
            temperature=0.7,
        ),
    )

    text = response.text.strip()
    return extract_json(text)


def save_plan(plan: dict) -> Path:
    """Save a training plan to data/plans/ with a timestamp filename."""
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = PLANS_DIR / f"plan_{timestamp}.json"
    path.write_text(json.dumps(plan, indent=2))
    return path
