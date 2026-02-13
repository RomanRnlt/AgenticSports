"""Graduated autonomy: classify adjustment impact and decide what to auto-apply.

P7 enhancement: Impact classification uses LLM-based contextual analysis
instead of keyword matching. Falls back to keyword heuristic if LLM fails.
"""

import json
from enum import Enum

from src.agent.json_utils import extract_json
from src.agent.llm import MODEL, get_client


class ImpactLevel(Enum):
    LOW = "low"       # auto-apply, inform user
    MEDIUM = "medium"  # propose, wait for approval
    HIGH = "high"     # explain in detail, ask for decision


# Keyword fallbacks (used when LLM is unavailable)
_HIGH_IMPACT_KEYWORDS = [
    "restructure", "periodization", "injury", "race target", "goal change",
    "cancel", "significant", "surgery", "stop training",
]
_MEDIUM_IMPACT_KEYWORDS = [
    "volume", "reduce", "increase", "rest day", "add session", "remove session",
    "skip", "swap week",
]


IMPACT_CLASSIFICATION_PROMPT = """\
Classify the impact level of this coaching adjustment for an endurance athlete.

Impact levels:
- low: Minor tweaks that don't change the training structure (pace target change, HR zone nudge, swap equivalent sessions)
- medium: Changes that modify weekly structure (volume change >15%, add/remove rest day, change session type)
- high: Fundamental changes to the training program (restructure periodization, change race goal, injury accommodation, stop/cancel training)

Adjustment: "{description}"

You MUST respond with ONLY a valid JSON object:
{{"impact": "low|medium|high", "reasoning": "1 sentence why"}}
"""


def classify_impact(adjustment: dict, use_llm: bool = True) -> ImpactLevel:
    """Classify the impact level of a proposed adjustment.

    Uses LLM-based contextual classification (P7), falling back to keyword
    heuristic if LLM is unavailable or fails.

    Args:
        adjustment: Dict with 'description' and optionally 'impact' fields.
        use_llm: Whether to attempt LLM classification (disable for testing fallback).
    """
    # Trust the LLM's pre-classified impact if provided
    impact_str = adjustment.get("impact", "").lower()
    if impact_str == "high":
        return ImpactLevel.HIGH
    if impact_str == "medium":
        return ImpactLevel.MEDIUM
    if impact_str == "low":
        return ImpactLevel.LOW

    desc = adjustment.get("description", "")

    # Primary: LLM-based classification (P7)
    if use_llm and desc:
        result = _classify_impact_llm(desc)
        if result is not None:
            return result

    # Fallback: keyword heuristic
    return _classify_impact_keywords(desc)


def _classify_impact_llm(description: str) -> ImpactLevel | None:
    """Classify impact using LLM. Returns None on failure."""
    try:
        from google import genai

        client = get_client()
        prompt = IMPACT_CLASSIFICATION_PROMPT.format(description=description[:500])

        response = client.models.generate_content(
            model=MODEL,
            contents=[
                genai.types.Content(
                    role="user",
                    parts=[genai.types.Part(text=prompt)],
                ),
            ],
            config=genai.types.GenerateContentConfig(
                temperature=0.1,
            ),
        )

        result = extract_json(response.text.strip())
        impact = result.get("impact", "").lower()

        if impact == "high":
            return ImpactLevel.HIGH
        if impact == "medium":
            return ImpactLevel.MEDIUM
        if impact == "low":
            return ImpactLevel.LOW
    except Exception:
        pass

    return None


def _classify_impact_keywords(description: str) -> ImpactLevel:
    """Fallback: classify impact using keyword matching."""
    desc = description.lower()
    for kw in _HIGH_IMPACT_KEYWORDS:
        if kw in desc:
            return ImpactLevel.HIGH
    for kw in _MEDIUM_IMPACT_KEYWORDS:
        if kw in desc:
            return ImpactLevel.MEDIUM
    return ImpactLevel.LOW


def classify_and_apply(adjustments: list[dict]) -> dict:
    """Classify all adjustments and split into auto-applied vs proposals.

    Returns:
        dict with 'auto_applied' and 'proposals' lists,
        plus counts by impact level.
    """
    auto_applied = []
    proposals = []

    for adj in adjustments:
        level = classify_impact(adj)
        adj["classified_impact"] = level.value

        if level == ImpactLevel.LOW:
            auto_applied.append(adj)
        else:
            proposals.append(adj)

    return {
        "auto_applied": auto_applied,
        "proposals": proposals,
        "counts": {
            "low": len([a for a in adjustments if a.get("classified_impact") == "low"]),
            "medium": len([a for a in adjustments if a.get("classified_impact") == "medium"]),
            "high": len([a for a in adjustments if a.get("classified_impact") == "high"]),
        },
    }
