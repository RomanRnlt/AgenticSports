"""Graduated autonomy: classify adjustment impact and decide what to auto-apply."""

from enum import Enum


class ImpactLevel(Enum):
    LOW = "low"       # auto-apply, inform user
    MEDIUM = "medium"  # propose, wait for approval
    HIGH = "high"     # explain in detail, ask for decision


# Keywords that indicate higher-impact adjustments
_HIGH_IMPACT_KEYWORDS = [
    "restructure", "periodization", "injury", "race target", "goal change",
    "cancel", "significant", "surgery", "stop training",
]
_MEDIUM_IMPACT_KEYWORDS = [
    "volume", "reduce", "increase", "rest day", "add session", "remove session",
    "skip", "swap week",
]


def classify_impact(adjustment: dict) -> ImpactLevel:
    """Classify the impact level of a proposed adjustment.

    Uses the adjustment's own 'impact' field if present, otherwise infers from description.

    Low: single session pace/HR target change, swap equivalent sessions
    Medium: volume change >15%, add/remove rest day, change session type
    High: restructure periodization, change race target, injury flag
    """
    # Trust the LLM's classification if provided
    impact_str = adjustment.get("impact", "").lower()
    if impact_str == "high":
        return ImpactLevel.HIGH
    if impact_str == "medium":
        return ImpactLevel.MEDIUM
    if impact_str == "low":
        return ImpactLevel.LOW

    # Fallback: infer from description
    desc = adjustment.get("description", "").lower()
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
