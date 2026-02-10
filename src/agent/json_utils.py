"""Utilities for extracting and repairing JSON from LLM responses."""

import json
import re


def extract_json(text: str) -> dict:
    """Extract a JSON object from LLM response text.

    Handles:
    - Markdown code fences (```json ... ```)
    - Leading/trailing whitespace and text
    - Common LLM JSON errors (missing closing braces, trailing commas)

    Raises ValueError if no valid JSON can be extracted.
    """
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    match = re.search(r"\{", text)
    if match:
        candidate = text[match.start():]
        # Try progressively fixing common issues
        for attempt in [candidate, _fix_trailing_commas(candidate), _fix_missing_braces(candidate)]:
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not extract valid JSON from LLM response:\n{text[:500]}")


def _fix_trailing_commas(text: str) -> str:
    """Remove trailing commas before closing braces/brackets."""
    # ,} -> }   and ,] -> ]
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)
    return text


def _fix_missing_braces(text: str) -> str:
    """Try to fix unbalanced braces by appending missing closing braces."""
    text = _fix_trailing_commas(text)
    open_count = text.count("{") - text.count("}")
    if open_count > 0:
        text = text.rstrip() + "}" * open_count
    return text
