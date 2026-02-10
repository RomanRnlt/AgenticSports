"""Utilities for extracting and repairing JSON from LLM responses."""

import json
import re


def extract_json(text: str) -> dict:
    """Extract a JSON object from LLM response text.

    Handles:
    - Markdown code fences (```json ... ```)
    - Leading/trailing whitespace and text
    - Common LLM JSON errors (missing closing braces, trailing commas)
    - Unescaped control characters in string values
    - Text before/after JSON object

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

    # Try to find JSON object bounded by first { and last }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace:last_brace + 1]
        # Try progressively more aggressive fixes
        for fixer in [lambda t: t, _fix_trailing_commas, _fix_missing_braces, _fix_control_chars]:
            try:
                return json.loads(fixer(candidate))
            except json.JSONDecodeError:
                continue

        # Try combining fixes
        try:
            fixed = _fix_control_chars(_fix_trailing_commas(candidate))
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        try:
            fixed = _fix_control_chars(_fix_missing_braces(candidate))
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

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
    bracket_count = text.count("[") - text.count("]")
    if bracket_count > 0:
        text = text.rstrip() + "]" * bracket_count
    return text


def _fix_control_chars(text: str) -> str:
    """Escape unescaped control characters inside JSON string values."""
    # Replace literal tab/newline chars that might appear inside strings
    # but preserve \n and \t that are already escaped
    result = []
    in_string = False
    escape_next = False
    for char in text:
        if escape_next:
            result.append(char)
            escape_next = False
            continue
        if char == '\\' and in_string:
            escape_next = True
            result.append(char)
            continue
        if char == '"':
            in_string = not in_string
            result.append(char)
            continue
        if in_string:
            if char == '\n':
                result.append('\\n')
                continue
            if char == '\r':
                result.append('\\r')
                continue
            if char == '\t':
                result.append('\\t')
                continue
        result.append(char)
    return ''.join(result)
