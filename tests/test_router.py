"""Priority 5 tests: Input routing and message classification.

Validates audit finding #8 ("Kein dynamisches Routing"). Tests verify:
- Messages get classified into correct route types
- Different routes produce different context budgets
- Router integrates into conversation pipeline
- Fallback to general_chat on classification failure
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from src.agent.router import (
    classify_message,
    get_budget_overrides,
    ROUTE_TYPES,
    ROUTE_CONTEXT_WEIGHTS,
)


# ── Fixtures ────────────────────────────────────────────────────────

BASE_BUDGETS = {
    "system": 4000,
    "model": 8000,
    "activity": 4000,
    "cross": 4000,
    "rolling": 3200,
    "recent": 16000,
}


def _mock_classification_response(route: str):
    """Create a mock Gemini response for message classification."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({"route": route})
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    return mock_client


# ── Route Types Tests ───────────────────────────────────────────────

class TestRouteTypes:
    """Verify all route types are defined and have context weights."""

    def test_all_route_types_defined(self):
        expected = {
            "activity_question", "plan_question", "constraint_update",
            "goal_discussion", "motivation", "general_chat",
        }
        assert expected == set(ROUTE_TYPES)

    def test_all_routes_have_context_weights(self):
        for route in ROUTE_TYPES:
            assert route in ROUTE_CONTEXT_WEIGHTS, f"Missing weights for {route}"

    def test_general_chat_has_minimal_activity_context(self):
        weights = ROUTE_CONTEXT_WEIGHTS["general_chat"]
        assert weights["activity"] == 0.0

    def test_activity_question_boosts_activity_context(self):
        weights = ROUTE_CONTEXT_WEIGHTS["activity_question"]
        assert weights["activity"] > 1.0


# ── Classification Tests ────────────────────────────────────────────

class TestClassifyMessage:
    """Verify LLM-based message classification."""

    def test_activity_question(self):
        mock = _mock_classification_response("activity_question")
        with patch("src.agent.router.get_client", return_value=mock):
            route = classify_message("How was my last run?")
        assert route == "activity_question"

    def test_plan_question(self):
        mock = _mock_classification_response("plan_question")
        with patch("src.agent.router.get_client", return_value=mock):
            route = classify_message("What should I do tomorrow?")
        assert route == "plan_question"

    def test_constraint_update(self):
        mock = _mock_classification_response("constraint_update")
        with patch("src.agent.router.get_client", return_value=mock):
            route = classify_message("I have knee pain since yesterday")
        assert route == "constraint_update"

    def test_goal_discussion(self):
        mock = _mock_classification_response("goal_discussion")
        with patch("src.agent.router.get_client", return_value=mock):
            route = classify_message("Am I on track for my race?")
        assert route == "goal_discussion"

    def test_motivation(self):
        mock = _mock_classification_response("motivation")
        with patch("src.agent.router.get_client", return_value=mock):
            route = classify_message("I don't feel like training today")
        assert route == "motivation"

    def test_general_chat(self):
        mock = _mock_classification_response("general_chat")
        with patch("src.agent.router.get_client", return_value=mock):
            route = classify_message("Thanks, that helps!")
        assert route == "general_chat"

    def test_unknown_route_falls_back_to_general_chat(self):
        mock = _mock_classification_response("nonexistent_route")
        with patch("src.agent.router.get_client", return_value=mock):
            route = classify_message("something weird")
        assert route == "general_chat"

    def test_malformed_response_falls_back(self):
        mock_response = MagicMock()
        mock_response.text = "not json at all"
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        with patch("src.agent.router.get_client", return_value=mock_client):
            route = classify_message("test")
        assert route == "general_chat"

    def test_api_error_falls_back(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API error")
        with patch("src.agent.router.get_client", return_value=mock_client):
            route = classify_message("test")
        assert route == "general_chat"


# ── Budget Override Tests ───────────────────────────────────────────

class TestBudgetOverrides:
    """Verify route-specific budget adjustments."""

    def test_general_chat_zeros_activity_budget(self):
        overrides = get_budget_overrides("general_chat", BASE_BUDGETS)
        assert overrides["activity"] == 0

    def test_activity_question_boosts_activity_budget(self):
        overrides = get_budget_overrides("activity_question", BASE_BUDGETS)
        assert overrides["activity"] > BASE_BUDGETS["activity"]

    def test_system_budget_unchanged(self):
        """System budget should never be modified by routing."""
        for route in ROUTE_TYPES:
            overrides = get_budget_overrides(route, BASE_BUDGETS)
            assert overrides["system"] == BASE_BUDGETS["system"]

    def test_constraint_update_boosts_model_budget(self):
        overrides = get_budget_overrides("constraint_update", BASE_BUDGETS)
        assert overrides["model"] > BASE_BUDGETS["model"]

    def test_unknown_route_returns_base_budgets(self):
        overrides = get_budget_overrides("nonexistent", BASE_BUDGETS)
        assert overrides == BASE_BUDGETS

    def test_all_routes_produce_valid_budgets(self):
        """All budget values must be non-negative integers."""
        for route in ROUTE_TYPES:
            overrides = get_budget_overrides(route, BASE_BUDGETS)
            for key, value in overrides.items():
                assert isinstance(value, int), f"{route}:{key} is not int"
                assert value >= 0, f"{route}:{key} is negative"


# ── Different Routes Produce Different Budgets ──────────────────────

class TestRouteDifferentiation:
    """Verify routes actually produce DIFFERENT context configurations.

    This is the core of the routing value proposition: different messages
    should get different treatment.
    """

    def test_activity_vs_general_have_different_activity_budgets(self):
        act_budgets = get_budget_overrides("activity_question", BASE_BUDGETS)
        gen_budgets = get_budget_overrides("general_chat", BASE_BUDGETS)
        assert act_budgets["activity"] != gen_budgets["activity"]
        assert act_budgets["activity"] > gen_budgets["activity"]

    def test_constraint_vs_motivation_have_different_model_budgets(self):
        con_budgets = get_budget_overrides("constraint_update", BASE_BUDGETS)
        mot_budgets = get_budget_overrides("motivation", BASE_BUDGETS)
        assert con_budgets["model"] != mot_budgets["model"]

    def test_at_least_3_distinct_budget_profiles(self):
        """At least 3 route types should produce distinct budget configurations."""
        budget_sets = set()
        for route in ROUTE_TYPES:
            b = get_budget_overrides(route, BASE_BUDGETS)
            # Use a tuple of key values as a hashable representation
            budget_sets.add(tuple(sorted(b.items())))
        assert len(budget_sets) >= 3, f"Only {len(budget_sets)} distinct budget profiles"
