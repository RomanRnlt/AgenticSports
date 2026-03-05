"""Tests for onboarding-mode system prompt injection.

Verifies that:
- context="onboarding" appends ONBOARDING_MODE_INSTRUCTIONS to runtime context
- context="coach" (default) does NOT include onboarding instructions
- build_system_prompt returns only the static prompt (no runtime data)
- Runtime context is a separate string from the system prompt
"""

from unittest.mock import MagicMock

from src.agent.system_prompt import (
    ONBOARDING_MODE_INSTRUCTIONS,
    STATIC_SYSTEM_PROMPT,
    build_runtime_context,
    build_system_prompt,
)


def _make_mock_user_model(
    name: str = "TestUser",
    sports: list | None = None,
    goal_event: str | None = None,
    training_days: int | None = None,
    max_minutes: int | None = None,
) -> MagicMock:
    """Create a mock user_model with a configurable profile."""
    profile = {
        "name": name,
        "sports": sports or [],
        "goal": {"event": goal_event, "target_date": None},
        "constraints": {
            "training_days_per_week": training_days,
            "max_session_minutes": max_minutes,
        },
        "fitness": {},
    }
    mock = MagicMock()
    mock.project_profile.return_value = profile
    mock.get_active_beliefs.return_value = []
    mock.get_active_plan_summary.return_value = None
    return mock


class TestOnboardingPromptInjection:
    """Test that ONBOARDING_MODE_INSTRUCTIONS are injected based on context."""

    def test_onboarding_context_includes_instructions(self) -> None:
        """context='onboarding' appends onboarding instructions."""
        user_model = _make_mock_user_model()
        result = build_runtime_context(user_model, context="onboarding")

        assert "ONBOARDING MODE" in result
        assert "complete_onboarding" in result
        assert "define_session_schema" in result

    def test_coach_context_excludes_onboarding_instructions(self) -> None:
        """context='coach' (default) does NOT include onboarding instructions."""
        user_model = _make_mock_user_model()
        result = build_runtime_context(user_model, context="coach")

        assert "ONBOARDING MODE" not in result

    def test_default_context_is_coach(self) -> None:
        """Default context is 'coach' -- no onboarding instructions."""
        user_model = _make_mock_user_model()
        result = build_runtime_context(user_model)

        assert "ONBOARDING MODE" not in result

    def test_build_system_prompt_returns_only_static(self) -> None:
        """build_system_prompt returns ONLY the static prompt, no runtime data."""
        user_model = _make_mock_user_model()

        result = build_system_prompt(user_model, context="onboarding")

        # The system prompt is always the static prompt -- no runtime data
        assert result == STATIC_SYSTEM_PROMPT
        # No onboarding instructions in the system prompt itself
        assert "ONBOARDING MODE" not in result
        # No user-specific data
        assert "TestUser" not in result

    def test_build_system_prompt_identical_for_all_contexts(self) -> None:
        """build_system_prompt returns the same string regardless of context."""
        user_model = _make_mock_user_model()

        onboarding_prompt = build_system_prompt(user_model, context="onboarding")
        coach_prompt = build_system_prompt(user_model, context="coach")

        assert onboarding_prompt == coach_prompt
        assert onboarding_prompt == STATIC_SYSTEM_PROMPT

    def test_onboarding_instructions_mention_completion_sequence(self) -> None:
        """ONBOARDING_MODE_INSTRUCTIONS describes the full setup sequence."""
        assert "define_session_schema" in ONBOARDING_MODE_INSTRUCTIONS
        assert "define_metric" in ONBOARDING_MODE_INSTRUCTIONS
        assert "define_eval_criteria" in ONBOARDING_MODE_INSTRUCTIONS
        assert "create_training_plan" in ONBOARDING_MODE_INSTRUCTIONS
        assert "complete_onboarding" in ONBOARDING_MODE_INSTRUCTIONS

    def test_onboarding_still_shows_missing_fields(self) -> None:
        """Onboarding context still includes the missing-fields section."""
        user_model = _make_mock_user_model(name="", sports=[], goal_event=None)
        result = build_runtime_context(user_model, context="onboarding")

        assert "Missing:" in result or "missing" in result.lower()
        assert "ONBOARDING MODE" in result

    def test_coach_context_with_startup_context(self) -> None:
        """Coach context with startup_context works normally."""
        user_model = _make_mock_user_model(name="Marco", sports=["running"])
        result = build_runtime_context(
            user_model,
            startup_context="Recent: 3 runs this week",
            context="coach",
        )

        assert "Recent: 3 runs this week" in result
        assert "ONBOARDING MODE" not in result


class TestSystemPromptCaching:
    """Test that system prompt is truly static and cacheable."""

    def test_system_prompt_has_no_runtime_data(self) -> None:
        """The static system prompt contains no date-specific or f-string data."""
        # Should not contain any real date references
        assert "2026" not in STATIC_SYSTEM_PROMPT
        assert "Monday" not in STATIC_SYSTEM_PROMPT

        # Should not contain real user names (note: "Marco" appears in
        # few-shot examples which is fine -- it's static example data)
        assert "Roman" not in STATIC_SYSTEM_PROMPT

    def test_runtime_context_is_separate_from_system_prompt(self) -> None:
        """Runtime context and system prompt are fully separate strings."""
        user_model = _make_mock_user_model(name="TestAthlete99", sports=["running"])

        system = build_system_prompt(user_model)
        runtime = build_runtime_context(user_model, date="2026-03-05")

        # System prompt should NOT contain runtime data
        assert "TestAthlete99" not in system
        assert "2026-03-05" not in system

        # Runtime context SHOULD contain runtime data
        assert "TestAthlete99" in runtime
        assert "2026-03-05" in runtime
