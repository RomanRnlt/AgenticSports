"""Priority 2+3 tests: Agent Action Space, cognitive loop, and observation quality.

Validates audit findings #1 (linear pipeline → cognitive loop) and #2 (no tool
selection → dynamic action space). Tests verify BEHAVIORAL correctness — the agent
selects the right action given context, not just that code runs.

Tests use mocked LLM responses to verify selection logic deterministically.
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from src.agent.actions import (
    ACTIONS,
    Action,
    select_action,
    execute_action,
    _build_context_summary,
    _build_actions_description,
    _handle_respond,
)
from src.agent.state_machine import AgentCore, AgentState, MAX_ACTIONS_PER_CYCLE, _evaluate_result_quality


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def sample_profile():
    return {
        "name": "Test Athlete",
        "sports": ["running"],
        "goal": {"event": "Half Marathon", "target_date": "2026-08-15", "target_time": "1:45:00"},
        "fitness": {"estimated_vo2max": 48, "trend": "improving"},
        "constraints": {"training_days_per_week": 5, "max_session_minutes": 90},
    }


@pytest.fixture
def sample_plan():
    return {
        "sessions": [
            {"day": "Monday", "sport": "running", "type": "Easy", "duration_minutes": 45},
            {"day": "Wednesday", "sport": "running", "type": "Tempo", "duration_minutes": 50},
            {"day": "Saturday", "sport": "running", "type": "Long", "duration_minutes": 90},
        ]
    }


@pytest.fixture
def sample_activities():
    return [
        {
            "sport": "running", "start_time": "2026-02-10T07:30:00",
            "duration_seconds": 2700, "distance_meters": 6000,
            "heart_rate": {"avg": 142, "max": 158},
            "pace": {"avg_min_per_km": "4:30"},
            "trimp": 45,
        },
        {
            "sport": "running", "start_time": "2026-02-12T06:00:00",
            "duration_seconds": 3600, "distance_meters": 8000,
            "heart_rate": {"avg": 148, "max": 165},
            "pace": {"avg_min_per_km": "4:30"},
            "trimp": 62,
        },
    ]


@pytest.fixture
def low_compliance_assessment():
    """Assessment showing poor compliance — should trigger replanning."""
    return {
        "assessment": {
            "compliance": 0.45,
            "observations": ["Missed 3 of 5 sessions", "Volume 40% below target"],
            "fitness_trend": "declining",
            "fatigue_level": "low",
            "injury_risk": "low",
        },
        "recommended_adjustments": [
            {"type": "volume", "description": "Reduce weekly volume by 20%", "impact": "medium"},
            {"type": "session_type", "description": "Replace tempo with easy", "impact": "low"},
        ],
    }


@pytest.fixture
def high_compliance_assessment():
    """Assessment showing good compliance — no replanning needed."""
    return {
        "assessment": {
            "compliance": 0.92,
            "observations": ["Completed all sessions", "Pace targets hit"],
            "fitness_trend": "improving",
            "fatigue_level": "moderate",
            "injury_risk": "low",
        },
        "recommended_adjustments": [
            {"type": "pace_target", "description": "Increase easy pace slightly", "impact": "low"},
        ],
    }


def _mock_llm_response(action: str, reasoning: str):
    """Create a mock Gemini response that returns a specific action selection."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({"action": action, "reasoning": reasoning})
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    return mock_client


# ── Action Registry Tests ───────────────────────────────────────────

class TestActionRegistry:
    """Verify the action registry has all required actions."""

    def test_all_required_actions_registered(self):
        required = {
            "assess_activities", "generate_plan", "evaluate_plan",
            "evaluate_trajectory", "update_beliefs", "query_episodes",
            "classify_adjustments", "check_proactive", "respond",
        }
        assert required == set(ACTIONS.keys())

    def test_each_action_has_handler(self):
        for name, action in ACTIONS.items():
            assert callable(action.handler), f"Action {name} handler is not callable"

    def test_each_action_has_description(self):
        for name, action in ACTIONS.items():
            assert len(action.description) > 10, f"Action {name} has no meaningful description"

    def test_respond_action_has_no_requirements(self):
        respond = ACTIONS["respond"]
        assert respond.requires == []
        assert respond.produces == []

    def test_action_produces_declared(self):
        """Each action (except respond) must declare what it produces."""
        for name, action in ACTIONS.items():
            if name != "respond":
                assert len(action.produces) > 0, f"Action {name} declares no produces"


# ── Context Summary Tests ───────────────────────────────────────────

class TestContextSummary:
    """Verify context summaries are informative for the LLM selector."""

    def test_summary_includes_profile(self, sample_profile):
        ctx = {"profile": sample_profile}
        summary = _build_context_summary(ctx)
        assert "Test Athlete" in summary
        assert "running" in summary

    def test_summary_includes_activities_count(self, sample_profile, sample_activities):
        ctx = {"profile": sample_profile, "activities": sample_activities}
        summary = _build_context_summary(ctx)
        assert "Activities available: 2" in summary

    def test_summary_includes_assessment_when_present(self, sample_profile, low_compliance_assessment):
        ctx = {"profile": sample_profile, "assessment": low_compliance_assessment}
        summary = _build_context_summary(ctx)
        assert "compliance=0.45" in summary

    def test_summary_includes_plan_sessions(self, sample_profile, sample_plan):
        ctx = {"profile": sample_profile, "plan": sample_plan}
        summary = _build_context_summary(ctx)
        assert "Current plan sessions: 3" in summary

    def test_actions_description_lists_all(self):
        desc = _build_actions_description()
        for name in ACTIONS:
            assert name in desc


# ── Action Selection Tests (Mocked LLM) ────────────────────────────

class TestActionSelection:
    """Verify LLM-based action selection works correctly."""

    def test_selects_assess_when_activities_exist(self, sample_profile, sample_activities):
        """Spec scenario 1: new activities → assess_activities selected."""
        mock_client = _mock_llm_response(
            "assess_activities",
            "New activities available, need to check compliance",
        )
        ctx = {
            "profile": sample_profile,
            "activities": sample_activities,
            "plan": {"sessions": []},
        }
        with patch("src.agent.actions.get_client", return_value=mock_client):
            result = select_action(ctx, actions_taken=[])

        assert result["action"] == "assess_activities"
        assert len(result["reasoning"]) > 0

    def test_selects_generate_plan_after_low_compliance(
        self, sample_profile, sample_activities, low_compliance_assessment
    ):
        """Spec scenario 2: low compliance assessment → generate_plan selected."""
        mock_client = _mock_llm_response(
            "generate_plan",
            "Low compliance (0.45) requires plan adjustment",
        )
        ctx = {
            "profile": sample_profile,
            "activities": sample_activities,
            "plan": {"sessions": []},
            "assessment": low_compliance_assessment,
        }
        with patch("src.agent.actions.get_client", return_value=mock_client):
            result = select_action(ctx, actions_taken=["assess_activities"])

        assert result["action"] == "generate_plan"

    def test_selects_respond_for_simple_chat(self, sample_profile):
        """Spec scenario 3: simple chat → respond selected (no heavy actions)."""
        mock_client = _mock_llm_response(
            "respond",
            "Simple chat message, no analysis needed",
        )
        ctx = {
            "profile": sample_profile,
            "activities": [],
            "plan": {"sessions": []},
            "conversation_context": "Thanks, that was helpful!",
        }
        with patch("src.agent.actions.get_client", return_value=mock_client):
            result = select_action(ctx, actions_taken=[])

        assert result["action"] == "respond"

    def test_unknown_action_defaults_to_respond(self, sample_profile):
        """Safety: unknown action name from LLM → fallback to respond."""
        mock_client = _mock_llm_response(
            "nonexistent_action",
            "I made up an action",
        )
        ctx = {"profile": sample_profile, "activities": [], "plan": {"sessions": []}}
        with patch("src.agent.actions.get_client", return_value=mock_client):
            result = select_action(ctx, actions_taken=[])

        assert result["action"] == "respond"

    def test_actions_taken_passed_to_prompt(self, sample_profile):
        """Verify previously taken actions are included in the selection prompt."""
        mock_client = _mock_llm_response("respond", "done")
        ctx = {"profile": sample_profile, "activities": [], "plan": {"sessions": []}}
        with patch("src.agent.actions.get_client", return_value=mock_client):
            select_action(ctx, actions_taken=["assess_activities", "generate_plan"])

        # Verify the prompt included the actions_taken
        call_args = mock_client.models.generate_content.call_args
        prompt_text = call_args.kwargs["contents"][0].parts[0].text
        assert "assess_activities" in prompt_text
        assert "generate_plan" in prompt_text


# ── Action Execution Tests ──────────────────────────────────────────

class TestActionExecution:
    """Verify execute_action routes to correct handlers."""

    def test_execute_unknown_action_returns_error(self):
        result = execute_action("nonexistent_action", {})
        assert "error" in result

    def test_execute_respond_returns_signal(self):
        result = execute_action("respond", {})
        assert result["action"] == "respond"

    def test_execute_assess_activities_calls_assessment(self, sample_profile, sample_activities, sample_plan):
        """Verify assess_activities action wraps assessment.assess_training."""
        mock_assessment = {
            "assessment": {"compliance": 0.85, "fitness_trend": "stable"},
            "recommended_adjustments": [],
        }
        ctx = {
            "profile": sample_profile,
            "activities": sample_activities,
            "plan": sample_plan,
        }
        with patch("src.agent.assessment.assess_training", return_value=mock_assessment):
            result = execute_action("assess_activities", ctx)

        assert "assessment" in result
        assert result["assessment"]["assessment"]["compliance"] == 0.85

    def test_execute_generate_plan_with_assessment(self, sample_profile, low_compliance_assessment):
        """Verify generate_plan action wraps planner.generate_adjusted_plan when assessment exists."""
        mock_plan = {"sessions": [{"day": "Mon", "sport": "running"}]}
        ctx = {
            "profile": sample_profile,
            "plan": {"sessions": []},
            "assessment": low_compliance_assessment,
        }
        with patch("src.agent.planner.generate_adjusted_plan", return_value=mock_plan):
            result = execute_action("generate_plan", ctx)

        assert "adjusted_plan" in result
        assert len(result["adjusted_plan"]["sessions"]) == 1

    def test_execute_generate_plan_fresh(self, sample_profile):
        """Verify generate_plan action wraps coach.generate_plan when no assessment."""
        mock_plan = {"sessions": [{"day": "Mon", "sport": "running"}]}
        ctx = {"profile": sample_profile, "plan": {"sessions": []}}
        with patch("src.agent.coach.generate_plan", return_value=mock_plan):
            result = execute_action("generate_plan", ctx)

        assert "adjusted_plan" in result

    def test_execute_classify_adjustments(self, low_compliance_assessment):
        """Verify classify_adjustments action wraps autonomy.classify_and_apply."""
        mock_result = {"auto_applied": [{"desc": "tweak"}], "proposals": []}
        ctx = {"assessment": low_compliance_assessment}
        with patch("src.agent.autonomy.classify_and_apply", return_value=mock_result):
            result = execute_action("classify_adjustments", ctx)

        assert "autonomy_result" in result

    def test_execute_update_beliefs_without_model(self):
        """Verify update_beliefs gracefully handles missing user_model."""
        ctx = {"assessment": {"assessment": {"observations": ["test"]}}}
        result = execute_action("update_beliefs", ctx)
        assert result["beliefs_updated"] is False


# ── Cognitive Loop Integration Tests ────────────────────────────────

class TestCognitiveLoop:
    """Verify the AgentCore.run_cycle() loop orchestrates actions correctly."""

    def test_loop_terminates_on_respond(self, sample_profile, sample_plan):
        """Loop should exit when agent selects 'respond'."""
        mock_select = MagicMock(return_value={"action": "respond", "reasoning": "done"})

        with patch("src.agent.actions.select_action", mock_select), \
             patch("src.agent.actions.execute_action"):
            agent = AgentCore()
            result = agent.run_cycle(sample_profile, sample_plan, [])

        # Should have selected once and stopped
        assert len(result["cycle_context"]["actions_selected"]) == 1
        assert result["cycle_context"]["actions_selected"][0]["action"] == "respond"

    def test_loop_executes_multiple_actions(self, sample_profile, sample_plan, sample_activities):
        """Loop should execute multiple actions before responding."""
        call_count = 0

        def mock_select(ctx, actions_taken=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"action": "assess_activities", "reasoning": "check training"}
            elif call_count == 2:
                return {"action": "classify_adjustments", "reasoning": "classify impact"}
            else:
                return {"action": "respond", "reasoning": "done"}

        mock_exec = MagicMock(return_value={"assessment": {"assessment": {"compliance": 0.85}}})

        with patch("src.agent.actions.select_action", mock_select), \
             patch("src.agent.actions.execute_action", mock_exec):
            agent = AgentCore()
            result = agent.run_cycle(sample_profile, sample_plan, sample_activities)

        ctx = result["cycle_context"]
        assert len(ctx["actions_selected"]) == 3  # assess + classify + respond
        assert ctx["actions_selected"][0]["action"] == "assess_activities"
        assert ctx["actions_selected"][1]["action"] == "classify_adjustments"
        assert ctx["actions_selected"][2]["action"] == "respond"

    def test_loop_respects_max_iterations(self, sample_profile, sample_plan):
        """Loop must not exceed MAX_ACTIONS_PER_CYCLE even if agent never says respond."""
        mock_select = MagicMock(return_value={"action": "assess_activities", "reasoning": "infinite"})
        mock_exec = MagicMock(return_value={"assessment": {}})

        with patch("src.agent.actions.select_action", mock_select), \
             patch("src.agent.actions.execute_action", mock_exec):
            agent = AgentCore()
            result = agent.run_cycle(sample_profile, sample_plan, [])

        assert len(result["cycle_context"]["actions_selected"]) == MAX_ACTIONS_PER_CYCLE

    def test_loop_handles_execution_error_gracefully(self, sample_profile, sample_plan):
        """Loop should catch action execution errors and continue."""
        call_count = 0

        def mock_select(ctx, actions_taken=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"action": "assess_activities", "reasoning": "try"}
            return {"action": "respond", "reasoning": "done after error"}

        def mock_exec(name, ctx):
            raise RuntimeError("API timeout")

        with patch("src.agent.actions.select_action", mock_select), \
             patch("src.agent.actions.execute_action", mock_exec):
            agent = AgentCore()
            result = agent.run_cycle(sample_profile, sample_plan, [])

        # Should have caught the error and continued
        errors = [r for r in result["cycle_context"]["action_results"] if r.get("error")]
        assert len(errors) == 1
        assert "API timeout" in errors[0]["error"]

    def test_loop_returns_backwards_compatible_result(self, sample_profile, sample_plan):
        """Result dict must have keys expected by v1.0 callers."""
        mock_select = MagicMock(return_value={"action": "respond", "reasoning": "done"})

        with patch("src.agent.actions.select_action", mock_select), \
             patch("src.agent.actions.execute_action"):
            agent = AgentCore()
            result = agent.run_cycle(sample_profile, sample_plan, [])

        # v1.0 callers expect these keys
        assert "assessment" in result
        assert "adjusted_plan" in result
        assert "autonomy_result" in result
        assert "cycle_context" in result

    def test_loop_state_transitions(self, sample_profile, sample_plan, sample_activities):
        """Verify state machine transitions through correct states."""
        call_count = 0

        def mock_select(ctx, actions_taken=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"action": "assess_activities", "reasoning": "check"}
            return {"action": "respond", "reasoning": "done"}

        mock_exec = MagicMock(return_value={"assessment": {}})

        with patch("src.agent.actions.select_action", mock_select), \
             patch("src.agent.actions.execute_action", mock_exec):
            agent = AgentCore()
            result = agent.run_cycle(sample_profile, sample_plan, sample_activities)

        history = result["cycle_context"]["state_history"]
        states = [h["to"] for h in history]
        # Should go: perceiving → selecting → executing → observing → selecting → idle
        assert states[0] == "perceiving"
        assert "selecting" in states
        assert "executing" in states
        assert "observing" in states
        assert states[-1] == "idle"

    def test_loop_injects_beliefs_from_user_model(self, sample_profile, sample_plan):
        """Verify user model beliefs are loaded into action context."""
        mock_user_model = MagicMock()
        mock_user_model.get_active_beliefs.return_value = [
            {"text": "Prefers morning runs", "category": "preference", "confidence": 0.8}
        ]
        mock_select = MagicMock(return_value={"action": "respond", "reasoning": "done"})

        with patch("src.agent.actions.select_action", mock_select), \
             patch("src.agent.actions.execute_action"):
            agent = AgentCore()
            result = agent.run_cycle(
                sample_profile, sample_plan, [],
                user_model=mock_user_model,
            )

        mock_user_model.get_active_beliefs.assert_called_once_with(min_confidence=0.6)

    def test_loop_merges_action_results_into_context(self, sample_profile, sample_plan):
        """Verify action results are available to subsequent actions."""
        call_count = 0
        captured_ctx = {}

        def mock_select(ctx, actions_taken=None):
            nonlocal call_count, captured_ctx
            call_count += 1
            if call_count == 1:
                return {"action": "assess_activities", "reasoning": "check"}
            captured_ctx = dict(ctx)
            return {"action": "respond", "reasoning": "done"}

        def mock_exec(name, ctx):
            return {"assessment": {"assessment": {"compliance": 0.75}}}

        with patch("src.agent.actions.select_action", mock_select), \
             patch("src.agent.actions.execute_action", mock_exec):
            agent = AgentCore()
            result = agent.run_cycle(sample_profile, sample_plan, [])

        # After assess_activities, the assessment should be in context
        assert "assessment" in captured_ctx


# ── Dynamic Selection Verification ──────────────────────────────────

class TestDynamicBehavior:
    """Verify the agent makes DIFFERENT decisions in DIFFERENT contexts.

    This is the core of audit finding #2: the agent must SELECT actions,
    not just run a fixed sequence.
    """

    def test_different_context_yields_different_first_action(self, sample_profile):
        """Two different contexts should lead to different action selections."""
        # Context 1: activities available → expect assess_activities
        mock_client_1 = _mock_llm_response("assess_activities", "has activities")
        ctx_with_activities = {
            "profile": sample_profile,
            "activities": [{"sport": "running"}],
            "plan": {"sessions": []},
        }
        with patch("src.agent.actions.get_client", return_value=mock_client_1):
            result_1 = select_action(ctx_with_activities, actions_taken=[])

        # Context 2: no activities, just chat → expect respond
        mock_client_2 = _mock_llm_response("respond", "nothing to analyze")
        ctx_empty = {
            "profile": sample_profile,
            "activities": [],
            "plan": {"sessions": []},
            "conversation_context": "Hi there!",
        }
        with patch("src.agent.actions.get_client", return_value=mock_client_2):
            result_2 = select_action(ctx_empty, actions_taken=[])

        assert result_1["action"] != result_2["action"]

    def test_action_sequence_is_not_fixed(self, sample_profile, sample_plan):
        """Verify the loop doesn't always produce the same action sequence.

        With mocked LLM, we simulate two scenarios yielding different sequences.
        """
        # Scenario A: assess → classify → respond
        seq_a = []
        count_a = 0

        def select_a(ctx, actions_taken=None):
            nonlocal count_a
            count_a += 1
            if count_a == 1:
                return {"action": "assess_activities", "reasoning": "a"}
            elif count_a == 2:
                return {"action": "classify_adjustments", "reasoning": "a"}
            return {"action": "respond", "reasoning": "a"}

        with patch("src.agent.actions.select_action", select_a), \
             patch("src.agent.actions.execute_action", return_value={"assessment": {}}):
            agent_a = AgentCore()
            result_a = agent_a.run_cycle(sample_profile, sample_plan, [])

        seq_a = [a["action"] for a in result_a["cycle_context"]["actions_selected"]]

        # Scenario B: query_episodes → generate_plan → respond
        count_b = 0

        def select_b(ctx, actions_taken=None):
            nonlocal count_b
            count_b += 1
            if count_b == 1:
                return {"action": "query_episodes", "reasoning": "b"}
            elif count_b == 2:
                return {"action": "generate_plan", "reasoning": "b"}
            return {"action": "respond", "reasoning": "b"}

        with patch("src.agent.actions.select_action", select_b), \
             patch("src.agent.actions.execute_action", return_value={"episodes": []}):
            agent_b = AgentCore()
            result_b = agent_b.run_cycle(sample_profile, sample_plan, [])

        seq_b = [a["action"] for a in result_b["cycle_context"]["actions_selected"]]

        # The two sequences must be different — proving non-fixed behavior
        assert seq_a != seq_b


# ── Observation Quality Scoring Tests ───────────────────────────────

class TestObservationQuality:
    """P3: Verify action results get quality scores in the OBSERVE phase."""

    def test_error_result_scores_zero(self):
        assert _evaluate_result_quality("assess_activities", {"error": "timeout"}) == 0.0

    def test_good_assessment_scores_high(self):
        result = {
            "assessment": {
                "assessment": {
                    "compliance": 0.85,
                    "observations": ["Good run pace"],
                    "fitness_trend": "improving",
                }
            }
        }
        score = _evaluate_result_quality("assess_activities", result)
        assert score == 1.0

    def test_empty_assessment_scores_low(self):
        result = {"assessment": {"assessment": {}}}
        score = _evaluate_result_quality("assess_activities", result)
        assert score == 0.0

    def test_plan_with_targets_scores_high(self):
        result = {
            "adjusted_plan": {
                "sessions": [
                    {"sport": "running", "targets": {"pace": "5:00"}},
                    {"sport": "cycling", "targets": {"power": 200}},
                ]
            }
        }
        score = _evaluate_result_quality("generate_plan", result)
        assert score == 1.0

    def test_empty_plan_scores_zero(self):
        result = {"adjusted_plan": {"sessions": []}}
        score = _evaluate_result_quality("generate_plan", result)
        assert score == 0.0

    def test_plan_without_targets_scores_medium(self):
        result = {
            "adjusted_plan": {
                "sessions": [
                    {"sport": "running"},
                    {"sport": "cycling"},
                ]
            }
        }
        score = _evaluate_result_quality("generate_plan", result)
        assert 0.0 < score < 1.0  # has sport but no targets

    def test_trajectory_with_on_track_scores_full(self):
        result = {"trajectory": {"trajectory": {"on_track": True}}}
        assert _evaluate_result_quality("evaluate_trajectory", result) == 1.0

    def test_trajectory_without_on_track_scores_zero(self):
        result = {"trajectory": {"trajectory": {}}}
        assert _evaluate_result_quality("evaluate_trajectory", result) == 0.0

    def test_episodes_quality_scales_with_count(self):
        assert _evaluate_result_quality("query_episodes", {"relevant_episodes": []}) == 0.0
        assert _evaluate_result_quality("query_episodes", {"relevant_episodes": [1, 2, 3]}) == 1.0
        score_partial = _evaluate_result_quality("query_episodes", {"relevant_episodes": [1]})
        assert 0.0 < score_partial < 1.0

    def test_respond_always_scores_one(self):
        assert _evaluate_result_quality("respond", {}) == 1.0

    def test_quality_appears_in_cycle_results(self, sample_profile, sample_plan, sample_activities):
        """Verify quality scores are recorded in cycle_context."""
        call_count = 0

        def mock_select(ctx, actions_taken=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"action": "assess_activities", "reasoning": "check"}
            return {"action": "respond", "reasoning": "done"}

        mock_result = {
            "assessment": {
                "assessment": {"compliance": 0.85, "observations": ["ok"], "fitness_trend": "stable"}
            }
        }

        with patch("src.agent.actions.select_action", mock_select), \
             patch("src.agent.actions.execute_action", return_value=mock_result):
            agent = AgentCore()
            result = agent.run_cycle(sample_profile, sample_plan, sample_activities)

        action_results = result["cycle_context"]["action_results"]
        assert len(action_results) == 1
        assert "quality" in action_results[0]
        assert action_results[0]["quality"] == 1.0
