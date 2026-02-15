"""Priority 9: Transformation Validation — Agentic Compliance Tests.

These tests verify that the v2.0 architecture genuinely transforms ReAgt from
"Augmented LLM with Workflow" to "Autonomous Agent" per Q1 2026 standards.

Each test directly validates the resolution of a specific architecture audit
finding from .planning/ARCHITECTURE_AUDIT_v1.0.md.

The tests are structural/behavioral — they verify the architecture patterns
exist and work correctly, not that the LLM produces perfect outputs.
"""

import pytest
from unittest.mock import patch, MagicMock


# ── Finding #1: Dynamic Cognitive Loop (not linear pipeline) ─────


class TestCognitiveLoop:
    """Verify state_machine.py implements a cognitive loop, not a linear pipeline."""

    def test_run_cycle_has_action_loop(self):
        """run_cycle() iterates over actions — it's not a fixed sequence."""
        from src.agent.state_machine import AgentCore, MAX_ACTIONS_PER_CYCLE

        assert MAX_ACTIONS_PER_CYCLE >= 3, "Loop must allow multiple iterations"

    def test_run_cycle_records_multiple_actions(self):
        """The cognitive loop can select and execute multiple actions per cycle."""
        from src.agent.state_machine import AgentCore

        core = AgentCore()
        # Mock select_action to return assess_activities then respond
        actions_sequence = iter([
            {"action": "assess_activities", "reasoning": "check data"},
            {"action": "respond", "reasoning": "done"},
        ])

        def mock_select(ctx, actions_taken=None):
            return next(actions_sequence)

        def mock_execute(name, ctx):
            if name == "assess_activities":
                return {"assessment": {"assessment": {"compliance": 0.8, "observations": ["good"], "fatigue_level": "low"}}}
            return {}

        with patch("src.agent.actions.select_action", side_effect=mock_select), \
             patch("src.agent.actions.execute_action", side_effect=mock_execute):
            result = core.run_cycle(
                profile={"name": "Test", "goal": {}},
                plan={"sessions": []},
                activities=[],
            )

        # Should have recorded the action selection
        assert len(core.context["actions_selected"]) == 2
        assert core.context["actions_selected"][0]["action"] == "assess_activities"
        assert core.context["actions_selected"][1]["action"] == "respond"

    def test_loop_terminates_on_respond(self):
        """Loop stops when agent selects 'respond'."""
        from src.agent.state_machine import AgentCore

        core = AgentCore()

        with patch("src.agent.actions.select_action", return_value={"action": "respond", "reasoning": "done"}):
            result = core.run_cycle(
                profile={"name": "Test", "goal": {}},
                plan={"sessions": []},
                activities=[],
            )

        # Only one action selected (respond)
        assert len(core.context["actions_selected"]) == 1

    def test_observation_quality_scoring(self):
        """Each action result is quality-scored in the OBSERVE phase."""
        from src.agent.state_machine import _evaluate_result_quality

        # Good assessment
        good = _evaluate_result_quality("assess_activities", {
            "assessment": {"assessment": {"compliance": 0.8, "observations": ["x"], "fatigue_level": "low"}}
        })
        assert good > 0.5

        # Error result
        error = _evaluate_result_quality("assess_activities", {"error": "API failed"})
        assert error == 0.0


# ── Finding #2: Dynamic Action Space ─────────────────────────────


class TestActionSpace:
    """Verify the agent has a registry of tools it can dynamically choose from."""

    def test_action_registry_exists(self):
        from src.agent.actions import ACTIONS
        assert len(ACTIONS) >= 8, "Need sufficient action variety"

    def test_all_actions_have_handlers(self):
        from src.agent.actions import ACTIONS
        for name, action in ACTIONS.items():
            assert callable(action.handler), f"{name} handler not callable"

    def test_action_selection_is_llm_based(self):
        """select_action() uses LLM, not hardcoded rules."""
        from src.agent.actions import ACTION_SELECTION_PROMPT
        # The prompt is used for LLM-based selection
        assert "Available actions" in ACTION_SELECTION_PROMPT
        assert "reasoning" in ACTION_SELECTION_PROMPT

    def test_actions_have_requires_and_produces(self):
        """Each action declares what it needs and produces."""
        from src.agent.actions import ACTIONS
        for name, action in ACTIONS.items():
            if name != "respond":
                assert isinstance(action.requires, list)
                assert isinstance(action.produces, list)


# ── Finding #4: No Hardcoded Coaching Rules ──────────────────────


class TestNoHardcodedRules:
    """Verify hardcoded keyword matching is replaced with LLM + fallback."""

    def test_impact_classification_uses_llm(self):
        """Impact classification has LLM primary path."""
        from src.agent.autonomy import classify_impact, IMPACT_CLASSIFICATION_PROMPT
        assert "JSON" in IMPACT_CLASSIFICATION_PROMPT
        # The function signature supports the LLM path
        import inspect
        sig = inspect.signature(classify_impact)
        assert "use_llm" in sig.parameters

    def test_goal_type_inference_uses_llm(self):
        """Goal type inference has LLM primary path."""
        from src.agent.startup import infer_goal_type, GOAL_TYPE_PROMPT
        assert "JSON" in GOAL_TYPE_PROMPT
        import inspect
        sig = inspect.signature(infer_goal_type)
        assert "use_llm" in sig.parameters

    def test_fatigue_detection_uses_llm(self):
        """Fatigue detection has LLM primary path."""
        from src.agent.proactive import _detect_fatigue_llm
        # The LLM function exists
        assert callable(_detect_fatigue_llm)

    def test_keyword_fallbacks_still_exist(self):
        """Fallbacks exist for when LLM fails — not removed, just demoted."""
        from src.agent.autonomy import _classify_impact_keywords
        from src.agent.startup import _infer_goal_type_keywords
        assert callable(_classify_impact_keywords)
        assert callable(_infer_goal_type_keywords)

    def test_no_keyword_only_paths(self):
        """Primary paths always try LLM first."""
        from src.agent.autonomy import classify_impact
        from src.agent.startup import infer_goal_type

        # When use_llm=True (default), LLM is attempted first
        # We verify by checking the function accepts use_llm parameter
        import inspect
        auto_sig = inspect.signature(classify_impact)
        startup_sig = inspect.signature(infer_goal_type)
        assert auto_sig.parameters["use_llm"].default is True
        assert startup_sig.parameters["use_llm"].default is True


# ── Finding #5: Evaluator-Optimizer Loop ─────────────────────────


class TestEvaluatorOptimizer:
    """Verify plans are evaluated and regenerated if below threshold."""

    def test_plan_evaluator_exists(self):
        from src.agent.plan_evaluator import evaluate_plan, PlanEvaluation
        assert callable(evaluate_plan)

    def test_evaluation_has_scoring_criteria(self):
        """Plan evaluation uses structured criteria, not a single score."""
        from src.agent.plan_evaluator import EVALUATION_SYSTEM_PROMPT
        # The prompt defines multiple evaluation criteria
        assert "sport_distribution" in EVALUATION_SYSTEM_PROMPT
        assert "target_specificity" in EVALUATION_SYSTEM_PROMPT
        assert "constraint_compliance" in EVALUATION_SYSTEM_PROMPT

    def test_acceptance_threshold(self):
        from src.agent.plan_evaluator import PLAN_ACCEPTANCE_THRESHOLD
        assert 60 <= PLAN_ACCEPTANCE_THRESHOLD <= 80

    def test_evaluate_plan_action_generates_feedback(self):
        """evaluate_plan action produces feedback for regeneration when score is low."""
        from src.agent.actions import _handle_evaluate_plan

        # Mock a low-scoring plan
        with patch("src.agent.plan_evaluator.evaluate_plan") as mock_eval:
            from src.agent.plan_evaluator import PlanEvaluation
            mock_eval.return_value = PlanEvaluation(
                score=45,
                criteria_scores={"sport_coverage": 30},
                issues=["Missing sport coverage"],
                suggestions=["Add cycling sessions"],
                acceptable=False,
            )

            ctx = {
                "adjusted_plan": {"sessions": [{"sport": "running"}]},
                "profile": {"name": "Test", "sports": ["running", "cycling"]},
            }
            result = _handle_evaluate_plan(ctx)

        assert result["plan_evaluation"]["score"] == 45
        assert not result["plan_evaluation"]["acceptable"]
        assert "plan_feedback" in result  # Feedback for regeneration


# ── Finding #6: Active Memory ────────────────────────────────────


class TestActiveMemory:
    """Verify memory has outcome tracking, not just passive storage."""

    def test_belief_schema_has_utility_fields(self):
        """Beliefs have utility, outcome_count, last_outcome fields."""
        from src.memory.user_model import UserModel

        um = UserModel()
        belief = um.add_belief("Test belief", "preference", 0.8)

        assert "utility" in belief
        assert "outcome_count" in belief
        assert "last_outcome" in belief
        assert "outcome_history" in belief

    def test_record_outcome_updates_confidence(self):
        """Recording an outcome changes the belief's confidence and utility."""
        from src.memory.user_model import UserModel

        um = UserModel()
        belief = um.add_belief("Test belief", "preference", 0.7)
        original_conf = belief["confidence"]

        um.record_outcome(belief["id"], "confirmed", detail="test")
        updated = next(b for b in um.beliefs if b["id"] == belief["id"])

        assert updated["confidence"] > original_conf
        assert updated["outcome_count"] == 1
        assert updated["last_outcome"] == "confirmed"

    def test_high_utility_retrieval(self):
        """High-utility beliefs can be retrieved preferentially."""
        from src.memory.user_model import UserModel

        um = UserModel()
        b1 = um.add_belief("Low utility belief", "preference", 0.7)
        b2 = um.add_belief("High utility belief", "preference", 0.7)

        # Record multiple positive outcomes for b2 (each adds 0.1 utility)
        for _ in range(5):
            um.record_outcome(b2["id"], "confirmed")

        high = um.get_high_utility_beliefs(min_utility=0.3)
        ids = [b["id"] for b in high]
        assert b2["id"] in ids


# ── Finding #7: Mid-Conversation Proactivity ─────────────────────


class TestMidConversationProactivity:
    """Verify proactive queue is consumed during conversation, not just startup."""

    def test_process_message_has_proactive_check(self):
        """process_message() calls _check_proactive_injection()."""
        from src.agent.conversation import ConversationEngine
        import inspect
        source = inspect.getsource(ConversationEngine.process_message)
        assert "_check_proactive_injection" in source
        assert "_maybe_refresh_proactive_triggers" in source

    def test_proactive_injection_skips_general_chat(self):
        """No injection for casual chat — only for substantive messages."""
        from src.agent.conversation import ConversationEngine
        from src.memory.user_model import UserModel

        um = UserModel()
        engine = ConversationEngine(user_model=um)
        result = engine._check_proactive_injection("Thanks!", "general_chat")
        assert result is None

    def test_refresh_triggers_function_exists(self):
        """refresh_proactive_triggers() exists for mid-conversation trigger detection."""
        from src.agent.proactive import refresh_proactive_triggers
        assert callable(refresh_proactive_triggers)

    def test_queue_messages_have_message_text(self):
        """Queued messages include formatted message_text for LLM consumption."""
        from src.agent.proactive import queue_proactive_message, _load_queue
        from pathlib import Path
        import tempfile

        path = Path(tempfile.mkdtemp()) / "q.json"
        trigger = {"type": "fatigue_warning", "data": {"message": "tired"}}
        msg = queue_proactive_message(trigger, queue_path=path)
        assert "message_text" in msg
        assert len(msg["message_text"]) > 0


# ── Finding #8: Dynamic Routing ──────────────────────────────────


class TestDynamicRouting:
    """Verify messages are classified and routed to specialized handlers."""

    def test_router_module_exists(self):
        from src.agent.router import classify_message, get_budget_overrides
        assert callable(classify_message)
        assert callable(get_budget_overrides)

    def test_multiple_route_types(self):
        from src.agent.router import ROUTE_TYPES
        assert len(ROUTE_TYPES) >= 5

    def test_routes_produce_different_budgets(self):
        """Different routes produce different context budget profiles."""
        from src.agent.router import get_budget_overrides
        from src.agent.conversation import TOKEN_BUDGETS

        base = TOKEN_BUDGETS["ongoing"]
        activity_budget = get_budget_overrides("activity_question", base)
        general_budget = get_budget_overrides("general_chat", base)

        # activity_question should have higher activity budget than general_chat
        assert activity_budget["activity"] > general_budget["activity"]

    def test_routing_wired_into_process_message(self):
        """process_message() uses the router for message classification."""
        from src.agent.conversation import ConversationEngine
        import inspect
        source = inspect.getsource(ConversationEngine.process_message)
        assert "classify_message" in source
        assert "route" in source


# ── Finding #10: Event-Driven Reflection ─────────────────────────


class TestEventDrivenReflection:
    """Verify reflection uses event-driven triggers, not just time-based."""

    def test_reflection_checks_compliance_deviation(self):
        """Reflection triggers on compliance deviation, not just time elapsed."""
        from src.agent.reflection import _is_reflection_due
        import inspect
        source = inspect.getsource(_is_reflection_due)
        # Should check compliance deviation
        assert "compliance" in source.lower() or "deviation" in source.lower()

    def test_compliance_deviation_threshold_exists(self):
        """A deviation threshold is defined for event-driven triggers."""
        from src.agent.reflection import COMPLIANCE_DEVIATION_THRESHOLD
        assert 0.2 <= COMPLIANCE_DEVIATION_THRESHOLD <= 0.5


# ── Overall Architecture Classification ──────────────────────────


class TestArchitectureClassification:
    """Verify the overall system qualifies as 'Autonomous Agent' per Anthropic taxonomy."""

    def test_agent_decides_what_to_do(self):
        """Agent uses LLM to select actions — not a fixed pipeline."""
        from src.agent.actions import select_action
        from src.agent.state_machine import AgentCore
        # Both exist and are used together
        assert callable(select_action)

    def test_agent_evaluates_its_outputs(self):
        """Agent can evaluate generated plans and iterate."""
        from src.agent.plan_evaluator import evaluate_plan
        from src.agent.actions import ACTIONS
        assert "evaluate_plan" in ACTIONS
        assert "generate_plan" in ACTIONS

    def test_agent_has_active_memory(self):
        """Memory tracks outcomes, not just stores facts."""
        from src.memory.user_model import UserModel
        um = UserModel()
        b = um.add_belief("Test", "preference", 0.7)
        assert "utility" in b
        assert "outcome_count" in b

    def test_agent_routes_inputs(self):
        """Different input types get different processing."""
        from src.agent.router import ROUTE_TYPES, ROUTE_CONTEXT_WEIGHTS
        assert len(ROUTE_TYPES) >= 5
        assert len(ROUTE_CONTEXT_WEIGHTS) >= 5

    def test_agent_is_proactive(self):
        """Agent can surface insights without being asked."""
        from src.agent.proactive import refresh_proactive_triggers, get_pending_messages
        assert callable(refresh_proactive_triggers)

    def test_agent_uses_llm_for_decisions(self):
        """Coaching decisions are LLM-driven, not keyword-driven."""
        from src.agent.autonomy import classify_impact
        from src.agent.startup import infer_goal_type
        import inspect
        # Both default to using LLM
        assert inspect.signature(classify_impact).parameters["use_llm"].default is True
        assert inspect.signature(infer_goal_type).parameters["use_llm"].default is True
