"""Tests for Step 6 Phase B: ConversationEngine."""

import json
import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.agent.conversation import (
    ConversationEngine,
    TOKEN_BUDGETS,
    CONVERSATION_SYSTEM_PROMPT,
)
from src.memory.user_model import UserModel


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def tmp_dirs(tmp_path):
    """Create temporary directories for model and sessions."""
    model_dir = tmp_path / "user_model"
    model_dir.mkdir()
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    return model_dir, sessions_dir


@pytest.fixture
def model(tmp_dirs):
    """Create a UserModel with some data."""
    model_dir, _ = tmp_dirs
    m = UserModel(data_dir=model_dir)
    m.update_structured_core("name", "Test Athlete")
    m.update_structured_core("sports", ["running"])
    m.update_structured_core("goal.event", "Half Marathon")
    m.update_structured_core("goal.target_date", "2026-10-15")
    m.update_structured_core("goal.target_time", "1:45:00")
    m.update_structured_core("constraints.training_days_per_week", 5)
    m.update_structured_core("constraints.max_session_minutes", 90)
    m.save()
    return m


@pytest.fixture
def engine(model, tmp_dirs):
    """Create a ConversationEngine with mocked Gemini."""
    model_dir, sessions_dir = tmp_dirs
    data_dir = model_dir.parent  # use tmp_path as data root so plans aren't found
    return ConversationEngine(user_model=model, sessions_dir=sessions_dir, data_dir=data_dir)


def _mock_llm_response(response_json: dict) -> MagicMock:
    """Create a mock LLM response that returns JSON."""
    mock_response = MagicMock()
    mock_response.text = json.dumps(response_json)
    return mock_response


def _mock_embedding_response(values=None) -> MagicMock:
    """Create a mock embedding response."""
    mock_response = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.values = values or [0.1] * 768
    mock_response.embeddings = [mock_embedding]
    return mock_response


# ── Session Lifecycle ─────────────────────────────────────────────


class TestSessionLifecycle:
    def test_start_session_creates_id(self, engine):
        session_id = engine.start_session()
        assert session_id.startswith("session_")
        assert engine.session_id == session_id

    def test_start_session_resets_state(self, engine):
        engine.start_session()
        assert engine.turn_count == 0
        assert engine._rolling_summary == ""
        assert engine.cycle_triggered is False

    def test_end_session_without_start_returns_none(self, engine):
        result = engine.end_session()
        assert result is None

    def test_end_session_increments_session_count(self, engine, tmp_dirs):
        _, sessions_dir = tmp_dirs

        engine.start_session()

        # Simulate a turn by writing directly to JSONL
        turn = {"role": "user", "content": "Hi", "timestamp": "2026-02-10T10:00:00", "artifacts": []}
        with open(engine._session_file, "w") as f:
            f.write(json.dumps(turn) + "\n")
        engine._turn_count = 1

        # Mock the LLM call for session summary
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_llm_response({
            "summary": "Quick test session",
            "key_topics": ["greeting"],
            "decisions_made": [],
            "athlete_mood": "neutral",
            "next_session_context": "Continue onboarding",
        })

        with patch("src.agent.conversation.get_client", return_value=mock_client):
            summary = engine.end_session()

        assert summary is not None
        assert summary["duration_turns"] == 1
        assert engine.user_model.meta["sessions_completed"] == 1

    def test_end_session_saves_summary_file(self, engine, tmp_dirs):
        _, sessions_dir = tmp_dirs

        engine.start_session()
        session_id = engine.session_id

        # Write a turn
        turn = {"role": "user", "content": "Hi", "timestamp": "2026-02-10T10:00:00", "artifacts": []}
        with open(engine._session_file, "w") as f:
            f.write(json.dumps(turn) + "\n")
        engine._turn_count = 1

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_llm_response({
            "summary": "Session summary",
            "key_topics": ["test"],
            "decisions_made": [],
            "athlete_mood": "good",
            "next_session_context": "",
        })

        with patch("src.agent.conversation.get_client", return_value=mock_client):
            engine.end_session()

        summary_files = list(sessions_dir.glob("summary_*.json"))
        assert len(summary_files) == 1

    def test_end_session_promotes_stable_session_beliefs(self, engine):
        engine.start_session()

        # Add a session belief with high confidence (should promote)
        engine.user_model.add_belief("Runs fast", "fitness", confidence=0.8, durability="session")
        # Add a session belief with low confidence (should NOT promote)
        engine.user_model.add_belief("Maybe prefers hills", "preference", confidence=0.5, durability="session")

        # Write a turn so end_session proceeds
        turn = {"role": "user", "content": "test", "timestamp": "2026-02-10T10:00:00", "artifacts": []}
        with open(engine._session_file, "w") as f:
            f.write(json.dumps(turn) + "\n")
        engine._turn_count = 1

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_llm_response({
            "summary": "Test", "key_topics": [], "decisions_made": [],
            "athlete_mood": "ok", "next_session_context": "",
        })

        with patch("src.agent.conversation.get_client", return_value=mock_client):
            engine.end_session()

        beliefs = engine.user_model.beliefs
        fast_belief = next(b for b in beliefs if b["text"] == "Runs fast")
        assert fast_belief["durability"] == "global"  # promoted!


# ── Process Message ──────────────────────────────────────────────


class TestProcessMessage:
    def test_process_message_returns_text(self, engine):
        engine.start_session()

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_llm_response({
            "response_text": "Hello! Tell me about your training.",
            "extracted_beliefs": [],
            "structured_core_updates": {},
            "trigger_cycle": False,
            "cycle_reason": None,
            "onboarding_complete": False,
        })

        with patch("src.agent.conversation.get_client", return_value=mock_client):
            response = engine.process_message("Hi there!")

        assert response == "Hello! Tell me about your training."

    def test_process_message_writes_jsonl(self, engine, tmp_dirs):
        _, sessions_dir = tmp_dirs
        engine.start_session()

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_llm_response({
            "response_text": "Great to hear!",
            "extracted_beliefs": [],
            "structured_core_updates": {},
            "trigger_cycle": False,
            "cycle_reason": None,
            "onboarding_complete": False,
        })

        with patch("src.agent.conversation.get_client", return_value=mock_client):
            engine.process_message("I like running")

        # Should have 2 turns in JSONL (user + agent)
        turns = engine.get_recent_turns(n=10)
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert turns[0]["content"] == "I like running"
        assert turns[1]["role"] == "agent"
        assert turns[1]["content"] == "Great to hear!"

    def test_process_message_extracts_beliefs(self, engine):
        engine.start_session()

        mock_client = MagicMock()
        # First call: main conversation LLM
        main_response = _mock_llm_response({
            "response_text": "Morning runs are great!",
            "extracted_beliefs": [
                {
                    "text": "Prefers morning training",
                    "category": "scheduling",
                    "confidence": 0.8,
                    "reasoning": "User said they run in the morning",
                }
            ],
            "structured_core_updates": {},
            "trigger_cycle": False,
            "cycle_reason": None,
            "onboarding_complete": False,
        })
        # Second call: belief merge LLM
        merge_response = _mock_llm_response({
            "operation": "ADD",
            "target_belief_id": None,
            "new_text": None,
            "new_confidence": 0.8,
            "reasoning": "New information",
        })
        # Third call: embedding (for new belief)
        embed_response = _mock_embedding_response()

        mock_client.models.generate_content.side_effect = [main_response, merge_response]
        mock_client.models.embed_content.return_value = embed_response

        with patch("src.agent.conversation.get_client", return_value=mock_client):
            with patch("src.memory.user_model.get_client", return_value=mock_client):
                engine.process_message("I usually run in the morning before work")

        # Should have the new belief
        beliefs = engine.user_model.get_active_beliefs(category="scheduling")
        assert len(beliefs) >= 1
        assert any("morning" in b["text"].lower() for b in beliefs)

    def test_process_message_updates_structured_core(self, engine):
        engine.start_session()

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_llm_response({
            "response_text": "Great, I'll note that!",
            "extracted_beliefs": [],
            "structured_core_updates": {
                "fitness.weekly_volume_km": 40,
            },
            "trigger_cycle": False,
            "cycle_reason": None,
            "onboarding_complete": False,
        })

        with patch("src.agent.conversation.get_client", return_value=mock_client):
            engine.process_message("I run about 40km per week")

        assert engine.user_model.structured_core["fitness"]["weekly_volume_km"] == 40

    def test_process_message_sets_cycle_trigger(self, engine):
        engine.start_session()

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_llm_response({
            "response_text": "Let me create a plan for you!",
            "extracted_beliefs": [],
            "structured_core_updates": {},
            "trigger_cycle": True,
            "cycle_reason": "Athlete requested a new plan",
            "onboarding_complete": True,
        })

        with patch("src.agent.conversation.get_client", return_value=mock_client):
            engine.process_message("Can you make me a training plan?")

        assert engine.cycle_triggered is True

    def test_process_message_auto_starts_session(self, engine):
        """If no session started, process_message should auto-start one."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_llm_response({
            "response_text": "Hi!",
            "extracted_beliefs": [],
            "structured_core_updates": {},
            "trigger_cycle": False,
            "cycle_reason": None,
            "onboarding_complete": False,
        })

        with patch("src.agent.conversation.get_client", return_value=mock_client):
            engine.process_message("Hello")

        assert engine.session_id is not None
        assert engine.turn_count == 2  # user + agent

    def test_process_message_handles_malformed_llm_response(self, engine):
        engine.start_session()

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "I'm sorry, I can't respond in JSON right now."
        mock_client.models.generate_content.return_value = mock_response

        with patch("src.agent.conversation.get_client", return_value=mock_client):
            response = engine.process_message("Hi")

        # Should return the raw text as fallback
        assert "sorry" in response.lower() or len(response) > 0


# ── Phase Detection ──────────────────────────────────────────────


class TestPhaseDetection:
    def test_onboarding_phase_empty_model(self, tmp_dirs):
        model_dir, sessions_dir = tmp_dirs
        empty_model = UserModel(data_dir=model_dir)
        engine = ConversationEngine(user_model=empty_model, sessions_dir=sessions_dir)

        assert engine._detect_conversation_phase() == "onboarding"

    def test_onboarding_phase_incomplete_core(self, tmp_dirs):
        model_dir, sessions_dir = tmp_dirs
        m = UserModel(data_dir=model_dir)
        m.update_structured_core("sports", ["running"])
        # Missing goal and constraints
        engine = ConversationEngine(user_model=m, sessions_dir=sessions_dir)

        assert engine._detect_conversation_phase() == "onboarding"

    def test_early_phase(self, engine):
        # model fixture has complete core, 0 sessions
        assert engine._detect_conversation_phase() == "early"

    def test_ongoing_phase(self, engine):
        engine.user_model.meta["sessions_completed"] = 5
        assert engine._detect_conversation_phase() == "ongoing"

    def test_planning_phase(self, engine):
        engine._cycle_triggered = True
        assert engine._detect_conversation_phase() == "planning"


# ── Context Builder ──────────────────────────────────────────────


class TestContextBuilder:
    def test_prompt_includes_model_summary(self, engine):
        prompt = engine._build_conversation_prompt("Hi", "early")
        assert "ATHLETE PROFILE" in prompt
        assert "Test Athlete" in prompt

    def test_prompt_includes_current_message(self, engine):
        prompt = engine._build_conversation_prompt("How's my progress?", "ongoing")
        assert "How's my progress?" in prompt

    def test_onboarding_skips_cross_session(self, engine):
        prompt = engine._build_conversation_prompt("Hi", "onboarding")
        assert "PREVIOUS SESSION" not in prompt

    def test_prompt_includes_recent_turns(self, engine, tmp_dirs):
        _, sessions_dir = tmp_dirs
        engine.start_session()

        # Write some turns
        for i in range(3):
            engine._append_to_session("user", f"Message {i}")
            engine._append_to_session("agent", f"Reply {i}")

        prompt = engine._build_conversation_prompt("Latest message", "early")
        assert "Message 0" in prompt
        assert "Reply 0" in prompt


# ── Belief Compare/Merge ─────────────────────────────────────────


class TestBeliefMerge:
    def test_add_new_belief(self, engine):
        engine.start_session()

        candidate = {
            "text": "Runs best in cool weather",
            "category": "preference",
            "confidence": 0.7,
        }

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_llm_response({
            "operation": "ADD",
            "target_belief_id": None,
            "new_text": None,
            "new_confidence": 0.7,
            "reasoning": "New information",
        })
        mock_client.models.embed_content.return_value = _mock_embedding_response()

        with patch("src.agent.conversation.get_client", return_value=mock_client):
            with patch("src.memory.user_model.get_client", return_value=mock_client):
                engine._process_belief_candidates([candidate])

        beliefs = engine.user_model.get_active_beliefs(category="preference")
        assert any("cool weather" in b["text"] for b in beliefs)

    def test_update_existing_belief(self, engine):
        engine.start_session()
        old_belief = engine.user_model.add_belief(
            "Runs 3 days per week", "scheduling", confidence=0.7
        )

        candidate = {
            "text": "Runs 5 days per week",
            "category": "scheduling",
            "confidence": 0.85,
        }

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_llm_response({
            "operation": "UPDATE",
            "target_belief_id": old_belief["id"],
            "new_text": "Runs 5 days per week (increased from 3)",
            "new_confidence": 0.85,
            "reasoning": "Updated frequency",
        })
        mock_client.models.embed_content.return_value = _mock_embedding_response()

        with patch("src.agent.conversation.get_client", return_value=mock_client):
            with patch("src.memory.user_model.get_client", return_value=mock_client):
                engine._process_belief_candidates([candidate])

        updated = next(b for b in engine.user_model.beliefs if b["id"] == old_belief["id"])
        assert "5 days" in updated["text"]
        assert updated["confidence"] == 0.85

    def test_delete_superseded_belief(self, engine):
        engine.start_session()
        old_belief = engine.user_model.add_belief(
            "Does not like morning runs", "scheduling", confidence=0.6
        )

        candidate = {
            "text": "Actually prefers morning runs now",
            "category": "scheduling",
            "confidence": 0.8,
        }

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_llm_response({
            "operation": "DELETE",
            "target_belief_id": old_belief["id"],
            "new_text": None,
            "new_confidence": 0.8,
            "reasoning": "Contradicts old belief",
        })
        mock_client.models.embed_content.return_value = _mock_embedding_response()

        with patch("src.agent.conversation.get_client", return_value=mock_client):
            with patch("src.memory.user_model.get_client", return_value=mock_client):
                engine._process_belief_candidates([candidate])

        # Old belief should be inactive
        assert old_belief["active"] is False
        assert old_belief["superseded_by"] is not None

        # New belief should exist
        active = engine.user_model.get_active_beliefs(category="scheduling")
        assert any("morning runs" in b["text"] for b in active)

    def test_noop_skips_duplicate(self, engine):
        engine.start_session()
        engine.user_model.add_belief("Likes running", "preference", confidence=0.8)

        candidate = {
            "text": "Enjoys running",
            "category": "preference",
            "confidence": 0.7,
        }

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_llm_response({
            "operation": "NOOP",
            "target_belief_id": None,
            "reasoning": "Already captured",
        })

        with patch("src.agent.conversation.get_client", return_value=mock_client):
            initial_count = len(engine.user_model.beliefs)
            engine._process_belief_candidates([candidate])

        assert len(engine.user_model.beliefs) == initial_count


# ── Session Storage ──────────────────────────────────────────────


class TestSessionStorage:
    def test_jsonl_format(self, engine, tmp_dirs):
        _, sessions_dir = tmp_dirs
        engine.start_session()

        engine._append_to_session("user", "Hello coach!")
        engine._append_to_session("agent", "Hello athlete!", {"artifacts": ["plans/plan.json"]})

        turns = engine.get_recent_turns(n=10)
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert turns[0]["content"] == "Hello coach!"
        assert turns[1]["artifacts"] == ["plans/plan.json"]

    def test_session_file_naming(self, engine):
        session_id = engine.start_session()
        assert engine._session_file.name.startswith("session_")
        assert engine._session_file.suffix == ".jsonl"

    def test_get_recent_turns_limits(self, engine, tmp_dirs):
        engine.start_session()

        for i in range(20):
            engine._append_to_session("user", f"Message {i}")

        recent = engine.get_recent_turns(n=5)
        assert len(recent) == 5
        assert recent[0]["content"] == "Message 15"


# ── Rolling Summary ──────────────────────────────────────────────


class TestRollingSummary:
    def test_update_rolling_summary(self, engine):
        engine.start_session()

        # Write enough turns to trigger summary
        for i in range(15):
            engine._append_to_session("user", f"Training update {i}")
            engine._append_to_session("agent", f"Noted, update {i}")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "The athlete provided 15 training updates covering various sessions."
        mock_client.models.generate_content.return_value = mock_response

        with patch("src.agent.conversation.get_client", return_value=mock_client):
            engine._update_rolling_summary()

        assert len(engine._rolling_summary) > 0
        assert "training updates" in engine._rolling_summary.lower()

    def test_rolling_summary_only_for_older_turns(self, engine):
        engine.start_session()

        # Only 5 turns -- fewer than RECENT_TURNS_COUNT
        for i in range(5):
            engine._append_to_session("user", f"Message {i}")

        engine._update_rolling_summary()
        # Should not produce a summary (all turns are "recent")
        assert engine._rolling_summary == ""


# ── Session Search (BM25) ───────────────────────────────────────


class TestSessionSearch:
    def test_search_empty_returns_empty(self, engine):
        results = engine.search_sessions("training plan")
        assert results == []

    def test_search_finds_summary(self, engine, tmp_dirs):
        _, sessions_dir = tmp_dirs

        # Create a session summary
        summary = {
            "session_id": "session_2026-02-08_100000",
            "date": "2026-02-08",
            "summary": "Discussed marathon training and tapering strategy for the August race.",
            "key_topics": ["marathon", "tapering"],
            "decisions_made": [],
            "athlete_mood": "motivated",
            "next_session_context": "Review tapering timeline",
        }
        summary_path = sessions_dir / "summary_2026-02-08_100000.json"
        summary_path.write_text(json.dumps(summary))

        results = engine.search_sessions("tapering")
        assert len(results) >= 1
        assert any("tapering" in r["text"].lower() for r in results)

    def test_search_finds_transcript(self, engine, tmp_dirs):
        _, sessions_dir = tmp_dirs

        # Create a session JSONL
        session_file = sessions_dir / "session_2026-02-07_090000.jsonl"
        turns = [
            {"role": "user", "content": "My knee hurts after long runs", "timestamp": "2026-02-07T09:00:00", "artifacts": []},
            {"role": "agent", "content": "Let's adjust your plan", "timestamp": "2026-02-07T09:01:00", "artifacts": []},
        ]
        with open(session_file, "w") as f:
            for turn in turns:
                f.write(json.dumps(turn) + "\n")

        results = engine.search_sessions("knee hurts")
        assert len(results) >= 1


# ── Token Budget Validation ──────────────────────────────────────


class TestTokenBudgets:
    def test_all_phases_have_budgets(self):
        for phase in ["onboarding", "early", "ongoing", "planning"]:
            assert phase in TOKEN_BUDGETS
            budget = TOKEN_BUDGETS[phase]
            assert "system" in budget
            assert "model" in budget
            assert "cross" in budget
            assert "rolling" in budget
            assert "recent" in budget

    def test_onboarding_has_no_cross_session(self):
        assert TOKEN_BUDGETS["onboarding"]["cross"] == 0

    def test_ongoing_has_largest_recent(self):
        assert TOKEN_BUDGETS["ongoing"]["recent"] > TOKEN_BUDGETS["onboarding"]["recent"]


# ── Integration Test (requires Gemini API) ───────────────────────


@pytest.mark.integration
class TestConversationIntegration:
    def test_five_turn_conversation(self, tmp_dirs):
        """Full integration test: 5 turns, verify beliefs and session storage."""
        model_dir, sessions_dir = tmp_dirs
        user_model = UserModel(data_dir=model_dir)
        engine = ConversationEngine(user_model=user_model, sessions_dir=sessions_dir)

        engine.start_session()

        # Turn 1: Introduction
        r1 = engine.process_message(
            "Hi! I'm training for a half marathon in October. I run about 30km per week."
        )
        assert len(r1) > 0

        # Turn 2: More details
        r2 = engine.process_message(
            "I usually run 4 times a week, mostly in the morning before work. "
            "My target time is 1:50."
        )
        assert len(r2) > 0

        # Turn 3: Constraint
        r3 = engine.process_message(
            "I had a knee injury last year that sometimes bothers me on runs over 15km."
        )
        assert len(r3) > 0

        # Turn 4: Preference
        r4 = engine.process_message(
            "I really enjoy interval training but I know I should do more easy runs."
        )
        assert len(r4) > 0

        # Turn 5: Question
        r5 = engine.process_message(
            "What do you think my training should look like for the next few weeks?"
        )
        assert len(r5) > 0

        # Verify beliefs were extracted
        active_beliefs = user_model.get_active_beliefs()
        assert len(active_beliefs) >= 1, f"Expected beliefs, got {len(active_beliefs)}"

        # Verify session JSONL exists and has turns
        jsonl_files = list(sessions_dir.glob("session_*.jsonl"))
        assert len(jsonl_files) == 1

        turns = engine.get_recent_turns(n=100)
        assert len(turns) == 10  # 5 user + 5 agent

        # Verify structured core got some updates
        core = user_model.structured_core
        # At minimum the LLM should have picked up running and half marathon
        has_updates = (
            core.get("sports")
            or core.get("goal", {}).get("event")
            or core.get("constraints", {}).get("training_days_per_week")
        )
        assert has_updates, f"Expected structured_core updates, got {core}"

        # End session and verify summary
        summary = engine.end_session()
        assert summary is not None
        assert summary["duration_turns"] == 10
        assert len(summary.get("summary", "")) > 0

        # Verify summary file was written
        summary_files = list(sessions_dir.glob("summary_*.json"))
        assert len(summary_files) == 1
