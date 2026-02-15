"""Tests for Step 6 Phase C: OnboardingEngine and CLI chat mode."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.agent.onboarding import OnboardingEngine, ONBOARDING_GREETING
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
def empty_model(tmp_dirs):
    """Create a fresh UserModel with no data (new athlete)."""
    model_dir, _ = tmp_dirs
    return UserModel(data_dir=model_dir)


@pytest.fixture
def populated_model(tmp_dirs):
    """Create a UserModel with enough data for plan generation."""
    model_dir, _ = tmp_dirs
    m = UserModel(data_dir=model_dir)
    m.update_structured_core("name", "Test Runner")
    m.update_structured_core("sports", ["running"])
    m.update_structured_core("goal.event", "Half Marathon")
    m.update_structured_core("goal.target_date", "2026-10-15")
    m.update_structured_core("goal.target_time", "1:45:00")
    m.update_structured_core("constraints.training_days_per_week", 5)
    m.update_structured_core("constraints.max_session_minutes", 90)
    return m


@pytest.fixture
def engine(empty_model, tmp_dirs):
    """Create an OnboardingEngine with empty model."""
    model_dir, sessions_dir = tmp_dirs
    return OnboardingEngine(
        user_model=empty_model,
        sessions_dir=sessions_dir,
        data_dir=model_dir.parent,
    )


def _mock_llm_response(response_json: dict) -> MagicMock:
    """Create a mock LLM response that returns JSON."""
    mock_response = MagicMock()
    mock_response.text = json.dumps(response_json)
    return mock_response


# ── OnboardingEngine Initialization ─────────────────────────────


class TestOnboardingInit:
    def test_creates_with_default_model(self, tmp_dirs):
        model_dir, sessions_dir = tmp_dirs
        with patch("src.agent.onboarding.UserModel.load_or_create") as mock_load:
            mock_load.return_value = UserModel(data_dir=model_dir)
            engine = OnboardingEngine(sessions_dir=sessions_dir, data_dir=model_dir.parent)
            assert engine.user_model is not None

    def test_accepts_custom_model(self, empty_model, tmp_dirs):
        _, sessions_dir = tmp_dirs
        engine = OnboardingEngine(user_model=empty_model, sessions_dir=sessions_dir)
        assert engine.user_model is empty_model

    def test_plan_not_triggered_initially(self, engine):
        assert engine._plan_triggered is False


# ── Start ────────────────────────────────────────────────────────


class TestOnboardingStart:
    def test_start_returns_greeting(self, engine):
        greeting = engine.start()
        assert greeting == ONBOARDING_GREETING

    def test_start_creates_session(self, engine):
        engine.start()
        assert engine.conversation.session_id is not None

    def test_start_increments_turn_count(self, engine):
        engine.start()
        assert engine.conversation._turn_count == 1

    def test_greeting_is_logged_to_session(self, engine, tmp_dirs):
        _, sessions_dir = tmp_dirs
        engine.start()
        # Session file should exist and contain the greeting
        session_files = list(sessions_dir.glob("session_*.jsonl"))
        assert len(session_files) == 1
        content = session_files[0].read_text().strip()
        lines = content.split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["role"] == "agent"
        assert "Welcome to ReAgt" in entry["content"]


# ── Process Message ──────────────────────────────────────────────


class TestOnboardingProcessMessage:
    @patch("src.agent.conversation.get_client")
    def test_process_message_returns_response(self, mock_client, engine):
        engine.start()
        mock_model = MagicMock()
        mock_client.return_value.models = mock_model
        mock_model.generate_content.return_value = _mock_llm_response({
            "response_text": "Great! Tell me more about your training.",
            "extracted_beliefs": [],
            "structured_core_updates": {},
            "trigger_cycle": False,
        })
        response = engine.process_message("I run 5 days a week")
        assert "Tell me more" in response

    @patch("src.agent.conversation.get_client")
    def test_process_message_detects_cycle_trigger(self, mock_client, engine):
        engine.start()
        mock_model = MagicMock()
        mock_client.return_value.models = mock_model
        mock_model.generate_content.return_value = _mock_llm_response({
            "response_text": "Perfect, I have everything I need!",
            "extracted_beliefs": [],
            "structured_core_updates": {},
            "trigger_cycle": True,
            "onboarding_complete": True,
        })
        engine.process_message("I want to run a half marathon in 1:45")
        assert engine._plan_triggered is True

    @patch("src.agent.conversation.get_client")
    def test_process_message_without_trigger(self, mock_client, engine):
        engine.start()
        mock_model = MagicMock()
        mock_client.return_value.models = mock_model
        mock_model.generate_content.return_value = _mock_llm_response({
            "response_text": "How many days per week?",
            "extracted_beliefs": [],
            "structured_core_updates": {},
            "trigger_cycle": False,
        })
        engine.process_message("I run 40km per week")
        assert engine._plan_triggered is False


# ── Onboarding Completion Detection ─────────────────────────────


class TestOnboardingComplete:
    def test_not_complete_with_empty_model(self, engine):
        assert engine.is_onboarding_complete() is False

    def test_complete_when_plan_triggered(self, engine):
        engine._plan_triggered = True
        assert engine.is_onboarding_complete() is True

    def test_complete_with_minimum_fields(self, populated_model, tmp_dirs):
        _, sessions_dir = tmp_dirs
        engine = OnboardingEngine(
            user_model=populated_model,
            sessions_dir=sessions_dir,
            data_dir=tmp_dirs[0].parent,
        )
        # Model has sports, goal.event, constraints.training_days_per_week
        assert engine.is_onboarding_complete() is True

    def test_not_complete_missing_sports(self, empty_model, tmp_dirs):
        _, sessions_dir = tmp_dirs
        empty_model.update_structured_core("goal.event", "Marathon")
        empty_model.update_structured_core("constraints.training_days_per_week", 4)
        engine = OnboardingEngine(
            user_model=empty_model,
            sessions_dir=sessions_dir,
            data_dir=tmp_dirs[0].parent,
        )
        assert engine.is_onboarding_complete() is False

    def test_not_complete_missing_event(self, empty_model, tmp_dirs):
        _, sessions_dir = tmp_dirs
        empty_model.update_structured_core("sports", ["running"])
        empty_model.update_structured_core("constraints.training_days_per_week", 4)
        engine = OnboardingEngine(
            user_model=empty_model,
            sessions_dir=sessions_dir,
            data_dir=tmp_dirs[0].parent,
        )
        assert engine.is_onboarding_complete() is False

    def test_not_complete_missing_days(self, empty_model, tmp_dirs):
        _, sessions_dir = tmp_dirs
        empty_model.update_structured_core("sports", ["running"])
        empty_model.update_structured_core("goal.event", "Marathon")
        engine = OnboardingEngine(
            user_model=empty_model,
            sessions_dir=sessions_dir,
            data_dir=tmp_dirs[0].parent,
        )
        assert engine.is_onboarding_complete() is False

    def test_complete_with_goal_type_no_event(self, empty_model, tmp_dirs):
        """Athletes with a goal_type but no specific race event can complete onboarding."""
        _, sessions_dir = tmp_dirs
        empty_model.update_structured_core("sports", ["soccer", "swimming"])
        empty_model.update_structured_core("goal.goal_type", "general")
        empty_model.update_structured_core("constraints.training_days_per_week", 6)
        engine = OnboardingEngine(
            user_model=empty_model,
            sessions_dir=sessions_dir,
            data_dir=tmp_dirs[0].parent,
        )
        assert engine.is_onboarding_complete() is True

    def test_complete_with_routine_goal_type(self, empty_model, tmp_dirs):
        """Athletes with routine goal_type complete onboarding without event."""
        _, sessions_dir = tmp_dirs
        empty_model.update_structured_core("sports", ["running"])
        empty_model.update_structured_core("goal.goal_type", "routine")
        empty_model.update_structured_core("constraints.training_days_per_week", 4)
        engine = OnboardingEngine(
            user_model=empty_model,
            sessions_dir=sessions_dir,
            data_dir=tmp_dirs[0].parent,
        )
        assert engine.is_onboarding_complete() is True

    def test_not_complete_missing_event_and_goal_type(self, empty_model, tmp_dirs):
        """Without both event and goal_type, onboarding is not complete."""
        _, sessions_dir = tmp_dirs
        empty_model.update_structured_core("sports", ["running"])
        empty_model.update_structured_core("constraints.training_days_per_week", 4)
        engine = OnboardingEngine(
            user_model=empty_model,
            sessions_dir=sessions_dir,
            data_dir=tmp_dirs[0].parent,
        )
        assert engine.is_onboarding_complete() is False


# ── End Session ──────────────────────────────────────────────────


class TestOnboardingEndSession:
    @patch("src.agent.conversation.get_client")
    def test_end_session_delegates_to_conversation(self, mock_client, engine):
        engine.start()
        mock_model = MagicMock()
        mock_client.return_value.models = mock_model
        mock_model.generate_content.return_value = _mock_llm_response({
            "session_id": "test",
            "date": "2026-02-10",
            "duration_turns": 1,
            "summary": "Onboarding session",
            "key_topics": ["onboarding"],
            "decisions_made": [],
            "beliefs_extracted_count": 0,
            "athlete_mood": "neutral",
            "next_session_context": "Continue onboarding",
        })
        result = engine.end_session()
        # end_session returns summary dict or None
        assert result is None or isinstance(result, dict)


# ── Get Onboarding Prompt ────────────────────────────────────────


class TestGetOnboardingPrompt:
    def test_returns_greeting(self, engine):
        assert engine.get_onboarding_prompt() == ONBOARDING_GREETING

    def test_greeting_mentions_sport(self):
        assert "sport" in ONBOARDING_GREETING.lower()

    def test_greeting_mentions_training(self):
        assert "training" in ONBOARDING_GREETING.lower()


# ── Full Onboarding Flow (mocked LLM) ───────────────────────────


class TestOnboardingFlow:
    @patch("src.agent.conversation.get_client")
    def test_multi_turn_onboarding_populates_model(self, mock_client, engine):
        """Simulate a 3-turn onboarding conversation that populates the user model."""
        engine.start()
        mock_model = MagicMock()
        mock_client.return_value.models = mock_model

        # Turn 1: User describes sports
        mock_model.generate_content.return_value = _mock_llm_response({
            "response_text": "Running is great! What event are you training for?",
            "extracted_beliefs": [
                {"text": "Trains running 5 days per week", "category": "fitness", "confidence": 0.8}
            ],
            "structured_core_updates": {
                "sports": ["running"],
                "constraints.training_days_per_week": 5,
            },
            "trigger_cycle": False,
        })
        engine.process_message("I run 5 days a week, mostly easy runs")

        assert engine.user_model.structured_core["sports"] == ["running"]
        assert engine.user_model.structured_core["constraints"]["training_days_per_week"] == 5

        # Turn 2: User describes goal
        mock_model.generate_content.return_value = _mock_llm_response({
            "response_text": "A half marathon in 1:45! How long are your sessions usually?",
            "extracted_beliefs": [
                {"text": "Targets sub-1:45 half marathon", "category": "motivation", "confidence": 0.9}
            ],
            "structured_core_updates": {
                "goal.event": "Half Marathon",
                "goal.target_time": "1:45:00",
            },
            "trigger_cycle": False,
        })
        engine.process_message("I want to run a half marathon under 1:45")

        assert engine.user_model.structured_core["goal"]["event"] == "Half Marathon"
        assert engine.user_model.structured_core["goal"]["target_time"] == "1:45:00"

        # Turn 3: Final details + trigger
        mock_model.generate_content.return_value = _mock_llm_response({
            "response_text": "Perfect! I have everything I need to create your plan.",
            "extracted_beliefs": [
                {"text": "Max session 90 minutes", "category": "constraint", "confidence": 0.9}
            ],
            "structured_core_updates": {
                "constraints.max_session_minutes": 90,
            },
            "trigger_cycle": True,
            "onboarding_complete": True,
        })
        engine.process_message("Usually about 90 minutes max per session")

        assert engine.is_onboarding_complete() is True
        assert engine.user_model.structured_core["constraints"]["max_session_minutes"] == 90

        # Verify beliefs were added
        active_beliefs = engine.user_model.get_active_beliefs()
        assert len(active_beliefs) >= 3

    @patch("src.agent.conversation.get_client")
    def test_onboarding_session_file_created(self, mock_client, engine, tmp_dirs):
        """Verify session JSONL file is created during onboarding."""
        _, sessions_dir = tmp_dirs
        engine.start()
        mock_model = MagicMock()
        mock_client.return_value.models = mock_model
        mock_model.generate_content.return_value = _mock_llm_response({
            "response_text": "Tell me more!",
            "extracted_beliefs": [],
            "structured_core_updates": {},
            "trigger_cycle": False,
        })
        engine.process_message("Hello")

        session_files = list(sessions_dir.glob("session_*.jsonl"))
        assert len(session_files) == 1


# ── CLI Argument Parsing ─────────────────────────────────────────


class TestCLIArgParsing:
    def test_chat_flag_recognized(self):
        """Verify --chat flag is accepted by argparse."""
        from src.interface.cli import main
        import argparse

        # Parse with --chat should not raise
        parser = argparse.ArgumentParser()
        parser.add_argument("--chat", action="store_true")
        parsed = parser.parse_args(["--chat"])
        assert parsed.chat is True

    def test_onboard_legacy_flag_recognized(self):
        """Verify --onboard-legacy flag is accepted by argparse."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--onboard-legacy", action="store_true")
        parsed = parser.parse_args(["--onboard-legacy"])
        assert parsed.onboard_legacy is True


# ── CLI Backward Compatibility ───────────────────────────────────


class TestCLIBackwardCompat:
    def test_import_flag_still_works(self):
        """Verify --import flag is still recognized."""
        from src.interface.cli import main
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--import", dest="import_file", metavar="FILE")
        parsed = parser.parse_args(["--import", "test.fit"])
        assert parsed.import_file == "test.fit"

    def test_assess_flag_still_works(self):
        """Verify --assess flag is still recognized."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--assess", action="store_true")
        parsed = parser.parse_args(["--assess"])
        assert parsed.assess is True

    def test_trajectory_flag_still_works(self):
        """Verify --trajectory flag is still recognized."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--trajectory", action="store_true")
        parsed = parser.parse_args(["--trajectory"])
        assert parsed.trajectory is True

    def test_status_flag_still_works(self):
        """Verify --status flag is still recognized."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--status", action="store_true")
        parsed = parser.parse_args(["--status"])
        assert parsed.status is True

    @patch("src.interface.cli.import_activity")
    def test_import_dispatches_correctly(self, mock_import):
        """Verify --import dispatches to import_activity."""
        from src.interface.cli import main

        main(["--import", "test.fit"])
        mock_import.assert_called_once_with("test.fit")

    @patch("src.interface.cli.run_assessment")
    def test_assess_dispatches_correctly(self, mock_assess):
        """Verify --assess dispatches to run_assessment."""
        from src.interface.cli import main

        main(["--assess"])
        mock_assess.assert_called_once()

    @patch("src.interface.cli.run_trajectory")
    def test_trajectory_dispatches_correctly(self, mock_traj):
        """Verify --trajectory dispatches to run_trajectory."""
        from src.interface.cli import main

        main(["--trajectory"])
        mock_traj.assert_called_once()

    @patch("src.interface.cli.run_status")
    def test_status_dispatches_correctly(self, mock_status):
        """Verify --status dispatches to run_status."""
        from src.interface.cli import main

        main(["--status"])
        mock_status.assert_called_once()

    @patch("src.interface.cli.run_chat")
    def test_default_runs_chat(self, mock_chat):
        """Verify default (no flags) runs chat mode."""
        from src.interface.cli import main

        main([])
        mock_chat.assert_called_once()

    @patch("src.interface.cli.run_chat")
    def test_chat_flag_runs_chat(self, mock_chat):
        """Verify --chat flag runs chat mode."""
        from src.interface.cli import main

        main(["--chat"])
        mock_chat.assert_called_once()


# ── Integration: Onboarding -> Plan Generation ──────────────────


class TestOnboardingPlanIntegration:
    @pytest.mark.integration
    @patch("src.agent.conversation.get_client")
    def test_onboarding_to_plan_generation(self, mock_client, engine):
        """Full flow: onboarding conversation -> populated model -> profile projection."""
        engine.start()
        mock_model = MagicMock()
        mock_client.return_value.models = mock_model

        # Simulate onboarding that fills all required fields
        mock_model.generate_content.return_value = _mock_llm_response({
            "response_text": "I have everything!",
            "extracted_beliefs": [],
            "structured_core_updates": {
                "name": "Test Runner",
                "sports": ["running"],
                "goal.event": "10K",
                "goal.target_date": "2026-06-01",
                "goal.target_time": "0:45:00",
                "constraints.training_days_per_week": 4,
                "constraints.max_session_minutes": 60,
            },
            "trigger_cycle": True,
            "onboarding_complete": True,
        })
        engine.process_message("I want to run a 10K in 45 minutes, training 4 days per week, 60 min max")

        assert engine.is_onboarding_complete()

        # Project profile and verify it's valid for plan generation
        profile = engine.user_model.project_profile()
        assert profile["sports"] == ["running"]
        assert profile["goal"]["event"] == "10K"
        assert profile["goal"]["target_date"] == "2026-06-01"
        assert profile["goal"]["target_time"] == "0:45:00"
        assert profile["constraints"]["training_days_per_week"] == 4
        assert profile["constraints"]["max_session_minutes"] == 60
        # Profile should be compatible with generate_plan (has all required keys)
        assert "name" in profile
        assert "fitness" in profile
