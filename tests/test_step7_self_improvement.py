"""Tests for Self-Improvement Mechanisms.

Tests meta-belief extraction from reflection cycles and their injection
into future coaching behavior via the PrefEval pattern.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.memory.episodes import extract_meta_beliefs, META_REFLECTION_PROMPT
from src.memory.user_model import UserModel


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def sample_episode():
    return {
        "id": "ep_2026-02-03",
        "block": "2026-W06",
        "period": "2026-02-03",
        "prescribed_sessions": 5,
        "actual_sessions": 3,
        "compliance_rate": 0.6,
        "key_observations": [
            "Easy runs consistently executed at Zone 3 instead of Zone 2",
            "Interval sessions were well-paced with good recovery",
        ],
        "lessons": [
            "Emphasize easy pace targets more strongly in the plan",
            "Thursday sessions missed 3 weeks in a row — schedule conflict",
        ],
        "patterns_detected": [
            "Athlete runs easy sessions 15-20 seconds/km too fast",
            "Weekly volume drops when Thursday is missed",
        ],
        "fitness_delta": {
            "estimated_vo2max_change": "stable",
            "weekly_volume_trend": "decreasing",
        },
        "confidence": 0.8,
    }


def _mock_llm_response(response_json: dict) -> MagicMock:
    mock = MagicMock()
    mock.text = json.dumps(response_json)
    return mock


# ── extract_meta_beliefs ─────────────────────────────────────────


class TestExtractMetaBeliefs:
    @patch("src.memory.episodes.get_client")
    def test_extracts_meta_beliefs(self, mock_client, sample_episode):
        mock_model = MagicMock()
        mock_client.return_value.models = mock_model
        mock_model.generate_content.return_value = _mock_llm_response({
            "meta_beliefs": [
                {
                    "text": "Athlete runs easy sessions too fast — emphasize zone discipline",
                    "category": "meta",
                    "confidence": 0.85,
                    "reasoning": "Pattern persists across 3 weeks",
                },
                {
                    "text": "Thursday sessions should be optional or light to match schedule",
                    "category": "meta",
                    "confidence": 0.75,
                    "reasoning": "Recurring Thursday misses",
                },
            ]
        })

        beliefs = extract_meta_beliefs(sample_episode)
        assert len(beliefs) == 2
        assert beliefs[0]["category"] == "meta"
        assert "zone discipline" in beliefs[0]["text"]

    @patch("src.memory.episodes.get_client")
    def test_returns_empty_for_no_insights(self, mock_client, sample_episode):
        mock_model = MagicMock()
        mock_client.return_value.models = mock_model
        mock_model.generate_content.return_value = _mock_llm_response({
            "meta_beliefs": []
        })

        beliefs = extract_meta_beliefs(sample_episode)
        assert beliefs == []

    @patch("src.memory.episodes.get_client")
    def test_handles_malformed_response(self, mock_client, sample_episode):
        mock_model = MagicMock()
        mock_client.return_value.models = mock_model
        mock_model.generate_content.return_value = MagicMock(text="not json at all")

        beliefs = extract_meta_beliefs(sample_episode)
        assert beliefs == []

    @patch("src.memory.episodes.get_client")
    def test_prompt_includes_episode_data(self, mock_client, sample_episode):
        mock_model = MagicMock()
        mock_client.return_value.models = mock_model
        mock_model.generate_content.return_value = _mock_llm_response({"meta_beliefs": []})

        extract_meta_beliefs(sample_episode)

        # Verify the prompt was built with episode data
        call_args = mock_model.generate_content.call_args
        prompt_content = call_args.kwargs.get("contents") or call_args[1].get("contents")
        prompt_text = prompt_content[0].parts[0].text
        assert "2026-W06" in prompt_text
        assert "Zone 3 instead of Zone 2" in prompt_text

    @patch("src.memory.episodes.get_client")
    def test_uses_meta_reflection_system_prompt(self, mock_client, sample_episode):
        mock_model = MagicMock()
        mock_client.return_value.models = mock_model
        mock_model.generate_content.return_value = _mock_llm_response({"meta_beliefs": []})

        extract_meta_beliefs(sample_episode)

        call_args = mock_model.generate_content.call_args
        config = call_args.kwargs.get("config") or call_args[1].get("config")
        assert "coaching effectiveness" in config.system_instruction.lower()


# ── Meta-Belief Storage in UserModel ─────────────────────────────


class TestMetaBeliefStorage:
    def test_meta_beliefs_stored_like_regular_beliefs(self, tmp_path):
        model_dir = tmp_path / "user_model"
        model_dir.mkdir()
        model = UserModel(data_dir=model_dir)

        belief = model.add_belief(
            text="Athlete responds better to specific pace targets than HR zones",
            category="meta",
            confidence=0.8,
            source="reflection",
        )
        assert belief["category"] == "meta"
        assert belief["active"] is True

    def test_meta_beliefs_appear_in_active_beliefs(self, tmp_path):
        model_dir = tmp_path / "user_model"
        model_dir.mkdir()
        model = UserModel(data_dir=model_dir)

        model.add_belief("Meta insight 1", "meta", confidence=0.8)
        model.add_belief("Regular belief", "fitness", confidence=0.9)

        active = model.get_active_beliefs()
        categories = [b["category"] for b in active]
        assert "meta" in categories

    def test_meta_beliefs_in_model_summary(self, tmp_path):
        model_dir = tmp_path / "user_model"
        model_dir.mkdir()
        model = UserModel(data_dir=model_dir)
        model.update_structured_core("sports", ["running"])

        model.add_belief(
            "Plan compliance improves with explicit rest day messaging",
            "meta",
            confidence=0.75,
        )

        summary = model.get_model_summary()
        assert "[META]" in summary
        assert "rest day messaging" in summary

    def test_meta_beliefs_filtered_by_confidence(self, tmp_path):
        model_dir = tmp_path / "user_model"
        model_dir.mkdir()
        model = UserModel(data_dir=model_dir)

        model.add_belief("High confidence meta", "meta", confidence=0.9)
        model.add_belief("Low confidence meta", "meta", confidence=0.3)

        high_conf = model.get_active_beliefs(category="meta", min_confidence=0.6)
        assert len(high_conf) == 1
        assert high_conf[0]["text"] == "High confidence meta"


# ── Meta-Beliefs in Plan Prompt (PrefEval) ───────────────────────


class TestMetaBeliefInjection:
    def test_meta_beliefs_injected_in_plan_prompt(self):
        from src.agent.prompts import build_plan_prompt

        beliefs = [
            {"text": "Athlete runs easy sessions too fast", "category": "meta", "confidence": 0.85},
            {"text": "Prefers morning training", "category": "scheduling", "confidence": 0.8},
        ]

        prompt = build_plan_prompt(
            {"sports": ["running"], "goal": {}, "constraints": {}, "fitness": {}},
            beliefs=beliefs,
        )
        assert "COACH'S NOTES" in prompt
        assert "[META]" in prompt
        assert "easy sessions too fast" in prompt

    def test_meta_beliefs_injected_in_assessment(self):
        from src.agent.assessment import _build_assessment_prompt

        beliefs = [
            {"text": "Thursday compliance is historically low", "category": "meta", "confidence": 0.7},
        ]

        prompt = _build_assessment_prompt(
            {"goal": {}, "fitness": {}},
            {"sessions": []},
            [],
            beliefs=beliefs,
        )
        assert "Thursday compliance" in prompt


# ── Meta-Belief Lifecycle ────────────────────────────────────────


class TestMetaBeliefLifecycle:
    @patch("src.memory.episodes.get_client")
    def test_full_lifecycle_extract_and_store(self, mock_client, sample_episode, tmp_path):
        """Full flow: reflection -> extract meta-beliefs -> store in user model."""
        mock_model = MagicMock()
        mock_client.return_value.models = mock_model
        mock_model.generate_content.return_value = _mock_llm_response({
            "meta_beliefs": [
                {
                    "text": "Zone 2 discipline needs reinforcement in plans",
                    "category": "meta",
                    "confidence": 0.8,
                    "reasoning": "Persistent easy-session pacing issue",
                }
            ]
        })

        # Extract meta-beliefs from reflection
        meta_beliefs = extract_meta_beliefs(sample_episode)
        assert len(meta_beliefs) == 1

        # Store in user model
        model_dir = tmp_path / "user_model"
        model_dir.mkdir()
        model = UserModel(data_dir=model_dir)

        for mb in meta_beliefs:
            model.add_belief(
                text=mb["text"],
                category="meta",
                confidence=mb["confidence"],
                source="reflection",
                source_ref=sample_episode.get("id"),
            )

        # Verify meta-belief is in model
        active_meta = model.get_active_beliefs(category="meta")
        assert len(active_meta) == 1
        assert active_meta[0]["source"] == "reflection"
        assert active_meta[0]["source_ref"] == "ep_2026-02-03"

        # Verify it appears in model summary (PrefEval injection)
        model.update_structured_core("sports", ["running"])
        summary = model.get_model_summary()
        assert "Zone 2 discipline" in summary


# ── META_REFLECTION_PROMPT ───────────────────────────────────────


class TestMetaReflectionPrompt:
    def test_prompt_instructs_meta_analysis(self):
        assert "coaching effectiveness" in META_REFLECTION_PROMPT.lower()

    def test_prompt_requires_json(self):
        assert "JSON" in META_REFLECTION_PROMPT

    def test_prompt_includes_examples(self):
        assert "zone discipline" in META_REFLECTION_PROMPT.lower() or "easy sessions" in META_REFLECTION_PROMPT.lower()

    def test_prompt_output_schema(self):
        assert "meta_beliefs" in META_REFLECTION_PROMPT
        assert "category" in META_REFLECTION_PROMPT
        assert "confidence" in META_REFLECTION_PROMPT
