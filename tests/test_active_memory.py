"""Priority 6 tests: Active Memory with Outcome Tracking.

Validates audit finding #6 ("Memory ist passiv, nicht aktiv"). Tests verify:
- Beliefs gain outcome tracking fields (utility, outcome_count, last_outcome)
- Confirmed beliefs increase confidence and utility
- Contradicted beliefs decrease confidence
- Old beliefs without outcome fields load correctly (backwards compatible)
- Episode retrieval prefers high-utility episodes over just recent ones
- Outcome recording integrates into the cognitive loop via update_beliefs action
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.memory.user_model import UserModel
from src.memory.episodes import (
    retrieve_relevant_episodes,
    record_episode_outcome,
    store_episode,
)


# ── Belief Outcome Tracking Tests ──────────────────────────────────

class TestBeliefOutcomeFields:
    """Verify new beliefs have outcome tracking fields."""

    def test_new_belief_has_utility_field(self):
        model = UserModel()
        b = model.add_belief("Test belief", "preference")
        assert "utility" in b
        assert b["utility"] == 0.0

    def test_new_belief_has_outcome_count(self):
        model = UserModel()
        b = model.add_belief("Test belief", "preference")
        assert b["outcome_count"] == 0

    def test_new_belief_has_last_outcome_null(self):
        model = UserModel()
        b = model.add_belief("Test belief", "preference")
        assert b["last_outcome"] is None

    def test_new_belief_has_empty_outcome_history(self):
        model = UserModel()
        b = model.add_belief("Test belief", "preference")
        assert b["outcome_history"] == []


class TestRecordOutcome:
    """Verify outcome recording updates beliefs correctly."""

    def test_confirmed_outcome_increases_confidence(self):
        model = UserModel()
        b = model.add_belief("Athlete runs easy too fast", "pattern", confidence=0.7)
        model.record_outcome(b["id"], "confirmed")
        assert b["confidence"] == pytest.approx(0.75)

    def test_confirmed_outcome_increases_utility(self):
        model = UserModel()
        b = model.add_belief("Test", "pattern", confidence=0.7)
        model.record_outcome(b["id"], "confirmed")
        assert b["utility"] == pytest.approx(0.1)

    def test_contradicted_outcome_decreases_confidence(self):
        model = UserModel()
        b = model.add_belief("Prefers morning runs", "preference", confidence=0.7)
        model.record_outcome(b["id"], "contradicted")
        assert b["confidence"] == pytest.approx(0.6)

    def test_contradicted_outcome_decreases_utility(self):
        model = UserModel()
        b = model.add_belief("Test", "preference", confidence=0.7)
        # First give it some utility
        b["utility"] = 0.3
        model.record_outcome(b["id"], "contradicted")
        assert b["utility"] == pytest.approx(0.25)

    def test_outcome_count_increments(self):
        model = UserModel()
        b = model.add_belief("Test", "pattern")
        model.record_outcome(b["id"], "confirmed")
        model.record_outcome(b["id"], "confirmed")
        assert b["outcome_count"] == 2

    def test_last_outcome_is_set(self):
        model = UserModel()
        b = model.add_belief("Test", "pattern")
        model.record_outcome(b["id"], "confirmed")
        assert b["last_outcome"] == "confirmed"
        model.record_outcome(b["id"], "contradicted")
        assert b["last_outcome"] == "contradicted"

    def test_outcome_history_grows(self):
        model = UserModel()
        b = model.add_belief("Test", "pattern")
        model.record_outcome(b["id"], "confirmed", detail="good compliance")
        model.record_outcome(b["id"], "contradicted", detail="skipped sessions")
        assert len(b["outcome_history"]) == 2
        assert b["outcome_history"][0]["type"] == "confirmed"
        assert b["outcome_history"][1]["type"] == "contradicted"
        assert "good compliance" in b["outcome_history"][0]["detail"]

    def test_confidence_capped_at_one(self):
        model = UserModel()
        b = model.add_belief("Test", "pattern", confidence=0.98)
        model.record_outcome(b["id"], "confirmed")
        assert b["confidence"] <= 1.0

    def test_confidence_floors_at_zero(self):
        model = UserModel()
        b = model.add_belief("Test", "pattern", confidence=0.05)
        model.record_outcome(b["id"], "contradicted")
        assert b["confidence"] >= 0.0

    def test_inactive_belief_not_updated(self):
        model = UserModel()
        b = model.add_belief("Test", "pattern", confidence=0.7)
        model.invalidate_belief(b["id"])
        result = model.record_outcome(b["id"], "confirmed")
        assert result is None

    def test_unknown_belief_returns_none(self):
        model = UserModel()
        result = model.record_outcome("nonexistent_id", "confirmed")
        assert result is None


class TestHighUtilityBeliefs:
    """Verify retrieval of beliefs with proven utility."""

    def test_returns_beliefs_above_utility_threshold(self):
        model = UserModel()
        b1 = model.add_belief("High utility", "pattern", confidence=0.8)
        b1["utility"] = 0.5
        b2 = model.add_belief("Low utility", "pattern", confidence=0.8)
        b2["utility"] = 0.1

        results = model.get_high_utility_beliefs(min_utility=0.3)
        assert len(results) == 1
        assert results[0]["text"] == "High utility"

    def test_filters_by_confidence(self):
        model = UserModel()
        b = model.add_belief("Good belief", "pattern", confidence=0.4)
        b["utility"] = 0.5
        results = model.get_high_utility_beliefs(min_utility=0.3, min_confidence=0.6)
        assert len(results) == 0

    def test_sorted_by_utility_descending(self):
        model = UserModel()
        b1 = model.add_belief("Medium", "pattern", confidence=0.8)
        b1["utility"] = 0.5
        b2 = model.add_belief("High", "pattern", confidence=0.8)
        b2["utility"] = 0.9

        results = model.get_high_utility_beliefs(min_utility=0.3)
        assert results[0]["text"] == "High"
        assert results[1]["text"] == "Medium"


# ── Backwards Compatibility Tests ──────────────────────────────────

class TestBackwardsCompatibility:
    """Verify old belief files without outcome fields load correctly."""

    def test_load_old_beliefs_without_utility(self):
        """Old beliefs missing utility fields get defaults on load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.json"
            # Simulate old belief format (no utility, outcome_count, etc.)
            old_data = {
                "structured_core": {"name": "Test", "sports": ["running"]},
                "beliefs": [
                    {
                        "id": "belief_old1",
                        "text": "Old belief without utility",
                        "category": "preference",
                        "confidence": 0.8,
                        "active": True,
                        "stability": "stable",
                        "durability": "global",
                        "source": "conversation",
                        "source_ref": None,
                        "first_observed": "2026-01-01T00:00:00",
                        "last_confirmed": "2026-01-01T00:00:00",
                        "valid_from": "2026-01-01",
                        "valid_until": None,
                        "learned_at": "2026-01-01T00:00:00",
                        "archived_at": None,
                        "superseded_by": None,
                        "embedding": None,
                        # NOTE: No utility, outcome_count, last_outcome, outcome_history
                    }
                ],
                "meta": {"created_at": "2026-01-01T00:00:00", "updated_at": "2026-01-01T00:00:00"},
            }
            model_path.write_text(json.dumps(old_data))

            model = UserModel(data_dir=Path(tmpdir))
            model.load()

            assert len(model.beliefs) == 1
            b = model.beliefs[0]
            assert b["utility"] == 0.0
            assert b["outcome_count"] == 0
            assert b["last_outcome"] is None
            assert b["outcome_history"] == []

    def test_load_and_record_outcome_on_old_belief(self):
        """Old beliefs can have outcomes recorded after loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.json"
            old_data = {
                "structured_core": {"name": "Test"},
                "beliefs": [
                    {
                        "id": "belief_old2",
                        "text": "Old belief",
                        "category": "preference",
                        "confidence": 0.7,
                        "active": True,
                        "stability": "stable",
                        "durability": "global",
                        "source": "conversation",
                        "source_ref": None,
                        "first_observed": "2026-01-01T00:00:00",
                        "last_confirmed": "2026-01-01T00:00:00",
                        "valid_from": "2026-01-01",
                        "valid_until": None,
                        "learned_at": "2026-01-01T00:00:00",
                        "archived_at": None,
                        "superseded_by": None,
                        "embedding": None,
                    }
                ],
                "meta": {},
            }
            model_path.write_text(json.dumps(old_data))

            model = UserModel(data_dir=Path(tmpdir))
            model.load()
            result = model.record_outcome("belief_old2", "confirmed", "test outcome")
            assert result is not None
            assert result["confidence"] == pytest.approx(0.75)
            assert result["utility"] == pytest.approx(0.1)


# ── Utility-Weighted Episode Retrieval Tests ────────────────────────

class TestUtilityWeightedRetrieval:
    """Verify episodes with high utility score higher in retrieval."""

    def _make_episode(self, ep_id, lessons, utility=0.0, generated_at="2026-02-01"):
        return {
            "id": ep_id,
            "block": "2026-W06",
            "period": "2026-02-01",
            "key_observations": [],
            "lessons": lessons,
            "patterns_detected": [],
            "compliance_rate": 0.8,
            "utility": utility,
            "generated_at": generated_at,
        }

    def test_high_utility_episode_preferred_over_recent(self):
        """Episode A (high utility) should rank above Episode B (recent but low utility)."""
        ep_a = self._make_episode(
            "ep_a", ["Easy pace adjustment worked well for running"], utility=0.8,
            generated_at="2026-01-15",
        )
        ep_b = self._make_episode(
            "ep_b", ["Introduced cycling cross-training"], utility=0.1,
            generated_at="2026-02-10",
        )
        # ep_b is more recent (index 0 = most recent)
        episodes = [ep_b, ep_a]

        context = {"goal": {"event": "Marathon"}, "sports": ["running"]}
        results = retrieve_relevant_episodes(context, episodes, max_results=2)

        # ep_a should come first due to higher utility despite being older
        assert results[0]["id"] == "ep_a"

    def test_utility_zero_episodes_still_retrieved(self):
        """Episodes with zero utility are still returned if they match keywords."""
        ep = self._make_episode("ep_1", ["Long run fatigue management"], utility=0.0)
        context = {"goal": {"event": "Marathon"}, "sports": ["running"]}
        results = retrieve_relevant_episodes(context, [ep], max_results=5)
        assert len(results) == 1

    def test_retrieval_with_mixed_utility(self):
        """Multiple episodes with different utilities are ranked correctly."""
        ep_high = self._make_episode("ep_h", ["Running pace zones"], utility=0.9)
        ep_mid = self._make_episode("ep_m", ["Running interval training"], utility=0.5)
        ep_low = self._make_episode("ep_l", ["Running easy recovery"], utility=0.1)
        # All same recency (same index order), same keywords
        episodes = [ep_low, ep_mid, ep_high]

        context = {"goal": {"event": "5K"}, "sports": ["running"]}
        results = retrieve_relevant_episodes(context, episodes, max_results=3)

        # Highest utility should be first
        assert results[0]["id"] == "ep_h"


# ── Episode Outcome Recording Tests ─────────────────────────────────

class TestEpisodeOutcomeRecording:
    """Verify episode utility can be updated from outcomes."""

    def test_record_positive_outcome(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ep = {
                "id": "ep_test1",
                "block": "2026-W06",
                "lessons": ["Easy pace"],
                "utility": 0.0,
            }
            store_episode(ep, storage_dir=tmpdir)

            result = record_episode_outcome("ep_test1", 0.2, storage_dir=tmpdir)
            assert result is not None
            assert result["utility"] == pytest.approx(0.2)
            assert result["referenced_count"] == 1
            assert result["outcome_validated"] is True

    def test_utility_capped_at_one(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ep = {"id": "ep_cap", "block": "W1", "lessons": [], "utility": 0.95}
            store_episode(ep, storage_dir=tmpdir)

            result = record_episode_outcome("ep_cap", 0.2, storage_dir=tmpdir)
            assert result["utility"] <= 1.0

    def test_utility_floors_at_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ep = {"id": "ep_floor", "block": "W1", "lessons": [], "utility": 0.02}
            store_episode(ep, storage_dir=tmpdir)

            result = record_episode_outcome("ep_floor", -0.1, storage_dir=tmpdir)
            assert result["utility"] >= 0.0

    def test_unknown_episode_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = record_episode_outcome("nonexistent", 0.1, storage_dir=tmpdir)
            assert result is None


# ── Action Integration Tests ────────────────────────────────────────

class TestUpdateBeliefsAction:
    """Verify update_beliefs action records outcomes from assessment."""

    def test_outcomes_recorded_on_matching_beliefs(self):
        from src.agent.actions import execute_action

        model = UserModel()
        b = model.add_belief(
            "Athlete runs easy sessions too fast", "pattern", confidence=0.7,
        )

        ctx = {
            "profile": {"name": "Test"},
            "user_model": model,
            "assessment": {
                "assessment": {
                    "compliance": 0.85,
                    "observations": [
                        "Athlete ran easy sessions at correct pace this week",
                    ],
                },
            },
        }
        with patch.object(model, "save"):
            result = execute_action("update_beliefs", ctx)

        assert result["beliefs_updated"] is True
        assert result["outcomes_recorded"] >= 1
        # Belief confidence should have increased (confirmed with good compliance)
        assert b["confidence"] > 0.7

    def test_no_outcomes_without_assessment(self):
        from src.agent.actions import execute_action

        model = UserModel()
        ctx = {"profile": {"name": "Test"}, "user_model": model}
        result = execute_action("update_beliefs", ctx)
        assert result["beliefs_updated"] is False

    def test_contradicted_on_low_compliance(self):
        from src.agent.actions import execute_action

        model = UserModel()
        b = model.add_belief(
            "Athlete trains consistently six days per week", "scheduling", confidence=0.8,
        )

        ctx = {
            "profile": {"name": "Test"},
            "user_model": model,
            "assessment": {
                "assessment": {
                    "compliance": 0.4,
                    "observations": [
                        "Athlete only completed 2 of 6 planned sessions this week",
                    ],
                },
            },
        }
        with patch.object(model, "save"):
            result = execute_action("update_beliefs", ctx)

        assert result["outcomes_recorded"] >= 1
        # Low compliance should contradict → decrease confidence
        assert b["confidence"] < 0.8
