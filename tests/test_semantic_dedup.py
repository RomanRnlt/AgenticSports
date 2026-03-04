"""Tests for semantic config deduplication (Visionplan 8.12 D).

Covers:
- compute_config_similarity: raw text similarity via SequenceMatcher
- compute_weighted_config_similarity: weighted formula + name/desc similarity
- define_metric integration: semantic dedup blocks similar formulas (> 0.88)
- Edge cases: empty strings, single chars, DB errors
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.services.config_gc import (
    SIMILARITY_THRESHOLD,
    compute_config_similarity,
    compute_weighted_config_similarity,
)


USER_ID = "test-user-dedup"


# ---------------------------------------------------------------------------
# compute_config_similarity — unit tests
# ---------------------------------------------------------------------------


class TestComputeConfigSimilarity:
    def test_identical_strings(self) -> None:
        score = compute_config_similarity("hr * duration * 0.5", "hr * duration * 0.5")
        assert score == 1.0

    def test_very_similar_formulas(self) -> None:
        """Whitespace/decimal variations should produce high similarity."""
        score = compute_config_similarity(
            "hr * duration * 0.5",
            "hr*duration*0.50",
        )
        assert score > 0.85

    def test_different_formulas(self) -> None:
        score = compute_config_similarity(
            "hr * duration * 0.5",
            "rmssd * log(rmssd) + sdnn",
        )
        assert score < 0.5

    def test_empty_strings_identical(self) -> None:
        score = compute_config_similarity("", "")
        assert score == 1.0

    def test_one_empty_one_not(self) -> None:
        score = compute_config_similarity("", "hr * duration")
        assert score == 0.0

    def test_one_empty_one_not_reversed(self) -> None:
        score = compute_config_similarity("hr * duration", "")
        assert score == 0.0

    def test_single_character_identical(self) -> None:
        score = compute_config_similarity("a", "a")
        assert score == 1.0

    def test_single_character_different(self) -> None:
        score = compute_config_similarity("a", "z")
        assert score == 0.0

    def test_whitespace_normalization(self) -> None:
        """Extra whitespace should be collapsed before comparison."""
        score = compute_config_similarity(
            "  hr  *  duration  ",
            "hr * duration",
        )
        assert score == 1.0

    def test_case_normalization(self) -> None:
        """Comparison should be case-insensitive."""
        score = compute_config_similarity("HR * Duration", "hr * duration")
        assert score == 1.0

    def test_near_symmetry(self) -> None:
        """Similarity should be approximately symmetric.

        SequenceMatcher is not perfectly symmetric by design (it optimizes
        for one direction), but results should be close.
        """
        a = "hr * duration * 0.5"
        b = "heart_rate * time * 0.5"
        score_ab = compute_config_similarity(a, b)
        score_ba = compute_config_similarity(b, a)
        assert abs(score_ab - score_ba) < 0.1


# ---------------------------------------------------------------------------
# compute_weighted_config_similarity — unit tests
# ---------------------------------------------------------------------------


class TestComputeWeightedConfigSimilarity:
    def test_identical_formula_and_name(self) -> None:
        score = compute_weighted_config_similarity(
            formula1="hr * duration * 0.5",
            formula2="hr * duration * 0.5",
            name1="trimp",
            name2="trimp",
        )
        assert score == 1.0

    def test_similar_formula_different_name(self) -> None:
        """High formula similarity + low name similarity = moderate score."""
        score = compute_weighted_config_similarity(
            formula1="hr * duration * 0.5",
            formula2="hr * duration * 0.50",
            name1="trimp_score",
            name2="completely_different_name",
        )
        # Formula weight is 0.7, so even with different names,
        # high formula sim dominates
        assert score > 0.6

    def test_different_formula_similar_name(self) -> None:
        """Low formula similarity + high name similarity = moderate score."""
        score = compute_weighted_config_similarity(
            formula1="hr * duration * 0.5",
            formula2="rmssd * log(rmssd)",
            name1="training_load",
            name2="training_load_v2",
        )
        # Different formula keeps total score down
        assert score < SIMILARITY_THRESHOLD

    def test_both_different(self) -> None:
        score = compute_weighted_config_similarity(
            formula1="hr * duration * 0.5",
            formula2="rmssd * log(rmssd)",
            name1="trimp",
            name2="hrv_score",
        )
        assert score < 0.5

    def test_description_included_in_identity(self) -> None:
        """Description should influence the identity component."""
        score_without_desc = compute_weighted_config_similarity(
            formula1="a + b",
            formula2="a + b",
            name1="metric_a",
            name2="metric_b",
        )
        score_with_matching_desc = compute_weighted_config_similarity(
            formula1="a + b",
            formula2="a + b",
            name1="metric_a",
            name2="metric_b",
            desc1="Calculate training load",
            desc2="Calculate training load",
        )
        # Adding matching descriptions should increase identity similarity
        assert score_with_matching_desc >= score_without_desc


# ---------------------------------------------------------------------------
# define_metric semantic dedup — integration tests
# ---------------------------------------------------------------------------


def _make_registry(mock_settings, mock_get, mock_upsert, existing_metrics):
    """Helper to set up a registry with mocked dependencies."""
    settings = MagicMock()
    settings.use_supabase = True
    settings.agenticsports_user_id = USER_ID
    mock_settings.return_value = settings
    mock_get.return_value = existing_metrics
    mock_upsert.return_value = {"id": "new-id", "name": "new_metric"}

    from src.agent.tools.config_tools import register_config_tools
    from src.agent.tools.registry import ToolRegistry

    registry = ToolRegistry()
    user_model = MagicMock()
    user_model.user_id = USER_ID
    register_config_tools(registry, user_model)
    return registry


class TestDefineMetricSemanticDedup:
    @patch("src.db.agent_config_db.get_metric_definitions")
    @patch("src.db.agent_config_db.upsert_metric_definition")
    @patch("src.agent.tools.config_tools.get_settings")
    def test_identical_formulas_detected(self, mock_settings, mock_upsert, mock_get) -> None:
        """Identical formula under different name should be blocked as duplicate."""
        registry = _make_registry(mock_settings, mock_get, mock_upsert, [
            {"name": "existing_trimp", "formula": "hr * duration * 0.5", "description": ""},
        ])

        result = registry.execute("define_metric", {
            "name": "new_trimp",
            "formula": "hr * duration * 0.5",
        })

        assert result["status"] == "duplicate"
        assert result["existing_name"] == "existing_trimp"
        assert "similarity" in result
        assert result["similarity"] > SIMILARITY_THRESHOLD
        mock_upsert.assert_not_called()

    @patch("src.db.agent_config_db.get_metric_definitions")
    @patch("src.db.agent_config_db.upsert_metric_definition")
    @patch("src.agent.tools.config_tools.get_settings")
    def test_very_similar_formulas_detected(self, mock_settings, mock_upsert, mock_get) -> None:
        """Formulas with only whitespace/decimal differences should be detected.

        Names and descriptions are also similar to ensure the weighted score
        crosses the 0.88 threshold.
        """
        registry = _make_registry(mock_settings, mock_get, mock_upsert, [
            {"name": "trimp_score", "formula": "hr * duration * 0.5", "description": "TRIMP training load"},
        ])

        result = registry.execute("define_metric", {
            "name": "trimp_scores",
            "formula": "hr * duration * 0.50",
            "description": "TRIMP training load score",
        })

        assert result["status"] == "duplicate"
        assert result["existing_name"] == "trimp_score"
        assert result["similarity"] > SIMILARITY_THRESHOLD
        mock_upsert.assert_not_called()

    @patch("src.db.agent_config_db.get_metric_definitions")
    @patch("src.db.agent_config_db.upsert_metric_definition")
    @patch("src.agent.tools.config_tools.get_settings")
    def test_different_formulas_allowed(self, mock_settings, mock_upsert, mock_get) -> None:
        """Formulas that are genuinely different should be allowed."""
        registry = _make_registry(mock_settings, mock_get, mock_upsert, [
            {"name": "trimp", "formula": "hr * duration * 0.5", "description": ""},
        ])

        result = registry.execute("define_metric", {
            "name": "hrv_score",
            "formula": "rmssd * log(rmssd) + sdnn",
        })

        assert result["status"] == "success"
        mock_upsert.assert_called_once()

    @patch("src.db.agent_config_db.get_metric_definitions")
    @patch("src.db.agent_config_db.upsert_metric_definition")
    @patch("src.agent.tools.config_tools.get_settings")
    def test_similar_names_different_formula_allowed(self, mock_settings, mock_upsert, mock_get) -> None:
        """Similar names but completely different formulas should be allowed."""
        registry = _make_registry(mock_settings, mock_get, mock_upsert, [
            {"name": "training_load", "formula": "hr * duration * 0.5", "description": ""},
        ])

        result = registry.execute("define_metric", {
            "name": "training_load_v2",
            "formula": "power * duration / weight",
        })

        assert result["status"] == "success"
        mock_upsert.assert_called_once()

    @patch("src.db.agent_config_db.get_metric_definitions")
    @patch("src.db.agent_config_db.upsert_metric_definition")
    @patch("src.agent.tools.config_tools.get_settings")
    def test_same_name_update_not_blocked(self, mock_settings, mock_upsert, mock_get) -> None:
        """Updating an existing metric by the same name should skip dedup entirely."""
        registry = _make_registry(mock_settings, mock_get, mock_upsert, [
            {"name": "trimp", "formula": "hr * duration * 0.5", "description": ""},
        ])

        result = registry.execute("define_metric", {
            "name": "trimp",
            "formula": "hr * duration * 0.6",
        })

        assert result["status"] == "success"
        mock_upsert.assert_called_once()

    @patch("src.db.agent_config_db.get_metric_definitions", side_effect=Exception("DB error"))
    @patch("src.db.agent_config_db.upsert_metric_definition")
    @patch("src.agent.tools.config_tools.get_settings")
    def test_dedup_check_failure_proceeds(self, mock_settings, mock_upsert, mock_get) -> None:
        """If the dedup check fails (DB error), metric should still be created."""
        settings = MagicMock()
        settings.use_supabase = True
        settings.agenticsports_user_id = USER_ID
        mock_settings.return_value = settings
        mock_upsert.return_value = {"id": "789", "name": "test_metric"}

        from src.agent.tools.config_tools import register_config_tools
        from src.agent.tools.registry import ToolRegistry

        registry = ToolRegistry()
        user_model = MagicMock()
        user_model.user_id = USER_ID
        register_config_tools(registry, user_model)

        result = registry.execute("define_metric", {
            "name": "test_metric",
            "formula": "a + b",
        })

        assert result["status"] == "success"
        mock_upsert.assert_called_once()

    @patch("src.db.agent_config_db.get_metric_definitions")
    @patch("src.db.agent_config_db.upsert_metric_definition")
    @patch("src.agent.tools.config_tools.get_settings")
    def test_duplicate_returns_similarity_score(self, mock_settings, mock_upsert, mock_get) -> None:
        """Duplicate response should include the similarity score as a float."""
        registry = _make_registry(mock_settings, mock_get, mock_upsert, [
            {"name": "existing_trimp", "formula": "hr * duration * 0.5", "description": ""},
        ])

        result = registry.execute("define_metric", {
            "name": "new_trimp",
            "formula": "hr * duration * 0.5",
        })

        assert result["status"] == "duplicate"
        assert isinstance(result["similarity"], float)
        assert 0.0 <= result["similarity"] <= 1.0

    @patch("src.db.agent_config_db.get_metric_definitions")
    @patch("src.db.agent_config_db.upsert_metric_definition")
    @patch("src.agent.tools.config_tools.get_settings")
    def test_no_existing_metrics_allows_creation(self, mock_settings, mock_upsert, mock_get) -> None:
        """When no existing metrics, creation should succeed without issues."""
        registry = _make_registry(mock_settings, mock_get, mock_upsert, [])

        result = registry.execute("define_metric", {
            "name": "brand_new_metric",
            "formula": "a + b + c",
        })

        assert result["status"] == "success"
        mock_upsert.assert_called_once()
