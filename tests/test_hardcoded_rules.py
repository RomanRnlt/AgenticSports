"""Priority 7 tests: Eliminate Hardcoded Rules.

Validates audit finding #4 ("Hardcoded Rules trotz 'No Rules' Claim"). Tests verify:
- Impact classification uses LLM with keyword fallback (Group A)
- Goal type inference uses LLM with keyword fallback (Group B)
- Reflection triggers are event-driven with time-based fallback (Group C)
- Proactive triggers use LLM with heuristic fallback (Group D)
- All replacements maintain backwards compatibility via fallbacks
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import pytest

from src.agent.autonomy import (
    classify_impact,
    classify_and_apply,
    ImpactLevel,
    _classify_impact_keywords,
)
from src.agent.startup import (
    infer_goal_type,
    _infer_goal_type_keywords,
    _detect_triggers,
    _detect_triggers_heuristic,
    GOAL_TYPES,
)
from src.agent.reflection import (
    _is_reflection_due,
    REFLECTION_MIN_DAYS,
    REFLECTION_MIN_ACTIVITIES,
    COMPLIANCE_DEVIATION_THRESHOLD,
)
from src.agent.proactive import (
    _detect_high_fatigue,
)


# ── Helper ───────────────────────────────────────────────────────────

def _mock_llm_response(result_dict):
    """Create a mock Gemini response returning given JSON."""
    mock_response = MagicMock()
    mock_response.text = json.dumps(result_dict)
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    return mock_client


# ── Group A: Impact Classification ───────────────────────────────────

class TestImpactClassificationLLM:
    """Verify LLM-based impact classification with keyword fallback."""

    def test_llm_classifies_novel_high_impact(self):
        """Novel adjustment without keywords classified correctly by LLM."""
        mock = _mock_llm_response({"impact": "high", "reasoning": "fundamental goal change"})
        with patch("src.agent.autonomy.get_client", return_value=mock):
            level = classify_impact({"description": "Switch from 5K to marathon training"})
        assert level == ImpactLevel.HIGH

    def test_llm_classifies_medium_impact(self):
        mock = _mock_llm_response({"impact": "medium", "reasoning": "volume change"})
        with patch("src.agent.autonomy.get_client", return_value=mock):
            level = classify_impact({"description": "Add one more easy run per week"})
        assert level == ImpactLevel.MEDIUM

    def test_llm_classifies_low_impact(self):
        mock = _mock_llm_response({"impact": "low", "reasoning": "minor tweak"})
        with patch("src.agent.autonomy.get_client", return_value=mock):
            level = classify_impact({"description": "Adjust pace target by 5 sec/km"})
        assert level == ImpactLevel.LOW

    def test_falls_back_to_keywords_on_llm_failure(self):
        mock = MagicMock()
        mock.models.generate_content.side_effect = Exception("API error")
        with patch("src.agent.autonomy.get_client", return_value=mock):
            level = classify_impact({"description": "restructure periodization"})
        assert level == ImpactLevel.HIGH  # keyword fallback

    def test_pre_classified_impact_trusted(self):
        """If adjustment already has 'impact' field, LLM is not called."""
        level = classify_impact({"impact": "high", "description": "anything"})
        assert level == ImpactLevel.HIGH

    def test_keyword_fallback_directly(self):
        """Verify keyword fallback works standalone."""
        assert _classify_impact_keywords("injury protocol") == ImpactLevel.HIGH
        assert _classify_impact_keywords("reduce volume") == ImpactLevel.MEDIUM
        assert _classify_impact_keywords("adjust pace zone") == ImpactLevel.LOW

    def test_classify_and_apply_with_llm(self):
        mock = _mock_llm_response({"impact": "medium", "reasoning": "test"})
        adjustments = [
            {"description": "Add extra recovery day"},
            {"impact": "low", "description": "minor tweak"},
        ]
        with patch("src.agent.autonomy.get_client", return_value=mock):
            result = classify_and_apply(adjustments)
        assert len(result["auto_applied"]) >= 1  # low-impact auto-applied
        assert len(result["proposals"]) >= 1  # medium-impact proposed


# ── Group B: Goal Type Inference ─────────────────────────────────────

class TestGoalTypeInferenceLLM:
    """Verify LLM-based goal type inference with keyword fallback."""

    def test_llm_infers_race_target(self):
        mock = _mock_llm_response({"goal_type": "race_target"})
        with patch("src.agent.startup.get_client", return_value=mock):
            gt = infer_goal_type({"event": "Berlin Marathon", "target_time": "3:30", "target_date": "2026-09-27"})
        assert gt == "race_target"

    def test_llm_infers_performance_target(self):
        mock = _mock_llm_response({"goal_type": "performance_target"})
        with patch("src.agent.startup.get_client", return_value=mock):
            gt = infer_goal_type({"event": "Break my 5K PR"})
        assert gt == "performance_target"

    def test_llm_infers_routine(self):
        mock = _mock_llm_response({"goal_type": "routine"})
        with patch("src.agent.startup.get_client", return_value=mock):
            gt = infer_goal_type({"event": "Stay healthy and active"})
        assert gt == "routine"

    def test_falls_back_to_keywords_on_llm_failure(self):
        mock = MagicMock()
        mock.models.generate_content.side_effect = Exception("API error")
        with patch("src.agent.startup.get_client", return_value=mock):
            gt = infer_goal_type({"event": "marathon", "target_time": "3:30", "target_date": "2026-10-01"})
        assert gt == "race_target"  # keyword fallback

    def test_keyword_fallback_directly(self):
        assert _infer_goal_type_keywords({"event": "5k run"}) == "performance_target"
        assert _infer_goal_type_keywords({"event": "run 3 times a week"}) == "routine"
        assert _infer_goal_type_keywords({"event": "be happy"}) == "general"

    def test_all_goal_types_valid(self):
        assert GOAL_TYPES == {"race_target", "performance_target", "routine", "general"}


# ── Group C: Event-Driven Reflection Triggers ────────────────────────

class TestEventDrivenReflection:
    """Verify event-driven reflection triggers replace fixed time-only triggers."""

    def _make_activities(self, count, start_date="2026-02-01"):
        return [
            {"start_time": f"{start_date}T{10+i}:00:00"}
            for i in range(count)
        ]

    def test_no_activities_no_reflection(self):
        assert _is_reflection_due(None, []) is False

    def test_first_reflection_with_enough_data(self):
        """First reflection triggers when enough time span and activities."""
        activities = [
            {"start_time": "2026-01-01T10:00:00"},
            {"start_time": "2026-01-05T10:00:00"},
            {"start_time": "2026-01-10T10:00:00"},
        ]
        assert _is_reflection_due(None, activities) is True

    def test_first_reflection_too_few_activities(self):
        activities = [{"start_time": "2026-01-01T10:00:00"}]
        assert _is_reflection_due(None, activities) is False

    def test_compliance_deviation_triggers_reflection(self):
        """Significant compliance deviation triggers reflection early (P7)."""
        last_episode = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "period": "2026-02-01",
        }
        # Plan has 5 sessions but only 1 activity since last reflection
        plan = {"sessions": [{"day": "Mon"}, {"day": "Tue"}, {"day": "Wed"}, {"day": "Thu"}, {"day": "Fri"}]}
        activities = [
            {"start_time": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()},
            {"start_time": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()},
        ]
        # compliance = 2/5 = 0.4, deviation = |1.0 - 0.4| = 0.6 > 0.3 threshold
        assert _is_reflection_due(last_episode, activities, plan=plan) is True

    def test_good_compliance_no_early_trigger(self):
        """Good compliance doesn't trigger early reflection."""
        last_episode = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        plan = {"sessions": [{"day": "Mon"}, {"day": "Tue"}, {"day": "Wed"}]}
        activities = [
            {"start_time": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()},
            {"start_time": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()},
            {"start_time": (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()},
        ]
        # compliance = 3/3 = 1.0, deviation = 0.0 < 0.3
        assert _is_reflection_due(last_episode, activities, plan=plan) is False

    def test_time_based_fallback_still_works(self):
        """Time-based trigger works when compliance check doesn't trigger."""
        old_episode = {
            "generated_at": (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(timespec="seconds"),
        }
        activities = [
            {"start_time": (datetime.now(timezone.utc) - timedelta(days=i)).isoformat()}
            for i in range(5)
        ]
        assert _is_reflection_due(old_episode, activities) is True

    def test_threshold_value(self):
        assert COMPLIANCE_DEVIATION_THRESHOLD == 0.3


# ── Group D: Proactive Triggers ──────────────────────────────────────

class TestProactiveTriggers:
    """Verify LLM-based proactive trigger detection with heuristic fallback."""

    def test_llm_detects_fatigue(self):
        mock = _mock_llm_response({"triggers": [
            {"type": "fatigue_warning", "priority": "high", "reasoning": "TRIMP up 40%"},
        ]})
        with patch("src.agent.startup.get_client", return_value=mock):
            triggers = _detect_triggers(
                None,
                {"trimp_direction": "increasing (+40%)", "volume_direction": "stable", "sessions_per_week": [4, 4]},
                [],
            )
        assert any(t["type"] == "fatigue_warning" for t in triggers)

    def test_llm_detects_no_triggers(self):
        mock = _mock_llm_response({"triggers": []})
        with patch("src.agent.startup.get_client", return_value=mock):
            triggers = _detect_triggers(
                None,
                {"trimp_direction": "stable", "volume_direction": "stable", "sessions_per_week": [3, 3]},
                [],
            )
        assert len(triggers) == 0

    def test_heuristic_fallback_on_llm_failure(self):
        mock = MagicMock()
        mock.models.generate_content.side_effect = Exception("API error")
        with patch("src.agent.startup.get_client", return_value=mock):
            triggers = _detect_triggers(
                {"compliance_rate": 0.4, "matched_count": 2, "planned_count": 5},
                {"trimp_direction": "increasing (+35%)", "volume_direction": "stable", "sessions_per_week": [3]},
                [],
            )
        # Heuristic should catch TRIMP >30% and low compliance
        types = [t["type"] for t in triggers]
        assert "fatigue_warning" in types
        assert "compliance_low" in types

    def test_heuristic_directly(self):
        triggers = _detect_triggers_heuristic(
            {"compliance_rate": 0.95, "matched_count": 5, "planned_count": 5},
            {"trimp_direction": "stable", "volume_direction": "increasing (+20%)", "sessions_per_week": [4, 5]},
        )
        types = [t["type"] for t in triggers]
        assert "great_consistency" in types
        assert "fitness_improving" in types


class TestFatigueDetection:
    """Verify LLM-based fatigue detection in proactive.py."""

    def test_llm_detects_fatigue(self):
        mock = _mock_llm_response({"fatigued": True, "reasoning": "elevated HR"})
        activities = [
            {"sport": "running", "hr_zone": 3, "heart_rate": {"avg": 155}, "duration_seconds": 3000}
            for _ in range(3)
        ]
        with patch("src.agent.llm.get_client", return_value=mock):
            assert _detect_high_fatigue(activities, []) is True

    def test_llm_says_not_fatigued(self):
        mock = _mock_llm_response({"fatigued": False, "reasoning": "normal HR"})
        activities = [
            {"sport": "running", "hr_zone": 2, "heart_rate": {"avg": 130}, "duration_seconds": 3000}
            for _ in range(3)
        ]
        with patch("src.agent.llm.get_client", return_value=mock):
            assert _detect_high_fatigue(activities, []) is False

    def test_heuristic_fallback_on_llm_failure(self):
        mock = MagicMock()
        mock.models.generate_content.side_effect = Exception("API error")
        activities = [
            {"sport": "running", "hr_zone": 4, "heart_rate": {"avg": 155}, "duration_seconds": 3000}
            for _ in range(3)
        ]
        with patch("src.agent.llm.get_client", return_value=mock):
            assert _detect_high_fatigue(activities, []) is True  # heuristic: 3 activities with zone>=3 and HR>140

    def test_no_activities(self):
        assert _detect_high_fatigue([], []) is False
