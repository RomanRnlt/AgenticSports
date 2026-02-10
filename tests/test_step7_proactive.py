"""Tests for Step 7 Phase E: Proactive Outreach via Chat.

Tests the proactive message queue, engagement tracking,
silence-decay mechanism, and conversation-based triggers.
"""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from src.agent.proactive import (
    check_proactive_triggers,
    format_proactive_message,
    queue_proactive_message,
    get_pending_messages,
    deliver_message,
    record_engagement,
    expire_stale_messages,
    calculate_silence_decay,
    check_conversation_triggers,
    _load_queue,
    _save_queue,
)


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def queue_path(tmp_path):
    """Create a temporary queue file path."""
    return tmp_path / "proactive_queue.json"


@pytest.fixture
def sample_trigger():
    return {
        "type": "goal_at_risk",
        "priority": "high",
        "data": {"predicted_time": "1:52:00", "target_time": "1:45:00"},
    }


# ── Queue Operations ─────────────────────────────────────────────


class TestQueueOperations:
    def test_load_empty_queue(self, queue_path):
        assert _load_queue(queue_path) == []

    def test_save_and_load_queue(self, queue_path):
        queue = [{"id": "msg_test", "status": "pending"}]
        _save_queue(queue, queue_path)
        loaded = _load_queue(queue_path)
        assert len(loaded) == 1
        assert loaded[0]["id"] == "msg_test"

    def test_queue_proactive_message(self, queue_path, sample_trigger):
        msg = queue_proactive_message(sample_trigger, priority=0.8, queue_path=queue_path)
        assert msg["id"].startswith("msg_")
        assert msg["trigger_type"] == "goal_at_risk"
        assert msg["priority"] == 0.8
        assert msg["status"] == "pending"
        assert msg["delivered_at"] is None
        assert msg["engagement_tracking"]["user_responded_at"] is None

    def test_queue_creates_file(self, queue_path, sample_trigger):
        queue_proactive_message(sample_trigger, queue_path=queue_path)
        assert queue_path.exists()

    def test_queue_multiple_messages(self, queue_path):
        for i in range(3):
            queue_proactive_message(
                {"type": f"trigger_{i}", "data": {}},
                priority=i * 0.3,
                queue_path=queue_path,
            )
        queue = _load_queue(queue_path)
        assert len(queue) == 3

    def test_queue_default_priority(self, queue_path, sample_trigger):
        msg = queue_proactive_message(sample_trigger, queue_path=queue_path)
        assert msg["priority"] == 0.5


# ── Get Pending Messages ─────────────────────────────────────────


class TestGetPendingMessages:
    def test_empty_queue(self, queue_path):
        assert get_pending_messages(queue_path) == []

    def test_returns_only_pending(self, queue_path):
        _save_queue([
            {"id": "m1", "status": "pending", "priority": 0.5},
            {"id": "m2", "status": "delivered", "priority": 0.9},
            {"id": "m3", "status": "pending", "priority": 0.7},
        ], queue_path)

        pending = get_pending_messages(queue_path)
        assert len(pending) == 2
        ids = [m["id"] for m in pending]
        assert "m2" not in ids

    def test_sorted_by_priority_descending(self, queue_path):
        _save_queue([
            {"id": "low", "status": "pending", "priority": 0.2},
            {"id": "high", "status": "pending", "priority": 0.9},
            {"id": "med", "status": "pending", "priority": 0.5},
        ], queue_path)

        pending = get_pending_messages(queue_path)
        assert [m["id"] for m in pending] == ["high", "med", "low"]


# ── Deliver Message ──────────────────────────────────────────────


class TestDeliverMessage:
    def test_deliver_marks_as_delivered(self, queue_path, sample_trigger):
        msg = queue_proactive_message(sample_trigger, queue_path=queue_path)
        result = deliver_message(msg["id"], queue_path=queue_path)
        assert result["status"] == "delivered"
        assert result["delivered_at"] is not None

    def test_deliver_nonexistent_returns_none(self, queue_path):
        assert deliver_message("msg_nonexistent", queue_path=queue_path) is None

    def test_deliver_persists(self, queue_path, sample_trigger):
        msg = queue_proactive_message(sample_trigger, queue_path=queue_path)
        deliver_message(msg["id"], queue_path=queue_path)
        queue = _load_queue(queue_path)
        assert queue[0]["status"] == "delivered"


# ── Engagement Tracking ──────────────────────────────────────────


class TestEngagementTracking:
    def test_record_engagement_basic(self, queue_path, sample_trigger):
        msg = queue_proactive_message(sample_trigger, queue_path=queue_path)
        deliver_message(msg["id"], queue_path=queue_path)
        result = record_engagement(
            msg["id"], responded=True, continued_session=True, turns_after=5,
            queue_path=queue_path,
        )
        assert result is not None
        tracking = result["engagement_tracking"]
        assert tracking["user_responded_at"] is not None
        assert tracking["user_continued_session"] is True
        assert tracking["session_turns_after_delivery"] == 5

    def test_record_engagement_undelivered_returns_none(self, queue_path, sample_trigger):
        msg = queue_proactive_message(sample_trigger, queue_path=queue_path)
        # Not delivered yet
        result = record_engagement(msg["id"], responded=True, queue_path=queue_path)
        assert result is None

    def test_record_engagement_nonexistent_returns_none(self, queue_path):
        assert record_engagement("msg_nope", queue_path=queue_path) is None

    def test_engagement_response_latency(self, queue_path, sample_trigger):
        msg = queue_proactive_message(sample_trigger, queue_path=queue_path)
        deliver_message(msg["id"], queue_path=queue_path)
        result = record_engagement(msg["id"], responded=True, queue_path=queue_path)
        tracking = result["engagement_tracking"]
        # Latency should be >= 0 (delivered just now)
        assert tracking["response_latency_seconds"] is not None
        assert tracking["response_latency_seconds"] >= 0


# ── Expire Stale Messages ───────────────────────────────────────


class TestExpireStaleMessages:
    def test_expire_old_messages(self, queue_path):
        old_time = (datetime.now() - timedelta(days=10)).isoformat(timespec="seconds")
        _save_queue([
            {"id": "m_old", "status": "pending", "created_at": old_time},
            {"id": "m_new", "status": "pending", "created_at": datetime.now().isoformat(timespec="seconds")},
        ], queue_path)

        expired = expire_stale_messages(max_age_days=7, queue_path=queue_path)
        assert len(expired) == 1
        assert expired[0]["id"] == "m_old"

        queue = _load_queue(queue_path)
        assert queue[0]["status"] == "expired"
        assert queue[1]["status"] == "pending"

    def test_expire_nothing_when_all_fresh(self, queue_path):
        now = datetime.now().isoformat(timespec="seconds")
        _save_queue([{"id": "m1", "status": "pending", "created_at": now}], queue_path)
        expired = expire_stale_messages(max_age_days=7, queue_path=queue_path)
        assert len(expired) == 0

    def test_expire_skips_delivered(self, queue_path):
        old_time = (datetime.now() - timedelta(days=10)).isoformat(timespec="seconds")
        _save_queue([
            {"id": "m1", "status": "delivered", "created_at": old_time},
        ], queue_path)
        expired = expire_stale_messages(max_age_days=7, queue_path=queue_path)
        assert len(expired) == 0


# ── Silence Decay ────────────────────────────────────────────────


class TestSilenceDecay:
    def test_no_interaction_returns_moderate(self):
        assert calculate_silence_decay(None) == 0.5

    def test_recent_interaction_no_boost(self):
        now = datetime.now().isoformat(timespec="seconds")
        assert calculate_silence_decay(now) == 0.0

    def test_one_day_silence(self):
        one_day_ago = (datetime.now() - timedelta(hours=25)).isoformat(timespec="seconds")
        result = calculate_silence_decay(one_day_ago)
        assert result == 0.1

    def test_four_days_silence(self):
        four_days = (datetime.now() - timedelta(days=4)).isoformat(timespec="seconds")
        result = calculate_silence_decay(four_days)
        assert result == 0.3

    def test_seven_days_silence(self):
        seven_days = (datetime.now() - timedelta(days=7)).isoformat(timespec="seconds")
        result = calculate_silence_decay(seven_days)
        assert result == 0.5

    def test_two_weeks_silence(self):
        two_weeks = (datetime.now() - timedelta(days=14)).isoformat(timespec="seconds")
        result = calculate_silence_decay(two_weeks)
        assert result == 0.7


# ── Conversation-Based Triggers ──────────────────────────────────


class TestConversationTriggers:
    def test_no_triggers_for_active_user(self):
        now = datetime.now().isoformat(timespec="seconds")
        triggers = check_conversation_triggers({}, last_interaction=now)
        assert len(triggers) == 0

    def test_silence_checkin_after_5_days(self):
        six_days = (datetime.now() - timedelta(days=6)).isoformat(timespec="seconds")
        triggers = check_conversation_triggers({}, last_interaction=six_days)
        assert len(triggers) == 1
        assert triggers[0]["type"] == "silence_checkin"
        assert triggers[0]["data"]["days_since_last_chat"] >= 5

    def test_no_checkin_at_4_days(self):
        four_days = (datetime.now() - timedelta(days=4)).isoformat(timespec="seconds")
        triggers = check_conversation_triggers({}, last_interaction=four_days)
        assert len(triggers) == 0

    def test_no_interaction_history(self):
        triggers = check_conversation_triggers({}, last_interaction=None)
        assert len(triggers) == 0

    def test_silence_checkin_has_urgency(self):
        ten_days = (datetime.now() - timedelta(days=10)).isoformat(timespec="seconds")
        triggers = check_conversation_triggers({}, last_interaction=ten_days)
        assert triggers[0]["urgency"] >= 0.5


# ── Backward Compatibility ───────────────────────────────────────


class TestProactiveBackwardCompat:
    def test_check_proactive_triggers_still_works(self):
        """Existing check_proactive_triggers function unchanged."""
        profile = {"goal": {"event": "10K"}}
        activities = []
        episodes = []
        trajectory = {"trajectory": {"on_track": True}, "confidence": 0.7}

        triggers = check_proactive_triggers(profile, activities, episodes, trajectory)
        assert isinstance(triggers, list)

    def test_format_proactive_message_still_works(self):
        """Existing format_proactive_message function unchanged."""
        trigger = {
            "type": "on_track",
            "data": {"predicted_time": "0:48:00", "confidence": 0.8},
        }
        context = {"goal": {"event": "10K"}}
        msg = format_proactive_message(trigger, context)
        assert "Looking good" in msg


# ── Integration: Queue Lifecycle ─────────────────────────────────


class TestQueueLifecycle:
    @pytest.mark.integration
    def test_full_queue_lifecycle(self, queue_path):
        """Full lifecycle: queue -> get_pending -> deliver -> record -> expire."""
        # 1. Queue messages
        msg1 = queue_proactive_message(
            {"type": "goal_at_risk", "data": {}}, priority=0.9, queue_path=queue_path,
        )
        msg2 = queue_proactive_message(
            {"type": "fitness_improving", "data": {}}, priority=0.3, queue_path=queue_path,
        )

        # 2. Get pending — sorted by priority
        pending = get_pending_messages(queue_path)
        assert len(pending) == 2
        assert pending[0]["id"] == msg1["id"]

        # 3. Deliver top message
        delivered = deliver_message(msg1["id"], queue_path=queue_path)
        assert delivered["status"] == "delivered"

        # 4. Pending count decreases
        pending_after = get_pending_messages(queue_path)
        assert len(pending_after) == 1

        # 5. Record engagement
        engaged = record_engagement(
            msg1["id"], responded=True, continued_session=True, turns_after=3,
            queue_path=queue_path,
        )
        assert engaged["engagement_tracking"]["session_turns_after_delivery"] == 3

        # 6. Second message stays pending
        assert pending_after[0]["id"] == msg2["id"]
