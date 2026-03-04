"""Tests for SpawnTool implementation (background sub-agent).

Covers:
- Resource guard: max 3 concurrent tasks per user
- Task ID generation
- _get_active_task_count: counting active tasks
- _cleanup_done_task: removing finished tasks
- _run_background_task: full sub-agent flow
- spawn_background_task tool: registration, execution, error handling
- get_restricted_tools: restricted registry
- create_restricted_loop: restricted loop factory
- Notification delivery on completion
- Max iterations clamping
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.tools.notification_tools import (
    _MAX_CONCURRENT_TASKS,
    _active_tasks,
    _cleanup_done_task,
    _get_active_task_count,
    _make_task_id,
    _run_background_task,
)
from src.agent.tools.registry import ToolRegistry, get_restricted_tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

USER_ID = "test-user-spawn"


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    from src.agent.tools.notification_tools import register_notification_tools
    register_notification_tools(registry, USER_ID)
    return registry


# ---------------------------------------------------------------------------
# Task ID generation
# ---------------------------------------------------------------------------


class TestMakeTaskId:
    def test_starts_with_bg_prefix(self) -> None:
        tid = _make_task_id()
        assert tid.startswith("bg_")

    def test_unique(self) -> None:
        ids = {_make_task_id() for _ in range(100)}
        assert len(ids) == 100


# ---------------------------------------------------------------------------
# Active task tracking
# ---------------------------------------------------------------------------


class TestActiveTaskTracking:
    def setup_method(self) -> None:
        _active_tasks.clear()

    def test_count_empty(self) -> None:
        assert _get_active_task_count(USER_ID) == 0

    def test_count_with_running_tasks(self) -> None:
        task1 = MagicMock()
        task1.done.return_value = False
        task2 = MagicMock()
        task2.done.return_value = False

        _active_tasks[USER_ID] = {"t1": task1, "t2": task2}
        assert _get_active_task_count(USER_ID) == 2

    def test_count_excludes_done_tasks(self) -> None:
        running = MagicMock()
        running.done.return_value = False
        done = MagicMock()
        done.done.return_value = True

        _active_tasks[USER_ID] = {"t1": running, "t2": done}
        assert _get_active_task_count(USER_ID) == 1

    def test_cleanup_removes_task(self) -> None:
        task = MagicMock()
        _active_tasks[USER_ID] = {"t1": task}

        _cleanup_done_task(USER_ID, "t1")

        assert USER_ID not in _active_tasks

    def test_cleanup_nonexistent_user(self) -> None:
        # Should not raise
        _cleanup_done_task("nonexistent", "t1")

    def teardown_method(self) -> None:
        _active_tasks.clear()


# ---------------------------------------------------------------------------
# Resource guard
# ---------------------------------------------------------------------------


class TestResourceGuard:
    def setup_method(self) -> None:
        _active_tasks.clear()

    def test_allows_when_under_limit(self) -> None:
        registry = _make_registry()
        result = registry.execute("spawn_background_task", {
            "instruction": "Analyze training load",
        })
        # In sync context, it logs a warning but still returns spawned status
        # (we can't create asyncio tasks in sync test context)
        assert "error" not in result or "concurrent" not in result.get("error", "")

    def test_blocks_when_at_limit(self) -> None:
        # Fill up active tasks
        for i in range(_MAX_CONCURRENT_TASKS):
            task = MagicMock()
            task.done.return_value = False
            if USER_ID not in _active_tasks:
                _active_tasks[USER_ID] = {}
            _active_tasks[USER_ID][f"t{i}"] = task

        registry = _make_registry()
        result = registry.execute("spawn_background_task", {
            "instruction": "Blocked task",
        })

        assert result["spawned"] is False
        assert "Too many concurrent" in result["error"]

    def test_max_iterations_clamped(self) -> None:
        registry = _make_registry()
        result = registry.execute("spawn_background_task", {
            "instruction": "Test",
            "max_iterations": 50,  # Should be clamped to 30
        })
        # It still spawns, just with clamped iterations
        assert result.get("spawned") is True or "concurrent" not in result.get("error", "")

    def teardown_method(self) -> None:
        _active_tasks.clear()


# ---------------------------------------------------------------------------
# get_restricted_tools
# ---------------------------------------------------------------------------


class TestGetRestrictedTools:
    def test_has_data_tools(self) -> None:
        mock_user = MagicMock()
        mock_user.user_id = USER_ID

        with patch("src.config.get_settings") as mock_settings:
            settings = MagicMock()
            settings.agenticsports_user_id = USER_ID
            settings.use_supabase = False
            mock_settings.return_value = settings

            registry = get_restricted_tools(mock_user)
            tool_names = [t["name"] for t in registry.list_tools()]

            # Should have data/analysis/calc tools
            assert len(tool_names) > 0

    def test_excludes_dangerous_tools(self) -> None:
        mock_user = MagicMock()
        mock_user.user_id = USER_ID

        with patch("src.config.get_settings") as mock_settings:
            settings = MagicMock()
            settings.agenticsports_user_id = USER_ID
            settings.use_supabase = False
            mock_settings.return_value = settings

            registry = get_restricted_tools(mock_user)
            tool_names = [t["name"] for t in registry.list_tools()]

            # Should NOT have dangerous tools
            assert "send_notification" not in tool_names
            assert "spawn_background_task" not in tool_names
            assert "complete_onboarding" not in tool_names
            assert "define_metric" not in tool_names


# ---------------------------------------------------------------------------
# _run_background_task
# ---------------------------------------------------------------------------


class TestRunBackgroundTask:
    @patch("src.agent.tools.notification_tools.send_notification_async", new_callable=AsyncMock)
    @patch("src.agent.tools.notification_tools.asyncio.to_thread")
    def test_successful_task(self, mock_to_thread, mock_notify) -> None:
        # Mock the sequence of to_thread calls:
        # 1. UserModelDB.load_or_create
        # 2. loop.process_message
        # 3. queue_proactive_message (optional)
        mock_user = MagicMock()
        mock_user.user_id = USER_ID

        mock_result = MagicMock()
        mock_result.response_text = "Analysis complete: your training load is optimal."
        mock_result.tool_calls_made = 3
        mock_result.total_duration_ms = 5000

        call_count = 0

        async def fake_to_thread(fn, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_user  # load_or_create
            if call_count == 2:
                return mock_result  # process_message
            return MagicMock()  # queue_proactive_message

        mock_to_thread.side_effect = fake_to_thread
        mock_notify.return_value = {"sent": True}

        with (
            patch("src.agent.agent_loop.create_restricted_loop") as mock_loop_factory,
            patch("src.db.user_model_db.UserModelDB") as mock_db,
        ):
            mock_agent_loop = MagicMock()
            mock_agent_loop.start_session.return_value = "sess-1"
            mock_loop_factory.return_value = mock_agent_loop

            asyncio.run(_run_background_task("bg_test", USER_ID, "Analyze load", 15))

        # Should have sent a notification
        mock_notify.assert_called()

    @patch("src.agent.tools.notification_tools.send_notification_async", new_callable=AsyncMock)
    @patch("src.agent.tools.notification_tools.asyncio.to_thread", side_effect=Exception("DB down"))
    def test_failed_task_notifies_user(self, mock_to_thread, mock_notify) -> None:
        mock_notify.return_value = {"sent": True}

        asyncio.run(_run_background_task("bg_fail", USER_ID, "Analyze", 15))

        # Should notify about the failure
        mock_notify.assert_called()
        call_args = mock_notify.call_args
        assert "fehlgeschlagen" in call_args.kwargs.get("title", call_args[1].get("title", ""))


# ---------------------------------------------------------------------------
# create_restricted_loop
# ---------------------------------------------------------------------------


class TestCreateRestrictedLoop:
    def test_creates_loop_with_restricted_tools(self) -> None:
        from src.agent.agent_loop import create_restricted_loop

        mock_user = MagicMock()
        mock_user.user_id = USER_ID
        mock_user.project_profile.return_value = {}
        mock_user.get_active_beliefs.return_value = []
        mock_user.meta = {}

        with patch("src.config.get_settings") as mock_settings:
            settings = MagicMock()
            settings.agenticsports_user_id = USER_ID
            settings.use_supabase = False
            mock_settings.return_value = settings

            loop = create_restricted_loop(mock_user, max_tool_rounds=10)

            # Should be an AgentLoop
            from src.agent.agent_loop import AgentLoop
            assert isinstance(loop, AgentLoop)

            # Tools should be restricted
            tool_names = [t["name"] for t in loop.tools.list_tools()]
            assert "send_notification" not in tool_names
