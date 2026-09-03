"""Unit tests for Session Manager, Task Tracking, and Result Fencing."""

import asyncio
import pytest

from app.models.events import EventType, ToolResultEnvelope
from app.models.state import SessionState
from app.models.tool import ToolResult
from app.services.session_manager import SessionManager


@pytest.fixture
def manager():
    """Create a fresh SessionManager instance for each test."""
    mgr = SessionManager()
    yield mgr
    mgr.clear()


# ---------------------------------------------------------------------------
# Test 1: Normal tool completion updates state when versions match
# ---------------------------------------------------------------------------
def test_normal_tool_completion_updates_state(manager: SessionManager):
    session_id = "sess_001"

    # Start turn 1
    session = manager.start_turn(session_id, transcript="Book a ride to SFO")
    assert session.current_version == 1
    assert session.state == SessionState.THINKING

    # Transition to tool running if appropriate
    manager.set_state(session_id, SessionState.TOOL_RUNNING)
    assert session.state == SessionState.TOOL_RUNNING

    # Create matching envelope for version 1
    tool_result = ToolResult.ok(
        tool_name="book_ride",
        output={"ride_id": "ride_123", "status": "confirmed"},
    )
    envelope = ToolResultEnvelope(
        session_id=session_id,
        version=1,
        tool_result=tool_result,
    )

    # Process result
    accepted = manager.process_tool_result(envelope)
    assert accepted is True
    assert session.state == SessionState.COMPLETED

    # Verify event emission
    accepted_events = manager.get_events(session_id=session_id, event_type=EventType.RESULT_ACCEPTED)
    assert len(accepted_events) == 1
    assert accepted_events[0].version == 1
    assert accepted_events[0].payload["tool_name"] == "book_ride"
    assert accepted_events[0].payload["success"] is True

    completed_events = manager.get_events(session_id=session_id, event_type=EventType.TOOL_COMPLETED)
    assert len(completed_events) == 1
    assert completed_events[0].payload["output"]["ride_id"] == "ride_123"


# ---------------------------------------------------------------------------
# Test 2: Obsolete results from V1 are rejected when session has advanced to V2
# ---------------------------------------------------------------------------
def test_stale_result_rejected_when_version_advanced(manager: SessionManager):
    session_id = "sess_002"

    # Turn 1: User asks for weather
    session = manager.start_turn(session_id, transcript="What is the weather in Seattle?")
    assert session.current_version == 1

    # Turn 2: User supersedes before Turn 1 tool result arrives
    session = manager.start_turn(session_id, transcript="Never mind, tell me a joke")
    assert session.current_version == 2
    assert session.state == SessionState.THINKING

    # Now, delayed ToolResult arrives with old Version 1
    delayed_tool_result = ToolResult.ok(
        tool_name="get_weather",
        output={"location": "Seattle", "temp": 65},
    )
    stale_envelope = ToolResultEnvelope(
        session_id=session_id,
        version=1,  # Stale version!
        tool_result=delayed_tool_result,
    )

    accepted = manager.process_tool_result(stale_envelope)
    assert accepted is False
    # Session state should remain THINKING (not updated to COMPLETED by stale result)
    assert session.state == SessionState.THINKING

    # Verify RESULT_REJECTED_STALE event was emitted
    stale_events = manager.get_events(session_id=session_id, event_type=EventType.RESULT_REJECTED_STALE)
    assert len(stale_events) == 1
    assert stale_events[0].version == 1
    assert stale_events[0].payload["session_current_version"] == 2
    assert stale_events[0].payload["result_version"] == 1
    assert stale_events[0].payload["tool_name"] == "get_weather"

    # Turn 2 result arrives with matching Version 2
    turn2_result = ToolResult.ok(
        tool_name="tell_joke",
        output={"joke": "Why did the chicken cross the road?"},
    )
    valid_envelope = ToolResultEnvelope(
        session_id=session_id,
        version=2,
        tool_result=turn2_result,
    )
    accepted_v2 = manager.process_tool_result(valid_envelope)
    assert accepted_v2 is True
    assert session.state == SessionState.COMPLETED


# ---------------------------------------------------------------------------
# Test 3: Task cancellation triggers cleanly on superseding turns
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_task_cancellation_on_superseding_turn(manager: SessionManager):
    session_id = "sess_003"
    task_cancelled = False

    async def long_running_task():
        nonlocal task_cancelled
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            task_cancelled = True
            raise

    # Launch background task for Turn 1
    task = asyncio.create_task(long_running_task())
    session = manager.start_turn(
        session_id,
        transcript="Send an email to team",
        state=SessionState.TOOL_RUNNING,
        active_task=task,
    )
    assert session.current_version == 1
    assert session.active_task is task
    assert not task.done()

    # Yield control briefly to ensure the task has started
    await asyncio.sleep(0.01)

    # Start Turn 2 (supersedes Turn 1)
    manager.start_turn(session_id, transcript="Cancel that email!")

    # Verify Turn 1 task is cancelled
    assert task.cancelled() or task.cancelling()
    # Await cancellation to handle the CancelledError cleanly
    with pytest.raises(asyncio.CancelledError):
        await task

    assert task_cancelled is True
    assert session.current_version == 2


# ---------------------------------------------------------------------------
# Test 4: Interrupt marks state as INTERRUPTED and cancels active task
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_interrupt_cancels_active_task_and_updates_state(manager: SessionManager):
    session_id = "sess_004"
    cancelled = False

    async def background_operation():
        nonlocal cancelled
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled = True
            raise

    task = asyncio.create_task(background_operation())
    session = manager.start_turn(
        session_id,
        transcript="Initiate transfer",
        state=SessionState.TOOL_RUNNING,
        active_task=task,
    )

    await asyncio.sleep(0.01)

    # Interrupt call
    interrupted_session = manager.interrupt(session_id)
    assert interrupted_session.state == SessionState.INTERRUPTED
    assert task.cancelled() or task.cancelling()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancelled is True

    # Check interrupt event was emitted
    interrupt_events = manager.get_events(session_id=session_id, event_type=EventType.INTERRUPTED)
    assert len(interrupt_events) == 1
    assert interrupt_events[0].payload["state"] == "INTERRUPTED"


# ---------------------------------------------------------------------------
# Test 5: Reject result when session does not exist
# ---------------------------------------------------------------------------
def test_reject_result_for_unknown_session(manager: SessionManager):
    envelope = ToolResultEnvelope(
        session_id="ghost_session",
        version=1,
        tool_result=ToolResult.ok(tool_name="test_tool", output="data"),
    )
    accepted = manager.process_tool_result(envelope)
    assert accepted is False

    events = manager.get_events(session_id="ghost_session", event_type=EventType.RESULT_REJECTED_STALE)
    assert len(events) == 1
    assert events[0].payload["reason"] == "Session not found"
