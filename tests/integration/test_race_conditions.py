"""Integration Stress Testing and Race Condition Test Suite for VoxGuard.

Verifies result fencing, async task cancellation, rapid interruptions,
and delayed/uncancellable tool races against the SessionManager.
"""

import asyncio
import pytest

from app.models.events import EventType, ToolResultEnvelope
from app.models.state import SessionState
from app.models.tool import ToolResult
from app.services.session_manager import SessionManager
from app.tools.mock_tools import MockHotelSearchTool


@pytest.fixture
def manager():
    """Provide a fresh SessionManager instance for each test."""
    mgr = SessionManager()
    yield mgr
    mgr.clear()


# ---------------------------------------------------------------------------
# Test 1: The Core Stress Test (Fast Fractional Delays)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_core_stress_test_delayed_hotel_search_interrupted(manager: SessionManager):
    """The Core Stress Test:

    1. Start V1 calling hotel_search with simulated delay (delay=0.1s, max_price: 5000).
    2. Wait 0.02s, then trigger an interruption / new turn V2 (max_price: 3000).
    3. Let V1 complete (or attempt cancel). Ensure its result returns version 1.
    4. Process V1 result through the session manager and assert it is rejected/dropped
       as stale (RESULT_REJECTED_STALE).
    5. Complete V2 and assert its result is accepted (RESULT_ACCEPTED).
    6. Verify session state only reflects V2 data, not V1.
    """
    session_id = "sess_stress_001"
    tool = MockHotelSearchTool()

    v1_completed = asyncio.Event()
    v1_result_envelope = None

    async def v1_worker():
        nonlocal v1_result_envelope
        cur = asyncio.current_task()
        try:
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            if cur and hasattr(cur, "uncancel"):
                cur.uncancel()
        res = tool.run(destination="Downtown", max_price=5000)
        v1_result_envelope = ToolResultEnvelope(
            session_id=session_id,
            version=1,
            tool_result=res,
        )
        v1_completed.set()

    # Step 1: Start V1 turn
    task_v1 = asyncio.create_task(v1_worker())
    session_v1 = manager.start_turn(
        session_id=session_id,
        transcript="Find hotels with max price 5000",
        state=SessionState.TOOL_RUNNING,
        active_task=task_v1,
    )
    assert session_v1.current_version == 1

    # Step 2: Wait 0.02s, then trigger interruption / new turn V2 (max_price: 3000)
    await asyncio.sleep(0.02)

    session_v2 = manager.start_turn(
        session_id=session_id,
        transcript="Wait, find cheaper hotels with max price 3000 instead",
        state=SessionState.TOOL_RUNNING,
    )
    assert session_v2.current_version == 2

    # Step 3: Let V1 complete and ensure its result returns version 1
    await asyncio.wait_for(v1_completed.wait(), timeout=1.0)
    await task_v1
    assert v1_result_envelope is not None
    assert v1_result_envelope.version == 1
    assert v1_result_envelope.tool_result.output["max_price"] == 5000

    # Step 4: Process V1 result through the session manager -> assert rejected as stale
    v1_accepted = manager.process_tool_result(v1_result_envelope)
    assert v1_accepted is False

    stale_events = manager.get_events(session_id=session_id, event_type=EventType.RESULT_REJECTED_STALE)
    assert len(stale_events) >= 1
    assert stale_events[-1].version == 1
    assert stale_events[-1].payload["result_version"] == 1
    assert stale_events[-1].payload["session_current_version"] == 2

    # Step 5: Complete V2 (max_price: 3000) and assert its result is accepted
    res_v2 = tool.run(destination="Downtown", max_price=3000)
    v2_result_envelope = ToolResultEnvelope(
        session_id=session_id,
        version=2,
        tool_result=res_v2,
    )
    v2_accepted = manager.process_tool_result(v2_result_envelope)
    assert v2_accepted is True

    accepted_events = manager.get_events(session_id=session_id, event_type=EventType.RESULT_ACCEPTED)
    assert len(accepted_events) == 1
    assert accepted_events[0].version == 2
    assert accepted_events[0].payload["tool_name"] == "hotel_search"

    # Step 6: Verify session state only reflects V2 data, not V1
    current_session = manager.get_session(session_id)
    assert current_session.state == SessionState.COMPLETED
    assert current_session.committed_version == 2
    assert current_session.last_result.output["max_price"] == 3000
    assert current_session.committed_data["max_price"] == 3000
    assert current_session.committed_data["max_price"] != 5000


# ---------------------------------------------------------------------------
# Test 2: Rapid Interruptions (<100ms apart)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_rapid_interruptions(manager: SessionManager):
    """Rapid Interruptions:

    Send V1, V2, V3 in rapid succession (<100ms apart) and verify
    only V3 can commit to session state.
    """
    session_id = "sess_rapid_002"
    tool = MockHotelSearchTool()

    # V1 started
    task_v1 = asyncio.create_task(asyncio.sleep(0.2))
    manager.start_turn(session_id, "Turn 1: Find hotels", active_task=task_v1)
    assert manager.get_session(session_id).current_version == 1

    await asyncio.sleep(0.02)  # 20ms apart (<100ms)

    # V2 started
    task_v2 = asyncio.create_task(asyncio.sleep(0.2))
    manager.start_turn(session_id, "Turn 2: Change destination", active_task=task_v2)
    assert manager.get_session(session_id).current_version == 2
    assert task_v1.cancelled() or task_v1.cancelling()

    await asyncio.sleep(0.02)  # 20ms apart (<100ms)

    # V3 started
    task_v3 = asyncio.create_task(asyncio.sleep(0.2))
    manager.start_turn(session_id, "Turn 3: Final preference", active_task=task_v3)
    assert manager.get_session(session_id).current_version == 3
    assert task_v2.cancelled() or task_v2.cancelling()

    # Clean up background tasks
    for task in (task_v1, task_v2, task_v3):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Create results for all 3 turns
    env_v1 = ToolResultEnvelope(
        session_id=session_id,
        version=1,
        tool_result=tool.run(destination="City A", max_price=1000),
    )
    env_v2 = ToolResultEnvelope(
        session_id=session_id,
        version=2,
        tool_result=tool.run(destination="City B", max_price=2000),
    )
    env_v3 = ToolResultEnvelope(
        session_id=session_id,
        version=3,
        tool_result=tool.run(destination="City C", max_price=3000),
    )

    # V1 and V2 must be rejected
    assert manager.process_tool_result(env_v1) is False
    assert manager.process_tool_result(env_v2) is False

    # Only V3 must be accepted
    assert manager.process_tool_result(env_v3) is True

    # Check session state
    session = manager.get_session(session_id)
    assert session.state == SessionState.COMPLETED
    assert session.committed_version == 3
    assert session.committed_data["destination"] == "City C"
    assert session.committed_data["max_price"] == 3000

    # Ensure stale events logged for V1 and V2
    stale_events = manager.get_events(session_id=session_id, event_type=EventType.RESULT_REJECTED_STALE)
    stale_versions = [e.payload["result_version"] for e in stale_events]
    assert 1 in stale_versions
    assert 2 in stale_versions


# ---------------------------------------------------------------------------
# Test 3: Uncancellable / Delayed Tool Race
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_uncancellable_delayed_tool_race(manager: SessionManager):
    """Uncancellable/Delayed Tool Race:

    Force V1 to finish after V2 has already completed.
    Assert V1 still cannot overwrite V2's committed state.
    """
    session_id = "sess_uncancellable_003"
    tool = MockHotelSearchTool()

    v1_result_holder = []
    v1_finished = asyncio.Event()

    async def uncancellable_v1_worker():
        cur = asyncio.current_task()
        try:
            await asyncio.sleep(0.08)
        except asyncio.CancelledError:
            if cur and hasattr(cur, "uncancel"):
                cur.uncancel()

        res = tool.run(destination="Miami", max_price=9000)
        v1_result_holder.append(
            ToolResultEnvelope(session_id=session_id, version=1, tool_result=res)
        )
        v1_finished.set()

    task_v1 = asyncio.create_task(uncancellable_v1_worker())
    manager.start_turn(session_id, "Turn 1: Book Miami", active_task=task_v1)
    assert manager.get_session(session_id).current_version == 1

    # Yield briefly so task_v1 starts running
    await asyncio.sleep(0.01)

    # Turn 2 starts
    manager.start_turn(session_id, "Turn 2: Fast override to Denver")
    assert manager.get_session(session_id).current_version == 2

    # V2 completes FAST and commits to session state
    env_v2 = ToolResultEnvelope(
        session_id=session_id,
        version=2,
        tool_result=tool.run(destination="Denver", max_price=2500),
    )
    v2_accepted = manager.process_tool_result(env_v2)
    assert v2_accepted is True

    # Verify V2 is committed
    session = manager.get_session(session_id)
    assert session.state == SessionState.COMPLETED
    assert session.committed_version == 2
    assert session.committed_data["destination"] == "Denver"
    assert session.committed_data["max_price"] == 2500

    # Wait for the uncancellable V1 worker to complete
    await asyncio.wait_for(v1_finished.wait(), timeout=1.0)
    await task_v1
    assert len(v1_result_holder) == 1
    env_v1 = v1_result_holder[0]
    assert env_v1.version == 1

    # Attempt to process V1 AFTER V2 has already completed
    v1_accepted = manager.process_tool_result(env_v1)
    assert v1_accepted is False

    # Verify V1 CANNOT overwrite V2's committed state
    assert session.state == SessionState.COMPLETED
    assert session.committed_version == 2
    assert session.committed_data["destination"] == "Denver"
    assert session.committed_data["max_price"] == 2500
    assert session.last_result.output["destination"] == "Denver"
