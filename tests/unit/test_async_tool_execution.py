"""Unit tests for VoxGuard Phase 1: Async Tool Execution and Cancellation.

Tests verify:
- Normal async tool completion
- Distinct cancellation status (ToolResult.cancelled vs ToolResult.fail)
- Cancellation of a running tool on turn supersession
- Tool completion before cancellation
- Cancellation failure / late tool completion rejected by Result Fence
- Tool failure semantic distinction
- Configurable non-blocking delay across mock tools
"""

import asyncio
import time
import pytest

from app.models.events import EventType, ToolResultEnvelope
from app.models.state import SessionState
from app.models.tool import ToolResult
from app.services.agent_controller import AgentController
from app.services.session_manager import SessionManager
from app.tools.mock_tools import (
    MockFlightSearchTool,
    MockHotelSearchTool,
    MockRestaurantSearchTool,
)
from app.tools.registry import ToolRegistry


@pytest.fixture
def session_manager():
    mgr = SessionManager()
    yield mgr
    mgr.clear()


@pytest.fixture
def tool_registry():
    reg = ToolRegistry()
    reg.register(MockHotelSearchTool())
    reg.register(MockFlightSearchTool())
    reg.register(MockRestaurantSearchTool())
    return reg


@pytest.fixture
def controller(session_manager, tool_registry):
    return AgentController(manager=session_manager, registry=tool_registry)


# ---------------------------------------------------------------------------
# 1. Normal async tool completion
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_normal_async_tool_completion(controller: AgentController, session_manager: SessionManager):
    session_id = "sess_async_normal_001"
    session = controller.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Delhi under 3000",
        simulated_delay=0.05,
    )

    assert session.current_version == 1
    assert session.active_task is not None

    # Wait for the async tool worker to complete
    await session.active_task

    updated_session = session_manager.get_session(session_id)
    assert updated_session is not None
    assert updated_session.state == SessionState.COMPLETED
    assert updated_session.committed_version == 1
    assert updated_session.last_result is not None
    assert updated_session.last_result.success is True
    assert updated_session.last_result.status == "completed"
    assert updated_session.last_result.is_cancelled is False
    assert updated_session.committed_data["destination"] == "Delhi"
    assert updated_session.committed_data["max_price"] == 3000.0

    # Verify event emission
    events = session_manager.get_events(session_id=session_id)
    event_types = [e.event_type for e in events]
    assert EventType.TURN_STARTED in event_types
    assert EventType.TOOL_STARTED in event_types
    assert EventType.RESULT_ACCEPTED in event_types
    assert EventType.RESPONSE_READY in event_types


# ---------------------------------------------------------------------------
# 2. Successful cancellation semantics
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_successful_cancellation_semantics(tool_registry: ToolRegistry):
    hotel_tool = tool_registry.get_tool("hotel_search")

    task = asyncio.create_task(hotel_tool.execute(destination="Delhi", max_price=5000, delay=0.5))
    await asyncio.sleep(0.02)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    # Confirm ToolResult.cancelled provides distinct CANCELLED status, not generic failure
    cancelled_res = ToolResult.cancelled(
        tool_name="hotel_search",
        message="Execution was cancelled by a superseding turn.",
    )
    assert cancelled_res.status == "cancelled"
    assert cancelled_res.is_cancelled is True
    assert cancelled_res.success is False
    assert cancelled_res.status != "failed"
    assert "cancelled" in cancelled_res.error.lower()


# ---------------------------------------------------------------------------
# 3. Cancellation of a running tool on superseding turn
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cancellation_of_running_tool_on_superseding_turn(
    controller: AgentController,
    session_manager: SessionManager,
):
    session_id = "sess_cancel_running_002"

    # Start Turn 1 with 0.2s delay
    session_v1 = controller.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Delhi under 5000",
        simulated_delay=0.2,
    )
    task_v1 = session_v1.active_task
    assert task_v1 is not None

    # Wait briefly while V1 is mid-flight
    await asyncio.sleep(0.03)
    assert not task_v1.done()

    # Start Turn 2 (supersedes V1)
    session_v2 = controller.handle_turn(
        session_id=session_id,
        transcript="Actually make that under 3000",
        simulated_delay=0.01,
    )
    task_v2 = session_v2.active_task
    assert session_v2.current_version == 2

    # Await both tasks
    await asyncio.gather(task_v1, task_v2, return_exceptions=True)

    # Verify V1 task received cancellation and Result Fence rejected V1
    stale_events = session_manager.get_events(session_id=session_id, event_type=EventType.RESULT_REJECTED_STALE)
    assert len(stale_events) >= 1
    assert stale_events[0].payload["result_version"] == 1

    # Verify V2 accepted
    final_session = session_manager.get_session(session_id)
    assert final_session.state == SessionState.COMPLETED
    assert final_session.committed_version == 2
    assert final_session.committed_data["max_price"] == 3000.0


# ---------------------------------------------------------------------------
# 4. Tool completion before cancellation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_tool_completion_before_cancellation(
    controller: AgentController,
    session_manager: SessionManager,
):
    session_id = "sess_complete_before_cancel_003"

    # Fast tool execution
    session = controller.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Delhi under 3000",
        simulated_delay=0.01,
    )
    await session.active_task

    # Tool finished cleanly
    assert session.state == SessionState.COMPLETED
    assert session.committed_version == 1

    # Attempting to cancel an already completed task is a no-op
    was_cancelled = session.cancel_active_task()
    assert was_cancelled is False
    assert session.state == SessionState.COMPLETED
    assert session.committed_version == 1


# ---------------------------------------------------------------------------
# 5. Cancellation failure / late completion rejected by Result Fence
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cancellation_failure_late_completion_rejected_by_fence(
    session_manager: SessionManager,
    tool_registry: ToolRegistry,
):
    """Verifies Acceptance Criterion 8:

    Even if a tool cannot be cancelled (or completes late after cancellation),
    the Result Fence strictly drops the obsolete result and preserves V2 state.
    """
    session_id = "sess_late_completion_004"
    hotel_tool = tool_registry.get_tool("hotel_search")

    v1_envelope = None

    async def uncancellable_v1_worker():
        nonlocal v1_envelope
        cur = asyncio.current_task()
        try:
            await asyncio.sleep(0.12)
        except asyncio.CancelledError:
            # Tool ignores cancellation and continues running!
            if cur and hasattr(cur, "uncancel"):
                cur.uncancel()
            await asyncio.sleep(0.05)

        res = await hotel_tool.execute(destination="Delhi", max_price=9999)
        v1_envelope = ToolResultEnvelope(
            session_id=session_id,
            version=1,
            tool_result=res,
        )

    # Start V1
    task_v1 = asyncio.create_task(uncancellable_v1_worker())
    session_manager.start_turn(
        session_id=session_id,
        transcript="Search with max price 9999",
        state=SessionState.TOOL_RUNNING,
        active_task=task_v1,
    )

    # Wait 0.02s, then start V2
    await asyncio.sleep(0.02)
    session_manager.start_turn(
        session_id=session_id,
        transcript="Search with max price 2000",
        state=SessionState.TOOL_RUNNING,
    )

    # V2 completes first
    res_v2 = await hotel_tool.execute(destination="Delhi", max_price=2000)
    v2_envelope = ToolResultEnvelope(
        session_id=session_id,
        version=2,
        tool_result=res_v2,
    )
    v2_accepted = session_manager.process_tool_result(v2_envelope)
    assert v2_accepted is True
    session = session_manager.get_session(session_id)
    assert session.committed_version == 2
    assert session.committed_data["max_price"] == 2000.0

    # Wait for the uncancellable V1 to finally complete
    await task_v1
    assert v1_envelope is not None
    assert v1_envelope.version == 1

    # Attempt to process late V1 result
    v1_accepted = session_manager.process_tool_result(v1_envelope)
    assert v1_accepted is False  # Must be rejected by Result Fence!

    # Verify session state was NOT overwritten by late V1
    final_session = session_manager.get_session(session_id)
    assert final_session.committed_version == 2
    assert final_session.committed_data["max_price"] == 2000.0
    assert final_session.committed_data["max_price"] != 9999.0


# ---------------------------------------------------------------------------
# 6. Tool failure semantic distinction
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_tool_failure_semantic_distinction(tool_registry: ToolRegistry):
    """Verifies requirement 6: Tool failure is NOT converted to cancellation."""
    hotel_tool = tool_registry.get_tool("hotel_search")

    # Missing required argument raises error and produces failed ToolResult
    result = await hotel_tool.arun()
    assert result.success is False
    assert result.status == "failed"
    assert result.is_cancelled is False
    assert "Missing required parameter" in result.error

    # Explicit fail constructor
    failed_res = ToolResult.fail(tool_name="hotel_search", error="Upstream service 503")
    assert failed_res.status == "failed"
    assert failed_res.is_cancelled is False
    assert failed_res.success is False


# ---------------------------------------------------------------------------
# 7. Configurable non-blocking delay across mock tools
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_configurable_delay_across_mock_tools(tool_registry: ToolRegistry):
    """Verifies requirements 3 & 4: hotel_search, flight_search, restaurant_search

    support configurable non-blocking delay via asyncio.sleep.
    """
    hotel_tool = tool_registry.get_tool("hotel_search")
    flight_tool = tool_registry.get_tool("flight_search")
    restaurant_tool = tool_registry.get_tool("restaurant_search")

    # Hotel search with delay
    t0 = time.perf_counter()
    res_hotel = await hotel_tool.execute(destination="Delhi", max_price=5000, delay=0.06)
    elapsed_hotel = time.perf_counter() - t0
    assert res_hotel.success is True
    assert elapsed_hotel >= 0.05

    # Flight search with delay
    t0 = time.perf_counter()
    res_flight = await flight_tool.execute(destination="Mumbai", delay=0.06)
    elapsed_flight = time.perf_counter() - t0
    assert res_flight.success is True
    assert elapsed_flight >= 0.05

    # Restaurant search with delay
    t0 = time.perf_counter()
    res_rest = await restaurant_tool.execute(location="Delhi", delay=0.06)
    elapsed_rest = time.perf_counter() - t0
    assert res_rest.success is True
    assert elapsed_rest >= 0.05
