"""VoxGuard Phase 4 — Concurrency, Stress Testing, and Race Conditions Test Suite.

Verifies that the backend remains strictly correct when multiple asynchronous requests
overlap, cancel, fail, and complete in arbitrary orders.

Enforces the core correctness invariant:
ONLY THE CURRENT VERSION MAY MODIFY CURRENT CONVERSATION STATE OR TRIGGER RESPONSE GENERATION.
"""

import asyncio
from unittest.mock import MagicMock
import pytest

from app.models.events import EventType, ToolResultEnvelope
from app.models.state import SessionState
from app.models.tool import ToolResult
from app.services.agent_controller import AgentController
from app.services.session_manager import SessionManager
from app.tools.registry import ToolRegistry, create_default_registry


@pytest.fixture
def clean_manager() -> SessionManager:
    """Provide a pristine SessionManager for each test."""
    manager = SessionManager()
    yield manager
    manager.clear()


@pytest.fixture
def clean_registry() -> ToolRegistry:
    """Provide a fresh ToolRegistry with mock tools."""
    return create_default_registry()


@pytest.fixture
def controller(clean_manager: SessionManager, clean_registry: ToolRegistry) -> AgentController:
    """Provide an AgentController wired with clean manager and registry."""
    return AgentController(manager=clean_manager, registry=clean_registry)


# ---------------------------------------------------------------------------
# Test 1: Normal completion
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_normal_completion(controller: AgentController):
    """Start V1. Allow tool to complete normally.

    Verify result accepted, state updated, response generation proceeds.
    """
    session_id = "sess_p4_normal"
    synth_spy = MagicMock(side_effect=controller.synthesize_response)
    controller.synthesize_response = synth_spy

    session = controller.handle_turn(
        session_id=session_id,
        transcript="What is the weather in Boston?",
        simulated_delay=0.02,
        request_id="req_norm_v1",
    )
    assert session.current_version == 1
    assert session.current_request_id == "req_norm_v1"

    if session.active_task:
        await session.active_task

    # 1. Result accepted
    final_session = controller.session_manager.get_session(session_id)
    assert final_session.state == SessionState.COMPLETED
    assert final_session.committed_version == 1
    assert final_session.committed_request_id == "req_norm_v1"

    # 2. Conversation state updated correctly
    assert final_session.last_response is not None
    assert "Boston" in final_session.last_response

    # 3. Response generation proceeded and RESPONSE_READY emitted
    assert synth_spy.call_count == 1
    events = controller.session_manager.get_events(session_id=session_id, event_type=EventType.RESPONSE_READY)
    assert len(events) == 1
    assert events[0].request_id == "req_norm_v1"


# ---------------------------------------------------------------------------
# Test 2: Cancellation on superseding turn
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cancellation(controller: AgentController):
    """Start V1. Start V2 before V1 completes.

    Verify V1 becomes obsolete, cancellation requested for V1,
    V2 becomes current request, cancellation does not corrupt state.
    """
    session_id = "sess_p4_cancel"
    session = controller.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Delhi under 5000",
        simulated_delay=0.5,
        request_id="req_v1_cancel",
    )
    v1_task = session.active_task

    await asyncio.sleep(0.02)

    # Start V2 before V1 completes
    session_v2 = controller.handle_turn(
        session_id=session_id,
        transcript="Actually find hotels in Delhi under 2500",
        simulated_delay=0.02,
        request_id="req_v2_active",
    )

    # V2 is now current request and version
    assert session_v2.current_version == 2
    assert session_v2.current_request_id == "req_v2_active"
    assert session_v2.requests["req_v1_cancel"].status in ("superseded", "cancelled", "interrupted")

    # Verify cancellation requested for V1
    cancel_events = controller.session_manager.get_events(
        session_id=session_id,
        event_type=EventType.CANCELLATION_REQUESTED,
    )
    assert len(cancel_events) >= 1
    assert cancel_events[0].request_id == "req_v1_cancel"
    assert cancel_events[0].payload["superseded_version"] == 1

    # Ensure V1 task is cancelled
    assert v1_task.cancelled() or v1_task.cancelling() or v1_task.done()

    if session_v2.active_task:
        await session_v2.active_task

    # Session state remains uncorrupted and reflects V2
    assert session_v2.committed_version == 2
    assert session_v2.committed_request_id == "req_v2_active"


# ---------------------------------------------------------------------------
# Test 3: Late V1 result
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_late_v1_result(controller: AgentController):
    """Start V1 with a long delay. Start V2 afterward.

    Allow V2 to complete first. Then allow V1 to complete.
    Verify: V1 -> STALE -> REJECTED, V2 -> ACCEPTED.
    """
    session_id = "sess_p4_late"

    # Start V1
    controller.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Delhi under 5000",
        simulated_delay=0.18,
        request_id="req_v1_late",
    )

    await asyncio.sleep(0.02)

    # Start V2 with faster delay
    controller.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Delhi under 3000",
        simulated_delay=0.02,
        request_id="req_v2_fast",
    )

    # Wait for both to complete
    await asyncio.sleep(0.25)

    session = controller.session_manager.get_session(session_id)
    assert session.committed_version == 2
    assert session.committed_request_id == "req_v2_fast"
    assert session.committed_data["max_price"] == 3000

    # V1 rejected as stale
    stale_events = controller.session_manager.get_events(
        session_id=session_id,
        event_type=EventType.RESULT_REJECTED_STALE,
    )
    assert len(stale_events) >= 1
    assert stale_events[0].request_id == "req_v1_late"

    # V2 accepted
    accepted_events = controller.session_manager.get_events(
        session_id=session_id,
        event_type=EventType.RESULT_ACCEPTED,
    )
    assert len(accepted_events) == 1
    assert accepted_events[0].request_id == "req_v2_fast"


# ---------------------------------------------------------------------------
# Test 4: Cancellation failure
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cancellation_failure(controller: AgentController):
    """Simulate a tool that cannot be cancelled or ignores cancellation.

    Start V1. Start V2. Attempt cancellation of V1. Allow V1 to finish.
    Verify the Result Fence still rejects V1.
    """
    session_id = "sess_p4_uncancellable"
    v1_finished = asyncio.Event()

    # Custom uncancellable worker for V1
    async def uncancellable_worker():
        cur = asyncio.current_task()
        try:
            await asyncio.sleep(0.12)
        except asyncio.CancelledError:
            # Explicitly uncancel to simulate uncancellable external library
            if cur and hasattr(cur, "uncancel"):
                cur.uncancel()
            await asyncio.sleep(0.02)
        v1_finished.set()
        # Submit result directly through fencing
        envelope = ToolResultEnvelope(
            session_id=session_id,
            version=1,
            request_id="req_v1_uncancel",
            tool_result=ToolResult.ok(
                tool_name="hotel_search",
                output={"hotels": ["H1"], "max_price": 9999},
                request_id="req_v1_uncancel",
                version=1,
            ),
        )
        return controller.session_manager.process_tool_result(envelope)

    # Start V1
    task_v1 = asyncio.create_task(uncancellable_worker())
    controller.session_manager.start_turn(
        session_id=session_id,
        transcript="Hotels 9999",
        active_task=task_v1,
        request_id="req_v1_uncancel",
    )

    await asyncio.sleep(0.02)

    # Start V2 (triggers cancellation attempt on V1)
    controller.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Delhi under 3000",
        simulated_delay=0.02,
        request_id="req_v2_wins",
    )

    # Wait for V1 to finish
    await v1_finished.wait()
    v1_accepted = await task_v1
    assert v1_accepted is False

    # Wait for V2
    await asyncio.sleep(0.05)

    session = controller.session_manager.get_session(session_id)
    assert session.committed_version == 2
    assert session.committed_request_id == "req_v2_wins"
    assert session.committed_data["max_price"] == 3000
    assert session.committed_data["max_price"] != 9999


# ---------------------------------------------------------------------------
# Test 5: Three versions out of order
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_three_versions_out_of_order(clean_manager: SessionManager):
    """Run V1 -> V2 -> V3 with overlapping async tools.

    Complete them in order: V2, then V1, then V3.
    Verify only V3 can modify final current conversation state.
    """
    session_id = "sess_p4_three_versions"
    clean_manager.start_turn(session_id, "Turn 1", request_id="req_v1")
    clean_manager.start_turn(session_id, "Turn 2", request_id="req_v2")
    clean_manager.start_turn(session_id, "Turn 3", request_id="req_v3")

    session = clean_manager.get_session(session_id)
    assert session.current_version == 3
    assert session.current_request_id == "req_v3"

    # 1. V2 completes first -> Must be rejected (current is 3)
    env2 = ToolResultEnvelope(
        session_id=session_id,
        version=2,
        request_id="req_v2",
        tool_result=ToolResult.ok(tool_name="hotel_search", output={"max_price": 2000}),
    )
    assert clean_manager.process_tool_result(env2) is False

    # 2. V1 completes second -> Must be rejected
    env1 = ToolResultEnvelope(
        session_id=session_id,
        version=1,
        request_id="req_v1",
        tool_result=ToolResult.ok(tool_name="hotel_search", output={"max_price": 1000}),
    )
    assert clean_manager.process_tool_result(env1) is False

    # 3. V3 completes third -> Must be accepted!
    env3 = ToolResultEnvelope(
        session_id=session_id,
        version=3,
        request_id="req_v3",
        tool_result=ToolResult.ok(tool_name="hotel_search", output={"max_price": 3000}),
    )
    assert clean_manager.process_tool_result(env3) is True

    # Final state is strictly V3
    assert session.committed_version == 3
    assert session.committed_request_id == "req_v3"
    assert session.committed_data["max_price"] == 3000


# ---------------------------------------------------------------------------
# Test 6: Rapid successive changes (V1 to V5)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_rapid_successive_changes(clean_manager: SessionManager):
    """Create V1 -> V2 -> V3 -> V4 -> V5 quickly while tools are running.

    Allow results to complete in arbitrary order.
    Verify ONLY V5 can affect the current conversation.
    """
    session_id = "sess_p4_rapid_5"
    for i in range(1, 6):
        clean_manager.start_turn(session_id, f"Turn {i}", request_id=f"req_{i}")

    session = clean_manager.get_session(session_id)
    assert session.current_version == 5
    assert session.current_request_id == "req_5"

    # Arbitrary completion order: 3, 1, 4, 2, 5
    completion_order = [3, 1, 4, 2, 5]
    for v in completion_order:
        env = ToolResultEnvelope(
            session_id=session_id,
            version=v,
            request_id=f"req_{v}",
            tool_result=ToolResult.ok(tool_name="hotel_search", output={"v": v, "price": v * 1000}),
        )
        accepted = clean_manager.process_tool_result(env)
        if v == 5:
            assert accepted is True
        else:
            assert accepted is False

    assert session.committed_version == 5
    assert session.committed_request_id == "req_5"
    assert session.committed_data["price"] == 5000


# ---------------------------------------------------------------------------
# Test 7: Current tool failure
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_current_tool_failure(clean_manager: SessionManager):
    """Start a current request. Make its tool fail.

    Verify failure is represented correctly (TOOL_FAILED, state ERROR),
    session remains consistent, no invalid response is generated.
    """
    session_id = "sess_p4_tool_fail"
    session = clean_manager.start_turn(session_id, "Make transfer", request_id="req_fail_curr")

    failed_result = ToolResult.fail(
        tool_name="transfer_funds",
        error="Insufficient funds in source account",
        request_id="req_fail_curr",
        version=1,
    )
    envelope = ToolResultEnvelope(
        session_id=session_id,
        version=1,
        request_id="req_fail_curr",
        tool_result=failed_result,
    )

    clean_manager.process_tool_result(envelope)

    assert session.state == SessionState.ERROR
    assert session.requests["req_fail_curr"].status == "failed"
    assert "amount" not in session.committed_data

    fail_events = clean_manager.get_events(session_id=session_id, event_type=EventType.TOOL_FAILED)
    assert len(fail_events) == 1
    assert fail_events[0].payload["error"] == "Insufficient funds in source account"


# ---------------------------------------------------------------------------
# Test 8: Stale tool failure
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stale_tool_failure(clean_manager: SessionManager):
    """Start V1. Start V2. Make V1 fail after V2 becomes current.

    Verify V1's failure cannot overwrite V2's current state.
    """
    session_id = "sess_p4_stale_fail"
    clean_manager.start_turn(session_id, "Turn 1", request_id="req_v1")
    clean_manager.start_turn(session_id, "Turn 2", request_id="req_v2")

    session = clean_manager.get_session(session_id)
    assert session.current_version == 2
    assert session.current_request_id == "req_v2"
    assert session.state == SessionState.THINKING

    # V1 fails late
    stale_failure = ToolResult.fail(
        tool_name="book_ride",
        error="Database connection timeout",
        request_id="req_v1",
        version=1,
    )
    stale_envelope = ToolResultEnvelope(
        session_id=session_id,
        version=1,
        request_id="req_v1",
        tool_result=stale_failure,
    )

    accepted = clean_manager.process_tool_result(stale_envelope)
    assert accepted is False

    # State must NOT be set to ERROR; must remain THINKING for V2
    assert session.state == SessionState.THINKING
    assert session.current_version == 2
    assert session.current_request_id == "req_v2"

    # TOOL_FAILED must NOT be emitted for stale failure
    fail_events = clean_manager.get_events(session_id=session_id, event_type=EventType.TOOL_FAILED)
    assert len(fail_events) == 0

    # Instead, RESULT_REJECTED_STALE must be emitted
    stale_events = clean_manager.get_events(session_id=session_id, event_type=EventType.RESULT_REJECTED_STALE)
    assert len(stale_events) == 1
    assert stale_events[0].request_id == "req_v1"


# ---------------------------------------------------------------------------
# Test 9: Concurrent result processing
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_concurrent_result_processing(clean_manager: SessionManager):
    """Simulate multiple tool results arriving concurrently across threads/tasks.

    Verify the version check and state mutation remain strictly correct.
    """
    session_id = "sess_p4_concurrent"

    # Advance to version 4
    for i in range(1, 5):
        clean_manager.start_turn(session_id, f"Turn {i}", request_id=f"req_{i}")

    accepted_list = []

    async def submit_envelope(v: int):
        await asyncio.sleep(0.001)
        env = ToolResultEnvelope(
            session_id=session_id,
            version=v,
            request_id=f"req_{v}",
            tool_result=ToolResult.ok(tool_name="hotel_search", output={"version": v}),
        )
        is_acc = clean_manager.process_tool_result(env)
        if is_acc:
            accepted_list.append(v)

    # 12 concurrent workers racing with versions 1, 2, 3, 4
    tasks = [asyncio.create_task(submit_envelope(v)) for v in [1, 2, 3, 4, 3, 2, 1, 4, 2, 3, 4, 1]]
    await asyncio.gather(*tasks)

    # Only version 4 may ever be accepted
    assert set(accepted_list) == {4}
    session = clean_manager.get_session(session_id)
    assert session.committed_version == 4
    assert session.committed_request_id == "req_4"


# ---------------------------------------------------------------------------
# Test 10: Deterministic stress test
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_deterministic_stress_loop(controller: AgentController):
    """Multi-iteration deterministic stress test:

    Repeatedly performs:
    request -> async tool -> superseding request -> cancellation -> delayed completion.
    Runs 10 iterations cleanly and asserts zero state corruption.
    """
    session_id = "sess_p4_stress_loop"

    for iteration in range(1, 11):
        v1_id = f"req_iter_{iteration}_v1"
        v2_id = f"req_iter_{iteration}_v2"

        # Start V1
        controller.handle_turn(
            session_id=session_id,
            transcript="Find hotels in Delhi under 5000",
            simulated_delay=0.08,
            request_id=v1_id,
        )

        # Brief delay to allow V1 to register and start
        await asyncio.sleep(0.01)

        # Supersede with V2
        controller.handle_turn(
            session_id=session_id,
            transcript="Find hotels in Delhi under 3000",
            simulated_delay=0.01,
            request_id=v2_id,
        )

        # Allow both tasks to complete and fence decisions to apply
        await asyncio.sleep(0.1)

        session = controller.session_manager.get_session(session_id)
        # In each iteration, V2 must be the committed request
        assert session.committed_request_id == v2_id
        assert session.committed_data["max_price"] == 3000.0


# ---------------------------------------------------------------------------
# Test 11: Critical Acceptance Test Scenario
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_critical_acceptance_test(controller: AgentController):
    """Critical VoxGuard Acceptance Test:

    V1:
    request_id = A
    version = 1
    hotel_search delay (long)

    V2:
    request_id = B
    version = 2
    hotel_search delay (fast)

    Then:
    1. V1 starts.
    2. V2 supersedes V1.
    3. Cancellation of V1 is attempted.
    4. V2 completes.
    5. V1 completes afterward.
    6. V2 must be accepted.
    7. V1 must be rejected as stale.

    Verify:
    V1:
    - rejected
    - does not mutate conversation state
    - does not overwrite V2
    - does not trigger response generation
    - does not produce RESPONSE_READY
    - does not reach the Rime response path

    V2:
    - accepted
    - remains current
    - may trigger response generation
    - may produce RESPONSE_READY
    """
    session_id = "sess_critical_p4_exact"

    # Spy on synthesize_response to verify response generation function calls
    synth_spy = MagicMock(side_effect=controller.synthesize_response)
    controller.synthesize_response = synth_spy

    # Step 1: V1 starts with 0.18s delay
    controller.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Delhi under 5000",
        simulated_delay=0.18,
        request_id="req_A",
    )
    s = controller.session_manager.get_session(session_id)
    assert s.current_version == 1
    assert s.current_request_id == "req_A"

    # Step 2: V2 supersedes V1
    await asyncio.sleep(0.02)
    controller.handle_turn(
        session_id=session_id,
        transcript="Actually make that under 3000",
        simulated_delay=0.02,
        request_id="req_B",
    )
    assert s.current_version == 2
    assert s.current_request_id == "req_B"

    # Step 3: Cancellation of V1 was attempted
    cancel_events = controller.session_manager.get_events(
        session_id=session_id,
        event_type=EventType.CANCELLATION_REQUESTED,
    )
    assert len(cancel_events) >= 1
    assert cancel_events[0].request_id == "req_A"

    # Step 4: V2 completes first
    await asyncio.sleep(0.05)
    assert s.committed_version == 2
    assert s.committed_request_id == "req_B"
    assert s.committed_data["max_price"] == 3000.0

    # Step 5: Wait for V1 to finish afterward
    await asyncio.sleep(0.18)

    # Step 6 & 7 Verification:
    # V1 rejected
    stale_events = controller.session_manager.get_events(
        session_id=session_id,
        event_type=EventType.RESULT_REJECTED_STALE,
    )
    assert len(stale_events) >= 1
    assert stale_events[0].request_id == "req_A"
    assert stale_events[0].version == 1

    # V1 does NOT mutate conversation state or overwrite V2
    assert s.committed_version == 2
    assert s.committed_request_id == "req_B"
    assert s.committed_data["max_price"] == 3000.0
    assert s.committed_data["max_price"] != 5000.0

    # synthesize_response was called EXACTLY ONCE, ONLY for V2, NEVER for V1
    assert synth_spy.call_count == 1
    called_result: ToolResult = synth_spy.call_args[0][1]
    assert called_result.request_id == "req_B"
    assert called_result.version == 2

    # RESPONSE_READY produced ONLY for V2
    ready_events = controller.session_manager.get_events(
        session_id=session_id,
        event_type=EventType.RESPONSE_READY,
    )
    assert len(ready_events) == 1
    assert ready_events[0].request_id == "req_B"
    assert ready_events[0].version == 2
