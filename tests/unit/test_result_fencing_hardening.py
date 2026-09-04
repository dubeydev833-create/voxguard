"""Comprehensive Unit Tests for VoxGuard Result Fencing Hardening (Phase 3).

Guarantees the fundamental invariant:
ONLY THE CURRENT VERSION MAY MODIFY CURRENT CONVERSATION STATE OR TRIGGER RESPONSE GENERATION.

Tests cover:
1. Current result accepted.
2. Stale result rejected.
3. V1 finishes after V2.
4. V1 cancellation fails but V1 still finishes.
5. Stale result does not modify conversation state.
6. Stale result does not trigger response generation.
7. Stale result does not produce RESPONSE_READY.
8. Current V2 result is accepted.
9. V1 -> V2 -> V3 with results completing out of order.
10. Rapid successive versions.
11. Concurrent result processing under locking.
12. Critical acceptance test scenario.
"""

import asyncio
from unittest.mock import MagicMock
import pytest

from app.models.events import EventType, ToolResultEnvelope
from app.models.state import SessionState
from app.models.tool import ToolResult
from app.services.agent_controller import AgentController
from app.services.result_fence import FenceDecision, ResultFence
from app.services.session_manager import SessionManager
from app.tools.registry import ToolRegistry, create_default_registry


@pytest.fixture
def clean_manager() -> SessionManager:
    """Provide an isolated SessionManager for each test."""
    manager = SessionManager()
    yield manager
    manager.clear()


@pytest.fixture
def clean_registry() -> ToolRegistry:
    """Provide a fresh ToolRegistry with mock tools."""
    return create_default_registry()


@pytest.fixture
def controller(clean_manager: SessionManager, clean_registry: ToolRegistry) -> AgentController:
    """Provide an AgentController configured with clean manager and registry."""
    return AgentController(manager=clean_manager, registry=clean_registry)


# ---------------------------------------------------------------------------
# Test 1: Current result accepted
# ---------------------------------------------------------------------------
def test_current_result_accepted(clean_manager: SessionManager):
    session = clean_manager.start_turn("sess_1", "Turn 1", request_id="req_A")
    assert session.current_version == 1

    envelope = ToolResultEnvelope(
        session_id="sess_1",
        version=1,
        request_id="req_A",
        tool_result=ToolResult.ok(tool_name="hotel_search", output={"max_price": 5000}),
    )

    accepted = clean_manager.process_tool_result(envelope)
    assert accepted is True
    assert session.state == SessionState.COMPLETED
    assert session.committed_version == 1
    assert session.committed_request_id == "req_A"
    assert session.committed_data["max_price"] == 5000


# ---------------------------------------------------------------------------
# Test 2: Stale result rejected
# ---------------------------------------------------------------------------
def test_stale_result_rejected(clean_manager: SessionManager):
    clean_manager.start_turn("sess_2", "Turn 1", request_id="req_A")
    clean_manager.start_turn("sess_2", "Turn 2", request_id="req_B")

    session = clean_manager.get_session("sess_2")
    assert session.current_version == 2
    assert session.current_request_id == "req_B"

    # Stale V1 result arrives
    stale_envelope = ToolResultEnvelope(
        session_id="sess_2",
        version=1,
        request_id="req_A",
        tool_result=ToolResult.ok(tool_name="hotel_search", output={"max_price": 5000}),
    )

    accepted = clean_manager.process_tool_result(stale_envelope)
    assert accepted is False
    assert session.current_version == 2
    assert session.committed_version is None


# ---------------------------------------------------------------------------
# Test 3: V1 finishes after V2
# ---------------------------------------------------------------------------
def test_v1_finishes_after_v2(clean_manager: SessionManager):
    session_id = "sess_3"
    clean_manager.start_turn(session_id, "Turn 1", request_id="req_A")
    clean_manager.start_turn(session_id, "Turn 2", request_id="req_B")

    # V2 arrives first
    v2_envelope = ToolResultEnvelope(
        session_id=session_id,
        version=2,
        request_id="req_B",
        tool_result=ToolResult.ok(tool_name="hotel_search", output={"max_price": 3000}),
    )
    assert clean_manager.process_tool_result(v2_envelope) is True

    session = clean_manager.get_session(session_id)
    assert session.committed_version == 2
    assert session.committed_request_id == "req_B"
    assert session.committed_data["max_price"] == 3000

    # V1 arrives after V2
    v1_envelope = ToolResultEnvelope(
        session_id=session_id,
        version=1,
        request_id="req_A",
        tool_result=ToolResult.ok(tool_name="hotel_search", output={"max_price": 5000}),
    )
    assert clean_manager.process_tool_result(v1_envelope) is False

    # Session committed state remains V2, V1 data is never committed
    assert session.committed_version == 2
    assert session.committed_request_id == "req_B"
    assert session.committed_data["max_price"] == 3000


# ---------------------------------------------------------------------------
# Test 4: V1 cancellation fails but V1 still finishes
# ---------------------------------------------------------------------------
def test_v1_cancellation_fails_but_v1_still_rejected_by_fence(clean_manager: SessionManager):
    session_id = "sess_4"
    clean_manager.start_turn(session_id, "Turn 1", request_id="req_A")
    clean_manager.start_turn(session_id, "Turn 2", request_id="req_B")

    # Tool ignored cancellation and finished with success
    uncancelled_v1_result = ToolResult.ok(
        tool_name="hotel_search",
        output={"hotels": ["Grand Palace"], "max_price": 5000},
        request_id="req_A",
        version=1,
    )
    envelope = ToolResultEnvelope(
        session_id=session_id,
        version=1,
        request_id="req_A",
        tool_result=uncancelled_v1_result,
    )

    decision = ResultFence.evaluate(envelope, clean_manager.get_session(session_id))
    assert decision.decision == FenceDecision.STALE
    assert clean_manager.process_tool_result(envelope) is False


# ---------------------------------------------------------------------------
# Test 5: Stale result does not modify conversation state
# ---------------------------------------------------------------------------
def test_stale_result_does_not_modify_conversation_state(clean_manager: SessionManager):
    session_id = "sess_5"
    clean_manager.start_turn(session_id, "Turn 1", request_id="req_A")
    clean_manager.start_turn(session_id, "Turn 2", request_id="req_B")

    session = clean_manager.get_session(session_id)
    initial_updated_at = session.updated_at
    initial_transcript = session.last_transcript

    stale_envelope = ToolResultEnvelope(
        session_id=session_id,
        version=1,
        request_id="req_A",
        tool_result=ToolResult.ok(
            tool_name="transfer_funds",
            output={"amount": 999999, "status": "stolen"},
        ),
    )

    accepted = clean_manager.process_tool_result(stale_envelope)
    assert accepted is False

    # Confirm complete state isolation
    assert session.current_version == 2
    assert session.current_request_id == "req_B"
    assert session.committed_version is None
    assert session.committed_request_id is None
    assert "amount" not in session.committed_data
    assert session.last_transcript == initial_transcript
    assert session.last_result is None


# ---------------------------------------------------------------------------
# Test 6: Stale result does not trigger response generation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stale_result_does_not_trigger_response_generation(controller: AgentController):
    session_id = "sess_no_synth"

    original_synth = controller.synthesize_response
    synth_spy = MagicMock(side_effect=original_synth)
    controller.synthesize_response = synth_spy

    controller.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Delhi under 5000",
        simulated_delay=0.15,
        request_id="req_v1",
    )
    await asyncio.sleep(0.01)
    controller.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Delhi under 3000",
        simulated_delay=0.01,
        request_id="req_v2",
    )

    await asyncio.sleep(0.2)

    # synthesize_response called exactly once for V2, NEVER for V1
    assert synth_spy.call_count == 1
    call_args = synth_spy.call_args_list[0]
    result_arg: ToolResult = call_args[0][1]
    assert result_arg.request_id == "req_v2"
    assert result_arg.version == 2


# ---------------------------------------------------------------------------
# Test 7: Stale result does not produce RESPONSE_READY
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stale_result_does_not_produce_response_ready(controller: AgentController):
    session_id = "sess_no_ready"

    controller.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Delhi under 5000",
        simulated_delay=0.15,
        request_id="req_v1",
    )
    await asyncio.sleep(0.01)
    controller.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Delhi under 3000",
        simulated_delay=0.01,
        request_id="req_v2",
    )

    await asyncio.sleep(0.2)

    ready_events = controller.session_manager.get_events(
        session_id=session_id,
        event_type=EventType.RESPONSE_READY,
    )
    # Only 1 RESPONSE_READY event, belonging to V2
    assert len(ready_events) == 1
    assert ready_events[0].request_id == "req_v2"
    assert ready_events[0].version == 2


# ---------------------------------------------------------------------------
# Test 8: Current V2 result is accepted
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_current_v2_result_accepted(controller: AgentController):
    session_id = "sess_v2_accept"
    controller.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Delhi under 5000",
        simulated_delay=0.1,
        request_id="req_v1",
    )
    controller.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Delhi under 3000",
        simulated_delay=0.01,
        request_id="req_v2",
    )

    await asyncio.sleep(0.05)
    session = controller.session_manager.get_session(session_id)
    assert session.committed_version == 2
    assert session.committed_request_id == "req_v2"
    assert session.state == SessionState.COMPLETED


# ---------------------------------------------------------------------------
# Test 9: V1 -> V2 -> V3 with results completing out of order
# ---------------------------------------------------------------------------
def test_v1_v2_v3_out_of_order_completion(clean_manager: SessionManager):
    session_id = "sess_out_of_order"
    clean_manager.start_turn(session_id, "Turn 1", request_id="req_v1")
    clean_manager.start_turn(session_id, "Turn 2", request_id="req_v2")
    clean_manager.start_turn(session_id, "Turn 3", request_id="req_v3")

    session = clean_manager.get_session(session_id)
    assert session.current_version == 3

    # Arrival 1: V1 completes -> Stale
    env1 = ToolResultEnvelope(
        session_id=session_id,
        version=1,
        request_id="req_v1",
        tool_result=ToolResult.ok(tool_name="hotel_search", output={"max_price": 5000}),
    )
    assert clean_manager.process_tool_result(env1) is False

    # Arrival 2: V3 completes -> Accepted
    env3 = ToolResultEnvelope(
        session_id=session_id,
        version=3,
        request_id="req_v3",
        tool_result=ToolResult.ok(tool_name="hotel_search", output={"max_price": 2000}),
    )
    assert clean_manager.process_tool_result(env3) is True
    assert session.committed_version == 3
    assert session.committed_request_id == "req_v3"
    assert session.committed_data["max_price"] == 2000

    # Arrival 3: V2 completes late -> Stale
    env2 = ToolResultEnvelope(
        session_id=session_id,
        version=2,
        request_id="req_v2",
        tool_result=ToolResult.ok(tool_name="hotel_search", output={"max_price": 3000}),
    )
    assert clean_manager.process_tool_result(env2) is False

    # State still commits V3
    assert session.committed_version == 3
    assert session.committed_request_id == "req_v3"
    assert session.committed_data["max_price"] == 2000


# ---------------------------------------------------------------------------
# Test 10: Rapid successive versions
# ---------------------------------------------------------------------------
def test_rapid_successive_versions(clean_manager: SessionManager):
    session_id = "sess_rapid"
    for i in range(1, 11):
        clean_manager.start_turn(session_id, f"Turn {i}", request_id=f"req_{i}")

    session = clean_manager.get_session(session_id)
    assert session.current_version == 10
    assert session.current_request_id == "req_10"

    # Submit results for versions 1 to 9 -> all must be rejected
    for i in range(1, 10):
        env = ToolResultEnvelope(
            session_id=session_id,
            version=i,
            request_id=f"req_{i}",
            tool_result=ToolResult.ok(tool_name="hotel_search", output={"v": i}),
        )
        assert clean_manager.process_tool_result(env) is False
        assert session.committed_version is None

    # Submit version 10 -> accepted
    env10 = ToolResultEnvelope(
        session_id=session_id,
        version=10,
        request_id="req_10",
        tool_result=ToolResult.ok(tool_name="hotel_search", output={"v": 10}),
    )
    assert clean_manager.process_tool_result(env10) is True
    assert session.committed_version == 10
    assert session.committed_request_id == "req_10"
    assert session.committed_data["v"] == 10


# ---------------------------------------------------------------------------
# Test 11: Concurrent result processing under session lock
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_concurrent_result_processing_under_lock(clean_manager: SessionManager):
    session_id = "sess_concurrent"

    # Start session at version 5
    for i in range(1, 6):
        clean_manager.start_turn(session_id, f"Turn {i}", request_id=f"req_{i}")

    results = []

    async def submit_result(v: int):
        await asyncio.sleep(0.001)
        env = ToolResultEnvelope(
            session_id=session_id,
            version=v,
            request_id=f"req_{v}",
            tool_result=ToolResult.ok(tool_name="hotel_search", output={"version": v}),
        )
        acc = clean_manager.process_tool_result(env)
        results.append((v, acc))

    # Run tasks concurrently in parallel
    tasks = [asyncio.create_task(submit_result(v)) for v in [1, 2, 3, 4, 5, 2, 1, 4]]
    await asyncio.gather(*tasks)

    # Exactly one result (v=5) must be accepted
    accepted_versions = [v for v, acc in results if acc]
    assert accepted_versions == [5]

    session = clean_manager.get_session(session_id)
    assert session.committed_version == 5
    assert session.committed_request_id == "req_5"


# ---------------------------------------------------------------------------
# Test 12: Critical Acceptance Scenario
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_critical_acceptance_scenario(controller: AgentController):
    """Critical Acceptance Scenario:

    V1 starts: request_id=A, version=1, simulated delay
    20ms later: V2 starts: request_id=B, version=2
    V1 cancellation attempted
    V2 completes first
    Then V1 completes
    Expected:
    V2 result -> accepted, current conversation remains V2, response generation allowed
    V1 result -> version mismatch, stale rejection, NO state mutation, NO response generation, NO RESPONSE_READY
    """
    session_id = "sess_critical_acceptance"

    # Spy on synthesize_response
    synth_spy = MagicMock(side_effect=controller.synthesize_response)
    controller.synthesize_response = synth_spy

    # Step 1: V1 starts with 0.15s delay
    controller.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Delhi under 5000",
        simulated_delay=0.15,
        request_id="req_A",
    )
    s = controller.session_manager.get_session(session_id)
    assert s.current_version == 1
    assert s.current_request_id == "req_A"

    # Step 2: V2 starts shortly after
    await asyncio.sleep(0.02)
    controller.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Delhi under 3000",
        simulated_delay=0.01,
        request_id="req_B",
    )
    assert s.current_version == 2
    assert s.current_request_id == "req_B"

    # Step 3: V2 completes first
    await asyncio.sleep(0.05)
    assert s.committed_version == 2
    assert s.committed_request_id == "req_B"
    assert s.committed_data["max_price"] == 3000

    # Step 4: Wait for V1 to complete late and hit fence
    await asyncio.sleep(0.15)

    # V1 rejected
    stale_events = controller.session_manager.get_events(
        session_id=session_id,
        event_type=EventType.RESULT_REJECTED_STALE,
    )
    assert len(stale_events) >= 1
    assert stale_events[0].request_id == "req_A"
    assert stale_events[0].version == 1

    # State still strictly committed to V2
    assert s.committed_version == 2
    assert s.committed_request_id == "req_B"
    assert s.committed_data["max_price"] == 3000
    assert s.committed_data["max_price"] != 5000

    # synthesize_response was called ONLY for V2, NEVER for V1
    assert synth_spy.call_count == 1
    assert synth_spy.call_args[0][1].request_id == "req_B"

    # RESPONSE_READY emitted ONLY for V2
    ready_events = controller.session_manager.get_events(
        session_id=session_id,
        event_type=EventType.RESPONSE_READY,
    )
    assert len(ready_events) == 1
    assert ready_events[0].request_id == "req_B"
    assert ready_events[0].version == 2
