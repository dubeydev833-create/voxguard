"""VoxGuard LLM Integration, Agent Controller, and Tool Layer Completeness Tests.

Verifies the 14 core requirements:
1. hotel_search intent extraction
2. restaurant_search intent extraction
3. flight_search intent extraction
4. LLM provider mock behavior (delay, override, mock responses)
5. Tool registry registration, lookup, and dispatch
6. Unknown / invalid tool handling (graceful validation before execution)
7. Tool failure simulation (TOOL_FAILED event, no corrupted session state)
8. LLM provider failure handling (safe ERROR transition, no crash)
9. Request identity preservation across entire lifecycle
10. Version propagation across entire lifecycle
11. Stale LLM/tool result protection (pre-dispatch and post-dispatch fencing)
12. Current result response generation and session completion
13. Cancellation separation from tool failure
14. Complete AgentController turn lifecycle end-to-end
"""

import asyncio
import pytest
from typing import Any, Dict, List

from app.llm.base import StructuredIntent
from app.llm.mock_provider import MockLLMProvider
from app.models.events import EventType, ToolResultEnvelope
from app.models.state import SessionState
from app.models.tool import RiskLevel, ToolResult
from app.services.agent_controller import AgentController
from app.services.session_manager import SessionManager
from app.tools.base import BaseTool
from app.tools.mock_tools import (
    MockFlightSearchTool,
    MockHotelSearchTool,
    MockRestaurantSearchTool,
    get_mock_tools,
)
from app.tools.registry import ToolRegistry, create_default_registry


@pytest.fixture
def fresh_runtime():
    """Provides a fresh isolated SessionManager, ToolRegistry, and AgentController."""
    mgr = SessionManager()
    reg = create_default_registry()
    llm = MockLLMProvider()
    ctrl = AgentController(manager=mgr, registry=reg, llm_provider=llm)
    yield mgr, reg, llm, ctrl
    mgr.clear()


# =========================================================================
# 1. Hotel search intent mapping
# =========================================================================
def test_intent_mapping_hotel_search(fresh_runtime):
    _, _, llm, ctrl = fresh_runtime

    # Standard query
    intent1 = llm._parse_regex("Find hotels in Delhi under 5000")
    assert intent1.intent == "hotel_search"
    assert intent1.arguments["destination"] == "Delhi"
    assert intent1.arguments["max_price"] == 5000.0

    # Follow-up / refinement query: "Actually make that under 3000"
    intent2 = llm._parse_regex("Actually make that under 3000")
    assert intent2.intent == "hotel_search"
    assert intent2.arguments["max_price"] == 3000.0

    # AgentController default parser consistency
    ctrl_parsed = ctrl.default_intent_parser("Find hotels in Delhi under 5000")
    assert ctrl_parsed is not None
    assert ctrl_parsed[0] == "hotel_search"
    assert ctrl_parsed[1]["max_price"] == 5000.0


# =========================================================================
# 2. Restaurant search intent mapping
# =========================================================================
def test_intent_mapping_restaurant_search(fresh_runtime):
    _, _, llm, ctrl = fresh_runtime

    intent = llm._parse_regex("Find restaurants in Delhi")
    assert intent.intent == "restaurant_search"
    assert intent.arguments["location"] == "Delhi"

    # With cuisine and price limit
    intent_custom = llm._parse_regex("Find Italian restaurants in Mumbai under 2000")
    assert intent_custom.intent == "restaurant_search"
    assert intent_custom.arguments["location"] == "Mumbai"
    assert intent_custom.arguments["cuisine"] == "Italian"
    assert intent_custom.arguments["max_price"] == 2000.0

    # AgentController default parser consistency
    ctrl_parsed = ctrl.default_intent_parser("Find restaurants in Delhi")
    assert ctrl_parsed is not None
    assert ctrl_parsed[0] == "restaurant_search"
    assert ctrl_parsed[1]["location"] == "Delhi"


# =========================================================================
# 3. Flight search intent mapping
# =========================================================================
def test_intent_mapping_flight_search(fresh_runtime):
    _, _, llm, ctrl = fresh_runtime

    intent = llm._parse_regex("Find a flight from Delhi to Mumbai")
    assert intent.intent == "flight_search"
    assert intent.arguments["origin"] == "Delhi"
    assert intent.arguments["destination"] == "Mumbai"

    # AgentController default parser consistency
    ctrl_parsed = ctrl.default_intent_parser("Find a flight from Delhi to Mumbai")
    assert ctrl_parsed is not None
    assert ctrl_parsed[0] == "flight_search"
    assert ctrl_parsed[1]["destination"] == "Mumbai"
    assert ctrl_parsed[1]["origin"] == "Delhi"


# =========================================================================
# 4. LLM provider mock behavior (delay, override, mock responses)
# =========================================================================
@pytest.mark.asyncio
async def test_llm_provider_mock_behavior():
    # Test delay and call logging
    llm = MockLLMProvider(delay=0.02)
    intent = await llm.parse_intent(
        transcript="Find hotels in Delhi under 4000",
        session_id="sess_mock_llm",
        request_id="req_001",
        version=1,
    )
    assert intent.intent == "hotel_search"
    assert len(llm.parse_calls) == 1
    assert llm.parse_calls[0]["request_id"] == "req_001"

    # Test override intent
    custom_intent = StructuredIntent(intent="book_ride", arguments={"pickup_location": "Airport", "dropoff_location": "Hotel"})
    llm_override = MockLLMProvider(override_intent=custom_intent)
    override_res = await llm_override.parse_intent(
        transcript="any random transcript",
        session_id="sess_mock_llm",
        request_id="req_002",
        version=2,
    )
    assert override_res.intent == "book_ride"
    assert override_res.arguments["pickup_location"] == "Airport"

    # Test mock response dictionary
    llm_responses = MockLLMProvider(mock_responses={"hotel_search": "Custom synthesized hotel response"})
    res = ToolResult.ok("hotel_search", output={"count": 1})
    synth = await llm_responses.synthesize_response(
        tool_name="hotel_search",
        tool_result=res,
        transcript="find hotel",
        session_id="sess_mock_llm",
        request_id="req_003",
        version=1,
    )
    assert synth == "Custom synthesized hotel response"


# =========================================================================
# 5. Tool registry registration, lookup, and dispatch
# =========================================================================
@pytest.mark.asyncio
async def test_tool_registry_registration_and_dispatch():
    reg = ToolRegistry()
    assert len(reg) == 0

    # Custom tool definition
    class EchoTool(BaseTool):
        name: str = "echo_tool"
        description: str = "Echo back parameters"
        risk_level: RiskLevel = RiskLevel.LOW
        requires_confirmation: bool = False
        parameters: Dict[str, Any] = {"type": "object", "properties": {"msg": {"type": "string"}}}

        def execute(self, **kwargs: Any) -> ToolResult:
            return ToolResult.ok(self.name, output={"echo": kwargs.get("msg")})

    echo = EchoTool()
    reg.register(echo)
    assert reg.has_tool("echo_tool") is True
    assert reg.get("echo_tool") is echo
    assert "echo_tool" in reg.list_names()

    # Sync and async execution via registry
    res_sync = reg.execute("echo_tool", msg="hello sync")
    assert res_sync.success is True
    assert res_sync.output["echo"] == "hello sync"

    res_async = await reg.execute_async("echo_tool", msg="hello async")
    assert res_async.success is True
    assert res_async.output["echo"] == "hello async"

    # Duplicate registration fails without overwrite=True
    with pytest.raises(ValueError):
        reg.register(EchoTool())


# =========================================================================
# 6. Unknown / invalid tool handling (graceful validation before execution)
# =========================================================================
def test_unknown_and_invalid_tool_validation(fresh_runtime):
    mgr, reg, _, ctrl = fresh_runtime

    # Unregistered tool
    err_unregistered = ctrl.validate_tool_request("non_existent_tool", {})
    assert err_unregistered is not None
    assert "not registered" in err_unregistered

    # Missing required argument for flight_search (requires destination)
    err_missing = ctrl.validate_tool_request("flight_search", {})
    assert err_missing is not None
    assert "destination" in err_missing

    # Invalid parameter type for hotel_search (max_price must be numeric)
    err_type = ctrl.validate_tool_request("hotel_search", {"max_price": "invalid_number_abc"})
    assert err_type is not None
    assert "numeric" in err_type

    # Negative amount for transfer_funds
    err_amt = ctrl.validate_tool_request("transfer_funds", {"recipient_account": "ACC_123", "amount": -50})
    assert err_amt is not None
    assert "strictly positive" in err_amt


# =========================================================================
# 7. Tool failure simulation (TOOL_FAILED event, no invalid data)
# =========================================================================
@pytest.mark.asyncio
async def test_tool_failure_simulation(fresh_runtime):
    mgr, _, _, ctrl = fresh_runtime
    session_id = "sess_tool_fail"

    # Turn with fail=True passed into arguments via override
    ctrl.llm_provider.override_intent = StructuredIntent(
        intent="hotel_search",
        arguments={"destination": "Delhi", "max_price": 5000, "fail": True},
    )

    session = ctrl.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Delhi under 5000",
        simulated_delay=0.01,
        request_id="req_fail_001",
    )
    if session.active_task:
        await session.active_task

    updated = mgr.get_session(session_id)
    # Session state should be ERROR when a tool fails
    assert updated.state == SessionState.ERROR
    # Committed data must not have false success data
    assert "hotels" not in updated.committed_data

    events = [e.event_type for e in mgr.get_events(session_id)]
    assert EventType.TOOL_FAILED in events
    assert EventType.TOOL_COMPLETED not in events


# =========================================================================
# 8. LLM provider failure handling (safe ERROR transition, no crash)
# =========================================================================
@pytest.mark.asyncio
async def test_llm_provider_failure_handling(fresh_runtime):
    mgr, _, _, ctrl = fresh_runtime
    session_id = "sess_llm_fail"

    # Case A: Synchronous LLM failure
    failing_llm = MockLLMProvider(fail=True)
    ctrl.llm_provider = failing_llm

    session = ctrl.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Delhi under 5000",
        request_id="req_llm_sync_fail",
    )
    assert session.state == SessionState.ERROR
    assert "error interpreting your request" in session.last_response

    # Case B: Asynchronous LLM failure
    async_failing_llm = MockLLMProvider(delay=0.01, fail=True)
    ctrl.llm_provider = async_failing_llm

    session_async = ctrl.handle_turn(
        session_id="sess_llm_async_fail",
        transcript="Find a flight from Delhi to Mumbai",
        request_id="req_llm_async_fail",
    )
    if session_async.active_task:
        await session_async.active_task

    updated_async = mgr.get_session("sess_llm_async_fail")
    assert updated_async.state == SessionState.ERROR
    assert "error interpreting your request" in updated_async.last_response


# =========================================================================
# 9. Request identity preservation end-to-end
# =========================================================================
@pytest.mark.asyncio
async def test_request_identity_preservation_end_to_end(fresh_runtime):
    mgr, _, llm, ctrl = fresh_runtime
    session_id = "sess_req_id_test"
    target_request_id = "req_custom_xyz_999"

    session = ctrl.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Delhi under 5000",
        simulated_delay=0.01,
        request_id=target_request_id,
    )
    assert session.current_request_id == target_request_id

    if session.active_task:
        await session.active_task

    updated = mgr.get_session(session_id)
    assert updated.committed_request_id == target_request_id
    assert target_request_id in updated.requests
    assert updated.requests[target_request_id].status == "completed"

    # Check request_id in recorded LLM calls
    assert llm.parse_calls[0]["request_id"] == target_request_id
    # Check request_id in all turn-related emitted events
    turn_events = [e for e in mgr.get_events(session_id) if e.event_type != EventType.SESSION_CREATED]
    assert len(turn_events) > 0
    for ev in turn_events:
        assert ev.request_id == target_request_id


# =========================================================================
# 10. Version propagation end-to-end
# =========================================================================
@pytest.mark.asyncio
async def test_version_propagation_end_to_end(fresh_runtime):
    mgr, _, llm, ctrl = fresh_runtime
    session_id = "sess_ver_prop_test"

    # Turn 1
    s1 = ctrl.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Delhi under 5000",
        simulated_delay=0.01,
        request_id="req_v1",
    )
    assert s1.current_version == 1
    if s1.active_task:
        await s1.active_task
    assert mgr.get_session(session_id).committed_version == 1

    # Turn 2
    s2 = ctrl.handle_turn(
        session_id=session_id,
        transcript="Find a flight from Delhi to Mumbai",
        simulated_delay=0.01,
        request_id="req_v2",
    )
    assert s2.current_version == 2
    if s2.active_task:
        await s2.active_task
    assert mgr.get_session(session_id).committed_version == 2


# =========================================================================
# 11. Stale LLM/tool result protection (pre-dispatch and post-dispatch)
# =========================================================================
@pytest.mark.asyncio
async def test_stale_result_protection_pre_and_post_dispatch(fresh_runtime):
    mgr, _, _, ctrl = fresh_runtime
    session_id = "sess_stale_fence_test"

    # --- Part A: Pre-Dispatch Fence ---
    # LLM has delay. Turn 1 starts LLM parsing. Turn 2 arrives before LLM parsing finishes.
    ctrl.llm_provider = MockLLMProvider(delay=0.05)
    s1 = ctrl.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Delhi under 5000",
        request_id="req_v1",
    )
    assert s1.current_version == 1

    # Turn 2 interrupts while Turn 1 is still parsing in LLM
    await asyncio.sleep(0.01)
    ctrl.llm_provider = MockLLMProvider(delay=0.0)  # Fast parser for Turn 2
    s2 = ctrl.handle_turn(
        session_id=session_id,
        transcript="Find restaurants in Delhi",
        simulated_delay=0.01,
        request_id="req_v2",
    )
    assert s2.current_version == 2

    # Await both tasks
    if s1.active_task:
        try:
            await s1.active_task
        except asyncio.CancelledError:
            pass
    if s2.active_task:
        await s2.active_task

    session_final = mgr.get_session(session_id)
    assert session_final.committed_version == 2
    assert session_final.committed_request_id == "req_v2"
    assert "restaurants" in session_final.last_response.lower()

    # --- Part B: Post-Dispatch Fence ---
    # Manual stale envelope injection for obsolete version
    stale_envelope = ToolResultEnvelope(
        session_id=session_id,
        version=1,
        request_id="req_v1",
        tool_result=ToolResult.ok("hotel_search", output={"hotels": ["Stale Hotel"]}),
    )
    accepted = mgr.process_tool_result(stale_envelope)
    assert accepted is False
    # Verify session state was not corrupted by stale hotel result
    assert "Stale Hotel" not in str(session_final.committed_data)


# =========================================================================
# 12. Current result response generation allowed
# =========================================================================
@pytest.mark.asyncio
async def test_current_result_response_generation(fresh_runtime):
    mgr, _, _, ctrl = fresh_runtime
    session_id = "sess_response_gen"

    session = ctrl.handle_turn(
        session_id=session_id,
        transcript="Find restaurants in Delhi",
        simulated_delay=0.01,
        request_id="req_rest_01",
    )
    if session.active_task:
        await session.active_task

    updated = mgr.get_session(session_id)
    assert updated.state == SessionState.COMPLETED
    assert updated.last_response is not None
    assert "restaurants in Delhi" in updated.last_response
    assert updated.committed_data.get("response") == updated.last_response

    event_types = [e.event_type for e in mgr.get_events(session_id)]
    assert EventType.RESPONSE_READY in event_types


# =========================================================================
# 13. Cancellation + async tool execution cooperation
# =========================================================================
@pytest.mark.asyncio
async def test_cancellation_distinct_from_tool_failure(fresh_runtime):
    mgr, _, _, ctrl = fresh_runtime
    session_id = "sess_cancel_test"

    # Start long-running tool execution
    session = ctrl.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Delhi under 5000",
        simulated_delay=0.2,
        request_id="req_to_be_cancelled",
    )
    assert session.active_task is not None

    # Immediate superseding turn triggers cancellation of the previous task
    await asyncio.sleep(0.01)
    new_session = ctrl.handle_turn(
        session_id=session_id,
        transcript="Actually make that under 3000",
        simulated_delay=0.01,
        request_id="req_superseding",
    )
    assert new_session.current_version == 2
    assert new_session.current_request_id == "req_superseding"

    if new_session.active_task:
        await new_session.active_task

    final_session = mgr.get_session(session_id)
    assert final_session.committed_version == 2
    assert final_session.committed_request_id == "req_superseding"
    assert final_session.state == SessionState.COMPLETED

    # Verify request status in session tracking - superseded request is stale/superseded
    assert final_session.requests["req_to_be_cancelled"].status in ("superseded", "stale", "cancelled")
    assert final_session.requests["req_superseding"].status == "completed"

    # Verify CANCELLATION_REQUESTED was emitted specifically for req_to_be_cancelled
    cancel_events = [
        e for e in mgr.get_events(session_id)
        if e.event_type == EventType.CANCELLATION_REQUESTED and e.request_id == "req_to_be_cancelled"
    ]
    assert len(cancel_events) >= 1


# =========================================================================
# 14. Complete AgentController turn lifecycle flow
# =========================================================================
@pytest.mark.asyncio
async def test_agent_controller_turn_lifecycle_flow(fresh_runtime):
    mgr, _, _, ctrl = fresh_runtime
    session_id = "sess_lifecycle_flow"

    # Step 1: User says "Find hotels in Delhi under 5000"
    session = ctrl.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Delhi under 5000",
        simulated_delay=0.01,
        request_id="req_lifecycle_01",
    )
    assert session.state == SessionState.TOOL_RUNNING
    assert session.current_version == 1

    if session.active_task:
        await session.active_task

    session_v1 = mgr.get_session(session_id)
    assert session_v1.state == SessionState.COMPLETED
    assert session_v1.committed_version == 1
    assert "hotels in Delhi under ₹5000" in session_v1.last_response

    # Step 2: User says "Find a flight from Delhi to Mumbai"
    session2 = ctrl.handle_turn(
        session_id=session_id,
        transcript="Find a flight from Delhi to Mumbai",
        simulated_delay=0.01,
        request_id="req_lifecycle_02",
    )
    assert session2.current_version == 2
    assert session2.state == SessionState.TOOL_RUNNING

    if session2.active_task:
        await session2.active_task

    session_v2 = mgr.get_session(session_id)
    assert session_v2.state == SessionState.COMPLETED
    assert session_v2.committed_version == 2
    assert "flights to Mumbai" in session_v2.last_response

    # Verify event sequencing: chronological progression
    event_seq = [e.event_type for e in mgr.get_events(session_id)]
    assert EventType.TURN_STARTED in event_seq
    assert EventType.TOOL_STARTED in event_seq
    assert EventType.TOOL_COMPLETED in event_seq
    assert EventType.RESPONSE_READY in event_seq
