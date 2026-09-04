"""Unit tests for VoxGuard Phase 5 — LLM Integration.

Verifies:
1. Natural-language input produces valid structured intent.
2. Valid intent dispatches correct registered tool.
3. Invalid tool name is rejected without execution.
4. Invalid tool arguments are rejected before execution.
5. LLM processing retains request_id.
6. LLM processing retains version.
7. Obsolete LLM operation cannot dispatch a tool.
8. Current LLM operation can dispatch a tool.
9. Current tool result reaches response generation.
10. Stale tool result does NOT reach response generation.
11. V1 -> V2 interruption during LLM processing remains safe.
12. Crucial Acceptance Test: V1 interrupted during LLM processing, tool never dispatched,
    V2 completes and commits.
"""

import asyncio
from unittest.mock import MagicMock
import pytest

from app.llm.base import StructuredIntent
from app.llm.mock_provider import MockLLMProvider
from app.llm.provider import get_llm_provider
from app.models.events import EventType
from app.models.state import SessionState
from app.models.tool import ToolResult
from app.services.agent_controller import AgentController
from app.services.session_manager import SessionManager
from app.tools.registry import create_default_registry


@pytest.fixture
def session_manager():
    mgr = SessionManager()
    yield mgr
    mgr.clear()


@pytest.fixture
def tool_registry():
    return create_default_registry()


@pytest.fixture
def mock_llm():
    return MockLLMProvider()


@pytest.fixture
def controller(session_manager, tool_registry, mock_llm):
    return AgentController(
        manager=session_manager,
        registry=tool_registry,
        llm_provider=mock_llm,
    )


# ---------------------------------------------------------------------------
# Test 1: Natural-language input produces valid structured intent
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_nl_input_produces_valid_structured_intent(mock_llm: MockLLMProvider):
    # Hotel search
    intent1 = await mock_llm.parse_intent(
        transcript="Find hotels in Paris under 2500",
        session_id="s1",
        request_id="req_1",
        version=1,
    )
    assert intent1.intent == "hotel_search"
    assert intent1.arguments["destination"] == "Paris"
    assert intent1.arguments["max_price"] == 2500.0

    # Book ride
    intent2 = await mock_llm.parse_intent(
        transcript="Book a ride from Airport to Downtown",
        session_id="s1",
        request_id="req_2",
        version=2,
    )
    assert intent2.intent == "book_ride"
    assert "Airport" in intent2.arguments["pickup_location"]
    assert "Downtown" in intent2.arguments["dropoff_location"]

    # Weather
    intent3 = await mock_llm.parse_intent(
        transcript="What is the weather in Rome?",
        session_id="s1",
        request_id="req_3",
        version=3,
    )
    assert intent3.intent == "get_weather"
    assert intent3.arguments["location"] == "Rome"

    # Conversational / direct
    intent4 = await mock_llm.parse_intent(
        transcript="Hello, tell me what you can do.",
        session_id="s1",
        request_id="req_4",
        version=4,
    )
    assert intent4.intent is None
    assert intent4.direct_response is not None
    assert "Hello" in intent4.direct_response


# ---------------------------------------------------------------------------
# Test 2: Valid intent dispatches correct registered tool
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_valid_intent_dispatches_correct_tool(controller: AgentController):
    session_id = "sess_dispatch_valid"
    session = controller.handle_turn(
        session_id=session_id,
        transcript="What is the weather in Chicago?",
        simulated_delay=0.01,
        request_id="req_weather_valid",
    )

    assert session.current_version == 1
    assert session.state == SessionState.TOOL_RUNNING

    if session.active_task:
        await session.active_task

    updated_session = controller.session_manager.get_session(session_id)
    assert updated_session.state == SessionState.COMPLETED
    assert updated_session.committed_version == 1
    assert updated_session.committed_request_id == "req_weather_valid"
    assert "Chicago" in updated_session.last_response
    assert "weather" in updated_session.last_response.lower()


# ---------------------------------------------------------------------------
# Test 3: Invalid tool name is rejected without execution
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_invalid_tool_name_rejected_without_execution(
    session_manager: SessionManager, tool_registry
):
    bad_llm = MockLLMProvider(
        override_intent=StructuredIntent(
            intent="non_existent_tool_xyz",
            arguments={"param1": "val1"},
        )
    )
    ctrl = AgentController(
        manager=session_manager,
        registry=tool_registry,
        llm_provider=bad_llm,
    )

    execute_spy = MagicMock(side_effect=tool_registry.execute_async)
    tool_registry.execute_async = execute_spy

    session_id = "sess_invalid_tool"
    session = ctrl.handle_turn(
        session_id=session_id,
        transcript="Perform unknown operation",
        request_id="req_invalid_tool",
    )

    # Tool execution was NEVER dispatched
    execute_spy.assert_not_called()

    # Session remains safe, request recorded with failure
    assert session.state in [SessionState.ERROR, SessionState.THINKING]
    assert "req_invalid_tool" in session.requests
    assert session.requests["req_invalid_tool"].status == "failed"


# ---------------------------------------------------------------------------
# Test 4: Invalid tool arguments are rejected before execution
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_invalid_tool_arguments_rejected_before_execution(
    session_manager: SessionManager, tool_registry
):
    # send_email requires recipient, subject, and body
    missing_args_llm = MockLLMProvider(
        override_intent=StructuredIntent(
            intent="send_email",
            arguments={"recipient": "user@example.com"},  # missing subject and body
        )
    )
    ctrl = AgentController(
        manager=session_manager,
        registry=tool_registry,
        llm_provider=missing_args_llm,
    )

    execute_spy = MagicMock(side_effect=tool_registry.execute_async)
    tool_registry.execute_async = execute_spy

    session_id = "sess_missing_args"
    session = ctrl.handle_turn(
        session_id=session_id,
        transcript="Send an email to user@example.com",
        request_id="req_missing_args",
    )

    # Tool execution was NEVER dispatched due to pre-validation
    execute_spy.assert_not_called()
    assert session.requests["req_missing_args"].status == "failed"


# ---------------------------------------------------------------------------
# Test 5: LLM processing retains request_id
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_llm_processing_retains_request_id(controller: AgentController, mock_llm: MockLLMProvider):
    session_id = "sess_req_id_retention"
    custom_req_id = "custom_req_llm_99"

    session = controller.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Delhi under 3000",
        simulated_delay=0.01,
        request_id=custom_req_id,
    )

    if session.active_task:
        await session.active_task

    # Verify MockLLM recorded the exact request_id
    assert len(mock_llm.parse_calls) > 0
    assert mock_llm.parse_calls[0]["request_id"] == custom_req_id

    # Verify session committed the exact request_id
    updated_session = controller.session_manager.get_session(session_id)
    assert updated_session.committed_request_id == custom_req_id
    assert updated_session.requests[custom_req_id].status == "completed"


# ---------------------------------------------------------------------------
# Test 6: LLM processing retains version
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_llm_processing_retains_version(controller: AgentController, mock_llm: MockLLMProvider):
    session_id = "sess_version_retention"

    # Turn 1
    session1 = controller.handle_turn(
        session_id=session_id,
        transcript="What is the weather in Seattle?",
        simulated_delay=0.01,
        request_id="req_v1",
    )
    if session1.active_task:
        await session1.active_task

    # Turn 2
    session2 = controller.handle_turn(
        session_id=session_id,
        transcript="What is the weather in Chicago?",
        simulated_delay=0.01,
        request_id="req_v2",
    )
    if session2.active_task:
        await session2.active_task

    assert mock_llm.parse_calls[0]["version"] == 1
    assert mock_llm.parse_calls[1]["version"] == 2
    assert session2.committed_version == 2


# ---------------------------------------------------------------------------
# Test 7: Obsolete LLM operation cannot dispatch a tool
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_obsolete_llm_operation_cannot_dispatch_tool(
    session_manager: SessionManager, tool_registry
):
    # LLM simulates latency (0.1s)
    delayed_llm = MockLLMProvider(delay=0.1)
    ctrl = AgentController(
        manager=session_manager,
        registry=tool_registry,
        llm_provider=delayed_llm,
    )

    execute_spy = MagicMock(side_effect=tool_registry.execute_async)
    tool_registry.execute_async = execute_spy

    session_id = "sess_obsolete_llm"
    # V1 starts with delayed LLM parsing
    session_v1 = ctrl.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Paris with price under 5000",
        request_id="req_v1",
    )
    assert session_v1.current_version == 1
    assert session_v1.state == SessionState.THINKING

    # While V1 is in-flight parsing LLM, user interrupts by starting V2
    await asyncio.sleep(0.02)
    session_v2 = ctrl.handle_turn(
        session_id=session_id,
        transcript="Actually, hello there!",
        request_id="req_v2",
    )
    assert session_v2.current_version == 2

    # Wait for V1's delayed LLM completion
    await asyncio.sleep(0.15)

    # Verify: V1's tool execution was NEVER dispatched
    execute_spy.assert_not_called()


# ---------------------------------------------------------------------------
# Test 8: Current LLM operation can dispatch a tool
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_current_llm_operation_dispatches_tool(
    session_manager: SessionManager, tool_registry
):
    delayed_llm = MockLLMProvider(delay=0.02)
    ctrl = AgentController(
        manager=session_manager,
        registry=tool_registry,
        llm_provider=delayed_llm,
    )

    session_id = "sess_current_dispatches"
    session = ctrl.handle_turn(
        session_id=session_id,
        transcript="What is the weather in Seattle?",
        simulated_delay=0.01,
        request_id="req_current",
    )

    assert session.state == SessionState.THINKING

    # Allow LLM and tool execution to finish
    await asyncio.sleep(0.08)

    updated_session = session_manager.get_session(session_id)
    assert updated_session.state == SessionState.COMPLETED
    assert updated_session.committed_version == 1
    assert "Seattle" in updated_session.last_response


# ---------------------------------------------------------------------------
# Test 9: Current tool result reaches response generation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_current_tool_result_reaches_response_generation(controller: AgentController):
    synth_spy = MagicMock(side_effect=controller.synthesize_response)
    controller.synthesize_response = synth_spy

    session_id = "sess_current_synth"
    session = controller.handle_turn(
        session_id=session_id,
        transcript="What is the weather in Seattle?",
        simulated_delay=0.01,
        request_id="req_v1",
    )

    if session.active_task:
        await session.active_task

    assert synth_spy.call_count == 1
    args, _ = synth_spy.call_args
    assert args[0] == "get_weather"
    assert args[1].version == 1
    assert args[1].request_id == "req_v1"


# ---------------------------------------------------------------------------
# Test 10: Stale tool result does NOT reach response generation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stale_tool_result_does_not_reach_response_generation(controller: AgentController):
    synth_spy = MagicMock(side_effect=controller.synthesize_response)
    controller.synthesize_response = synth_spy

    session_id = "sess_stale_no_synth"
    # V1 has long tool delay
    controller.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Delhi under 5000",
        simulated_delay=0.15,
        request_id="req_v1",
    )

    # Wait brief moment, then supersede with V2
    await asyncio.sleep(0.01)
    controller.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Delhi under 3000",
        simulated_delay=0.01,
        request_id="req_v2",
    )

    # Wait for both to settle
    await asyncio.sleep(0.2)

    # synthesize_response called exactly once for V2, NEVER for V1
    assert synth_spy.call_count == 1
    args, _ = synth_spy.call_args
    assert args[1].version == 2
    assert args[1].request_id == "req_v2"


# ---------------------------------------------------------------------------
# Test 11: V1 -> V2 interruption during LLM processing remains safe
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_v1_v2_interruption_during_llm_processing_safe(
    session_manager: SessionManager, tool_registry
):
    delayed_llm = MockLLMProvider(delay=0.1)
    ctrl = AgentController(
        manager=session_manager,
        registry=tool_registry,
        llm_provider=delayed_llm,
    )

    session_id = "sess_interruption_safe"
    ctrl.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Paris",
        request_id="req_v1",
    )

    await asyncio.sleep(0.02)
    # User barge-in: explicit interrupt
    session_manager.interrupt(session_id)

    # Wait for LLM delay to elapse
    await asyncio.sleep(0.12)

    session = session_manager.get_session(session_id)
    assert session.state == SessionState.INTERRUPTED
    assert session.committed_version is None  # V1 was never committed


# ---------------------------------------------------------------------------
# Test 12: CRUCIAL ACCEPTANCE TEST
#
# Sequence:
# 1. Session starts at version 1.
# 2. V1 user turn arrives: "Find hotels in Paris".
# 3. LLM begins parsing V1.
# 4. User interrupts mid-flight with V2: "Cancel that, what is the weather in Rome?".
# 5. V1's LLM completion arrives after V2 has already bumped session version.
# 6. Verify: V1's tool execution is NEVER dispatched (prevented before launch).
# 7. V2 parses and dispatches get_weather tool.
# 8. Only V2's result is committed to the session.
# 9. Verify session final state reflects version 2, weather in Rome, and V1 data is absent.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_critical_acceptance_test(session_manager: SessionManager, tool_registry):
    # Step 1: Initialize controller with delayed LLM (0.1s)
    delayed_llm = MockLLMProvider(delay=0.1)
    ctrl = AgentController(
        manager=session_manager,
        registry=tool_registry,
        llm_provider=delayed_llm,
    )

    execute_spy = MagicMock(side_effect=tool_registry.execute_async)
    tool_registry.execute_async = execute_spy

    session_id = "sess_critical_acceptance_phase5"

    # Step 2 & 3: V1 arrives: "Find hotels in Paris" (LLM begins parsing V1 with 0.1s delay)
    session_v1 = ctrl.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Paris",
        request_id="req_v1_hotels",
    )
    assert session_v1.current_version == 1
    assert session_v1.state == SessionState.THINKING

    # Step 4: Mid-flight (0.02s in), user interrupts with V2: "Cancel that, what is the weather in Rome?"
    await asyncio.sleep(0.02)
    session_v2 = ctrl.handle_turn(
        session_id=session_id,
        transcript="Cancel that, what is the weather in Rome?",
        simulated_delay=0.01,
        request_id="req_v2_weather",
    )
    assert session_v2.current_version == 2

    # Step 5: Wait for all processing to complete
    # V1 finishes LLM at ~0.10s (sees version 2, aborts tool dispatch)
    # V2 finishes LLM at ~0.12s, dispatches get_weather, finishes tool at ~0.14s
    await asyncio.sleep(0.25)

    # Step 6: Verify V1's tool execution was NEVER dispatched
    executed_tools = [call.args[0] for call in execute_spy.call_args_list]
    assert "hotel_search" not in executed_tools, (
        f"hotel_search should NEVER have been dispatched! Dispatched tools: {executed_tools}"
    )

    # Step 7: V2 parses and dispatches get_weather tool
    assert "get_weather" in executed_tools

    # Step 8: Only V2's result is committed to the session
    final_session = session_manager.get_session(session_id)
    assert final_session.state == SessionState.COMPLETED
    assert final_session.committed_version == 2
    assert final_session.committed_request_id == "req_v2_weather"

    # Step 9: Verify session final state reflects version 2, weather in Rome, and V1 data is absent
    assert "Rome" in final_session.last_response
    assert "weather" in final_session.last_response.lower()
    assert "hotel" not in final_session.last_response.lower()
    assert "Paris" not in str(final_session.committed_data)

    # Verify event trail has rejection for V1 and acceptance for V2
    rejected_events = session_manager.get_events(session_id=session_id, event_type=EventType.RESULT_REJECTED_STALE)
    accepted_events = session_manager.get_events(session_id=session_id, event_type=EventType.RESULT_ACCEPTED)

    assert len(rejected_events) >= 1
    assert rejected_events[0].version == 1
    assert len(accepted_events) >= 1
    assert accepted_events[0].version == 2
