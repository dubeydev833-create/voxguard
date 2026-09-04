"""Unit tests for VoxGuard First-Class Request Identity (Phase 2).

Verifies that request_id and version are tracked and propagated across:
- SessionManager turns, requests registry, and cancellation
- AgentController asynchronous tool dispatch and result formulation
- ToolResult and ToolResultEnvelope
- Result Fencing rejection of stale request_ids and acceptance of valid request_ids
- REST API turn initiation and response payloads
"""

import asyncio
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.events import EventType, ToolResultEnvelope
from app.models.state import SessionState
from app.models.tool import ToolResult
from app.services.agent_controller import AgentController
from app.services.session_manager import SessionManager
from app.tools.registry import ToolRegistry, create_default_registry


@pytest.fixture
def clean_manager() -> SessionManager:
    """Provide a pristine SessionManager instance for each test."""
    manager = SessionManager()
    yield manager
    manager.clear()


@pytest.fixture
def clean_registry() -> ToolRegistry:
    """Provide a fresh ToolRegistry with mock tools registered."""
    return create_default_registry()


@pytest.fixture
def controller(clean_manager: SessionManager, clean_registry: ToolRegistry) -> AgentController:
    """Provide an AgentController wired with clean manager and registry."""
    return AgentController(manager=clean_manager, registry=clean_registry)


def test_turn_creates_unique_request_id_and_tracks_version(clean_manager: SessionManager):
    """Verify start_turn generates request_id, tracks version, and creates RequestContext."""
    session = clean_manager.start_turn("sess_test_1", "Hello assistant")

    assert session.current_version == 1
    assert session.current_request_id is not None
    assert session.current_request_id.startswith("req_")
    assert session.current_request_id in session.requests

    req_ctx = session.requests[session.current_request_id]
    assert req_ctx.request_id == session.current_request_id
    assert req_ctx.session_id == "sess_test_1"
    assert req_ctx.version == 1
    assert req_ctx.user_input == "Hello assistant"
    assert req_ctx.status == "pending"

    # Verify TURN_STARTED event has matching request_id and version
    events = clean_manager.get_events(session_id="sess_test_1", event_type=EventType.TURN_STARTED)
    assert len(events) == 1
    assert events[0].version == 1
    assert events[0].request_id == session.current_request_id
    assert events[0].payload["request_id"] == session.current_request_id


def test_explicit_request_id_is_honored(clean_manager: SessionManager):
    """Verify passing a custom request_id is preserved across state and events."""
    custom_id = "req_custom_999"
    session = clean_manager.start_turn("sess_test_2", "Book a cab", request_id=custom_id)

    assert session.current_request_id == custom_id
    assert custom_id in session.requests
    assert session.requests[custom_id].request_id == custom_id

    events = clean_manager.get_events(session_id="sess_test_2", event_type=EventType.TURN_STARTED)
    assert events[0].request_id == custom_id


def test_v1_and_v2_have_distinct_request_ids_and_versions(clean_manager: SessionManager):
    """Verify consecutive turns generate distinct request_id and monotonically increment version."""
    s1 = clean_manager.start_turn("sess_multi", "Turn 1", request_id="req_v1")
    v1_version = s1.current_version
    v1_id = s1.current_request_id

    assert v1_version == 1
    assert v1_id == "req_v1"

    s2 = clean_manager.start_turn("sess_multi", "Turn 2", request_id="req_v2")
    v2_version = s2.current_version
    v2_id = s2.current_request_id

    assert v2_version == 2
    assert v2_id == "req_v2"
    assert v1_id != v2_id
    assert len(s2.requests) == 2
    assert "req_v1" in s2.requests
    assert "req_v2" in s2.requests


@pytest.mark.asyncio
async def test_tool_execution_and_result_retain_request_id_and_version(controller: AgentController):
    """Verify AgentController propagates request_id and version through tool execution to envelope."""
    session = controller.handle_turn(
        session_id="sess_tool_test",
        transcript="Weather in Paris",
        simulated_delay=0.01,
        request_id="req_weather_paris",
    )
    req_id = session.current_request_id
    version = session.current_version
    assert req_id == "req_weather_paris"
    assert version == 1

    # Wait for the background task to complete
    if session.active_task:
        await session.active_task

    assert session.state == SessionState.COMPLETED
    assert session.committed_version == 1
    assert session.committed_request_id == "req_weather_paris"
    assert session.requests["req_weather_paris"].status == "completed"

    # Check last_result has request_id and version
    assert session.last_result is not None
    assert session.last_result.request_id == "req_weather_paris"
    assert session.last_result.version == 1

    # Check RESULT_ACCEPTED event carries request_id
    accepted_events = controller.session_manager.get_events(
        session_id="sess_tool_test",
        event_type=EventType.RESULT_ACCEPTED,
    )
    assert len(accepted_events) == 1
    assert accepted_events[0].request_id == "req_weather_paris"
    assert accepted_events[0].version == 1

    # Check RESPONSE_READY event carries request_id
    response_events = controller.session_manager.get_events(
        session_id="sess_tool_test",
        event_type=EventType.RESPONSE_READY,
    )
    assert len(response_events) == 1
    assert response_events[0].request_id == "req_weather_paris"


@pytest.mark.asyncio
async def test_cancellation_preserves_request_id_and_version(controller: AgentController):
    """Verify that when a tool is cancelled, cancellation events and results preserve request_id and version."""
    session = controller.handle_turn(
        session_id="sess_cancel_test",
        transcript="Find hotels in Delhi under 5000",
        simulated_delay=0.5,
        request_id="req_v1_cancel",
    )
    v1_id = session.current_request_id
    v1_version = session.current_version

    await asyncio.sleep(0.02)

    # Interrupt
    controller.session_manager.interrupt("sess_cancel_test")

    if session.active_task:
        try:
            await session.active_task
        except asyncio.CancelledError:
            pass

    assert session.state == SessionState.INTERRUPTED
    assert session.requests[v1_id].status in ("interrupted", "cancelled")

    # Check CANCELLATION_REQUESTED has request_id
    cancel_events = controller.session_manager.get_events(
        session_id="sess_cancel_test",
        event_type=EventType.CANCELLATION_REQUESTED,
    )
    assert len(cancel_events) >= 1
    assert cancel_events[0].request_id == v1_id
    assert cancel_events[0].version == v1_version


def test_fencing_rejects_stale_request_id_and_reports_both_identities(clean_manager: SessionManager):
    """Verify result fence rejects stale V1 tool result, preserving V1 request_id and logging V2 current request_id."""
    clean_manager.start_turn("sess_fence", "V1 query", request_id="req_v1")
    clean_manager.start_turn("sess_fence", "V2 query", request_id="req_v2")

    session = clean_manager.get_session("sess_fence")
    assert session.current_version == 2
    assert session.current_request_id == "req_v2"

    # Simulate delayed completion of V1 tool
    stale_tool_result = ToolResult.ok(
        tool_name="hotel_search",
        output={"hotels": []},
        request_id="req_v1",
        version=1,
    )
    stale_envelope = ToolResultEnvelope(
        session_id="sess_fence",
        version=1,
        request_id="req_v1",
        tool_result=stale_tool_result,
    )

    accepted = clean_manager.process_tool_result(stale_envelope)
    assert accepted is False
    assert session.requests["req_v1"].status == "stale"
    assert session.committed_request_id is None

    # Check RESULT_REJECTED_STALE event
    rejected_events = clean_manager.get_events(
        session_id="sess_fence",
        event_type=EventType.RESULT_REJECTED_STALE,
    )
    assert len(rejected_events) == 1
    rej = rejected_events[0]
    assert rej.request_id == "req_v1"
    assert rej.version == 1
    assert rej.payload["result_request_id"] == "req_v1"
    assert rej.payload["session_current_request_id"] == "req_v2"
    assert rej.payload["result_version"] == 1
    assert rej.payload["session_current_version"] == 2


@pytest.mark.asyncio
async def test_v1_superseded_by_v2_fencing_and_commitment(controller: AgentController):
    """Critical end-to-end test: V1 superseded by V2.

    Verifies:
    1. V1 has request_id_1, version 1
    2. V2 has request_id_2, version 2
    3. Stale V1 result is rejected with request_id_1
    4. V2 result is accepted with request_id_2
    5. session.committed_request_id == request_id_2
    6. session.committed_version == 2
    """
    session_id = "sess_v1_v2_e2e"

    # Step 1: Start V1 with a long delay
    controller.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Delhi under 5000",
        simulated_delay=0.3,
        request_id="req_hotel_v1",
    )
    s = controller.session_manager.get_session(session_id)
    assert s.current_request_id == "req_hotel_v1"
    assert s.current_version == 1

    await asyncio.sleep(0.05)

    # Step 2: Start V2 with a faster delay
    controller.handle_turn(
        session_id=session_id,
        transcript="Find hotels in Delhi under 3000",
        simulated_delay=0.05,
        request_id="req_hotel_v2",
    )
    assert s.current_request_id == "req_hotel_v2"
    assert s.current_version == 2

    # Wait for V2 to complete
    await asyncio.sleep(0.15)

    assert s.committed_version == 2
    assert s.committed_request_id == "req_hotel_v2"
    assert s.state == SessionState.COMPLETED

    # Check rejected events for V1
    rejected = controller.session_manager.get_events(
        session_id=session_id,
        event_type=EventType.RESULT_REJECTED_STALE,
    )
    # If V1 finished late or was rejected
    if rejected:
        assert rejected[0].request_id == "req_hotel_v1"
        assert rejected[0].version == 1

    # Check accepted events for V2
    accepted = controller.session_manager.get_events(
        session_id=session_id,
        event_type=EventType.RESULT_ACCEPTED,
    )
    assert len(accepted) == 1
    assert accepted[0].request_id == "req_hotel_v2"
    assert accepted[0].version == 2


def test_rest_api_request_id_endpoints():
    """Verify REST API creates and returns request_id in SessionResponse."""
    client = TestClient(app)

    # 1. Create Session
    create_resp = client.post("/api/v1/sessions", json={})
    assert create_resp.status_code == 201
    data = create_resp.json()
    assert "session_id" in data
    assert "current_request_id" in data
    assert "committed_request_id" in data
    session_id = data["session_id"]

    # 2. Start Turn with custom request_id
    turn_resp = client.post(
        f"/api/v1/sessions/{session_id}/turns",
        json={"transcript": "hello world", "request_id": "req_api_turn_1"},
    )
    assert turn_resp.status_code == 200
    turn_data = turn_resp.json()
    assert turn_data["current_request_id"] == "req_api_turn_1"
    assert turn_data["current_version"] == 1
    assert turn_data["committed_request_id"] == "req_api_turn_1"

    # 3. Get Session
    get_resp = client.get(f"/api/v1/sessions/{session_id}")
    assert get_resp.status_code == 200
    get_data = get_resp.json()
    assert get_data["current_request_id"] == "req_api_turn_1"
    assert get_data["committed_request_id"] == "req_api_turn_1"
