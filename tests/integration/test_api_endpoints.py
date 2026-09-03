"""Integration tests for FastAPI REST Endpoints and WebSocket Streaming."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.session_manager import session_manager


@pytest.fixture
def client():
    session_manager.clear()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Test 1: Verify POST /api/v1/sessions returns 201 with IDLE and version 0
# ---------------------------------------------------------------------------
def test_create_session_endpoint(client: TestClient):
    resp = client.post("/api/v1/sessions", json={"metadata": {"test_client": "integration"}})
    assert resp.status_code == 201
    data = resp.json()

    assert data["state"] == "IDLE"
    assert data["current_version"] == 0
    assert "session_id" in data
    assert data["session_id"].startswith("sess_")
    assert data["committed_version"] is None
    assert data["last_transcript"] is None


# ---------------------------------------------------------------------------
# Test 2: Verify POST /api/v1/sessions/{session_id}/turns initiates a turn and increments version
# ---------------------------------------------------------------------------
def test_turns_initiation_and_version_increment(client: TestClient):
    session_id = "sess_api_turn_001"

    # Initialize session
    create_resp = client.post("/api/v1/sessions", json={"session_id": session_id})
    assert create_resp.status_code == 201
    assert create_resp.json()["current_version"] == 0

    # Turn 1
    t1_resp = client.post(
        f"/api/v1/sessions/{session_id}/turns",
        json={"transcript": "Find hotels in Seattle with max price 4000"},
    )
    assert t1_resp.status_code == 200
    t1_data = t1_resp.json()
    assert t1_data["current_version"] == 1
    assert t1_data["last_transcript"] == "Find hotels in Seattle with max price 4000"

    # Turn 2 (superseding turn)
    t2_resp = client.post(
        f"/api/v1/sessions/{session_id}/turns",
        json={"transcript": "Book a cab to downtown"},
    )
    assert t2_resp.status_code == 200
    t2_data = t2_resp.json()
    assert t2_data["current_version"] == 2
    assert t2_data["last_transcript"] == "Book a cab to downtown"

    # Turn 3
    t3_resp = client.post(
        f"/api/v1/sessions/{session_id}/turns",
        json={"transcript": "What is the weather in Tokyo?"},
    )
    assert t3_resp.status_code == 200
    t3_data = t3_resp.json()
    assert t3_data["current_version"] == 3


# ---------------------------------------------------------------------------
# Test 3: Verify POST /api/v1/sessions/{session_id}/interrupt switches state to INTERRUPTED
# ---------------------------------------------------------------------------
def test_interrupt_endpoint(client: TestClient):
    session_id = "sess_api_interrupt_001"

    # Start a turn
    client.post(
        f"/api/v1/sessions/{session_id}/turns",
        json={"transcript": "Send an email to alice@example.com"},
    )

    # Interrupt
    resp = client.post(f"/api/v1/sessions/{session_id}/interrupt")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == session_id
    assert data["state"] == "INTERRUPTED"

    # Confirm session retrieval also returns INTERRUPTED
    get_resp = client.get(f"/api/v1/sessions/{session_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["state"] == "INTERRUPTED"


# ---------------------------------------------------------------------------
# Test 4: Verify WebSocket streams TURN_STARTED, TOOL_STARTED,
#         RESULT_REJECTED_STALE, and RESULT_ACCEPTED
# ---------------------------------------------------------------------------
def test_websocket_streams_lifecycle_events(client: TestClient):
    session_id = "sess_api_ws_stream_001"

    from app.models.events import EventType, ToolResultEnvelope
    from app.models.tool import ToolResult

    with client.websocket_connect(f"/api/v1/sessions/{session_id}/events") as ws:
        # Initial connection event
        conn_event = ws.receive_json()
        assert conn_event["event_type"] == "CONNECTED"
        assert conn_event["session_id"] == session_id

        # 1. Turn 1 starts -> broadcasts TURN_STARTED
        session_manager.start_turn(session_id, transcript="Find hotels with price 5000")
        ev1 = ws.receive_json()
        assert ev1["event_type"] == "TURN_STARTED"
        assert ev1["version"] == 1

        # 2. Tool execution starts -> broadcasts TOOL_STARTED
        session_manager._emit(
            EventType.TOOL_STARTED,
            session_id=session_id,
            version=1,
            payload={"tool_name": "hotel_search"},
        )
        ev2 = ws.receive_json()
        assert ev2["event_type"] == "TOOL_STARTED"
        assert ev2["version"] == 1
        assert ev2["payload"]["tool_name"] == "hotel_search"

        # 3. Advance to Turn 2 -> version becomes 2
        session_manager.start_turn(session_id, transcript="Check Seattle weather")
        ev3 = ws.receive_json()
        assert ev3["event_type"] == "TURN_STARTED"
        assert ev3["version"] == 2

        # 4. Submit stale result (v1) -> broadcasts RESULT_REJECTED_STALE
        stale_env = ToolResultEnvelope(
            session_id=session_id,
            version=1,
            tool_result=ToolResult.ok(tool_name="hotel_search", output={"price": 5000}),
        )
        accepted = session_manager.process_tool_result(stale_env)
        assert accepted is False

        ev4 = ws.receive_json()
        assert ev4["event_type"] == "RESULT_REJECTED_STALE"
        assert ev4["version"] == 1

        # 5. Submit valid result (v2) -> broadcasts RESULT_ACCEPTED
        valid_env = ToolResultEnvelope(
            session_id=session_id,
            version=2,
            tool_result=ToolResult.ok(tool_name="get_weather", output={"temp": 68}),
        )
        accepted_v2 = session_manager.process_tool_result(valid_env)
        assert accepted_v2 is True

        ev5 = ws.receive_json()
        assert ev5["event_type"] == "RESULT_ACCEPTED"
        assert ev5["version"] == 2

        ws.close()
