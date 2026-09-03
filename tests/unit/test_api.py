"""Sanity and integration tests for FastAPI REST and WebSocket endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.session_manager import session_manager


@pytest.fixture
def client():
    session_manager.clear()
    return TestClient(app)


def test_health_and_root(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    resp_root = client.get("/")
    assert resp_root.status_code == 200
    assert resp_root.json()["service"] == "VoxGuard API"


def test_session_lifecycle_rest_endpoints(client: TestClient):
    # 1. Create session
    resp = client.post("/api/v1/sessions", json={"metadata": {"user": "alice"}})
    assert resp.status_code == 201
    data = resp.json()
    session_id = data["session_id"]
    assert data["current_version"] == 0
    assert data["state"] == "IDLE"

    # 2. Get session
    resp_get = client.get(f"/api/v1/sessions/{session_id}")
    assert resp_get.status_code == 200
    assert resp_get.json()["session_id"] == session_id

    # 3. Start turn 1
    resp_turn1 = client.post(
        f"/api/v1/sessions/{session_id}/turns",
        json={"transcript": "Book a cab to downtown"},
    )
    assert resp_turn1.status_code == 200
    t1_data = resp_turn1.json()
    assert t1_data["current_version"] == 1
    assert t1_data["state"] in ["THINKING", "TOOL_RUNNING"]
    assert t1_data["last_transcript"] == "Book a cab to downtown"

    # 4. Start turn 2 (supersedes turn 1)
    resp_turn2 = client.post(
        f"/api/v1/sessions/{session_id}/turns",
        json={"transcript": "Actually make it to airport"},
    )
    assert resp_turn2.status_code == 200
    t2_data = resp_turn2.json()
    assert t2_data["current_version"] == 2
    assert t2_data["last_transcript"] == "Actually make it to airport"

    # 5. Submit stale result (version 1) -> rejected!
    resp_stale = client.post(
        f"/api/v1/sessions/{session_id}/results",
        json={
            "version": 1,
            "tool_result": {
                "tool_name": "book_ride",
                "success": True,
                "output": {"destination": "downtown"},
            },
        },
    )
    assert resp_stale.status_code == 200
    assert resp_stale.json()["accepted"] is False

    # 6. Submit valid result (version 2) -> accepted!
    resp_valid = client.post(
        f"/api/v1/sessions/{session_id}/results",
        json={
            "version": 2,
            "tool_result": {
                "tool_name": "book_ride",
                "success": True,
                "output": {"destination": "airport"},
            },
        },
    )
    assert resp_valid.status_code == 200
    assert resp_valid.json()["accepted"] is True

    # 7. Explicit interruption
    resp_interrupt = client.post(f"/api/v1/sessions/{session_id}/interrupt")
    assert resp_interrupt.status_code == 200
    assert resp_interrupt.json()["state"] == "INTERRUPTED"


def test_websocket_stream_events(client: TestClient):
    session_id = "sess_ws_test"

    with client.websocket_connect(f"/api/v1/sessions/{session_id}/events") as ws:
        # Initial connection message
        init_msg = ws.receive_json()
        assert init_msg["event_type"] == "CONNECTED"
        assert init_msg["session_id"] == session_id

        # Trigger turn via manager
        session_manager.start_turn(session_id, transcript="Streaming test prompt")

        # Verify broadcast
        event_data = ws.receive_json()
        assert event_data["event_type"] == "TURN_STARTED"
        assert event_data["session_id"] == session_id
        assert event_data["version"] == 1
        assert event_data["payload"]["transcript"] == "Streaming test prompt"
