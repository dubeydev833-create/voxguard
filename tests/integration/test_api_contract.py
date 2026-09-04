"""Integration tests for VoxGuard Phase 6 — Backend API Integration Contract.

Verifies the complete FastAPI API contract:
1. Create session (POST /api/v1/sessions)
2. Submit turn (POST /api/v1/sessions/{session_id}/turns)
3. Retrieve session status (GET /api/v1/sessions/{session_id})
4. Request_id generation (auto-generated vs explicit)
5. Version increment (monotonic $0 \to 1 \to 2$)
6. Superseding request (mid-flight turn superseding)
7. Cancellation attempt (CANCELLATION_REQUESTED emission)
8. Stale result rejection (POST /results with obsolete version -> accepted=False)
9. Current result acceptance (POST /results with current version -> accepted=True)
10. V1 -> V2 API race condition / critical interruption flow
11. Unknown session error handling (HTTP 404)
12. Invalid payload validation (HTTP 422)
13. Tool failure handling (safe ERROR state without crashing)
14. REST & WebSocket observability
"""

import asyncio
import time
import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models.events import EventType, ToolResultEnvelope
from app.models.state import SessionState
from app.models.tool import ToolResult
from app.services.session_manager import session_manager


@pytest.fixture
def client():
    session_manager.clear()
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Create Session
# ---------------------------------------------------------------------------
def test_create_session(client: TestClient):
    resp = client.post("/api/v1/sessions", json={"metadata": {"source": "voice_client"}})
    assert resp.status_code == 201
    data = resp.json()

    assert data["session_id"].startswith("sess_")
    assert data["current_version"] == 0
    assert data["state"] == "IDLE"
    assert data["current_request_id"] is None
    assert data["committed_version"] is None
    assert data["last_response"] is None
    assert "created_at" in data
    assert "updated_at" in data

    # Verify no secrets or sensitive internal fields are exposed
    assert "api_key" not in data
    assert "secret" not in data
    assert "token" not in data


# ---------------------------------------------------------------------------
# 2. Submit Turn
# ---------------------------------------------------------------------------
def test_submit_turn(client: TestClient):
    # Initialize session
    create_resp = client.post("/api/v1/sessions")
    session_id = create_resp.json()["session_id"]

    # Submit turn
    turn_resp = client.post(
        f"/api/v1/sessions/{session_id}/turns",
        json={"transcript": "What is the weather in Seattle?"},
    )
    assert turn_resp.status_code == 200
    data = turn_resp.json()

    assert data["session_id"] == session_id
    assert data["current_version"] == 1
    assert data["last_transcript"] == "What is the weather in Seattle?"
    assert data["state"] in ["THINKING", "TOOL_RUNNING"]
    assert data["current_request_id"] is not None


# ---------------------------------------------------------------------------
# 3. Retrieve Session Status
# ---------------------------------------------------------------------------
def test_retrieve_session_status(client: TestClient):
    create_resp = client.post("/api/v1/sessions")
    session_id = create_resp.json()["session_id"]

    # Post turn
    client.post(
        f"/api/v1/sessions/{session_id}/turns",
        json={"transcript": "Hello, what can you do?"},
    )

    # Get status
    get_resp = client.get(f"/api/v1/sessions/{session_id}")
    assert get_resp.status_code == 200
    status_data = get_resp.json()

    assert status_data["session_id"] == session_id
    assert status_data["current_version"] == 1
    assert status_data["state"] in ["THINKING", "TOOL_RUNNING", "COMPLETED"]
    assert status_data["last_event"] is not None


# ---------------------------------------------------------------------------
# 4. Request ID Generation (Auto-generated vs Explicit)
# ---------------------------------------------------------------------------
def test_request_id_generation(client: TestClient):
    create_resp = client.post("/api/v1/sessions")
    session_id = create_resp.json()["session_id"]

    # Auto-generated request_id
    t1_resp = client.post(
        f"/api/v1/sessions/{session_id}/turns",
        json={"transcript": "Find hotels in Paris"},
    )
    t1_data = t1_resp.json()
    assert t1_data["current_request_id"].startswith("req_")

    # Explicit request_id
    custom_id = "voice_layer_req_9988"
    t2_resp = client.post(
        f"/api/v1/sessions/{session_id}/turns",
        json={"transcript": "Find hotels in Rome", "request_id": custom_id},
    )
    t2_data = t2_resp.json()
    assert t2_data["current_request_id"] == custom_id


# ---------------------------------------------------------------------------
# 5. Version Increment (Monotonic 0 -> 1 -> 2 -> 3)
# ---------------------------------------------------------------------------
def test_version_increment(client: TestClient):
    create_resp = client.post("/api/v1/sessions")
    session_id = create_resp.json()["session_id"]
    assert create_resp.json()["current_version"] == 0

    t1 = client.post(f"/api/v1/sessions/{session_id}/turns", json={"transcript": "Turn 1"})
    assert t1.json()["current_version"] == 1

    t2 = client.post(f"/api/v1/sessions/{session_id}/turns", json={"transcript": "Turn 2"})
    assert t2.json()["current_version"] == 2

    t3 = client.post(f"/api/v1/sessions/{session_id}/turns", json={"transcript": "Turn 3"})
    assert t3.json()["current_version"] == 3


# ---------------------------------------------------------------------------
# 6. Superseding Request
# ---------------------------------------------------------------------------
def test_superseding_request(client: TestClient):
    create_resp = client.post("/api/v1/sessions")
    session_id = create_resp.json()["session_id"]

    # V1 starts with simulated delay
    t1 = client.post(
        f"/api/v1/sessions/{session_id}/turns",
        json={"transcript": "Find hotels in Delhi under 5000", "simulated_delay": 0.5},
    )
    assert t1.json()["current_version"] == 1

    # V2 supersedes V1 mid-flight
    t2 = client.post(
        f"/api/v1/sessions/{session_id}/turns",
        json={"transcript": "Actually find hotels in Mumbai under 4000", "simulated_delay": 0.01},
    )
    assert t2.json()["current_version"] == 2
    assert t2.json()["last_transcript"] == "Actually find hotels in Mumbai under 4000"


# ---------------------------------------------------------------------------
# 7. Cancellation Attempt on Superseding Turn
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cancellation_attempt():
    session_manager.clear()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        create_resp = await ac.post("/api/v1/sessions")
        session_id = create_resp.json()["session_id"]

        # V1 starts long task
        await ac.post(
            f"/api/v1/sessions/{session_id}/turns",
            json={"transcript": "Find hotels in Delhi under 5000", "simulated_delay": 0.5},
        )
        await asyncio.sleep(0.02)

        # V2 arrives
        await ac.post(
            f"/api/v1/sessions/{session_id}/turns",
            json={"transcript": "Cancel that", "simulated_delay": 0.01},
        )

        # Check session events for CANCELLATION_REQUESTED
        events_resp = await ac.get(f"/api/v1/sessions/{session_id}/events")
        assert events_resp.status_code == 200
        event_types = [e["event_type"] for e in events_resp.json()]
        assert "CANCELLATION_REQUESTED" in event_types


# ---------------------------------------------------------------------------
# 8. Stale Result Rejection (POST /results)
# ---------------------------------------------------------------------------
def test_stale_result_rejection(client: TestClient):
    create_resp = client.post("/api/v1/sessions")
    session_id = create_resp.json()["session_id"]

    # Advance session to version 2
    client.post(f"/api/v1/sessions/{session_id}/turns", json={"transcript": "Turn 1"})
    client.post(f"/api/v1/sessions/{session_id}/turns", json={"transcript": "Turn 2"})

    # Submit result with stale version 1
    resp = client.post(
        f"/api/v1/sessions/{session_id}/results",
        json={
            "version": 1,
            "tool_result": {
                "tool_name": "hotel_search",
                "success": True,
                "output": {"hotels": ["Old Hotel"]},
            },
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["accepted"] is False
    assert data["submitted_version"] == 1
    assert data["current_session_version"] == 2


# ---------------------------------------------------------------------------
# 9. Current Result Acceptance (POST /results)
# ---------------------------------------------------------------------------
def test_current_result_acceptance(client: TestClient):
    create_resp = client.post("/api/v1/sessions")
    session_id = create_resp.json()["session_id"]

    # Advance to version 1
    client.post(f"/api/v1/sessions/{session_id}/turns", json={"transcript": "Turn 1"})

    # Submit result matching version 1
    resp = client.post(
        f"/api/v1/sessions/{session_id}/results",
        json={
            "version": 1,
            "tool_result": {
                "tool_name": "hotel_search",
                "success": True,
                "output": {"hotels": ["Current Hotel"]},
            },
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["accepted"] is True
    assert data["submitted_version"] == 1
    assert data["current_session_version"] == 1


# ---------------------------------------------------------------------------
# 10. Critical V1 -> V2 Interruption Flow (Full API Verification)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_v1_v2_api_interruption_flow():
    session_manager.clear()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        create_resp = await ac.post("/api/v1/sessions")
        session_id = create_resp.json()["session_id"]

        # V1: "Find hotels in Delhi under 5000" (starts async tool with 0.15s delay)
        t1_resp = await ac.post(
            f"/api/v1/sessions/{session_id}/turns",
            json={"transcript": "Find hotels in Delhi under 5000", "simulated_delay": 0.15},
        )
        assert t1_resp.status_code == 200
        assert t1_resp.json()["current_version"] == 1

        # Non-blocking check: yielded to event loop
        await asyncio.sleep(0.02)

        # V2: "Actually under 3000" (supersedes V1 mid-flight with 0.01s delay)
        t2_resp = await ac.post(
            f"/api/v1/sessions/{session_id}/turns",
            json={"transcript": "Actually under 3000", "simulated_delay": 0.01},
        )
        assert t2_resp.status_code == 200
        assert t2_resp.json()["current_version"] == 2

        # Wait for all background processing to settle on active loop
        await asyncio.sleep(0.25)

        # Verify final session status
        final_resp = await ac.get(f"/api/v1/sessions/{session_id}")
        assert final_resp.status_code == 200
        final_data = final_resp.json()

        # V2 committed, V1 obsolete
        assert final_data["current_version"] == 2
        assert final_data["committed_version"] == 2
        assert final_data["state"] == "COMPLETED"

        # Response contains V2 results (under 3000), not V1 (under 5000)
        assert "3000" in final_data["last_response"]
        assert "5000" not in final_data["last_response"]

        # Events verification: V1 was rejected as stale, V2 was accepted
        events_resp = await ac.get(f"/api/v1/sessions/{session_id}/events")
        events = events_resp.json()
        stale_events = [e for e in events if e["event_type"] == "RESULT_REJECTED_STALE"]
        accepted_events = [e for e in events if e["event_type"] == "RESULT_ACCEPTED"]

        assert len(stale_events) >= 1
        assert stale_events[0]["version"] == 1
        assert len(accepted_events) >= 1
        assert accepted_events[0]["version"] == 2


# ---------------------------------------------------------------------------
# 11. Unknown Session Error Handling (HTTP 404)
# ---------------------------------------------------------------------------
def test_unknown_session_404(client: TestClient):
    ghost_id = "sess_non_existent_9999"

    # GET /sessions/{id} -> 404
    resp_get = client.get(f"/api/v1/sessions/{ghost_id}")
    assert resp_get.status_code == 404
    assert f"'{ghost_id}' not found" in resp_get.json()["detail"]

    # POST /sessions/{id}/interrupt -> 404
    resp_interrupt = client.post(f"/api/v1/sessions/{ghost_id}/interrupt")
    assert resp_interrupt.status_code == 404

    # GET /sessions/{id}/events -> 404
    resp_events = client.get(f"/api/v1/sessions/{ghost_id}/events")
    assert resp_events.status_code == 404

    # POST /sessions/{id}/results -> 404
    resp_results = client.post(
        f"/api/v1/sessions/{ghost_id}/results",
        json={"version": 1, "tool_result": {"tool_name": "hotel_search", "success": True}},
    )
    assert resp_results.status_code == 404


# ---------------------------------------------------------------------------
# 12. Invalid Payload Validation (HTTP 422)
# ---------------------------------------------------------------------------
def test_invalid_payload_422(client: TestClient):
    create_resp = client.post("/api/v1/sessions")
    session_id = create_resp.json()["session_id"]

    # Missing transcript
    resp_missing = client.post(f"/api/v1/sessions/{session_id}/turns", json={})
    assert resp_missing.status_code == 422

    # Empty transcript (min_length=1)
    resp_empty = client.post(f"/api/v1/sessions/{session_id}/turns", json={"transcript": ""})
    assert resp_empty.status_code == 422

    # Malformed JSON in results submission
    resp_bad_res = client.post(
        f"/api/v1/sessions/{session_id}/results",
        json={"version": "not_an_int"},
    )
    assert resp_bad_res.status_code == 422


# ---------------------------------------------------------------------------
# 13. Tool Failure Handling (Safe ERROR state without 500 crash)
# ---------------------------------------------------------------------------
def test_tool_failure_handling(client: TestClient):
    create_resp = client.post("/api/v1/sessions")
    session_id = create_resp.json()["session_id"]

    # Submit tool failure result through Result Fence
    fail_resp = client.post(
        f"/api/v1/sessions/{session_id}/results",
        json={
            "version": 0,
            "tool_result": {
                "tool_name": "transfer_funds",
                "success": False,
                "error": "Payment gateway unreachable",
            },
        },
    )
    assert fail_resp.status_code == 200
    assert fail_resp.json()["accepted"] is True

    # Retrieve status: safe ERROR state, no 500 crash
    status_resp = client.get(f"/api/v1/sessions/{session_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["state"] == "ERROR"
    assert status_resp.json()["last_event"] == "TOOL_FAILED"


# ---------------------------------------------------------------------------
# 14. REST & WebSocket Observability
# ---------------------------------------------------------------------------
def test_observability_rest_and_websocket(client: TestClient):
    create_resp = client.post("/api/v1/sessions")
    session_id = create_resp.json()["session_id"]

    with client.websocket_connect(f"/api/v1/sessions/{session_id}/events") as ws:
        init_evt = ws.receive_json()
        assert init_evt["event_type"] == "CONNECTED"

        # Submit turn
        client.post(
            f"/api/v1/sessions/{session_id}/turns",
            json={"transcript": "What is the weather in Boston?"},
        )

        # Receive broadcast over WS
        ws_evt = ws.receive_json()
        assert ws_evt["event_type"] == "TURN_STARTED"
        assert ws_evt["version"] == 1

        # Also inspect via REST GET /events
        rest_events = client.get(f"/api/v1/sessions/{session_id}/events").json()
        assert len(rest_events) >= 1
        assert any(e["event_type"] == "TURN_STARTED" for e in rest_events)

        ws.close()
