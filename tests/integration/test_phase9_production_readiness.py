"""VoxGuard Phase 9 — Production Readiness, API Hardening & Final Backend Verification Tests.

Validates:
1. API contract hardening (request validation, whitespace rejection, negative version rejection, 422/404 handling)
2. Safe error isolation and generic 500 responses without stack traces or secret leakage
3. Session lifecycle, request identity propagation, and committed data consistency
4. ResultFence as final authority (stale versions, mismatched request IDs, cancellation failures)
5. Zero secrets exposure in responses, events, or configuration templates
6. Complete end-to-end regression of the V1 -> V2 barge-in lifecycle
"""

import asyncio
import os
from unittest.mock import patch
import pytest
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from app.main import app
from app.models.events import EventType, ToolResultEnvelope
from app.models.state import SessionState
from app.models.tool import ToolResult
from app.services.agent_controller import AgentController
from app.services.session_manager import SessionManager


# =====================================================================
# 1. API CONTRACT HARDENING & REQUEST VALIDATION
# =====================================================================

def test_p9_whitespace_only_transcript_rejected():
    """Verify whitespace-only transcript is rejected with HTTP 422."""
    client = TestClient(app)
    sess_resp = client.post("/api/v1/sessions")
    assert sess_resp.status_code == 201
    s_id = sess_resp.json()["session_id"]

    # Test whitespace-only transcripts
    for bad_transcript in ["   ", "\t", "\n  \t  "]:
        resp = client.post(
            f"/api/v1/sessions/{s_id}/turns",
            json={"transcript": bad_transcript},
        )
        assert resp.status_code == 422
        assert "empty or whitespace" in str(resp.json())


def test_p9_negative_version_rejected():
    """Verify negative version in process result request is rejected with HTTP 422."""
    client = TestClient(app)
    sess_resp = client.post("/api/v1/sessions")
    s_id = sess_resp.json()["session_id"]

    resp = client.post(
        f"/api/v1/sessions/{s_id}/results",
        json={
            "version": -1,
            "tool_result": {
                "tool_name": "hotel_search",
                "success": True,
            },
        },
    )
    assert resp.status_code == 422


def test_p9_session_id_whitespace_sanitized():
    """Verify whitespace-only session_id in create_session is cleanly sanitized to default uuid."""
    client = TestClient(app)
    resp = client.post("/api/v1/sessions", json={"session_id": "   "})
    assert resp.status_code == 201
    sid = resp.json()["session_id"]
    assert sid.startswith("sess_")
    assert sid.strip() == sid


def test_p9_malformed_json_body_rejected():
    """Verify malformed JSON requests return HTTP 422 without server error."""
    client = TestClient(app)
    sess_resp = client.post("/api/v1/sessions")
    s_id = sess_resp.json()["session_id"]

    resp = client.post(
        f"/api/v1/sessions/{s_id}/turns",
        content="not valid json {{{",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422


# =====================================================================
# 2. SAFE ERROR ISOLATION & GENERIC 500
# =====================================================================

def test_p9_sanitized_500_on_unhandled_exception():
    """Verify unhandled internal exceptions return clean JSON 500 without stack trace or secrets."""
    client = TestClient(app, raise_server_exceptions=False)
    leak_secret = "CRITICAL_INTERNAL_DB_SECRET_KEY_999"

    with patch(
        "app.services.agent_controller.agent_controller.handle_turn",
        side_effect=RuntimeError(f"Internal crash with secret: {leak_secret}"),
    ):
        resp = client.post("/api/v1/sessions/sess_crash/turns", json={"transcript": "Test crash"})
        assert resp.status_code == 500
        data = resp.json()
        assert data == {"detail": "Internal server error"}
        assert leak_secret not in str(data)
        assert "Traceback" not in str(data)


def test_p9_unknown_session_consistent_404():
    """Verify accessing unknown sessions consistently returns 404 with structured detail."""
    client = TestClient(app)
    ghost_id = "sess_p9_ghost_404"

    endpoints = [
        ("GET", f"/api/v1/sessions/{ghost_id}"),
        ("GET", f"/api/v1/sessions/{ghost_id}/events"),
        ("POST", f"/api/v1/sessions/{ghost_id}/interrupt"),
        ("POST", f"/api/v1/sessions/{ghost_id}/results"),
    ]

    for method, path in endpoints:
        if method == "GET":
            resp = client.get(path)
        else:
            payload = {"version": 1, "tool_result": {"tool_name": "hotel_search", "success": True}} if "results" in path else None
            resp = client.post(path, json=payload)
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


# =====================================================================
# 3. SESSION LIFECYCLE & REQUEST IDENTITY
# =====================================================================

@pytest.mark.asyncio
async def test_p9_session_creation_and_request_identity():
    """Verify session creation defaults and request_id propagation."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        create = await ac.post("/api/v1/sessions")
        assert create.status_code == 201
        data = create.json()
        assert data["current_version"] == 0
        assert data["state"] == "IDLE"
        assert data["committed_version"] is None
        s_id = data["session_id"]

        # Submit turn with explicit request_id
        turn = await ac.post(
            f"/api/v1/sessions/{s_id}/turns",
            json={"transcript": "Find hotels in Delhi under 3000", "request_id": "req_custom_p9", "simulated_delay": 0.0},
        )
        assert turn.status_code == 200
        turn_data = turn.json()
        assert turn_data["current_version"] == 1
        assert turn_data["current_request_id"] == "req_custom_p9"

        await asyncio.sleep(0.05)
        status = await ac.get(f"/api/v1/sessions/{s_id}")
        st_data = status.json()
        assert st_data["committed_version"] == 1
        assert st_data["committed_request_id"] == "req_custom_p9"
        assert st_data["state"] == "COMPLETED"


# =====================================================================
# 4. RESULT FENCING FINAL AUTHORITY
# =====================================================================

@pytest.mark.asyncio
async def test_p9_result_fence_rejects_stale_versions_and_requests():
    """Verify Result Fence is the final authority rejecting both stale versions and mismatched request IDs."""
    manager = SessionManager()
    sess_id = "sess_p9_fencing"
    manager.start_turn(sess_id, "Turn 1", request_id="req_turn_1")
    manager.start_turn(sess_id, "Turn 2", request_id="req_turn_2")

    sess = manager.get_session(sess_id)
    assert sess.current_version == 2
    assert sess.current_request_id == "req_turn_2"

    # 1. Stale version rejection
    env_stale_v = ToolResultEnvelope(
        session_id=sess_id,
        version=1,
        request_id="req_turn_1",
        tool_result=ToolResult.ok("hotel_search", {"data": "stale"}),
    )
    assert manager.process_tool_result(env_stale_v) is False

    # 2. Matching version but mismatched request_id rejection
    env_stale_req = ToolResultEnvelope(
        session_id=sess_id,
        version=2,
        request_id="req_turn_mismatched",
        tool_result=ToolResult.ok("hotel_search", {"data": "mismatched"}),
    )
    assert manager.process_tool_result(env_stale_req) is False

    # 3. Accepted result (version and request_id match)
    env_accepted = ToolResultEnvelope(
        session_id=sess_id,
        version=2,
        request_id="req_turn_2",
        tool_result=ToolResult.ok("hotel_search", {"data": "valid_v2"}),
    )
    assert manager.process_tool_result(env_accepted) is True
    assert sess.committed_data["data"] == "valid_v2"


@pytest.mark.asyncio
async def test_p9_cancellation_failure_does_not_compromise_fence():
    """Verify that if a tool actively ignores cancellation and completes, the fence drops it."""
    manager = SessionManager()
    controller = AgentController(manager=manager)
    sess_id = "sess_p9_uncancellable"

    # Turn 1: 0.1s simulated delay
    controller.handle_turn(sess_id, "Find hotels under 5000", simulated_delay=0.1, request_id="req_v1")
    await asyncio.sleep(0.01)

    # Turn 2: supersedes immediately
    controller.handle_turn(sess_id, "Find hotels under 3000", simulated_delay=0.0, request_id="req_v2")
    await asyncio.sleep(0.04)

    sess = manager.get_session(sess_id)
    assert sess.committed_version == 2
    assert sess.committed_request_id == "req_v2"

    # Wait for Turn 1 to complete late
    await asyncio.sleep(0.12)

    # Check that Turn 1 never committed
    assert sess.committed_version == 2
    assert sess.committed_request_id == "req_v2"
    stale_events = manager.get_events(sess_id, event_type=EventType.RESULT_REJECTED_STALE)
    assert len(stale_events) >= 1
    assert stale_events[0].request_id == "req_v1"


# =====================================================================
# 5. CONFIGURATION & SECURITY
# =====================================================================

def test_p9_configuration_and_secrets_hygiene():
    """Verify secrets configuration: .env.example contains only placeholders, no real secrets in repo."""
    # Check .env.example
    assert os.path.exists(".env.example")
    with open(".env.example", "r", encoding="utf-8") as f:
        content = f.read()

    # Ensure placeholders only
    assert "your-openai-api-key-here" in content
    assert "your-anthropic-api-key-here" in content
    assert "sk-" not in content

    # Check .gitignore
    assert os.path.exists(".gitignore")
    with open(".gitignore", "r", encoding="utf-8") as f:
        git_content = f.read()
    assert ".env" in git_content


def test_p9_zero_secrets_in_all_api_responses():
    """Verify that API responses never leak credentials or tokens."""
    client = TestClient(app)
    sess = client.post("/api/v1/sessions").json()
    sid = sess["session_id"]

    turn = client.post(
        f"/api/v1/sessions/{sid}/turns",
        json={"transcript": "Find hotels in Delhi under 3000"},
    ).json()

    forbidden = ["sk-", "secret_key", "password", "token", "authorization"]
    for obj in [sess, turn]:
        serialized = str(obj).lower()
        for word in forbidden:
            assert word not in serialized


# =====================================================================
# 6. FULL PHASE 1–8 BEHAVIORAL REGRESSION (V1 -> V2 BARGE-IN)
# =====================================================================

@pytest.mark.asyncio
async def test_p9_full_end_to_end_v1_v2_barge_in():
    """Complete end-to-end verification of Phase 1-8 behavior:
    1. V1 turn with delay
    2. V2 superseding barge-in
    3. Cancellation attempted
    4. V2 accepted and produces RESPONSE_READY
    5. V1 rejected as stale upon late arrival
    6. State strictly reflects V2
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Create session
        create = await ac.post("/api/v1/sessions")
        s_id = create.json()["session_id"]

        # Turn 1
        t1 = await ac.post(
            f"/api/v1/sessions/{s_id}/turns",
            json={"transcript": "Find hotels in Delhi under 5000", "simulated_delay": 0.15, "request_id": "req_p9_v1"},
        )
        assert t1.status_code == 200
        assert t1.json()["current_version"] == 1

        await asyncio.sleep(0.02)

        # Turn 2 (barge-in)
        t2 = await ac.post(
            f"/api/v1/sessions/{s_id}/turns",
            json={"transcript": "Actually make that under 3000", "simulated_delay": 0.0, "request_id": "req_p9_v2"},
        )
        assert t2.status_code == 200
        assert t2.json()["current_version"] == 2

        # Allow execution to settle
        await asyncio.sleep(0.2)

        # Final state check
        status = await ac.get(f"/api/v1/sessions/{s_id}")
        assert status.status_code == 200
        data = status.json()
        assert data["committed_version"] == 2
        assert data["committed_request_id"] == "req_p9_v2"
        assert data["state"] == "COMPLETED"
        assert "3000" in data["last_response"]

        # Events check
        evts_resp = await ac.get(f"/api/v1/sessions/{s_id}/events")
        evts = evts_resp.json()
        types = [e["event_type"] for e in evts]
        assert "TURN_STARTED" in types
        assert "CANCELLATION_REQUESTED" in types
        assert "RESULT_REJECTED_STALE" in types
        assert "RESULT_ACCEPTED" in types
        assert "RESPONSE_READY" in types
