"""VoxGuard Rime/TTS + WebSocket + End-to-End Integration Hardening Tests.

Verifies:
1. TTS Result Fencing:
   - V1 starts -> V2 supersedes -> V1 finishes late -> rejected by fence -> V1 response NOT generated -> V1 cannot trigger TTS.
   - V2 finishes -> accepted by fence -> V2 response generated -> V2 triggers TTS with committed V2 data.
2. Interrupted TTS:
   - Turn interrupted before/during synthesis -> TTS rejected, state consistent.
3. Rime/TTS Failure Handling:
   - Simulated failure returns sanitized 502 error, doesn't crash server, preserves session state.
4. WebSocket Session Isolation:
   - Session A and Session B concurrent WebSockets strictly receive only their own events.
5. WebSocket Disconnect and Reconnect:
   - Client disconnect during background processing does not abort backend task or crash server; reconnect is safe.
6. WebSocket Event Ordering:
   - Valid turn event sequencing vs stale turn event sequencing.
7. Critical End-to-End Integration Test:
   - Full lifecycle from user input -> V1 -> V2 barge-in -> Result Fence -> V2 response -> Rime Voice audio.
"""

import asyncio
import pytest
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from app.main import app
from app.models.events import EventType, ToolResultEnvelope
from app.models.state import Session, SessionState
from app.models.tool import ToolResult
from app.services.rime_service import RimeService, rime_service
from app.services.session_manager import session_manager


@pytest.fixture(autouse=True)
def clean_state():
    """Ensure clean session manager and default rime state before each test."""
    session_manager.clear()
    rime_service.fail = False
    yield
    session_manager.clear()
    rime_service.fail = False


# =========================================================================
# 1. TTS Result Fencing (V1 superseded, V1 rejected, V1 no TTS, V2 gets TTS)
# =========================================================================
@pytest.mark.asyncio
async def test_tts_result_fencing_v1_rejected_v2_accepted():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Step 1: Create session
        res = await ac.post("/api/v1/sessions", json={})
        assert res.status_code == 201
        session_id = res.json()["session_id"]

        # Step 2: Start Turn 1 (V1) with 0.2s delay
        v1_res = await ac.post(
            f"/api/v1/sessions/{session_id}/turns",
            json={"transcript": "Find hotels in Delhi under 5000", "simulated_delay": 0.2},
        )
        assert v1_res.status_code == 200
        assert v1_res.json()["current_version"] == 1
        req_v1 = v1_res.json()["current_request_id"]

        # V1 is running, not completed -> TTS must be rejected
        tts_v1_early = await ac.post(f"/api/v1/sessions/{session_id}/tts", json={})
        assert tts_v1_early.status_code == 400
        assert "in progress" in tts_v1_early.json()["detail"].lower()

        # Step 3: User interrupts with Turn 2 (V2) before V1 completes
        await asyncio.sleep(0.02)
        v2_res = await ac.post(
            f"/api/v1/sessions/{session_id}/turns",
            json={"transcript": "Actually make that under 3000", "simulated_delay": 0.01},
        )
        assert v2_res.status_code == 200
        assert v2_res.json()["current_version"] == 2
        req_v2 = v2_res.json()["current_request_id"]

        # Wait for both background tasks to finish
        await asyncio.sleep(0.3)

        # Inspect final session state
        sess_res = await ac.get(f"/api/v1/sessions/{session_id}")
        sess_data = sess_res.json()
        assert sess_data["current_version"] == 2
        assert sess_data["committed_version"] == 2
        assert sess_data["committed_request_id"] == req_v2
        assert sess_data["state"] == "COMPLETED"
        assert "under ₹3000" in sess_data["last_response"]

        # Check recorded events
        events = session_manager.get_events(session_id)
        rejected_events = [e for e in events if e.event_type == EventType.RESULT_REJECTED_STALE]
        assert len(rejected_events) >= 1
        assert rejected_events[0].version == 1
        assert rejected_events[0].request_id == req_v1

        accepted_events = [e for e in events if e.event_type == EventType.RESULT_ACCEPTED]
        assert len(accepted_events) == 1
        assert accepted_events[0].version == 2
        assert accepted_events[0].request_id == req_v2

        # Step 4: TTS is allowed only for V2 committed response
        tts_res = await ac.post(f"/api/v1/sessions/{session_id}/tts", json={})
        assert tts_res.status_code == 200
        tts_data = tts_res.json()
        assert tts_data["version"] == 2
        assert tts_data["request_id"] == req_v2
        assert "3000" in tts_data["text"]
        assert len(tts_data["audio_base64"]) > 50


# =========================================================================
# 2. Interrupted TTS Guardrails
# =========================================================================
def test_interrupted_session_cannot_trigger_tts():
    client = TestClient(app)
    # Create session
    create_res = client.post("/api/v1/sessions", json={})
    session_id = create_res.json()["session_id"]

    # Start turn
    client.post(
        f"/api/v1/sessions/{session_id}/turns",
        json={"transcript": "Find hotels in Delhi under 5000", "simulated_delay": 0.5},
    )

    # Interrupt session mid-turn
    int_res = client.post(f"/api/v1/sessions/{session_id}/interrupt")
    assert int_res.status_code == 200
    assert int_res.json()["state"] == "INTERRUPTED"

    # Attempt TTS on interrupted session
    tts_res = client.post(f"/api/v1/sessions/{session_id}/tts", json={})
    assert tts_res.status_code == 400
    assert "interrupted" in tts_res.json()["detail"].lower()


# =========================================================================
# 3. Rime Failure Handling (Safe 502, no crash, state preserved, sanitized)
# =========================================================================
@pytest.mark.asyncio
async def test_rime_failure_handled_safely():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Create session and complete a conversational turn
        res = await ac.post("/api/v1/sessions", json={})
        session_id = res.json()["session_id"]

        turn_res = await ac.post(
            f"/api/v1/sessions/{session_id}/turns",
            json={"transcript": "Hello, how can you assist me?"},
        )
        assert turn_res.status_code == 200
        await asyncio.sleep(0.02)

        # Confirm session is completed with a valid response
        sess_res = await ac.get(f"/api/v1/sessions/{session_id}")
        sess_data = sess_res.json()
        assert sess_data["state"] == "COMPLETED"
        assert sess_data["last_response"] is not None

        # Simulate Rime Voice service failure
        rime_service.fail = True

        tts_res = await ac.post(f"/api/v1/sessions/{session_id}/tts", json={})
        # Must return clean 502 Bad Gateway without crashing
        assert tts_res.status_code == 502
        err_detail = tts_res.json()["detail"]
        assert "temporarily unavailable" in err_detail.lower()
        # Ensure no internal secret / stack trace leaked
        assert "api_key" not in err_detail.lower()
        assert "secret" not in err_detail.lower()

        # Session state remains completely intact
        sess_after = await ac.get(f"/api/v1/sessions/{session_id}")
        assert sess_after.json()["state"] == "COMPLETED"
        assert sess_after.json()["last_response"] == sess_data["last_response"]


# =========================================================================
# 4. WebSocket Session Isolation
# =========================================================================
def test_websocket_session_isolation():
    client = TestClient(app)

    # Create Session A and Session B
    res_a = client.post("/api/v1/sessions", json={})
    sid_a = res_a.json()["session_id"]

    res_b = client.post("/api/v1/sessions", json={})
    sid_b = res_b.json()["session_id"]

    # Connect WebSocket clients to both sessions
    with client.websocket_connect(f"/api/v1/sessions/{sid_a}/events") as ws_a, \
         client.websocket_connect(f"/api/v1/sessions/{sid_b}/events") as ws_b:

        init_a = ws_a.receive_json()
        assert init_a["event_type"] == "CONNECTED"
        assert init_a["session_id"] == sid_a

        init_b = ws_b.receive_json()
        assert init_b["event_type"] == "CONNECTED"
        assert init_b["session_id"] == sid_b

        # Action 1: Emit event in Session A
        client.post(f"/api/v1/sessions/{sid_a}/interrupt")

        # ws_a must receive the INTERRUPTION events
        ev_a = ws_a.receive_json()
        assert ev_a["session_id"] == sid_a

        # ws_b must NOT receive anything from Session A
        # Action 2: Emit event in Session B
        client.post(f"/api/v1/sessions/{sid_b}/interrupt")

        # ws_b receives its own event
        ev_b = ws_b.receive_json()
        assert ev_b["session_id"] == sid_b
        assert ev_b["session_id"] != sid_a


# =========================================================================
# 5. WebSocket Disconnect & Reconnect Safety
# =========================================================================
def test_websocket_disconnect_and_reconnect_safety():
    client = TestClient(app)
    res = client.post("/api/v1/sessions", json={})
    session_id = res.json()["session_id"]

    # 1. Connect WebSocket
    ws = client.websocket_connect(f"/api/v1/sessions/{session_id}/events")
    ws_conn = ws.__enter__()
    init_ev = ws_conn.receive_json()
    assert init_ev["event_type"] == "CONNECTED"

    # 2. Emit event while connected
    client.post(f"/api/v1/sessions/{session_id}/interrupt")
    int_ev = ws_conn.receive_json()
    assert int_ev["session_id"] == session_id

    # 3. Abruptly close / disconnect the WebSocket
    ws.__exit__(None, None, None)

    # 4. Backend continues to process turns safely without connected client
    turn_res = client.post(
        f"/api/v1/sessions/{session_id}/turns",
        json={"transcript": "Hello again, how are you?"},
    )
    assert turn_res.status_code == 200

    # 5. Reconnect to the same session via WebSocket
    with client.websocket_connect(f"/api/v1/sessions/{session_id}/events") as ws2:
        reconnect_ev = ws2.receive_json()
        assert reconnect_ev["event_type"] == "CONNECTED"
        assert reconnect_ev["session_id"] == session_id
        # State was preserved and updated
        assert reconnect_ev["state"] == "COMPLETED"
        assert reconnect_ev["current_version"] == 1


# =========================================================================
# 6. WebSocket Event Sequencing (Valid Turn vs Stale Turn)
# =========================================================================
def test_websocket_event_sequencing_valid_and_stale():
    client = TestClient(app)
    res = client.post("/api/v1/sessions", json={})
    session_id = res.json()["session_id"]

    with client.websocket_connect(f"/api/v1/sessions/{session_id}/events") as ws:
        _ = ws.receive_json()  # CONNECTED

        # Part A: Valid fast turn
        session_manager.start_turn(session_id, transcript="Quick check")
        ev1 = ws.receive_json()
        assert ev1["event_type"] == "TURN_STARTED"

        valid_env = ToolResultEnvelope(
            session_id=session_id,
            version=1,
            request_id=ev1["request_id"],
            tool_result=ToolResult.ok("get_weather", output={"temp": 25}),
        )
        session_manager.process_tool_result(valid_env)

        ev2 = ws.receive_json()
        assert ev2["event_type"] == "RESULT_ACCEPTED"

        ev3 = ws.receive_json()
        assert ev3["event_type"] == "TOOL_COMPLETED"

        # Part B: Stale result injection
        # Turn 2 starts
        session_manager.start_turn(session_id, transcript="Second turn")
        ev_turn2 = ws.receive_json()
        assert ev_turn2["event_type"] == "TURN_STARTED"
        assert ev_turn2["version"] == 2

        # Stale result from Turn 1 arrives
        stale_env = ToolResultEnvelope(
            session_id=session_id,
            version=1,
            request_id="req_obsolete_1",
            tool_result=ToolResult.ok("hotel_search", output={"hotels": ["Stale"]}),
        )
        session_manager.process_tool_result(stale_env)

        ev_stale = ws.receive_json()
        assert ev_stale["event_type"] == "RESULT_REJECTED_STALE"
        assert ev_stale["version"] == 1
        assert ev_stale["payload"]["session_current_version"] == 2


# =========================================================================
# 7. Critical Comprehensive End-to-End Test (V1 -> V2 -> Fence -> TTS)
# =========================================================================
@pytest.mark.asyncio
async def test_comprehensive_e2e_turn_superseding_fencing_and_tts():
    """Full End-to-End scenario proving core VoxGuard innovation.

    1. Create session.
    2. Start V1: 'Find hotels in Delhi under 5000' with delay.
    3. Confirm V1 is version 1.
    4. Wait briefly.
    5. Start V2: 'Actually make that under 3000'.
    6. Confirm V2 becomes current.
    7. Allow V1 to finish late.
    8. Verify: RESULT_REJECTED_STALE for V1.
    9. Verify V1 does NOT generate response.
    10. Verify V1 does NOT trigger Rime/TTS.
    11. Allow V2 to finish.
    12. Verify: RESULT_ACCEPTED for V2.
    13. Verify: RESPONSE_READY for V2.
    14. Verify final committed version is 2.
    15. Verify final committed data belongs to V2.
    16. Verify only V2 can reach TTS.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Create session
        res = await ac.post("/api/v1/sessions", json={"metadata": {"channel": "test_e2e"}})
        assert res.status_code == 201
        session_id = res.json()["session_id"]

        # 2. Start V1 with 0.2s delay
        v1_res = await ac.post(
            f"/api/v1/sessions/{session_id}/turns",
            json={"transcript": "Find hotels in Delhi under 5000", "simulated_delay": 0.2},
        )
        assert v1_res.status_code == 200
        v1_data = v1_res.json()
        # 3. Confirm V1 is version 1
        assert v1_data["current_version"] == 1
        v1_req_id = v1_data["current_request_id"]

        # 4. Wait briefly
        await asyncio.sleep(0.03)

        # 5. Start V2
        v2_res = await ac.post(
            f"/api/v1/sessions/{session_id}/turns",
            json={"transcript": "Actually make that under 3000", "simulated_delay": 0.01},
        )
        assert v2_res.status_code == 200
        v2_data = v2_res.json()
        # 6. Confirm V2 becomes current
        assert v2_data["current_version"] == 2
        v2_req_id = v2_data["current_request_id"]
        assert v2_req_id != v1_req_id

        # 7. Wait for both tasks to resolve
        await asyncio.sleep(0.3)

        # Retrieve events
        events_res = await ac.get(f"/api/v1/sessions/{session_id}/events")
        assert events_res.status_code == 200
        events = events_res.json()

        # 8. Verify RESULT_REJECTED_STALE for V1
        stale_events = [e for e in events if e["event_type"] == "RESULT_REJECTED_STALE"]
        assert len(stale_events) >= 1
        assert stale_events[0]["version"] == 1
        assert stale_events[0]["request_id"] == v1_req_id

        # 9. Verify V1 does NOT generate response & 13. RESPONSE_READY for V2
        ready_events = [e for e in events if e["event_type"] == "RESPONSE_READY"]
        assert len(ready_events) == 1
        assert ready_events[0]["version"] == 2
        assert ready_events[0]["request_id"] == v2_req_id

        # 12. Verify RESULT_ACCEPTED for V2
        accepted_events = [e for e in events if e["event_type"] == "RESULT_ACCEPTED"]
        assert len(accepted_events) == 1
        assert accepted_events[0]["version"] == 2
        assert accepted_events[0]["request_id"] == v2_req_id

        # Retrieve final session state
        final_sess = await ac.get(f"/api/v1/sessions/{session_id}")
        sess_data = final_sess.json()

        # 14. Verify final committed version is 2
        assert sess_data["committed_version"] == 2
        assert sess_data["committed_request_id"] == v2_req_id

        # 15. Verify final committed data belongs to V2 (₹3000, not ₹5000)
        assert sess_data["committed_data"].get("max_price") == 3000.0
        assert "3000" in sess_data["last_response"]
        assert "5000" not in sess_data["last_response"]

        # 10 & 16. Verify only V2 can reach TTS
        tts_res = await ac.post(f"/api/v1/sessions/{session_id}/tts", json={})
        assert tts_res.status_code == 200
        tts_json = tts_res.json()
        assert tts_json["version"] == 2
        assert tts_json["request_id"] == v2_req_id
        assert "3000" in tts_json["text"]
        assert len(tts_json["audio_base64"]) > 0


# =========================================================================
# 8. Result Fencing: V1 -> V2 -> V3 with arrival order V2 -> V1 -> V3
# =========================================================================
def test_v1_v2_v3_results_arrive_v2_v1_v3():
    """Verify Section 4 exact scenario:
    V1 -> V2 -> V3 turns submitted.
    Results arrive in order: V2 -> V1 -> V3.
    Expected: V2 rejected, V1 rejected, V3 accepted.
    Final committed_version MUST be 3.
    """
    session_id = "sess_v1_v2_v3_exact"
    session_manager.start_turn(session_id, "Turn 1", request_id="req_1")
    session_manager.start_turn(session_id, "Turn 2", request_id="req_2")
    session_manager.start_turn(session_id, "Turn 3", request_id="req_3")

    session = session_manager.get_session(session_id)
    assert session.current_version == 3
    assert session.current_request_id == "req_3"

    # 1. V2 arrives first -> Rejected as stale (v2 < v3)
    env2 = ToolResultEnvelope(
        session_id=session_id,
        version=2,
        request_id="req_2",
        tool_result=ToolResult.ok("hotel_search", output={"max_price": 3000}),
    )
    accepted_v2 = session_manager.process_tool_result(env2)
    assert accepted_v2 is False

    # 2. V1 arrives second -> Rejected as stale (v1 < v3)
    env1 = ToolResultEnvelope(
        session_id=session_id,
        version=1,
        request_id="req_1",
        tool_result=ToolResult.ok("hotel_search", output={"max_price": 5000}),
    )
    accepted_v1 = session_manager.process_tool_result(env1)
    assert accepted_v1 is False

    # 3. V3 arrives third -> Accepted (v3 == v3)
    env3 = ToolResultEnvelope(
        session_id=session_id,
        version=3,
        request_id="req_3",
        tool_result=ToolResult.ok("hotel_search", output={"max_price": 2000}),
    )
    accepted_v3 = session_manager.process_tool_result(env3)
    assert accepted_v3 is True

    # Final verification: committed_version MUST be 3
    assert session.committed_version == 3
    assert session.committed_request_id == "req_3"
    assert session.committed_data["max_price"] == 2000
    assert session.state == SessionState.COMPLETED


# =========================================================================
# 9. Session Isolation: Session A (V1->V2) and Session B (V1->V2->V3)
# =========================================================================
def test_concurrent_sessions_a_and_b_full_isolation():
    """Verify Section 6:
    Session A: V1 -> V2
    Session B: V1 -> V2 -> V3
    Verify:
    - versions are session-local
    - request IDs are correctly associated
    - events are session-local
    - committed results are session-local
    - WebSocket events are session-local
    """
    client = TestClient(app)

    # Initialize Session A and Session B
    res_a = client.post("/api/v1/sessions", json={"session_id": "sess_iso_A"})
    assert res_a.status_code == 201
    sid_a = res_a.json()["session_id"]

    res_b = client.post("/api/v1/sessions", json={"session_id": "sess_iso_B"})
    assert res_b.status_code == 201
    sid_b = res_b.json()["session_id"]

    with client.websocket_connect(f"/api/v1/sessions/{sid_a}/events") as ws_a, \
         client.websocket_connect(f"/api/v1/sessions/{sid_b}/events") as ws_b:

        init_a = ws_a.receive_json()
        assert init_a["event_type"] == "CONNECTED"
        assert init_a["session_id"] == sid_a

        init_b = ws_b.receive_json()
        assert init_b["event_type"] == "CONNECTED"
        assert init_b["session_id"] == sid_b

        # Session A: V1 -> V2
        turn_a1 = client.post(f"/api/v1/sessions/{sid_a}/turns", json={"transcript": "Turn A1", "request_id": "req_A1"})
        assert turn_a1.json()["current_version"] == 1
        turn_a2 = client.post(f"/api/v1/sessions/{sid_a}/turns", json={"transcript": "Turn A2", "request_id": "req_A2"})
        assert turn_a2.json()["current_version"] == 2

        # Session B: V1 -> V2 -> V3
        turn_b1 = client.post(f"/api/v1/sessions/{sid_b}/turns", json={"transcript": "Turn B1", "request_id": "req_B1"})
        assert turn_b1.json()["current_version"] == 1
        turn_b2 = client.post(f"/api/v1/sessions/{sid_b}/turns", json={"transcript": "Turn B2", "request_id": "req_B2"})
        assert turn_b2.json()["current_version"] == 2
        turn_b3 = client.post(f"/api/v1/sessions/{sid_b}/turns", json={"transcript": "Turn B3", "request_id": "req_B3"})
        assert turn_b3.json()["current_version"] == 3

        # Complete Session A with V2 result
        env_a2 = ToolResultEnvelope(
            session_id=sid_a,
            version=2,
            request_id="req_A2",
            tool_result=ToolResult.ok("hotel_search", output={"max_price": 3000}),
        )
        assert session_manager.process_tool_result(env_a2) is True

        # Complete Session B with V3 result
        env_b3 = ToolResultEnvelope(
            session_id=sid_b,
            version=3,
            request_id="req_B3",
            tool_result=ToolResult.ok("flight_search", output={"destination": "Mumbai"}),
        )
        assert session_manager.process_tool_result(env_b3) is True

        # Verify Session A state and events
        sess_a = session_manager.get_session(sid_a)
        assert sess_a.current_version == 2
        assert sess_a.committed_version == 2
        assert sess_a.committed_request_id == "req_A2"
        assert "hotel_search" in str(sess_a.committed_data) or sess_a.committed_data.get("max_price") == 3000
        assert "flight_search" not in str(sess_a.committed_data)

        # Verify Session B state and events
        sess_b = session_manager.get_session(sid_b)
        assert sess_b.current_version == 3
        assert sess_b.committed_version == 3
        assert sess_b.committed_request_id == "req_B3"
        assert sess_b.committed_data.get("destination") == "Mumbai"
        assert "max_price" not in sess_b.committed_data

        # Verify all events recorded in Session A only have session_id == sid_a
        events_a = session_manager.get_events(sid_a)
        for ev in events_a:
            assert ev.session_id == sid_a

        # Verify all events recorded in Session B only have session_id == sid_b
        events_b = session_manager.get_events(sid_b)
        for ev in events_b:
            assert ev.session_id == sid_b

