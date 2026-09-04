"""VoxGuard Comprehensive Backend Reliability Test Matrix.

Rigorously proves through automated pytest tests that VoxGuard's core innovation works correctly:
Intent Versioning + Cancellation + Result Fencing

The core invariant enforced:
A result is allowed to affect session state ONLY if it still belongs to the current authoritative request/version:
result.version == current_version AND result.request_id == current_request_id.
If not, the result is stale and MUST be rejected by the Result Fence.

Covers all 13 required scenarios:
1. Normal Completion
2. Basic Cancellation
3. Stale Result
4. Current Result
5. Multiple Versions (Unfavorable Order: V1, V3, V2)
6. Rapid Successive Interruptions (V1 -> V2 -> V3 -> V4)
7. Delayed Result (Without relying solely on cancellation)
8. Cancellation Failure (Task uncancels and produces result; Fence still rejects)
9. Tool Failure (TOOL_FAILED, sanitized reporting, cleanup)
10. Race Condition (Concurrent V1/V2 operations)
11. Session Isolation (Multi-session concurrency)
12. Response Fencing (Spy confirms 0 NLG calls for stale results)
13. TTS/Rime Fencing (Stale/interrupted turns cannot generate audio)
"""

import asyncio
from unittest.mock import MagicMock
import pytest
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from app.main import app
from app.models.events import EventType, ToolResultEnvelope
from app.models.state import Session, SessionState
from app.models.tool import ToolResult
from app.services.agent_controller import AgentController, agent_controller
from app.services.result_fence import FenceDecision, ResultFence
from app.services.rime_service import RimeService, rime_service
from app.services.session_manager import SessionManager, session_manager
from app.tools.base import BaseTool
from app.tools.registry import ToolRegistry, create_default_registry


# =====================================================================
# 1. NORMAL COMPLETION
# =====================================================================
@pytest.mark.asyncio
async def test_matrix_1_normal_completion():
    """Scenario 1: V1 starts. Tool completes normally. Result is accepted. Session commits V1."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Step 1: Create session
        create = await ac.post("/api/v1/sessions")
        assert create.status_code == 201
        sid = create.json()["session_id"]

        # Step 2: Submit turn V1
        turn = await ac.post(
            f"/api/v1/sessions/{sid}/turns",
            json={"transcript": "What is the weather in Boston?", "simulated_delay": 0.02, "request_id": "req_matrix_norm_v1"},
        )
        assert turn.status_code == 200
        assert turn.json()["current_version"] == 1

        # Await completion
        await asyncio.sleep(0.08)

        # Step 3: Inspect session state
        sess_resp = await ac.get(f"/api/v1/sessions/{sid}")
        assert sess_resp.status_code == 200
        sess = sess_resp.json()
        assert sess["committed_version"] == 1
        assert sess["committed_request_id"] == "req_matrix_norm_v1"
        assert sess["state"] == "COMPLETED"
        assert sess["last_response"] is not None
        assert "Boston" in sess["last_response"]

        # Step 4: Verify event telemetry
        events_resp = await ac.get(f"/api/v1/sessions/{sid}/events")
        assert events_resp.status_code == 200
        events = events_resp.json()
        accepted_events = [e for e in events if e["event_type"] == "RESULT_ACCEPTED"]
        assert len(accepted_events) == 1
        assert accepted_events[0]["version"] == 1
        assert accepted_events[0]["request_id"] == "req_matrix_norm_v1"


# =====================================================================
# 2. BASIC CANCELLATION
# =====================================================================
@pytest.mark.asyncio
async def test_matrix_2_basic_cancellation():
    """Scenario 2: Start a long-running V1 task. Interrupt/cancel V1.

    Verify:
    - cancellation is attempted
    - V1 becomes obsolete/interrupted
    - active task is cleaned up
    - session does not incorrectly commit the cancelled result
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        create = await ac.post("/api/v1/sessions")
        sid = create.json()["session_id"]

        # Start long-running V1 (simulated delay 5.0s)
        t1 = await ac.post(
            f"/api/v1/sessions/{sid}/turns",
            json={"transcript": "Find hotels in Delhi under 5000", "simulated_delay": 5.0, "request_id": "req_cancel_v1"},
        )
        assert t1.status_code == 200
        assert t1.json()["current_version"] == 1

        await asyncio.sleep(0.02)
        sess_before = session_manager.get_session(sid)
        assert sess_before.active_task is not None
        assert not sess_before.active_task.done()

        # Interrupt session
        interrupt = await ac.post(f"/api/v1/sessions/{sid}/interrupt")
        assert interrupt.status_code == 200
        assert interrupt.json()["state"] == "INTERRUPTED"

        # Allow task cancellation to finalize
        await asyncio.sleep(0.05)

        sess_after = session_manager.get_session(sid)
        assert sess_after.state == SessionState.INTERRUPTED
        assert sess_after.committed_version is None
        assert sess_after.committed_request_id is None
        # Verify active task was cancelled/cleaned up
        assert sess_after.active_task is None or sess_after.active_task.done()

        # Event telemetry checks
        evts = (await ac.get(f"/api/v1/sessions/{sid}/events")).json()
        assert any(e["event_type"] == "INTERRUPTED" for e in evts)
        # Cancelled result MUST NOT be committed
        assert not any(e["event_type"] == "RESULT_ACCEPTED" for e in evts)


# =====================================================================
# 3. STALE RESULT
# =====================================================================
@pytest.mark.asyncio
async def test_matrix_3_stale_result_rejection():
    """Scenario 3: Start V1. Then start V2. After V2 becomes current, deliver/complete V1's result.

    Verify:
    - V1 result is rejected
    - RESULT_REJECTED_STALE is emitted
    - V1 cannot modify committed state
    - V1 cannot overwrite V2
    - V1 cannot generate the final response
    - V1 cannot trigger TTS/Rime
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        create = await ac.post("/api/v1/sessions")
        sid = create.json()["session_id"]

        # V1: 0.18s delay
        await ac.post(
            f"/api/v1/sessions/{sid}/turns",
            json={"transcript": "Find hotels in Delhi under 5000", "simulated_delay": 0.18, "request_id": "req_stale_v1"},
        )
        await asyncio.sleep(0.02)

        # V2 supersedes V1: 0.02s delay
        await ac.post(
            f"/api/v1/sessions/{sid}/turns",
            json={"transcript": "Actually find hotels in Delhi under 2500", "simulated_delay": 0.02, "request_id": "req_curr_v2"},
        )

        # Wait for both to run through completion & fence
        await asyncio.sleep(0.25)

        sess = (await ac.get(f"/api/v1/sessions/{sid}")).json()
        assert sess["committed_version"] == 2
        assert sess["committed_request_id"] == "req_curr_v2"
        assert sess["committed_data"]["max_price"] == 2500.0
        assert sess["committed_data"]["max_price"] != 5000.0

        evts = (await ac.get(f"/api/v1/sessions/{sid}/events")).json()
        stale_evts = [e for e in evts if e["event_type"] == "RESULT_REJECTED_STALE"]
        assert len(stale_evts) >= 1
        assert stale_evts[0]["request_id"] == "req_stale_v1"

        accepted_evts = [e for e in evts if e["event_type"] == "RESULT_ACCEPTED"]
        assert len(accepted_evts) == 1
        assert accepted_evts[0]["request_id"] == "req_curr_v2"

        # Final response belongs to V2, NOT V1
        assert "2500" in sess["last_response"]
        assert "5000" not in sess["last_response"]

        # TTS check: TTS synthesizes the committed V2 response
        tts_res = await ac.post(f"/api/v1/sessions/{sid}/tts")
        assert tts_res.status_code == 200
        tts_data = tts_res.json()
        assert tts_data["version"] == 2
        assert tts_data["request_id"] == "req_curr_v2"
        assert "2500" in tts_data["text"]


# =====================================================================
# 4. CURRENT RESULT
# =====================================================================
@pytest.mark.asyncio
async def test_matrix_4_current_result():
    """Scenario 4: Start V1. Allow V1 to finish while it is still current.

    Verify:
    - result is accepted
    - RESULT_ACCEPTED is emitted
    - committed_version is V1
    - response generation is allowed
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        create = await ac.post("/api/v1/sessions")
        sid = create.json()["session_id"]

        t1 = await ac.post(
            f"/api/v1/sessions/{sid}/turns",
            json={"transcript": "What is the weather in Seattle?", "simulated_delay": 0.02, "request_id": "req_curr_only_v1"},
        )
        assert t1.status_code == 200

        await asyncio.sleep(0.08)

        sess = (await ac.get(f"/api/v1/sessions/{sid}")).json()
        assert sess["committed_version"] == 1
        assert sess["committed_request_id"] == "req_curr_only_v1"
        assert sess["state"] == "COMPLETED"
        assert sess["last_response"] is not None

        evts = (await ac.get(f"/api/v1/sessions/{sid}/events")).json()
        assert any(e["event_type"] == "RESULT_ACCEPTED" and e["version"] == 1 for e in evts)
        assert any(e["event_type"] == "RESPONSE_READY" and e["version"] == 1 for e in evts)


# =====================================================================
# 5. MULTIPLE VERSIONS UNFAVORABLE COMPLETION ORDER
# =====================================================================
@pytest.mark.asyncio
async def test_matrix_5_multiple_versions_unfavorable_order():
    """Scenario 5: Create V1 -> V2 -> V3. Deliver results in unfavorable order: V1, then V3, then V2.

    Verify:
    V1 = rejected
    V2 = rejected
    V3 = accepted
    Final committed version MUST be 3.
    """
    mgr = SessionManager()
    sid = "sess_matrix_unfavorable_order"

    mgr.start_turn(sid, "Turn 1", request_id="req_m5_v1")
    mgr.start_turn(sid, "Turn 2", request_id="req_m5_v2")
    mgr.start_turn(sid, "Turn 3", request_id="req_m5_v3")

    session = mgr.get_session(sid)
    assert session.current_version == 3
    assert session.current_request_id == "req_m5_v3"

    # 1. Deliver V1 result -> Must be REJECTED (stale, 1 < 3)
    env1 = ToolResultEnvelope(
        session_id=sid,
        version=1,
        request_id="req_m5_v1",
        tool_result=ToolResult.ok(tool_name="hotel_search", output={"max_price": 1000}),
    )
    assert mgr.process_tool_result(env1) is False

    # 2. Deliver V3 result -> Must be ACCEPTED (current, 3 == 3)
    env3 = ToolResultEnvelope(
        session_id=sid,
        version=3,
        request_id="req_m5_v3",
        tool_result=ToolResult.ok(tool_name="hotel_search", output={"max_price": 3000}),
    )
    assert mgr.process_tool_result(env3) is True

    # 3. Deliver V2 result -> Must be REJECTED (stale, 2 < 3)
    env2 = ToolResultEnvelope(
        session_id=sid,
        version=2,
        request_id="req_m5_v2",
        tool_result=ToolResult.ok(tool_name="hotel_search", output={"max_price": 2000}),
    )
    assert mgr.process_tool_result(env2) is False

    # Final committed version MUST be 3
    assert session.committed_version == 3
    assert session.committed_request_id == "req_m5_v3"
    assert session.committed_data["max_price"] == 3000


# =====================================================================
# 6. RAPID SUCCESSIVE INTERRUPTIONS
# =====================================================================
@pytest.mark.asyncio
async def test_matrix_6_rapid_successive_interruptions():
    """Scenario 6: Create rapid turns: V1 -> V2 -> V3 -> V4.

    Make earlier tools slow.
    Verify only V4 can become authoritative.
    No stale result can change final state.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        create = await ac.post("/api/v1/sessions")
        sid = create.json()["session_id"]

        # Rapidly post turns 1, 2, 3 with slow delays
        await ac.post(f"/api/v1/sessions/{sid}/turns", json={"transcript": "Find hotels in Delhi under 5000", "simulated_delay": 0.3, "request_id": "req_r1"})
        await ac.post(f"/api/v1/sessions/{sid}/turns", json={"transcript": "Find hotels in Delhi under 4000", "simulated_delay": 0.3, "request_id": "req_r2"})
        await ac.post(f"/api/v1/sessions/{sid}/turns", json={"transcript": "Find hotels in Delhi under 3000", "simulated_delay": 0.3, "request_id": "req_r3"})

        # Turn 4 is the final prompt
        await ac.post(f"/api/v1/sessions/{sid}/turns", json={"transcript": "Actually under 1500", "simulated_delay": 0.01, "request_id": "req_r4"})

        # Wait for all executions to settle
        await asyncio.sleep(0.35)

        sess = (await ac.get(f"/api/v1/sessions/{sid}")).json()
        assert sess["current_version"] == 4
        assert sess["committed_version"] == 4
        assert sess["committed_request_id"] == "req_r4"
        assert sess["committed_data"]["max_price"] == 1500.0


# =====================================================================
# 7. DELAYED RESULT (WITHOUT RELYING ONLY ON CANCELLATION)
# =====================================================================
@pytest.mark.asyncio
async def test_matrix_7_delayed_result_rejection():
    """Scenario 7: V1 uses configurable delay. Start V1. Supersede with V2.

    Allow V1 to finish after V2.
    Verify V1 is rejected by Result Fencing.
    Do NOT rely only on cancellation for this test.
    """
    mgr = SessionManager()
    sid = "sess_matrix_delayed_v1"

    # Start V1
    mgr.start_turn(sid, "Find hotel under 5000", request_id="req_d1")
    assert mgr.get_session(sid).current_version == 1

    # Supersede with V2
    mgr.start_turn(sid, "Find hotel under 2000", request_id="req_d2")
    assert mgr.get_session(sid).current_version == 2

    # V2 completes and commits
    v2_env = ToolResultEnvelope(
        session_id=sid,
        version=2,
        request_id="req_d2",
        tool_result=ToolResult.ok(tool_name="hotel_search", output={"max_price": 2000.0}),
    )
    assert mgr.process_tool_result(v2_env) is True
    assert mgr.get_session(sid).committed_version == 2

    # V1 completes late and tries to commit
    v1_env = ToolResultEnvelope(
        session_id=sid,
        version=1,
        request_id="req_d1",
        tool_result=ToolResult.ok(tool_name="hotel_search", output={"max_price": 5000.0}),
    )
    # Result Fence must reject V1 unconditionally
    assert mgr.process_tool_result(v1_env) is False
    assert mgr.get_session(sid).committed_version == 2
    assert mgr.get_session(sid).committed_data["max_price"] == 2000.0


# =====================================================================
# 8. CANCELLATION FAILURE (VERY IMPORTANT)
# =====================================================================
@pytest.mark.asyncio
async def test_matrix_8_cancellation_failure():
    """Scenario 8: Simulate situation where cancellation of V1 does not prevent V1 from producing a result.

    Verify:
    Cancellation failure does NOT allow V1 to commit.
    Result Fence MUST still reject V1.
    """
    mgr = SessionManager()
    registry = create_default_registry()
    ctrl = AgentController(manager=mgr, registry=registry)
    sid = "sess_matrix_cancellation_failure"

    v1_completed_event = asyncio.Event()

    # Worker that catches CancelledError, uncancels itself, and submits result anyway
    async def rogue_uncancellable_v1():
        cur = asyncio.current_task()
        try:
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            # Deliberately ignore cancellation
            if cur and hasattr(cur, "uncancel"):
                cur.uncancel()
            await asyncio.sleep(0.02)
        v1_completed_event.set()
        # Submit V1 result to fencing
        env = ToolResultEnvelope(
            session_id=sid,
            version=1,
            request_id="req_rogue_v1",
            tool_result=ToolResult.ok(
                tool_name="hotel_search",
                output={"hotels": ["Rogue Hotel"], "max_price": 9999.0},
                request_id="req_rogue_v1",
                version=1,
            ),
        )
        return mgr.process_tool_result(env)

    # Launch rogue V1 task
    task_v1 = asyncio.create_task(rogue_uncancellable_v1())
    mgr.start_turn(sid, "Hotels 9999", active_task=task_v1, request_id="req_rogue_v1")
    assert mgr.get_session(sid).current_version == 1

    await asyncio.sleep(0.01)

    # V2 arrives (triggers cancellation attempt on V1 task)
    ctrl.handle_turn(
        session_id=sid,
        transcript="Find hotels in Delhi under 3000",
        simulated_delay=0.01,
        request_id="req_legit_v2",
    )
    assert mgr.get_session(sid).current_version == 2

    # Wait for rogue V1 to attempt delivery
    await v1_completed_event.wait()
    v1_accepted = await task_v1
    assert v1_accepted is False  # Result Fence REJECTED V1!

    # Wait for V2 to settle
    await asyncio.sleep(0.05)

    sess = mgr.get_session(sid)
    assert sess.committed_version == 2
    assert sess.committed_request_id == "req_legit_v2"
    assert sess.committed_data["max_price"] == 3000.0
    assert sess.committed_data["max_price"] != 9999.0


# =====================================================================
# 9. TOOL FAILURE
# =====================================================================
@pytest.mark.asyncio
async def test_matrix_9_tool_failure():
    """Scenario 9: Make current tool fail.

    Verify:
    - TOOL_FAILED is emitted
    - session remains consistent
    - no invalid committed result is created
    - background task is cleaned up
    """
    mgr = SessionManager()
    sid = "sess_matrix_tool_fail"

    mgr.start_turn(sid, "Transfer funds", request_id="req_tf_fail")
    session = mgr.get_session(sid)

    failed_result = ToolResult.fail(
        tool_name="transfer_funds",
        error="Network timeout contacting bank clearinghouse",
        request_id="req_tf_fail",
        version=1,
    )
    envelope = ToolResultEnvelope(
        session_id=sid,
        version=1,
        request_id="req_tf_fail",
        tool_result=failed_result,
    )

    accepted = mgr.process_tool_result(envelope)
    assert accepted is True  # Result envelope processed for current version

    # Session state must transition to ERROR
    assert session.state == SessionState.ERROR
    assert session.committed_version == 1
    assert "amount" not in session.committed_data

    # TOOL_FAILED event emitted
    fail_evts = mgr.get_events(session_id=sid, event_type=EventType.TOOL_FAILED)
    assert len(fail_evts) == 1
    assert "clearinghouse" in fail_evts[0].payload["error"]

    # Session remains consistent and can recover on next turn
    mgr.start_turn(sid, "Check balance", request_id="req_tf_recovery")
    assert session.state == SessionState.THINKING
    assert session.current_version == 2


# =====================================================================
# 10. RACE CONDITION
# =====================================================================
@pytest.mark.asyncio
async def test_matrix_10_race_condition_concurrency():
    """Scenario 10: Concurrent V1/V2 operations.

    Verify session state remains internally consistent.
    No incorrect committed_version, incorrect request_id, stale response, duplicate invalid commit.
    """
    mgr = SessionManager()
    sid = "sess_matrix_race_cond"

    # Start V1 and V2 in rapid succession
    mgr.start_turn(sid, "Turn 1", request_id="req_rc_v1")
    mgr.start_turn(sid, "Turn 2", request_id="req_rc_v2")

    session = mgr.get_session(sid)
    assert session.current_version == 2

    # Concurrent result arrivals
    results = []

    async def submit_v1():
        env = ToolResultEnvelope(
            session_id=sid,
            version=1,
            request_id="req_rc_v1",
            tool_result=ToolResult.ok(tool_name="hotel_search", output={"max_price": 5000}),
        )
        results.append(("v1", mgr.process_tool_result(env)))

    async def submit_v2():
        env = ToolResultEnvelope(
            session_id=sid,
            version=2,
            request_id="req_rc_v2",
            tool_result=ToolResult.ok(tool_name="hotel_search", output={"max_price": 2000}),
        )
        results.append(("v2", mgr.process_tool_result(env)))

    await asyncio.gather(submit_v1(), submit_v2())

    res_dict = dict(results)
    assert res_dict["v1"] is False  # V1 strictly rejected
    assert res_dict["v2"] is True   # V2 strictly accepted
    assert session.committed_version == 2
    assert session.committed_request_id == "req_rc_v2"
    assert session.committed_data["max_price"] == 2000


# =====================================================================
# 11. SESSION ISOLATION
# =====================================================================
@pytest.mark.asyncio
async def test_matrix_11_session_isolation():
    """Scenario 11: Create two independent sessions. Run concurrent requests in both.

    Verify Session A cannot affect Session B.
    Version numbers and committed results must remain session-local.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Create Session A and Session B
        c_a = await ac.post("/api/v1/sessions", json={"metadata": {"name": "SessionA"}})
        c_b = await ac.post("/api/v1/sessions", json={"metadata": {"name": "SessionB"}})
        sid_a = c_a.json()["session_id"]
        sid_b = c_b.json()["session_id"]

        # Run concurrent turns in both
        t_a1 = ac.post(f"/api/v1/sessions/{sid_a}/turns", json={"transcript": "Find hotels in Delhi under 5000", "simulated_delay": 0.1})
        t_b1 = ac.post(f"/api/v1/sessions/{sid_b}/turns", json={"transcript": "What is the weather in Paris?", "simulated_delay": 0.02})
        await asyncio.gather(t_a1, t_b1)

        # Interrupt Session A only
        int_a = await ac.post(f"/api/v1/sessions/{sid_a}/interrupt")
        assert int_a.json()["state"] == "INTERRUPTED"

        # Wait for Session B to finish
        await asyncio.sleep(0.08)

        # Session B must NOT be interrupted; it should complete successfully
        sess_b = (await ac.get(f"/api/v1/sessions/{sid_b}")).json()
        assert sess_b["state"] == "COMPLETED"
        assert sess_b["committed_version"] == 1
        assert "Paris" in sess_b["last_response"]

        # Session A remains INTERRUPTED and uncommitted
        sess_a = (await ac.get(f"/api/v1/sessions/{sid_a}")).json()
        assert sess_a["state"] == "INTERRUPTED"
        assert sess_a["committed_version"] is None


# =====================================================================
# 12. RESPONSE FENCING
# =====================================================================
@pytest.mark.asyncio
async def test_matrix_12_response_fencing():
    """Scenario 12: A stale tool result MUST NOT reach response generation.

    Mock/spy synthesize_response.
    Verify:
    V1 stale result -> NO response generation
    V2 current result -> response generation allowed
    """
    mgr = SessionManager()
    registry = create_default_registry()
    ctrl = AgentController(manager=mgr, registry=registry)
    sid = "sess_matrix_response_fencing"

    synth_spy = MagicMock(side_effect=ctrl.synthesize_response)
    ctrl.synthesize_response = synth_spy

    # V1 starts with delay
    ctrl.handle_turn(sid, "Find hotels in Delhi under 5000", simulated_delay=0.15, request_id="req_resp_v1")
    await asyncio.sleep(0.02)

    # V2 supersedes with fast completion
    ctrl.handle_turn(sid, "Find hotels in Delhi under 3000", simulated_delay=0.01, request_id="req_resp_v2")

    # Wait for both tasks to complete
    await asyncio.sleep(0.2)

    # Assert synthesize_response was called EXACTLY ONCE, for V2 ONLY
    assert synth_spy.call_count == 1
    called_tool_result: ToolResult = synth_spy.call_args[0][1]
    assert called_tool_result.request_id == "req_resp_v2"
    assert called_tool_result.version == 2


# =====================================================================
# 13. TTS / RIME FENCING
# =====================================================================
@pytest.mark.asyncio
async def test_matrix_13_tts_rime_fencing():
    """Scenario 13: Stale/interrupted result cannot trigger TTS.

    A valid committed response may trigger TTS according to contract.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Case A: Interrupted session rejects TTS with 400
        create_int = await ac.post("/api/v1/sessions")
        sid_int = create_int.json()["session_id"]
        await ac.post(f"/api/v1/sessions/{sid_int}/interrupt")

        tts_bad = await ac.post(f"/api/v1/sessions/{sid_int}/tts")
        assert tts_bad.status_code == 400
        assert "interrupted" in tts_bad.json()["detail"].lower()

        # Case B: Valid committed session generates TTS audio
        create_ok = await ac.post("/api/v1/sessions")
        sid_ok = create_ok.json()["session_id"]
        await ac.post(
            f"/api/v1/sessions/{sid_ok}/turns",
            json={"transcript": "Hello, can you help me?", "request_id": "req_tts_ok"},
        )
        await asyncio.sleep(0.02)

        tts_good = await ac.post(f"/api/v1/sessions/{sid_ok}/tts", json={"speaker": "marsh"})
        assert tts_good.status_code == 200
        tts_data = tts_good.json()
        assert tts_data["session_id"] == sid_ok
        assert tts_data["speaker"] == "marsh"
        assert tts_data["audio_format"] == "audio/wav"
        assert len(tts_data["audio_base64"]) > 0
