"""VoxGuard Production Reliability and Hardening Tests (Phase 7).

Verifies backend reliability under:
- Cancellation edge cases (clean cancel, exception during cancel, completed task, ignored cancellation, multi-version cancellations)
- Version race conditions and out-of-order completions
- Exception isolation and sanitized API error reporting
- Task lifecycle and memory management
- Multi-session isolation
- API reliability under concurrency and invalid inputs
- Telemetry observability and secrets hygiene
- Critical Delhi hotels interruption stress scenario
- Concurrency performance sanity check
"""

import asyncio
import uuid
import pytest
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from app.main import app
from app.models.events import EventType, ToolResultEnvelope
from app.models.state import Session, SessionState
from app.models.tool import ToolResult
from app.services.agent_controller import AgentController
from app.services.result_fence import FenceDecision, ResultFence
from app.services.session_manager import SessionManager
from app.tools.base import BaseTool
from app.tools.registry import ToolRegistry


# --- Custom Mock Tools for Reliability Edge Cases ---

class ResilientIgnoringCancelTool(BaseTool):
    """Tool that catches CancelledError, uncancels, finishes execution late."""

    name = "resilient_ignoring_cancel"
    description = "A tool that ignores cancellation and completes late."

    def __init__(self, completion_delay: float = 0.08) -> None:
        super().__init__()
        self.completion_delay = completion_delay
        self.executed_to_completion = False

    async def execute(self, **kwargs) -> ToolResult:
        cur_task = asyncio.current_task()
        try:
            await asyncio.sleep(self.completion_delay)
        except asyncio.CancelledError:
            if cur_task and hasattr(cur_task, "uncancel"):
                cur_task.uncancel()
            # Finish late anyway
            await asyncio.sleep(0.04)
        self.executed_to_completion = True
        return ToolResult.ok(
            tool_name=self.name,
            output={"status": "completed_late", "data": "late_payload"},
        )


class ExplodingTool(BaseTool):
    """Tool that raises an unexpected runtime exception."""

    name = "exploding_tool"
    description = "A tool that always raises an unhandled exception."

    async def execute(self, **kwargs) -> ToolResult:
        raise RuntimeError("Severe tool database disconnection failure")


# =====================================================================
# 1. CANCELLATION ROBUSTNESS
# =====================================================================

@pytest.mark.asyncio
async def test_cancellation_robustness_clean_cancel():
    """Case A: Tool cancels cleanly when superseding turn arrives."""
    manager = SessionManager()
    controller = AgentController(manager=manager)
    sess_id = f"sess_clean_cancel_{uuid.uuid4().hex[:8]}"

    controller.handle_turn(sess_id, "Find hotels under 5000", simulated_delay=0.3)
    sess = manager.get_session(sess_id)
    assert sess.current_version == 1
    assert sess.active_task is not None

    # Supersede with V2
    await asyncio.sleep(0.02)
    controller.handle_turn(sess_id, "Actually under 3000", simulated_delay=0.0)

    assert sess.current_version == 2
    events = manager.get_events(sess_id)
    cancel_events = [e for e in events if e.event_type == EventType.CANCELLATION_REQUESTED]
    assert len(cancel_events) >= 1
    assert cancel_events[0].payload["superseded_version"] == 1


@pytest.mark.asyncio
async def test_cancellation_robustness_exception_during_cancel():
    """Case B: If task cancellation raises an exception, Result Fence remains final authority."""
    manager = SessionManager()
    sess_id = f"sess_exc_cancel_{uuid.uuid4().hex[:8]}"
    session = manager.get_or_create_session(sess_id)

    class FlakyTask:
        def done(self):
            return False
        def cancel(self):
            raise RuntimeError("Task cancellation hook failed")

    session.active_task = FlakyTask()
    # cancel_active_task catches exception and returns False without crashing
    cancelled = session.cancel_active_task()
    assert cancelled is False

    # Now verify Result Fence still protects against stale results
    session.current_version = 2
    session.current_request_id = "req_v2"

    stale_envelope = ToolResultEnvelope(
        session_id=sess_id,
        version=1,
        request_id="req_v1",
        tool_result=ToolResult.ok(tool_name="hotel_search", output={"data": "v1"}),
    )
    assert manager.process_tool_result(stale_envelope) is False
    assert session.committed_version is None


@pytest.mark.asyncio
async def test_cancellation_robustness_already_completed():
    """Case C: Task has already completed when cancellation is requested."""
    manager = SessionManager()
    sess_id = f"sess_already_done_{uuid.uuid4().hex[:8]}"
    session = manager.get_or_create_session(sess_id)

    async def quick_work():
        return 42

    task = asyncio.create_task(quick_work())
    await task
    session.active_task = task
    assert task.done() is True

    # cancel_active_task detects it is done, clears active_task, and returns False
    res = session.cancel_active_task()
    assert res is False
    assert session.active_task is None


@pytest.mark.asyncio
async def test_cancellation_robustness_tool_ignores_cancellation():
    """Case D: Tool actively ignores cancellation and finishes late.
    Result Fence MUST reject it and preserve active session state.
    """
    manager = SessionManager()
    registry = ToolRegistry()
    resilient_tool = ResilientIgnoringCancelTool(completion_delay=0.04)
    registry.register(resilient_tool)
    controller = AgentController(manager=manager, registry=registry)
    sess_id = f"sess_ignore_cancel_{uuid.uuid4().hex[:8]}"

    # V1 starts resilient tool
    controller.handle_turn(sess_id, "Find hotels under 5000")
    # Manually associate resilient tool
    task1 = asyncio.create_task(registry.execute_async("resilient_ignoring_cancel"))
    manager.set_task(sess_id, task1)

    # Immediately supersede with V2
    await asyncio.sleep(0.01)
    controller.handle_turn(sess_id, "Actually under 3000")
    sess = manager.get_session(sess_id)
    assert sess.current_version == 2

    # Wait for the resilient task to finish late despite cancellation
    raw_res = await task1
    assert resilient_tool.executed_to_completion is True

    # Submit the late V1 result
    envelope = ToolResultEnvelope(
        session_id=sess_id,
        version=1,
        request_id="req_v1_stale",
        tool_result=raw_res,
    )
    accepted = manager.process_tool_result(envelope)
    assert accepted is False
    assert sess.state != SessionState.COMPLETED
    assert "data" not in sess.committed_data

    events = manager.get_events(sess_id, event_type=EventType.RESULT_REJECTED_STALE)
    assert len(events) >= 1
    assert events[0].payload["result_version"] == 1


@pytest.mark.asyncio
async def test_cancellation_robustness_multiple_old_tasks_cancelled():
    """Case E: Multiple old tasks are cancelled when several newer versions arrive in rapid succession."""
    manager = SessionManager()
    controller = AgentController(manager=manager)
    sess_id = f"sess_multi_cancel_{uuid.uuid4().hex[:8]}"

    # Rapid succession: V1 -> V2 -> V3
    controller.handle_turn(sess_id, "Find hotels under 5000", simulated_delay=0.3)
    controller.handle_turn(sess_id, "Find hotels under 4000", simulated_delay=0.3)
    controller.handle_turn(sess_id, "Find hotels under 3000", simulated_delay=0.0)

    sess = manager.get_session(sess_id)
    assert sess.current_version == 3

    await asyncio.sleep(0.1)
    # Both V1 and V2 are stale
    events = manager.get_events(sess_id)
    cancel_events = [e for e in events if e.event_type == EventType.CANCELLATION_REQUESTED]
    assert len(cancel_events) >= 2


# =====================================================================
# 2. VERSION RACE PROTECTION
# =====================================================================

@pytest.mark.asyncio
async def test_version_race_out_of_order_completions():
    """V1 starts, V2 starts, V3 starts.
    Finish order: V1 finishes, V3 finishes, V2 finishes.
    Only V3 result mutates conversation state or triggers response generation.
    """
    manager = SessionManager()
    sess_id = f"sess_ooo_{uuid.uuid4().hex[:8]}"
    session = manager.get_or_create_session(sess_id)

    # Simulate turns advancing to version 3
    manager.start_turn(sess_id, "Turn 1", request_id="req_1")
    manager.start_turn(sess_id, "Turn 2", request_id="req_2")
    manager.start_turn(sess_id, "Turn 3", request_id="req_3")
    assert session.current_version == 3

    # V1 completes first
    env1 = ToolResultEnvelope(
        session_id=sess_id,
        version=1,
        request_id="req_1",
        tool_result=ToolResult.ok("hotel_search", {"result": "v1_data"}),
    )
    assert manager.process_tool_result(env1) is False

    # V3 completes second (current version)
    env3 = ToolResultEnvelope(
        session_id=sess_id,
        version=3,
        request_id="req_3",
        tool_result=ToolResult.ok("hotel_search", {"result": "v3_data"}),
    )
    assert manager.process_tool_result(env3) is True
    assert session.committed_version == 3
    assert session.committed_data["result"] == "v3_data"

    # V2 completes third (late)
    env2 = ToolResultEnvelope(
        session_id=sess_id,
        version=2,
        request_id="req_2",
        tool_result=ToolResult.ok("hotel_search", {"result": "v2_data"}),
    )
    assert manager.process_tool_result(env2) is False

    # Verify committed state strictly reflects V3
    assert session.committed_version == 3
    assert session.committed_data["result"] == "v3_data"


@pytest.mark.asyncio
async def test_rapid_version_progression_no_overwrite():
    """V1 -> V2 -> V3 -> V4 rapidly.
    No older version can overwrite newest valid state.
    """
    manager = SessionManager()
    sess_id = f"sess_rapid_{uuid.uuid4().hex[:8]}"

    for i in range(1, 5):
        manager.start_turn(sess_id, f"Turn {i}", request_id=f"req_{i}")

    session = manager.get_session(sess_id)
    assert session.current_version == 4

    # Try to process V1, V2, V3
    for v in [1, 2, 3]:
        env = ToolResultEnvelope(
            session_id=sess_id,
            version=v,
            request_id=f"req_{v}",
            tool_result=ToolResult.ok("hotel_search", {"v": v}),
        )
        assert manager.process_tool_result(env) is False

    # Process V4
    env4 = ToolResultEnvelope(
        session_id=sess_id,
        version=4,
        request_id="req_4",
        tool_result=ToolResult.ok("hotel_search", {"v": 4}),
    )
    assert manager.process_tool_result(env4) is True
    assert session.committed_version == 4
    assert session.committed_data["v"] == 4


# =====================================================================
# 3. EXCEPTION ISOLATION
# =====================================================================

@pytest.mark.asyncio
async def test_exception_isolation_tool_failed_vs_cancelled():
    """Verify TOOL_FAILED is distinct from TOOL_CANCELLED."""
    manager = SessionManager()
    sess_id = f"sess_iso_{uuid.uuid4().hex[:8]}"
    manager.get_or_create_session(sess_id)

    # 1. Cancelled result
    env_cancel = ToolResultEnvelope(
        session_id=sess_id,
        version=0,
        request_id="req_c",
        tool_result=ToolResult.cancelled("hotel_search", "Cancelled by user"),
    )
    manager.process_tool_result(env_cancel)

    # 2. Failed result
    env_fail = ToolResultEnvelope(
        session_id=sess_id,
        version=0,
        request_id="req_f",
        tool_result=ToolResult.fail("hotel_search", "API 500 error"),
    )
    manager.process_tool_result(env_fail)

    events = manager.get_events(sess_id)
    types = [e.event_type for e in events]
    assert EventType.TOOL_CANCELLED in types
    assert EventType.TOOL_FAILED in types


@pytest.mark.asyncio
async def test_exception_isolation_failed_tool_does_not_crash_manager():
    """Verify tool raising unhandled exception does not crash session manager or controller."""
    manager = SessionManager()
    registry = ToolRegistry()
    registry.register(ExplodingTool())
    controller = AgentController(manager=manager, registry=registry)
    sess_id = f"sess_explode_{uuid.uuid4().hex[:8]}"

    # Submit turn that will invoke exploding_tool
    manager.start_turn(sess_id, "Test exploding tool")
    session = manager.get_session(sess_id)

    # Execute directly via controller
    res = await registry.execute_async("exploding_tool")
    assert res.success is False
    assert "disconnection" in res.error

    envelope = ToolResultEnvelope(
        session_id=sess_id,
        version=session.current_version,
        request_id=session.current_request_id,
        tool_result=res,
    )
    accepted = manager.process_tool_result(envelope)
    assert accepted is True
    assert session.state == SessionState.ERROR


@pytest.mark.asyncio
async def test_exception_isolation_new_request_continues_after_tool_failure():
    """Verify a newer request can succeed normally after an earlier tool failed."""
    manager = SessionManager()
    controller = AgentController(manager=manager)
    sess_id = f"sess_recovery_{uuid.uuid4().hex[:8]}"

    # Turn 1 fails
    manager.start_turn(sess_id, "Turn 1", request_id="req_1")
    fail_env = ToolResultEnvelope(
        session_id=sess_id,
        version=1,
        request_id="req_1",
        tool_result=ToolResult.fail("hotel_search", "Temporary service outage"),
    )
    manager.process_tool_result(fail_env)
    sess = manager.get_session(sess_id)
    assert sess.state == SessionState.ERROR

    # Turn 2 initiates and recovers
    controller.handle_turn(sess_id, "Find hotels in Delhi under 3000", simulated_delay=0.0)
    await asyncio.sleep(0.05)

    sess = manager.get_session(sess_id)
    assert sess.current_version == 2
    assert sess.state == SessionState.COMPLETED
    assert "hotels" in sess.committed_data


@pytest.mark.asyncio
async def test_exception_isolation_exceptions_do_not_bypass_fence():
    """Verify an older failed tool result cannot set state to ERROR if superseded."""
    manager = SessionManager()
    sess_id = f"sess_fence_exc_{uuid.uuid4().hex[:8]}"
    manager.start_turn(sess_id, "Turn 1", request_id="req_1")
    manager.start_turn(sess_id, "Turn 2", request_id="req_2")

    sess = manager.get_session(sess_id)
    assert sess.current_version == 2

    # Late failure from V1
    v1_fail = ToolResultEnvelope(
        session_id=sess_id,
        version=1,
        request_id="req_1",
        tool_result=ToolResult.fail("hotel_search", "Fatal crash in V1"),
    )
    accepted = manager.process_tool_result(v1_fail)
    assert accepted is False
    # V2 state remains unaffected
    assert sess.state != SessionState.ERROR


def test_api_500_exception_handler_sanitized():
    """Verify unhandled internal exceptions return clean JSON 500 without stack traces."""
    client = TestClient(app, raise_server_exceptions=False)
    # Trigger an unexpected exception by patching an internal method
    from unittest.mock import patch

    with patch("app.services.agent_controller.agent_controller.handle_turn", side_effect=Exception("Database password: secret_key_12345")):
        response = client.post("/api/v1/sessions/sess_500_test/turns", json={"transcript": "Test"})
        assert response.status_code == 500
        data = response.json()
        assert data == {"detail": "Internal server error"}
        assert "secret_key" not in str(data)
        assert "Traceback" not in str(data)


# =====================================================================
# 4. TASK LIFECYCLE & MEMORY MANAGEMENT
# =====================================================================

@pytest.mark.asyncio
async def test_task_lifecycle_memory_cleanup():
    """Verify finished background tasks are automatically cleared from session.active_task."""
    manager = SessionManager()
    controller = AgentController(manager=manager)
    sess_id = f"sess_lifecycle_{uuid.uuid4().hex[:8]}"

    controller.handle_turn(sess_id, "Find hotels under 3000", simulated_delay=0.01)
    session = manager.get_session(sess_id)

    # Wait for the task to complete
    await asyncio.sleep(0.06)

    # After completion, active_task should be cleared to prevent memory leaks
    assert session.active_task is None


# =====================================================================
# 5. SESSION ISOLATION
# =====================================================================

@pytest.mark.asyncio
async def test_session_isolation_independent_sessions():
    """Verify operations on Session A cannot mutate Session B."""
    manager = SessionManager()
    controller = AgentController(manager=manager)

    sess_a = f"sess_a_{uuid.uuid4().hex[:8]}"
    sess_b = f"sess_b_{uuid.uuid4().hex[:8]}"

    # Session A: V1 -> V2
    controller.handle_turn(sess_a, "Find hotels in Mumbai under 5000", simulated_delay=0.1)
    controller.handle_turn(sess_a, "Find hotels in Mumbai under 3000", simulated_delay=0.0)

    # Session B: V1
    controller.handle_turn(sess_b, "What is the weather in Delhi", simulated_delay=0.0)

    await asyncio.sleep(0.05)

    a = manager.get_session(sess_a)
    b = manager.get_session(sess_b)

    # Session A assertions
    assert a.current_version == 2
    assert "Mumbai" in a.last_transcript

    # Session B assertions
    assert b.current_version == 1
    assert "Delhi" in b.last_transcript
    assert "weather" in str(b.committed_data).lower()
    assert "Mumbai" not in str(b.committed_data)


# =====================================================================
# 6. API RELIABILITY & CONCURRENCY
# =====================================================================

@pytest.mark.asyncio
async def test_api_reliability_concurrent_requests():
    """Verify concurrent requests to FastAPI endpoints succeed without deadlocks."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create 10 sessions concurrently
        tasks = [client.post("/api/v1/sessions") for _ in range(10)]
        responses = await asyncio.gather(*tasks)
        for resp in responses:
            assert resp.status_code == 201
            assert resp.json()["current_version"] == 0


@pytest.mark.asyncio
async def test_api_reliability_invalid_session_and_payloads():
    """Verify API handles invalid session IDs and malformed payloads cleanly."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 404 for unknown session
        resp_404 = await client.get("/api/v1/sessions/non_existent_12345")
        assert resp_404.status_code == 404
        assert "not found" in resp_404.json()["detail"].lower()

        # 422 for empty transcript
        sess_create = await client.post("/api/v1/sessions")
        s_id = sess_create.json()["session_id"]
        resp_422 = await client.post(f"/api/v1/sessions/{s_id}/turns", json={"transcript": ""})
        assert resp_422.status_code == 422


@pytest.mark.asyncio
async def test_api_reliability_repeated_requests():
    """Verify repeated requests to the same session do not corrupt state."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        create = await client.post("/api/v1/sessions")
        s_id = create.json()["session_id"]

        for _ in range(3):
            turn = await client.post(
                f"/api/v1/sessions/{s_id}/turns",
                json={"transcript": "Find hotels in Delhi under 3000", "simulated_delay": 0.0},
            )
            assert turn.status_code == 200

        await asyncio.sleep(0.05)
        status = await client.get(f"/api/v1/sessions/{s_id}")
        assert status.status_code == 200
        assert status.json()["current_version"] == 3


# =====================================================================
# 7. OBSERVABILITY & SECRETS AUDIT
# =====================================================================

@pytest.mark.asyncio
async def test_observability_event_diagnostics():
    """Verify diagnostic event stream records all expected lifecycle phases."""
    manager = SessionManager()
    controller = AgentController(manager=manager)
    sess_id = f"sess_obs_{uuid.uuid4().hex[:8]}"

    # Turn with mock delay
    controller.handle_turn(sess_id, "Find hotels under 5000", simulated_delay=0.1)
    controller.handle_turn(sess_id, "Actually under 3000", simulated_delay=0.0)
    await asyncio.sleep(0.05)

    events = manager.get_events(sess_id)
    event_types = {e.event_type for e in events}

    expected_types = {
        EventType.SESSION_CREATED,
        EventType.TURN_STARTED,
        EventType.CANCELLATION_REQUESTED,
        EventType.TOOL_STARTED,
        EventType.RESULT_ACCEPTED,
        EventType.TOOL_COMPLETED,
        EventType.RESPONSE_READY,
    }
    assert expected_types.issubset(event_types)


@pytest.mark.asyncio
async def test_security_no_secrets_in_events_or_state():
    """Verify events, payloads, and session state never contain API keys or auth secrets."""
    manager = SessionManager()
    controller = AgentController(manager=manager)
    sess_id = f"sess_sec_{uuid.uuid4().hex[:8]}"

    controller.handle_turn(sess_id, "Find hotels under 3000", simulated_delay=0.0)
    await asyncio.sleep(0.05)

    session = manager.get_session(sess_id)
    events = manager.get_events(sess_id)

    forbidden_keywords = ["api_key", "secret", "password", "token", "sk-", "bearer"]

    # Check session
    sess_str = str(session.model_dump()).lower()
    for kw in forbidden_keywords:
        assert kw not in sess_str

    # Check all events
    for evt in events:
        evt_str = str(evt.model_dump()).lower()
        for kw in forbidden_keywords:
            assert kw not in evt_str


# =====================================================================
# 8. CRITICAL STRESS SCENARIO (SCENARIO 9)
# =====================================================================

@pytest.mark.asyncio
async def test_critical_stress_delhi_hotels_scenario():
    """Exact test for Section 9:
    V1: 'Find hotels in Delhi under 5000.' (Tool delay simulated)
    After brief delay:
    V2: 'Actually under 3000.'

    Expected:
    1. V1 exists with version 1.
    2. V2 supersedes V1.
    3. Current version becomes 2.
    4. Cancellation of V1 is attempted.
    5. V1 is allowed to finish if cancellation fails.
    6. V1 reaches Result Fence.
    7. V1 is classified as stale.
    8. V1 does not mutate current conversation state.
    9. V1 does not trigger response generation.
    10. V2 completes.
    11. V2 result is accepted.
    12. Only V2 can produce RESPONSE_READY.
    """
    manager = SessionManager()
    controller = AgentController(manager=manager)
    sess_id = f"sess_delhi_stress_{uuid.uuid4().hex[:8]}"

    # Step 1: V1 starts with delay
    controller.handle_turn(
        sess_id,
        "Find hotels in Delhi under 5000",
        simulated_delay=0.15,
        request_id="req_delhi_v1",
    )
    sess = manager.get_session(sess_id)
    assert sess.current_version == 1
    assert sess.current_request_id == "req_delhi_v1"

    # Step 2 & 3: V2 supersedes V1 after 0.03s
    await asyncio.sleep(0.03)
    controller.handle_turn(
        sess_id,
        "Actually under 3000",
        simulated_delay=0.0,
        request_id="req_delhi_v2",
    )
    assert sess.current_version == 2
    assert sess.current_request_id == "req_delhi_v2"

    # Step 4: Cancellation was attempted
    cancel_evts = manager.get_events(sess_id, event_type=EventType.CANCELLATION_REQUESTED)
    assert len(cancel_evts) >= 1
    assert cancel_evts[0].payload["superseded_version"] == 1

    # Step 10 & 11: V2 completes and is accepted
    await asyncio.sleep(0.05)
    assert sess.state == SessionState.COMPLETED
    assert sess.committed_version == 2
    assert sess.committed_request_id == "req_delhi_v2"

    # Step 5, 6, 7: Simulate V1 finishing late and reaching Result Fence
    v1_late_result = ToolResult.ok(
        tool_name="hotel_search",
        output={"hotels": [{"name": "Expensive Hotel", "price": 4800}]},
        request_id="req_delhi_v1",
        version=1,
    )
    v1_envelope = ToolResultEnvelope(
        session_id=sess_id,
        version=1,
        request_id="req_delhi_v1",
        tool_result=v1_late_result,
    )
    accepted_v1 = manager.process_tool_result(v1_envelope)

    # Step 7, 8, 9: V1 classified as stale; does not mutate conversation state or trigger response
    assert accepted_v1 is False
    stale_evts = manager.get_events(sess_id, event_type=EventType.RESULT_REJECTED_STALE)
    assert len(stale_evts) >= 1
    assert stale_evts[0].payload["result_version"] == 1

    # Step 12: Only V2 produced RESPONSE_READY
    response_ready_evts = manager.get_events(sess_id, event_type=EventType.RESPONSE_READY)
    assert len(response_ready_evts) == 1
    assert response_ready_evts[0].version == 2
    assert response_ready_evts[0].request_id == "req_delhi_v2"
    assert "3000" in response_ready_evts[0].payload["response"]


# =====================================================================
# 9. CONCURRENCY PERFORMANCE SANITY CHECK
# =====================================================================

@pytest.mark.asyncio
async def test_concurrency_performance_multi_session():
    """Verify 20 concurrent sessions execute without deadlocking or blocking."""
    manager = SessionManager()
    controller = AgentController(manager=manager)

    async def run_session(idx: int):
        s_id = f"sess_perf_{idx}_{uuid.uuid4().hex[:6]}"
        controller.handle_turn(s_id, f"Find hotels in Delhi under {2000 + idx * 100}", simulated_delay=0.01)
        await asyncio.sleep(0.03)
        s = manager.get_session(s_id)
        assert s.state == SessionState.COMPLETED
        assert s.committed_version == 1

    tasks = [run_session(i) for i in range(20)]
    await asyncio.gather(*tasks)
