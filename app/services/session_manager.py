"""VoxGuard Session Manager and Result Fencing Service.

Manages conversational sessions, version tracking, active task cancellation,
and result fencing across asynchronous turns.
"""

import asyncio
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Union
import uuid

from app.models.events import Event, EventType, ToolResultEnvelope
from app.models.state import RequestContext, Session, SessionState
from app.services.result_fence import FenceDecision, ResultFence


class SessionManager:
    """In-memory session manager with version tracking and result fencing."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self._events: List[Event] = []
        self._listeners: List[Callable[[Event], Any]] = []
        self._locks: Dict[str, threading.RLock] = {}
        self._global_lock = threading.Lock()

    def get_session_lock(self, session_id: str) -> threading.RLock:
        """Retrieve or create a reentrant lock for per-session synchronization."""
        with self._global_lock:
            if session_id not in self._locks:
                self._locks[session_id] = threading.RLock()
            return self._locks[session_id]

    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve a session by its ID, or None if it does not exist."""
        return self._sessions.get(session_id)

    def get_or_create_session(self, session_id: str) -> Session:
        """Retrieve an existing session or create a new one."""
        if session_id not in self._sessions:
            session = Session(
                session_id=session_id,
                current_version=0,
                state=SessionState.IDLE,
            )
            self._sessions[session_id] = session
            self._emit(
                EventType.SESSION_CREATED,
                session_id=session_id,
                version=0,
                payload={"session_id": session_id},
            )
        return self._sessions[session_id]

    def start_turn(
        self,
        session_id: str,
        transcript: str,
        state: SessionState = SessionState.THINKING,
        active_task: Optional[asyncio.Task] = None,
        request_id: Optional[str] = None,
    ) -> Session:
        """Start a new conversational turn.

        Increments current_version, cancels any currently running task for the session,
        updates the session state, and emits TURN_STARTED.

        Args:
            session_id: The unique session identifier.
            transcript: User transcript or prompt initiating the turn.
            state: The target state (defaults to THINKING, can also be TOOL_RUNNING).
            active_task: Optional asyncio.Task to track for this turn.
            request_id: Optional unique request ID. If None, one will be generated.

        Returns:
            The updated Session object.
        """
        with self.get_session_lock(session_id):
            session = self.get_or_create_session(session_id)
            superseded_request_id = session.current_request_id

            # 1. Cancel any active task from previous turns
            if session.cancel_active_task():
                if superseded_request_id and superseded_request_id in session.requests:
                    session.requests[superseded_request_id].status = "superseded"
                self._emit(
                    EventType.CANCELLATION_REQUESTED,
                    session_id=session_id,
                    version=session.current_version,
                    request_id=superseded_request_id,
                    payload={
                        "reason": "Superseding turn received",
                        "superseded_version": session.current_version,
                        "superseded_request_id": superseded_request_id,
                    },
                )

            # 2. Increment version to invalidate previous in-flight results
            session.current_version += 1

            # 3. Assign new request_id and store RequestContext
            req_id = request_id or f"req_{uuid.uuid4().hex[:10]}"
            session.current_request_id = req_id
            session.requests[req_id] = RequestContext(
                request_id=req_id,
                session_id=session_id,
                version=session.current_version,
                user_input=transcript,
                status="pending",
            )

            # 4. Update session properties
            session.state = state
            session.last_transcript = transcript
            session.active_task = active_task
            if active_task is not None:
                self._attach_task_cleanup(session_id, active_task)
            session.updated_at = time.time()

            # 5. Emit TURN_STARTED event
            self._emit(
                EventType.TURN_STARTED,
                session_id=session_id,
                version=session.current_version,
                request_id=req_id,
                payload={
                    "transcript": transcript,
                    "state": session.state.value,
                    "version": session.current_version,
                    "request_id": req_id,
                },
            )

            return session

    def interrupt(self, session_id: str) -> Session:
        """Interrupt an active session turn.

        Cancels any active running task, updates state to INTERRUPTED,
        and emits the INTERRUPTED and INTERRUPTION_DETECTED events.
        """
        with self.get_session_lock(session_id):
            session = self.get_or_create_session(session_id)
            current_req_id = session.current_request_id

            # Cancel any active running task
            if session.cancel_active_task():
                if current_req_id and current_req_id in session.requests:
                    session.requests[current_req_id].status = "interrupted"
                self._emit(
                    EventType.CANCELLATION_REQUESTED,
                    session_id=session_id,
                    version=session.current_version,
                    request_id=current_req_id,
                    payload={
                        "reason": "User interruption",
                        "request_id": current_req_id,
                    },
                )

            if current_req_id and current_req_id in session.requests:
                session.requests[current_req_id].status = "interrupted"

            # Update state
            session.state = SessionState.INTERRUPTED
            session.updated_at = time.time()

            self._emit(
                EventType.INTERRUPTION_DETECTED,
                session_id=session_id,
                version=session.current_version,
                request_id=current_req_id,
                payload={
                    "session_id": session_id,
                    "state": session.state.value,
                    "request_id": current_req_id,
                },
            )
            self._emit(
                EventType.INTERRUPTED,
                session_id=session_id,
                version=session.current_version,
                request_id=current_req_id,
                payload={
                    "session_id": session_id,
                    "state": session.state.value,
                    "request_id": current_req_id,
                },
            )

            return session

    def _attach_task_cleanup(self, session_id: str, task: asyncio.Task) -> None:
        """Attach a done callback to clear finished task from session to prevent memory leaks."""
        def _on_done(t: asyncio.Task) -> None:
            sess = self.get_session(session_id)
            if sess and sess.active_task is t:
                sess.active_task = None

        try:
            task.add_done_callback(_on_done)
        except Exception:
            pass

    def set_task(self, session_id: str, task: asyncio.Task) -> None:
        """Associate an active asyncio task with the session."""
        session = self.get_or_create_session(session_id)
        session.active_task = task
        self._attach_task_cleanup(session_id, task)

    def set_state(self, session_id: str, state: SessionState) -> Session:
        """Update session state and emit STATE_CHANGED."""
        session = self.get_or_create_session(session_id)
        session.state = state
        session.updated_at = time.time()

        self._emit(
            EventType.STATE_CHANGED,
            session_id=session_id,
            version=session.current_version,
            request_id=session.current_request_id,
            payload={"state": state.value},
        )
        return session

    def process_tool_result(self, result: ToolResultEnvelope) -> bool:
        """Process an incoming tool result with explicit Result Fencing.

        Accepts the result only if ResultFence.evaluate() approves it.
        Otherwise, rejects the result as stale and emits RESULT_REJECTED_STALE.
        Enforces atomic check-and-mutation under the per-session lock.

        Args:
            result: The ToolResultEnvelope containing result and version metadata.

        Returns:
            True if accepted, False if rejected as stale.
        """
        with self.get_session_lock(result.session_id):
            session = self.get_session(result.session_id)
            eval_result = ResultFence.evaluate(result, session)

            if not eval_result.is_accepted:
                if session and result.request_id and result.request_id in session.requests:
                    session.requests[result.request_id].status = "stale"
                self._emit(
                    EventType.RESULT_REJECTED_STALE,
                    session_id=result.session_id,
                    version=result.version,
                    request_id=result.request_id,
                    payload={
                        "reason": eval_result.reason,
                        "result_version": result.version,
                        "session_current_version": eval_result.session_current_version,
                        "result_request_id": result.request_id,
                        "session_current_request_id": eval_result.session_current_request_id,
                        "tool_name": result.tool_result.tool_name,
                        "status": result.tool_result.status,
                        "is_cancelled": result.tool_result.is_cancelled,
                    },
                )
                return False

            # If result was cancelled
            if result.tool_result.is_cancelled:
                if result.request_id and result.request_id in session.requests:
                    session.requests[result.request_id].status = "cancelled"
                session.last_result = result.tool_result
                session.updated_at = time.time()
                self._emit(
                    EventType.TOOL_CANCELLED,
                    session_id=result.session_id,
                    version=result.version,
                    request_id=result.request_id,
                    payload={
                        "tool_name": result.tool_result.tool_name,
                        "status": "cancelled",
                        "error": result.tool_result.error,
                        "version": result.version,
                        "request_id": result.request_id,
                    },
                )
                return True

            # If result failed (tool failure)
            if not result.tool_result.success:
                session.state = SessionState.ERROR
                session.committed_version = result.version
                session.committed_request_id = result.request_id
                if result.request_id and result.request_id in session.requests:
                    session.requests[result.request_id].status = "failed"
                session.last_result = result.tool_result
                session.updated_at = time.time()
                self._emit(
                    EventType.TOOL_FAILED,
                    session_id=result.session_id,
                    version=result.version,
                    request_id=result.request_id,
                    payload={
                        "tool_name": result.tool_result.tool_name,
                        "error": result.tool_result.error,
                        "status": "failed",
                        "version": result.version,
                        "request_id": result.request_id,
                    },
                )
                return True

            # Accepted result: update session state to COMPLETED and commit data
            session.state = SessionState.COMPLETED
            session.committed_version = result.version
            session.committed_request_id = result.request_id
            if result.request_id and result.request_id in session.requests:
                session.requests[result.request_id].status = "completed"
            session.last_result = result.tool_result
            if isinstance(result.tool_result.output, dict):
                session.committed_data.update(result.tool_result.output)
                session.metadata.update(result.tool_result.output)
            session.updated_at = time.time()

            self._emit(
                EventType.RESULT_ACCEPTED,
                session_id=result.session_id,
                version=result.version,
                request_id=result.request_id,
                payload={
                    "tool_name": result.tool_result.tool_name,
                    "success": result.tool_result.success,
                    "version": result.version,
                    "request_id": result.request_id,
                },
            )

            self._emit(
                EventType.TOOL_COMPLETED,
                session_id=result.session_id,
                version=result.version,
                request_id=result.request_id,
                payload={
                    "tool_name": result.tool_result.tool_name,
                    "output": result.tool_result.output,
                    "error": result.tool_result.error,
                    "request_id": result.request_id,
                },
            )

            return True

    def add_listener(self, listener: Callable[[Event], Any]) -> None:
        """Register a callback listener for emitted events."""
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[Event], Any]) -> None:
        """Unregister a callback listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def get_events(
        self,
        session_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
    ) -> List[Event]:
        """Query recorded events by session_id and/or event_type."""
        events = self._events
        if session_id is not None:
            events = [e for e in events if e.session_id == session_id]
        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]
        return events

    def clear(self) -> None:
        """Clear all sessions, tasks, and event history."""
        with self._global_lock:
            for session in list(self._sessions.values()):
                session.cancel_active_task()
            self._sessions.clear()
            self._events.clear()
            self._listeners.clear()
            self._locks.clear()

    def _emit(
        self,
        event_type: EventType,
        session_id: str,
        version: int,
        payload: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> Event:
        """Internal helper to create, record, and dispatch an event."""
        session = self.get_session(session_id)
        effective_req_id = request_id or (session.current_request_id if session else None)
        event = Event(
            event_type=event_type,
            session_id=session_id,
            version=version,
            request_id=effective_req_id,
            payload=payload or {},
        )
        self._events.append(event)
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass
        return event


# Global singleton SessionManager
session_manager = SessionManager()
