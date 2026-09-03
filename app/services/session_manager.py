"""VoxGuard Session Manager and Result Fencing Service.

Manages conversational sessions, version tracking, active task cancellation,
and result fencing across asynchronous turns.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Union
import time

from app.models.events import Event, EventType, ToolResultEnvelope
from app.models.state import Session, SessionState


class SessionManager:
    """In-memory session manager with version tracking and result fencing."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self._events: List[Event] = []
        self._listeners: List[Callable[[Event], Any]] = []

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
    ) -> Session:
        """Start a new conversational turn.

        Increments current_version, cancels any currently running task for the session,
        updates the session state, and emits TURN_STARTED.

        Args:
            session_id: The unique session identifier.
            transcript: User transcript or prompt initiating the turn.
            state: The target state (defaults to THINKING, can also be TOOL_RUNNING).
            active_task: Optional asyncio.Task to track for this turn.

        Returns:
            The updated Session object.
        """
        session = self.get_or_create_session(session_id)

        # 1. Cancel any active task from previous turns
        if session.cancel_active_task():
            self._emit(
                EventType.CANCELLATION_REQUESTED,
                session_id=session_id,
                version=session.current_version,
                payload={
                    "reason": "Superseding turn received",
                    "superseded_version": session.current_version,
                },
            )

        # 2. Increment version to invalidate previous in-flight results
        session.current_version += 1

        # 3. Update session properties
        session.state = state
        session.last_transcript = transcript
        session.active_task = active_task
        session.updated_at = time.time()

        # 4. Emit TURN_STARTED event
        self._emit(
            EventType.TURN_STARTED,
            session_id=session_id,
            version=session.current_version,
            payload={
                "transcript": transcript,
                "state": session.state.value,
                "version": session.current_version,
            },
        )

        return session

    def interrupt(self, session_id: str) -> Session:
        """Interrupt an active session turn.

        Cancels any active running task, updates state to INTERRUPTED,
        and emits the INTERRUPTED and INTERRUPTION_DETECTED events.
        """
        session = self.get_or_create_session(session_id)

        # Cancel any active running task
        if session.cancel_active_task():
            self._emit(
                EventType.CANCELLATION_REQUESTED,
                session_id=session_id,
                version=session.current_version,
                payload={"reason": "User interruption"},
            )

        # Update state
        session.state = SessionState.INTERRUPTED
        session.updated_at = time.time()

        self._emit(
            EventType.INTERRUPTION_DETECTED,
            session_id=session_id,
            version=session.current_version,
            payload={"session_id": session_id, "state": session.state.value},
        )
        self._emit(
            EventType.INTERRUPTED,
            session_id=session_id,
            version=session.current_version,
            payload={"session_id": session_id, "state": session.state.value},
        )

        return session

    def set_task(self, session_id: str, task: asyncio.Task) -> None:
        """Associate an active asyncio task with the session."""
        session = self.get_or_create_session(session_id)
        session.active_task = task

    def set_state(self, session_id: str, state: SessionState) -> Session:
        """Update session state and emit STATE_CHANGED."""
        session = self.get_or_create_session(session_id)
        session.state = state
        session.updated_at = time.time()

        self._emit(
            EventType.STATE_CHANGED,
            session_id=session_id,
            version=session.current_version,
            payload={"state": state.value},
        )
        return session

    def process_tool_result(self, result: ToolResultEnvelope) -> bool:
        """Process an incoming tool result with result fencing.

        Accepts the result only if result.version == session.current_version.
        Otherwise, rejects the result as stale and emits RESULT_REJECTED_STALE.

        Args:
            result: The ToolResultEnvelope containing result and version metadata.

        Returns:
            True if accepted, False if rejected as stale.
        """
        session = self.get_session(result.session_id)
        if session is None:
            # Session does not exist; cannot accept
            self._emit(
                EventType.RESULT_REJECTED_STALE,
                session_id=result.session_id,
                version=result.version,
                payload={
                    "reason": "Session not found",
                    "result_version": result.version,
                    "tool_name": result.tool_result.tool_name,
                },
            )
            return False

        # Result fencing check: version must strictly match current_version
        if result.version != session.current_version:
            self._emit(
                EventType.RESULT_REJECTED_STALE,
                session_id=result.session_id,
                version=result.version,
                payload={
                    "reason": "Stale result version",
                    "result_version": result.version,
                    "session_current_version": session.current_version,
                    "tool_name": result.tool_result.tool_name,
                },
            )
            return False

        # Accepted result: update session state to COMPLETED and commit data
        session.state = SessionState.COMPLETED
        session.committed_version = result.version
        session.last_result = result.tool_result
        if isinstance(result.tool_result.output, dict):
            session.committed_data.update(result.tool_result.output)
            session.metadata.update(result.tool_result.output)
        session.updated_at = time.time()

        self._emit(
            EventType.RESULT_ACCEPTED,
            session_id=result.session_id,
            version=result.version,
            payload={
                "tool_name": result.tool_result.tool_name,
                "success": result.tool_result.success,
                "version": result.version,
            },
        )

        self._emit(
            EventType.TOOL_COMPLETED,
            session_id=result.session_id,
            version=result.version,
            payload={
                "tool_name": result.tool_result.tool_name,
                "output": result.tool_result.output,
                "error": result.tool_result.error,
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
        for session in self._sessions.values():
            session.cancel_active_task()
        self._sessions.clear()
        self._events.clear()
        self._listeners.clear()

    def _emit(
        self,
        event_type: EventType,
        session_id: str,
        version: int,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Event:
        """Internal helper to create, record, and dispatch an event."""
        event = Event(
            event_type=event_type,
            session_id=session_id,
            version=version,
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
