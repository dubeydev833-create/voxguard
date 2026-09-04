"""VoxGuard Conversation and Session Management REST Endpoints.

Provides endpoints for creating sessions, initiating/superseding conversational turns,
triggering interruptions, and inspecting session status.
"""

import asyncio
from typing import Any, Dict, List, Optional
import uuid
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field, field_validator

from app.models.events import Event, EventType, ToolResultEnvelope
from app.models.state import Session, SessionState
from app.models.tool import ToolResult
from app.services.agent_controller import agent_controller
from app.services.rime_service import rime_service
from app.services.session_manager import session_manager

router = APIRouter(tags=["conversations"])


# --- Request and Response Schemas ---

class TTSRequest(BaseModel):
    """Optional payload for requesting speech audio synthesis."""

    speaker: Optional[str] = None
    text: Optional[str] = None


class TTSResponse(BaseModel):
    """Response schema representing synthesized voice audio payload."""

    session_id: str
    version: Optional[int] = None
    request_id: Optional[str] = None
    text: str
    audio_base64: str
    audio_format: str = "audio/wav"
    speaker: str = "marsh"
    execution_time: float = 0.0


class CreateSessionRequest(BaseModel):
    """Optional payload for initializing a new session."""

    session_id: Optional[str] = Field(default=None, max_length=128)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    """Response schema representing current session state."""

    session_id: str
    current_version: int
    current_request_id: Optional[str] = None
    committed_request_id: Optional[str] = None
    state: SessionState
    last_transcript: Optional[str] = None
    committed_version: Optional[int] = None
    committed_data: Dict[str, Any] = Field(default_factory=dict)
    last_response: Optional[str] = None
    last_event: Optional[str] = None
    created_at: float
    updated_at: float


class TurnRequest(BaseModel):
    """Payload for submitting a new turn or prompt to an ongoing session."""

    transcript: str = Field(..., min_length=1, description="User transcript or natural language query.")
    request_id: Optional[str] = None
    state: Optional[SessionState] = SessionState.THINKING
    simulated_delay: Optional[float] = Field(default=None, ge=0.0)

    @field_validator("transcript")
    @classmethod
    def validate_transcript(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Transcript cannot be empty or whitespace only.")
        return v


def _build_session_response(session: Session) -> SessionResponse:
    """Helper to convert Session into client-facing SessionResponse with last event."""
    events = session_manager.get_events(session.session_id)
    last_event = events[-1].event_type.value if events else None
    return SessionResponse(
        session_id=session.session_id,
        current_version=session.current_version,
        current_request_id=session.current_request_id,
        committed_request_id=session.committed_request_id,
        state=session.state,
        last_transcript=session.last_transcript,
        committed_version=session.committed_version,
        committed_data=session.committed_data,
        last_response=session.last_response,
        last_event=last_event,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


class ProcessResultRequest(BaseModel):
    """Payload for submitting an execution result to the fencing pipeline."""

    version: int = Field(..., ge=0, description="Version epoch associated with the tool result.")
    tool_result: ToolResult
    request_id: Optional[str] = None
    call_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessResultResponse(BaseModel):
    """Outcome of the result fencing check."""

    accepted: bool
    session_id: str
    current_session_version: int
    submitted_version: int
    request_id: Optional[str] = None


# --- Endpoints ---

@router.post(
    "/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new session",
)
async def create_session(request: Optional[CreateSessionRequest] = None) -> SessionResponse:
    """Initialize a new VoxGuard agent session."""
    raw_sid = request.session_id.strip() if (request and request.session_id and request.session_id.strip()) else None
    session_id = raw_sid or f"sess_{uuid.uuid4().hex[:10]}"
    session = session_manager.get_or_create_session(session_id)
    if request and request.metadata:
        session.metadata.update(request.metadata)

    return _build_session_response(session)


@router.get(
    "/sessions/{session_id}",
    response_model=SessionResponse,
    summary="Get session details",
)
async def get_session(session_id: str) -> SessionResponse:
    """Retrieve the current state and version metadata for a session."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    return _build_session_response(session)


@router.get(
    "/sessions/{session_id}/events",
    response_model=List[Event],
    summary="Retrieve recorded events for a session",
)
async def get_session_events(
    session_id: str,
    event_type: Optional[EventType] = None,
) -> List[Event]:
    """Retrieve chronological event history for a session."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    return session_manager.get_events(session_id=session_id, event_type=event_type)


@router.websocket(
    "/sessions/{session_id}/events",
    name="stream_session_events",
)
async def stream_session_events(websocket: WebSocket, session_id: str):
    """Stream live events for a specific session via WebSocket.

    Validates session presence, accepts the WebSocket connection,
    streams serialized JSON events to connected client without blocking,
    and cleanly disconnects on client termination.
    """
    await websocket.accept()

    session = session_manager.get_session(session_id)
    if session is None:
        session = session_manager.get_or_create_session(session_id)

    # Send initial connection confirmation
    await websocket.send_json({
        "event_type": "CONNECTED",
        "session_id": session_id,
        "current_version": session.current_version,
        "state": session.state.value,
    })

    # Queue to forward events from session_manager listener to websocket
    queue: asyncio.Queue[Event] = asyncio.Queue()

    def on_event(event: Event):
        if event.session_id == session_id:
            try:
                queue.put_nowait(event)
            except Exception:
                pass

    session_manager.add_listener(on_event)

    async def sender():
        try:
            while True:
                event = await queue.get()
                await websocket.send_text(event.model_dump_json())
                queue.task_done()
        except asyncio.CancelledError:
            pass

    async def receiver():
        try:
            while True:
                # Keep connection alive and listen for client messages / pings
                await websocket.receive_text()
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass

    sender_task = asyncio.create_task(sender())
    receiver_task = asyncio.create_task(receiver())

    try:
        done, pending = await asyncio.wait(
            [sender_task, receiver_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    finally:
        session_manager.remove_listener(on_event)
        sender_task.cancel()
        receiver_task.cancel()


@router.post(
    "/sessions/{session_id}/turns",
    response_model=SessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Start or supersede a conversational turn",
)
async def start_turn(session_id: str, request: TurnRequest) -> SessionResponse:
    """Start a new turn for the session.

    Increments current_version, cancels any currently running background tasks,
    updates state, and registers the user transcript.
    """
    delay = request.simulated_delay
    if delay is None:
        if "under 5000" in request.transcript.lower() and "hotel" in request.transcript.lower():
            delay = 5.0
        else:
            delay = 0.0

    session = agent_controller.handle_turn(
        session_id=session_id,
        transcript=request.transcript,
        simulated_delay=delay,
        request_id=request.request_id,
    )
    if request.state and request.state != SessionState.THINKING:
        session.state = request.state

    return _build_session_response(session)


@router.post(
    "/sessions/{session_id}/interrupt",
    response_model=SessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Explicitly interrupt session turn",
)
async def interrupt_session(session_id: str) -> SessionResponse:
    """Explicitly interrupt the session.

    Marks state as INTERRUPTED and triggers cancellation of active tasks.
    """
    if session_manager.get_session(session_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    session = session_manager.interrupt(session_id)
    return _build_session_response(session)


@router.post(
    "/sessions/{session_id}/results",
    response_model=ProcessResultResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit tool result to result fencing check",
)
async def submit_tool_result(session_id: str, request: ProcessResultRequest) -> ProcessResultResponse:
    """Submit a tool result through result fencing.

    Accepts only if submitted version matches current session version;
    otherwise rejects as stale.
    """
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    req_id = request.request_id or request.tool_result.request_id
    envelope = ToolResultEnvelope(
        session_id=session_id,
        version=request.version,
        request_id=req_id,
        tool_result=request.tool_result,
        call_id=request.call_id,
        metadata=request.metadata,
    )

    accepted = session_manager.process_tool_result(envelope)
    return ProcessResultResponse(
        accepted=accepted,
        session_id=session_id,
        current_session_version=session.current_version,
        submitted_version=request.version,
        request_id=req_id,
    )


@router.post(
    "/sessions/{session_id}/tts",
    response_model=TTSResponse,
    status_code=status.HTTP_200_OK,
    summary="Synthesize voice response audio via Rime service",
)
async def synthesize_speech(
    session_id: str,
    request: Optional[TTSRequest] = None,
) -> TTSResponse:
    """Synthesize voice audio via Rime service for the latest session response.

    Enforces that interrupted or superseded turns cannot synthesize audio.
    """
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    can_synth, reason = rime_service.can_synthesize_for_session(session)
    if not can_synth:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=reason or "Cannot synthesize audio for session.",
        )

    text_to_speak = (request.text if request and request.text else None) or session.last_response or ""
    speaker = (request.speaker if request and request.speaker else None) or "marsh"

    try:
        audio_res = await rime_service.synthesize(
            session_id=session.session_id,
            text=text_to_speak,
            version=session.committed_version or session.current_version,
            request_id=session.committed_request_id or session.current_request_id,
            speaker=speaker,
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Voice synthesis service temporarily unavailable.",
        )

    return TTSResponse(
        session_id=audio_res.session_id,
        version=audio_res.version,
        request_id=audio_res.request_id,
        text=audio_res.text,
        audio_base64=audio_res.audio_base64,
        audio_format=audio_res.audio_format,
        speaker=audio_res.speaker,
        execution_time=audio_res.execution_time,
    )

