"""VoxGuard Conversation and Session Management REST Endpoints.

Provides endpoints for creating sessions, initiating/superseding conversational turns,
triggering interruptions, and inspecting session status.
"""

from typing import Any, Dict, Optional
import uuid
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.models.events import ToolResultEnvelope
from app.models.state import SessionState
from app.models.tool import ToolResult
from app.services.agent_controller import agent_controller
from app.services.session_manager import session_manager

router = APIRouter(tags=["conversations"])


# --- Request and Response Schemas ---

class CreateSessionRequest(BaseModel):
    """Optional payload for initializing a new session."""

    session_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    """Response schema representing current session state."""

    session_id: str
    current_version: int
    state: SessionState
    last_transcript: Optional[str] = None
    committed_version: Optional[int] = None
    committed_data: Dict[str, Any] = Field(default_factory=dict)
    last_response: Optional[str] = None
    created_at: float
    updated_at: float


class TurnRequest(BaseModel):
    """Payload for submitting a new turn or prompt to an ongoing session."""

    transcript: str
    state: Optional[SessionState] = SessionState.THINKING
    simulated_delay: Optional[float] = None


class ProcessResultRequest(BaseModel):
    """Payload for submitting an execution result to the fencing pipeline."""

    version: int
    tool_result: ToolResult
    call_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessResultResponse(BaseModel):
    """Outcome of the result fencing check."""

    accepted: bool
    session_id: str
    current_session_version: int
    submitted_version: int


# --- Endpoints ---

@router.post(
    "/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new session",
)
async def create_session(request: Optional[CreateSessionRequest] = None) -> SessionResponse:
    """Initialize a new VoxGuard agent session."""
    session_id = (request.session_id if request and request.session_id else None) or f"sess_{uuid.uuid4().hex[:10]}"
    session = session_manager.get_or_create_session(session_id)
    if request and request.metadata:
        session.metadata.update(request.metadata)

    return SessionResponse(
        session_id=session.session_id,
        current_version=session.current_version,
        state=session.state,
        last_transcript=session.last_transcript,
        committed_version=session.committed_version,
        committed_data=session.committed_data,
        last_response=session.last_response,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


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
    return SessionResponse(
        session_id=session.session_id,
        current_version=session.current_version,
        state=session.state,
        last_transcript=session.last_transcript,
        committed_version=session.committed_version,
        committed_data=session.committed_data,
        last_response=session.last_response,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


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
    )
    if request.state and request.state != SessionState.THINKING:
        session.state = request.state

    return SessionResponse(
        session_id=session.session_id,
        current_version=session.current_version,
        state=session.state,
        last_transcript=session.last_transcript,
        committed_version=session.committed_version,
        committed_data=session.committed_data,
        last_response=session.last_response,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


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
    session = session_manager.interrupt(session_id)
    return SessionResponse(
        session_id=session.session_id,
        current_version=session.current_version,
        state=session.state,
        last_transcript=session.last_transcript,
        committed_version=session.committed_version,
        committed_data=session.committed_data,
        last_response=session.last_response,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


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

    envelope = ToolResultEnvelope(
        session_id=session_id,
        version=request.version,
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
    )
