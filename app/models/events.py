"""VoxGuard Event and Envelope Models.

Defines the event types, event structures, and result envelopes
used for event emission and result fencing across asynchronous turns.
"""

from enum import Enum
import time
from typing import Any, Dict, Optional
import uuid
from pydantic import BaseModel, Field

from app.models.tool import ToolResult


class EventType(str, Enum):
    """Event types emitted throughout session lifecycle and tool operations."""

    SESSION_CREATED = "SESSION_CREATED"
    TURN_STARTED = "TURN_STARTED"
    INTERRUPTED = "INTERRUPTED"
    INTERRUPTION_DETECTED = "INTERRUPTION_DETECTED"
    CANCELLATION_REQUESTED = "CANCELLATION_REQUESTED"
    TOOL_STARTED = "TOOL_STARTED"
    TOOL_COMPLETED = "TOOL_COMPLETED"
    TOOL_CANCELLED = "TOOL_CANCELLED"
    TOOL_FAILED = "TOOL_FAILED"
    RESULT_ACCEPTED = "RESULT_ACCEPTED"
    RESULT_REJECTED_STALE = "RESULT_REJECTED_STALE"
    RESPONSE_READY = "RESPONSE_READY"
    STATE_CHANGED = "STATE_CHANGED"


class Event(BaseModel):
    """Event emitted by the session manager and tool execution pipeline."""

    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    event_type: EventType
    session_id: str
    version: int
    request_id: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
    payload: Dict[str, Any] = Field(default_factory=dict)


class ToolResultEnvelope(BaseModel):
    """Versioned envelope enclosing a tool result for fencing checks."""

    session_id: str
    version: int
    request_id: Optional[str] = None
    tool_result: ToolResult
    call_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
