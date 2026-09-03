"""VoxGuard Session State Models.

Defines the session states and session tracking models for managing
agent conversational turns, background task execution, and result fencing.
"""

import asyncio
from enum import Enum
import time
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class SessionState(str, Enum):
    """Lifecycle states of an active VoxGuard session."""

    IDLE = "IDLE"
    THINKING = "THINKING"
    TOOL_RUNNING = "TOOL_RUNNING"
    INTERRUPTED = "INTERRUPTED"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


from app.models.tool import ToolResult


class Session(BaseModel):
    """In-memory state and task tracking for a user agent session."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    current_version: int = 0
    state: SessionState = SessionState.IDLE
    last_transcript: Optional[str] = None
    active_task: Optional[asyncio.Task] = Field(default=None, exclude=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    committed_version: Optional[int] = None
    committed_data: Dict[str, Any] = Field(default_factory=dict)
    last_result: Optional[ToolResult] = None
    last_response: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def cancel_active_task(self) -> bool:
        """Cancel the active background task if one is running."""
        if self.active_task is not None and not self.active_task.done():
            self.active_task.cancel()
            return True
        return False
