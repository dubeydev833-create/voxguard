"""VoxGuard Result Fence Service.

Enforces the core correctness invariant of the VoxGuard voice agent:
ONLY THE CURRENT VERSION MAY MODIFY CURRENT CONVERSATION STATE OR TRIGGER RESPONSE GENERATION.

Any asynchronous tool result whose version or request_id does not match the active
session context is strictly classified as STALE and prohibited from mutating state
or triggering response synthesis.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from app.models.events import ToolResultEnvelope
from app.models.state import Session, SessionState


class FenceDecision(str, Enum):
    """Result classification from the Result Fence evaluation."""

    ACCEPTED = "ACCEPTED"
    STALE = "STALE"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    INVALID_STATE = "INVALID_STATE"


@dataclass
class FenceEvaluation:
    """Detailed outcome of a Result Fence validation check."""

    decision: FenceDecision
    reason: str
    result_version: int
    session_current_version: Optional[int] = None
    result_request_id: Optional[str] = None
    session_current_request_id: Optional[str] = None

    @property
    def is_accepted(self) -> bool:
        """Helper property to test if result was accepted."""
        return self.decision == FenceDecision.ACCEPTED


class ResultFence:
    """Explicit barrier protecting conversational session state and response generation."""

    @staticmethod
    def evaluate(envelope: ToolResultEnvelope, session: Optional[Session]) -> FenceEvaluation:
        """Evaluate incoming tool result envelope against active session state.

        Guarantees:
        1. Session must exist.
        2. Result version must strictly match current_version (fundamental invariant).
        3. Result request_id must match current_request_id if both are provided.
        4. Interrupted sessions cannot accept normal execution results.

        Args:
            envelope: ToolResultEnvelope containing result, version, and request_id.
            session: Target Session or None if not found.

        Returns:
            FenceEvaluation with decision and diagnostic metadata.
        """
        if session is None:
            return FenceEvaluation(
                decision=FenceDecision.SESSION_NOT_FOUND,
                reason="Session not found",
                result_version=envelope.version,
                result_request_id=envelope.request_id,
            )

        # 1. Primary Invariant Barrier: Version must strictly match
        if envelope.version != session.current_version:
            return FenceEvaluation(
                decision=FenceDecision.STALE,
                reason=f"Stale result version: result v{envelope.version} != session v{session.current_version}",
                result_version=envelope.version,
                session_current_version=session.current_version,
                result_request_id=envelope.request_id,
                session_current_request_id=session.current_request_id,
            )

        # 2. Request Identity Barrier: Request ID must match if present
        if (
            envelope.request_id is not None
            and session.current_request_id is not None
            and envelope.request_id != session.current_request_id
        ):
            return FenceEvaluation(
                decision=FenceDecision.STALE,
                reason=f"Stale request identity: result req '{envelope.request_id}' != session req '{session.current_request_id}'",
                result_version=envelope.version,
                session_current_version=session.current_version,
                result_request_id=envelope.request_id,
                session_current_request_id=session.current_request_id,
            )

        # 3. Interruption Barrier: Interrupted sessions cannot accept normal results
        if session.state == SessionState.INTERRUPTED and not envelope.tool_result.is_cancelled:
            return FenceEvaluation(
                decision=FenceDecision.STALE,
                reason="Session was interrupted before tool result completed",
                result_version=envelope.version,
                session_current_version=session.current_version,
                result_request_id=envelope.request_id,
                session_current_request_id=session.current_request_id,
            )

        # Approved
        return FenceEvaluation(
            decision=FenceDecision.ACCEPTED,
            reason="Result matches active session version and request identity",
            result_version=envelope.version,
            session_current_version=session.current_version,
            result_request_id=envelope.request_id,
            session_current_request_id=session.current_request_id,
        )

    @classmethod
    def is_accepted(cls, envelope: ToolResultEnvelope, session: Optional[Session]) -> bool:
        """Evaluate and return True if accepted, False otherwise."""
        return cls.evaluate(envelope, session).is_accepted
