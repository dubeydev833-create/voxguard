"""VoxGuard LLM Base Interfaces and Structured Data Models.

Defines the abstract LLM provider contract and structured intent models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from app.models.tool import ToolResult


class StructuredIntent(BaseModel):
    """Structured intent and arguments extracted from natural language transcript."""

    intent: Optional[str] = None
    arguments: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    direct_response: Optional[str] = None


class LLMProvider(ABC):
    """Abstract interface for LLM provider adapters in VoxGuard."""

    @abstractmethod
    async def parse_intent(
        self,
        transcript: str,
        session_id: str,
        request_id: str,
        version: int,
    ) -> StructuredIntent:
        """Parse natural-language transcript into a structured tool intent and arguments.

        Args:
            transcript: User natural language speech transcript or text.
            session_id: Conversation session identifier.
            request_id: Unique identifier for this turn request.
            version: Monotonic version of this turn request.

        Returns:
            StructuredIntent with requested tool name and validated arguments, or
            direct_response if no tool is necessary.
        """
        raise NotImplementedError

    @abstractmethod
    async def synthesize_response(
        self,
        tool_name: str,
        tool_result: ToolResult,
        transcript: str,
        session_id: str,
        request_id: str,
        version: int,
    ) -> str:
        """Synthesize natural language response from execution result.

        Args:
            tool_name: The name of the executed tool.
            tool_result: The ToolResult produced by tool execution.
            transcript: Original user input transcript.
            session_id: Conversation session identifier.
            request_id: Unique identifier for this turn request.
            version: Monotonic version of this turn request.

        Returns:
            Natural language response string suitable for voice/text output.
        """
        raise NotImplementedError
