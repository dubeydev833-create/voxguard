"""VoxGuard Tool Data Models.

Defines the schemas and models for tool definitions, parameters,
risk classifications, invocations, and execution results.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """Risk classification levels for tool execution security guardrails."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ToolParameter(BaseModel):
    """Schema definition for a single parameter accepted by a tool."""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[str]] = None


class ToolDefinition(BaseModel):
    """Complete metadata and parameter schema for an agent tool."""

    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.LOW
    requires_confirmation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize definition to standard dictionary format."""
        return self.model_dump()


class ToolCall(BaseModel):
    """Representation of an incoming tool invocation request."""

    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    call_id: Optional[str] = None


class ToolResult(BaseModel):
    """Output generated from executing a tool."""

    tool_name: str
    success: bool
    output: Optional[Any] = None
    error: Optional[str] = None
    call_id: Optional[str] = None
    request_id: Optional[str] = None
    version: Optional[int] = None
    status: str = "completed"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def is_cancelled(self) -> bool:
        """Check if execution result represents a cancellation."""
        return self.status == "cancelled" or self.metadata.get("cancelled", False) is True

    @classmethod
    def ok(
        cls,
        tool_name: str,
        output: Any,
        call_id: Optional[str] = None,
        request_id: Optional[str] = None,
        version: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ToolResult":
        """Convenience constructor for successful execution."""
        return cls(
            tool_name=tool_name,
            success=True,
            status="completed",
            output=output,
            call_id=call_id,
            request_id=request_id,
            version=version,
            metadata=metadata or {},
        )

    @classmethod
    def fail(
        cls,
        tool_name: str,
        error: str,
        call_id: Optional[str] = None,
        request_id: Optional[str] = None,
        version: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ToolResult":
        """Convenience constructor for failed execution."""
        return cls(
            tool_name=tool_name,
            success=False,
            status="failed",
            error=error,
            call_id=call_id,
            request_id=request_id,
            version=version,
            metadata=metadata or {},
        )

    @classmethod
    def cancelled(
        cls,
        tool_name: str,
        message: str = "Tool execution was cancelled.",
        call_id: Optional[str] = None,
        request_id: Optional[str] = None,
        version: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ToolResult":
        """Convenience constructor for cancelled execution."""
        meta = {**(metadata or {}), "cancelled": True, "status": "cancelled"}
        return cls(
            tool_name=tool_name,
            success=False,
            status="cancelled",
            error=message,
            call_id=call_id,
            request_id=request_id,
            version=version,
            metadata=meta,
        )
