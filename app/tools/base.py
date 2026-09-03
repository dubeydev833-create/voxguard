"""VoxGuard Base Tool Interface.

Defines the abstract BaseTool class from which all concrete tools,
mock tools, and integration tools inherit.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from app.models.tool import RiskLevel, ToolDefinition, ToolResult


class BaseTool(ABC):
    """Abstract base class for all VoxGuard executable tools."""

    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}
    risk_level: RiskLevel = RiskLevel.LOW
    requires_confirmation: bool = False

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        risk_level: Optional[RiskLevel] = None,
        requires_confirmation: Optional[bool] = None,
    ) -> None:
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        if parameters is not None:
            self.parameters = parameters
        if risk_level is not None:
            self.risk_level = risk_level
        if requires_confirmation is not None:
            self.requires_confirmation = requires_confirmation

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given keyword arguments and return a ToolResult.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Tool execution must be implemented by subclasses.")

    def run(self, **kwargs: Any) -> ToolResult:
        """Alias for execute with argument validation."""
        try:
            self.validate_arguments(**kwargs)
            return self.execute(**kwargs)
        except Exception as exc:
            return ToolResult.fail(
                tool_name=self.name,
                error=str(exc),
            )

    def __call__(self, **kwargs: Any) -> ToolResult:
        """Allow calling the tool instance directly."""
        return self.run(**kwargs)

    def validate_arguments(self, **kwargs: Any) -> None:
        """Validate input arguments against the tool's required parameters schema."""
        if not self.parameters:
            return

        required_params: List[str] = self.parameters.get("required", [])
        missing = [param for param in required_params if param not in kwargs or kwargs[param] is None]
        if missing:
            raise ValueError(
                f"Missing required parameter(s) for tool '{self.name}': {', '.join(missing)}"
            )

    def get_definition(self) -> ToolDefinition:
        """Generate a ToolDefinition model for this tool."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            risk_level=self.risk_level,
            requires_confirmation=self.requires_confirmation,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return the dictionary representation of the tool's definition."""
        return self.get_definition().to_dict()
