"""VoxGuard Tool Registry.

Provides central registration, lookup, schema extraction,
and execution dispatch for agent tools.
"""

from typing import Any, Callable, Dict, List, Optional, Type, Union

from app.models.tool import ToolDefinition, ToolResult
from app.tools.base import BaseTool
from app.tools.mock_tools import get_mock_tools


class ToolRegistry:
    """Central registry managing tool instances for the agent runtime."""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: Union[BaseTool, Type[BaseTool]], overwrite: bool = False) -> BaseTool:
        """Register a tool instance or class in the registry.

        Args:
            tool: BaseTool instance or subclass.
            overwrite: Whether to overwrite an existing tool with the same name.

        Returns:
            The registered BaseTool instance.
        """
        instance = tool() if isinstance(tool, type) else tool
        if not isinstance(instance, BaseTool):
            raise TypeError(f"Expected BaseTool instance or class, got {type(tool)}")

        if not instance.name:
            raise ValueError("Cannot register a tool with an empty name.")

        if instance.name in self._tools and not overwrite:
            raise ValueError(
                f"Tool '{instance.name}' is already registered. Use overwrite=True to replace it."
            )

        self._tools[instance.name] = instance
        return instance

    def register_tool(self, tool_or_class: Optional[Union[BaseTool, Type[BaseTool]]] = None, overwrite: bool = False) -> Any:
        """Decorator or method to register a tool."""
        def decorator(cls_or_inst: Union[BaseTool, Type[BaseTool]]) -> Union[BaseTool, Type[BaseTool]]:
            self.register(cls_or_inst, overwrite=overwrite)
            return cls_or_inst

        if tool_or_class is not None:
            return decorator(tool_or_class)
        return decorator

    def unregister(self, name: str) -> BaseTool:
        """Unregister and return a tool by name."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered.")
        return self._tools.pop(name)

    def get(self, name: str, default: Optional[BaseTool] = None) -> Optional[BaseTool]:
        """Retrieve a tool by name, returning default if not found."""
        return self._tools.get(name, default)

    def get_tool(self, name: str) -> BaseTool:
        """Retrieve a tool by name or raise KeyError if not found."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered.")
        return self._tools[name]

    def has_tool(self, name: str) -> bool:
        """Check whether a tool is registered."""
        return name in self._tools

    def list_tools(self) -> List[BaseTool]:
        """Return a list of all registered tool instances."""
        return list(self._tools.values())

    def list_names(self) -> List[str]:
        """Return a list of all registered tool names."""
        return list(self._tools.keys())

    def get_definitions(self) -> List[ToolDefinition]:
        """Retrieve definitions for all registered tools."""
        return [tool.get_definition() for tool in self._tools.values()]

    def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Dispatch execution to the registered tool by name."""
        tool = self.get(tool_name)
        if tool is None:
            return ToolResult.fail(
                tool_name=tool_name,
                error=f"Tool '{tool_name}' is not registered in the tool registry.",
            )
        return tool.run(**kwargs)

    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()

    def __contains__(self, name: str) -> bool:
        return self.has_tool(name)

    def __len__(self) -> int:
        return len(self._tools)

    def __getitem__(self, name: str) -> BaseTool:
        return self.get_tool(name)

    def __iter__(self):
        return iter(self._tools.values())


def create_default_registry() -> ToolRegistry:
    """Create a registry instance pre-loaded with all standard mock tools."""
    registry = ToolRegistry()
    for tool in get_mock_tools():
        registry.register(tool)
    return registry


# Default global tool registry pre-loaded with mock tools
tool_registry = create_default_registry()
