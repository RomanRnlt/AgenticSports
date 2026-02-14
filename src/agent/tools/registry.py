"""Tool Registry -- manages all tools available to the agent.

Equivalent to Claude Code's tool system. Each tool is:
1. A Gemini FunctionDeclaration (schema for the model)
2. A Python callable (implementation)

Tools can be:
- Native (Python functions in this codebase)
- MCP (loaded from external MCP servers)
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from google import genai


@dataclass
class Tool:
    """A single tool available to the agent."""
    name: str
    description: str
    handler: Callable[..., dict]
    parameters: dict = field(default_factory=dict)  # JSON Schema
    category: str = "general"     # data, analysis, research, planning, memory, meta
    source: str = "native"        # native | mcp


class ToolRegistry:
    """Registry of all tools available to the agent."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        """Register a tool."""
        self._tools[tool.name] = tool

    def register_mcp_tools(self, mcp_tools: list[Tool]):
        """Register tools loaded from an MCP server."""
        for tool in mcp_tools:
            tool.source = "mcp"
            self._tools[tool.name] = tool

    def get_declarations(self) -> list[genai.types.FunctionDeclaration]:
        """Get Gemini FunctionDeclaration objects for all registered tools."""
        declarations = []
        for tool in self._tools.values():
            decl = genai.types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=_schema_to_genai(tool.parameters) if tool.parameters else None,
            )
            declarations.append(decl)
        return declarations

    def execute(self, name: str, args: dict) -> dict:
        """Execute a tool by name with given arguments."""
        if name not in self._tools:
            return {"error": f"Unknown tool: {name}"}
        tool = self._tools[name]
        try:
            return tool.handler(**args)
        except TypeError as e:
            return {"error": f"Invalid arguments for {name}: {e}"}
        except Exception as e:
            return {"error": f"Tool {name} failed: {e}"}

    def list_tools(self) -> list[dict]:
        """List all registered tools (for debugging)."""
        return [
            {"name": t.name, "category": t.category, "source": t.source}
            for t in self._tools.values()
        ]


def _schema_to_genai(schema: dict) -> genai.types.Schema:
    """Convert a JSON Schema dict to Gemini Schema object.

    Handles nested objects, arrays, and primitive types.
    """
    type_map = {
        "string": genai.types.Type.STRING,
        "integer": genai.types.Type.INTEGER,
        "number": genai.types.Type.NUMBER,
        "boolean": genai.types.Type.BOOLEAN,
        "array": genai.types.Type.ARRAY,
        "object": genai.types.Type.OBJECT,
    }

    schema_type = type_map.get(schema.get("type", "string"), genai.types.Type.STRING)

    kwargs = {"type": schema_type}

    if "description" in schema:
        kwargs["description"] = schema["description"]

    if "enum" in schema:
        kwargs["enum"] = schema["enum"]

    if "properties" in schema:
        kwargs["properties"] = {
            k: _schema_to_genai(v)
            for k, v in schema["properties"].items()
        }

    if "required" in schema:
        kwargs["required"] = schema["required"]

    if "items" in schema:
        kwargs["items"] = _schema_to_genai(schema["items"])

    if schema.get("nullable"):
        kwargs["nullable"] = True

    return genai.types.Schema(**kwargs)


def get_default_tools(user_model) -> ToolRegistry:
    """Create the default tool registry with all native tools.

    This is called once at agent startup. MCP tools are added separately.
    """
    registry = ToolRegistry()

    # Import and register all tool modules
    from src.agent.tools.data_tools import register_data_tools
    from src.agent.tools.analysis_tools import register_analysis_tools
    from src.agent.tools.planning_tools import register_planning_tools
    from src.agent.tools.memory_tools import register_memory_tools
    from src.agent.tools.research_tools import register_research_tools
    from src.agent.tools.meta_tools import register_meta_tools

    register_data_tools(registry, user_model)
    register_analysis_tools(registry)
    register_planning_tools(registry, user_model)
    register_memory_tools(registry, user_model)
    register_research_tools(registry)
    register_meta_tools(registry, user_model)

    # Register MCP tools (overrides native fallbacks if available)
    from src.agent.mcp.client import load_mcp_tools
    mcp_tools = load_mcp_tools()
    if mcp_tools:
        registry.register_mcp_tools(mcp_tools)

    return registry
