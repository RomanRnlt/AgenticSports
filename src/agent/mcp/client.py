"""MCP Client -- connects to external MCP servers for additional tools.

Phase 1: Manual tool bridging (load MCP tools as native Tool objects)
Phase 2: Full MCP client protocol when more servers are needed

Current MCP servers we can connect to:
- Brave Search MCP (web search)
- Fetch MCP (web page fetching)
- Future: Garmin MCP, Fitbit MCP, Strava MCP
"""

import os
from src.agent.tools.registry import Tool


def load_mcp_tools() -> list[Tool]:
    """Load tools from configured MCP servers.

    Reads MCP server configuration from environment or config file.
    Returns a list of Tool objects that can be registered in the ToolRegistry.
    """
    tools = []

    # Check for Brave Search MCP
    brave_key = os.environ.get("BRAVE_SEARCH_API_KEY")
    if brave_key:
        tools.append(_create_brave_search_tool(brave_key))

    return tools


def _create_brave_search_tool(api_key: str) -> Tool:
    """Create a web search tool using Brave Search API."""

    def brave_search(query: str) -> dict:
        try:
            import requests
        except ImportError:
            return {"error": "requests library not installed. Run: uv add requests"}

        try:
            resp = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": 5},
                headers={"X-Subscription-Token": api_key},
                timeout=10,
            )
            data = resp.json()
            results = []
            for item in data.get("web", {}).get("results", [])[:5]:
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("description", ""),
                    "url": item.get("url", ""),
                })
            return {"results": results, "source": "brave_search_mcp"}
        except Exception as e:
            return {"error": str(e)}

    return Tool(
        name="web_search",
        description="Search the web using Brave Search",
        handler=brave_search,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
        category="research",
        source="mcp",
    )
