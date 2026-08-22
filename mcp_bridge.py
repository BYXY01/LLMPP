"""
mcp_bridge.py - Standalone MCP client bridge for LLMPP.

LLMPP connects to external MCP servers through this module. It is entirely
optional: if the `mcp` package is not installed (or this file is missing),
LLMPP simply runs without MCP tools.

This module reads its own configuration file `mcp_config.json` (same
directory) listing the external MCP servers to connect to:

    {
        "servers": [
            {"name": "math", "command": ["python", "/path/to/mcp_server.py"]},
            {"name": "remote", "url": "http://127.0.0.1:8000/mcp"}
        ]
    }

Transports:
    command            -> stdio (spawn a local process)
    url ending /sse    -> SSE
    url (otherwise)    -> streamable HTTP

It can be run standalone to inspect tools:
    python mcp_bridge.py

Interface (mirrors PluginManager style):
    from mcp_bridge import MCPClient
    client = MCPClient()
    tools  = client.load()          # List[Dict] OpenAI tool schemas
    result = client.call(name, args)  # str
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.client.sse import sse_client

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MCP_CONFIG_PATH = os.path.join(BASE_DIR, "mcp_config.json")

DEFAULT_SERVERS: List[Dict[str, Any]] = []


def load_config() -> List[Dict[str, Any]]:
    """Read the MCP server list from mcp_config.json."""
    if not os.path.exists(MCP_CONFIG_PATH):
        return list(DEFAULT_SERVERS)
    try:
        with open(MCP_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("servers", [])
    except Exception as e:
        print(f"[mcp] failed to read {MCP_CONFIG_PATH}: {e}")
        return list(DEFAULT_SERVERS)


class MCPClient:
    """Connect to external MCP servers and expose their tools to LLMPP."""

    def __init__(self, servers: Optional[List[Dict[str, Any]]] = None):
        self.servers = servers if servers is not None else load_config()
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._tool_owner: Dict[str, str] = {}

    # --- public -----------------------------------------------------------

    def load(self) -> List[Dict[str, Any]]:
        """Connect to every configured server and fetch its tools.

        Returns OpenAI-format tool schemas. Tools are cached; a server that
        fails to connect is skipped with a log message.
        """
        self._tools.clear()
        self._tool_owner.clear()
        for srv in self.servers:
            name = srv.get("name", "unnamed")
            try:
                schemas = asyncio.run(self._fetch_server_tools(srv))
            except BaseException as e:
                print(f"[mcp] server '{name}' failed: {e}")
                continue
            for schema in schemas:
                tool_name = schema["function"]["name"]
                self._tools[tool_name] = schema
                self._tool_owner[tool_name] = name
        return list(self._tools.values())

    def call(self, name: str, args: Dict[str, Any]) -> str:
        """Execute a tool on its owning MCP server; return the result as text."""
        owner = self._tool_owner.get(name)
        if owner is None:
            return f"[error] MCP tool not found: {name}"
        srv = next((s for s in self.servers if s.get("name") == owner), None)
        if srv is None:
            return f"[error] MCP server not found: {owner}"
        try:
            result = asyncio.run(self._call_server_tool(srv, name, args))
            return result
        except BaseException as e:
            return f"[error] MCP tool call failed: {e}"

    async def call_async(self, name: str, args: Dict[str, Any]) -> str:
        """Async version of call(); for use inside a running event loop."""
        owner = self._tool_owner.get(name)
        if owner is None:
            return f"[error] MCP tool not found: {name}"
        srv = next((s for s in self.servers if s.get("name") == owner), None)
        if srv is None:
            return f"[error] MCP server not found: {owner}"
        try:
            return await self._call_server_tool(srv, name, args)
        except BaseException as e:
            return f"[error] MCP tool call failed: {e}"

    # --- internals --------------------------------------------------------

    @staticmethod
    def _open_client(srv: Dict[str, Any]):
        """Return an async context manager for a server connection."""
        if "command" in srv:
            params = StdioServerParameters(command=srv["command"][0], args=srv["command"][1:])
            return stdio_client(params)
        url = srv["url"]
        if url.endswith("/sse"):
            return sse_client(url)
        return streamable_http_client(url)

    @staticmethod
    def _to_openai_schema(tool) -> Dict[str, Any]:
        schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": schema or {"type": "object", "properties": {}},
            },
        }

    async def _fetch_server_tools(self, srv: Dict[str, Any]) -> List[Dict[str, Any]]:
        async with self._open_client(srv) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                return [self._to_openai_schema(t) for t in tools.tools]

    async def _call_server_tool(self, srv: Dict[str, Any], name: str, args: Dict[str, Any]) -> str:
        async with self._open_client(srv) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, args)
                parts = []
                for item in result.content:
                    if hasattr(item, "text"):
                        parts.append(item.text)
                    else:
                        parts.append(str(item))
                return "\n".join(parts)


def main():
    """Standalone: print the tools exposed by all configured servers."""
    client = MCPClient()
    tools = client.load()
    print(f"[mcp] loaded {len(tools)} tool(s) from {len(client.servers)} server(s)")
    for t in tools:
        print(f"  - {t['function']['name']}: {t['function']['description']}")
    if tools and len(sys.argv) > 1:
        name, args = sys.argv[1], {}
        if len(sys.argv) > 2:
            try:
                args = json.loads(sys.argv[2])
            except json.JSONDecodeError:
                pass
        print(f"[mcp] {name}{args} -> {client.call(name, args)}")


if __name__ == "__main__":
    main()
