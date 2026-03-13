"""MCP Bridge — bidirectional Model Context Protocol integration.

Exposes anchor capabilities as MCP tools/resources/prompts and
consumes external MCP servers from anchor's Agent.
"""

from __future__ import annotations

from anchor.mcp.errors import (
    MCPConfigError,
    MCPConnectionError,
    MCPError,
    MCPTimeoutError,
    MCPToolError,
)
from anchor.mcp.models import (
    MCPPrompt,
    MCPPromptArgument,
    MCPResource,
    MCPServerConfig,
)
from anchor.mcp.protocols import MCPClient, MCPServer

__all__ = [
    "MCPClient",
    "MCPConfigError",
    "MCPConnectionError",
    "MCPError",
    "MCPPrompt",
    "MCPPromptArgument",
    "MCPResource",
    "MCPServer",
    "MCPServerConfig",
    "MCPTimeoutError",
    "MCPToolError",
]
