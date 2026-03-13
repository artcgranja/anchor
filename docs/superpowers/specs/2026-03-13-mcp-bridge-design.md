# MCP Bridge Design Spec

**Date:** 2026-03-13
**Status:** Draft
**Version:** anchor v0.2.0
**Approach:** Protocol-First with FastMCP 3.0 Backend

## Overview

Bidirectional MCP Bridge for anchor: expose anchor's full toolkit (RAG, memory, pipeline, agent, ingestion, observability) as MCP tools/resources/prompts to external clients, AND consume external MCP servers from anchor's Agent.

Built on FastMCP 3.0 (2026 state-of-the-art) with anchor-native PEP 544 Protocols for swap-ability.

## Research Basis

Designed from analysis of 2026 MCP integration patterns across:

- **FastMCP 3.0** — Provider/Transform architecture, ProxyProvider, OAuth 2.1
- **LangGraph** — `MultiServerMCPClient`, dict-based config, tool name prefixing
- **CrewAI** — `mcps=[]` on Agent, auto-managed lifecycle
- **Agno** — `MCPTools`/`MultiMCPTools`, command strings, async context manager
- **OpenAI Agents SDK** — `MCPServerStdio`/`MCPServerSse`, `cache_tools_list`
- **MCP Spec 2026** — Streamable HTTP transport, OAuth 2.1 + PKCE, Client Identifier Metadata

Key best practices adopted:
1. Async context manager for connection lifecycle (universal pattern)
2. Tool name prefixing to prevent collisions (LangGraph)
3. Tool list caching to avoid re-fetching on every turn (OpenAI SDK)
4. Simple API: accept command strings and URLs directly (Agno/CrewAI)
5. Full MCP primitives: tools, resources, AND prompts (not tools-only)
6. Deferred/lazy loading of tool schemas (2026 context-efficiency pattern)

## Module Structure

```
src/anchor/mcp/
├── __init__.py          # Public exports
├── models.py            # MCPServerConfig, MCPResource, MCPPrompt models
├── protocols.py         # MCPClient, MCPServer protocols (PEP 544)
├── client.py            # FastMCPClientBridge — consumes external MCP servers
├── server.py            # FastMCPServerBridge — exposes anchor as MCP server
├── tools.py             # Adapter: MCP tools <-> AgentTool conversion
└── errors.py            # MCPError hierarchy
```

New optional dependency in `pyproject.toml`:
```toml
[project.optional-dependencies]
mcp = ["fastmcp>=3.0,<4"]
```

## Protocols

### MCPClient Protocol

```python
from __future__ import annotations

from typing import Any, Protocol, Self, runtime_checkable

from anchor.llm.models import ToolSchema
from anchor.mcp.models import MCPPrompt, MCPResource


@runtime_checkable
class MCPClient(Protocol):
    """Protocol for connecting to and consuming external MCP servers.

    Any object implementing this interface can be used to bridge
    external MCP servers into anchor's Agent tool loop.

    Follows the async context manager pattern (universal across
    LangGraph, Agno, OpenAI Agents SDK, FastMCP Client).
    """

    async def connect(self) -> None:
        """Establish connection to the MCP server."""
        ...

    async def disconnect(self) -> None:
        """Close the connection to the MCP server."""
        ...

    async def list_tools(self) -> list[ToolSchema]:
        """List all tools available on the connected MCP server.

        Returns anchor-native ToolSchema objects.
        """
        ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool on the MCP server and return the result."""
        ...

    async def list_resources(self) -> list[MCPResource]:
        """List all resources available on the connected MCP server."""
        ...

    async def read_resource(self, uri: str) -> str:
        """Read a resource by URI from the MCP server."""
        ...

    async def list_prompts(self) -> list[MCPPrompt]:
        """List all prompt templates on the connected MCP server."""
        ...

    async def get_prompt(self, name: str, arguments: dict[str, Any]) -> str:
        """Get a rendered prompt template from the MCP server."""
        ...

    async def __aenter__(self) -> Self:
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        ...
```

### MCPServer Protocol

```python
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from anchor.agent.models import AgentTool


@runtime_checkable
class MCPServer(Protocol):
    """Protocol for exposing anchor capabilities as an MCP server.

    Registration methods are synchronous (setup-time), while
    the server run loop is async.
    """

    def expose_tool(self, tool: AgentTool) -> None:
        """Register an AgentTool as an MCP tool."""
        ...

    def expose_tools(self, tools: list[AgentTool]) -> None:
        """Register multiple AgentTools as MCP tools."""
        ...

    def expose_resource(
        self, uri: str, handler: Callable[..., str] | Callable[..., Any],
    ) -> None:
        """Register a resource handler at the given URI.

        Accepts both sync and async handlers.
        """
        ...

    def expose_prompt(
        self, name: str, handler: Callable[..., str] | Callable[..., Any],
    ) -> None:
        """Register a prompt template handler.

        Accepts both sync and async handlers.
        """
        ...

    async def run(self, transport: str = "stdio") -> None:
        """Start serving on the specified transport.

        Supported transports: "stdio", "http", "sse".
        """
        ...
```

## Models

All models use Pydantic v2 with `frozen=True`, matching anchor conventions.

```python
from __future__ import annotations

from pydantic import BaseModel


class MCPServerConfig(BaseModel, frozen=True):
    """Configuration for connecting to an external MCP server.

    Accepts both command (STDIO) and url (HTTP) — transport is
    auto-inferred from which field is provided, following
    FastMCP Client's pattern.
    """

    name: str
    """Unique server name. Used for tool name prefixing."""

    command: str | None = None
    """STDIO transport: command to spawn (e.g., 'npx', 'python')."""

    args: list[str] | None = None
    """STDIO transport: command arguments."""

    url: str | None = None
    """HTTP/SSE transport: server URL."""

    env: dict[str, str] | None = None
    """Environment variables passed to the server process."""

    headers: dict[str, str] | None = None
    """HTTP headers for authentication (e.g., Bearer tokens)."""

    cache_tools: bool = True
    """Cache the tool list after first discovery. Avoids re-fetching
    on every agent turn. Pattern from OpenAI Agents SDK."""

    prefix_tools: bool = True
    """Prefix tool names with server name to prevent collisions.
    E.g., 'filesystem_read_file'. Pattern from LangGraph."""

    timeout: float = 30.0
    """Timeout in seconds for MCP server operations (connect, tool calls)."""

    @model_validator(mode="after")
    def _check_transport(self) -> Self:
        if self.command is None and self.url is None:
            msg = "MCPServerConfig requires either 'command' (STDIO) or 'url' (HTTP)"
            raise ValueError(msg)
        return self


class MCPResource(BaseModel, frozen=True):
    """Descriptor for an MCP resource."""

    uri: str
    name: str
    description: str | None = None
    mime_type: str | None = None


class MCPPromptArgument(BaseModel, frozen=True):
    """A single argument for an MCP prompt template."""

    name: str
    description: str | None = None
    required: bool = False


class MCPPrompt(BaseModel, frozen=True):
    """Descriptor for an MCP prompt template."""

    name: str
    description: str | None = None
    arguments: list[MCPPromptArgument] | None = None
```

## Client Implementation: FastMCPClientBridge

```python
class FastMCPClientBridge:
    """Connects to external MCP servers using FastMCP Client.

    Implements the MCPClient protocol. Adapts MCP tools to
    anchor's AgentTool format for seamless Agent integration.

    Usage::

        config = MCPServerConfig(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        )
        async with FastMCPClientBridge(config) as client:
            tools = await client.as_agent_tools()
            # tools are AgentTool instances ready for Agent.with_tools()
    """

    def __init__(self, config: MCPServerConfig) -> None:
        self._config = config
        self._client: Client | None = None  # FastMCP Client
        self._cached_tools: list[ToolSchema] | None = None

    async def connect(self) -> None:
        """Connect using auto-inferred transport.

        - config.command → STDIO transport
        - config.url → HTTP/Streamable HTTP transport
        """
        # Build FastMCP Client from config
        # Enter its async context

    async def disconnect(self) -> None:
        """Close the FastMCP Client connection."""

    async def list_tools(self) -> list[ToolSchema]:
        """List tools, with optional caching."""
        if self._config.cache_tools and self._cached_tools is not None:
            return self._cached_tools
        # Fetch from FastMCP Client, convert to ToolSchema
        # Cache if config.cache_tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute tool via FastMCP Client, return result as string."""

    async def list_resources(self) -> list[MCPResource]: ...
    async def read_resource(self, uri: str) -> str: ...
    async def list_prompts(self) -> list[MCPPrompt]: ...
    async def get_prompt(self, name: str, arguments: dict[str, Any]) -> str: ...

    async def as_agent_tools(self) -> list[AgentTool]:
        """Convert all MCP tools to AgentTool instances.

        Each AgentTool wraps the MCP tool's schema and provides an async-aware
        callable. Since Agent._execute_tool() is synchronous, the Agent class
        must be extended with an _aexecute_tool() async method that handles
        MCP tools natively via await. The achat() path will use _aexecute_tool()
        instead of _execute_tool() for MCP-sourced tools.

        AgentTool.fn for MCP tools raises MCPError if called synchronously,
        guiding users to use achat() instead of chat().

        If config.prefix_tools is True, tool names are prefixed:
        'read_file' → 'filesystem_read_file'
        """

    # Async context manager
    async def __aenter__(self) -> Self: ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None: ...
```

## Multi-Server Pool: MCPClientPool

```python
class MCPClientPool:
    """Manages connections to multiple MCP servers.

    Convenience class for connecting to several servers at once
    and collecting all their tools as AgentTool instances.

    Usage::

        pool = MCPClientPool([
            MCPServerConfig(name="fs", command="npx", args=[...]),
            MCPServerConfig(name="db", url="http://localhost:8080/mcp"),
        ])
        async with pool:
            tools = await pool.all_agent_tools()
            agent = Agent().with_tools(tools)
    """

    def __init__(self, configs: list[MCPServerConfig]) -> None:
        self._configs = configs
        self._clients: list[MCPClient] = []  # Protocol-typed, not concrete class

    async def connect_all(self) -> None:
        """Connect to all configured servers concurrently."""
        # asyncio.gather for parallel connection

    async def disconnect_all(self) -> None:
        """Disconnect all servers."""

    async def all_agent_tools(self) -> list[AgentTool]:
        """Collect AgentTool instances from all connected servers."""

    async def __aenter__(self) -> Self: ...
    async def __aexit__(self, *args) -> None: ...
```

## Server Implementation: FastMCPServerBridge

```python
class FastMCPServerBridge:
    """Exposes anchor capabilities as an MCP server via FastMCP.

    Usage::

        # From an Agent
        server = FastMCPServerBridge.from_agent(agent)
        await server.run(transport="stdio")

        # Manual registration
        server = FastMCPServerBridge("my-anchor-server")
        server.expose_tools(my_tools)
        server.expose_resource("context://memory", memory_handler)
        await server.run(transport="http")
    """

    def __init__(self, name: str = "anchor") -> None:
        self._mcp = FastMCP(name)

    def expose_tool(self, tool: AgentTool) -> None:
        """Register an AgentTool as an MCP tool.

        Converts AgentTool's fn + input_schema to FastMCP tool format.
        """

    def expose_tools(self, tools: list[AgentTool]) -> None:
        """Register multiple tools."""

    def expose_resource(
        self, uri: str, handler: Callable[..., str] | Callable[..., Any],
    ) -> None:
        """Register a resource at the given URI pattern.

        Accepts both sync and async handlers.
        """

    def expose_prompt(
        self, name: str, handler: Callable[..., str] | Callable[..., Any],
    ) -> None:
        """Register a prompt template.

        Accepts both sync and async handlers.
        """

    @classmethod
    def from_agent(cls, agent: Agent) -> FastMCPServerBridge:
        """Create server from an Agent, exposing all its tools.

        Also exposes:
        - resource 'context://pipeline' → pipeline query
        - resource 'context://memory' → memory state
        - prompt 'chat' → send a message through the agent
        """

    @classmethod
    def from_pipeline(cls, pipeline: ContextPipeline) -> FastMCPServerBridge:
        """Create server exposing pipeline as retrieval tools.

        Exposes:
        - tool 'query' → run pipeline.build() and return results
        - resource 'context://result' → last pipeline result
        """

    async def run(self, transport: str = "stdio") -> None:
        """Start the MCP server.

        Args:
            transport: "stdio" (for Claude Desktop, local dev),
                      "http" (Streamable HTTP, 2026 standard for remote),
                      "sse" (legacy, supported for compatibility).
        """
```

## Agent Integration

New fluent method on `Agent`:

```python
# In agent.py
def with_mcp_servers(
    self,
    servers: list[str | MCPServerConfig],
) -> Agent:
    """Connect to external MCP servers. Returns self for chaining.

    Accepts mixed list of:
    - Command strings: "npx -y @modelcontextprotocol/server-filesystem /tmp"
    - URLs: "http://localhost:8080/mcp"
    - MCPServerConfig objects for full control

    Tools from MCP servers are lazily loaded on first achat() call
    and cached for subsequent calls (deferred loading pattern).

    Usage::

        agent = (
            Agent(model="claude-sonnet-4-5-20250514")
            .with_system_prompt("You are helpful.")
            .with_mcp_servers([
                "npx -y @modelcontextprotocol/server-filesystem /tmp",
                "http://localhost:8080/mcp",
                MCPServerConfig(name="db", command="python", args=["db_server.py"]),
            ])
        )

        async for chunk in agent.achat("List files in /tmp"):
            print(chunk, end="")
    """
```

### Config Parsing Logic

String arguments are auto-parsed into `MCPServerConfig`:

- Strings starting with `http://` or `https://` → URL-based config (HTTP transport)
- Other strings → split by whitespace, first token is `command`, rest are `args` (STDIO transport)
- `MCPServerConfig` objects passed through directly
- Server names auto-generated from command/URL when not specified

### Lazy Connection

MCP servers connect lazily on first `achat()` call:

1. `achat()` checks if MCP pool is connected
2. If not, connects all servers via `MCPClientPool.connect_all()`
3. Discovers tools via `pool.all_agent_tools()`
4. Adds to `_all_active_tools()` alongside regular tools and skill tools
5. Subsequent calls use cached tools (if `cache_tools=True`)

### Sync `chat()` Limitation

The sync `chat()` method cannot natively support MCP (which is async-only). Options:

1. Raise `TypeError` if MCP servers are configured but `chat()` is called (not `achat()`)
2. Use `asyncio.run()` internally for the connection phase only

Recommendation: Option 1 with a clear error message pointing to `achat()`.

## Error Handling

```python
class MCPError(AstroContextError):
    """Base error for all MCP operations."""

class MCPConnectionError(MCPError):
    """Failed to connect to an MCP server.

    Includes server name and transport details in the message.
    """

class MCPToolError(MCPError):
    """Error executing an MCP tool.

    Includes tool name, server name, and the original error.
    """

class MCPTimeoutError(MCPError):
    """MCP server operation timed out."""

class MCPConfigError(MCPError):
    """Invalid MCP server configuration.

    E.g., neither command nor url specified.
    """
```

Follows anchor's existing `exceptions.py` pattern where all errors inherit from `AstroContextError`.

## Testing Strategy

### Unit Tests (`tests/test_mcp/`)

```
tests/test_mcp/
├── test_models.py           # MCPServerConfig, MCPResource, MCPPrompt
├── test_protocols.py         # Protocol conformance checks
├── test_client.py            # FastMCPClientBridge with mocked FastMCP Client
├── test_server.py            # FastMCPServerBridge with in-memory FastMCP
├── test_tools.py             # AgentTool <-> MCP tool conversion
├── test_pool.py              # MCPClientPool multi-server management
├── test_agent_integration.py # Agent.with_mcp_servers() behavior
└── test_errors.py            # Error hierarchy and messages
```

### Test Approach

1. **Unit tests** — mock FastMCP Client internals, test conversion logic
2. **Integration tests** — use FastMCP's in-process server (no real subprocess/network)
3. **Round-trip test** — Agent → expose as MCP server → consume from another Agent
4. **Edge cases** — server disconnect during tool call, tool name collisions, empty tool lists
5. **Async-only** — all tests use pytest-asyncio with `asyncio_mode = "auto"`
6. **Coverage target** — 94%+ (matching project standard)

## Public API Exports

Added to `src/anchor/__init__.py`:

```python
# MCP Bridge
from anchor.mcp import (
    FastMCPClientBridge,
    FastMCPServerBridge,
    MCPClient,
    MCPClientPool,
    MCPConfigError,
    MCPConnectionError,
    MCPError,
    MCPPrompt,
    MCPPromptArgument,
    MCPResource,
    MCPServer,
    MCPServerConfig,
    MCPTimeoutError,
    MCPToolError,
)
```

## Dependencies

```toml
[project.optional-dependencies]
mcp = ["fastmcp>=3.0,<4"]
```

FastMCP 3.0 brings in the official `mcp` SDK as a transitive dependency. No need to depend on both.

## Implementation Notes

### Agent `__slots__` Update

The `Agent` class uses `__slots__`. New slots needed:
- `_mcp_configs` — stored `MCPServerConfig` list
- `_mcp_pool` — lazy `MCPClientPool` instance
- `_mcp_tools` — cached `list[AgentTool]` from MCP servers

### Agent Async Tool Execution

Add `_aexecute_tool()` async method alongside existing `_execute_tool()`.
The `achat()` path calls `_aexecute_tool()` which can `await` MCP tool calls
natively. The sync `chat()` path continues to use `_execute_tool()`.

MCP-sourced `AgentTool` instances are tagged (e.g., via a sentinel `fn` that
raises `MCPError`) so `_execute_tool()` can detect and reject them with a
clear error pointing to `achat()`.

### Circular Import Prevention

`FastMCPServerBridge.from_agent()` and `from_pipeline()` use `TYPE_CHECKING`
guard for the `Agent` and `ContextPipeline` imports:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anchor.agent.agent import Agent
    from anchor.pipeline.pipeline import ContextPipeline
```

### Module `__all__`

`mcp/__init__.py` defines `__all__` with all public names, following the
pattern in `exceptions.py` and other anchor modules.

### Field Naming

`MCPResource.mime_type` uses snake_case (Pythonic) rather than MCP spec's
`mimeType` (camelCase). This is intentional and consistent with how anchor
handles all external protocol fields.

## Future Considerations (Out of Scope for v0.2.0)

- OAuth 2.1 + PKCE auth provider integration (FastMCP has built-in support)
- MCP resource subscriptions (real-time updates)
- Tool search tool pattern (lazy discovery for large tool sets)
- MCP Apps (interactive UI components, SEP-1865)
- Agent-to-Agent via MCP (2026 roadmap extension)
