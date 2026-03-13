# MCP Bridge Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bidirectional MCP Bridge to anchor — expose anchor capabilities as MCP server AND consume external MCP servers from Agent.

**Architecture:** Protocol-First design with PEP 544 Protocols backed by FastMCP 3.0. New `src/anchor/mcp/` module with errors, models, protocols, tool adapters, client bridge, server bridge, and Agent integration.

**Tech Stack:** Python 3.11+, Pydantic v2 (frozen models), FastMCP 3.0 (`fastmcp>=3.0,<4`), pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-03-13-mcp-bridge-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `src/anchor/mcp/__init__.py` | Public exports with `__all__` |
| `src/anchor/mcp/errors.py` | `MCPError` hierarchy inheriting from `AstroContextError` |
| `src/anchor/mcp/models.py` | `MCPServerConfig`, `MCPResource`, `MCPPrompt`, `MCPPromptArgument` |
| `src/anchor/mcp/protocols.py` | `MCPClient` and `MCPServer` PEP 544 Protocols |
| `src/anchor/mcp/tools.py` | `mcp_tool_to_agent_tool()` and `parse_server_string()` adapters |
| `src/anchor/mcp/client.py` | `FastMCPClientBridge` and `MCPClientPool` |
| `src/anchor/mcp/server.py` | `FastMCPServerBridge` with `from_agent()` / `from_pipeline()` |
| `tests/test_mcp/__init__.py` | Test package marker |
| `tests/test_mcp/test_errors.py` | Error hierarchy tests |
| `tests/test_mcp/test_models.py` | Model validation tests |
| `tests/test_mcp/test_protocols.py` | Protocol conformance tests |
| `tests/test_mcp/test_tools.py` | Tool adapter tests |
| `tests/test_mcp/test_client.py` | Client bridge tests |
| `tests/test_mcp/test_pool.py` | Multi-server pool tests |
| `tests/test_mcp/test_server.py` | Server bridge tests |
| `tests/test_mcp/test_agent_integration.py` | Agent.with_mcp_servers() tests |

### Modified Files

| File | Change |
|------|--------|
| `pyproject.toml` | Add `mcp = ["fastmcp>=3.0,<4"]` to optional-dependencies |
| `src/anchor/__init__.py` | Add MCP exports to `__all__` |
| `src/anchor/agent/agent.py` | Add `with_mcp_servers()`, `_aexecute_tool()`, `_arun_tools()`, new `__slots__` |

---

## Chunk 1: Foundation (Errors, Models, Protocols)

### Task 1: Add FastMCP dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add mcp optional dependency**

In `pyproject.toml`, add to `[project.optional-dependencies]`:

```toml
mcp = ["fastmcp>=3.0,<4"]
```

- [ ] **Step 2: Install the dependency**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-panini && uv sync --extra mcp`
Expected: fastmcp 3.x installed successfully

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add fastmcp>=3.0 as optional mcp dependency"
```

---

### Task 2: Create MCP error hierarchy

**Files:**
- Create: `src/anchor/mcp/__init__.py`
- Create: `src/anchor/mcp/errors.py`
- Create: `tests/test_mcp/__init__.py`
- Create: `tests/test_mcp/test_errors.py`

- [ ] **Step 1: Write failing tests for error hierarchy**

Create `tests/test_mcp/__init__.py` (empty file).

Create `tests/test_mcp/test_errors.py`:

```python
"""Tests for MCP error hierarchy."""

from __future__ import annotations

from anchor.exceptions import AstroContextError
from anchor.mcp.errors import (
    MCPConfigError,
    MCPConnectionError,
    MCPError,
    MCPTimeoutError,
    MCPToolError,
)


class TestMCPErrorHierarchy:
    """All MCP errors must inherit from AstroContextError."""

    def test_mcp_error_is_astro_context_error(self) -> None:
        err = MCPError("test")
        assert isinstance(err, AstroContextError)

    def test_mcp_connection_error_is_mcp_error(self) -> None:
        err = MCPConnectionError("failed", server_name="fs", transport="stdio")
        assert isinstance(err, MCPError)
        assert err.server_name == "fs"
        assert err.transport == "stdio"

    def test_mcp_tool_error_is_mcp_error(self) -> None:
        cause = RuntimeError("boom")
        err = MCPToolError("failed", tool_name="read", server_name="fs", cause=cause)
        assert isinstance(err, MCPError)
        assert err.tool_name == "read"
        assert err.server_name == "fs"
        assert err.cause is cause

    def test_mcp_timeout_error_is_mcp_error(self) -> None:
        err = MCPTimeoutError("timed out")
        assert isinstance(err, MCPError)

    def test_mcp_config_error_is_mcp_error(self) -> None:
        err = MCPConfigError("bad config")
        assert isinstance(err, MCPError)

    def test_all_errors_catchable_as_base(self) -> None:
        errors = [
            MCPConnectionError("x", server_name="s", transport="t"),
            MCPToolError("x", tool_name="t", server_name="s"),
            MCPTimeoutError("x"),
            MCPConfigError("x"),
        ]
        for err in errors:
            try:
                raise err
            except MCPError:
                pass  # All should be caught
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-panini && python -m pytest tests/test_mcp/test_errors.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'anchor.mcp'`

- [ ] **Step 3: Create the mcp package and errors module**

Create `src/anchor/mcp/__init__.py`:

```python
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

__all__ = [
    "MCPConfigError",
    "MCPConnectionError",
    "MCPError",
    "MCPTimeoutError",
    "MCPToolError",
]
```

Create `src/anchor/mcp/errors.py`:

```python
"""MCP error hierarchy.

All errors inherit from AstroContextError for consistent
exception handling across the anchor toolkit.
"""

from __future__ import annotations

from anchor.exceptions import AstroContextError


class MCPError(AstroContextError):
    """Base error for all MCP operations."""


class MCPConnectionError(MCPError):
    """Failed to connect to an MCP server."""

    def __init__(
        self,
        message: str,
        *,
        server_name: str,
        transport: str = "",
    ) -> None:
        full_msg = f"[{server_name}] {message}" if server_name else message
        super().__init__(full_msg)
        self.server_name = server_name
        self.transport = transport


class MCPToolError(MCPError):
    """Error executing an MCP tool."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        server_name: str,
        cause: BaseException | None = None,
    ) -> None:
        full_msg = f"[{server_name}/{tool_name}] {message}"
        super().__init__(full_msg)
        self.tool_name = tool_name
        self.server_name = server_name
        self.cause = cause


class MCPTimeoutError(MCPError):
    """MCP server operation timed out."""


class MCPConfigError(MCPError):
    """Invalid MCP server configuration."""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-panini && python -m pytest tests/test_mcp/test_errors.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/anchor/mcp/__init__.py src/anchor/mcp/errors.py tests/test_mcp/__init__.py tests/test_mcp/test_errors.py
git commit -m "feat(mcp): add error hierarchy inheriting from AstroContextError"
```

---

### Task 3: Create MCP models

**Files:**
- Create: `src/anchor/mcp/models.py`
- Create: `tests/test_mcp/test_models.py`
- Modify: `src/anchor/mcp/__init__.py`

- [ ] **Step 1: Write failing tests for models**

Create `tests/test_mcp/test_models.py`:

```python
"""Tests for MCP data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from anchor.mcp.models import (
    MCPPrompt,
    MCPPromptArgument,
    MCPResource,
    MCPServerConfig,
)


class TestMCPServerConfig:
    """MCPServerConfig validation and behavior."""

    def test_stdio_config(self) -> None:
        cfg = MCPServerConfig(name="fs", command="npx", args=["-y", "server"])
        assert cfg.name == "fs"
        assert cfg.command == "npx"
        assert cfg.args == ["-y", "server"]
        assert cfg.url is None
        assert cfg.cache_tools is True
        assert cfg.prefix_tools is True
        assert cfg.timeout == 30.0

    def test_http_config(self) -> None:
        cfg = MCPServerConfig(name="api", url="http://localhost:8080/mcp")
        assert cfg.url == "http://localhost:8080/mcp"
        assert cfg.command is None

    def test_requires_command_or_url(self) -> None:
        with pytest.raises(ValidationError, match="command.*STDIO.*url.*HTTP"):
            MCPServerConfig(name="bad")

    def test_frozen(self) -> None:
        cfg = MCPServerConfig(name="fs", command="npx")
        with pytest.raises(ValidationError):
            cfg.name = "other"  # type: ignore[misc]

    def test_with_headers(self) -> None:
        cfg = MCPServerConfig(
            name="api",
            url="http://example.com",
            headers={"Authorization": "Bearer tok"},
        )
        assert cfg.headers == {"Authorization": "Bearer tok"}

    def test_with_env(self) -> None:
        cfg = MCPServerConfig(
            name="db",
            command="python",
            args=["server.py"],
            env={"DB_URL": "postgres://localhost"},
        )
        assert cfg.env == {"DB_URL": "postgres://localhost"}

    def test_custom_timeout(self) -> None:
        cfg = MCPServerConfig(name="slow", url="http://slow.example.com", timeout=120.0)
        assert cfg.timeout == 120.0


class TestMCPResource:
    """MCPResource model."""

    def test_basic(self) -> None:
        r = MCPResource(uri="file:///tmp/data.txt", name="data")
        assert r.uri == "file:///tmp/data.txt"
        assert r.name == "data"
        assert r.description is None
        assert r.mime_type is None

    def test_with_metadata(self) -> None:
        r = MCPResource(
            uri="db://users",
            name="users",
            description="User table",
            mime_type="application/json",
        )
        assert r.description == "User table"
        assert r.mime_type == "application/json"

    def test_frozen(self) -> None:
        r = MCPResource(uri="x", name="y")
        with pytest.raises(ValidationError):
            r.uri = "z"  # type: ignore[misc]


class TestMCPPrompt:
    """MCPPrompt and MCPPromptArgument models."""

    def test_basic_prompt(self) -> None:
        p = MCPPrompt(name="analyze")
        assert p.name == "analyze"
        assert p.description is None
        assert p.arguments is None

    def test_prompt_with_arguments(self) -> None:
        p = MCPPrompt(
            name="summarize",
            description="Summarize content",
            arguments=[
                MCPPromptArgument(name="text", required=True),
                MCPPromptArgument(name="max_length", description="Max words"),
            ],
        )
        assert len(p.arguments) == 2
        assert p.arguments[0].required is True
        assert p.arguments[1].required is False

    def test_frozen(self) -> None:
        p = MCPPrompt(name="x")
        with pytest.raises(ValidationError):
            p.name = "y"  # type: ignore[misc]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-panini && python -m pytest tests/test_mcp/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'anchor.mcp.models'`

- [ ] **Step 3: Implement models**

Create `src/anchor/mcp/models.py`:

```python
"""MCP data models.

All models use Pydantic v2 with frozen=True, matching anchor conventions.
Field names use snake_case (Pythonic) rather than MCP spec's camelCase.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, model_validator


class MCPServerConfig(BaseModel, frozen=True):
    """Configuration for connecting to an external MCP server.

    Accepts both command (STDIO) and url (HTTP) — transport is
    auto-inferred from which field is provided.
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
    """Cache the tool list after first discovery."""

    prefix_tools: bool = True
    """Prefix tool names with server name to prevent collisions."""

    timeout: float = 30.0
    """Timeout in seconds for MCP server operations."""

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

- [ ] **Step 4: Update `__init__.py` with model exports**

Add to `src/anchor/mcp/__init__.py`:

```python
from anchor.mcp.models import (
    MCPPrompt,
    MCPPromptArgument,
    MCPResource,
    MCPServerConfig,
)
```

And add these names to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-panini && python -m pytest tests/test_mcp/test_models.py -v`
Expected: All 13 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/anchor/mcp/models.py src/anchor/mcp/__init__.py tests/test_mcp/test_models.py
git commit -m "feat(mcp): add MCPServerConfig, MCPResource, MCPPrompt models"
```

---

### Task 4: Create MCP protocols

**Files:**
- Create: `src/anchor/mcp/protocols.py`
- Create: `tests/test_mcp/test_protocols.py`
- Modify: `src/anchor/mcp/__init__.py`

- [ ] **Step 1: Write failing tests for protocols**

Create `tests/test_mcp/test_protocols.py`:

```python
"""Tests for MCP protocol conformance."""

from __future__ import annotations

from typing import Any, Self
from unittest.mock import AsyncMock

from anchor.agent.models import AgentTool
from anchor.llm.models import ToolSchema
from anchor.mcp.models import MCPPrompt, MCPResource
from anchor.mcp.protocols import MCPClient, MCPServer


class _FakeMCPClient:
    """Minimal class that structurally conforms to MCPClient."""

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def list_tools(self) -> list[ToolSchema]:
        return []
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        return ""
    async def list_resources(self) -> list[MCPResource]:
        return []
    async def read_resource(self, uri: str) -> str:
        return ""
    async def list_prompts(self) -> list[MCPPrompt]:
        return []
    async def get_prompt(self, name: str, arguments: dict[str, Any]) -> str:
        return ""
    async def __aenter__(self) -> Self:
        return self
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None: ...


class _FakeMCPServer:
    """Minimal class that structurally conforms to MCPServer."""

    def expose_tool(self, tool: AgentTool) -> None: ...
    def expose_tools(self, tools: list[AgentTool]) -> None: ...
    def expose_resource(self, uri: str, handler: Any) -> None: ...
    def expose_prompt(self, name: str, handler: Any) -> None: ...
    async def run(self, transport: str = "stdio") -> None: ...


class TestMCPClientProtocol:
    def test_fake_client_satisfies_protocol(self) -> None:
        client = _FakeMCPClient()
        assert isinstance(client, MCPClient)

    def test_object_does_not_satisfy_protocol(self) -> None:
        assert not isinstance(object(), MCPClient)


class TestMCPServerProtocol:
    def test_fake_server_satisfies_protocol(self) -> None:
        server = _FakeMCPServer()
        assert isinstance(server, MCPServer)

    def test_object_does_not_satisfy_protocol(self) -> None:
        assert not isinstance(object(), MCPServer)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-panini && python -m pytest tests/test_mcp/test_protocols.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'anchor.mcp.protocols'`

- [ ] **Step 3: Implement protocols**

Create `src/anchor/mcp/protocols.py`:

```python
"""MCP protocol definitions (PEP 544).

Any object with matching methods can be used — no inheritance required.
Follows anchor's established pattern from protocols/retriever.py.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, Self, runtime_checkable

from anchor.agent.models import AgentTool
from anchor.llm.models import ToolSchema
from anchor.mcp.models import MCPPrompt, MCPResource


@runtime_checkable
class MCPClient(Protocol):
    """Protocol for connecting to and consuming external MCP servers."""

    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...

    async def list_tools(self) -> list[ToolSchema]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str: ...

    async def list_resources(self) -> list[MCPResource]: ...

    async def read_resource(self, uri: str) -> str: ...

    async def list_prompts(self) -> list[MCPPrompt]: ...

    async def get_prompt(self, name: str, arguments: dict[str, Any]) -> str: ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None: ...


@runtime_checkable
class MCPServer(Protocol):
    """Protocol for exposing anchor capabilities as an MCP server."""

    def expose_tool(self, tool: AgentTool) -> None: ...

    def expose_tools(self, tools: list[AgentTool]) -> None: ...

    def expose_resource(
        self, uri: str, handler: Callable[..., str] | Callable[..., Any],
    ) -> None: ...

    def expose_prompt(
        self, name: str, handler: Callable[..., str] | Callable[..., Any],
    ) -> None: ...

    async def run(self, transport: str = "stdio") -> None: ...
```

- [ ] **Step 4: Update `__init__.py` with protocol exports**

Add `MCPClient` and `MCPServer` to `src/anchor/mcp/__init__.py` imports and `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-panini && python -m pytest tests/test_mcp/test_protocols.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/anchor/mcp/protocols.py src/anchor/mcp/__init__.py tests/test_mcp/test_protocols.py
git commit -m "feat(mcp): add MCPClient and MCPServer PEP 544 protocols"
```

---

## Chunk 2: Tool Adapters

### Task 5: Create tool conversion utilities

**Files:**
- Create: `src/anchor/mcp/tools.py`
- Create: `tests/test_mcp/test_tools.py`
- Modify: `src/anchor/mcp/__init__.py`

- [ ] **Step 1: Write failing tests for tool adapters**

Create `tests/test_mcp/test_tools.py`:

```python
"""Tests for MCP tool <-> AgentTool conversion."""

from __future__ import annotations

import pytest

from anchor.agent.models import AgentTool
from anchor.llm.models import ToolSchema
from anchor.mcp.errors import MCPError
from anchor.mcp.models import MCPServerConfig
from anchor.mcp.tools import mcp_tool_to_agent_tool, parse_server_string


class TestMCPToolToAgentTool:
    """Convert ToolSchema + async caller into AgentTool."""

    def test_basic_conversion(self) -> None:
        schema = ToolSchema(
            name="read_file",
            description="Read a file",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )

        async def caller(name: str, args: dict) -> str:
            return "content"

        tool = mcp_tool_to_agent_tool(
            schema=schema,
            async_caller=caller,
            server_name="fs",
            prefix=True,
        )

        assert isinstance(tool, AgentTool)
        assert tool.name == "fs_read_file"
        assert tool.description == "Read a file"
        assert "path" in tool.input_schema.get("properties", {})

    def test_no_prefix(self) -> None:
        schema = ToolSchema(
            name="read_file",
            description="Read a file",
            input_schema={"type": "object", "properties": {}},
        )

        async def caller(name: str, args: dict) -> str:
            return ""

        tool = mcp_tool_to_agent_tool(
            schema=schema,
            async_caller=caller,
            server_name="fs",
            prefix=False,
        )
        assert tool.name == "read_file"

    def test_sync_fn_raises_mcp_error(self) -> None:
        schema = ToolSchema(
            name="test_tool",
            description="test",
            input_schema={"type": "object", "properties": {}},
        )

        async def caller(name: str, args: dict) -> str:
            return ""

        tool = mcp_tool_to_agent_tool(
            schema=schema,
            async_caller=caller,
            server_name="s",
            prefix=False,
        )
        with pytest.raises(MCPError, match="achat"):
            tool.fn()

    def test_async_caller_and_original_name_attached(self) -> None:
        schema = ToolSchema(
            name="read_file",
            description="Read a file",
            input_schema={"type": "object", "properties": {}},
        )

        async def caller(name: str, args: dict) -> str:
            return "content"

        tool = mcp_tool_to_agent_tool(
            schema=schema,
            async_caller=caller,
            server_name="fs",
            prefix=True,
        )
        # These attributes are used by Agent._aexecute_tool()
        assert tool._mcp_async_caller is caller
        assert tool._mcp_original_name == "read_file"

    def test_empty_description(self) -> None:
        schema = ToolSchema(
            name="noop",
            description="",
            input_schema={"type": "object", "properties": {}},
        )

        async def caller(name: str, args: dict) -> str:
            return ""

        tool = mcp_tool_to_agent_tool(
            schema=schema,
            async_caller=caller,
            server_name="s",
            prefix=False,
        )
        assert tool.description == ""


class TestParseServerString:
    """Parse convenience strings into MCPServerConfig."""

    def test_http_url(self) -> None:
        cfg = parse_server_string("http://localhost:8080/mcp")
        assert cfg.url == "http://localhost:8080/mcp"
        assert cfg.command is None
        assert cfg.name == "localhost"

    def test_https_url(self) -> None:
        cfg = parse_server_string("https://api.example.com/mcp")
        assert cfg.url == "https://api.example.com/mcp"
        assert cfg.name == "api.example.com"

    def test_command_string(self) -> None:
        cfg = parse_server_string("npx -y @mcp/server-filesystem /tmp")
        assert cfg.command == "npx"
        assert cfg.args == ["-y", "@mcp/server-filesystem", "/tmp"]
        assert cfg.url is None
        assert cfg.name == "npx"

    def test_single_command(self) -> None:
        cfg = parse_server_string("python")
        assert cfg.command == "python"
        assert cfg.args == []
        assert cfg.name == "python"

    def test_defaults_applied(self) -> None:
        cfg = parse_server_string("http://localhost:8080/mcp")
        assert cfg.cache_tools is True
        assert cfg.prefix_tools is True
        assert cfg.timeout == 30.0
        assert cfg.env is None
        assert cfg.headers is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-panini && python -m pytest tests/test_mcp/test_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'anchor.mcp.tools'`

- [ ] **Step 3: Implement tool adapters**

Create `src/anchor/mcp/tools.py`:

```python
"""Adapters between MCP tools and anchor AgentTool.

Provides conversion utilities and config string parsing.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any
from urllib.parse import urlparse

from anchor.agent.models import AgentTool
from anchor.llm.models import ToolSchema
from anchor.mcp.errors import MCPError
from anchor.mcp.models import MCPServerConfig


def mcp_tool_to_agent_tool(
    *,
    schema: ToolSchema,
    async_caller: Callable[[str, dict[str, Any]], Coroutine[Any, Any, str]],
    server_name: str,
    prefix: bool,
) -> AgentTool:
    """Convert an MCP ToolSchema into an anchor AgentTool.

    The returned AgentTool's ``fn`` raises MCPError if called
    synchronously — MCP tools require ``achat()`` (async path).
    The ``_async_caller`` attribute is used by ``Agent._aexecute_tool()``.

    Parameters
    ----------
    schema:
        The tool schema from the MCP server.
    async_caller:
        Async callable: (tool_name, arguments) -> result string.
    server_name:
        Name of the originating MCP server.
    prefix:
        If True, prefix tool name with ``server_name_``.
    """
    tool_name = f"{server_name}_{schema.name}" if prefix else schema.name

    # The actual MCP tool name (without prefix) for the async caller
    original_name = schema.name

    def _sync_sentinel(**_kwargs: Any) -> str:
        msg = (
            f"MCP tool '{tool_name}' requires async execution. "
            f"Use agent.achat() instead of agent.chat()."
        )
        raise MCPError(msg)

    tool = AgentTool(
        name=tool_name,
        description=schema.description,
        input_schema=schema.input_schema,
        fn=_sync_sentinel,
    )

    # Attach async caller for _aexecute_tool() to use
    object.__setattr__(tool, "_mcp_async_caller", async_caller)
    object.__setattr__(tool, "_mcp_original_name", original_name)

    return tool


def parse_server_string(server: str) -> MCPServerConfig:
    """Parse a convenience string into MCPServerConfig.

    - Strings starting with http:// or https:// → URL config
    - Other strings → command + args (split by whitespace)
    """
    stripped = server.strip()

    if stripped.startswith(("http://", "https://")):
        parsed = urlparse(stripped)
        name = parsed.hostname or "mcp-server"
        return MCPServerConfig(name=name, url=stripped)

    parts = stripped.split()
    command = parts[0]
    args = parts[1:] if len(parts) > 1 else []
    return MCPServerConfig(name=command, command=command, args=args)
```

- [ ] **Step 4: Update `__init__.py` exports**

Add `mcp_tool_to_agent_tool` and `parse_server_string` to imports and `__all__` in `src/anchor/mcp/__init__.py`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-panini && python -m pytest tests/test_mcp/test_tools.py -v`
Expected: All 10 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/anchor/mcp/tools.py src/anchor/mcp/__init__.py tests/test_mcp/test_tools.py
git commit -m "feat(mcp): add tool adapters and config string parser"
```

---

## Chunk 3: Client Bridge

### Task 6: Implement FastMCPClientBridge

**Files:**
- Create: `src/anchor/mcp/client.py`
- Create: `tests/test_mcp/test_client.py`
- Modify: `src/anchor/mcp/__init__.py`

- [ ] **Step 1: Write failing tests for client bridge**

Create `tests/test_mcp/test_client.py`:

```python
"""Tests for FastMCPClientBridge."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anchor.llm.models import ToolSchema
from anchor.mcp.client import FastMCPClientBridge
from anchor.mcp.errors import MCPConnectionError, MCPToolError
from anchor.mcp.models import MCPServerConfig
from anchor.mcp.protocols import MCPClient


@pytest.fixture
def stdio_config() -> MCPServerConfig:
    return MCPServerConfig(name="test", command="echo", args=["hello"])


@pytest.fixture
def http_config() -> MCPServerConfig:
    return MCPServerConfig(name="api", url="http://localhost:8080/mcp")


class TestFastMCPClientBridge:
    async def test_init(self, stdio_config: MCPServerConfig) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        assert bridge._config is stdio_config
        assert bridge._fastmcp_client is None

    async def test_connect_creates_client_for_stdio(
        self, stdio_config: MCPServerConfig,
    ) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            await bridge.connect()
            assert bridge._fastmcp_client is not None

    async def test_connect_creates_client_for_http(
        self, http_config: MCPServerConfig,
    ) -> None:
        bridge = FastMCPClientBridge(http_config)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            await bridge.connect()
            assert bridge._fastmcp_client is not None

    async def test_disconnect(self, stdio_config: MCPServerConfig) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            await bridge.connect()
            await bridge.disconnect()
            assert bridge._fastmcp_client is None

    async def test_list_tools_returns_tool_schemas(
        self, stdio_config: MCPServerConfig,
    ) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        # FastMCP Client returns mcp.types.Tool objects
        mock_mcp_tool = MagicMock()
        mock_mcp_tool.name = "read_file"
        mock_mcp_tool.description = "Read a file"
        mock_mcp_tool.inputSchema = {"type": "object", "properties": {"path": {"type": "string"}}}
        mock_client.list_tools.return_value = [mock_mcp_tool]

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            await bridge.connect()
            tools = await bridge.list_tools()

        assert len(tools) == 1
        assert tools[0].name == "read_file"
        assert isinstance(tools[0], ToolSchema)

    async def test_list_tools_caching(self, stdio_config: MCPServerConfig) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.list_tools.return_value = []

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            await bridge.connect()
            await bridge.list_tools()
            await bridge.list_tools()

        # Called only once due to caching
        assert mock_client.list_tools.call_count == 1

    async def test_call_tool(self, stdio_config: MCPServerConfig) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_result = MagicMock()
        mock_result.content = [MagicMock(text="file content")]
        mock_client.call_tool.return_value = mock_result

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            await bridge.connect()
            result = await bridge.call_tool("read_file", {"path": "/tmp/x"})

        assert result == "file content"

    async def test_as_agent_tools(self, stdio_config: MCPServerConfig) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_mcp_tool = MagicMock()
        mock_mcp_tool.name = "greet"
        mock_mcp_tool.description = "Greet someone"
        mock_mcp_tool.inputSchema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        mock_client.list_tools.return_value = [mock_mcp_tool]

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            await bridge.connect()
            tools = await bridge.as_agent_tools()

        assert len(tools) == 1
        assert tools[0].name == "test_greet"  # prefixed with server name

    async def test_satisfies_protocol(self, stdio_config: MCPServerConfig) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        assert isinstance(bridge, MCPClient)

    async def test_connect_failure_raises_connection_error(
        self, stdio_config: MCPServerConfig,
    ) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        with patch("anchor.mcp.client.Client", side_effect=OSError("refused")):
            with pytest.raises(MCPConnectionError, match="refused"):
                await bridge.connect()

    async def test_call_tool_failure_raises_tool_error(
        self, stdio_config: MCPServerConfig,
    ) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.call_tool.side_effect = RuntimeError("tool broke")

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            await bridge.connect()
            with pytest.raises(MCPToolError, match="tool broke"):
                await bridge.call_tool("bad_tool", {})

    async def test_list_tools_no_caching(self, stdio_config: MCPServerConfig) -> None:
        no_cache_config = MCPServerConfig(
            name="test", command="echo", args=["hello"], cache_tools=False,
        )
        bridge = FastMCPClientBridge(no_cache_config)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.list_tools.return_value = []

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            await bridge.connect()
            await bridge.list_tools()
            await bridge.list_tools()

        # Called twice since caching is off
        assert mock_client.list_tools.call_count == 2

    async def test_list_resources(self, stdio_config: MCPServerConfig) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_resource = MagicMock()
        mock_resource.uri = "file:///tmp/test"
        mock_resource.name = "test"
        mock_resource.description = "A test resource"
        mock_resource.mimeType = "text/plain"
        mock_client.list_resources.return_value = [mock_resource]

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            await bridge.connect()
            resources = await bridge.list_resources()

        assert len(resources) == 1
        assert resources[0].uri == "file:///tmp/test"

    async def test_list_prompts(self, stdio_config: MCPServerConfig) -> None:
        bridge = FastMCPClientBridge(stdio_config)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_prompt = MagicMock()
        mock_prompt.name = "analyze"
        mock_prompt.description = "Analyze code"
        mock_prompt.arguments = []
        mock_client.list_prompts.return_value = [mock_prompt]

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            await bridge.connect()
            prompts = await bridge.list_prompts()

        assert len(prompts) == 1
        assert prompts[0].name == "analyze"

    async def test_context_manager(self, stdio_config: MCPServerConfig) -> None:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.list_tools.return_value = []

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            async with FastMCPClientBridge(stdio_config) as bridge:
                assert bridge._fastmcp_client is not None
            assert bridge._fastmcp_client is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-panini && python -m pytest tests/test_mcp/test_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'anchor.mcp.client'`

- [ ] **Step 3: Implement FastMCPClientBridge**

Create `src/anchor/mcp/client.py`:

```python
"""FastMCP-backed client bridge for consuming external MCP servers.

Uses FastMCP's Client class with auto-inferred transport.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Self

from anchor.agent.models import AgentTool
from anchor.llm.models import ToolSchema
from anchor.mcp.errors import MCPConnectionError, MCPToolError
from anchor.mcp.models import (
    MCPPrompt,
    MCPPromptArgument,
    MCPResource,
    MCPServerConfig,
)
from anchor.mcp.protocols import MCPClient
from anchor.mcp.tools import mcp_tool_to_agent_tool

try:
    from fastmcp.client import Client
except ImportError as exc:
    _import_error = exc

    class Client:  # type: ignore[no-redef]
        """Placeholder when fastmcp is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "MCP bridge requires the 'fastmcp' package. "
                "Install with: pip install astro-anchor[mcp]"
            ) from _import_error

logger = logging.getLogger(__name__)


class FastMCPClientBridge:
    """Connects to external MCP servers using FastMCP Client.

    Implements the MCPClient protocol. Adapts MCP tools to
    anchor's AgentTool format for seamless Agent integration.
    """

    __slots__ = ("_config", "_fastmcp_client", "_cached_tools")

    def __init__(self, config: MCPServerConfig) -> None:
        self._config = config
        self._fastmcp_client: Client | None = None
        self._cached_tools: list[ToolSchema] | None = None

    async def connect(self) -> None:
        """Connect using auto-inferred transport."""
        try:
            if self._config.url is not None:
                client = Client(self._config.url)
            elif self._config.command is not None:
                cmd = self._config.command
                if self._config.args:
                    cmd = f"{cmd} {' '.join(self._config.args)}"
                client = Client(cmd)
            else:
                msg = "No command or URL configured"
                raise MCPConnectionError(
                    msg,
                    server_name=self._config.name,
                    transport="unknown",
                )

            self._fastmcp_client = await client.__aenter__()
        except MCPConnectionError:
            raise
        except Exception as exc:
            msg = f"Failed to connect to MCP server '{self._config.name}': {exc}"
            raise MCPConnectionError(
                msg,
                server_name=self._config.name,
                transport="http" if self._config.url else "stdio",
            ) from exc

    async def disconnect(self) -> None:
        """Close the FastMCP Client connection."""
        if self._fastmcp_client is not None:
            try:
                await self._fastmcp_client.__aexit__(None, None, None)
            except Exception:
                logger.warning(
                    "Error disconnecting from MCP server '%s'",
                    self._config.name,
                    exc_info=True,
                )
            finally:
                self._fastmcp_client = None
                self._cached_tools = None

    async def list_tools(self) -> list[ToolSchema]:
        """List tools, with optional caching."""
        if self._config.cache_tools and self._cached_tools is not None:
            return self._cached_tools

        assert self._fastmcp_client is not None  # noqa: S101
        mcp_tools = await self._fastmcp_client.list_tools()
        schemas = [
            ToolSchema(
                name=t.name,
                description=t.description or "",
                input_schema=t.inputSchema,
            )
            for t in mcp_tools
        ]

        if self._config.cache_tools:
            self._cached_tools = schemas

        return schemas

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute tool via FastMCP Client, return result as string."""
        assert self._fastmcp_client is not None  # noqa: S101
        try:
            result = await self._fastmcp_client.call_tool(name, arguments)
            # FastMCP returns a list of content blocks
            if hasattr(result, "content") and isinstance(result.content, list):
                texts = [
                    block.text for block in result.content
                    if hasattr(block, "text") and block.text
                ]
                return " ".join(texts) if texts else str(result)
            return str(result)
        except Exception as exc:
            raise MCPToolError(
                f"MCP tool '{name}' failed: {exc}",
                tool_name=name,
                server_name=self._config.name,
                cause=exc,
            ) from exc

    async def list_resources(self) -> list[MCPResource]:
        """List resources from the MCP server."""
        assert self._fastmcp_client is not None  # noqa: S101
        mcp_resources = await self._fastmcp_client.list_resources()
        return [
            MCPResource(
                uri=str(r.uri),
                name=r.name,
                description=r.description,
                mime_type=r.mimeType if hasattr(r, "mimeType") else None,
            )
            for r in mcp_resources
        ]

    async def read_resource(self, uri: str) -> str:
        """Read a resource by URI."""
        assert self._fastmcp_client is not None  # noqa: S101
        result = await self._fastmcp_client.read_resource(uri)
        if isinstance(result, list) and result:
            first = result[0]
            if hasattr(first, "content"):
                return str(first.content)
        return str(result)

    async def list_prompts(self) -> list[MCPPrompt]:
        """List prompt templates from the MCP server."""
        assert self._fastmcp_client is not None  # noqa: S101
        mcp_prompts = await self._fastmcp_client.list_prompts()
        return [
            MCPPrompt(
                name=p.name,
                description=p.description,
                arguments=[
                    MCPPromptArgument(
                        name=a.name,
                        description=a.description,
                        required=a.required or False,
                    )
                    for a in (p.arguments or [])
                ],
            )
            for p in mcp_prompts
        ]

    async def get_prompt(self, name: str, arguments: dict[str, Any]) -> str:
        """Get a rendered prompt."""
        assert self._fastmcp_client is not None  # noqa: S101
        result = await self._fastmcp_client.get_prompt(name, arguments)
        if hasattr(result, "messages") and result.messages:
            texts = []
            for msg in result.messages:
                if hasattr(msg.content, "text"):
                    texts.append(msg.content.text)
            return " ".join(texts) if texts else str(result)
        return str(result)

    async def as_agent_tools(self) -> list[AgentTool]:
        """Convert all MCP tools to AgentTool instances."""
        schemas = await self.list_tools()
        return [
            mcp_tool_to_agent_tool(
                schema=schema,
                async_caller=self.call_tool,
                server_name=self._config.name,
                prefix=self._config.prefix_tools,
            )
            for schema in schemas
        ]

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.disconnect()
```

- [ ] **Step 4: Update `__init__.py` exports**

Add `FastMCPClientBridge` to `src/anchor/mcp/__init__.py` imports and `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-panini && python -m pytest tests/test_mcp/test_client.py -v`
Expected: All 15 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/anchor/mcp/client.py src/anchor/mcp/__init__.py tests/test_mcp/test_client.py
git commit -m "feat(mcp): add FastMCPClientBridge for consuming external MCP servers"
```

---

### Task 7: Implement MCPClientPool

**Files:**
- Create: `tests/test_mcp/test_pool.py`
- Modify: `src/anchor/mcp/client.py`
- Modify: `src/anchor/mcp/__init__.py`

- [ ] **Step 1: Write failing tests for pool**

Create `tests/test_mcp/test_pool.py`:

```python
"""Tests for MCPClientPool."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anchor.mcp.client import FastMCPClientBridge, MCPClientPool
from anchor.mcp.models import MCPServerConfig


@pytest.fixture
def two_configs() -> list[MCPServerConfig]:
    return [
        MCPServerConfig(name="fs", command="echo", args=["fs"]),
        MCPServerConfig(name="db", command="echo", args=["db"]),
    ]


class TestMCPClientPool:
    async def test_connect_all(self, two_configs: list[MCPServerConfig]) -> None:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.list_tools.return_value = []

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            pool = MCPClientPool(two_configs)
            await pool.connect_all()
            assert len(pool._clients) == 2

    async def test_disconnect_all(self, two_configs: list[MCPServerConfig]) -> None:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.list_tools.return_value = []

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            pool = MCPClientPool(two_configs)
            await pool.connect_all()
            await pool.disconnect_all()
            assert len(pool._clients) == 0

    async def test_all_agent_tools_collects_from_all(
        self, two_configs: list[MCPServerConfig],
    ) -> None:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        mock_tool = MagicMock()
        mock_tool.name = "tool1"
        mock_tool.description = "A tool"
        mock_tool.inputSchema = {"type": "object", "properties": {}}
        mock_client.list_tools.return_value = [mock_tool]

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            pool = MCPClientPool(two_configs)
            await pool.connect_all()
            tools = await pool.all_agent_tools()

        # 1 tool per server × 2 servers = 2 tools
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "fs_tool1" in names
        assert "db_tool1" in names

    async def test_context_manager(self, two_configs: list[MCPServerConfig]) -> None:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.list_tools.return_value = []

        with patch("anchor.mcp.client.Client", return_value=mock_client):
            async with MCPClientPool(two_configs) as pool:
                assert len(pool._clients) == 2
            assert len(pool._clients) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-panini && python -m pytest tests/test_mcp/test_pool.py -v`
Expected: FAIL — `ImportError: cannot import name 'MCPClientPool'`

- [ ] **Step 3: Add MCPClientPool to client.py**

Append to `src/anchor/mcp/client.py`:

```python
class MCPClientPool:
    """Manages connections to multiple MCP servers.

    Convenience class for connecting to several servers at once
    and collecting all their tools as AgentTool instances.
    """

    __slots__ = ("_configs", "_clients")

    def __init__(self, configs: list[MCPServerConfig]) -> None:
        self._configs = configs
        self._clients: list[MCPClient] = []

    async def connect_all(self) -> None:
        """Connect to all configured servers concurrently."""
        bridges = [FastMCPClientBridge(cfg) for cfg in self._configs]
        await asyncio.gather(*(b.connect() for b in bridges))
        self._clients = bridges

    async def disconnect_all(self) -> None:
        """Disconnect all servers."""
        await asyncio.gather(
            *(c.disconnect() for c in self._clients),
            return_exceptions=True,
        )
        self._clients = []

    async def all_agent_tools(self) -> list[AgentTool]:
        """Collect AgentTool instances from all connected servers."""
        tool_lists = await asyncio.gather(
            *(c.as_agent_tools() for c in self._clients),
        )
        return [tool for tools in tool_lists for tool in tools]

    async def __aenter__(self) -> Self:
        await self.connect_all()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.disconnect_all()
```

- [ ] **Step 4: Update `__init__.py` exports**

Add `MCPClientPool` to imports and `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-panini && python -m pytest tests/test_mcp/test_pool.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/anchor/mcp/client.py src/anchor/mcp/__init__.py tests/test_mcp/test_pool.py
git commit -m "feat(mcp): add MCPClientPool for multi-server management"
```

---

## Chunk 4: Server Bridge

### Task 8: Implement FastMCPServerBridge

**Files:**
- Create: `src/anchor/mcp/server.py`
- Create: `tests/test_mcp/test_server.py`
- Modify: `src/anchor/mcp/__init__.py`

- [ ] **Step 1: Write failing tests for server bridge**

Create `tests/test_mcp/test_server.py`:

```python
"""Tests for FastMCPServerBridge."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anchor.agent.models import AgentTool
from anchor.agent.tool_decorator import tool
from anchor.mcp.protocols import MCPServer
from anchor.mcp.server import FastMCPServerBridge


@tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello {name}"


@tool
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


class TestFastMCPServerBridge:
    def test_satisfies_protocol(self) -> None:
        server = FastMCPServerBridge("test")
        assert isinstance(server, MCPServer)

    def test_expose_tool(self) -> None:
        server = FastMCPServerBridge("test")
        server.expose_tool(greet)
        # Tool should be registered with internal FastMCP server
        assert len(server._registered_tools) == 1

    def test_expose_tools(self) -> None:
        server = FastMCPServerBridge("test")
        server.expose_tools([greet, add])
        assert len(server._registered_tools) == 2

    def test_expose_resource(self) -> None:
        server = FastMCPServerBridge("test")

        def handler() -> str:
            return "resource data"

        server.expose_resource("data://config", handler)
        assert len(server._registered_resources) == 1

    def test_expose_prompt(self) -> None:
        server = FastMCPServerBridge("test")

        def handler(topic: str) -> str:
            return f"Analyze {topic}"

        server.expose_prompt("analyze", handler)
        assert len(server._registered_prompts) == 1

    def test_from_agent(self) -> None:
        # Test that from_agent creates a server with the agent's tools,
        # resources (context://pipeline, context://memory), and prompt (chat)
        from anchor.agent.agent import Agent

        agent = Agent(model="claude-haiku-4-5-20251001").with_tools([greet, add])
        server = FastMCPServerBridge.from_agent(agent)
        assert isinstance(server, FastMCPServerBridge)
        # Agent has 2 tools
        assert len(server._registered_tools) >= 2
        # Should expose context://pipeline and context://memory resources
        assert "context://pipeline" in server._registered_resources
        assert "context://memory" in server._registered_resources
        # Should expose chat prompt
        assert "chat" in server._registered_prompts

    def test_from_pipeline(self) -> None:
        # Test that from_pipeline creates a server with query tool and context://result resource
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.formatted_output = "pipeline output"
        mock_pipeline.build.return_value = mock_result

        server = FastMCPServerBridge.from_pipeline(mock_pipeline)
        assert isinstance(server, FastMCPServerBridge)
        assert "query" in server._registered_tools
        assert "context://result" in server._registered_resources

    @pytest.mark.asyncio
    async def test_run_unknown_transport_raises(self) -> None:
        server = FastMCPServerBridge("test")
        with pytest.raises(ValueError, match="Unknown transport"):
            await server.run(transport="invalid")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-panini && python -m pytest tests/test_mcp/test_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'anchor.mcp.server'`

- [ ] **Step 3a: Create server.py with core class and expose methods**

Create `src/anchor/mcp/server.py` with the class skeleton, `__init__`, `expose_tool`, `expose_tools`, `expose_resource`, `expose_prompt`, and `run`:

Create `src/anchor/mcp/server.py`:

```python
"""FastMCP-backed server bridge for exposing anchor as MCP server."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from anchor.agent.models import AgentTool

try:
    from fastmcp import FastMCP
except ImportError as exc:
    _import_error = exc

    class FastMCP:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "MCP bridge requires the 'fastmcp' package. "
                "Install with: pip install astro-anchor[mcp]"
            ) from _import_error

if TYPE_CHECKING:
    from anchor.agent.agent import Agent
    from anchor.pipeline.pipeline import ContextPipeline

logger = logging.getLogger(__name__)


class FastMCPServerBridge:
    """Exposes anchor capabilities as an MCP server via FastMCP."""

    __slots__ = (
        "_mcp",
        "_registered_tools",
        "_registered_resources",
        "_registered_prompts",
    )

    def __init__(self, name: str = "anchor") -> None:
        self._mcp = FastMCP(name)
        self._registered_tools: list[str] = []
        self._registered_resources: list[str] = []
        self._registered_prompts: list[str] = []

    def expose_tool(self, tool: AgentTool) -> None:
        """Register an AgentTool as an MCP tool."""
        fn = tool.fn

        # Register with FastMCP using the tool decorator
        self._mcp.tool(name=tool.name, description=tool.description)(fn)
        self._registered_tools.append(tool.name)

    def expose_tools(self, tools: list[AgentTool]) -> None:
        """Register multiple tools."""
        for t in tools:
            self.expose_tool(t)

    def expose_resource(
        self,
        uri: str,
        handler: Callable[..., str] | Callable[..., Any],
    ) -> None:
        """Register a resource at the given URI pattern."""
        self._mcp.resource(uri)(handler)
        self._registered_resources.append(uri)

    def expose_prompt(
        self,
        name: str,
        handler: Callable[..., str] | Callable[..., Any],
    ) -> None:
        """Register a prompt template."""
        self._mcp.prompt(name=name)(handler)
        self._registered_prompts.append(name)

    @classmethod
    def from_agent(cls, agent: Agent) -> FastMCPServerBridge:
        """Create server from an Agent, exposing all its tools.

        Also exposes:
        - resource ``context://pipeline`` -> pipeline query
        - resource ``context://memory`` -> memory state
        - prompt ``chat`` -> send a message through the agent
        """
        server = cls(name="anchor-agent")
        server.expose_tools(agent._tools)

        # Expose pipeline resource
        def _pipeline_resource() -> str:
            """Return the agent's pipeline configuration."""
            return str(getattr(agent, "_pipeline", "No pipeline configured"))

        server.expose_resource("context://pipeline", _pipeline_resource)

        # Expose memory resource
        def _memory_resource() -> str:
            """Return the agent's memory state."""
            return str(getattr(agent, "_memory", "No memory configured"))

        server.expose_resource("context://memory", _memory_resource)

        # Expose chat prompt
        def _chat_prompt(message: str) -> str:
            """Send a message through the agent."""
            return message

        server.expose_prompt("chat", _chat_prompt)

        return server

    @classmethod
    def from_pipeline(cls, pipeline: ContextPipeline) -> FastMCPServerBridge:
        """Create server exposing pipeline as retrieval tools.

        Exposes:
        - tool ``query`` -> run pipeline.build() and return results
        - resource ``context://result`` -> last pipeline result
        """
        server = cls(name="anchor-pipeline")
        _last_result: list[str] = []

        def query(text: str) -> str:
            """Run the context pipeline and return formatted results."""
            result = pipeline.build(text)
            output = str(result.formatted_output)
            _last_result.clear()
            _last_result.append(output)
            return output

        server._mcp.tool(name="query", description="Query the context pipeline")(query)
        server._registered_tools.append("query")

        def _result_resource() -> str:
            """Return the last pipeline result."""
            return _last_result[0] if _last_result else "No results yet"

        server.expose_resource("context://result", _result_resource)

        return server

    async def run(self, transport: str = "stdio") -> None:
        """Start the MCP server."""
        if transport == "stdio":
            await self._mcp.run_async(transport="stdio")
        elif transport == "http":
            await self._mcp.run_async(transport="streamable-http")
        elif transport == "sse":
            await self._mcp.run_async(transport="sse")
        else:
            msg = f"Unknown transport: {transport!r}. Use 'stdio', 'http', or 'sse'."
            raise ValueError(msg)
```

- [ ] **Step 3b: Verify core expose methods pass**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-panini && python -m pytest tests/test_mcp/test_server.py::TestFastMCPServerBridge::test_satisfies_protocol tests/test_mcp/test_server.py::TestFastMCPServerBridge::test_expose_tool tests/test_mcp/test_server.py::TestFastMCPServerBridge::test_expose_tools tests/test_mcp/test_server.py::TestFastMCPServerBridge::test_expose_resource tests/test_mcp/test_server.py::TestFastMCPServerBridge::test_expose_prompt tests/test_mcp/test_server.py::TestFastMCPServerBridge::test_run_unknown_transport_raises -v`
Expected: These 6 tests PASS. Factory method tests (`from_agent`, `from_pipeline`) may still fail — that's fine, they depend on the full class.

- [ ] **Step 4: Update `__init__.py` exports**

Add `FastMCPServerBridge` to imports and `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-panini && python -m pytest tests/test_mcp/test_server.py -v`
Expected: All 8 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/anchor/mcp/server.py src/anchor/mcp/__init__.py tests/test_mcp/test_server.py
git commit -m "feat(mcp): add FastMCPServerBridge for exposing anchor as MCP server"
```

---

## Chunk 5: Agent Integration

### Task 9: Add with_mcp_servers() and async tool execution to Agent

**Files:**
- Create: `tests/test_mcp/test_agent_integration.py`
- Modify: `src/anchor/agent/agent.py`

- [ ] **Step 1: Write failing tests for Agent MCP integration**

Create `tests/test_mcp/test_agent_integration.py`:

```python
"""Tests for Agent MCP integration."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anchor.agent.agent import Agent
from anchor.mcp.models import MCPServerConfig


class TestAgentWithMCPServers:
    def test_with_mcp_servers_accepts_strings(self) -> None:
        agent = Agent(model="claude-haiku-4-5-20251001")
        result = agent.with_mcp_servers([
            "npx -y @mcp/server-filesystem /tmp",
            "http://localhost:8080/mcp",
        ])
        assert result is agent  # fluent API
        assert len(agent._mcp_configs) == 2

    def test_with_mcp_servers_accepts_configs(self) -> None:
        cfg = MCPServerConfig(name="test", command="echo")
        agent = Agent(model="claude-haiku-4-5-20251001").with_mcp_servers([cfg])
        assert len(agent._mcp_configs) == 1
        assert agent._mcp_configs[0] is cfg

    def test_with_mcp_servers_accepts_mixed(self) -> None:
        cfg = MCPServerConfig(name="test", command="echo")
        agent = Agent(model="claude-haiku-4-5-20251001").with_mcp_servers([
            cfg,
            "http://localhost:8080",
        ])
        assert len(agent._mcp_configs) == 2

    def test_chat_raises_when_mcp_configured(self) -> None:
        agent = Agent(model="claude-haiku-4-5-20251001").with_mcp_servers([
            "http://localhost:8080",
        ])
        with pytest.raises(TypeError, match="achat"):
            list(agent.chat("hello"))

    def test_with_mcp_servers_is_additive(self) -> None:
        agent = Agent(model="claude-haiku-4-5-20251001")
        agent.with_mcp_servers(["http://a.com"])
        agent.with_mcp_servers(["http://b.com"])
        assert len(agent._mcp_configs) == 2


class TestAgentAsyncMCPExecution:
    """Tests for _aexecute_tool() and _arun_tools() async paths."""

    @pytest.mark.asyncio
    async def test_aexecute_mcp_tool_uses_async_caller(self) -> None:
        """MCP tools dispatch through _mcp_async_caller."""
        from anchor.agent.models import AgentTool
        from anchor.mcp.errors import MCPError

        agent = Agent(model="claude-haiku-4-5-20251001")

        # Create a mock MCP AgentTool with _mcp_async_caller
        async_caller = AsyncMock(return_value="mcp result")

        def sentinel(**_kwargs: Any) -> str:
            raise MCPError("sync")

        tool = AgentTool(
            name="server_read",
            description="Read",
            input_schema={"type": "object", "properties": {}},
            fn=sentinel,
        )
        object.__setattr__(tool, "_mcp_async_caller", async_caller)
        object.__setattr__(tool, "_mcp_original_name", "read")

        agent._mcp_tools = [tool]
        result = await agent._aexecute_tool("server_read", {"path": "/tmp"})

        assert result == "mcp result"
        async_caller.assert_called_once_with("read", {"path": "/tmp"})

    @pytest.mark.asyncio
    async def test_aexecute_regular_tool_uses_sync_fn(self) -> None:
        """Regular (non-MCP) tools use sync fn in _aexecute_tool."""
        from anchor.agent.tool_decorator import tool

        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello {name}"

        agent = Agent(model="claude-haiku-4-5-20251001").with_tools([greet])
        result = await agent._aexecute_tool("greet", {"name": "World"})
        assert result == "Hello World"

    @pytest.mark.asyncio
    async def test_aexecute_mcp_tool_returns_error_on_failure(self) -> None:
        """MCP tool failure returns error string, doesn't raise."""
        from anchor.agent.models import AgentTool
        from anchor.mcp.errors import MCPError

        agent = Agent(model="claude-haiku-4-5-20251001")

        async_caller = AsyncMock(side_effect=RuntimeError("boom"))

        def sentinel(**_kwargs: Any) -> str:
            raise MCPError("sync")

        tool = AgentTool(
            name="broken",
            description="Broken tool",
            input_schema={"type": "object", "properties": {}},
            fn=sentinel,
        )
        object.__setattr__(tool, "_mcp_async_caller", async_caller)
        object.__setattr__(tool, "_mcp_original_name", "broken")

        agent._mcp_tools = [tool]
        result = await agent._aexecute_tool("broken", {})
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_aexecute_unknown_tool_returns_unknown(self) -> None:
        agent = Agent(model="claude-haiku-4-5-20251001")
        result = await agent._aexecute_tool("nonexistent", {})
        assert "Unknown tool" in result

    @pytest.mark.asyncio
    async def test_all_active_tools_includes_mcp(self) -> None:
        """_all_active_tools() includes MCP tools."""
        from anchor.agent.models import AgentTool

        agent = Agent(model="claude-haiku-4-5-20251001")
        mcp_tool = AgentTool(
            name="mcp_tool",
            description="MCP",
            input_schema={"type": "object", "properties": {}},
            fn=lambda: "",
        )
        agent._mcp_tools = [mcp_tool]
        all_tools = agent._all_active_tools()
        assert any(t.name == "mcp_tool" for t in all_tools)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-panini && python -m pytest tests/test_mcp/test_agent_integration.py -v`
Expected: FAIL — `AttributeError: 'Agent' object has no attribute '_mcp_configs'`

- [ ] **Step 3a: Add __slots__, __init__, with_mcp_servers(), and chat() guard**

Modify `src/anchor/agent/agent.py`:

1. Add to `__slots__`:
   - `"_mcp_configs"`
   - `"_mcp_pool"`
   - `"_mcp_tools"`

2. In `__init__()`, initialize:
   ```python
   self._mcp_configs: list[MCPServerConfig] = []
   self._mcp_pool: MCPClientPool | None = None
   self._mcp_tools: list[AgentTool] = []
   ```

3. Add `with_mcp_servers()` method:
   ```python
   def with_mcp_servers(
       self,
       servers: list[str | MCPServerConfig],
   ) -> Agent:
       """Connect to external MCP servers. Returns self for chaining."""
       from anchor.mcp.models import MCPServerConfig
       from anchor.mcp.tools import parse_server_string

       for server in servers:
           if isinstance(server, str):
               self._mcp_configs.append(parse_server_string(server))
           else:
               self._mcp_configs.append(server)
       return self
   ```

4. Add sync `chat()` guard at the top of `chat()`:
   ```python
   if self._mcp_configs:
       msg = (
           "MCP servers require async execution. "
           "Use agent.achat() instead of agent.chat()."
       )
       raise TypeError(msg)
   ```

- [ ] **Step 3b: Verify config tests pass**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-panini && python -m pytest tests/test_mcp/test_agent_integration.py::TestAgentWithMCPServers -v`
Expected: All 5 TestAgentWithMCPServers tests PASS

- [ ] **Step 3c: Add _aexecute_tool(), _arun_tools(), and _all_active_tools() extension**

Continue modifying `src/anchor/agent/agent.py`:

5. Add `_aexecute_tool()` async method:
   ```python
   async def _aexecute_tool(self, name: str, tool_input: dict[str, Any]) -> str:
       """Async tool execution — supports both regular and MCP tools."""
       for tool in self._all_active_tools():
           if tool.name == name:
               # Check if this is an MCP tool with async caller
               async_caller = getattr(tool, "_mcp_async_caller", None)
               if async_caller is not None:
                   original_name = getattr(tool, "_mcp_original_name", name)
                   try:
                       return await async_caller(original_name, tool_input)
                   except Exception:
                       logger.exception("MCP tool '%s' failed", name)
                       return f"Error: MCP tool '{name}' failed."

               # Regular tool — run sync
               valid, err = tool.validate_input(tool_input)
               if not valid:
                   return f"Error: invalid input for tool '{name}': {err}"
               try:
                   return tool.fn(**tool_input)
               except Exception:
                   logger.exception("Tool '%s' failed", name)
                   return f"Error: tool '{name}' failed."
       return f"Unknown tool: {name}"
   ```

6. Add `_arun_tools()` async method:
   ```python
   async def _arun_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
       """Execute tool calls asynchronously."""
       results: list[ToolResult] = []
       for tc in tool_calls:
           result_text = await self._aexecute_tool(tc.name, tc.arguments)
           self._record_tool_call(tc.name, tc.arguments, result_text)
           results.append(ToolResult(tool_call_id=tc.id, content=result_text))
       return results
   ```

7. In `_all_active_tools()`, include MCP tools:
   ```python
   tools.extend(self._mcp_tools)
   ```

- [ ] **Step 3d: Verify async execution tests pass**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-panini && python -m pytest tests/test_mcp/test_agent_integration.py::TestAgentAsyncMCPExecution -v`
Expected: All 5 TestAgentAsyncMCPExecution tests PASS

- [ ] **Step 3e: Add lazy MCP connection to achat()**

Continue modifying `src/anchor/agent/agent.py`:

8. In `achat()`, add lazy MCP connection before the tool loop:
   ```python
   # Lazy MCP connection
   if self._mcp_configs and not self._mcp_pool:
       from anchor.mcp.client import MCPClientPool
       self._mcp_pool = MCPClientPool(self._mcp_configs)
       await self._mcp_pool.connect_all()
       self._mcp_tools = await self._mcp_pool.all_agent_tools()
   ```

9. In `achat()`, replace `self._run_tools(tool_calls)` with `await self._arun_tools(tool_calls)`.

- [ ] **Step 4: Run all integration tests to verify they pass**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-panini && python -m pytest tests/test_mcp/test_agent_integration.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Run existing agent tests to check for regressions**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-panini && python -m pytest tests/test_agent/ -v`
Expected: All existing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add src/anchor/agent/agent.py tests/test_mcp/test_agent_integration.py
git commit -m "feat(mcp): integrate MCP servers into Agent with lazy loading"
```

---

## Chunk 6: Public API and Final Verification

### Task 10: Add MCP exports to anchor's public API

**Files:**
- Modify: `src/anchor/__init__.py`

- [ ] **Step 1: Add MCP imports and exports**

Add to `src/anchor/__init__.py` (within a try/except for optional dependency):

```python
# MCP Bridge (optional — requires pip install astro-anchor[mcp])
try:
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

    __all__ += [
        "FastMCPClientBridge",
        "FastMCPServerBridge",
        "MCPClient",
        "MCPClientPool",
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
except ImportError:
    pass  # fastmcp not installed — MCP bridge unavailable
```

Note: Use try/except because fastmcp is an optional dependency. Users who don't install `astro-anchor[mcp]` should not get ImportErrors when importing anchor.

- [ ] **Step 2: Verify imports work**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-panini && python -c "from anchor import MCPServerConfig, FastMCPClientBridge; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/anchor/__init__.py
git commit -m "feat(mcp): export MCP bridge types from anchor public API"
```

---

### Task 11: Run full test suite

**Files:** None (verification only)

- [ ] **Step 1: Run all MCP tests**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-panini && python -m pytest tests/test_mcp/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 2: Run full test suite for regressions**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-panini && python -m pytest tests/ -v --tb=short -x`
Expected: No regressions. All existing tests PASS.

- [ ] **Step 3: Run linter**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-panini && ruff check src/anchor/mcp/ tests/test_mcp/`
Expected: No errors

- [ ] **Step 4: Run type checker**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/distracted-panini && mypy src/anchor/mcp/`
Expected: No errors (or only expected issues with FastMCP type stubs)
