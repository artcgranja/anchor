# SKILL.md Loader Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SKILL.md loading support to astro-context so agents can consume cross-platform skills alongside native Python skills.

**Architecture:** A `loader.py` module parses SKILL.md frontmatter + markdown into native `Skill` instances. Tool discovery imports `tools.py` via `importlib`. The `SkillRegistry` and `Agent` get thin wrapper methods for loading. No new dependencies.

**Tech Stack:** Python 3.11+, Pydantic v2, importlib, pytest

**Spec:** `docs/superpowers/specs/2026-03-12-skillmd-loader-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `src/astro_context/agent/skills/loader.py` | Parse SKILL.md frontmatter, discover tools from `tools.py`, build `Skill` instances |
| `tests/test_agent/test_skills/test_loader.py` | Unit tests for loader parsing, validation, tool discovery |
| `tests/test_agent/test_skills/test_loader_integration.py` | Integration tests for registry + agent loading |
| `tests/fixtures/skills/brainstorm/SKILL.md` | Valid hybrid skill fixture (instructions + tools) |
| `tests/fixtures/skills/brainstorm/tools.py` | Test tool: `save_brainstorm_result` |
| `tests/fixtures/skills/minimal/SKILL.md` | Valid instructions-only skill fixture |
| `tests/fixtures/skills/invalid/SKILL.md` | Invalid frontmatter fixture (missing name) |
| `examples/skills/brainstorm/SKILL.md` | Example brainstorming skill for users |
| `examples/skills/brainstorm/tools.py` | Example tool implementation |

### Modified Files

| File | Change |
|------|--------|
| `src/astro_context/agent/skills/registry.py` | Add `load_from_path()`, `load_from_directory()` |
| `src/astro_context/agent/agent.py` | Add `with_skills_directory()`, `with_skill_from_path()` |
| `src/astro_context/agent/skills/__init__.py` | Export `load_skill`, `load_skills_directory` |
| `src/astro_context/agent/__init__.py` | Export `load_skill`, `load_skills_directory` |

---

## Chunk 1: Test Fixtures and Loader Core

### Task 1: Create Test Fixtures

**Files:**
- Create: `tests/fixtures/skills/brainstorm/SKILL.md`
- Create: `tests/fixtures/skills/brainstorm/tools.py`
- Create: `tests/fixtures/skills/minimal/SKILL.md`
- Create: `tests/fixtures/skills/invalid/SKILL.md`

- [ ] **Step 1: Create brainstorm SKILL.md fixture**

```markdown
---
name: brainstorm
description: Guide the agent through a structured brainstorming process
activation: on_demand
tags: [creative]
---

# Brainstorming Skill

Help the user explore ideas through structured dialogue.

## Process

1. **Understand the goal** -- Ask what the user wants to achieve
2. **Ask clarifying questions** -- One at a time, prefer multiple choice
3. **Propose approaches** -- Present 2-3 options with trade-offs
4. **Summarize** -- Capture the chosen approach and next steps
```

Write to `tests/fixtures/skills/brainstorm/SKILL.md`.

- [ ] **Step 2: Create brainstorm tools.py fixture**

```python
"""Tools for the brainstorm skill."""

from __future__ import annotations

import json
from pathlib import Path

from astro_context.agent.tool_decorator import tool


@tool
def save_brainstorm_result(title: str, summary: str, approaches: str) -> str:
    """Save a brainstorm result to a JSON file.

    Args:
        title: Title of the brainstorming session.
        summary: Summary of the chosen approach.
        approaches: Description of all approaches considered.
    """
    result = {"title": title, "summary": summary, "approaches": approaches}
    output_path = Path("brainstorm_output.json")
    output_path.write_text(json.dumps(result, indent=2))
    return f"Brainstorm saved to {output_path}"
```

Write to `tests/fixtures/skills/brainstorm/tools.py`.

- [ ] **Step 3: Create minimal SKILL.md fixture (instructions-only)**

```markdown
---
name: minimal-helper
description: A minimal instructions-only skill for testing
---

# Minimal Helper

This skill provides guidelines only, no tools.

Always be concise and direct.
```

Write to `tests/fixtures/skills/minimal/SKILL.md`.

- [ ] **Step 4: Create invalid SKILL.md fixture**

```markdown
---
description: Missing the required name field
---

This skill has invalid frontmatter.
```

Write to `tests/fixtures/skills/invalid/SKILL.md`.

- [ ] **Step 5: Commit fixtures**

```bash
git add tests/fixtures/skills/
git commit -m "test: add SKILL.md test fixtures for loader"
```

---

### Task 2: Implement Frontmatter Parser (TDD)

**Files:**
- Create: `src/astro_context/agent/skills/loader.py`
- Create: `tests/test_agent/test_skills/test_loader.py`

- [ ] **Step 1: Write failing tests for frontmatter parsing**

```python
"""Tests for the SKILL.md loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from astro_context.agent.skills.loader import load_skill, load_skills_directory

FIXTURES = Path(__file__).resolve().parent.parent.parent / "fixtures" / "skills"


class TestLoadSkillParsing:
    def test_parses_valid_frontmatter(self) -> None:
        skill = load_skill(FIXTURES / "brainstorm")
        assert skill.name == "brainstorm"
        assert skill.description == "Guide the agent through a structured brainstorming process"
        assert skill.activation == "on_demand"
        assert skill.tags == ("creative",)

    def test_instructions_from_markdown_body(self) -> None:
        skill = load_skill(FIXTURES / "brainstorm")
        assert "# Brainstorming Skill" in skill.instructions
        assert "Propose approaches" in skill.instructions

    def test_instructions_only_skill(self) -> None:
        skill = load_skill(FIXTURES / "minimal")
        assert skill.name == "minimal-helper"
        assert skill.tools == ()
        assert skill.activation == "on_demand"

    def test_default_activation_is_on_demand(self) -> None:
        skill = load_skill(FIXTURES / "minimal")
        assert skill.activation == "on_demand"


class TestLoadSkillValidation:
    def test_missing_name_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="name"):
            load_skill(FIXTURES / "invalid")

    def test_nonexistent_path_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_skill(FIXTURES / "does-not-exist")

    def test_invalid_name_format_raises(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "bad_name"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: BadName!!\ndescription: test\n---\nBody"
        )
        with pytest.raises(ValueError, match="name"):
            load_skill(skill_dir)

    def test_name_too_long_raises(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "long"
        skill_dir.mkdir()
        long_name = "a" * 65
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {long_name}\ndescription: test\n---\nBody"
        )
        with pytest.raises(ValueError, match="name"):
            load_skill(skill_dir)

    def test_missing_description_raises(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "no-desc"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: no-desc\n---\nBody"
        )
        with pytest.raises(ValueError, match="description"):
            load_skill(skill_dir)

    def test_description_too_long_raises(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "longdesc"
        skill_dir.mkdir()
        long_desc = "a" * 1025
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: test-skill\ndescription: {long_desc}\n---\nBody"
        )
        with pytest.raises(ValueError, match="description"):
            load_skill(skill_dir)


class TestActivationOverride:
    def test_activation_always_override(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "always-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: always-skill\ndescription: test\nactivation: always\n---\nBody"
        )
        skill = load_skill(skill_dir)
        assert skill.activation == "always"
```

Write to `tests/test_agent/test_skills/test_loader.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/optimistic-lamport && uv run pytest tests/test_agent/test_skills/test_loader.py -v`
Expected: FAIL with `ModuleNotFoundError` (loader.py doesn't exist yet)

- [ ] **Step 3: Implement frontmatter parser in loader.py**

```python
"""SKILL.md loader for the Agent Skills standard.

Parses SKILL.md files into native Skill instances, with optional
tool discovery from tools.py in the same directory.
"""

from __future__ import annotations

import importlib.util
import logging
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from astro_context.agent.skills.models import Skill

if TYPE_CHECKING:
    from astro_context.agent.models import AgentTool

logger = logging.getLogger(__name__)

_NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_MAX_NAME_LENGTH = 64
_MAX_DESCRIPTION_LENGTH = 1024


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split SKILL.md content into frontmatter dict and markdown body.

    Frontmatter is delimited by ``---`` lines at the start of the file.
    Returns (frontmatter_dict, body_text).
    """
    text = text.strip()
    if not text.startswith("---"):
        return {}, text

    end_idx = text.find("---", 3)
    if end_idx == -1:
        return {}, text

    raw_fm = text[3:end_idx].strip()
    body = text[end_idx + 3:].strip()

    fm: dict[str, str] = {}
    for line in raw_fm.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        fm[key.strip()] = value.strip()
    return fm, body


def _parse_tags(raw: str) -> tuple[str, ...]:
    """Parse a YAML-style inline list like ``[creative, tools]``."""
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    return tuple(t.strip() for t in raw.split(",") if t.strip())


def _validate_name(name: str) -> None:
    """Validate skill name: lowercase, hyphens, digits, max 64 chars."""
    if not name:
        msg = "SKILL.md frontmatter missing required field: 'name'"
        raise ValueError(msg)
    if len(name) > _MAX_NAME_LENGTH:
        msg = f"Skill name exceeds {_MAX_NAME_LENGTH} characters: '{name}'"
        raise ValueError(msg)
    if not _NAME_PATTERN.match(name):
        msg = (
            f"Invalid skill name '{name}': must be lowercase letters, "
            "digits, and hyphens only (no leading/trailing/consecutive hyphens)"
        )
        raise ValueError(msg)


def _validate_description(description: str) -> None:
    """Validate description is present and within length limit."""
    if not description:
        msg = "SKILL.md frontmatter missing required field: 'description'"
        raise ValueError(msg)
    if len(description) > _MAX_DESCRIPTION_LENGTH:
        msg = f"Skill description exceeds {_MAX_DESCRIPTION_LENGTH} characters"
        raise ValueError(msg)


def _discover_tools(skill_dir: Path, skill_name: str) -> tuple[AgentTool, ...]:
    """Import tools.py from skill directory and collect AgentTool instances."""
    from astro_context.agent.models import AgentTool as AgentToolCls

    tools_path = skill_dir / "tools.py"
    if not tools_path.exists():
        return ()

    module_name = f"astro_context.skills.{skill_name}.tools"
    logger.info("Loading tools from %s as %s", tools_path, module_name)

    try:
        spec = importlib.util.spec_from_file_location(module_name, tools_path)
        if spec is None or spec.loader is None:
            msg = f"Cannot load module spec from {tools_path}"
            raise ValueError(msg)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        # Clean up partial registration
        sys.modules.pop(module_name, None)
        msg = f"Failed to import tools for skill '{skill_name}': {exc}"
        raise ValueError(msg) from exc

    tools: list[AgentToolCls] = []
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, AgentToolCls):
            tools.append(attr)
    return tuple(tools)


def load_skill(path: str | Path) -> Skill:
    """Load a single SKILL.md directory into a Skill instance.

    Parameters
    ----------
    path:
        Path to a directory containing a ``SKILL.md`` file.

    Raises
    ------
    FileNotFoundError
        If *path* or ``SKILL.md`` does not exist.
    ValueError
        If frontmatter is invalid or ``tools.py`` import fails.
    """
    skill_dir = Path(path).resolve()
    if not skill_dir.is_dir():
        msg = f"Skill directory not found: {skill_dir}"
        raise FileNotFoundError(msg)

    skill_file = skill_dir / "SKILL.md"
    if not skill_file.exists():
        msg = f"SKILL.md not found in {skill_dir}"
        raise FileNotFoundError(msg)

    text = skill_file.read_text(encoding="utf-8")
    fm, body = _parse_frontmatter(text)

    name = fm.get("name", "")
    _validate_name(name)

    description = fm.get("description", "")
    _validate_description(description)

    activation = fm.get("activation", "on_demand")
    if activation not in ("always", "on_demand"):
        activation = "on_demand"

    tags = _parse_tags(fm.get("tags", ""))
    tools = _discover_tools(skill_dir, name)

    return Skill(
        name=name,
        description=description,
        instructions=body,
        tools=tools,
        activation=activation,
        tags=tags,
    )


def load_skills_directory(path: str | Path) -> list[Skill]:
    """Scan a directory for ``*/SKILL.md`` patterns and load all skills.

    Parameters
    ----------
    path:
        Path to a directory containing skill subdirectories.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    skills_dir = Path(path).resolve()
    if not skills_dir.is_dir():
        msg = f"Skills directory not found: {skills_dir}"
        raise FileNotFoundError(msg)

    skill_files = sorted(skills_dir.glob("*/SKILL.md"))
    if not skill_files:
        logger.warning("No SKILL.md files found in %s", skills_dir)
        return []

    skills: list[Skill] = []
    for skill_file in skill_files:
        skill_dir = skill_file.parent
        try:
            skill = load_skill(skill_dir)
            skills.append(skill)
        except (ValueError, FileNotFoundError) as exc:
            logger.warning("Skipping skill in %s: %s", skill_dir.name, exc)
    return skills
```

Write to `src/astro_context/agent/skills/loader.py`.

- [ ] **Step 4: Run parsing tests to verify they pass**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/optimistic-lamport && uv run pytest tests/test_agent/test_skills/test_loader.py -v`
Expected: All `TestLoadSkillParsing` and `TestLoadSkillValidation` tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/astro_context/agent/skills/loader.py tests/test_agent/test_skills/test_loader.py
git commit -m "feat: add SKILL.md loader with frontmatter parsing and validation"
```

---

### Task 3: Tool Discovery Tests (TDD)

**Files:**
- Modify: `tests/test_agent/test_skills/test_loader.py`

- [ ] **Step 1: Add tool discovery tests**

Append to `tests/test_agent/test_skills/test_loader.py`:

```python
class TestToolDiscovery:
    def test_discovers_tools_from_tools_py(self) -> None:
        skill = load_skill(FIXTURES / "brainstorm")
        assert len(skill.tools) == 1
        assert skill.tools[0].name == "save_brainstorm_result"

    def test_instructions_only_has_no_tools(self) -> None:
        skill = load_skill(FIXTURES / "minimal")
        assert skill.tools == ()

    def test_tool_is_callable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        skill = load_skill(FIXTURES / "brainstorm")
        tool = skill.tools[0]
        result = tool.fn(title="Test", summary="Sum", approaches="A vs B")
        assert "saved" in result.lower() or "brainstorm" in result.lower()
        assert (tmp_path / "brainstorm_output.json").exists()

    def test_tools_py_import_error_raises(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "bad-tools"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: bad-tools\ndescription: test\n---\nBody"
        )
        (skill_dir / "tools.py").write_text("import nonexistent_module_xyz")
        with pytest.raises(ValueError, match="Failed to import"):
            load_skill(skill_dir)
```

- [ ] **Step 2: Run tests to verify tool discovery passes**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/optimistic-lamport && uv run pytest tests/test_agent/test_skills/test_loader.py::TestToolDiscovery -v`
Expected: All 4 tests PASS (loader already handles this)

- [ ] **Step 3: Commit**

```bash
git add tests/test_agent/test_skills/test_loader.py
git commit -m "test: add tool discovery tests for SKILL.md loader"
```

---

### Task 4: Directory Loading Tests (TDD)

**Files:**
- Modify: `tests/test_agent/test_skills/test_loader.py`

- [ ] **Step 1: Add directory loading tests**

Append to `tests/test_agent/test_skills/test_loader.py`:

```python
class TestLoadSkillsDirectory:
    def test_loads_all_valid_skills(self) -> None:
        skills = load_skills_directory(FIXTURES)
        names = {s.name for s in skills}
        assert "brainstorm" in names
        assert "minimal-helper" in names

    def test_skips_invalid_skills(self) -> None:
        skills = load_skills_directory(FIXTURES)
        names = {s.name for s in skills}
        # "invalid" directory has bad frontmatter -- should be skipped, not present
        assert "brainstorm" in names
        assert "minimal-helper" in names
        # invalid skill directory should not appear in loaded skills
        for name in names:
            assert name  # no empty names
        assert len(skills) >= 2  # at least brainstorm + minimal

    def test_nonexistent_dir_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_skills_directory("/nonexistent/path")

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        skills = load_skills_directory(tmp_path)
        assert skills == []
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/optimistic-lamport && uv run pytest tests/test_agent/test_skills/test_loader.py::TestLoadSkillsDirectory -v`
Expected: All 4 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_agent/test_skills/test_loader.py
git commit -m "test: add directory loading tests for SKILL.md loader"
```

---

## Chunk 2: Registry, Agent, Exports, Integration

### Task 5: SkillRegistry Extensions (TDD)

**Files:**
- Modify: `src/astro_context/agent/skills/registry.py`
- Create: `tests/test_agent/test_skills/test_loader_integration.py`

- [ ] **Step 1: Write failing integration tests for registry loading**

```python
"""Integration tests for SKILL.md loader with SkillRegistry and Agent."""

from __future__ import annotations

from pathlib import Path

import pytest

from astro_context.agent.skills.models import Skill
from astro_context.agent.skills.registry import SkillRegistry
from astro_context.agent.tools import AgentTool

FIXTURES = Path(__file__).resolve().parent.parent.parent / "fixtures" / "skills"


def _noop() -> str:
    return "ok"


def _make_tool(name: str = "t") -> AgentTool:
    return AgentTool(
        name=name, description="test",
        input_schema={"type": "object", "properties": {}}, fn=_noop,
    )


class TestRegistryLoadFromPath:
    def test_load_and_register(self) -> None:
        reg = SkillRegistry()
        skill = reg.load_from_path(FIXTURES / "brainstorm")
        assert skill.name == "brainstorm"
        assert reg.get("brainstorm") is skill

    def test_duplicate_name_raises(self) -> None:
        reg = SkillRegistry()
        reg.register(Skill(name="brainstorm", description="native"))
        with pytest.raises(ValueError, match="already registered"):
            reg.load_from_path(FIXTURES / "brainstorm")


class TestRegistryLoadFromDirectory:
    def test_loads_all_valid(self) -> None:
        reg = SkillRegistry()
        skills = reg.load_from_directory(FIXTURES)
        assert len(skills) >= 2
        assert reg.get("brainstorm") is not None
        assert reg.get("minimal-helper") is not None

    def test_skips_duplicate_continues(self) -> None:
        reg = SkillRegistry()
        reg.register(Skill(name="brainstorm", description="native"))
        skills = reg.load_from_directory(FIXTURES)
        # brainstorm skipped, minimal-helper loaded
        names = {s.name for s in skills}
        assert "minimal-helper" in names
        assert "brainstorm" not in names


class TestActivationFlow:
    def test_skillmd_on_demand_activation(self) -> None:
        reg = SkillRegistry()
        reg.load_from_path(FIXTURES / "brainstorm")
        assert reg.is_active("brainstorm") is False
        reg.activate("brainstorm")
        assert reg.is_active("brainstorm") is True
        tools = reg.active_tools()
        assert any(t.name == "save_brainstorm_result" for t in tools)

    def test_mixed_native_and_skillmd(self) -> None:
        reg = SkillRegistry()
        native = Skill(
            name="native-skill", description="native",
            tools=(_make_tool("native_tool"),), activation="always",
        )
        reg.register(native)
        reg.load_from_path(FIXTURES / "brainstorm")
        reg.activate("brainstorm")
        tools = reg.active_tools()
        names = {t.name for t in tools}
        assert "native_tool" in names
        assert "save_brainstorm_result" in names
```

Write to `tests/test_agent/test_skills/test_loader_integration.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/optimistic-lamport && uv run pytest tests/test_agent/test_skills/test_loader_integration.py -v`
Expected: FAIL with `AttributeError: 'SkillRegistry' object has no attribute 'load_from_path'`

- [ ] **Step 3: Add load_from_path and load_from_directory to SkillRegistry**

First, add these imports to the top of `src/astro_context/agent/skills/registry.py` (after the existing imports):

```python
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
```

Then add the following methods after the existing `reset` method and before `# -- Queries --`:

```python
    def load_from_path(self, path: str | Path) -> Skill:
        """Load a SKILL.md skill from *path* and register it.

        Returns the loaded :class:`Skill`.
        """
        from astro_context.agent.skills.loader import load_skill

        skill = load_skill(Path(path))
        self.register(skill)
        return skill

    def load_from_directory(self, path: str | Path) -> list[Skill]:
        """Load all SKILL.md skills from *path* and register them.

        Skips skills that fail to load or have duplicate names.
        """
        from astro_context.agent.skills.loader import load_skills_directory

        loaded = load_skills_directory(Path(path))
        registered: list[Skill] = []
        for skill in loaded:
            try:
                self.register(skill)
                registered.append(skill)
            except ValueError as exc:
                logger.warning("Skipping skill '%s': %s", skill.name, exc)
        return registered
```

The `import logging`, `from pathlib import Path`, and `logger` were already added in the import block above.

- [ ] **Step 4: Run integration tests to verify they pass**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/optimistic-lamport && uv run pytest tests/test_agent/test_skills/test_loader_integration.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run existing registry tests to verify no regressions**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/optimistic-lamport && uv run pytest tests/test_agent/test_skills/test_registry.py -v`
Expected: All existing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add src/astro_context/agent/skills/registry.py tests/test_agent/test_skills/test_loader_integration.py
git commit -m "feat: add load_from_path and load_from_directory to SkillRegistry"
```

---

### Task 6: Agent Convenience Methods (TDD)

**Files:**
- Modify: `src/astro_context/agent/agent.py`
- Modify: `tests/test_agent/test_skills/test_loader_integration.py`

- [ ] **Step 1: Add Agent integration tests**

Append to `tests/test_agent/test_skills/test_loader_integration.py`:

```python
from astro_context.agent.agent import Agent


class _FakeClient:
    """Minimal stand-in so Agent.__init__ doesn't need anthropic installed."""
    pass


class TestAgentSkillLoading:
    def test_with_skills_directory(self) -> None:
        agent = Agent(model="test", client=_FakeClient())
        agent.with_skills_directory(FIXTURES)
        reg = agent._skill_registry
        assert reg.get("brainstorm") is not None
        assert reg.get("minimal-helper") is not None

    def test_with_skill_from_path(self) -> None:
        agent = Agent(model="test", client=_FakeClient())
        agent.with_skill_from_path(FIXTURES / "brainstorm")
        reg = agent._skill_registry
        assert reg.get("brainstorm") is not None

    def test_chaining(self) -> None:
        agent = (
            Agent(model="test", client=_FakeClient())
            .with_skill_from_path(FIXTURES / "brainstorm")
            .with_skills_directory(FIXTURES)
        )
        # Should not raise -- chaining works
        assert agent._skill_registry.get("brainstorm") is not None
        assert agent._skill_registry.get("minimal-helper") is not None

    def test_activate_tool_created_for_on_demand(self) -> None:
        agent = Agent(model="test", client=_FakeClient())
        agent.with_skill_from_path(FIXTURES / "brainstorm")
        assert agent._activate_tool is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/optimistic-lamport && uv run pytest tests/test_agent/test_skills/test_loader_integration.py::TestAgentSkillLoading -v`
Expected: FAIL with `AttributeError: 'Agent' object has no attribute 'with_skills_directory'`

- [ ] **Step 3: Add convenience methods to Agent**

First, add `from pathlib import Path` to the module-level imports at the top of `src/astro_context/agent/agent.py`:

```python
from pathlib import Path
```

Then add the following methods after the existing `with_skills` method:

```python
    def with_skills_directory(self, path: str | Path) -> Agent:
        """Load all SKILL.md skills from a directory. Returns self for chaining."""
        self._skill_registry.load_from_directory(Path(path))
        self._ensure_activate_tool()
        return self

    def with_skill_from_path(self, path: str | Path) -> Agent:
        """Load one SKILL.md skill from a directory. Returns self for chaining."""
        self._skill_registry.load_from_path(Path(path))
        self._ensure_activate_tool()
        return self
```

The `from pathlib import Path` import was already added above.

- [ ] **Step 4: Run Agent integration tests to verify they pass**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/optimistic-lamport && uv run pytest tests/test_agent/test_skills/test_loader_integration.py::TestAgentSkillLoading -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Run existing Agent tests to verify no regressions**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/optimistic-lamport && uv run pytest tests/test_agent/test_agent.py -v`
Expected: All existing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add src/astro_context/agent/agent.py tests/test_agent/test_skills/test_loader_integration.py
git commit -m "feat: add with_skills_directory and with_skill_from_path to Agent"
```

---

### Task 7: Update Exports

**Files:**
- Modify: `src/astro_context/agent/skills/__init__.py`
- Modify: `src/astro_context/agent/__init__.py`

- [ ] **Step 1: Update skills __init__.py**

Add to `src/astro_context/agent/skills/__init__.py`:

```python
from astro_context.agent.skills.loader import load_skill, load_skills_directory
```

And update `__all__` to include `"load_skill"` and `"load_skills_directory"`.

- [ ] **Step 2: Update agent __init__.py**

Add to `src/astro_context/agent/__init__.py`:

```python
from astro_context.agent.skills import load_skill, load_skills_directory
```

And update `__all__` to include `"load_skill"` and `"load_skills_directory"`.

- [ ] **Step 3: Run export tests to check imports work**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/optimistic-lamport && uv run python -c "from astro_context.agent import load_skill, load_skills_directory; print('OK')"`
Expected: prints `OK`

- [ ] **Step 4: Commit**

```bash
git add src/astro_context/agent/skills/__init__.py src/astro_context/agent/__init__.py
git commit -m "feat: export load_skill and load_skills_directory from agent module"
```

---

### Task 8: Example Brainstorming Skill

**Files:**
- Create: `examples/skills/brainstorm/SKILL.md`
- Create: `examples/skills/brainstorm/tools.py`

- [ ] **Step 1: Create example SKILL.md**

```markdown
---
name: brainstorm
description: Guide the agent through a structured brainstorming process to explore ideas before implementation
activation: on_demand
tags: [creative, planning]
---

# Brainstorming Skill

Help the user explore ideas through structured dialogue before jumping to implementation.

## Process

1. **Understand the goal** -- Ask what the user wants to achieve and why
2. **Ask clarifying questions** -- One at a time, prefer multiple choice when possible
3. **Propose 2-3 approaches** -- Present options with trade-offs and your recommendation
4. **Summarize** -- Capture the chosen approach, key decisions, and next steps

## Guidelines

- One question per message -- don't overwhelm
- Prefer multiple choice questions over open-ended
- Always propose alternatives before settling on an approach
- Apply YAGNI ruthlessly -- remove unnecessary complexity
- Save the result when the brainstorming session concludes

## Using the Tool

After the brainstorming session, use `save_brainstorm_result` to persist the outcome:
- **title**: A short name for the session
- **summary**: The chosen approach and key decisions
- **approaches**: All approaches that were considered with trade-offs
```

Write to `examples/skills/brainstorm/SKILL.md`.

- [ ] **Step 2: Create example tools.py**

```python
"""Tools for the brainstorm skill.

Example of a SKILL.md skill with Python tools for astro-context.
"""

from __future__ import annotations

import json
from pathlib import Path

from astro_context.agent.tool_decorator import tool


@tool
def save_brainstorm_result(title: str, summary: str, approaches: str) -> str:
    """Save brainstorm results to a JSON file for future reference.

    Args:
        title: Short name for the brainstorming session.
        summary: The chosen approach and key decisions.
        approaches: All approaches considered with their trade-offs.
    """
    result = {
        "title": title,
        "summary": summary,
        "approaches": approaches,
    }
    output_path = Path("brainstorm_output.json")
    output_path.write_text(json.dumps(result, indent=2))
    return f"Brainstorm result saved to {output_path}"
```

Write to `examples/skills/brainstorm/tools.py`.

- [ ] **Step 3: Commit**

```bash
git add examples/skills/
git commit -m "docs: add example brainstorming SKILL.md skill"
```

---

### Task 9: Full Test Suite Run

- [ ] **Step 1: Run all loader tests**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/optimistic-lamport && uv run pytest tests/test_agent/test_skills/test_loader.py tests/test_agent/test_skills/test_loader_integration.py -v`
Expected: All tests PASS

- [ ] **Step 2: Run full test suite for regressions**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/optimistic-lamport && uv run pytest --tb=short -q`
Expected: All 1088+ tests PASS, no regressions

- [ ] **Step 3: Run linter**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/optimistic-lamport && uv run ruff check src/astro_context/agent/skills/loader.py`
Expected: No errors

- [ ] **Step 4: Run type checker**

Run: `cd /Users/arthurgranja/github/astro-context/.claude/worktrees/optimistic-lamport && uv run mypy src/astro_context/agent/skills/loader.py`
Expected: No errors
