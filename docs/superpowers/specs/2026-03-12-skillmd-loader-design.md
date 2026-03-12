# SKILL.md Loader for astro-context

**Date:** 2026-03-12
**Status:** Approved
**Scope:** Add SKILL.md loading support to astro-context's agent skill system

## Summary

Add a SKILL.md loader that parses the open Agent Skills standard (agentskills.io) into native `Skill` instances, enabling astro-context agents to consume skills from the 2026 cross-platform skill ecosystem alongside existing Python-native skills.

## Context

The SKILL.md format (published by Anthropic, Dec 2025) is the de facto standard for AI coding assistant skills in 2026, adopted by Claude Code, Codex, Copilot, Cursor, Windsurf, and Antigravity. astro-context already has a progressive tool disclosure system (`Skill`, `SkillRegistry`, `@tool` decorator) that aligns naturally with SKILL.md's design.

No other context engineering toolkit bridges development-time SKILL.md files with runtime agent skills. This is a differentiator.

## Design Decisions

1. **Unified registry** -- SKILL.md skills and Python-native skills coexist in the same `SkillRegistry`, same activation model, same API. No subclasses, no type markers.
2. **Hybrid skills** -- A skill can be instructions-only (LLM guidelines) OR instructions + executable Python tools. The loader detects what's available.
3. **Both auto-scan and explicit loading** -- `with_skills_directory()` loads all skills from a path; `with_skill_from_path()` loads one specific skill.
4. **No new dependencies** -- Uses PyYAML (transitive via Pydantic) or a lightweight regex-based frontmatter parser.
5. **Approach 1 (Loader as utility)** -- Chosen over subclass or plugin system approaches for simplicity.

## Architecture

### SKILL.md to Skill Mapping

| SKILL.md field       | Skill field    | Notes                                                    |
|----------------------|----------------|----------------------------------------------------------|
| `name` (frontmatter) | `name`         | Required. Validated: lowercase, hyphens, max 64 chars    |
| `description`        | `description`  | Required. Max 1024 chars                                 |
| Markdown body        | `instructions` | Full markdown content after frontmatter                  |
| `scripts/` or `tools.py` | `tools`   | Auto-discovered `@tool`-decorated functions              |
| `activation`         | `activation`   | Defaults to `"on_demand"`, overridable in frontmatter    |
| `tags`               | `tags`         | Passed through if present                                |

### New Module: `agent/skills/loader.py`

The loader performs three steps:

1. **Parse** -- Read SKILL.md, extract YAML frontmatter + markdown body
2. **Discover tools** -- If `tools.py` or `scripts/*.py` exists in the skill directory, import and collect all `AgentTool` instances
3. **Build Skill** -- Create a `Skill(name=..., description=..., instructions=..., tools=..., activation=..., tags=...)`

Key functions:

```python
def load_skill(path: Path) -> Skill:
    """Load a single SKILL.md directory into a Skill instance."""

def load_skills_directory(path: Path) -> list[Skill]:
    """Scan a directory for */SKILL.md patterns, return all loaded skills."""
```

### SkillRegistry Extensions

Two new methods on `SkillRegistry`:

```python
def load_from_path(self, path: Path) -> Skill:
    """Load a SKILL.md skill and register it. Returns the skill."""

def load_from_directory(self, path: Path) -> list[Skill]:
    """Load all SKILL.md skills from a directory and register them."""
```

No changes to existing methods (`register`, `activate`, `active_tools`, `skill_discovery_prompt`).

### Agent Convenience Methods

Two new methods on `Agent`:

```python
def with_skills_directory(self, path: str | Path) -> Agent:
    """Load all SKILL.md skills from a directory. Returns self for chaining."""

def with_skill_from_path(self, path: str | Path) -> Agent:
    """Load one SKILL.md skill from a directory. Returns self for chaining."""
```

Usage:

```python
agent = (
    Agent(model="claude-sonnet-4-5-20251001")
    .with_system_prompt("You are a helpful assistant.")
    .with_skills_directory("./skills")           # all SKILL.md skills
    .with_skill_from_path("./extras/brainstorm") # one specific skill
    .with_skill(memory_skill(memory))            # native Python skill
)
```

### Security Considerations

- Only load from explicitly configured directories (not arbitrary URLs)
- Log what Python modules get imported during tool discovery
- Tool discovery uses controlled `importlib` imports, not `exec`

## Test Brainstorming Skill

A hybrid SKILL.md skill that validates both loader paths:

```
examples/skills/brainstorm/
    SKILL.md     # Instructions for brainstorming flow
    tools.py     # save_brainstorm_result tool
```

**SKILL.md frontmatter:**
- `name: brainstorm`
- `description: Guide the agent through a structured brainstorming process`
- `activation: on_demand`
- `tags: [creative]`

**SKILL.md body:**
Instructions guiding the agent through: understand goal, ask clarifying questions, propose 2-3 approaches with trade-offs, summarize chosen approach.

**tools.py:**
- `save_brainstorm_result(title: str, summary: str, approaches: str) -> str` -- Persists brainstorm output to a JSON file.

## File Changes

### New Files

| File | Purpose |
|------|---------|
| `src/astro_context/agent/skills/loader.py` | SKILL.md parser + tool discovery |
| `tests/agent/skills/test_loader.py` | Unit tests for loader |
| `tests/agent/skills/test_loader_integration.py` | Integration tests |
| `tests/fixtures/skills/brainstorm/SKILL.md` | Test brainstorming skill |
| `tests/fixtures/skills/brainstorm/tools.py` | Test tool (save_brainstorm_result) |
| `tests/fixtures/skills/minimal/SKILL.md` | Instructions-only test skill |
| `tests/fixtures/skills/invalid/SKILL.md` | Invalid frontmatter test case |
| `examples/skills/brainstorm/SKILL.md` | Example brainstorming skill |
| `examples/skills/brainstorm/tools.py` | Example tool implementation |

### Modified Files

| File | Change |
|------|--------|
| `src/astro_context/agent/skills/registry.py` | Add `load_from_path()`, `load_from_directory()` |
| `src/astro_context/agent/agent.py` | Add `with_skills_directory()`, `with_skill_from_path()` |
| `src/astro_context/agent/skills/__init__.py` | Export loader functions |
| `src/astro_context/agent/__init__.py` | Export loader functions |

## Testing Strategy

### Unit Tests (`tests/agent/skills/test_loader.py`)

- Parse SKILL.md with valid frontmatter -> correct Skill fields
- Parse SKILL.md without tools -> instructions-only skill (empty tools tuple)
- Parse SKILL.md with tools.py -> discovers @tool-decorated functions
- Invalid frontmatter (missing name/description) -> clear error
- Invalid SKILL.md path -> clear error
- `load_skills_directory` -> finds all */SKILL.md in directory

### Integration Tests (`tests/agent/skills/test_loader_integration.py`)

- Load brainstorm skill from SKILL.md -> registers in SkillRegistry
- Activate brainstorm skill -> tools become available via active_tools()
- Execute save_brainstorm_result tool -> produces output
- Agent.with_skills_directory() -> loads and registers all skills
- Agent.with_skill_from_path() -> loads one specific skill
- Mix of native Python skill + SKILL.md skill in same registry -> no conflicts

### Test Fixtures

Directory `tests/fixtures/skills/` with sample skills: valid, invalid, instructions-only, with-tools.

## Out of Scope

- Plugin system with versioning/dependency resolution (future)
- Skills marketplace or registry service (future)
- Separate skills repository with diverse skills (future, after loader is validated)
- Full SKILL.md spec support (assets/, references/) -- start with frontmatter + markdown + Python tools
