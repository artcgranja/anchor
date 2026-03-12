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
