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
