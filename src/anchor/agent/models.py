"""Base model for agent tools."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError


class AgentTool(BaseModel):
    """A tool that the Agent can use during conversation.

    Each tool has a name, description, JSON Schema for inputs,
    and a callable that executes the tool logic.

    Supports three tiers of creation:

    1. ``@tool`` decorator (auto-generates schema from type hints)
    2. ``@tool(input_model=MyModel)`` (explicit Pydantic model)
    3. Direct construction with a raw ``input_schema`` dict
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., str]
    input_model: type[BaseModel] | None = None

    def to_tool_schema(self) -> ToolSchema:
        """Convert to provider-agnostic ToolSchema."""
        from anchor.llm.models import ToolSchema

        return ToolSchema(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )

    def validate_input(self, tool_input: dict[str, Any]) -> tuple[bool, str]:
        """Validate tool input against the schema.

        When ``input_model`` is set, uses full Pydantic validation.
        Otherwise falls back to basic JSON Schema type checking.

        Returns
        -------
        tuple[bool, str]
            ``(True, "")`` when valid, ``(False, error_message)`` otherwise.
        """
        if self.input_model is not None:
            return self._pydantic_validate(tool_input)
        return self._basic_validate(tool_input)

    def _pydantic_validate(self, tool_input: dict[str, Any]) -> tuple[bool, str]:
        """Validate using the attached Pydantic input model."""
        assert self.input_model is not None  # noqa: S101
        try:
            self.input_model.model_validate(tool_input)
        except ValidationError as exc:
            return False, str(exc)
        return True, ""

    def _basic_validate(self, tool_input: dict[str, Any]) -> tuple[bool, str]:
        """Fallback validation using basic JSON Schema type checking.

        Checks required fields and basic type matching (string, number,
        integer, boolean).  Extra fields are allowed (lenient mode).
        """
        properties: dict[str, Any] = self.input_schema.get("properties", {})
        required: list[str] = self.input_schema.get("required", [])

        for field_name in required:
            if field_name not in tool_input:
                return False, f"Missing required field: '{field_name}'"

        _type_map: dict[str, type | tuple[type, ...]] = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
        }
        for key, value in tool_input.items():
            if key not in properties:
                continue
            expected_type_name = properties[key].get("type")
            if expected_type_name is None:
                continue
            expected = _type_map.get(expected_type_name)
            if expected is None:
                continue
            if not isinstance(value, expected):
                return (
                    False,
                    f"Field '{key}' expected type '{expected_type_name}', "
                    f"got '{type(value).__name__}'",
                )

        return True, ""
