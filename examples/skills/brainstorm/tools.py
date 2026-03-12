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
