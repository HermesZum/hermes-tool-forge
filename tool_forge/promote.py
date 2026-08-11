"""Promote successful forged tools to SKILL.md files.

When a forged tool proves useful (use_count threshold or manual promotion),
it can be promoted to a skill that loads automatically in future sessions.
The skill includes the tool's code, schema, and description, so the tool
is available from session start.
"""

import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

AUTO_PROMOTE_THRESHOLD = 3


def promote_to_skill(
    tool: Dict[str, Any],
    skills_dir: str,
) -> Optional[str]:
    """Promote a forged tool to a SKILL.md file.

    Args:
        tool: The forged tool record from the store.
        skills_dir: Path to the skills directory (e.g. ~/.hermes/skills/).

    Returns:
        The path to the created SKILL.md file, or None on failure.
    """
    try:
        name = tool.get("name", "")
        # Sanitize name — only allow [a-z0-9_-] for the directory
        safe_name = re.sub(r'[^a-z0-9_-]', '', name.lower())
        if not safe_name:
            logger.error("forge-promote: invalid tool name '%s'", name)
            return None

        skill_dir = Path(skills_dir) / "forged" / f"forged-{safe_name}"
        skills_root = Path(skills_dir).resolve()

        # Path traversal protection — ensure skill_dir is within skills_dir
        if not str(skill_dir.resolve()).startswith(str(skills_root)):
            logger.error("forge-promote: path traversal detected for '%s'", name)
            return None

        skill_dir.mkdir(parents=True, exist_ok=True)

        # Build content WITHOUT .format() — use string concatenation to avoid
        # format-string injection from tool fields containing { } chars
        params = tool.get("params_schema", {})
        params_table = _build_params_table(params)

        # Escape braces in tool fields to prevent format-string issues
        safe_desc = tool.get("description", "").replace("{", "{{").replace("}", "}}")
        safe_code = tool.get("python_code", "").replace("{", "{{").replace("}", "}}")

        content = (
            "---\n"
            f"name: forged-{safe_name}\n"
            f'description: "Auto-promoted from forged tool. {safe_desc}"\n'
            "version: 1.0.0\n"
            "origin: forged\n"
            f"forged_tool_id: {tool.get('id', '')}\n"
            "---\n\n"
            f"# Forged Tool: {safe_name}\n\n"
            f"{safe_desc}\n\n"
            "## Parameters\n\n"
            f"{params_table}\n\n"
            "## Implementation\n\n"
            "The following Python code defines the tool's `execute()` function.\n"
            "It is called with a dict of parameters and returns a JSON-serializable result.\n\n"
            f"```python\n{safe_code}\n```\n\n"
            "## Usage\n\n"
            "Call `execute(args)` where args is a dict matching the parameters above.\n"
            "The function returns a dict with the result.\n\n"
            "## Origin\n\n"
            f"- Forged: {time.strftime('%Y-%m-%d', time.localtime(tool.get('created_at', time.time())))}\n"
            f"- Judge approved: {tool.get('judge_approved', False)}\n"
            f"- Test passed: {tool.get('test_passed', False)}\n"
            f"- Use count at promotion: {tool.get('use_count', 0)}\n"
        )

        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text(content, encoding="utf-8")

        logger.info("forge-promote: promoted %s to %s", name, skill_path)
        return str(skill_path)

    except Exception as e:
        logger.error("forge-promote: failed to promote %s: %s", tool.get("name"), e)
        return None


def should_auto_promote(tool: Dict[str, Any]) -> bool:
    """Check if a tool meets the criteria for auto-promotion."""
    if tool.get("promoted", False):
        return False
    if not tool.get("judge_approved", False) or not tool.get("test_passed", False):
        return False
    if tool.get("use_count", 0) < AUTO_PROMOTE_THRESHOLD:
        return False
    return True


def _build_params_table(params_schema: Dict[str, Any]) -> str:
    """Build a markdown table from a JSON schema's parameters."""
    properties = params_schema.get("properties", {})
    if not properties:
        return "No parameters."

    required = set(params_schema.get("required", []))

    lines = [
        "| Parameter | Type | Required | Description |",
        "|---|---|---|---|",
    ]
    for name, prop in properties.items():
        ptype = prop.get("type", "any")
        req = "Yes" if name in required else "No"
        desc = prop.get("description", "").replace("|", "\\|")
        lines.append(f"| {name} | {ptype} | {req} | {desc} |")

    return "\n".join(lines)