"""Promote successful forged tools to SKILL.md files.

When a forged tool proves useful (use_count threshold or manual promotion),
it can be promoted to a skill that loads automatically in future sessions.
The skill includes the tool's code, schema, and description, so the tool
is available from session start — no mid-session registration needed,
no prompt-caching break.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Minimum use_count before a tool is eligible for auto-promotion
AUTO_PROMOTE_THRESHOLD = 3

# Skill template — the tool's execute() function becomes the skill's
# core logic, wrapped in a handler that the agent can call.
SKILL_TEMPLATE = """---
name: forged-{name}
description: "Auto-promoted from forged tool. {description}"
version: 1.0.0
origin: forged
forged_tool_id: {tool_id}
---

# Forged Tool: {name}

{description}

## Parameters

{params_table}

## Implementation

The following Python code defines the tool's ``execute()`` function.
It is called with a dict of parameters and returns a JSON-serializable result.

```python
{python_code}
```

## Usage

Call ``execute(args)`` where args is a dict matching the parameters above.
The function returns a dict with the result.

## Origin

- Forged: {created_at}
- Judge approved: {judge_approved}
- Test passed: {test_passed}
- Use count at promotion: {use_count}
"""


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
        # Create the skill directory
        skill_name = f"forged-{tool['name']}"
        skill_dir = Path(skills_dir) / "forged" / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Build the params table
        params = tool.get("params_schema", {})
        params_table = _build_params_table(params)

        # Format the SKILL.md
        content = SKILL_TEMPLATE.format(
            name=tool["name"],
            description=tool.get("description", ""),
            tool_id=tool["id"],
            params_table=params_table,
            python_code=tool.get("python_code", ""),
            created_at=time.strftime(
                "%Y-%m-%d", time.localtime(tool.get("created_at", time.time()))
            ),
            judge_approved=tool.get("judge_approved", False),
            test_passed=tool.get("test_passed", False),
            use_count=tool.get("use_count", 0),
        )

        # Write the SKILL.md
        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text(content, encoding="utf-8")

        logger.info("forge-promote: promoted %s to %s", tool["name"], skill_path)
        return str(skill_path)

    except Exception as e:
        logger.error("forge-promote: failed to promote %s: %s", tool.get("name"), e)
        return None


def should_auto_promote(tool: Dict[str, Any]) -> bool:
    """Check if a tool meets the criteria for auto-promotion.

    A tool is auto-promotion eligible if:
    - It is judge-approved and test-passed
    - It has not already been promoted
    - Its use_count has crossed the threshold
    """
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
        desc = prop.get("description", "")
        lines.append(f"| {name} | {ptype} | {req} | {desc} |")

    return "\n".join(lines)