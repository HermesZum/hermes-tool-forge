"""Tests for the promote module — promoting forged tools to SKILL.md."""

import os
import sys
import tempfile
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tool_forge.promote import promote_to_skill, should_auto_promote, _build_params_table


class TestPromoteToSkill:
    """Promote a forged tool to a SKILL.md file."""

    def test_successful_promotion(self, tmp_path):
        tool = {
            "id": "abc123",
            "name": "doubler",
            "description": "Doubles a number",
            "params_schema": {
                "type": "object",
                "properties": {"x": {"type": "number", "description": "Number to double"}},
                "required": ["x"],
            },
            "python_code": "def execute(args):\n    return {'result': args.get('x', 0) * 2}",
            "judge_approved": True,
            "test_passed": True,
            "use_count": 5,
            "created_at": time.time(),
        }
        path = promote_to_skill(tool, str(tmp_path / "skills"))
        assert path is not None
        assert os.path.exists(path)
        assert "forged-doubler" in path
        assert path.endswith("SKILL.md")

        # Check file content
        with open(path) as f:
            content = f.read()
        assert "doubler" in content
        assert "Doubles a number" in content
        assert "def execute(args)" in content
        assert "abc123" in content

    def test_promotion_creates_nested_dirs(self, tmp_path):
        tool = {
            "id": "xyz",
            "name": "test_tool",
            "description": "Test",
            "params_schema": {"type": "object"},
            "python_code": "def execute(args):\n    return {}",
            "judge_approved": True,
            "test_passed": True,
            "use_count": 1,
            "created_at": time.time(),
        }
        path = promote_to_skill(tool, str(tmp_path / "deep" / "skills"))
        assert path is not None
        assert os.path.exists(path)


class TestShouldAutoPromote:
    """Check auto-promotion criteria."""

    def test_eligible_tool(self):
        tool = {
            "judge_approved": True,
            "test_passed": True,
            "use_count": 3,
            "promoted": False,
        }
        assert should_auto_promote(tool) is True

    def test_not_enough_uses(self):
        tool = {
            "judge_approved": True,
            "test_passed": True,
            "use_count": 2,
            "promoted": False,
        }
        assert should_auto_promote(tool) is False

    def test_already_promoted(self):
        tool = {
            "judge_approved": True,
            "test_passed": True,
            "use_count": 10,
            "promoted": True,
        }
        assert should_auto_promote(tool) is False

    def test_not_approved(self):
        tool = {
            "judge_approved": False,
            "test_passed": True,
            "use_count": 10,
            "promoted": False,
        }
        assert should_auto_promote(tool) is False

    def test_not_tested(self):
        tool = {
            "judge_approved": True,
            "test_passed": False,
            "use_count": 10,
            "promoted": False,
        }
        assert should_auto_promote(tool) is False


class TestBuildParamsTable:
    """Params table generation from JSON schema."""

    def test_with_properties(self):
        schema = {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "Input value"},
                "y": {"type": "string", "description": "Label"},
            },
            "required": ["x"],
        }
        table = _build_params_table(schema)
        assert "| x | number | Yes | Input value |" in table
        assert "| y | string | No | Label |" in table

    def test_empty_properties(self):
        schema = {"type": "object", "properties": {}}
        table = _build_params_table(schema)
        assert "No parameters" in table