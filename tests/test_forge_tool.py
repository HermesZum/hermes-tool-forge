"""Tests for the forge_tool handler — the main tool lifecycle."""

import json
import os
import sys
import tempfile
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tool_forge.forge_tool import ForgeHandler
from tool_forge.store import ForgeStore


@pytest.fixture
def handler(tmp_path):
    """Create a ForgeHandler with a temp store and no LLM."""
    db_path = str(tmp_path / "test_forge.db")
    store = ForgeStore(db_path)
    store.connect()
    h = ForgeHandler(
        store=store,
        llm=None,
        skills_dir=str(tmp_path / "skills"),
        session_id="test-session",
    )
    yield h
    store.close()


class TestForgeTool:
    """forge_tool — create, judge, test, register."""

    def test_successful_forge(self, handler):
        result = handler.handle("forge_tool", {
            "name": "double_it",
            "description": "Doubles a number",
            "params_schema": {
                "type": "object",
                "properties": {"x": {"type": "number"}},
                "required": ["x"],
            },
            "python_code": "def execute(args):\n    return {'result': args.get('x', 0) * 2}",
            "test_args": {"x": 21},
        })
        data = json.loads(result)
        assert data["status"] == "forged"
        assert data["name"] == "double_it"
        assert data["judge_verdict"]["approved"] is True

    def test_forge_rejects_empty_name(self, handler):
        result = handler.handle("forge_tool", {
            "name": "",
            "description": "test",
            "params_schema": {"type": "object"},
            "python_code": "def execute(args):\n    return {}",
        })
        data = json.loads(result)
        assert "error" in data

    def test_forge_rejects_forbidden_code(self, handler):
        result = handler.handle("forge_tool", {
            "name": "hack",
            "description": "Network tool",
            "params_schema": {"type": "object"},
            "python_code": "import socket\ndef execute(args):\n    return {}",
            "test_args": {},
        })
        data = json.loads(result)
        assert data["status"] == "rejected"
        assert data["stage"] == "static_validation"

    def test_forge_rejects_no_execute_function(self, handler):
        result = handler.handle("forge_tool", {
            "name": "no_exec",
            "description": "Missing execute",
            "params_schema": {"type": "object"},
            "python_code": "def not_execute(args):\n    return {}",
            "test_args": {},
        })
        data = json.loads(result)
        assert data["status"] == "test_failed"

    def test_forge_name_collision(self, handler):
        # First forge succeeds
        handler.handle("forge_tool", {
            "name": "unique_tool",
            "description": "First version",
            "params_schema": {"type": "object"},
            "python_code": "def execute(args):\n    return {}",
            "test_args": {},
        })
        # Second with same name fails
        result = handler.handle("forge_tool", {
            "name": "unique_tool",
            "description": "Second version",
            "params_schema": {"type": "object"},
            "python_code": "def execute(args):\n    return {}",
            "test_args": {},
        })
        data = json.loads(result)
        assert "error" in data
        assert "already exists" in data["error"]


class TestForgeCall:
    """forge_call — call a previously forged tool."""

    def test_call_forged_tool(self, handler):
        # Forge a tool first
        handler.handle("forge_tool", {
            "name": "adder",
            "description": "Adds two numbers",
            "params_schema": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}},
            "python_code": "def execute(args):\n    return {'sum': args.get('a', 0) + args.get('b', 0)}",
            "test_args": {"a": 1, "b": 2},
        })

        # Call it
        result = handler.handle("forge_call", {
            "tool_name": "adder",
            "args": {"a": 10, "b": 20},
        })
        data = json.loads(result)
        assert "result" in data
        assert data["result"]["sum"] == 30

    def test_call_nonexistent_tool(self, handler):
        result = handler.handle("forge_call", {
            "tool_name": "nonexistent",
            "args": {},
        })
        data = json.loads(result)
        assert "error" in data
        assert "No forged tool" in data["error"]

    def test_call_increments_use_count(self, handler):
        handler.handle("forge_tool", {
            "name": "counter",
            "description": "Test counter",
            "params_schema": {"type": "object"},
            "python_code": "def execute(args):\n    return {'ok': True}",
            "test_args": {},
        })

        # Call it 3 times
        for _ in range(3):
            handler.handle("forge_call", {"tool_name": "counter", "args": {}})

        tools = json.loads(handler.handle("forge_list", {}))
        tool = [t for t in tools["tools"] if t["name"] == "counter"][0]
        assert tool["use_count"] == 3


class TestForgeList:
    """forge_list — list all forged tools."""

    def test_empty_list(self, handler):
        result = handler.handle("forge_list", {})
        data = json.loads(result)
        assert data["tools"] == []
        assert data["total"] == 0

    def test_list_with_tools(self, handler):
        handler.handle("forge_tool", {
            "name": "tool_a",
            "description": "Tool A",
            "params_schema": {"type": "object"},
            "python_code": "def execute(args):\n    return {}",
            "test_args": {},
        })
        handler.handle("forge_tool", {
            "name": "tool_b",
            "description": "Tool B",
            "params_schema": {"type": "object"},
            "python_code": "def execute(args):\n    return {}",
            "test_args": {},
        })

        result = handler.handle("forge_list", {})
        data = json.loads(result)
        assert data["total"] == 2
        names = {t["name"] for t in data["tools"]}
        assert names == {"tool_a", "tool_b"}


class TestForgePromote:
    """forge_promote — promote a tool to a skill."""

    def test_promote_approved_tool(self, handler, tmp_path):
        handler.handle("forge_tool", {
            "name": "promotable",
            "description": "Can be promoted",
            "params_schema": {"type": "object"},
            "python_code": "def execute(args):\n    return {'ok': True}",
            "test_args": {},
        })

        result = handler.handle("forge_promote", {"tool_name": "promotable"})
        data = json.loads(result)
        assert data["status"] == "promoted"
        assert os.path.exists(data["skill_path"])

    def test_promote_nonexistent(self, handler):
        result = handler.handle("forge_promote", {"tool_name": "ghost"})
        data = json.loads(result)
        assert "error" in data

    def test_double_promote(self, handler):
        handler.handle("forge_tool", {
            "name": "double",
            "description": "Double promote",
            "params_schema": {"type": "object"},
            "python_code": "def execute(args):\n    return {}",
            "test_args": {},
        })

        # First promote
        handler.handle("forge_promote", {"tool_name": "double"})
        # Second promote
        result = handler.handle("forge_promote", {"tool_name": "double"})
        data = json.loads(result)
        assert "already promoted" in data.get("message", "")


class TestReloadFromStore:
    """reload_from_store — restore forged tools from previous session."""

    def test_reload_restores_tools(self, tmp_path):
        db_path = str(tmp_path / "test_forge.db")
        store1 = ForgeStore(db_path)
        store1.connect()

        h1 = ForgeHandler(store=store1, llm=None, skills_dir=str(tmp_path / "skills"))
        h1.handle("forge_tool", {
            "name": "persisted_tool",
            "description": "Survives restart",
            "params_schema": {"type": "object"},
            "python_code": "def execute(args):\n    return {'persisted': True}",
            "test_args": {},
        })
        store1.close()

        # New handler with same DB
        store2 = ForgeStore(db_path)
        store2.connect()
        h2 = ForgeHandler(store=store2, llm=None, skills_dir=str(tmp_path / "skills"))
        loaded = h2.reload_from_store()
        assert loaded == 1

        # Call the reloaded tool
        result = h2.handle("forge_call", {
            "tool_name": "persisted_tool",
            "args": {},
        })
        data = json.loads(result)
        assert data["result"]["persisted"] is True

        store2.close()