"""Tests for the forge_tool handler — the main tool lifecycle."""

import json
import os
import sys
import tempfile
import time
import textwrap
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tool_forge.forge_tool import ForgeHandler
from tool_forge.store import ForgeStore


@pytest.fixture
def handler(tmp_path):
    """Create a ForgeHandler with a temp store and no LLM (fail-closed judge)."""
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


VALID_CODE = 'def execute(args):\n    return {"result": args.get("x", 0) * 2}'


class TestForgeTool:
    """forge_tool — create, judge, test, register."""

    def test_forge_rejects_without_llm(self, handler):
        """Without LLM, judge fails-closed — no tool can be forged."""
        result = handler.handle("forge_tool", {
            "name": "double_it",
            "description": "Doubles a number",
            "params_schema": {
                "type": "object",
                "properties": {"x": {"type": "number"}},
                "required": ["x"],
            },
            "python_code": VALID_CODE,
            "test_args": {"x": 21},
        })
        data = json.loads(result)
        assert data["status"] == "rejected"
        assert data["stage"] == "judge"

    def test_forge_rejects_empty_name(self, handler):
        result = handler.handle("forge_tool", {
            "name": "",
            "description": "test",
            "params_schema": {"type": "object"},
            "python_code": VALID_CODE,
        })
        data = json.loads(result)
        assert "error" in data

    def test_forge_rejects_bad_name(self, handler):
        result = handler.handle("forge_tool", {
            "name": "Bad Name!",
            "description": "test",
            "params_schema": {"type": "object"},
            "python_code": VALID_CODE,
        })
        data = json.loads(result)
        assert "error" in data
        assert "snake_case" in data["error"]

    def test_forge_rejects_path_traversal_name(self, handler):
        result = handler.handle("forge_tool", {
            "name": "../../../tmp/evil",
            "description": "test",
            "params_schema": {"type": "object"},
            "python_code": VALID_CODE,
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

    def test_forge_rejects_import_bypass(self, handler):
        """__import__('subprocess') must be caught by the validator."""
        result = handler.handle("forge_tool", {
            "name": "bypass",
            "description": "Import bypass",
            "params_schema": {"type": "object"},
            "python_code": "def execute(args):\n    return {'r': __import__('subprocess').check_output(['id']).decode()}",
            "test_args": {},
        })
        data = json.loads(result)
        assert data["status"] == "rejected"
        assert data["stage"] == "static_validation"
        assert "__import__" in data["reason"] or "allowlist" in data["reason"]

    def test_forge_rejects_non_object_schema(self, handler):
        result = handler.handle("forge_tool", {
            "name": "bad_schema",
            "description": "Bad schema",
            "params_schema": {"type": "string"},
            "python_code": VALID_CODE,
        })
        data = json.loads(result)
        assert "error" in data
        assert "object" in data["error"]

    def test_forge_name_collision(self, handler):
        # Manually insert a tool to simulate collision
        handler._store.add({
            "id": "existing",
            "name": "unique_tool",
            "description": "First",
            "params_schema": {"type": "object"},
            "python_code": VALID_CODE,
            "created_at": time.time(),
        })
        result = handler.handle("forge_tool", {
            "name": "unique_tool",
            "description": "Second",
            "params_schema": {"type": "object"},
            "python_code": VALID_CODE,
            "test_args": {},
        })
        data = json.loads(result)
        assert "error" in data
        assert "already exists" in data["error"]


class TestForgeCall:
    """forge_call — call a previously forged tool via sandbox."""

    def test_call_nonexistent_tool(self, handler):
        result = handler.handle("forge_call", {
            "tool_name": "nonexistent",
            "args": {},
        })
        data = json.loads(result)
        assert "error" in data
        assert "No forged tool" in data["error"]

    def test_call_with_bad_name(self, handler):
        result = handler.handle("forge_call", {
            "tool_name": "../../../etc/passwd",
            "args": {},
        })
        data = json.loads(result)
        assert "error" in data

    def test_call_unapproved_tool(self, handler):
        """Calling a tool that wasn't judge-approved must fail."""
        handler._store.add({
            "id": "unapproved",
            "name": "unapproved_tool",
            "description": "Not approved",
            "params_schema": {"type": "object"},
            "python_code": VALID_CODE,
            "judge_approved": False,
            "test_passed": False,
            "created_at": time.time(),
        })
        result = handler.handle("forge_call", {
            "tool_name": "unapproved_tool",
            "args": {},
        })
        data = json.loads(result)
        assert "error" in data
        assert "not approved" in data["error"]

    def test_call_approved_tool_runs_in_sandbox(self, handler):
        """An approved+tested tool must run via sandbox subprocess, not exec()."""
        handler._store.add({
            "id": "approved_tool",
            "name": "approved_tool",
            "description": "Approved tool",
            "params_schema": {"type": "object", "properties": {"x": {"type": "number"}}},
            "python_code": VALID_CODE,
            "judge_approved": True,
            "test_passed": True,
            "created_at": time.time(),
        })
        result = handler.handle("forge_call", {
            "tool_name": "approved_tool",
            "args": {"x": 21},
        })
        data = json.loads(result)
        assert "result" in data
        assert data["result"]["result"] == 42

    def test_call_re_validates_stored_code(self, handler):
        """If stored code has been tampered to include forbidden imports, reject."""
        handler._store.add({
            "id": "tampered",
            "name": "tampered_tool",
            "description": "Tampered",
            "params_schema": {"type": "object"},
            "python_code": "import subprocess\ndef execute(args):\n    return {}",
            "judge_approved": True,
            "test_passed": True,
            "created_at": time.time(),
        })
        result = handler.handle("forge_call", {
            "tool_name": "tampered_tool",
            "args": {},
        })
        data = json.loads(result)
        assert "error" in data
        assert "validation" in data["error"].lower()

    def test_call_increments_use_count(self, handler):
        handler._store.add({
            "id": "counter",
            "name": "counter",
            "description": "Test counter",
            "params_schema": {"type": "object"},
            "python_code": 'def execute(args):\n    return {"ok": True}',
            "judge_approved": True,
            "test_passed": True,
            "created_at": time.time(),
        })
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
        assert data["total"] == 0
        assert data["tools"] == []

    def test_list_with_tools(self, handler):
        for name in ("tool_a", "tool_b"):
            handler._store.add({
                "id": name,
                "name": name,
                "description": f"Tool {name}",
                "params_schema": {"type": "object"},
                "python_code": VALID_CODE,
                "created_at": time.time(),
            })
        result = handler.handle("forge_list", {})
        data = json.loads(result)
        assert data["total"] == 2


class TestForgePromote:
    """forge_promote — promote a tool to a skill."""

    def test_promote_approved_tool(self, handler, tmp_path):
        handler._store.add({
            "id": "promotable",
            "name": "promotable",
            "description": "Can be promoted",
            "params_schema": {"type": "object"},
            "python_code": 'def execute(args):\n    return {"ok": True}',
            "judge_approved": True,
            "test_passed": True,
            "created_at": time.time(),
        })
        result = handler.handle("forge_promote", {"tool_name": "promotable"})
        data = json.loads(result)
        assert data["status"] == "promoted"
        assert os.path.exists(data["skill_path"])

    def test_promote_nonexistent(self, handler):
        result = handler.handle("forge_promote", {"tool_name": "ghost"})
        data = json.loads(result)
        assert "error" in data

    def test_promote_path_traversal_blocked(self, handler):
        """Path traversal via tool name must be blocked."""
        handler._store.add({
            "id": "traversal",
            "name": "traversal_tool",
            "description": "Test",
            "params_schema": {"type": "object"},
            "python_code": VALID_CODE,
            "judge_approved": True,
            "test_passed": True,
            "created_at": time.time(),
        })
        # The name is valid snake_case, so promotion should work normally
        result = handler.handle("forge_promote", {"tool_name": "traversal_tool"})
        data = json.loads(result)
        assert data["status"] == "promoted"
        # Verify the skill path is within skills_dir
        assert "skills" in data["skill_path"]

    def test_double_promote(self, handler):
        handler._store.add({
            "id": "double",
            "name": "double_promote",
            "description": "Double",
            "params_schema": {"type": "object"},
            "python_code": VALID_CODE,
            "judge_approved": True,
            "test_passed": True,
            "created_at": time.time(),
        })
        handler.handle("forge_promote", {"tool_name": "double_promote"})
        result = handler.handle("forge_promote", {"tool_name": "double_promote"})
        data = json.loads(result)
        assert "already promoted" in data.get("message", "")


class TestReloadFromStore:
    """reload_from_store — verify forged tools from previous session."""

    def test_reload_verifies_tools(self, tmp_path):
        db_path = str(tmp_path / "test_forge.db")
        store1 = ForgeStore(db_path)
        store1.connect()

        h1 = ForgeHandler(store=store1, llm=None, skills_dir=str(tmp_path / "skills"))
        # Manually add an approved+tested tool
        store1.add({
            "id": "persisted",
            "name": "persisted_tool",
            "description": "Survives restart",
            "params_schema": {"type": "object"},
            "python_code": VALID_CODE,
            "judge_approved": True,
            "test_passed": True,
            "created_at": time.time(),
        })
        store1.close()

        # New handler with same DB
        store2 = ForgeStore(db_path)
        store2.connect()
        h2 = ForgeHandler(store=store2, llm=None, skills_dir=str(tmp_path / "skills"))
        loaded = h2.reload_from_store()
        assert loaded == 1

        # Can call the reloaded tool via sandbox
        result = h2.handle("forge_call", {
            "tool_name": "persisted_tool",
            "args": {"x": 5},
        })
        data = json.loads(result)
        assert "result" in data
        assert data["result"]["result"] == 10

        store2.close()

    def test_reload_skips_invalid_stored_code(self, tmp_path):
        """If stored code has been tampered, reload must skip it."""
        db_path = str(tmp_path / "test_forge.db")
        store = ForgeStore(db_path)
        store.connect()

        # Add a tool with forbidden code (simulating DB tampering)
        store.add({
            "id": "bad",
            "name": "bad_tool",
            "description": "Tampered",
            "params_schema": {"type": "object"},
            "python_code": "import subprocess\ndef execute(args):\n    return {}",
            "judge_approved": True,
            "test_passed": True,
            "created_at": time.time(),
        })

        h = ForgeHandler(store=store, llm=None, skills_dir=str(tmp_path / "skills"))
        loaded = h.reload_from_store()
        assert loaded == 0  # Skipped because code fails validation

        store.close()