"""Tests for the store module — SQLite persistence for forged tools."""

import json
import os
import sys
import tempfile
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tool_forge.store import ForgeStore


@pytest.fixture
def store(tmp_path):
    """Create a fresh store with a temp DB."""
    db_path = str(tmp_path / "test_forge.db")
    s = ForgeStore(db_path)
    s.connect()
    yield s
    s.close()


class TestStoreCRUD:
    """Basic CRUD operations on the forge store."""

    def test_connect_creates_schema(self, store):
        assert store.connected
        assert store.count() == 0

    def test_add_and_get(self, store):
        tool = {
            "id": "abc123",
            "name": "double_it",
            "description": "Doubles a number",
            "params_schema": {"type": "object", "properties": {"x": {"type": "number"}}},
            "python_code": "def execute(args):\n    return {'result': args.get('x', 0) * 2}",
            "created_at": time.time(),
        }
        store.add(tool)
        assert store.count() == 1

        fetched = store.get("abc123")
        assert fetched is not None
        assert fetched["name"] == "double_it"
        assert fetched["description"] == "Doubles a number"
        assert fetched["params_schema"]["type"] == "object"

    def test_get_by_name(self, store):
        tool = {
            "id": "def456",
            "name": "reverse_string",
            "description": "Reverses a string",
            "params_schema": {"type": "object"},
            "python_code": "def execute(args):\n    return {}",
            "created_at": time.time(),
        }
        store.add(tool)

        fetched = store.get_by_name("reverse_string")
        assert fetched is not None
        assert fetched["id"] == "def456"

    def test_get_nonexistent(self, store):
        assert store.get("nonexistent") is None
        assert store.get_by_name("nonexistent") is None

    def test_update_fields(self, store):
        tool = {
            "id": "ghi789",
            "name": "counter",
            "description": "Counts things",
            "params_schema": {"type": "object"},
            "python_code": "def execute(args):\n    return {}",
            "created_at": time.time(),
        }
        store.add(tool)

        store.update("ghi789", judge_approved=True, test_passed=True, use_count=5)

        fetched = store.get("ghi789")
        assert fetched["judge_approved"] is True
        assert fetched["test_passed"] is True
        assert fetched["use_count"] == 5

    def test_increment_use(self, store):
        tool = {
            "id": "jkl012",
            "name": "tool_x",
            "description": "Test tool",
            "params_schema": {"type": "object"},
            "python_code": "def execute(args):\n    return {}",
            "created_at": time.time(),
        }
        store.add(tool)

        store.increment_use("jkl012")
        store.increment_use("jkl012")
        store.increment_use("jkl012")

        fetched = store.get("jkl012")
        assert fetched["use_count"] == 3
        assert fetched["last_used_at"] is not None

    def test_remove(self, store):
        tool = {
            "id": "mno345",
            "name": "deleteme",
            "description": "Temporary",
            "params_schema": {"type": "object"},
            "python_code": "def execute(args):\n    return {}",
            "created_at": time.time(),
        }
        store.add(tool)
        assert store.count() == 1

        assert store.remove("mno345") is True
        assert store.count() == 0
        assert store.get("mno345") is None

    def test_remove_nonexistent(self, store):
        assert store.remove("nonexistent") is False

    def test_list_all(self, store):
        for i in range(3):
            store.add({
                "id": f"tool_{i}",
                "name": f"tool_{i}",
                "description": f"Tool {i}",
                "params_schema": {"type": "object"},
                "python_code": "def execute(args):\n    return {}",
                "created_at": time.time() + i,
            })
        tools = store.list_all()
        assert len(tools) == 3
        # Newest first
        assert tools[0]["name"] == "tool_2"

    def test_list_approved(self, store):
        # Add one approved+tested and one not
        store.add({
            "id": "approved1",
            "name": "approved_tool",
            "description": "Approved",
            "params_schema": {"type": "object"},
            "python_code": "def execute(args):\n    return {}",
            "judge_approved": True,
            "test_passed": True,
            "use_count": 5,
            "created_at": time.time(),
        })
        store.add({
            "id": "unapproved1",
            "name": "unapproved_tool",
            "description": "Not approved",
            "params_schema": {"type": "object"},
            "python_code": "def execute(args):\n    return {}",
            "judge_approved": False,
            "test_passed": False,
            "created_at": time.time(),
        })

        approved = store.list_approved()
        assert len(approved) == 1
        assert approved[0]["name"] == "approved_tool"

    def test_name_collision_overwrites(self, store):
        """Adding a tool with the same name should overwrite (INSERT OR REPLACE)."""
        store.add({
            "id": "v1",
            "name": "same_name",
            "description": "Version 1",
            "params_schema": {"type": "object"},
            "python_code": "def execute(args):\n    return {}",
            "created_at": time.time(),
        })
        store.add({
            "id": "v2",
            "name": "same_name",
            "description": "Version 2",
            "params_schema": {"type": "object"},
            "python_code": "def execute(args):\n    return {}",
            "created_at": time.time(),
        })
        assert store.count() == 1
        fetched = store.get("v2")
        assert fetched["description"] == "Version 2"