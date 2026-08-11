"""Forge tool handler — the main tool the LLM calls to create new tools.

Exposes the ``forge_tool`` agent tool which:
1. Validates the generated code (static analysis with import allowlist)
2. Runs the LLM judge for safety review (fail-closed if LLM unavailable)
3. Tests the code in a sandbox subprocess
4. Persists the tool definition for future sessions

Forged tools ALWAYS run in the sandbox subprocess — never in the main
Hermes process. This is enforced by using run_code() for both testing
and normal forge_call invocations.

The agent calls forged tools via forge_call, which re-runs the code in
the sandbox with the provided arguments.
"""

import json
import logging
import re
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from .judge import judge_code
from .promote import promote_to_skill, should_auto_promote
from .sandbox import run_code, validate_code, SandboxError
from .store import ForgeStore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Tool name validation — snake_case, 1-64 chars, starts with letter
_TOOL_NAME_RE = re.compile(r'^[a-z][a-z0-9_]{0,63}$')

# Default test arguments for sandbox testing
_DEFAULT_TEST_ARGS = {"test": True}

# Forge tool schema (what the LLM sees in the system prompt)
FORGE_TOOL_SCHEMA = {
    "name": "forge_tool",
    "description": (
        "Create a new tool at runtime. Write a Python function called "
        "'execute' that takes a dict of arguments and returns a "
        "JSON-serializable dict. A safety judge reviews the code, then "
        "it is tested in a sandbox. If approved and tested, the tool is "
        "registered and callable via forge_call. Use this when you need "
        "a capability that no existing tool provides."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Unique tool name (snake_case, no spaces, 1-64 chars)",
            },
            "description": {
                "type": "string",
                "description": "What the tool does, for the agent catalog",
            },
            "params_schema": {
                "type": "object",
                "description": (
                    "JSON schema for the tool's parameters. "
                    "Must have 'type': 'object', 'properties': {...}, "
                    "and optionally 'required': [...]."
                ),
            },
            "python_code": {
                "type": "string",
                "description": (
                    "Python code defining execute(args: dict) -> dict. "
                    "Only stdlib modules from the allowlist: json, math, re, "
                    "collections, datetime, itertools, functools, etc. "
                    "No network, no filesystem writes, no subprocess."
                ),
            },
            "test_args": {
                "type": "object",
                "description": "Arguments to test the tool with in the sandbox",
            },
        },
        "required": ["name", "description", "params_schema", "python_code"],
    },
}

FORGE_CALL_SCHEMA = {
    "name": "forge_call",
    "description": (
        "Call a previously forged tool by name. The tool must have been "
        "created with forge_tool and passed judge review + sandbox test."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "description": "Name of the forged tool to call",
            },
            "args": {
                "type": "object",
                "description": "Arguments to pass to the tool's execute() function",
            },
        },
        "required": ["tool_name"],
    },
}

FORGE_LIST_SCHEMA = {
    "name": "forge_list",
    "description": "List all forged tools with their status and usage stats.",
    "parameters": {"type": "object", "properties": {}},
}

FORGE_PROMOTE_SCHEMA = {
    "name": "forge_promote",
    "description": (
        "Promote a forged tool to a SKILL.md file so it loads automatically "
        "in future sessions. The tool must be judge-approved and test-passed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "description": "Name of the forged tool to promote",
            },
        },
        "required": ["tool_name"],
    },
}

ALL_TOOL_SCHEMAS = [
    FORGE_TOOL_SCHEMA,
    FORGE_CALL_SCHEMA,
    FORGE_LIST_SCHEMA,
    FORGE_PROMOTE_SCHEMA,
]


class ForgeHandler:
    """Manages the forge tool lifecycle — create, judge, test, call, promote.

    Forged tools are NEVER exec'd in the main process. Their code is stored
    in SQLite and re-run in the sandbox subprocess on every forge_call.
    """

    def __init__(
        self,
        store: ForgeStore,
        llm: Any = None,
        skills_dir: str = "",
        session_id: str = "",
    ):
        self._store = store
        self._llm = llm
        self._skills_dir = skills_dir
        self._session_id = session_id
        self._lock = threading.RLock()

    def handle(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Dispatch a forge_* tool call."""
        try:
            if tool_name == "forge_tool":
                return self._handle_forge(args)
            elif tool_name == "forge_call":
                return self._handle_call(args)
            elif tool_name == "forge_list":
                return self._handle_list(args)
            elif tool_name == "forge_promote":
                return self._handle_promote(args)
            else:
                return json.dumps({"error": f"Unknown forge tool: {tool_name}"})
        except Exception as e:
            logger.error("forge: %s failed: %s", tool_name, e, exc_info=True)
            return json.dumps({"error": str(e)})

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return ALL_TOOL_SCHEMAS

    def _validate_name(self, name: str) -> Optional[str]:
        """Validate a tool name. Returns None if OK, error message if invalid."""
        if not name:
            return "name is required"
        if not _TOOL_NAME_RE.match(name):
            return ("name must be snake_case: 1-64 chars, lowercase letters, "
                    "digits, and underscores only, starting with a letter")
        return None

    def _handle_forge(self, args: Dict[str, Any]) -> str:
        """Create, judge, test, and persist a new forged tool."""
        name = args.get("name", "").strip()
        description = args.get("description", "").strip()
        params_schema = args.get("params_schema", {})
        python_code = args.get("python_code", "").strip()
        test_args = args.get("test_args", _DEFAULT_TEST_ARGS)

        # Validate name
        name_err = self._validate_name(name)
        if name_err:
            return json.dumps({"error": name_err})

        if not description or not python_code:
            return json.dumps({
                "error": "description and python_code are required",
            })

        if not isinstance(params_schema, dict):
            return json.dumps({"error": "params_schema must be a JSON object"})

        # Validate params_schema has required structure
        if params_schema.get("type") != "object":
            return json.dumps({
                "error": "params_schema must have 'type': 'object'",
            })

        # Check for name collision
        with self._lock:
            existing = self._store.get_by_name(name)
            if existing:
                return json.dumps({
                    "error": f"A forged tool named '{name}' already exists. "
                             f"Use a different name or delete the old one first.",
                    "existing_tool": {
                        "id": existing["id"],
                        "approved": existing.get("judge_approved", False),
                        "tested": existing.get("test_passed", False),
                    },
                })

        tool_id = uuid.uuid4().hex[:12]

        # Step 1: Static validation
        static_ok, static_msg = validate_code(python_code)
        if not static_ok:
            # Store the rejected tool for audit trail
            self._store.add({
                "id": tool_id,
                "name": name,
                "description": description,
                "params_schema": params_schema,
                "python_code": python_code,
                "judge_verdict": json.dumps({"static": static_msg}),
                "judge_approved": False,
                "test_passed": False,
                "created_at": time.time(),
                "session_id": self._session_id,
            })
            return json.dumps({
                "tool_id": tool_id,
                "status": "rejected",
                "stage": "static_validation",
                "reason": static_msg,
            })

        # Step 2: LLM judge (fail-closed if unavailable)
        verdict = judge_code(python_code, description, params_schema, self._llm)

        if not verdict["approved"]:
            self._store.add({
                "id": tool_id,
                "name": name,
                "description": description,
                "params_schema": params_schema,
                "python_code": python_code,
                "judge_verdict": json.dumps(verdict),
                "judge_approved": False,
                "test_passed": False,
                "created_at": time.time(),
                "session_id": self._session_id,
            })
            return json.dumps({
                "tool_id": tool_id,
                "status": "rejected",
                "stage": "judge",
                "verdict": verdict,
            })

        # Step 3: Sandbox test
        passed, stdout, stderr = run_code(python_code, test_args)

        if not passed:
            self._store.add({
                "id": tool_id,
                "name": name,
                "description": description,
                "params_schema": params_schema,
                "python_code": python_code,
                "judge_verdict": json.dumps(verdict),
                "judge_approved": True,
                "test_passed": False,
                "test_output": stderr or stdout,
                "created_at": time.time(),
                "session_id": self._session_id,
            })
            return json.dumps({
                "tool_id": tool_id,
                "status": "test_failed",
                "stage": "sandbox_test",
                "stdout": stdout[:1000],
                "stderr": stderr[:1000],
                "judge_verdict": verdict,
            })

        # Step 4: Persist (no exec in main process — code runs in sandbox)
        with self._lock:
            self._store.add({
                "id": tool_id,
                "name": name,
                "description": description,
                "params_schema": params_schema,
                "python_code": python_code,
                "judge_verdict": json.dumps(verdict),
                "judge_approved": True,
                "test_passed": True,
                "test_output": stdout[:500],
                "created_at": time.time(),
                "session_id": self._session_id,
            })

        logger.info("forge: tool '%s' forged successfully (id=%s)", name, tool_id)

        return json.dumps({
            "tool_id": tool_id,
            "status": "forged",
            "name": name,
            "description": description,
            "judge_verdict": verdict,
            "test_output": stdout[:500],
            "message": (
                f"Tool '{name}' is now available. Call it with forge_call "
                f"using tool_name='{name}'."
            ),
        })

    def _handle_call(self, args: Dict[str, Any]) -> str:
        """Call a previously forged tool — runs in sandbox subprocess."""
        tool_name = args.get("tool_name", "").strip()
        call_args = args.get("args", {})

        if not tool_name:
            return json.dumps({"error": "tool_name is required"})

        name_err = self._validate_name(tool_name)
        if name_err:
            return json.dumps({"error": f"Invalid tool name: {name_err}"})

        with self._lock:
            tool = self._store.get_by_name(tool_name)

        if tool is None:
            return json.dumps({
                "error": f"No forged tool named '{tool_name}'. "
                         f"Use forge_list to see available tools.",
            })

        if not tool.get("judge_approved") or not tool.get("test_passed"):
            return json.dumps({
                "error": f"Tool '{tool_name}' was not approved or tested.",
            })

        # Re-validate stored code before execution (safety: DB could be tampered)
        code = tool["python_code"]
        ok, msg = validate_code(code)
        if not ok:
            logger.error("forge: stored code for '%s' failed re-validation: %s",
                         tool_name, msg)
            return json.dumps({
                "error": f"Stored code for '{tool_name}' failed safety validation: {msg}",
            })

        # Run in sandbox — NEVER in the main process
        passed, stdout, stderr = run_code(code, call_args)

        if not passed:
            return json.dumps({
                "error": f"Tool execution failed: {stderr or stdout}",
            })

        # Update use count
        with self._lock:
            self._store.increment_use(tool["id"])
            # Check auto-promotion
            updated = self._store.get_by_name(tool_name)
            if updated and should_auto_promote(updated):
                path = promote_to_skill(updated, self._skills_dir)
                if path:
                    self._store.update(tool["id"], promoted=True, promoted_path=path)
                    logger.info("forge: auto-promoted '%s' to %s", tool_name, path)

        # Parse the sandbox output — it's the JSON result from execute()
        try:
            result = json.loads(stdout)
        except json.JSONDecodeError:
            result = {"raw_output": stdout}

        return json.dumps({"result": result}, default=str)

    def _handle_list(self, args: Dict[str, Any]) -> str:
        """List all forged tools."""
        with self._lock:
            tools = self._store.list_all()
        if not tools:
            return json.dumps({"tools": [], "total": 0, "message": "No forged tools yet."})

        return json.dumps({
            "tools": [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "id": t["id"],
                    "judge_approved": t.get("judge_approved", False),
                    "test_passed": t.get("test_passed", False),
                    "use_count": t.get("use_count", 0),
                    "promoted": t.get("promoted", False),
                    "created_at": t.get("created_at"),
                }
                for t in tools
            ],
            "total": len(tools),
        })

    def _handle_promote(self, args: Dict[str, Any]) -> str:
        """Manually promote a forged tool to a skill."""
        tool_name = args.get("tool_name", "").strip()
        if not tool_name:
            return json.dumps({"error": "tool_name is required"})

        name_err = self._validate_name(tool_name)
        if name_err:
            return json.dumps({"error": f"Invalid tool name: {name_err}"})

        with self._lock:
            tool = self._store.get_by_name(tool_name)

        if tool is None:
            return json.dumps({"error": f"No forged tool named '{tool_name}'"})

        if not tool.get("judge_approved") or not tool.get("test_passed"):
            return json.dumps({
                "error": "Tool must be judge-approved and test-passed to promote",
            })

        if tool.get("promoted"):
            return json.dumps({
                "message": f"Tool '{tool_name}' is already promoted",
                "path": tool.get("promoted_path"),
            })

        path = promote_to_skill(tool, self._skills_dir)
        if path is None:
            return json.dumps({"error": "Promotion failed — check logs"})

        with self._lock:
            self._store.update(tool["id"], promoted=True, promoted_path=path)
        return json.dumps({
            "status": "promoted",
            "tool_name": tool_name,
            "skill_path": path,
            "message": (
                f"Tool '{tool_name}' promoted to a skill. It will load "
                f"automatically in future sessions."
            ),
        })

    def reload_from_store(self) -> int:
        """Verify all approved+tested forged tools from the store.

        This does NOT load handlers (no exec). It just counts available
        tools so the agent knows what's available via forge_call.
        """
        with self._lock:
            tools = self._store.list_approved()
        count = 0
        for tool in tools:
            # Re-validate stored code
            ok, _ = validate_code(tool["python_code"])
            if ok:
                count += 1
                logger.info("forge: verified '%s' from store", tool["name"])
            else:
                logger.warning(
                    "forge: stored code for '%s' failed validation — skipped",
                    tool["name"],
                )
        return count