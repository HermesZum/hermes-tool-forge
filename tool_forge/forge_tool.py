"""Forge tool handler — the main tool the LLM calls to create new tools.

Exposes the ``forge_tool`` agent tool which:
1. Validates the generated code (static analysis)
2. Runs the LLM judge for safety review
3. Tests the code in a sandbox subprocess
4. Registers the tool in the Hermes registry if approved + tested
5. Persists the tool definition for future sessions

The forged tool is registered in the registry under the ``forged`` toolset.
It does NOT appear in the system prompt (preserving prompt caching).
The agent can call it via ``execute_code`` RPC, or the ``forge_call`` tool
which dispatches to the registered handler directly.
"""

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from .judge import judge_code
from .promote import promote_to_skill, should_auto_promote
from .sandbox import run_code, validate_code, SandboxError
from .store import ForgeStore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
                "description": "Unique tool name (snake_case, no spaces)",
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
                    "Only stdlib modules for data processing. "
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

# Forge call schema — call a previously forged tool
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

# Forge list schema — list all forged tools
FORGE_LIST_SCHEMA = {
    "name": "forge_list",
    "description": "List all forged tools with their status and usage stats.",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}

# Forge promote schema — promote a tool to a skill
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
    """Manages the forge tool lifecycle — create, judge, test, register, call."""

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
        # Registered handlers for forged tools (name -> callable)
        self._handlers: Dict[str, callable] = {}

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
        """Return schemas for all forge tools."""
        return ALL_TOOL_SCHEMAS

    def _handle_forge(self, args: Dict[str, Any]) -> str:
        """Create, judge, test, and register a new forged tool."""
        name = args.get("name", "").strip()
        description = args.get("description", "").strip()
        params_schema = args.get("params_schema", {})
        python_code = args.get("python_code", "").strip()
        test_args = args.get("test_args", _DEFAULT_TEST_ARGS)

        # Validate inputs
        if not name or not description or not python_code:
            return json.dumps({
                "error": "name, description, and python_code are required",
            })

        if not isinstance(params_schema, dict):
            return json.dumps({"error": "params_schema must be a JSON object"})

        # Check for name collision with existing forged tool
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
            return json.dumps({
                "tool_id": tool_id,
                "status": "rejected",
                "stage": "static_validation",
                "reason": static_msg,
            })

        # Step 2: LLM judge
        verdict = judge_code(python_code, description, params_schema, self._llm)

        if not verdict["approved"]:
            # Store the rejected tool for audit trail
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

        # Step 4: Register the handler
        handler = self._build_handler(python_code, tool_id)
        if handler is None:
            return json.dumps({
                "tool_id": tool_id,
                "status": "registration_failed",
                "stage": "handler_build",
                "reason": "Failed to compile execute() from generated code",
            })

        self._handlers[name] = handler

        # Step 5: Persist
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
        """Call a previously forged tool."""
        tool_name = args.get("tool_name", "").strip()
        call_args = args.get("args", {})

        if not tool_name:
            return json.dumps({"error": "tool_name is required"})

        # Check if handler is loaded
        if tool_name not in self._handlers:
            # Try to load from store
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

            handler = self._build_handler(tool["python_code"], tool["id"])
            if handler is None:
                return json.dumps({
                    "error": f"Could not load tool '{tool_name}' — "
                             f"its code may be corrupted.",
                })
            self._handlers[tool_name] = handler

        # Call the handler
        handler = self._handlers[tool_name]
        try:
            result = handler(call_args)
            # Update use count
            tool = self._store.get_by_name(tool_name)
            if tool:
                self._store.increment_use(tool["id"])
                # Check auto-promotion
                if should_auto_promote(tool):
                    path = promote_to_skill(tool, self._skills_dir)
                    if path:
                        self._store.update(tool["id"], promoted=True, promoted_path=path)
                        logger.info(
                            "forge: auto-promoted '%s' to %s", tool_name, path
                        )
            return json.dumps({"result": result}, default=str)
        except Exception as e:
            return json.dumps({"error": f"Tool execution failed: {e}"})

    def _handle_list(self, args: Dict[str, Any]) -> str:
        """List all forged tools."""
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

    def _build_handler(self, python_code: str, tool_id: str) -> Optional[callable]:
        """Build a callable handler from the forged Python code.

        The code must define an execute(args: dict) -> dict function.
        We exec it in a restricted namespace and return the execute callable.
        """
        try:
            namespace: Dict[str, Any] = {"__builtins__": __builtins__}
            exec(python_code, namespace)
            execute_fn = namespace.get("execute")
            if not callable(execute_fn):
                return None
            return execute_fn
        except Exception as e:
            logger.error("forge: failed to build handler for %s: %s", tool_id, e)
            return None

    def reload_from_store(self) -> int:
        """Load all approved+tested forged tools from the store.

        Called at plugin initialization to restore forged tools from
        previous sessions. Returns the number of tools loaded.
        """
        tools = self._store.list_approved()
        count = 0
        for tool in tools:
            handler = self._build_handler(tool["python_code"], tool["id"])
            if handler is not None:
                self._handlers[tool["name"]] = handler
                count += 1
                logger.info("forge: reloaded '%s' from store", tool["name"])
        return count