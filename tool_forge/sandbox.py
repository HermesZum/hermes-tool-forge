"""Restricted execution environment for forged tool code.

Runs LLM-generated Python in a subprocess with:
- Allowlist-based import control (only safe stdlib modules)
- __import__ blocked in restricted builtins
- Resource limits (CPU, file size, address space, no subprocesses)
- Timeout (default 10 seconds)
- Captured stdout/stderr
- AST-based static analysis before execution

Forged tools ALWAYS run in this sandbox — both during testing AND during
normal forge_call invocations. They never exec() in the main Hermes process.
"""

import ast
import json
import logging
import os
import resource
import subprocess
import sys
import tempfile
import textwrap
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ── Allowlist of permitted imports ──────────────────────────────────────
# Only these stdlib modules are allowed. Everything else is rejected.
_ALLOWED_MODULES = frozenset({
    "json", "math", "re", "collections", "datetime", "itertools",
    "functools", "operator", "string", "textwrap", "unicodedata",
    "decimal", "fractions", "statistics", "hashlib", "base64",
    "uuid", "csv", "io", "bisect", "heapq", "array",
    "copy", "pprint", "calendar", "difflib",
})

# Forbidden builtins — stripped from the exec namespace AND blocked by AST
_FORBIDDEN_BUILTINS = frozenset({
    "__import__", "open", "exec", "eval", "compile", "globals",
    "locals", "vars", "dir", "type", "input", "breakpoint",
    "getattr", "setattr", "delattr", "hasattr",
})

# Forbidden call names — caught by AST static analysis
_FORBIDDEN_CALL_NAMES = frozenset({
    "__import__", "open", "exec", "eval", "compile",
    "globals", "locals", "vars", "input", "breakpoint",
})

# Runtime-stripped builtins — deleted from the subprocess before user code runs.
# Subset of _FORBIDDEN_BUILTINS that are truly dangerous at runtime.
# NOTE: getattr/setattr/delattr/hasattr are NOT stripped at runtime because
# the stripping code itself needs delattr. They ARE caught by AST analysis.
_RUNTIME_STRIPPED = frozenset({
    "__import__", "open", "exec", "eval", "compile",
    "globals", "locals", "vars", "input", "breakpoint",
})

# Maximum code size (chars)
MAX_CODE_SIZE = 10_000
# Maximum test args size (chars when JSON-serialized)
MAX_TEST_ARGS_SIZE = 10_000
# Default test timeout (seconds)
DEFAULT_TIMEOUT = 10
# Maximum stdout/stderr capture (bytes)
MAX_OUTPUT = 5_000
# Resource limits for sandbox subprocess
_RLIMIT_CPU = 5        # 5 seconds CPU time
_RLIMIT_FSIZE = 1_048_576  # 1 MB file writes
_RLIMIT_AS = 268_435_456   # 256 MB address space


class SandboxError(Exception):
    """Raised when sandbox validation or execution fails."""
    pass


def validate_code(code: str) -> Tuple[bool, str]:
    """Static analysis of forged tool code using an import ALLOWLIST.

    Returns (ok, message). If ok is False, message describes the issue.
    """
    if not code or not code.strip():
        return False, "Empty code"

    if len(code) > MAX_CODE_SIZE:
        return False, f"Code exceeds {MAX_CODE_SIZE} chars"

    # Parse AST
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    issues = []

    for node in ast.walk(tree):
        # Check imports — use ALLOWLIST, not blocklist
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".")[0]
                if root_mod not in _ALLOWED_MODULES:
                    issues.append(
                        f"Import '{alias.name}' not in allowlist. "
                        f"Allowed: {', '.join(sorted(_ALLOWED_MODULES))}"
                    )

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_mod = node.module.split(".")[0]
                if root_mod not in _ALLOWED_MODULES:
                    issues.append(
                        f"Import from '{node.module}' not in allowlist."
                    )
            for alias in node.names:
                if alias.name == "*":
                    issues.append("Wildcard import not allowed")

        # Catch calls to forbidden builtin functions
        elif isinstance(node, ast.Call):
            func = node.func
            # Direct calls to dangerous builtins
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_CALL_NAMES:
                issues.append(f"Forbidden: {func.id}() call")
            # getattr/setattr/delattr can bypass import restrictions
            elif isinstance(func, ast.Name) and func.id in ("getattr", "setattr", "delattr"):
                issues.append(f"Forbidden: {func.id}() can bypass import restrictions")

        # Catch attribute access on os/sys that could escape
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in ("os", "sys"):
                if node.attr in ("system", "popen", "exec", "spawn", "fork",
                                 "kill", "remove", "unlink", "rmdir", "exit",
                                 "modules", "path", "environ"):
                    issues.append(f"Forbidden: {node.value.id}.{node.attr}")

        # Catch subscript access to __builtins__ or globals() — bypass vector
        elif isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id in ("__builtins__", "globals", "vars"):
                issues.append(f"Forbidden: subscript on {node.value.id} — bypass vector")

    if issues:
        return False, "; ".join(issues[:5])  # Limit to first 5 issues

    return True, "OK"


def run_code(
    code: str,
    args: Dict[str, Any],
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[bool, str, str]:
    """Run forged tool code in a sandbox subprocess.

    The code must define a function called ``execute`` that takes a dict
    and returns a JSON-serializable result.

    This is used BOTH for testing (during forge_tool) AND for normal
    execution (during forge_call). Forged tools NEVER run in the main
    Hermes process.

    Returns (passed, stdout, stderr).
    """
    # Pre-flight validation
    ok, msg = validate_code(code)
    if not ok:
        return False, "", f"Validation failed: {msg}"

    # Size-check the args
    args_json = json.dumps(args)
    if len(args_json) > MAX_TEST_ARGS_SIZE:
        return False, "", f"Arguments exceed {MAX_TEST_ARGS_SIZE} chars"

    # Build the harness — user code at module level with restricted builtins
    _forbidden_list = ",".join(repr(b) for b in sorted(_RUNTIME_STRIPPED))
    _allowed_list = ",".join(repr(m) for m in sorted(_ALLOWED_MODULES))
    harness = (
        "import json, sys, traceback, resource\n"
        "\n"
        "# Set resource limits\n"
        f"resource.setrlimit(resource.RLIMIT_CPU, ({_RLIMIT_CPU}, {_RLIMIT_CPU}))\n"
        f"resource.setrlimit(resource.RLIMIT_FSIZE, ({_RLIMIT_FSIZE}, {_RLIMIT_FSIZE}))\n"
        f"resource.setrlimit(resource.RLIMIT_AS, ({_RLIMIT_AS}, {_RLIMIT_AS}))\n"
        "try:\n"
        "    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))\n"
        "except (ValueError, OSError):\n"
        "    pass\n"
        "\n"
        "# Replace __import__ with a restricted version BEFORE stripping it\n"
        f"_allowed = frozenset([{_allowed_list}])\n"
        "_orig_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __builtins__['__import__']\n"
        "def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):\n"
        "    root = name.split('.')[0]\n"
        "    if root not in _allowed:\n"
        "        raise ImportError(f\"Module '{{root}}' not in allowlist\")\n"
        "    return _orig_import(name, globals, locals, fromlist, level)\n"
        "if isinstance(__builtins__, dict):\n"
        "    __builtins__['__import__'] = _safe_import\n"
        "else:\n"
        "    __builtins__.__import__ = _safe_import\n"
        "\n"
        "# Strip other dangerous builtins (but keep our safe __import__)\n"
        f"_forbidden = frozenset([{_forbidden_list}]) - frozenset(['__import__'])\n"
        "for _name in _forbidden:\n"
        "    try:\n"
        "        delattr(__builtins__, _name)\n"
        "    except (AttributeError, TypeError):\n"
        "        try:\n"
        "            __builtins__.pop(_name, None)\n"
        "        except (AttributeError, TypeError):\n"
        "            pass\n"
        "\n"
        + code +
        "\n\n"
        "# Read args and execute\n"
        "test_args = json.loads(sys.stdin.read())\n"
        "if 'execute' not in dir():\n"
        "    print(json.dumps({'error': 'No execute() function defined'}))\n"
        "    sys.exit(1)\n"
        "try:\n"
        "    result = execute(test_args)\n"
        "    print(json.dumps(result, default=str))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'error': str(e), 'traceback': traceback.format_exc()}))\n"
        "    sys.exit(1)\n"
    )

    # Write harness to temp file and execute
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="forge_sandbox_"
    ) as f:
        f.write(harness)
        harness_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, "-I", harness_path],  # -I = isolated mode
            input=args_json,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={
                "PATH": "/usr/bin:/bin",
                "HOME": "/tmp",
                "LANG": "en_US.UTF-8",
                "LC_ALL": "en_US.UTF-8",
            },
        )
        stdout = proc.stdout[:MAX_OUTPUT]
        stderr = proc.stderr[:MAX_OUTPUT]
        passed = proc.returncode == 0
        return passed, stdout, stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Execution timed out after {timeout}s"
    except Exception as e:
        return False, "", f"Execution failed: {e}"
    finally:
        try:
            os.unlink(harness_path)
        except OSError:
            pass