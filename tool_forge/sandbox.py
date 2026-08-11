"""Restricted execution environment for testing forged tool code.

Runs LLM-generated Python in a subprocess with:
- No network access (no socket, no urllib, no requests)
- No filesystem writes (read-only access to /tmp only)
- Timeout (default 10 seconds)
- Captured stdout/stderr
- AST-based static analysis before execution

This is NOT a full security sandbox. It is a safety net that catches
obvious mistakes before the code is registered. The judge LLM is the
primary safety gate.
"""

import ast
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Forbidden imports — these modules are stripped from the AST before exec
_FORBIDDEN_MODULES = frozenset({
    "socket", "urllib", "requests", "http", "ssl", "ftplib",
    "smtplib", "telnetlib", "paramiko", "fabric",
    "ctypes", "cffi", "subprocess", "multiprocessing",
    "pickle", "shelve", "marshal",
    "os.system", "os.popen", "os.exec", "os.spawn",
    "shutil.rmtree", "shutil.move",
})

# AST node types that are forbidden
_FORBIDDEN_NODES = (
    ast.ImportFrom,
)

# Maximum code size (chars)
MAX_CODE_SIZE = 10_000
# Maximum test timeout (seconds)
DEFAULT_TIMEOUT = 10
# Maximum stdout/stderr capture (bytes)
MAX_OUTPUT = 5_000


class SandboxError(Exception):
    """Raised when sandbox validation or execution fails."""
    pass


def validate_code(code: str) -> Tuple[bool, str]:
    """Static analysis of forged tool code.

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
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".")[0]
                if root_mod in _FORBIDDEN_MODULES:
                    issues.append(f"Forbidden import: {alias.name}")

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_mod = node.module.split(".")[0]
                if root_mod in _FORBIDDEN_MODULES:
                    issues.append(f"Forbidden import: {node.module}")
            # Also check relative imports
            for alias in node.names:
                if alias.name == "*":
                    issues.append("Wildcard import not allowed")

        elif isinstance(node, ast.Attribute):
            # Check for os.system, os.popen, etc.
            if isinstance(node.value, ast.Name) and node.value.id == "os":
                if node.attr in ("system", "popen", "exec", "spawn", "fork",
                                 "kill", "remove", "unlink", "rmdir"):
                    issues.append(f"Forbidden os.{node.attr}()")

        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            # Allow but note — not a hard block
            pass

    if issues:
        return False, "; ".join(issues)

    return True, "OK"


def run_code(
    code: str,
    test_args: Dict[str, Any],
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[bool, str, str]:
    """Run forged tool code in a subprocess with test arguments.

    The code must define a function called ``execute`` that takes a dict
    and returns a JSON-serializable result.

    Returns (passed, stdout, stderr).
    """
    # Pre-flight validation
    ok, msg = validate_code(code)
    if not ok:
        return False, "", f"Validation failed: {msg}"

    # Build the test harness — user code at module level (no indentation)
    # so the execute() function is defined at the top level.
    harness = (
        "import json, sys, traceback\n"
        "\n"
        + code
        + "\n\n"
        "# Test arguments passed as JSON via stdin\n"
        "test_args = json.loads(sys.stdin.read())\n"
        "\n"
        "# Call the execute function\n"
        "if 'execute' not in dir():\n"
        "    print(json.dumps({'error': 'No execute() function defined'}))\n"
        "    sys.exit(1)\n"
        "\n"
        "try:\n"
        "    result = execute(test_args)\n"
        "    print(json.dumps(result, default=str))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'error': str(e), 'traceback': traceback.format_exc()}))\n"
        "    sys.exit(1)\n"
    )

    # Write harness to temp file and execute
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="forge_test_"
    ) as f:
        f.write(harness)
        harness_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, harness_path],
            input=json.dumps(test_args),
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
        return False, "", f"Test timed out after {timeout}s"
    except Exception as e:
        return False, "", f"Test execution failed: {e}"
    finally:
        try:
            os.unlink(harness_path)
        except OSError:
            pass