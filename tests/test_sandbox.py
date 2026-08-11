"""Tests for the sandbox module — static validation and code execution."""

import os
import sys
import textwrap
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tool_forge.sandbox import validate_code, run_code, SandboxError, MAX_CODE_SIZE


def _code(src: str) -> str:
    """Dedent a code string so def execute() is at module level."""
    return textwrap.dedent(src).strip()


class TestValidateCode:
    """Static analysis of forged tool code."""

    def test_valid_simple_function(self):
        code = """
def execute(args):
    return {"result": args.get("x", 0) * 2}
"""
        ok, msg = validate_code(code)
        assert ok, f"Expected OK, got: {msg}"

    def test_empty_code(self):
        ok, msg = validate_code("")
        assert not ok
        assert "Empty" in msg

    def test_syntax_error(self):
        ok, msg = validate_code("def execute(args):\n    return {")
        assert not ok
        assert "Syntax error" in msg

    def test_forbidden_import_socket(self):
        ok, msg = validate_code("import socket\ndef execute(args):\n    return {}")
        assert not ok
        assert "socket" in msg

    def test_forbidden_import_requests(self):
        ok, msg = validate_code("import requests\ndef execute(args):\n    return {}")
        assert not ok
        assert "requests" in msg

    def test_forbidden_os_system(self):
        ok, msg = validate_code(
            "import os\ndef execute(args):\n    os.system('ls')\n    return {}"
        )
        assert not ok
        assert "os.system" in msg

    def test_forbidden_subprocess(self):
        ok, msg = validate_code("import subprocess\ndef execute(args):\n    return {}")
        assert not ok
        assert "subprocess" in msg

    def test_allowed_stdlib(self):
        code = """
import json
import math
import re
from collections import Counter

def execute(args):
    text = args.get("text", "")
    words = re.findall(r'\\w+', text.lower())
    counts = Counter(words)
    return {"word_count": len(words), "top": counts.most_common(5)}
"""
        ok, msg = validate_code(code)
        assert ok, f"Expected OK, got: {msg}"

    def test_code_too_large(self):
        code = "x = 0\n" * (MAX_CODE_SIZE // 5)
        ok, msg = validate_code(code)
        assert not ok
        assert "exceeds" in msg

    def test_wildcard_import_rejected(self):
        ok, msg = validate_code("from os import *\ndef execute(args):\n    return {}")
        assert not ok
        assert "Wildcard" in msg


class TestTestCode:
    """Sandbox execution of forged tool code."""

    def test_successful_execution(self):
        code = _code("""
            def execute(args):
                return {"doubled": args.get("x", 0) * 2}
        """)
        passed, stdout, stderr = run_code(code, {"x": 21})
        assert passed
        assert "42" in stdout

    def test_no_execute_function(self):
        code = _code("def not_execute(args):\n    return {}")
        passed, stdout, stderr = run_code(code, {})
        assert not passed
        assert "execute" in stdout or "execute" in stderr

    def test_runtime_error(self):
        code = _code("""
            def execute(args):
                return 1 / 0
        """)
        passed, stdout, stderr = run_code(code, {})
        assert not passed
        assert "error" in stdout.lower()

    def test_forbidden_code_rejected_before_exec(self):
        code = "import socket\ndef execute(args):\n    return {}"
        passed, stdout, stderr = run_code(code, {})
        assert not passed
        assert "Validation failed" in stderr

    def test_timeout(self):
        code = _code("""
            def execute(args):
                while True:
                    pass
                return {}
        """)
        passed, stdout, stderr = run_code(code, {}, timeout=2)
        assert not passed
        assert "timed out" in stderr.lower()

    def test_returns_complex_dict(self):
        code = _code("""
            def execute(args):
                items = args.get("items", [])
                return {
                    "count": len(items),
                    "first": items[0] if items else None,
                    "summary": f"Processed {len(items)} items",
                }
        """)
        passed, stdout, stderr = run_code(code, {"items": ["a", "b", "c"]})
        assert passed
        import json
        result = json.loads(stdout)
        assert result["count"] == 3
        assert result["first"] == "a"