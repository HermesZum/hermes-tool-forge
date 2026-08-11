"""Tests for the judge module — LLM safety review of forged tool code."""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tool_forge.judge import judge_code, _parse_verdict, _normalize_verdict


class TestJudgeCode:
    """Judge review of forged tool code — fail-closed when LLM unavailable."""

    def test_static_rejection_forbidden_import(self):
        code = "import socket\ndef execute(args):\n    return {}"
        verdict = judge_code(code, "Network tool", {}, llm=None)
        assert not verdict["approved"]
        assert verdict["method"] == "static"
        assert "socket" in verdict["risks"][0]

    def test_static_rejection_syntax_error(self):
        code = "def execute(args):\n    return {"
        verdict = judge_code(code, "Broken tool", {}, llm=None)
        assert not verdict["approved"]
        assert "Syntax" in verdict["risks"][0]

    def test_no_llm_fail_closed(self):
        """When LLM is unavailable, the judge must REJECT, not approve."""
        code = "def execute(args):\n    return {'result': args.get('x', 0) * 2}"
        verdict = judge_code(code, "Doubler", {"type": "object"}, llm=None)
        assert not verdict["approved"]
        assert verdict["confidence"] == 0.0
        assert verdict["method"] == "static_only"
        assert "unavailable" in verdict["summary"].lower() or "available" in verdict["summary"].lower()

    def test_llm_unavailable_fallback_rejects(self):
        """When LLM call fails, the judge must REJECT (fail-closed)."""
        code = "def execute(args):\n    return {}"
        # Pass an object that has no chat/complete methods — _call_llm returns None
        verdict = judge_code(code, "Empty tool", {}, llm=object())
        assert not verdict["approved"]
        assert verdict["method"] in ("static_fallback", "error_fallback", "static_only")


class TestParseVerdict:
    """Parsing of LLM judge responses."""

    def test_parse_clean_json(self):
        response = '{"approved": true, "confidence": 0.9, "risks": [], "recommendations": [], "summary": "Safe"}'
        verdict = _parse_verdict(response)
        assert verdict["approved"] is True
        assert verdict["confidence"] == 0.9
        assert verdict["summary"] == "Safe"

    def test_parse_markdown_fenced_json(self):
        response = '```json\n{"approved": false, "confidence": 0.8, "risks": ["network"], "recommendations": [], "summary": "Rejected"}\n```'
        verdict = _parse_verdict(response)
        assert verdict["approved"] is False
        assert "network" in verdict["risks"]

    def test_parse_json_with_extra_text(self):
        response = 'I reviewed the code.\n{"approved": true, "confidence": 0.7, "risks": [], "recommendations": [], "summary": "OK"}\nThat is my verdict.'
        verdict = _parse_verdict(response)
        assert verdict["approved"] is True

    def test_parse_approved_pattern_fallback(self):
        response = 'The code looks good. "approved": true. No issues found.'
        verdict = _parse_verdict(response)
        assert verdict["approved"] is True

    def test_parse_unparseable_rejects(self):
        verdict = _parse_verdict("This is not JSON at all.")
        assert verdict["approved"] is False
        assert "parse" in verdict["risks"][0].lower()

    def test_normalize_verdict_defaults(self):
        normalized = _normalize_verdict({"approved": 1})
        assert normalized["approved"] is True
        assert normalized["confidence"] == 0.5
        assert normalized["risks"] == []
        assert normalized["recommendations"] == []