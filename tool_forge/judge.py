"""LLM judge — second-opinion safety review of forged tool code.

Uses the plugin's host-owned LLM facade (ctx.llm) to run a structured
safety review of LLM-generated code before it is tested or registered.

The judge prompt asks the model to:
1. Identify any security risks (network access, file writes, code injection)
2. Identify any logic errors or edge cases
3. Approve or reject with a structured verdict

The verdict is parsed from the model's response and returned as a dict.
If the LLM is unavailable (no ctx.llm, no API key), the judge falls back
to static analysis only (sandbox.validate_code) and issues a conditional
approval with reduced confidence.
"""

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

from .sandbox import validate_code

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

JUDGE_SYSTEM_PROMPT = """You are a code safety judge. You review Python tool code
that an AI agent wants to register as a runtime tool. Your job is to catch
security risks and logic errors before the code is tested.

Respond with a JSON object (no markdown, no explanation outside JSON):
{
  "approved": true/false,
  "confidence": 0.0-1.0,
  "risks": ["list of identified risks, empty if none"],
  "recommendations": ["list of suggestions, empty if none"],
  "summary": "one-line verdict"
}

Reject if the code:
- Makes network connections (socket, urllib, requests, http)
- Writes to the filesystem outside an explicit temp dir
- Uses subprocess, os.system, or eval/exec
- Imports pickle, marshal, or ctypes
- Has obvious infinite loops or resource exhaustion
- Attempts to access environment variables for secrets

Approve if the code:
- Only reads from its input arguments
- Returns a JSON-serializable result
- Uses only stdlib modules for data processing
- Has clear, bounded logic with no side effects
- Defines an execute(args: dict) -> dict function"""


def judge_code(
    code: str,
    description: str,
    params_schema: Dict[str, Any],
    llm: Any = None,
) -> Dict[str, Any]:
    """Run the judge review on forged tool code.

    Args:
        code: The Python source code to review.
        description: What the tool claims to do.
        params_schema: The JSON schema for the tool's parameters.
        llm: The plugin's LLM facade (ctx.llm). If None, falls back to
             static analysis only.

    Returns:
        Dict with keys: approved (bool), confidence (float), risks (list),
        recommendations (list), summary (str), method (str).
    """
    # Always run static analysis first
    static_ok, static_msg = validate_code(code)

    if not static_ok:
        return {
            "approved": False,
            "confidence": 1.0,
            "risks": [static_msg],
            "recommendations": ["Fix the identified issue before resubmitting"],
            "summary": f"Static analysis rejected: {static_msg}",
            "method": "static",
        }

    # If no LLM available, FAIL CLOSED — do not approve without judge review
    if llm is None:
        return {
            "approved": False,
            "confidence": 0.0,
            "risks": ["LLM judge unavailable — cannot review code safety"],
            "recommendations": ["Provide LLM access to the plugin or review code manually"],
            "summary": "Rejected (no LLM judge available — fail-closed)",
            "method": "static_only",
        }

    # Run LLM judge
    try:
        user_prompt = (
            f"Tool description: {description}\n\n"
            f"Parameters schema: {json.dumps(params_schema, indent=2)}\n\n"
            f"Python code:\n```python\n{code}\n```\n\n"
            f"Review this code and respond with the JSON verdict."
        )

        response = _call_llm(llm, user_prompt)
        if response is None:
            # LLM call failed — FAIL CLOSED
            return {
                "approved": False,
                "confidence": 0.0,
                "risks": ["LLM judge call returned no response"],
                "recommendations": ["Check LLM configuration and retry"],
                "summary": "Rejected (LLM judge call failed — fail-closed)",
                "method": "static_fallback",
            }

        verdict = _parse_verdict(response)
        verdict["method"] = "llm"
        return verdict

    except Exception as e:
        logger.error("forge-judge: LLM review failed: %s", e, exc_info=True)
        # FAIL CLOSED on any exception
        return {
            "approved": False,
            "confidence": 0.0,
            "risks": [f"LLM judge error: {e}"],
            "recommendations": ["Check LLM access and retry, or review manually"],
            "summary": f"Rejected (judge error: {e} — fail-closed)",
            "method": "error_fallback",
        }


def _call_llm(llm: Any, prompt: str) -> Optional[str]:
    """Call the plugin LLM facade with the judge prompt.

    The PluginLlm facade supports both chat() and structured completion
    patterns. We try chat first, then fall back to complete.
    """
    # Try chat() method first (most common)
    try:
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        result = llm.chat(messages)
        if isinstance(result, str):
            return result
        if isinstance(result, dict) and "content" in result:
            return result["content"]
        if hasattr(result, "content"):
            return result.content
        return str(result)
    except (AttributeError, TypeError):
        pass

    # Try complete() method
    try:
        full_prompt = f"{JUDGE_SYSTEM_PROMPT}\n\n{prompt}"
        result = llm.complete(full_prompt)
        if isinstance(result, str):
            return result
        return str(result)
    except (AttributeError, TypeError):
        return None


def _parse_verdict(response: str) -> Dict[str, Any]:
    """Parse the LLM judge response into a verdict dict.

    The judge is instructed to return JSON, but we handle common
    formatting issues (markdown code fences, extra text).
    """
    # Strip markdown code fences if present
    text = response.strip()
    if text.startswith("```"):
        # Remove opening fence
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    # Try to extract JSON from the response
    # First try direct parse
    try:
        data = json.loads(text)
        return _normalize_verdict(data)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    json_match = re.search(r'\{[^{}]*"(?:approved|confidence)"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return _normalize_verdict(data)
        except json.JSONDecodeError:
            pass

    # Fallback: look for approved: true/false pattern
    approved_match = re.search(r'"approved"\s*:\s*(true|false)', text, re.IGNORECASE)
    if approved_match:
        approved = approved_match.group(1).lower() == "true"
        return {
            "approved": approved,
            "confidence": 0.6,
            "risks": ["Judge response was not valid JSON"],
            "recommendations": [],
            "summary": f"Parsed from non-JSON response (approved={approved})",
        }

    # Could not parse — be conservative and reject
    return {
        "approved": False,
        "confidence": 0.3,
        "risks": ["Could not parse judge response"],
        "recommendations": ["Re-run the forge or review manually"],
        "summary": "Judge response unparseable — rejected for safety",
    }


def _normalize_verdict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a parsed verdict dict to the expected shape."""
    return {
        "approved": bool(data.get("approved", False)),
        "confidence": float(data.get("confidence", 0.5)),
        "risks": list(data.get("risks", [])),
        "recommendations": list(data.get("recommendations", [])),
        "summary": str(data.get("summary", "")),
    }