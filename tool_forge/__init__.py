"""Hermes Tool Forge Plugin

Runtime tool forging for Hermes Agent. Agents can write their own tools
mid-conversation, have them safety-reviewed by an LLM judge, tested in a
sandbox, and registered for use — all without breaking prompt caching.

Inspired by AgentOS (framerslab) runtime tool forging.

Plugin kind: standalone (provides tools + hooks).
Activation: add 'tool_forge' to plugins.enabled in config.yaml.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Resolve paths
_HERMES_HOME = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
_DB_PATH = os.path.join(_HERMES_HOME, "tool_forge", "forge.db")
_SKILLS_DIR = os.path.join(_HERMES_HOME, "skills")

# Module-level singleton
_handler: Optional[Any] = None


def register(ctx) -> None:
    """Plugin registration entry point — called by PluginManager."""
    global _handler

    from .forge_tool import ForgeHandler
    from .store import ForgeStore

    # Initialize the store
    store = ForgeStore(_DB_PATH)
    try:
        store.connect()
    except Exception as e:
        logger.error("tool-forge: failed to connect store: %s", e)
        return

    # Get the LLM facade for the judge
    try:
        llm = ctx.llm
    except Exception:
        llm = None
        logger.warning("tool-forge: LLM facade unavailable — judge will fail-closed (reject all)")

    # Create the handler
    handler = ForgeHandler(
        store=store,
        llm=llm,
        skills_dir=_SKILLS_DIR,
        session_id="",
    )

    # Reload forged tools from previous sessions
    loaded = handler.reload_from_store()
    if loaded:
        logger.info("tool-forge: reloaded %d forged tools from store", loaded)

    _handler = handler

    # Register the 4 forge tools
    for schema in handler.get_tool_schemas():
        try:
            ctx.register_tool(
                name=schema["name"],
                toolset="forge",
                schema=schema,
                handler=_make_dispatcher(schema["name"]),
                check_fn=lambda: True,
                description=schema.get("description", ""),
            )
            logger.info("tool-forge: registered tool '%s'", schema["name"])
        except Exception as e:
            logger.error("tool-forge: failed to register %s: %s", schema["name"], e)


def _make_dispatcher(tool_name: str):
    """Create a handler callable that dispatches to the ForgeHandler."""
    def _dispatch(args: Dict[str, Any], **kwargs) -> str:
        if _handler is None:
            return json.dumps({"error": "Tool forge not initialized"})
        return _handler.handle(tool_name, args)
    return _dispatch