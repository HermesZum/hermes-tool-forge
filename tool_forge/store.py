"""SQLite persistence for forged tools.

Stores forged tool definitions so they survive /reset and can be promoted
to skills. Each tool record captures the name, description, JSON schema,
Python source, judge verdict, and usage statistics.
"""

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


_BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS forged_tools (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    description     TEXT NOT NULL,
    params_schema   TEXT NOT NULL,
    python_code     TEXT NOT NULL,
    toolset         TEXT NOT NULL DEFAULT 'forged',
    judge_verdict   TEXT,
    judge_approved  INTEGER DEFAULT 0,
    test_passed     INTEGER DEFAULT 0,
    test_output     TEXT,
    created_at      REAL NOT NULL,
    last_used_at    REAL,
    use_count       INTEGER DEFAULT 0,
    promoted        INTEGER DEFAULT 0,
    promoted_path   TEXT,
    session_id      TEXT,
    metadata        TEXT
);

CREATE INDEX IF NOT EXISTS idx_forged_name ON forged_tools(name);
CREATE INDEX IF NOT EXISTS idx_forged_promoted ON forged_tools(promoted);
"""


class ForgeStore:
    """Thread-safe SQLite store for forged tool definitions."""

    def __init__(self, db_path: str):
        self._db_path = str(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()

    def connect(self) -> None:
        """Open the database connection and create schema if needed."""
        with self._lock:
            if self._conn is not None:
                return
            path = Path(self._db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            try:
                conn.executescript(_BASE_SCHEMA)
                conn.commit()
            except Exception:
                conn.close()
                raise
            self._conn = conn
            logger.info("forge-store: connected (db=%s)", self._db_path)

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    @property
    def connected(self) -> bool:
        return self._conn is not None

    def add(self, tool: Dict[str, Any]) -> None:
        """Insert a new forged tool record."""
        with self._lock:
            if not self._conn:
                return
            self._conn.execute(
                """INSERT OR REPLACE INTO forged_tools
                   (id, name, description, params_schema, python_code, toolset,
                    judge_verdict, judge_approved, test_passed, test_output,
                    created_at, last_used_at, use_count, promoted, promoted_path,
                    session_id, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    tool["id"],
                    tool["name"],
                    tool["description"],
                    json.dumps(tool["params_schema"]),
                    tool["python_code"],
                    tool.get("toolset", "forged"),
                    tool.get("judge_verdict"),
                    int(tool.get("judge_approved", False)),
                    int(tool.get("test_passed", False)),
                    tool.get("test_output"),
                    tool.get("created_at", time.time()),
                    tool.get("last_used_at"),
                    tool.get("use_count", 0),
                    int(tool.get("promoted", False)),
                    tool.get("promoted_path"),
                    tool.get("session_id"),
                    json.dumps(tool.get("metadata", {})),
                ),
            )
            self._conn.commit()

    def update(self, tool_id: str, **fields) -> None:
        """Update specific fields on an existing tool."""
        with self._lock:
            if not self._conn:
                return
            allowed = {
                "judge_verdict", "judge_approved", "test_passed", "test_output",
                "last_used_at", "use_count", "promoted", "promoted_path",
                "description", "python_code", "params_schema", "metadata",
            }
            updates = {k: v for k, v in fields.items() if k in allowed}
            if not updates:
                return
            # Serialize JSON fields
            if "metadata" in updates and isinstance(updates["metadata"], dict):
                updates["metadata"] = json.dumps(updates["metadata"])
            if "params_schema" in updates and isinstance(updates["params_schema"], dict):
                updates["params_schema"] = json.dumps(updates["params_schema"])
            # Convert bools to ints
            for k in ("judge_approved", "test_passed", "promoted"):
                if k in updates:
                    updates[k] = int(bool(updates[k]))
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            values = list(updates.values()) + [tool_id]
            self._conn.execute(
                f"UPDATE forged_tools SET {set_clause} WHERE id = ?",
                values,
            )
            self._conn.commit()

    def get(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a tool by ID."""
        with self._lock:
            if not self._conn:
                return None
            row = self._conn.execute(
                "SELECT * FROM forged_tools WHERE id = ?", (tool_id,)
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Fetch a tool by name."""
        with self._lock:
            if not self._conn:
                return None
            row = self._conn.execute(
                "SELECT * FROM forged_tools WHERE name = ?", (name,)
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def list_all(self) -> List[Dict[str, Any]]:
        """List all forged tools, newest first."""
        with self._lock:
            if not self._conn:
                return []
            rows = self._conn.execute(
                "SELECT * FROM forged_tools ORDER BY created_at DESC"
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def list_approved(self) -> List[Dict[str, Any]]:
        """List all judge-approved, tested tools."""
        with self._lock:
            if not self._conn:
                return []
            rows = self._conn.execute(
                """SELECT * FROM forged_tools
                   WHERE judge_approved = 1 AND test_passed = 1
                   ORDER BY use_count DESC"""
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def increment_use(self, tool_id: str) -> None:
        """Increment use count and update last_used_at."""
        with self._lock:
            if not self._conn:
                return
            self._conn.execute(
                """UPDATE forged_tools
                   SET use_count = use_count + 1, last_used_at = ?
                   WHERE id = ?""",
                (time.time(), tool_id),
            )
            self._conn.commit()

    def remove(self, tool_id: str) -> bool:
        """Delete a tool by ID. Returns True if a row was deleted."""
        with self._lock:
            if not self._conn:
                return False
            cur = self._conn.execute(
                "DELETE FROM forged_tools WHERE id = ?", (tool_id,)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def count(self) -> int:
        with self._lock:
            if not self._conn:
                return 0
            return self._conn.execute(
                "SELECT COUNT(*) FROM forged_tools"
            ).fetchone()[0]

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        # Parse JSON fields
        if d.get("params_schema"):
            try:
                d["params_schema"] = json.loads(d["params_schema"])
            except (json.JSONDecodeError, TypeError):
                pass
        if d.get("metadata"):
            try:
                d["metadata"] = json.loads(d["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass
        # Convert int flags to bool
        for k in ("judge_approved", "test_passed", "promoted"):
            if k in d:
                d[k] = bool(d[k])
        return d