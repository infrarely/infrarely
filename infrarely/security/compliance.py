"""
aos/compliance.py — Immutable Compliance Audit Log
═══════════════════════════════════════════════════════════════════════════════
SCALE GAP 4: Healthcare/finance-grade audit trail that cannot be altered.

Usage:
    import infrarely
    result = agent.run("Process claim #1234",
                       session_id="sess-abc",
                       user_id="dr-smith")
    # Every action is automatically logged

    entries = aos.compliance_log.trail(agent="claims-agent")
    aos.compliance_log.export("audit-2024.json")
    aos.compliance_log.export("audit-2024.csv", format="csv")

The compliance log is SEPARATE from security.AuditLog:
  - security.AuditLog: Tracks security screening (PII, injection, etc.)
  - compliance.ComplianceLog: Tracks ALL agent actions for regulatory compliance
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# ACTION TYPES
# ═══════════════════════════════════════════════════════════════════════════════


class ActionType(str, Enum):
    """Types of auditable actions."""

    AGENT_RUN = "agent_run"
    AGENT_DELEGATE = "agent_delegate"
    TOOL_CALL = "tool_call"
    LLM_CALL = "llm_call"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    SECURITY_SCREEN = "security_screen"
    SECURITY_BLOCK = "security_block"
    MEMORY_STORE = "memory_store"
    KNOWLEDGE_QUERY = "knowledge_query"
    SANDBOX_VIOLATION = "sandbox_violation"
    CONFIG_CHANGE = "config_change"
    AGENT_CREATED = "agent_created"
    AGENT_SHUTDOWN = "agent_shutdown"
    ERROR = "error"
    CUSTOM = "custom"


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLIANCE ENTRY
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ComplianceEntry:
    """
    An immutable (frozen) audit record.

    Fields:
        action      : What happened (ActionType)
        agent_name  : Which agent performed the action
        timestamp   : When (epoch seconds)
        details     : Action-specific metadata dict
        user_id     : Who initiated (for human-in-the-loop)
        session_id  : Session identifier
        trace_id    : Trace ID for correlation
        task_query  : The task/prompt that triggered this
        human_approved : Whether a human approved this action
        approver_id : ID of the human who approved
        hash        : SHA-256 chain hash for tamper detection
    """

    action: str
    agent_name: str = ""
    timestamp: float = 0.0
    details: str = ""  # JSON string (frozen→no dict)
    user_id: str = ""
    session_id: str = ""
    trace_id: str = ""
    task_query: str = ""
    human_approved: bool = False
    approver_id: str = ""
    hash: str = ""  # Chain hash for tamper detection

    @property
    def chain_hash(self) -> str:
        """Alias for hash field."""
        return self.hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp,
            "details": json.loads(self.details) if self.details else {},
            "user_id": self.user_id,
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "task_query": self.task_query,
            "human_approved": self.human_approved,
            "approver_id": self.approver_id,
            "hash": self.hash,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLIANCE AUDIT LOG
# ═══════════════════════════════════════════════════════════════════════════════


class ComplianceLog:
    """
    Immutable, append-only compliance audit log.

    Every agent action is recorded with a chain hash for tamper detection.
    Supports SQLite persistence and export to JSON/CSV.
    """

    def __init__(self, db_path: Optional[str] = None, enabled: bool = True):
        self._enabled = enabled
        self._entries: List[ComplianceEntry] = []
        self._lock = threading.Lock()
        self._last_hash = "0" * 64  # genesis hash
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

        if db_path:
            self._init_db(db_path)

    def _init_db(self, db_path: str) -> None:
        """Initialize SQLite persistence for compliance records."""
        os.makedirs(
            os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True
        )
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS compliance_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                agent_name TEXT,
                timestamp REAL NOT NULL,
                details TEXT,
                user_id TEXT,
                session_id TEXT,
                trace_id TEXT,
                task_query TEXT,
                human_approved INTEGER DEFAULT 0,
                approver_id TEXT,
                hash TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_compliance_agent ON compliance_log(agent_name);
            CREATE INDEX IF NOT EXISTS idx_compliance_action ON compliance_log(action);
            CREATE INDEX IF NOT EXISTS idx_compliance_timestamp ON compliance_log(timestamp);
            CREATE INDEX IF NOT EXISTS idx_compliance_user ON compliance_log(user_id);
            CREATE INDEX IF NOT EXISTS idx_compliance_session ON compliance_log(session_id);
        """
        )
        self._conn.commit()

    def _compute_hash(
        self, action: str, agent_name: str, timestamp: float, details: str
    ) -> str:
        """Compute chain hash: H(prev_hash + action + agent + timestamp + details)."""
        payload = f"{self._last_hash}|{action}|{agent_name}|{timestamp}|{details}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def record(
        self,
        action: str,
        agent_name: str = "",
        details: Optional[Dict[str, Any]] = None,
        user_id: str = "",
        session_id: str = "",
        trace_id: str = "",
        task_query: str = "",
        human_approved: bool = False,
        approver_id: str = "",
    ) -> ComplianceEntry:
        """
        Record an immutable compliance entry.

        Parameters
        ----------
        action : str
            Action type (use ActionType enum values).
        agent_name : str
            The agent that performed the action.
        details : dict, optional
            Action-specific metadata.
        user_id : str
            User who initiated the action.
        session_id : str
            Session identifier.
        trace_id : str
            Trace ID for correlation.
        task_query : str
            The task prompt that triggered this.
        human_approved : bool
            Whether a human approved this action.
        approver_id : str
            ID of the approver.

        Returns
        -------
        ComplianceEntry
            The recorded entry.
        """
        if not self._enabled:
            # Return a dummy entry
            return ComplianceEntry(action=action, agent_name=agent_name)

        ts = time.time()
        details_json = json.dumps(details or {}, default=str)

        with self._lock:
            chain_hash = self._compute_hash(action, agent_name, ts, details_json)

            entry = ComplianceEntry(
                action=action,
                agent_name=agent_name,
                timestamp=ts,
                details=details_json,
                user_id=user_id,
                session_id=session_id,
                trace_id=trace_id,
                task_query=task_query,
                human_approved=human_approved,
                approver_id=approver_id,
                hash=chain_hash,
            )

            self._entries.append(entry)
            self._last_hash = chain_hash

            if self._conn:
                self._conn.execute(
                    """INSERT INTO compliance_log
                       (action, agent_name, timestamp, details, user_id,
                        session_id, trace_id, task_query, human_approved,
                        approver_id, hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        action,
                        agent_name,
                        ts,
                        details_json,
                        user_id,
                        session_id,
                        trace_id,
                        task_query,
                        int(human_approved),
                        approver_id,
                        chain_hash,
                    ),
                )
                self._conn.commit()

        return entry

    def trail(
        self,
        agent: Optional[str] = None,
        action: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[ComplianceEntry]:
        """
        Query the compliance trail with filters.

        Parameters
        ----------
        agent : str, optional
            Filter by agent name.
        action : str, optional
            Filter by action type.
        user_id : str, optional
            Filter by user.
        session_id : str, optional
            Filter by session.
        since : float, optional
            Only entries after this timestamp.
        limit : int
            Maximum entries to return.
        """
        with self._lock:
            result = list(self._entries)

        if agent:
            result = [e for e in result if e.agent_name == agent]
        if action:
            result = [e for e in result if e.action == action]
        if user_id:
            result = [e for e in result if e.user_id == user_id]
        if session_id:
            result = [e for e in result if e.session_id == session_id]
        if since is not None:
            result = [e for e in result if e.timestamp >= since]

        return result[-limit:]

    def verify_integrity(self) -> bool:
        """
        Verify the chain hash integrity of the entire log.

        Returns True if no tampering detected, False otherwise.
        """
        with self._lock:
            prev_hash = "0" * 64
            for entry in self._entries:
                payload = (
                    f"{prev_hash}|{entry.action}|{entry.agent_name}"
                    f"|{entry.timestamp}|{entry.details}"
                )
                expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()
                if entry.hash != expected:
                    return False
                prev_hash = entry.hash
            return True

    def export(self, path: str, format: str = "json") -> str:
        """
        Export the compliance log to a file.

        Parameters
        ----------
        path : str
            Output file path.
        format : str
            "json" or "csv".

        Returns
        -------
        str
            The output path.
        """
        with self._lock:
            entries = [e.to_dict() for e in self._entries]

        if format == "json":
            with open(path, "w") as f:
                json.dump(entries, f, indent=2, default=str)
        elif format == "csv":
            if entries:
                with open(path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=entries[0].keys())
                    writer.writeheader()
                    for row in entries:
                        # Flatten details dict to string for CSV
                        row["details"] = json.dumps(row["details"])
                        writer.writerow(row)
        else:
            raise ValueError(f"Unknown export format: {format!r}. Use 'json' or 'csv'.")

        return path

    @property
    def count(self) -> int:
        """Total number of entries."""
        return len(self._entries)

    def verify_chain(self) -> bool:
        """Alias for verify_integrity()."""
        return self.verify_integrity()

    def get_trail(self, **kwargs) -> list:
        """Alias for trail() with keyword mapping."""
        if "agent_name" in kwargs:
            kwargs["agent"] = kwargs.pop("agent_name")
        return self.trail(**kwargs)

    def export_json(self, path: str) -> str:
        """Export as JSON. Alias for export(path, format='json')."""
        return self.export(path, format="json")

    def reset(self) -> None:
        """
        Reset the compliance log. USE ONLY IN TESTS.

        In production, compliance logs should never be cleared.
        """
        with self._lock:
            self._entries.clear()
            self._last_hash = "0" * 64
            if self._conn:
                self._conn.execute("DELETE FROM compliance_log")
                self._conn.commit()


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_compliance_log: Optional[ComplianceLog] = None


def get_compliance_log() -> ComplianceLog:
    """Get module-level ComplianceLog singleton."""
    global _compliance_log
    if _compliance_log is None:
        _compliance_log = ComplianceLog()
    return _compliance_log
