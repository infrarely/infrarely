"""
aos/observability.py — Traces, Metrics, Health
═══════════════════════════════════════════════════════════════════════════════
Philosophy 5: Observable by default. Zero setup required.

Every agent execution produces a trace. Every tool call is logged.
Developers can call result.explain() at any time.
Metrics are always collecting: LLM bypass rate, failure rate, latency.

  agent.state                     → current cognitive state
  agent.health()                  → full health report
  agent.get_trace(trace_id)       → complete execution trace
  infrarely.metrics.llm_bypass_rate()   → % tasks solved without LLM
  infrarely.metrics.failure_rate()      → % tasks that failed
  infrarely.metrics.avg_task_duration() → average execution time
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# TRACE DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TraceStep:
    """One step in an execution trace."""

    name: str
    tool: str = ""
    duration_ms: float = 0.0
    success: bool = True
    skipped: bool = False
    error: str = ""
    output_preview: str = ""  # first 200 chars of output


@dataclass
class TraceLLMCall:
    """Record of an LLM invocation."""

    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: float = 0.0
    reason: str = ""  # why LLM was called


@dataclass
class TraceKnowledgeQuery:
    """Record of a knowledge query."""

    query: str = ""
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)
    decision: str = ""  # bypass_llm | ground_llm | low_confidence
    duration_ms: float = 0.0


@dataclass
class TraceStateTransition:
    """Record of a state machine transition."""

    from_state: str = ""
    to_state: str = ""
    reason: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class TraceToolCall:
    """Record of a tool invocation."""

    tool_name: str = ""
    inputs: Dict[str, Any] = field(default_factory=dict)
    output_preview: str = ""
    duration_ms: float = 0.0
    success: bool = True
    cached: bool = False


@dataclass
class ExecutionTrace:
    """
    Complete execution trace for one agent.run() call.
    Retrieved via agent.get_trace(trace_id).
    """

    trace_id: str = field(default_factory=lambda: f"trace_{uuid.uuid4().hex[:12]}")
    agent_name: str = ""
    goal: str = ""
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: str = ""
    duration_ms: float = 0.0
    success: bool = True

    steps: List[TraceStep] = field(default_factory=list)
    llm_calls: List[TraceLLMCall] = field(default_factory=list)
    knowledge_queries: List[TraceKnowledgeQuery] = field(default_factory=list)
    state_transitions: List[TraceStateTransition] = field(default_factory=list)
    tool_calls: List[TraceToolCall] = field(default_factory=list)

    errors: List[str] = field(default_factory=list)
    output_preview: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "agent_name": self.agent_name,
            "goal": self.goal,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "steps": [asdict(s) for s in self.steps],
            "llm_calls": [asdict(c) for c in self.llm_calls],
            "knowledge_queries": [asdict(q) for q in self.knowledge_queries],
            "state_transitions": [asdict(t) for t in self.state_transitions],
            "tool_calls": [asdict(t) for t in self.tool_calls],
            "errors": self.errors,
            "output_preview": self.output_preview,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TRACE STORE — persists traces to SQLite
# ═══════════════════════════════════════════════════════════════════════════════


class TraceStore:
    """Stores and retrieves execution traces."""

    def __init__(self, db_path: str = "./aos_traces.db"):
        self._db_path = db_path
        os.makedirs(
            os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True
        )
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        self._init_tables()

    def _init_tables(self):
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS traces (
                trace_id    TEXT PRIMARY KEY,
                agent_name  TEXT NOT NULL,
                goal        TEXT NOT NULL,
                started_at  TEXT NOT NULL,
                completed_at TEXT,
                duration_ms REAL DEFAULT 0,
                success     INTEGER DEFAULT 1,
                data        TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_traces_agent
                ON traces(agent_name, started_at);
        """
        )
        self._conn.commit()

    def save(self, trace: ExecutionTrace) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO traces
                    (trace_id, agent_name, goal, started_at, completed_at,
                     duration_ms, success, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    trace.trace_id,
                    trace.agent_name,
                    trace.goal,
                    trace.started_at,
                    trace.completed_at,
                    trace.duration_ms,
                    int(trace.success),
                    json.dumps(trace.to_dict()),
                ),
            )
            self._conn.commit()

    def get(self, trace_id: str) -> Optional[ExecutionTrace]:
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM traces WHERE trace_id=?", (trace_id,)
            ).fetchone()
            if not row:
                return None
            data = json.loads(row[0])
            return self._dict_to_trace(data)

    def list_recent(self, agent_name: str = "", limit: int = 20) -> list:
        with self._lock:
            if agent_name:
                rows = self._conn.execute(
                    """
                    SELECT trace_id, agent_name, goal, started_at, duration_ms, success
                    FROM traces WHERE agent_name=? ORDER BY started_at DESC LIMIT ?
                """,
                    (agent_name, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT trace_id, agent_name, goal, started_at, duration_ms, success
                    FROM traces ORDER BY started_at DESC LIMIT ?
                """,
                    (limit,),
                ).fetchall()
            from types import SimpleNamespace

            class _TraceRecord(SimpleNamespace):
                """SimpleNamespace that also supports dict-style access."""

                def __getitem__(self, key):
                    return getattr(self, key)

                def __contains__(self, key):
                    return hasattr(self, key)

                def get(self, key, default=None):
                    return getattr(self, key, default)

            return [
                _TraceRecord(
                    trace_id=r[0],
                    agent_name=r[1],
                    goal=r[2],
                    started_at=r[3],
                    duration_ms=r[4],
                    success=bool(r[5]),
                )
                for r in rows
            ]

    @staticmethod
    def _dict_to_trace(d: Dict) -> ExecutionTrace:
        trace = ExecutionTrace(
            trace_id=d.get("trace_id", ""),
            agent_name=d.get("agent_name", ""),
            goal=d.get("goal", ""),
            started_at=d.get("started_at", ""),
            completed_at=d.get("completed_at", ""),
            duration_ms=d.get("duration_ms", 0),
            success=d.get("success", True),
            errors=d.get("errors", []),
            output_preview=d.get("output_preview", ""),
        )
        for s in d.get("steps", []):
            trace.steps.append(TraceStep(**s))
        for c in d.get("llm_calls", []):
            trace.llm_calls.append(TraceLLMCall(**c))
        for q in d.get("knowledge_queries", []):
            trace.knowledge_queries.append(TraceKnowledgeQuery(**q))
        for t in d.get("state_transitions", []):
            trace.state_transitions.append(TraceStateTransition(**t))
        for t in d.get("tool_calls", []):
            trace.tool_calls.append(TraceToolCall(**t))
        return trace

    def close(self):
        self._conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH REPORT
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class HealthReport:
    """Full health report for an agent."""

    agent_name: str = ""
    state: str = "IDLE"
    uptime_seconds: float = 0.0
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    avg_duration_ms: float = 0.0
    memory_entries: int = 0
    tools_registered: int = 0
    llm_calls_total: int = 0
    knowledge_queries_total: int = 0
    circuit_breakers_open: int = 0
    last_error: str = ""

    def __str__(self) -> str:
        lines = [
            f"Agent: {self.agent_name}",
            f"State: {self.state}",
            f"Uptime: {self.uptime_seconds:.0f}s",
            f"Tasks: {self.successful_tasks}/{self.total_tasks} successful",
            f"Avg Duration: {self.avg_duration_ms:.0f}ms",
            f"Tools: {self.tools_registered}",
            f"LLM Calls: {self.llm_calls_total}",
        ]
        if self.last_error:
            lines.append(f"Last Error: {self.last_error}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS COLLECTOR — system-wide metrics
# ═══════════════════════════════════════════════════════════════════════════════


class MetricsCollector:
    """
    System-wide metrics. Singleton per process.
    Accessed via infrarely.metrics.
    """

    _instance: Optional["MetricsCollector"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "MetricsCollector":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._total_tasks = 0
                cls._instance._successful_tasks = 0
                cls._instance._failed_tasks = 0
                cls._instance._llm_calls = 0
                cls._instance._llm_bypassed = 0
                cls._instance._total_duration_ms = 0.0
                cls._instance._recoveries = 0
                cls._instance._low_confidence_count = 0
                cls._instance._tool_metrics: Dict[str, Dict[str, Any]] = defaultdict(
                    lambda: {"calls": 0, "errors": 0, "total_ms": 0.0}
                )
                cls._instance._start_time = time.time()
            return cls._instance

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_task(
        self,
        success: bool,
        used_llm: bool,
        duration_ms: float,
        recovered: bool = False,
        low_confidence: bool = False,
    ):
        with self._lock:
            self._total_tasks += 1
            self._total_duration_ms += duration_ms
            if success:
                self._successful_tasks += 1
            else:
                self._failed_tasks += 1
            if used_llm:
                self._llm_calls += 1
            else:
                self._llm_bypassed += 1
            if recovered:
                self._recoveries += 1
            if low_confidence:
                self._low_confidence_count += 1

    def record_tool_call(self, tool_name: str, duration_ms: float, success: bool):
        with self._lock:
            m = self._tool_metrics[tool_name]
            m["calls"] += 1
            m["total_ms"] += duration_ms
            if not success:
                m["errors"] += 1

    # ── Queries ───────────────────────────────────────────────────────────────

    def llm_bypass_rate(self) -> float:
        """% of tasks solved without any LLM call."""
        total = self._llm_calls + self._llm_bypassed
        return (self._llm_bypassed / total * 100) if total > 0 else 0.0

    def hallucination_risk(self) -> float:
        """% of responses with low confidence (proxy for hallucination)."""
        return (
            (self._low_confidence_count / self._total_tasks * 100)
            if self._total_tasks > 0
            else 0.0
        )

    def avg_task_duration(self) -> float:
        """Average task execution time in ms."""
        return (
            self._total_duration_ms / self._total_tasks
            if self._total_tasks > 0
            else 0.0
        )

    def failure_rate(self) -> float:
        """% of tasks that failed."""
        return (
            (self._failed_tasks / self._total_tasks * 100)
            if self._total_tasks > 0
            else 0.0
        )

    def recovery_rate(self) -> float:
        """% of failures that auto-recovered."""
        return (
            (self._recoveries / self._failed_tasks * 100)
            if self._failed_tasks > 0
            else 0.0
        )

    def total_tasks(self) -> int:
        return self._total_tasks

    def summary(self) -> Dict[str, Any]:
        """Get a summary dict of all key metrics."""
        total = self._total_tasks or 1  # avoid division by zero
        return {
            "total_tasks": self._total_tasks,
            "successful_tasks": self._successful_tasks,
            "failed_tasks": self._failed_tasks,
            "success_rate": (
                round(self._successful_tasks / total, 3) if self._total_tasks else 0.0
            ),
            "failure_rate": round(self.failure_rate(), 3),
            "llm_calls": self._llm_calls,
            "llm_bypass_rate": round(self.llm_bypass_rate(), 3),
            "avg_duration_ms": round(self.avg_task_duration(), 1),
            "hallucination_risk": round(self.hallucination_risk(), 3),
            "recovery_rate": round(self.recovery_rate(), 3),
            "uptime_seconds": round(self.uptime_seconds(), 1),
            "total_tokens": 0,  # placeholder — future token tracking
        }

    def tool_stats(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._tool_metrics)

    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    def export(self, format: str = "json", port: int = 9090) -> Any:
        """Export metrics in various formats."""
        data = {
            "total_tasks": self._total_tasks,
            "successful_tasks": self._successful_tasks,
            "failed_tasks": self._failed_tasks,
            "llm_bypass_rate": self.llm_bypass_rate(),
            "hallucination_risk": self.hallucination_risk(),
            "avg_task_duration_ms": self.avg_task_duration(),
            "failure_rate": self.failure_rate(),
            "recovery_rate": self.recovery_rate(),
            "uptime_seconds": self.uptime_seconds(),
            "tool_stats": dict(self._tool_metrics),
        }
        if format == "json":
            return data
        elif format == "prometheus":
            lines = []
            for key, val in data.items():
                if isinstance(val, (int, float)):
                    lines.append(f"aos_{key} {val}")
            return "\n".join(lines)
        return data

    def reset(self):
        """Reset all metrics (for testing)."""
        with self._lock:
            self._total_tasks = 0
            self._successful_tasks = 0
            self._failed_tasks = 0
            self._llm_calls = 0
            self._llm_bypassed = 0
            self._total_duration_ms = 0.0
            self._recoveries = 0
            self._low_confidence_count = 0
            self._tool_metrics.clear()
            self._start_time = time.time()


# ─── Logger ───────────────────────────────────────────────────────────────────


class _Logger:
    """Structured logger with in-memory buffer AND persistent file logging.

    Logs are written to:
      logs/aos.log          — main log (rotated at 5 MB, 3 backups)
      logs/errors/error.log — ERROR-only log (rotated at 2 MB, 3 backups)

    The log directory defaults to ``./logs`` relative to the working directory.
    Override with ``infrarely.configure(log_dir="/custom/path")``.
    """

    _instance: Optional["_Logger"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "_Logger":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._level = "INFO"
                cls._instance._entries: List[Dict[str, Any]] = []
                cls._instance._max_entries = 10_000
                cls._instance._log_dir: Optional[str] = None
                cls._instance._file_handle: Optional[Any] = None
                cls._instance._error_handle: Optional[Any] = None
                cls._instance._max_file_bytes = 5 * 1024 * 1024  # 5 MB
                cls._instance._max_error_bytes = 2 * 1024 * 1024  # 2 MB
                cls._instance._backup_count = 3
                cls._instance._file_enabled = False
            return cls._instance

    _LEVELS = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}

    # ── File logging setup ────────────────────────────────────────────────────

    def enable_file_logging(self, log_dir: Optional[str] = None):
        """Enable persistent file logging with rotation.

        Args:
            log_dir: Directory for log files. Defaults to ``./logs``.
        """
        with self._lock:
            self._log_dir = log_dir or os.path.join(os.getcwd(), "logs")
            os.makedirs(self._log_dir, exist_ok=True)
            error_dir = os.path.join(self._log_dir, "errors")
            os.makedirs(error_dir, exist_ok=True)

            main_path = os.path.join(self._log_dir, "aos.log")
            error_path = os.path.join(error_dir, "error.log")

            self._file_handle = open(main_path, "a", encoding="utf-8")
            self._error_handle = open(error_path, "a", encoding="utf-8")
            self._file_enabled = True

    def _rotate_if_needed(self, handle, path: str, max_bytes: int):
        """Rotate a log file if it exceeds max_bytes."""
        try:
            if handle and not handle.closed and handle.tell() > max_bytes:
                handle.flush()
                handle.close()
                # Rotate: aos.log.3 → delete, .2 → .3, .1 → .2, current → .1
                for i in range(self._backup_count, 0, -1):
                    src = f"{path}.{i}" if i > 1 else path
                    dst = f"{path}.{i}"
                    if i == self._backup_count and os.path.exists(dst):
                        os.remove(dst)
                    if i > 1:
                        src = f"{path}.{i - 1}"
                    if os.path.exists(src):
                        os.rename(src, dst)
                # Reopen
                return open(path, "a", encoding="utf-8")
        except OSError:
            pass
        return handle

    def _write_to_file(self, entry: Dict[str, Any]):
        """Write a log entry to disk (main log + error log if applicable)."""
        if not self._file_enabled:
            return

        line = self._format_line(entry)

        try:
            # Main log
            if self._file_handle and not self._file_handle.closed:
                self._file_handle.write(line)
                self._file_handle.flush()
                main_path = os.path.join(self._log_dir, "aos.log")
                self._file_handle = self._rotate_if_needed(
                    self._file_handle, main_path, self._max_file_bytes
                )

            # Error log (ERROR level only)
            if (
                entry["level"] == "ERROR"
                and self._error_handle
                and not self._error_handle.closed
            ):
                self._error_handle.write(line)
                self._error_handle.flush()
                error_path = os.path.join(self._log_dir, "errors", "error.log")
                self._error_handle = self._rotate_if_needed(
                    self._error_handle, error_path, self._max_error_bytes
                )
        except OSError:
            pass  # Never crash the app due to logging

    @staticmethod
    def _format_line(entry: Dict[str, Any]) -> str:
        """Format a log entry as a single line: [TIMESTAMP] LEVEL | message | key=val ..."""
        ts = entry.get("ts", "")
        level = entry.get("level", "INFO")
        msg = entry.get("message", "")
        extras = {k: v for k, v in entry.items() if k not in ("ts", "level", "message")}
        extra_str = " | ".join(f"{k}={v}" for k, v in extras.items()) if extras else ""
        parts = [f"[{ts}] {level:7s} | {msg}"]
        if extra_str:
            parts.append(f" | {extra_str}")
        parts.append("\n")
        return "".join(parts)

    # ── Core API ──────────────────────────────────────────────────────────────

    def set_level(self, level: str):
        self._level = level.upper()

    def _should_log(self, level: str) -> bool:
        return self._LEVELS.get(level.upper(), 0) >= self._LEVELS.get(self._level, 1)

    def _log(self, level: str, message: str, **kwargs):
        if not self._should_log(level):
            return
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level.upper(),
            "message": message,
            **kwargs,
        }
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries // 2 :]
            self._write_to_file(entry)

    def debug(self, msg: str, **kw):
        self._log("DEBUG", msg, **kw)

    def info(self, msg: str, **kw):
        self._log("INFO", msg, **kw)

    def warning(self, msg: str, **kw):
        self._log("WARNING", msg, **kw)

    def error(self, msg: str, **kw):
        self._log("ERROR", msg, **kw)

    def get_entries(self, level: Optional[str] = None, limit: int = 100) -> List[Dict]:
        with self._lock:
            entries = list(self._entries)
        if level:
            entries = [e for e in entries if e["level"] == level.upper()]
        return entries[-limit:]

    def clear(self):
        with self._lock:
            self._entries.clear()

    @property
    def log_dir(self) -> Optional[str]:
        """Return the active log directory, or None if file logging is off."""
        return self._log_dir if self._file_enabled else None

    def close(self):
        """Flush and close file handles."""
        with self._lock:
            for h in (self._file_handle, self._error_handle):
                if h and not h.closed:
                    try:
                        h.flush()
                        h.close()
                    except OSError:
                        pass
            self._file_enabled = False


# ── Module-level singletons ──────────────────────────────────────────────────

_metrics = MetricsCollector()
_logger = _Logger()


def get_metrics() -> MetricsCollector:
    return _metrics


def get_logger() -> _Logger:
    return _logger
