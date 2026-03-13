"""
observability/logger.py  (v4 — full log system)
═══════════════════════════════════════════════════════════════════════════════
Three-stream logging architecture:

  Stream 1 — logs/agent.log
    Append-only JSON-lines master log. Every event from every component.
    Never truncated. Rotated by size (10 MB → agent.log.1, .2, .3).
    Machine-readable. Used by monitoring, audits, post-mortems.

  Stream 2 — logs/sessions/session_YYYYMMDD_HHMMSS_<id>.log
    Human-readable plain-text log for exactly ONE agent session.
    Created when the session starts, closed when it ends.
    Contains: session header, all events with aligned columns, session footer.
    Readable with `cat`, `less`, `tail -f`.

  Stream 3 — logs/errors/errors_YYYYMMDD.log
    ERROR and WARNING lines only, one per calendar day.
    Quick triage file — "what went wrong today?"

All three streams receive every log call.
No log call is ever lost (writes are flushed immediately).

Log levels (in priority order):
  DEBUG   dim — internal tracing, off in production
  INFO    bright green — normal operations
  WARNING yellow — degraded but recoverable
  ERROR   bold red — failure, needs attention
  TOOL    blue — tool execution events
  ROUTER  magenta — routing decisions
  MEMORY  yellow — memory layer operations
  LLM     cyan — LLM calls and token usage
  TRACE   white — execution trace events (Gap 10)
  METRIC  white — metric snapshots

Rotation:
  agent.log rotates at 10 MB → keeps 5 backups
  session logs: one per session, never rotated
  error log: one per calendar day
"""

from __future__ import annotations

import json
import os
import sys
import time
import threading
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional

try:
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

import infrarely.core.app_config as config

# ── directory setup ───────────────────────────────────────────────────────────
_LOG_DIR         = config.LOG_DIR
_SESSION_DIR     = os.path.join(_LOG_DIR, "sessions")
_TRACE_DIR       = getattr(config, "TRACE_DIR", None) or os.path.join(_LOG_DIR, "traces")
_ERROR_DIR       = os.path.join(_LOG_DIR, "errors")
_TERMINAL_LOG    = os.path.join(_TRACE_DIR, "logs.log")

for _d in (_LOG_DIR, _SESSION_DIR, _TRACE_DIR, _ERROR_DIR):
    os.makedirs(_d, exist_ok=True)

# ── master log — rotating JSON lines ─────────────────────────────────────────
_MASTER_LOG_PATH = config.LOG_FILE   # logs/agent.log
_master_handler  = RotatingFileHandler(
    _MASTER_LOG_PATH,
    maxBytes    = 10 * 1024 * 1024,   # 10 MB
    backupCount = 5,
    encoding    = "utf-8",
)

# ── error log — daily plain text ──────────────────────────────────────────────
def _error_log_path() -> str:
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    return os.path.join(_ERROR_DIR, f"errors_{date_str}.log")

# ── session log — per-session human-readable ──────────────────────────────────
_session_file:    Optional[Any]  = None
_session_path:    Optional[str]  = None
_session_id:      Optional[str]  = None
_session_start:   Optional[float] = None
_session_lock = threading.Lock()
_terminal_lock = threading.Lock()

# ── Rich console ──────────────────────────────────────────────────────────────
_console = Console(stderr=False) if RICH_AVAILABLE else None

# ── colour palette ────────────────────────────────────────────────────────────
_LEVEL_STYLE = {
    "DEBUG":   "dim cyan",
    "INFO":    "bright_green",
    "WARNING": "yellow",
    "ERROR":   "bold red",
    "TOOL":    "bright_blue",
    "ROUTER":  "bright_magenta",
    "MEMORY":  "bright_yellow",
    "LLM":     "bright_cyan",
    "TRACE":   "white",
    "METRIC":  "bright_white",
    "CAP":     "magenta",
}

_LEVEL_ICON = {
    "DEBUG":   "·",
    "INFO":    "✓",
    "WARNING": "⚠",
    "ERROR":   "✗",
    "TOOL":    "⚙",
    "ROUTER":  "→",
    "MEMORY":  "◉",
    "LLM":     "◈",
    "TRACE":   "◎",
    "METRIC":  "◆",
    "CAP":     "⬡",
}

# ── column widths for human-readable session log ──────────────────────────────
_COL_TS    = 23   # 2026-03-08T18:22:01.123
_COL_LVL   =  7   # WARNING
_COL_ICON  =  2   # ⚠
_LINE_W    = 120


# ═════════════════════════════════════════════════════════════════════════════
# Session management
# ═════════════════════════════════════════════════════════════════════════════

def start_session(session_id: str, student_id: str = "unknown") -> str:
    """
    Open a new per-session log file.
    Returns the path of the session log.
    Call this once at agent startup (from main.py).
    """
    global _session_file, _session_path, _session_id, _session_start

    with _session_lock:
        # close any previously open session
        if _session_file and not _session_file.closed:
            _session_file.flush()
            _session_file.close()

        _session_id    = session_id
        _session_start = time.monotonic()
        ts_str         = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        fname          = f"session_{ts_str}_{session_id[:8]}.log"
        _session_path  = os.path.join(_SESSION_DIR, fname)
        _session_file  = open(_session_path, "w", encoding="utf-8")

        _write_session_header(student_id)

    return _session_path


def end_session(app_cfg=None):
    """
    Write the session footer and close the session log file.
    Call this at agent shutdown (from main.py).
    """
    global _session_file

    with _session_lock:
        if _session_file and not _session_file.closed:
            _write_session_footer(app_cfg)
            _session_file.flush()
            _session_file.close()


def _write_session_header(student_id: str):
    now = datetime.now(timezone.utc).isoformat()
    lines = [
        "=" * _LINE_W,
        f"  STUDENT LIFE ASSISTANT — SESSION LOG",
        f"  Session ID : {_session_id}",
        f"  Student    : {student_id}",
        f"  Started    : {now}",
        f"  Log file   : {_session_path}",
        f"  Master log : {_MASTER_LOG_PATH}",
        "=" * _LINE_W,
        f"{'TIMESTAMP':<{_COL_TS}}  {'LEVEL':<{_COL_LVL}}  {'MESSAGE'}",
        "-" * _LINE_W,
    ]
    _session_file.write("\n".join(lines) + "\n")
    _session_file.flush()


def _write_session_footer(app_cfg=None):
    elapsed = time.monotonic() - (_session_start or time.monotonic())
    now     = datetime.now(timezone.utc).isoformat()
    lines = [
        "",
        "-" * _LINE_W,
        f"  SESSION SUMMARY",
        f"  Ended      : {now}",
        f"  Duration   : {elapsed:.1f}s",
    ]
    if app_cfg:
        lines += [
            f"  LLM calls  : {app_cfg.llm_call_count_session}",
            f"  Tool calls : {app_cfg.tool_call_count_session}",
            f"  Tokens     : {app_cfg.token_count_session}",
        ]
    lines += ["=" * _LINE_W, ""]
    _session_file.write("\n".join(lines) + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# Core log writers
# ═════════════════════════════════════════════════════════════════════════════

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _write_master(record: Dict[str, Any]):
    """Append JSON line to the rotating master log."""
    line = json.dumps(record, ensure_ascii=False) + "\n"
    try:
        _master_handler.stream.write(line)
        _master_handler.stream.flush()
        # Size check without passing a fake LogRecord
        if os.path.exists(_MASTER_LOG_PATH):
            if os.path.getsize(_MASTER_LOG_PATH) >= 10 * 1024 * 1024:
                _master_handler.doRollover()
    except Exception:
        pass  # never let master log crash the agent


def _write_session(ts: str, level: str, message: str, extra: Optional[Dict]):
    """Append a human-readable line to the session log."""
    with _session_lock:
        if _session_file and not _session_file.closed:
            icon     = _LEVEL_ICON.get(level, " ")
            extra_s  = ""
            if extra:
                parts   = [f"{k}={v}" for k, v in extra.items()]
                extra_s = "  " + "  ".join(parts)
            line = f"{ts:<{_COL_TS}}  {level:<{_COL_LVL}}  {icon} {message}{extra_s}\n"
            _session_file.write(line)
            _session_file.flush()


def _write_error_log(ts: str, level: str, message: str, extra: Optional[Dict]):
    """Append WARNING/ERROR to today's error log."""
    if level not in ("WARNING", "ERROR"):
        return
    try:
        path     = _error_log_path()
        icon     = _LEVEL_ICON.get(level, " ")
        extra_s  = ("  " + "  ".join(f"{k}={v}" for k, v in extra.items())) if extra else ""
        line     = f"{ts}  {level:<7}  {icon} {message}{extra_s}\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass   # never let logging crash the agent


def terminal_log_path() -> str:
    """Path to the full terminal transcript log file."""
    return _TERMINAL_LOG


def terminal_line(text: str):
    """Append one or more plain terminal lines to logs/traces/logs.log."""
    try:
        if text is None:
            return
        payload = str(text).replace("\r\n", "\n").replace("\r", "\n")
        lines = payload.split("\n")
        with _terminal_lock:
            with open(_TERMINAL_LOG, "a", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
    except Exception:
        pass


def _console_print(level: str, message: str, extra: Optional[Dict], ts: str):
    """Print to terminal via Rich (or plain stderr fallback)."""
    if not config.LOG_TO_FILE and level == "DEBUG":
        return   # suppress DEBUG in non-file mode

    extra_plain = ("  " + "  ".join(f"{k}={v}" for k, v in extra.items())) if extra else ""
    plain_line  = f"[{ts[11:19]}] [{level}] {message}{extra_plain}"
    terminal_line(plain_line)

    if RICH_AVAILABLE and _console:
        style  = _LEVEL_STYLE.get(level, "white")
        icon   = _LEVEL_ICON.get(level, " ")
        label  = f"[{style}][{level:6s}][/{style}]"
        ts_s   = f"[dim]{ts[11:19]}[/dim]"   # HH:MM:SS only on terminal
        msg_s  = f"[{style}]{message}[/{style}]"
        extras = ("  " + "  ".join(
            f"[dim]{k}=[/dim][white]{v}[/white]" for k, v in extra.items()
        )) if extra else ""
        _console.print(f"{ts_s} {label} {msg_s}{extras}")
    else:
        hms = ts[11:19]
        print(f"[{hms}] [{level}] {message}", file=sys.stderr)


# ═════════════════════════════════════════════════════════════════════════════
# Primary entry-point
# ═════════════════════════════════════════════════════════════════════════════

def log(level: str, message: str, extra: Optional[Dict] = None):
    """
    Single entry-point used by every module.
    Writes to all three streams simultaneously.
    Never raises — logging must never crash the agent.
    """
    try:
        ts = _now_iso()

        # ── Stream 1: master JSON log ─────────────────────────────────────
        record: Dict[str, Any] = {
            "ts":      ts,
            "session": _session_id or "no-session",
            "level":   level,
            "msg":     message,
        }
        if extra:
            record.update(extra)
        _write_master(record)

        # ── Stream 2: session human-readable log ──────────────────────────
        if config.LOG_TO_FILE:
            _write_session(ts, level, message, extra)

        # ── Stream 3: error/warning daily log ────────────────────────────
        _write_error_log(ts, level, message, extra)

        # ── Terminal ──────────────────────────────────────────────────────
        _console_print(level, message, extra, ts)

    except Exception:
        pass   # absorb all logging failures


# ═════════════════════════════════════════════════════════════════════════════
# Convenience wrappers
# ═════════════════════════════════════════════════════════════════════════════

def debug(msg: str, **kw):   log("DEBUG",   msg, kw or None)
def info(msg: str,  **kw):   log("INFO",    msg, kw or None)
def warn(msg: str,  **kw):   log("WARNING", msg, kw or None)
def error(msg: str, **kw):   log("ERROR",   msg, kw or None)

def tool_log(tool: str, action: str, duration_ms: float = 0, **kw):
    log("TOOL", f"{tool} → {action}", {"duration_ms": f"{duration_ms:.1f}", **kw})

def router_log(intent: str, tool: str, confidence: float, **kw):
    log("ROUTER", f"intent={intent} → tool={tool}", {"conf": f"{confidence:.2f}", **kw})

def memory_log(op: str, layer: str, **kw):
    log("MEMORY", f"{op} [{layer}]", kw or None)

def llm_log(prompt_tokens: int, completion_tokens: int, reason: str, **kw):
    total = prompt_tokens + completion_tokens
    log("LLM", f"call reason={reason}", {
        "prompt_tok":     prompt_tokens,
        "completion_tok": completion_tokens,
        "total_tok":      total,
        **kw,
    })
    if total > config.TOKEN_WARN_THRESHOLD:
        warn(f"Token usage {total} exceeds threshold {config.TOKEN_WARN_THRESHOLD}")

def capability_log(cap_name: str, event: str, **kw):
    log("CAP", f"{cap_name} → {event}", kw or None)

def trace_log(run_id: str, event: str, **kw):
    log("TRACE", f"[{run_id}] {event}", kw or None)


# ═════════════════════════════════════════════════════════════════════════════
# Session summary (terminal, end of session)
# ═════════════════════════════════════════════════════════════════════════════

def print_session_summary(app_cfg):
    if RICH_AVAILABLE and _console:
        from rich.table import Table
        t = Table(title="Session Summary", show_header=True, header_style="bold magenta")
        t.add_column("Metric",    style="cyan")
        t.add_column("Value",     style="bright_white")
        t.add_row("LLM calls",    str(app_cfg.llm_call_count_session))
        t.add_row("Tool calls",   str(app_cfg.tool_call_count_session))
        t.add_row("Tokens used",  str(app_cfg.token_count_session))
        if _session_path:
            t.add_row("Session log", _session_path)
        t.add_row("Master log",   _MASTER_LOG_PATH)
        _console.print(t)
    else:
        print(f"\nSession: LLM={app_cfg.llm_call_count_session} "
              f"Tools={app_cfg.tool_call_count_session} "
              f"Tokens={app_cfg.token_count_session}")
        if _session_path:
            print(f"Session log: {_session_path}")