"""
memory/memory_manager.py — Three-tier memory hierarchy for the Student Agent.

┌─────────────────────────────────────────────────────────────────────────────┐
│  Tier 1 — Working Memory   (in-process deque, last N messages)              │
│  Tier 2 — Summary Memory   (SQLite, compressed conversation summaries)      │
│  Tier 3 — Long-term Memory (SQLite, student profile, key facts, flashcards) │
└─────────────────────────────────────────────────────────────────────────────┘

WHY THIS DESIGN
───────────────
Problem addressed: Context-window collapse — long conversations cause the LLM
to lose earlier information and increase per-call token count (cost + latency).

Solution:
  • Working memory holds the last WORKING_MEMORY_WINDOW raw messages.  These go
    directly into the LLM context every call (fast, full fidelity).
  • When the raw history exceeds SUMMARY_TRIGGER_COUNT messages, the oldest
    half is compressed into a single summary entry in SQLite.  The summary is
    injected into the context as a short "session recap" block.
  • Long-term memory stores the student profile and any explicitly persisted
    facts (e.g. "user mentioned exam is Friday").  It is read once at session
    start and injected into the system prompt.

This approach keeps active context under ~1,000 tokens for typical sessions
while preserving full semantic continuity — critical on resource-limited
hardware with expensive API calls.
"""

import json
import os
import sqlite3
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from config import (
    LONG_TERM_DB_PATH,
    WORKING_MEMORY_WINDOW,
    SUMMARY_TRIGGER_COUNT,
    MAX_SUMMARY_TOKENS,
)
from infrarely.observability.logger import get_logger

log = get_logger("memory")


# ─────────────────────────────────────────────────────────────────────────────
#  DATABASE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _init_db(path: str) -> sqlite3.Connection:
    """Create tables if they don't already exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # concurrent reads during writes

    conn.executescript("""
    CREATE TABLE IF NOT EXISTS summaries (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id   TEXT NOT NULL,
        content     TEXT NOT NULL,
        created_at  TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS long_term_facts (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id   TEXT NOT NULL,
        category    TEXT NOT NULL,
        key         TEXT NOT NULL,
        value       TEXT NOT NULL,
        confidence  REAL DEFAULT 1.0,
        updated_at  TEXT NOT NULL,
        UNIQUE(thread_id, category, key) ON CONFLICT REPLACE
    );

    CREATE TABLE IF NOT EXISTS student_profiles (
        student_id      TEXT PRIMARY KEY,
        profile_json    TEXT NOT NULL,
        updated_at      TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_summaries_thread
        ON summaries(thread_id);
    CREATE INDEX IF NOT EXISTS idx_facts_thread
        ON long_term_facts(thread_id, category);
    """)
    conn.commit()
    return conn


# ─────────────────────────────────────────────────────────────────────────────
#  WORKING MEMORY  (Tier 1)
# ─────────────────────────────────────────────────────────────────────────────

class WorkingMemory:
    """
    Fixed-size sliding window of the most recent raw messages.

    Each entry: {"role": "human"|"assistant"|"tool", "content": str, "ts": float}
    """

    def __init__(self, window: int = WORKING_MEMORY_WINDOW):
        self._window  = window
        self._buffer: deque = deque(maxlen=window)

    def add(self, role: str, content: str):
        entry = {"role": role, "content": content, "ts": time.time()}
        self._buffer.append(entry)
        log.mem_write("working", role, size_bytes=len(content))

    def get_messages(self) -> List[Dict]:
        log.mem_read("working", f"last_{len(self._buffer)}_messages")
        return list(self._buffer)

    def count(self) -> int:
        return len(self._buffer)

    def clear(self):
        self._buffer.clear()


# ─────────────────────────────────────────────────────────────────────────────
#  SUMMARY MEMORY  (Tier 2)
# ─────────────────────────────────────────────────────────────────────────────

class SummaryMemory:
    """
    Stores LLM-generated summaries of older conversation segments in SQLite.

    A summary is created every time working memory overflows.  The full
    history is compressed into a paragraph, then the raw messages are dropped.
    The latest summary is prepended to the system context as a "session recap."
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def save_summary(self, thread_id: str, summary_text: str):
        ts = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO summaries (thread_id, content, created_at) VALUES (?,?,?)",
            (thread_id, summary_text, ts),
        )
        self._conn.commit()
        log.mem_write("summary", thread_id, size_bytes=len(summary_text))

    def get_latest_summary(self, thread_id: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT content FROM summaries WHERE thread_id=? ORDER BY id DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
        if row:
            log.mem_read("summary", thread_id)
            return row["content"]
        return None

    def get_all_summaries(self, thread_id: str) -> List[str]:
        rows = self._conn.execute(
            "SELECT content FROM summaries WHERE thread_id=? ORDER BY id ASC",
            (thread_id,),
        ).fetchall()
        return [r["content"] for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
#  LONG-TERM MEMORY  (Tier 3)
# ─────────────────────────────────────────────────────────────────────────────

class LongTermMemory:
    """
    Persistent facts and student profiles stored in SQLite.

    Facts are keyed by (thread_id, category, key) and carry a confidence score
    (0.0–1.0) to support the hallucination guard.  When the agent retrieves a
    fact, it checks confidence; low-confidence facts are flagged to the user.
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    # ── Facts ────────────────────────────────────────────────────────────────

    def store_fact(self, thread_id: str, category: str, key: str,
                   value: Any, confidence: float = 1.0):
        ts    = datetime.now(timezone.utc).isoformat()
        val_s = json.dumps(value) if not isinstance(value, str) else value
        self._conn.execute(
            """INSERT INTO long_term_facts
               (thread_id, category, key, value, confidence, updated_at)
               VALUES (?,?,?,?,?,?)""",
            (thread_id, category, key, val_s, confidence, ts),
        )
        self._conn.commit()
        log.mem_write("long_term", f"{category}/{key}", size_bytes=len(val_s))

    def retrieve_fact(self, thread_id: str, category: str,
                      key: str) -> Tuple[Optional[Any], float]:
        """Returns (value, confidence).  value=None if not found."""
        row = self._conn.execute(
            """SELECT value, confidence FROM long_term_facts
               WHERE thread_id=? AND category=? AND key=?""",
            (thread_id, category, key),
        ).fetchone()
        if row:
            log.mem_read("long_term", f"{category}/{key}")
            try:
                val = json.loads(row["value"])
            except (json.JSONDecodeError, TypeError):
                val = row["value"]
            return val, row["confidence"]
        return None, 0.0

    def retrieve_by_category(self, thread_id: str, category: str) -> Dict[str, Any]:
        rows = self._conn.execute(
            """SELECT key, value, confidence FROM long_term_facts
               WHERE thread_id=? AND category=?""",
            (thread_id, category),
        ).fetchall()
        result = {}
        for row in rows:
            try:
                val = json.loads(row["value"])
            except (json.JSONDecodeError, TypeError):
                val = row["value"]
            result[row["key"]] = {"value": val, "confidence": row["confidence"]}
        return result

    # ── Student profiles ─────────────────────────────────────────────────────

    def save_profile(self, student_id: str, profile: Dict):
        ts = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT OR REPLACE INTO student_profiles
               (student_id, profile_json, updated_at) VALUES (?,?,?)""",
            (student_id, json.dumps(profile), ts),
        )
        self._conn.commit()
        log.mem_write("long_term", f"profile/{student_id}")

    def load_profile(self, student_id: str) -> Optional[Dict]:
        row = self._conn.execute(
            "SELECT profile_json FROM student_profiles WHERE student_id=?",
            (student_id,),
        ).fetchone()
        if row:
            log.mem_read("long_term", f"profile/{student_id}")
            return json.loads(row["profile_json"])
        return None

    def list_profiles(self) -> List[str]:
        rows = self._conn.execute(
            "SELECT student_id FROM student_profiles ORDER BY updated_at DESC"
        ).fetchall()
        return [r["student_id"] for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
#  MEMORY MANAGER  (facade over all three tiers)
# ─────────────────────────────────────────────────────────────────────────────

class MemoryManager:
    """
    Single entry-point for all memory operations.

    Responsibilities:
      1. Add messages to working memory.
      2. Detect overflow → trigger compression via the LLM.
      3. Build the context block injected into each LLM call.
      4. Persist / retrieve long-term facts and profiles.
    """

    def __init__(self, thread_id: str, llm=None):
        self.thread_id     = thread_id
        self._llm          = llm   # used for summarisation; can be set later
        self._conn         = _init_db(LONG_TERM_DB_PATH)
        self.working       = WorkingMemory()
        self.summaries     = SummaryMemory(self._conn)
        self.long_term     = LongTermMemory(self._conn)
        self._raw_history: List[Dict] = []   # full unsummarised buffer

    def set_llm(self, llm):
        """Inject LLM after construction (avoids circular import at startup)."""
        self._llm = llm

    # ── Add a message ────────────────────────────────────────────────────────

    def add_message(self, role: str, content: str):
        """Record a new conversation turn."""
        self.working.add(role, content)
        self._raw_history.append({"role": role, "content": content})

        if len(self._raw_history) >= SUMMARY_TRIGGER_COUNT:
            self._compress()

    # ── Compression ─────────────────────────────────────────────────────────

    def _compress(self):
        """Summarise the oldest half of raw history and drop those messages."""
        half       = len(self._raw_history) // 2
        to_compress = self._raw_history[:half]
        self._raw_history = self._raw_history[half:]

        before_tokens = sum(len(m["content"].split()) for m in to_compress)

        if self._llm is None:
            # Fallback: simple extractive summary
            summary = self._extractive_summary(to_compress)
        else:
            summary = self._llm_summary(to_compress)

        after_tokens = len(summary.split())
        log.mem_compress(before_tokens, after_tokens)
        self.summaries.save_summary(self.thread_id, summary)

    def _extractive_summary(self, messages: List[Dict]) -> str:
        """Lightweight summary when no LLM is available."""
        lines = [f"[{m['role'].upper()}]: {m['content'][:120]}" for m in messages]
        return "Earlier conversation summary:\n" + "\n".join(lines[:10])

    def _llm_summary(self, messages: List[Dict]) -> str:
        """Ask the LLM to compress the messages into a concise paragraph."""
        transcript = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )
        prompt = (
            f"Summarise the following conversation into at most "
            f"{MAX_SUMMARY_TOKENS} words, preserving all key facts, "
            f"decisions, and student-specific information:\n\n{transcript}"
        )
        try:
            from langchain_core.messages import HumanMessage
            resp = self._llm.invoke([HumanMessage(content=prompt)])
            return resp.content
        except Exception as exc:
            log.error("memory", f"LLM summarisation failed: {exc}")
            return self._extractive_summary(messages)

    # ── Context builder ──────────────────────────────────────────────────────

    def build_context_block(self) -> str:
        """
        Produce the memory context string to inject into the system prompt.

        Structure:
          [LONG-TERM PROFILE]  → student name, courses, preferences
          [SESSION RECAP]      → latest compression summary
          [RECENT MESSAGES]    → working memory window (last N turns)
        """
        parts = []

        # Tier 3 — student profile
        profile = self.load_active_profile()
        if profile:
            parts.append(
                f"[STUDENT PROFILE]\n"
                f"Name: {profile.get('name', 'Unknown')}\n"
                f"Programme: {profile.get('degree_program', 'N/A')}\n"
                f"Courses: {', '.join(profile.get('courses', []))}\n"
                f"Learning style: {profile.get('learning_style', 'N/A')}\n"
                f"Study session length: {profile.get('study_session_length', 'N/A')} min"
            )

        # Tier 2 — session recap
        latest_summary = self.summaries.get_latest_summary(self.thread_id)
        if latest_summary:
            parts.append(f"[SESSION RECAP]\n{latest_summary}")

        # Tier 1 — recent messages already handled by LangGraph message list
        # (We just note how many are in context for the logger)
        n_working = self.working.count()
        parts.append(f"[WORKING MEMORY] {n_working} recent messages in active context.")

        return "\n\n".join(parts)

    # ── Profile helpers ──────────────────────────────────────────────────────

    def load_active_profile(self) -> Optional[Dict]:
        """Load profile keyed by thread_id (one profile per thread)."""
        return self.long_term.load_profile(self.thread_id)

    def save_active_profile(self, profile: Dict):
        profile["thread_id"] = self.thread_id
        self.long_term.save_profile(self.thread_id, profile)

    # ── Quick fact persistence ────────────────────────────────────────────────

    def remember(self, category: str, key: str, value: Any,
                 confidence: float = 1.0):
        """Persist a fact with optional confidence score."""
        self.long_term.store_fact(self.thread_id, category, key,
                                  value, confidence)

    def recall(self, category: str, key: str) -> Tuple[Optional[Any], float]:
        """Retrieve a fact.  Returns (value, confidence)."""
        return self.long_term.retrieve_fact(self.thread_id, category, key)