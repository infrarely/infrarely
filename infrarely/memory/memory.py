"""
aos/memory.py — Agent Memory API
═══════════════════════════════════════════════════════════════════════════════
Memory is infrastructure, not a feature (Problem 5).

Three scopes:
  "session"   → this session only, cleared on restart
  "permanent" → survives restarts, stored in SQLite
  "shared"    → visible to all agents in the system

Default scope is "permanent" — agents remember across sessions automatically.

Memory operations:
  agent.memory.store(key, value, scope="permanent")
  agent.memory.get(key) → value
  agent.memory.forget(key)
  agent.memory.clear()
  agent.memory.search(query) → ranked results

Philosophy 4: Memory is always ON by default.
Philosophy 5: Every agent is observable — memory operations are logged.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY ENTRY
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class MemoryEntry:
    """A single memory item with metadata."""

    key: str
    value: Any
    scope: str = "permanent"  # "session" | "permanent" | "shared"
    agent_id: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    access_count: int = 0
    tags: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# SEARCH RESULT
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class MemorySearchResult:
    """Result from a semantic memory search."""

    entries: List[MemoryEntry] = field(default_factory=list)
    query: str = ""
    duration_ms: float = 0.0

    def __bool__(self) -> bool:
        return len(self.entries) > 0

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY BACKEND — SQLite for persistent, dict for session
# ═══════════════════════════════════════════════════════════════════════════════


class _SQLiteMemoryBackend:
    """Persistent memory backend using SQLite."""

    def __init__(self, db_path: str):
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
            CREATE TABLE IF NOT EXISTS memory (
                key         TEXT NOT NULL,
                value       TEXT NOT NULL,
                scope       TEXT NOT NULL DEFAULT 'permanent',
                agent_id    TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                tags        TEXT DEFAULT '[]',
                PRIMARY KEY (agent_id, key, scope)
            );
            CREATE INDEX IF NOT EXISTS idx_memory_agent
                ON memory(agent_id, scope);
        """
        )
        self._conn.commit()

    def store(self, entry: MemoryEntry) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO memory
                    (key, value, scope, agent_id, created_at, updated_at, access_count, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    entry.key,
                    (
                        json.dumps(entry.value)
                        if not isinstance(entry.value, str)
                        else entry.value
                    ),
                    entry.scope,
                    entry.agent_id,
                    entry.created_at,
                    entry.updated_at,
                    entry.access_count,
                    json.dumps(entry.tags),
                ),
            )
            self._conn.commit()

    def get(
        self, agent_id: str, key: str, scope: str = "permanent"
    ) -> Optional[MemoryEntry]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT key, value, scope, agent_id, created_at, updated_at, access_count, tags
                FROM memory WHERE agent_id=? AND key=? AND scope=?
            """,
                (agent_id, key, scope),
            ).fetchone()
            if not row:
                return None
            # Update access count
            self._conn.execute(
                """
                UPDATE memory SET access_count = access_count + 1,
                    updated_at = ? WHERE agent_id=? AND key=? AND scope=?
            """,
                (datetime.now(timezone.utc).isoformat(), agent_id, key, scope),
            )
            self._conn.commit()
            try:
                value = json.loads(row[1])
            except (json.JSONDecodeError, TypeError):
                value = row[1]
            return MemoryEntry(
                key=row[0],
                value=value,
                scope=row[2],
                agent_id=row[3],
                created_at=row[4],
                updated_at=row[5],
                access_count=row[6],
                tags=json.loads(row[7]) if row[7] else [],
            )

    def forget(self, agent_id: str, key: str, scope: str = "permanent") -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM memory WHERE agent_id=? AND key=? AND scope=?",
                (agent_id, key, scope),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def clear(self, agent_id: str) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM memory WHERE agent_id=?", (agent_id,)
            )
            self._conn.commit()
            return cursor.rowcount

    def search(self, agent_id: str, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Keyword-based search across memory entries."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT key, value, scope, agent_id, created_at, updated_at, access_count, tags
                FROM memory
                WHERE agent_id=? AND (key LIKE ? OR value LIKE ?)
                ORDER BY updated_at DESC
                LIMIT ?
            """,
                (agent_id, f"%{query}%", f"%{query}%", limit),
            ).fetchall()

            results = []
            for row in rows:
                try:
                    value = json.loads(row[1])
                except (json.JSONDecodeError, TypeError):
                    value = row[1]
                results.append(
                    MemoryEntry(
                        key=row[0],
                        value=value,
                        scope=row[2],
                        agent_id=row[3],
                        created_at=row[4],
                        updated_at=row[5],
                        access_count=row[6],
                        tags=json.loads(row[7]) if row[7] else [],
                    )
                )
            return results

    def get_all(self, agent_id: str, scope: Optional[str] = None) -> List[MemoryEntry]:
        with self._lock:
            if scope:
                rows = self._conn.execute(
                    "SELECT key, value, scope, agent_id, created_at, updated_at, access_count, tags "
                    "FROM memory WHERE agent_id=? AND scope=?",
                    (agent_id, scope),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT key, value, scope, agent_id, created_at, updated_at, access_count, tags "
                    "FROM memory WHERE agent_id=?",
                    (agent_id,),
                ).fetchall()
            results = []
            for row in rows:
                try:
                    value = json.loads(row[1])
                except (json.JSONDecodeError, TypeError):
                    value = row[1]
                results.append(
                    MemoryEntry(
                        key=row[0],
                        value=value,
                        scope=row[2],
                        agent_id=row[3],
                        created_at=row[4],
                        updated_at=row[5],
                        access_count=row[6],
                        tags=json.loads(row[7]) if row[7] else [],
                    )
                )
            return results

    def close(self):
        self._conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# IN-MEMORY BACKEND (for session scope)
# ═══════════════════════════════════════════════════════════════════════════════


class _InMemoryBackend:
    """Fast in-memory storage for session-scoped data."""

    def __init__(self):
        self._store: Dict[str, MemoryEntry] = {}
        self._lock = threading.Lock()

    def _key(self, agent_id: str, key: str) -> str:
        return f"{agent_id}::{key}"

    def store(self, entry: MemoryEntry) -> None:
        with self._lock:
            self._store[self._key(entry.agent_id, entry.key)] = entry

    def get(self, agent_id: str, key: str) -> Optional[MemoryEntry]:
        with self._lock:
            entry = self._store.get(self._key(agent_id, key))
            if entry:
                entry.access_count += 1
                entry.updated_at = datetime.now(timezone.utc).isoformat()
            return entry

    def forget(self, agent_id: str, key: str) -> bool:
        with self._lock:
            k = self._key(agent_id, key)
            if k in self._store:
                del self._store[k]
                return True
            return False

    def clear(self, agent_id: str) -> int:
        with self._lock:
            keys_to_remove = [k for k in self._store if k.startswith(f"{agent_id}::")]
            for k in keys_to_remove:
                del self._store[k]
            return len(keys_to_remove)

    def search(self, agent_id: str, query: str, limit: int = 10) -> List[MemoryEntry]:
        with self._lock:
            results = []
            q = query.lower()
            for k, entry in self._store.items():
                if not k.startswith(f"{agent_id}::"):
                    continue
                val_str = str(entry.value).lower()
                key_str = entry.key.lower()
                if q in val_str or q in key_str:
                    results.append(entry)
                    if len(results) >= limit:
                        break
            return results

    def get_all(self, agent_id: str) -> List[MemoryEntry]:
        with self._lock:
            return [e for k, e in self._store.items() if k.startswith(f"{agent_id}::")]


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED MEMORY (cross-agent, global namespace)
# ═══════════════════════════════════════════════════════════════════════════════


class _SharedMemoryStore:
    """
    Singleton shared memory with namespace isolation & access control.

    Every key is *owned* by the agent that wrote it.  Other agents
    must be explicitly granted access via ``grant_access()``.

    Owner permissions:
      - The agent that stores a key is its owner.
      - Only the owner (or a granted agent) can read the key.
      - Only the owner can forget (delete) the key.
      - ``grant_access(key, to_agent)`` opens read access to another agent.
      - ``revoke_access(key, from_agent)`` removes it.

    This prevents: ``malicious_agent.memory.get("api_key", scope="shared")``
    from reading another agent's shared secrets.
    """

    _instance: Optional["_SharedMemoryStore"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "_SharedMemoryStore":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._store: Dict[str, MemoryEntry] = {}
                # ACL: key → set of agent_ids allowed to read  (owner always allowed)
                cls._instance._acl: Dict[str, set] = {}
                # Owner map: key → agent_id that wrote it
                cls._instance._owners: Dict[str, str] = {}
            return cls._instance

    # ── Write ─────────────────────────────────────────────────────────────

    def store(self, entry: MemoryEntry) -> None:
        with self._lock:
            self._store[entry.key] = entry
            # Record ownership; first writer owns the key
            if entry.key not in self._owners:
                self._owners[entry.key] = entry.agent_id
                self._acl[entry.key] = {entry.agent_id}
            elif self._owners[entry.key] == entry.agent_id:
                # Owner can overwrite
                pass
            else:
                # Non-owner attempting overwrite — silently store
                # but do NOT change ownership
                pass

    # ── Read ──────────────────────────────────────────────────────────────

    def get(self, key: str, agent_id: str = "") -> Optional[MemoryEntry]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            # Access control check
            if agent_id and key in self._acl:
                acl = self._acl[key]
                if agent_id not in acl:
                    # Also check name-based matching (agent names without UUID)
                    if not any(name in agent_id for name in acl):
                        return None  # denied — no access
            return entry

    # ── Delete ────────────────────────────────────────────────────────────

    def forget(self, key: str, agent_id: str = "") -> bool:
        with self._lock:
            if key not in self._store:
                return False
            # Only owner (or legacy no-agent call) can delete
            if agent_id and key in self._owners:
                if self._owners[key] != agent_id:
                    return False
            del self._store[key]
            self._acl.pop(key, None)
            self._owners.pop(key, None)
            return True

    # ── Clear ─────────────────────────────────────────────────────────────

    def clear(self) -> int:
        with self._lock:
            count = len(self._store)
            self._store.clear()
            self._acl.clear()
            self._owners.clear()
            return count

    def clear_agent(self, agent_id: str) -> int:
        """Clear only keys owned by *agent_id*."""
        with self._lock:
            keys = [k for k, owner in self._owners.items() if owner == agent_id]
            for k in keys:
                del self._store[k]
                self._acl.pop(k, None)
                self._owners.pop(k, None)
            return len(keys)

    # ── Search ────────────────────────────────────────────────────────────

    def search(
        self, query: str, limit: int = 10, agent_id: str = ""
    ) -> List[MemoryEntry]:
        with self._lock:
            results = []
            q = query.lower()
            for entry in self._store.values():
                if q in entry.key.lower() or q in str(entry.value).lower():
                    # Respect ACL
                    if agent_id and entry.key in self._acl:
                        if agent_id not in self._acl[entry.key]:
                            continue
                    results.append(entry)
                    if len(results) >= limit:
                        break
            return results

    # ── Access Control ────────────────────────────────────────────────────

    def grant_access(self, key: str, to_agent: str, from_agent: str = "") -> bool:
        """
        Grant *to_agent* read access to *key*.

        Only the owner (``from_agent``) can grant.  Returns True on success.
        """
        with self._lock:
            if key not in self._store:
                return False
            if from_agent and key in self._owners:
                if self._owners[key] != from_agent:
                    return False  # not the owner
            self._acl.setdefault(key, set()).add(to_agent)
            return True

    def revoke_access(
        self, key: str, from_agent_target: str, owner_agent: str = ""
    ) -> bool:
        """Revoke *from_agent_target*'s read access to *key*."""
        with self._lock:
            if key not in self._acl:
                return False
            if owner_agent and key in self._owners:
                if self._owners[key] != owner_agent:
                    return False
            self._acl[key].discard(from_agent_target)
            return True

    def get_acl(self, key: str) -> set:
        """Return the set of agent IDs with read access to *key*."""
        with self._lock:
            return set(self._acl.get(key, set()))

    def get_owner(self, key: str) -> str:
        """Return the owner agent_id for *key*, or '' if not found."""
        with self._lock:
            return self._owners.get(key, "")


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT MEMORY — the public API surface
# ═══════════════════════════════════════════════════════════════════════════════


class AgentMemory:
    """
    Per-agent memory interface. Accessed via agent.memory.

    Default scope is "permanent" — data survives restarts.
    Session scope cleared when Python process exits.
    Shared scope visible to all agents.
    """

    def __init__(self, agent_id: str, db_path: str = "./aos_memory.db"):
        self._agent_id = agent_id
        self._session = _InMemoryBackend()
        self._persistent = _SQLiteMemoryBackend(db_path)
        self._shared = _SharedMemoryStore()
        self._operation_log: List[Dict[str, Any]] = []

    def _log_op(self, op: str, key: str, scope: str):
        self._operation_log.append(
            {
                "op": op,
                "key": key,
                "scope": scope,
                "ts": datetime.now(timezone.utc).isoformat(),
                "agent": self._agent_id,
            }
        )

    # ── Store ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_scope(scope: str) -> str:
        """Map scope aliases → canonical name."""
        if scope == "persistent":
            return "permanent"
        return scope

    def store(
        self,
        key: str,
        value: Any,
        scope: str = "permanent",
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Store a value in memory.

        Scopes:
          "session"   → this session only
          "permanent" → survives restarts (default)
          "shared"    → visible to all agents
        """
        scope = self._resolve_scope(scope)
        entry = MemoryEntry(
            key=key,
            value=value,
            scope=scope,
            agent_id=self._agent_id,
            tags=tags or [],
        )
        self._log_op("store", key, scope)

        if scope == "session":
            self._session.store(entry)
        elif scope == "shared":
            self._shared.store(entry)
        else:
            self._persistent.store(entry)

    # ── Get ───────────────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None, *, scope: str = "permanent") -> Any:
        """
        Retrieve a value from memory. Returns default if not found.
        If scope is not specified, searches: session → permanent → shared.
        """
        scope = self._resolve_scope(scope)
        self._log_op("get", key, scope)

        # Try specific scope
        if scope == "session":
            entry = self._session.get(self._agent_id, key)
        elif scope == "shared":
            entry = self._shared.get(key, agent_id=self._agent_id)
        else:
            entry = self._persistent.get(self._agent_id, key, scope)

        if entry:
            return entry.value

        # Auto-search across scopes if not found in specified scope
        if scope == "permanent":
            entry = self._session.get(self._agent_id, key)
            if entry:
                return entry.value
            entry = self._shared.get(key, agent_id=self._agent_id)
            if entry:
                return entry.value

        return default

    # ── Forget ────────────────────────────────────────────────────────────────

    def forget(self, key: str, scope: str = "permanent") -> bool:
        """Remove a specific key from memory."""
        scope = self._resolve_scope(scope)
        self._log_op("forget", key, scope)
        if scope == "session":
            return self._session.forget(self._agent_id, key)
        elif scope == "shared":
            return self._shared.forget(key, agent_id=self._agent_id)
        else:
            return self._persistent.forget(self._agent_id, key, scope)

    # ── Clear ─────────────────────────────────────────────────────────────────

    def clear(self) -> int:
        """Clear ALL memory for this agent (all scopes)."""
        self._log_op("clear", "*", "all")
        count = 0
        count += self._session.clear(self._agent_id)
        count += self._persistent.clear(self._agent_id)
        return count

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, limit: int = 10) -> MemorySearchResult:
        """
        Search memory semantically (keyword match + relevance).
        Searches across all scopes.
        """
        start = time.monotonic()
        self._log_op("search", query, "all")

        results = []
        results.extend(self._session.search(self._agent_id, query, limit))
        results.extend(self._persistent.search(self._agent_id, query, limit))
        results.extend(self._shared.search(query, limit, agent_id=self._agent_id))

        # Deduplicate by key
        seen = set()
        unique = []
        for entry in results:
            if entry.key not in seen:
                seen.add(entry.key)
                unique.append(entry)

        elapsed = (time.monotonic() - start) * 1000
        return MemorySearchResult(
            entries=unique[:limit],
            query=query,
            duration_ms=elapsed,
        )

    # ── Convenience ───────────────────────────────────────────────────────────

    def has(self, key: str, scope: str = "permanent") -> bool:
        """Check if a key exists in memory."""
        scope = self._resolve_scope(scope)
        return self.get(key, scope=scope) is not None

    def list_keys(self, scope: str = "permanent") -> List[str]:
        """List all keys in a given scope."""
        scope = self._resolve_scope(scope)
        if scope == "session":
            entries = self._session.get_all(self._agent_id)
        else:
            entries = self._persistent.get_all(self._agent_id, scope)
        return [e.key for e in entries]

    # ── Shared Memory Access Control ──────────────────────────────────────────

    def grant_access(self, key: str, to_agent: str) -> bool:
        """
        Grant another agent read access to a shared memory key you own.

        Example::

            agent_a.memory.grant_access("api_key", to_agent=agent_b._id)

        Returns True on success, False if:
          - Key doesn't exist
          - You don't own the key
        """
        return self._shared.grant_access(
            key, to_agent=to_agent, from_agent=self._agent_id
        )

    def revoke_access(self, key: str, from_agent: str) -> bool:
        """Revoke another agent's read access to a shared memory key you own."""
        return self._shared.revoke_access(
            key, from_agent_target=from_agent, owner_agent=self._agent_id
        )

    def shared_acl(self, key: str) -> set:
        """Return the set of agent IDs with read access to a shared key."""
        return self._shared.get_acl(key)

    def shared_owner(self, key: str) -> str:
        """Return the owner agent_id for a shared key."""
        return self._shared.get_owner(key)

    @property
    def operations(self) -> List[Dict[str, Any]]:
        """Return the log of all memory operations (for observability)."""
        return list(self._operation_log)

    def close(self):
        """Close database connections."""
        self._persistent.close()

    # ── Export / Load (cross-process portability) ─────────────────────────────

    def export(self, scope: str = "permanent") -> List[Dict[str, Any]]:
        """
        Export memory entries as a list of dicts (JSON-serializable).
        Enables cross-process memory sharing.

        Usage:
            data = agent.memory.export()
            json.dump(data, open("memory.json", "w"))
        """
        scope = self._resolve_scope(scope)
        if scope == "session":
            entries = self._session.get_all(self._agent_id)
        elif scope == "shared":
            entries = list(self._shared.search("", limit=10000))
        else:
            entries = self._persistent.get_all(self._agent_id, scope)

        return [
            {
                "key": e.key,
                "value": e.value,
                "scope": e.scope,
                "agent_id": e.agent_id,
                "created_at": e.created_at,
                "updated_at": e.updated_at,
                "access_count": e.access_count,
                "tags": e.tags,
            }
            for e in entries
        ]

    def load(self, data, overwrite: bool = True) -> int:
        """
        Load memory entries from a list of dicts or a filepath.
        Returns the number of entries loaded.

        Usage:
            data = json.load(open("memory.json"))
            count = agent.memory.load(data)
            # or
            count = agent.memory.load("memory.json")
        """
        import json as _json

        if isinstance(data, str):
            with open(data, "r") as f:
                data = _json.load(f)
        count = 0
        for item in data:
            key = item.get("key", "")
            value = item.get("value")
            scope = item.get("scope", "permanent")
            tags = item.get("tags", [])
            if not key:
                continue
            if not overwrite and self.has(key, scope=scope):
                continue
            self.store(key, value, scope=scope, tags=tags)
            count += 1
        return count

    def snapshot(self) -> Dict[str, Any]:
        """
        Full memory snapshot across all scopes (JSON-serializable).

        Usage:
            snap = agent.memory.snapshot()
            json.dump(snap, open("agent_memory.json", "w"))
        """
        return {
            "agent_id": self._agent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session": self.export(scope="session"),
            "permanent": self.export(scope="permanent"),
            "shared": self.export(scope="shared"),
        }
