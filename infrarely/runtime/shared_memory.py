"""
runtime/shared_memory.py — Module 4: Shared Memory Layer
═══════════════════════════════════════════════════════════════════════════════
Cross-agent knowledge sharing with access control.
Acts like /dev/shm in Linux — shared state that agents can read/write.

Gap solutions:
  Gap 4  — Shared memory corruption: versioned writes, atomic compare-and-swap,
           conflict detection on concurrent writes

Design:
  Every entry has a version counter. Writes only succeed if the caller
  provides the expected version (optimistic concurrency). Results are
  namespaced per agent with a shared global namespace.
"""

from __future__ import annotations
import copy
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from infrarely.observability import logger


@dataclass
class MemoryEntry:
    """A single shared memory entry with versioning."""

    key: str
    value: Any
    version: int = 1
    owner: str = ""  # agent_id who last wrote
    namespace: str = "global"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    read_count: int = 0
    write_count: int = 1
    readers: Set[str] = field(default_factory=set)  # agent_ids with read permission
    writers: Set[str] = field(default_factory=set)  # agent_ids with write permission
    locked_by: str = ""  # agent_id holding exclusive lock
    lock_time: float = 0.0
    history: List[Dict[str, Any]] = field(default_factory=list)  # last N versions

    MAX_HISTORY = 5

    def record_history(self):
        """Keep last N versions for rollback."""
        self.history.append(
            {
                "version": self.version,
                "value": copy.deepcopy(self.value),
                "owner": self.owner,
                "timestamp": self.updated_at,
            }
        )
        if len(self.history) > self.MAX_HISTORY:
            self.history = self.history[-self.MAX_HISTORY :]


class SharedMemory:
    """
    Versioned, access-controlled shared memory for multi-agent state.

    Invariants:
      • Max entries: 1000
      • Writes require version match (optimistic concurrency — Gap 4)
      • Locks auto-expire after 30 seconds
      • Namespace isolation: agents see only permitted keys
    """

    MAX_ENTRIES = 1000
    LOCK_TIMEOUT_S = 30.0

    def __init__(self):
        self._store: Dict[str, MemoryEntry] = {}  # full_key → entry
        self._conflict_count = 0
        self._total_reads = 0
        self._total_writes = 0

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _full_key(namespace: str, key: str) -> str:
        return f"{namespace}::{key}"

    def _check_lock(self, entry: MemoryEntry, agent_id: str) -> bool:
        """Check if entry is locked by another agent. Auto-expire stale locks."""
        if not entry.locked_by:
            return True
        if entry.locked_by == agent_id:
            return True
        # Check lock timeout
        if time.time() - entry.lock_time > self.LOCK_TIMEOUT_S:
            logger.warn(
                f"SharedMemory: lock expired for '{entry.key}' "
                f"(held by '{entry.locked_by}') — releasing"
            )
            entry.locked_by = ""
            entry.lock_time = 0.0
            return True
        return False

    def _check_permission(self, entry: MemoryEntry, agent_id: str, write: bool) -> bool:
        """Check read/write permission. Empty sets mean public access."""
        if write:
            if (
                entry.writers
                and agent_id not in entry.writers
                and agent_id != entry.owner
            ):
                return False
        else:
            if (
                entry.readers
                and agent_id not in entry.readers
                and agent_id != entry.owner
            ):
                return False
        return True

    # ── Write (with optimistic concurrency — Gap 4) ──────────────────────────
    def write(
        self,
        key: str,
        value: Any,
        agent_id: str,
        namespace: str = "global",
        expected_version: Optional[int] = None,
        readers: Set[str] = None,
        writers: Set[str] = None,
    ) -> Tuple[bool, int]:
        """
        Write a value. Returns (success, current_version).

        If expected_version is provided, write only succeeds if it matches
        the entry's current version (compare-and-swap, Gap 4).
        New keys are always created (version=1).
        """
        fk = self._full_key(namespace, key)
        existing = self._store.get(fk)

        if existing:
            # Lock check
            if not self._check_lock(existing, agent_id):
                logger.warn(
                    f"SharedMemory: write blocked — '{fk}' locked by '{existing.locked_by}'"
                )
                return False, existing.version

            # Permission check
            if not self._check_permission(existing, agent_id, write=True):
                logger.warn(
                    f"SharedMemory: write denied — '{agent_id}' has no write access to '{fk}'"
                )
                return False, existing.version

            # Version check (Gap 4: optimistic concurrency)
            if expected_version is not None and existing.version != expected_version:
                self._conflict_count += 1
                logger.warn(
                    f"SharedMemory: version conflict on '{fk}' — "
                    f"expected v{expected_version}, actual v{existing.version} "
                    f"(conflict #{self._conflict_count})"
                )
                return False, existing.version

            # Record history before overwrite
            existing.record_history()
            existing.value = copy.deepcopy(value)
            existing.version += 1
            existing.owner = agent_id
            existing.updated_at = time.time()
            existing.write_count += 1
            self._total_writes += 1
            return True, existing.version

        else:
            # New entry
            if len(self._store) >= self.MAX_ENTRIES:
                logger.warn(
                    f"SharedMemory: at capacity ({self.MAX_ENTRIES}) — evicting oldest"
                )
                self._evict_oldest()

            entry = MemoryEntry(
                key=key,
                value=copy.deepcopy(value),
                version=1,
                owner=agent_id,
                namespace=namespace,
                readers=readers or set(),
                writers=writers or set(),
            )
            self._store[fk] = entry
            self._total_writes += 1
            return True, 1

    # ── Read ──────────────────────────────────────────────────────────────────
    def read(
        self, key: str, agent_id: str, namespace: str = "global"
    ) -> Optional[Tuple[Any, int]]:
        """
        Read a value. Returns (value, version) or None if not found.
        Returns deep copy to prevent external mutation (Gap 4).
        """
        fk = self._full_key(namespace, key)
        entry = self._store.get(fk)
        if not entry:
            return None

        if not self._check_permission(entry, agent_id, write=False):
            logger.warn(f"SharedMemory: read denied for '{agent_id}' on '{fk}'")
            return None

        entry.read_count += 1
        self._total_reads += 1
        return copy.deepcopy(entry.value), entry.version

    # ── Delete ────────────────────────────────────────────────────────────────
    def delete(self, key: str, agent_id: str, namespace: str = "global") -> bool:
        """Delete an entry. Only owner or writer can delete."""
        fk = self._full_key(namespace, key)
        entry = self._store.get(fk)
        if not entry:
            return False
        if not self._check_lock(entry, agent_id):
            return False
        if agent_id != entry.owner and agent_id not in entry.writers:
            return False
        del self._store[fk]
        return True

    # ── Locking ───────────────────────────────────────────────────────────────
    def lock(self, key: str, agent_id: str, namespace: str = "global") -> bool:
        """Acquire exclusive lock on a key."""
        fk = self._full_key(namespace, key)
        entry = self._store.get(fk)
        if not entry:
            return False
        if not self._check_lock(entry, agent_id):
            return False
        entry.locked_by = agent_id
        entry.lock_time = time.time()
        return True

    def unlock(self, key: str, agent_id: str, namespace: str = "global") -> bool:
        """Release exclusive lock."""
        fk = self._full_key(namespace, key)
        entry = self._store.get(fk)
        if not entry:
            return False
        if entry.locked_by != agent_id:
            return False
        entry.locked_by = ""
        entry.lock_time = 0.0
        return True

    # ── Rollback (Gap 4) ─────────────────────────────────────────────────────
    def rollback(self, key: str, agent_id: str, namespace: str = "global") -> bool:
        """Rollback to previous version."""
        fk = self._full_key(namespace, key)
        entry = self._store.get(fk)
        if not entry or not entry.history:
            return False
        if not self._check_permission(entry, agent_id, write=True):
            return False
        prev = entry.history.pop()
        entry.value = prev["value"]
        entry.version = prev["version"]
        entry.updated_at = time.time()
        logger.info(f"SharedMemory: rolled back '{fk}' to v{entry.version}")
        return True

    # ── Namespace listing ─────────────────────────────────────────────────────
    def list_keys(self, namespace: str = "global", agent_id: str = "") -> List[str]:
        """List all keys in a namespace visible to agent."""
        keys = []
        prefix = f"{namespace}::"
        for fk, entry in self._store.items():
            if fk.startswith(prefix):
                if not agent_id or self._check_permission(entry, agent_id, write=False):
                    keys.append(entry.key)
        return keys

    # ── Eviction ──────────────────────────────────────────────────────────────
    def _evict_oldest(self):
        """Evict oldest entry by creation time."""
        if not self._store:
            return
        oldest_key = min(self._store, key=lambda k: self._store[k].created_at)
        del self._store[oldest_key]

    # ── Cleanup for removed agent ─────────────────────────────────────────────
    def cleanup_agent(self, agent_id: str):
        """Remove all locks held by agent and clean up private namespace."""
        for entry in self._store.values():
            if entry.locked_by == agent_id:
                entry.locked_by = ""
                entry.lock_time = 0.0
        # Remove private namespace entries
        private_prefix = f"{agent_id}::"
        to_remove = [k for k in self._store if k.startswith(private_prefix)]
        for k in to_remove:
            del self._store[k]

    # ── Snapshot ──────────────────────────────────────────────────────────────
    def snapshot(self) -> Dict[str, Any]:
        namespaces = defaultdict(int)
        for fk in self._store:
            ns = fk.split("::")[0] if "::" in fk else "unknown"
            namespaces[ns] += 1

        return {
            "total_entries": len(self._store),
            "total_reads": self._total_reads,
            "total_writes": self._total_writes,
            "conflict_count": self._conflict_count,
            "namespaces": dict(namespaces),
            "recent_entries": [
                {
                    "key": e.key,
                    "namespace": e.namespace,
                    "version": e.version,
                    "owner": e.owner,
                    "locked": bool(e.locked_by),
                }
                for e in sorted(
                    self._store.values(), key=lambda e: e.updated_at, reverse=True
                )[:10]
            ],
        }
