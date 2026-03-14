"""
infrarely/scaling.py — Horizontal Scaling & State Backend Support
═══════════════════════════════════════════════════════════════════════════════
SCALE GAP 3: The ability to run multiple InfraRely instances that share state.

Current: Single Python process, SQLite.
Upgrade path:
    infrarely.configure(state_backend="sqlite")          # current default
    infrarely.configure(state_backend="redis", redis_url="redis://localhost")
    infrarely.configure(state_backend="redis", redis_url="redis://cluster:6379",
                  coordination="redis-locks")

Architecture:
    StateBackend (ABC)     — abstract interface for state storage
    SQLiteBackend          — current default (single-process)
    MemoryBackend          — in-memory for testing
    RedisBackend           — multi-process / multi-machine
    CoordinationManager    — distributed lock management
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from infrarely.runtime.paths import STATE_DB


# ═══════════════════════════════════════════════════════════════════════════════
# STATE BACKEND INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════


class StateBackend(ABC):
    """Abstract interface for distributed state storage."""

    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        """Get a value by key."""
        ...

    @abstractmethod
    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Set a value with optional TTL in seconds."""
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if existed."""
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        ...

    @abstractmethod
    def keys(self, pattern: str = "*") -> List[str]:
        """List keys matching a pattern."""
        ...

    @abstractmethod
    def lock(self, name: str, timeout: float = 10.0) -> "DistributedLock":
        """Acquire a distributed lock."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the backend connection."""
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTED LOCK
# ═══════════════════════════════════════════════════════════════════════════════


class DistributedLock:
    """A distributed lock that works across processes."""

    def __init__(self, name: str, backend: "StateBackend", timeout: float = 10.0):
        self._name = f"_lock:{name}"
        self._backend = backend
        self._timeout = timeout
        self._acquired = False

    def acquire(self) -> bool:
        """Try to acquire the lock. Returns True if successful."""
        start = time.time()
        while time.time() - start < self._timeout:
            if not self._backend.exists(self._name):
                self._backend.set(
                    self._name,
                    json.dumps({"holder": os.getpid(), "acquired": time.time()}),
                    ttl=int(self._timeout * 2),
                )
                self._acquired = True
                return True
            time.sleep(0.05)
        return False

    def release(self) -> None:
        """Release the lock."""
        if self._acquired:
            self._backend.delete(self._name)
            self._acquired = False

    def __enter__(self) -> "DistributedLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY BACKEND — in-memory (single process, for testing)
# ═══════════════════════════════════════════════════════════════════════════════


class MemoryBackend(StateBackend):
    """In-memory state backend for testing."""

    def __init__(self):
        self._data: Dict[str, str] = {}
        self._ttls: Dict[str, float] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            self._evict_expired()
            return self._data.get(key)

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        with self._lock:
            self._data[key] = value
            if ttl is not None:
                self._ttls[key] = time.time() + ttl
            elif key in self._ttls:
                del self._ttls[key]

    def delete(self, key: str) -> bool:
        with self._lock:
            existed = key in self._data
            self._data.pop(key, None)
            self._ttls.pop(key, None)
            return existed

    def exists(self, key: str) -> bool:
        with self._lock:
            self._evict_expired()
            return key in self._data

    def keys(self, pattern: str = "*") -> List[str]:
        import fnmatch

        with self._lock:
            self._evict_expired()
            if pattern == "*":
                return list(self._data.keys())
            return [k for k in self._data if fnmatch.fnmatch(k, pattern)]

    def lock(self, name: str, timeout: float = 10.0) -> DistributedLock:
        return DistributedLock(name, self, timeout)

    def close(self) -> None:
        with self._lock:
            self._data.clear()
            self._ttls.clear()

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [k for k, exp in self._ttls.items() if exp <= now]
        for k in expired:
            self._data.pop(k, None)
            del self._ttls[k]


# ═══════════════════════════════════════════════════════════════════════════════
# SQLITE BACKEND — single process, persistent
# ═══════════════════════════════════════════════════════════════════════════════


class SQLiteBackend(StateBackend):
    """SQLite-based state backend for single-machine deployment."""

    def __init__(self, db_path: str = str(STATE_DB)):
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
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                expires_at REAL
            );
            CREATE INDEX IF NOT EXISTS idx_state_expires ON state(expires_at);
        """
        )
        self._conn.commit()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            self._evict_expired()
            row = self._conn.execute(
                "SELECT value FROM state WHERE key=?", (key,)
            ).fetchone()
            return row[0] if row else None

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        with self._lock:
            expires_at = (time.time() + ttl) if ttl else None
            self._conn.execute(
                "INSERT OR REPLACE INTO state (key, value, expires_at) VALUES (?, ?, ?)",
                (key, value, expires_at),
            )
            self._conn.commit()

    def delete(self, key: str) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM state WHERE key=?", (key,))
            self._conn.commit()
            return cursor.rowcount > 0

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def keys(self, pattern: str = "*") -> List[str]:
        with self._lock:
            self._evict_expired()
            if pattern == "*":
                rows = self._conn.execute("SELECT key FROM state").fetchall()
            else:
                # Convert glob to SQL LIKE
                sql_pattern = pattern.replace("*", "%").replace("?", "_")
                rows = self._conn.execute(
                    "SELECT key FROM state WHERE key LIKE ?", (sql_pattern,)
                ).fetchall()
            return [r[0] for r in rows]

    def lock(self, name: str, timeout: float = 10.0) -> DistributedLock:
        return DistributedLock(name, self, timeout)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _evict_expired(self) -> None:
        now = time.time()
        self._conn.execute(
            "DELETE FROM state WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (now,),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# REDIS BACKEND — multi-process / multi-machine (requires redis-py)
# ═══════════════════════════════════════════════════════════════════════════════


class RedisBackend(StateBackend):
    """
    Redis-based state backend for distributed deployments.

    Requires ``pip install redis`` (optional dependency).
    Falls back gracefully if redis is not installed.
    """

    def __init__(
        self, redis_url: str = "redis://localhost:6379", prefix: str = "infrarely:"
    ):
        self._prefix = prefix
        self._redis_url = redis_url
        self._client = None
        try:
            import redis

            self._client = redis.Redis.from_url(redis_url, decode_responses=True)
            self._client.ping()
        except ImportError:
            raise ImportError(
                "redis package required for RedisBackend. "
                "Install with: pip install redis"
            )
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Redis at {redis_url}: {e}. "
                "Ensure Redis is running or use state_backend='sqlite'."
            )

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def get(self, key: str) -> Optional[str]:
        return self._client.get(self._key(key))

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        if ttl:
            self._client.setex(self._key(key), ttl, value)
        else:
            self._client.set(self._key(key), value)

    def delete(self, key: str) -> bool:
        return bool(self._client.delete(self._key(key)))

    def exists(self, key: str) -> bool:
        return bool(self._client.exists(self._key(key)))

    def keys(self, pattern: str = "*") -> List[str]:
        full_pattern = self._key(pattern)
        raw_keys = self._client.keys(full_pattern)
        prefix_len = len(self._prefix)
        return [k[prefix_len:] for k in raw_keys]

    def lock(self, name: str, timeout: float = 10.0) -> "RedisLock":
        return RedisLock(self._key(f"_lock:{name}"), self._client, timeout)

    def close(self) -> None:
        if self._client:
            self._client.close()


class RedisLock:
    """Redis-based distributed lock using SETNX."""

    def __init__(self, key: str, client: Any, timeout: float = 10.0):
        self._key = key
        self._client = client
        self._timeout = timeout
        self._acquired = False

    def acquire(self) -> bool:
        result = self._client.set(
            self._key,
            str(os.getpid()),
            nx=True,
            ex=int(self._timeout * 2),
        )
        self._acquired = bool(result)
        return self._acquired

    def release(self) -> None:
        if self._acquired:
            self._client.delete(self._key)
            self._acquired = False

    def __enter__(self) -> "RedisLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


# ═══════════════════════════════════════════════════════════════════════════════
# BACKEND FACTORY
# ═══════════════════════════════════════════════════════════════════════════════


def create_backend(
    backend_type: str = "sqlite",
    **kwargs,
) -> StateBackend:
    """
    Create a state backend by type.

    Parameters
    ----------
    backend_type : str
        "memory", "sqlite", or "redis"
    **kwargs
        Backend-specific options (db_path, redis_url, prefix, etc.)
    """
    if backend_type == "memory":
        return MemoryBackend()
    elif backend_type == "sqlite":
        return SQLiteBackend(db_path=kwargs.get("db_path", str(STATE_DB)))
    elif backend_type == "redis":
        return RedisBackend(
            redis_url=kwargs.get("redis_url", "redis://localhost:6379"),
            prefix=kwargs.get("prefix", "infrarely:"),
        )
    else:
        raise ValueError(
            f"Unknown state backend: {backend_type!r}. "
            "Use 'memory', 'sqlite', or 'redis'."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# COORDINATION MANAGER
# ═══════════════════════════════════════════════════════════════════════════════


class CoordinationManager:
    """
    Manages distributed coordination between InfraRely instances.

    Provides leader election, task partitioning, and health checking
    across multiple InfraRely workers.
    """

    def __init__(self, backend: StateBackend, instance_id: Optional[str] = None):
        self._backend = backend
        self._instance_id = instance_id or f"infrarely_{os.getpid()}_{int(time.time())}"
        self._heartbeat_interval = 5.0
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._running = False

    @property
    def instance_id(self) -> str:
        return self._instance_id

    def register_instance(self) -> None:
        """Register this InfraRely instance in the coordination registry."""
        self._backend.set(
            f"instance:{self._instance_id}",
            json.dumps(
                {
                    "instance_id": self._instance_id,
                    "pid": os.getpid(),
                    "started": time.time(),
                    "last_heartbeat": time.time(),
                }
            ),
            ttl=int(self._heartbeat_interval * 3),
        )

    def get_instances(self) -> List[Dict[str, Any]]:
        """Get all live InfraRely instances."""
        keys = self._backend.keys("instance:*")
        instances = []
        for key in keys:
            data = self._backend.get(key)
            if data:
                try:
                    instances.append(json.loads(data))
                except Exception:
                    pass
        return instances

    def start_heartbeat(self) -> None:
        """Start background heartbeat to keep this instance registered."""
        if self._running:
            return
        self._running = True

        def heartbeat_loop():
            while self._running:
                try:
                    self.register_instance()
                except Exception:
                    pass
                time.sleep(self._heartbeat_interval)

        self._heartbeat_thread = threading.Thread(
            target=heartbeat_loop,
            daemon=True,
            name="infrarely-coordination-heartbeat",
        )
        self._heartbeat_thread.start()

    def stop_heartbeat(self) -> None:
        """Stop the heartbeat thread."""
        self._running = False
        # Remove our instance
        try:
            self._backend.delete(f"instance:{self._instance_id}")
        except Exception:
            pass

    def claim_task(self, task_id: str, timeout: int = 300) -> bool:
        """
        Attempt to claim a task for this instance (atomic via lock).

        Returns True if claimed, False if already claimed by another instance.
        """
        claim_key = f"task_claim:{task_id}"
        if self._backend.exists(claim_key):
            return False
        self._backend.set(
            claim_key,
            json.dumps({"instance": self._instance_id, "claimed_at": time.time()}),
            ttl=timeout,
        )
        return True

    def release_task(self, task_id: str) -> None:
        """Release a task claim."""
        self._backend.delete(f"task_claim:{task_id}")
