"""
infrarely/sync.py — Knowledge Source Auto-Sync
═══════════════════════════════════════════════════════════════════════════════
Knowledge sources that automatically refresh when data changes.

Without this, agents give stale answers forever.

Usage::

    agent.knowledge.add_database(
        "products",
        connection=db,
        query="SELECT * FROM products",
        refresh_interval=3600,
        change_detection=True,
    )

    # Webhook-triggered refresh
    await agent.knowledge.refresh("products")
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# SYNC SOURCE — Definition of a syncable knowledge source
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SyncSource:
    """A knowledge source that can auto-refresh."""

    name: str = ""
    source_type: str = "static"  # "static" | "database" | "api" | "file" | "webhook"

    # ── Database sources ──────────────────────────────────────────────────────
    connection: Any = None  # DB connection object
    query: str = ""  # SQL query

    # ── API sources ───────────────────────────────────────────────────────────
    url: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    method: str = "GET"

    # ── File sources ──────────────────────────────────────────────────────────
    file_path: str = ""

    # ── Sync settings ─────────────────────────────────────────────────────────
    refresh_interval: float = 0  # seconds, 0 = no auto-refresh
    change_detection: bool = True  # only refresh if data changed
    last_sync: float = 0.0
    last_hash: str = ""
    sync_count: int = 0
    enabled: bool = True

    # ── Callbacks ─────────────────────────────────────────────────────────────
    on_refresh: Optional[Callable[[int], None]] = None  # called with count of records
    on_error: Optional[Callable[[Exception], None]] = None
    fetch_fn: Optional[Callable[[], Any]] = None  # custom fetch function

    @property
    def is_due(self) -> bool:
        """Check if sync is due based on interval."""
        if self.refresh_interval <= 0:
            return False
        if self.last_sync == 0:
            return True
        return (time.time() - self.last_sync) >= self.refresh_interval


# ═══════════════════════════════════════════════════════════════════════════════
# SYNC RESULT — Result of a sync operation
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SyncResult:
    """Result of a knowledge sync operation."""

    source_name: str = ""
    success: bool = True
    records_count: int = 0
    changed: bool = False  # was data actually different?
    duration_ms: float = 0.0
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


# ═══════════════════════════════════════════════════════════════════════════════
# SYNC SCHEDULER — Background thread that refreshes sources
# ═══════════════════════════════════════════════════════════════════════════════


class SyncScheduler:
    """
    Background scheduler that periodically refreshes knowledge sources.

    Runs in a daemon thread, checking each source's refresh_interval.
    """

    def __init__(self, knowledge_manager: Any):
        self._km = knowledge_manager
        self._sources: Dict[str, SyncSource] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._check_interval = 10.0  # check every 10 seconds

    def add_source(self, source: SyncSource) -> None:
        """Register a syncable source."""
        with self._lock:
            self._sources[source.name] = source

        # Auto-start if we have interval sources
        if source.refresh_interval > 0 and not self._running:
            self.start()

    def remove_source(self, name: str) -> None:
        """Remove a source from sync scheduling."""
        with self._lock:
            self._sources.pop(name, None)

    def get_source(self, name: str) -> Optional[SyncSource]:
        """Get a sync source by name."""
        with self._lock:
            return self._sources.get(name)

    def list_sources(self) -> List[SyncSource]:
        """List all sync sources."""
        with self._lock:
            return list(self._sources.values())

    def refresh(self, name: str) -> SyncResult:
        """
        Manually trigger a refresh for a specific source.

        Returns SyncResult with refresh status.
        """
        with self._lock:
            source = self._sources.get(name)

        if source is None:
            return SyncResult(
                source_name=name,
                success=False,
                error=f"Unknown source: {name}",
            )

        return self._sync_source(source)

    def refresh_all(self) -> List[SyncResult]:
        """Refresh all sources."""
        with self._lock:
            sources = list(self._sources.values())
        return [self._sync_source(s) for s in sources if s.enabled]

    def _sync_source(self, source: SyncSource) -> SyncResult:
        """Perform sync for a single source."""
        start = time.time()

        try:
            # Fetch data based on source type
            data = self._fetch_data(source)

            if data is None:
                return SyncResult(
                    source_name=source.name,
                    success=False,
                    error="No data returned from source",
                    duration_ms=(time.time() - start) * 1000,
                )

            # Check if data changed
            data_str = str(data)
            data_hash = hashlib.md5(
                data_str.encode("utf-8", errors="replace")
            ).hexdigest()

            changed = data_hash != source.last_hash
            if source.change_detection and not changed:
                return SyncResult(
                    source_name=source.name,
                    success=True,
                    records_count=0,
                    changed=False,
                    duration_ms=(time.time() - start) * 1000,
                )

            # Update knowledge manager
            record_count = self._ingest_data(source.name, data)

            # Update source state
            source.last_sync = time.time()
            source.last_hash = data_hash
            source.sync_count += 1

            # Callback
            if source.on_refresh:
                try:
                    source.on_refresh(record_count)
                except Exception:
                    pass

            return SyncResult(
                source_name=source.name,
                success=True,
                records_count=record_count,
                changed=True,
                duration_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            if source.on_error:
                try:
                    source.on_error(e)
                except Exception:
                    pass

            return SyncResult(
                source_name=source.name,
                success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def _fetch_data(self, source: SyncSource) -> Any:
        """Fetch data from a source."""
        # Custom fetch function
        if source.fetch_fn:
            return source.fetch_fn()

        # Database
        if source.source_type == "database" and source.connection and source.query:
            cursor = source.connection.cursor()
            cursor.execute(source.query)
            rows = cursor.fetchall()
            return rows

        # API
        if source.source_type == "api" and source.url:
            import urllib.request
            import json

            req = urllib.request.Request(
                source.url, headers=source.headers, method=source.method
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())

        # File
        if source.source_type == "file" and source.file_path:
            with open(source.file_path, "r") as f:
                return f.read()

        return None

    def _ingest_data(self, name: str, data: Any) -> int:
        """Ingest fetched data into the knowledge manager."""
        if isinstance(data, str):
            self._km.add_data(name, data)
            return 1
        elif isinstance(data, list):
            for i, item in enumerate(data):
                self._km.add_data(f"{name}_{i}", str(item))
            return len(data)
        elif isinstance(data, dict):
            for key, value in data.items():
                self._km.add_data(f"{name}_{key}", str(value))
            return len(data)
        else:
            self._km.add_data(name, str(data))
            return 1

    def start(self) -> None:
        """Start the background sync scheduler."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="infrarely-sync-scheduler",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background sync scheduler."""
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None

    def _scheduler_loop(self) -> None:
        """Background loop that checks and refreshes due sources."""
        while self._running and not self._stop_event.is_set():
            with self._lock:
                sources = [s for s in self._sources.values() if s.enabled and s.is_due]

            for source in sources:
                if self._stop_event.is_set():
                    break
                try:
                    self._sync_source(source)
                except Exception:
                    pass

            self._stop_event.wait(timeout=self._check_interval)

    @property
    def running(self) -> bool:
        return self._running
