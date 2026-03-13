"""
runtime/lifecycle_manager.py — Module 9: Lifecycle Manager
═══════════════════════════════════════════════════════════════════════════════
Manages the full lifecycle of agents: spawn, pause, resume, terminate, restart.
Like systemd or init for the agent OS.

Gap solutions:
  Gap 8  — Agent crash recovery: detect crashed agents, restart with
           state restoration, reassign their pending tasks
  Gap 9  — Lifecycle leaks: idle agents auto-terminated, zombie detection,
           periodic GC sweep

Lifecycle states:
  SPAWNING → ACTIVE → BUSY → (PAUSED) → DRAINING → TERMINATED
                ↓                                         ↑
             CRASHED ──────── (restart) ──────────> ACTIVE
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set
from infrarely.observability import logger


class LifecycleEvent(Enum):
    SPAWNED = auto()
    ACTIVATED = auto()
    PAUSED = auto()
    RESUMED = auto()
    DRAINING = auto()
    TERMINATED = auto()
    CRASHED = auto()
    RESTARTED = auto()
    IDLE_TERMINATED = auto()
    GC_COLLECTED = auto()


@dataclass
class LifecycleRecord:
    """Tracks lifecycle events for an agent."""

    agent_id: str
    events: List[Dict[str, Any]] = field(default_factory=list)
    spawn_count: int = 0
    crash_count: int = 0
    last_activity: float = field(default_factory=time.time)
    idle_since: float = 0.0
    max_restarts: int = 3
    auto_restart: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_event(self, event: LifecycleEvent, details: str = ""):
        self.events.append(
            {
                "event": event.name,
                "details": details,
                "timestamp": time.time(),
            }
        )
        if len(self.events) > 100:
            self.events = self.events[-50:]
        self.last_activity = time.time()

    @property
    def is_idle(self) -> bool:
        return self.idle_since > 0

    @property
    def idle_duration_ms(self) -> float:
        if self.idle_since <= 0:
            return 0.0
        return (time.time() - self.idle_since) * 1000


class LifecycleManager:
    """
    Manages agent lifecycles with crash recovery and leak prevention.

    Invariants:
      • Max agents: controlled by resource_isolation
      • Idle timeout: auto-terminate after configurable period (Gap 9)
      • Max restarts: agents with >N crashes don't auto-restart (Gap 8)
      • GC sweep: runs periodically to clean up zombies (Gap 9)
    """

    DEFAULT_IDLE_TIMEOUT_MS = 300_000.0  # 5 minutes
    GC_INTERVAL_MS = 60_000.0  # 1 minute

    def __init__(
        self,
        idle_timeout_ms: float = DEFAULT_IDLE_TIMEOUT_MS,
    ):
        self._records: Dict[str, LifecycleRecord] = {}
        self._idle_timeout_ms = idle_timeout_ms
        self._last_gc = time.time()
        self._total_spawns = 0
        self._total_terminations = 0
        self._total_crashes = 0
        self._total_restarts = 0
        self._total_gc_collections = 0

    # ── Spawn ─────────────────────────────────────────────────────────────────
    def spawn(
        self,
        agent_id: str,
        auto_restart: bool = True,
        max_restarts: int = 3,
        metadata: Dict[str, Any] = None,
    ) -> LifecycleRecord:
        """
        Register an agent lifecycle.
        Actually spawning the agent happens at the registry/core level;
        this tracks the lifecycle state.
        """
        record = LifecycleRecord(
            agent_id=agent_id,
            auto_restart=auto_restart,
            max_restarts=max_restarts,
            metadata=metadata or {},
        )
        record.spawn_count = 1
        record.add_event(LifecycleEvent.SPAWNED)
        self._records[agent_id] = record
        self._total_spawns += 1

        logger.info(f"Lifecycle: spawned '{agent_id}'")
        return record

    # ── Activate ──────────────────────────────────────────────────────────────
    def activate(self, agent_id: str) -> bool:
        record = self._records.get(agent_id)
        if not record:
            return False
        record.add_event(LifecycleEvent.ACTIVATED)
        record.idle_since = 0  # no longer idle
        return True

    # ── Pause / Resume ────────────────────────────────────────────────────────
    def pause(self, agent_id: str, reason: str = "") -> bool:
        record = self._records.get(agent_id)
        if not record:
            return False
        record.add_event(LifecycleEvent.PAUSED, reason)
        record.idle_since = time.time()
        logger.info(f"Lifecycle: paused '{agent_id}' — {reason}")
        return True

    def resume(self, agent_id: str) -> bool:
        record = self._records.get(agent_id)
        if not record:
            return False
        record.add_event(LifecycleEvent.RESUMED)
        record.idle_since = 0
        return True

    # ── Terminate ─────────────────────────────────────────────────────────────
    def terminate(self, agent_id: str, reason: str = "") -> bool:
        """Gracefully terminate an agent."""
        record = self._records.get(agent_id)
        if not record:
            return False
        record.add_event(LifecycleEvent.TERMINATED, reason)
        self._total_terminations += 1
        logger.info(f"Lifecycle: terminated '{agent_id}' — {reason}")
        return True

    # ── Crash detection & recovery (Gap 8) ────────────────────────────────────
    def report_crash(self, agent_id: str, error: str = "") -> Dict[str, Any]:
        """
        Report an agent crash. Decides whether to auto-restart.
        Gap 8: crash recovery with restart limits and state restoration.
        Returns action taken.
        """
        record = self._records.get(agent_id)
        if not record:
            return {"action": "ignored", "reason": "unknown agent"}

        record.crash_count += 1
        record.add_event(LifecycleEvent.CRASHED, error)
        self._total_crashes += 1

        logger.error(
            f"Lifecycle: '{agent_id}' CRASHED (#{record.crash_count}) — {error}"
        )

        # Check auto-restart eligibility (Gap 8)
        if record.auto_restart and record.crash_count <= record.max_restarts:
            record.spawn_count += 1
            record.add_event(
                LifecycleEvent.RESTARTED, f"Auto-restart #{record.crash_count}"
            )
            self._total_restarts += 1
            logger.info(
                f"Lifecycle: auto-restarting '{agent_id}' "
                f"(crash #{record.crash_count}/{record.max_restarts})"
            )
            return {
                "action": "restart",
                "crash_count": record.crash_count,
                "max_restarts": record.max_restarts,
            }
        else:
            record.add_event(
                LifecycleEvent.TERMINATED,
                f"Max restarts ({record.max_restarts}) exceeded",
            )
            self._total_terminations += 1
            logger.error(
                f"Lifecycle: '{agent_id}' permanently terminated — "
                f"exceeded max restarts ({record.max_restarts})"
            )
            return {
                "action": "terminate",
                "crash_count": record.crash_count,
                "reason": "max restarts exceeded",
            }

    # ── Mark activity ─────────────────────────────────────────────────────────
    def heartbeat(self, agent_id: str):
        """Record agent activity to prevent idle termination."""
        record = self._records.get(agent_id)
        if record:
            record.last_activity = time.time()
            record.idle_since = 0

    def mark_idle(self, agent_id: str):
        """Mark agent as idle (start idle timer)."""
        record = self._records.get(agent_id)
        if record and not record.idle_since:
            record.idle_since = time.time()

    # ── GC sweep (Gap 9: lifecycle leaks) ─────────────────────────────────────
    def gc_sweep(self) -> List[str]:
        """
        Garbage collect idle and zombie agents (Gap 9).
        Returns list of agent_ids that should be terminated.
        """
        self._last_gc = time.time()
        to_terminate = []

        for agent_id, record in self._records.items():
            # Check idle timeout (Gap 9)
            if record.idle_since > 0:
                if record.idle_duration_ms > self._idle_timeout_ms:
                    record.add_event(
                        LifecycleEvent.IDLE_TERMINATED,
                        f"Idle for {record.idle_duration_ms:.0f}ms",
                    )
                    to_terminate.append(agent_id)
                    self._total_gc_collections += 1
                    logger.info(
                        f"Lifecycle GC: '{agent_id}' idle-terminated "
                        f"({record.idle_duration_ms:.0f}ms idle)"
                    )

            # Check zombie: no activity for 2× idle timeout
            elif record.last_activity > 0:
                stale_ms = (time.time() - record.last_activity) * 1000
                if stale_ms > self._idle_timeout_ms * 2:
                    record.add_event(
                        LifecycleEvent.GC_COLLECTED,
                        f"Zombie detected — no activity for {stale_ms:.0f}ms",
                    )
                    to_terminate.append(agent_id)
                    self._total_gc_collections += 1
                    logger.warn(
                        f"Lifecycle GC: '{agent_id}' zombie-collected "
                        f"({stale_ms:.0f}ms stale)"
                    )

        return to_terminate

    def should_run_gc(self) -> bool:
        """Check if enough time has passed for GC."""
        return (time.time() - self._last_gc) * 1000 > self.GC_INTERVAL_MS

    # ── Drain (graceful shutdown) ─────────────────────────────────────────────
    def drain(self, agent_id: str) -> bool:
        """Put agent in draining mode — finish current tasks, accept no new ones."""
        record = self._records.get(agent_id)
        if not record:
            return False
        record.add_event(LifecycleEvent.DRAINING)
        logger.info(f"Lifecycle: '{agent_id}' draining")
        return True

    # ── Cleanup after termination ─────────────────────────────────────────────
    def remove(self, agent_id: str) -> Optional[LifecycleRecord]:
        """Remove lifecycle record after agent is fully cleaned up."""
        return self._records.pop(agent_id, None)

    # ── Query ─────────────────────────────────────────────────────────────────
    def get_record(self, agent_id: str) -> Optional[LifecycleRecord]:
        return self._records.get(agent_id)

    def all_records(self) -> Dict[str, LifecycleRecord]:
        return dict(self._records)

    def idle_agents(self) -> List[str]:
        return [aid for aid, rec in self._records.items() if rec.idle_since > 0]

    def crashed_agents(self) -> List[str]:
        return [
            aid
            for aid, rec in self._records.items()
            if rec.events and rec.events[-1].get("event") == "CRASHED"
        ]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_agents_tracked": len(self._records),
            "total_spawns": self._total_spawns,
            "total_terminations": self._total_terminations,
            "total_crashes": self._total_crashes,
            "total_restarts": self._total_restarts,
            "total_gc_collections": self._total_gc_collections,
            "idle_agents": self.idle_agents(),
            "crashed_agents": self.crashed_agents(),
            "agents": {
                aid: {
                    "spawn_count": rec.spawn_count,
                    "crash_count": rec.crash_count,
                    "idle": rec.is_idle,
                    "idle_ms": round(rec.idle_duration_ms, 0) if rec.is_idle else 0,
                    "events": len(rec.events),
                    "last_event": rec.events[-1]["event"] if rec.events else "none",
                }
                for aid, rec in self._records.items()
            },
        }
