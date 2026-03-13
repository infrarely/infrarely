"""
runtime/state_persistence.py — GAP 5: Agent State Persistence
═══════════════════════════════════════════════════════════════════════════════
State snapshots, task progress checkpoints, and memory checkpoints for
durable agent state across crashes and restarts.

Features:
  • Snapshot capture: full agent state at a point in time
  • Task progress checkpoints: save partial work
  • Memory checkpoints: save working memory state
  • Restore: reload state from latest checkpoint
  • Checkpoint history with configurable retention
  • In-memory store (can be extended to disk/DB)
"""

from __future__ import annotations

import copy
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class StateSnapshot:
    """A point-in-time snapshot of an agent's state."""

    snapshot_id: str = field(default_factory=lambda: f"snap_{uuid.uuid4().hex[:6]}")
    agent_id: str = ""
    snapshot_type: str = "full"  # full | task | memory
    state_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    size_bytes: int = 0

    def compute_size(self) -> int:
        """Rough estimate of snapshot size."""
        self.size_bytes = len(str(self.state_data))
        return self.size_bytes


@dataclass
class TaskCheckpoint:
    """Checkpoint for an in-progress task."""

    checkpoint_id: str = field(default_factory=lambda: f"cp_{uuid.uuid4().hex[:6]}")
    agent_id: str = ""
    task_id: str = ""
    step: int = 0
    total_steps: int = 0
    partial_result: Any = None
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @property
    def progress_pct(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return (self.step / self.total_steps) * 100


@dataclass
class MemoryCheckpoint:
    """Checkpoint for working memory contents."""

    checkpoint_id: str = field(default_factory=lambda: f"memcp_{uuid.uuid4().hex[:6]}")
    agent_id: str = ""
    memory_data: Dict[str, Any] = field(default_factory=dict)
    turn_count: int = 0
    created_at: float = field(default_factory=time.time)


class StatePersistence:
    """
    Manages state persistence for all agents.

    Invariants:
      • Max snapshots per agent: 20 (FIFO eviction)
      • Max total snapshots: 500
      • Checkpoints are immutable once created
      • Restore always uses most recent snapshot of requested type
    """

    MAX_SNAPSHOTS_PER_AGENT = 20
    MAX_TOTAL_SNAPSHOTS = 500
    MAX_CHECKPOINTS_PER_TASK = 10

    def __init__(self):
        self._snapshots: Dict[str, List[StateSnapshot]] = defaultdict(list)
        self._task_checkpoints: Dict[str, List[TaskCheckpoint]] = defaultdict(list)
        self._memory_checkpoints: Dict[str, List[MemoryCheckpoint]] = defaultdict(list)
        self._total_snapshots = 0
        self._total_restores = 0

    # ── State snapshots ───────────────────────────────────────────────────────

    def save_snapshot(
        self,
        agent_id: str,
        state_data: Dict[str, Any],
        snapshot_type: str = "full",
        metadata: Dict[str, Any] = None,
    ) -> StateSnapshot:
        """Save a state snapshot for an agent."""
        snap = StateSnapshot(
            agent_id=agent_id,
            snapshot_type=snapshot_type,
            state_data=copy.deepcopy(state_data),
            metadata=metadata or {},
        )
        snap.compute_size()

        agent_snaps = self._snapshots[agent_id]
        agent_snaps.append(snap)
        self._total_snapshots += 1

        # Evict oldest if over per-agent limit
        if len(agent_snaps) > self.MAX_SNAPSHOTS_PER_AGENT:
            self._snapshots[agent_id] = agent_snaps[-self.MAX_SNAPSHOTS_PER_AGENT :]

        # Global limit
        total = sum(len(s) for s in self._snapshots.values())
        if total > self.MAX_TOTAL_SNAPSHOTS:
            self._evict_oldest_global()

        return snap

    def restore_snapshot(
        self, agent_id: str, snapshot_type: str = "full"
    ) -> Optional[Dict[str, Any]]:
        """Restore the most recent snapshot of a given type."""
        snaps = self._snapshots.get(agent_id, [])
        for snap in reversed(snaps):
            if snap.snapshot_type == snapshot_type:
                self._total_restores += 1
                return copy.deepcopy(snap.state_data)
        return None

    def get_snapshots(self, agent_id: str) -> List[StateSnapshot]:
        return list(self._snapshots.get(agent_id, []))

    # ── Task checkpoints ──────────────────────────────────────────────────────

    def save_task_checkpoint(
        self,
        agent_id: str,
        task_id: str,
        step: int,
        total_steps: int = 0,
        partial_result: Any = None,
        context: Dict[str, Any] = None,
    ) -> TaskCheckpoint:
        """Save a task progress checkpoint."""
        cp = TaskCheckpoint(
            agent_id=agent_id,
            task_id=task_id,
            step=step,
            total_steps=total_steps,
            partial_result=copy.deepcopy(partial_result) if partial_result else None,
            context=copy.deepcopy(context) if context else {},
        )
        key = f"{agent_id}::{task_id}"
        cps = self._task_checkpoints[key]
        cps.append(cp)
        if len(cps) > self.MAX_CHECKPOINTS_PER_TASK:
            self._task_checkpoints[key] = cps[-self.MAX_CHECKPOINTS_PER_TASK :]
        return cp

    def restore_task_checkpoint(
        self, agent_id: str, task_id: str
    ) -> Optional[TaskCheckpoint]:
        """Restore the latest checkpoint for a task."""
        key = f"{agent_id}::{task_id}"
        cps = self._task_checkpoints.get(key, [])
        if cps:
            self._total_restores += 1
            return cps[-1]
        return None

    def task_progress(self, agent_id: str, task_id: str) -> float:
        """Get task progress percentage."""
        cp = self.restore_task_checkpoint(agent_id, task_id)
        if cp:
            return cp.progress_pct
        return 0.0

    # ── Memory checkpoints ────────────────────────────────────────────────────

    def save_memory_checkpoint(
        self,
        agent_id: str,
        memory_data: Dict[str, Any],
        turn_count: int = 0,
    ) -> MemoryCheckpoint:
        """Save a memory checkpoint."""
        mcp = MemoryCheckpoint(
            agent_id=agent_id,
            memory_data=copy.deepcopy(memory_data),
            turn_count=turn_count,
        )
        self._memory_checkpoints[agent_id].append(mcp)
        if len(self._memory_checkpoints[agent_id]) > self.MAX_SNAPSHOTS_PER_AGENT:
            self._memory_checkpoints[agent_id] = self._memory_checkpoints[agent_id][
                -self.MAX_SNAPSHOTS_PER_AGENT :
            ]
        return mcp

    def restore_memory_checkpoint(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Restore the latest memory checkpoint."""
        mcps = self._memory_checkpoints.get(agent_id, [])
        if mcps:
            self._total_restores += 1
            return copy.deepcopy(mcps[-1].memory_data)
        return None

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def cleanup_agent(self, agent_id: str) -> int:
        """Remove all persistence data for an agent. Returns count removed."""
        count = 0
        count += len(self._snapshots.pop(agent_id, []))
        count += len(self._memory_checkpoints.pop(agent_id, []))
        keys_to_remove = [
            k for k in self._task_checkpoints if k.startswith(f"{agent_id}::")
        ]
        for k in keys_to_remove:
            count += len(self._task_checkpoints.pop(k, []))
        return count

    def _evict_oldest_global(self):
        """Evict oldest snapshots globally until under limit."""
        all_snaps = []
        for agent_id, snaps in self._snapshots.items():
            for s in snaps:
                all_snaps.append((s.created_at, agent_id, s.snapshot_id))
        all_snaps.sort()
        # Remove oldest 10%
        remove_count = max(1, len(all_snaps) // 10)
        to_remove = set()
        for _, agent_id, snap_id in all_snaps[:remove_count]:
            to_remove.add((agent_id, snap_id))
        for agent_id in list(self._snapshots):
            self._snapshots[agent_id] = [
                s
                for s in self._snapshots[agent_id]
                if (agent_id, s.snapshot_id) not in to_remove
            ]

    # ── Query ─────────────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        total_snaps = sum(len(s) for s in self._snapshots.values())
        total_cps = sum(len(c) for c in self._task_checkpoints.values())
        total_mcps = sum(len(m) for m in self._memory_checkpoints.values())
        return {
            "total_snapshots": total_snaps,
            "total_task_checkpoints": total_cps,
            "total_memory_checkpoints": total_mcps,
            "total_restores": self._total_restores,
            "agents_tracked": len(self._snapshots),
            "per_agent": {
                agent_id: len(snaps) for agent_id, snaps in self._snapshots.items()
            },
        }
