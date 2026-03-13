"""
runtime/priority_scheduler.py — GAP 3: Agent Priority Scheduling
═══════════════════════════════════════════════════════════════════════════════
Priority-queue based scheduling with fair scheduling and starvation protection.

Features:
  • Min-heap priority queue (lower number = higher priority)
  • Fair share scheduling: guaranteed minimum slice per agent
  • Starvation protection: priority boost for agents waiting too long
  • Aging mechanism: priority improves over time if not scheduled
  • Per-agent scheduling stats
"""

from __future__ import annotations

import heapq
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PriorityTask:
    """A task with priority for the priority queue."""

    task_id: str = field(default_factory=lambda: f"ptask_{uuid.uuid4().hex[:6]}")
    label: str = ""
    agent_id: str = ""
    priority: int = 5  # 1=highest, 10=lowest
    effective_priority: int = 5  # after aging adjustments
    enqueued_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    payload: Dict[str, Any] = field(default_factory=dict)
    status: str = "queued"  # queued | running | done | failed

    @property
    def wait_time_ms(self) -> float:
        if self.started_at:
            return (self.started_at - self.enqueued_at) * 1000
        return (time.time() - self.enqueued_at) * 1000

    def __lt__(self, other: "PriorityTask") -> bool:
        """For heap comparison — lower effective_priority = higher urgency."""
        return self.effective_priority < other.effective_priority


@dataclass
class AgentSchedulingStats:
    """Per-agent scheduling statistics."""

    agent_id: str
    tasks_assigned: int = 0
    total_wait_ms: float = 0.0
    last_assigned_at: float = 0.0
    starvation_boosts: int = 0
    fair_share_tokens: int = 0

    @property
    def avg_wait_ms(self) -> float:
        if self.tasks_assigned == 0:
            return 0.0
        return self.total_wait_ms / self.tasks_assigned


class PriorityScheduler:
    """
    Priority-queue scheduler with fairness guarantees.

    Invariants:
      • Tasks are dispatched in priority order (lowest number first)
      • Agents get at least 1 task per FAIR_SHARE_WINDOW dispatches
      • Tasks waiting > STARVATION_THRESHOLD_MS get priority boost
      • Aging: priority improves by 1 every AGING_INTERVAL_MS
      • Max queue: 500
    """

    MAX_QUEUE = 500
    FAIR_SHARE_WINDOW = 10  # per agent: at least 1 task per N dispatches
    STARVATION_THRESHOLD_MS = 5000.0  # 5s wait → boost
    AGING_INTERVAL_MS = 2000.0  # boost priority by 1 every 2s
    MIN_PRIORITY = 1
    MAX_PRIORITY = 10

    def __init__(self):
        self._queue: List[PriorityTask] = []  # min-heap
        self._running: Dict[str, PriorityTask] = {}
        self._completed: List[PriorityTask] = []
        self._agent_stats: Dict[str, AgentSchedulingStats] = {}
        self._dispatch_count = 0
        self._total_starvation_boosts = 0

    # ── Enqueue ───────────────────────────────────────────────────────────────

    def enqueue(
        self,
        label: str = "",
        priority: int = 5,
        agent_id: str = "",
        payload: Dict[str, Any] = None,
    ) -> PriorityTask:
        """Add a task to the priority queue."""
        if len(self._queue) >= self.MAX_QUEUE:
            raise ValueError(f"Priority queue full ({self.MAX_QUEUE})")

        task = PriorityTask(
            label=label,
            priority=max(self.MIN_PRIORITY, min(self.MAX_PRIORITY, priority)),
            effective_priority=max(self.MIN_PRIORITY, min(self.MAX_PRIORITY, priority)),
            agent_id=agent_id,
            payload=payload or {},
        )
        heapq.heappush(self._queue, task)
        return task

    # ── Aging & starvation protection ─────────────────────────────────────────

    def apply_aging(self):
        """
        Boost effective priority of tasks that have been waiting too long.
        Prevents starvation of low-priority tasks.
        """
        now = time.time()
        changed = False
        for task in self._queue:
            if task.status != "queued":
                continue
            wait_ms = (now - task.enqueued_at) * 1000
            if wait_ms > self.STARVATION_THRESHOLD_MS:
                age_steps = int(wait_ms / self.AGING_INTERVAL_MS)
                new_prio = max(self.MIN_PRIORITY, task.priority - age_steps)
                if new_prio < task.effective_priority:
                    task.effective_priority = new_prio
                    self._total_starvation_boosts += 1
                    changed = True
        if changed:
            heapq.heapify(self._queue)

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def dispatch(self, agent_id: str = "") -> Optional[PriorityTask]:
        """
        Pop the highest-priority task from the queue.
        If agent_id is specified, applies fairness logic.
        """
        self.apply_aging()

        if not self._queue:
            return None

        # Fair share check: if agent hasn't been served recently, prefer their tasks
        if agent_id:
            stats = self._agent_stats.get(agent_id)
            if (
                stats
                and (self._dispatch_count - stats.tasks_assigned)
                > self.FAIR_SHARE_WINDOW
            ):
                # Look for a task for this agent
                for i, task in enumerate(self._queue):
                    if task.agent_id == agent_id and task.status == "queued":
                        task = self._queue.pop(i)
                        heapq.heapify(self._queue)
                        return self._assign(task, agent_id)

        # Normal: pop highest priority
        task = heapq.heappop(self._queue)
        return self._assign(task, agent_id or task.agent_id)

    def _assign(self, task: PriorityTask, agent_id: str) -> PriorityTask:
        task.status = "running"
        task.started_at = time.time()
        task.agent_id = agent_id
        self._running[task.task_id] = task
        self._dispatch_count += 1

        # Update agent stats
        if agent_id not in self._agent_stats:
            self._agent_stats[agent_id] = AgentSchedulingStats(agent_id=agent_id)
        stats = self._agent_stats[agent_id]
        stats.tasks_assigned += 1
        stats.total_wait_ms += task.wait_time_ms
        stats.last_assigned_at = time.time()

        return task

    # ── Complete ──────────────────────────────────────────────────────────────

    def complete(self, task_id: str, success: bool = True) -> bool:
        task = self._running.pop(task_id, None)
        if not task:
            return False
        task.status = "done" if success else "failed"
        task.completed_at = time.time()
        self._completed.append(task)
        if len(self._completed) > 500:
            self._completed = self._completed[-250:]
        return True

    # ── Fair share reporting ──────────────────────────────────────────────────

    def starvation_report(self) -> List[Dict[str, Any]]:
        """Identify agents that haven't been served fairly."""
        starved = []
        for agent_id, stats in self._agent_stats.items():
            gap = self._dispatch_count - stats.tasks_assigned
            if gap > self.FAIR_SHARE_WINDOW:
                starved.append(
                    {
                        "agent_id": agent_id,
                        "tasks_assigned": stats.tasks_assigned,
                        "dispatch_gap": gap,
                        "avg_wait_ms": round(stats.avg_wait_ms, 1),
                    }
                )
        return starved

    # ── Query ─────────────────────────────────────────────────────────────────

    def queue_depth(self) -> int:
        return len(self._queue)

    def running_count(self) -> int:
        return len(self._running)

    def get_stats(self, agent_id: str) -> Optional[Dict[str, Any]]:
        stats = self._agent_stats.get(agent_id)
        if not stats:
            return None
        return {
            "tasks_assigned": stats.tasks_assigned,
            "avg_wait_ms": round(stats.avg_wait_ms, 1),
            "starvation_boosts": stats.starvation_boosts,
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "queue_depth": self.queue_depth(),
            "running": self.running_count(),
            "total_dispatched": self._dispatch_count,
            "completed": len(self._completed),
            "starvation_boosts": self._total_starvation_boosts,
            "agents": {
                aid: {
                    "assigned": s.tasks_assigned,
                    "avg_wait_ms": round(s.avg_wait_ms, 1),
                }
                for aid, s in self._agent_stats.items()
            },
        }
