"""
runtime/agent_scheduler.py — Module 2: Agent Scheduler
═══════════════════════════════════════════════════════════════════════════════
Controls which agent runs which task.  Like a process scheduler in an OS.

Scheduling strategies:
  • PRIORITY       — highest priority agent goes first
  • ROUND_ROBIN    — fair rotation across agents
  • CAPABILITY     — best-matching agent for the task's required capability

Gap solutions:
  Gap 1  — Deadlock detection: dependency graph cycle check + timeout
  Gap 3  — Agent starvation: fairness tokens ensure all agents run eventually

Design:
  Tasks arrive → enqueued → scheduler picks agent → dispatches.
  Agents never call each other directly: Agent → Bus → Scheduler → Target.
"""

from __future__ import annotations
import time
import uuid
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from infrarely.observability import logger


class SchedulingStrategy(Enum):
    PRIORITY = auto()
    ROUND_ROBIN = auto()
    CAPABILITY = auto()


class TaskStatus(Enum):
    QUEUED = auto()
    ASSIGNED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    TIMED_OUT = auto()
    REASSIGNED = auto()


@dataclass
class AgentTask:
    """A unit of work for the scheduler."""

    task_id: str
    intent: str
    payload: Dict[str, Any] = field(default_factory=dict)
    required_capability: str = ""
    required_role: str = ""
    priority: int = 5  # 1–10
    submitter: str = ""  # agent_id or "user"
    assigned_to: str = ""  # agent_id
    status: TaskStatus = TaskStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    timeout_ms: float = 30_000.0  # 30s default
    result: Any = None
    error: str = ""
    retries: int = 0
    max_retries: int = 2
    depends_on: List[str] = field(default_factory=list)  # task_ids

    @property
    def elapsed_ms(self) -> float:
        if self.started_at:
            end = self.completed_at or time.time()
            return (end - self.started_at) * 1000
        return 0.0

    @property
    def is_timed_out(self) -> bool:
        if self.status == TaskStatus.RUNNING and self.started_at:
            return (time.time() - self.started_at) * 1000 > self.timeout_ms
        return False


class AgentScheduler:
    """
    Multi-agent task scheduler with deadlock detection and fairness.

    Invariants:
      • Max queue depth: 200 tasks
      • Deadlock detection runs on every dispatch cycle
      • Fairness tokens: each agent gets at least 1 task per 10 dispatches
    """

    MAX_QUEUE = 200
    FAIRNESS_WINDOW = 10  # ensure each agent runs at least once per N dispatches

    def __init__(self, strategy: SchedulingStrategy = SchedulingStrategy.CAPABILITY):
        self._strategy = strategy
        self._queue: deque[AgentTask] = deque()
        self._active: Dict[str, AgentTask] = {}  # task_id → running task
        self._completed: List[AgentTask] = []
        self._dispatch_count = 0
        self._agent_dispatch_count: Dict[str, int] = defaultdict(int)
        self._fairness_tokens: Dict[str, int] = defaultdict(int)
        self._dependency_graph: Dict[str, Set[str]] = {}  # task_id → depends_on set

    # ── Submit task ───────────────────────────────────────────────────────────
    def submit(
        self,
        intent: str,
        payload: Dict[str, Any] = None,
        required_capability: str = "",
        required_role: str = "",
        priority: int = 5,
        submitter: str = "user",
        timeout_ms: float = 30_000.0,
        depends_on: List[str] = None,
    ) -> AgentTask:
        """
        Submit a task to the scheduler queue.
        Raises ValueError if queue is full (Gap 2: backpressure).
        """
        if len(self._queue) >= self.MAX_QUEUE:
            raise ValueError(
                f"Scheduler queue full ({self.MAX_QUEUE} tasks). "
                "Backpressure applied — retry later."
            )

        task = AgentTask(
            task_id=f"task_{uuid.uuid4().hex[:8]}",
            intent=intent,
            payload=payload or {},
            required_capability=required_capability,
            required_role=required_role,
            priority=priority,
            submitter=submitter,
            timeout_ms=timeout_ms,
            depends_on=depends_on or [],
        )

        # Track dependencies for deadlock detection (Gap 1)
        if task.depends_on:
            self._dependency_graph[task.task_id] = set(task.depends_on)

        self._queue.append(task)
        logger.debug(
            f"Scheduler: queued '{task.task_id}' intent='{intent}' "
            f"priority={priority} deps={task.depends_on}"
        )
        return task

    # ── Dispatch next task ────────────────────────────────────────────────────
    def dispatch(self, registry) -> Optional[Tuple[AgentTask, str]]:
        """
        Pick the next task from queue and assign to an agent.
        Returns (task, agent_id) or None if nothing to dispatch.

        Implements:
          • Dependency ordering — blocked tasks stay in queue
          • Deadlock detection (Gap 1)
          • Timeout detection for running tasks
          • Fairness tokens (Gap 3)
        """
        # Check for timed-out running tasks (Gap 1)
        self._check_timeouts()

        # Deadlock detection (Gap 1)
        if self._dependency_graph:
            deadlocked = self._detect_deadlocks()
            if deadlocked:
                self._break_deadlocks(deadlocked)

        # Find dispatchable task (dependencies satisfied)
        task = self._pick_next_task()
        if not task:
            return None

        # Find best agent
        agent_id = self._select_agent(task, registry)
        if not agent_id:
            # No available agent — put task back
            self._queue.appendleft(task)
            logger.debug(f"Scheduler: no agent available for '{task.task_id}'")
            return None

        # Assign
        task.assigned_to = agent_id
        task.status = TaskStatus.ASSIGNED
        task.started_at = time.time()
        self._active[task.task_id] = task
        self._dispatch_count += 1
        self._agent_dispatch_count[agent_id] += 1
        self._fairness_tokens[agent_id] = self._dispatch_count

        registry.set_status(
            agent_id, registry.get(agent_id).status.__class__(2)
        )  # BUSY
        logger.info(
            f"Scheduler: dispatched '{task.task_id}' → '{agent_id}' "
            f"(intent='{task.intent}')"
        )
        return task, agent_id

    # ── Complete task ─────────────────────────────────────────────────────────
    def complete(
        self, task_id: str, success: bool, result: Any = None, error: str = ""
    ):
        """Mark a task as completed or failed."""
        task = self._active.pop(task_id, None)
        if not task:
            return

        task.completed_at = time.time()
        task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        task.result = result
        task.error = error

        # Remove from dependency graph
        self._dependency_graph.pop(task_id, None)
        # Remove satisfied dependencies
        for deps in self._dependency_graph.values():
            deps.discard(task_id)

        self._completed.append(task)
        if len(self._completed) > 500:
            self._completed = self._completed[-250:]

        logger.info(
            f"Scheduler: task '{task_id}' {'completed' if success else 'failed'} "
            f"in {task.elapsed_ms:.0f}ms"
        )

    # ── Timeout detection (Gap 1) ─────────────────────────────────────────────
    def _check_timeouts(self):
        """Detect and handle timed-out tasks."""
        timed_out = [tid for tid, task in self._active.items() if task.is_timed_out]
        for tid in timed_out:
            task = self._active.pop(tid)
            task.status = TaskStatus.TIMED_OUT
            task.error = f"Timed out after {task.timeout_ms}ms"
            task.completed_at = time.time()

            # Retry if allowed (Gap 8: crash recovery)
            if task.retries < task.max_retries:
                task.retries += 1
                task.status = TaskStatus.REASSIGNED
                task.started_at = 0
                task.assigned_to = ""
                self._queue.appendleft(task)
                logger.warn(
                    f"Scheduler: '{tid}' timed out — requeueing "
                    f"(retry {task.retries}/{task.max_retries})"
                )
            else:
                self._completed.append(task)
                logger.error(f"Scheduler: '{tid}' timed out — max retries exceeded")

    # ── Deadlock detection (Gap 1) ────────────────────────────────────────────
    def _detect_deadlocks(self) -> List[str]:
        """
        Detect cycles in task dependency graph using DFS.
        Returns list of task_ids involved in a cycle, or empty list.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        colour: Dict[str, int] = {t: WHITE for t in self._dependency_graph}
        cycle_members: List[str] = []

        def dfs(tid: str) -> bool:
            colour[tid] = GRAY
            for dep in self._dependency_graph.get(tid, set()):
                if dep in colour:
                    if colour[dep] == GRAY:
                        cycle_members.append(dep)
                        cycle_members.append(tid)
                        return True
                    if colour[dep] == WHITE and dfs(dep):
                        return True
            colour[tid] = BLACK
            return False

        for tid in list(colour):
            if colour[tid] == WHITE:
                if dfs(tid):
                    return cycle_members
        return []

    def _break_deadlocks(self, deadlocked: List[str]):
        """Break deadlock by failing the lowest-priority task in the cycle."""
        candidates = []
        for tid in set(deadlocked):
            task = self._active.get(tid)
            if task:
                candidates.append(task)
        if not candidates:
            # Check queue
            for tid in set(deadlocked):
                for qt in self._queue:
                    if qt.task_id == tid:
                        candidates.append(qt)
        if candidates:
            victim = min(candidates, key=lambda t: t.priority)
            victim.status = TaskStatus.FAILED
            victim.error = "Deadlock detected — task cancelled"
            victim.completed_at = time.time()
            self._active.pop(victim.task_id, None)
            self._dependency_graph.pop(victim.task_id, None)
            self._completed.append(victim)
            logger.warn(f"Scheduler: broke deadlock by cancelling '{victim.task_id}'")

    # ── Task selection ────────────────────────────────────────────────────────
    def _pick_next_task(self) -> Optional[AgentTask]:
        """Pick next dispatchable task from queue."""
        completed_ids = {
            t.task_id for t in self._completed if t.status == TaskStatus.COMPLETED
        }
        active_ids = set(self._active.keys())

        dispatchable = []
        remaining = deque()

        for task in self._queue:
            # Check dependencies satisfied
            unmet = [d for d in task.depends_on if d not in completed_ids]
            if unmet:
                remaining.append(task)
            else:
                dispatchable.append(task)

        if not dispatchable:
            return None

        # Sort by priority (highest first)
        dispatchable.sort(key=lambda t: t.priority, reverse=True)
        chosen = dispatchable[0]

        # Rebuild queue: remaining + other dispatchable
        self._queue = remaining
        for t in dispatchable[1:]:
            self._queue.append(t)

        return chosen

    # ── Agent selection ───────────────────────────────────────────────────────
    def _select_agent(self, task: AgentTask, registry) -> Optional[str]:
        """
        Select best agent for task.
        Implements fairness (Gap 3): agents that haven't run recently get a boost.
        """
        if self._strategy == SchedulingStrategy.CAPABILITY and task.required_capability:
            agent = registry.best_agent_for_capability(task.required_capability)
            if agent:
                return agent.agent_id

        available = registry.find_available(
            capability=task.required_capability or None,
            role=task.required_role or None,
        )
        if not available:
            # Broaden search: any available agent
            available = registry.active_agents()

        if not available:
            return None

        # Fairness boost (Gap 3): agents with oldest fairness token get priority
        def _fairness_score(a):
            last_dispatch = self._fairness_tokens.get(a.agent_id, 0)
            starvation_bonus = max(
                0, self._dispatch_count - last_dispatch - self.FAIRNESS_WINDOW
            )
            return a.priority + starvation_bonus * 0.5

        ranked = sorted(available, key=_fairness_score, reverse=True)
        return ranked[0].agent_id

    # ── Query ─────────────────────────────────────────────────────────────────
    def queue_depth(self) -> int:
        return len(self._queue)

    def active_tasks(self) -> int:
        return len(self._active)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "strategy": self._strategy.name,
            "queue_depth": len(self._queue),
            "active_tasks": len(self._active),
            "total_dispatched": self._dispatch_count,
            "completed_tasks": len(self._completed),
            "agent_dispatch_counts": dict(self._agent_dispatch_count),
            "active_task_ids": list(self._active.keys()),
            "recent_completed": [
                {
                    "task_id": t.task_id,
                    "intent": t.intent,
                    "status": t.status.name,
                    "agent": t.assigned_to,
                    "elapsed_ms": round(t.elapsed_ms, 1),
                }
                for t in self._completed[-10:]
            ],
        }
