"""
agent/scheduler.py  — Gap 8: Execution Scheduler
Priority task queue for future concurrent capability workflows.
Currently operates synchronously; async workers can be added later.
"""
from __future__ import annotations
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional
import infrarely.core.app_config as config
from infrarely.observability import logger


class Priority(IntEnum):
    CRITICAL      = 0
    USER_REQUEST  = 1
    BACKGROUND    = 5
    MAINTENANCE   = 9


@dataclass(order=True)
class ScheduledTask:
    priority:   int
    task_id:    str         = field(compare=False)
    fn:         Callable    = field(compare=False)
    args:       tuple       = field(compare=False, default_factory=tuple)
    kwargs:     dict        = field(compare=False, default_factory=dict)
    submitted:  float       = field(compare=False, default_factory=time.monotonic)
    label:      str         = field(compare=False, default="task")


@dataclass
class TaskResult:
    task_id:    str
    label:      str
    success:    bool
    value:      Any   = None
    error:      str   = ""
    duration_ms: float = 0.0


class ExecutionScheduler:
    """
    Priority queue for task execution.
    submit() enqueues; run_next() executes the highest-priority task.
    run_all() drains the queue synchronously (used in tests and sequential mode).
    """

    def __init__(self, max_queue: int = None):
        self._queue: queue.PriorityQueue = queue.PriorityQueue(
            maxsize=max_queue or getattr(config, "SCHEDULER_MAX_QUEUE", 50)
        )
        self._results: Dict[str, TaskResult] = {}
        self._lock = threading.Lock()

    def submit(self, fn: Callable, *args,
               priority: Priority = Priority.USER_REQUEST,
               label: str = "task", **kwargs) -> str:
        task_id = uuid.uuid4().hex[:8]
        task = ScheduledTask(
            priority=priority.value, task_id=task_id,
            fn=fn, args=args, kwargs=kwargs, label=label,
        )
        try:
            self._queue.put_nowait(task)
            logger.debug(f"Scheduler: queued '{label}' [{task_id}] priority={priority.name}")
        except queue.Full:
            logger.warn(f"Scheduler: queue full — dropping '{label}' [{task_id}]")
        return task_id

    def run_next(self) -> Optional[TaskResult]:
        try:
            task = self._queue.get_nowait()
        except queue.Empty:
            return None
        return self._execute(task)

    def run_all(self) -> List[TaskResult]:
        results = []
        while not self._queue.empty():
            r = self.run_next()
            if r:
                results.append(r)
        return results

    def _execute(self, task: ScheduledTask) -> TaskResult:
        t0 = time.monotonic()
        try:
            value   = task.fn(*task.args, **task.kwargs)
            elapsed = (time.monotonic() - t0) * 1000
            result  = TaskResult(task_id=task.task_id, label=task.label,
                                 success=True, value=value, duration_ms=elapsed)
            logger.debug(f"Scheduler: '{task.label}' completed in {elapsed:.1f}ms")
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            result  = TaskResult(task_id=task.task_id, label=task.label,
                                 success=False, error=str(e), duration_ms=elapsed)
            logger.error(f"Scheduler: '{task.label}' failed: {e}")
        with self._lock:
            self._results[task.task_id] = result
        return result

    def get_result(self, task_id: str) -> Optional[TaskResult]:
        with self._lock:
            return self._results.get(task_id)

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()