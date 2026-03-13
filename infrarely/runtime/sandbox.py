"""
aos/sandbox.py — Agent Sandboxing & Resource Isolation
═══════════════════════════════════════════════════════════════════════════════
SCALE GAP 2: Resource isolation between agents so one misbehaving agent
can't take down others.

Provides per-agent limits on:
  - Memory usage (approximate, tracked by allocation)
  - Execution time per task
  - Tool calls per task
  - Tool allow/block lists (whitelist/blacklist)
  - Network access control

Usage::

    agent = infrarely.agent("untrusted-agent",
        sandbox=aos.Sandbox(
            max_memory_mb=512,
            max_execution_time=30,
            max_tool_calls_per_task=20,
            allowed_tools=["search", "summarize"],
            blocked_tools=["execute_code"],
            network_access=False,
        )
    )

Architecture:
    Sandbox       — configuration dataclass
    SandboxGuard  — enforces limits at runtime, called by execution engine
    ResourceMeter — tracks resource usage per agent
"""

from __future__ import annotations

import resource
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set


# ═══════════════════════════════════════════════════════════════════════════════
# SANDBOX CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Sandbox:
    """
    Resource sandbox configuration for an agent.

    Parameters
    ----------
    max_memory_mb : int
        Maximum memory in megabytes (0 = unlimited).
    max_execution_time : float
        Maximum seconds per task (0 = unlimited).
    max_tool_calls_per_task : int
        Maximum tool invocations per task (0 = unlimited).
    allowed_tools : list[str], optional
        Whitelist of tool names. If set, only these tools can be called.
    blocked_tools : list[str], optional
        Blacklist of tool names. These tools are always blocked.
    network_access : bool
        Whether the agent can make outbound network calls.
    max_output_size_kb : int
        Maximum output size in kilobytes (0 = unlimited).
    max_concurrent_tasks : int
        Maximum concurrent tasks this agent can run (0 = unlimited).
    """

    max_memory_mb: int = 0
    max_execution_time: float = 0.0
    max_tool_calls_per_task: int = 0
    allowed_tools: Optional[List[str]] = None
    blocked_tools: Optional[List[str]] = None
    network_access: bool = True
    max_output_size_kb: int = 0
    max_concurrent_tasks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_memory_mb": self.max_memory_mb,
            "max_execution_time": self.max_execution_time,
            "max_tool_calls_per_task": self.max_tool_calls_per_task,
            "allowed_tools": self.allowed_tools,
            "blocked_tools": self.blocked_tools,
            "network_access": self.network_access,
            "max_output_size_kb": self.max_output_size_kb,
            "max_concurrent_tasks": self.max_concurrent_tasks,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SANDBOX VIOLATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SandboxViolation:
    """Record of a sandbox limit being hit."""

    agent_name: str = ""
    violation_type: str = (
        ""  # "memory" | "time" | "tool_calls" | "tool_blocked" | "network" | "output_size"
    )
    limit: Any = None
    actual: Any = None
    message: str = ""
    timestamp: float = field(default_factory=time.time)


# ═══════════════════════════════════════════════════════════════════════════════
# RESOURCE METER — tracks per-agent resource usage within a task
# ═══════════════════════════════════════════════════════════════════════════════


class ResourceMeter:
    """
    Tracks resource consumption for a single agent task execution.

    Created by SandboxGuard at the start of each task.
    """

    def __init__(self, agent_name: str, sandbox: Sandbox):
        self._agent_name = agent_name
        self._sandbox = sandbox
        self._start_time = time.time()
        self._tool_call_count = 0
        self._output_bytes = 0
        self._violations: List[SandboxViolation] = []
        self._active = True

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self._start_time

    @property
    def tool_calls(self) -> int:
        return self._tool_call_count

    @property
    def violations(self) -> List[SandboxViolation]:
        return list(self._violations)

    @property
    def is_within_limits(self) -> bool:
        """Check if all resource limits are still satisfied."""
        return (
            len(self._violations) == 0 and self._check_time() and self._check_memory()
        )

    def _check_time(self) -> bool:
        if self._sandbox.max_execution_time <= 0:
            return True
        return self.elapsed_seconds <= self._sandbox.max_execution_time

    def _check_memory(self) -> bool:
        if self._sandbox.max_memory_mb <= 0:
            return True
        try:
            # Get current RSS in MB (Linux)
            usage = resource.getrusage(resource.RUSAGE_SELF)
            rss_mb = usage.ru_maxrss / 1024  # ru_maxrss is in KB on Linux
            return rss_mb <= self._sandbox.max_memory_mb
        except Exception:
            return True  # Can't measure, assume OK

    def check_tool_allowed(self, tool_name: str) -> Optional[SandboxViolation]:
        """
        Check if a tool call is allowed. Returns a violation if blocked.
        """
        sb = self._sandbox

        # Check blocked list first
        if sb.blocked_tools and tool_name in sb.blocked_tools:
            v = SandboxViolation(
                agent_name=self._agent_name,
                violation_type="tool_blocked",
                limit=sb.blocked_tools,
                actual=tool_name,
                message=f"Tool '{tool_name}' is blocked by sandbox policy",
            )
            self._violations.append(v)
            return v

        # Check allowed list
        if sb.allowed_tools is not None and tool_name not in sb.allowed_tools:
            v = SandboxViolation(
                agent_name=self._agent_name,
                violation_type="tool_blocked",
                limit=sb.allowed_tools,
                actual=tool_name,
                message=f"Tool '{tool_name}' is not in the allowed list",
            )
            self._violations.append(v)
            return v

        return None

    def check_tool_call_limit(self) -> Optional[SandboxViolation]:
        """Check if tool call limit is reached before executing."""
        sb = self._sandbox
        if sb.max_tool_calls_per_task <= 0:
            return None

        if self._tool_call_count >= sb.max_tool_calls_per_task:
            v = SandboxViolation(
                agent_name=self._agent_name,
                violation_type="tool_calls",
                limit=sb.max_tool_calls_per_task,
                actual=self._tool_call_count,
                message=f"Tool call limit ({sb.max_tool_calls_per_task}) exceeded",
            )
            self._violations.append(v)
            return v

        return None

    def record_tool_call(self) -> None:
        """Record that a tool call happened."""
        self._tool_call_count += 1

    def check_time_limit(self) -> Optional[SandboxViolation]:
        """Check if execution time limit is reached."""
        sb = self._sandbox
        if sb.max_execution_time <= 0:
            return None

        elapsed = self.elapsed_seconds
        if elapsed > sb.max_execution_time:
            v = SandboxViolation(
                agent_name=self._agent_name,
                violation_type="time",
                limit=sb.max_execution_time,
                actual=elapsed,
                message=f"Execution time ({elapsed:.1f}s) exceeded limit ({sb.max_execution_time}s)",
            )
            self._violations.append(v)
            return v

        return None

    def check_output_size(self, output: Any) -> Optional[SandboxViolation]:
        """Check if output exceeds size limit."""
        sb = self._sandbox
        if sb.max_output_size_kb <= 0:
            return None

        size_bytes = sys.getsizeof(output) if output is not None else 0
        size_kb = size_bytes / 1024
        if size_kb > sb.max_output_size_kb:
            v = SandboxViolation(
                agent_name=self._agent_name,
                violation_type="output_size",
                limit=sb.max_output_size_kb,
                actual=size_kb,
                message=f"Output size ({size_kb:.1f}KB) exceeds limit ({sb.max_output_size_kb}KB)",
            )
            self._violations.append(v)
            return v

        return None

    def check_network_access(self) -> Optional[SandboxViolation]:
        """Check if network access is permitted."""
        if self._sandbox.network_access:
            return None
        v = SandboxViolation(
            agent_name=self._agent_name,
            violation_type="network",
            limit=False,
            actual="network_attempt",
            message="Network access is blocked by sandbox policy",
        )
        self._violations.append(v)
        return v


# ═══════════════════════════════════════════════════════════════════════════════
# SANDBOX GUARD — enforces sandbox limits
# ═══════════════════════════════════════════════════════════════════════════════


class SandboxGuard:
    """
    Enforces sandbox policies for agents.

    Created per-agent, provides ResourceMeter instances for each task,
    and tracks violations across those tasks.
    """

    def __init__(self, agent_name: str, sandbox: Sandbox):
        self._agent_name = agent_name
        self._sandbox = sandbox
        self._active_meters: Dict[str, ResourceMeter] = {}
        self._violation_log: List[SandboxViolation] = []
        self._lock = threading.Lock()
        self._concurrent_tasks = 0

    @property
    def sandbox(self) -> Sandbox:
        return self._sandbox

    @property
    def violation_count(self) -> int:
        return len(self._violation_log)

    @property
    def violations(self) -> List[SandboxViolation]:
        return list(self._violation_log)

    def start_task(self, task_id: str) -> ResourceMeter:
        """
        Begin resource tracking for a new task.

        Returns a ResourceMeter to track and enforce limits.
        """
        with self._lock:
            # Check concurrency limit
            sb = self._sandbox
            if (
                sb.max_concurrent_tasks > 0
                and self._concurrent_tasks >= sb.max_concurrent_tasks
            ):
                v = SandboxViolation(
                    agent_name=self._agent_name,
                    violation_type="concurrent",
                    limit=sb.max_concurrent_tasks,
                    actual=self._concurrent_tasks,
                    message=f"Concurrent task limit ({sb.max_concurrent_tasks}) exceeded",
                )
                self._violation_log.append(v)
                # Still return a meter, but it has a violation
                meter = ResourceMeter(self._agent_name, sb)
                meter._violations.append(v)
                return meter

            self._concurrent_tasks += 1
            meter = ResourceMeter(self._agent_name, sb)
            self._active_meters[task_id] = meter
            return meter

    def end_task(self, task_id: str) -> Optional[ResourceMeter]:
        """End resource tracking for a task. Returns the meter for inspection."""
        with self._lock:
            self._concurrent_tasks = max(0, self._concurrent_tasks - 1)
            meter = self._active_meters.pop(task_id, None)
            if meter:
                self._violation_log.extend(meter.violations)
            return meter

    def check_tool(self, task_id: str, tool_name: str) -> Optional[SandboxViolation]:
        """
        Pre-flight check before a tool call.

        Checks:
          1. Tool is allowed (not blocked / in whitelist)
          2. Tool call count limit not exceeded
          3. Time limit not exceeded

        Returns SandboxViolation if blocked, None if ok.
        """
        meter = self._active_meters.get(task_id)
        if meter is None:
            return None

        # Check tool access
        v = meter.check_tool_allowed(tool_name)
        if v:
            return v

        # Check call count
        v = meter.check_tool_call_limit()
        if v:
            return v

        # Check time
        v = meter.check_time_limit()
        if v:
            return v

        # All good — record the call
        meter.record_tool_call()
        return None

    def get_meter(self, task_id: str) -> Optional[ResourceMeter]:
        """Get the resource meter for an active task."""
        return self._active_meters.get(task_id)

    def reset(self) -> None:
        """Reset all tracking data."""
        with self._lock:
            self._active_meters.clear()
            self._violation_log.clear()
            self._concurrent_tasks = 0
