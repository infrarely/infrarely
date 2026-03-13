"""
runtime/resource_isolation.py — Module 6: Resource Isolation
═══════════════════════════════════════════════════════════════════════════════
Per-agent budgets and global resource limits.
Like Linux cgroups — prevents any agent from monopolising system resources.

Gap solutions:
  Gap 7  — Resource exhaustion: hard limits on tokens, time, memory, depth
           per agent.  Global ceiling ensures total doesn't exceed capacity.

Resources tracked per agent:
  • token_budget     — LLM tokens consumed
  • execution_depth  — call stack depth
  • wall_time_ms     — total execution time
  • task_count       — tasks executed
  • memory_entries   — shared memory entries owned
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from infrarely.observability import logger


@dataclass
class ResourceQuota:
    """Per-agent resource limits."""

    max_tokens: int = 2000
    max_depth: int = 5
    max_wall_time_ms: float = 60_000.0  # 60s
    max_tasks_per_minute: int = 20
    max_memory_entries: int = 50

    def scale(self, factor: float) -> "ResourceQuota":
        """Return scaled copy of quotas."""
        return ResourceQuota(
            max_tokens=int(self.max_tokens * factor),
            max_depth=max(1, int(self.max_depth * factor)),
            max_wall_time_ms=self.max_wall_time_ms * factor,
            max_tasks_per_minute=max(1, int(self.max_tasks_per_minute * factor)),
            max_memory_entries=max(5, int(self.max_memory_entries * factor)),
        )


@dataclass
class ResourceUsage:
    """Actual resource usage for an agent."""

    tokens_used: int = 0
    current_depth: int = 0
    wall_time_ms: float = 0.0
    task_timestamps: List[float] = field(default_factory=list)
    memory_entries_owned: int = 0
    violations: int = 0
    last_violation: str = ""
    start_time: float = field(default_factory=time.time)

    @property
    def tasks_last_minute(self) -> int:
        cutoff = time.time() - 60
        return sum(1 for t in self.task_timestamps if t > cutoff)

    def record_task(self):
        self.task_timestamps.append(time.time())
        # Trim old timestamps
        if len(self.task_timestamps) > 200:
            cutoff = time.time() - 120
            self.task_timestamps = [t for t in self.task_timestamps if t > cutoff]


class ResourceIsolation:
    """
    Enforces per-agent resource limits and global resource ceilings.

    Invariants:
      • Global token ceiling: sum of all agent budgets ≤ global limit
      • Hard limits: agents that exceed quota get blocked
      • Soft warnings at 80% threshold
      • Auto-throttle: reduce quota when global pressure is high
    """

    WARN_THRESHOLD = 0.8  # 80% usage triggers warning
    GLOBAL_TOKEN_CEILING = 10000
    GLOBAL_MAX_AGENTS = 50

    def __init__(self):
        self._quotas: Dict[str, ResourceQuota] = {}
        self._usage: Dict[str, ResourceUsage] = {}
        self._global_tokens_allocated = 0
        self._throttle_active = False
        self._total_violations = 0

    # ── Register agent ────────────────────────────────────────────────────────
    def register(self, agent_id: str, quota: ResourceQuota = None) -> ResourceQuota:
        """
        Register an agent with a resource quota.
        Gap 7: checks global ceiling before allocating.
        """
        if quota is None:
            quota = ResourceQuota()

        # Check global token ceiling (Gap 7)
        if self._global_tokens_allocated + quota.max_tokens > self.GLOBAL_TOKEN_CEILING:
            available = self.GLOBAL_TOKEN_CEILING - self._global_tokens_allocated
            if available <= 0:
                raise ValueError(
                    f"Global token ceiling reached ({self.GLOBAL_TOKEN_CEILING}) — "
                    "cannot allocate more tokens"
                )
            # Scale down quota to fit
            factor = available / quota.max_tokens
            quota = quota.scale(factor)
            logger.warn(
                f"ResourceIsolation: scaled quota for '{agent_id}' to "
                f"{quota.max_tokens} tokens (global pressure)"
            )

        self._quotas[agent_id] = quota
        self._usage[agent_id] = ResourceUsage()
        self._global_tokens_allocated += quota.max_tokens
        return quota

    # ── Unregister ────────────────────────────────────────────────────────────
    def unregister(self, agent_id: str):
        """Release agent's resource allocation."""
        quota = self._quotas.pop(agent_id, None)
        self._usage.pop(agent_id, None)
        if quota:
            self._global_tokens_allocated -= quota.max_tokens

    # ── Checks ────────────────────────────────────────────────────────────────
    def check_tokens(self, agent_id: str, tokens_needed: int = 1) -> bool:
        """
        Check if agent can spend tokens. Returns False if over budget.
        Gap 7: hard limit enforcement.
        """
        quota = self._quotas.get(agent_id)
        usage = self._usage.get(agent_id)
        if not quota or not usage:
            return False

        if usage.tokens_used + tokens_needed > quota.max_tokens:
            self._record_violation(agent_id, "token_limit", usage, quota)
            return False

        # Soft warning
        ratio = (usage.tokens_used + tokens_needed) / quota.max_tokens
        if ratio >= self.WARN_THRESHOLD:
            logger.warn(
                f"ResourceIsolation: '{agent_id}' at {ratio:.0%} token budget "
                f"({usage.tokens_used}/{quota.max_tokens})"
            )
        return True

    def check_depth(self, agent_id: str, current_depth: int = 0) -> bool:
        """Check if agent can go deeper in execution."""
        quota = self._quotas.get(agent_id)
        if not quota:
            return False
        if current_depth >= quota.max_depth:
            usage = self._usage.get(agent_id)
            if usage:
                self._record_violation(agent_id, "depth_limit", usage, quota)
            return False
        return True

    def check_task_rate(self, agent_id: str) -> bool:
        """Check if agent is within task rate limit."""
        quota = self._quotas.get(agent_id)
        usage = self._usage.get(agent_id)
        if not quota or not usage:
            return False
        if usage.tasks_last_minute >= quota.max_tasks_per_minute:
            self._record_violation(agent_id, "task_rate_limit", usage, quota)
            return False
        return True

    def check_wall_time(self, agent_id: str) -> bool:
        """Check if agent is within wall time limit."""
        quota = self._quotas.get(agent_id)
        usage = self._usage.get(agent_id)
        if not quota or not usage:
            return False
        elapsed = (time.time() - usage.start_time) * 1000
        if elapsed > quota.max_wall_time_ms:
            self._record_violation(agent_id, "wall_time_limit", usage, quota)
            return False
        return True

    def check_all(self, agent_id: str) -> Dict[str, bool]:
        """Run all resource checks at once."""
        return {
            "tokens": self.check_tokens(agent_id),
            "depth": self.check_depth(agent_id),
            "task_rate": self.check_task_rate(agent_id),
            "wall_time": self.check_wall_time(agent_id),
        }

    # ── Consume ───────────────────────────────────────────────────────────────
    def consume_tokens(self, agent_id: str, tokens: int) -> bool:
        """Record token consumption. Returns False if over budget."""
        if not self.check_tokens(agent_id, tokens):
            return False
        usage = self._usage.get(agent_id)
        if usage:
            usage.tokens_used += tokens
        return True

    def record_task(self, agent_id: str):
        """Record a task execution."""
        usage = self._usage.get(agent_id)
        if usage:
            usage.record_task()

    def set_depth(self, agent_id: str, depth: int):
        """Update current execution depth."""
        usage = self._usage.get(agent_id)
        if usage:
            usage.current_depth = depth

    # ── Global pressure ───────────────────────────────────────────────────────
    def global_pressure(self) -> float:
        """
        Calculate global resource pressure (0.0 = no pressure, 1.0 = at capacity).
        """
        if not self._usage:
            return 0.0
        total_used = sum(u.tokens_used for u in self._usage.values())
        return min(1.0, total_used / max(1, self.GLOBAL_TOKEN_CEILING))

    def apply_throttle(self, force: bool = False):
        """
        Gap 7: if global pressure > 80%, scale down all quotas by 50%.
        """
        pressure = self.global_pressure()
        if pressure >= self.WARN_THRESHOLD or force:
            self._throttle_active = True
            for agent_id, quota in self._quotas.items():
                self._quotas[agent_id] = quota.scale(0.5)
            logger.warn(
                f"ResourceIsolation: throttle active — global pressure {pressure:.0%}"
            )
        else:
            self._throttle_active = False

    # ── Violation tracking ────────────────────────────────────────────────────
    def _record_violation(
        self,
        agent_id: str,
        violation_type: str,
        usage: ResourceUsage,
        quota: ResourceQuota,
    ):
        usage.violations += 1
        usage.last_violation = violation_type
        self._total_violations += 1
        logger.warn(
            f"ResourceIsolation: violation '{violation_type}' for '{agent_id}' "
            f"(total violations: {usage.violations})"
        )

    # ── Query ─────────────────────────────────────────────────────────────────
    def get_usage(self, agent_id: str) -> Optional[Dict[str, Any]]:
        usage = self._usage.get(agent_id)
        quota = self._quotas.get(agent_id)
        if not usage or not quota:
            return None
        return {
            "tokens": f"{usage.tokens_used}/{quota.max_tokens}",
            "depth": f"{usage.current_depth}/{quota.max_depth}",
            "tasks_last_min": f"{usage.tasks_last_minute}/{quota.max_tasks_per_minute}",
            "violations": usage.violations,
            "last_violation": usage.last_violation,
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "agents_tracked": len(self._quotas),
            "global_tokens_allocated": self._global_tokens_allocated,
            "global_token_ceiling": self.GLOBAL_TOKEN_CEILING,
            "global_pressure": f"{self.global_pressure():.1%}",
            "throttle_active": self._throttle_active,
            "total_violations": self._total_violations,
            "per_agent": {aid: self.get_usage(aid) for aid in self._quotas},
        }
