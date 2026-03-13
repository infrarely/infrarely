"""
runtime/global_resource_governance.py — GAP 9
═══════════════════════════════════════════════════════════════════════════════
Cluster-wide resource governance: global token budget, CPU budget,
system-pressure back-pressure, and per-agent quotas.

Subsystems:
  • TokenBudgetPool      — cluster-wide token ceiling (LLM tokens)
  • CpuBudgetPool        — cluster-wide CPU-time budget (seconds)
  • SystemPressure       — aggregate pressure metric (0.0 – 1.0)
  • ResourceGovernor     — orchestrator with snapshot()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


# ─── Enums ────────────────────────────────────────────────────────────────────
class PressureLevel(Enum):
    NORMAL = auto()  # < 0.60
    ELEVATED = auto()  # 0.60 – 0.80
    CRITICAL = auto()  # > 0.80


# ─── Data ─────────────────────────────────────────────────────────────────────
@dataclass
class AgentQuota:
    agent_id: str
    token_limit: int = 2000
    cpu_limit_s: float = 30.0
    tokens_used: int = 0
    cpu_used_s: float = 0.0
    throttled: bool = False

    @property
    def token_remaining(self) -> int:
        return max(0, self.token_limit - self.tokens_used)

    @property
    def cpu_remaining(self) -> float:
        return max(0.0, self.cpu_limit_s - self.cpu_used_s)


# ─── Token Budget Pool ───────────────────────────────────────────────────────
class TokenBudgetPool:
    """Cluster-level token ceiling shared across all agents."""

    def __init__(self, ceiling: int = 100_000):
        self.ceiling = ceiling
        self._used = 0

    def consume(self, amount: int) -> bool:
        if self._used + amount > self.ceiling:
            return False
        self._used += amount
        return True

    def release(self, amount: int) -> None:
        self._used = max(0, self._used - amount)

    @property
    def remaining(self) -> int:
        return max(0, self.ceiling - self._used)

    @property
    def utilization(self) -> float:
        return self._used / max(self.ceiling, 1)

    def reset(self) -> None:
        self._used = 0


# ─── CPU Budget Pool ─────────────────────────────────────────────────────────
class CpuBudgetPool:
    """Cluster-level CPU-time budget (seconds)."""

    def __init__(self, ceiling_s: float = 600.0):
        self.ceiling_s = ceiling_s
        self._used_s = 0.0

    def consume(self, seconds: float) -> bool:
        if self._used_s + seconds > self.ceiling_s:
            return False
        self._used_s += seconds
        return True

    def release(self, seconds: float) -> None:
        self._used_s = max(0.0, self._used_s - seconds)

    @property
    def remaining(self) -> float:
        return max(0.0, self.ceiling_s - self._used_s)

    @property
    def utilization(self) -> float:
        return self._used_s / max(self.ceiling_s, 0.001)

    def reset(self) -> None:
        self._used_s = 0.0


# ─── System Pressure ─────────────────────────────────────────────────────────
class SystemPressure:
    """Aggregate pressure metric from token + cpu utilization."""

    def __init__(
        self,
        token_pool: TokenBudgetPool,
        cpu_pool: CpuBudgetPool,
        token_weight: float = 0.6,
        cpu_weight: float = 0.4,
    ):
        self._token_pool = token_pool
        self._cpu_pool = cpu_pool
        self._tw = token_weight
        self._cw = cpu_weight

    @property
    def value(self) -> float:
        return (
            self._tw * self._token_pool.utilization
            + self._cw * self._cpu_pool.utilization
        )

    @property
    def level(self) -> PressureLevel:
        v = self.value
        if v > 0.80:
            return PressureLevel.CRITICAL
        elif v > 0.60:
            return PressureLevel.ELEVATED
        return PressureLevel.NORMAL

    @property
    def should_throttle(self) -> bool:
        return self.level == PressureLevel.CRITICAL


# ─── Resource Governor ────────────────────────────────────────────────────────
class ResourceGovernor:
    """
    Orchestrates cluster token budget, CPU budget, pressure, and per-agent
    quotas.
    """

    def __init__(
        self,
        token_ceiling: int = 100_000,
        cpu_ceiling_s: float = 600.0,
        default_agent_token_limit: int = 2000,
        default_agent_cpu_limit_s: float = 30.0,
    ):
        self.token_pool = TokenBudgetPool(ceiling=token_ceiling)
        self.cpu_pool = CpuBudgetPool(ceiling_s=cpu_ceiling_s)
        self.pressure = SystemPressure(self.token_pool, self.cpu_pool)
        self._default_token_limit = default_agent_token_limit
        self._default_cpu_limit_s = default_agent_cpu_limit_s
        self._quotas: Dict[str, AgentQuota] = {}

    # ── agent quotas ─────────────────────────────────────────────────────
    def register_agent(
        self, agent_id: str, token_limit: int = 0, cpu_limit_s: float = 0.0
    ) -> AgentQuota:
        tl = token_limit if token_limit > 0 else self._default_token_limit
        cl = cpu_limit_s if cpu_limit_s > 0 else self._default_cpu_limit_s
        q = AgentQuota(agent_id=agent_id, token_limit=tl, cpu_limit_s=cl)
        self._quotas[agent_id] = q
        return q

    def consume_tokens(self, agent_id: str, amount: int) -> bool:
        q = self._quotas.get(agent_id)
        if not q:
            return False
        if q.throttled:
            return False
        if q.tokens_used + amount > q.token_limit:
            q.throttled = True
            return False
        if not self.token_pool.consume(amount):
            return False
        q.tokens_used += amount
        # check pressure after consumption
        if self.pressure.should_throttle:
            q.throttled = True
        return True

    def consume_cpu(self, agent_id: str, seconds: float) -> bool:
        q = self._quotas.get(agent_id)
        if not q:
            return False
        if q.throttled:
            return False
        if q.cpu_used_s + seconds > q.cpu_limit_s:
            q.throttled = True
            return False
        if not self.cpu_pool.consume(seconds):
            return False
        q.cpu_used_s += seconds
        return True

    def unthrottle(self, agent_id: str) -> bool:
        q = self._quotas.get(agent_id)
        if not q:
            return False
        q.throttled = False
        return True

    def reset_agent(self, agent_id: str) -> bool:
        q = self._quotas.get(agent_id)
        if not q:
            return False
        q.tokens_used = 0
        q.cpu_used_s = 0.0
        q.throttled = False
        return True

    def reset_all(self) -> None:
        self.token_pool.reset()
        self.cpu_pool.reset()
        for q in self._quotas.values():
            q.tokens_used = 0
            q.cpu_used_s = 0.0
            q.throttled = False

    # ── introspection ────────────────────────────────────────────────────
    def agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        q = self._quotas.get(agent_id)
        if not q:
            return None
        return {
            "agent_id": q.agent_id,
            "tokens_used": q.tokens_used,
            "token_remaining": q.token_remaining,
            "cpu_used_s": round(q.cpu_used_s, 3),
            "cpu_remaining_s": round(q.cpu_remaining, 3),
            "throttled": q.throttled,
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "token_pool_utilization": round(self.token_pool.utilization, 4),
            "token_pool_remaining": self.token_pool.remaining,
            "cpu_pool_utilization": round(self.cpu_pool.utilization, 4),
            "cpu_pool_remaining_s": round(self.cpu_pool.remaining, 3),
            "system_pressure": round(self.pressure.value, 4),
            "pressure_level": self.pressure.level.name,
            "agents_registered": len(self._quotas),
            "agents_throttled": sum(1 for q in self._quotas.values() if q.throttled),
        }
