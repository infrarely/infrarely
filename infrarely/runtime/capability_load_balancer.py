"""
runtime/capability_load_balancer.py — GAP 7
═══════════════════════════════════════════════════════════════════════════════
Capability-level load balancing.

When multiple agents can fulfil the same capability, this module distributes
work across them using configurable strategies:

  • ROUND_ROBIN   — simple rotation
  • LEAST_LOADED  — pick agent with fewest in-flight tasks
  • RANDOM        — random selection for stochastic fairness

Subsystems:
  • PoolEntry / CapabilityPool   — per-capability agent pool
  • LoadBalancer                 — top-level coordinator with snapshot()
"""

from __future__ import annotations

import random as _random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set


# ─── Strategy Enum ────────────────────────────────────────────────────────────
class LBStrategy(Enum):
    ROUND_ROBIN = auto()
    LEAST_LOADED = auto()
    RANDOM = auto()


# ─── Data ─────────────────────────────────────────────────────────────────────
@dataclass
class PoolEntry:
    agent_id: str
    healthy: bool = True
    in_flight: int = 0
    total_dispatched: int = 0
    last_dispatched: float = 0.0
    max_concurrent: int = 10


# ─── Capability Pool ─────────────────────────────────────────────────────────
class CapabilityPool:
    """Manages the agent pool for a single capability."""

    def __init__(
        self,
        capability: str,
        strategy: LBStrategy = LBStrategy.ROUND_ROBIN,
        max_per_agent: int = 10,
    ):
        self.capability = capability
        self.strategy = strategy
        self.max_per_agent = max_per_agent
        self._members: Dict[str, PoolEntry] = {}
        self._rr_index = 0

    # ── membership ────────────────────────────────────────────────────────
    def add(self, agent_id: str, max_concurrent: int = 0) -> bool:
        if agent_id in self._members:
            return False
        cap = max_concurrent if max_concurrent > 0 else self.max_per_agent
        self._members[agent_id] = PoolEntry(agent_id=agent_id, max_concurrent=cap)
        return True

    def remove(self, agent_id: str) -> bool:
        return self._members.pop(agent_id, None) is not None

    def set_health(self, agent_id: str, healthy: bool) -> bool:
        entry = self._members.get(agent_id)
        if not entry:
            return False
        entry.healthy = healthy
        return True

    # ── selection ─────────────────────────────────────────────────────────
    def _healthy_available(self) -> List[PoolEntry]:
        return [
            e
            for e in self._members.values()
            if e.healthy and e.in_flight < e.max_concurrent
        ]

    def select(self) -> Optional[str]:
        """Pick the next agent according to current strategy."""
        pool = self._healthy_available()
        if not pool:
            return None

        if self.strategy == LBStrategy.ROUND_ROBIN:
            self._rr_index = self._rr_index % len(pool)
            chosen = pool[self._rr_index]
            self._rr_index = (self._rr_index + 1) % len(pool)
        elif self.strategy == LBStrategy.LEAST_LOADED:
            chosen = min(pool, key=lambda e: e.in_flight)
        elif self.strategy == LBStrategy.RANDOM:
            chosen = _random.choice(pool)
        else:
            chosen = pool[0]

        chosen.in_flight += 1
        chosen.total_dispatched += 1
        chosen.last_dispatched = time.time()
        return chosen.agent_id

    def release(self, agent_id: str) -> bool:
        entry = self._members.get(agent_id)
        if not entry:
            return False
        entry.in_flight = max(0, entry.in_flight - 1)
        return True

    def size(self) -> int:
        return len(self._members)

    def healthy_count(self) -> int:
        return sum(1 for e in self._members.values() if e.healthy)


# ─── Load Balancer ────────────────────────────────────────────────────────────
class LoadBalancer:
    """
    Top-level load balancer across all capabilities.
    """

    def __init__(
        self,
        default_strategy: LBStrategy = LBStrategy.ROUND_ROBIN,
        default_max_per_agent: int = 10,
    ):
        self.default_strategy = default_strategy
        self.default_max_per_agent = default_max_per_agent
        self._pools: Dict[str, CapabilityPool] = {}
        self._total_dispatched = 0

    # ── pool management ───────────────────────────────────────────────────
    def create_pool(
        self,
        capability: str,
        strategy: Optional[LBStrategy] = None,
        max_per_agent: int = 0,
    ) -> CapabilityPool:
        if capability in self._pools:
            return self._pools[capability]
        s = strategy or self.default_strategy
        m = max_per_agent if max_per_agent > 0 else self.default_max_per_agent
        pool = CapabilityPool(capability=capability, strategy=s, max_per_agent=m)
        self._pools[capability] = pool
        return pool

    def register_agent(
        self, capability: str, agent_id: str, max_concurrent: int = 0
    ) -> bool:
        pool = self._pools.get(capability) or self.create_pool(capability)
        return pool.add(agent_id, max_concurrent)

    def unregister_agent(self, capability: str, agent_id: str) -> bool:
        pool = self._pools.get(capability)
        if not pool:
            return False
        return pool.remove(agent_id)

    def set_agent_health(self, capability: str, agent_id: str, healthy: bool) -> bool:
        pool = self._pools.get(capability)
        if not pool:
            return False
        return pool.set_health(agent_id, healthy)

    # ── dispatch ──────────────────────────────────────────────────────────
    def dispatch(self, capability: str) -> Optional[str]:
        pool = self._pools.get(capability)
        if not pool:
            return None
        agent_id = pool.select()
        if agent_id:
            self._total_dispatched += 1
        return agent_id

    def release(self, capability: str, agent_id: str) -> bool:
        pool = self._pools.get(capability)
        if not pool:
            return False
        return pool.release(agent_id)

    # ── introspection ─────────────────────────────────────────────────────
    def pool_stats(self, capability: str) -> Dict[str, Any]:
        pool = self._pools.get(capability)
        if not pool:
            return {}
        return {
            "capability": capability,
            "strategy": pool.strategy.name,
            "size": pool.size(),
            "healthy": pool.healthy_count(),
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_pools": len(self._pools),
            "total_dispatched": self._total_dispatched,
            "pools": {
                cap: {
                    "strategy": p.strategy.name,
                    "size": p.size(),
                    "healthy": p.healthy_count(),
                }
                for cap, p in self._pools.items()
            },
        }
