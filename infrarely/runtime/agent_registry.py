"""
runtime/agent_registry.py — Module 1: Agent Registry
═══════════════════════════════════════════════════════════════════════════════
Central store for all active agents in the runtime.

Like /proc in Linux — every agent must register here to exist.

Provides:
  • Agent discovery by id, role, capability
  • Agent lifecycle state tracking
  • Capability lookup for the marketplace
  • Gap 5 solved: capability conflict resolution via scoring

Design:
  The registry is the ONLY authority on which agents exist.
  No component may access an agent without going through the registry.
"""

from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set
from infrarely.observability import logger


class AgentStatus(Enum):
    """Agent lifecycle states."""

    REGISTERED = auto()  # created but not started
    ACTIVE = auto()  # running and accepting tasks
    BUSY = auto()  # currently executing a task
    PAUSED = auto()  # temporarily suspended
    TERMINATED = auto()  # shut down
    CRASHED = auto()  # failed unexpectedly
    DRAINING = auto()  # finishing current work before shutdown


@dataclass
class AgentRecord:
    """
    Complete record for one agent in the registry.

    agent_id        — globally unique identifier
    role            — functional role (student_agent, research_agent, etc.)
    capabilities    — set of capabilities this agent can execute
    tools           — set of tools this agent has access to
    token_budget    — max tokens per session
    permissions     — set of permission tags
    status          — current lifecycle state
    priority        — scheduling priority (higher = more important)
    owner           — who created this agent
    metadata        — arbitrary key-value tags
    """

    agent_id: str
    role: str
    capabilities: Set[str] = field(default_factory=set)
    tools: Set[str] = field(default_factory=set)
    token_budget: int = 5000
    permissions: Set[str] = field(default_factory=set)
    status: AgentStatus = AgentStatus.REGISTERED
    priority: int = 5  # 1=lowest, 10=highest
    owner: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_active: float = 0.0
    task_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_tokens_used: int = 0
    version: int = 1

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total else 1.0

    @property
    def is_available(self) -> bool:
        return self.status in (AgentStatus.ACTIVE, AgentStatus.REGISTERED)


class AgentRegistry:
    """
    Central agent registry.  Thread-safe for single-threaded async use.
    All agent lookup, creation, and status changes flow through here.
    """

    MAX_AGENTS = 50  # hard cap on concurrent agents

    def __init__(self):
        self._agents: Dict[str, AgentRecord] = {}
        self._roles_index: Dict[str, Set[str]] = {}  # role → set of agent_ids
        self._caps_index: Dict[str, Set[str]] = {}  # capability → set of agent_ids
        self._creation_order: List[str] = []

    # ── Registration ──────────────────────────────────────────────────────────
    def register(
        self,
        role: str,
        capabilities: Set[str] = None,
        tools: Set[str] = None,
        token_budget: int = 5000,
        permissions: Set[str] = None,
        priority: int = 5,
        owner: str = "system",
        metadata: Dict[str, Any] = None,
        agent_id: str = None,
    ) -> AgentRecord:
        """
        Register a new agent.  Returns the AgentRecord.
        Raises ValueError if MAX_AGENTS exceeded.
        """
        if len(self._agents) >= self.MAX_AGENTS:
            raise ValueError(
                f"Agent registry full ({self.MAX_AGENTS} agents). "
                "Terminate idle agents first."
            )

        aid = agent_id or f"agent_{uuid.uuid4().hex[:8]}"
        if aid in self._agents:
            raise ValueError(f"Agent '{aid}' already registered")

        caps = capabilities or set()
        record = AgentRecord(
            agent_id=aid,
            role=role,
            capabilities=caps,
            tools=tools or set(),
            token_budget=token_budget,
            permissions=permissions or set(),
            priority=priority,
            owner=owner,
            metadata=metadata or {},
        )
        self._agents[aid] = record
        self._creation_order.append(aid)

        # Update indices
        self._roles_index.setdefault(role, set()).add(aid)
        for cap in caps:
            self._caps_index.setdefault(cap, set()).add(aid)

        logger.info(f"AgentRegistry: registered '{aid}' role={role} caps={caps}")
        return record

    # ── Lookup ────────────────────────────────────────────────────────────────
    def get(self, agent_id: str) -> Optional[AgentRecord]:
        return self._agents.get(agent_id)

    def get_by_role(self, role: str) -> List[AgentRecord]:
        """All agents with given role."""
        ids = self._roles_index.get(role, set())
        return [self._agents[aid] for aid in ids if aid in self._agents]

    def get_by_capability(self, capability: str) -> List[AgentRecord]:
        """All agents that offer a given capability (Gap 5: conflict candidates)."""
        ids = self._caps_index.get(capability, set())
        return [self._agents[aid] for aid in ids if aid in self._agents]

    def find_available(
        self, capability: str = None, role: str = None
    ) -> List[AgentRecord]:
        """Find available agents, optionally filtered by capability or role."""
        candidates = list(self._agents.values())
        if capability:
            cap_ids = self._caps_index.get(capability, set())
            candidates = [a for a in candidates if a.agent_id in cap_ids]
        if role:
            candidates = [a for a in candidates if a.role == role]
        return [a for a in candidates if a.is_available]

    # ── Status management ─────────────────────────────────────────────────────
    def set_status(self, agent_id: str, status: AgentStatus) -> bool:
        rec = self._agents.get(agent_id)
        if not rec:
            return False
        old = rec.status
        rec.status = status
        if status == AgentStatus.ACTIVE:
            rec.last_active = time.time()
        logger.debug(f"AgentRegistry: '{agent_id}' {old.name} → {status.name}")
        return True

    def activate(self, agent_id: str) -> bool:
        return self.set_status(agent_id, AgentStatus.ACTIVE)

    def terminate(self, agent_id: str) -> bool:
        """Mark agent as terminated and remove from indices."""
        rec = self._agents.get(agent_id)
        if not rec:
            return False
        rec.status = AgentStatus.TERMINATED
        # Don't remove from _agents (keep for history), but remove from indices
        self._roles_index.get(rec.role, set()).discard(agent_id)
        for cap in rec.capabilities:
            self._caps_index.get(cap, set()).discard(agent_id)
        logger.info(f"AgentRegistry: terminated '{agent_id}'")
        return True

    # ── Record keeping ────────────────────────────────────────────────────────
    def record_task(self, agent_id: str, success: bool, tokens: int = 0):
        rec = self._agents.get(agent_id)
        if not rec:
            return
        rec.task_count += 1
        rec.total_tokens_used += tokens
        if success:
            rec.success_count += 1
        else:
            rec.failure_count += 1
        rec.last_active = time.time()

    # ── Capability conflict resolution (Gap 5) ───────────────────────────────
    def best_agent_for_capability(self, capability: str) -> Optional[AgentRecord]:
        """
        When multiple agents offer the same capability, rank by:
          score = priority × success_rate × (1 / (1 + tokens_used/budget))
        Returns best available agent or None.
        """
        candidates = self.find_available(capability=capability)
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        def _score(a: AgentRecord) -> float:
            budget_factor = 1.0 / (1.0 + a.total_tokens_used / max(a.token_budget, 1))
            return a.priority * a.success_rate * budget_factor

        ranked = sorted(candidates, key=_score, reverse=True)
        if len(ranked) > 1:
            logger.info(
                f"AgentRegistry: capability conflict '{capability}' — "
                f"{len(ranked)} agents, winner='{ranked[0].agent_id}'"
            )
        return ranked[0]

    # ── Query ─────────────────────────────────────────────────────────────────
    def all_agents(self) -> List[AgentRecord]:
        return list(self._agents.values())

    def active_agents(self) -> List[AgentRecord]:
        return [a for a in self._agents.values() if a.status == AgentStatus.ACTIVE]

    def agent_count(self) -> int:
        return len(self._agents)

    def active_count(self) -> int:
        return sum(1 for a in self._agents.values() if a.status == AgentStatus.ACTIVE)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_agents": len(self._agents),
            "active_agents": self.active_count(),
            "max_agents": self.MAX_AGENTS,
            "roles": {role: len(ids) for role, ids in self._roles_index.items()},
            "capabilities_offered": {
                cap: len(ids) for cap, ids in self._caps_index.items()
            },
            "agents": {
                aid: {
                    "role": a.role,
                    "status": a.status.name,
                    "priority": a.priority,
                    "capabilities": sorted(a.capabilities),
                    "tasks": a.task_count,
                    "success_rate": round(a.success_rate, 3),
                    "tokens_used": a.total_tokens_used,
                }
                for aid, a in self._agents.items()
            },
        }
