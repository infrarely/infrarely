"""
evolution/architecture_optimizer.py — Layer 7, Module 4
═══════════════════════════════════════════════════════════════════════════════
Proposes architecture-level improvements.

Targets:
    scheduler strategy, load-balancing policy, bus configuration,
    memory retention policy, agent pool sizing, token limits

All changes are PROPOSALS — never directly applied.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


# ── Data types ────────────────────────────────────────────────────────────────


class OptimizationTarget(Enum):
    SCHEDULER_STRATEGY = auto()
    LOAD_BALANCING = auto()
    BUS_CAPACITY = auto()
    MEMORY_RETENTION = auto()
    AGENT_POOL_SIZE = auto()
    TOKEN_LIMITS = auto()
    REPLICATION = auto()


class ArchProposalStatus(Enum):
    PROPOSED = auto()
    APPROVED = auto()
    REJECTED = auto()
    SIMULATED = auto()
    APPLIED = auto()
    ROLLED_BACK = auto()


@dataclass
class SystemProfile:
    """Current system configuration profile."""

    scheduler_strategy: str = "capability"
    lb_strategy: str = "round_robin"
    bus_capacity: int = 2000
    max_agents: int = 50
    token_ceiling: int = 10000
    avg_queue_depth: float = 0.0
    avg_bus_utilization: float = 0.0
    avg_agent_utilization: float = 0.0
    avg_health_score: float = 100.0


@dataclass
class ArchitectureProposal:
    """A proposed architecture change."""

    proposal_id: str = field(default_factory=lambda: f"arch_{uuid.uuid4().hex[:8]}")
    target: OptimizationTarget = OptimizationTarget.SCHEDULER_STRATEGY
    description: str = ""
    current_value: Any = None
    proposed_value: Any = None
    expected_impact: str = ""
    status: ArchProposalStatus = ArchProposalStatus.PROPOSED
    confidence: float = 0.0
    created_at: float = field(default_factory=time.time)
    simulation_result: Optional[Dict[str, Any]] = None


# ── Engine ────────────────────────────────────────────────────────────────────


class ArchitectureOptimizer:
    """
    Analyses system profiles and proposes architecture-level changes.
    """

    # Configuration thresholds
    HIGH_QUEUE_UTILIZATION = 0.70  # queue > 70% capacity → suggest increase
    HIGH_BUS_UTILIZATION = 0.75
    LOW_AGENT_UTILIZATION = 0.20  # agents < 20% utilized → suggest shrink
    HIGH_AGENT_UTILIZATION = 0.85  # agents > 85% utilized → suggest grow
    HEALTH_CONCERN = 60.0
    MIN_OBSERVATIONS = 3

    def __init__(self):
        self._profiles: List[SystemProfile] = []
        self._proposals: List[ArchitectureProposal] = []
        self._applied: List[ArchitectureProposal] = []
        self._analysis_count = 0

    # ── Profile recording ─────────────────────────────────────────────────────

    def record_profile(self, profile: SystemProfile) -> None:
        self._profiles.append(profile)
        if len(self._profiles) > 200:
            self._profiles = self._profiles[-100:]

    def build_profile(
        self,
        scheduler_snapshot: Dict[str, Any] = None,
        bus_snapshot: Dict[str, Any] = None,
        monitoring_snapshot: Dict[str, Any] = None,
        isolation_snapshot: Dict[str, Any] = None,
    ) -> SystemProfile:
        """Build a SystemProfile from Layer-6 snapshots."""
        p = SystemProfile()
        if scheduler_snapshot:
            p.avg_queue_depth = scheduler_snapshot.get("queue_depth", 0)
            p.scheduler_strategy = scheduler_snapshot.get("strategy", "capability")
        if bus_snapshot:
            total = bus_snapshot.get("total_messages", 0)
            cap = bus_snapshot.get("bus_capacity", 2000)
            p.bus_capacity = cap
            p.avg_bus_utilization = total / max(1, cap)
        if monitoring_snapshot:
            p.avg_health_score = monitoring_snapshot.get("health_score", 100.0)
            p.max_agents = monitoring_snapshot.get("agents_monitored", 50)
        if isolation_snapshot:
            p.token_ceiling = isolation_snapshot.get("global_token_ceiling", 10000)
        self.record_profile(p)
        return p

    # ── Analysis ──────────────────────────────────────────────────────────────

    def analyse(self) -> List[ArchitectureProposal]:
        """Analyse profiles and generate architecture proposals."""
        self._analysis_count += 1
        if len(self._profiles) < self.MIN_OBSERVATIONS:
            return []

        new_proposals = []
        recent = self._profiles[-self.MIN_OBSERVATIONS :]

        avg_queue = sum(p.avg_queue_depth for p in recent) / len(recent)
        avg_bus = sum(p.avg_bus_utilization for p in recent) / len(recent)
        avg_agent = sum(p.avg_agent_utilization for p in recent) / len(recent)
        avg_health = sum(p.avg_health_score for p in recent) / len(recent)

        latest = self._profiles[-1]

        # Queue depth → scheduler or pool adjustment
        if avg_queue > latest.bus_capacity * self.HIGH_QUEUE_UTILIZATION:
            new_proposals.append(
                ArchitectureProposal(
                    target=OptimizationTarget.AGENT_POOL_SIZE,
                    description="Queue utilization high — increase agent pool",
                    current_value=latest.max_agents,
                    proposed_value=int(latest.max_agents * 1.5),
                    expected_impact="Reduce queue depth by ~30%",
                    confidence=0.7,
                )
            )

        # Bus congestion
        if avg_bus > self.HIGH_BUS_UTILIZATION:
            new_cap = int(latest.bus_capacity * 1.5)
            new_proposals.append(
                ArchitectureProposal(
                    target=OptimizationTarget.BUS_CAPACITY,
                    description="Bus utilization high — increase capacity",
                    current_value=latest.bus_capacity,
                    proposed_value=new_cap,
                    expected_impact=f"Increase bus capacity from {latest.bus_capacity} to {new_cap}",
                    confidence=0.8,
                )
            )

        # Agent pool sizing
        if avg_agent > self.HIGH_AGENT_UTILIZATION:
            new_proposals.append(
                ArchitectureProposal(
                    target=OptimizationTarget.AGENT_POOL_SIZE,
                    description="Agent utilization high — scale up",
                    current_value=latest.max_agents,
                    proposed_value=int(latest.max_agents * 1.3),
                    expected_impact="Reduce agent overload",
                    confidence=0.7,
                )
            )
        elif avg_agent < self.LOW_AGENT_UTILIZATION and latest.max_agents > 5:
            new_proposals.append(
                ArchitectureProposal(
                    target=OptimizationTarget.AGENT_POOL_SIZE,
                    description="Agent utilization low — scale down to save resources",
                    current_value=latest.max_agents,
                    proposed_value=max(5, int(latest.max_agents * 0.7)),
                    expected_impact="Reduce idle agent overhead",
                    confidence=0.6,
                )
            )

        # Health concerns
        if avg_health < self.HEALTH_CONCERN:
            new_proposals.append(
                ArchitectureProposal(
                    target=OptimizationTarget.SCHEDULER_STRATEGY,
                    description="Low health score — consider switching scheduler strategy",
                    current_value=latest.scheduler_strategy,
                    proposed_value="weighted_priority",
                    expected_impact="Prioritize healthy agents, improve success rate",
                    confidence=0.5,
                )
            )

        # Token ceiling
        if latest.token_ceiling < 15000 and avg_health < 80:
            new_proposals.append(
                ArchitectureProposal(
                    target=OptimizationTarget.TOKEN_LIMITS,
                    description="Token ceiling may be constraining throughput",
                    current_value=latest.token_ceiling,
                    proposed_value=int(latest.token_ceiling * 1.5),
                    expected_impact="Reduce throttling, improve throughput",
                    confidence=0.5,
                )
            )

        self._proposals.extend(new_proposals)
        return new_proposals

    # ── Proposal management ───────────────────────────────────────────────────

    def approve(self, proposal_id: str) -> bool:
        for p in self._proposals:
            if p.proposal_id == proposal_id and p.status == ArchProposalStatus.PROPOSED:
                p.status = ArchProposalStatus.APPROVED
                return True
        return False

    def reject(self, proposal_id: str) -> bool:
        for p in self._proposals:
            if p.proposal_id == proposal_id and p.status == ArchProposalStatus.PROPOSED:
                p.status = ArchProposalStatus.REJECTED
                return True
        return False

    def record_simulation(self, proposal_id: str, result: Dict[str, Any]) -> bool:
        for p in self._proposals:
            if p.proposal_id == proposal_id and p.status == ArchProposalStatus.APPROVED:
                p.simulation_result = result
                p.status = ArchProposalStatus.SIMULATED
                return True
        return False

    def mark_applied(self, proposal_id: str) -> bool:
        for p in self._proposals:
            if (
                p.proposal_id == proposal_id
                and p.status == ArchProposalStatus.SIMULATED
            ):
                p.status = ArchProposalStatus.APPLIED
                self._applied.append(p)
                return True
        return False

    def mark_rolled_back(self, proposal_id: str) -> bool:
        for p in self._proposals:
            if p.proposal_id == proposal_id and p.status == ArchProposalStatus.APPLIED:
                p.status = ArchProposalStatus.ROLLED_BACK
                return True
        return False

    def pending_proposals(self) -> List[ArchitectureProposal]:
        return [p for p in self._proposals if p.status == ArchProposalStatus.PROPOSED]

    def get_proposal(self, proposal_id: str) -> Optional[ArchitectureProposal]:
        for p in self._proposals:
            if p.proposal_id == proposal_id:
                return p
        return None

    # ── Query ─────────────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        return {
            "profiles_recorded": len(self._profiles),
            "analysis_count": self._analysis_count,
            "total_proposals": len(self._proposals),
            "pending": len(self.pending_proposals()),
            "applied": len(self._applied),
            "latest_health": (
                round(self._profiles[-1].avg_health_score, 1)
                if self._profiles
                else None
            ),
        }
