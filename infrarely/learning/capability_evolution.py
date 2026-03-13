"""
evolution/capability_evolution.py — Layer 7, Module 3
═══════════════════════════════════════════════════════════════════════════════
Automatically improves capabilities based on performance data.

Evolution strategies:
    • Graph optimization (reorder steps)
    • Tool replacement (swap slow/unreliable tools)
    • Parameter tuning (adjust defaults)
    • Step parallelization (run independent steps together)

All changes are PROPOSALS — never applied directly.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


# ── Data types ────────────────────────────────────────────────────────────────


class EvolutionStrategy(Enum):
    REORDER_STEPS = auto()
    REPLACE_TOOL = auto()
    TUNE_PARAMETERS = auto()
    PARALLELIZE = auto()
    CACHE_OUTPUT = auto()
    ADD_FALLBACK = auto()
    REMOVE_STEP = auto()


class ProposalStatus(Enum):
    PENDING = auto()
    APPROVED = auto()
    REJECTED = auto()
    TESTED = auto()
    APPLIED = auto()
    ROLLED_BACK = auto()


@dataclass
class CapabilityProfile:
    """Performance profile for one capability."""

    name: str
    steps: List[str] = field(default_factory=list)
    avg_latency_ms: float = 0.0
    success_rate: float = 1.0
    token_cost: int = 0
    step_latencies: Dict[str, float] = field(default_factory=dict)
    step_failure_rates: Dict[str, float] = field(default_factory=dict)
    invocations: int = 0


@dataclass
class EvolutionProposal:
    """A proposed capability change."""

    proposal_id: str = field(default_factory=lambda: f"evo_{uuid.uuid4().hex[:8]}")
    capability: str = ""
    strategy: EvolutionStrategy = EvolutionStrategy.TUNE_PARAMETERS
    description: str = ""
    expected_improvement: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    status: ProposalStatus = ProposalStatus.PENDING
    created_at: float = field(default_factory=time.time)
    confidence: float = 0.0  # 0–1


# ── Engine ────────────────────────────────────────────────────────────────────


class CapabilityEvolutionEngine:
    """
    Analyses capability performance and proposes improvements.
    Never modifies capabilities directly — only generates proposals.
    """

    # Thresholds that trigger proposals
    SLOW_STEP_THRESHOLD_MS = 500.0
    LOW_SUCCESS_THRESHOLD = 0.80
    HIGH_TOKEN_THRESHOLD = 500
    MIN_INVOCATIONS = 5

    def __init__(self):
        self._profiles: Dict[str, CapabilityProfile] = {}
        self._proposals: List[EvolutionProposal] = []
        self._applied: List[EvolutionProposal] = []
        self._analysis_count = 0

    # ── Profile management ────────────────────────────────────────────────────

    def register_capability(self, name: str, steps: List[str]) -> CapabilityProfile:
        profile = CapabilityProfile(name=name, steps=list(steps))
        self._profiles[name] = profile
        return profile

    def record_execution(
        self,
        capability: str,
        latency_ms: float,
        success: bool,
        tokens: int = 0,
        step_latencies: Dict[str, float] = None,
        step_failures: List[str] = None,
    ) -> None:
        """Record a capability execution for profiling."""
        profile = self._profiles.get(capability)
        if not profile:
            profile = CapabilityProfile(name=capability)
            self._profiles[capability] = profile

        profile.invocations += 1
        # Running averages
        n = profile.invocations
        profile.avg_latency_ms = (profile.avg_latency_ms * (n - 1) + latency_ms) / n
        profile.success_rate = (
            profile.success_rate * (n - 1) + (1.0 if success else 0.0)
        ) / n
        profile.token_cost = (profile.token_cost * (n - 1) + tokens) // n

        if step_latencies:
            for step, lat in step_latencies.items():
                prev = profile.step_latencies.get(step, lat)
                profile.step_latencies[step] = (prev * (n - 1) + lat) / n

        if step_failures:
            for step in step_failures:
                prev = profile.step_failure_rates.get(step, 0.0)
                profile.step_failure_rates[step] = (prev * (n - 1) + 1.0) / n

    # ── Analysis ──────────────────────────────────────────────────────────────

    def analyse(self) -> List[EvolutionProposal]:
        """Analyse all profiles and generate proposals."""
        self._analysis_count += 1
        new_proposals = []

        for name, profile in self._profiles.items():
            if profile.invocations < self.MIN_INVOCATIONS:
                continue
            new_proposals.extend(self._analyse_profile(profile))

        self._proposals.extend(new_proposals)
        return new_proposals

    def _analyse_profile(self, profile: CapabilityProfile) -> List[EvolutionProposal]:
        proposals = []

        # 1. Slow steps → suggest replacement or caching
        for step, lat in profile.step_latencies.items():
            if lat > self.SLOW_STEP_THRESHOLD_MS:
                proposals.append(
                    EvolutionProposal(
                        capability=profile.name,
                        strategy=EvolutionStrategy.CACHE_OUTPUT,
                        description=f"Step '{step}' averages {lat:.0f}ms — consider caching",
                        expected_improvement=f"Reduce latency by ~{lat * 0.6:.0f}ms",
                        details={"step": step, "current_latency_ms": lat},
                        confidence=min(1.0, profile.invocations / 20),
                    )
                )

        # 2. High failure-rate steps → suggest replacement or fallback
        for step, rate in profile.step_failure_rates.items():
            if rate > (1 - self.LOW_SUCCESS_THRESHOLD):
                proposals.append(
                    EvolutionProposal(
                        capability=profile.name,
                        strategy=EvolutionStrategy.ADD_FALLBACK,
                        description=(
                            f"Step '{step}' has {rate:.0%} failure rate — add fallback"
                        ),
                        expected_improvement=f"Improve success rate from {profile.success_rate:.0%}",
                        details={"step": step, "failure_rate": rate},
                        confidence=min(1.0, profile.invocations / 10),
                    )
                )

        # 3. Overall low success → suggest tool replacement
        if profile.success_rate < self.LOW_SUCCESS_THRESHOLD:
            proposals.append(
                EvolutionProposal(
                    capability=profile.name,
                    strategy=EvolutionStrategy.REPLACE_TOOL,
                    description=(
                        f"Capability '{profile.name}' success rate {profile.success_rate:.0%} — "
                        f"review tool chain"
                    ),
                    expected_improvement="Improve overall success rate",
                    details={"current_success_rate": profile.success_rate},
                    confidence=min(1.0, profile.invocations / 15),
                )
            )

        # 4. High token cost → suggest parameter tuning
        if profile.token_cost > self.HIGH_TOKEN_THRESHOLD:
            proposals.append(
                EvolutionProposal(
                    capability=profile.name,
                    strategy=EvolutionStrategy.TUNE_PARAMETERS,
                    description=(
                        f"'{profile.name}' uses ~{profile.token_cost} tokens/exec — "
                        f"tune parameters to reduce cost"
                    ),
                    expected_improvement=f"Reduce token usage by ~30%",
                    details={"current_token_cost": profile.token_cost},
                    confidence=0.5,
                )
            )

        # 5. Parallelization: if ≥2 consecutive steps have no data dependency
        if len(profile.steps) >= 3:
            for i in range(len(profile.steps) - 1):
                step_a = profile.steps[i]
                step_b = profile.steps[i + 1]
                lat_a = profile.step_latencies.get(step_a, 0)
                lat_b = profile.step_latencies.get(step_b, 0)
                if lat_a > 100 and lat_b > 100:
                    proposals.append(
                        EvolutionProposal(
                            capability=profile.name,
                            strategy=EvolutionStrategy.PARALLELIZE,
                            description=(
                                f"Steps '{step_a}' + '{step_b}' may run in parallel"
                            ),
                            expected_improvement=(
                                f"Save ~{min(lat_a, lat_b):.0f}ms by parallelizing"
                            ),
                            details={"steps": [step_a, step_b]},
                            confidence=0.4,
                        )
                    )

        return proposals

    # ── Proposal management ───────────────────────────────────────────────────

    def approve(self, proposal_id: str) -> bool:
        for p in self._proposals:
            if p.proposal_id == proposal_id and p.status == ProposalStatus.PENDING:
                p.status = ProposalStatus.APPROVED
                return True
        return False

    def reject(self, proposal_id: str) -> bool:
        for p in self._proposals:
            if p.proposal_id == proposal_id and p.status == ProposalStatus.PENDING:
                p.status = ProposalStatus.REJECTED
                return True
        return False

    def mark_tested(self, proposal_id: str) -> bool:
        for p in self._proposals:
            if p.proposal_id == proposal_id and p.status == ProposalStatus.APPROVED:
                p.status = ProposalStatus.TESTED
                return True
        return False

    def mark_applied(self, proposal_id: str) -> bool:
        for p in self._proposals:
            if p.proposal_id == proposal_id and p.status == ProposalStatus.TESTED:
                p.status = ProposalStatus.APPLIED
                self._applied.append(p)
                return True
        return False

    def mark_rolled_back(self, proposal_id: str) -> bool:
        for p in self._proposals:
            if p.proposal_id == proposal_id and p.status == ProposalStatus.APPLIED:
                p.status = ProposalStatus.ROLLED_BACK
                return True
        return False

    def pending_proposals(self) -> List[EvolutionProposal]:
        return [p for p in self._proposals if p.status == ProposalStatus.PENDING]

    def get_proposal(self, proposal_id: str) -> Optional[EvolutionProposal]:
        for p in self._proposals:
            if p.proposal_id == proposal_id:
                return p
        return None

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_profile(self, capability: str) -> Optional[CapabilityProfile]:
        return self._profiles.get(capability)

    def all_profiles(self) -> Dict[str, CapabilityProfile]:
        return dict(self._profiles)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "capabilities_tracked": len(self._profiles),
            "analysis_count": self._analysis_count,
            "total_proposals": len(self._proposals),
            "pending_proposals": len(self.pending_proposals()),
            "applied_proposals": len(self._applied),
            "profiles": {
                name: {
                    "invocations": p.invocations,
                    "avg_latency_ms": round(p.avg_latency_ms, 1),
                    "success_rate": round(p.success_rate, 3),
                    "token_cost": p.token_cost,
                }
                for name, p in self._profiles.items()
            },
        }
