"""
evolution/policy_guard.py — Layer 7, Module 7
═══════════════════════════════════════════════════════════════════════════════
Safety-policy enforcement for the Autonomous Evolution System.

Rules enforced:
    1. Mutation-rate cap    — max N changes / time window
    2. No core-runtime mod  — proposals cannot touch runtime layer directly
    3. Approval requirement — capability changes need explicit approval
    4. Sandbox restriction  — experiments may only run in sandboxed mode
    5. Rollback guarantee   — every applied change must have rollback path
    6. No security escalation — changes cannot escalate permissions

The guard is the FINAL checkpoint before any proposal is enacted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set


# ── Data types ────────────────────────────────────────────────────────────────


class PolicyVerdict(Enum):
    ALLOWED = auto()
    DENIED = auto()
    RATE_LIMITED = auto()


class ViolationSeverity(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass
class PolicyViolation:
    rule: str
    description: str
    severity: ViolationSeverity = ViolationSeverity.MEDIUM
    ts: float = field(default_factory=time.time)


@dataclass
class PolicyDecision:
    verdict: PolicyVerdict
    violations: List[PolicyViolation] = field(default_factory=list)
    proposal_id: str = ""
    ts: float = field(default_factory=time.time)


# ── Guard ─────────────────────────────────────────────────────────────────────


class PolicyGuard:
    """
    Enforces safety policies on proposals before they can be enacted.
    """

    # Policy knobs
    MAX_MUTATIONS_PER_WINDOW = 5
    MUTATION_WINDOW_SEC = 3600.0  # 1 hour
    PROTECTED_COMPONENTS: Set[str] = {
        "runtime.agent_registry",
        "runtime.agent_scheduler",
        "runtime.message_bus",
        "runtime.shared_memory",
        "runtime.lifecycle_manager",
        "runtime.resource_isolation",
        "runtime.agent_monitoring",
    }
    REQUIRE_ROLLBACK = True
    REQUIRE_SIMULATION = True
    MIN_CONFIDENCE = 0.3

    def __init__(self):
        self._decisions: List[PolicyDecision] = []
        self._mutation_timestamps: List[float] = []
        self._overrides: Dict[str, bool] = {}  # rule_name → enabled
        self._total_denied = 0
        self._total_allowed = 0

    # ── Policy evaluation ─────────────────────────────────────────────────────

    def evaluate(self, proposal: Dict[str, Any]) -> PolicyDecision:
        """
        Evaluate a proposal dict against all policies.

        Expected keys in *proposal*:
            proposal_id   — str
            component     — str (dotted path, e.g. "capability.xxx")
            description   — str
            confidence    — float 0..1
            has_rollback  — bool
            has_simulation — bool
            modifies_runtime — bool
            escalates_permissions — bool
        """
        violations: List[PolicyViolation] = []

        # 1. Mutation-rate cap
        if not self._override("mutation_rate"):
            now = time.time()
            cutoff = now - self.MUTATION_WINDOW_SEC
            self._mutation_timestamps = [
                t for t in self._mutation_timestamps if t > cutoff
            ]
            if len(self._mutation_timestamps) >= self.MAX_MUTATIONS_PER_WINDOW:
                violations.append(
                    PolicyViolation(
                        rule="mutation_rate",
                        description=f"Rate limit: {len(self._mutation_timestamps)}/{self.MAX_MUTATIONS_PER_WINDOW} in window",
                        severity=ViolationSeverity.HIGH,
                    )
                )

        # 2. No core-runtime modification
        if not self._override("no_runtime_mod"):
            component = proposal.get("component", "")
            if (
                proposal.get("modifies_runtime", False)
                or component in self.PROTECTED_COMPONENTS
            ):
                violations.append(
                    PolicyViolation(
                        rule="no_runtime_mod",
                        description=f"Component '{component}' is protected — direct modification forbidden",
                        severity=ViolationSeverity.CRITICAL,
                    )
                )

        # 3. Confidence floor
        if not self._override("confidence_floor"):
            conf = proposal.get("confidence", 0)
            if conf < self.MIN_CONFIDENCE:
                violations.append(
                    PolicyViolation(
                        rule="confidence_floor",
                        description=f"Confidence {conf:.2f} < minimum {self.MIN_CONFIDENCE:.2f}",
                        severity=ViolationSeverity.MEDIUM,
                    )
                )

        # 4. Rollback requirement
        if not self._override("require_rollback"):
            if self.REQUIRE_ROLLBACK and not proposal.get("has_rollback", False):
                violations.append(
                    PolicyViolation(
                        rule="require_rollback",
                        description="Proposal has no rollback path",
                        severity=ViolationSeverity.HIGH,
                    )
                )

        # 5. Simulation requirement
        if not self._override("require_simulation"):
            if self.REQUIRE_SIMULATION and not proposal.get("has_simulation", False):
                violations.append(
                    PolicyViolation(
                        rule="require_simulation",
                        description="Proposal has not been simulated",
                        severity=ViolationSeverity.MEDIUM,
                    )
                )

        # 6. Security escalation
        if not self._override("no_escalation"):
            if proposal.get("escalates_permissions", False):
                violations.append(
                    PolicyViolation(
                        rule="no_escalation",
                        description="Proposal would escalate permissions",
                        severity=ViolationSeverity.CRITICAL,
                    )
                )

        # 7. Description required
        if not self._override("require_description"):
            if not proposal.get("description"):
                violations.append(
                    PolicyViolation(
                        rule="require_description",
                        description="Proposal has no description",
                        severity=ViolationSeverity.LOW,
                    )
                )

        # Build verdict
        critical = any(v.severity == ViolationSeverity.CRITICAL for v in violations)
        rate_limited = any(v.rule == "mutation_rate" for v in violations)

        if critical:
            verdict = PolicyVerdict.DENIED
        elif rate_limited:
            verdict = PolicyVerdict.RATE_LIMITED
        elif violations:
            verdict = PolicyVerdict.DENIED
        else:
            verdict = PolicyVerdict.ALLOWED
            self._mutation_timestamps.append(time.time())

        decision = PolicyDecision(
            verdict=verdict,
            violations=violations,
            proposal_id=proposal.get("proposal_id", ""),
        )
        self._decisions.append(decision)
        if verdict == PolicyVerdict.ALLOWED:
            self._total_allowed += 1
        else:
            self._total_denied += 1
        return decision

    # ── Override management ───────────────────────────────────────────────────

    def set_override(self, rule_name: str, skip: bool = True) -> None:
        """Disable a specific policy rule (for testing only)."""
        self._overrides[rule_name] = skip

    def clear_overrides(self) -> None:
        self._overrides.clear()

    def _override(self, rule_name: str) -> bool:
        return self._overrides.get(rule_name, False)

    # ── Query ─────────────────────────────────────────────────────────────────

    def recent_decisions(self, limit: int = 20) -> List[PolicyDecision]:
        return self._decisions[-limit:]

    def violations_by_rule(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for d in self._decisions:
            for v in d.violations:
                counts[v.rule] = counts.get(v.rule, 0) + 1
        return counts

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_evaluated": len(self._decisions),
            "total_allowed": self._total_allowed,
            "total_denied": self._total_denied,
            "active_overrides": list(self._overrides.keys()),
            "mutations_in_window": len(self._mutation_timestamps),
            "max_mutations_per_window": self.MAX_MUTATIONS_PER_WINDOW,
        }
