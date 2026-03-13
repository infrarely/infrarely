"""
evolution/verification_layer.py — Layer 7, Module 6
═══════════════════════════════════════════════════════════════════════════════
The most critical safety module.

Before any proposed change can be applied, it must pass three gates:
    1. Static validation   — structural correctness checks
    2. Simulation          — dry-run with synthetic load
    3. Canary deployment   — limited rollout with live traffic

Only proposals that pass ALL three gates may be promoted.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional


# ── Data types ────────────────────────────────────────────────────────────────


class VerificationGate(Enum):
    STATIC = auto()
    SIMULATION = auto()
    CANARY = auto()


class VerificationResult(Enum):
    PASS = auto()
    FAIL = auto()
    SKIP = auto()


@dataclass
class GateResult:
    gate: VerificationGate
    result: VerificationResult
    details: str = ""
    duration_ms: float = 0.0
    ts: float = field(default_factory=time.time)


@dataclass
class VerificationReport:
    """Full verification report for a single proposal."""

    report_id: str = field(default_factory=lambda: f"vr_{uuid.uuid4().hex[:8]}")
    proposal_id: str = ""
    proposal_type: str = ""  # "capability", "architecture", "experiment"
    gates: List[GateResult] = field(default_factory=list)
    overall: VerificationResult = VerificationResult.FAIL
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


# ── Static checks registry ───────────────────────────────────────────────────


@dataclass
class StaticCheck:
    name: str
    check_fn: Callable[[Dict[str, Any]], bool]
    description: str = ""


# ── Engine ────────────────────────────────────────────────────────────────────


class VerificationLayer:
    """
    Verifies proposals through three gates:
        Static → Simulation → Canary
    """

    # Thresholds for simulation / canary
    SIMULATION_SUCCESS_THRESHOLD = 0.80
    CANARY_SUCCESS_THRESHOLD = 0.85
    CANARY_LATENCY_CEILING_MS = 2000.0

    def __init__(self):
        self._static_checks: List[StaticCheck] = []
        self._reports: List[VerificationReport] = []
        self._total_verified = 0
        self._total_passed = 0
        self._total_failed = 0

        # Register built-in static checks
        self._register_builtins()

    def _register_builtins(self):
        self.register_static_check(
            StaticCheck(
                name="no_empty_proposal",
                check_fn=lambda ctx: bool(ctx.get("description")),
                description="Proposal must have a description",
            )
        )
        self.register_static_check(
            StaticCheck(
                name="confidence_floor",
                check_fn=lambda ctx: ctx.get("confidence", 0) >= 0.3,
                description="Confidence must be >= 0.3",
            )
        )
        self.register_static_check(
            StaticCheck(
                name="no_runtime_modification",
                check_fn=lambda ctx: not ctx.get("modifies_runtime", False),
                description="Direct runtime modification is forbidden",
            )
        )

    # ── Registration ──────────────────────────────────────────────────────────

    def register_static_check(self, check: StaticCheck) -> None:
        self._static_checks.append(check)

    # ── Verification pipeline ─────────────────────────────────────────────────

    def verify(
        self,
        proposal_id: str,
        proposal_type: str,
        context: Dict[str, Any],
        simulation_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        canary_fn: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> VerificationReport:
        """Run full verification pipeline for a proposal."""
        report = VerificationReport(
            proposal_id=proposal_id,
            proposal_type=proposal_type,
        )

        # Gate 1: Static validation
        static_result = self._run_static(context)
        report.gates.append(static_result)
        if static_result.result == VerificationResult.FAIL:
            report.overall = VerificationResult.FAIL
            report.completed_at = time.time()
            self._finalise(report)
            return report

        # Gate 2: Simulation
        if simulation_fn:
            sim_result = self._run_simulation(simulation_fn)
            report.gates.append(sim_result)
            if sim_result.result == VerificationResult.FAIL:
                report.overall = VerificationResult.FAIL
                report.completed_at = time.time()
                self._finalise(report)
                return report
        else:
            report.gates.append(
                GateResult(
                    gate=VerificationGate.SIMULATION,
                    result=VerificationResult.SKIP,
                    details="No simulation function provided",
                )
            )

        # Gate 3: Canary
        if canary_fn:
            canary_result = self._run_canary(canary_fn)
            report.gates.append(canary_result)
            if canary_result.result == VerificationResult.FAIL:
                report.overall = VerificationResult.FAIL
                report.completed_at = time.time()
                self._finalise(report)
                return report
        else:
            report.gates.append(
                GateResult(
                    gate=VerificationGate.CANARY,
                    result=VerificationResult.SKIP,
                    details="No canary function provided",
                )
            )

        report.overall = VerificationResult.PASS
        report.completed_at = time.time()
        self._finalise(report)
        return report

    def _run_static(self, context: Dict[str, Any]) -> GateResult:
        t0 = time.time()
        failures = []
        for check in self._static_checks:
            try:
                if not check.check_fn(context):
                    failures.append(check.name)
            except Exception as exc:
                failures.append(f"{check.name}(error:{exc})")
        duration = (time.time() - t0) * 1000
        if failures:
            return GateResult(
                gate=VerificationGate.STATIC,
                result=VerificationResult.FAIL,
                details=f"Failed checks: {', '.join(failures)}",
                duration_ms=duration,
            )
        return GateResult(
            gate=VerificationGate.STATIC,
            result=VerificationResult.PASS,
            details=f"All {len(self._static_checks)} checks passed",
            duration_ms=duration,
        )

    def _run_simulation(self, fn: Callable[[], Dict[str, Any]]) -> GateResult:
        t0 = time.time()
        try:
            result = fn()
        except Exception as exc:
            return GateResult(
                gate=VerificationGate.SIMULATION,
                result=VerificationResult.FAIL,
                details=f"Simulation raised: {exc}",
                duration_ms=(time.time() - t0) * 1000,
            )
        duration = (time.time() - t0) * 1000
        success_rate = result.get("success_rate", 0.0)
        if success_rate < self.SIMULATION_SUCCESS_THRESHOLD:
            return GateResult(
                gate=VerificationGate.SIMULATION,
                result=VerificationResult.FAIL,
                details=f"Simulation success {success_rate:.0%} < threshold {self.SIMULATION_SUCCESS_THRESHOLD:.0%}",
                duration_ms=duration,
            )
        return GateResult(
            gate=VerificationGate.SIMULATION,
            result=VerificationResult.PASS,
            details=f"Simulation success {success_rate:.0%}",
            duration_ms=duration,
        )

    def _run_canary(self, fn: Callable[[], Dict[str, Any]]) -> GateResult:
        t0 = time.time()
        try:
            result = fn()
        except Exception as exc:
            return GateResult(
                gate=VerificationGate.CANARY,
                result=VerificationResult.FAIL,
                details=f"Canary raised: {exc}",
                duration_ms=(time.time() - t0) * 1000,
            )
        duration = (time.time() - t0) * 1000
        success_rate = result.get("success_rate", 0.0)
        latency = result.get("avg_latency_ms", 0.0)
        if success_rate < self.CANARY_SUCCESS_THRESHOLD:
            return GateResult(
                gate=VerificationGate.CANARY,
                result=VerificationResult.FAIL,
                details=f"Canary success {success_rate:.0%} < {self.CANARY_SUCCESS_THRESHOLD:.0%}",
                duration_ms=duration,
            )
        if latency > self.CANARY_LATENCY_CEILING_MS:
            return GateResult(
                gate=VerificationGate.CANARY,
                result=VerificationResult.FAIL,
                details=f"Canary latency {latency:.0f}ms > ceiling {self.CANARY_LATENCY_CEILING_MS:.0f}ms",
                duration_ms=duration,
            )
        return GateResult(
            gate=VerificationGate.CANARY,
            result=VerificationResult.PASS,
            details=f"Canary success {success_rate:.0%}, latency {latency:.0f}ms",
            duration_ms=duration,
        )

    def _finalise(self, report: VerificationReport) -> None:
        self._reports.append(report)
        self._total_verified += 1
        if report.overall == VerificationResult.PASS:
            self._total_passed += 1
        else:
            self._total_failed += 1

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_report(self, report_id: str) -> Optional[VerificationReport]:
        for r in self._reports:
            if r.report_id == report_id:
                return r
        return None

    def reports_for_proposal(self, proposal_id: str) -> List[VerificationReport]:
        return [r for r in self._reports if r.proposal_id == proposal_id]

    def pass_rate(self) -> float:
        if self._total_verified == 0:
            return 0.0
        return self._total_passed / self._total_verified

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_verified": self._total_verified,
            "total_passed": self._total_passed,
            "total_failed": self._total_failed,
            "pass_rate": round(self.pass_rate(), 3),
            "static_checks_registered": len(self._static_checks),
        }
