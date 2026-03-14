"""
agent/planning_engine.py — CAPABILITY 2: Deterministic Planning Engine
═══════════════════════════════════════════════════════════════════════════════
Two-phase planning architecture with deterministic compilation boundary.

PROBLEM SOLVED:
  Current agent planners use LLM to plan AND execute.
  This causes: hallucinated steps, non-deterministic ordering,
  unvalidated tool references, infinite loops, unrecoverable failures.

THIS IMPLEMENTATION:
  Phase 1 — PLAN GENERATION (LLM permitted here ONLY)
    Goal → PlannerLLM → RawPlan (strict JSON schema) → PlanValidator
    If FAIL: re-prompt with specific errors (max 3 attempts)
    If PASS: proceed to Phase 2

  Phase 2 — PLAN COMPILATION (FULLY DETERMINISTIC)
    RawPlan → CapabilityCompiler → ExecutionGraph → RuntimeExecution

  PlanValidator — 6 Deterministic Gates (ALL must pass):
    Gate 1: capability_exists_check
    Gate 2: dag_integrity_check
    Gate 3: input_reference_check
    Gate 4: token_budget_check
    Gate 5: tool_availability_check
    Gate 6: resource_budget_check

RULES (from InfraRely spec):
  RULE 4 — PLANNING IS NOT EXECUTION
    Plans are generated once, compiled once, validated once.
    Execution is always deterministic after compilation.
    A plan that fails validation is re-planned with specific error context.
    A plan is NEVER re-generated during execution.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from infrarely.observability import logger


# ═══════════════════════════════════════════════════════════════════════════════
# RAW PLAN SCHEMA — LLM must output this exactly
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RawPlanStep:
    """A single step in the raw plan from LLM or deterministic planner."""

    id: str  # "step_NNN"
    capability: str  # must exist in CapabilityRegistry
    inputs: Dict[str, Any] = field(
        default_factory=dict
    )  # static or "step_NNN.output" refs
    depends_on: List[str] = field(default_factory=list)  # step IDs
    can_parallelize: bool = False
    fallback: Optional[str] = None  # fallback capability if this fails
    timeout_seconds: int = 30
    required: bool = True  # if False, SKIPPED on failure is OK

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "capability": self.capability,
            "inputs": self.inputs,
            "depends_on": self.depends_on,
            "can_parallelize": self.can_parallelize,
            "fallback": self.fallback,
            "timeout_seconds": self.timeout_seconds,
            "required": self.required,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RawPlanStep":
        return RawPlanStep(
            id=d.get("id", ""),
            capability=d.get("capability", ""),
            inputs=d.get("inputs", {}),
            depends_on=d.get("depends_on", []),
            can_parallelize=d.get("can_parallelize", False),
            fallback=d.get("fallback"),
            timeout_seconds=d.get("timeout_seconds", 30),
            required=d.get("required", True),
        )


@dataclass
class RawPlan:
    """
    The raw plan structure — output of Phase 1 (LLM or deterministic).
    This is the ONLY format accepted by the PlanValidator.
    """

    goal: str
    steps: List[RawPlanStep] = field(default_factory=list)
    validation_rules: List[str] = field(default_factory=list)
    max_retries: int = 3
    estimated_tokens: int = 0
    confidence_score: float = 0.0
    plan_id: str = field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}")
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source: str = "deterministic"  # "deterministic" | "llm" | "cached"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "validation_rules": self.validation_rules,
            "max_retries": self.max_retries,
            "estimated_tokens": self.estimated_tokens,
            "confidence_score": self.confidence_score,
            "created_at": self.created_at,
            "source": self.source,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RawPlan":
        return RawPlan(
            goal=d.get("goal", ""),
            steps=[RawPlanStep.from_dict(s) for s in d.get("steps", [])],
            validation_rules=d.get("validation_rules", []),
            max_retries=d.get("max_retries", 3),
            estimated_tokens=d.get("estimated_tokens", 0),
            confidence_score=d.get("confidence_score", 0.0),
            plan_id=d.get("plan_id", f"plan_{uuid.uuid4().hex[:8]}"),
            created_at=d.get("created_at", datetime.now(timezone.utc).isoformat()),
            source=d.get("source", "deterministic"),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION GATE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════


class GateStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class GateResult:
    """Result of a single validation gate."""

    gate_name: str
    gate_number: int
    status: GateStatus
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "gate_number": self.gate_number,
            "status": self.status.value,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details,
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass
class ValidationResult:
    """Aggregate result of all 6 validation gates."""

    plan_id: str
    passed: bool
    gates: List[GateResult] = field(default_factory=list)
    total_errors: int = 0
    total_warnings: int = 0
    validation_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "passed": self.passed,
            "gates": [g.to_dict() for g in self.gates],
            "total_errors": self.total_errors,
            "total_warnings": self.total_warnings,
            "validation_ms": round(self.validation_ms, 2),
        }

    def failed_gates(self) -> List[GateResult]:
        return [g for g in self.gates if g.status == GateStatus.FAILED]

    def failure_context(self) -> List[Dict[str, Any]]:
        """Generate specific failure context for LLM re-prompting."""
        context = []
        for gate in self.failed_gates():
            context.append(
                {
                    "failed_gate": gate.gate_name,
                    "gate_number": gate.gate_number,
                    "errors": gate.errors,
                    "details": gate.details,
                }
            )
        return context


# ═══════════════════════════════════════════════════════════════════════════════
# PLAN VALIDATOR — 6 Deterministic Gates (ALL must pass)
# ═══════════════════════════════════════════════════════════════════════════════


class PlanValidator:
    """
    Validates a RawPlan through 6 deterministic gates.
    NO LLM involved — pure structural/logical validation.

    Gate 1: capability_exists_check     → all capabilities in registry
    Gate 2: dag_integrity_check         → no cycles, no dangling nodes
    Gate 3: input_reference_check       → all step_NNN.output refs are valid
    Gate 4: token_budget_check          → estimated_tokens within limits
    Gate 5: tool_availability_check     → all required tools ONLINE
    Gate 6: resource_budget_check       → execution depth + token spend within policy
    """

    def __init__(
        self,
        capability_registry=None,
        tool_registry=None,
        token_budget_limit: int = 5000,
        max_execution_depth: int = 8,
        max_steps: int = 20,
    ):
        self._capability_registry = capability_registry
        self._tool_registry = tool_registry
        self._token_budget_limit = token_budget_limit
        self._max_execution_depth = max_execution_depth
        self._max_steps = max_steps

    def validate(self, plan: RawPlan) -> ValidationResult:
        """
        Run all 6 validation gates sequentially.
        All gates run even if early gates fail (for complete error reporting).
        """
        t_start = time.monotonic()
        gates: List[GateResult] = []

        # Gate 1: Capability exists check
        gates.append(self._gate_1_capability_exists(plan))

        # Gate 2: DAG integrity check
        gates.append(self._gate_2_dag_integrity(plan))

        # Gate 3: Input reference check
        gates.append(self._gate_3_input_references(plan))

        # Gate 4: Token budget check
        gates.append(self._gate_4_token_budget(plan))

        # Gate 5: Tool availability check
        gates.append(self._gate_5_tool_availability(plan))

        # Gate 6: Resource budget check
        gates.append(self._gate_6_resource_budget(plan))

        total_errors = sum(len(g.errors) for g in gates)
        total_warnings = sum(len(g.warnings) for g in gates)
        all_passed = all(g.status != GateStatus.FAILED for g in gates)

        result = ValidationResult(
            plan_id=plan.plan_id,
            passed=all_passed,
            gates=gates,
            total_errors=total_errors,
            total_warnings=total_warnings,
            validation_ms=(time.monotonic() - t_start) * 1000,
        )

        if all_passed:
            logger.info(
                f"PlanValidator: all 6 gates PASSED for plan '{plan.plan_id}'",
                warnings=total_warnings,
            )
        else:
            logger.warn(
                f"PlanValidator: validation FAILED for plan '{plan.plan_id}'",
                failed_gates=[g.gate_name for g in result.failed_gates()],
                errors=total_errors,
            )

        return result

    # ── Gate 1: Capability Exists Check ───────────────────────────────────────

    def _gate_1_capability_exists(self, plan: RawPlan) -> GateResult:
        """Verify all step capabilities exist in the registry."""
        t = time.monotonic()
        errors = []
        warnings = []
        details = {"checked": [], "missing": []}

        for step in plan.steps:
            details["checked"].append(step.capability)

            # Check in tool registry if no capability registry
            found = False
            if self._tool_registry and step.capability in self._tool_registry:
                found = True
            elif self._capability_registry:
                # Check if it's a registered capability name
                cap = (
                    self._capability_registry.get(step.capability)
                    if hasattr(self._capability_registry, "get")
                    else self._capability_registry.match(step.capability)
                )
                if cap:
                    found = True

            # Also accept if it exists as tool name
            if (
                not found
                and self._tool_registry
                and step.capability in self._tool_registry
            ):
                found = True

            if not found:
                # Soft check: if no registries available, warn > error
                if self._tool_registry is None and self._capability_registry is None:
                    warnings.append(
                        f"Step '{step.id}': capability '{step.capability}' "
                        f"cannot be verified (no registry available)"
                    )
                else:
                    errors.append(
                        f"Step '{step.id}': capability '{step.capability}' "
                        f"not found in registry"
                    )
                    details["missing"].append(step.capability)

            # Check fallback too
            if step.fallback:
                fallback_found = False
                if self._tool_registry and step.fallback in self._tool_registry:
                    fallback_found = True
                if not fallback_found:
                    warnings.append(
                        f"Step '{step.id}': fallback '{step.fallback}' not in registry"
                    )

        return GateResult(
            gate_name="capability_exists_check",
            gate_number=1,
            status=GateStatus.FAILED if errors else GateStatus.PASSED,
            errors=errors,
            warnings=warnings,
            details=details,
            duration_ms=(time.monotonic() - t) * 1000,
        )

    # ── Gate 2: DAG Integrity Check ───────────────────────────────────────────

    def _gate_2_dag_integrity(self, plan: RawPlan) -> GateResult:
        """Verify the step dependency graph has no cycles and no dangling refs."""
        t = time.monotonic()
        errors = []
        warnings = []
        details = {}

        step_ids = {s.id for s in plan.steps}
        details["step_ids"] = list(step_ids)

        # Check for duplicate step IDs
        all_ids = [s.id for s in plan.steps]
        seen = set()
        for sid in all_ids:
            if sid in seen:
                errors.append(f"Duplicate step ID: '{sid}'")
            seen.add(sid)

        # Check for dangling dependency references
        for step in plan.steps:
            for dep in step.depends_on:
                if dep not in step_ids:
                    errors.append(
                        f"Step '{step.id}' depends on '{dep}' which does not exist"
                    )

        # Cycle detection (DFS-based)
        graph: Dict[str, List[str]] = {s.id: s.depends_on for s in plan.steps}
        cycle = self._detect_cycle(graph, step_ids)
        if cycle:
            errors.append(f"Dependency cycle detected: {' → '.join(cycle)}")
            details["cycle"] = cycle

        # Check for self-dependencies
        for step in plan.steps:
            if step.id in step.depends_on:
                errors.append(f"Step '{step.id}' depends on itself")

        return GateResult(
            gate_name="dag_integrity_check",
            gate_number=2,
            status=GateStatus.FAILED if errors else GateStatus.PASSED,
            errors=errors,
            warnings=warnings,
            details=details,
            duration_ms=(time.monotonic() - t) * 1000,
        )

    # ── Gate 3: Input Reference Check ─────────────────────────────────────────

    def _gate_3_input_references(self, plan: RawPlan) -> GateResult:
        """Verify all step_NNN.output references point to valid steps."""
        t = time.monotonic()
        errors = []
        warnings = []
        details = {"references_checked": 0, "valid": 0, "invalid": 0}

        step_ids = {s.id for s in plan.steps}

        # Build execution order to check reference ordering
        step_order = {s.id: i for i, s in enumerate(plan.steps)}

        for step in plan.steps:
            for key, value in step.inputs.items():
                if isinstance(value, str) and "." in value:
                    # Check for step_NNN.output pattern
                    parts = value.split(".")
                    ref_step = parts[0]
                    details["references_checked"] += 1

                    if ref_step.startswith("step_") or ref_step in step_ids:
                        if ref_step not in step_ids:
                            errors.append(
                                f"Step '{step.id}' input '{key}' references "
                                f"'{ref_step}' which does not exist"
                            )
                            details["invalid"] += 1
                        else:
                            # Check ordering: referenced step must come before
                            ref_idx = step_order.get(ref_step, -1)
                            cur_idx = step_order.get(step.id, -1)
                            if ref_idx >= cur_idx and ref_step not in step.depends_on:
                                warnings.append(
                                    f"Step '{step.id}' references '{ref_step}' "
                                    f"that isn't declared as a dependency"
                                )
                            details["valid"] += 1

        return GateResult(
            gate_name="input_reference_check",
            gate_number=3,
            status=GateStatus.FAILED if errors else GateStatus.PASSED,
            errors=errors,
            warnings=warnings,
            details=details,
            duration_ms=(time.monotonic() - t) * 1000,
        )

    # ── Gate 4: Token Budget Check ────────────────────────────────────────────

    def _gate_4_token_budget(self, plan: RawPlan) -> GateResult:
        """Verify estimated token usage is within budget limits."""
        t = time.monotonic()
        errors = []
        warnings = []
        details = {
            "estimated_tokens": plan.estimated_tokens,
            "budget_limit": self._token_budget_limit,
        }

        if plan.estimated_tokens > self._token_budget_limit:
            errors.append(
                f"Estimated tokens ({plan.estimated_tokens}) exceeds budget "
                f"({self._token_budget_limit})"
            )

        # Warn at 80% threshold
        if plan.estimated_tokens > self._token_budget_limit * 0.8:
            warnings.append(
                f"Estimated tokens ({plan.estimated_tokens}) is >80% of budget "
                f"({self._token_budget_limit})"
            )

        return GateResult(
            gate_name="token_budget_check",
            gate_number=4,
            status=GateStatus.FAILED if errors else GateStatus.PASSED,
            errors=errors,
            warnings=warnings,
            details=details,
            duration_ms=(time.monotonic() - t) * 1000,
        )

    # ── Gate 5: Tool Availability Check ───────────────────────────────────────

    def _gate_5_tool_availability(self, plan: RawPlan) -> GateResult:
        """Verify all required tools are ONLINE (circuit breakers CLOSED)."""
        t = time.monotonic()
        errors = []
        warnings = []
        details = {"tools_checked": [], "unavailable": []}

        for step in plan.steps:
            tool_name = step.capability
            details["tools_checked"].append(tool_name)

            if self._tool_registry:
                tool = (
                    self._tool_registry.get(tool_name)
                    if hasattr(self._tool_registry, "get")
                    else None
                )

                # Check circuit breaker state if available
                if hasattr(self._tool_registry, "is_available"):
                    if not self._tool_registry.is_available(tool_name):
                        if step.required:
                            errors.append(
                                f"Required tool '{tool_name}' is unavailable "
                                f"(circuit breaker OPEN)"
                            )
                        else:
                            warnings.append(
                                f"Optional tool '{tool_name}' is unavailable"
                            )
                        details["unavailable"].append(tool_name)

        return GateResult(
            gate_name="tool_availability_check",
            gate_number=5,
            status=GateStatus.FAILED if errors else GateStatus.PASSED,
            errors=errors,
            warnings=warnings,
            details=details,
            duration_ms=(time.monotonic() - t) * 1000,
        )

    # ── Gate 6: Resource Budget Check ─────────────────────────────────────────

    def _gate_6_resource_budget(self, plan: RawPlan) -> GateResult:
        """Verify execution depth and resource consumption within policy."""
        t = time.monotonic()
        errors = []
        warnings = []
        details = {
            "step_count": len(plan.steps),
            "max_steps": self._max_steps,
            "max_execution_depth": self._max_execution_depth,
        }

        # Step count check
        if len(plan.steps) > self._max_steps:
            errors.append(
                f"Step count ({len(plan.steps)}) exceeds max_steps ({self._max_steps})"
            )

        # Execution depth check (longest dependency chain)
        depth = self._compute_max_depth(plan)
        details["computed_depth"] = depth
        if depth > self._max_execution_depth:
            errors.append(
                f"Execution depth ({depth}) exceeds max_execution_depth "
                f"({self._max_execution_depth})"
            )

        # Timeout budget check
        total_timeout = sum(s.timeout_seconds for s in plan.steps)
        details["total_timeout_seconds"] = total_timeout
        if total_timeout > 300:  # 5 minute hard cap
            warnings.append(
                f"Total timeout ({total_timeout}s) exceeds 5 minute soft cap"
            )

        return GateResult(
            gate_name="resource_budget_check",
            gate_number=6,
            status=GateStatus.FAILED if errors else GateStatus.PASSED,
            errors=errors,
            warnings=warnings,
            details=details,
            duration_ms=(time.monotonic() - t) * 1000,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _detect_cycle(
        self,
        graph: Dict[str, List[str]],
        all_nodes: Set[str],
    ) -> Optional[List[str]]:
        """DFS-based cycle detection. Returns cycle path or None."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in all_nodes}
        path: List[str] = []

        def dfs(node: str) -> bool:
            color[node] = GRAY
            path.append(node)
            for dep in graph.get(node, []):
                if dep not in color:
                    continue
                if color[dep] == GRAY:
                    path.append(dep)
                    return True
                if color[dep] == WHITE and dfs(dep):
                    return True
            path.pop()
            color[node] = BLACK
            return False

        for node in all_nodes:
            if color[node] == WHITE and dfs(node):
                cycle_start = path[-1]
                idx = path.index(cycle_start)
                return path[idx:]
        return None

    def _compute_max_depth(self, plan: RawPlan) -> int:
        """Compute the longest dependency chain (execution depth).
        Handles cycles gracefully by tracking visited nodes."""
        graph = {s.id: s.depends_on for s in plan.steps}
        memo: Dict[str, int] = {}

        def depth(node: str, visiting: Optional[Set[str]] = None) -> int:
            if node in memo:
                return memo[node]
            if visiting is None:
                visiting = set()
            if node in visiting:
                # Cycle detected — return 1 to avoid infinite recursion
                # (Gate 2 catches the actual cycle error)
                return 1
            visiting = visiting | {node}
            deps = [d for d in graph.get(node, []) if d in graph]
            if not deps:
                memo[node] = 1
                return 1
            max_dep = max(depth(d, visiting) for d in deps)
            memo[node] = max_dep + 1
            return memo[node]

        if not plan.steps:
            return 0
        return max(depth(s.id) for s in plan.steps)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPILED PLAN — Output of Phase 2 (fully deterministic)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CompiledExecutionGraph:
    """
    The fully compiled, validated, frozen execution graph.
    Once this is created, execution is 100% deterministic.
    NO LLM is ever consulted during execution of a compiled graph.
    """

    plan_id: str
    goal: str
    nodes: List["ExecutionNode"] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)  # topological order
    parallelizable_groups: List[List[str]] = field(default_factory=list)
    estimated_tokens: int = 0
    compiled_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    validation_result: Optional[ValidationResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "nodes": [n.to_dict() for n in self.nodes],
            "execution_order": self.execution_order,
            "parallelizable_groups": self.parallelizable_groups,
            "estimated_tokens": self.estimated_tokens,
            "compiled_at": self.compiled_at,
        }


@dataclass
class ExecutionNode:
    """A single node in the compiled execution graph."""

    node_id: str
    capability: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    fallback: Optional[str] = None
    timeout_seconds: int = 30
    required: bool = True
    can_parallelize: bool = False

    # Execution state (mutable during execution)
    status: str = "pending"  # pending | running | success | failed | skipped
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "capability": self.capability,
            "inputs": self.inputs,
            "dependencies": self.dependencies,
            "fallback": self.fallback,
            "timeout_seconds": self.timeout_seconds,
            "required": self.required,
            "can_parallelize": self.can_parallelize,
            "status": self.status,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# RE-PROMPT PROTOCOL — When validation fails
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RePlanRequest:
    """
    Context passed back to the LLM planner when a plan fails validation.
    Contains SPECIFIC gate failures — not generic "try again".
    """

    original_goal: str
    attempt_number: int
    max_attempts: int = 3
    failed_gates: List[Dict[str, Any]] = field(default_factory=list)
    previous_plan: Optional[Dict[str, Any]] = None
    constraints: Dict[str, Any] = field(default_factory=dict)

    def to_prompt_context(self) -> str:
        """Generate a structured prompt context for the LLM re-planner."""
        lines = [
            f"PLAN RE-GENERATION (attempt {self.attempt_number}/{self.max_attempts})",
            f"Goal: {self.original_goal}",
            "",
            "VALIDATION FAILURES (you must fix ALL of these):",
        ]
        for fg in self.failed_gates:
            lines.append(f"  Gate: {fg.get('failed_gate', 'unknown')}")
            for err in fg.get("errors", []):
                lines.append(f"    ERROR: {err}")
            details = fg.get("details", {})
            if details:
                lines.append(f"    Context: {json.dumps(details, default=str)}")
            lines.append("")

        if self.constraints:
            lines.append("CONSTRAINTS:")
            for k, v in self.constraints.items():
                lines.append(f"  {k}: {v}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINISTIC PLANNING ENGINE — The two-phase architecture
# ═══════════════════════════════════════════════════════════════════════════════


class DeterministicPlanningEngine:
    """
    The core planning engine implementing the two-phase architecture.

    Phase 1: Plan Generation (LLM permitted for novel goals ONLY)
      - If goal maps to registered capability → skip LLM entirely
      - Router confidence > 0.9 on known capability → use compiled graph directly
      - LLM planner only invoked for novel goals not in registry

    Phase 2: Plan Compilation (FULLY DETERMINISTIC)
      - RawPlan → PlanValidator (6 gates) → CompiledExecutionGraph
      - If validation fails → re-prompt with specific errors (max 3 attempts)
      - Compiled plans are cacheable and reusable

    SACRED BOUNDARY: Nothing crosses from Phase 1 to Phase 2 without validation.
    """

    MAX_REPLAN_ATTEMPTS = 3

    def __init__(
        self,
        capability_registry=None,
        tool_registry=None,
        token_budget_limit: int = 5000,
        max_execution_depth: int = 8,
    ):
        self._capability_registry = capability_registry
        self._tool_registry = tool_registry
        self._validator = PlanValidator(
            capability_registry=capability_registry,
            tool_registry=tool_registry,
            token_budget_limit=token_budget_limit,
            max_execution_depth=max_execution_depth,
        )
        self._plan_cache: OrderedDict[str, CompiledExecutionGraph] = OrderedDict()
        self._cache_max = 200
        self._stats = {
            "plans_generated": 0,
            "plans_validated": 0,
            "plans_failed": 0,
            "plans_cached_hits": 0,
            "plans_compiled": 0,
            "deterministic_bypasses": 0,
            "llm_plans_generated": 0,
            "replan_attempts": 0,
        }

    # ── Phase 1: Plan Generation ──────────────────────────────────────────────

    def generate_plan(
        self,
        goal: str,
        intent: str = "",
        params: Dict[str, Any] = None,
        router_confidence: float = 0.0,
        llm_planner: Optional[Callable] = None,
    ) -> Tuple[RawPlan, str]:
        """
        Phase 1: Generate a RawPlan.

        Priority order:
          1. Cache hit → return cached plan
          2. Registered capability graph → deterministic plan (no LLM)
          3. Router confidence > 0.9 → deterministic routing plan
          4. LLM planner (novel goals only)

        Returns: (RawPlan, generation_path)
        generation_path is one of: "cached", "deterministic", "router", "llm"
        """
        params = params or {}

        # ── 1. Check plan cache ───────────────────────────────────────────
        cache_key = self._cache_key(goal, intent, params)
        if cache_key in self._plan_cache:
            self._plan_cache.move_to_end(cache_key)
            self._stats["plans_cached_hits"] += 1
            cached = self._plan_cache[cache_key]
            logger.debug(f"PlanningEngine: cache HIT for '{goal[:40]}'")
            # Reconstruct RawPlan from cached graph
            raw = self._graph_to_raw_plan(cached)
            raw.source = "cached"
            return raw, "cached"

        # ── 2. Direct capability match (deterministic, no LLM) ────────────
        if self._capability_registry:
            match_fn = (
                self._capability_registry.match
                if hasattr(self._capability_registry, "match")
                else None
            )
            cap = match_fn(goal) if match_fn else None
            if cap:
                raw = self._capability_to_raw_plan(cap, goal, params)
                self._stats["deterministic_bypasses"] += 1
                logger.info(
                    f"PlanningEngine: deterministic plan from capability '{cap.name}'",
                )
                return raw, "deterministic"

        # ── 3. High-confidence router → deterministic plan ────────────────
        if router_confidence > 0.9 and intent:
            raw = self._intent_to_raw_plan(intent, goal, params)
            self._stats["deterministic_bypasses"] += 1
            logger.info(
                f"PlanningEngine: deterministic plan from router "
                f"(confidence={router_confidence:.2f})",
            )
            return raw, "router"

        # ── 4. LLM planner (novel goals) ─────────────────────────────────
        if llm_planner:
            raw = self._llm_generate_plan(goal, params, llm_planner)
            self._stats["llm_plans_generated"] += 1
            return raw, "llm"

        # ── Fallback: minimal single-step plan ───────────────────────────
        raw = self._fallback_plan(goal, intent, params)
        return raw, "fallback"

    # ── Phase 2: Plan Compilation (FULLY DETERMINISTIC) ───────────────────────

    def compile_plan(
        self,
        raw_plan: RawPlan,
    ) -> Tuple[Optional[CompiledExecutionGraph], ValidationResult]:
        """
        Phase 2: Validate and compile a RawPlan into a CompiledExecutionGraph.

        This phase is FULLY DETERMINISTIC. No LLM involved.

        Returns: (CompiledExecutionGraph | None, ValidationResult)
        If validation fails, returns (None, result_with_errors).
        """
        self._stats["plans_validated"] += 1

        # ── Run all 6 validation gates ────────────────────────────────────
        validation = self._validator.validate(raw_plan)

        if not validation.passed:
            self._stats["plans_failed"] += 1
            logger.warn(
                f"PlanningEngine: compilation failed — "
                f"{validation.total_errors} errors in "
                f"{len(validation.failed_gates())} gates",
            )
            return None, validation

        # ── Compile to execution graph ────────────────────────────────────
        graph = self._compile_to_graph(raw_plan, validation)
        self._stats["plans_compiled"] += 1

        # ── Cache the compiled graph ──────────────────────────────────────
        cache_key = self._cache_key(raw_plan.goal, "", {})
        self._plan_cache[cache_key] = graph
        self._plan_cache.move_to_end(cache_key)
        while len(self._plan_cache) > self._cache_max:
            self._plan_cache.popitem(last=False)

        logger.info(
            f"PlanningEngine: compiled plan '{raw_plan.plan_id}' → "
            f"{len(graph.nodes)} nodes, order={graph.execution_order}",
        )

        return graph, validation

    # ── Full pipeline: generate + validate + compile (with re-plan) ───────────

    def plan(
        self,
        goal: str,
        intent: str = "",
        params: Dict[str, Any] = None,
        router_confidence: float = 0.0,
        llm_planner: Optional[Callable] = None,
    ) -> Tuple[Optional[CompiledExecutionGraph], ValidationResult, Dict[str, Any]]:
        """
        Full planning pipeline: Phase 1 → Validate → Phase 2.
        With re-plan protocol on validation failure (max 3 attempts).

        Returns: (CompiledExecutionGraph | None, ValidationResult, metadata)
        """
        params = params or {}
        self._stats["plans_generated"] += 1
        metadata = {
            "attempts": 0,
            "generation_path": "",
            "replan_attempts": 0,
        }

        for attempt in range(1, self.MAX_REPLAN_ATTEMPTS + 1):
            metadata["attempts"] = attempt

            # Phase 1: Generate
            raw_plan, gen_path = self.generate_plan(
                goal=goal,
                intent=intent,
                params=params,
                router_confidence=router_confidence,
                llm_planner=llm_planner,
            )
            metadata["generation_path"] = gen_path

            # Phase 2: Validate + Compile
            graph, validation = self.compile_plan(raw_plan)

            if graph is not None:
                return graph, validation, metadata

            # ── Re-plan protocol ──────────────────────────────────────────
            if attempt < self.MAX_REPLAN_ATTEMPTS and llm_planner:
                self._stats["replan_attempts"] += 1
                metadata["replan_attempts"] += 1

                replan_request = RePlanRequest(
                    original_goal=goal,
                    attempt_number=attempt + 1,
                    max_attempts=self.MAX_REPLAN_ATTEMPTS,
                    failed_gates=validation.failure_context(),
                    previous_plan=raw_plan.to_dict(),
                    constraints={
                        "max_steps": self._validator._max_steps,
                        "max_execution_depth": self._validator._max_execution_depth,
                        "token_budget": self._validator._token_budget_limit,
                    },
                )

                logger.info(
                    f"PlanningEngine: re-plan attempt {attempt + 1}/"
                    f"{self.MAX_REPLAN_ATTEMPTS}",
                    failed_gates=[
                        g["failed_gate"] for g in replan_request.failed_gates
                    ],
                )

                # Wrap the re-plan context into a new LLM call
                replan_context = replan_request.to_prompt_context()
                try:
                    raw_plan = llm_planner(goal, params, replan_context)
                    if isinstance(raw_plan, dict):
                        raw_plan = RawPlan.from_dict(raw_plan)
                    raw_plan.source = "llm_replan"
                    continue
                except Exception as e:
                    logger.error(f"PlanningEngine: re-plan LLM failed: {e}")
                    break
            else:
                break

        # All attempts exhausted
        logger.error(
            f"PlanningEngine: all {self.MAX_REPLAN_ATTEMPTS} attempts failed "
            f"for goal: '{goal[:60]}'"
        )
        return None, validation, metadata

    # ── Internal: Convert capability to RawPlan ───────────────────────────────

    def _capability_to_raw_plan(
        self, capability, goal: str, params: Dict[str, Any]
    ) -> RawPlan:
        """Convert a registered Capability to a RawPlan (deterministic)."""
        steps = []
        for i, step in enumerate(capability.steps):
            merged_inputs = dict(params)
            merged_inputs.update(step.base_params)
            steps.append(
                RawPlanStep(
                    id=f"step_{i+1:03d}",
                    capability=step.tool_name,
                    inputs=merged_inputs,
                    depends_on=[],
                    can_parallelize=False,
                    timeout_seconds=30,
                    required=step.failure_policy.name != "SKIP",
                )
            )
        return RawPlan(
            goal=goal,
            steps=steps,
            max_retries=3,
            estimated_tokens=0,
            confidence_score=0.95,
            source="deterministic",
        )

    def _intent_to_raw_plan(
        self, intent: str, goal: str, params: Dict[str, Any]
    ) -> RawPlan:
        """Convert a high-confidence intent to a simple single-step plan."""
        tool_name = intent  # intent typically maps to tool name
        return RawPlan(
            goal=goal,
            steps=[
                RawPlanStep(
                    id="step_001",
                    capability=tool_name,
                    inputs=params,
                    depends_on=[],
                    timeout_seconds=30,
                    required=True,
                )
            ],
            estimated_tokens=0,
            confidence_score=0.95,
            source="deterministic",
        )

    def _fallback_plan(self, goal: str, intent: str, params: Dict[str, Any]) -> RawPlan:
        """Minimal fallback plan when no other generation path works."""
        tool_name = intent or "general"
        return RawPlan(
            goal=goal,
            steps=[
                RawPlanStep(
                    id="step_001",
                    capability=tool_name,
                    inputs=params,
                    depends_on=[],
                    timeout_seconds=30,
                    required=True,
                )
            ],
            estimated_tokens=0,
            confidence_score=0.5,
            source="fallback",
        )

    def _llm_generate_plan(
        self,
        goal: str,
        params: Dict[str, Any],
        llm_planner: Callable,
    ) -> RawPlan:
        """Invoke the LLM planner for novel goals."""
        try:
            result = llm_planner(goal, params)
            if isinstance(result, RawPlan):
                result.source = "llm"
                return result
            if isinstance(result, dict):
                plan = RawPlan.from_dict(result)
                plan.source = "llm"
                return plan
            # If LLM returned a string (raw JSON), try parsing
            if isinstance(result, str):
                data = json.loads(result)
                plan = RawPlan.from_dict(data)
                plan.source = "llm"
                return plan
        except Exception as e:
            logger.error(f"LLM planner failed: {e}")

        # Fallback if LLM fails
        return self._fallback_plan(goal, "", params)

    def _graph_to_raw_plan(self, graph: CompiledExecutionGraph) -> RawPlan:
        """Reconstruct a RawPlan from a cached CompiledExecutionGraph."""
        steps = [
            RawPlanStep(
                id=node.node_id,
                capability=node.capability,
                inputs=node.inputs,
                depends_on=node.dependencies,
                can_parallelize=node.can_parallelize,
                fallback=node.fallback,
                timeout_seconds=node.timeout_seconds,
                required=node.required,
            )
            for node in graph.nodes
        ]
        return RawPlan(
            goal=graph.goal,
            steps=steps,
            estimated_tokens=graph.estimated_tokens,
            confidence_score=1.0,
            source="cached",
        )

    # ── Internal: Compile RawPlan to ExecutionGraph ───────────────────────────

    def _compile_to_graph(
        self,
        plan: RawPlan,
        validation: ValidationResult,
    ) -> CompiledExecutionGraph:
        """
        Compile a validated RawPlan into a CompiledExecutionGraph.
        This is FULLY DETERMINISTIC — no LLM involved.
        """
        # Build execution nodes
        nodes = []
        for step in plan.steps:
            nodes.append(
                ExecutionNode(
                    node_id=step.id,
                    capability=step.capability,
                    inputs=copy.deepcopy(step.inputs),
                    dependencies=list(step.depends_on),
                    fallback=step.fallback,
                    timeout_seconds=step.timeout_seconds,
                    required=step.required,
                    can_parallelize=step.can_parallelize,
                )
            )

        # Compute topological execution order
        execution_order = self._topological_sort(plan)

        # Identify parallelizable groups
        parallel_groups = self._find_parallel_groups(plan)

        return CompiledExecutionGraph(
            plan_id=plan.plan_id,
            goal=plan.goal,
            nodes=nodes,
            execution_order=execution_order,
            parallelizable_groups=parallel_groups,
            estimated_tokens=plan.estimated_tokens,
            validation_result=validation,
        )

    def _topological_sort(self, plan: RawPlan) -> List[str]:
        """Kahn's algorithm for topological sorting."""
        # Build adjacency and in-degree
        in_degree: Dict[str, int] = {s.id: 0 for s in plan.steps}
        adj: Dict[str, List[str]] = {s.id: [] for s in plan.steps}

        for step in plan.steps:
            for dep in step.depends_on:
                if dep in adj:
                    adj[dep].append(step.id)
                    in_degree[step.id] += 1

        # Start with zero in-degree nodes
        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        queue.sort()  # deterministic ordering
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in sorted(adj.get(node, [])):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
            queue.sort()

        return result

    def _find_parallel_groups(self, plan: RawPlan) -> List[List[str]]:
        """Find groups of steps that can run in parallel."""
        groups = []
        dep_map = {s.id: set(s.depends_on) for s in plan.steps}
        can_par = {s.id for s in plan.steps if s.can_parallelize}

        completed: Set[str] = set()
        remaining = set(dep_map.keys())

        while remaining:
            # Find all steps whose deps are all completed
            ready = []
            for sid in remaining:
                if dep_map[sid].issubset(completed):
                    ready.append(sid)

            if not ready:
                break  # shouldn't happen in a valid DAG

            # Split into parallel and sequential
            parallel = [s for s in ready if s in can_par]
            sequential = [s for s in ready if s not in can_par]

            if parallel:
                groups.append(sorted(parallel))
            for s in sequential:
                groups.append([s])

            for s in ready:
                remaining.discard(s)
                completed.add(s)

        return groups

    # ── Cache key ─────────────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(goal: str, intent: str, params: Dict[str, Any]) -> str:
        sig = json.dumps(
            {"goal": goal, "intent": intent, "params": params},
            sort_keys=True,
            default=str,
        )
        return hashlib.md5(sig.encode()).hexdigest()

    # ── Stats / Snapshot ──────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "cache_size": len(self._plan_cache),
            "cache_max": self._cache_max,
            "validator_config": {
                "token_budget_limit": self._validator._token_budget_limit,
                "max_execution_depth": self._validator._max_execution_depth,
                "max_steps": self._validator._max_steps,
            },
        }
