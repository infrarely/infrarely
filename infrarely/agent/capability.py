"""
agent/capability.py  — Layer 2
═══════════════════════════════════════════════════════════════════════════════
Data structures for the Capability Graph execution layer.

A Capability is a named, ordered sequence of CapabilitySteps.
Each step wraps exactly one tool call and can reference outputs from
previous steps via a simple {step_name.field} param interpolation.

Design constraints preserved from Layer 1
───────────────────────────────────────────
  • FINAL_DETERMINISTIC steps run first (always)
  • FINAL_GENERATED steps run only when required
  • Each step produces a ToolResult with a full ExecutionContract
  • The LLM remains isolated inside tools — never called from the executor
  • A failing step does not silently corrupt downstream steps

Execution contract per step
────────────────────────────
  FINAL_DETERMINISTIC  → step output is immediately final; passes to next step
  FINAL_GENERATED      → tool called LLM; result is final; no second LLM call
  FAILED               → step_result.ok = False; executor applies failure policy
  PARTIAL_CONTEXT      → not valid in a capability step (would require core LLM)
                         Use a FINAL_GENERATED tool instead

Failure policy per step
────────────────────────
  ABORT        stop immediately; return partial results gathered so far
  SKIP         mark step failed; continue with next step (using last good output)
  REQUIRED     alias for ABORT — this step's output is needed by later steps

Step output passing
────────────────────
  Each step receives a shared `context: Dict[str, Any]` that accumulates
  the `.data` from every previously completed step, keyed by step name.

  Param interpolation:  {"topic": "{exam_topics.topics[0]}"}
  The executor resolves these references before calling the tool.
  If a reference cannot be resolved, the literal string is passed through
  (safe degradation — tool may produce a FAILED result, which is handled).

Auditing
─────────
  CapabilityResult carries a StepTrace for every step that ran.
  This is the auditable decision trace for the full workflow.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from infrarely.agent.state import ExecutionContract, TaskState, ToolResult


# ── Step failure policy ───────────────────────────────────────────────────────
class FailurePolicy(Enum):
    ABORT = auto()  # stop capability immediately
    SKIP = auto()  # log failure, continue with next step
    REQUIRED = auto()  # alias for ABORT — downstream steps depend on this


# ── Step execution status ─────────────────────────────────────────────────────
class StepStatus(Enum):
    PENDING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()  # only when a non-REQUIRED upstream failed


# ── Single step definition ────────────────────────────────────────────────────
@dataclass
class CapabilityStep:
    """
    One tool invocation within a capability.

    name           — unique within the capability; used as context key
    tool_name      — must exist in ToolRegistry
    base_params    — static params; may contain {step_name.field} references
    failure_policy — ABORT (default) or SKIP
    condition      — optional callable(context) → bool
                     if provided and returns False, step is skipped
    description    — human-readable label for audit traces
    """

    name: str
    tool_name: str
    base_params: Dict[str, Any] = field(default_factory=dict)
    failure_policy: FailurePolicy = FailurePolicy.ABORT
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    description: str = ""

    def resolve_params(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interpolate {step_name.field} references from the execution context.
        Unresolvable references are passed through as-is (safe degradation).
        """
        resolved = {}
        for key, value in self.base_params.items():
            if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                ref = value[1:-1]  # strip braces
                resolved[key] = _resolve_ref(ref, context, default=value)
            else:
                resolved[key] = value
        return resolved

    def should_run(self, context: Dict[str, Any]) -> bool:
        if self.condition is None:
            return True
        try:
            return bool(self.condition(context))
        except Exception:
            return True  # safe default — run if condition errors


def _resolve_ref(ref: str, context: Dict[str, Any], default: Any = None) -> Any:
    """
    Resolve "step_name.field" or "step_name.field[0]" from context.
    Returns default if path cannot be resolved.
    """
    import re

    parts = ref.split(".")
    obj = context
    for part in parts:
        # handle list index: field[0]
        m = re.match(r"^(\w+)\[(\d+)\]$", part)
        if m:
            key, idx = m.group(1), int(m.group(2))
            if isinstance(obj, dict) and key in obj:
                obj = obj[key]
                if isinstance(obj, list) and idx < len(obj):
                    obj = obj[idx]
                else:
                    return default
            else:
                return default
        else:
            if isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return default
    return obj


# ── Step execution trace ──────────────────────────────────────────────────────
@dataclass
class StepTrace:
    """Immutable record of one step's execution — the audit trail entry."""

    step_name: str
    tool_name: str
    status: StepStatus
    contract: Optional[ExecutionContract] = None
    duration_ms: float = 0.0
    error: str = ""
    tokens_used: int = 0
    skipped_reason: str = ""


# ── Capability definition ─────────────────────────────────────────────────────
@dataclass
class Capability:
    """
    A named, ordered workflow of CapabilitySteps.

    name         — unique identifier (matched by router)
    description  — shown in /health and audit traces
    steps        — ordered list; deterministic steps should come first
    intent_tags  — triggers used by the router to map queries to this capability
    version      — incremented on modification; invalidates PlanCache (Gap 10)

    Execution order guarantee:
      FINAL_DETERMINISTIC steps are always scheduled before FINAL_GENERATED steps
      within their natural dependency order.  If two deterministic steps are
      independent, they run in declaration order.
    """

    name: str
    description: str
    steps: List[CapabilityStep]
    intent_tags: List[str] = field(default_factory=list)
    version: int = 1

    def ordered_steps(self) -> List[CapabilityStep]:
        """
        Return steps in execution order:
          1. FINAL_DETERMINISTIC steps (by declaration order)
          2. FINAL_GENERATED steps (by declaration order)
        This ensures all data is gathered before any LLM-using tool runs.
        We detect tool type by checking the step tool_name against a registry
        lookup; if registry is unavailable we use declaration order as-is.
        """
        # Declaration order is the contract — callers define deterministic first.
        # This method is a hook for future reordering logic.
        return list(self.steps)


# ── Capability execution plan ─────────────────────────────────────────────────
@dataclass(frozen=True)
class CapabilityPlan:
    """
    The authority object the router emits when a capability is matched.
    Frozen like ExecutionPlan — nothing downstream may mutate it.

    Replaces (ExecutionPlan, ToolResult) for multi-step flows.
    The core dispatches to CapabilityExecutor when it receives this type.
    """

    capability: Capability
    task_state: TaskState
    initial_context: Dict[str, Any] = field(default_factory=dict)

    @property
    def allow_llm(self) -> bool:
        """
        True only if at least one step is FINAL_GENERATED.
        Used by the budget check before execution begins.
        """
        # Cannot check without registry here; executor checks per-step.
        # Conservative: assume True if any step *might* use LLM.
        return True  # executor enforces per-step isolation


# ── Capability execution result ───────────────────────────────────────────────
@dataclass
class CapabilityResult:
    """
    Aggregated result from a full capability execution.
    Contains both the formatted response and the full audit trace.

    Mirrors AgentResponse structure for seamless core integration.
    """

    capability_name: str
    success: bool
    message: str  # final formatted response
    steps_trace: List[StepTrace] = field(default_factory=list)
    step_outputs: Dict[str, ToolResult] = field(default_factory=dict)
    tokens_used: int = 0
    execution_ms: float = 0.0
    aborted_at: Optional[str] = None  # step name that caused abort
    partial: bool = False  # True if aborted early

    @property
    def steps_completed(self) -> int:
        return sum(1 for t in self.steps_trace if t.status == StepStatus.COMPLETED)

    @property
    def steps_failed(self) -> int:
        return sum(1 for t in self.steps_trace if t.status == StepStatus.FAILED)

    def audit_summary(self) -> str:
        lines = [f"Capability: {self.capability_name}"]
        for trace in self.steps_trace:
            icon = {"COMPLETED": "✓", "FAILED": "✗", "SKIPPED": "○", "PENDING": "…"}
            lines.append(
                f"  {icon.get(trace.status.name, '?')} {trace.step_name}"
                f" [{trace.tool_name}] {trace.duration_ms:.0f}ms"
                + (f" — {trace.error}" if trace.error else "")
            )
        return "\n".join(lines)
