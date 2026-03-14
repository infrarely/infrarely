"""
infrarely/result.py — Structured Result & Error objects
═══════════════════════════════════════════════════════════════════════════════
Every SDK operation returns a Result. Never a bare exception.

Philosophy 3: Errors are data, not exceptions.
  Every error has: type, message, step, recovered, suggestion.
  Every result has: output, confidence, used_llm, sources, duration_ms, trace_id.

Result.explain() produces a human-readable execution summary
that a non-technical person can read and understand.
"""

from __future__ import annotations

import textwrap
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR TYPE TAXONOMY
# ═══════════════════════════════════════════════════════════════════════════════


class ErrorType(Enum):
    """Every failure maps to exactly one of these types."""

    TOOL_FAILURE = "TOOL_FAILURE"
    PLAN_INVALID = "PLAN_INVALID"
    KNOWLEDGE_GAP = "KNOWLEDGE_GAP"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    TIMEOUT = "TIMEOUT"
    STATE_CORRUPTED = "STATE_CORRUPTED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    DELEGATION_FAILED = "DELEGATION_FAILED"
    SECURITY_VIOLATION = "SECURITY_VIOLATION"
    VALIDATION = "VALIDATION"
    APPROVAL_TIMEOUT = "APPROVAL_TIMEOUT"
    APPROVAL_REJECTED = "APPROVAL_REJECTED"
    UNKNOWN = "UNKNOWN"


# ── Human-readable suggestions per error type ────────────────────────────────

_SUGGESTIONS: Dict[str, str] = {
    "TOOL_FAILURE": (
        "Check that the tool is properly decorated with @infrarely.tool and "
        "returns the expected type. Verify network/API connectivity if the "
        "tool calls external services."
    ),
    "PLAN_INVALID": (
        "The planning engine could not build a valid execution plan. "
        "Ensure the agent has tools or capabilities that match the goal. "
        "Try simplifying the goal or adding more specific tools."
    ),
    "KNOWLEDGE_GAP": (
        "The agent's knowledge sources did not have enough information and "
        "LLM confidence was also low. Add more knowledge via "
        "infrarely.knowledge.add_data() or infrarely.knowledge.add_documents()."
    ),
    "VERIFICATION_FAILED": (
        "The output failed quality verification. This usually means the "
        "generated response didn't pass factual or structural checks. "
        "Improve knowledge sources or adjust knowledge_threshold."
    ),
    "BUDGET_EXCEEDED": (
        "Token or execution depth limit was reached. Increase token_budget "
        "in config or simplify the task."
    ),
    "TIMEOUT": (
        "The task exceeded its time limit. Increase default_timeout in "
        "config or break the task into smaller subtasks."
    ),
    "STATE_CORRUPTED": (
        "The agent's state machine entered an invalid state but attempted "
        "auto-recovery. If this persists, call agent.reset()."
    ),
    "PERMISSION_DENIED": (
        "The agent tried to access a resource outside its permitted scope. "
        "Check agent permissions and memory scope settings."
    ),
    "CONFIGURATION_ERROR": (
        "SDK is not properly configured. Call infrarely.configure() with at "
        "minimum an llm_provider and api_key."
    ),
    "DELEGATION_FAILED": (
        "The delegated agent failed to complete its subtask. Check the "
        "delegate agent's health with agent.health()."
    ),
    "SECURITY_VIOLATION": (
        "Input was blocked by the security policy. The prompt injection "
        "scanner detected a potential attack. Review the input or adjust "
        "the SecurityPolicy settings via infrarely.configure(security=...)."
    ),
    "VALIDATION": (
        "Tool input validation failed. One or more arguments don't match "
        "the expected types. Check the tool's type annotations and ensure "
        "inputs are correct. See the error details for the exact field."
    ),
    "APPROVAL_TIMEOUT": (
        "A human approval request timed out. Increase the timeout in "
        "agent.require_approval_for() or ensure approvers respond faster. "
        "Check pending requests via infrarely.approvals.get_pending()."
    ),
    "APPROVAL_REJECTED": (
        "A human reviewer rejected this operation. Review the approval "
        "request details and the rejection reason. Adjust the operation "
        "or approval rules as needed."
    ),
    "UNKNOWN": (
        "An unexpected error occurred. Check the full trace with "
        "agent.get_trace(result.trace_id) for details."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR OBJECT
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Error:
    """Structured error — never a bare exception string."""

    type: ErrorType = ErrorType.UNKNOWN
    message: str = ""
    step: str = ""  # which step/tool failed
    recovered: bool = False  # True if agent auto-recovered
    recovery_action: str = ""  # what recovery was attempted
    suggestion: str = ""  # what the developer should do
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.suggestion:
            self.suggestion = _SUGGESTIONS.get(self.type.value, _SUGGESTIONS["UNKNOWN"])

    def __str__(self) -> str:
        parts = [f"[{self.type.value}] {self.message}"]
        if self.step:
            parts.append(f"  Step: {self.step}")
        if self.recovered:
            parts.append(f"  Auto-recovered: {self.recovery_action}")
        parts.append(f"  Suggestion: {self.suggestion}")
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "message": self.message,
            "step": self.step,
            "recovered": self.recovered,
            "recovery_action": self.recovery_action,
            "suggestion": self.suggestion,
            "details": self.details,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT OBJECT
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Result:
    """
    Structured result returned by every agent.run() call.

    Beginners use:  result.output, print(result)
    Developers use: result.confidence, result.used_llm, result.sources
    Experts use:    result.explain(), result.trace_id, result.error
    """

    # ── Core fields ───────────────────────────────────────────────────────────
    output: Any = None  # the actual answer
    success: bool = True
    confidence: float = 1.0  # 0.0–1.0
    used_llm: bool = False  # was an LLM call made?
    sources: List[str] = field(default_factory=list)  # knowledge/tool sources
    duration_ms: float = 0.0
    trace_id: str = field(default_factory=lambda: f"trace_{uuid.uuid4().hex[:12]}")

    # ── Error (only populated on failure) ─────────────────────────────────────
    error: Optional[Error] = None

    # ── Internal metadata ─────────────────────────────────────────────────────
    _steps_executed: int = 0
    _steps_skipped: int = 0
    _knowledge_sources: List[str] = field(default_factory=list)
    _llm_calls: int = 0
    _state_transitions: List[str] = field(default_factory=list)
    _goal: str = ""
    _agent_name: str = ""
    _plan_source: str = ""  # "deterministic" | "llm" | "cached"

    # ── Token tracking ────────────────────────────────────────────────────────
    tokens_used: int = 0
    estimated_cost: float = 0.0

    # ── Beginner-friendly string representation ───────────────────────────────

    def __str__(self) -> str:
        if self.success:
            return str(self.output) if self.output is not None else ""
        return f"Error: {self.error}" if self.error else "Unknown error"

    def __repr__(self) -> str:
        status = "OK" if self.success else "FAILED"
        return (
            f"Result(status={status}, confidence={self.confidence:.2f}, "
            f"used_llm={self.used_llm}, duration_ms={self.duration_ms:.0f})"
        )

    def __bool__(self) -> bool:
        """Allow `if result:` for quick success checks."""
        return self.success

    # ── Explain — human-readable execution summary ────────────────────────────

    def explain(self) -> str:
        """
        Returns a human-readable, non-technical explanation of how
        this result was produced. Designed for GATE 3 compliance —
        readable by non-technical people.
        """
        lines = []
        border = "─" * 55

        lines.append(f"┌{border}┐")
        lines.append(f"│ {'Execution Summary':^53} │")
        lines.append(f"├{border}┤")

        # Goal
        goal_display = self._goal[:48] if self._goal else "(not set)"
        lines.append(f'│ Goal: "{goal_display}"')

        # Status
        if self.success:
            lines.append(f"│ Status: Completed successfully")
        elif self.error and self.error.recovered:
            lines.append(f"│ Status: Recovered from error")
        else:
            err_type = self.error.type.value if self.error else "UNKNOWN"
            lines.append(f"│ Status: Failed ({err_type})")

        # Duration
        if self.duration_ms < 1000:
            lines.append(f"│ Duration: {self.duration_ms:.0f}ms")
        else:
            lines.append(f"│ Duration: {self.duration_ms / 1000:.1f}s")

        # LLM usage
        if self.used_llm:
            lines.append(
                f"│ LLM called: Yes ({self._llm_calls} call"
                f"{'s' if self._llm_calls != 1 else ''})"
            )
        else:
            conf_str = f"knowledge confidence: {self.confidence:.2f}"
            lines.append(f"│ LLM called: No ({conf_str})")

        # Steps
        if self._steps_executed > 0 or self._steps_skipped > 0:
            lines.append(f"│ Steps executed: {self._steps_executed}")
            if self._steps_skipped:
                lines.append(
                    f"│ Steps skipped: {self._steps_skipped} (condition not met)"
                )

        # Knowledge sources
        if self._knowledge_sources:
            src_str = ", ".join(f'"{s}"' for s in self._knowledge_sources[:5])
            lines.append(f"│ Knowledge sources: [{src_str}]")
        elif self.sources:
            src_str = ", ".join(f'"{s}"' for s in self.sources[:5])
            lines.append(f"│ Sources used: [{src_str}]")

        # Plan source
        if self._plan_source:
            lines.append(f"│ Plan type: {self._plan_source}")

        # State transitions
        if self._state_transitions:
            flow = " → ".join(self._state_transitions)
            lines.append(f"│ State flow: {flow}")

        # Error details
        if self.error and not self.success:
            lines.append(f"├{border}┤")
            lines.append(f"│ Error: {self.error.message[:50]}")
            if self.error.suggestion:
                wrapped = textwrap.wrap(self.error.suggestion, width=50)
                lines.append(f"│ Fix: {wrapped[0]}")
                for w in wrapped[1:]:
                    lines.append(f"│      {w}")

        lines.append(f"│ Trace ID: {self.trace_id}")
        lines.append(f"└{border}┘")

        return "\n".join(lines)

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "success": self.success,
            "output": self.output,
            "confidence": self.confidence,
            "used_llm": self.used_llm,
            "sources": self.sources,
            "duration_ms": self.duration_ms,
            "trace_id": self.trace_id,
            "tokens_used": self.tokens_used,
            "estimated_cost": self.estimated_cost,
        }
        if self.error:
            d["error"] = self.error.to_dict()
        return d


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT FACTORY — Internal helpers for building results
# ═══════════════════════════════════════════════════════════════════════════════


def _ok(
    output: Any,
    confidence: float = 1.0,
    used_llm: bool = False,
    sources: Optional[List[str]] = None,
    duration_ms: float = 0.0,
    goal: str = "",
    agent_name: str = "",
    **kwargs,
) -> Result:
    """Build a success Result."""
    return Result(
        output=output,
        success=True,
        confidence=confidence,
        used_llm=used_llm,
        sources=sources or [],
        duration_ms=duration_ms,
        _goal=goal,
        _agent_name=agent_name,
        **kwargs,
    )


def _fail(
    error_type: ErrorType,
    message: str,
    step: str = "",
    recovered: bool = False,
    recovery_action: str = "",
    goal: str = "",
    agent_name: str = "",
    duration_ms: float = 0.0,
    **kwargs,
) -> Result:
    """Build a failure Result."""
    return Result(
        output=None,
        success=False,
        confidence=0.0,
        error=Error(
            type=error_type,
            message=message,
            step=step,
            recovered=recovered,
            recovery_action=recovery_action,
        ),
        duration_ms=duration_ms,
        _goal=goal,
        _agent_name=agent_name,
        **kwargs,
    )
