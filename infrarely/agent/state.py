"""
agent/state.py  (v2 — refactored)
═══════════════════════════════════════════════════════════════════════════════
Typed state objects that flow through the entire pipeline.

Key additions over v1:
  • ResponseType enum  — classifies how a response should be produced
  • ToolResult.is_complete — tool asserts "I produced the final answer"
  • ToolResult.response_type — carried from tool through to core
  • ExecutionPlan  — the SINGLE authority object the router hands to the core.
                     Once set, nothing in the pipeline may override it.

Design contract
───────────────
  DETERMINISTIC   Tool data IS the complete answer. LLM is NEVER called.
  TOOL_GENERATIVE Tool called LLM internally. Result is final. Core skips LLM.
  GENERATIVE      Core will call LLM once, using tool result as context.
  ERROR           Something failed. Format error message, no LLM.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone


# ─── Response classification ──────────────────────────────────────────────────
class ResponseType(Enum):
    DETERMINISTIC   = auto()   # tool result → direct format → done
    TOOL_GENERATIVE = auto()   # tool used LLM internally → result is final
    GENERATIVE      = auto()   # core calls LLM once with tool context
    ERROR           = auto()   # structured error, no LLM


# ─── Tool result status (Layer 4) ─────────────────────────────────────────────
class ToolStatus(Enum):
    """Fine-grained outcome status carried on every ToolResult."""
    SUCCESS              = "success"                # tool produced complete data
    PARTIAL              = "partial"                 # some data returned, not everything
    ERROR                = "error"                   # tool failed entirely
    GENERATION_REQUIRED  = "generation_required"     # tool needs LLM to finish


class VerificationType(Enum):
    """Categories of verification applied by the Verification Layer."""
    STRUCTURAL  = "structural"    # schema / type / required-field checks
    LOGICAL     = "logical"       # business-rule coherence
    KNOWLEDGE   = "knowledge"     # factual plausibility
    POLICY      = "policy"        # permission / budget / safety


# ─── Contract violation exception ──────────────────────────────────────────────
class ToolContractViolation(Exception):
    """Raised when a ToolResult violates its declared contract."""
    def __init__(self, tool_name: str, violations: List[str]):
        self.tool_name  = tool_name
        self.violations = violations
        super().__init__(f"[{tool_name}] contract violations: {'; '.join(violations)}")


class ExecutionContract(Enum):
    """Backward-compatible execution contract enum used by older modules."""
    FINAL_DETERMINISTIC = "final_deterministic"
    FINAL_GENERATED     = "final_generated"
    PARTIAL_CONTEXT     = "partial_context"
    FAILED              = "failed"


_CONTRACT_TO_RESPONSE_TYPE = {
    ExecutionContract.FINAL_DETERMINISTIC: ResponseType.DETERMINISTIC,
    ExecutionContract.FINAL_GENERATED: ResponseType.TOOL_GENERATIVE,
    ExecutionContract.PARTIAL_CONTEXT: ResponseType.GENERATIVE,
    ExecutionContract.FAILED: ResponseType.ERROR,
}

_RESPONSE_TYPE_TO_CONTRACT = {
    ResponseType.DETERMINISTIC: ExecutionContract.FINAL_DETERMINISTIC,
    ResponseType.TOOL_GENERATIVE: ExecutionContract.FINAL_GENERATED,
    ResponseType.GENERATIVE: ExecutionContract.PARTIAL_CONTEXT,
    ResponseType.ERROR: ExecutionContract.FAILED,
}


def contract_to_response_type(contract: ExecutionContract) -> ResponseType:
    return _CONTRACT_TO_RESPONSE_TYPE.get(contract, ResponseType.DETERMINISTIC)


def response_type_to_contract(response_type: ResponseType) -> ExecutionContract:
    return _RESPONSE_TYPE_TO_CONTRACT.get(
        response_type,
        ExecutionContract.FINAL_DETERMINISTIC,
    )


# ─── Conversation message ─────────────────────────────────────────────────────
@dataclass
class Message:
    role:      str             # "user" | "assistant" | "system"
    content:   str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_api_dict(self) -> Dict[str, str]:
        """Returns the dict expected by LLM API message lists."""
        return {"role": self.role, "content": self.content}


# ─── Intent / routing state ───────────────────────────────────────────────────
@dataclass
class TaskState:
    """
    Extracted intent + parameters.  Set by the router, read-only downstream.
    """
    task:          str                    # e.g. "assignments_upcoming"
    tool:          str                    # e.g. "assignment_tracker"
    params:        Dict[str, Any]         = field(default_factory=dict)
    requires_llm:  bool                   = False
    confidence:    float                  = 1.0
    raw_input:     str                    = ""
    student_id:    str                    = "student_1"
    timestamp:     str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ─── Tool return contract ─────────────────────────────────────────────────────
@dataclass
class ToolResult:
    """
    Standardised return value from every tool.

    is_complete:   When True the tool has produced the FINAL human-readable
                   response in `data` or `message`.  The core MUST NOT call
                   the LLM afterward, even if allow_llm=True in the plan.

    response_type: Mirrors the ExecutionPlan value; carried here so the
                   formatter can make rendering decisions without the plan.

    status:        Fine-grained outcome (SUCCESS / PARTIAL / ERROR / GENERATION_REQUIRED).
    confidence:    0.0–1.0 self-assessed quality score for verification layer.
    """
    success:       bool                   = True
    tool_name:     str                    = ""
    data:          Any                    = None
    message:       str                    = ""
    error:         str                    = ""
    duration_ms:   float                  = 0.0
    is_complete:   bool                   = False
    response_type: ResponseType           = ResponseType.DETERMINISTIC
    contract:      Optional[ExecutionContract] = None
    status:        ToolStatus             = ToolStatus.SUCCESS
    confidence:    float                  = 1.0
    metadata:      Dict[str, Any]         = field(default_factory=dict)

    def __post_init__(self):
        # Legacy path: contract provided, derive response_type/success/is_complete.
        if self.contract is not None:
            self.response_type = contract_to_response_type(self.contract)

            if self.contract == ExecutionContract.FAILED:
                self.success = False
                self.is_complete = True
                self.status = ToolStatus.ERROR
            elif self.contract in (
                ExecutionContract.FINAL_DETERMINISTIC,
                ExecutionContract.FINAL_GENERATED,
            ):
                self.success = True
                self.is_complete = True
                self.status = ToolStatus.SUCCESS
            else:  # PARTIAL_CONTEXT
                self.success = True
                self.is_complete = False
                self.status = ToolStatus.GENERATION_REQUIRED
        else:
            # New path: derive legacy contract from response_type.
            self.contract = response_type_to_contract(self.response_type)

        # Synchronise status from success flag if caller didn't set status.
        if not self.success and self.status == ToolStatus.SUCCESS:
            self.status = ToolStatus.ERROR

        # Keep core invariants coherent.
        if not self.success and not self.error:
            self.error = "Tool execution failed"
        if self.response_type == ResponseType.ERROR:
            self.success = False
            self.status = ToolStatus.ERROR

    @property
    def allow_llm(self) -> bool:
        """Backward-compatible shim used by older tests/callers."""
        return self.contract == ExecutionContract.PARTIAL_CONTEXT

    @property
    def is_final(self) -> bool:
        """Backward-compatible shim used by older tests/callers."""
        return self.contract in (
            ExecutionContract.FINAL_DETERMINISTIC,
            ExecutionContract.FINAL_GENERATED,
            ExecutionContract.FAILED,
        )

    # ── helpers ───────────────────────────────────────────────────────────────
    def is_empty(self) -> bool:
        """True when the tool ran successfully but found nothing."""
        if not self.success:
            return False
        if isinstance(self.data, (list, dict)) and not self.data:
            return True
        return self.data is None and not self.message

    def to_context_snippet(self, max_items: int = 10, max_chars: int = 600) -> str:
        """
        Compact, token-efficient representation safe to embed in an LLM prompt.
        Hard-caps total characters to protect the token budget.
        """
        if not self.success:
            return f"[{self.tool_name}] ERROR: {self.error}"

        if isinstance(self.data, list):
            lines = [f"  - {item}" for item in self.data[:max_items]]
            body  = "\n".join(lines)
        elif isinstance(self.data, dict):
            lines = [f"  {k}: {v}" for k, v in list(self.data.items())[:15]]
            body  = "\n".join(lines)
        elif self.data is not None:
            body = str(self.data)
        else:
            body = self.message

        snippet = f"[{self.tool_name}]\n{body}"
        return snippet[:max_chars]


# ─── Execution plan — THE single authority object ─────────────────────────────
@dataclass
class ExecutionPlan:
    """
    Produced by the router; consumed by the core.

    allow_llm is FINAL.  Nothing in the pipeline may override it.
    The core enforces this as a hard gate:
        if not plan.allow_llm → zero LLM calls, guaranteed.

    response_type encodes the intended rendering strategy:
        DETERMINISTIC   → format tool result directly
        TOOL_GENERATIVE → tool already called LLM; format result directly
        GENERATIVE      → one LLM call, tool result as context
        ERROR           → format error, no LLM
    """
    task_state:    TaskState
    response_type: Optional[ResponseType] = None
    allow_llm:     Optional[bool] = None
    tool_name:     str = ""
    contract:      Optional[ExecutionContract] = None

    def __post_init__(self):
        if self.contract is not None and self.response_type is None:
            self.response_type = contract_to_response_type(self.contract)
        if self.response_type is None:
            self.response_type = ResponseType.DETERMINISTIC

        if self.contract is None:
            self.contract = response_type_to_contract(self.response_type)

        if self.allow_llm is None:
            self.allow_llm = self.contract == ExecutionContract.PARTIAL_CONTEXT


# ─── Final agent response ─────────────────────────────────────────────────────
@dataclass
class AgentResponse:
    """Returned to the caller (CLI, API, test runner, etc.)."""
    message:        str
    response_type:  ResponseType           = ResponseType.DETERMINISTIC
    task_state:     Optional[TaskState]    = None
    tool_result:    Optional[ToolResult]   = None
    plan:           Optional[ExecutionPlan] = None
    llm_used:       bool                   = False
    tokens_used:    int                    = 0
    from_cache:     bool                   = False
    execution_ms:   float                  = 0.0

    @property
    def contract(self) -> ExecutionContract:
        """Backward-compatible view for callers still reading response.contract."""
        return response_type_to_contract(self.response_type)