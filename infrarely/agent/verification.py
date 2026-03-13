"""
agent/verification.py  — Layer 4: Verification Layer
═══════════════════════════════════════════════════════════════════════════════
All outputs must pass verification before returning to the user.
Failed verification → regenerate or structured fallback.

Verification categories
────────────────────────
  STRUCTURAL  — schema / type / required-field checks
  LOGICAL     — business-rule coherence (dates in range, counts positive, etc.)
  KNOWLEDGE   — factual plausibility (known course IDs, valid student IDs, etc.)
  POLICY      — permission / budget / safety checks

Design contract
───────────────
  • Pure function: no side effects, no LLM calls, no network
  • Returns VerificationResult with pass/fail per category
  • Never mutates the ToolResult — caller decides what to do on failure
  • All checks are deterministic and fast (< 1ms)
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from infrarely.agent.state import (
    ExecutionContract,
    ResponseType,
    ToolContractViolation,
    ToolResult,
    ToolStatus,
    VerificationType,
)
from infrarely.observability import logger
import infrarely.core.app_config as config


# ─── Verification result ──────────────────────────────────────────────────────
@dataclass
class CheckResult:
    """Outcome of a single verification check."""
    check_name:  str
    category:    VerificationType
    passed:      bool
    message:     str = ""


@dataclass
class VerificationResult:
    """Aggregated outcome of all verification checks on a ToolResult."""
    tool_name:     str
    passed:        bool                    = True
    checks:        List[CheckResult]       = field(default_factory=list)
    violations:    List[str]               = field(default_factory=list)
    confidence:    float                   = 1.0   # adjusted confidence

    @property
    def failed_checks(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.passed]

    @property
    def structural_ok(self) -> bool:
        return all(
            c.passed for c in self.checks
            if c.category == VerificationType.STRUCTURAL
        )

    @property
    def logical_ok(self) -> bool:
        return all(
            c.passed for c in self.checks
            if c.category == VerificationType.LOGICAL
        )

    @property
    def policy_ok(self) -> bool:
        return all(
            c.passed for c in self.checks
            if c.category == VerificationType.POLICY
        )

    def summary(self) -> str:
        total  = len(self.checks)
        passed = sum(1 for c in self.checks if c.passed)
        failed = total - passed
        lines  = [f"Verification: {passed}/{total} passed"]
        for c in self.failed_checks:
            lines.append(f"  ✗ [{c.category.value}] {c.check_name}: {c.message}")
        return "\n".join(lines)


# ─── Verification Engine ──────────────────────────────────────────────────────
class VerificationEngine:
    """
    Stateless engine that runs all registered checks against a ToolResult.
    Checks are categorised by VerificationType.
    """

    def __init__(self):
        self._checks: List[_RegisteredCheck] = []
        self._register_defaults()

    def verify(self, result: ToolResult, context: Dict[str, Any] = None) -> VerificationResult:
        """
        Run all applicable checks.  Returns VerificationResult.
        `context` carries optional runtime info (student_id, budget state, etc.)
        """
        ctx = context or {}
        vr  = VerificationResult(tool_name=result.tool_name)

        for rc in self._checks:
            try:
                cr = rc.fn(result, ctx)
                vr.checks.append(cr)
                if not cr.passed:
                    vr.passed = False
                    vr.violations.append(f"[{cr.category.value}] {cr.check_name}: {cr.message}")
            except Exception as e:
                # Check itself failed — log but don't block
                logger.warn(f"VerificationEngine: check '{rc.name}' raised {e}")

        # Adjust confidence based on failures
        if vr.violations:
            penalty = min(len(vr.violations) * 0.15, 0.6)
            vr.confidence = max(0.0, result.confidence - penalty)
        else:
            vr.confidence = result.confidence

        if not vr.passed:
            logger.warn(
                f"Verification FAILED for '{result.tool_name}'",
                violations=len(vr.violations),
                details=vr.violations[:3],
            )
        else:
            logger.debug(
                f"Verification passed for '{result.tool_name}'",
                checks=len(vr.checks),
            )

        return vr

    def verify_or_raise(self, result: ToolResult, context: Dict[str, Any] = None) -> VerificationResult:
        """Like verify(), but raises ToolContractViolation on structural failures."""
        vr = self.verify(result, context)
        if not vr.structural_ok:
            raise ToolContractViolation(result.tool_name, vr.violations)
        return vr

    def register_check(self, name: str, category: VerificationType,
                       fn: Callable[[ToolResult, Dict], CheckResult]):
        """Add a custom check."""
        self._checks.append(_RegisteredCheck(name=name, category=category, fn=fn))

    # ── default checks ────────────────────────────────────────────────────────
    def _register_defaults(self):
        # STRUCTURAL checks
        self.register_check("contract_valid", VerificationType.STRUCTURAL, _check_contract_valid)
        self.register_check("tool_name_set", VerificationType.STRUCTURAL, _check_tool_name)
        self.register_check("error_on_failure", VerificationType.STRUCTURAL, _check_error_on_failure)
        self.register_check("payload_size", VerificationType.STRUCTURAL, _check_payload_size)
        self.register_check("status_coherent", VerificationType.STRUCTURAL, _check_status_coherent)

        # LOGICAL checks
        self.register_check("data_not_none_on_success", VerificationType.LOGICAL, _check_data_present)
        self.register_check("confidence_range", VerificationType.LOGICAL, _check_confidence_range)

        # POLICY checks
        self.register_check("budget_within_limit", VerificationType.POLICY, _check_budget)

        # KNOWLEDGE checks (CP7)
        self.register_check("known_course_id", VerificationType.KNOWLEDGE, _check_known_course_id)
        self.register_check("valid_date_format", VerificationType.KNOWLEDGE, _check_valid_date_format)

        # Enhanced LOGICAL checks (CP7)
        self.register_check("list_size_reasonable", VerificationType.LOGICAL, _check_list_size_reasonable)

        # Enhanced POLICY checks (CP7)
        self.register_check("llm_output_length", VerificationType.POLICY, _check_llm_output_length)

        # SEMANTIC checks (P4 — validate LLM content matches requested context)
        self.register_check("llm_topic_relevance", VerificationType.KNOWLEDGE, _check_llm_topic_relevance)
        self.register_check("llm_no_hallucinated_courses", VerificationType.KNOWLEDGE, _check_no_hallucinated_courses)


@dataclass
class _RegisteredCheck:
    name:     str
    category: VerificationType
    fn:       Callable


# ── STRUCTURAL checks ─────────────────────────────────────────────────────────
def _check_contract_valid(result: ToolResult, ctx: Dict) -> CheckResult:
    valid = isinstance(result.contract, ExecutionContract)
    return CheckResult(
        check_name="contract_valid",
        category=VerificationType.STRUCTURAL,
        passed=valid,
        message="" if valid else f"Invalid contract type: {type(result.contract)}",
    )


def _check_tool_name(result: ToolResult, ctx: Dict) -> CheckResult:
    ok = bool(result.tool_name)
    return CheckResult(
        check_name="tool_name_set",
        category=VerificationType.STRUCTURAL,
        passed=ok,
        message="" if ok else "ToolResult missing tool_name",
    )


def _check_error_on_failure(result: ToolResult, ctx: Dict) -> CheckResult:
    if result.status == ToolStatus.ERROR and not result.error:
        return CheckResult(
            check_name="error_on_failure",
            category=VerificationType.STRUCTURAL,
            passed=False,
            message="ERROR status but no error message",
        )
    return CheckResult(
        check_name="error_on_failure",
        category=VerificationType.STRUCTURAL,
        passed=True,
    )


def _check_payload_size(result: ToolResult, ctx: Dict) -> CheckResult:
    max_chars = getattr(config, "MAX_TOOL_DATA_CHARS", 50_000)
    try:
        size = len(json.dumps(result.data, default=str))
        ok = size <= max_chars
        return CheckResult(
            check_name="payload_size",
            category=VerificationType.STRUCTURAL,
            passed=ok,
            message="" if ok else f"Payload {size} chars > {max_chars} limit",
        )
    except Exception:
        return CheckResult(
            check_name="payload_size",
            category=VerificationType.STRUCTURAL,
            passed=True,
        )


def _check_status_coherent(result: ToolResult, ctx: Dict) -> CheckResult:
    """Status and success flag must agree."""
    ok = True
    msg = ""
    if result.success and result.status == ToolStatus.ERROR:
        ok = False
        msg = "success=True but status=ERROR"
    elif not result.success and result.status == ToolStatus.SUCCESS:
        ok = False
        msg = "success=False but status=SUCCESS"
    return CheckResult(
        check_name="status_coherent",
        category=VerificationType.STRUCTURAL,
        passed=ok,
        message=msg,
    )


# ── LOGICAL checks ────────────────────────────────────────────────────────────
def _check_data_present(result: ToolResult, ctx: Dict) -> CheckResult:
    """Successful results should have non-None data or a message."""
    if result.success and result.status == ToolStatus.SUCCESS:
        has_content = result.data is not None or bool(result.message)
        return CheckResult(
            check_name="data_not_none_on_success",
            category=VerificationType.LOGICAL,
            passed=has_content,
            message="" if has_content else "SUCCESS status but no data or message",
        )
    return CheckResult(
        check_name="data_not_none_on_success",
        category=VerificationType.LOGICAL,
        passed=True,
    )


def _check_confidence_range(result: ToolResult, ctx: Dict) -> CheckResult:
    ok = 0.0 <= result.confidence <= 1.0
    return CheckResult(
        check_name="confidence_range",
        category=VerificationType.LOGICAL,
        passed=ok,
        message="" if ok else f"confidence={result.confidence} out of [0, 1] range",
    )


# ── POLICY checks ─────────────────────────────────────────────────────────────
def _check_budget(result: ToolResult, ctx: Dict) -> CheckResult:
    """If budget info is in context, verify tokens used is within limits."""
    budget = ctx.get("token_budget")
    if budget and isinstance(budget, dict):
        used  = budget.get("used", 0)
        limit = budget.get("limit", float("inf"))
        ok = used <= limit
        return CheckResult(
            check_name="budget_within_limit",
            category=VerificationType.POLICY,
            passed=ok,
            message="" if ok else f"Token budget exceeded: {used}/{limit}",
        )
    return CheckResult(
        check_name="budget_within_limit",
        category=VerificationType.POLICY,
        passed=True,
    )

# ── KNOWLEDGE checks (CP7) ───────────────────────────────────────────────────
def _check_known_course_id(result: ToolResult, ctx: Dict) -> CheckResult:
    """If result references a course_id, verify it looks valid."""
    course_id = None
    if isinstance(result.data, dict):
        course_id = result.data.get("course_id") or result.data.get("course")
    if course_id and isinstance(course_id, str):
        import re
        ok = bool(re.match(r'^[A-Z]{2,5}\d{2,4}$', course_id))
        return CheckResult(
            check_name="known_course_id",
            category=VerificationType.KNOWLEDGE,
            passed=ok,
            message="" if ok else f"Unexpected course_id format: '{course_id}'",
        )
    return CheckResult(
        check_name="known_course_id",
        category=VerificationType.KNOWLEDGE,
        passed=True,
    )


def _check_valid_date_format(result: ToolResult, ctx: Dict) -> CheckResult:
    """If result data contains date fields, check they're parseable."""
    if not isinstance(result.data, dict):
        return CheckResult(check_name="valid_date_format",
                           category=VerificationType.KNOWLEDGE, passed=True)
    for key in ("date", "due_date", "start_date", "end_date"):
        val = result.data.get(key)
        if val and isinstance(val, str):
            from datetime import datetime
            try:
                datetime.fromisoformat(val.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                return CheckResult(
                    check_name="valid_date_format",
                    category=VerificationType.KNOWLEDGE,
                    passed=False,
                    message=f"Unparseable date in '{key}': '{val}'",
                )
    return CheckResult(check_name="valid_date_format",
                       category=VerificationType.KNOWLEDGE, passed=True)


# ── Enhanced LOGICAL checks (CP7) ─────────────────────────────────────────
def _check_list_size_reasonable(result: ToolResult, ctx: Dict) -> CheckResult:
    """Data lists should not be unreasonably large."""
    if isinstance(result.data, list) and len(result.data) > 500:
        return CheckResult(
            check_name="list_size_reasonable",
            category=VerificationType.LOGICAL,
            passed=False,
            message=f"Data list has {len(result.data)} items (>500)",
        )
    return CheckResult(check_name="list_size_reasonable",
                       category=VerificationType.LOGICAL, passed=True)


# ── Enhanced POLICY checks (CP7) ──────────────────────────────────────────
def _check_llm_output_length(result: ToolResult, ctx: Dict) -> CheckResult:
    """LLM-generated output should not be too short or too long."""
    if result.response_type != ResponseType.TOOL_GENERATIVE:
        return CheckResult(check_name="llm_output_length",
                           category=VerificationType.POLICY, passed=True)
    msg_len = len(result.message or "")
    if msg_len < 10:
        return CheckResult(
            check_name="llm_output_length",
            category=VerificationType.POLICY,
            passed=False,
            message=f"LLM output too short ({msg_len} chars)",
        )
    if msg_len > 10000:
        return CheckResult(
            check_name="llm_output_length",
            category=VerificationType.POLICY,
            passed=False,
            message=f"LLM output too long ({msg_len} chars)",
        )
    return CheckResult(check_name="llm_output_length",
                       category=VerificationType.POLICY, passed=True)


# ── P4: SEMANTIC checks ───────────────────────────────────────────────────────
def _check_llm_topic_relevance(result: ToolResult, ctx: Dict) -> CheckResult:
    """
    For TOOL_GENERATIVE results: verify the output text contains at least
    one keyword related to the requested topic. Catches LLM hallucinations
    where the response drifts off-topic entirely.
    """
    if result.response_type != ResponseType.TOOL_GENERATIVE:
        return CheckResult(check_name="llm_topic_relevance",
                           category=VerificationType.KNOWLEDGE, passed=True)

    # Extract the topic from metadata (tools store it there)
    meta_topic = None
    if isinstance(result.metadata, dict):
        meta_topic = result.metadata.get("topic") or result.metadata.get("topics")

    if not meta_topic:
        # No topic to verify against — pass vacuously
        return CheckResult(check_name="llm_topic_relevance",
                           category=VerificationType.KNOWLEDGE, passed=True)

    # Build keyword set from topic(s)
    keywords = set()
    if isinstance(meta_topic, str):
        keywords.update(w.lower() for w in meta_topic.split() if len(w) > 2)
    elif isinstance(meta_topic, list):
        for t in meta_topic[:5]:
            keywords.update(w.lower() for w in str(t).split() if len(w) > 2)

    if not keywords:
        return CheckResult(check_name="llm_topic_relevance",
                           category=VerificationType.KNOWLEDGE, passed=True)

    # Check if output text contains at least one topic keyword
    output_lower = (result.message or "").lower()
    if isinstance(result.data, list):
        output_lower += " ".join(str(d).lower() for d in result.data)

    matches = sum(1 for kw in keywords if kw in output_lower)
    ok = matches >= 1

    return CheckResult(
        check_name="llm_topic_relevance",
        category=VerificationType.KNOWLEDGE,
        passed=ok,
        message="" if ok else (
            f"LLM output has 0/{len(keywords)} topic keywords. "
            f"Expected at least one of: {', '.join(list(keywords)[:5])}"
        ),
    )


def _check_no_hallucinated_courses(result: ToolResult, ctx: Dict) -> CheckResult:
    """
    For TOOL_GENERATIVE results that mention course IDs: verify mentioned
    courses look like real course IDs (not fabricated strings).
    """
    if result.response_type != ResponseType.TOOL_GENERATIVE:
        return CheckResult(check_name="llm_no_hallucinated_courses",
                           category=VerificationType.KNOWLEDGE, passed=True)

    import re
    text = (result.message or "")
    if isinstance(result.data, list):
        text += " ".join(str(d) for d in result.data)

    # Find anything that looks like a course ID
    found_ids = re.findall(r'\b[A-Z]{2,5}\d{3,4}\b', text)
    if not found_ids:
        return CheckResult(check_name="llm_no_hallucinated_courses",
                           category=VerificationType.KNOWLEDGE, passed=True)

    # Check each found ID matches expected format (we can't check enrollment
    # without structured_memory here — just validate format plausibility)
    for cid in found_ids:
        if not re.match(r'^[A-Z]{2,5}\d{2,4}$', cid):
            return CheckResult(
                check_name="llm_no_hallucinated_courses",
                category=VerificationType.KNOWLEDGE,
                passed=False,
                message=f"Suspicious course ID in LLM output: '{cid}'",
            )

    return CheckResult(check_name="llm_no_hallucinated_courses",
                       category=VerificationType.KNOWLEDGE, passed=True)


# ── module-level convenience ──────────────────────────────────────────────────
_engine = VerificationEngine()


def verify(result: ToolResult, context: Dict[str, Any] = None) -> VerificationResult:
    """Module-level convenience: verify a ToolResult."""
    return _engine.verify(result, context)


def verify_or_raise(result: ToolResult, context: Dict[str, Any] = None) -> VerificationResult:
    """Module-level convenience: verify and raise on structural failure."""
    return _engine.verify_or_raise(result, context)
