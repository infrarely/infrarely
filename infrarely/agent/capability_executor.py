"""
agent/capability_executor.py  — Layer 2 (upgraded for CapabilityGraph)
═══════════════════════════════════════════════════════════════════════════════
Executes a CapabilityPlan step-by-step using graph-based ordering.

Execution model
───────────────
  1. Receive CapabilityPlan (frozen — immutable authority)
  2. Build CapabilityGraph via CapabilityGraphBuilder (if not cached)
  3. Execute nodes in topological order (dependency-resolved)
  4. For each node:
       a. Evaluate node.condition(context) — skip if False
       b. Check all dependencies completed
       c. Resolve params via context interpolation
       d. Execute tool via ToolSandbox (timeout enforced)
       e. Verify result via VerificationEngine (Layer 4)
       f. Contract enforcement: FAILED → apply failure_policy + retry
       g. COMPLETED → write node output to context[node.name]
       h. Append StepTrace to audit log
  5. Format aggregated response from all completed node outputs
  6. Return CapabilityResult

LLM isolation guarantee (preserved from Layer 1)
──────────────────────────────────────────────────
  The executor never calls llm_call directly.
  FINAL_GENERATED tools call LLM internally (exactly once, inside the tool).
  The executor sees only the ToolResult — not the LLM call.

Integration with new infrastructure
─────────────────────────────────────
  • CapabilityGraph     — dependency-ordered execution
  • ToolSandbox         — timeout enforcement per node
  • VerificationEngine  — structural/logical checks on each step output
  • ErrorRecoveryEngine — recovery from FAILED steps
"""

from __future__ import annotations
import time
from typing import Any, Dict, Optional, Tuple

from infrarely.agent.capability import (
    CapabilityPlan,
    CapabilityResult,
    CapabilityStep,
    FailurePolicy,
    StepStatus,
    StepTrace,
)
from infrarely.agent.capability_graph import CapabilityGraphBuilder, CapabilityGraph, GraphNode
from infrarely.agent.state import ExecutionContract, TaskState, ToolResult, ToolStatus
from infrarely.agent.tool_sandbox import ToolSandbox
from infrarely.agent.verification import VerificationEngine
from infrarely.agent.error_recovery import ErrorRecoveryEngine
from infrarely.tools.registry import ToolRegistry
from infrarely.observability import logger
import infrarely.core.app_config as config


class CapabilityExecutor:
    """
    Graph-aware executor.  One instance per AgentCore, reused across requests.
    Each execute() call is fully isolated — no shared mutable state.

    New in v2:
      • Builds CapabilityGraph for dependency-ordered execution
      • Uses ToolSandbox for timeout enforcement per node
      • Runs VerificationEngine on each step output
      • Supports retry_policy per node
    """

    def __init__(self, registry: ToolRegistry, app_cfg, safety_controller=None):
        self._registry = registry
        self._cfg = app_cfg
        self._graph_builder = CapabilityGraphBuilder(registry)
        self._sandbox = ToolSandbox()
        self._verifier = VerificationEngine()
        self._recovery = ErrorRecoveryEngine()
        self._safety = safety_controller  # Layer 5 safety controller (Gap 5)

    def execute(
        self, plan: CapabilityPlan, trace=None, budget=None
    ) -> CapabilityResult:
        t_start = time.monotonic()
        context = dict(plan.initial_context)  # mutable copy; plan stays frozen
        traces: list[StepTrace] = []
        outputs: Dict[str, ToolResult] = {}
        tokens_total = 0
        aborted_at: Optional[str] = None

        # ── Safety: enter execution (blocks graph mutation) ───────────────────
        if self._safety:
            self._safety.enter_execution()

        try:
            return self._execute_inner(
                plan,
                trace,
                budget,
                t_start,
                context,
                traces,
                outputs,
                tokens_total,
                aborted_at,
            )
        finally:
            # ── Safety: exit execution (allows graph mutation) ────────────────
            if self._safety:
                self._safety.exit_execution()

    def _execute_inner(
        self,
        plan,
        trace,
        budget,
        t_start,
        context,
        traces,
        outputs,
        tokens_total,
        aborted_at,
    ) -> CapabilityResult:

        # ── Build execution graph ─────────────────────────────────────────────
        graph = self._graph_builder.build(plan.capability)

        # ── Validate graph integrity (Gap 5) ──────────────────────────────────
        graph_issues = graph.validate()
        if graph_issues:
            logger.warn(
                f"Graph validation issues for '{plan.capability.name}': "
                f"{graph_issues}"
            )

        logger.info(
            f"CapabilityExecutor starting '{plan.capability.name}'",
            steps=graph.node_count,
            execution_order=graph.execution_order,
        )

        # ── Execute in topological order ──────────────────────────────────────
        completed_nodes: set = set()

        for node_name in graph.execution_order:
            node = graph.nodes[node_name]

            # ── dependency check ──────────────────────────────────────────────
            unmet = [d for d in node.dependencies if d not in completed_nodes]
            if unmet:
                traces.append(
                    StepTrace(
                        step_name=node.name,
                        tool_name=node.tool_name,
                        status=StepStatus.SKIPPED,
                        skipped_reason=f"Unmet dependencies: {unmet}",
                    )
                )
                logger.debug(f"Node '{node.name}' skipped — unmet deps: {unmet}")
                if node.failure_policy in (FailurePolicy.ABORT, FailurePolicy.REQUIRED):
                    aborted_at = node.name
                    break
                continue

            # ── condition gate ────────────────────────────────────────────────
            if node.condition is not None:
                try:
                    should_run = bool(node.condition(context))
                except Exception:
                    should_run = True
                if not should_run:
                    traces.append(
                        StepTrace(
                            step_name=node.name,
                            tool_name=node.tool_name,
                            status=StepStatus.SKIPPED,
                            skipped_reason="condition evaluated False",
                        )
                    )
                    logger.debug(f"Node '{node.name}' skipped — condition False")
                    continue

            # ── tool lookup ───────────────────────────────────────────────────
            # Budget check for LLM-using nodes (CP5)
            if budget and node.allow_llm:
                if not budget.can_spend(
                    estimated_tokens=getattr(config, "LLM_MAX_TOKENS", 512)
                ):
                    traces.append(
                        StepTrace(
                            step_name=node.name,
                            tool_name=node.tool_name,
                            status=StepStatus.SKIPPED,
                            skipped_reason="Token budget exhausted",
                        )
                    )
                    if trace:
                        trace.add_step(
                            node.name, node.tool_name, "budget_blocked", skipped=True
                        )
                    logger.warn(f"Node '{node.name}' skipped — token budget exhausted")
                    continue

            tool = self._registry.get(node.tool_name)
            if tool is None:
                trace = StepTrace(
                    step_name=node.name,
                    tool_name=node.tool_name,
                    status=StepStatus.FAILED,
                    error=f"Tool '{node.tool_name}' not in registry",
                )
                traces.append(trace)
                if node.failure_policy in (FailurePolicy.ABORT, FailurePolicy.REQUIRED):
                    aborted_at = node.name
                    break
                continue

            # ── param resolution (context interpolation) ──────────────────────
            resolved_params = _resolve_node_params(node, context)
            # ── P2: Parameter pre-validation ──────────────────────────────────
            missing = _check_required_params(node, resolved_params)
            if missing:
                skip_msg = f"Missing required parameter(s): {', '.join(missing)}"
                traces.append(
                    StepTrace(
                        step_name=node.name,
                        tool_name=node.tool_name,
                        status=StepStatus.SKIPPED,
                        skipped_reason=skip_msg,
                    )
                )
                logger.debug(f"Node '{node.name}' skipped — {skip_msg}")
                if trace:
                    trace.add_step(
                        node.name,
                        node.tool_name,
                        "param_missing",
                        skipped=True,
                        error=skip_msg,
                    )
                if node.failure_policy in (FailurePolicy.ABORT, FailurePolicy.REQUIRED):
                    aborted_at = node.name
                    break
                continue
            # ── build TaskState ───────────────────────────────────────────────
            state = TaskState(
                task=f"{plan.capability.name}.{node.name}",
                tool=node.tool_name,
                params=resolved_params,
                student_id=plan.task_state.student_id,
                raw_input=plan.task_state.raw_input,
            )

            # ── execute (with retry) ──────────────────────────────────────────
            max_attempts = 1 + node.retry_policy
            result = None
            step_ms = 0.0
            step_tok = 0

            for attempt in range(max_attempts):
                t_step = time.monotonic()
                result = self._sandbox.run(tool, state)
                step_ms = (time.monotonic() - t_step) * 1000
                step_tok = result.metadata.get("tokens_used", 0)

                if result.status != ToolStatus.ERROR or attempt == max_attempts - 1:
                    break
                logger.debug(f"Node '{node.name}' attempt {attempt+1} failed, retrying")

            tokens_total += step_tok
            self._cfg.tool_call_count_session += 1

            # ── verification (Layer 4) ────────────────────────────────────────
            vr = self._verifier.verify(result, {"tool_name": node.tool_name})
            if not vr.passed:
                logger.warn(
                    f"Node '{node.name}' failed verification",
                    violations=vr.violations[:3],
                )
                # Attempt error recovery
                recovery_msg = self._recovery.recover(result)
                if recovery_msg and not result.success:
                    result.message = recovery_msg

            logger.info(
                f"Node '{node.name}' → {result.contract.value}",
                tool=node.tool_name,
                duration_ms=round(step_ms, 1),
                tokens=step_tok,
                status=result.status.value,
                verified=vr.passed,
            )

            # Record in execution trace (CP4/CP9)
            if trace:
                trace.add_step(
                    node.name,
                    node.tool_name,
                    result.contract.value,
                    duration_ms=step_ms,
                    tokens=step_tok,
                    error=result.error if not result.success else "",
                )

            # Record tokens in budget (CP5)
            if budget and step_tok > 0:
                budget.record(step_tok, reason=f"{plan.capability.name}.{node.name}")

            # ── contract enforcement ──────────────────────────────────────────
            if result.contract == ExecutionContract.FAILED:
                # P1: Run RecoveryEngine on failed capability nodes
                recovery_msg = self._recovery.recover(result)
                if recovery_msg:
                    result.message = recovery_msg
                    logger.info(
                        f"RecoveryEngine intercepted '{node.name}'",
                        recovery=recovery_msg[:80],
                    )
                    if trace:
                        trace.add_step(
                            f"{node.name}_recovery",
                            node.tool_name,
                            "recovered",
                            error=result.error,
                        )

                traces.append(
                    StepTrace(
                        step_name=node.name,
                        tool_name=node.tool_name,
                        status=StepStatus.FAILED,
                        contract=result.contract,
                        duration_ms=step_ms,
                        error=result.error,
                        tokens_used=step_tok,
                    )
                )
                # Store failed result (with recovery msg) so it appears in response
                outputs[node.name] = result

                if node.failure_policy in (FailurePolicy.ABORT, FailurePolicy.REQUIRED):
                    aborted_at = node.name
                    logger.warn(
                        f"Capability '{plan.capability.name}' aborted at node '{node.name}'",
                        error=result.error,
                    )
                    break
                logger.debug(f"Node '{node.name}' failed (SKIP policy): {result.error}")
                continue

            # ── success: write to context ─────────────────────────────────────
            traces.append(
                StepTrace(
                    step_name=node.name,
                    tool_name=node.tool_name,
                    status=StepStatus.COMPLETED,
                    contract=result.contract,
                    duration_ms=step_ms,
                    tokens_used=step_tok,
                )
            )
            outputs[node.name] = result
            completed_nodes.add(node.name)

            # Context receives the node's data payload for downstream nodes
            context[node.name] = _extract_context_data(result)

        # ── format aggregated response ────────────────────────────────────────
        total_ms = (time.monotonic() - t_start) * 1000
        message = _format_capability_response(
            capability_name=plan.capability.name,
            description=plan.capability.description,
            outputs=outputs,
            traces=traces,
            aborted_at=aborted_at,
        )

        result = CapabilityResult(
            capability_name=plan.capability.name,
            success=aborted_at is None,
            message=message,
            steps_trace=traces,
            step_outputs=outputs,
            tokens_used=tokens_total,
            execution_ms=round(total_ms, 1),
            aborted_at=aborted_at,
            partial=aborted_at is not None,
        )

        logger.info(
            f"CapabilityExecutor finished '{plan.capability.name}'",
            completed=result.steps_completed,
            failed=result.steps_failed,
            tokens=tokens_total,
            ms=round(total_ms, 1),
            aborted=aborted_at or "none",
        )
        return result


# ── P2: required param checker ────────────────────────────────────────────────
# Maps tool_name → list of required param names for pre-validation.
_TOOL_REQUIRED_PARAMS: Dict[str, list] = {
    "exam_topic_predictor": ["course_id"],
    "practice_question_generator": ["topic"],
    "course_material_search": ["course_id"],
}


def _check_required_params(node: GraphNode, resolved_params: Dict[str, Any]) -> list:
    """
    Return list of missing required parameters for a node.
    Empty list = all params present.
    Checks both the static registry and unresolved {ref} placeholders.
    """
    required = _TOOL_REQUIRED_PARAMS.get(node.tool_name, [])
    missing = []
    for p in required:
        val = resolved_params.get(p)
        if val is None:
            missing.append(p)
        elif isinstance(val, str) and val.startswith("{") and val.endswith("}"):
            # Still an unresolved reference — upstream step didn't produce it
            missing.append(f"{p} (unresolved: {val})")
    return missing


# ── node param resolution ─────────────────────────────────────────────────────
def _resolve_node_params(node: GraphNode, context: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve {step_name.field} references in node params from context."""
    from infrarely.agent.capability import _resolve_ref

    resolved = {}
    for key, value in node.base_params.items():
        if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
            ref = value[1:-1]
            resolved[key] = _resolve_ref(ref, context, default=value)
        else:
            resolved[key] = value
    return resolved


# ── context extraction ────────────────────────────────────────────────────────
def _extract_context_data(result: ToolResult) -> Any:
    """
    Extract the most useful data from a ToolResult for use in downstream steps.
    Returns a dict whenever possible so {step.field} references work.
    """
    if isinstance(result.data, dict):
        return result.data
    if isinstance(result.data, list):
        # Wrap list in a dict with a 'items' key for consistent interpolation
        return {"items": result.data, "count": len(result.data)}
    if result.data is not None:
        return {"value": result.data, "message": result.message}
    return {"message": result.message}


# ── response formatter ────────────────────────────────────────────────────────
def _format_capability_response(
    capability_name: str,
    description: str,
    outputs: Dict[str, ToolResult],
    traces: list[StepTrace],
    aborted_at: Optional[str],
) -> str:
    """
    Assembles a human-readable response from all completed step outputs.
    Each step contributes its formatted section.
    Deterministic layout — same inputs always produce the same output.
    """
    from infrarely.agent.response_formatter import format_result

    sections = []

    # Header
    header = f"📋 {description or capability_name}"
    if aborted_at:
        header += (
            f"\n⚠️  Workflow stopped at step '{aborted_at}' — partial results below"
        )
    sections.append(header)

    # One section per step (completed, failed with recovery, or skipped)
    for trace in traces:
        if trace.status == StepStatus.COMPLETED and trace.step_name in outputs:
            result = outputs[trace.step_name]
            content = format_result(result)
            label = trace.step_name.replace("_", " ").title()
            sections.append(f"\n── {label} ──\n{content}")
        elif trace.status == StepStatus.FAILED:
            # P1: Show recovery message if available, not just raw error
            if trace.step_name in outputs and outputs[trace.step_name].message:
                sections.append(
                    f"\n── {trace.step_name} ──\n{outputs[trace.step_name].message}"
                )
            else:
                sections.append(f"\n── {trace.step_name} ──\n⚠️  {trace.error}")
        elif trace.status == StepStatus.SKIPPED:
            reason = trace.skipped_reason or "Skipped"
            sections.append(f"\n── {trace.step_name} ──\n○ {reason}")

    return "\n".join(sections)
