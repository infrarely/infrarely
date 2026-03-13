"""
aos/workflow.py — Workflow & Step engine
═══════════════════════════════════════════════════════════════════════════════
Define multi-step capabilities as DAGs.

  @infrarely.capability("exam_prep")
  def exam_prep(topic: str):
      return infrarely.workflow([
          infrarely.step("search", tool=search_fn, inputs={"q": topic}),
          infrarely.step("write",  tool=write_fn,  depends_on=["search"]),
      ])

Steps execute in dependency order. Independent steps run in parallel.
Each step has: timeout, fallback, retry, condition.
"""

from __future__ import annotations

import inspect

import concurrent.futures
import re
import time
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# STEP DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Step:
    """
    A single step in a workflow DAG.

    Args:
        name:             Unique ID in this workflow
        tool:             @infrarely.tool decorated function
        inputs:           Static values or "{step_name}.output" references
        depends_on:       Step names this step waits for
        condition:        "if {step}.output.{field} {op} {value}"
        fallback:         Step name to run if this fails
        required:         If False, SKIPPED on failure is OK
        timeout_seconds:  Per-step timeout
        can_parallelize:  Hint for parallel group detection
    """

    name: str
    tool: Optional[Callable] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    condition: str = ""
    fallback: str = ""
    required: bool = True
    timeout_seconds: int = 30
    can_parallelize: bool = True


def step(
    name: str,
    tool: Optional[Callable] = None,
    inputs: Optional[Dict[str, Any]] = None,
    depends_on: Optional[List[str]] = None,
    condition: str = "",
    fallback: str = "",
    required: bool = True,
    timeout_seconds: int = 30,
    can_parallelize: bool = True,
) -> Step:
    """
    Create a workflow step.

    Usage:
        infrarely.step("search", tool=search_web, inputs={"query": topic})
        infrarely.step("write", tool=write_report, depends_on=["search"])
    """
    return Step(
        name=name,
        tool=tool,
        inputs=inputs or {},
        depends_on=depends_on or [],
        condition=condition,
        fallback=fallback,
        required=required,
        timeout_seconds=timeout_seconds,
        can_parallelize=can_parallelize,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STEP RESULT
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class StepResult:
    """Result of executing one step."""

    name: str
    output: Any = None
    success: bool = True
    skipped: bool = False
    error: str = ""
    duration_ms: float = 0.0
    used_fallback: bool = False

    def __bool__(self) -> bool:
        return self.success


# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW ENGINE
# ═══════════════════════════════════════════════════════════════════════════════


class Workflow:
    """
    Executable DAG of Steps with dependency resolution,
    parallel execution, condition evaluation, and fallback handling.
    """

    def __init__(self, steps: List[Step]):
        self._steps = {s.name: s for s in steps}
        self._order = self._topological_sort(steps)
        self._results: Dict[str, StepResult] = {}
        self.workflow_id = f"wf_{uuid.uuid4().hex[:8]}"

    # ── Topological sort ──────────────────────────────────────────────────────

    @staticmethod
    def _topological_sort(steps: List[Step]) -> List[List[str]]:
        """
        Returns execution levels (each level can run in parallel).
        Level 0 has no dependencies, Level 1 depends on Level 0, etc.
        """
        name_set = {s.name for s in steps}
        deps = {s.name: [d for d in s.depends_on if d in name_set] for s in steps}
        in_degree = {name: 0 for name in name_set}
        for name, dep_list in deps.items():
            in_degree[name] = len(dep_list)

        levels = []
        remaining = set(name_set)

        while remaining:
            # Find all nodes with in_degree == 0
            level = [n for n in remaining if in_degree.get(n, 0) == 0]
            if not level:
                # Cycle detected — force break
                level = [min(remaining)]
            levels.append(sorted(level))
            for n in level:
                remaining.discard(n)
                # Reduce in-degree of dependents
                for other in remaining:
                    if n in deps.get(other, []):
                        in_degree[other] = max(0, in_degree[other] - 1)

        return levels

    # ── Condition evaluation ──────────────────────────────────────────────────

    def _evaluate_condition(self, condition: str) -> bool:
        """
        Evaluate a condition string like:
          "if predict_topics.output.count > 0"
        """
        if not condition:
            return True

        condition = condition.strip()
        if condition.startswith("if "):
            condition = condition[3:].strip()

        # Pattern: step.output.field op value
        match = re.match(
            r"(\w+)\.output(?:\.(\w+))?\s*(==|!=|>|<|>=|<=)\s*(.+)",
            condition,
        )
        if not match:
            return True  # Can't parse → allow

        step_name, field_name, op, value_str = match.groups()
        value_str = value_str.strip().strip('"').strip("'")

        step_result = self._results.get(step_name)
        if not step_result or not step_result.success:
            return False

        # Get the actual value
        actual = step_result.output
        if field_name and isinstance(actual, dict):
            actual = actual.get(field_name)
        elif field_name and hasattr(actual, field_name):
            actual = getattr(actual, field_name)

        # Parse expected value
        try:
            expected = type(actual)(value_str) if actual is not None else value_str
        except (ValueError, TypeError):
            try:
                expected = json.loads(value_str)
            except Exception:
                expected = value_str

        # Compare
        ops = {
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
            ">": lambda a, b: a > b,
            "<": lambda a, b: a < b,
            ">=": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b,
        }
        try:
            return ops.get(op, lambda a, b: True)(actual, expected)
        except (TypeError, ValueError):
            return True

    # ── Input resolution ──────────────────────────────────────────────────────

    def _resolve_inputs(self, step: Step) -> Dict[str, Any]:
        """Resolve input references like "{step_name}.output" to actual values.

        Also auto-injects dependency outputs into the tool's parameters when
        the step has depends_on but no explicit inputs referencing those deps.
        For root steps (no depends_on), inject initial_context.
        """
        resolved = {}
        referenced_deps: Set[str] = set()

        # Inject initial_context for root steps (no dependencies)
        if (
            not step.depends_on
            and hasattr(self, "_initial_context")
            and self._initial_context
        ):
            if step.tool is not None:
                try:
                    sig = inspect.signature(step.tool)
                    for pname, param in sig.parameters.items():
                        if pname in self._initial_context and param.kind in (
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            inspect.Parameter.KEYWORD_ONLY,
                        ):
                            resolved[pname] = self._initial_context[pname]
                except (ValueError, TypeError):
                    pass

        for key, value in step.inputs.items():
            if isinstance(value, str) and "{" in value and "}" in value:
                # Pattern: {step_name.output} or {step_name.output.field}
                match = re.match(r"\{(\w+)\.output(?:\.(\w+))?\}", value)
                if match:
                    ref_step, ref_field = match.groups()
                    referenced_deps.add(ref_step)
                    ref_result = self._results.get(ref_step)
                    if ref_result and ref_result.success:
                        val = ref_result.output
                        if ref_field and isinstance(val, dict):
                            val = val.get(ref_field, val)
                        resolved[key] = val
                        continue
                # Also handle simple "step_name.output" without braces
                resolved[key] = value
            else:
                resolved[key] = value

        # Auto-inject dependency outputs for deps not already referenced
        unreferenced = [
            d
            for d in step.depends_on
            if d not in referenced_deps
            and d in self._results
            and self._results[d].success
        ]

        if unreferenced and step.tool is not None:
            try:
                sig = inspect.signature(step.tool)
                params = list(sig.parameters.values())
                # Filter to params not already filled
                unfilled = [
                    p
                    for p in params
                    if p.name not in resolved
                    and p.kind
                    in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                ]
                # Map dependency outputs to unfilled params by position
                for i, dep_name in enumerate(unreferenced):
                    if i < len(unfilled):
                        dep_output = self._results[dep_name].output
                        param_name = unfilled[i].name
                        # Smart extraction: if output is dict and param name matches a key, use that
                        if isinstance(dep_output, dict) and param_name in dep_output:
                            resolved[param_name] = dep_output[param_name]
                        else:
                            resolved[param_name] = dep_output
            except (ValueError, TypeError):
                pass

        return resolved

    # ── Execute one step ──────────────────────────────────────────────────────

    def _execute_step(self, step: Step) -> StepResult:
        """Execute a single step with timeout and error handling."""
        start = time.monotonic()

        # Check condition
        if step.condition and not self._evaluate_condition(step.condition):
            return StepResult(
                name=step.name,
                skipped=True,
                success=True,
                duration_ms=(time.monotonic() - start) * 1000,
            )

        if step.tool is None:
            return StepResult(
                name=step.name,
                success=False,
                error="No tool assigned",
                duration_ms=(time.monotonic() - start) * 1000,
            )

        # Resolve inputs
        inputs = self._resolve_inputs(step)

        try:
            # Execute with timeout
            result_container = [None]
            error_container = [None]

            def _run():
                try:
                    result_container[0] = step.tool(**inputs)
                except Exception as e:
                    error_container[0] = e

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            t.join(timeout=step.timeout_seconds)

            if t.is_alive():
                elapsed = (time.monotonic() - start) * 1000
                error_msg = (
                    f"Step '{step.name}' timed out after {step.timeout_seconds}s"
                )
                # Try fallback
                if step.fallback and step.fallback in self._steps:
                    fb_result = self._execute_step(self._steps[step.fallback])
                    fb_result.used_fallback = True
                    return fb_result
                if not step.required:
                    return StepResult(
                        name=step.name,
                        skipped=True,
                        success=True,
                        error=error_msg,
                        duration_ms=elapsed,
                    )
                return StepResult(
                    name=step.name, success=False, error=error_msg, duration_ms=elapsed
                )

            if error_container[0]:
                raise error_container[0]

            result = result_container[0]
            elapsed = (time.monotonic() - start) * 1000

            # Check if tool returned an __aos_error
            if isinstance(result, dict) and result.get("__aos_error"):
                error_msg = result.get("message", "Tool failed")
                if step.fallback and step.fallback in self._steps:
                    fb_result = self._execute_step(self._steps[step.fallback])
                    fb_result.used_fallback = True
                    return fb_result
                if not step.required:
                    return StepResult(
                        name=step.name,
                        skipped=True,
                        success=True,
                        error=error_msg,
                        duration_ms=elapsed,
                    )
                return StepResult(
                    name=step.name, success=False, error=error_msg, duration_ms=elapsed
                )

            return StepResult(
                name=step.name, output=result, success=True, duration_ms=elapsed
            )

        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            error_msg = f"Step '{step.name}' failed: {e}"

            # Try fallback
            if step.fallback and step.fallback in self._steps:
                fb_result = self._execute_step(self._steps[step.fallback])
                fb_result.used_fallback = True
                return fb_result

            if not step.required:
                return StepResult(
                    name=step.name,
                    skipped=True,
                    success=True,
                    error=error_msg,
                    duration_ms=elapsed,
                )

            return StepResult(
                name=step.name, success=False, error=error_msg, duration_ms=elapsed
            )

    # ── Execute full workflow ─────────────────────────────────────────────────

    def execute(
        self, initial_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, StepResult]:
        """
        Execute the workflow DAG level by level.
        Independent steps within a level run in parallel.
        Returns dict of {step_name: StepResult}.
        """
        self._results.clear()
        self._initial_context = initial_context or {}

        for level in self._order:
            if len(level) == 1:
                # Single step — run directly
                step = self._steps[level[0]]
                result = self._execute_step(step)
                self._results[step.name] = result
                if not result.success and not result.skipped and step.required:
                    break  # Stop on required step failure
            else:
                # Multiple steps — run in parallel
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=len(level)
                ) as executor:
                    futures = {}
                    for name in level:
                        step = self._steps[name]
                        futures[name] = executor.submit(self._execute_step, step)

                    has_failure = False
                    for name, future in futures.items():
                        try:
                            result = future.result(
                                timeout=self._steps[name].timeout_seconds + 5
                            )
                        except Exception as e:
                            result = StepResult(name=name, success=False, error=str(e))
                        self._results[name] = result
                        if (
                            not result.success
                            and not result.skipped
                            and self._steps[name].required
                        ):
                            has_failure = True

                    if has_failure:
                        break

        return dict(self._results)

    @property
    def steps(self) -> List[Step]:
        return list(self._steps.values())

    @property
    def results(self) -> Dict[str, StepResult]:
        return dict(self._results)

    @property
    def final_output(self) -> Any:
        """Return the output of the last executed step."""
        if not self._results:
            return None
        # Find the last step in topological order that has output
        for level in reversed(self._order):
            for name in reversed(level):
                r = self._results.get(name)
                if r and r.success and r.output is not None:
                    return r.output
        return None

    @property
    def all_succeeded(self) -> bool:
        return all(r.success for r in self._results.values())

    @property
    def total_duration_ms(self) -> float:
        return sum(r.duration_ms for r in self._results.values())

    # ── State serialization (for recovery/resume) ─────────────────────────────

    def export_state(self) -> Dict[str, Any]:
        """
        Export workflow execution state as a JSON-serializable dict.
        Enables workflow recovery, persistence, and debugging.

        Usage:
            results = wf.execute()
            state = wf.export_state()
            json.dump(state, open("wf_state.json", "w"))
        """
        step_states = {}
        for name, step in self._steps.items():
            r = self._results.get(name)
            step_states[name] = {
                "depends_on": step.depends_on,
                "required": step.required,
                "executed": r is not None,
                "success": r.success if r else None,
                "skipped": r.skipped if r else None,
                "output": r.output if r else None,
                "error": r.error if r else None,
                "duration_ms": r.duration_ms if r else None,
                "used_fallback": r.used_fallback if r else None,
            }
        return {
            "workflow_id": self.workflow_id,
            "steps": step_states,
            "all_succeeded": self.all_succeeded if self._results else None,
            "total_duration_ms": self.total_duration_ms,
            "final_output": self.final_output,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC FUNCTION — infrarely.workflow()
# ═══════════════════════════════════════════════════════════════════════════════


def workflow(steps: List[Step]) -> Workflow:
    """
    Create an executable workflow from a list of steps.

    Usage:
        wf = infrarely.workflow([
            infrarely.step("search", tool=search_fn, inputs={"q": topic}),
            infrarely.step("write",  tool=write_fn,  depends_on=["search"]),
        ])
    """
    return Workflow(steps)


# ═══════════════════════════════════════════════════════════════════════════════
# PARALLEL EXECUTION — aos.parallel()
# ═══════════════════════════════════════════════════════════════════════════════


import json


def parallel(tasks: List[Tuple[Any, str]], timeout: int = 60) -> List[Any]:
    """
    Run multiple (agent, goal) pairs in parallel.

    Usage:
        results = aos.parallel([
            (researcher, "Research company A"),
            (researcher, "Research company B"),
            (writer,     "Write summary"),
        ])
    """
    from infrarely.core.result import Result, _fail, ErrorType

    results = [None] * len(tasks)

    def _run_task(index: int, agent: Any, goal: str):
        try:
            results[index] = agent.run(goal)
        except Exception as e:
            results[index] = _fail(
                ErrorType.TOOL_FAILURE,
                f"Parallel task {index} failed: {e}",
                goal=goal,
            )

    threads = []
    for i, (agent, goal) in enumerate(tasks):
        t = threading.Thread(target=_run_task, args=(i, agent, goal), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=timeout)

    # Fill in any None results (timed out)
    for i, r in enumerate(results):
        if r is None:
            results[i] = _fail(
                ErrorType.TIMEOUT,
                f"Parallel task {i} timed out after {timeout}s",
                goal=tasks[i][1] if i < len(tasks) else "",
            )

    return results
