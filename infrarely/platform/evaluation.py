"""
aos/evaluation.py — Agent Evaluation & Regression Testing Framework
═══════════════════════════════════════════════════════════════════════════════
Built-in way to test whether agents give correct answers after changes.

62% of production teams say evaluation is their top need. This module
provides structured evaluation suites with regression tracking.

Usage::

    suite = aos.eval.suite("study-assistant-evals")

    suite.add(
        input="What is 144 / 12?",
        expected_output=12,
        expected_used_llm=False,
        expected_confidence_min=0.99,
    )
    suite.add(
        input="What do mitochondria do?",
        expected_sources=["biology"],
        expected_used_llm=False,
    )

    results = suite.run(agent)
    print(results.pass_rate)           # 0.95
    print(results.regression_report)   # what changed vs last run
    print(results.failed_cases)        # exactly which inputs broke
"""

from __future__ import annotations

import json
import os
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from infrarely.core.agent import Agent


# ═══════════════════════════════════════════════════════════════════════════════
# EVAL CASE — A single test/evaluation case
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class EvalCase:
    """A single evaluation case with expected behavior."""

    input: str = ""
    expected_output: Any = None
    expected_used_llm: Optional[bool] = None
    expected_sources: Optional[List[str]] = None
    expected_confidence_min: Optional[float] = None
    expected_confidence_max: Optional[float] = None
    expected_success: bool = True
    timeout: float = 30.0
    tags: List[str] = field(default_factory=list)
    description: str = ""
    custom_validator: Optional[Callable] = None  # fn(result) -> bool

    @property
    def case_id(self) -> str:
        """Deterministic ID based on input."""
        return hashlib.md5(self.input.encode()).hexdigest()[:12]


# ═══════════════════════════════════════════════════════════════════════════════
# EVAL RESULT — Result of a single case evaluation
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class EvalCaseResult:
    """Result of evaluating a single case."""

    case: EvalCase = field(default_factory=EvalCase)
    passed: bool = False
    actual_output: Any = None
    actual_used_llm: Optional[bool] = None
    actual_sources: Optional[List[str]] = None
    actual_confidence: float = 0.0
    actual_success: bool = False
    failures: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    error: Optional[str] = None

    @property
    def input(self) -> str:
        """Shortcut to case.input."""
        return self.case.input

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input": self.case.input,
            "passed": self.passed,
            "actual_output": str(self.actual_output),
            "actual_used_llm": self.actual_used_llm,
            "actual_sources": self.actual_sources,
            "actual_confidence": self.actual_confidence,
            "failures": self.failures,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EVAL SUITE RESULTS — Aggregate results for a full suite run
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class EvalSuiteResults:
    """Aggregate results from running an evaluation suite."""

    suite_name: str = ""
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases_count: int = 0
    skipped_cases: int = 0
    pass_rate: float = 0.0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    avg_confidence: float = 0.0
    llm_usage_rate: float = 0.0
    case_results: List[EvalCaseResult] = field(default_factory=list)
    regression_report: str = ""
    run_timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> int:
        """Alias for total_cases."""
        return self.total_cases

    @property
    def cases(self) -> List[EvalCaseResult]:
        """Alias for case_results."""
        return self.case_results

    @property
    def failed_cases(self) -> List[EvalCaseResult]:
        """Get all failed cases."""
        return [r for r in self.case_results if not r.passed]

    @property
    def passed_cases_list(self) -> List[EvalCaseResult]:
        """Get all passed cases."""
        return [r for r in self.case_results if r.passed]

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"═══ Evaluation Report: {self.suite_name} ═══",
            f"Pass Rate:   {self.pass_rate:.1f}% ({self.passed_cases}/{self.total_cases})",
            f"Duration:    {self.total_duration_ms:.0f}ms total, {self.avg_duration_ms:.0f}ms avg",
            f"Confidence:  {self.avg_confidence:.2f} avg",
            f"LLM Usage:   {self.llm_usage_rate:.1%}",
        ]
        if self.failed_cases:
            lines.append(f"\n── Failed Cases ({self.failed_cases_count}) ──")
            for r in self.failed_cases:
                lines.append(f'  ✗ "{r.case.input[:60]}"')
                for f in r.failures:
                    lines.append(f"    → {f}")
                if r.error:
                    lines.append(f"    → Error: {r.error}")
        if self.regression_report:
            lines.append(f"\n── Regression Report ──")
            lines.append(self.regression_report)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases_count,
            "pass_rate": self.pass_rate,
            "total_duration_ms": self.total_duration_ms,
            "avg_duration_ms": self.avg_duration_ms,
            "avg_confidence": self.avg_confidence,
            "llm_usage_rate": self.llm_usage_rate,
            "case_results": [r.to_dict() for r in self.case_results],
            "regression_report": self.regression_report,
            "run_timestamp": self.run_timestamp,
        }

    def __str__(self) -> str:
        return self.summary()


# ═══════════════════════════════════════════════════════════════════════════════
# EVAL SUITE — The main evaluation container
# ═══════════════════════════════════════════════════════════════════════════════


class EvalSuite:
    """
    A collection of evaluation cases that can be run against an agent.

    Usage::

        suite = EvalSuite("my-evals")
        suite.add(input="2+2", expected_output=4)
        results = suite.run(agent)
    """

    def __init__(self, name: str, *, description: str = ""):
        self._name = name
        self._description = description
        self._cases: List[EvalCase] = []
        self._history: List[EvalSuiteResults] = []
        self._results_dir: Optional[str] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def cases(self) -> List[EvalCase]:
        return list(self._cases)

    def add(
        self,
        input: str,
        *,
        expected_output: Any = None,
        expected_used_llm: Optional[bool] = None,
        expected_sources: Optional[List[str]] = None,
        expected_confidence_min: Optional[float] = None,
        expected_confidence_max: Optional[float] = None,
        expected_success: bool = True,
        timeout: float = 30.0,
        tags: Optional[List[str]] = None,
        description: str = "",
        custom_validator: Optional[Callable] = None,
    ) -> "EvalSuite":
        """Add an evaluation case. Returns self for chaining."""
        self._cases.append(
            EvalCase(
                input=input,
                expected_output=expected_output,
                expected_used_llm=expected_used_llm,
                expected_sources=expected_sources,
                expected_confidence_min=expected_confidence_min,
                expected_confidence_max=expected_confidence_max,
                expected_success=expected_success,
                timeout=timeout,
                tags=tags or [],
                description=description,
                custom_validator=custom_validator,
            )
        )
        return self

    def run(
        self,
        agent: "Agent",
        *,
        tags: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> EvalSuiteResults:
        """
        Run all evaluation cases against the given agent.

        Parameters
        ----------
        agent : Agent
            The agent to evaluate.
        tags : list[str], optional
            If provided, only run cases with matching tags.
        verbose : bool
            If True, print progress as cases are evaluated.

        Returns
        -------
        EvalSuiteResults
        """
        cases = self._cases
        if tags:
            cases = [c for c in cases if any(t in c.tags for t in tags)]

        case_results: List[EvalCaseResult] = []
        total_start = time.time()

        for i, case in enumerate(cases):
            if verbose:
                print(
                    f"  [{i+1}/{len(cases)}] {case.input[:60]}...", end=" ", flush=True
                )

            case_result = self._evaluate_case(agent, case)
            case_results.append(case_result)

            if verbose:
                status = "PASS" if case_result.passed else "FAIL"
                print(f"{status} ({case_result.duration_ms:.0f}ms)")

        total_duration = (time.time() - total_start) * 1000

        # Calculate aggregate metrics
        passed = sum(1 for r in case_results if r.passed)
        total = len(case_results)
        confidences = [r.actual_confidence for r in case_results]
        llm_used_count = sum(1 for r in case_results if r.actual_used_llm)

        results = EvalSuiteResults(
            suite_name=self._name,
            total_cases=total,
            passed_cases=passed,
            failed_cases_count=total - passed,
            pass_rate=(passed / total * 100) if total > 0 else 0.0,
            total_duration_ms=total_duration,
            avg_duration_ms=total_duration / total if total > 0 else 0.0,
            avg_confidence=sum(confidences) / len(confidences) if confidences else 0.0,
            llm_usage_rate=llm_used_count / total if total > 0 else 0.0,
            case_results=case_results,
        )

        # Generate regression report if we have history
        if self._history:
            results.regression_report = self._generate_regression_report(
                results, self._history[-1]
            )

        # Store in history
        self._history.append(results)

        # Save to disk if configured
        if self._results_dir:
            self._save_results(results)

        return results

    def _evaluate_case(self, agent: "Agent", case: EvalCase) -> EvalCaseResult:
        """Evaluate a single case against the agent."""
        failures: List[str] = []
        start = time.time()

        try:
            result = agent.run(case.input)
            duration_ms = (time.time() - start) * 1000

            cr = EvalCaseResult(
                case=case,
                actual_output=result.output,
                actual_used_llm=result.used_llm,
                actual_sources=result.sources,
                actual_confidence=result.confidence,
                actual_success=result.success,
                duration_ms=duration_ms,
            )

            # Check expected_success
            if case.expected_success != result.success:
                failures.append(
                    f"Expected success={case.expected_success}, got {result.success}"
                )

            # Check expected_output
            if case.expected_output is not None:
                if not self._output_matches(result.output, case.expected_output):
                    failures.append(
                        f"Expected output={case.expected_output!r}, got {result.output!r}"
                    )

            # Check expected_used_llm
            if case.expected_used_llm is not None:
                if result.used_llm != case.expected_used_llm:
                    failures.append(
                        f"Expected used_llm={case.expected_used_llm}, got {result.used_llm}"
                    )

            # Check expected_sources
            if case.expected_sources is not None:
                for src in case.expected_sources:
                    found = any(src.lower() in s.lower() for s in result.sources)
                    if not found:
                        failures.append(
                            f"Expected source '{src}' not found in {result.sources}"
                        )

            # Check expected_confidence_min
            if case.expected_confidence_min is not None:
                if result.confidence < case.expected_confidence_min:
                    failures.append(
                        f"Confidence {result.confidence:.2f} below min {case.expected_confidence_min:.2f}"
                    )

            # Check expected_confidence_max
            if case.expected_confidence_max is not None:
                if result.confidence > case.expected_confidence_max:
                    failures.append(
                        f"Confidence {result.confidence:.2f} above max {case.expected_confidence_max:.2f}"
                    )

            # Check custom validator
            if case.custom_validator is not None:
                try:
                    if not case.custom_validator(result):
                        failures.append("Custom validator returned False")
                except Exception as e:
                    failures.append(f"Custom validator raised: {e}")

            cr.failures = failures
            cr.passed = len(failures) == 0
            return cr

        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            return EvalCaseResult(
                case=case,
                passed=False,
                duration_ms=duration_ms,
                error=str(e),
                failures=[f"Exception during evaluation: {e}"],
            )

    def _output_matches(self, actual: Any, expected: Any) -> bool:
        """Flexible output comparison."""
        # Exact match
        if actual == expected:
            return True

        # Numeric comparison with tolerance
        try:
            a_num = float(actual)
            e_num = float(expected)
            if abs(a_num - e_num) < 1e-6:
                return True
        except (TypeError, ValueError):
            pass

        # String containment (if expected is string)
        if isinstance(expected, str) and isinstance(actual, str):
            if expected.lower() in actual.lower():
                return True

        # String representation
        if str(actual) == str(expected):
            return True

        return False

    def _generate_regression_report(
        self,
        current: EvalSuiteResults,
        previous: EvalSuiteResults,
    ) -> str:
        """Generate a regression report comparing current vs previous run."""
        lines = []

        # Pass rate change
        rate_change = current.pass_rate - previous.pass_rate
        if abs(rate_change) < 0.1:
            lines.append(f"Pass rate unchanged: {current.pass_rate:.1f}%")
        elif rate_change > 0:
            lines.append(
                f"Pass rate improved: {previous.pass_rate:.1f}% → {current.pass_rate:.1f}% (+{rate_change:.1f}%)"
            )
        else:
            lines.append(
                f"Pass rate REGRESSED: {previous.pass_rate:.1f}% → {current.pass_rate:.1f}% ({rate_change:.1f}%)"
            )

        # Duration change
        dur_change = current.avg_duration_ms - previous.avg_duration_ms
        if abs(dur_change) > 10:
            direction = "slower" if dur_change > 0 else "faster"
            lines.append(
                f"Avg duration: {previous.avg_duration_ms:.0f}ms → {current.avg_duration_ms:.0f}ms ({direction})"
            )

        # LLM usage change
        llm_change = current.llm_usage_rate - previous.llm_usage_rate
        if abs(llm_change) > 0.01:
            direction = "more" if llm_change > 0 else "less"
            lines.append(
                f"LLM usage: {previous.llm_usage_rate:.1%} → {current.llm_usage_rate:.1%} ({direction} LLM)"
            )

        # New failures
        prev_inputs = {r.case.input for r in previous.case_results if r.passed}
        curr_failed = {r.case.input for r in current.case_results if not r.passed}
        new_failures = prev_inputs & curr_failed
        if new_failures:
            lines.append(f"\nNew failures ({len(new_failures)}):")
            for inp in list(new_failures)[:5]:
                lines.append(f'  - "{inp[:60]}"')

        # New passes
        prev_failed = {r.case.input for r in previous.case_results if not r.passed}
        curr_passed = {r.case.input for r in current.case_results if r.passed}
        new_passes = prev_failed & curr_passed
        if new_passes:
            lines.append(f"\nNew passes ({len(new_passes)}):")
            for inp in list(new_passes)[:5]:
                lines.append(f'  + "{inp[:60]}"')

        return "\n".join(lines) if lines else "No previous run to compare."

    def save_results_to(self, directory: str) -> None:
        """Configure directory for saving results to disk."""
        self._results_dir = directory
        os.makedirs(directory, exist_ok=True)

    def _save_results(self, results: EvalSuiteResults) -> None:
        """Save results to disk."""
        if not self._results_dir:
            return
        timestamp = int(results.run_timestamp)
        filename = f"{self._name}_{timestamp}.json"
        filepath = os.path.join(self._results_dir, filename)
        try:
            with open(filepath, "w") as f:
                json.dump(results.to_dict(), f, indent=2, default=str)
        except Exception:
            pass

    def load_from_yaml(self, filepath: str) -> "EvalSuite":
        """
        Load evaluation cases from a YAML file.

        YAML format::

            cases:
              - input: "What is 2+2?"
                expected_output: 4
                expected_used_llm: false
        """
        # Use simple YAML-like parser (no dependency needed)
        with open(filepath, "r") as f:
            content = f.read()
        cases = _parse_simple_yaml(content)
        for case_data in cases:
            self.add(**case_data)
        return self

    def __len__(self) -> int:
        return len(self._cases)

    def __repr__(self) -> str:
        return f"EvalSuite(name={self._name!r}, cases={len(self._cases)})"


# ═══════════════════════════════════════════════════════════════════════════════
# EVAL NAMESPACE — Top-level aos.eval interface
# ═══════════════════════════════════════════════════════════════════════════════


class EvalNamespace:
    """
    Namespace for evaluation functionality.

    Usage::

        suite = aos.eval.suite("my-evals")
        results = suite.run(agent)
    """

    def suite(self, name: str, *, description: str = "") -> EvalSuite:
        """Create a new evaluation suite."""
        return EvalSuite(name, description=description)

    def quick_eval(
        self,
        agent: "Agent",
        inputs: List[str],
        *,
        verbose: bool = False,
    ) -> EvalSuiteResults:
        """
        Quick evaluation with just inputs (no expected values).

        Useful for smoke testing — just checks that the agent
        doesn't crash and produces output.
        """
        suite = EvalSuite("quick-eval")
        for inp in inputs:
            suite.add(input=inp, expected_success=True)
        return suite.run(agent, verbose=verbose)

    def quick(
        self,
        agent: "Agent" = None,
        cases: list = None,
        *,
        verbose: bool = False,
    ) -> EvalSuiteResults:
        """
        Quick evaluation with cases as (input, expected_output) tuples.
        """
        suite = EvalSuite("quick-eval")
        if cases:
            for item in cases:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    suite.add(
                        input=item[0], expected_output=item[1], expected_success=True
                    )
                else:
                    suite.add(input=str(item), expected_success=True)
        return suite.run(agent, verbose=verbose)


# ═══════════════════════════════════════════════════════════════════════════════
# SIMPLE YAML PARSER (no dependency)
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_simple_yaml(content: str) -> List[Dict[str, Any]]:
    """
    Parse a simple YAML-like format for eval cases.
    Supports basic key: value pairs under a 'cases:' list.
    """
    cases: List[Dict[str, Any]] = []
    current_case: Dict[str, Any] = {}
    in_cases = False

    for line in content.split("\n"):
        stripped = line.strip()

        if stripped == "cases:":
            in_cases = True
            continue

        if not in_cases:
            continue

        if stripped.startswith("- "):
            if current_case:
                cases.append(current_case)
            current_case = {}
            stripped = stripped[2:].strip()

        if ":" in stripped and not stripped.startswith("#"):
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip()

            # Type coercion
            if value.lower() in ("true", "yes"):
                value = True
            elif value.lower() in ("false", "no"):
                value = False
            elif value.lower() in ("null", "none", "~"):
                value = None
            else:
                try:
                    if "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                except (ValueError, TypeError):
                    # Strip quotes
                    if (value.startswith('"') and value.endswith('"')) or (
                        value.startswith("'") and value.endswith("'")
                    ):
                        value = value[1:-1]
                    # Handle lists [a, b, c]
                    if (
                        isinstance(value, str)
                        and value.startswith("[")
                        and value.endswith("]")
                    ):
                        items = value[1:-1].split(",")
                        value = [i.strip().strip("'\"") for i in items if i.strip()]

            if key in current_case or key == "input":
                current_case[key] = value
            else:
                current_case[key] = value

    if current_case:
        cases.append(current_case)

    return cases
