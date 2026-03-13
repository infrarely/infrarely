"""
adaptive/failure_analyzer.py — Module 4: Failure Analyzer
═══════════════════════════════════════════════════════════════════════════════
Detects repeating failure patterns and produces actionable reports.

Tracks:
  tool_failures, verification_failures, missing_parameters, timeout_events

Outputs:
  Failure reports to logs/failure_reports/.
  In-memory pattern detection with mitigation suggestions.
"""

from __future__ import annotations
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import infrarely.core.app_config as config
from infrarely.observability import logger

_REPORT_DIR = os.path.join(config.LOG_DIR, "failure_reports")
os.makedirs(_REPORT_DIR, exist_ok=True)


@dataclass
class FailureRecord:
    """Single failure event."""

    ts: str
    tool: str
    error_type: str  # "tool_error" | "verification" | "missing_param" | "timeout"
    error_msg: str
    intent: str = ""
    capability: str = ""
    params: Dict = field(default_factory=dict)


@dataclass
class FailurePattern:
    """Detected repeating pattern."""

    tool: str
    error_type: str
    occurrences: int
    first_seen: str
    last_seen: str
    failure_rate: float
    mitigation: str


class FailureAnalyzer:
    """
    Collects failure events, detects patterns, writes reports.
    Pure observability — never blocks execution.
    """

    PATTERN_THRESHOLD = 3  # min failures to flag a pattern
    REPORT_EVERY = 50  # write report every N recorded failures

    def __init__(self):
        self._failures: List[FailureRecord] = []
        self._tool_counts: Dict[str, int] = defaultdict(int)
        self._tool_failures: Dict[str, int] = defaultdict(int)
        self._error_type_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._patterns: List[FailurePattern] = []
        self._total_recorded = 0

    # ── Record failure ────────────────────────────────────────────────────────
    def record_failure(
        self,
        tool: str,
        error_type: str,
        error_msg: str,
        intent: str = "",
        capability: str = "",
        params: Dict = None,
    ) -> None:
        rec = FailureRecord(
            ts=datetime.now(timezone.utc).isoformat(),
            tool=tool,
            error_type=error_type,
            error_msg=error_msg[:200],
            intent=intent,
            capability=capability,
            params=params or {},
        )
        self._failures.append(rec)
        if len(self._failures) > 1000:
            self._failures = self._failures[-500:]

        self._tool_failures[tool] += 1
        self._error_type_counts[tool][error_type] += 1
        self._total_recorded += 1

        # Periodic pattern detection
        if self._total_recorded % 10 == 0:
            self._detect_patterns()

        # Periodic report
        if self._total_recorded % self.REPORT_EVERY == 0:
            self._write_report()

    def record_tool_execution(self, tool: str, success: bool):
        """Record total executions for rate calculation."""
        self._tool_counts[tool] += 1
        if not success:
            self._tool_failures[tool] += 1

    # ── Pattern detection ─────────────────────────────────────────────────────
    def _detect_patterns(self):
        self._patterns.clear()
        for tool, type_counts in self._error_type_counts.items():
            total = self._tool_counts.get(tool, self._tool_failures.get(tool, 1))
            for error_type, count in type_counts.items():
                if count >= self.PATTERN_THRESHOLD:
                    # Find first/last occurrence
                    matching = [
                        f
                        for f in self._failures
                        if f.tool == tool and f.error_type == error_type
                    ]
                    rate = count / total if total else 1.0
                    self._patterns.append(
                        FailurePattern(
                            tool=tool,
                            error_type=error_type,
                            occurrences=count,
                            first_seen=matching[0].ts if matching else "",
                            last_seen=matching[-1].ts if matching else "",
                            failure_rate=round(rate, 3),
                            mitigation=self._suggest_mitigation(tool, error_type, rate),
                        )
                    )

        if self._patterns:
            logger.info(
                f"FailureAnalyzer: {len(self._patterns)} pattern(s) detected",
            )

    def _suggest_mitigation(self, tool: str, error_type: str, rate: float) -> str:
        if error_type == "missing_param":
            return f"Add stronger parameter extraction rule for '{tool}'."
        if error_type == "timeout":
            return f"Increase timeout or add caching for '{tool}'."
        if error_type == "verification":
            return f"Review verification checks for '{tool}' output format."
        if rate > 0.5:
            return f"Tool '{tool}' has >{rate:.0%} failure rate. Consider circuit-breaking."
        return f"Monitor '{tool}' — {error_type} failures trending up."

    # ── Report generation ─────────────────────────────────────────────────────
    def _write_report(self):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = os.path.join(_REPORT_DIR, f"failure_report_{ts}.json")
        try:
            report = {
                "generated_at": ts,
                "total_failures": self._total_recorded,
                "tool_failure_counts": dict(self._tool_failures),
                "patterns": [
                    {
                        "tool": p.tool,
                        "error_type": p.error_type,
                        "occurrences": p.occurrences,
                        "failure_rate": p.failure_rate,
                        "mitigation": p.mitigation,
                    }
                    for p in self._patterns
                ],
                "recent_failures": [
                    {
                        "ts": f.ts,
                        "tool": f.tool,
                        "type": f.error_type,
                        "msg": f.error_msg,
                    }
                    for f in self._failures[-20:]
                ],
            }
            with open(path, "w") as fh:
                json.dump(report, fh, indent=2)
            logger.info(f"FailureAnalyzer: report written to {path}")
        except Exception as e:
            logger.error(f"FailureAnalyzer: report write failed: {e}")

    # ── Query ─────────────────────────────────────────────────────────────────
    def get_patterns(self) -> List[Dict[str, Any]]:
        return [
            {
                "tool": p.tool,
                "error_type": p.error_type,
                "occurrences": p.occurrences,
                "failure_rate": p.failure_rate,
                "mitigation": p.mitigation,
            }
            for p in self._patterns
        ]

    def recent_failures(self, n: int = 10) -> List[Dict[str, Any]]:
        return [
            {"ts": f.ts, "tool": f.tool, "type": f.error_type, "msg": f.error_msg}
            for f in self._failures[-n:]
        ]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_failures": self._total_recorded,
            "tool_failure_counts": dict(self._tool_failures),
            "patterns_detected": len(self._patterns),
            "patterns": self.get_patterns(),
        }
