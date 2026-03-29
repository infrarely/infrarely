"""
agent/execution_trace.py  — Gap 10: Execution Trace System
═══════════════════════════════════════════════════════════════════════════════
Every agent request produces a structured trace file.

Trace file: logs/traces/trace_<run_id>.json
Contains:
  run_id, session_id, student_id, query
  steps[]  — each tool/capability step with contract, duration, tokens
  llm_calls — count and total tokens
  outcome  — final contract, total duration, total tokens
  errors[] — any errors encountered

The trace is written atomically at the end of the request.
It never affects execution — pure observation.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import List

import infrarely.core.app_config as config
from infrarely.observability import logger
from infrarely.runtime.paths import TRACE_DIR

_TRACE_DIR = str(TRACE_DIR)
os.makedirs(_TRACE_DIR, exist_ok=True)

# ── Trace retention policy ────────────────────────────────────────────────────
MAX_TRACE_FILES = getattr(config, "MAX_TRACE_FILES", 200)
TRACE_CLEANUP_BATCH = 50  # remove this many when limit exceeded


def enforce_trace_retention():
    """
    Delete oldest trace files when directory exceeds MAX_TRACE_FILES.
    Called after each trace save to prevent trace explosion (Gap 3).
    """
    try:
        files = [
            f
            for f in os.listdir(_TRACE_DIR)
            if f.startswith("trace_") and f.endswith(".json")
        ]
        if len(files) <= MAX_TRACE_FILES:
            return
        # Sort by modification time (oldest first)
        full_paths = [os.path.join(_TRACE_DIR, f) for f in files]
        full_paths.sort(key=lambda p: os.path.getmtime(p))
        to_remove = len(files) - MAX_TRACE_FILES + TRACE_CLEANUP_BATCH
        removed = 0
        for p in full_paths[:to_remove]:
            try:
                os.remove(p)
                removed += 1
            except OSError:
                pass
        if removed:
            logger.info(
                f"TraceRetention: cleaned {removed} old traces "
                f"(limit={MAX_TRACE_FILES}, was={len(files)})"
            )
    except Exception as e:
        logger.error(f"TraceRetention: cleanup failed: {e}")


@dataclass
class TraceStep:
    step: str
    tool: str
    contract: str
    duration_ms: float = 0.0
    tokens: int = 0
    error: str = ""
    skipped: bool = False


@dataclass
class ExecutionTrace:
    run_id: str
    session_id: str
    student_id: str
    query: str
    ts_start: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    steps: List[TraceStep] = field(default_factory=list)
    llm_calls: int = 0
    total_tokens: int = 0
    total_ms: float = 0.0
    outcome: str = "pending"  # success | partial | failed
    final_contract: str = ""
    errors: List[str] = field(default_factory=list)
    capability: str = ""  # set if multi-step workflow

    def add_step(
        self,
        step: str,
        tool: str,
        contract: str,
        duration_ms: float = 0,
        tokens: int = 0,
        error: str = "",
        skipped: bool = False,
    ):
        self.steps.append(
            TraceStep(
                step=step,
                tool=tool,
                contract=contract,
                duration_ms=duration_ms,
                tokens=tokens,
                error=error,
                skipped=skipped,
            )
        )
        self.total_tokens += tokens
        if error:
            self.errors.append(f"{step}: {error}")

    def finalise(self, outcome: str, final_contract: str, total_ms: float):
        self.outcome = outcome
        self.final_contract = final_contract
        self.total_ms = total_ms

    def save(self):
        """Write trace atomically to logs/traces/."""
        if not getattr(config, "ENABLE_EXECUTION_TRACE", True):
            return
        path = os.path.join(_TRACE_DIR, f"trace_{self.run_id}.json")
        try:
            payload = {
                "run_id": self.run_id,
                "session_id": self.session_id,
                "student_id": self.student_id,
                "query": self.query,
                "ts_start": self.ts_start,
                "capability": self.capability,
                "steps": [asdict(s) for s in self.steps],
                "llm_calls": self.llm_calls,
                "total_tokens": self.total_tokens,
                "total_ms": round(self.total_ms, 2),
                "outcome": self.outcome,
                "final_contract": self.final_contract,
                "errors": self.errors,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            logger.trace_log(
                self.run_id,
                "saved",
                outcome=self.outcome,
                steps=len(self.steps),
                tokens=self.total_tokens,
                ms=round(self.total_ms, 1),
            )
            # Enforce retention policy after each save (Gap 3)
            enforce_trace_retention()
        except Exception as e:
            logger.error(f"ExecutionTrace.save failed: {e}")

    @staticmethod
    def new(query: str, student_id: str, session_id: str = "") -> "ExecutionTrace":
        return ExecutionTrace(
            run_id=uuid.uuid4().hex[:12],
            session_id=session_id or "no-session",
            student_id=student_id,
            query=query,
        )

    def render(self) -> str:
        """
        Render the execution trace as formatted ASCII art.

        Example output:
        ┌─ Run: support-bot [42ms] ──────────────────────────
        │  Input: "What's the status of order #1042?"
        │
        ├─ [1] Intent Classification [0ms]
        │     matched: get_order_status (score: 0.94)
        │     rejected: search_web (score: 0.12)
        │
        ├─ [2] Parameter Extraction [5ms]
        │     order_id = "1042" ✓ (type: str)
        │
        ├─ [3] Tool Execution: get_order_status [38ms]
        │     status: "shipped"
        │     tracking_number: "UPS-9283"
        │
        └─ Result: 1 tool call · 0 LLM calls · 42ms
        """
        lines: List[str] = []

        # ── Header ────────────────────────────────────────────────────────────
        header_line = f"┌─ Run: {self.run_id} [{self.total_ms:.0f}ms]"
        padding = "─" * max(1, 50 - len(header_line))
        lines.append(f"{header_line} {padding}")

        # ── Input ────────────────────────────────────────────────────────────
        query_short = self.query[:48] if len(self.query) > 48 else self.query
        lines.append(f'│  Input: "{query_short}"')

        if self.steps or self.llm_calls > 0 or self.errors:
            lines.append("│")

        # ── Steps ────────────────────────────────────────────────────────────
        for i, step in enumerate(self.steps):
            is_last_step = (
                i == len(self.steps) - 1 and not self.llm_calls and not self.errors
            )

            prefix = "└─" if is_last_step else "├─"
            step_title = (
                f"[{i + 1}] {step.step}: {step.tool} [{step.duration_ms:.0f}ms]"
            )
            lines.append(f"{prefix} {step_title}")

            # Step details
            if step.contract:
                lines.append(f"│     contract: {step.contract}")
            if step.tokens > 0:
                lines.append(f"│     tokens: {step.tokens}")
            if step.error:
                lines.append(f"│     error: {step.error} ✗")
            if step.skipped:
                lines.append(f"│     status: skipped")

        # ── LLM calls ────────────────────────────────────────────────────────
        if self.llm_calls > 0:
            is_last = not self.errors
            prefix = "└─" if is_last else "├─"
            lines.append(f"{prefix} LLM Calls: {self.llm_calls} calls")

        # ── Errors ───────────────────────────────────────────────────────────
        if self.errors:
            for i, error in enumerate(self.errors):
                is_last = i == len(self.errors) - 1
                prefix = "└─" if is_last else "├─"
                lines.append(f"{prefix} Error: {error}")

        # ── Footer ───────────────────────────────────────────────────────────
        tool_count = len([s for s in self.steps if not s.skipped])
        outcome_str = self.outcome.upper() if self.outcome else "UNKNOWN"
        status_icon = "✓" if self.outcome == "success" else "✗"

        footer = (
            f"│ Result: {status_icon} {outcome_str} · "
            f"{tool_count} tool{'s' if tool_count != 1 else ''} · "
            f"{self.llm_calls} LLM call{'s' if self.llm_calls != 1 else ''} · "
            f"{self.total_tokens} tokens · {self.total_ms:.0f}ms"
        )

        if not self.errors and not self.steps:
            lines.append("└─ (no steps recorded)")
        else:
            lines.append(footer)

        return "\n".join(lines)

    def __str__(self) -> str:
        """Print the trace as formatted ASCII art."""
        return self.render()

    def __repr__(self) -> str:
        """Print the trace as formatted ASCII art."""
        return self.render()
