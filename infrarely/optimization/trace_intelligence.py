"""
adaptive/trace_intelligence.py — Module 10: Trace Intelligence Engine
═══════════════════════════════════════════════════════════════════════════════
Analyses execution traces to produce analytics.

Analytics:
  • Capability heatmaps (usage frequency)
  • Tool reliability scores
  • Latency distributions
  • Error trends

Output stored in logs/analytics/.
"""

from __future__ import annotations
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import infrarely.core.app_config as config
from infrarely.observability import logger
from infrarely.runtime.paths import ANALYTICS_DIR

_ANALYTICS_DIR = str(ANALYTICS_DIR)
os.makedirs(_ANALYTICS_DIR, exist_ok=True)


@dataclass
class TraceRecord:
    """Lightweight record from one execution trace."""

    trace_id: str
    capability: str
    tools_used: List[str]
    tokens: int
    latency_ms: float
    outcome: str  # "success" | "partial" | "failed"
    errors: List[str] = field(default_factory=list)
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TraceIntelligenceEngine:
    """
    Analyses execution traces for patterns, reliability scores,
    and latency distributions.
    """

    ANALYTICS_EVERY = 25  # write analytics report every N traces

    def __init__(self):
        self._traces: List[TraceRecord] = []
        self._capability_usage: Counter = Counter()
        self._tool_usage: Counter = Counter()
        self._tool_successes: Counter = Counter()
        self._tool_failures: Counter = Counter()
        self._latency_buckets: Dict[str, List[float]] = defaultdict(list)
        self._error_trends: Dict[str, int] = defaultdict(int)

    # ── Record ────────────────────────────────────────────────────────────────
    def record_trace(
        self,
        trace_id: str,
        capability: str,
        tools_used: List[str],
        tokens: int,
        latency_ms: float,
        outcome: str,
        errors: List[str] = None,
    ) -> None:
        rec = TraceRecord(
            trace_id=trace_id,
            capability=capability or "single_tool",
            tools_used=tools_used,
            tokens=tokens,
            latency_ms=latency_ms,
            outcome=outcome,
            errors=errors or [],
        )
        self._traces.append(rec)
        if len(self._traces) > 1000:
            self._traces = self._traces[-500:]

        # Update counters
        if capability:
            self._capability_usage[capability] += 1
        for tool in tools_used:
            self._tool_usage[tool] += 1
            if outcome == "success":
                self._tool_successes[tool] += 1
            else:
                self._tool_failures[tool] += 1

        # Latency tracking
        bucket = capability or "single_tool"
        self._latency_buckets[bucket].append(latency_ms)
        if len(self._latency_buckets[bucket]) > 200:
            self._latency_buckets[bucket] = self._latency_buckets[bucket][-100:]

        # Error trends
        for err in errors or []:
            # Extract first word as category
            category = err.split(":")[0].strip() if ":" in err else err[:30]
            self._error_trends[category] += 1

        # Periodic analytics
        if len(self._traces) % self.ANALYTICS_EVERY == 0:
            self._write_analytics()

    # ── Analytics ─────────────────────────────────────────────────────────────
    def capability_heatmap(self) -> Dict[str, int]:
        """Usage frequency per capability."""
        return dict(self._capability_usage.most_common())

    def tool_reliability(self) -> Dict[str, Dict[str, Any]]:
        """Reliability score per tool."""
        result = {}
        for tool in self._tool_usage:
            total = self._tool_usage[tool]
            successes = self._tool_successes.get(tool, 0)
            failures = self._tool_failures.get(tool, 0)
            result[tool] = {
                "total_uses": total,
                "successes": successes,
                "failures": failures,
                "reliability": round(successes / total, 3) if total else 1.0,
            }
        return result

    def latency_distribution(self) -> Dict[str, Dict[str, float]]:
        """Latency stats per capability."""
        result = {}
        for bucket, latencies in self._latency_buckets.items():
            if not latencies:
                continue
            sorted_lat = sorted(latencies)
            n = len(sorted_lat)
            result[bucket] = {
                "count": n,
                "min_ms": round(sorted_lat[0], 1),
                "max_ms": round(sorted_lat[-1], 1),
                "avg_ms": round(sum(sorted_lat) / n, 1),
                "p50_ms": round(sorted_lat[n // 2], 1),
                "p95_ms": (
                    round(sorted_lat[int(n * 0.95)], 1)
                    if n >= 20
                    else round(sorted_lat[-1], 1)
                ),
            }
        return result

    def error_trends(self) -> Dict[str, int]:
        """Error frequency by category."""
        return dict(sorted(self._error_trends.items(), key=lambda x: -x[1]))

    # ── Report generation ─────────────────────────────────────────────────────
    def _write_analytics(self):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = os.path.join(_ANALYTICS_DIR, f"analytics_{ts}.json")
        try:
            report = {
                "generated_at": ts,
                "total_traces": len(self._traces),
                "capability_heatmap": self.capability_heatmap(),
                "tool_reliability": self.tool_reliability(),
                "latency_distribution": self.latency_distribution(),
                "error_trends": self.error_trends(),
            }
            with open(path, "w") as fh:
                json.dump(report, fh, indent=2)
            logger.info(f"TraceIntelligence: analytics written to {path}")
        except Exception as e:
            logger.error(f"TraceIntelligence: analytics write failed: {e}")

    # ── Query ─────────────────────────────────────────────────────────────────
    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_traces": len(self._traces),
            "capability_heatmap": self.capability_heatmap(),
            "tool_reliability": self.tool_reliability(),
            "latency_distribution": self.latency_distribution(),
            "error_trends": self.error_trends(),
        }
