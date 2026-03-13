"""
evolution/failure_intelligence.py — Layer 7, Module 2
═══════════════════════════════════════════════════════════════════════════════
Understands WHY failures happen.

Consumes data from:
    adaptive/failure_analyzer, adaptive/trace_intelligence,
    adaptive/quality_scorer

Produces:
    failure_patterns, failure_rate_by_capability, root_cause_suggestions
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


# ── Data types ────────────────────────────────────────────────────────────────


class FailureCategory(Enum):
    TOOL_FAILURE = auto()
    CAPABILITY_GRAPH_ERROR = auto()
    PARAMETER_INFERENCE_ERROR = auto()
    AGENT_CRASH = auto()
    RESOURCE_EXHAUSTION = auto()
    TIMEOUT = auto()
    VERIFICATION_FAILURE = auto()


@dataclass
class FailureEvent:
    """One failure occurrence ingested."""

    event_id: str = field(default_factory=lambda: f"fe_{uuid.uuid4().hex[:8]}")
    category: FailureCategory = FailureCategory.TOOL_FAILURE
    component: str = ""  # tool name, capability name, agent_id
    error_msg: str = ""
    capability: str = ""
    ts: float = field(default_factory=time.time)


@dataclass
class RootCause:
    """Root-cause analysis result."""

    cause_id: str = field(default_factory=lambda: f"rc_{uuid.uuid4().hex[:8]}")
    component: str = ""
    category: FailureCategory = FailureCategory.TOOL_FAILURE
    description: str = ""
    occurrences: int = 0
    failure_rate: float = 0.0
    suggestion: str = ""
    confidence: float = 0.0  # 0–1


@dataclass
class FailureCorrelation:
    """Two components whose failures correlate."""

    component_a: str
    component_b: str
    correlation: float  # 0–1
    shared_failures: int


# ── Engine ────────────────────────────────────────────────────────────────────


class FailureIntelligence:
    """
    Deep failure analysis: patterns, rates, root causes, correlations.
    Sits above the FailureAnalyzer and cross-references trace/quality data.
    """

    PATTERN_THRESHOLD = 3  # min events to flag a pattern
    CORRELATION_THRESHOLD = 0.5  # co-occurrence ratio to flag correlation
    MAX_EVENTS = 2000

    def __init__(self):
        self._events: List[FailureEvent] = []
        self._root_causes: List[RootCause] = []
        self._correlations: List[FailureCorrelation] = []
        # Counters
        self._component_total: Dict[str, int] = defaultdict(int)
        self._component_failures: Dict[str, int] = defaultdict(int)
        self._category_counts: Dict[FailureCategory, int] = defaultdict(int)
        self._capability_failures: Dict[str, int] = defaultdict(int)
        self._capability_total: Dict[str, int] = defaultdict(int)
        self._analysis_count = 0

    # ── Ingest ────────────────────────────────────────────────────────────────

    def record_event(self, event: FailureEvent) -> None:
        self._events.append(event)
        if len(self._events) > self.MAX_EVENTS:
            self._events = self._events[-self.MAX_EVENTS // 2 :]
        self._component_failures[event.component] += 1
        self._category_counts[event.category] += 1
        if event.capability:
            self._capability_failures[event.capability] += 1

    def record_success(self, component: str, capability: str = "") -> None:
        """Record a successful execution for rate calculation."""
        self._component_total[component] += 1
        if capability:
            self._capability_total[capability] += 1

    def ingest_from_analyzer(self, analyzer_snapshot: Dict[str, Any]) -> int:
        """
        Ingest failure data from adaptive/failure_analyzer.snapshot().
        Returns number of new events ingested.
        """
        patterns = analyzer_snapshot.get("patterns", [])
        count = 0
        for p in patterns:
            if not isinstance(p, dict):
                continue
            cat = self._classify_error_type(p.get("error_type", ""))
            ev = FailureEvent(
                category=cat,
                component=p.get("tool", "unknown"),
                error_msg=p.get("mitigation", ""),
                capability="",
            )
            self.record_event(ev)
            count += 1
        return count

    # ── Analysis ──────────────────────────────────────────────────────────────

    def analyse(self) -> Dict[str, Any]:
        """Run full failure analysis. Returns summary dict."""
        self._analysis_count += 1
        self._root_causes.clear()
        self._correlations.clear()

        self._detect_root_causes()
        self._detect_correlations()

        return {
            "analysis_id": self._analysis_count,
            "total_events": len(self._events),
            "root_causes": [self._rc_dict(rc) for rc in self._root_causes],
            "correlations": [self._corr_dict(c) for c in self._correlations],
            "failure_rate_by_capability": self.failure_rate_by_capability(),
            "category_breakdown": {
                cat.name: cnt for cat, cnt in self._category_counts.items()
            },
        }

    def failure_rate_by_capability(self) -> Dict[str, float]:
        """Per-capability failure rate."""
        rates = {}
        all_caps = set(self._capability_failures) | set(self._capability_total)
        for cap in all_caps:
            fails = self._capability_failures.get(cap, 0)
            total = self._capability_total.get(cap, 0) + fails
            rates[cap] = round(fails / max(1, total), 3)
        return rates

    # ── Root-cause detection ──────────────────────────────────────────────────

    def _detect_root_causes(self) -> None:
        # Group by (component, category)
        groups: Dict[tuple, List[FailureEvent]] = defaultdict(list)
        for ev in self._events:
            groups[(ev.component, ev.category)].append(ev)

        for (component, category), events in groups.items():
            if len(events) < self.PATTERN_THRESHOLD:
                continue
            total = self._component_total.get(component, 0) + len(events)
            rate = len(events) / max(1, total)
            self._root_causes.append(
                RootCause(
                    component=component,
                    category=category,
                    description=self._describe_root_cause(
                        component, category, len(events)
                    ),
                    occurrences=len(events),
                    failure_rate=round(rate, 3),
                    suggestion=self._suggest_fix(component, category, rate),
                    confidence=min(1.0, len(events) / 10),
                )
            )

    def _describe_root_cause(
        self, component: str, category: FailureCategory, count: int
    ) -> str:
        return (
            f"'{component}' has {count} {category.name.lower().replace('_', ' ')} "
            f"events"
        )

    def _suggest_fix(
        self, component: str, category: FailureCategory, rate: float
    ) -> str:
        suggestions = {
            FailureCategory.TOOL_FAILURE: f"Review tool '{component}' implementation or add circuit-breaker",
            FailureCategory.CAPABILITY_GRAPH_ERROR: f"Check capability graph for '{component}' — possible cycle or missing step",
            FailureCategory.PARAMETER_INFERENCE_ERROR: f"Improve parameter extraction rules for '{component}'",
            FailureCategory.AGENT_CRASH: f"Increase max_restarts or investigate root crash cause for '{component}'",
            FailureCategory.RESOURCE_EXHAUSTION: f"Increase token/resource quota for '{component}'",
            FailureCategory.TIMEOUT: f"Increase timeout or add caching for '{component}'",
            FailureCategory.VERIFICATION_FAILURE: f"Review verification checks for '{component}' — output format may have changed",
        }
        base = suggestions.get(category, f"Investigate '{component}'")
        if rate > 0.5:
            base += f" — CRITICAL: {rate:.0%} failure rate"
        return base

    # ── Correlation detection ─────────────────────────────────────────────────

    def _detect_correlations(self) -> None:
        """Find components that fail together (within ±2s windows)."""
        WINDOW = 2.0
        components = set(ev.component for ev in self._events)
        comp_list = sorted(components)

        for i, comp_a in enumerate(comp_list):
            events_a = [ev for ev in self._events if ev.component == comp_a]
            for comp_b in comp_list[i + 1 :]:
                events_b = [ev for ev in self._events if ev.component == comp_b]
                shared = 0
                for ea in events_a:
                    for eb in events_b:
                        if abs(ea.ts - eb.ts) <= WINDOW:
                            shared += 1
                            break
                total = min(len(events_a), len(events_b))
                if total == 0:
                    continue
                corr = shared / total
                if corr >= self.CORRELATION_THRESHOLD and shared >= 2:
                    self._correlations.append(
                        FailureCorrelation(
                            component_a=comp_a,
                            component_b=comp_b,
                            correlation=round(corr, 3),
                            shared_failures=shared,
                        )
                    )

    # ── Classify ──────────────────────────────────────────────────────────────

    @staticmethod
    def _classify_error_type(error_type: str) -> FailureCategory:
        mapping = {
            "tool_error": FailureCategory.TOOL_FAILURE,
            "missing_param": FailureCategory.PARAMETER_INFERENCE_ERROR,
            "timeout": FailureCategory.TIMEOUT,
            "verification": FailureCategory.VERIFICATION_FAILURE,
            "resource": FailureCategory.RESOURCE_EXHAUSTION,
            "crash": FailureCategory.AGENT_CRASH,
            "graph": FailureCategory.CAPABILITY_GRAPH_ERROR,
        }
        return mapping.get(error_type, FailureCategory.TOOL_FAILURE)

    # ── Query ─────────────────────────────────────────────────────────────────

    def root_causes(self) -> List[RootCause]:
        return list(self._root_causes)

    def correlations(self) -> List[FailureCorrelation]:
        return list(self._correlations)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_events": len(self._events),
            "analysis_count": self._analysis_count,
            "root_causes": len(self._root_causes),
            "correlations": len(self._correlations),
            "category_breakdown": {
                cat.name: cnt for cat, cnt in self._category_counts.items()
            },
            "failure_rate_by_capability": self.failure_rate_by_capability(),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _rc_dict(rc: RootCause) -> Dict[str, Any]:
        return {
            "cause_id": rc.cause_id,
            "component": rc.component,
            "category": rc.category.name,
            "description": rc.description,
            "occurrences": rc.occurrences,
            "failure_rate": rc.failure_rate,
            "suggestion": rc.suggestion,
            "confidence": rc.confidence,
        }

    @staticmethod
    def _corr_dict(c: FailureCorrelation) -> Dict[str, Any]:
        return {
            "component_a": c.component_a,
            "component_b": c.component_b,
            "correlation": c.correlation,
            "shared_failures": c.shared_failures,
        }
