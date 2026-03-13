"""
evolution/performance_intelligence.py — Layer 7, Module 1
═══════════════════════════════════════════════════════════════════════════════
Understands how well the system is performing.

Consumes data from Layer-5/6:
    agent_monitoring, trace_intelligence, token_optimizer

Produces:
    performance trends, bottleneck detection, degradation alerts
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


# ── Data types ────────────────────────────────────────────────────────────────


class TrendDirection(Enum):
    IMPROVING = auto()
    STABLE = auto()
    DEGRADING = auto()


class AlertSeverity(Enum):
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()


@dataclass
class PerformanceMetrics:
    """One snapshot of system-wide performance."""

    ts: float = field(default_factory=time.time)
    agent_latency_avg: float = 0.0
    task_success_rate: float = 1.0
    scheduler_queue_depth: int = 0
    bus_message_rate: int = 0
    tool_reliability: Dict[str, float] = field(default_factory=dict)
    token_usage: int = 0
    health_score: float = 100.0


@dataclass
class PerformanceTrend:
    """Detected trend in a metric."""

    metric: str
    direction: TrendDirection
    current_value: float
    previous_value: float
    change_pct: float
    window_size: int
    detected_at: float = field(default_factory=time.time)


@dataclass
class Bottleneck:
    """Detected performance bottleneck."""

    component: str
    metric: str
    severity: AlertSeverity
    current_value: float
    threshold: float
    suggestion: str
    detected_at: float = field(default_factory=time.time)


@dataclass
class DegradationAlert:
    """Alert for performance degradation."""

    alert_id: str = field(default_factory=lambda: f"alert_{uuid.uuid4().hex[:8]}")
    component: str = ""
    metric: str = ""
    severity: AlertSeverity = AlertSeverity.WARNING
    message: str = ""
    detected_at: float = field(default_factory=time.time)
    acknowledged: bool = False


# ── Engine ────────────────────────────────────────────────────────────────────


class PerformanceIntelligence:
    """
    Analyses system metrics to detect trends, bottlenecks, and degradation.

    Works entirely from in-memory snapshots; never touches the runtime.
    """

    # Thresholds
    LATENCY_DEGRADATION_PCT = 0.40  # 40 % increase → alert
    SUCCESS_RATE_FLOOR = 0.85  # < 85 % → bottleneck
    QUEUE_DEPTH_CEILING = 150  # queue > 150 → bottleneck
    BUS_RATE_CEILING = 1500  # messages/sec > 1500 → congestion
    TOOL_RELIABILITY_FLOOR = 0.70  # per-tool reliability < 70 % → bottleneck
    HEALTH_SCORE_FLOOR = 40.0  # overall health < 40 → critical
    TREND_WINDOW = 5  # snapshots to compute trend

    def __init__(self):
        self._history: deque[PerformanceMetrics] = deque(maxlen=200)
        self._trends: List[PerformanceTrend] = []
        self._bottlenecks: List[Bottleneck] = []
        self._alerts: List[DegradationAlert] = []
        self._analysis_count = 0

    # ── Ingest ────────────────────────────────────────────────────────────────

    def record_snapshot(self, metrics: PerformanceMetrics) -> None:
        """Record a performance snapshot from Layer-6 data."""
        self._history.append(metrics)

    def ingest_from_monitoring(
        self,
        monitoring_snapshot: Dict[str, Any],
        scheduler_snapshot: Dict[str, Any] = None,
        bus_snapshot: Dict[str, Any] = None,
        trace_snapshot: Dict[str, Any] = None,
    ) -> PerformanceMetrics:
        """
        Build a PerformanceMetrics from existing Layer-5/6 snapshots.
        Accepts the dict returned by each module's .snapshot() method.
        """
        m = PerformanceMetrics()
        m.health_score = monitoring_snapshot.get("health_score", 100.0)

        # Derive avg latency from top agents
        top = monitoring_snapshot.get("top_agents", [])
        if top:
            latencies = [a.get("avg_latency_ms", 0) for a in top if isinstance(a, dict)]
            m.agent_latency_avg = sum(latencies) / len(latencies) if latencies else 0.0

        # Success rate
        global_tasks = monitoring_snapshot.get("global_tasks", 0)
        global_failures = monitoring_snapshot.get("global_failures", 0)
        m.task_success_rate = 1 - global_failures / max(1, global_tasks)

        if scheduler_snapshot:
            m.scheduler_queue_depth = scheduler_snapshot.get("queue_depth", 0)
        if bus_snapshot:
            m.bus_message_rate = bus_snapshot.get("total_messages", 0)
        if trace_snapshot:
            tr = trace_snapshot.get("tool_reliability", {})
            m.tool_reliability = {
                t: info.get("reliability", 1.0)
                for t, info in tr.items()
                if isinstance(info, dict)
            }
        m.token_usage = monitoring_snapshot.get("global_tasks", 0) * 10  # estimate

        self.record_snapshot(m)
        return m

    # ── Analysis ──────────────────────────────────────────────────────────────

    def analyse(self) -> Dict[str, Any]:
        """
        Run full analysis: trends, bottlenecks, alerts.
        Returns a summary dict.
        """
        self._analysis_count += 1
        self._trends.clear()
        self._bottlenecks.clear()

        self._compute_trends()
        self._detect_bottlenecks()
        self._check_degradation()

        return {
            "analysis_id": self._analysis_count,
            "snapshots": len(self._history),
            "trends": [self._trend_dict(t) for t in self._trends],
            "bottlenecks": [self._bottleneck_dict(b) for b in self._bottlenecks],
            "alerts": [self._alert_dict(a) for a in self._alerts if not a.acknowledged],
        }

    # ── Trend detection ───────────────────────────────────────────────────────

    def _compute_trends(self) -> None:
        if len(self._history) < self.TREND_WINDOW * 2:
            return
        history = list(self._history)
        recent = history[-self.TREND_WINDOW :]
        prior = history[-(self.TREND_WINDOW * 2) : -self.TREND_WINDOW]

        self._add_trend("agent_latency_avg", recent, prior, higher_is_worse=True)
        self._add_trend("task_success_rate", recent, prior, higher_is_worse=False)
        self._add_trend("scheduler_queue_depth", recent, prior, higher_is_worse=True)
        self._add_trend("health_score", recent, prior, higher_is_worse=False)

    def _add_trend(
        self,
        metric: str,
        recent: List[PerformanceMetrics],
        prior: List[PerformanceMetrics],
        higher_is_worse: bool,
    ) -> None:
        r_avg = sum(getattr(m, metric, 0) for m in recent) / max(1, len(recent))
        p_avg = sum(getattr(m, metric, 0) for m in prior) / max(1, len(prior))
        if p_avg == 0:
            return
        change_pct = (r_avg - p_avg) / abs(p_avg)

        if abs(change_pct) < 0.05:
            direction = TrendDirection.STABLE
        elif (change_pct > 0) == higher_is_worse:
            direction = TrendDirection.DEGRADING
        else:
            direction = TrendDirection.IMPROVING

        self._trends.append(
            PerformanceTrend(
                metric=metric,
                direction=direction,
                current_value=round(r_avg, 3),
                previous_value=round(p_avg, 3),
                change_pct=round(change_pct * 100, 1),
                window_size=self.TREND_WINDOW,
            )
        )

    # ── Bottleneck detection ──────────────────────────────────────────────────

    def _detect_bottlenecks(self) -> None:
        if not self._history:
            return
        latest = self._history[-1]

        if latest.task_success_rate < self.SUCCESS_RATE_FLOOR:
            self._bottlenecks.append(
                Bottleneck(
                    component="task_pipeline",
                    metric="task_success_rate",
                    severity=AlertSeverity.WARNING,
                    current_value=latest.task_success_rate,
                    threshold=self.SUCCESS_RATE_FLOOR,
                    suggestion="Investigate failing tools or capability steps",
                )
            )

        if latest.scheduler_queue_depth > self.QUEUE_DEPTH_CEILING:
            self._bottlenecks.append(
                Bottleneck(
                    component="scheduler",
                    metric="scheduler_queue_depth",
                    severity=AlertSeverity.WARNING,
                    current_value=latest.scheduler_queue_depth,
                    threshold=self.QUEUE_DEPTH_CEILING,
                    suggestion="Increase agent pool or adjust task priorities",
                )
            )

        if latest.bus_message_rate > self.BUS_RATE_CEILING:
            self._bottlenecks.append(
                Bottleneck(
                    component="message_bus",
                    metric="bus_message_rate",
                    severity=AlertSeverity.WARNING,
                    current_value=latest.bus_message_rate,
                    threshold=self.BUS_RATE_CEILING,
                    suggestion="Increase bus capacity or reduce broadcast frequency",
                )
            )

        if latest.health_score < self.HEALTH_SCORE_FLOOR:
            self._bottlenecks.append(
                Bottleneck(
                    component="system",
                    metric="health_score",
                    severity=AlertSeverity.CRITICAL,
                    current_value=latest.health_score,
                    threshold=self.HEALTH_SCORE_FLOOR,
                    suggestion="Critical health — review agent monitoring dashboard",
                )
            )

        for tool, rel in latest.tool_reliability.items():
            if rel < self.TOOL_RELIABILITY_FLOOR:
                self._bottlenecks.append(
                    Bottleneck(
                        component=f"tool:{tool}",
                        metric="tool_reliability",
                        severity=AlertSeverity.WARNING,
                        current_value=rel,
                        threshold=self.TOOL_RELIABILITY_FLOOR,
                        suggestion=f"Tool '{tool}' reliability low — consider replacement or circuit-breaking",
                    )
                )

    # ── Degradation alerts ────────────────────────────────────────────────────

    def _check_degradation(self) -> None:
        for trend in self._trends:
            if trend.direction == TrendDirection.DEGRADING:
                severity = (
                    AlertSeverity.CRITICAL
                    if abs(trend.change_pct) > 50
                    else AlertSeverity.WARNING
                )
                self._alerts.append(
                    DegradationAlert(
                        component=trend.metric,
                        metric=trend.metric,
                        severity=severity,
                        message=(
                            f"{trend.metric} degraded by {abs(trend.change_pct):.1f}% "
                            f"({trend.previous_value} → {trend.current_value})"
                        ),
                    )
                )

    def acknowledge_alert(self, alert_id: str) -> bool:
        for a in self._alerts:
            if a.alert_id == alert_id:
                a.acknowledged = True
                return True
        return False

    # ── Query ─────────────────────────────────────────────────────────────────

    def active_alerts(self) -> List[DegradationAlert]:
        return [a for a in self._alerts if not a.acknowledged]

    def latest_metrics(self) -> Optional[PerformanceMetrics]:
        return self._history[-1] if self._history else None

    def trends(self) -> List[PerformanceTrend]:
        return list(self._trends)

    def bottlenecks(self) -> List[Bottleneck]:
        return list(self._bottlenecks)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "snapshots_recorded": len(self._history),
            "analysis_count": self._analysis_count,
            "active_trends": len(self._trends),
            "active_bottlenecks": len(self._bottlenecks),
            "active_alerts": len(self.active_alerts()),
            "latest_health": (
                round(self._history[-1].health_score, 1) if self._history else None
            ),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _trend_dict(t: PerformanceTrend) -> Dict[str, Any]:
        return {
            "metric": t.metric,
            "direction": t.direction.name,
            "current": t.current_value,
            "previous": t.previous_value,
            "change_pct": t.change_pct,
        }

    @staticmethod
    def _bottleneck_dict(b: Bottleneck) -> Dict[str, Any]:
        return {
            "component": b.component,
            "metric": b.metric,
            "severity": b.severity.name,
            "current": b.current_value,
            "threshold": b.threshold,
            "suggestion": b.suggestion,
        }

    @staticmethod
    def _alert_dict(a: DegradationAlert) -> Dict[str, Any]:
        return {
            "alert_id": a.alert_id,
            "component": a.component,
            "severity": a.severity.name,
            "message": a.message,
        }
