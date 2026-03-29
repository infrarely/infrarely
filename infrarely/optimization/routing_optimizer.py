"""
adaptive/routing_optimizer.py — Module 1: Routing Optimizer
═══════════════════════════════════════════════════════════════════════════════
Improves router accuracy using execution feedback.

Tracks per-intent statistics:
  success_rate, average_latency, verification_failures

Produces a dynamic confidence multiplier:
  adjusted_conf = rule_confidence × success_rate × latency_factor

Safety:
  If success_rate < 0.5 → disable adaptive weighting, fall back to static rules.
  Never overrides routing decision — only re-ranks.
"""

from __future__ import annotations
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from infrarely.observability import logger


@dataclass
class IntentStats:
    """Running statistics for one intent."""

    intent: str
    total_executions: int = 0
    successes: int = 0
    failures: int = 0
    verification_fails: int = 0
    total_latency_ms: float = 0.0
    last_execution_ts: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.total_executions if self.total_executions else 1.0

    @property
    def avg_latency_ms(self) -> float:
        return (
            self.total_latency_ms / self.total_executions
            if self.total_executions
            else 0.0
        )

    @property
    def latency_factor(self) -> float:
        """Score 1.0 for fast, decays towards 0.7 for slow intents (>1000ms)."""
        avg = self.avg_latency_ms
        if avg <= 100:
            return 1.0
        if avg >= 1000:
            return 0.7
        return 1.0 - 0.3 * ((avg - 100) / 900)


class RoutingOptimizer:
    """
    Maintains per-intent routing statistics and produces adjusted confidence.

    Thread-safety: single-threaded (same as the rest of the agent).
    """

    def __init__(self):
        self._stats: Dict[str, IntentStats] = defaultdict(
            lambda: IntentStats(intent="")
        )
        self._enabled = True  # can be disabled by safety controller

    # ── Record execution outcome ──────────────────────────────────────────────
    def record(
        self,
        intent: str,
        success: bool,
        latency_ms: float,
        verification_passed: bool = True,
        capability: str = "",
    ) -> None:
        s = self._stats[intent]
        s.intent = intent
        s.total_executions += 1
        s.total_latency_ms += latency_ms
        s.last_execution_ts = time.time()
        if success:
            s.successes += 1
        else:
            s.failures += 1
        if not verification_passed:
            s.verification_fails += 1

    # ── Adjust confidence ─────────────────────────────────────────────────────
    def adjust_confidence(self, intent: str, base_confidence: float) -> float:
        """
        Returns adjusted confidence.
        If adaptive is disabled or insufficient data, returns base unchanged.
        """
        if not self._enabled:
            return base_confidence
        s = self._stats.get(intent)
        if s is None or s.total_executions < 3:
            return base_confidence  # insufficient data

        # Safety: disable adaptive if success rate is dangerously low
        if s.success_rate < 0.5:
            logger.warn(
                f"RoutingOptimizer: intent '{intent}' success_rate={s.success_rate:.2f} "
                f"— falling back to static confidence",
            )
            return base_confidence

        adjusted = base_confidence * s.success_rate * s.latency_factor
        return round(min(adjusted, 1.0), 4)

    # ── Query ─────────────────────────────────────────────────────────────────
    def get_stats(self, intent: str) -> Optional[IntentStats]:
        return self._stats.get(intent)

    def all_stats(self) -> Dict[str, Dict[str, Any]]:
        return {
            intent: {
                "executions": s.total_executions,
                "success_rate": round(s.success_rate, 3),
                "avg_latency_ms": round(s.avg_latency_ms, 1),
                "latency_factor": round(s.latency_factor, 3),
                "verification_fails": s.verification_fails,
            }
            for intent, s in self._stats.items()
        }

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        logger.info(f"RoutingOptimizer adaptive={'ON' if enabled else 'OFF'}")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled,
            "intents_tracked": len(self._stats),
            "stats": self.all_stats(),
        }
