"""
adaptive/capability_optimizer.py — Module 3: Capability Optimizer
═══════════════════════════════════════════════════════════════════════════════
Improves execution plan performance using historical metrics.

Tracks per-capability:
  execution_time, failure_rate, tool_latency, token_usage

Optimisation strategies:
  • Defer frequently-failing nodes until params verified
  • Reorder independent nodes by latency (fast first)
  • Skip nodes with persistent failures (with SKIP policy)

Optimisation frequency: every N executions (configurable).
Never mutates capability graph during execution.
"""

from __future__ import annotations
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from infrarely.observability import logger


@dataclass
class NodeMetrics:
    """Per-node execution metrics."""

    tool_name: str
    executions: int = 0
    successes: int = 0
    failures: int = 0
    total_ms: float = 0.0
    total_tokens: int = 0
    param_missing: int = 0
    last_failure: str = ""

    @property
    def success_rate(self) -> float:
        return self.successes / self.executions if self.executions else 1.0

    @property
    def failure_rate(self) -> float:
        return self.failures / self.executions if self.executions else 0.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.executions if self.executions else 0.0

    @property
    def avg_tokens(self) -> float:
        return self.total_tokens / self.executions if self.executions else 0.0


@dataclass
class CapabilityMetrics:
    """Per-capability execution metrics."""

    name: str
    total_executions: int = 0
    full_successes: int = 0
    partial_successes: int = 0
    aborted: int = 0
    total_ms: float = 0.0
    total_tokens: int = 0
    nodes: Dict[str, NodeMetrics] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return (
            self.full_successes / self.total_executions
            if self.total_executions
            else 1.0
        )

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.total_executions if self.total_executions else 0.0


class CapabilityOptimizer:
    """
    Collects capability execution metrics and suggests optimisations.
    Pure advisory — never mutates the capability graph itself.
    """

    OPTIMISE_EVERY = 100  # check for optimisations every N total executions
    DEFER_THRESHOLD = 0.40  # defer a node if failure_rate > this

    def __init__(self):
        self._capabilities: Dict[str, CapabilityMetrics] = {}
        self._total_executions = 0
        self._recommendations: List[Dict[str, Any]] = []

    # ── Record execution outcome ──────────────────────────────────────────────
    def record_capability(
        self,
        capability_name: str,
        success: bool,
        partial: bool,
        latency_ms: float,
        tokens: int,
        node_results: Dict[str, Dict[str, Any]] = None,
    ) -> None:
        if capability_name not in self._capabilities:
            self._capabilities[capability_name] = CapabilityMetrics(
                name=capability_name
            )

        cm = self._capabilities[capability_name]
        cm.total_executions += 1
        cm.total_ms += latency_ms
        cm.total_tokens += tokens

        if success:
            cm.full_successes += 1
        elif partial:
            cm.partial_successes += 1
        else:
            cm.aborted += 1

        # Record node-level metrics
        if node_results:
            for node_name, nr in node_results.items():
                if node_name not in cm.nodes:
                    cm.nodes[node_name] = NodeMetrics(
                        tool_name=nr.get("tool", node_name)
                    )
                nm = cm.nodes[node_name]
                nm.executions += 1
                nm.total_ms += nr.get("duration_ms", 0)
                nm.total_tokens += nr.get("tokens", 0)
                if nr.get("success", True):
                    nm.successes += 1
                else:
                    nm.failures += 1
                    nm.last_failure = nr.get("error", "")
                if nr.get("param_missing"):
                    nm.param_missing += 1

        self._total_executions += 1

        # Periodic optimisation check
        if self._total_executions % self.OPTIMISE_EVERY == 0:
            self._analyse_and_recommend()

    # ── Get optimisation advice ───────────────────────────────────────────────
    def get_node_advice(self, capability_name: str, node_name: str) -> Optional[str]:
        """
        Returns optimisation advice for a specific node.
        'defer' | 'skip' | None (no advice).
        """
        cm = self._capabilities.get(capability_name)
        if not cm or cm.total_executions < 5:
            return None
        nm = cm.nodes.get(node_name)
        if not nm or nm.executions < 3:
            return None

        if nm.failure_rate > self.DEFER_THRESHOLD:
            if nm.param_missing > nm.failures * 0.5:
                return "defer"  # often fails due to missing params
            return "skip"  # persistent failures
        return None

    def should_defer_node(self, capability_name: str, node_name: str) -> bool:
        """Check if a node should be deferred until params verified."""
        return self.get_node_advice(capability_name, node_name) == "defer"

    # ── Analytics ─────────────────────────────────────────────────────────────
    def _analyse_and_recommend(self):
        """Generate optimisation recommendations."""
        self._recommendations.clear()
        for cap_name, cm in self._capabilities.items():
            if cm.total_executions < 10:
                continue
            for node_name, nm in cm.nodes.items():
                if nm.failure_rate > self.DEFER_THRESHOLD:
                    self._recommendations.append(
                        {
                            "type": "high_failure_node",
                            "capability": cap_name,
                            "node": node_name,
                            "tool": nm.tool_name,
                            "failure_rate": round(nm.failure_rate, 3),
                            "suggestion": (
                                f"Node '{node_name}' in '{cap_name}' has {nm.failure_rate:.0%} "
                                f"failure rate. Consider adding parameter pre-validation."
                            ),
                        }
                    )
                if nm.avg_ms > 500 and nm.executions > 5:
                    self._recommendations.append(
                        {
                            "type": "slow_node",
                            "capability": cap_name,
                            "node": node_name,
                            "avg_ms": round(nm.avg_ms, 1),
                            "suggestion": (
                                f"Node '{node_name}' averages {nm.avg_ms:.0f}ms. "
                                f"Consider caching or deferring."
                            ),
                        }
                    )

        if self._recommendations:
            logger.info(
                f"CapabilityOptimizer: {len(self._recommendations)} recommendation(s)",
            )

    def get_recommendations(self) -> List[Dict[str, Any]]:
        return list(self._recommendations)

    # ── Execution Cost Model (Gap 7) ──────────────────────────────────────────
    def estimate_cost(self, capability_name: str) -> Dict[str, Any]:
        """
        Estimate execution cost for a capability based on historical data.
        Cost = latency_weight × avg_ms + token_weight × avg_tokens + failure_penalty.
        Returns cost dict or defaults for unknown capabilities.
        """
        LATENCY_WEIGHT = 0.3
        TOKEN_WEIGHT = 0.5
        FAILURE_PENALTY = 100.0

        cm = self._capabilities.get(capability_name)
        if not cm or cm.total_executions < 1:
            return {
                "capability": capability_name,
                "estimated_cost": 50.0,  # default mid-range cost
                "confidence": "low",
                "avg_ms": 0,
                "avg_tokens": 0,
                "failure_rate": 0,
                "executions": 0,
                "note": "Insufficient data — using default estimate",
            }

        avg_ms = cm.avg_ms
        avg_tokens = cm.total_tokens / cm.total_executions if cm.total_executions else 0
        failure_rate = 1 - cm.success_rate

        cost = (
            LATENCY_WEIGHT * (avg_ms / 100)
            + TOKEN_WEIGHT * (avg_tokens / 100)
            + FAILURE_PENALTY * failure_rate
        )

        confidence = (
            "high"
            if cm.total_executions >= 10
            else "medium" if cm.total_executions >= 3 else "low"
        )

        return {
            "capability": capability_name,
            "estimated_cost": round(cost, 2),
            "confidence": confidence,
            "avg_ms": round(avg_ms, 1),
            "avg_tokens": round(avg_tokens, 1),
            "failure_rate": round(failure_rate, 3),
            "executions": cm.total_executions,
            "node_costs": {
                nn: {
                    "avg_ms": round(nm.avg_ms, 1),
                    "avg_tokens": round(nm.avg_tokens, 1),
                    "failure_rate": round(nm.failure_rate, 3),
                }
                for nn, nm in cm.nodes.items()
            },
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_executions": self._total_executions,
            "capabilities": {
                name: {
                    "executions": cm.total_executions,
                    "success_rate": round(cm.success_rate, 3),
                    "avg_ms": round(cm.avg_ms, 1),
                    "total_tokens": cm.total_tokens,
                    "nodes": {
                        nn: {
                            "executions": nm.executions,
                            "success_rate": round(nm.success_rate, 3),
                            "failure_rate": round(nm.failure_rate, 3),
                            "avg_ms": round(nm.avg_ms, 1),
                            "avg_tokens": round(nm.avg_tokens, 1),
                        }
                        for nn, nm in cm.nodes.items()
                    },
                }
                for name, cm in self._capabilities.items()
            },
            "recommendations": self._recommendations,
        }
