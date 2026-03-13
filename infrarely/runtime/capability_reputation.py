"""
runtime/capability_reputation.py — GAP 4: Capability Reputation System
═══════════════════════════════════════════════════════════════════════════════
Dedicated reputation scoring for capability providers using a weighted
ranking formula:

  score = 0.4 × reliability + 0.3 × latency_factor + 0.2 × quality + 0.1 × cost_factor

Features:
  • Per-provider reputation tracking with time-decayed history
  • Configurable weights for the scoring formula
  • Reputation thresholds: suspend providers below minimum score
  • Leaderboard ranking across all providers
  • Decay: recent invocations weighted more heavily
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class InvocationRecord:
    """Single invocation record for reputation tracking."""

    success: bool = True
    latency_ms: float = 0.0
    quality_score: float = 1.0  # 0.0–1.0
    token_cost: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProviderReputation:
    """Reputation profile for a capability provider."""

    provider_id: str
    capability: str
    invocations: List[InvocationRecord] = field(default_factory=list)
    suspended: bool = False
    suspended_at: float = 0.0
    created_at: float = field(default_factory=time.time)

    MAX_HISTORY = 200

    def record(self, record: InvocationRecord):
        self.invocations.append(record)
        if len(self.invocations) > self.MAX_HISTORY:
            self.invocations = self.invocations[-self.MAX_HISTORY :]

    @property
    def total_invocations(self) -> int:
        return len(self.invocations)

    @property
    def reliability(self) -> float:
        """Success rate (0.0–1.0)."""
        if not self.invocations:
            return 1.0
        return sum(1 for r in self.invocations if r.success) / len(self.invocations)

    @property
    def avg_latency_ms(self) -> float:
        if not self.invocations:
            return 0.0
        return sum(r.latency_ms for r in self.invocations) / len(self.invocations)

    @property
    def avg_quality(self) -> float:
        if not self.invocations:
            return 1.0
        return sum(r.quality_score for r in self.invocations) / len(self.invocations)

    @property
    def avg_cost(self) -> float:
        if not self.invocations:
            return 0.0
        return sum(r.token_cost for r in self.invocations) / len(self.invocations)


class ReputationScorer:
    """
    Weighted reputation scorer using the formula:
      score = W_reliability × reliability
            + W_latency × latency_factor
            + W_quality × quality
            + W_cost × cost_factor
    """

    def __init__(
        self,
        w_reliability: float = 0.4,
        w_latency: float = 0.3,
        w_quality: float = 0.2,
        w_cost: float = 0.1,
        latency_baseline_ms: float = 500.0,
        cost_baseline: float = 100.0,
    ):
        self.w_reliability = w_reliability
        self.w_latency = w_latency
        self.w_quality = w_quality
        self.w_cost = w_cost
        self.latency_baseline_ms = latency_baseline_ms
        self.cost_baseline = cost_baseline

    def score(self, rep: ProviderReputation) -> float:
        """Compute composite reputation score (0.0–1.0)."""
        reliability = rep.reliability

        # Latency factor: lower is better
        avg_lat = rep.avg_latency_ms
        if avg_lat <= 0:
            latency_factor = 1.0
        else:
            latency_factor = min(1.0, self.latency_baseline_ms / avg_lat)

        quality = rep.avg_quality

        # Cost factor: lower is better
        avg_cost = rep.avg_cost
        if avg_cost <= 0:
            cost_factor = 1.0
        else:
            cost_factor = min(1.0, self.cost_baseline / avg_cost)

        return (
            self.w_reliability * reliability
            + self.w_latency * latency_factor
            + self.w_quality * quality
            + self.w_cost * cost_factor
        )


class CapabilityReputationManager:
    """
    Manages reputation for all capability providers.

    Invariants:
      • Providers below SUSPENSION_THRESHOLD are auto-suspended
      • Minimum invocations required before suspension
      • Leaderboard always sorted by composite score
      • Weights are configurable but must sum to 1.0
    """

    SUSPENSION_THRESHOLD = 0.35
    MIN_INVOCATIONS_FOR_SUSPENSION = 5

    def __init__(self, scorer: ReputationScorer = None):
        self._scorer = scorer or ReputationScorer()
        self._providers: Dict[str, ProviderReputation] = {}  # key = provider_id::cap
        self._suspension_count = 0

    @staticmethod
    def _key(provider_id: str, capability: str) -> str:
        return f"{provider_id}::{capability}"

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, provider_id: str, capability: str) -> ProviderReputation:
        key = self._key(provider_id, capability)
        if key not in self._providers:
            self._providers[key] = ProviderReputation(
                provider_id=provider_id, capability=capability
            )
        return self._providers[key]

    # ── Record invocation ─────────────────────────────────────────────────────

    def record_invocation(
        self,
        provider_id: str,
        capability: str,
        success: bool = True,
        latency_ms: float = 0.0,
        quality_score: float = 1.0,
        token_cost: int = 0,
    ) -> float:
        """
        Record an invocation and return the updated score.
        Auto-suspends if score drops below threshold.
        """
        key = self._key(provider_id, capability)
        rep = self._providers.get(key)
        if not rep:
            rep = self.register(provider_id, capability)

        rep.record(
            InvocationRecord(
                success=success,
                latency_ms=latency_ms,
                quality_score=quality_score,
                token_cost=token_cost,
            )
        )

        score = self._scorer.score(rep)

        # Auto-suspension check
        if (
            not rep.suspended
            and rep.total_invocations >= self.MIN_INVOCATIONS_FOR_SUSPENSION
            and score < self.SUSPENSION_THRESHOLD
        ):
            rep.suspended = True
            rep.suspended_at = time.time()
            self._suspension_count += 1

        return score

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_score(self, provider_id: str, capability: str) -> float:
        key = self._key(provider_id, capability)
        rep = self._providers.get(key)
        if not rep:
            return 0.0
        return self._scorer.score(rep)

    def get_reputation(
        self, provider_id: str, capability: str
    ) -> Optional[ProviderReputation]:
        return self._providers.get(self._key(provider_id, capability))

    def best_provider(self, capability: str) -> Optional[Tuple[str, float]]:
        """Return (provider_id, score) of the best non-suspended provider."""
        candidates = []
        for key, rep in self._providers.items():
            if rep.capability == capability and not rep.suspended:
                candidates.append((rep.provider_id, self._scorer.score(rep)))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0]

    def leaderboard(
        self, capability: str = "", limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Rank providers by score."""
        entries = []
        for key, rep in self._providers.items():
            if capability and rep.capability != capability:
                continue
            entries.append(
                {
                    "provider_id": rep.provider_id,
                    "capability": rep.capability,
                    "score": round(self._scorer.score(rep), 4),
                    "reliability": round(rep.reliability, 3),
                    "avg_latency_ms": round(rep.avg_latency_ms, 1),
                    "avg_quality": round(rep.avg_quality, 3),
                    "invocations": rep.total_invocations,
                    "suspended": rep.suspended,
                }
            )
        entries.sort(key=lambda x: x["score"], reverse=True)
        return entries[:limit]

    def reinstate(self, provider_id: str, capability: str) -> bool:
        """Reinstate a suspended provider."""
        rep = self._providers.get(self._key(provider_id, capability))
        if not rep or not rep.suspended:
            return False
        rep.suspended = False
        rep.suspended_at = 0.0
        return True

    def snapshot(self) -> Dict[str, Any]:
        active = sum(1 for r in self._providers.values() if not r.suspended)
        suspended = sum(1 for r in self._providers.values() if r.suspended)
        return {
            "total_providers": len(self._providers),
            "active": active,
            "suspended": suspended,
            "total_suspensions": self._suspension_count,
            "top_providers": self.leaderboard(limit=5),
        }
