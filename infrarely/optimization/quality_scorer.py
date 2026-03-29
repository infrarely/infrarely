"""
adaptive/quality_scorer.py — Module 8: Execution Quality Scorer
═══════════════════════════════════════════════════════════════════════════════
Measures and scores execution quality per capability and per session.

Metrics:
  completion_rate    — fraction of steps that completed
  verification_rate  — fraction that passed verification
  latency_score      — 1.0 if fast, decays for slow executions
  token_efficiency   — lower tokens per execution = higher score

Composite quality_score = weighted average of all metrics.
Capabilities below 0.6 trigger review.
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from infrarely.observability import logger


@dataclass
class QualityRecord:
    """Quality metrics for one execution."""

    capability: str
    steps_total: int
    steps_completed: int
    steps_failed: int
    verification_ok: int
    verification_fail: int
    latency_ms: float
    tokens_used: int

    @property
    def completion_rate(self) -> float:
        return self.steps_completed / self.steps_total if self.steps_total else 1.0

    @property
    def verification_rate(self) -> float:
        total = self.verification_ok + self.verification_fail
        return self.verification_ok / total if total else 1.0

    @property
    def latency_score(self) -> float:
        """1.0 for ≤100ms, decays to 0.5 at 2000ms+."""
        if self.latency_ms <= 100:
            return 1.0
        if self.latency_ms >= 2000:
            return 0.5
        return 1.0 - 0.5 * ((self.latency_ms - 100) / 1900)

    @property
    def token_score(self) -> float:
        """1.0 for 0 tokens, decays to 0.5 at 1000+ tokens."""
        if self.tokens_used == 0:
            return 1.0
        if self.tokens_used >= 1000:
            return 0.5
        return 1.0 - 0.5 * (self.tokens_used / 1000)

    @property
    def quality_score(self) -> float:
        """Weighted composite quality score."""
        return round(
            0.40 * self.completion_rate
            + 0.25 * self.verification_rate
            + 0.20 * self.latency_score
            + 0.15 * self.token_score,
            4,
        )


class ExecutionQualityScorer:
    """
    Tracks and scores execution quality per capability.
    Flags capabilities below quality threshold for review.
    """

    REVIEW_THRESHOLD = 0.60
    MIN_RECORDS = 3  # don't flag until enough data

    def __init__(self):
        self._records: Dict[str, List[QualityRecord]] = defaultdict(list)
        self._session_records: List[QualityRecord] = []
        self._flagged: List[str] = []

    # ── Record ────────────────────────────────────────────────────────────────
    def record(
        self,
        capability: str,
        steps_total: int,
        steps_completed: int,
        steps_failed: int,
        verification_ok: int,
        verification_fail: int,
        latency_ms: float,
        tokens_used: int,
    ) -> float:
        """Record one execution and return its quality score."""
        rec = QualityRecord(
            capability=capability,
            steps_total=steps_total,
            steps_completed=steps_completed,
            steps_failed=steps_failed,
            verification_ok=verification_ok,
            verification_fail=verification_fail,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
        )
        self._records[capability].append(rec)
        self._session_records.append(rec)

        # Keep bounded
        if len(self._records[capability]) > 200:
            self._records[capability] = self._records[capability][-100:]

        # Check for review
        self._check_flagging(capability)

        return rec.quality_score

    # ── Scoring ───────────────────────────────────────────────────────────────
    def get_score(self, capability: str) -> Optional[float]:
        """Get average quality score for a capability."""
        records = self._records.get(capability)
        if not records:
            return None
        scores = [r.quality_score for r in records]
        return round(sum(scores) / len(scores), 4)

    def get_session_score(self) -> float:
        """Get overall session quality score."""
        if not self._session_records:
            return 1.0
        scores = [r.quality_score for r in self._session_records]
        return round(sum(scores) / len(scores), 4)

    # ── Flagging ──────────────────────────────────────────────────────────────
    def _check_flagging(self, capability: str):
        records = self._records[capability]
        if len(records) < self.MIN_RECORDS:
            return
        avg = sum(r.quality_score for r in records[-10:]) / min(len(records), 10)
        if avg < self.REVIEW_THRESHOLD:
            if capability not in self._flagged:
                self._flagged.append(capability)
                logger.warn(
                    f"QualityScorer: '{capability}' flagged for review — "
                    f"avg quality={avg:.3f} (threshold={self.REVIEW_THRESHOLD})"
                )
        elif capability in self._flagged and avg >= self.REVIEW_THRESHOLD + 0.1:
            self._flagged.remove(capability)
            logger.info(f"QualityScorer: '{capability}' recovered above threshold")

    def get_flagged(self) -> List[str]:
        return list(self._flagged)

    # ── Query ─────────────────────────────────────────────────────────────────
    def snapshot(self) -> Dict[str, Any]:
        return {
            "session_score": self.get_session_score(),
            "session_executions": len(self._session_records),
            "flagged_capabilities": self._flagged,
            "capabilities": {
                cap: {
                    "executions": len(recs),
                    "avg_quality": self.get_score(cap),
                    "last_quality": recs[-1].quality_score if recs else None,
                    "avg_completion": round(
                        sum(r.completion_rate for r in recs) / len(recs), 3
                    ),
                    "avg_latency_ms": round(
                        sum(r.latency_ms for r in recs) / len(recs), 1
                    ),
                }
                for cap, recs in self._records.items()
            },
        }
