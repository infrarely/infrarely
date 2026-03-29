"""
evolution/evolution_memory.py — Layer 7, Module 8
═══════════════════════════════════════════════════════════════════════════════
Learning history for the evolution system.

Stores every evolution attempt (whether it succeeded, failed, or was
rolled back) so the system can:
    • avoid repeating failed experiments
    • accelerate adoption of patterns that have worked before
    • track the cumulative impact of all changes over time
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


# ── Data types ────────────────────────────────────────────────────────────────


class ChangeOutcome(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    ROLLED_BACK = auto()
    INCONCLUSIVE = auto()


@dataclass
class ImpactMetric:
    """Before/after snapshot of a single KPI."""

    metric_name: str
    before: float
    after: float

    @property
    def delta(self) -> float:
        return self.after - self.before

    @property
    def delta_pct(self) -> float:
        if self.before == 0:
            return 0.0
        return (self.after - self.before) / abs(self.before) * 100


@dataclass
class EvolutionRecord:
    """One evolution event in the system's history."""

    record_id: str = field(default_factory=lambda: f"evo_{uuid.uuid4().hex[:8]}")
    proposal_id: str = ""
    component: str = ""
    change_type: str = ""  # "capability", "architecture", "experiment"
    description: str = ""
    reason: str = ""
    outcome: ChangeOutcome = ChangeOutcome.INCONCLUSIVE
    impact: List[ImpactMetric] = field(default_factory=list)
    experiment_id: Optional[str] = None
    verification_report_id: Optional[str] = None
    applied_at: float = field(default_factory=time.time)
    rolled_back_at: Optional[float] = None
    tags: List[str] = field(default_factory=list)


# ── Engine ────────────────────────────────────────────────────────────────────


class EvolutionMemory:
    """
    Persistent (in-memory) store of all evolution records.
    Provides queries to learn from past changes.
    """

    MAX_RECORDS = 5000

    def __init__(self):
        self._records: List[EvolutionRecord] = []
        self._index_by_component: Dict[str, List[str]] = {}  # component → [record_id]
        self._index_by_proposal: Dict[str, str] = {}  # proposal_id → record_id

    # ── Recording ─────────────────────────────────────────────────────────────

    def record(self, rec: EvolutionRecord) -> str:
        self._records.append(rec)
        self._index_by_component.setdefault(rec.component, []).append(rec.record_id)
        if rec.proposal_id:
            self._index_by_proposal[rec.proposal_id] = rec.record_id
        if len(self._records) > self.MAX_RECORDS:
            self._records = self._records[-self.MAX_RECORDS :]
        return rec.record_id

    def mark_outcome(self, record_id: str, outcome: ChangeOutcome) -> bool:
        rec = self.get(record_id)
        if rec is None:
            return False
        rec.outcome = outcome
        if outcome == ChangeOutcome.ROLLED_BACK:
            rec.rolled_back_at = time.time()
        return True

    def add_impact(self, record_id: str, metric: ImpactMetric) -> bool:
        rec = self.get(record_id)
        if rec is None:
            return False
        rec.impact.append(metric)
        return True

    # ── Query ─────────────────────────────────────────────────────────────────

    def get(self, record_id: str) -> Optional[EvolutionRecord]:
        for r in self._records:
            if r.record_id == record_id:
                return r
        return None

    def get_by_proposal(self, proposal_id: str) -> Optional[EvolutionRecord]:
        rid = self._index_by_proposal.get(proposal_id)
        if rid:
            return self.get(rid)
        return None

    def history_for_component(self, component: str) -> List[EvolutionRecord]:
        ids = self._index_by_component.get(component, [])
        return [r for r in self._records if r.record_id in ids]

    def failures_for_component(self, component: str) -> List[EvolutionRecord]:
        return [
            r
            for r in self.history_for_component(component)
            if r.outcome in (ChangeOutcome.FAILURE, ChangeOutcome.ROLLED_BACK)
        ]

    def successes_for_component(self, component: str) -> List[EvolutionRecord]:
        return [
            r
            for r in self.history_for_component(component)
            if r.outcome == ChangeOutcome.SUCCESS
        ]

    def was_tried_before(
        self, component: str, change_type: str, description: str
    ) -> bool:
        """Check if a similar change was tried before (to avoid repeats)."""
        for r in self.history_for_component(component):
            if (
                r.change_type == change_type
                and r.description == description
                and r.outcome in (ChangeOutcome.FAILURE, ChangeOutcome.ROLLED_BACK)
            ):
                return True
        return False

    def cumulative_impact(self) -> Dict[str, float]:
        """Sum of all impact deltas grouped by metric name."""
        totals: Dict[str, float] = {}
        for r in self._records:
            if r.outcome == ChangeOutcome.SUCCESS:
                for m in r.impact:
                    totals[m.metric_name] = totals.get(m.metric_name, 0.0) + m.delta
        return totals

    def success_rate(self) -> float:
        decided = [
            r
            for r in self._records
            if r.outcome
            in (ChangeOutcome.SUCCESS, ChangeOutcome.FAILURE, ChangeOutcome.ROLLED_BACK)
        ]
        if not decided:
            return 0.0
        wins = sum(1 for r in decided if r.outcome == ChangeOutcome.SUCCESS)
        return wins / len(decided)

    def recent(self, limit: int = 20) -> List[EvolutionRecord]:
        return self._records[-limit:]

    def all_records(self) -> List[EvolutionRecord]:
        return list(self._records)

    def snapshot(self) -> Dict[str, Any]:
        outcomes: Dict[str, int] = {}
        for r in self._records:
            key = r.outcome.name.lower()
            outcomes[key] = outcomes.get(key, 0) + 1
        return {
            "total_records": len(self._records),
            "outcomes": outcomes,
            "success_rate": round(self.success_rate(), 3),
            "cumulative_impact": self.cumulative_impact(),
            "components_touched": len(self._index_by_component),
        }
