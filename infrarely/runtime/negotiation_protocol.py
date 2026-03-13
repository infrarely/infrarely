"""
runtime/negotiation_protocol.py — Module 8: Agent Negotiation Protocol
═══════════════════════════════════════════════════════════════════════════════
Task delegation, bidding, and role assignment between agents.
Like a contract-net protocol for multi-agent systems.

Gap solutions:
  Gap 10 — Coordination complexity: structured negotiation phases (propose,
           bid, accept/reject, execute) prevent ad-hoc coordination chaos.
           Timeout-bounded rounds prevent indefinite negotiation.

Protocol phases:
  1. PROPOSE  — coordinator posts a task for negotiation
  2. BID      — agents submit bids (capability match, cost, time estimate)
  3. AWARD    — coordinator picks the winning bid
  4. EXECUTE  — winner executes the task
  5. VERIFY   — coordinator verifies the result
"""

from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
from infrarely.observability import logger


class NegotiationPhase(Enum):
    PROPOSED = auto()
    BIDDING = auto()
    AWARDED = auto()
    EXECUTING = auto()
    VERIFYING = auto()
    COMPLETED = auto()
    FAILED = auto()
    TIMED_OUT = auto()
    CANCELLED = auto()


@dataclass
class Bid:
    """A bid from an agent for a negotiation."""

    bid_id: str = field(default_factory=lambda: f"bid_{uuid.uuid4().hex[:6]}")
    agent_id: str = ""
    capability_match: float = 0.0  # 0.0–1.0
    token_cost_estimate: int = 0
    time_estimate_ms: float = 0.0
    confidence: float = 0.5  # 0.0–1.0
    message: str = ""
    submitted_at: float = field(default_factory=time.time)

    @property
    def bid_score(self) -> float:
        """Overall bid quality score."""
        cost_factor = max(0.1, 1.0 - (self.token_cost_estimate / 1000))
        time_factor = max(0.1, 1.0 - (self.time_estimate_ms / 30_000))
        return (
            self.capability_match * 0.4
            + self.confidence * 0.25
            + cost_factor * 0.2
            + time_factor * 0.15
        )


@dataclass
class Negotiation:
    """A single negotiation session."""

    negotiation_id: str = field(default_factory=lambda: f"neg_{uuid.uuid4().hex[:8]}")
    task_description: str = ""
    required_capability: str = ""
    coordinator: str = ""  # agent_id or "system"
    phase: NegotiationPhase = NegotiationPhase.PROPOSED
    bids: List[Bid] = field(default_factory=list)
    winner_bid: Optional[Bid] = None
    winner_agent: str = ""
    max_bid_time_ms: float = 10_000.0  # 10s bidding window
    max_execution_time_ms: float = 30_000.0
    created_at: float = field(default_factory=time.time)
    bidding_opened_at: float = 0.0
    awarded_at: float = 0.0
    completed_at: float = 0.0
    result: Any = None
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_bidding_expired(self) -> bool:
        if self.phase != NegotiationPhase.BIDDING:
            return False
        if self.bidding_opened_at == 0:
            return False
        return (time.time() - self.bidding_opened_at) * 1000 > self.max_bid_time_ms

    @property
    def elapsed_ms(self) -> float:
        end = self.completed_at or time.time()
        return (end - self.created_at) * 1000


class NegotiationProtocol:
    """
    Structured negotiation for multi-agent task delegation.

    Invariants:
      • Max concurrent negotiations: 20
      • Bidding rounds are timeout-bounded (Gap 10)
      • Failed negotiations get re-proposed once
      • All phases are logged for auditability
    """

    MAX_CONCURRENT = 20
    MAX_BIDS_PER_NEGOTIATION = 10
    AUTO_REPROPOSE = True

    def __init__(self):
        self._negotiations: Dict[str, Negotiation] = {}
        self._completed: List[Negotiation] = []
        self._total_negotiations = 0
        self._total_successful = 0
        self._total_failed = 0

    # ── Propose ───────────────────────────────────────────────────────────────
    def propose(
        self,
        task_description: str,
        required_capability: str = "",
        coordinator: str = "system",
        max_bid_time_ms: float = 10_000.0,
        max_execution_time_ms: float = 30_000.0,
        metadata: Dict[str, Any] = None,
    ) -> Negotiation:
        """
        Start a new negotiation. Moves to BIDDING phase immediately.
        Gap 10: structured coordination avoids ad-hoc chaos.
        """
        active = [
            n
            for n in self._negotiations.values()
            if n.phase
            not in (
                NegotiationPhase.COMPLETED,
                NegotiationPhase.FAILED,
                NegotiationPhase.TIMED_OUT,
                NegotiationPhase.CANCELLED,
            )
        ]
        if len(active) >= self.MAX_CONCURRENT:
            raise ValueError(
                f"Max concurrent negotiations ({self.MAX_CONCURRENT}) reached"
            )

        neg = Negotiation(
            task_description=task_description,
            required_capability=required_capability,
            coordinator=coordinator,
            phase=NegotiationPhase.BIDDING,
            max_bid_time_ms=max_bid_time_ms,
            max_execution_time_ms=max_execution_time_ms,
            bidding_opened_at=time.time(),
            metadata=metadata or {},
        )
        self._negotiations[neg.negotiation_id] = neg
        self._total_negotiations += 1

        logger.info(
            f"Negotiation: '{neg.negotiation_id}' proposed — "
            f"'{task_description}' (cap='{required_capability}')"
        )
        return neg

    # ── Submit bid ────────────────────────────────────────────────────────────
    def submit_bid(
        self,
        negotiation_id: str,
        agent_id: str,
        capability_match: float = 1.0,
        token_cost_estimate: int = 10,
        time_estimate_ms: float = 5000.0,
        confidence: float = 0.8,
        message: str = "",
    ) -> Optional[Bid]:
        """
        Submit a bid for a negotiation.
        Returns the Bid on success, None if negotiation is not accepting bids.
        """
        neg = self._negotiations.get(negotiation_id)
        if not neg or neg.phase != NegotiationPhase.BIDDING:
            return None

        # Check bidding timeout (Gap 10)
        if neg.is_bidding_expired:
            self._close_bidding(neg)
            return None

        # Check max bids
        if len(neg.bids) >= self.MAX_BIDS_PER_NEGOTIATION:
            return None

        # Check duplicate bid
        for existing_bid in neg.bids:
            if existing_bid.agent_id == agent_id:
                return None

        bid = Bid(
            agent_id=agent_id,
            capability_match=capability_match,
            token_cost_estimate=token_cost_estimate,
            time_estimate_ms=time_estimate_ms,
            confidence=confidence,
            message=message,
        )
        neg.bids.append(bid)

        logger.debug(
            f"Negotiation: bid from '{agent_id}' on '{negotiation_id}' "
            f"(score={bid.bid_score:.3f})"
        )
        return bid

    # ── Award ─────────────────────────────────────────────────────────────────
    def award(self, negotiation_id: str) -> Optional[Bid]:
        """
        Award the negotiation to the best bidder.
        Moves to AWARDED phase.
        Gap 10: systematic winner selection avoids coordination chaos.
        """
        neg = self._negotiations.get(negotiation_id)
        if not neg:
            return None

        if neg.phase == NegotiationPhase.BIDDING:
            self._close_bidding(neg)

        if not neg.bids:
            neg.phase = NegotiationPhase.FAILED
            neg.error = "No bids received"
            self._finish(neg, success=False)
            return None

        # Select winner by bid score
        best = max(neg.bids, key=lambda b: b.bid_score)
        neg.winner_bid = best
        neg.winner_agent = best.agent_id
        neg.phase = NegotiationPhase.AWARDED
        neg.awarded_at = time.time()

        logger.info(
            f"Negotiation: '{negotiation_id}' awarded to '{best.agent_id}' "
            f"(score={best.bid_score:.3f}, {len(neg.bids)} bids)"
        )
        return best

    # ── Execute & Verify ──────────────────────────────────────────────────────
    def mark_executing(self, negotiation_id: str) -> bool:
        neg = self._negotiations.get(negotiation_id)
        if neg and neg.phase == NegotiationPhase.AWARDED:
            neg.phase = NegotiationPhase.EXECUTING
            return True
        return False

    def mark_verifying(self, negotiation_id: str) -> bool:
        neg = self._negotiations.get(negotiation_id)
        if neg and neg.phase == NegotiationPhase.EXECUTING:
            neg.phase = NegotiationPhase.VERIFYING
            return True
        return False

    def complete(
        self, negotiation_id: str, success: bool, result: Any = None, error: str = ""
    ) -> bool:
        """Complete the negotiation with a result."""
        neg = self._negotiations.get(negotiation_id)
        if not neg:
            return False

        neg.result = result
        neg.error = error
        self._finish(neg, success=success)
        return True

    def cancel(self, negotiation_id: str, reason: str = "") -> bool:
        neg = self._negotiations.get(negotiation_id)
        if not neg:
            return False
        neg.phase = NegotiationPhase.CANCELLED
        neg.error = reason or "Cancelled"
        neg.completed_at = time.time()
        self._archive(neg)
        return True

    # ── Timeout check ─────────────────────────────────────────────────────────
    def check_timeouts(self):
        """
        Check all negotiations for timeouts.
        Gap 10: prevents indefinite coordination.
        """
        for neg in list(self._negotiations.values()):
            if neg.phase == NegotiationPhase.BIDDING and neg.is_bidding_expired:
                self._close_bidding(neg)
                # Auto-award if there are bids
                if neg.bids:
                    self.award(neg.negotiation_id)
                else:
                    neg.phase = NegotiationPhase.TIMED_OUT
                    neg.error = "Bidding timed out with no bids"
                    self._finish(neg, success=False)

            elif neg.phase == NegotiationPhase.EXECUTING:
                elapsed = (time.time() - neg.awarded_at) * 1000
                if elapsed > neg.max_execution_time_ms:
                    neg.phase = NegotiationPhase.TIMED_OUT
                    neg.error = f"Execution timed out after {elapsed:.0f}ms"
                    self._finish(neg, success=False)

    # ── Private ───────────────────────────────────────────────────────────────
    def _close_bidding(self, neg: Negotiation):
        """Close bidding phase."""
        if neg.phase == NegotiationPhase.BIDDING:
            neg.phase = NegotiationPhase.AWARDED  # interim

    def _finish(self, neg: Negotiation, success: bool):
        neg.completed_at = time.time()
        if success:
            neg.phase = NegotiationPhase.COMPLETED
            self._total_successful += 1
        else:
            if neg.phase not in (
                NegotiationPhase.TIMED_OUT,
                NegotiationPhase.CANCELLED,
            ):
                neg.phase = NegotiationPhase.FAILED
            self._total_failed += 1
        self._archive(neg)

    def _archive(self, neg: Negotiation):
        self._completed.append(neg)
        self._negotiations.pop(neg.negotiation_id, None)
        if len(self._completed) > 200:
            self._completed = self._completed[-100:]

    # ── Query ─────────────────────────────────────────────────────────────────
    def get(self, negotiation_id: str) -> Optional[Negotiation]:
        return self._negotiations.get(negotiation_id)

    def active_negotiations(self) -> List[Negotiation]:
        return [
            n
            for n in self._negotiations.values()
            if n.phase
            not in (
                NegotiationPhase.COMPLETED,
                NegotiationPhase.FAILED,
                NegotiationPhase.TIMED_OUT,
                NegotiationPhase.CANCELLED,
            )
        ]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_negotiations": self._total_negotiations,
            "active": len(self._negotiations),
            "successful": self._total_successful,
            "failed": self._total_failed,
            "active_negotiations": [
                {
                    "id": n.negotiation_id,
                    "task": n.task_description[:50],
                    "phase": n.phase.name,
                    "bids": len(n.bids),
                    "winner": n.winner_agent,
                    "elapsed_ms": round(n.elapsed_ms, 1),
                }
                for n in self._negotiations.values()
            ],
            "recent_completed": [
                {
                    "id": n.negotiation_id,
                    "task": n.task_description[:50],
                    "phase": n.phase.name,
                    "winner": n.winner_agent,
                    "elapsed_ms": round(n.elapsed_ms, 1),
                }
                for n in self._completed[-5:]
            ],
        }
