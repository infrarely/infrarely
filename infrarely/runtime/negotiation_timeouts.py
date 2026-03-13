"""
runtime/negotiation_timeouts.py — GAP 8
═══════════════════════════════════════════════════════════════════════════════
Timeout management for the auction / negotiation subsystem.

Ensures that no bid or auction can hang forever:

  • BidTimer          — per-bid deadline; auto-expires
  • AuctionTimer      — per-auction closing timer
  • TimeoutPolicy     — configurable defaults
  • NegotiationTimeoutManager — orchestrator with snapshot()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set


# ─── Enums ────────────────────────────────────────────────────────────────────
class BidStatus(Enum):
    PENDING = auto()
    ACCEPTED = auto()
    EXPIRED = auto()
    REJECTED = auto()


class AuctionPhase(Enum):
    OPEN = auto()
    CLOSED = auto()
    EXPIRED = auto()
    AWARDED = auto()


# ─── Data ─────────────────────────────────────────────────────────────────────
@dataclass
class BidTimer:
    bid_id: str
    auction_id: str
    bidder: str
    status: BidStatus = BidStatus.PENDING
    submitted_at: float = field(default_factory=time.time)
    timeout_ms: float = 10_000.0  # 10 s default
    resolved_at: Optional[float] = None

    @property
    def deadline(self) -> float:
        return self.submitted_at + self.timeout_ms / 1000.0

    @property
    def is_expired(self) -> bool:
        return self.status == BidStatus.PENDING and time.time() > self.deadline


@dataclass
class AuctionTimer:
    auction_id: str
    capability: str
    phase: AuctionPhase = AuctionPhase.OPEN
    opened_at: float = field(default_factory=time.time)
    close_after_ms: float = 15_000.0  # 15 s default
    bids: Dict[str, BidTimer] = field(default_factory=dict)
    winner: Optional[str] = None
    fallback: Optional[str] = None  # default agent if no bids

    @property
    def close_deadline(self) -> float:
        return self.opened_at + self.close_after_ms / 1000.0


@dataclass
class TimeoutPolicy:
    bid_timeout_ms: float = 10_000.0
    auction_close_ms: float = 15_000.0
    fallback_agent: Optional[str] = None


# ─── Manager ─────────────────────────────────────────────────────────────────
class NegotiationTimeoutManager:
    """
    Enforces bid and auction timeouts with optional default fallback.
    """

    def __init__(self, policy: Optional[TimeoutPolicy] = None):
        self.policy = policy or TimeoutPolicy()
        self._auctions: Dict[str, AuctionTimer] = {}
        self._total_expired_bids = 0
        self._total_expired_auctions = 0

    # ── auction lifecycle ─────────────────────────────────────────────────
    def open_auction(
        self,
        auction_id: str,
        capability: str,
        close_after_ms: float = 0,
        fallback: Optional[str] = None,
    ) -> AuctionTimer:
        if auction_id in self._auctions:
            raise ValueError(f"Auction '{auction_id}' already exists")
        t = close_after_ms if close_after_ms > 0 else self.policy.auction_close_ms
        fb = fallback or self.policy.fallback_agent
        a = AuctionTimer(
            auction_id=auction_id, capability=capability, close_after_ms=t, fallback=fb
        )
        self._auctions[auction_id] = a
        return a

    def submit_bid(
        self, auction_id: str, bid_id: str, bidder: str, timeout_ms: float = 0
    ) -> BidTimer:
        auction = self._auctions.get(auction_id)
        if not auction:
            raise KeyError(f"Auction '{auction_id}' not found")
        if auction.phase != AuctionPhase.OPEN:
            raise RuntimeError(f"Auction '{auction_id}' is not open")
        t = timeout_ms if timeout_ms > 0 else self.policy.bid_timeout_ms
        bid = BidTimer(
            bid_id=bid_id, auction_id=auction_id, bidder=bidder, timeout_ms=t
        )
        auction.bids[bid_id] = bid
        return bid

    def accept_bid(self, auction_id: str, bid_id: str) -> bool:
        auction = self._auctions.get(auction_id)
        if not auction:
            return False
        bid = auction.bids.get(bid_id)
        if not bid or bid.status != BidStatus.PENDING:
            return False
        bid.status = BidStatus.ACCEPTED
        bid.resolved_at = time.time()
        auction.winner = bid.bidder
        auction.phase = AuctionPhase.AWARDED
        # reject remaining pending bids
        for b in auction.bids.values():
            if b.bid_id != bid_id and b.status == BidStatus.PENDING:
                b.status = BidStatus.REJECTED
                b.resolved_at = time.time()
        return True

    def close_auction(self, auction_id: str) -> Optional[str]:
        """Close auction. Returns winner or fallback."""
        auction = self._auctions.get(auction_id)
        if not auction:
            return None
        if auction.phase == AuctionPhase.AWARDED:
            return auction.winner
        auction.phase = AuctionPhase.CLOSED
        # expire remaining pending bids
        for b in auction.bids.values():
            if b.status == BidStatus.PENDING:
                b.status = BidStatus.EXPIRED
                b.resolved_at = time.time()
                self._total_expired_bids += 1
        return auction.fallback

    # ── timeout enforcement ──────────────────────────────────────────────
    def enforce_bid_timeouts(self) -> List[str]:
        """Expire overdue bids. Returns list of expired bid_ids."""
        expired = []
        for auction in self._auctions.values():
            if auction.phase != AuctionPhase.OPEN:
                continue
            for bid in auction.bids.values():
                if bid.is_expired:
                    bid.status = BidStatus.EXPIRED
                    bid.resolved_at = time.time()
                    self._total_expired_bids += 1
                    expired.append(bid.bid_id)
        return expired

    def enforce_auction_timeouts(self) -> List[str]:
        """Close overdue auctions. Returns list of expired auction_ids."""
        now = time.time()
        expired = []
        for a in self._auctions.values():
            if a.phase == AuctionPhase.OPEN and now > a.close_deadline:
                self.close_auction(a.auction_id)
                a.phase = AuctionPhase.EXPIRED
                self._total_expired_auctions += 1
                expired.append(a.auction_id)
        return expired

    def enforce_all(self) -> Dict[str, List[str]]:
        return {
            "expired_bids": self.enforce_bid_timeouts(),
            "expired_auctions": self.enforce_auction_timeouts(),
        }

    # ── introspection ────────────────────────────────────────────────────
    def auction_info(self, auction_id: str) -> Optional[Dict[str, Any]]:
        a = self._auctions.get(auction_id)
        if not a:
            return None
        return {
            "auction_id": a.auction_id,
            "capability": a.capability,
            "phase": a.phase.name,
            "bids": len(a.bids),
            "winner": a.winner,
            "fallback": a.fallback,
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_auctions": len(self._auctions),
            "open_auctions": sum(
                1 for a in self._auctions.values() if a.phase == AuctionPhase.OPEN
            ),
            "total_expired_bids": self._total_expired_bids,
            "total_expired_auctions": self._total_expired_auctions,
            "policy": {
                "bid_timeout_ms": self.policy.bid_timeout_ms,
                "auction_close_ms": self.policy.auction_close_ms,
                "fallback_agent": self.policy.fallback_agent,
            },
        }
