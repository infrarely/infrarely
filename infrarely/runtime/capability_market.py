"""
Module 7 — Capability Marketplace
===================================
Lets agents **publish** capabilities they can fulfil, and lets the scheduler
**discover** the best provider via composite scoring (Gap 5).

Key features
------------
* Publish / discover / best_provider API
* Composite score  = quality×0.4 + success_rate×0.3 + cost_factor×0.2 + latency_factor×0.1
* Auto-deprecation when success rate drops below threshold
* Per-agent listing limit  (MAX_LISTINGS_PER_AGENT)
* Duplicate listing prevention  (same capability + agent)
* Invocation tracking  (success / failure / latency)
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("capability_market")


# ─── Enums ────────────────────────────────────────────────────────────────────


class ListingStatus(enum.Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    REMOVED = "removed"


# ─── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class CapabilityListing:
    listing_id: str
    capability: str
    provider_agent: str
    description: str = ""
    token_cost: int = 0
    tags: Set[str] = field(default_factory=set)
    status: ListingStatus = ListingStatus.ACTIVE
    quality: float = 1.0  # 0‑1 quality rating
    invocation_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_latency_ms: float = 0.0
    created_ts: float = field(default_factory=time.time)

    # ── derived scores ──────────────────────────────────────────────────

    @property
    def success_rate(self) -> float:
        if self.invocation_count == 0:
            return 1.0
        return self.success_count / self.invocation_count

    @property
    def avg_latency_ms(self) -> float:
        if self.invocation_count == 0:
            return 0.0
        return self.total_latency_ms / self.invocation_count

    @property
    def cost_factor(self) -> float:
        """Lower cost → higher factor (inverse, capped at 1.0)."""
        if self.token_cost <= 0:
            return 1.0
        return min(1.0, 100.0 / self.token_cost)

    @property
    def latency_factor(self) -> float:
        """Lower latency → higher factor."""
        avg = self.avg_latency_ms
        if avg <= 0:
            return 1.0
        return min(1.0, 500.0 / avg)

    @property
    def composite_score(self) -> float:
        """Gap 5 — weighted composite score for ranking."""
        return (
            self.quality * 0.4
            + self.success_rate * 0.3
            + self.cost_factor * 0.2
            + self.latency_factor * 0.1
        )


# ─── Marketplace ──────────────────────────────────────────────────────────────


class CapabilityMarketplace:
    """
    Central marketplace where agents advertise capabilities and the
    scheduler discovers the best provider via composite scoring.
    """

    MAX_LISTINGS_PER_AGENT: int = 20
    DEPRECATION_THRESHOLD: float = 0.50  # success rate below → auto-deprecate
    MIN_INVOCATIONS_FOR_DEPRECATION: int = 4

    def __init__(self) -> None:
        self._listings: Dict[str, CapabilityListing] = {}  # listing_id → listing
        self._by_capability: Dict[str, List[str]] = {}  # cap → [listing_ids]
        self._by_agent: Dict[str, List[str]] = {}  # agent → [listing_ids]
        self._counter: int = 0

    # ── publish ─────────────────────────────────────────────────────────

    def publish(
        self,
        capability: str,
        provider_agent: str,
        description: str = "",
        *,
        token_cost: int = 0,
        quality: float = 1.0,
        tags: Set[str] = None,
    ) -> CapabilityListing:
        """
        Publish a capability listing.

        Raises ValueError if:
        * duplicate (same capability + same agent already active)
        * per-agent listing limit exceeded
        """
        # Duplicate check
        for lid in self._by_agent.get(provider_agent, []):
            existing = self._listings[lid]
            if (
                existing.capability == capability
                and existing.status == ListingStatus.ACTIVE
            ):
                raise ValueError(
                    f"Agent '{provider_agent}' already has an active listing "
                    f"for capability '{capability}'"
                )

        # Per-agent limit
        active_count = sum(
            1
            for lid in self._by_agent.get(provider_agent, [])
            if self._listings[lid].status == ListingStatus.ACTIVE
        )
        if active_count >= self.MAX_LISTINGS_PER_AGENT:
            raise ValueError(
                f"Agent '{provider_agent}' reached listing limit "
                f"({self.MAX_LISTINGS_PER_AGENT})"
            )

        self._counter += 1
        listing_id = f"listing_{self._counter:04d}"

        listing = CapabilityListing(
            listing_id=listing_id,
            capability=capability,
            provider_agent=provider_agent,
            description=description,
            token_cost=token_cost,
            quality=quality,
            tags=tags or set(),
        )

        self._listings[listing_id] = listing
        self._by_capability.setdefault(capability, []).append(listing_id)
        self._by_agent.setdefault(provider_agent, []).append(listing_id)

        logger.info(
            "Published listing '%s': cap='%s' by '%s' cost=%d",
            listing_id,
            capability,
            provider_agent,
            token_cost,
        )
        return listing

    # ── discover ────────────────────────────────────────────────────────

    def discover(
        self,
        capability: str,
        *,
        tags: Set[str] = None,
        only_active: bool = True,
    ) -> List[CapabilityListing]:
        """
        Return listings for *capability*, ranked by composite score (desc).
        Optionally filter by tags and active status.
        """
        ids = self._by_capability.get(capability, [])
        results: List[CapabilityListing] = []
        for lid in ids:
            listing = self._listings[lid]
            if only_active and listing.status != ListingStatus.ACTIVE:
                continue
            if tags and not tags.issubset(listing.tags):
                continue
            results.append(listing)

        results.sort(key=lambda l: l.composite_score, reverse=True)
        return results

    # ── best provider ───────────────────────────────────────────────────

    def best_provider(
        self,
        capability: str,
        *,
        tags: Set[str] = None,
    ) -> Optional[CapabilityListing]:
        """Shorthand — return the highest-ranked active listing or None."""
        results = self.discover(capability, tags=tags)
        return results[0] if results else None

    # ── invocation tracking ─────────────────────────────────────────────

    def record_invocation(
        self,
        listing_id: str,
        *,
        success: bool = True,
        latency_ms: float = 0.0,
    ) -> None:
        """
        Record one invocation.  If success rate drops below threshold
        after enough invocations, auto-deprecate the listing.
        """
        listing = self._listings.get(listing_id)
        if listing is None:
            logger.warning("record_invocation: listing '%s' not found", listing_id)
            return

        listing.invocation_count += 1
        listing.total_latency_ms += latency_ms
        if success:
            listing.success_count += 1
        else:
            listing.failure_count += 1

        # Auto-deprecation check
        if (
            listing.invocation_count >= self.MIN_INVOCATIONS_FOR_DEPRECATION
            and listing.success_rate < self.DEPRECATION_THRESHOLD
            and listing.status == ListingStatus.ACTIVE
        ):
            listing.status = ListingStatus.DEPRECATED
            logger.warning(
                "Auto-deprecated listing '%s' (cap='%s', success_rate=%.2f)",
                listing_id,
                listing.capability,
                listing.success_rate,
            )

    # ── removal ─────────────────────────────────────────────────────────

    def remove_listing(self, listing_id: str) -> None:
        listing = self._listings.get(listing_id)
        if listing:
            listing.status = ListingStatus.REMOVED
            logger.info("Removed listing '%s'", listing_id)

    def remove_agent_listings(self, agent_id: str) -> int:
        """Remove all listings by a specific agent. Returns count removed."""
        count = 0
        for lid in self._by_agent.get(agent_id, []):
            listing = self._listings[lid]
            if listing.status == ListingStatus.ACTIVE:
                listing.status = ListingStatus.REMOVED
                count += 1
        return count

    # ── snapshot ────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        active = [
            l for l in self._listings.values() if l.status == ListingStatus.ACTIVE
        ]
        caps = {l.capability for l in active}
        return {
            "total_listings": len(self._listings),
            "active_listings": len(active),
            "unique_capabilities": len(caps),
            "capabilities": sorted(caps),
            "agents_publishing": len({l.provider_agent for l in active}),
        }
