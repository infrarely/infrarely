"""
runtime/message_bus.py — Module 3: Agent Messaging Bus
═══════════════════════════════════════════════════════════════════════════════
In-memory message queue that all agents communicate through.
Agents NEVER call each other directly: Agent → Message Bus → Scheduler → Target.

Gap solutions:
  Gap 1  — Deadlock prevention: message TTL + circular dependency check
  Gap 2  — Message storms: per-agent rate limiting, queue cap, backpressure

Message flow:
  1. Sender creates AgentMessage with routing info
  2. Bus validates (rate limits, queue depth)
  3. Bus delivers to recipient inbox or broadcasts to subscribers
"""

from __future__ import annotations
import time
import uuid
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set
from infrarely.observability import logger


class MessageType(Enum):
    REQUEST = auto()
    RESPONSE = auto()
    BROADCAST = auto()
    EVENT = auto()
    HEARTBEAT = auto()
    CANCEL = auto()


class MessagePriority(Enum):
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class AgentMessage:
    """A structured message on the bus."""

    msg_id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:8]}")
    msg_type: MessageType = MessageType.REQUEST
    sender: str = ""
    recipient: str = ""  # "" for broadcast
    topic: str = ""  # subscription topic for events
    intent: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: str = ""  # link request ↔ response
    ttl_ms: float = 60_000  # 60s time-to-live (Gap 1)
    created_at: float = field(default_factory=time.time)
    delivered: bool = False
    delivery_time: float = 0.0

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) * 1000 > self.ttl_ms


class MessageBus:
    """
    Central messaging infrastructure for multi-agent communication.

    Invariants:
      • Per-agent inbox cap: 100 messages
      • Per-agent send rate: max 50 messages per second (Gap 2)
      • Total bus capacity: 2000 messages
      • Expired messages auto-purged
    """

    INBOX_CAP = 100  # per-agent max
    RATE_LIMIT = 50  # messages per second per agent
    RATE_WINDOW = 1.0  # seconds
    BUS_CAPACITY = 2000  # total messages across all inboxes

    def __init__(self):
        self._inboxes: Dict[str, deque] = defaultdict(deque)
        self._subscriptions: Dict[str, Set[str]] = defaultdict(set)  # topic → agent_ids
        self._rate_tracker: Dict[str, List[float]] = defaultdict(
            list
        )  # agent → timestamps
        self._total_messages = 0
        self._total_delivered = 0
        self._total_dropped = 0
        self._total_expired = 0
        self._dead_letter: List[AgentMessage] = []  # undeliverable messages
        self._message_log: List[AgentMessage] = []

    # ── Send ──────────────────────────────────────────────────────────────────
    def send(self, message: AgentMessage) -> bool:
        """
        Send a message. Returns True on success.
        Enforces rate limiting (Gap 2) and queue caps (Gap 2).
        """
        # Rate limit check (Gap 2)
        if not self._check_rate(message.sender):
            self._total_dropped += 1
            logger.warn(
                f"Bus: rate limit exceeded for '{message.sender}' — "
                f"message '{message.msg_id}' dropped"
            )
            return False

        # Expired TTL check (Gap 1)
        if message.is_expired:
            self._total_expired += 1
            self._dead_letter_add(message, "expired before send")
            return False

        # Global capacity (Gap 2)
        total_queued = sum(len(q) for q in self._inboxes.values())
        if total_queued >= self.BUS_CAPACITY:
            self._total_dropped += 1
            self._dead_letter_add(message, "bus at capacity")
            logger.warn(f"Bus: at capacity ({self.BUS_CAPACITY}) — message dropped")
            return False

        self._total_messages += 1

        if message.msg_type == MessageType.BROADCAST:
            return self._broadcast(message)
        elif message.msg_type == MessageType.EVENT and message.topic:
            return self._publish_event(message)
        else:
            return self._deliver(message, message.recipient)

    # ── Receive ───────────────────────────────────────────────────────────────
    def receive(self, agent_id: str, max_messages: int = 10) -> List[AgentMessage]:
        """
        Retrieve messages from an agent's inbox.
        Auto-purges expired messages (Gap 1).
        """
        inbox = self._inboxes.get(agent_id)
        if not inbox:
            return []

        # Purge expired (Gap 1)
        self._purge_expired(agent_id)

        # Re-fetch inbox after purge (purge replaces the deque)
        inbox = self._inboxes.get(agent_id)
        if not inbox:
            return []

        messages = []
        for _ in range(min(max_messages, len(inbox))):
            if inbox:
                msg = inbox.popleft()
                msg.delivered = True
                msg.delivery_time = time.time()
                self._total_delivered += 1
                messages.append(msg)
        return messages

    # ── Subscribe/Unsubscribe ─────────────────────────────────────────────────
    def subscribe(self, agent_id: str, topic: str):
        """Subscribe agent to a topic."""
        self._subscriptions[topic].add(agent_id)
        logger.debug(f"Bus: '{agent_id}' subscribed to topic '{topic}'")

    def unsubscribe(self, agent_id: str, topic: str):
        """Unsubscribe agent from a topic."""
        self._subscriptions[topic].discard(agent_id)

    def unsubscribe_all(self, agent_id: str):
        """Remove agent from all subscriptions."""
        for topic in list(self._subscriptions):
            self._subscriptions[topic].discard(agent_id)
            if not self._subscriptions[topic]:
                del self._subscriptions[topic]

    # ── Inbox management ──────────────────────────────────────────────────────
    def inbox_depth(self, agent_id: str) -> int:
        return len(self._inboxes.get(agent_id, []))

    def clear_inbox(self, agent_id: str):
        """Clear an agent's inbox."""
        self._inboxes.pop(agent_id, None)

    def remove_agent(self, agent_id: str):
        """Remove agent from bus entirely."""
        self.clear_inbox(agent_id)
        self.unsubscribe_all(agent_id)

    # ── Private delivery ──────────────────────────────────────────────────────
    def _deliver(self, message: AgentMessage, recipient: str) -> bool:
        """Deliver to a single recipient's inbox."""
        if not recipient:
            self._dead_letter_add(message, "no recipient")
            return False

        inbox = self._inboxes[recipient]
        if len(inbox) >= self.INBOX_CAP:
            self._total_dropped += 1
            self._dead_letter_add(message, f"inbox full for '{recipient}'")
            logger.warn(
                f"Bus: inbox cap ({self.INBOX_CAP}) reached for '{recipient}' "
                f"— message dropped (backpressure)"
            )
            return False

        inbox.append(message)
        self._log_message(message)
        return True

    def _broadcast(self, message: AgentMessage) -> bool:
        """Deliver to all inboxes."""
        delivered_any = False
        for agent_id in list(self._inboxes.keys()):
            if agent_id != message.sender:
                msg_copy = AgentMessage(
                    msg_id=f"{message.msg_id}_{agent_id[:6]}",
                    msg_type=message.msg_type,
                    sender=message.sender,
                    recipient=agent_id,
                    topic=message.topic,
                    intent=message.intent,
                    payload=message.payload.copy(),
                    priority=message.priority,
                    correlation_id=message.correlation_id,
                    ttl_ms=message.ttl_ms,
                    created_at=message.created_at,
                )
                if self._deliver(msg_copy, agent_id):
                    delivered_any = True
        return delivered_any

    def _publish_event(self, message: AgentMessage) -> bool:
        """Deliver to subscribers of a topic."""
        subscribers = self._subscriptions.get(message.topic, set())
        if not subscribers:
            self._dead_letter_add(message, f"no subscribers for '{message.topic}'")
            return False

        delivered_any = False
        for agent_id in subscribers:
            if agent_id != message.sender:
                if self._deliver(message, agent_id):
                    delivered_any = True
        return delivered_any

    # ── Rate limiting (Gap 2) ─────────────────────────────────────────────────
    def _check_rate(self, sender: str) -> bool:
        """Enforce per-agent rate limit. Returns True if send allowed."""
        if not sender:
            return True
        now = time.time()
        timestamps = self._rate_tracker[sender]
        # Trim old timestamps
        cutoff = now - self.RATE_WINDOW
        self._rate_tracker[sender] = [t for t in timestamps if t > cutoff]
        if len(self._rate_tracker[sender]) >= self.RATE_LIMIT:
            return False
        self._rate_tracker[sender].append(now)
        return True

    # ── TTL purging (Gap 1) ───────────────────────────────────────────────────
    def _purge_expired(self, agent_id: str):
        """Remove expired messages from an inbox."""
        inbox = self._inboxes.get(agent_id)
        if not inbox:
            return
        fresh = deque()
        for msg in inbox:
            if msg.is_expired:
                self._total_expired += 1
            else:
                fresh.append(msg)
        self._inboxes[agent_id] = fresh

    def purge_all_expired(self):
        """Global expired message cleanup."""
        for agent_id in list(self._inboxes.keys()):
            self._purge_expired(agent_id)

    # ── Dead letter queue ─────────────────────────────────────────────────────
    def _dead_letter_add(self, message: AgentMessage, reason: str):
        """Add undeliverable message to dead letter queue."""
        message.payload["_dead_reason"] = reason
        self._dead_letter.append(message)
        if len(self._dead_letter) > 200:
            self._dead_letter = self._dead_letter[-100:]

    # ── Logging ───────────────────────────────────────────────────────────────
    def _log_message(self, message: AgentMessage):
        self._message_log.append(message)
        if len(self._message_log) > 1000:
            self._message_log = self._message_log[-500:]

    # ── Snapshot ──────────────────────────────────────────────────────────────
    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_messages": self._total_messages,
            "total_delivered": self._total_delivered,
            "total_dropped": self._total_dropped,
            "total_expired": self._total_expired,
            "dead_letter_count": len(self._dead_letter),
            "inbox_depths": {aid: len(inbox) for aid, inbox in self._inboxes.items()},
            "subscriptions": {
                topic: list(subs) for topic, subs in self._subscriptions.items()
            },
            "recent_messages": [
                {
                    "msg_id": m.msg_id,
                    "type": m.msg_type.name,
                    "sender": m.sender,
                    "recipient": m.recipient,
                    "intent": m.intent,
                    "delivered": m.delivered,
                }
                for m in self._message_log[-10:]
            ],
        }
