"""
aos/hitl.py — Human-in-the-Loop (HITL) Approval System
═══════════════════════════════════════════════════════════════════════════════
Enables agents to pause mid-execution, request human approval, and resume.

Enterprise research shows 42% of regulated enterprises require human
approval gates. This module makes that possible.

Usage::

    @infrarely.tool
    def process_payment(amount: float, account: str) -> dict:
        return {"status": "processed", "amount": amount, "account": account}

    agent = infrarely.agent("payment-processor")
    agent.require_approval_for(
        tools=["process_payment"],
        when=lambda amount, **_: amount > 1000,
        timeout=3600,
    )

    result = agent.run("Process payment of $5000 to account X")
    # → Execution pauses, sends approval request
    # → Human approves via API/dashboard
    # → Agent resumes
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# APPROVAL STATUS
# ═══════════════════════════════════════════════════════════════════════════════


class ApprovalStatus(Enum):
    """Status of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"
    AUTO_APPROVED = "auto_approved"


# ═══════════════════════════════════════════════════════════════════════════════
# APPROVAL REQUEST
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ApprovalRequest:
    """A human approval request created when an agent hits an approval gate."""

    request_id: str = field(default_factory=lambda: f"approval_{uuid.uuid4().hex[:12]}")
    agent_name: str = ""
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    resolved_by: Optional[str] = None
    timeout: float = 3600.0  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def approved_by(self) -> Optional[str]:
        """Who approved/rejected this request (alias for resolved_by)."""
        return self.resolved_by

    @property
    def is_pending(self) -> bool:
        return self.status == ApprovalStatus.PENDING

    @property
    def is_expired(self) -> bool:
        if self.status != ApprovalStatus.PENDING:
            return False
        return (time.time() - self.created_at) > self.timeout

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "agent_name": self.agent_name,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "reason": self.reason,
            "status": self.status.value,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "resolved_by": self.resolved_by,
            "timeout": self.timeout,
            "metadata": self.metadata,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# APPROVAL RULE
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ApprovalRule:
    """A rule that determines when human approval is required."""

    tools: List[str]
    condition: Optional[Callable[..., bool]] = None
    timeout: float = 3600.0
    reason_template: str = "Agent wants to execute {tool_name} with args: {args}"
    auto_approve_if: Optional[Callable[..., bool]] = None  # bypass condition
    notify: Optional[Callable[["ApprovalRequest"], None]] = None


# ═══════════════════════════════════════════════════════════════════════════════
# APPROVAL MANAGER — Singleton that manages all approval requests
# ═══════════════════════════════════════════════════════════════════════════════


class ApprovalManager:
    """
    Manages approval requests across all agents.

    Provides both blocking (synchronous) and callback-based approval flows.
    Approval requests can be resolved via:
    - Direct API call: ``approval_manager.approve(request_id)``
    - Dashboard UI
    - Webhook callback
    - Auto-approval rules
    """

    _instance: Optional["ApprovalManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ApprovalManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._requests: Dict[str, ApprovalRequest] = {}
                cls._instance._events: Dict[str, threading.Event] = {}
                cls._instance._callbacks: List[Callable[[ApprovalRequest], None]] = []
                cls._instance._request_lock = threading.Lock()
            return cls._instance

    def create_request(
        self,
        agent_name: str,
        tool_name: str = "",
        arguments: Dict[str, Any] = None,
        timeout: float = 3600.0,
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        *,
        action_type: str = "",
        description: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> ApprovalRequest:
        """Create a new approval request and notify listeners."""
        # Detect positional (agent_name, action_type, description, context)
        # convention: arguments would be a string (description) and timeout
        # would be a dict (context).
        if isinstance(arguments, str):
            # Positional form: (agent, action_type, description, context)
            if isinstance(timeout, dict):
                context = timeout
                timeout = 3600.0
            description = arguments
            arguments = None
            if not action_type:
                action_type = tool_name

        # Handle alternate calling convention: request(agent_name, action_type, description, context)
        if action_type and not tool_name:
            tool_name = action_type
        if description and not reason:
            reason = description
        if context is not None and arguments is None:
            arguments = context
        request = ApprovalRequest(
            agent_name=agent_name,
            tool_name=tool_name,
            arguments=arguments or {},
            reason=reason
            or f"Agent '{agent_name}' wants to execute '{tool_name}' with: {arguments}",
            timeout=timeout,
            metadata=metadata or {},
        )

        event = threading.Event()

        with self._request_lock:
            self._requests[request.request_id] = request
            self._events[request.request_id] = event

        # Notify all registered callbacks
        for cb in self._callbacks:
            try:
                cb(request)
            except Exception:
                pass

        return request

    def wait_for_approval(
        self,
        request_id: str,
        timeout: Optional[float] = None,
    ) -> ApprovalRequest:
        """
        Block until the request is resolved or times out.

        Returns the updated ApprovalRequest with final status.
        """
        with self._request_lock:
            request = self._requests.get(request_id)
            event = self._events.get(request_id)

        if request is None or event is None:
            raise ValueError(f"Unknown approval request: {request_id}")

        wait_timeout = timeout or request.timeout

        # Block until signal or timeout
        resolved = event.wait(timeout=wait_timeout)

        if not resolved:
            # Timed out
            with self._request_lock:
                request.status = ApprovalStatus.TIMED_OUT
                request.resolved_at = time.time()

        return request

    def approve(
        self,
        request_id: str,
        *,
        approved_by: str = "human",
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ApprovalRequest:
        """Approve a pending request."""
        meta = metadata or {}
        if reason:
            meta["approval_reason"] = reason
        return self._resolve(request_id, ApprovalStatus.APPROVED, approved_by, meta)

    def reject(
        self,
        request_id: str,
        *,
        rejected_by: str = "human",
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ApprovalRequest:
        """Reject a pending request."""
        meta = metadata or {}
        if reason:
            meta["rejection_reason"] = reason
        return self._resolve(request_id, ApprovalStatus.REJECTED, rejected_by, meta)

    def _resolve(
        self,
        request_id: str,
        status: ApprovalStatus,
        resolved_by: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ApprovalRequest:
        """Resolve a request with the given status."""
        with self._request_lock:
            request = self._requests.get(request_id)
            event = self._events.get(request_id)

        if request is None:
            raise ValueError(f"Unknown approval request: {request_id}")

        if not request.is_pending:
            raise ValueError(
                f"Request {request_id} already resolved with status: {request.status.value}"
            )

        request.status = status
        request.resolved_at = time.time()
        request.resolved_by = resolved_by
        if metadata:
            request.metadata.update(metadata)

        # Signal blocking waiters
        if event:
            event.set()

        return request

    def get_pending(self, agent_name: Optional[str] = None) -> List[ApprovalRequest]:
        """Get all pending approval requests, optionally filtered by agent."""
        with self._request_lock:
            pending = [
                r for r in self._requests.values() if r.is_pending and not r.is_expired
            ]
            if agent_name:
                pending = [r for r in pending if r.agent_name == agent_name]
            return pending

    def request(self, *args, **kwargs) -> ApprovalRequest:
        """Alias for create_request()."""
        return self.create_request(*args, **kwargs)

    def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get a specific approval request."""
        with self._request_lock:
            return self._requests.get(request_id)

    def get_all(
        self,
        agent_name: Optional[str] = None,
        status: Optional[ApprovalStatus] = None,
        limit: int = 100,
    ) -> List[ApprovalRequest]:
        """Get approval requests with optional filters."""
        with self._request_lock:
            results = list(self._requests.values())

        if agent_name:
            results = [r for r in results if r.agent_name == agent_name]
        if status:
            results = [r for r in results if r.status == status]

        results.sort(key=lambda r: r.created_at, reverse=True)
        return results[:limit]

    def on_request(self, callback: Callable[[ApprovalRequest], None]) -> None:
        """Register a callback for new approval requests (e.g., send notification)."""
        self._callbacks.append(callback)

    def clear(self) -> None:
        """Clear all requests (for testing)."""
        with self._request_lock:
            for event in self._events.values():
                event.set()
            self._requests.clear()
            self._events.clear()
            self._callbacks.clear()

    def reset(self) -> None:
        """Reset singleton state (for testing)."""
        self.clear()


# ── Module-level accessor ────────────────────────────────────────────────────


def get_approval_manager() -> ApprovalManager:
    """Get the global approval manager singleton."""
    return ApprovalManager()


# ═══════════════════════════════════════════════════════════════════════════════
# HITL-AWARE TOOL WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════


class HITLGate:
    """
    Gate that wraps tool execution with human approval checks.

    Used internally by Agent.require_approval_for().
    """

    def __init__(self, agent_name: str):
        self._agent_name = agent_name
        self._rules: List[ApprovalRule] = []
        self._manager = get_approval_manager()

    def add_rule(self, rule: ApprovalRule) -> None:
        """Add an approval rule."""
        self._rules.append(rule)

    def check(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Optional[ApprovalRequest]:
        """
        Check if a tool call requires approval.

        Returns an ApprovalRequest if approval is needed, None otherwise.
        """
        for rule in self._rules:
            if tool_name not in rule.tools:
                continue

            # Check auto-approve bypass
            if rule.auto_approve_if:
                try:
                    if rule.auto_approve_if(**arguments):
                        return None  # Auto-approved, no gate needed
                except Exception:
                    pass

            # Check condition
            needs_approval = True
            if rule.condition is not None:
                try:
                    needs_approval = rule.condition(**arguments)
                except Exception:
                    # If condition check fails, require approval (safe default)
                    needs_approval = True

            if needs_approval:
                reason = rule.reason_template.format(
                    tool_name=tool_name,
                    args=arguments,
                )
                request = self._manager.create_request(
                    agent_name=self._agent_name,
                    tool_name=tool_name,
                    arguments=arguments,
                    timeout=rule.timeout,
                    reason=reason,
                )

                # Notify via rule-specific callback
                if rule.notify:
                    try:
                        rule.notify(request)
                    except Exception:
                        pass

                return request

        return None  # No rule matched → no approval needed

    def wait_and_check(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> tuple[bool, Optional[ApprovalRequest]]:
        """
        Check approval and block until resolved.

        Returns (approved: bool, request: ApprovalRequest or None).
        If no approval needed, returns (True, None).
        """
        request = self.check(tool_name, arguments)
        if request is None:
            return True, None

        # Block until resolved
        request = self._manager.wait_for_approval(request.request_id)

        approved = request.status == ApprovalStatus.APPROVED
        return approved, request

    @property
    def rules(self) -> List[ApprovalRule]:
        """Get all approval rules."""
        return list(self._rules)

    def clear(self) -> None:
        """Clear all rules."""
        self._rules.clear()
