"""
runtime/security_sandbox.py — GAP 10
═══════════════════════════════════════════════════════════════════════════════
Security sandbox for agent execution: permission scoping, tool access
control, and execution isolation.

Subsystems:
  • PermissionScope    — per-agent allow/deny lists for tools, data, actions
  • SandboxPolicy      — configurable defaults
  • AgentSandbox       — execution environment for a single agent
  • SecuritySandboxManager — orchestrator with snapshot()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set


# ─── Enums ────────────────────────────────────────────────────────────────────
class AccessDecision(Enum):
    ALLOW = auto()
    DENY = auto()


class ViolationSeverity(Enum):
    WARNING = auto()
    BLOCKED = auto()
    CRITICAL = auto()


# ─── Data ─────────────────────────────────────────────────────────────────────
@dataclass
class SecurityViolation:
    agent_id: str
    resource: str
    action: str
    severity: ViolationSeverity
    ts: float = field(default_factory=time.time)
    details: str = ""


@dataclass
class PermissionScope:
    """Per-agent permission scope."""

    agent_id: str
    allowed_tools: Set[str] = field(default_factory=set)
    denied_tools: Set[str] = field(default_factory=set)
    allowed_data: Set[str] = field(default_factory=set)
    denied_data: Set[str] = field(default_factory=set)
    allowed_actions: Set[str] = field(default_factory=set)
    denied_actions: Set[str] = field(default_factory=set)
    max_memory_mb: float = 512.0
    max_cpu_s: float = 60.0
    network_access: bool = False

    def check_tool(self, tool_name: str) -> AccessDecision:
        if tool_name in self.denied_tools:
            return AccessDecision.DENY
        if self.allowed_tools and tool_name not in self.allowed_tools:
            return AccessDecision.DENY
        return AccessDecision.ALLOW

    def check_data(self, data_key: str) -> AccessDecision:
        if data_key in self.denied_data:
            return AccessDecision.DENY
        if self.allowed_data and data_key not in self.allowed_data:
            return AccessDecision.DENY
        return AccessDecision.ALLOW

    def check_action(self, action: str) -> AccessDecision:
        if action in self.denied_actions:
            return AccessDecision.DENY
        if self.allowed_actions and action not in self.allowed_actions:
            return AccessDecision.DENY
        return AccessDecision.ALLOW


@dataclass
class SandboxPolicy:
    """Default sandbox policy applied when no per-agent scope exists."""

    default_allowed_tools: Set[str] = field(default_factory=set)
    default_denied_tools: Set[str] = field(default_factory=set)
    default_network_access: bool = False
    max_violations_before_kill: int = 5
    enable_isolation: bool = True


# ─── Agent Sandbox ────────────────────────────────────────────────────────────
class AgentSandbox:
    """Execution sandbox for a single agent."""

    def __init__(self, agent_id: str, scope: PermissionScope, max_violations: int = 5):
        self.agent_id = agent_id
        self.scope = scope
        self.max_violations = max_violations
        self._violations: List[SecurityViolation] = []
        self._killed = False
        self._created_at = time.time()

    @property
    def is_killed(self) -> bool:
        return self._killed

    @property
    def violation_count(self) -> int:
        return len(self._violations)

    def request_tool(self, tool_name: str) -> AccessDecision:
        decision = self.scope.check_tool(tool_name)
        if decision == AccessDecision.DENY:
            self._record_violation(tool_name, "tool_access", ViolationSeverity.BLOCKED)
        return decision if not self._killed else AccessDecision.DENY

    def request_data(self, data_key: str) -> AccessDecision:
        decision = self.scope.check_data(data_key)
        if decision == AccessDecision.DENY:
            self._record_violation(data_key, "data_access", ViolationSeverity.BLOCKED)
        return decision if not self._killed else AccessDecision.DENY

    def request_action(self, action: str) -> AccessDecision:
        decision = self.scope.check_action(action)
        if decision == AccessDecision.DENY:
            self._record_violation(action, "action_access", ViolationSeverity.BLOCKED)
        return decision if not self._killed else AccessDecision.DENY

    def _record_violation(
        self, resource: str, action: str, severity: ViolationSeverity
    ):
        v = SecurityViolation(
            agent_id=self.agent_id, resource=resource, action=action, severity=severity
        )
        self._violations.append(v)
        if len(self._violations) >= self.max_violations:
            self._killed = True

    def kill(self) -> None:
        self._killed = True

    def violations(self) -> List[SecurityViolation]:
        return list(self._violations)


# ─── Security Sandbox Manager ────────────────────────────────────────────────
class SecuritySandboxManager:
    """
    Top-level manager for creating/tracking agent sandboxes.
    """

    def __init__(self, policy: Optional[SandboxPolicy] = None):
        self.policy = policy or SandboxPolicy()
        self._sandboxes: Dict[str, AgentSandbox] = {}
        self._scopes: Dict[str, PermissionScope] = {}

    # ── scope management ─────────────────────────────────────────────────
    def define_scope(
        self,
        agent_id: str,
        allowed_tools: Optional[Set[str]] = None,
        denied_tools: Optional[Set[str]] = None,
        allowed_data: Optional[Set[str]] = None,
        denied_data: Optional[Set[str]] = None,
        allowed_actions: Optional[Set[str]] = None,
        denied_actions: Optional[Set[str]] = None,
        network_access: bool = False,
    ) -> PermissionScope:
        scope = PermissionScope(
            agent_id=agent_id,
            allowed_tools=allowed_tools or set(self.policy.default_allowed_tools),
            denied_tools=denied_tools or set(self.policy.default_denied_tools),
            allowed_data=allowed_data or set(),
            denied_data=denied_data or set(),
            allowed_actions=allowed_actions or set(),
            denied_actions=denied_actions or set(),
            network_access=(
                network_access if network_access else self.policy.default_network_access
            ),
        )
        self._scopes[agent_id] = scope
        return scope

    # ── sandbox lifecycle ────────────────────────────────────────────────
    def create_sandbox(self, agent_id: str) -> AgentSandbox:
        scope = self._scopes.get(agent_id)
        if not scope:
            scope = self.define_scope(agent_id)
        sb = AgentSandbox(
            agent_id=agent_id,
            scope=scope,
            max_violations=self.policy.max_violations_before_kill,
        )
        self._sandboxes[agent_id] = sb
        return sb

    def get_sandbox(self, agent_id: str) -> Optional[AgentSandbox]:
        return self._sandboxes.get(agent_id)

    def destroy_sandbox(self, agent_id: str) -> bool:
        return self._sandboxes.pop(agent_id, None) is not None

    def kill_agent(self, agent_id: str) -> bool:
        sb = self._sandboxes.get(agent_id)
        if not sb:
            return False
        sb.kill()
        return True

    # ── convenience ──────────────────────────────────────────────────────
    def check_tool(self, agent_id: str, tool_name: str) -> AccessDecision:
        sb = self._sandboxes.get(agent_id)
        if not sb:
            return AccessDecision.DENY
        return sb.request_tool(tool_name)

    def check_data(self, agent_id: str, data_key: str) -> AccessDecision:
        sb = self._sandboxes.get(agent_id)
        if not sb:
            return AccessDecision.DENY
        return sb.request_data(data_key)

    def all_violations(self) -> List[SecurityViolation]:
        violations = []
        for sb in self._sandboxes.values():
            violations.extend(sb.violations())
        return violations

    # ── introspection ────────────────────────────────────────────────────
    def snapshot(self) -> Dict[str, Any]:
        total_v = sum(sb.violation_count for sb in self._sandboxes.values())
        killed = sum(1 for sb in self._sandboxes.values() if sb.is_killed)
        return {
            "total_sandboxes": len(self._sandboxes),
            "killed_sandboxes": killed,
            "total_violations": total_v,
            "isolation_enabled": self.policy.enable_isolation,
            "max_violations_before_kill": self.policy.max_violations_before_kill,
        }
