"""
runtime/identity_permissions.py — Module 5: Agent Identity & Permissions
═══════════════════════════════════════════════════════════════════════════════
RBAC system that controls what agents can do.
Like Linux user/group permissions or SELinux policies.

Gap solutions:
  Gap 6  — Security boundaries: agents cannot escalate privileges,
           roles are immutable after assignment, permissions enforced
           at every operation

Roles:
  ADMIN    — full control over all agents and runtime
  WORKER   — can execute tools and capabilities
  OBSERVER — read-only access to shared memory and monitoring
  SANDBOX  — restricted execution, cannot access shared memory
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set
from infrarely.observability import logger


class AgentRole(Enum):
    ADMIN = auto()
    WORKER = auto()
    OBSERVER = auto()
    SANDBOX = auto()


class Permission(Enum):
    """Atomic permissions that can be assigned to roles."""

    # Tool permissions
    TOOL_EXECUTE = auto()
    TOOL_REGISTER = auto()
    # Capability permissions
    CAPABILITY_EXECUTE = auto()
    CAPABILITY_REGISTER = auto()
    # Memory permissions
    MEMORY_READ = auto()
    MEMORY_WRITE = auto()
    MEMORY_DELETE = auto()
    MEMORY_LOCK = auto()
    # Bus permissions
    BUS_SEND = auto()
    BUS_RECEIVE = auto()
    BUS_BROADCAST = auto()
    # Agent management
    AGENT_SPAWN = auto()
    AGENT_TERMINATE = auto()
    AGENT_PAUSE = auto()
    # Scheduler
    TASK_SUBMIT = auto()
    TASK_CANCEL = auto()
    # Monitoring
    MONITOR_READ = auto()
    MONITOR_CONFIGURE = auto()
    # Runtime
    RUNTIME_CONFIGURE = auto()


# ── Role → Permission mapping ─────────────────────────────────────────────────

ROLE_PERMISSIONS: Dict[AgentRole, FrozenSet[Permission]] = {
    AgentRole.ADMIN: frozenset(Permission),  # all permissions
    AgentRole.WORKER: frozenset(
        [
            Permission.TOOL_EXECUTE,
            Permission.CAPABILITY_EXECUTE,
            Permission.MEMORY_READ,
            Permission.MEMORY_WRITE,
            Permission.BUS_SEND,
            Permission.BUS_RECEIVE,
            Permission.TASK_SUBMIT,
            Permission.MONITOR_READ,
        ]
    ),
    AgentRole.OBSERVER: frozenset(
        [
            Permission.MEMORY_READ,
            Permission.BUS_RECEIVE,
            Permission.MONITOR_READ,
        ]
    ),
    AgentRole.SANDBOX: frozenset(
        [
            Permission.TOOL_EXECUTE,
            Permission.CAPABILITY_EXECUTE,
            Permission.BUS_SEND,
            Permission.BUS_RECEIVE,
        ]
    ),
}


@dataclass
class AgentIdentity:
    """Immutable identity assigned to an agent at registration time."""

    agent_id: str
    role: AgentRole
    extra_permissions: FrozenSet[Permission] = frozenset()
    denied_permissions: FrozenSet[Permission] = frozenset()
    created_at: float = field(default_factory=time.time)
    created_by: str = "system"
    tags: Dict[str, str] = field(default_factory=dict)

    @property
    def effective_permissions(self) -> FrozenSet[Permission]:
        """Role permissions + extras − denied."""
        base = ROLE_PERMISSIONS.get(self.role, frozenset())
        return (base | self.extra_permissions) - self.denied_permissions


class IdentityManager:
    """
    Manages agent identities and enforces permission checks.

    Invariants (Gap 6: security boundaries):
      • Roles cannot be changed after creation
      • Only ADMIN can grant extra permissions
      • Permission checks are O(1)
      • No privilege escalation: cannot grant permissions you don't have
    """

    def __init__(self):
        self._identities: Dict[str, AgentIdentity] = {}
        self._audit_log: List[Dict[str, Any]] = []

    # ── Create identity ───────────────────────────────────────────────────────
    def create_identity(
        self,
        agent_id: str,
        role: AgentRole,
        created_by: str = "system",
        extra_permissions: FrozenSet[Permission] = frozenset(),
        denied_permissions: FrozenSet[Permission] = frozenset(),
        tags: Dict[str, str] = None,
    ) -> AgentIdentity:
        """
        Create an identity for an agent. Immutable after creation.
        Raises if agent_id already has an identity.
        """
        if agent_id in self._identities:
            raise ValueError(
                f"Identity already exists for '{agent_id}' — "
                "roles are immutable (Gap 6)"
            )

        # Gap 6: no privilege escalation — creator must have all extras
        if created_by != "system" and extra_permissions:
            creator = self._identities.get(created_by)
            if creator:
                creator_perms = creator.effective_permissions
                escalated = extra_permissions - creator_perms
                if escalated:
                    self._audit(
                        agent_id,
                        "escalation_blocked",
                        {
                            "by": created_by,
                            "attempted": [p.name for p in escalated],
                        },
                    )
                    raise PermissionError(
                        f"Privilege escalation blocked: '{created_by}' cannot grant "
                        f"{[p.name for p in escalated]} — they don't have those permissions"
                    )

        identity = AgentIdentity(
            agent_id=agent_id,
            role=role,
            extra_permissions=extra_permissions,
            denied_permissions=denied_permissions,
            created_by=created_by,
            tags=tags or {},
        )
        self._identities[agent_id] = identity
        self._audit(agent_id, "identity_created", {"role": role.name, "by": created_by})
        return identity

    # ── Check permission ──────────────────────────────────────────────────────
    def check(self, agent_id: str, permission: Permission) -> bool:
        """
        Check if agent has a specific permission. O(1).
        Returns False if agent has no identity.
        """
        identity = self._identities.get(agent_id)
        if not identity:
            return False
        return permission in identity.effective_permissions

    def require(self, agent_id: str, permission: Permission) -> bool:
        """Check permission and raise PermissionError if denied."""
        if not self.check(agent_id, permission):
            self._audit(agent_id, "permission_denied", {"permission": permission.name})
            raise PermissionError(
                f"Agent '{agent_id}' lacks permission {permission.name}"
            )
        return True

    # ── Query ─────────────────────────────────────────────────────────────────
    def get_identity(self, agent_id: str) -> Optional[AgentIdentity]:
        return self._identities.get(agent_id)

    def get_role(self, agent_id: str) -> Optional[AgentRole]:
        identity = self._identities.get(agent_id)
        return identity.role if identity else None

    def agents_with_role(self, role: AgentRole) -> List[str]:
        return [aid for aid, ident in self._identities.items() if ident.role == role]

    def agents_with_permission(self, permission: Permission) -> List[str]:
        return [
            aid
            for aid, ident in self._identities.items()
            if permission in ident.effective_permissions
        ]

    # ── Remove identity ───────────────────────────────────────────────────────
    def remove_identity(self, agent_id: str) -> bool:
        if agent_id in self._identities:
            self._audit(agent_id, "identity_removed", {})
            del self._identities[agent_id]
            return True
        return False

    # ── Grant/deny extra permissions (admin only, Gap 6) ──────────────────────
    def grant_extra(
        self, agent_id: str, permission: Permission, granted_by: str
    ) -> bool:
        """
        Grant an extra permission to an agent.
        Only ADMIN can grant. Cannot escalate (Gap 6).
        """
        grantor = self._identities.get(granted_by)
        if not grantor or grantor.role != AgentRole.ADMIN:
            self._audit(
                agent_id,
                "grant_denied",
                {
                    "by": granted_by,
                    "permission": permission.name,
                },
            )
            return False

        identity = self._identities.get(agent_id)
        if not identity:
            return False

        # Rebuild as frozen sets are immutable
        new_extras = set(identity.extra_permissions)
        new_extras.add(permission)
        identity_dict = {
            "agent_id": identity.agent_id,
            "role": identity.role,
            "extra_permissions": frozenset(new_extras),
            "denied_permissions": identity.denied_permissions,
            "created_at": identity.created_at,
            "created_by": identity.created_by,
            "tags": identity.tags,
        }
        self._identities[agent_id] = AgentIdentity(**identity_dict)
        self._audit(
            agent_id,
            "permission_granted",
            {
                "by": granted_by,
                "permission": permission.name,
            },
        )
        return True

    def deny_permission(
        self, agent_id: str, permission: Permission, denied_by: str
    ) -> bool:
        """Deny a permission. Only ADMIN can deny."""
        denier = self._identities.get(denied_by)
        if not denier or denier.role != AgentRole.ADMIN:
            return False

        identity = self._identities.get(agent_id)
        if not identity:
            return False

        new_denied = set(identity.denied_permissions)
        new_denied.add(permission)
        identity_dict = {
            "agent_id": identity.agent_id,
            "role": identity.role,
            "extra_permissions": identity.extra_permissions,
            "denied_permissions": frozenset(new_denied),
            "created_at": identity.created_at,
            "created_by": identity.created_by,
            "tags": identity.tags,
        }
        self._identities[agent_id] = AgentIdentity(**identity_dict)
        self._audit(
            agent_id,
            "permission_denied_by_admin",
            {
                "by": denied_by,
                "permission": permission.name,
            },
        )
        return True

    # ── Audit log ─────────────────────────────────────────────────────────────
    def _audit(self, agent_id: str, action: str, details: Dict[str, Any]):
        self._audit_log.append(
            {
                "agent_id": agent_id,
                "action": action,
                "details": details,
                "timestamp": time.time(),
            }
        )
        if len(self._audit_log) > 500:
            self._audit_log = self._audit_log[-250:]

    # ── Snapshot ──────────────────────────────────────────────────────────────
    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_identities": len(self._identities),
            "roles": {
                role.name: len(self.agents_with_role(role)) for role in AgentRole
            },
            "recent_audit": self._audit_log[-10:],
            "identities": {
                aid: {
                    "role": ident.role.name,
                    "permissions": len(ident.effective_permissions),
                    "created_by": ident.created_by,
                }
                for aid, ident in self._identities.items()
            },
        }
