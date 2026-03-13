"""
aos/multitenancy.py — Multi-Tenancy Support
═══════════════════════════════════════════════════════════════════════════════
SCALE GAP 5: Tenant isolation for knowledge, memory, tools, and billing.

Usage:
    import infrarely

    t = aos.create_tenant("acme-corp", config={
        "model": "gpt-4o-mini",
        "token_budget": 1_000_000,
    })

    agent = aos.Agent("support", tenant_id="acme-corp")
    result = agent.run("Help customer")       # isolated knowledge/memory/billing

    usage = aos.tenant_manager.usage("acme-corp")
    assert usage["tokens_used"] <= 1_000_000

Architecture:
    TenantConfig     — Per-tenant config (model overrides, budgets, tool ACLs)
    TenantContext    — Runtime context for tenant isolation
    TenantManager    — Registry & lifecycle management
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# ═══════════════════════════════════════════════════════════════════════════════
# TENANT CONFIG
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TenantConfig:
    """
    Per-tenant configuration.

    Attributes
    ----------
    tenant_id : str
        Unique tenant identifier.
    display_name : str
        Human-readable name.
    model : str
        Default LLM model for this tenant.
    token_budget : int
        Monthly/total token budget (0 = unlimited).
    allowed_tools : set
        Tool whitelist (empty = all tools allowed).
    blocked_tools : set
        Tool blacklist.
    max_agents : int
        Maximum concurrent agents (0 = unlimited).
    max_tasks_per_minute : int
        Rate limit (0 = unlimited).
    metadata : dict
        Arbitrary tenant metadata.
    created_at : float
        Tenant creation timestamp.
    active : bool
        Whether the tenant is active.
    """

    tenant_id: str = ""
    display_name: str = ""
    model: str = ""
    token_budget: int = 0
    allowed_tools: Set[str] = field(default_factory=set)
    blocked_tools: Set[str] = field(default_factory=set)
    max_agents: int = 0
    max_tasks_per_minute: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    active: bool = True


# Wrap TenantConfig __init__ to support aliases
_TenantConfig_original_init = TenantConfig.__init__


def _tenantconfig_patched_init(self, *args, **kwargs):
    if "token_budget_daily" in kwargs:
        kwargs.setdefault("token_budget", kwargs.pop("token_budget_daily"))
    if "rate_limit_per_minute" in kwargs:
        kwargs.setdefault("max_tasks_per_minute", kwargs.pop("rate_limit_per_minute"))
    _TenantConfig_original_init(self, *args, **kwargs)


TenantConfig.__init__ = _tenantconfig_patched_init  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════════
# TENANT CONTEXT — runtime state per tenant
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TenantContext:
    """
    Runtime context for an active tenant.

    Provides isolation for:
    - Token usage tracking
    - Agent count tracking
    - Rate limiting
    - Knowledge/memory namespace prefixing
    """

    config: TenantConfig
    tokens_used: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    agent_count: int = 0
    _task_timestamps: List[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def tenant_id(self) -> str:
        return self.config.tenant_id

    @property
    def namespace(self) -> str:
        """Namespace prefix for knowledge/memory isolation."""
        return f"tenant:{self.config.tenant_id}:"

    def record_tokens(self, count: int) -> None:
        """Record token usage. Raises if budget exceeded."""
        with self._lock:
            if (
                self.config.token_budget > 0
                and self.tokens_used + count > self.config.token_budget
            ):
                raise TokenBudgetExceeded(
                    self.config.tenant_id,
                    self.config.token_budget,
                    self.tokens_used + count,
                )
            self.tokens_used += count

    def record_task(self, success: bool = True) -> None:
        """Record a completed task."""
        with self._lock:
            if success:
                self.tasks_completed += 1
            else:
                self.tasks_failed += 1
            self._task_timestamps.append(time.time())

    def check_rate_limit(self) -> bool:
        """Check if the tenant is within rate limits. Returns True if OK."""
        if self.config.max_tasks_per_minute <= 0:
            return True
        with self._lock:
            cutoff = time.time() - 60
            self._task_timestamps = [t for t in self._task_timestamps if t > cutoff]
            return len(self._task_timestamps) < self.config.max_tasks_per_minute

    def check_agent_limit(self) -> bool:
        """Check if the tenant can create more agents."""
        if self.config.max_agents <= 0:
            return True
        return self.agent_count < self.config.max_agents

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed for this tenant."""
        if self.config.blocked_tools and tool_name in self.config.blocked_tools:
            return False
        if self.config.allowed_tools and tool_name not in self.config.allowed_tools:
            return False
        return True

    def usage(self) -> Dict[str, Any]:
        """Get usage summary for this tenant."""
        return {
            "tenant_id": self.config.tenant_id,
            "tokens_used": self.tokens_used,
            "token_budget": self.config.token_budget,
            "tokens_remaining": (
                max(0, self.config.token_budget - self.tokens_used)
                if self.config.token_budget > 0
                else -1
            ),
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "active_agents": self.agent_count,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════════


class TenantError(Exception):
    """Base class for tenant errors."""

    pass


class TenantNotFound(TenantError):
    """Raised when a tenant_id doesn't exist."""

    def __init__(self, tenant_id: str):
        super().__init__(f"Tenant not found: {tenant_id!r}")
        self.tenant_id = tenant_id


class TokenBudgetExceeded(TenantError):
    """Raised when a tenant exceeds their token budget."""

    def __init__(self, tenant_id: str, budget: int, requested: int):
        super().__init__(
            f"Tenant {tenant_id!r} token budget exceeded: "
            f"budget={budget:,}, requested={requested:,}"
        )
        self.tenant_id = tenant_id
        self.budget = budget
        self.requested = requested


class RateLimitExceeded(TenantError):
    """Raised when a tenant hits rate limits."""

    def __init__(self, tenant_id: str, limit: int):
        super().__init__(
            f"Tenant {tenant_id!r} rate limit exceeded: " f"{limit} tasks/min"
        )
        self.tenant_id = tenant_id
        self.limit = limit


class AgentLimitExceeded(TenantError):
    """Raised when a tenant hits max agent count."""

    def __init__(self, tenant_id: str, limit: int):
        super().__init__(
            f"Tenant {tenant_id!r} agent limit exceeded: " f"max {limit} agents"
        )
        self.tenant_id = tenant_id
        self.limit = limit


# ═══════════════════════════════════════════════════════════════════════════════
# TENANT MANAGER
# ═══════════════════════════════════════════════════════════════════════════════


class TenantManager:
    """
    Manages tenant lifecycle: creation, configuration, usage tracking.

    Example
    -------
        manager = TenantManager()
        manager.create("acme-corp", model="gpt-4o-mini", token_budget=500_000)
        ctx = manager.get("acme-corp")
        ctx.record_tokens(1200)
        print(ctx.usage())
    """

    def __init__(self):
        self._tenants: Dict[str, TenantContext] = {}
        self._lock = threading.Lock()

    def create(
        self,
        tenant_id: str,
        display_name: Any = "",
        model: str = "",
        token_budget: int = 0,
        allowed_tools: Optional[Set[str]] = None,
        blocked_tools: Optional[Set[str]] = None,
        max_agents: int = 0,
        max_tasks_per_minute: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> TenantContext:
        """
        Create a new tenant.

        Parameters
        ----------
        tenant_id : str
            Unique tenant ID.
        display_name : str or TenantConfig
            Human-readable name, or a TenantConfig object.
        config : dict, optional
            Convenience: pass config keys as dict.
        """
        # Accept TenantConfig object as second positional arg
        if isinstance(display_name, TenantConfig):
            tc = display_name
            if not tc.tenant_id:
                tc.tenant_id = tenant_id
            if not tc.display_name:
                tc.display_name = tenant_id
            ctx = TenantContext(config=tc)
            with self._lock:
                self._tenants[tenant_id] = ctx
            return ctx

        display_name_str = display_name if isinstance(display_name, str) else ""

        # Merge config dict if provided
        if config:
            model = config.get("model", model)
            token_budget = config.get("token_budget", token_budget)
            max_agents = config.get("max_agents", max_agents)
            max_tasks_per_minute = config.get(
                "max_tasks_per_minute", max_tasks_per_minute
            )

        tc = TenantConfig(
            tenant_id=tenant_id,
            display_name=display_name_str or tenant_id,
            model=model,
            token_budget=token_budget,
            allowed_tools=allowed_tools or set(),
            blocked_tools=blocked_tools or set(),
            max_agents=max_agents,
            max_tasks_per_minute=max_tasks_per_minute,
            metadata=metadata or {},
        )

        ctx = TenantContext(config=tc)
        with self._lock:
            self._tenants[tenant_id] = ctx
        return ctx

    def get(self, tenant_id: str) -> TenantContext:
        """Get tenant context. Raises TenantNotFound if missing."""
        with self._lock:
            ctx = self._tenants.get(tenant_id)
        if ctx is None:
            raise TenantNotFound(tenant_id)
        return ctx

    def exists(self, tenant_id: str) -> bool:
        """Check if a tenant exists."""
        with self._lock:
            return tenant_id in self._tenants

    def list_tenants(self) -> List[str]:
        """List all tenant IDs."""
        with self._lock:
            return list(self._tenants.keys())

    def usage(self, tenant_id: str) -> Dict[str, Any]:
        """Get usage summary for a tenant."""
        return self.get(tenant_id).usage()

    def usage_all(self) -> List[Dict[str, Any]]:
        """Get usage summary for all tenants."""
        with self._lock:
            return [ctx.usage() for ctx in self._tenants.values()]

    def deactivate(self, tenant_id: str) -> None:
        """Deactivate a tenant (soft-delete)."""
        ctx = self.get(tenant_id)
        ctx.config.active = False

    def activate(self, tenant_id: str) -> None:
        """Re-activate a tenant."""
        ctx = self.get(tenant_id)
        ctx.config.active = True

    def delete(self, tenant_id: str) -> None:
        """Permanently remove a tenant."""
        with self._lock:
            if tenant_id in self._tenants:
                del self._tenants[tenant_id]

    def record_tokens(self, tenant_id: str, count: int) -> None:
        """Record token usage for a tenant."""
        self.get(tenant_id).record_tokens(count)

    def get_usage(self, tenant_id: str) -> Dict[str, Any]:
        """Get usage summary for a tenant. Alias for usage()."""
        return self.usage(tenant_id)

    def reset(self) -> None:
        """Reset all tenants. USE ONLY IN TESTS."""
        with self._lock:
            self._tenants.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_tenant_manager: Optional[TenantManager] = None


def get_tenant_manager() -> TenantManager:
    """Get module-level TenantManager singleton."""
    global _tenant_manager
    if _tenant_manager is None:
        _tenant_manager = TenantManager()
    return _tenant_manager
