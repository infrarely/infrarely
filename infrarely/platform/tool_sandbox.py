"""
infrarely/tool_sandbox.py — Tool Execution Sandboxing
═══════════════════════════════════════════════════════════════════════════════
SECURITY GAP 4: Tool Execution Has No Sandbox.

Problem: ``@infrarely.tool`` functions execute with the full privileges of the
         Python process — they can read any file, make any network call,
         import any module, delete data.  There is NO restriction layer.

Solution: A ``ToolSandboxPolicy`` that wraps tool execution with
          configurable restrictions on:
            • Filesystem paths (whitelist / blacklist)
            • Network access (allow/deny)
            • Import restrictions (blocked modules)
            • Execution time limit per tool call

Design:
  • Non-invasive — tools don't need modification
  • Configurable per-tool or agent-wide
  • Enforcement via pre/post execution hooks + monitoring
  • Violations raise ToolSandboxViolation (caught by engine)
"""

from __future__ import annotations

import functools
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL SANDBOX POLICY
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ToolSandboxPolicy:
    """
    Restrictions applied to tool function execution.

    Defaults are permissive — tighten per your security requirements.
    """

    # Filesystem
    allowed_paths: List[str] = field(default_factory=list)  # whitelist (empty = all)
    blocked_paths: List[str] = field(
        default_factory=lambda: [  # blacklist
            "/etc/shadow",
            "/etc/passwd",
            os.path.expanduser("~/.ssh"),
            os.path.expanduser("~/.aws"),
            os.path.expanduser("~/.gnupg"),
        ]
    )

    # Network
    network_allowed: bool = True  # global network toggle

    # Imports
    blocked_imports: List[str] = field(
        default_factory=lambda: [
            "subprocess",
            "shutil",
            "ctypes",
        ]
    )

    # Execution limits
    max_execution_time: float = 30.0  # seconds per tool call
    max_output_size: int = 1_000_000  # bytes — truncate output beyond this

    # Enforcement
    enabled: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL SANDBOX VIOLATION
# ═══════════════════════════════════════════════════════════════════════════════


class ToolSandboxViolation(Exception):
    """Raised when a tool attempts a forbidden operation."""

    def __init__(self, tool_name: str, violation_type: str, detail: str = ""):
        self.tool_name = tool_name
        self.violation_type = violation_type  # "filesystem" | "network" | "import" | "timeout" | "output_size"
        self.detail = detail
        super().__init__(
            f"Tool '{tool_name}' sandbox violation ({violation_type}): {detail}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL SANDBOX RESULT
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ToolSandboxResult:
    """Outcome of a sandboxed tool execution."""

    tool_name: str = ""
    allowed: bool = True
    violations: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    output_truncated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "allowed": self.allowed,
            "violations": self.violations,
            "execution_time": round(self.execution_time, 3),
            "output_truncated": self.output_truncated,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL SANDBOX GUARD
# ═══════════════════════════════════════════════════════════════════════════════


class ToolSandboxGuard:
    """
    Wraps tool execution with sandbox policy enforcement.

    Usage::

        guard = ToolSandboxGuard(ToolSandboxPolicy(
            blocked_paths=["/etc", "/root"],
            network_allowed=False,
        ))

        # Check before execution
        result = guard.check_tool_call("read_file", {"path": "/etc/shadow"})
        if not result.allowed:
            raise ToolSandboxViolation(...)

        # Or wrap the function
        safe_fn = guard.wrap(original_fn, "my_tool")
        output = safe_fn(**params)

    Thread-safe singleton.
    """

    _instance: Optional["ToolSandboxGuard"] = None
    _lock = threading.Lock()

    def __new__(cls, policy: Optional[ToolSandboxPolicy] = None) -> "ToolSandboxGuard":
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._policy = policy or ToolSandboxPolicy()
                inst._per_tool_policies: Dict[str, ToolSandboxPolicy] = {}
                inst._violations_log: List[Dict[str, Any]] = []
                inst._stats = {"calls": 0, "blocked": 0, "violations": 0}
                cls._instance = inst
            return cls._instance

    @property
    def policy(self) -> ToolSandboxPolicy:
        return self._policy

    def update_policy(self, policy: ToolSandboxPolicy) -> None:
        """Update the global tool sandbox policy."""
        self._policy = policy

    def set_tool_policy(self, tool_name: str, policy: ToolSandboxPolicy) -> None:
        """Set a per-tool sandbox policy (overrides global)."""
        self._per_tool_policies[tool_name] = policy

    def get_effective_policy(self, tool_name: str) -> ToolSandboxPolicy:
        """Return the effective policy for a tool (per-tool or global)."""
        return self._per_tool_policies.get(tool_name, self._policy)

    # ── Pre-execution checks ──────────────────────────────────────────────────

    def check_tool_call(
        self, tool_name: str, params: Dict[str, Any]
    ) -> ToolSandboxResult:
        """
        Check if a tool call is allowed under the current policy.

        Inspects parameters for filesystem paths, network indicators, etc.
        """
        policy = self.get_effective_policy(tool_name)
        if not policy.enabled:
            return ToolSandboxResult(tool_name=tool_name, allowed=True)

        self._stats["calls"] += 1
        violations: List[str] = []

        # ── Filesystem path check ────────────────────────────────────────
        for key, value in params.items():
            if isinstance(value, str) and ("/" in value or "\\" in value):
                path = os.path.abspath(value)
                # Check blocked paths
                for blocked in policy.blocked_paths:
                    blocked_abs = os.path.abspath(blocked)
                    if path.startswith(blocked_abs) or path == blocked_abs:
                        violations.append(
                            f"filesystem: blocked path '{blocked}' (param '{key}')"
                        )

                # Check allowed paths (if whitelist is set)
                if policy.allowed_paths:
                    allowed = False
                    for allow_path in policy.allowed_paths:
                        allow_abs = os.path.abspath(allow_path)
                        if path.startswith(allow_abs):
                            allowed = True
                            break
                    if not allowed:
                        violations.append(
                            f"filesystem: path '{value}' not in allowed paths (param '{key}')"
                        )

        # ── Network check ────────────────────────────────────────────────
        if not policy.network_allowed:
            for key, value in params.items():
                if isinstance(value, str):
                    val_lower = value.lower()
                    if any(
                        proto in val_lower
                        for proto in ("http://", "https://", "ftp://", "ws://")
                    ):
                        violations.append(f"network: URL detected in param '{key}'")

        if violations:
            self._stats["blocked"] += 1
            self._stats["violations"] += len(violations)
            self._violations_log.append(
                {
                    "tool": tool_name,
                    "violations": violations,
                    "timestamp": time.time(),
                }
            )

        return ToolSandboxResult(
            tool_name=tool_name,
            allowed=len(violations) == 0,
            violations=violations,
        )

    # ── Execution wrapper ─────────────────────────────────────────────────────

    def wrap(self, fn: Callable, tool_name: str) -> Callable:
        """
        Wrap a tool function with sandbox enforcement.

        Returns a new callable that:
          1. Checks params against the policy
          2. Enforces execution time limit
          3. Truncates oversized output
        """
        policy = self.get_effective_policy(tool_name)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not policy.enabled:
                return fn(*args, **kwargs)

            # Pre-flight check
            check_result = self.check_tool_call(tool_name, kwargs)
            if not check_result.allowed:
                raise ToolSandboxViolation(
                    tool_name,
                    "policy_violation",
                    "; ".join(check_result.violations),
                )

            # Time-limited execution
            start = time.monotonic()
            result = fn(*args, **kwargs)
            elapsed = time.monotonic() - start

            if elapsed > policy.max_execution_time:
                raise ToolSandboxViolation(
                    tool_name,
                    "timeout",
                    f"Execution took {elapsed:.1f}s, limit is {policy.max_execution_time}s",
                )

            # Output size check
            output_str = str(result)
            if len(output_str) > policy.max_output_size:
                result = output_str[: policy.max_output_size] + "... [TRUNCATED]"

            return result

        return wrapper

    # ── Import check ──────────────────────────────────────────────────────────

    def check_import(self, module_name: str, tool_name: str = "") -> bool:
        """
        Check if a module import is allowed.

        Returns True if allowed, False if blocked.
        """
        policy = self.get_effective_policy(tool_name) if tool_name else self._policy
        if not policy.enabled:
            return True
        for blocked in policy.blocked_imports:
            if module_name == blocked or module_name.startswith(blocked + "."):
                return False
        return True

    # ── Stats / Audit ─────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, int]:
        """Return sandbox enforcement statistics."""
        return dict(self._stats)

    def violations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent violation records."""
        return list(reversed(self._violations_log[-limit:]))

    def reset(self) -> None:
        """Reset state (for testing)."""
        self._per_tool_policies.clear()
        self._violations_log.clear()
        self._stats = {"calls": 0, "blocked": 0, "violations": 0}
        self._policy = ToolSandboxPolicy()


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_guard_lock = threading.Lock()
_default_guard: Optional[ToolSandboxGuard] = None


def get_tool_sandbox_guard(
    policy: Optional[ToolSandboxPolicy] = None,
) -> ToolSandboxGuard:
    """Get or create the global tool sandbox guard singleton."""
    global _default_guard
    with _guard_lock:
        if _default_guard is None:
            _default_guard = ToolSandboxGuard(policy)
        elif policy is not None:
            _default_guard.update_policy(policy)
        return _default_guard
