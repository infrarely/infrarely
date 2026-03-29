"""
adaptive/safety_controller.py — Module 9: Safety Controller
═══════════════════════════════════════════════════════════════════════════════
Enforces deterministic safety constraints on all adaptive behaviour.

Invariants (NEVER overridden):
  • No new tools auto-registered
  • No capability graph mutation during execution
  • No routing override if confidence < 0.8
  • All adaptive changes pass verification

The SafetyController acts as a gatekeeper that Layer 5 modules
consult before applying any adaptive change.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from infrarely.observability import logger


@dataclass
class SafetyViolation:
    """Record of a blocked adaptive action."""

    module: str
    action: str
    reason: str
    blocked: bool = True


class SafetyController:
    """
    Central safety gatekeeper for Layer 5 adaptive intelligence.
    All adaptive modules must check with this controller before applying changes.

    Constraints:
      1. No auto-registration of tools
      2. No graph mutation during execution
      3. Routing override only if confidence ≥ 0.8
      4. Parameter inference only if confidence ≥ 0.75
      5. Capability discovery max = 50
      6. Skill memory max = 200
      7. LLM never called by any Layer 5 module
    """

    ROUTING_OVERRIDE_MIN_CONF = 0.80
    PARAM_INFERENCE_MIN_CONF = 0.75
    MAX_CAPABILITIES = 50
    MAX_SKILLS = 200

    def __init__(self):
        self._violations: List[SafetyViolation] = []
        self._checks_performed = 0
        self._checks_blocked = 0
        self._executing = False  # set True during capability execution

    # ── Execution lifecycle ───────────────────────────────────────────────────
    def enter_execution(self):
        """Mark execution start — blocks graph mutations."""
        self._executing = True

    def exit_execution(self):
        """Mark execution end — allows graph mutations."""
        self._executing = False

    # ── Safety checks ─────────────────────────────────────────────────────────
    def can_override_routing(self, confidence: float) -> bool:
        """Check if routing can be overridden with given confidence."""
        self._checks_performed += 1
        if confidence < self.ROUTING_OVERRIDE_MIN_CONF:
            self._record_violation(
                "routing_optimizer",
                f"override with conf={confidence:.2f}",
                f"Below minimum {self.ROUTING_OVERRIDE_MIN_CONF}",
            )
            return False
        return True

    def can_infer_parameter(self, param_name: str, confidence: float) -> bool:
        """Check if a parameter can be auto-filled."""
        self._checks_performed += 1
        if confidence < self.PARAM_INFERENCE_MIN_CONF:
            self._record_violation(
                "parameter_inference",
                f"infer '{param_name}' with conf={confidence:.2f}",
                f"Below minimum {self.PARAM_INFERENCE_MIN_CONF}",
            )
            return False
        return True

    def can_mutate_graph(self) -> bool:
        """Check if capability graph can be modified."""
        self._checks_performed += 1
        if self._executing:
            self._record_violation(
                "capability_optimizer",
                "mutate graph",
                "Graph mutation blocked during execution",
            )
            return False
        return True

    def can_register_tool(self) -> bool:
        """Auto tool registration is ALWAYS blocked."""
        self._checks_performed += 1
        self._record_violation(
            "capability_discovery",
            "auto-register tool",
            "Auto tool registration permanently disabled",
        )
        return False

    def can_add_capability(self, current_count: int) -> bool:
        """Check if a new capability can be added."""
        self._checks_performed += 1
        if current_count >= self.MAX_CAPABILITIES:
            self._record_violation(
                "capability_discovery",
                f"add capability (count={current_count})",
                f"Max capabilities ({self.MAX_CAPABILITIES}) reached",
            )
            return False
        return True

    def can_add_skill(self, current_count: int) -> bool:
        """Check if a new skill can be added."""
        self._checks_performed += 1
        if current_count >= self.MAX_SKILLS:
            self._record_violation(
                "skill_memory",
                f"add skill (count={current_count})",
                f"Max skills ({self.MAX_SKILLS}) reached",
            )
            return False
        return True

    # ── Violation tracking ────────────────────────────────────────────────────
    def _record_violation(self, module: str, action: str, reason: str):
        self._checks_blocked += 1
        v = SafetyViolation(module=module, action=action, reason=reason)
        self._violations.append(v)
        if len(self._violations) > 500:
            self._violations = self._violations[-250:]
        logger.debug(f"SafetyController BLOCKED: [{module}] {action} — {reason}")

    # ── Query ─────────────────────────────────────────────────────────────────
    def recent_violations(self, n: int = 10) -> List[Dict[str, Any]]:
        return [
            {"module": v.module, "action": v.action, "reason": v.reason}
            for v in self._violations[-n:]
        ]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "executing": self._executing,
            "checks_performed": self._checks_performed,
            "checks_blocked": self._checks_blocked,
            "block_rate": (
                f"{self._checks_blocked / self._checks_performed:.1%}"
                if self._checks_performed
                else "0%"
            ),
            "recent_violations": self.recent_violations(5),
            "constraints": {
                "routing_override_min_conf": self.ROUTING_OVERRIDE_MIN_CONF,
                "param_inference_min_conf": self.PARAM_INFERENCE_MIN_CONF,
                "max_capabilities": self.MAX_CAPABILITIES,
                "max_skills": self.MAX_SKILLS,
                "auto_tool_registration": "PERMANENTLY DISABLED",
                "graph_mutation_during_exec": "BLOCKED",
            },
        }
