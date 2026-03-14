"""
capabilities/registry.py  — Layer 2
═══════════════════════════════════════════════════════════════════════════════
CapabilityRegistry: maps intent strings to Capability definitions.

The router queries this registry first.
If a match is found, it returns a CapabilityPlan.
If no match, it falls through to the existing single-tool routing (unchanged).

This means Layer 2 is purely additive — single-tool routing is untouched.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

from infrarely.agent.capability import Capability, CapabilityPlan, CapabilityStep
from infrarely.agent.state import TaskState
from infrarely.observability import logger


class CapabilityRegistry:
    """
    Holds all registered Capabilities and provides O(n_tags) intent matching.

    Matching strategy
    ──────────────────
      Any intent_tag that appears as a substring of the lowercased query
      triggers a capability match.  The first match wins (declaration order).
      This is intentionally simple — the router already handles ambiguity
      resolution for single tools; capabilities claim distinct intent surfaces.
    """

    def __init__(self):
        self._capabilities: Dict[str, Capability] = {}
        self._tag_index: Dict[str, str] = {}  # tag → capability name

    def register(self, capability: Capability) -> "CapabilityRegistry":
        self._capabilities[capability.name] = capability
        for tag in capability.intent_tags:
            self._tag_index[tag.lower()] = capability.name
        logger.debug(
            f"CapabilityRegistry: registered '{capability.name}'",
            steps=len(capability.steps),
            tags=len(capability.intent_tags),
        )
        return self

    def match(self, query: str) -> Optional[Capability]:
        """
        Return the first Capability whose intent_tags match the query.
        Returns None if no capability matches — router falls back to single tool.
        """
        lower = query.lower()
        for tag, cap_name in self._tag_index.items():
            if tag in lower:
                cap = self._capabilities[cap_name]
                logger.debug(
                    f"CapabilityRegistry: matched '{cap_name}' via tag '{tag}'"
                )
                return cap
        return None

    def build_plan(
        self,
        capability: Capability,
        task_state: TaskState,
        extra_params: Dict = None,
    ) -> CapabilityPlan:
        """
        Construct a CapabilityPlan, injecting query-extracted params into steps.

        extra_params (e.g. {"course_id": "CS301", "topic": "trees"}) are
        merged into each step's base_params where the step doesn't already
        define that key.  This lets the router's param extraction (course_id,
        topic, etc.) flow into capability steps without coupling the
        capability definition to the router.
        """
        if extra_params:
            # Return a plan with enriched steps (non-mutating — creates new steps)
            enriched_steps = []
            for step in capability.steps:
                merged = dict(extra_params)
                merged.update(step.base_params)  # step params win over query params
                enriched_steps.append(
                    CapabilityStep(
                        name=step.name,
                        tool_name=step.tool_name,
                        base_params=merged,
                        failure_policy=step.failure_policy,
                        condition=step.condition,
                        description=step.description,
                    )
                )
            enriched_cap = Capability(
                name=capability.name,
                description=capability.description,
                steps=enriched_steps,
                intent_tags=capability.intent_tags,
            )
        else:
            enriched_cap = capability

        return CapabilityPlan(
            capability=enriched_cap,
            task_state=task_state,
            initial_context=dict(task_state.params),
        )

    def get(self, name: str) -> Optional[Capability]:
        return self._capabilities.get(name)

    def get_meta(self, name: str) -> Optional[Capability]:
        """Return the Capability for *name*, or None.  Mirrors ToolRegistry.get_meta."""
        return self._capabilities.get(name)

    def names(self) -> List[str]:
        return list(self._capabilities.keys())

    def __len__(self) -> int:
        return len(self._capabilities)

    def __contains__(self, name: str) -> bool:
        return name in self._capabilities


def build_student_capability_registry() -> CapabilityRegistry:
    """Factory — creates and registers all student capabilities."""
    from infrarely.capabilities.student_capabilities import ALL_CAPABILITIES

    registry = CapabilityRegistry()
    for cap in ALL_CAPABILITIES:
        registry.register(cap)
    logger.info(
        "CapabilityRegistry built",
        count=len(registry),
        capabilities=registry.names(),
    )
    return registry
