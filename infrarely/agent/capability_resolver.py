"""
agent/capability_resolver.py  — Capability Resolver
═══════════════════════════════════════════════════════════════════════════════
Maps reasoning engine output + intent to a specific Capability.

Sits between the Reasoning Engine and the Capability Compiler:
  Router → IntentMatch → ReasoningEngine → ReasoningResult
    → CapabilityResolver → Capability (or None for single-tool fallback)
    → CapabilityCompiler → CapabilityPlan → Executor

Resolution strategy
────────────────────
  1. If reasoning engine recommends specific steps, look up capabilities
     whose step tool names match those recommendations.
  2. If intent matches a registered capability tag, select that capability.
  3. If neither resolves, return None → single-tool routing proceeds as before.

Design contract
───────────────
  • No LLM calls — purely rule-based matching
  • O(n_capabilities × n_steps) worst case — fast for ≤20 capabilities
  • Returns Optional[Capability] — caller must handle None
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from infrarely.agent.reasoning_engine import ReasoningResult
from infrarely.capabilities.registry import CapabilityRegistry
from infrarely.agent.capability import Capability
from infrarely.observability import logger


# ── Direct intent → capability mapping (CP1 fix) ─────────────────────────────
_INTENT_TO_CAPABILITY = {
    "exam_topics": "exam_preparation",
    "practice_questions": "exam_preparation",
    "study_schedule": "weekly_planning",
    "calendar_week": "weekly_planning",
    "note_search": "study_session",
    "study_session": "study_session",
}


@dataclass
class ResolverResult:
    """Outcome of resolution."""

    capability: Optional[Capability] = None
    resolution_path: str = "none"  # "reasoning" | "intent" | "none"
    matched_rules: List[str] = field(default_factory=list)
    confidence: float = 0.0


class CapabilityResolver:
    """
    Resolves a ReasoningResult + query into a Capability for execution.

    Two-pass resolution:
      Pass 1 — reasoning-engine recommendations → capability step match
      Pass 2 — fallback intent-tag match on the raw query
    """

    def __init__(self, capability_registry: CapabilityRegistry):
        self._registry = capability_registry

    def resolve(
        self,
        reasoning: ReasoningResult,
        query: str,
        intent: str = "",
        params: Dict[str, Any] = None,
    ) -> ResolverResult:
        """
        Attempt to resolve to a Capability.

        Priority:
          1. Reasoning engine recommended steps → match to capability
          2. Raw query → capability intent tag match (existing registry.match)
        """
        # ── Pass 0: direct "capability:xxx" from reasoning actions ────────────
        for action in reasoning.recommended_steps:
            if action.startswith("capability:"):
                cap_name = action.split(":", 1)[1]
                cap = self._registry.get(cap_name)
                if cap:
                    logger.debug(
                        f"CapabilityResolver: resolved '{cap_name}' via direct reasoning action",
                        triggered_rules=reasoning.triggered_rules,
                    )
                    return ResolverResult(
                        capability=cap,
                        resolution_path="reasoning_direct",
                        matched_rules=reasoning.triggered_rules,
                        confidence=0.90,
                    )

        # ── Pass 1: reasoning-engine match ────────────────────────────────────
        if reasoning.triggered_rules:
            cap = self._match_from_reasoning(reasoning)
            if cap:
                logger.debug(
                    f"CapabilityResolver: resolved '{cap.name}' via reasoning",
                    triggered_rules=reasoning.triggered_rules,
                )
                return ResolverResult(
                    capability=cap,
                    resolution_path="reasoning",
                    matched_rules=reasoning.triggered_rules,
                    confidence=0.85,
                )

        # ── Pass 2: intent-tag match on raw query ─────────────────────────────
        cap = self._registry.match(query)
        if cap:
            logger.debug(
                f"CapabilityResolver: resolved '{cap.name}' via intent tag",
            )
            return ResolverResult(
                capability=cap,
                resolution_path="intent",
                confidence=0.75,
            )

        # ── Pass 3: direct intent → capability mapping ────────────────────────
        if intent in _INTENT_TO_CAPABILITY:
            cap_name = _INTENT_TO_CAPABILITY[intent]
            cap = self._registry.get(cap_name)
            if cap:
                logger.debug(
                    f"CapabilityResolver: resolved '{cap_name}' via intent map ('{intent}')",
                )
                return ResolverResult(
                    capability=cap,
                    resolution_path="intent_map",
                    confidence=0.70,
                )

        # ── No match — single-tool routing ────────────────────────────────────
        logger.debug("CapabilityResolver: no capability matched — single-tool path")
        return ResolverResult()

    def _match_from_reasoning(self, reasoning: ReasoningResult) -> Optional[Capability]:
        """
        Map reasoning recommended_steps to a capability.

        The reasoning engine produces action strings like:
          "load_student_profile_first", "ask_user_for_course_id"
        We check if any capability has steps whose tool_name or step.name
        aligns with the recommended actions.
        """
        # Strategy: for each capability, count how many reasoning actions
        # relate to its steps.  Return the capability with most matches.
        best_cap: Optional[Capability] = None
        best_score: int = 0

        for cap_name in self._registry.names():
            cap = self._registry.get(cap_name)
            if not cap:
                continue

            score = 0
            step_names = {s.name for s in cap.steps}
            tool_names = {s.tool_name for s in cap.steps}

            for action in reasoning.recommended_steps:
                action_lower = action.lower()
                # Direct step name match
                if action_lower in step_names:
                    score += 2
                # Tool name substring match
                for tn in tool_names:
                    if tn in action_lower or action_lower in tn:
                        score += 1

            if score > best_score:
                best_score = score
                best_cap = cap

        return best_cap if best_score > 0 else None
