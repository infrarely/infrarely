"""
router/tool_router.py
═══════════════════════════════════════════════════════════════════════════════
Central dispatch layer.  Emits an ExecutionPlan — the sole authority
governing LLM usage downstream.

Strategic design
─────────────────
  The router NEVER falls back to LLM for unknown queries.
  Unknown queries → StaticResponder("unknown") → 0 tokens, offline-safe.

  The ONLY way an LLM call reaches the system is:
    a) practice_question_generator (TOOL_GENERATIVE) — scoped, 1 call, in tool
    b) A future explicit GENERATIVE intent added with deliberate justification

  This means /run 1 (Basic Interaction) uses 0 tokens, even for "Hello".

ExecutionPlan invariants
─────────────────────────
  DETERMINISTIC   → allow_llm = False (guaranteed)
  TOOL_GENERATIVE → allow_llm = False (tool already called LLM once)
  is_complete=True → allow_llm = False (Gate A in core)
"""

from __future__ import annotations
from typing import Tuple

from infrarely.agent.state import ExecutionPlan, ResponseType, TaskState, ToolResult
from infrarely.router.intent_classifier import IntentClassifier, IntentMatch
from infrarely.tools.registry import ToolRegistry
from infrarely.observability import logger
from infrarely.observability.metrics import collector
import infrarely.core.app_config as config


class ToolRouter:
    def __init__(self, registry: ToolRegistry, app_cfg):
        self._classifier = IntentClassifier()
        self._registry   = registry
        self._app_cfg    = app_cfg

    def classify_only(self, query: str) -> Tuple[IntentMatch, TaskState]:
        """
        Classify intent without executing any tool.
        Used by core.py for capability-first routing.
        Returns (IntentMatch, TaskState).
        """
        match = self._classifier.classify(query)
        collector.record_intent(match.intent)
        logger.router_log(
            match.intent, match.tool, match.confidence,
            response_type=match.response_type.name,
        )
        state = TaskState(
            task         = match.intent,
            tool         = match.tool,
            params       = match.params,
            requires_llm = match.requires_llm,
            confidence   = match.confidence,
            raw_input    = query,
            student_id   = self._app_cfg.student_id,
        )
        return match, state

    def execute_tool(self, match: IntentMatch, state: TaskState) -> Tuple[ExecutionPlan, ToolResult]:
        """
        Execute the tool for a pre-classified intent.
        Returns (ExecutionPlan, ToolResult).
        """
        tool = self._registry.get(match.tool)

        if tool is None:
            logger.warn(f"Tool '{match.tool}' not in registry — using static fallback")
            static_tool = self._registry.get("static_responder")
            if static_tool:
                fallback_state = TaskState(
                    task="unknown_query", tool="static_responder",
                    params={"variant": "unknown"},
                    student_id=self._app_cfg.student_id,
                )
                result = static_tool.run(fallback_state)
            else:
                result = ToolResult(
                    success=True, tool_name="static_responder",
                    data="I'm not sure how to help with that. Type /help for a list of capabilities.",
                    is_complete=True, response_type=ResponseType.DETERMINISTIC,
                )
            plan = self._make_plan(state, result, match)
            return plan, result

        self._app_cfg.tool_call_count_session += 1
        result = tool.run(state)

        if (result.metadata.get("llm_used")
                and match.response_type == ResponseType.DETERMINISTIC):
            result.response_type = ResponseType.TOOL_GENERATIVE
            result.is_complete   = True

        plan = self._make_plan(state, result, match)
        return plan, result

    def route(self, query: str) -> Tuple[ExecutionPlan, ToolResult]:
        """Classify + execute in one call (backward-compatible)."""
        match, state = self.classify_only(query)
        return self.execute_tool(match, state)

    def _make_plan(
        self,
        state:  TaskState,
        result: ToolResult,
        match:  IntentMatch,
    ) -> ExecutionPlan:
        """
        Determine final allow_llm and response_type.

        Priority (descending):
          a) result.is_complete → NEVER allow LLM
          b) DETERMINISTIC or TOOL_GENERATIVE → never allow LLM
          c) GENERATIVE → allow LLM (one call max, enforced in core)
             NOTE: No current rule emits GENERATIVE, so this is unused in
             standard workflows.
        """
        rtype = (
            result.response_type
            if result.response_type != ResponseType.DETERMINISTIC
            else match.response_type
        )

        if result.is_complete:
            rtype     = result.response_type
            allow_llm = False
        elif rtype in (ResponseType.DETERMINISTIC, ResponseType.TOOL_GENERATIVE):
            allow_llm = False
        elif rtype == ResponseType.GENERATIVE:
            allow_llm = True
        else:
            allow_llm = False

        plan = ExecutionPlan(
            task_state    = state,
            response_type = rtype,
            allow_llm     = allow_llm,
            tool_name     = result.tool_name,
        )
        logger.debug(
            "ExecutionPlan",
            allow_llm     = allow_llm,
            response_type = rtype.name,
            is_complete   = result.is_complete,
        )
        return plan