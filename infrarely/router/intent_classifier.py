"""
router/intent_classifier.py
═══════════════════════════════════════════════════════════════════════════════
Rule-based intent classifier.  Zero LLM tokens.  Zero network calls.

Strategic design principle
───────────────────────────
  GENERATIVE is the EXCEPTION, not the default.
  Every intent that has a deterministic answer MUST be DETERMINISTIC.
  LLM is called ONLY when language must be generated that cannot be
  precomputed — currently only practice_questions (via TOOL_GENERATIVE).

  Before this refactor:
    greeting → GENERATIVE (LLM) ← waste
    unknown  → GENERATIVE (LLM) ← waste + breaks when LLM offline
    courses  → llm_general      ← miss

  After:
    greeting → StaticResponder (DETERMINISTIC, 0 tokens)
    help     → StaticResponder (DETERMINISTIC, 0 tokens)
    unknown  → StaticResponder (DETERMINISTIC, 0 tokens)
    courses  → student_profile_manager or course_material_search
    practice → PracticeQuestionGenerator (TOOL_GENERATIVE, 1 LLM call in tool)

  The only path that reaches `llm_client.llm_call` in the CORE is
  truly open-ended summarisation — which is not exercised by any
  standard student workflow.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import infrarely.core.app_config as config
from infrarely.agent.state import ResponseType

# ── Rule schema ───────────────────────────────────────────────────────────────
# (intent, tool, triggers, boosts, default_params, response_type)
#
# ResponseType contract per rule:
#   DETERMINISTIC   → tool data IS the complete answer (0 LLM tokens, always)
#   TOOL_GENERATIVE → tool calls LLM internally once; core never calls LLM
#   GENERATIVE      → RESERVED; not used for any known student workflow
#
RouteRule = Tuple[str, str, List[str], List[str], Dict, ResponseType]

_RULES: List[RouteRule] = []  # populated by user via infrarely.route()


def register_route(
    intent: str,
    tool: str,
    triggers: List[str],
    response_type: ResponseType = ResponseType.DETERMINISTIC,
    params: Optional[Dict] = None,
) -> None:
    _RULES.append(
        (
            intent,
            tool,
            triggers,
            [],
            dict(params or {}),
            response_type,
        )
    )


@dataclass
class IntentMatch:
    intent: str
    tool: str
    confidence: float
    params: Dict = field(default_factory=dict)
    response_type: ResponseType = ResponseType.DETERMINISTIC

    @property
    def requires_llm(self) -> bool:
        """Derived — no separate bool field, no dual source of truth."""
        return self.response_type in (
            ResponseType.GENERATIVE,
            ResponseType.TOOL_GENERATIVE,
        )


# ── Unknown-query fallback ────────────────────────────────────────────────────
def _unknown_match() -> IntentMatch:
    return IntentMatch(
        intent="unknown_query",
        tool="__unresolved__",
        confidence=0.0,
        params={},
        response_type=ResponseType.DETERMINISTIC,
    )


class IntentClassifier:
    """
    Trigger/boost scoring classifier.
    O(n_rules) per query.  Zero external dependencies.

    Scoring:
      ANY trigger matches  → base score 0.65
      Each boost keyword   → +0.15 (capped at 1.0)

    Fallback: StaticResponder("unknown") — NOT llm_general.
    Rationale: an unknown query should get a helpful canned response,
    not an expensive LLM call that fails when offline.
    """

    def __init__(self):
        self._rules = _RULES

    def classify(self, text: str) -> IntentMatch:
        lower = text.lower()
        best: Optional[IntentMatch] = None
        best_score: float = 0.0

        for intent_name, tool, triggers, boosts, default_params, rtype in self._rules:
            if not any(t in lower for t in triggers):
                continue

            score = 0.65 + min(
                sum(0.15 for b in boosts if b in lower),
                0.35,
            )

            if score > best_score:
                best_score = score
                params = self._extract_params(lower, dict(default_params))
                best = IntentMatch(
                    intent=intent_name,
                    tool=tool,
                    confidence=score,
                    params=params,
                    response_type=rtype,
                )

        if best is None or best_score < config.ROUTER_CONFIDENCE_THRESHOLD:
            return _unknown_match()

        return best

    def _extract_params(self, lower: str, base: dict) -> dict:
        params = dict(base)

        m = re.search(r"\b([A-Z]{2,4}\d{3,4})\b", lower.upper())
        if m:
            params["course_id"] = m.group(1)

        m = re.search(r"(?:next|in)\s+(\d+)\s+days?", lower)
        if m:
            params["days"] = int(m.group(1))

        m = re.search(r"\b(?:about|on|for)\s+([\w\s]{3,40})", lower)
        if m:
            topic_val = m.group(1).strip()
            # Don't treat course IDs (e.g. "cs301") as topics
            if not re.match(r"^[a-z]{2,4}\d{3,4}$", topic_val):
                params["topic"] = topic_val

        m = re.search(r"(?:search|find|look for)\s+([\w\s]{3,40})", lower)
        if m:
            params["query"] = m.group(1).strip()

        for diff in ("easy", "medium", "hard"):
            if diff in lower:
                params["difficulty"] = diff
                break

        return params
