"""
agent/reasoning_engine.py  — Gap 6: Deterministic Reasoning Engine
Rule-based goal inference. No LLM. Pure if-then logic.

The reasoning engine sits between the router and the capability executor.
It evaluates the current context against a rule table and emits
a list of precondition checks before capabilities execute.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from infrarely.observability import logger


@dataclass
class Rule:
    """IF condition(context) THEN action is recommended."""
    name:      str
    condition: Callable[[Dict[str, Any]], bool]
    action:    str         # human-readable action label
    priority:  int = 0     # higher = evaluated first


@dataclass
class ReasoningResult:
    triggered_rules:   List[str]
    blocked_rules:     List[str]
    recommended_steps: List[str]
    context_flags:     Dict[str, bool] = field(default_factory=dict)


# ── default rule table ────────────────────────────────────────────────────────
_DEFAULT_RULES: List[Rule] = [
    # ── Capability routing rules (CP6 — map intents to capabilities) ──────
    Rule(
        name      = "route_to_exam_preparation",
        condition = lambda ctx: (
            ctx.get("intent") in ("exam_topics", "practice_questions")
            or any(kw in (ctx.get("query") or "").lower()
                   for kw in ("exam prep", "prepare for exam", "exam coming",
                              "help me prepare", "study for exam", "revise for",
                              "get ready for exam", "exam preparation"))
        ),
        action    = "capability:exam_preparation",
        priority  = 20,
    ),
    Rule(
        name      = "route_to_weekly_planning",
        condition = lambda ctx: (
            ctx.get("intent") == "study_schedule"
            or any(kw in (ctx.get("query") or "").lower()
                   for kw in ("plan my week", "weekly overview", "week ahead",
                              "weekly plan", "this week overview",
                              "help me plan this week",
                              "what do i have this week"))
        ),
        action    = "capability:weekly_planning",
        priority  = 19,
    ),
    Rule(
        name      = "route_to_study_session",
        condition = lambda ctx: (
            any(kw in (ctx.get("query") or "").lower()
                for kw in ("study session", "help me study", "let's study",
                           "study with me", "i want to study",
                           "start a study session"))
        ),
        action    = "capability:study_session",
        priority  = 18,
    ),
    # ── Precondition rules (existing) ─────────────────────────────────────
    Rule(
        name      = "exam_requires_course",
        condition = lambda ctx: ctx.get("intent") == "exam_preparation"
                               and not ctx.get("course_id"),
        action    = "ask_user_for_course_id",
        priority  = 10,
    ),
    Rule(
        name      = "schedule_requires_profile",
        condition = lambda ctx: ctx.get("intent") in ("study_session", "weekly_planning")
                               and not ctx.get("profile_loaded"),
        action    = "load_student_profile_first",
        priority  = 9,
    ),
    Rule(
        name      = "practice_requires_topic",
        condition = lambda ctx: ctx.get("intent") == "practice_questions"
                               and not ctx.get("topic"),
        action    = "ask_user_for_topic",
        priority  = 8,
    ),
    Rule(
        name      = "generative_requires_llm_online",
        condition = lambda ctx: ctx.get("requires_llm") and not ctx.get("llm_online", True),
        action    = "warn_llm_offline_fallback_deterministic",
        priority  = 7,
    ),
]


class DeterministicReasoningEngine:
    """
    Evaluates rules against a context dict and returns
    ReasoningResult with triggered rules and recommended steps.
    """

    def __init__(self, rules: List[Rule] = None):
        self._rules = sorted(rules or _DEFAULT_RULES, key=lambda r: -r.priority)

    def evaluate(self, context: Dict[str, Any]) -> ReasoningResult:
        triggered, blocked, steps = [], [], []
        for rule in self._rules:
            try:
                fired = rule.condition(context)
            except Exception as e:
                logger.warn(f"ReasoningEngine: rule '{rule.name}' raised {e}")
                fired = False
            if fired:
                triggered.append(rule.name)
                steps.append(rule.action)
                logger.debug(f"ReasoningEngine: rule '{rule.name}' TRIGGERED → {rule.action}")
            else:
                blocked.append(rule.name)
        return ReasoningResult(
            triggered_rules   = triggered,
            blocked_rules     = blocked,
            recommended_steps = steps,
        )

    def add_rule(self, rule: Rule):
        self._rules.append(rule)
        self._rules.sort(key=lambda r: -r.priority)


_engine = DeterministicReasoningEngine()

def evaluate(context: Dict[str, Any]) -> ReasoningResult:
    return _engine.evaluate(context)