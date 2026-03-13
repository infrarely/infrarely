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
_RULES: List[Tuple] = [
    # ── Greetings / Help → StaticResponder (0 tokens, offline-safe) ───────────
    (
        "greeting",
        "static_responder",
        [
            "hello",
            "hi ",
            "hey ",
            "good morning",
            "good evening",
            "good afternoon",
            "howdy",
            "what's up",
            "sup ",
        ],
        [],
        {"variant": "greeting"},
        ResponseType.DETERMINISTIC,
    ),
    # ── Exam preparation (BEFORE generic help so 'help me prepare' wins) ──────
    (
        "exam_preparation",
        "exam_topic_predictor",
        [
            "prepare for exam",
            "prepare for my exam",
            "help me prepare",
            "exam prep",
            "study for exam",
            "study for my exam",
            "revise for",
            "get ready for exam",
            "exam coming",
            "ready for my exam",
            "prepare for my",
        ],
        ["exam", "prepare", "study", "revise"],
        {},
        ResponseType.DETERMINISTIC,
    ),
    (
        "help",
        "static_responder",
        [
            "help",
            "what can you do",
            "what do you do",
            "commands",
            "how do i",
            "capabilities",
        ],
        [],
        {"variant": "help"},
        ResponseType.DETERMINISTIC,
    ),
    # ── Calendar ──────────────────────────────────────────────────────────────
    (
        "calendar_today",
        "calendar_tool",
        ["today", "today's schedule", "on today"],
        ["schedule", "events", "calendar"],
        {"action": "today"},
        ResponseType.DETERMINISTIC,
    ),
    (
        "calendar_week",
        "calendar_tool",
        ["this week", "week events", "weekly schedule", "week schedule"],
        ["schedule", "events", "calendar"],
        {"action": "week"},
        ResponseType.DETERMINISTIC,
    ),
    (
        "calendar_list",
        "calendar_tool",
        ["calendar", "my calendar", "upcoming events"],
        ["events", "schedule"],
        {"action": "list"},
        ResponseType.DETERMINISTIC,
    ),
    # ── Assignments ───────────────────────────────────────────────────────────
    (
        "assignments_upcoming",
        "assignment_tracker",
        [
            "due soon",
            "what's due",
            "upcoming assignments",
            "deadlines",
            "due this week",
            "due tomorrow",
        ],
        ["assignments", "homework", "tasks"],
        {"action": "upcoming"},
        ResponseType.DETERMINISTIC,
    ),
    (
        "assignments_overdue",
        "assignment_tracker",
        ["overdue", "past due", "missed", "late assignment", "behind on"],
        ["assignments", "tasks"],
        {"action": "overdue"},
        ResponseType.DETERMINISTIC,
    ),
    (
        "assignments_list",
        "assignment_tracker",
        ["assignments", "homework", "my tasks"],
        ["show", "list", "my", "all"],
        {"action": "list"},
        ResponseType.DETERMINISTIC,
    ),
    # ── Courses (new: "what courses" maps to profile + course search) ─────────
    (
        "courses_enrolled",
        "student_profile_manager",
        [
            "what courses",
            "my courses",
            "enrolled courses",
            "taking courses",
            "courses i'm taking",
            "which courses",
            "courses enrolled",
            "course list",
        ],
        ["show", "list", "enrolled", "current"],
        {"action": "summary"},
        ResponseType.DETERMINISTIC,
    ),
    (
        "course_search",
        "course_material_search",
        [
            "course material",
            "syllabus",
            "course content",
            "course topics",
            "what's in",
            "lecture notes for",
        ],
        ["find", "search", "show"],
        {},
        ResponseType.DETERMINISTIC,
    ),
    # ── Study session (capability trigger) ────────────────────────────────────
    (
        "study_session",
        "note_search",
        [
            "start a study session",
            "study session",
            "let's study",
            "help me study",
            "study with me",
            "i want to study",
        ],
        ["study", "session"],
        {"action": "search"},
        ResponseType.DETERMINISTIC,
    ),
    # ── Study schedule ────────────────────────────────────────────────────────
    (
        "study_schedule",
        "study_schedule_generator",
        [
            "study schedule",
            "study plan",
            "generate schedule",
            "create schedule",
            "make a schedule",
            "plan my study",
            "study this week",
        ],
        ["study", "week", "plan"],
        {"days": 7},
        ResponseType.DETERMINISTIC,
    ),
    # ── Notes ─────────────────────────────────────────────────────────────────
    (
        "note_search",
        "note_search",
        ["notes", "note about", "lecture notes", "my notes", "show notes"],
        ["find", "search", "show", "all", "about"],
        {"action": "search"},
        ResponseType.DETERMINISTIC,
    ),
    # ── Practice questions (ONLY workflow that calls LLM) ─────────────────────
    (
        "practice_questions",
        "practice_question_generator",
        [
            "practice questions",
            "quiz me",
            "test me",
            "practice problems",
            "give me questions",
            "practice test",
            "self test",
        ],
        ["questions", "practice", "quiz", "exam prep"],
        {},
        ResponseType.TOOL_GENERATIVE,
    ),
    # ── Exam prediction (rule-based, 0 tokens) ────────────────────────────────
    (
        "exam_topics",
        "exam_topic_predictor",
        [
            "exam topics",
            "predict exam",
            "exam prediction",
            "likely topics",
            "what's on the exam",
            "exam focus",
        ],
        ["exam", "predict", "topics", "test"],
        {},
        ResponseType.DETERMINISTIC,
    ),
    # ── Profile ───────────────────────────────────────────────────────────────
    (
        "profile_view",
        "student_profile_manager",
        [
            "my profile",
            "student profile",
            "show profile",
            "view profile",
            "profile info",
            "my info",
        ],
        ["show", "view", "display", "info"],
        {"action": "summary"},
        ResponseType.DETERMINISTIC,
    ),
    (
        "profile_update",
        "student_profile_manager",
        ["update profile", "edit profile", "change profile", "set my"],
        ["update", "change", "edit"],
        {"action": "update"},
        ResponseType.DETERMINISTIC,
    ),
    # ── GPA / academic standing ───────────────────────────────────────────────
    (
        "profile_gpa",
        "student_profile_manager",
        ["my gpa", "gpa", "academic standing", "my grades"],
        ["show", "what", "current"],
        {"action": "summary"},
        ResponseType.DETERMINISTIC,
    ),
    # ── Course enrollment management (P3) ─────────────────────────────────────
    (
        "add_course",
        "course_manager",
        [
            "add course",
            "enroll in",
            "enroll course",
            "register for",
            "sign up for",
            "take course",
            "new course",
        ],
        ["add", "enroll", "register"],
        {"action": "add"},
        ResponseType.DETERMINISTIC,
    ),
    (
        "remove_course",
        "course_manager",
        ["remove course", "drop course", "unenroll", "withdraw from"],
        ["remove", "drop"],
        {"action": "remove"},
        ResponseType.DETERMINISTIC,
    ),
    (
        "list_courses",
        "course_manager",
        [
            "list courses",
            "show courses",
            "my courses",
            "enrolled courses",
            "what courses",
        ],
        ["list", "show", "enrolled"],
        {"action": "list"},
        ResponseType.DETERMINISTIC,
    ),
]


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
def _unknown_match(params: Dict = {}) -> IntentMatch:
    """
    When nothing matches above threshold → StaticResponder with 'unknown' variant.
    0 tokens. Works offline. Never fails.
    """
    return IntentMatch(
        intent="unknown_query",
        tool="static_responder",
        confidence=0.1,
        params={"variant": "unknown"},
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
