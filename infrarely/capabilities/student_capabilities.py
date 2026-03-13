"""
capabilities/student_capabilities.py  — Layer 2
═══════════════════════════════════════════════════════════════════════════════
All student workflow capabilities.

Each capability is a named sequence of CapabilitySteps.
Declaration order = execution order.
Deterministic steps are always declared first.

Capabilities defined here
──────────────────────────
  exam_preparation   — predict topics → schedule → practice questions
  weekly_planning    — assignments → calendar → study schedule
  study_session      — notes → profile (learning style) → practice questions

Token budget per capability (approximate, with LLM online)
───────────────────────────────────────────────────────────
  exam_preparation   ~300 tok  (2 det + 1 gen)
  weekly_planning    0 tok     (3 det)
  study_session      ~300 tok  (2 det + 1 gen)
"""

from __future__ import annotations

from infrarely.agent.capability import Capability, CapabilityStep, FailurePolicy


# ── exam_preparation ──────────────────────────────────────────────────────────
# Triggered by: "help me prepare for my CS301 exam",
#               "exam prep for CS301", "I have an exam coming up"
#
# Step 1 — predict exam topics (FINAL_DETERMINISTIC, 0 tokens)
#           output: context["exam_topics"] = {"items": [...topics...], ...}
# Step 2 — generate study schedule (FINAL_DETERMINISTIC, 0 tokens)
#           uses: nothing from step 1 (schedule is always useful)
# Step 3 — practice questions on first topic (FINAL_GENERATED, ~300 tokens)
#           uses: context["exam_topics.items[0]"] as topic
#           skipped if no exam_topics were produced
#
EXAM_PREPARATION = Capability(
    name="exam_preparation",
    description="Exam Preparation Workflow",
    intent_tags=[
        "help me prepare",
        "exam prep",
        "prepare for exam",
        "study for exam",
        "exam coming",
        "revise for",
        "get ready for exam",
        "exam preparation",
    ],
    steps=[
        CapabilityStep(
            name="exam_topics",
            tool_name="exam_topic_predictor",
            description="Predict likely exam topics",
            base_params={},  # course_id injected by router from query
            failure_policy=FailurePolicy.ABORT,
        ),
        CapabilityStep(
            name="study_schedule",
            tool_name="study_schedule_generator",
            description="Generate a study schedule",
            base_params={"days": 7},
            failure_policy=FailurePolicy.SKIP,  # schedule failure is not fatal
        ),
        CapabilityStep(
            name="practice_questions",
            tool_name="practice_question_generator",
            description="Generate practice questions on top exam topic",
            base_params={"topic": "{exam_topics.topics[0]}"},
            failure_policy=FailurePolicy.SKIP,  # LLM offline → still useful result
            # Only run if exam_topics produced results
            condition=lambda ctx: bool(
                isinstance(ctx.get("exam_topics"), dict)
                and ctx["exam_topics"].get("topics")
            ),
        ),
    ],
)


# ── weekly_planning ───────────────────────────────────────────────────────────
# Triggered by: "plan my week", "weekly overview", "what do I have this week",
#               "help me plan my study week"
#
# All three steps are FINAL_DETERMINISTIC — 0 tokens guaranteed.
#
WEEKLY_PLANNING = Capability(
    name="weekly_planning",
    description="Weekly Planning Overview",
    intent_tags=[
        "plan my week",
        "weekly overview",
        "this week overview",
        "what do i have this week",
        "help me plan this week",
        "weekly plan",
        "week ahead",
    ],
    steps=[
        CapabilityStep(
            name="assignments",
            tool_name="assignment_tracker",
            description="Check upcoming assignments",
            base_params={"action": "upcoming", "days": 7},
            failure_policy=FailurePolicy.SKIP,
        ),
        CapabilityStep(
            name="calendar",
            tool_name="calendar_tool",
            description="Show this week's calendar",
            base_params={"action": "week"},
            failure_policy=FailurePolicy.SKIP,
        ),
        CapabilityStep(
            name="study_schedule",
            tool_name="study_schedule_generator",
            description="Generate study schedule for the week",
            base_params={"days": 7},
            failure_policy=FailurePolicy.SKIP,
        ),
    ],
)


# ── study_session ─────────────────────────────────────────────────────────────
# Triggered by: "start a study session", "let's study", "help me study",
#               "study session for CS301"
#
# Step 1 — fetch notes on topic (FINAL_DETERMINISTIC, 0 tokens)
# Step 2 — fetch profile (learning style for personalisation, 0 tokens)
# Step 3 — practice questions (FINAL_GENERATED, ~300 tokens, skippable)
#
STUDY_SESSION = Capability(
    name="study_session",
    description="Interactive Study Session",
    intent_tags=[
        "start a study session",
        "let's study",
        "help me study",
        "study session",
        "study with me",
        "i want to study",
    ],
    steps=[
        CapabilityStep(
            name="notes",
            tool_name="note_search",
            description="Find relevant notes",
            base_params={"action": "search"},  # query injected by router
            failure_policy=FailurePolicy.SKIP,
        ),
        CapabilityStep(
            name="profile",
            tool_name="student_profile_manager",
            description="Load student profile for personalisation",
            base_params={"action": "summary"},
            failure_policy=FailurePolicy.SKIP,
        ),
        CapabilityStep(
            name="practice_questions",
            tool_name="practice_question_generator",
            description="Generate practice questions",
            base_params={
                "topic": "{topic}"
            },  # resolved from router extraction in initial_context
            failure_policy=FailurePolicy.SKIP,
            # Only run when the router actually extracted a topic from the query
            condition=lambda ctx: bool(ctx.get("topic")),
        ),
    ],
)


# ── registry of all capabilities ─────────────────────────────────────────────
ALL_CAPABILITIES = [
    EXAM_PREPARATION,
    WEEKLY_PLANNING,
    STUDY_SESSION,
]
