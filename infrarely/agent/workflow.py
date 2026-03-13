"""
agent/workflow.py
═══════════════════════════════════════════════════════════════════════════════
Test scenarios demonstrating the agent's capabilities.

Each scenario documents expected token usage:
  [0 tok]  = fully deterministic, works offline
  [~N tok] = LLM called once inside a tool (TOOL_GENERATIVE)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Scenario:
    id:          int
    name:        str
    description: str
    queries:     List[str]
    token_note:  str = ""


SCENARIOS: List[Scenario] = [

    Scenario(
        id=1,
        name="Basic Interaction",
        description="Greeting, profile, and courses — ALL zero-token, works fully offline.",
        token_note="Expected: 0 tokens",
        queries=[
            "Hello!",
            "Show my profile",
            "What courses am I enrolled in?",
            "What can you do?",
        ],
    ),

    Scenario(
        id=2,
        name="Assignment Management",
        description="List, overdue, and upcoming assignment queries.",
        token_note="Expected: 0 tokens",
        queries=[
            "Show all my assignments",
            "What assignments are overdue?",
            "What's due this week?",
            "Show assignments for CS301",
        ],
    ),

    Scenario(
        id=3,
        name="Tool Reliability",
        description="Fault injection: retry + circuit breaker under simulated failures.",
        token_note="Expected: 0 tokens (resilience test)",
        queries=[
            "Show my assignments",
            "Show my calendar today",
            "Show my assignments",
        ],
    ),

    Scenario(
        id=4,
        name="Study Planning",
        description="Schedule generation and exam topic prediction — all algorithmic.",
        token_note="Expected: 0 tokens",
        queries=[
            "Generate a study schedule",
            "Predict exam topics for CS301",
            "Show my notes",
            "Find notes about trees",
        ],
    ),

    Scenario(
        id=5,
        name="Practice Questions (LLM)",
        description="The ONLY scenario that calls the LLM — scoped to 1 call inside the tool.",
        token_note="Expected: ~130-300 tokens (1 scoped LLM call in tool)",
        queries=[
            "Practice questions about binary trees",
            "Give me hard practice questions about dynamic programming",
            "Quiz me on CS301 sorting algorithms",
        ],
    ),

    Scenario(
        id=6,
        name="Unknown Query Handling",
        description="Unknown queries return deterministic fallback — 0 tokens, no LLM.",
        token_note="Expected: 0 tokens",
        queries=[
            "What is the weather like?",
            "Tell me a joke",
            "Who won the world cup?",
        ],
    ),

    Scenario(
        id=7,
        name="Memory Persistence",
        description="Working memory and long-term compression across many turns.",
        token_note="Expected: 0 tokens",
        queries=[
            "Show my assignments",
            "Show my calendar today",
            "Show my profile",
            "Show my notes",
            "Show all my assignments",
            "Generate a study schedule",
            "What assignments are overdue?",
        ],
    ),

    Scenario(
        id=8,
        name="Full Workflow",
        description="Complete student session covering all deterministic tools.",
        token_note="Expected: 0 tokens",
        queries=[
            "Hello!",
            "Show my profile",
            "What courses am I enrolled in?",
            "Show all my assignments",
            "What's overdue?",
            "Show my calendar today",
            "Generate a study schedule",
            "Predict exam topics for CS301",
            "Find notes about graphs",
        ],
    ),
]


def get_scenario(n: int) -> Optional[Scenario]:
    return next((s for s in SCENARIOS if s.id == n), None)


def list_scenarios() -> List[str]:
    return [
        f"  [{s.id}] {s.name:<30} {s.token_note}"
        for s in SCENARIOS
    ]