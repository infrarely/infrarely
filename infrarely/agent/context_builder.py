"""
agent/context_builder.py
═══════════════════════════════════════════════════════════════════════════════
Builds the minimal, token-budgeted context string sent to the LLM.

Design principles
─────────────────
  1. Hard character budget (→ token budget).  Never exceed MAX_CONTEXT_CHARS.
  2. Priority order: tool_result > profile_snippet > lt_summary > history
     Earlier items get their full allocation; later items are trimmed first.
  3. Conversation history: last 2 turns only, each capped at 250 chars.
     No full window, no raw memory dumps.
  4. All assembly happens here — the core never touches strings directly.
"""

from __future__ import annotations
from typing import List, Dict, Optional

import infrarely.core.app_config as config
from infrarely.agent.state import ToolResult, TaskState


# ── token-approximate budgets (4 chars ≈ 1 token) ────────────────────────────
_MAX_CONTEXT_CHARS  = config.MAX_CONTEXT_TOKENS_FOR_LLM * 4   # e.g. 3200 chars
_HISTORY_MAX_CHARS  = 250   # per individual message
_PROFILE_MAX_CHARS  = 180
_LT_SUMMARY_CHARS   = 200
_TOOL_SNIPPET_CHARS = 600


def build(
    task_state:   TaskState,
    tool_result:  ToolResult,
    user_input:   str,
    history:      List[Dict[str, str]],    # already-trimmed [{role, content}]
    profile:      Optional[dict],
    lt_summary:   str,
) -> List[Dict[str, str]]:
    """
    Assembles the messages list to send to the LLM.
    Returns a list ready to pass directly to llm_call(messages=...).

    Structure:
      [history turns (≤2)]  +  [user turn with embedded context]

    The context block inside the user turn is:
      [Tool data]  (if any)
      [Student]    (if profile exists)
      [History]    (compressed summary from LTM, if any)
      ---
      User: <original query>
    """
    context_parts: List[str] = []
    budget_remaining = _MAX_CONTEXT_CHARS

    # 1 ── tool result snippet (highest priority)
    if tool_result.success and not tool_result.is_empty():
        snippet = tool_result.to_context_snippet(
            max_items=8,
            max_chars=min(_TOOL_SNIPPET_CHARS, budget_remaining),
        )
        context_parts.append(snippet)
        budget_remaining -= len(snippet)

    # 2 ── profile micro-snippet
    if profile and budget_remaining > 100:
        p = profile
        snippet = (
            f"[Student] {p.get('name','?')} | {p.get('degree','?')} yr{p.get('year','?')} "
            f"| style:{p.get('learning_style','?')} | peak:{p.get('peak_hours','?')}"
        )[:_PROFILE_MAX_CHARS]
        context_parts.append(snippet)
        budget_remaining -= len(snippet)

    # 3 ── long-term memory summary (lowest priority, trimmed first)
    if lt_summary and budget_remaining > 80:
        snippet = f"[Context] {lt_summary}"[:min(_LT_SUMMARY_CHARS, budget_remaining)]
        context_parts.append(snippet)
        budget_remaining -= len(snippet)

    # ── assemble user turn ────────────────────────────────────────────────────
    if context_parts:
        context_block = "\n".join(context_parts)
        user_content  = f"{context_block}\n\n{user_input}"
    else:
        user_content  = user_input

    # ── trim history (last 2 turns, chars-capped) ────────────────────────────
    trimmed_history: List[Dict[str, str]] = []
    for msg in history[-2:]:
        trimmed_history.append({
            "role":    msg["role"],
            "content": msg["content"][:_HISTORY_MAX_CHARS],
        })

    return trimmed_history + [{"role": "user", "content": user_content}]


def estimate_tokens(messages: List[Dict[str, str]]) -> int:
    """Rough token estimate: 4 chars ≈ 1 token."""
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return total_chars // 4