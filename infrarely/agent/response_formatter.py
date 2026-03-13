"""
agent/response_formatter.py
═══════════════════════════════════════════════════════════════════════════════
Single authoritative place where ToolResult → human-readable string.

The LLM is NEVER called here.  This module handles all DETERMINISTIC and
TOOL_GENERATIVE rendering.  It is the guaranteed fallback if the LLM is
unavailable or disabled.

Design rules
────────────
  • Consistent prefix emoji per result type
  • Empty results get a clear "nothing found" message (never triggers LLM)
  • Error results are formatted structurally, not passed to LLM for rewording
  • List output is line-by-line with an indexed count header
  • Dict output is key: value pairs
"""

from __future__ import annotations
from typing import Optional

from infrarely.agent.state import ToolResult, ResponseType


# ── per-tool "nothing found" messages ────────────────────────────────────────
_EMPTY_MESSAGES = {
    "assignment_tracker":         "No assignments found matching your query.",
    "calendar_tool":              "No calendar events found for that period.",
    "note_search":                "No notes found matching your query.",
    "course_material_search":     "No course material matched your search.",
    "exam_topic_predictor":       "Could not predict topics — no course data available.",
    "study_schedule_generator":   "Could not generate schedule — profile not found.",
    "student_profile_manager":    "No student profile found. Run /profile to create one.",
}

_DEFAULT_EMPTY = "No results found."
_ICON_OK    = "✅"
_ICON_WARN  = "⚠️ "
_ICON_EMPTY = "📭"
_ICON_INFO  = "ℹ️ "


def format_result(result: ToolResult) -> str:
    """
    Convert a ToolResult into a clean human-readable string.
    Called for DETERMINISTIC and TOOL_GENERATIVE responses.
    Never calls the LLM.
    """
    # ── error ─────────────────────────────────────────────────────────────────
    if not result.success:
        return f"{_ICON_WARN} {result.error or 'An unknown error occurred.'}"

    # ── empty success (nothing found) ─────────────────────────────────────────
    if result.is_empty():
        msg = _EMPTY_MESSAGES.get(result.tool_name, _DEFAULT_EMPTY)
        return f"{_ICON_EMPTY} {msg}"

    # ── assemble header ───────────────────────────────────────────────────────
    header = f"{_ICON_OK} {result.message}" if result.message else ""

    # If is_complete or message already contains multi-line detail, skip body
    # to avoid duplicating structured data that the message already summarises.
    if result.is_complete or (result.message and "\n" in result.message):
        return header.strip()

    # ── body by data type ─────────────────────────────────────────────────────
    if isinstance(result.data, list):
        body = _format_list(result.data)
    elif isinstance(result.data, dict):
        body = _format_dict(result.data)
    elif result.data is not None:
        body = str(result.data).strip()
    else:
        body = result.message   # message IS the content

    if header and body:
        return f"{header}\n{body}"
    return (header or body).strip()


def format_error(tool_name: str, error: str) -> str:
    return f"{_ICON_WARN} [{tool_name}] {error}"


# ── private helpers ───────────────────────────────────────────────────────────
def _format_list(items: list) -> str:
    if not items:
        return ""
    lines = []
    for item in items:
        if isinstance(item, dict):
            lines.append("  " + "  |  ".join(f"{k}: {v}" for k, v in item.items()))
        else:
            lines.append(str(item))
    return "\n".join(lines)


def _format_dict(d: dict) -> str:
    if not d:
        return ""
    return "\n".join(f"  {k}: {v}" for k, v in d.items())