"""
agent/error_recovery.py  — Gap 5: Error Recovery Engine
Maps structured errors to deterministic recovery actions.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
from infrarely.agent.state import ExecutionContract, ToolResult
from infrarely.observability import logger


@dataclass
class RecoveryAction:
    error_key:   str          # substring match against result.error
    description: str          # what this recovery does
    handler:     Callable     # (result, context) → Optional[str]  (user message or None)


def _suggest_add_course(result: ToolResult, context: dict) -> str:
    # Extract course_id from error text if possible
    import re
    m = re.search(r"['\"]([A-Z]{2,5}\d{2,4})['\"]", result.error or "")
    course_id = m.group(1) if m else "<course_id>"
    return (f"📚 Course '{course_id}' is not in your enrolled courses.\n"
            f"   👉 To add it, type:  /courses add {course_id}\n"
            f"   Or say: 'enroll in {course_id}'\n"
            f"   Once enrolled, exam tools and study features will work for that course.")

def _suggest_add_assignment(result: ToolResult, context: dict) -> str:
    return "📝 No assignments found. Type 'add assignment' to create one."

def _suggest_check_schedule(result: ToolResult, context: dict) -> str:
    return "📅 No schedule data. Type 'show my profile' to verify your setup."

def _llm_offline_recovery(result: ToolResult, context: dict) -> str:
    return ("🔌 LLM is currently offline. Deterministic results shown above.\n"
            "   Start Ollama with: ollama serve")

def _generic_recovery(result: ToolResult, context: dict) -> str:
    return f"⚠️  Operation incomplete: {result.error}\n   Please try again or rephrase your request."


def _missing_param_recovery(result: ToolResult, context: dict) -> str:
    return (f"❓ {result.error}\n"
            f"   Please include the missing information in your request.\n"
            f"   Example: 'practice questions on algorithms for CS301'")


_RECOVERY_TABLE: List[RecoveryAction] = [
    RecoveryAction("course not found",     "suggest add course",        _suggest_add_course),
    RecoveryAction("course_not_found",     "suggest add course",        _suggest_add_course),
    RecoveryAction("not found in your enrolled", "suggest add course",  _suggest_add_course),
    RecoveryAction("missing required param","suggest provide param",    _missing_param_recovery),
    RecoveryAction("no assignments",       "suggest add assignment",    _suggest_add_assignment),
    RecoveryAction("no schedule",          "suggest check schedule",    _suggest_check_schedule),
    RecoveryAction("connection refused",   "LLM offline message",       _llm_offline_recovery),
    RecoveryAction("llm",                  "LLM offline message",       _llm_offline_recovery),
]


class ErrorRecoveryEngine:
    def __init__(self, table: List[RecoveryAction] = None):
        self._table = table or _RECOVERY_TABLE

    def recover(self, result: ToolResult, context: dict = None) -> Optional[str]:
        """
        Attempt recovery from a FAILED ToolResult.
        Returns a user-facing recovery message, or None if no strategy matches.
        """
        if result.contract != ExecutionContract.FAILED:
            return None
        error_lower = (result.error or "").lower()
        for action in self._table:
            if action.error_key.lower() in error_lower:
                logger.info(f"ErrorRecovery: matched '{action.error_key}' → {action.description}")
                return action.handler(result, context or {})
        logger.debug(f"ErrorRecovery: no strategy for '{result.error}' — using generic")
        return _generic_recovery(result, context or {})

    def register(self, action: RecoveryAction):
        self._table.insert(0, action)   # higher priority


_engine = ErrorRecoveryEngine()

def recover(result: ToolResult, context: dict = None) -> Optional[str]:
    return _engine.recover(result, context)