"""
agent/tool_validator.py  — Gap 4: Tool Output Validator
Rejects structurally invalid or suspiciously large ToolResults.
"""
from __future__ import annotations
import json
from typing import Optional
import infrarely.core.app_config as config
from infrarely.agent.state import ExecutionContract, ToolResult
from infrarely.observability import logger

_MAX_CHARS = getattr(config, "MAX_TOOL_DATA_CHARS", 50_000)
_STRICT    = getattr(config, "STRICT_TOOL_VALIDATION", True)


class ToolOutputValidator:
    """
    Called by BaseTool.run() after _execute() returns.
    Mutates nothing — only raises ValidationError on hard violations.
    """

    @staticmethod
    def validate(result: ToolResult) -> Optional[str]:
        """
        Returns None if valid, or an error message string if invalid.
        On hard failure the caller should replace result with a FAILED contract.
        """
        if not _STRICT:
            return None

        # 1. contract must be a valid ExecutionContract
        if not isinstance(result.contract, ExecutionContract):
            return f"Invalid contract type: {type(result.contract)}"

        # 2. FAILED results must have an error message
        if result.contract == ExecutionContract.FAILED and not result.error:
            return "FAILED result missing error message"

        # 3. payload size guard
        try:
            size = len(json.dumps(result.data, default=str))
            if size > _MAX_CHARS:
                return f"Tool output too large: {size} chars > {_MAX_CHARS} limit"
        except Exception:
            pass   # non-serialisable data — tool's responsibility to handle

        # 4. tool_name must be populated
        if not result.tool_name:
            return "ToolResult missing tool_name"

        return None   # all good


_validator = ToolOutputValidator()

def validate(result: ToolResult) -> Optional[str]:
    return _validator.validate(result)