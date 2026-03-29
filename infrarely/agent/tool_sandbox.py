"""
agent/tool_sandbox.py  — Gap 7: Tool Sandbox
Wraps tool.run() with timeout enforcement and output size limits.
"""
from __future__ import annotations
import threading
import time
from typing import Optional
import infrarely.core.app_config as config
from infrarely.agent.state import ExecutionContract, TaskState, ToolResult
from infrarely.observability import logger

_TIMEOUT     = getattr(config, "TOOL_TIMEOUT_SECONDS", 5.0)
_MAX_BYTES   = getattr(config, "SANDBOX_MAX_OUTPUT_BYTES", 102_400)
_ENABLED     = getattr(config, "ENABLE_TOOL_SANDBOX", True)


class ToolSandbox:
    """
    Wraps a tool's run() in a timed thread.
    If the tool exceeds TOOL_TIMEOUT_SECONDS the thread is abandoned
    and a FAILED ToolResult is returned immediately.
    """

    def __init__(self, timeout: float = _TIMEOUT, enabled: bool = _ENABLED):
        self._timeout = timeout
        self._enabled = enabled

    def run(self, tool, state: TaskState) -> ToolResult:
        if not self._enabled:
            return tool.run(state)

        result_holder: list = []
        exc_holder:    list = []

        def _worker():
            try:
                result_holder.append(tool.run(state))
            except Exception as e:
                exc_holder.append(e)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=self._timeout)

        if t.is_alive():
            logger.warn(
                f"ToolSandbox: '{tool.name}' timed out after {self._timeout}s",
                tool    = tool.name,
                timeout = self._timeout,
            )
            return ToolResult(
                contract  = ExecutionContract.FAILED,
                tool_name = tool.name,
                error     = f"Tool '{tool.name}' timed out after {self._timeout}s",
            )

        if exc_holder:
            logger.error(f"ToolSandbox: '{tool.name}' raised {exc_holder[0]}")
            return ToolResult(
                contract  = ExecutionContract.FAILED,
                tool_name = tool.name,
                error     = str(exc_holder[0]),
            )

        result = result_holder[0]

        # output size guard
        try:
            import json
            size = len(json.dumps(result.data, default=str).encode())
            if size > _MAX_BYTES:
                logger.warn(f"ToolSandbox: '{tool.name}' output {size}B > {_MAX_BYTES}B limit")
                result.data    = None
                result.message = f"[Output truncated — exceeded {_MAX_BYTES} byte limit]"
        except Exception:
            pass

        return result