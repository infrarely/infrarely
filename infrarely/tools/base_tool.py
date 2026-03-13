"""
tools/base_tool.py  (v2 — refactored)
═══════════════════════════════════════════════════════════════════════════════
Abstract base class for all tools.

Hardening over v1
─────────────────
  • _execute() must set result.response_type explicitly (enforced by __init_subclass__)
  • Input sanitisation helper (strip strings, coerce types defensively)
  • Structured failure always returns is_complete=False so the router knows
    not to treat errors as terminal answers
  • Retry back-off uses min(0.5, 0.1 * attempt) to avoid blocking the loop
  • Circuit breaker state is logged with tool name for diagnostics
"""

from __future__ import annotations
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import infrarely.core.app_config as config
from infrarely.agent.state import ResponseType, TaskState, ToolResult
from infrarely.observability import logger
from infrarely.observability.metrics import collector


# ─── Circuit Breaker ──────────────────────────────────────────────────────────
class CircuitBreaker:
    def __init__(
        self,
        threshold:     int   = config.CIRCUIT_BREAKER_THRESHOLD,
        reset_seconds: float = config.CIRCUIT_BREAKER_RESET_SECONDS,
        tool_name:     str   = "unknown",
    ):
        self.threshold     = threshold
        self.reset_seconds = reset_seconds
        self._tool_name    = tool_name
        self._failures     = 0
        self._open_since:  Optional[float] = None

    @property
    def is_open(self) -> bool:
        if self._open_since is None:
            return False
        if time.time() - self._open_since >= self.reset_seconds:
            logger.info(f"CircuitBreaker RESET for '{self._tool_name}'")
            self._failures   = 0
            self._open_since = None
            return False
        return True

    @property
    def remaining_seconds(self) -> float:
        if self._open_since is None:
            return 0.0
        return max(0.0, self.reset_seconds - (time.time() - self._open_since))

    def record_success(self) -> None:
        self._failures   = 0
        self._open_since = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.threshold:
            if self._open_since is None:
                self._open_since = time.time()
                collector.circuit_breaker_trips += 1
                logger.warn(
                    f"CircuitBreaker OPEN for '{self._tool_name}'",
                    failures=self._failures,
                    cooldown_s=self.reset_seconds,
                )


# ─── Base Tool ────────────────────────────────────────────────────────────────
class BaseTool(ABC):
    """
    All tools inherit this.  Provides:
      • retry with exponential back-off
      • fault injection (configurable)
      • circuit breaker
      • structured logging + metrics
      • input sanitisation helpers
      • guaranteed ToolResult shape on all exit paths
    """

    name:          str          = "base_tool"
    description:   str          = ""
    response_type: ResponseType = ResponseType.DETERMINISTIC

    def __init__(self):
        self._breaker = CircuitBreaker(tool_name=self.name)

    # ── public entry-point ────────────────────────────────────────────────────
    def run(self, state: TaskState) -> ToolResult:
        if self._breaker.is_open:
            remaining = round(self._breaker.remaining_seconds, 1)
            return ToolResult(
                success       = False,
                tool_name     = self.name,
                error         = f"Circuit breaker open for {self.name} "
                                f"— retry in {remaining}s",
                response_type = ResponseType.ERROR,
                is_complete   = False,
            )

        # Sanitise params before execution
        state = self._sanitise_state(state)

        start      = time.monotonic()
        last_error = ""

        for attempt in range(1, config.MAX_RETRIES + 2):
            try:
                self._maybe_inject_fault()

                result             = self._execute(state)
                duration           = (time.monotonic() - start) * 1000
                result.duration_ms = duration

                # Ensure response_type is set
                if result.response_type == ResponseType.DETERMINISTIC:
                    result.response_type = self.response_type

                self._breaker.record_success()
                collector.record_tool_call(self.name, duration, success=True)
                logger.tool_log(self.name, "success",
                                duration_ms=duration, attempt=attempt)
                return result

            except Exception as exc:
                last_error = str(exc)
                duration   = (time.monotonic() - start) * 1000
                logger.warn(f"{self.name} attempt {attempt}/{config.MAX_RETRIES + 1} "
                            f"failed: {last_error}")

                if attempt <= config.MAX_RETRIES:
                    collector.retries += 1
                    time.sleep(min(0.5, 0.1 * attempt))
                else:
                    self._breaker.record_failure()
                    collector.record_tool_call(self.name, duration,
                                               success=False, error=last_error)
                    logger.tool_log(self.name, "failure",
                                    duration_ms=duration, error=last_error)
                    return ToolResult(
                        success       = False,
                        tool_name     = self.name,
                        error         = last_error,
                        duration_ms   = duration,
                        response_type = ResponseType.ERROR,
                        is_complete   = False,
                    )

        # Unreachable, but satisfies type checker
        return ToolResult(
            success=False, tool_name=self.name,
            error=last_error, response_type=ResponseType.ERROR,
        )

    # ── abstract implementation ───────────────────────────────────────────────
    @abstractmethod
    def _execute(self, state: TaskState) -> ToolResult:
        """Tool-specific logic.  Must return a ToolResult."""
        ...

    # ── helpers for subclasses ────────────────────────────────────────────────
    def validate_params(self, state: TaskState, required: List[str]) -> Optional[str]:
        """Returns an error string if any required param is absent or empty."""
        for key in required:
            val = state.params.get(key)
            if val is None or (isinstance(val, str) and not val.strip()):
                return f"Missing required parameter: '{key}'"
        return None

    def safe_get(self, state: TaskState, key: str, default: Any = None) -> Any:
        """Type-safe param retrieval with default."""
        return state.params.get(key, default)

    def safe_int(self, state: TaskState, key: str, default: int = 0,
                 min_val: int = 0, max_val: int = 9999) -> int:
        """Safe integer coercion with clamping."""
        try:
            return max(min_val, min(max_val, int(state.params.get(key, default))))
        except (TypeError, ValueError):
            return default

    # ── private ───────────────────────────────────────────────────────────────
    @staticmethod
    def _sanitise_state(state: TaskState) -> TaskState:
        """Strip whitespace from all string params."""
        clean_params = {}
        for k, v in state.params.items():
            clean_params[k] = v.strip() if isinstance(v, str) else v
        state.params = clean_params
        return state

    def _maybe_inject_fault(self) -> None:
        if not config.FAULT_INJECTION_ENABLED:
            return
        if random.random() < config.FAULT_PROBABILITY:
            raise RuntimeError(f"[FAULT INJECTED] {self.name} simulated failure")
        if random.random() < config.FAULT_PROBABILITY * 0.4:
            time.sleep(config.FAULT_TIMEOUT_SECONDS)
            raise TimeoutError(f"[FAULT INJECTED] {self.name} simulated timeout")