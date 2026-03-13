"""
aos/testing.py — Testing Utilities for Agent-Based Code
═══════════════════════════════════════════════════════════════════════════════
Makes it easy to write unit tests for agent code.

Without testing utilities, developers cannot ship agent code to production.

Usage::

    from infrarely.platform.testing import AgentTestCase, mock_tool, mock_llm

    class TestMyAgent(AgentTestCase):
        def setUp(self):
            self.agent = infrarely.agent("test-agent")

        def test_math_bypass(self):
            result = self.agent.run("What is 2 + 2?")
            self.assertResult(result, output=4, used_llm=False)

        @mock_tool("get_weather", returns="72°F in NYC")
        def test_weather_tool(self):
            result = self.agent.run("Weather in NYC?")
            self.assertResult(result, used_llm=False)

        @mock_llm(returns="Quantum entanglement is...")
        def test_llm_fallback(self):
            result = self.agent.run("Explain quantum entanglement")
            self.assertResult(result, used_llm=True)
"""

from __future__ import annotations

import functools
import time
import unittest
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Union
from unittest.mock import MagicMock, patch

_UNSET = object()  # sentinel for unset default values


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT TEST CASE — Base class for agent unit tests
# ═══════════════════════════════════════════════════════════════════════════════


class AgentTestCase(unittest.TestCase):
    """
    Base test case class for testing AOS agents.

    Provides:
    - Automatic setup/teardown of test agents
    - assertResult() for checking Result objects
    - assertSuccess() / assertFailed()
    - Performance assertions (assertFast, assertNoLLM)
    - Automatic cleanup between tests

    Usage::

        class TestMyAgent(AgentTestCase):
            def setUp(self):
                self.agent = self.create_agent("test")

            def test_basic(self):
                result = self.agent.run("2 + 2")
                self.assertResult(result, output=4)
    """

    def setUp(self):
        """Set up test fixtures. Override in subclass."""
        import infrarely

        self._test_agents: list = []
        # Reset singletons for clean test state
        try:
            from infrarely.core.config import get_config

            get_config().reset()
        except Exception:
            pass

    def tearDown(self):
        """Clean up after each test."""
        for agent in self._test_agents:
            try:
                agent.shutdown()
            except Exception:
                pass
        self._test_agents.clear()

    def create_agent(
        self,
        name: str = "test-agent",
        *,
        tools: Optional[list] = None,
        capabilities: Optional[list] = None,
        **kwargs,
    ) -> Any:
        """Create a test agent with automatic cleanup."""
        import infrarely

        agent = infrarely.agent(name, tools=tools, capabilities=capabilities, **kwargs)
        self._test_agents.append(agent)
        return agent

    # ── Result assertions ─────────────────────────────────────────────────────

    def assertResult(
        self,
        result: Any,
        *,
        output: Any = _UNSET,
        success: Optional[bool] = None,
        used_llm: Optional[bool] = None,
        sources: Optional[List[str]] = None,
        confidence_min: Optional[float] = None,
        confidence_max: Optional[float] = None,
        error_type: Optional[str] = None,
        msg: str = "",
    ) -> None:
        """
        Assert multiple properties of a Result object at once.

        Parameters
        ----------
        result : Result
            The result to check.
        output : Any
            Expected output value.
        success : bool, optional
            Expected success status.
        used_llm : bool, optional
            Expected LLM usage.
        sources : list[str], optional
            Expected sources (checks containment).
        confidence_min : float, optional
            Minimum confidence value.
        confidence_max : float, optional
            Maximum confidence value.
        error_type : str, optional
            Expected error type.
        msg : str
            Extra message for failures.
        """
        prefix = f"{msg}: " if msg else ""

        if success is not None:
            self.assertEqual(
                result.success,
                success,
                f"{prefix}Expected success={success}, got {result.success}",
            )

        if output is not _UNSET:
            self._assertOutputEquals(result.output, output, prefix)

        if used_llm is not None:
            self.assertEqual(
                result.used_llm,
                used_llm,
                f"{prefix}Expected used_llm={used_llm}, got {result.used_llm}",
            )

        if sources is not None:
            for src in sources:
                found = any(src.lower() in s.lower() for s in (result.sources or []))
                self.assertTrue(
                    found,
                    f"{prefix}Expected source '{src}' not found in {result.sources}",
                )

        if confidence_min is not None:
            self.assertGreaterEqual(
                result.confidence,
                confidence_min,
                f"{prefix}Confidence {result.confidence} below min {confidence_min}",
            )

        if confidence_max is not None:
            self.assertLessEqual(
                result.confidence,
                confidence_max,
                f"{prefix}Confidence {result.confidence} above max {confidence_max}",
            )

        if error_type is not None and result.error:
            self.assertEqual(
                result.error.type.value,
                error_type,
                f"{prefix}Expected error type '{error_type}', got '{result.error.type.value}'",
            )

    def _assertOutputEquals(self, actual: Any, expected: Any, prefix: str = "") -> None:
        """Flexible output comparison with type coercion."""
        if actual == expected:
            return

        # Numeric comparison with tolerance
        try:
            a_num = float(actual)
            e_num = float(expected)
            if abs(a_num - e_num) < 1e-6:
                return
        except (TypeError, ValueError):
            pass

        # String comparison
        if str(actual) == str(expected):
            return

        self.assertEqual(actual, expected, f"{prefix}Output mismatch")

    def assertSuccess(self, result: Any, msg: str = "") -> None:
        """Assert the result was successful."""
        self.assertTrue(
            result.success, msg or f"Expected success but got: {result.error}"
        )

    def assertFailed(self, result: Any, msg: str = "") -> None:
        """Assert the result failed."""
        self.assertFalse(result.success, msg or "Expected failure but got success")

    def assertNoLLM(self, result: Any, msg: str = "") -> None:
        """Assert no LLM was used."""
        self.assertFalse(result.used_llm, msg or "Expected no LLM usage")

    def assertUsedLLM(self, result: Any, msg: str = "") -> None:
        """Assert LLM was used."""
        self.assertTrue(result.used_llm, msg or "Expected LLM usage")

    def assertFast(self, result: Any, max_ms: float = 100, msg: str = "") -> None:
        """Assert the result was fast (under max_ms)."""
        self.assertLessEqual(
            result.duration_ms,
            max_ms,
            msg or f"Expected under {max_ms}ms, got {result.duration_ms}ms",
        )

    def assertConfident(
        self, result: Any, min_confidence: float = 0.9, msg: str = ""
    ) -> None:
        """Assert high confidence."""
        self.assertGreaterEqual(
            result.confidence,
            min_confidence,
            msg or f"Expected confidence >= {min_confidence}, got {result.confidence}",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MOCK DECORATORS
# ═══════════════════════════════════════════════════════════════════════════════


def mock_tool(
    tool_name: str,
    *,
    returns: Any = None,
    side_effect: Optional[Callable] = None,
    raises: Optional[Exception] = None,
) -> Callable:
    """
    Decorator that mocks a specific tool for the duration of a test.

    Usage::

        @mock_tool("get_weather", returns="72°F in NYC")
        def test_weather(self):
            result = self.agent.run("Weather in NYC?")
            self.assertResult(result, output="72°F in NYC")
    """

    def decorator(test_fn: Callable) -> Callable:
        @functools.wraps(test_fn)
        def wrapper(*args, **kwargs):
            from infrarely.core.decorators import get_tool_registry

            registry = get_tool_registry()
            original = registry.get(tool_name)

            # Create mock function
            if raises:
                mock_fn = MagicMock(side_effect=raises)
            elif side_effect:
                mock_fn = MagicMock(side_effect=side_effect)
            else:
                mock_fn = MagicMock(return_value=returns)

            mock_fn.__name__ = tool_name

            # Replace in registry
            registry._tools[tool_name] = mock_fn

            try:
                return test_fn(*args, **kwargs)
            finally:
                # Restore
                if original:
                    registry._tools[tool_name] = original
                else:
                    registry._tools.pop(tool_name, None)

        return wrapper

    return decorator


def mock_llm(
    *,
    returns: str = "",
    side_effect: Optional[Callable] = None,
) -> Callable:
    """
    Decorator that mocks LLM calls for the duration of a test.

    Usage::

        @mock_llm(returns="Quantum entanglement is...")
        def test_llm_response(self):
            result = self.agent.run("Explain quantum entanglement")
            self.assertResult(result, used_llm=True)
    """

    def decorator(test_fn: Callable) -> Callable:
        @functools.wraps(test_fn)
        def wrapper(*args, **kwargs):
            with mock_llm_context(returns=returns, side_effect=side_effect):
                return test_fn(*args, **kwargs)

        return wrapper

    return decorator


@contextmanager
def mock_llm_context(
    *,
    returns: str = "",
    side_effect: Optional[Callable] = None,
):
    """
    Context manager that mocks LLM calls.

    Usage::

        with mock_llm_context(returns="Test response"):
            result = agent.run("Some LLM task")
    """
    mock_response = returns

    def mock_llm_call(*args, **kwargs):
        if side_effect:
            return side_effect(*args, **kwargs)
        return mock_response

    with patch("aos._internal.bridge.ExecutionEngine._call_llm", mock_llm_call):
        yield


@contextmanager
def mock_knowledge_context(
    *,
    returns: Optional[Dict[str, Any]] = None,
):
    """
    Context manager that mocks knowledge queries.

    Usage::

        with mock_knowledge_context(returns={"text": "...", "confidence": 0.9}):
            result = agent.run("Query knowledge")
    """
    mock_result = returns or {"text": "", "confidence": 0.0, "sources": []}

    original_query = None
    try:
        from infrarely.memory.knowledge import KnowledgeManager

        original_query = KnowledgeManager.query
        KnowledgeManager.query = lambda self, *a, **kw: mock_result
        yield
    finally:
        if original_query:
            KnowledgeManager.query = original_query


# ═══════════════════════════════════════════════════════════════════════════════
# TEST FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


def create_test_agent(
    name: str = "test-agent",
    *,
    tools: Optional[list] = None,
    knowledge_data: Optional[Dict[str, str]] = None,
) -> Any:
    """
    Create a pre-configured agent for testing.

    Usage::

        agent = create_test_agent(
            "tutor",
            knowledge_data={"bio": "Mitochondria are the powerhouse of the cell"},
        )
    """
    import infrarely

    agent = infrarely.agent(name, tools=tools)

    if knowledge_data:
        for key, value in knowledge_data.items():
            agent.knowledge.add_data(key, value)

    return agent


def assert_agent_invariants(agent: Any) -> None:
    """
    Run a battery of invariant checks on an agent.

    Useful in tearDown or end-to-end tests.
    """
    # Agent should be alive
    assert agent.alive, "Agent should be alive"

    # State should be IDLE after tasks complete
    assert agent.state == "IDLE", f"Agent state should be IDLE, got {agent.state}"

    # Health should be accessible
    health = agent.health()
    assert health is not None, "Health report should not be None"


class TimingContext:
    """
    Context manager for measuring execution time.

    Usage::

        with TimingContext() as timer:
            result = agent.run("task")
        print(f"Took {timer.duration_ms}ms")
    """

    def __init__(self):
        self.start_time: float = 0
        self.end_time: float = 0
        self.duration_ms: float = 0

    def __enter__(self) -> "TimingContext":
        self.start_time = time.time()
        return self

    def __exit__(self, *args) -> None:
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
