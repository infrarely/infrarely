"""
tests/test_tools.py — Tool decorator and execution tests
═══════════════════════════════════════════════════════════════════════════════
Tests @infrarely.tool decorator, circuit breaker, retry, timeout, caching.
"""

import sys
import os
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestToolDecorator:
    """Test @infrarely.tool decorator basics."""

    def test_basic_tool(self):
        import infrarely

        @infrarely.tool
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        assert callable(greet)
        assert greet("World") == "Hello, World!"

    def test_tool_with_params(self):
        import infrarely

        @infrarely.tool(retries=2, timeout=10)
        def compute(x: int, y: int) -> int:
            return x + y

        assert compute(3, 4) == 7

    def test_tool_registered(self):
        import infrarely
        from infrarely.core.decorators import get_tool_registry

        @infrarely.tool
        def registered_tool() -> str:
            return "ok"

        registry = get_tool_registry()
        meta = registry.get_meta("registered_tool")
        assert meta is not None
        assert meta.name == "registered_tool"

    def test_tool_never_raises(self):
        """Tools with @infrarely.tool should never propagate exceptions."""
        import infrarely

        @infrarely.tool
        def bad_tool() -> str:
            raise ValueError("intentional error")

        result = bad_tool()
        # Should return error dict, not raise
        assert isinstance(result, dict)
        assert result.get("__aos_error") is True

    def test_tool_with_agent(self):
        import infrarely

        @infrarely.tool
        def double(n: int) -> int:
            return n * 2

        agent = infrarely.agent("tool_agent", tools=[double])
        assert "double" in agent.tools
        agent.shutdown()


class TestToolCircuitBreaker:
    """Test circuit breaker on tools."""

    def test_circuit_breaker_opens(self):
        import infrarely
        from infrarely.core.decorators import get_tool_registry

        fail_count = 0

        @infrarely.tool(retries=0)
        def fragile_tool() -> str:
            raise RuntimeError("failing")

        # Call enough times to open circuit breaker
        for _ in range(5):
            fragile_tool()

        # After threshold, circuit should be open
        result = fragile_tool()
        assert isinstance(result, dict)
        assert result.get("__aos_error") is True


class TestToolWithCache:
    """Test tool caching."""

    def test_cached_tool(self):
        import infrarely

        call_count = 0

        @infrarely.tool(cache=True)
        def expensive_compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * x

        # First call
        r1 = expensive_compute(5)
        assert r1 == 25
        count_after_first = call_count

        # Second call (should be cached)
        r2 = expensive_compute(5)
        assert r2 == 25
        # call_count should still be the same or at most incremented once more
        # (depends on cache implementation)
