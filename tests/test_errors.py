"""
tests/test_errors.py — Error handling tests
═══════════════════════════════════════════════════════════════════════════════
Philosophy 3: Errors are data, not exceptions.
Every error has type, message, step, recovered, suggestion.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestErrorTypes:
    """Test error type taxonomy."""

    def test_all_error_types_exist(self):
        from infrarely.core.result import ErrorType

        expected = [
            "TOOL_FAILURE",
            "PLAN_INVALID",
            "KNOWLEDGE_GAP",
            "VERIFICATION_FAILED",
            "BUDGET_EXCEEDED",
            "TIMEOUT",
            "STATE_CORRUPTED",
            "PERMISSION_DENIED",
            "CONFIGURATION_ERROR",
            "DELEGATION_FAILED",
            "UNKNOWN",
        ]
        for name in expected:
            assert hasattr(ErrorType, name), f"Missing ErrorType.{name}"

    def test_error_has_suggestion(self):
        from infrarely.core.result import Error, ErrorType

        err = Error(
            type=ErrorType.TOOL_FAILURE,
            message="Tool X failed",
            step="step1",
        )
        assert err.suggestion is not None
        assert len(err.suggestion) > 0

    def test_result_never_raises(self):
        """agent.run() should NEVER raise an exception."""
        import infrarely

        agent = infrarely.agent("error_agent")
        # Even with no LLM configured, should return a Result
        result = agent.run("explain the meaning of life in 10000 words")
        # Should be a Result, successful or not
        assert hasattr(result, "success")
        assert hasattr(result, "output")
        assert hasattr(result, "error")
        agent.shutdown()


class TestShutdownAgent:
    """Test that shut-down agents return proper errors."""

    def test_run_after_shutdown(self):
        import infrarely

        agent = infrarely.agent("shutdown_err")
        agent.shutdown()
        result = agent.run("anything")
        assert not result.success
        assert result.error.type == aos.ErrorType.STATE_CORRUPTED


class TestResultExplain:
    """Test result.explain() produces readable output."""

    def test_explain_format(self):
        import infrarely

        agent = infrarely.agent("explain_err")
        result = agent.run("10 + 20")
        explanation = result.explain()
        assert isinstance(explanation, str)
        assert len(explanation) > 10
        # Should contain key info
        assert "10 + 20" in explanation or "completed" in explanation.lower()
        agent.shutdown()

    def test_error_explain(self):
        import infrarely

        agent = infrarely.agent("err_explain")
        agent.shutdown()
        result = agent.run("anything")
        explanation = result.explain()
        assert "error" in explanation.lower() or "fail" in explanation.lower()
