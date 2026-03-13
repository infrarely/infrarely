"""
tests/test_beginner.py — Quality Gate 1: Beginner-friendly tests
═══════════════════════════════════════════════════════════════════════════════
Could a developer who knows basic Python but has never heard of
LangChain, embeddings, or vector stores use this without reading docs?
"""

import sys
import os
import pytest

# Ensure aos package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestImport:
    """Test that the basic import works."""

    def test_import_aos(self):
        import infrarely

        assert hasattr(aos, "agent")
        assert hasattr(aos, "tool")
        assert hasattr(aos, "configure")
        assert hasattr(aos, "shutdown")

    def test_version(self):
        import infrarely

        assert aos.__version__ == "0.1.0"


class TestAgentCreation:
    """Test that agents can be created with minimal code."""

    def test_create_agent_one_line(self):
        import infrarely

        agent = infrarely.agent("test_beginner")
        assert agent.name == "test_beginner"
        assert agent.state == "IDLE"
        agent.shutdown()

    def test_agent_has_memory(self):
        import infrarely

        agent = infrarely.agent("test_mem_beginner")
        assert agent.memory is not None
        agent.shutdown()

    def test_agent_has_knowledge(self):
        import infrarely

        agent = infrarely.agent("test_know_beginner")
        assert agent.knowledge is not None
        agent.shutdown()

    def test_agent_repr(self):
        import infrarely

        agent = infrarely.agent("repr_test")
        r = repr(agent)
        assert "repr_test" in r
        assert "IDLE" in r
        agent.shutdown()


class TestMathEval:
    """Test deterministic math evaluation (no LLM needed)."""

    def test_simple_addition(self):
        import infrarely

        agent = infrarely.agent("math_beginner")
        result = agent.run("2 + 2")
        assert result.success
        assert result.output == 4
        assert result.used_llm is False
        assert result.confidence == 1.0
        agent.shutdown()

    def test_complex_math(self):
        import infrarely

        agent = infrarely.agent("math_complex")
        result = agent.run("(10 + 5) * 3")
        assert result.success
        assert result.output == 45
        assert result.used_llm is False
        agent.shutdown()


class TestResultStructure:
    """Test that Result is always structured and never raises."""

    def test_result_has_all_fields(self):
        import infrarely

        agent = infrarely.agent("result_test")
        result = agent.run("2 + 2")
        assert hasattr(result, "output")
        assert hasattr(result, "success")
        assert hasattr(result, "confidence")
        assert hasattr(result, "used_llm")
        assert hasattr(result, "sources")
        assert hasattr(result, "duration_ms")
        assert hasattr(result, "trace_id")
        assert isinstance(result.duration_ms, float)
        assert isinstance(result.trace_id, str)
        agent.shutdown()

    def test_result_explain(self):
        import infrarely

        agent = infrarely.agent("explain_test")
        result = agent.run("3 * 7")
        explanation = result.explain()
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        agent.shutdown()

    def test_agent_explain(self):
        import infrarely

        agent = infrarely.agent("agent_explain_test")
        result = agent.run("5 + 5")
        explanation = agent.explain()
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        agent.shutdown()

    def test_agent_no_run_explain(self):
        import infrarely

        agent = infrarely.agent("no_run_explain")
        explanation = agent.explain()
        assert "has not run" in explanation
        agent.shutdown()
