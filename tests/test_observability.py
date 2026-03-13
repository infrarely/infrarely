"""
tests/test_observability.py — Observability tests
═══════════════════════════════════════════════════════════════════════════════
Tests traces, metrics, health reports, dashboard.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestTraces:
    """Test execution trace recording and retrieval."""

    def test_trace_recorded(self):
        import infrarely

        agent = infrarely.agent("trace_test")
        result = agent.run("3 + 3")
        assert result.trace_id
        assert isinstance(result.trace_id, str)
        assert len(result.trace_id) > 0
        agent.shutdown()

    def test_get_trace(self):
        import infrarely

        agent = infrarely.agent("trace_get")
        result = agent.run("7 * 8")
        trace = agent.get_trace(result.trace_id)
        assert trace is not None
        assert trace.goal == "7 * 8"
        assert trace.agent_name == "trace_get"
        agent.shutdown()

    def test_recent_traces(self):
        import infrarely

        agent = infrarely.agent("trace_recent")
        agent.run("1 + 1")
        agent.run("2 + 2")
        agent.run("3 + 3")

        traces = agent.get_recent_traces(limit=10)
        assert len(traces) >= 3
        agent.shutdown()


class TestMetrics:
    """Test metrics collection."""

    def test_metrics_singleton(self):
        import infrarely

        m1 = aos.metrics
        m2 = aos.metrics
        assert m1 is m2

    def test_bypass_rate(self):
        import infrarely

        # Reset metrics for clean test
        infrarely.metrics.reset()

        agent = infrarely.agent("metrics_bypass")
        # Math tasks bypass LLM
        agent.run("2 + 2")
        agent.run("3 * 3")
        agent.run("10 / 2")

        rate = infrarely.metrics.llm_bypass_rate()
        assert rate > 0  # Should have bypassed LLM for math
        agent.shutdown()

    def test_export_json(self):
        import infrarely

        data = infrarely.metrics.export(format="json")
        assert isinstance(data, dict)
        assert "total_tasks" in data
        assert "llm_bypass_rate" in data
        assert "failure_rate" in data

    def test_export_prometheus(self):
        import infrarely

        data = infrarely.metrics.export(format="prometheus")
        assert isinstance(data, str)
        assert "aos_total_tasks" in data


class TestHealth:
    """Test health reporting."""

    def test_agent_health(self):
        import infrarely

        agent = infrarely.agent("health_test")
        agent.run("1 + 1")

        report = agent.health()
        assert report.agent_name == "health_test"
        assert report.state == "IDLE"
        assert report.uptime_seconds > 0
        assert report.tools_registered >= 0
        agent.shutdown()

    def test_system_health(self):
        import infrarely

        agent = infrarely.agent("sys_health")
        report = infrarely.health()
        assert "sdk_version" in report
        assert "agents" in report
        assert "metrics" in report
        agent.shutdown()

    def test_health_str(self):
        import infrarely

        agent = infrarely.agent("health_str")
        report = agent.health()
        s = str(report)
        assert "health_str" in s
        assert "State" in s
        agent.shutdown()
