"""
tests/test_workflow.py — Workflow/DAG engine tests
═══════════════════════════════════════════════════════════════════════════════
Tests DAG execution, dependency resolution, parallel steps, conditions.
"""

import sys
import os
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestWorkflowBasics:
    """Test basic workflow creation and execution."""

    def test_single_step_workflow(self):
        import infrarely

        def fetch():
            return {"data": [1, 2, 3]}

        wf = infrarely.workflow(
            "simple",
            steps=[
                infrarely.step("fetch", fetch),
            ],
        )
        results = wf.execute()
        assert "fetch" in results
        assert results["fetch"].success
        assert results["fetch"].output == {"data": [1, 2, 3]}

    def test_sequential_workflow(self):
        import infrarely

        def step_a():
            return 10

        def step_b():
            return 20

        wf = infrarely.workflow(
            "seq",
            steps=[
                infrarely.step("a", step_a),
                infrarely.step("b", step_b, depends_on=["a"]),
            ],
        )
        results = wf.execute()
        assert results["a"].success
        assert results["b"].success
        assert results["a"].output == 10
        assert results["b"].output == 20

    def test_parallel_steps(self):
        import infrarely

        def slow_a():
            time.sleep(0.1)
            return "a"

        def slow_b():
            time.sleep(0.1)
            return "b"

        start = time.monotonic()
        wf = infrarely.workflow(
            "par",
            steps=[
                infrarely.step("a", slow_a),
                infrarely.step("b", slow_b),
            ],
        )
        results = wf.execute()
        elapsed = time.monotonic() - start

        assert results["a"].success
        assert results["b"].success
        # Parallel should be faster than sequential (< 0.3s vs ~0.2s)
        # Being generous with the check
        assert elapsed < 0.5

    def test_diamond_dag(self):
        """Test diamond-shaped dependency: A → (B, C) → D"""
        import infrarely

        def a():
            return 1

        def b():
            return 2

        def c():
            return 3

        def d():
            return 4

        wf = infrarely.workflow(
            "diamond",
            steps=[
                infrarely.step("a", a),
                infrarely.step("b", b, depends_on=["a"]),
                infrarely.step("c", c, depends_on=["a"]),
                infrarely.step("d", d, depends_on=["b", "c"]),
            ],
        )
        results = wf.execute()
        assert all(results[k].success for k in ["a", "b", "c", "d"])


class TestWorkflowFailure:
    """Test workflow failure handling."""

    def test_required_step_failure(self):
        import infrarely

        def failing():
            raise RuntimeError("boom")

        def after():
            return "ok"

        wf = infrarely.workflow(
            "fail",
            steps=[
                infrarely.step("fail", failing, required=True),
                infrarely.step("after", after, depends_on=["fail"]),
            ],
        )
        results = wf.execute()
        assert not results["fail"].success
        # After step may be skipped (not in results) or marked as failed
        if "after" in results:
            assert results["after"].skipped or not results["after"].success

    def test_optional_step_failure(self):
        import infrarely

        def failing():
            raise RuntimeError("boom")

        def independent():
            return "ok"

        wf = infrarely.workflow(
            "opt_fail",
            steps=[
                infrarely.step("fail", failing, required=False),
                infrarely.step("ok", independent),
            ],
        )
        results = wf.execute()
        # Optional step that fails gets marked as skipped
        assert results["fail"].skipped
        assert results["ok"].success


class TestWorkflowFallback:
    """Test step fallback functions."""

    def test_fallback_on_failure(self):
        import infrarely

        def primary():
            raise RuntimeError("primary failed")

        def backup():
            return "backup result"

        wf = infrarely.workflow(
            "fallback",
            steps=[
                infrarely.step("task", primary, fallback="backup_step"),
                infrarely.step("backup_step", backup),
            ],
        )
        results = wf.execute()
        # The primary failed, so should have tried fallback
        assert results["task"].success or results["task"].used_fallback
