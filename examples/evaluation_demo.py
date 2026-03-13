"""
Example: Agent Evaluation & Regression Testing
═══════════════════════════════════════════════════════════════════════════════
Shows how to create evaluation suites and run regression tests.

Run: python examples/evaluation_demo.py
"""

import infrarely

infrarely.configure(llm_provider="openai", log_file_enabled=False)


@infrarely.tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@infrarely.tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


agent = infrarely.agent("math-agent", tools=[add, multiply])
agent.knowledge.add_data("pi", "Pi is approximately 3.14159")
agent.knowledge.add_data("euler", "Euler's number e is approximately 2.71828")


# ── Create evaluation suite ───────────────────────────────────────────────────
suite = aos.eval.suite("math-agent-evals")

# Test math operations (should bypass LLM)
suite.add(
    input="What is 2 + 2?",
    expected_output=4,
    expected_used_llm=False,
    expected_confidence_min=0.99,
)

suite.add(
    input="What is 10 * 5?",
    expected_output=50,
    expected_used_llm=False,
)

suite.add(
    input="What is 100 / 4?",
    expected_output=25.0,
    expected_used_llm=False,
)

# Test tool execution
suite.add(
    input="add 15 and 25",
    expected_output=40.0,
    expected_used_llm=False,
)

# Test knowledge retrieval
suite.add(
    input="What is pi?",
    expected_sources=["pi"],
    expected_used_llm=False,
)

# Test expected success
suite.add(
    input="What is 3 + 7?",
    expected_output=10,
    expected_success=True,
)


# ── Run evaluation ────────────────────────────────────────────────────────────
print("─── Running Evaluation Suite ───\n")
results = suite.run(agent)

# Print detailed results
for cr in results.case_results:
    status = "✓ PASS" if cr.passed else "✗ FAIL"
    print(f"  [{status}] {cr.case.input}")
    if not cr.passed:
        for failure in cr.failures:
            print(f"         → {failure}")
    else:
        print(f"         output={cr.actual_output}, llm={cr.actual_used_llm}")

# Summary
print(f"\n─── Summary ───")
print(f"  Pass Rate: {results.pass_rate:.0%}")
print(f"  Passed:    {results.passed_count}/{results.total_count}")
print(f"  Duration:  {results.total_duration_ms:.0f}ms")

if results.regression_report:
    print(f"\n  Regression Report:")
    print(f"  {results.regression_report}")

# ── Quick eval shortcut ──────────────────────────────────────────────────────
print(f"\n─── Quick Eval ───")
quick = aos.eval.quick_eval(
    agent,
    input="What is 5 * 5?",
    expected_output=25,
    expected_used_llm=False,
)
print(f"  Quick eval passed: {quick}")

infrarely.shutdown()
