"""
production_deploy.py — Production-ready agent with monitoring
═══════════════════════════════════════════════════════════════════════════════
Shows health endpoints, dashboards, metrics, error handling, and workflows.
"""

import infrarely
import time

# ── Configure for production ──────────────────────────────────────────────────

infrarely.configure(
    log_level="INFO",
    memory_backend="sqlite",
    max_retries=3,
)

# ── Define robust tools ───────────────────────────────────────────────────────


@infrarely.tool(
    retries=3,
    timeout=5.0,
    cache=True,
    cache_ttl=300,
    description="Fetch student grades from the database",
)
def get_grades(student_id: str) -> dict:
    """Simulated grade lookup with caching and retries."""
    return {
        "student": student_id,
        "courses": {"CS101": "A", "MATH201": "B+", "ENG102": "A-"},
        "gpa": 3.7,
    }


@infrarely.tool(retries=2, timeout=10.0, description="Generate a PDF report")
def generate_report(data: dict) -> str:
    """Simulated report generation."""
    return f"Report generated for {data.get('student', 'unknown')}: GPA {data.get('gpa', 'N/A')}"


@infrarely.tool(description="Send email notification")
def send_email(to: str, subject: str, body: str) -> dict:
    """Simulated email sending."""
    return {"sent": True, "to": to, "subject": subject}


# ── Build a workflow ──────────────────────────────────────────────────────────

report_workflow = infrarely.workflow(
    "academic_report",
    infrarely.step("fetch", get_grades, required=True),
    infrarely.step("report", generate_report, depends_on=["fetch"]),
    infrarely.step("notify", send_email, depends_on=["report"], required=False),
)

# ── Create the agent ──────────────────────────────────────────────────────────

agent = infrarely.agent(
    "academic_advisor",
    description="Handles academic queries and generates reports",
    tools=[get_grades, generate_report, send_email],
)

# ── Event hooks ───────────────────────────────────────────────────────────────


@agent.on("task_start")
def on_start(data):
    print(f"  [EVENT] Task started: {data.get('goal', '')[:50]}")


@agent.on("task_complete")
def on_complete(data):
    duration = data.get("duration_ms", 0)
    print(f"  [EVENT] Task completed in {duration:.0f}ms")


@agent.on("error")
def on_error(data):
    print(f"  [EVENT] Error: {data.get('error', 'unknown')}")


# ── Run tasks ─────────────────────────────────────────────────────────────────

print("=== Production Agent Demo ===\n")

# Task 1: Simple query
result = agent.run("What is 15% of 240?")
print(f"Math: {result.output}")
print(f"  Used LLM: {result.used_llm}")
print()

# Task 2: Tool-powered query (if tools are registered)
result2 = agent.run("get grades for student S12345")
print(f"Grades: {result2}")
print()

# ── Observability ─────────────────────────────────────────────────────────────

print("--- Metrics ---")
metrics = infrarely.metrics.summary()
print(f"  Total tasks:    {metrics.get('total_tasks', 0)}")
print(f"  Success rate:   {metrics.get('success_rate', 0):.1%}")
print(f"  LLM bypass:     {metrics.get('llm_bypass_rate', 0):.1%}")
print(f"  Avg latency:    {metrics.get('avg_duration_ms', 0):.0f}ms")
print()

# ── Health check ──────────────────────────────────────────────────────────────

print("--- Health ---")
health = infrarely.health()
print(f"  Status: {health.get('status', 'unknown')}")
print(f"  Agents: {health.get('agents', 0)}")
print(f"  Uptime: {health.get('uptime_seconds', 0):.0f}s")
print()

# ── Execution traces ─────────────────────────────────────────────────────────

print("--- Recent Traces ---")
traces = agent.get_recent_traces(limit=3)
for t in traces:
    print(
        f"  [{t.trace_id[:8]}] {t.goal[:40]} → {'OK' if t.success else 'FAIL'} ({t.duration_ms:.0f}ms)"
    )
print()

# ── Agent explanation ─────────────────────────────────────────────────────────

print("--- Agent Status ---")
print(agent.explain())

# ── Cleanup ───────────────────────────────────────────────────────────────────

infrarely.shutdown()
print("\nAll agents shut down cleanly.")
