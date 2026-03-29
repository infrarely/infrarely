"""
Example: Async Agent Execution
═══════════════════════════════════════════════════════════════════════════════
Shows non-blocking execution for web servers and async applications.

Run: python examples/async_demo.py
"""

import asyncio
import infrarely

infrarely.configure(llm_provider="openai", log_file_enabled=False)


@infrarely.tool
def lookup(topic: str) -> str:
    """Look up information about a topic."""
    info = {
        "mars": "Mars is the 4th planet from the Sun, known as the Red Planet.",
        "venus": "Venus is the 2nd planet from the Sun, hottest planet.",
        "jupiter": "Jupiter is the largest planet in our solar system.",
    }
    return info.get(topic.lower(), f"No info on {topic}")


researcher = infrarely.agent("researcher", tools=[lookup])
writer = infrarely.agent("writer")
analyst = infrarely.agent("analyst")


async def main():
    # ── Single async run ──────────────────────────────────────────────────
    print("─── Single Async Run ───")
    result = await researcher.arun("What is 2 + 2?")
    print(f"Result: {result.output}")
    print(f"Used LLM: {result.used_llm}")

    # ── Parallel async execution ──────────────────────────────────────────
    print("\n─── Parallel Async Execution (3 tasks) ───")
    results = await aos.async_gather(
        (researcher, "What is 10 * 10?"),
        (writer, "What is 7 + 3?"),
        (analyst, "What is 144 / 12?"),
    )

    for i, r in enumerate(results, 1):
        print(
            f"  Task {i}: {r.output} (success={r.success}, duration={r.duration_ms:.0f}ms)"
        )

    # ── Async delegation ──────────────────────────────────────────────────
    print("\n─── Async Delegation ───")
    result = await aos.async_delegate(researcher, writer, "What is 5 + 5?")
    print(f"Delegated result: {result.output}")

    # ── Async parallel with more tasks ────────────────────────────────────
    print("\n─── Async Parallel (batch) ───")
    tasks = [(researcher, f"What is {i} * {i}?") for i in range(2, 6)]
    results = await aos.async_parallel(tasks)
    for i, r in enumerate(results):
        print(f"  {i+2}² = {r.output}")

    print("\nAll async operations completed!")


if __name__ == "__main__":
    asyncio.run(main())
    infrarely.shutdown()
