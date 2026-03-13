"""
weather_agent.py — Agent with a custom tool
═══════════════════════════════════════════════════════════════════════════════
Shows how to add custom tools to an agent using @infrarely.tool.
"""

import infrarely

# ── Define a tool ─────────────────────────────────────────────────────────────


@infrarely.tool
def get_weather(city: str) -> dict:
    """Get current weather for a city."""
    # In production, call a real API here
    weather_data = {
        "New York": {"temp": 72, "condition": "Sunny"},
        "London": {"temp": 58, "condition": "Cloudy"},
        "Tokyo": {"temp": 80, "condition": "Humid"},
    }
    return weather_data.get(city, {"temp": 0, "condition": "Unknown"})


# ── Create agent with the tool ───────────────────────────────────────────────

agent = infrarely.agent("weather_bot", tools=[get_weather])

# ── Run ───────────────────────────────────────────────────────────────────────

result = agent.run("get weather New York")
print(f"Output: {result.output}")
print(f"Used LLM: {result.used_llm}")
print(f"Confidence: {result.confidence}")
print(f"Duration: {result.duration_ms:.0f}ms")

# ── Explain what happened ─────────────────────────────────────────────────────

print("\n" + result.explain())

agent.shutdown()
