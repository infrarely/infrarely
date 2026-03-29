"""
research_team.py — Multi-agent delegation and collaboration
═══════════════════════════════════════════════════════════════════════════════
Shows how multiple agents can delegate tasks and share memory.
"""

import infrarely

# ── Create a team of agents ──────────────────────────────────────────────────

researcher = infrarely.agent("researcher", description="Finds and organizes facts")
writer = infrarely.agent("writer", description="Writes clear summaries")
reviewer = infrarely.agent("reviewer", description="Reviews work for accuracy")

# ── Add specialized knowledge ─────────────────────────────────────────────────

researcher.knowledge.add_data(
    "mars_facts",
    "Mars is the fourth planet from the Sun. It has a thin atmosphere "
    "composed mostly of carbon dioxide. Mars has two small moons: "
    "Phobos and Deimos. The average temperature is -80°F (-62°C).",
)

researcher.knowledge.add_data(
    "mars_exploration",
    "NASA's Perseverance rover landed on Mars in February 2021. "
    "It is searching for signs of ancient microbial life. "
    "The Ingenuity helicopter was the first powered flight on another planet.",
)

# ── Delegation chain ──────────────────────────────────────────────────────────

print("=== Research Team ===\n")

# 1. Researcher gathers facts
facts = researcher.run("What do we know about Mars?")
print(f"Researcher found: {str(facts.output)[:100]}...")
print(f"  Confidence: {facts.confidence:.2f}")
print()

# 2. Writer summarizes (delegated by reviewer)
summary = reviewer.delegate(writer, "Summarize the Mars facts", context=facts)
print(f"Writer produced: {str(summary.output)[:100]}...")
print()

# ── Broadcast a message ──────────────────────────────────────────────────────

received = []


@writer.on_message
def writer_handler(from_agent, message, data):
    received.append(f"{from_agent}: {message}")


@reviewer.on_message
def reviewer_handler(from_agent, message, data):
    received.append(f"{from_agent}: {message}")


researcher.broadcast("Research phase complete!")
print(f"Broadcast delivered to {len(received)} agents")
for msg in received:
    print(f"  {msg}")

# ── Shared memory ────────────────────────────────────────────────────────────

researcher.memory.store("project_status", "research_complete", scope="shared")
status = writer.memory.get("project_status")
print(f"\nShared memory - project status: {status}")

# ── System health ─────────────────────────────────────────────────────────────

print(f"\nSystem health:")
health = infrarely.health()
print(f"  Active agents: {health['agents']}")
print(f"  Total tasks: {health['metrics']['total_tasks']}")

# ── Cleanup ───────────────────────────────────────────────────────────────────

infrarely.shutdown()
