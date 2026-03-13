"""
Example: Agent Versioning & Rollback
═══════════════════════════════════════════════════════════════════════════════
Shows how to version, compare, and rollback agent configurations.

Run: python examples/versioning_demo.py
"""

import infrarely

infrarely.configure(llm_provider="openai", log_file_enabled=False)


@infrarely.tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception:
        return "Error in expression"


# ── Create agent and save initial version ─────────────────────────────────────
agent = infrarely.agent("versioned-bot", tools=[calculate])
agent.knowledge.add_data("greeting", "Hello! I'm the versioned bot.")

# Save v1
aos.versions.save(agent, tag="v1.0-stable", description="Initial version")
print("Saved version v1.0-stable")

# ── Make changes ──────────────────────────────────────────────────────────────
agent.knowledge.add_data("physics", "E = mc² describes mass-energy equivalence.")
agent.knowledge.add_data("math", "The Pythagorean theorem: a² + b² = c²")

# Save v2
aos.versions.save(agent, tag="v1.1-beta", description="Added physics knowledge")
print("Saved version v1.1-beta")

# ── List versions ─────────────────────────────────────────────────────────────
print("\n─── All Versions ───")
all_versions = aos.versions.list_versions("versioned-bot")
for v in all_versions:
    print(f"  {v.tag}: {v.description} ({len(v.knowledge_data)} knowledge entries)")

# ── Compare versions ──────────────────────────────────────────────────────────
print("\n─── Version Comparison ───")
comparison = aos.versions.compare("v1.0-stable", "v1.1-beta")
print(f"  Knowledge added: {comparison.knowledge_added}")
print(f"  Knowledge removed: {comparison.knowledge_removed}")
print(f"  Tools changed: {comparison.tools_changed}")

# ── Rollback ──────────────────────────────────────────────────────────────────
print("\n─── Rollback to v1.0 ───")
aos.versions.rollback(agent, tag="v1.0-stable")
print("Rolled back!")

# Verify rollback worked
result = agent.run("What is the Pythagorean theorem?")
print(f"After rollback, physics query: confidence={result.confidence:.2f}")
# Should have lower confidence since physics knowledge was removed

infrarely.shutdown()
