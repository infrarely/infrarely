"""
study_assistant.py — Knowledge-powered study agent
═══════════════════════════════════════════════════════════════════════════════
Shows how an agent can use knowledge to answer questions without LLM calls.
"""

import infrarely

# ── Configure (optional — uses env vars or defaults) ──────────────────────────

infrarely.configure(
    llm_provider="openai",  # or "anthropic", "groq", "gemini", "ollama"
    # api_key="sk-..."      # or set INFRARELY_API_KEY env var
)

# ── Create agent ──────────────────────────────────────────────────────────────

tutor = infrarely.agent("study_tutor", description="Helps students study")

# ── Add knowledge ─────────────────────────────────────────────────────────────

tutor.knowledge.add_data(
    "photosynthesis",
    "Photosynthesis is the process by which plants convert sunlight, "
    "water, and carbon dioxide into glucose and oxygen. It occurs in "
    "the chloroplasts, specifically using chlorophyll pigments.",
)

tutor.knowledge.add_data(
    "mitosis",
    "Mitosis is cell division that produces two identical daughter cells. "
    "The stages are: prophase, metaphase, anaphase, and telophase. "
    "It is used for growth and repair.",
)

tutor.knowledge.add_data(
    "newton_laws",
    "Newton's three laws of motion: "
    "1) An object at rest stays at rest unless acted upon by a force. "
    "2) F = ma (force equals mass times acceleration). "
    "3) For every action, there is an equal and opposite reaction.",
)

# ── Ask questions ─────────────────────────────────────────────────────────────

print("=== Study Assistant ===\n")

# This should be answered from knowledge (no LLM needed)
result = tutor.run("What is photosynthesis?")
print(f"Q: What is photosynthesis?")
print(f"A: {result.output}")
print(f"   Used LLM: {result.used_llm}")
print(f"   Confidence: {result.confidence:.2f}")
print()

# Math is handled deterministically
result = tutor.run("What is 15 * 3 + 10?")
print(f"Q: What is 15 * 3 + 10?")
print(f"A: {result.output}")
print(f"   Used LLM: {result.used_llm}")
print()

# Memory persists across calls
tutor.memory.store("student_name", "Alice")
tutor.memory.store("weak_topics", ["photosynthesis", "mitosis"])
name = tutor.memory.get("student_name")
print(f"Remembered student: {name}")

# ── Check health ──────────────────────────────────────────────────────────────

print(f"\n{tutor.health()}")

# ── Metrics ───────────────────────────────────────────────────────────────────

print(f"\nLLM bypass rate: {infrarely.metrics.llm_bypass_rate():.0f}%")
print(f"Avg task duration: {infrarely.metrics.avg_task_duration():.0f}ms")

tutor.shutdown()
