"""
Example: Streaming Responses
═══════════════════════════════════════════════════════════════════════════════
Shows token-by-token streaming output for real-time UIs.

Run: python examples/streaming_demo.py
"""

import infrarely

infrarely.configure(llm_provider="openai", log_file_enabled=False)


@infrarely.tool
def explain_concept(topic: str) -> str:
    """Explain a concept."""
    explanations = {
        "photosynthesis": (
            "Photosynthesis is the process by which plants convert sunlight, "
            "water, and carbon dioxide into glucose and oxygen. It occurs in "
            "the chloroplasts of plant cells, using chlorophyll to capture "
            "light energy. The process has two stages: light-dependent "
            "reactions and the Calvin cycle."
        ),
        "gravity": (
            "Gravity is a fundamental force of nature that attracts objects "
            "with mass toward each other. On Earth, gravity gives weight to "
            "objects and causes them to fall when dropped. Einstein's general "
            "relativity describes gravity as the curvature of spacetime."
        ),
    }
    return explanations.get(topic.lower(), f"Information about {topic}")


agent = infrarely.agent("streamer", tools=[explain_concept])
agent.knowledge.add_data(
    "ml",
    (
        "Machine learning is a subset of artificial intelligence that enables "
        "computers to learn from data without being explicitly programmed. "
        "Key types include supervised, unsupervised, and reinforcement learning."
    ),
)


# ── Synchronous streaming ─────────────────────────────────────────────────────
print("─── Streaming: Explain Photosynthesis ───")
print("Output: ", end="")
stream = agent.stream("Explain photosynthesis")
for chunk in stream:
    print(chunk, end="", flush=True)
print()

# Access metadata after streaming
if stream.result:
    print(f"\nConfidence: {stream.result.confidence:.2f}")
    print(f"Used LLM: {stream.result.used_llm}")
    print(f"Duration: {stream.result.duration_ms:.0f}ms")

# ── Stream math (instant) ─────────────────────────────────────────────────────
print("\n─── Streaming: Math ───")
print("Output: ", end="")
for chunk in agent.stream("What is 15 * 7?"):
    print(chunk, end="", flush=True)
print()

# ── Stream knowledge ──────────────────────────────────────────────────────────
print("\n─── Streaming: Knowledge ───")
print("Output: ", end="")
for chunk in agent.stream("What is machine learning?"):
    print(chunk, end="", flush=True)
print()

# ── Async example (uncomment to run with asyncio) ────────────────────────────
# import asyncio
# async def async_demo():
#     print("\n─── Async Streaming ───")
#     print("Output: ", end="")
#     async for chunk in agent.astream("Explain gravity"):
#         print(chunk, end="", flush=True)
#     print()
# asyncio.run(async_demo())

infrarely.shutdown()
