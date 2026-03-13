"""
Example: Context Window Management
═══════════════════════════════════════════════════════════════════════════════
Shows automatic management of context when it exceeds model token limits.

Run: python examples/context_window_demo.py
"""

import infrarely
from infrarely.core.context import ContextWindowManager, ContextStrategy

infrarely.configure(
    llm_provider="openai",
    log_file_enabled=False,
    context_strategy="sliding_window",
    max_context_tokens=8000,
)


# ── Create and manage a context window ────────────────────────────────────────
ctx = ContextWindowManager(max_tokens=500, strategy=ContextStrategy.SLIDING_WINDOW)

# Add system prompt (always preserved)
ctx.add_system("You are a helpful tutor.")

# Add conversation history
ctx.add_user("What is photosynthesis?")
ctx.add_assistant(
    "Photosynthesis is the process by which plants convert light into energy."
)

ctx.add_user("How does the Calvin cycle work?")
ctx.add_assistant("The Calvin cycle is the light-independent stage of photosynthesis.")

ctx.add_user("What about cellular respiration?")
ctx.add_assistant(
    "Cellular respiration is the process of converting glucose to ATP energy."
)

# Check token usage
print(f"─── Context Window Stats ───")
print(f"  Total tokens: {ctx.total_tokens}")
print(f"  Max tokens: {ctx.max_tokens}")
print(f"  Messages: {len(ctx.messages)}")
print(f"  Strategy: {ctx.strategy.value}")

# Add more messages to trigger overflow
for i in range(10):
    ctx.add_user(f"Tell me about topic {i}: " + "x" * 50)
    ctx.add_assistant(f"Here is information about topic {i}: " + "y" * 100)

print(f"\n─── After Overflow ───")
print(f"  Total tokens: {ctx.total_tokens}")
print(f"  Messages: {len(ctx.messages)}")
print(f"  Overflow handled: tokens stayed under limit")

# Export messages for LLM
messages = ctx.to_messages()
print(f"\n─── Messages for LLM ({len(messages)}) ───")
for m in messages[:5]:
    content_preview = (
        m["content"][:50] + "..." if len(m["content"]) > 50 else m["content"]
    )
    print(f"  [{m['role']}] {content_preview}")
if len(messages) > 5:
    print(f"  ... and {len(messages) - 5} more")

# ── Priority strategy ─────────────────────────────────────────────────────────
print(f"\n─── Priority Strategy ───")
ctx2 = ContextWindowManager(max_tokens=300, strategy=ContextStrategy.PRIORITY)
ctx2.add_system("You are an expert.")
ctx2.add_user("What is ML?", priority=1.0)
ctx2.add_assistant("ML is machine learning.", priority=0.8)
ctx2.add_user("What is DL?", priority=0.5)
ctx2.add_assistant("DL is deep learning.", priority=0.3)
ctx2.add_user("What is AI?", priority=0.9)

print(f"  Messages: {len(ctx2.messages)}")
print(f"  Tokens: {ctx2.total_tokens}")

infrarely.shutdown()
