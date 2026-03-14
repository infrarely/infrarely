"""
infrarely/_internal/knowledge_bridge.py — SDK ↔ Knowledge Layer bridge
═══════════════════════════════════════════════════════════════════════════════
Connects SDK knowledge API to InfraRely Knowledge Layer infrastructure.
Handles knowledge source registration, indexing, and querying.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from infrarely.memory.knowledge import KnowledgeManager, KnowledgeResult, get_knowledge_manager


def get_or_create_knowledge() -> KnowledgeManager:
    """Get the global knowledge manager, creating if needed."""
    return get_knowledge_manager()


def build_grounded_prompt(query: str, knowledge_result: KnowledgeResult) -> str:
    """
    Build a grounded prompt that constrains LLM to use only provided facts.
    Prevents hallucination by instruction.
    """
    if not knowledge_result or not knowledge_result.chunks:
        return query

    context_lines = []
    for chunk in knowledge_result.chunks[:5]:
        context_lines.append(
            f"- {chunk.content} (source: {chunk.source}, confidence: {chunk.confidence:.2f})"
        )

    return (
        f"FACTS (ONLY use these, do NOT add anything else):\n"
        f"{'chr(10)'.join(context_lines)}\n\n"
        f"QUESTION: {query}\n\n"
        f"RULES:\n"
        f"- Answer using ONLY the facts above\n"
        f"- If the facts don't contain the answer, say 'I don't have enough information'\n"
        f"- Do NOT hallucinate or add information not in the facts"
    )
