"""
tests/test_knowledge.py — Knowledge layer tests
═══════════════════════════════════════════════════════════════════════════════
Tests knowledge ingestion, querying, decision gate, LLM bypass.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestKnowledgeBasics:
    """Test basic knowledge operations."""

    def test_add_data(self):
        import infrarely
        from infrarely.memory.knowledge import get_knowledge_manager

        km = get_knowledge_manager()
        km.add_data("test_fact", "The speed of light is 299,792,458 meters per second")

        result = km.query("speed of light")
        assert result is not None
        assert result.confidence > 0
        assert len(result.chunks) > 0

    def test_knowledge_decision_gate(self):
        import infrarely
        from infrarely.memory.knowledge import get_knowledge_manager

        km = get_knowledge_manager()
        km.add_data("water_boiling", "Water boils at 100 degrees Celsius at sea level")

        result = km.query("at what temperature does water boil")
        assert result.decision in (
            "bypass_llm",
            "ground_llm",
            "low_confidence",
            "no_knowledge",
        )

    def test_knowledge_no_match(self):
        from infrarely.memory.knowledge import get_knowledge_manager

        km = get_knowledge_manager()
        result = km.query("xyzzy_nonexistent_topic_12345")
        assert result.decision in ("low_confidence", "no_knowledge")

    def test_agent_knowledge_integration(self):
        """Test that agent uses knowledge to answer questions."""
        import infrarely

        agent = infrarely.agent("knowledge_agent")
        agent.knowledge.add_data(
            "python_creator", "Python was created by Guido van Rossum in 1991"
        )
        agent.knowledge.add_data(
            "python_typing", "Python is dynamically typed and supports duck typing"
        )

        # Query knowledge directly to verify it works
        kr = agent.knowledge.query("Who created Python?")
        assert kr.confidence > 0
        assert len(kr.chunks) > 0
        # At least one chunk should mention Guido
        found = any("Guido" in c.content for c in kr.chunks)
        assert found, f"Expected 'Guido' in chunks: {[c.content for c in kr.chunks]}"
        agent.shutdown()


class TestKnowledgeSource:
    """Test knowledge source management."""

    def test_add_multiple_facts(self):
        from infrarely.memory.knowledge import get_knowledge_manager

        km = get_knowledge_manager()
        facts = {
            "earth_mass": "Earth's mass is approximately 5.972 × 10^24 kg",
            "earth_radius": "Earth's mean radius is 6,371 km",
            "earth_age": "Earth is approximately 4.54 billion years old",
        }
        for key, value in facts.items():
            km.add_data(key, value)

        result = km.query("how old is Earth")
        assert result.confidence > 0
        assert any(
            "billion" in c.content.lower() or "4.54" in c.content for c in result.chunks
        )
