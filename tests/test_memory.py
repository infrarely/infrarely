"""
tests/test_memory.py — Memory system tests
═══════════════════════════════════════════════════════════════════════════════
Tests memory store/get/forget/clear/search across scopes.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestMemoryBasics:
    """Test basic memory operations."""

    def test_store_and_get(self):
        import infrarely

        agent = infrarely.agent("mem_test_1")
        agent.memory.store("name", "Alice")
        assert agent.memory.get("name") == "Alice"
        agent.shutdown()

    def test_store_dict(self):
        import infrarely

        agent = infrarely.agent("mem_test_2")
        data = {"course": "CS301", "grade": "A"}
        agent.memory.store("student_record", data)
        retrieved = agent.memory.get("student_record")
        assert retrieved["course"] == "CS301"
        assert retrieved["grade"] == "A"
        agent.shutdown()

    def test_get_nonexistent(self):
        import infrarely

        agent = infrarely.agent("mem_test_3")
        result = agent.memory.get("nonexistent_key_xyz")
        assert result is None
        agent.shutdown()

    def test_forget(self):
        import infrarely

        agent = infrarely.agent("mem_test_4")
        agent.memory.store("temp", "value")
        assert agent.memory.get("temp") == "value"
        agent.memory.forget("temp")
        assert agent.memory.get("temp") is None
        agent.shutdown()

    def test_has(self):
        import infrarely

        agent = infrarely.agent("mem_test_5")
        agent.memory.store("exists", "yes")
        assert agent.memory.has("exists") is True
        assert agent.memory.has("nope") is False
        agent.shutdown()


class TestMemoryScopes:
    """Test memory scopes: session, permanent, shared."""

    def test_session_scope(self):
        import infrarely

        agent = infrarely.agent("mem_scope_1")
        agent.memory.store("session_key", "session_val", scope="session")
        assert agent.memory.get("session_key") == "session_val"
        agent.shutdown()

    def test_permanent_scope(self):
        import infrarely

        agent = infrarely.agent("mem_scope_2")
        agent.memory.store("perm_key", "perm_val", scope="permanent")
        assert agent.memory.get("perm_key") == "perm_val"
        agent.shutdown()

    def test_shared_scope(self):
        import infrarely

        agent1 = infrarely.agent("mem_shared_1")
        agent2 = infrarely.agent("mem_shared_2")

        agent1.memory.store("shared_data", "hello_from_1", scope="shared")
        result = agent2.memory.get("shared_data")
        # Shared memory should be accessible across agents
        assert result == "hello_from_1"

        agent1.shutdown()
        agent2.shutdown()


class TestMemoryClear:
    """Test memory clearing."""

    def test_clear_all(self):
        import infrarely

        agent = infrarely.agent("mem_clear_1")
        agent.memory.store("s1", "v1", scope="session")
        agent.memory.store("s2", "v2", scope="session")
        agent.memory.clear()
        assert agent.memory.get("s1") is None
        assert agent.memory.get("s2") is None
        agent.shutdown()

    def test_list_keys(self):
        import infrarely

        agent = infrarely.agent("mem_list_1")
        agent.memory.store("key_a", "a")
        agent.memory.store("key_b", "b")
        keys = agent.memory.list_keys()
        assert "key_a" in keys
        assert "key_b" in keys
        agent.shutdown()
