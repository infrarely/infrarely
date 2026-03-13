"""
tests/test_multi_agent.py — Multi-agent tests
═══════════════════════════════════════════════════════════════════════════════
Tests delegation, broadcast, agent registry.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestMultiAgentCreation:
    """Test creating and managing multiple agents."""

    def test_create_multiple_agents(self):
        import infrarely

        a1 = infrarely.agent("multi_a1")
        a2 = infrarely.agent("multi_a2")
        a3 = infrarely.agent("multi_a3")

        assert a1.name == "multi_a1"
        assert a2.name == "multi_a2"
        assert a3.name == "multi_a3"

        a1.shutdown()
        a2.shutdown()
        a3.shutdown()

    def test_agent_registry(self):
        import infrarely
        from infrarely.core.agent import _all_agents

        a1 = infrarely.agent("reg_a1")
        a2 = infrarely.agent("reg_a2")

        agents = _all_agents()
        assert "reg_a1" in agents
        assert "reg_a2" in agents

        a1.shutdown()
        a2.shutdown()

    def test_shutdown_removes_from_registry(self):
        import infrarely
        from infrarely.core.agent import _all_agents, _get_agent

        a = infrarely.agent("shutdown_test")
        assert _get_agent("shutdown_test") is not None
        a.shutdown()
        assert _get_agent("shutdown_test") is None


class TestDelegation:
    """Test agent-to-agent delegation."""

    def test_delegate_math(self):
        import infrarely

        calculator = infrarely.agent("calc_delegate")
        coordinator = infrarely.agent("coord_delegate")

        # Coordinator delegates a math task to calculator
        result = coordinator.delegate(calculator, "5 * 8")
        assert result.success
        assert result.output == 40

        calculator.shutdown()
        coordinator.shutdown()

    def test_delegate_to_non_agent(self):
        import infrarely

        agent = infrarely.agent("delegate_bad")
        result = agent.delegate("not_an_agent", "do something")
        assert not result.success
        assert result.error.type == aos.ErrorType.DELEGATION_FAILED
        agent.shutdown()

    def test_delegate_to_shutdown_agent(self):
        import infrarely

        a = infrarely.agent("delegate_down_a")
        b = infrarely.agent("delegate_down_b")
        b.shutdown()

        result = a.delegate(b, "task")
        assert not result.success
        assert result.error.type == aos.ErrorType.DELEGATION_FAILED
        a.shutdown()


class TestBroadcast:
    """Test broadcast messaging."""

    def test_broadcast_reaches_agents(self):
        import infrarely

        received_messages = []

        a1 = infrarely.agent("bc_sender")
        a2 = infrarely.agent("bc_receiver1")
        a3 = infrarely.agent("bc_receiver2")

        @a2.on_message
        def handle2(from_agent, message, data):
            received_messages.append(("r1", from_agent, message))

        @a3.on_message
        def handle3(from_agent, message, data):
            received_messages.append(("r2", from_agent, message))

        count = a1.broadcast("hello everyone")
        assert count >= 2  # At least a2 and a3
        assert len(received_messages) >= 2
        assert all(m[2] == "hello everyone" for m in received_messages)

        a1.shutdown()
        a2.shutdown()
        a3.shutdown()

    def test_broadcast_with_data(self):
        import infrarely

        received_data = []

        a1 = infrarely.agent("bc_data_sender")
        a2 = infrarely.agent("bc_data_receiver")

        @a2.on_message
        def handler(from_agent, message, data):
            received_data.append(data)

        a1.broadcast("update", data={"score": 95})
        assert len(received_data) >= 1
        assert received_data[0]["score"] == 95

        a1.shutdown()
        a2.shutdown()
