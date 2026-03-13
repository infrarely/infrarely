"""
tests/test_config.py — Configuration tests
═══════════════════════════════════════════════════════════════════════════════
Tests configure(), auto-detection, zero-config defaults.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestConfiguration:
    """Test configuration system."""

    def test_configure_basic(self):
        import infrarely

        infrarely.configure(llm_provider="openai")
        from infrarely.core.config import get_config

        cfg = get_config()
        assert cfg.get("llm_provider") == "openai"

    def test_default_values(self):
        from infrarely.core.config import get_config

        cfg = get_config()
        # Should have sane defaults
        assert cfg.get("llm_provider") is not None
        assert cfg.get("token_budget") is not None
        assert cfg.get("token_budget") == 10000
        assert cfg.get("max_retries") == 2

    def test_configure_memory(self):
        import infrarely

        infrarely.configure(memory_enabled=True)
        from infrarely.core.config import get_config

        cfg = get_config()
        assert cfg.get("memory_enabled") is True

    def test_configure_knowledge_threshold(self):
        import infrarely

        infrarely.configure(knowledge_threshold=0.9)
        from infrarely.core.config import get_config

        cfg = get_config()
        assert cfg.get("knowledge_threshold") == 0.9


class TestAutoDetection:
    """Test auto-detection of LLM providers."""

    def test_auto_detect_from_env(self):
        """When env vars are set, configure should auto-detect."""
        import infrarely

        # Just verify configure doesn't crash
        infrarely.configure()
        from infrarely.core.config import get_config

        cfg = get_config()
        assert cfg.get("llm_provider") is not None
