"""
aos/config.py — SDK Configuration System
═══════════════════════════════════════════════════════════════════════════════
Philosophy 4: Zero Surprise Defaults.
  Every default is the safest, most correct choice.
  Beginners call: infrarely.configure(llm_provider="openai", api_key="...")
  Experts override: anything they want.

The _SDKConfig singleton stores all settings. The configure() function
is the only public way to change them. Per-agent overrides are merged
at agent creation time.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT VALUES — safest, most correct choice for every setting
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULTS = {
    # ── LLM ───────────────────────────────────────────────────────────────────
    "llm_provider": "openai",  # "openai" | "anthropic" | "groq" | "ollama" | "gemini" | "local"
    "llm_model": None,  # auto-selected per provider if None
    "api_key": "",
    "llm_temperature": 0.3,
    "llm_max_tokens": 512,
    "llm_base_url": None,  # for ollama / custom endpoints
    # ── Knowledge Layer ───────────────────────────────────────────────────────
    "knowledge_backend": "memory",  # "memory" | "chromadb" | "pinecone"
    "knowledge_threshold": 0.85,  # confidence to bypass LLM
    "knowledge_refresh_hours": 24,
    # ── State Machine ─────────────────────────────────────────────────────────
    "state_backend": "sqlite",  # "sqlite" | "redis" | "memory"
    "state_db_path": "./aos_state.db",
    # ── Planning Engine ───────────────────────────────────────────────────────
    "max_replan_attempts": 3,
    "token_budget": 10_000,
    # ── Execution ─────────────────────────────────────────────────────────────
    "max_retries": 2,
    "default_timeout": 10_000,  # ms
    "max_execution_depth": 8,
    # ── Memory ────────────────────────────────────────────────────────────────
    "memory_enabled": True,
    "memory_backend": "sqlite",  # "sqlite" | "memory"
    "memory_db_path": "./aos_memory.db",
    "working_memory_window": 20,
    # ── Observability ─────────────────────────────────────────────────────────
    "log_level": "INFO",  # "DEBUG" | "INFO" | "WARNING" | "ERROR"
    "log_dir": "./logs",  # Path for log files (set None to disable file logs)
    "log_file_enabled": True,  # Enable/disable file-based logging
    "trace_storage": "sqlite",  # "sqlite" | "memory" | "none"
    "trace_db_path": "./aos_traces.db",
    # ── Multi-agent ───────────────────────────────────────────────────────────
    "max_agents": 50,
    "message_bus_capacity": 1000,
    "agent_idle_timeout_ms": 300_000,
    # ── Cache ─────────────────────────────────────────────────────────────────
    "cache_enabled": True,
    "cache_ttl": 3600,  # seconds
    # ── Safety ────────────────────────────────────────────────────────────────
    "sandbox_enabled": True,
    "permission_policy_enabled": True,
    # ── Security ──────────────────────────────────────────────────────────────
    "security": None,  # SecurityPolicy object, None = disabled
    # ── Context Window Management ─────────────────────────────────────────────
    "context_strategy": "sliding_window",  # "sliding_window" | "summarize" | "priority"
    "max_context_tokens": 8000,
    "context_overflow_action": "drop_oldest",  # "truncate" | "summarize" | "drop_oldest" | "error"
    "preserve_system_prompt": True,
    # ── HITL ──────────────────────────────────────────────────────────────────
    "hitl_default_timeout": 3600,  # seconds, default approval timeout
    # ── Validation ────────────────────────────────────────────────────────────
    "tool_validation_enabled": True,
    "tool_validation_coerce": True,  # attempt safe type coercion
    # ── Versioning ────────────────────────────────────────────────────────────
    "versions_dir": "./.aos_versions",  # ── Token Tracking ────────────────────────────────────────────────────
    "token_tracking_enabled": True,
    # ── Sandbox ───────────────────────────────────────────────────────────
    "sandbox_max_memory_mb": 512,
    "sandbox_max_execution_time": 60,
    "sandbox_max_tool_calls": 50,
    "sandbox_network_access": True,
    # ── Horizontal Scaling ────────────────────────────────────────────────
    "scaling_backend": "memory",  # "memory" | "sqlite" | "redis"
    "scaling_redis_url": "",  # redis://localhost:6379
    # ── Compliance Audit ──────────────────────────────────────────────────
    "compliance_enabled": True,
    "compliance_db_path": "",  # empty = in-memory only
    # ── Multi-Tenancy ─────────────────────────────────────────────────────
    "multitenancy_enabled": False,
    # ── Events & Webhooks ─────────────────────────────────────────────────
    "events_enabled": True,
    "events_max_history": 1000,
    # ── Input Sanitization (always-on) ────────────────────────────────────
    "sanitization_enabled": True,
    "sanitization_max_input_length": 50_000,
    "sanitization_strip_null_bytes": True,
    "sanitization_strip_control_chars": True,
    # ── Memory Namespace Isolation ────────────────────────────────────────
    "memory_namespace_isolation": True,
    # ── Key Rotation ──────────────────────────────────────────────────────
    "key_rotation_enabled": True,
    # ── Tool Execution Sandbox ────────────────────────────────────────────
    "tool_sandbox_enabled": True,
    "tool_sandbox_max_execution_time": 30.0,
    "tool_sandbox_network_allowed": True,
    # ── Self-Healing (Evolution Engine) ───────────────────────────────────
    "self_healing_enabled": True,
    "self_healing_max_window": 2000,
    "self_healing_default_cooldown": 60.0,
    "self_healing_default_confidence_threshold": 0.6,
    # ── Agent Collaboration Protocol (ACP) ────────────────────────────────
    "acp_enabled": True,
    "acp_default_timeout_ms": 30_000,
    "acp_server_host": "0.0.0.0",
    "acp_server_port": 9000,
    "acp_auth_token": "",
    # ── Agent Marketplace ──────────────────────────────────────────────────
    "marketplace_enabled": True,
    "marketplace_install_dir": "./.aos_packages",
    "marketplace_registry_url": "",  # remote registry URL (empty = local only)
    "marketplace_auto_update": False,
    # ── Natural Language Agent Configuration ───────────────────────────────
    "nlconfig_enabled": True,
    "nlconfig_confidence_threshold": 0.3,  # min confidence to auto-apply
    "nlconfig_auto_tools": True,  # auto-suggest tools from description
    "nlconfig_auto_knowledge": True,  # auto-seed knowledge domains
    # ── Agent Performance Benchmarking ─────────────────────────────────────
    "benchmark_enabled": True,
    "benchmark_default_suite": "standard-suite-v1",
    "benchmark_verbose": False,
    "benchmark_use_color": True,
    "benchmark_output_format": "table",  # "table" | "json" | "summary"
}

# ── Auto-selected models per provider ─────────────────────────────────────────
_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-20250514",
    "groq": "llama-3.1-8b-instant",
    "ollama": "llama3.2",
    "gemini": "gemini-1.5-flash",
    "local": "llama3.2",
}


# ═══════════════════════════════════════════════════════════════════════════════
# SDK CONFIG SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════


class _SDKConfig:
    """
    Thread-safe singleton that holds ALL SDK configuration.
    Only mutated via configure(). Read via get() or attribute access.
    """

    _instance: Optional["_SDKConfig"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "_SDKConfig":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._values = dict(_DEFAULTS)
                cls._instance._configured = False
            return cls._instance

    @property
    def configured(self) -> bool:
        return self._configured

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._values[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def get_all(self) -> Dict[str, Any]:
        return dict(self._values)

    def mark_configured(self) -> None:
        self._configured = True

    def merge(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Return a merged config dict (base + overrides). Does not mutate."""
        merged = dict(self._values)
        merged.update(overrides)
        return merged

    def reset(self) -> None:
        """Reset to defaults (used in tests)."""
        with self._lock:
            self._values = dict(_DEFAULTS)
            self._configured = False

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        vals = super().__getattribute__("_values")
        if name in vals:
            return vals[name]
        raise AttributeError(f"No SDK config key: {name}")


# ── Module-level singleton accessor ──────────────────────────────────────────
_config = _SDKConfig()


def get_config() -> _SDKConfig:
    """Return the global SDK config singleton."""
    return _config


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC CONFIGURE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════


def configure(**kwargs) -> None:
    """
    Configure the InfraRely SDK. Call once at startup.

    Minimal (beginner):
        infrarely.configure(llm_provider="openai", api_key="sk-...")

    Full (expert):
        infrarely.configure(
            llm_provider="openai",
            llm_model="gpt-4o-mini",
            api_key="sk-...",
            knowledge_backend="chromadb",
            knowledge_threshold=0.90,
            state_backend="sqlite",
            token_budget=20000,
            log_level="DEBUG",
            max_agents=100,
        )
    """
    cfg = get_config()

    # ── Aliases — friendly names map to canonical keys ─────────────────────
    _ALIASES = {
        "provider": "llm_provider",
        "model": "llm_model",
    }

    resolved = {}
    for key, value in kwargs.items():
        canonical = _ALIASES.get(key, key)
        if canonical not in _DEFAULTS and canonical not in ("api_key",):
            # Warn but don't crash — forward compatibility
            import warnings

            warnings.warn(f"Unknown InfraRely config key: '{key}'. Ignoring.", stacklevel=2)
            continue
        resolved[canonical] = value

    for key, value in resolved.items():
        cfg.set(key, value)

    # ── Auto-resolve model if not explicitly set ──────────────────────────────
    if "llm_model" not in resolved and "llm_model" not in kwargs:
        provider = cfg.get("llm_provider", "openai")
        cfg.set("llm_model", _DEFAULT_MODELS.get(provider, "gpt-4o-mini"))

    # ── Resolve API key from env if not provided ──────────────────────────────
    if not cfg.get("api_key"):
        provider = cfg.get("llm_provider")
        env_keys = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "groq": "GROQ_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }
        env_name = env_keys.get(provider, "")
        if env_name:
            cfg.set("api_key", os.getenv(env_name, ""))

    cfg.mark_configured()

    # ── Enable file logging ──────────────────────────────────────────────────
    try:
        from infrarely.observability.observability import get_logger

        logger = get_logger()
        logger.set_level(cfg.get("log_level", "INFO"))
        if cfg.get("log_file_enabled", True):
            log_dir = cfg.get("log_dir", "./logs")
            if log_dir:
                logger.enable_file_logging(log_dir)
    except Exception:
        pass  # Don't crash on logging setup


def _ensure_configured() -> _SDKConfig:
    """
    Internal: ensure configure() was called. If not, auto-configure
    with env-based defaults (Philosophy 4: zero surprise defaults).
    """
    cfg = get_config()
    if not cfg.configured:
        # Auto-configure from environment
        provider = os.getenv("INFRARELY_LLM_PROVIDER", "openai")
        configure(llm_provider=provider)
    return cfg
