"""
agent/llm_client.py
═══════════════════════════════════════════════════════════════════════════════
LLM adapter layer.

Runtime code calls this module only. Provider SDK details live in
``infrarely.llm.*`` and are loaded through the registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from infrarely.core.config import get_config
from infrarely.llm.registry import load_provider
from infrarely.observability import logger


_token_governor = None


def register_token_governor(budget):
    """Register a TokenBudget instance to receive all LLM call records."""
    global _token_governor
    _token_governor = budget
    logger.info("Token governor registered for LLM client")


def _record_governance(tokens: int, reason: str):
    if _token_governor is not None:
        _token_governor.record(tokens, reason)


@dataclass
class LLMStatus:
    ok: bool
    backend: str
    model: str
    message: str = ""
    fix_hint: str = ""

    def __str__(self) -> str:
        if self.ok:
            return f"✅ LLM ready  [{self.backend}] {self.model}"
        return f"⚠️  LLM unavailable  [{self.backend}]  {self.message}\n   → {self.fix_hint}"


def _resolve_provider():
    cfg = get_config()
    provider = cfg.get("llm")

    if provider is not None:
        return provider

    provider_name = cfg.get("llm_provider", "openai")
    provider = load_provider(
        provider_name,
        cfg.get("api_key", ""),
        cfg.get("llm_model"),
        base_url=cfg.get("llm_base_url"),
    )
    cfg.set("llm", provider)
    cfg.set("llm_init_error", "")
    return provider


def check_llm_health() -> LLMStatus:
    """Validate provider initialization. Never raises."""
    cfg = get_config()
    backend = cfg.get("llm_provider", "openai")
    model = cfg.get("llm_model")

    try:
        _resolve_provider()
        return LLMStatus(
            ok=True, backend=backend, model=model, message="Provider initialized"
        )
    except Exception as exc:
        return LLMStatus(
            ok=False,
            backend=backend,
            model=model,
            message=str(exc),
            fix_hint=_build_fix_hint(backend, str(exc)),
        )


def llm_call(
    messages: List[Dict],
    system: str = "",
    max_tokens: Optional[int] = None,
    reason: str = "unknown",
) -> Tuple[str, int]:
    """
    Single inference call. Returns (response_text, total_tokens).
    On failure returns a clean, user-facing message and 0 tokens.
    """
    cfg = get_config()
    backend = cfg.get("llm_provider", "openai")
    model = cfg.get("llm_model")

    if max_tokens is None:
        max_tokens = cfg.get("llm_max_tokens", 512)

    try:
        provider = _resolve_provider()
        text, tokens = provider.chat(
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=cfg.get("llm_temperature", 0.3),
        )
        _record_governance(tokens, reason)
        logger.llm_log(0, tokens, reason=reason, backend=backend, model=model)
        return text, tokens
    except Exception as exc:
        clean_msg = _classify_error(exc, backend)
        logger.error(f"LLM call failed [{backend}/{reason}]: {exc}")
        return clean_msg, 0


def _build_fix_hint(backend: str, message: str) -> str:
    lower = message.lower()
    if "not installed" in lower:
        installs = {
            "groq": "pip install groq",
            "openai": "pip install openai",
            "anthropic": "pip install anthropic",
            "gemini": "pip install google-generativeai",
        }
        return installs.get(backend, "Install the provider SDK package")

    key_hints = {
        "groq": "Set GROQ_API_KEY.",
        "openai": "Set OPENAI_API_KEY.",
        "anthropic": "Set ANTHROPIC_API_KEY.",
        "gemini": "Set GEMINI_API_KEY.",
    }
    return key_hints.get(backend, "Check provider configuration and API key.")


def _classify_error(exc: Exception, backend: str) -> str:
    msg = str(exc).lower()

    if "unknown llm provider" in msg:
        return (
            f"Unknown LLM provider '{backend}'. "
            "Use one of: openai, anthropic, groq, gemini."
        )

    if "api key" in msg or "unauthorized" in msg or "401" in msg or "403" in msg:
        return _build_fix_hint(backend, str(exc))

    if "rate limit" in msg or "429" in msg:
        return f"Rate limit hit on {backend}. Wait a moment and try again."

    if "timeout" in msg or "timed out" in msg:
        return f"LLM request timed out ({backend}). Please try again in a moment."

    return (
        f"LLM unavailable ({backend}). "
        "Practice questions and open-ended answers need a configured LLM provider."
    )
