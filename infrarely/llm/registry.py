from __future__ import annotations

from infrarely.llm.anthropic_provider import AnthropicProvider
from infrarely.llm.base import BaseLLMProvider
from infrarely.llm.gemini_provider import GeminiProvider
from infrarely.llm.groq_provider import GroqProvider
from infrarely.llm.openai_provider import OpenAIProvider


_PROVIDERS = {
    "openai": OpenAIProvider,
    "groq": GroqProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "google": GeminiProvider,
}


def load_provider(
    name: str,
    api_key: str | None,
    model: str,
    *,
    base_url: str | None = None,
) -> BaseLLMProvider:
    provider_name = (name or "").lower().strip()
    provider_cls = _PROVIDERS.get(provider_name)
    if provider_cls is None:
        raise ValueError(f"Unknown LLM provider: {name}")
    return provider_cls(api_key=api_key or "", model=model, base_url=base_url)
