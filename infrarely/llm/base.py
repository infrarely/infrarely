from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Tuple


class BaseLLMProvider(ABC):
    """Common provider interface for chat-based LLM backends."""

    def __init__(self, api_key: str | None, model: str, base_url: str | None = None):
        self.api_key = api_key or ""
        self.model = model
        self.base_url = base_url

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        text, _ = self.chat(
            messages=[{"role": "user", "content": prompt}],
            system="",
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return text

    def stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> Iterable[str]:
        yield self.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        system: str = "",
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> Tuple[str, int]:
        """Return (text, total_tokens)."""
