from __future__ import annotations

from typing import Dict, List, Tuple

from .base import BaseLLMProvider


class GroqProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is required for groq provider")
        try:
            from groq import Groq
        except ImportError as exc:
            raise ImportError(
                "groq package not installed. Run: pip install groq"
            ) from exc

        super().__init__(api_key=api_key, model=model, base_url=base_url)
        self.client = Groq(api_key=api_key)

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        system: str = "",
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> Tuple[str, int]:
        full_messages: List[Dict[str, str]] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        chat = self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        text = chat.choices[0].message.content or ""
        prompt_tokens = getattr(chat.usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(chat.usage, "completion_tokens", 0) or 0
        return text, prompt_tokens + completion_tokens
