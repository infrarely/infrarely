from __future__ import annotations

from typing import Dict, List, Tuple

from .base import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for anthropic provider")
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            ) from exc

        super().__init__(api_key=api_key, model=model, base_url=base_url)
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = Anthropic(**kwargs)

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        system: str = "",
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> Tuple[str, int]:
        response = self.client.messages.create(
            model=self.model,
            system=system or None,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        chunks = [
            block.text for block in response.content if getattr(block, "text", None)
        ]
        text = "\n".join(chunks)
        input_tokens = getattr(response.usage, "input_tokens", 0) or 0
        output_tokens = getattr(response.usage, "output_tokens", 0) or 0
        return text, input_tokens + output_tokens
