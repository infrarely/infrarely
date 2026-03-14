from __future__ import annotations

from typing import Dict, List, Tuple

from .base import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required for gemini provider")
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "google-generativeai package not installed. Run: pip install google-generativeai"
            ) from exc

        super().__init__(api_key=api_key, model=model, base_url=base_url)
        self._genai = genai
        self._genai.configure(api_key=api_key)

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        system: str = "",
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> Tuple[str, int]:
        model = self._genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system or None,
        )

        history = []
        prompt = ""
        for idx, message in enumerate(messages):
            role = "user" if message.get("role") == "user" else "model"
            content = message.get("content", "")
            is_last = idx == len(messages) - 1
            if is_last and role == "user":
                prompt = content
            else:
                history.append({"role": role, "parts": [content]})

        if not prompt:
            prompt = messages[-1].get("content", "") if messages else ""

        chat = model.start_chat(history=history)
        response = chat.send_message(
            prompt,
            generation_config=self._genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )

        text = getattr(response, "text", "") or ""
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
        return text, prompt_tokens + completion_tokens
