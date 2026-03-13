"""
memory/long_term.py
Compressed conversation summaries stored as JSON.
Default compression is rule-based (zero tokens).
LLM compression is opt-in and uses the configured backend (Ollama/Groq/Gemini).
"""

from __future__ import annotations
import json
import os
from typing import List, Optional
from datetime import datetime

import infrarely.core.app_config as config
from infrarely.agent.state import Message
from infrarely.observability import logger
from infrarely.observability.metrics import collector


def _load_summaries() -> dict:
    if os.path.exists(config.SUMMARY_FILE):
        with open(config.SUMMARY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_summaries(data: dict):
    os.makedirs(os.path.dirname(config.SUMMARY_FILE), exist_ok=True)
    with open(config.SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


class LongTermMemory:
    def __init__(self):
        self._data: dict = _load_summaries()
        logger.memory_log("init", "long_term", students=len(self._data))

    def compress_and_store(
        self, student_id: str, messages: List[Message], use_llm: bool = False
    ) -> str:
        if use_llm:
            summary = self._llm_compress(messages)
        else:
            summary = self._rule_compress(messages)

        collector.record_memory_write()
        if student_id not in self._data:
            self._data[student_id] = []
        self._data[student_id].append(
            {
                "ts": datetime.utcnow().isoformat(),
                "summary": summary,
                "turns": len(messages),
            }
        )
        self._data[student_id] = self._data[student_id][-20:]
        _save_summaries(self._data)
        logger.memory_log(
            "compress", "long_term", student=student_id, chars=len(summary)
        )
        return summary

    def _rule_compress(self, messages: List[Message]) -> str:
        intents = [
            m.content[:80].replace("\n", " ")
            for m in messages
            if m.role == "user" and len(m.content) > 5
        ]
        return "Previous topics: " + "; ".join(intents[:5])

    def _llm_compress(self, messages: List[Message]) -> str:
        """Uses the configured LLM backend (Ollama/Groq/Gemini)."""
        try:
            from infrarely.agent.llm_client import llm_call

            convo = "\n".join(f"{m.role}: {m.content[:200]}" for m in messages)
            prompt = f"Summarise this conversation in ≤3 sentences:\n{convo}"
            text, _ = llm_call(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.LONG_TERM_SUMMARY_MAX_TOKENS,
                reason="long_term_compress",
            )
            return text
        except Exception as e:
            logger.error(f"LLM compression failed: {e}, falling back to rule-based")
            return self._rule_compress(messages)

    def get_summary(self, student_id: str, n: int = 3) -> str:
        collector.record_memory_read()
        recent = self._data.get(student_id, [])[-n:]
        return " | ".join(s["summary"] for s in recent) if recent else ""

    def get_all_summaries(self, student_id: str) -> List[dict]:
        collector.record_memory_read()
        return self._data.get(student_id, [])
