"""
memory/working.py
Sliding-window conversation buffer.
Only the last N turns are retained in-process; older turns are
compressed into long-term memory to keep context small.
"""

from __future__ import annotations
from collections import deque
from typing import List, Optional

import infrarely.core.app_config as config
from infrarely.agent.state import Message
from infrarely.observability import logger
from infrarely.observability.metrics import collector


class WorkingMemory:
    """
    Holds recent conversation turns as a fixed-size deque.
    When the window overflows the oldest message is evicted
    (caller is responsible for archiving it to long-term memory).
    """

    def __init__(self, max_turns: int = config.WORKING_MEMORY_MAX_TURNS):
        self._max_turns = max_turns
        self._buffer: deque[Message] = deque(maxlen=max_turns)
        self._evicted: List[Message] = []  # captured for compression

    # ── write ────────────────────────────────────────────────────────────
    def add(self, message: Message) -> Optional[Message]:
        """
        Add a message. Returns the evicted message if the window overflowed,
        otherwise returns None.
        """
        evicted = None
        if len(self._buffer) == self._max_turns:
            evicted = self._buffer[0]  # leftmost = oldest
        self._buffer.append(message)
        collector.record_memory_write()
        logger.memory_log(
            "write", "working", role=message.role, chars=len(message.content)
        )
        return evicted

    # ── read ─────────────────────────────────────────────────────────────
    def get_recent(self, n: Optional[int] = None) -> List[Message]:
        collector.record_memory_read()
        msgs = list(self._buffer)
        return msgs[-n:] if n else msgs

    def get_for_llm(self) -> List[dict]:
        """
        Returns the buffer formatted as Anthropic API messages.
        Only user/assistant roles — tool messages are stripped or summarised.
        Token budget: each message is counted as ~4 chars per token.
        """
        collector.record_memory_read()
        result = []
        for m in self._buffer:
            if m.role in ("user", "assistant"):
                result.append({"role": m.role, "content": m.content})
        return result

    # ── state ────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._buffer)

    def clear(self):
        self._buffer.clear()
        logger.memory_log("clear", "working")

    def turn_count(self) -> int:
        return len(self._buffer)
