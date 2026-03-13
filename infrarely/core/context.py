"""
aos/context.py — Context Window Management
═══════════════════════════════════════════════════════════════════════════════
Automatic management of what gets sent to the LLM when accumulated
context exceeds the model's token limit.

Research shows the "20,000-document cliff" — RAG systems fail
catastrophically when context grows beyond model limits. This module
prevents that.

Strategies:
- sliding_window: Keep most recent messages, drop oldest
- summarize: Summarize old context into compact form
- priority: Keep highest-relevance content

Usage::

    infrarely.configure(
        context_strategy="sliding_window",
        max_context_tokens=8000,
        context_overflow_action="summarize",
        preserve_system_prompt=True,
    )
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════


class ContextStrategy(Enum):
    """Strategy for managing context overflow."""

    SLIDING_WINDOW = "sliding_window"
    SUMMARIZE = "summarize"
    PRIORITY = "priority"


class OverflowAction(Enum):
    """What to do when context overflows."""

    TRUNCATE = "truncate"
    SUMMARIZE = "summarize"
    DROP_OLDEST = "drop_oldest"
    ERROR = "error"


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT MESSAGE — A single message in the context window
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ContextMessage:
    """A single message in the context window."""

    role: str = "user"  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    timestamp: float = field(default_factory=time.time)
    priority: float = 0.5  # 0.0 (lowest) to 1.0 (highest)
    token_count: int = 0  # auto-calculated if 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_system: bool = False  # protected from eviction
    is_summary: bool = False  # was this summarized from other messages?

    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = estimate_tokens(self.content)
        if self.role == "system":
            self.is_system = True


# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN ESTIMATION — Zero-dependency token counter
# ═══════════════════════════════════════════════════════════════════════════════


def estimate_tokens(text: str) -> int:
    """
    Estimate token count without tiktoken dependency.

    Uses the rule of thumb: ~4 characters per token for English text,
    with adjustments for code and special characters.
    """
    if not text:
        return 0

    # Count words and characters
    words = len(text.split())
    chars = len(text)

    # English text: ~0.75 tokens per word, or ~4 chars per token
    # Code: tends to have more tokens due to syntax
    has_code = bool(re.search(r"[{}\[\]();=<>]", text))

    if has_code:
        # Code is denser in tokens
        tokens = max(words, chars // 3)
    else:
        tokens = max(words * 3 // 4, chars // 4)

    # Minimum 1 token for non-empty text
    return max(1, tokens)


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT WINDOW MANAGER
# ═══════════════════════════════════════════════════════════════════════════════


class ContextWindowManager:
    """
    Manages the context window for LLM calls.

    Automatically handles:
    - Token counting / estimation
    - Context overflow with configurable strategies
    - System prompt preservation
    - Message priority for smart eviction
    - Context summarization

    Usage::

        manager = ContextWindowManager(
            max_tokens=8000,
            strategy=ContextStrategy.SLIDING_WINDOW,
            preserve_system_prompt=True,
        )

        manager.add_system("You are a helpful assistant.")
        manager.add_user("What is 2+2?")
        manager.add_assistant("2+2 = 4")
        manager.add_user("Now explain calculus in detail...")

        # Get messages that fit within token budget
        messages = manager.get_messages()
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        strategy: ContextStrategy = ContextStrategy.SLIDING_WINDOW,
        overflow_action: OverflowAction = OverflowAction.DROP_OLDEST,
        preserve_system_prompt: bool = True,
        reserve_tokens: int = 512,  # tokens reserved for response
        summarizer: Optional[Callable[[List[ContextMessage]], str]] = None,
    ):
        self._max_tokens = max_tokens
        self._strategy = strategy
        self._overflow_action = overflow_action
        self._preserve_system = preserve_system_prompt
        self._reserve_tokens = min(reserve_tokens, max_tokens // 2)
        self._summarizer = summarizer or self._default_summarizer
        self._messages: List[ContextMessage] = []
        self._total_tokens = 0
        self._summaries_created = 0

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def current_tokens(self) -> int:
        return self._total_tokens

    @property
    def available_tokens(self) -> int:
        return max(0, self._max_tokens - self._reserve_tokens - self._total_tokens)

    @property
    def message_count(self) -> int:
        return len(self._messages)

    @property
    def is_overflow(self) -> bool:
        return self._total_tokens > (self._max_tokens - self._reserve_tokens)

    def add_system(self, content: str, *, priority: float = 1.0) -> None:
        """Add a system message (protected from eviction)."""
        self._add(
            ContextMessage(
                role="system",
                content=content,
                priority=priority,
                is_system=True,
            )
        )

    def add_user(self, content: str, *, priority: float = 0.5) -> None:
        """Add a user message."""
        self._add(ContextMessage(role="user", content=content, priority=priority))

    def add_assistant(self, content: str, *, priority: float = 0.5) -> None:
        """Add an assistant message."""
        self._add(ContextMessage(role="assistant", content=content, priority=priority))

    def add_tool(
        self, content: str, *, tool_name: str = "", priority: float = 0.6
    ) -> None:
        """Add a tool result message."""
        self._add(
            ContextMessage(
                role="tool",
                content=content,
                priority=priority,
                metadata={"tool_name": tool_name},
            )
        )

    def add(self, role: str, content: str, *, priority: float = 0.5) -> None:
        """Add a generic message."""
        self._add(ContextMessage(role=role, content=content, priority=priority))

    def add_message(self, role: str, content: str, *, priority: float = 0.5) -> None:
        """Alias for add(). Add a message with role and content."""
        self.add(role, content, priority=priority)

    @property
    def total_tokens(self) -> int:
        """Alias for current_tokens."""
        return self._total_tokens

    def _add(self, message: ContextMessage) -> None:
        """Add a message and handle overflow if needed."""
        self._messages.append(message)
        self._total_tokens += message.token_count

        # Check for overflow
        budget = self._max_tokens - self._reserve_tokens
        if self._total_tokens > budget:
            self._handle_overflow(budget)

    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get messages formatted for LLM API calls.

        Returns messages that fit within the token budget,
        with system prompts preserved.
        """
        messages = []
        for msg in self._messages:
            messages.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                }
            )
        return messages

    def get_context_messages(self) -> List[ContextMessage]:
        """Get raw ContextMessage objects."""
        return list(self._messages)

    def _handle_overflow(self, budget: int) -> None:
        """Handle context overflow according to the configured strategy."""
        if self._strategy == ContextStrategy.SLIDING_WINDOW:
            self._sliding_window_eviction(budget)
        elif self._strategy == ContextStrategy.SUMMARIZE:
            self._summarize_eviction(budget)
        elif self._strategy == ContextStrategy.PRIORITY:
            self._priority_eviction(budget)

    def _sliding_window_eviction(self, budget: int) -> None:
        """Remove oldest non-system messages until within budget."""
        while self._total_tokens > budget and len(self._messages) > 1:
            # Find first non-system, non-recent message to remove
            removed = False
            for i, msg in enumerate(self._messages):
                if self._preserve_system and msg.is_system:
                    continue
                self._total_tokens -= msg.token_count
                self._messages.pop(i)
                removed = True
                break
            if not removed:
                break

    def _summarize_eviction(self, budget: int) -> None:
        """Summarize oldest messages instead of dropping them."""
        # Find messages to summarize (non-system, oldest half)
        non_system = [m for m in self._messages if not m.is_system]
        if len(non_system) <= 2:
            # Too few messages to summarize, fall back to sliding window
            self._sliding_window_eviction(budget)
            return

        # Summarize the oldest half
        half = len(non_system) // 2
        to_summarize = non_system[:half]

        summary_text = self._summarizer(to_summarize)
        summary_msg = ContextMessage(
            role="system",
            content=f"[Summary of earlier conversation]: {summary_text}",
            priority=0.7,
            is_summary=True,
        )

        # Remove summarized messages
        for msg in to_summarize:
            if msg in self._messages:
                self._total_tokens -= msg.token_count
                self._messages.remove(msg)

        # Insert summary after system messages
        insert_idx = 0
        for i, msg in enumerate(self._messages):
            if msg.is_system and not msg.is_summary:
                insert_idx = i + 1

        self._messages.insert(insert_idx, summary_msg)
        self._total_tokens += summary_msg.token_count
        self._summaries_created += 1

        # If still over budget, do sliding window
        if self._total_tokens > budget:
            self._sliding_window_eviction(budget)

    def _priority_eviction(self, budget: int) -> None:
        """Remove lowest-priority messages first."""
        while self._total_tokens > budget and len(self._messages) > 1:
            # Find lowest priority non-system message
            min_priority = float("inf")
            min_idx = -1
            for i, msg in enumerate(self._messages):
                if self._preserve_system and msg.is_system:
                    continue
                if msg.priority < min_priority:
                    min_priority = msg.priority
                    min_idx = i

            if min_idx == -1:
                break

            self._total_tokens -= self._messages[min_idx].token_count
            self._messages.pop(min_idx)

    @staticmethod
    def _default_summarizer(messages: List[ContextMessage]) -> str:
        """Default summarizer: compact key points from messages."""
        parts = []
        for msg in messages:
            content = msg.content[:200]
            if msg.role == "user":
                parts.append(f"User asked: {content}")
            elif msg.role == "assistant":
                parts.append(f"Response: {content}")
            elif msg.role == "tool":
                tool_name = msg.metadata.get("tool_name", "tool")
                parts.append(f"{tool_name} returned: {content}")

        return " | ".join(parts) if parts else "Previous conversation context."

    def clear(self) -> None:
        """Clear all messages."""
        self._messages.clear()
        self._total_tokens = 0

    def reset(self) -> None:
        """Reset to initial state, preserving only system messages."""
        system_msgs = [m for m in self._messages if m.is_system and not m.is_summary]
        self._messages = system_msgs
        self._total_tokens = sum(m.token_count for m in system_msgs)

    def stats(self) -> Dict[str, Any]:
        """Get context window statistics."""
        return {
            "total_messages": len(self._messages),
            "total_tokens": self._total_tokens,
            "max_tokens": self._max_tokens,
            "available_tokens": self.available_tokens,
            "utilization": (
                self._total_tokens / self._max_tokens if self._max_tokens > 0 else 0
            ),
            "system_messages": sum(1 for m in self._messages if m.is_system),
            "summaries_created": self._summaries_created,
            "strategy": self._strategy.value,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT CONFIG — Convenience for infrarely.configure()
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ContextConfig:
    """Context window configuration."""

    strategy: str = "sliding_window"
    max_context_tokens: int = 8000
    context_overflow_action: str = "drop_oldest"
    preserve_system_prompt: bool = True
    reserve_response_tokens: int = 512

    def to_manager(self) -> ContextWindowManager:
        """Create a ContextWindowManager from this config."""
        strategy_map = {
            "sliding_window": ContextStrategy.SLIDING_WINDOW,
            "summarize": ContextStrategy.SUMMARIZE,
            "priority": ContextStrategy.PRIORITY,
        }
        overflow_map = {
            "truncate": OverflowAction.TRUNCATE,
            "summarize": OverflowAction.SUMMARIZE,
            "drop_oldest": OverflowAction.DROP_OLDEST,
            "error": OverflowAction.ERROR,
        }
        return ContextWindowManager(
            max_tokens=self.max_context_tokens,
            strategy=strategy_map.get(self.strategy, ContextStrategy.SLIDING_WINDOW),
            overflow_action=overflow_map.get(
                self.context_overflow_action, OverflowAction.DROP_OLDEST
            ),
            preserve_system_prompt=self.preserve_system_prompt,
            reserve_tokens=self.reserve_response_tokens,
        )
