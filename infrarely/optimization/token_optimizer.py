"""
adaptive/token_optimizer.py — Module 6: Token Optimizer
═══════════════════════════════════════════════════════════════════════════════
Minimises LLM usage by tracking and optimising token consumption.

Tracks:
  tokens_per_capability, tokens_per_tool, LLM call frequency

Strategies:
  • Cache LLM outputs for repeated queries
  • Reuse generated questions for same topic/difficulty
  • Prefer deterministic tools over generative ones
  • Enforce policy: LLM usage < 10% of total executions

Policy enforcement:
  If LLM call ratio exceeds soft limit → warn.
  If exceeds hard limit → block new generative executions.
"""

from __future__ import annotations
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from infrarely.observability import logger


@dataclass
class LLMCacheEntry:
    """Cached LLM output."""

    key: str
    output: str
    tokens_used: int
    created_at: float
    hits: int = 0
    ttl_secs: float = 3600.0  # 1 hour default

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_secs


class TokenOptimizer:
    """
    Token usage tracking and optimisation.
    Enforces LLM usage ratio policy.
    """

    LLM_RATIO_SOFT = 0.10  # warn if > 10% of executions use LLM
    LLM_RATIO_HARD = 0.20  # block generative if > 20%
    CACHE_MAX = 100

    def __init__(self):
        self._tool_tokens: Dict[str, int] = defaultdict(int)
        self._tool_calls: Dict[str, int] = defaultdict(int)
        self._cap_tokens: Dict[str, int] = defaultdict(int)
        self._total_executions = 0
        self._llm_executions = 0
        self._total_tokens = 0
        self._cache: Dict[str, LLMCacheEntry] = {}
        self._tokens_saved = 0

    # ── Record ────────────────────────────────────────────────────────────────
    def record_execution(
        self,
        tool: str,
        tokens: int,
        llm_used: bool,
        capability: str = "",
    ) -> None:
        self._total_executions += 1
        self._tool_calls[tool] += 1
        self._tool_tokens[tool] += tokens
        self._total_tokens += tokens
        if llm_used:
            self._llm_executions += 1
        if capability:
            self._cap_tokens[capability] += tokens

    # ── LLM output cache ─────────────────────────────────────────────────────
    def cache_llm_output(
        self,
        tool: str,
        params: Dict[str, Any],
        output: str,
        tokens_used: int,
        ttl_secs: float = 3600.0,
    ) -> None:
        """Cache an LLM output for potential reuse."""
        key = self._cache_key(tool, params)
        self._cache[key] = LLMCacheEntry(
            key=key,
            output=output,
            tokens_used=tokens_used,
            created_at=time.time(),
            ttl_secs=ttl_secs,
        )
        # Evict oldest if over limit
        if len(self._cache) > self.CACHE_MAX:
            oldest_key = min(self._cache, key=lambda k: self._cache[k].created_at)
            del self._cache[oldest_key]

    def get_cached_output(
        self, tool: str, params: Dict[str, Any]
    ) -> Optional[Tuple[str, int]]:
        """Return (output, tokens_saved) if cached and not expired."""
        key = self._cache_key(tool, params)
        entry = self._cache.get(key)
        if entry is None or entry.is_expired():
            return None
        entry.hits += 1
        self._tokens_saved += entry.tokens_used
        logger.debug(
            f"TokenOptimizer: cache HIT for '{tool}' — saved {entry.tokens_used} tokens"
        )
        return entry.output, entry.tokens_used

    def _cache_key(self, tool: str, params: Dict[str, Any]) -> str:
        # Stable hash from tool + sorted params
        param_str = str(sorted((k, str(v)) for k, v in params.items()))
        raw = f"{tool}:{param_str}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ── Policy enforcement ────────────────────────────────────────────────────
    @property
    def llm_ratio(self) -> float:
        """Fraction of executions that used LLM."""
        return (
            self._llm_executions / self._total_executions
            if self._total_executions
            else 0.0
        )

    def should_allow_llm(self) -> bool:
        """Check if LLM usage is within policy."""
        if self._total_executions < 5:
            return True  # insufficient data
        if self.llm_ratio > self.LLM_RATIO_HARD:
            logger.warn(
                f"TokenOptimizer: LLM ratio {self.llm_ratio:.1%} exceeds hard limit "
                f"({self.LLM_RATIO_HARD:.0%}) — blocking generative execution"
            )
            return False
        if self.llm_ratio > self.LLM_RATIO_SOFT:
            logger.warn(
                f"TokenOptimizer: LLM ratio {self.llm_ratio:.1%} exceeds soft limit "
                f"({self.LLM_RATIO_SOFT:.0%})"
            )
        return True

    # ── Query ─────────────────────────────────────────────────────────────────
    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_executions": self._total_executions,
            "llm_executions": self._llm_executions,
            "llm_ratio": f"{self.llm_ratio:.1%}",
            "total_tokens": self._total_tokens,
            "tokens_saved": self._tokens_saved,
            "cache_entries": len(self._cache),
            "cache_hits": sum(e.hits for e in self._cache.values()),
            "policy_status": (
                "BLOCKED"
                if self.llm_ratio > self.LLM_RATIO_HARD
                else "WARNING" if self.llm_ratio > self.LLM_RATIO_SOFT else "OK"
            ),
            "tokens_by_tool": {
                tool: {"calls": self._tool_calls[tool], "tokens": toks}
                for tool, toks in self._tool_tokens.items()
            },
            "tokens_by_capability": dict(self._cap_tokens),
        }
