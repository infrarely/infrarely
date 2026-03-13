"""
observability/token_budget.py
═══════════════════════════════════════════════════════════════════════════════
Per-agent, per-session token accounting with hard and soft limits.

Responsibilities
─────────────────
  • Track token usage per agent instance
  • Enforce soft (warn) and hard (block) daily/session budgets
  • Provide a clean API the core uses before every LLM call
  • Support multi-agent deployments (each AgentCore gets its own budget)

Usage
─────
    budget = TokenBudget(agent_id="student_1", session_hard_limit=3000)

    # Before LLM call:
    if budget.can_spend(estimated_tokens=200):
        text, tokens = llm_call(...)
        budget.record(tokens, reason="practice_questions")
    else:
        return budget.over_limit_message()
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import infrarely.core.app_config as config
from infrarely.observability import logger


@dataclass
class SpendRecord:
    reason: str
    tokens: int
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class TokenBudget:
    """
    Token budget for one agent instance.
    Thread-safety note: single-threaded for now; add a Lock for async use.
    """

    def __init__(
        self,
        agent_id: str,
        session_soft_limit: int = config.TOKEN_WARN_THRESHOLD * 10,
        session_hard_limit: int = config.TOKEN_DAILY_BUDGET,
        call_hard_limit: int = config.LLM_MAX_TOKENS * 2,
    ):
        self.agent_id = agent_id
        self.session_soft_limit = session_soft_limit
        self.session_hard_limit = session_hard_limit
        self.call_hard_limit = call_hard_limit

        self._session_total: int = 0
        self._call_count: int = 0
        self._history: List[SpendRecord] = []
        self._hard_blocked: int = 0  # times hard limit prevented calls

    # ── budget pressure (Gap 9) ───────────────────────────────────────────────
    @property
    def pressure(self) -> float:
        """
        0.0 = no pressure, 1.0 = at hard limit.
        Used to dynamically reduce max_tokens on generative nodes.
        """
        if self.session_hard_limit <= 0:
            return 0.0
        return min(self._session_total / self.session_hard_limit, 1.0)

    def effective_max_tokens(self, base_max: int = None) -> int:
        """
        Return reduced max_tokens based on budget pressure (Gap 9).
        When >80% budget used, reduce max_tokens proportionally.
        At 90%+ → 50% of base. At 95%+ → 25% of base.
        """
        base = base_max or config.LLM_MAX_TOKENS
        p = self.pressure
        if p < 0.8:
            return base
        if p < 0.9:
            # Linear reduction from 100% to 50% over 0.8–0.9 range
            factor = 1.0 - 0.5 * ((p - 0.8) / 0.1)
            return max(int(base * factor), 64)
        if p < 0.95:
            return max(int(base * 0.5), 64)
        return max(int(base * 0.25), 64)

    # ── pre-call gate ─────────────────────────────────────────────────────────
    def can_spend(self, estimated_tokens: int = 0) -> bool:
        """
        Returns True if an LLM call is allowed.
        Logs a warning on soft-limit breach; returns False on hard limit.
        """
        projected = self._session_total + estimated_tokens
        if projected >= self.session_hard_limit:
            self._hard_blocked += 1
            logger.warn(
                f"TokenBudget HARD LIMIT reached for agent '{self.agent_id}'",
                used=self._session_total,
                limit=self.session_hard_limit,
            )
            return False
        if projected >= self.session_soft_limit:
            logger.warn(
                f"TokenBudget soft limit approaching for '{self.agent_id}'",
                used=self._session_total,
                soft_limit=self.session_soft_limit,
            )
        return True

    # ── post-call recording ───────────────────────────────────────────────────
    def record(self, tokens: int, reason: str = "unknown") -> None:
        self._session_total += tokens
        self._call_count += 1
        self._history.append(SpendRecord(reason=reason, tokens=tokens))

        if tokens > config.TOKEN_WARN_THRESHOLD:
            logger.warn(
                f"Single LLM call used {tokens} tokens (threshold: {config.TOKEN_WARN_THRESHOLD})",
                reason=reason,
                agent=self.agent_id,
            )

    # ── introspection ─────────────────────────────────────────────────────────
    def over_limit_message(self) -> str:
        return (
            f"Token budget exhausted for this session "
            f"({self._session_total} / {self.session_hard_limit} tokens used). "
            "Please start a new session with /new."
        )

    def snapshot(self) -> Dict:
        by_reason: Dict[str, int] = {}
        for r in self._history:
            by_reason[r.reason] = by_reason.get(r.reason, 0) + r.tokens
        return {
            "agent_id": self.agent_id,
            "session_total": self._session_total,
            "call_count": self._call_count,
            "soft_limit": self.session_soft_limit,
            "hard_limit": self.session_hard_limit,
            "pct_used": f"{100 * self._session_total / self.session_hard_limit:.1f}%",
            "pressure": round(self.pressure, 3),
            "effective_max_tokens": self.effective_max_tokens(),
            "hard_blocked": self._hard_blocked,
            "by_reason": by_reason,
        }

    @property
    def session_total(self) -> int:
        return self._session_total

    @property
    def call_count(self) -> int:
        return self._call_count

    def reset(self) -> None:
        self._session_total = 0
        self._call_count = 0
        self._history.clear()
        logger.info(f"TokenBudget reset for agent '{self.agent_id}'")
