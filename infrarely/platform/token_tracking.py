"""
aos/token_tracking.py — Token Cost Tracking
═══════════════════════════════════════════════════════════════════════════════
SCALE GAP 1: Track exactly how many tokens each agent uses and how much it costs.

LLM cost is the #1 reason teams pull agents from production. Without visibility
into which agent, which task, which tool call consumed how many tokens, you
cannot optimize or budget.

Usage::

    result = agent.run("some task")
    result.tokens_used       # 423
    result.estimated_cost    # $0.000846

    infrarely.metrics.total_tokens_today()        # 1,247,832
    infrarely.metrics.estimated_cost_today()      # $2.49
    infrarely.metrics.cost_by_agent("my-agent")   # $0.73 today
    infrarely.metrics.most_expensive_tasks(top=5)

Architecture:
    TokenTracker — singleton that accumulates token/cost data per agent, per task.
    TokenUsage   — dataclass for a single usage record.
    PricingTable — per-model pricing (input/output per 1M tokens).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# PRICING TABLE — per-model USD pricing (per 1M tokens)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ModelPricing:
    """Cost per 1M tokens for a specific model."""

    input_per_1m: float = 0.0
    output_per_1m: float = 0.0
    cached_input_per_1m: float = 0.0

    def cost(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> float:
        """Calculate cost in USD.

        Accepts both naming conventions:
            cost(prompt_tokens=1000, completion_tokens=500)
            cost(input_tokens=1000, output_tokens=500)
        """
        pt = prompt_tokens or input_tokens
        ct = completion_tokens or output_tokens
        input_cost = (pt / 1_000_000) * self.input_per_1m
        output_cost = (ct / 1_000_000) * self.output_per_1m
        return input_cost + output_cost


# Default pricing table (March 2026 estimates)
_PRICING: Dict[str, ModelPricing] = {
    # OpenAI
    "gpt-4o": ModelPricing(input_per_1m=2.50, output_per_1m=10.00),
    "gpt-4o-mini": ModelPricing(input_per_1m=0.15, output_per_1m=0.60),
    "gpt-4-turbo": ModelPricing(input_per_1m=10.00, output_per_1m=30.00),
    "gpt-4": ModelPricing(input_per_1m=30.00, output_per_1m=60.00),
    "gpt-3.5-turbo": ModelPricing(input_per_1m=0.50, output_per_1m=1.50),
    "o1": ModelPricing(input_per_1m=15.00, output_per_1m=60.00),
    "o1-mini": ModelPricing(input_per_1m=3.00, output_per_1m=12.00),
    # Anthropic
    "claude-opus-4-20250514": ModelPricing(input_per_1m=15.00, output_per_1m=75.00),
    "claude-sonnet-4-20250514": ModelPricing(input_per_1m=3.00, output_per_1m=15.00),
    "claude-3-haiku-20240307": ModelPricing(input_per_1m=0.25, output_per_1m=1.25),
    # Groq (typically free tier / very cheap)
    "llama-3.1-8b-instant": ModelPricing(input_per_1m=0.05, output_per_1m=0.08),
    "llama-3.1-70b-versatile": ModelPricing(input_per_1m=0.59, output_per_1m=0.79),
    # Gemini
    "gemini-1.5-flash": ModelPricing(input_per_1m=0.075, output_per_1m=0.30),
    "gemini-1.5-pro": ModelPricing(input_per_1m=1.25, output_per_1m=5.00),
    # Local / Ollama — zero cost
    "llama3.2": ModelPricing(input_per_1m=0.0, output_per_1m=0.0),
    "llama3": ModelPricing(input_per_1m=0.0, output_per_1m=0.0),
}


def get_pricing(model: str) -> ModelPricing:
    """Look up pricing for a model. Falls back to gpt-4o-mini pricing."""
    # Exact match
    if model in _PRICING:
        return _PRICING[model]
    # Partial match (e.g. "gpt-4o-2024-05-13" → "gpt-4o")
    for key in _PRICING:
        if model.startswith(key):
            return _PRICING[key]
    # Default: cheapest OpenAI
    return _PRICING.get(
        "gpt-4o-mini", ModelPricing(input_per_1m=0.15, output_per_1m=0.60)
    )


def set_pricing(
    model: str,
    input_per_1m: float = 0.0,
    output_per_1m: float = 0.0,
    *,
    pricing: "ModelPricing | None" = None,
) -> None:
    """Register custom pricing for a model.

    Accepts either explicit floats or a ModelPricing object:
        set_pricing("my-model", 1.0, 2.0)
        set_pricing("my-model", pricing=ModelPricing(input_per_1m=1.0, output_per_1m=2.0))
        set_pricing("my-model", ModelPricing(input_per_1m=1.0, output_per_1m=2.0))  # positional
    """
    if isinstance(input_per_1m, ModelPricing):
        # Called as set_pricing("model", ModelPricing(...))
        _PRICING[model] = input_per_1m
    elif pricing is not None:
        _PRICING[model] = pricing
    else:
        _PRICING[model] = ModelPricing(
            input_per_1m=input_per_1m, output_per_1m=output_per_1m
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN USAGE RECORD
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TokenUsage:
    """Record of token usage for a single operation."""

    agent_name: str = ""
    task_goal: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    timestamp: Any = field(default=None)
    trace_id: str = ""
    operation: str = ""  # "run" | "delegate" | "tool_call"
    # Aliases — accept both naming conventions
    input_tokens: int = 0
    output_tokens: int = 0
    task: str = ""

    def __post_init__(self):
        # Resolve aliases: input_tokens → prompt_tokens, output_tokens → completion_tokens
        if self.input_tokens and not self.prompt_tokens:
            self.prompt_tokens = self.input_tokens
        if self.output_tokens and not self.completion_tokens:
            self.completion_tokens = self.output_tokens
        if self.task and not self.task_goal:
            self.task_goal = self.task
        # Ensure canonical fields mirror aliases
        self.input_tokens = self.prompt_tokens
        self.output_tokens = self.completion_tokens
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens
        if self.estimated_cost == 0.0 and self.total_tokens > 0 and self.model:
            pricing = get_pricing(self.model)
            self.estimated_cost = pricing.cost(
                self.prompt_tokens, self.completion_tokens
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN TRACKER SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════


class TokenTracker:
    """
    System-wide token usage tracker.

    Thread-safe singleton that accumulates token/cost data per agent,
    per task, per day. Integrates with MetricsCollector for unified reporting.

    Usage::

        tracker = get_token_tracker()
        tracker.record(TokenUsage(agent_name="bot", prompt_tokens=100, ...))
        tracker.total_tokens_today()
        tracker.cost_by_agent("bot")
    """

    _instance: Optional["TokenTracker"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "TokenTracker":
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._records: List[TokenUsage] = []
                inst._record_lock = threading.Lock()
                inst._max_records = 100_000
                # Quick lookup caches
                inst._agent_totals: Dict[str, Dict[str, float]] = {}
                inst._daily_totals: Dict[str, Dict[str, float]] = {}
                cls._instance = inst
            return cls._instance

    # ── Recording ─────────────────────────────────────────────────────────────

    def record(self, usage: TokenUsage) -> None:
        """Record a token usage event."""
        with self._record_lock:
            self._records.append(usage)
            # Cap size
            if len(self._records) > self._max_records:
                self._records = self._records[-(self._max_records // 2) :]

            # Update agent totals
            agent = usage.agent_name or "_unknown"
            if agent not in self._agent_totals:
                self._agent_totals[agent] = {"tokens": 0, "cost": 0.0, "calls": 0}
            self._agent_totals[agent]["tokens"] += usage.total_tokens
            self._agent_totals[agent]["cost"] += usage.estimated_cost
            self._agent_totals[agent]["calls"] += 1

            # Update daily totals
            if isinstance(usage.timestamp, (int, float)):
                day = datetime.fromtimestamp(usage.timestamp, tz=timezone.utc).strftime(
                    "%Y-%m-%d"
                )
            else:
                day = str(usage.timestamp)[:10]  # "2026-03-10"
            if day not in self._daily_totals:
                self._daily_totals[day] = {"tokens": 0, "cost": 0.0, "calls": 0}
            self._daily_totals[day]["tokens"] += usage.total_tokens
            self._daily_totals[day]["cost"] += usage.estimated_cost
            self._daily_totals[day]["calls"] += 1

    # ── Queries ───────────────────────────────────────────────────────────────

    def total_tokens_today(self) -> int:
        """Total tokens used today."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return int(self._daily_totals.get(today, {}).get("tokens", 0))

    def estimated_cost_today(self) -> float:
        """Total estimated cost today in USD."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self._daily_totals.get(today, {}).get("cost", 0.0)

    def cost_by_agent(self, agent_name: str = "") -> Any:
        """Total cost for a specific agent, or dict of all agents if no name given."""
        if not agent_name:
            return {name: data["cost"] for name, data in self._agent_totals.items()}
        return self._agent_totals.get(agent_name, {}).get("cost", 0.0)

    def tokens_by_agent(self, agent_name: str = "") -> Any:
        """Total tokens for a specific agent, or dict of all agents if no name given."""
        if not agent_name:
            return {
                name: int(data["tokens"]) for name, data in self._agent_totals.items()
            }
        return int(self._agent_totals.get(agent_name, {}).get("tokens", 0))

    def most_expensive_tasks(self, top: int = 5) -> List[Dict[str, Any]]:
        """Return the top N most expensive individual task records."""
        with self._record_lock:
            sorted_records = sorted(
                self._records, key=lambda r: r.estimated_cost, reverse=True
            )
        return [
            {
                "agent": r.agent_name,
                "task": r.task_goal[:100],
                "tokens": r.total_tokens,
                "cost": r.estimated_cost,
                "model": r.model,
                "timestamp": r.timestamp,
            }
            for r in sorted_records[:top]
        ]

    def most_expensive_agents(self, top: int = 5) -> List[Dict[str, Any]]:
        """Return agents ranked by total cost."""
        ranked = sorted(
            self._agent_totals.items(),
            key=lambda kv: kv[1]["cost"],
            reverse=True,
        )
        return [
            {
                "agent": name,
                "tokens": int(data["tokens"]),
                "cost": data["cost"],
                "calls": int(data["calls"]),
            }
            for name, data in ranked[:top]
        ]

    def daily_summary(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get daily token/cost summary for the last N days."""
        sorted_days = sorted(self._daily_totals.items(), reverse=True)
        return [
            {
                "date": day,
                "tokens": int(data["tokens"]),
                "cost": data["cost"],
                "calls": int(data["calls"]),
            }
            for day, data in sorted_days[:days]
        ]

    def get_records(
        self,
        agent_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[TokenUsage]:
        """Retrieve raw usage records with optional filtering."""
        with self._record_lock:
            records = list(self._records)
        if agent_name:
            records = [r for r in records if r.agent_name == agent_name]
        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records[:limit]

    def export(self, format: str = "json") -> Any:
        """Export token tracking data as JSON string or dict."""
        import json as _json

        data = {
            "total_records": len(self._records),
            "agent_totals": dict(self._agent_totals),
            "daily_totals": dict(self._daily_totals),
        }
        if format == "json":
            return _json.dumps(data, indent=2, default=str)
        return data

    def reset(self) -> None:
        """Reset all tracking data (for testing)."""
        with self._record_lock:
            self._records.clear()
            self._agent_totals.clear()
            self._daily_totals.clear()


def get_token_tracker() -> TokenTracker:
    """Get the global token tracker singleton."""
    return TokenTracker()
