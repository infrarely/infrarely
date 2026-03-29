"""
observability/metrics.py
Lightweight in-process metrics counters (no Prometheus needed).
Provides a snapshot at any time and resets per session.
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List
import time


@dataclass
class ToolMetric:
    call_count:    int   = 0
    error_count:   int   = 0
    total_ms:      float = 0.0
    last_error:    str   = ""

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.call_count if self.call_count else 0.0

    @property
    def error_rate(self) -> float:
        return self.error_count / self.call_count if self.call_count else 0.0


class MetricsCollector:
    """Central metrics store — one instance per agent session."""

    def __init__(self):
        self._tools:        Dict[str, ToolMetric] = defaultdict(ToolMetric)
        self._intent_hits:  Dict[str, int]        = defaultdict(int)
        self._llm_tokens:   List[int]             = []
        self._start_time:   float                 = time.time()
        self.circuit_breaker_trips: int           = 0
        self.retries:       int                   = 0
        self.memory_reads:  int                   = 0
        self.memory_writes: int                   = 0

    # ── recording ────────────────────────────────────────────────────────
    def record_tool_call(self, tool: str, duration_ms: float, success: bool, error: str = ""):
        m = self._tools[tool]
        m.call_count  += 1
        m.total_ms    += duration_ms
        if not success:
            m.error_count += 1
            m.last_error   = error

    def record_intent(self, intent: str):
        self._intent_hits[intent] += 1

    def record_llm_tokens(self, tokens: int):
        self._llm_tokens.append(tokens)

    def record_memory_read(self):
        self.memory_reads += 1

    def record_memory_write(self):
        self.memory_writes += 1

    # ── snapshots ────────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        elapsed = time.time() - self._start_time
        total_tok = sum(self._llm_tokens)
        return {
            "elapsed_seconds":        round(elapsed, 1),
            "total_tokens":           total_tok,
            "avg_tokens_per_llm_call": round(total_tok / len(self._llm_tokens), 1)
                                        if self._llm_tokens else 0,
            "llm_call_count":         len(self._llm_tokens),
            "tool_stats":             {k: {
                                        "calls":      v.call_count,
                                        "errors":     v.error_count,
                                        "avg_ms":     round(v.avg_ms, 1),
                                        "error_rate": f"{v.error_rate:.0%}",
                                       } for k, v in self._tools.items()},
            "intent_distribution":    dict(self._intent_hits),
            "memory_reads":           self.memory_reads,
            "memory_writes":          self.memory_writes,
            "circuit_breaker_trips":  self.circuit_breaker_trips,
            "retries":                self.retries,
        }


# module-level singleton
collector = MetricsCollector()