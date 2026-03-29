"""
infrarely/_internal/plan_bridge.py — SDK ↔ Planning Engine bridge
═══════════════════════════════════════════════════════════════════════════════
Wraps the InfraRely Deterministic Planning Engine for SDK use.
Handles plan generation, validation, and caching.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PlanStep:
    """A compiled plan step."""

    id: str
    capability: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    timeout_seconds: int = 30
    required: bool = True


@dataclass
class Plan:
    """A validated execution plan."""

    plan_id: str = ""
    goal: str = ""
    steps: List[PlanStep] = field(default_factory=list)
    source: str = "deterministic"  # "deterministic" | "llm" | "cached"
    valid: bool = True
    errors: List[str] = field(default_factory=list)


class PlanCache:
    """LRU cache for compiled plans."""

    def __init__(self, max_size: int = 100):
        self._cache: OrderedDict[str, Plan] = OrderedDict()
        self._max_size = max_size

    def _make_key(self, goal: str, tools: List[str]) -> str:
        data = json.dumps({"goal": goal, "tools": sorted(tools)}, sort_keys=True)
        return hashlib.md5(data.encode()).hexdigest()

    def get(self, goal: str, tools: List[str]) -> Optional[Plan]:
        key = self._make_key(goal, tools)
        if key in self._cache:
            self._cache.move_to_end(key)
            plan = self._cache[key]
            plan.source = "cached"
            return plan
        return None

    def put(self, goal: str, tools: List[str], plan: Plan) -> None:
        key = self._make_key(goal, tools)
        if key in self._cache:
            del self._cache[key]
        self._cache[key] = plan
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)
