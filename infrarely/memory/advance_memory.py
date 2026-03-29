"""
memory/advanced_memory.py  — Gap 9: Advanced Memory Manager
Four-layer memory: Working, Episodic, Semantic, Procedural.

Working Memory   — current conversation turns (already exists, wrapped here)
Episodic Memory  — concrete events ("on 2026-03-08, user asked about CS301 exam")
Semantic Memory  — facts ("student is in year 2", "CS301 has exam in 4 weeks")
Procedural Memory— learned patterns ("when user asks about exam, show topics first")
"""
from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import infrarely.core.app_config as config
from infrarely.observability import logger

_EPISODIC_MAX  = getattr(config, "EPISODIC_MEMORY_MAX_EVENTS",  200)
_SEMANTIC_MAX  = getattr(config, "SEMANTIC_MEMORY_MAX_FACTS",    500)
_PROCEDURAL_MAX= getattr(config, "PROCEDURAL_MEMORY_MAX_RULES",  100)


@dataclass
class Episode:
    ts:       str
    agent_id: str
    event:    str
    context:  Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def now(agent_id: str, event: str, **ctx) -> "Episode":
        return Episode(
            ts=datetime.now(timezone.utc).isoformat(),
            agent_id=agent_id, event=event, context=ctx,
        )


@dataclass
class SemanticFact:
    key:      str
    value:    Any
    source:   str   = ""
    ts:       str   = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ttl_secs: Optional[float] = None   # None = permanent

    def is_expired(self) -> bool:
        if self.ttl_secs is None:
            return False
        age = time.time() - datetime.fromisoformat(self.ts).timestamp()
        return age > self.ttl_secs


@dataclass
class ProceduralRule:
    trigger:  str         # intent or pattern string
    action:   str         # recommended tool or workflow
    weight:   float = 1.0 # higher = stronger preference
    uses:     int   = 0


class AdvancedMemoryManager:
    """
    Provides all four memory layers for a single student_id.
    Backed by in-memory structures; episodic and semantic can be
    persisted to JSON via save()/load().
    """

    def __init__(self, agent_id: str):
        self._agent_id    = agent_id
        self._episodic:   List[Episode]         = []
        self._semantic:   Dict[str, SemanticFact] = {}
        self._procedural: List[ProceduralRule]  = []

    # ── Episodic ──────────────────────────────────────────────────────────────
    def record_episode(self, event: str, **context):
        ep = Episode.now(self._agent_id, event, **context)
        self._episodic.append(ep)
        if len(self._episodic) > _EPISODIC_MAX:
            self._episodic = self._episodic[-_EPISODIC_MAX:]
        logger.debug(f"EpisodicMemory: recorded '{event}'", agent=self._agent_id)

    def recent_episodes(self, n: int = 10) -> List[Episode]:
        return self._episodic[-n:]

    # ── Semantic ──────────────────────────────────────────────────────────────
    def set_fact(self, key: str, value: Any, source: str = "", ttl_secs: float = None):
        self._semantic[key] = SemanticFact(key=key, value=value,
                                           source=source, ttl_secs=ttl_secs)
        if len(self._semantic) > _SEMANTIC_MAX:
            # evict oldest
            oldest = sorted(self._semantic, key=lambda k: self._semantic[k].ts)
            del self._semantic[oldest[0]]

    def get_fact(self, key: str) -> Optional[Any]:
        fact = self._semantic.get(key)
        if fact is None:
            return None
        if fact.is_expired():
            del self._semantic[key]
            return None
        return fact.value

    def facts_snapshot(self) -> Dict[str, Any]:
        return {k: f.value for k, f in self._semantic.items() if not f.is_expired()}

    # ── Procedural ────────────────────────────────────────────────────────────
    def add_rule(self, trigger: str, action: str, weight: float = 1.0):
        for rule in self._procedural:
            if rule.trigger == trigger and rule.action == action:
                rule.weight = max(rule.weight, weight)
                return
        self._procedural.append(ProceduralRule(trigger=trigger, action=action, weight=weight))
        if len(self._procedural) > _PROCEDURAL_MAX:
            self._procedural.sort(key=lambda r: -r.weight)
            self._procedural = self._procedural[:_PROCEDURAL_MAX]

    def best_action(self, trigger: str) -> Optional[str]:
        candidates = [r for r in self._procedural if r.trigger in trigger]
        if not candidates:
            return None
        best = max(candidates, key=lambda r: r.weight * (r.uses + 1))
        best.uses += 1
        return best.action

    # ── CP10: Capability integration helpers ───────────────────────────────
    def store_capability_facts(self, capability_name: str, step_outputs: dict):
        """Store semantic facts from capability execution results."""
        for step_name, result in step_outputs.items():
            if result.success and result.data:
                key = f"{capability_name}.{step_name}"
                if isinstance(result.data, dict):
                    for dk, dv in list(result.data.items())[:10]:
                        self.set_fact(f"{key}.{dk}", dv, source=capability_name)
                else:
                    self.set_fact(key, result.data, source=capability_name)
        logger.debug(f"SemanticMemory: stored facts from '{capability_name}'",
                     agent=self._agent_id)

    def learn_pattern(self, intent: str, capability_name: str):
        """Record a successful intent→capability mapping in procedural memory."""
        self.add_rule(intent, capability_name, weight=1.0)
        logger.debug(f"ProceduralMemory: intent '{intent}' → capability '{capability_name}'",
                     agent=self._agent_id)

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: str):
        payload = {
            "agent_id":   self._agent_id,
            "episodic":   [vars(e) for e in self._episodic],
            "semantic":   {k: vars(f) for k, f in self._semantic.items()},
            "procedural": [vars(r) for r in self._procedural],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        self._episodic   = [Episode(**e) for e in data.get("episodic",   [])]
        self._semantic   = {k: SemanticFact(**v) for k, v in data.get("semantic", {}).items()}
        self._procedural = [ProceduralRule(**r)  for r in data.get("procedural", [])]