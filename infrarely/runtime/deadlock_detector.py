"""
runtime/deadlock_detector.py — GAP 2: Deadlock Detection
═══════════════════════════════════════════════════════════════════════════════
Standalone deadlock detection subsystem with wait-graph construction,
cycle detection via DFS, and automatic timeout-based deadlock breaking.

Features:
  • Wait-for graph: tracks which agent is waiting for which resource/agent
  • Cycle detection using Tarjan-style DFS colouring
  • Timeout breaker: auto-release waits older than threshold
  • History of detected deadlocks for diagnostics
  • Integration point for scheduler and shared memory locks
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class WaitEdge:
    """An edge in the wait-for graph."""

    waiter: str  # agent_id that is waiting
    holder: str  # agent_id or resource holding the lock
    resource: str = ""  # optional: what resource is being waited on
    wait_started: float = field(default_factory=time.time)
    timeout_ms: float = 10_000.0  # default 10s

    @property
    def wait_duration_ms(self) -> float:
        return (time.time() - self.wait_started) * 1000

    @property
    def is_timed_out(self) -> bool:
        return self.wait_duration_ms > self.timeout_ms


@dataclass
class DeadlockRecord:
    """Record of a detected deadlock."""

    deadlock_id: str = field(default_factory=lambda: f"dl_{uuid.uuid4().hex[:6]}")
    cycle: List[str] = field(default_factory=list)
    detected_at: float = field(default_factory=time.time)
    resolution: str = ""
    victim: str = ""


class DeadlockDetector:
    """
    Wait-graph based deadlock detection and resolution.

    Invariants:
      • Wait graph is updated on every lock/wait operation
      • Cycle detection runs in O(V + E)
      • Timed-out waits are auto-released
      • Deadlock history kept for last 100 incidents
    """

    DEFAULT_TIMEOUT_MS = 10_000.0
    MAX_HISTORY = 100

    def __init__(self, default_timeout_ms: float = DEFAULT_TIMEOUT_MS):
        self._wait_graph: Dict[str, List[WaitEdge]] = defaultdict(list)
        self._deadlock_history: List[DeadlockRecord] = []
        self._default_timeout_ms = default_timeout_ms
        self._total_deadlocks = 0
        self._total_timeouts = 0

    # ── Wait registration ─────────────────────────────────────────────────────

    def register_wait(
        self,
        waiter: str,
        holder: str,
        resource: str = "",
        timeout_ms: float = 0,
    ) -> WaitEdge:
        """Register that `waiter` is waiting for `holder`."""
        edge = WaitEdge(
            waiter=waiter,
            holder=holder,
            resource=resource,
            timeout_ms=timeout_ms or self._default_timeout_ms,
        )
        self._wait_graph[waiter].append(edge)
        return edge

    def release_wait(self, waiter: str, holder: str = "") -> int:
        """Remove wait edges. Returns count removed."""
        edges = self._wait_graph.get(waiter, [])
        if not edges:
            return 0
        if holder:
            before = len(edges)
            self._wait_graph[waiter] = [e for e in edges if e.holder != holder]
            removed = before - len(self._wait_graph[waiter])
        else:
            removed = len(edges)
            self._wait_graph.pop(waiter, None)
        if not self._wait_graph.get(waiter):
            self._wait_graph.pop(waiter, None)
        return removed

    def release_all(self, agent_id: str) -> int:
        """Release all waits involving this agent (as waiter or holder)."""
        count = 0
        # As waiter
        count += len(self._wait_graph.pop(agent_id, []))
        # As holder
        for waiter in list(self._wait_graph):
            before = len(self._wait_graph[waiter])
            self._wait_graph[waiter] = [
                e for e in self._wait_graph[waiter] if e.holder != agent_id
            ]
            count += before - len(self._wait_graph[waiter])
            if not self._wait_graph[waiter]:
                self._wait_graph.pop(waiter, None)
        return count

    # ── Cycle detection ───────────────────────────────────────────────────────

    def detect_cycles(self) -> List[List[str]]:
        """
        Detect all cycles in the wait-for graph using DFS colouring.
        Returns list of cycles (each cycle is a list of agent_ids).
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        # Build adjacency from wait graph
        adj: Dict[str, Set[str]] = defaultdict(set)
        all_nodes: Set[str] = set()
        for waiter, edges in self._wait_graph.items():
            all_nodes.add(waiter)
            for edge in edges:
                adj[waiter].add(edge.holder)
                all_nodes.add(edge.holder)

        color: Dict[str, int] = {n: WHITE for n in all_nodes}
        parent: Dict[str, str] = {}
        cycles: List[List[str]] = []

        def dfs(node: str):
            color[node] = GRAY
            for neighbor in adj.get(node, set()):
                if neighbor not in color:
                    continue
                if color[neighbor] == GRAY:
                    # Found cycle — reconstruct
                    cycle = [neighbor, node]
                    current = node
                    while current in parent and parent[current] != neighbor:
                        current = parent[current]
                        cycle.append(current)
                    cycles.append(cycle)
                elif color[neighbor] == WHITE:
                    parent[neighbor] = node
                    dfs(neighbor)
            color[node] = BLACK

        for node in all_nodes:
            if color[node] == WHITE:
                dfs(node)

        return cycles

    def has_deadlock(self) -> bool:
        """Quick check — is there any cycle?"""
        return len(self.detect_cycles()) > 0

    # ── Timeout breaker ───────────────────────────────────────────────────────

    def check_timeouts(self) -> List[WaitEdge]:
        """Find and release all timed-out waits. Returns released edges."""
        timed_out: List[WaitEdge] = []
        for waiter in list(self._wait_graph):
            edges = self._wait_graph.get(waiter, [])
            expired = [e for e in edges if e.is_timed_out]
            timed_out.extend(expired)
            if expired:
                self._wait_graph[waiter] = [e for e in edges if not e.is_timed_out]
                if not self._wait_graph[waiter]:
                    self._wait_graph.pop(waiter, None)
        self._total_timeouts += len(timed_out)
        return timed_out

    # ── Resolution ────────────────────────────────────────────────────────────

    def break_deadlock(self, cycle: List[str]) -> Optional[str]:
        """
        Break a deadlock by releasing the waits of the first agent in the cycle
        (the 'victim'). Returns the victim agent_id.
        """
        if not cycle:
            return None

        victim = cycle[0]
        self.release_wait(victim)

        record = DeadlockRecord(
            cycle=list(cycle),
            resolution="victim_released",
            victim=victim,
        )
        self._deadlock_history.append(record)
        if len(self._deadlock_history) > self.MAX_HISTORY:
            self._deadlock_history = self._deadlock_history[-self.MAX_HISTORY :]
        self._total_deadlocks += 1
        return victim

    def detect_and_resolve(self) -> List[str]:
        """
        Full detection + resolution cycle.
        Returns list of victim agent_ids whose waits were released.
        """
        # First check timeouts
        self.check_timeouts()

        # Then detect cycles
        cycles = self.detect_cycles()
        victims: List[str] = []
        for cycle in cycles:
            victim = self.break_deadlock(cycle)
            if victim:
                victims.append(victim)
        return victims

    # ── Query ─────────────────────────────────────────────────────────────────

    def wait_count(self) -> int:
        return sum(len(edges) for edges in self._wait_graph.values())

    def waiting_agents(self) -> List[str]:
        return list(self._wait_graph.keys())

    def waits_for(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get what an agent is waiting for."""
        return [
            {
                "holder": e.holder,
                "resource": e.resource,
                "wait_ms": round(e.wait_duration_ms),
            }
            for e in self._wait_graph.get(agent_id, [])
        ]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "active_waits": self.wait_count(),
            "waiting_agents": self.waiting_agents(),
            "total_deadlocks": self._total_deadlocks,
            "total_timeouts": self._total_timeouts,
            "recent_deadlocks": [
                {
                    "id": d.deadlock_id,
                    "cycle": d.cycle,
                    "victim": d.victim,
                    "resolution": d.resolution,
                }
                for d in self._deadlock_history[-5:]
            ],
        }
