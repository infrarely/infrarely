"""
runtime/task_graph.py — GAP 1: Global Task Graph (DAG)
═══════════════════════════════════════════════════════════════════════════════
A full Task DAG that tracks nodes, edges (dependencies), state transitions,
and retries for multi-agent orchestration.

Features:
  • TaskNode with state machine (PENDING → READY → RUNNING → DONE / FAILED)
  • Dependency edges with automatic readiness propagation
  • Topological ordering for execution planning
  • Retry with configurable max_retries per node
  • Cycle detection before execution
  • DAG-level progress and completion tracking
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple


class NodeState(Enum):
    PENDING = auto()  # waiting for dependencies
    READY = auto()  # all deps satisfied, can run
    RUNNING = auto()  # currently executing
    DONE = auto()  # completed successfully
    FAILED = auto()  # failed (may retry)
    SKIPPED = auto()  # skipped due to upstream failure
    RETRYING = auto()  # queued for retry


@dataclass
class TaskNode:
    """A single node in the task DAG."""

    node_id: str
    label: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    assigned_agent: str = ""
    state: NodeState = NodeState.PENDING
    priority: int = 5
    max_retries: int = 2
    retries: int = 0
    result: Any = None
    error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    timeout_ms: float = 30_000.0

    @property
    def elapsed_ms(self) -> float:
        if self.started_at:
            end = self.completed_at or time.time()
            return (end - self.started_at) * 1000
        return 0.0

    @property
    def is_terminal(self) -> bool:
        return self.state in (NodeState.DONE, NodeState.FAILED, NodeState.SKIPPED)

    @property
    def can_retry(self) -> bool:
        return self.state == NodeState.FAILED and self.retries < self.max_retries


class TaskGraph:
    """
    Directed Acyclic Graph of tasks for orchestrating multi-agent work.

    Invariants:
      • No cycles allowed — checked on add_edge and validated before run
      • Nodes auto-transition PENDING → READY when all deps are DONE
      • Failed upstream propagates SKIPPED to downstream unless retried
      • Max 500 nodes per graph
    """

    MAX_NODES = 500

    def __init__(self, graph_id: str = ""):
        self.graph_id = graph_id or f"dag_{uuid.uuid4().hex[:8]}"
        self._nodes: Dict[str, TaskNode] = {}
        self._edges: Dict[str, Set[str]] = defaultdict(set)  # node → deps
        self._reverse: Dict[str, Set[str]] = defaultdict(set)  # node → dependants
        self._created_at = time.time()

    # ── Node management ───────────────────────────────────────────────────────

    def add_node(
        self,
        node_id: str = "",
        label: str = "",
        priority: int = 5,
        max_retries: int = 2,
        payload: Dict[str, Any] = None,
        timeout_ms: float = 30_000.0,
    ) -> TaskNode:
        """Add a task node to the graph."""
        if len(self._nodes) >= self.MAX_NODES:
            raise ValueError(f"Task graph full ({self.MAX_NODES} nodes)")
        nid = node_id or f"node_{uuid.uuid4().hex[:6]}"
        if nid in self._nodes:
            raise ValueError(f"Node '{nid}' already exists")
        node = TaskNode(
            node_id=nid,
            label=label or nid,
            priority=priority,
            max_retries=max_retries,
            payload=payload or {},
            timeout_ms=timeout_ms,
        )
        self._nodes[nid] = node
        return node

    def get_node(self, node_id: str) -> Optional[TaskNode]:
        return self._nodes.get(node_id)

    def remove_node(self, node_id: str) -> bool:
        if node_id not in self._nodes:
            return False
        del self._nodes[node_id]
        # Clean edges
        self._edges.pop(node_id, None)
        self._reverse.pop(node_id, None)
        for deps in self._edges.values():
            deps.discard(node_id)
        for deps in self._reverse.values():
            deps.discard(node_id)
        return True

    # ── Edge management ───────────────────────────────────────────────────────

    def add_edge(self, from_node: str, to_node: str) -> bool:
        """
        Add dependency: to_node depends on from_node finishing first.
        Returns False if would create a cycle.
        """
        if from_node not in self._nodes or to_node not in self._nodes:
            return False
        if from_node == to_node:
            return False

        # Tentatively add and check cycle
        self._edges[to_node].add(from_node)
        self._reverse[from_node].add(to_node)

        if self._has_cycle():
            self._edges[to_node].discard(from_node)
            self._reverse[from_node].discard(to_node)
            return False
        return True

    def get_dependencies(self, node_id: str) -> Set[str]:
        """Get direct dependencies of a node."""
        return set(self._edges.get(node_id, set()))

    def get_dependants(self, node_id: str) -> Set[str]:
        """Get nodes that depend on this node."""
        return set(self._reverse.get(node_id, set()))

    # ── Cycle detection ───────────────────────────────────────────────────────

    def _has_cycle(self) -> bool:
        """Kahn's algorithm — returns True if graph has a cycle."""
        in_degree: Dict[str, int] = {nid: 0 for nid in self._nodes}
        for nid, deps in self._edges.items():
            if nid in in_degree:
                in_degree[nid] = len(deps & self._nodes.keys())
        queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
        visited = 0
        while queue:
            nid = queue.popleft()
            visited += 1
            for child in self._reverse.get(nid, set()):
                if child in in_degree:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
        return visited < len(self._nodes)

    def validate(self) -> Tuple[bool, str]:
        """Validate DAG: no cycles, all edge refs valid."""
        if self._has_cycle():
            return False, "Graph contains a cycle"
        for nid, deps in self._edges.items():
            for dep in deps:
                if dep not in self._nodes:
                    return False, f"Dangling dependency: {nid} → {dep}"
        return True, "OK"

    # ── Topological sort ──────────────────────────────────────────────────────

    def topological_order(self) -> List[str]:
        """Return nodes in topological order (dependencies first)."""
        in_degree: Dict[str, int] = {nid: 0 for nid in self._nodes}
        for nid, deps in self._edges.items():
            if nid in in_degree:
                in_degree[nid] = len(deps & self._nodes.keys())
        queue = deque(
            sorted(
                (nid for nid, deg in in_degree.items() if deg == 0),
                key=lambda n: self._nodes[n].priority,
                reverse=True,
            )
        )
        order: List[str] = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            children = sorted(
                self._reverse.get(nid, set()),
                key=lambda n: self._nodes[n].priority if n in self._nodes else 0,
                reverse=True,
            )
            for child in children:
                if child in in_degree:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
        return order

    # ── State transitions ─────────────────────────────────────────────────────

    def refresh_ready(self):
        """Promote PENDING nodes to READY if all deps are DONE."""
        for nid, node in self._nodes.items():
            if node.state != NodeState.PENDING:
                continue
            deps = self._edges.get(nid, set())
            if not deps:
                node.state = NodeState.READY
                continue
            all_done = all(
                self._nodes.get(d) and self._nodes[d].state == NodeState.DONE
                for d in deps
            )
            any_failed = any(
                self._nodes.get(d)
                and self._nodes[d].state in (NodeState.FAILED, NodeState.SKIPPED)
                for d in deps
            )
            if all_done:
                node.state = NodeState.READY
            elif any_failed:
                # Upstream failed and can't retry → skip
                upstream_unrecoverable = any(
                    self._nodes.get(d)
                    and self._nodes[d].state in (NodeState.FAILED, NodeState.SKIPPED)
                    and not self._nodes[d].can_retry
                    for d in deps
                )
                if upstream_unrecoverable:
                    node.state = NodeState.SKIPPED
                    node.error = "Upstream dependency failed"

    def mark_running(self, node_id: str, agent_id: str = "") -> bool:
        node = self._nodes.get(node_id)
        if not node or node.state != NodeState.READY:
            return False
        node.state = NodeState.RUNNING
        node.started_at = time.time()
        node.assigned_agent = agent_id
        return True

    def mark_done(self, node_id: str, result: Any = None) -> bool:
        node = self._nodes.get(node_id)
        if not node or node.state != NodeState.RUNNING:
            return False
        node.state = NodeState.DONE
        node.completed_at = time.time()
        node.result = result
        self.refresh_ready()
        return True

    def mark_failed(self, node_id: str, error: str = "") -> bool:
        node = self._nodes.get(node_id)
        if not node or node.state != NodeState.RUNNING:
            return False
        node.state = NodeState.FAILED
        node.completed_at = time.time()
        node.error = error
        self.refresh_ready()
        return True

    def retry_node(self, node_id: str) -> bool:
        """Retry a failed node if retries remain."""
        node = self._nodes.get(node_id)
        if not node or not node.can_retry:
            return False
        node.retries += 1
        node.state = NodeState.READY
        node.started_at = 0.0
        node.completed_at = 0.0
        node.error = ""
        node.result = None
        return True

    # ── Query ─────────────────────────────────────────────────────────────────

    def ready_nodes(self) -> List[TaskNode]:
        """Get all READY nodes, sorted by priority (desc)."""
        self.refresh_ready()
        nodes = [n for n in self._nodes.values() if n.state == NodeState.READY]
        nodes.sort(key=lambda n: n.priority, reverse=True)
        return nodes

    def running_nodes(self) -> List[TaskNode]:
        return [n for n in self._nodes.values() if n.state == NodeState.RUNNING]

    def is_complete(self) -> bool:
        """True if all nodes are in a terminal state."""
        return all(n.is_terminal for n in self._nodes.values())

    def progress(self) -> Dict[str, int]:
        """Count nodes by state."""
        counts: Dict[str, int] = defaultdict(int)
        for n in self._nodes.values():
            counts[n.state.name] += 1
        return dict(counts)

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return sum(len(deps) for deps in self._edges.values())

    def snapshot(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "nodes": self.node_count(),
            "edges": self.edge_count(),
            "progress": self.progress(),
            "is_complete": self.is_complete(),
            "ready": [n.node_id for n in self.ready_nodes()],
            "running": [n.node_id for n in self.running_nodes()],
        }
