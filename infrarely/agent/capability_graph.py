"""
agent/capability_graph.py  — CapabilityGraph with dependency resolution
═══════════════════════════════════════════════════════════════════════════════
Graph-based representation of capability workflows.

Each node is a tool invocation with:
  - dependencies  (other node names that must complete first)
  - timeout       (per-node override, defaults to config.TOOL_TIMEOUT_SECONDS)
  - retry_policy  (max_retries for transient failures)
  - allow_llm     (whether this node's tool may use LLM internally)

Execution order is determined by topological sort of the dependency DAG.
Independent nodes at the same depth level CAN run in parallel (future).
Currently executed sequentially in topological order.

Design contract
───────────────
  • The graph is immutable once built (frozen=True on GraphPlan).
  • No cycles — the compiler rejects cyclic capabilities.
  • Each node produces exactly one ToolResult.
  • Node outputs are merged into a shared context dict for downstream nodes.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from infrarely.agent.capability import CapabilityStep, FailurePolicy
from infrarely.agent.state import ResponseType
from infrarely.observability import logger
import infrarely.core.app_config as config


# ─── Graph node ───────────────────────────────────────────────────────────────
@dataclass
class GraphNode:
    """
    One node in the CapabilityGraph = one tool invocation.

    Fields beyond CapabilityStep:
      dependencies  — names of nodes that must complete before this one runs
      timeout       — per-node timeout in seconds (None = use global default)
      retry_policy  — max retries on transient failure (0 = no retry)
      allow_llm     — whether this node's tool is permitted to call LLM
      depth         — set by topological sort (0 = no dependencies)
    """

    name: str
    tool_name: str
    base_params: Dict[str, Any] = field(default_factory=dict)
    failure_policy: FailurePolicy = FailurePolicy.ABORT
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    retry_policy: int = 0  # max_retries
    allow_llm: bool = False
    depth: int = 0  # set by topo sort

    def effective_timeout(self) -> float:
        return self.timeout or getattr(config, "TOOL_TIMEOUT_SECONDS", 5.0)


# ─── Graph edge ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GraphEdge:
    """Directed dependency edge: `from_node` must complete before `to_node`."""

    from_node: str
    to_node: str


# ─── Capability Graph ─────────────────────────────────────────────────────────
@dataclass
class CapabilityGraph:
    """
    Immutable DAG of GraphNodes representing a workflow.

    Built by CapabilityGraphBuilder from a Capability definition.
    Consumed by the executor for dependency-ordered execution.
    """

    name: str
    description: str
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: List[GraphEdge] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)

    @property
    def depth_levels(self) -> Dict[int, List[str]]:
        """Group nodes by depth level for potential parallel execution."""
        levels: Dict[int, List[str]] = {}
        for name in self.execution_order:
            d = self.nodes[name].depth
            levels.setdefault(d, []).append(name)
        return levels

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    def validate(self) -> List[str]:
        """
        Validate graph integrity (Gap 4). Returns list of issues found.
        Checks:
          1. No cycles (topological sort covers all nodes)
          2. No dangling dependencies (references to non-existent nodes)
          3. No self-references
          4. All nodes have valid tool names (non-empty)
          5. Execution order covers all nodes
        """
        issues: List[str] = []
        node_names = set(self.nodes.keys())

        for name, node in self.nodes.items():
            # Self-reference check
            if name in node.dependencies:
                issues.append(f"Node '{name}' has self-dependency")
            # Dangling dependency check
            for dep in node.dependencies:
                if dep not in node_names:
                    issues.append(
                        f"Node '{name}' depends on '{dep}' which doesn't exist"
                    )
            # Empty tool name
            if not node.tool_name:
                issues.append(f"Node '{name}' has empty tool_name")

        # Cycle detection (re-verify via Kahn's)
        in_deg = {n: 0 for n in self.nodes}
        for node in self.nodes.values():
            for dep in node.dependencies:
                if dep in in_deg:
                    in_deg[node.name] += 1
        queue = deque(n for n, d in in_deg.items() if d == 0)
        visited = 0
        while queue:
            cur = queue.popleft()
            visited += 1
            for node in self.nodes.values():
                if cur in node.dependencies:
                    in_deg[node.name] -= 1
                    if in_deg[node.name] == 0:
                        queue.append(node.name)
        if visited < len(self.nodes):
            cycle_nodes = {n for n, d in in_deg.items() if d > 0}
            issues.append(f"Cycle detected involving nodes: {cycle_nodes}")

        # Execution order completeness
        if set(self.execution_order) != node_names:
            missing = node_names - set(self.execution_order)
            extra = set(self.execution_order) - node_names
            if missing:
                issues.append(f"Execution order missing nodes: {missing}")
            if extra:
                issues.append(f"Execution order has unknown nodes: {extra}")

        if issues:
            logger.warn(
                f"Graph validation for '{self.name}': {len(issues)} issue(s)",
            )
        return issues


# ─── Graph Builder ────────────────────────────────────────────────────────────
class CapabilityGraphBuilder:
    """
    Builds a CapabilityGraph from a Capability (list of CapabilitySteps).

    Dependencies are inferred from {step_name.field} param references
    AND from the ToolRegistry response_type (deterministic before generative).
    """

    def __init__(self, registry=None):
        self._registry = registry

    def build(self, capability) -> CapabilityGraph:
        """
        Convert a Capability into a CapabilityGraph.
        Infers dependencies from param references.
        Computes topological execution order.
        """
        from infrarely.agent.capability import Capability

        nodes: Dict[str, GraphNode] = {}
        step_names = {s.name for s in capability.steps}

        # ── 1. Create nodes with inferred dependencies ────────────────────────
        for step in capability.steps:
            deps = self._infer_dependencies(step, step_names)
            allow_llm = False
            if self._registry:
                meta = self._registry.get_meta(step.tool_name)
                if meta and meta.requires_llm:
                    allow_llm = True

            node = GraphNode(
                name=step.name,
                tool_name=step.tool_name,
                base_params=dict(step.base_params),
                failure_policy=step.failure_policy,
                condition=step.condition,
                description=step.description,
                dependencies=deps,
                allow_llm=allow_llm,
            )
            nodes[node.name] = node

        # ── 2. Build edges ────────────────────────────────────────────────────
        edges: List[GraphEdge] = []
        for node in nodes.values():
            for dep in node.dependencies:
                edges.append(GraphEdge(from_node=dep, to_node=node.name))

        # ── 3. Topological sort with depth assignment ─────────────────────────
        execution_order = self._topological_sort(nodes)

        # ── 4. Assign depth levels ────────────────────────────────────────────
        self._assign_depths(nodes, execution_order)

        graph = CapabilityGraph(
            name=capability.name,
            description=capability.description,
            nodes=nodes,
            edges=edges,
            execution_order=execution_order,
        )

        logger.debug(
            f"CapabilityGraph built for '{capability.name}'",
            nodes=len(nodes),
            edges=len(edges),
            depths=dict(graph.depth_levels),
        )

        # ── 5. Validate graph integrity (Gap 4) ──────────────────────────────
        issues = graph.validate()
        if issues:
            for issue in issues:
                logger.warn(f"  Graph issue: {issue}")

        return graph

    @staticmethod
    def _infer_dependencies(step: CapabilityStep, all_names: Set[str]) -> List[str]:
        """Extract dependency names from {step_name.field} param references."""
        deps: List[str] = []
        for _key, value in step.base_params.items():
            if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                ref = value[1:-1]
                root = ref.split(".")[0].split("[")[0]
                if root in all_names and root != step.name:
                    if root not in deps:
                        deps.append(root)
        return deps

    @staticmethod
    def _topological_sort(nodes: Dict[str, GraphNode]) -> List[str]:
        """
        Kahn's algorithm for topological ordering.
        Falls back to declaration order for nodes with equal in-degree.
        """
        in_degree: Dict[str, int] = {name: 0 for name in nodes}
        adj: Dict[str, List[str]] = {name: [] for name in nodes}

        for node in nodes.values():
            for dep in node.dependencies:
                if dep in adj:
                    adj[dep].append(node.name)
                    in_degree[node.name] += 1

        # Seed queue with nodes that have no dependencies
        queue = deque(name for name, deg in in_degree.items() if deg == 0)
        order: List[str] = []

        while queue:
            current = queue.popleft()
            order.append(current)
            for neighbour in adj.get(current, []):
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    queue.append(neighbour)

        if len(order) != len(nodes):
            # Cycle detected — shouldn't happen if compiler validated
            missing = set(nodes) - set(order)
            logger.error(f"Topological sort incomplete — possible cycle in {missing}")
            # Append remaining in declaration order
            for name in nodes:
                if name not in order:
                    order.append(name)

        return order

    @staticmethod
    def _assign_depths(nodes: Dict[str, GraphNode], order: List[str]):
        """
        Assign depth = 1 + max(depth of dependencies).
        Nodes with no deps get depth 0.
        """
        for name in order:
            node = nodes[name]
            if not node.dependencies:
                node.depth = 0
            else:
                node.depth = 1 + max(
                    nodes[dep].depth for dep in node.dependencies if dep in nodes
                )
