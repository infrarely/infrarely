"""
runtime/distributed_scalability.py — GAP 6
═══════════════════════════════════════════════════════════════════════════════
Distributed scalability layer — multi-node support, agent sharding, and
distributed message bus abstraction.

In single-process mode (default) everything runs in-memory.
The abstractions are designed so swapping to real distributed backends
(Redis, gRPC, NATS) only requires replacing the transport layer.

Subsystems:
  • NodeRegistry         — tracks cluster nodes (local + remote)
  • AgentShardMap        — maps agents → nodes for routing
  • DistributedBus       — fan-out bus with node-aware delivery
  • DistributedScalability — top-level coordinator with snapshot()
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set


# ─── Enums ────────────────────────────────────────────────────────────────────
class NodeStatus(Enum):
    ONLINE = auto()
    OFFLINE = auto()
    DRAINING = auto()


# ─── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class ClusterNode:
    node_id: str
    address: str = "localhost"
    status: NodeStatus = NodeStatus.ONLINE
    agents: Set[str] = field(default_factory=set)
    joined_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    capacity: int = 50  # max agents on this node
    load: float = 0.0  # 0.0–1.0


@dataclass
class ShardEntry:
    agent_id: str
    node_id: str
    assigned_at: float = field(default_factory=time.time)


@dataclass
class RemoteMessage:
    sender: str
    recipient: str
    payload: Any = None
    origin_node: str = ""
    target_node: str = ""
    ts: float = field(default_factory=time.time)


# ─── Node Registry ───────────────────────────────────────────────────────────
class NodeRegistry:
    """Track cluster nodes and heartbeats."""

    HEARTBEAT_TIMEOUT_S = 30.0

    def __init__(self):
        self._nodes: Dict[str, ClusterNode] = {}

    def register(
        self, node_id: str, address: str = "localhost", capacity: int = 50
    ) -> ClusterNode:
        if node_id in self._nodes:
            raise ValueError(f"Node '{node_id}' already registered")
        node = ClusterNode(node_id=node_id, address=address, capacity=capacity)
        self._nodes[node_id] = node
        return node

    def heartbeat(self, node_id: str) -> bool:
        node = self._nodes.get(node_id)
        if not node or node.status == NodeStatus.OFFLINE:
            return False
        node.last_heartbeat = time.time()
        node.load = len(node.agents) / max(node.capacity, 1)
        return True

    def deregister(self, node_id: str) -> bool:
        node = self._nodes.get(node_id)
        if not node:
            return False
        node.status = NodeStatus.OFFLINE
        return True

    def drain(self, node_id: str) -> bool:
        node = self._nodes.get(node_id)
        if not node:
            return False
        node.status = NodeStatus.DRAINING
        return True

    def online_nodes(self) -> List[ClusterNode]:
        return [n for n in self._nodes.values() if n.status == NodeStatus.ONLINE]

    def least_loaded(self) -> Optional[ClusterNode]:
        online = self.online_nodes()
        if not online:
            return None
        return min(online, key=lambda n: n.load)

    def detect_stale(self) -> List[str]:
        """Find nodes that missed heartbeats."""
        now = time.time()
        stale = []
        for n in self._nodes.values():
            if n.status == NodeStatus.ONLINE:
                if now - n.last_heartbeat > self.HEARTBEAT_TIMEOUT_S:
                    n.status = NodeStatus.OFFLINE
                    stale.append(n.node_id)
        return stale


# ─── Agent Shard Map ─────────────────────────────────────────────────────────
class AgentShardMap:
    """Map agents to cluster nodes."""

    def __init__(self, node_registry: NodeRegistry):
        self._registry = node_registry
        self._shards: Dict[str, ShardEntry] = {}

    def assign(self, agent_id: str, node_id: str = "") -> ShardEntry:
        if not node_id:
            node = self._registry.least_loaded()
            if not node:
                raise RuntimeError("No online nodes available")
            node_id = node.node_id
        node = self._registry._nodes.get(node_id)
        if not node or node.status != NodeStatus.ONLINE:
            raise ValueError(f"Node '{node_id}' not online")
        if len(node.agents) >= node.capacity:
            raise ValueError(f"Node '{node_id}' at capacity ({node.capacity})")
        entry = ShardEntry(agent_id=agent_id, node_id=node_id)
        self._shards[agent_id] = entry
        node.agents.add(agent_id)
        node.load = len(node.agents) / max(node.capacity, 1)
        return entry

    def locate(self, agent_id: str) -> Optional[str]:
        entry = self._shards.get(agent_id)
        return entry.node_id if entry else None

    def reassign(self, agent_id: str, new_node_id: str) -> bool:
        old = self._shards.get(agent_id)
        if not old:
            return False
        old_node = self._registry._nodes.get(old.node_id)
        if old_node:
            old_node.agents.discard(agent_id)
            old_node.load = len(old_node.agents) / max(old_node.capacity, 1)
        new_node = self._registry._nodes.get(new_node_id)
        if not new_node or new_node.status != NodeStatus.ONLINE:
            return False
        new_node.agents.add(agent_id)
        new_node.load = len(new_node.agents) / max(new_node.capacity, 1)
        self._shards[agent_id] = ShardEntry(agent_id=agent_id, node_id=new_node_id)
        return True

    def remove(self, agent_id: str) -> bool:
        entry = self._shards.pop(agent_id, None)
        if not entry:
            return False
        node = self._registry._nodes.get(entry.node_id)
        if node:
            node.agents.discard(agent_id)
            node.load = len(node.agents) / max(node.capacity, 1)
        return True

    def agents_on_node(self, node_id: str) -> List[str]:
        return [e.agent_id for e in self._shards.values() if e.node_id == node_id]


# ─── Distributed Bus ─────────────────────────────────────────────────────────
class DistributedBus:
    """Node-aware message routing (in-process simulation)."""

    def __init__(self, shard_map: AgentShardMap):
        self._shard_map = shard_map
        self._queues: Dict[str, list] = {}  # node_id → [RemoteMessage]
        self._total_routed = 0

    def route(self, msg: RemoteMessage) -> bool:
        target_node = self._shard_map.locate(msg.recipient)
        if not target_node:
            return False
        msg.target_node = target_node
        msg.origin_node = self._shard_map.locate(msg.sender) or "unknown"
        self._queues.setdefault(target_node, []).append(msg)
        self._total_routed += 1
        return True

    def receive(self, node_id: str, limit: int = 50) -> List[RemoteMessage]:
        q = self._queues.get(node_id, [])
        batch = q[:limit]
        self._queues[node_id] = q[limit:]
        return batch

    def pending_count(self, node_id: str) -> int:
        return len(self._queues.get(node_id, []))


# ─── Top-level Coordinator ────────────────────────────────────────────────────
class DistributedScalability:
    """
    Unified access to node registry, shard map, and distributed bus.
    """

    def __init__(self):
        self.nodes = NodeRegistry()
        self.shards = AgentShardMap(self.nodes)
        self.bus = DistributedBus(self.shards)

    def snapshot(self) -> Dict[str, Any]:
        online = self.nodes.online_nodes()
        return {
            "total_nodes": len(self.nodes._nodes),
            "online_nodes": len(online),
            "total_agents_sharded": len(self.shards._shards),
            "total_messages_routed": self.bus._total_routed,
            "node_loads": {n.node_id: round(n.load, 3) for n in online},
        }
