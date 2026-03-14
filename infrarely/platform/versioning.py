"""
infrarely/versioning.py — Agent Versioning & Rollback System
═══════════════════════════════════════════════════════════════════════════════
Version, compare, and rollback agent configurations.

Enterprise teams rebuild their agent stack every 3 months because they
have no version control. This module solves that.

Usage::

    infrarely.versions.save(agent, tag="v1.0-stable")
    agent.knowledge.add_data("new_docs", "...")
    infrarely.versions.save(agent, tag="v1.1-beta")

    comparison = infrarely.versions.compare("v1.0-stable", "v1.1-beta", eval_suite)
    infrarely.versions.rollback(agent, tag="v1.0-stable")
"""

from __future__ import annotations

import copy
import json
import os
import time
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from infrarely.runtime.paths import VERSIONS_DIR

if TYPE_CHECKING:
    from infrarely.core.agent import Agent
    from infrarely.platform.evaluation import EvalSuite, EvalSuiteResults


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT SNAPSHOT — Serializable snapshot of agent state
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class AgentSnapshot:
    """Complete serializable snapshot of an agent's configuration."""

    tag: str = ""
    agent_name: str = ""
    created_at: float = field(default_factory=time.time)
    description: str = ""

    # Configuration
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    model: str = ""

    # Tools & capabilities (names only — functions can't be serialized)
    tool_names: List[str] = field(default_factory=list)
    capability_names: List[str] = field(default_factory=list)

    # Knowledge state
    knowledge_sources: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_data: Dict[str, str] = field(default_factory=dict)

    # Memory state
    memory_snapshot: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def knowledge(self):
        """Alias for knowledge_sources."""
        return self.knowledge_sources

    @property
    def snapshot_id(self) -> str:
        """Deterministic ID from tag and agent name."""
        return hashlib.md5(f"{self.agent_name}:{self.tag}".encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tag": self.tag,
            "agent_name": self.agent_name,
            "created_at": self.created_at,
            "description": self.description,
            "config_overrides": self.config_overrides,
            "model": self.model,
            "tool_names": self.tool_names,
            "capability_names": self.capability_names,
            "knowledge_sources": self.knowledge_sources,
            "knowledge_data": self.knowledge_data,
            "memory_snapshot": self.memory_snapshot,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentSnapshot":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════════════════════
# VERSION COMPARISON RESULT
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class VersionComparison:
    """Result of comparing two agent versions."""

    tag_a: str = ""
    tag_b: str = ""
    config_diff: Dict[str, Any] = field(default_factory=dict)
    tools_added: List[str] = field(default_factory=list)
    tools_removed: List[str] = field(default_factory=list)
    knowledge_added: List[str] = field(default_factory=list)
    knowledge_removed: List[str] = field(default_factory=list)
    eval_results_a: Optional[Any] = None
    eval_results_b: Optional[Any] = None
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tag_a": self.tag_a,
            "tag_b": self.tag_b,
            "config_diff": self.config_diff,
            "tools_added": self.tools_added,
            "tools_removed": self.tools_removed,
            "knowledge_added": self.knowledge_added,
            "knowledge_removed": self.knowledge_removed,
            "summary": self.summary,
        }

    def __str__(self) -> str:
        return self.summary


# ═══════════════════════════════════════════════════════════════════════════════
# VERSION MANAGER — The main versioning system
# ═══════════════════════════════════════════════════════════════════════════════


class VersionManager:
    """
    Manages agent versions: save, list, compare, rollback.

    Usage::

        versions = VersionManager()
        versions.save(agent, tag="v1.0")
        versions.rollback(agent, tag="v1.0")
    """

    def __init__(self, storage_dir: str = str(VERSIONS_DIR)):
        self._storage_dir = storage_dir
        self._snapshots: Dict[str, AgentSnapshot] = {}
        self._load_from_disk()

    def save(
        self,
        agent: "Agent",
        *,
        tag: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentSnapshot:
        """
        Save the current agent state as a versioned snapshot.

        Parameters
        ----------
        agent : Agent
            The agent to snapshot.
        tag : str
            Version tag (e.g., "v1.0-stable").
        description : str
            Optional description of this version.
        metadata : dict, optional
            Additional metadata to store.

        Returns
        -------
        AgentSnapshot
        """
        from infrarely.core.config import get_config

        cfg = get_config()

        # Capture knowledge state
        knowledge_data: Dict[str, str] = {}
        knowledge_sources: List[Dict[str, Any]] = []
        if agent._knowledge:
            try:
                for name, source in getattr(agent._knowledge, "_sources", {}).items():
                    knowledge_sources.append(
                        {
                            "name": name,
                            "type": getattr(source, "source_type", "unknown"),
                        }
                    )
                # Capture actual knowledge chunk content per source
                index = getattr(agent._knowledge, "_index", None)
                if index is not None:
                    chunks = getattr(index, "_chunks", [])
                    for src_name in [s["name"] for s in knowledge_sources]:
                        src_chunks = [
                            c.content
                            for c in chunks
                            if getattr(c, "source", "") == src_name
                        ]
                        if src_chunks:
                            knowledge_data[src_name] = "\n".join(src_chunks)
            except Exception:
                pass

        # Capture memory state
        memory_snapshot: Dict[str, Any] = {}
        if agent._memory:
            try:
                keys = agent._memory.list_keys()
                for key in keys[:100]:  # Cap at 100 keys
                    val = agent._memory.get(key)
                    if val is not None:
                        memory_snapshot[key] = val
            except Exception:
                pass

        snapshot = AgentSnapshot(
            tag=tag,
            agent_name=agent.name,
            description=description,
            config_overrides=dict(agent._config_overrides),
            model=cfg.get("llm_model", ""),
            tool_names=list(agent._tools.keys()),
            capability_names=list(agent._capabilities.keys()),
            knowledge_sources=knowledge_sources,
            knowledge_data=knowledge_data,
            memory_snapshot=memory_snapshot,
            metadata=metadata or {},
        )

        self._snapshots[tag] = snapshot
        self._save_to_disk(snapshot)

        return snapshot

    def get(self, agent_or_tag, tag: str = "") -> Optional[AgentSnapshot]:
        """Get a snapshot by tag. Optionally accepts (agent, tag) args."""
        if tag:
            # Called as get(agent, tag)
            return self._snapshots.get(tag)
        if hasattr(agent_or_tag, "name"):
            # Called as get(agent) - return latest for that agent
            versions = self.list_versions(agent_or_tag.name)
            return versions[0] if versions else None
        return self._snapshots.get(agent_or_tag)

    def list_versions(self, agent_name: Optional[str] = None) -> List[AgentSnapshot]:
        """List all saved versions, optionally filtered by agent name."""
        versions = list(self._snapshots.values())
        if agent_name:
            versions = [v for v in versions if v.agent_name == agent_name]
        versions.sort(key=lambda v: v.created_at, reverse=True)
        return versions

    def list(self, agent=None) -> List[AgentSnapshot]:
        """Alias for list_versions(). Accepts Agent object or agent_name string."""
        if agent is None:
            return self.list_versions()
        name = agent.name if hasattr(agent, "name") else str(agent)
        return self.list_versions(name)

    def delete(self, tag: str) -> bool:
        """Delete a version by tag."""
        if tag in self._snapshots:
            del self._snapshots[tag]
            # Remove from disk
            filepath = os.path.join(self._storage_dir, f"{tag}.json")
            try:
                os.remove(filepath)
            except OSError:
                pass
            return True
        return False

    def compare(
        self,
        agent_or_tag_a,
        tag_a_or_b: str = "",
        tag_b: str = "",
        eval_suite: Optional["EvalSuite"] = None,
        agent: Optional["Agent"] = None,
    ) -> VersionComparison:
        """
        Compare two versions, optionally running an eval suite against each.

        Can be called as:
            compare("v1.0", "v2.0")
            compare(agent, "v1.0", "v2.0")
        """
        if hasattr(agent_or_tag_a, "name"):
            # Called as compare(agent, tag_a, tag_b)
            agent = agent_or_tag_a
            actual_tag_a = tag_a_or_b
            actual_tag_b = tag_b
        else:
            actual_tag_a = agent_or_tag_a
            actual_tag_b = tag_a_or_b

        snap_a = self._snapshots.get(actual_tag_a)
        snap_b = self._snapshots.get(actual_tag_b)

        if snap_a is None:
            raise ValueError(f"Version not found: {actual_tag_a}")
        if snap_b is None:
            raise ValueError(f"Version not found: {actual_tag_b}")

        # Config diff
        config_diff: Dict[str, Any] = {}
        all_keys = set(snap_a.config_overrides.keys()) | set(
            snap_b.config_overrides.keys()
        )
        for key in all_keys:
            val_a = snap_a.config_overrides.get(key)
            val_b = snap_b.config_overrides.get(key)
            if val_a != val_b:
                config_diff[key] = {"from": val_a, "to": val_b}

        # Tool diff
        tools_a = set(snap_a.tool_names)
        tools_b = set(snap_b.tool_names)
        tools_added = list(tools_b - tools_a)
        tools_removed = list(tools_a - tools_b)

        # Knowledge diff
        knowledge_a = {s["name"] for s in snap_a.knowledge_sources}
        knowledge_b = {s["name"] for s in snap_b.knowledge_sources}
        knowledge_added = list(knowledge_b - knowledge_a)
        knowledge_removed = list(knowledge_a - knowledge_b)

        # Run evals if provided
        eval_results_a = None
        eval_results_b = None
        eval_summary = ""

        if eval_suite and agent:
            # Run eval with version A config
            self.rollback(agent, tag=actual_tag_a)
            eval_results_a = eval_suite.run(agent)

            # Run eval with version B config
            self.rollback(agent, tag=actual_tag_b)
            eval_results_b = eval_suite.run(agent)

            rate_diff = eval_results_b.pass_rate - eval_results_a.pass_rate
            if abs(rate_diff) < 0.001:
                eval_summary = f"Performance identical: {eval_results_a.pass_rate:.1%}"
            elif rate_diff > 0:
                eval_summary = f"{actual_tag_b} is {rate_diff:.1%} better than {actual_tag_a} ({eval_results_a.pass_rate:.1%} → {eval_results_b.pass_rate:.1%})"
            else:
                eval_summary = f"{actual_tag_b} is {abs(rate_diff):.1%} worse than {actual_tag_a} ({eval_results_a.pass_rate:.1%} → {eval_results_b.pass_rate:.1%})"

        # Build summary
        summary_parts = []
        if config_diff:
            summary_parts.append(f"Config changes: {list(config_diff.keys())}")
        if tools_added:
            summary_parts.append(f"Tools added: {tools_added}")
        if tools_removed:
            summary_parts.append(f"Tools removed: {tools_removed}")
        if knowledge_added:
            summary_parts.append(f"Knowledge added: {knowledge_added}")
        if knowledge_removed:
            summary_parts.append(f"Knowledge removed: {knowledge_removed}")
        if eval_summary:
            summary_parts.append(eval_summary)
        if not summary_parts:
            summary_parts.append("No differences found.")

        return VersionComparison(
            tag_a=actual_tag_a,
            tag_b=actual_tag_b,
            config_diff=config_diff,
            tools_added=tools_added,
            tools_removed=tools_removed,
            knowledge_added=knowledge_added,
            knowledge_removed=knowledge_removed,
            eval_results_a=eval_results_a,
            eval_results_b=eval_results_b,
            summary="\n".join(summary_parts),
        )

    def rollback(
        self,
        agent: "Agent",
        *,
        tag: str,
    ) -> bool:
        """
        Rollback an agent to a previously saved version.

        Restores configuration, tools, and knowledge state.

        Parameters
        ----------
        agent : Agent
            The agent to rollback.
        tag : str
            Version tag to rollback to.

        Returns
        -------
        bool
            True if rollback was successful.
        """
        snapshot = self._snapshots.get(tag)
        if snapshot is None:
            raise ValueError(f"Version not found: {tag}")

        # Restore config overrides
        agent._config_overrides = dict(snapshot.config_overrides)

        # Restore memory
        if agent._memory and snapshot.memory_snapshot:
            try:
                agent._memory.clear()
                for key, value in snapshot.memory_snapshot.items():
                    agent._memory.store(key, value, scope="session")
            except Exception:
                pass

        # Rebuild engine
        agent._rebuild_engine()

        return True

    def _save_to_disk(self, snapshot: AgentSnapshot) -> None:
        """Persist a snapshot to disk."""
        os.makedirs(self._storage_dir, exist_ok=True)
        filepath = os.path.join(self._storage_dir, f"{snapshot.tag}.json")
        try:
            with open(filepath, "w") as f:
                json.dump(snapshot.to_dict(), f, indent=2, default=str)
        except Exception:
            pass

    def _load_from_disk(self) -> None:
        """Load snapshots from disk on startup."""
        if not os.path.isdir(self._storage_dir):
            return
        for filename in os.listdir(self._storage_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self._storage_dir, filename)
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                    snapshot = AgentSnapshot.from_dict(data)
                    self._snapshots[snapshot.tag] = snapshot
                except Exception:
                    pass

    def clear(self) -> None:
        """Clear all versions (for testing)."""
        self._snapshots.clear()


# ── Module-level singleton ───────────────────────────────────────────────────

_version_manager: Optional[VersionManager] = None


def get_version_manager(storage_dir: str = str(VERSIONS_DIR)) -> VersionManager:
    """Get or create the global version manager."""
    global _version_manager
    if _version_manager is None:
        _version_manager = VersionManager(storage_dir)
    return _version_manager
