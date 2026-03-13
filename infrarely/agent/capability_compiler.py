"""
agent/capability_compiler.py  — Capability Compiler + PlanCache
═══════════════════════════════════════════════════════════════════════════════
Validates and compiles a Capability into an executable CapabilityPlan.

Compilation checks
───────────────────
  1. Every step's tool_name exists in the ToolRegistry
  2. No circular dependency loops in param references
  3. Total step count ≤ MAX_EXECUTION_DEPTH
  4. Deterministic step ordering is enforced

PlanCache
──────────
  Compiled plans are cached by (capability_name, frozen_params).
  Cache is in-memory, bounded, and automatically invalidated on
  registry changes.  Hit = skip recompilation = ~0 overhead.
"""

from __future__ import annotations
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from infrarely.agent.capability import (
    Capability,
    CapabilityPlan,
    CapabilityStep,
    FailurePolicy,
)
from infrarely.agent.state import TaskState, ResponseType
from infrarely.tools.registry import ToolRegistry
from infrarely.observability import logger
import infrarely.core.app_config as config


# ─── Compilation errors ───────────────────────────────────────────────────────
class CompilationError(Exception):
    """Raised when a capability fails compilation."""

    def __init__(self, capability_name: str, errors: List[str]):
        self.capability_name = capability_name
        self.errors = errors
        super().__init__(
            f"Compilation failed for '{capability_name}': {'; '.join(errors)}"
        )


# ─── Compiled plan metadata ───────────────────────────────────────────────────
@dataclass
class CompiledPlanInfo:
    """Metadata about a compiled plan — for diagnostics and cache."""

    capability_name: str
    step_count: int
    tools_used: List[str]
    has_dependencies: bool = False
    has_conditions: bool = False
    estimated_tokens: int = 0  # rough estimate
    warnings: List[str] = field(default_factory=list)


# ─── Capability Compiler ──────────────────────────────────────────────────────
class CapabilityCompiler:
    """
    Validates and compiles Capability definitions.

    compile() performs:
      1. Tool existence check — each step.tool_name must be in registry
      2. Dependency cycle detection — {step_name.field} refs form a DAG
      3. Depth guard — total steps ≤ MAX_EXECUTION_DEPTH
      4. Deterministic ordering — deterministic tools before generative ones

    Returns a CapabilityPlan ready for the executor.
    """

    def __init__(self, registry: ToolRegistry):
        self._registry = registry
        self._max_depth = getattr(config, "MAX_EXECUTION_DEPTH", 8)

    def compile(
        self,
        capability: Capability,
        task_state: TaskState,
        extra_params: Dict[str, Any] = None,
    ) -> Tuple[CapabilityPlan, CompiledPlanInfo]:
        """
        Compile a capability into an executable plan.
        Returns (plan, info).
        Raises CompilationError on hard failures.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # ── 1. Depth guard ────────────────────────────────────────────────────
        if len(capability.steps) > self._max_depth:
            errors.append(
                f"Step count {len(capability.steps)} exceeds MAX_EXECUTION_DEPTH={self._max_depth}"
            )

        # ── 2. Tool existence ─────────────────────────────────────────────────
        tools_used: List[str] = []
        for step in capability.steps:
            if step.tool_name not in self._registry:
                errors.append(
                    f"Step '{step.name}': tool '{step.tool_name}' not in registry"
                )
            else:
                tools_used.append(step.tool_name)

        # ── 3. Dependency cycle detection ─────────────────────────────────────
        dep_graph = self._build_dependency_graph(capability.steps)
        cycle = self._detect_cycle(dep_graph)
        if cycle:
            errors.append(f"Dependency cycle detected: {' → '.join(cycle)}")

        # ── 4. Duplicate step names ───────────────────────────────────────────
        names = [s.name for s in capability.steps]
        seen: Set[str] = set()
        for n in names:
            if n in seen:
                errors.append(f"Duplicate step name: '{n}'")
            seen.add(n)

        # ── Fail fast ─────────────────────────────────────────────────────────
        if errors:
            logger.error(
                f"Compilation failed for '{capability.name}'",
                errors=errors,
            )
            raise CompilationError(capability.name, errors)

        # ── 5. Enrich steps with extra params ─────────────────────────────────
        if extra_params:
            enriched_steps = []
            for step in capability.steps:
                merged = dict(extra_params)
                merged.update(step.base_params)  # step params override
                enriched_steps.append(
                    CapabilityStep(
                        name=step.name,
                        tool_name=step.tool_name,
                        base_params=merged,
                        failure_policy=step.failure_policy,
                        condition=step.condition,
                        description=step.description,
                    )
                )
            enriched_cap = Capability(
                name=capability.name,
                description=capability.description,
                steps=enriched_steps,
                intent_tags=capability.intent_tags,
            )
        else:
            enriched_cap = capability

        # ── 6. Order: deterministic first, then generative ────────────────────
        det_steps, gen_steps = [], []
        for step in enriched_cap.steps:
            meta = self._registry.get_meta(step.tool_name)
            if meta and meta.response_type == ResponseType.TOOL_GENERATIVE:
                gen_steps.append(step)
            else:
                det_steps.append(step)

        ordered_steps = det_steps + gen_steps
        if [s.name for s in ordered_steps] != [s.name for s in enriched_cap.steps]:
            warnings.append("Steps reordered: deterministic before generative")
            enriched_cap = Capability(
                name=enriched_cap.name,
                description=enriched_cap.description,
                steps=ordered_steps,
                intent_tags=enriched_cap.intent_tags,
            )

        # ── 7. Estimate tokens ────────────────────────────────────────────────
        est_tokens = 0
        for step in enriched_cap.steps:
            meta = self._registry.get_meta(step.tool_name)
            if meta and meta.requires_llm:
                est_tokens += config.LLM_MAX_TOKENS

        # ── 8. Build plan ─────────────────────────────────────────────────────
        plan = CapabilityPlan(
            capability=enriched_cap,
            task_state=task_state,
            initial_context=dict(task_state.params),
        )

        info = CompiledPlanInfo(
            capability_name=capability.name,
            step_count=len(enriched_cap.steps),
            tools_used=tools_used,
            has_dependencies=bool(dep_graph),
            has_conditions=any(s.condition is not None for s in enriched_cap.steps),
            estimated_tokens=est_tokens,
            warnings=warnings,
        )

        logger.info(
            f"Compiled '{capability.name}'",
            steps=info.step_count,
            tools=info.tools_used,
            est_tokens=info.estimated_tokens,
            warnings=info.warnings or "none",
        )
        return plan, info

    # ── dependency helpers ────────────────────────────────────────────────────
    @staticmethod
    def _build_dependency_graph(steps: List[CapabilityStep]) -> Dict[str, Set[str]]:
        """
        Build a directed graph: step_name → set of step_names it depends on.
        Dependencies come from {step_name.field} references in base_params.
        """
        import re

        graph: Dict[str, Set[str]] = {}
        step_names = {s.name for s in steps}

        for step in steps:
            deps: Set[str] = set()
            for _key, value in step.base_params.items():
                if (
                    isinstance(value, str)
                    and value.startswith("{")
                    and value.endswith("}")
                ):
                    ref = value[1:-1]
                    root = ref.split(".")[0].split("[")[0]
                    if root in step_names:
                        deps.add(root)
            if deps:
                graph[step.name] = deps
        return graph

    @staticmethod
    def _detect_cycle(graph: Dict[str, Set[str]]) -> Optional[List[str]]:
        """DFS-based cycle detection. Returns cycle path or None."""
        WHITE, GRAY, BLACK = 0, 1, 2
        colour: Dict[str, int] = {n: WHITE for n in graph}
        # Also add nodes that are only targets
        for deps in graph.values():
            for d in deps:
                if d not in colour:
                    colour[d] = WHITE

        path: List[str] = []

        def dfs(node: str) -> bool:
            colour[node] = GRAY
            path.append(node)
            for dep in graph.get(node, set()):
                if colour.get(dep, WHITE) == GRAY:
                    path.append(dep)
                    return True
                if colour.get(dep, WHITE) == WHITE:
                    if dfs(dep):
                        return True
            path.pop()
            colour[node] = BLACK
            return False

        for node in list(colour):
            if colour[node] == WHITE:
                if dfs(node):
                    # Extract the cycle
                    cycle_start = path[-1]
                    idx = path.index(cycle_start)
                    return path[idx:]
        return None


# ─── PlanCache ─────────────────────────────────────────────────────────────────
class PlanCache:
    """
    In-memory LRU cache for compiled CapabilityPlans.

    Key: (capability_name, frozen_params_hash)
    Value: (CapabilityPlan, CompiledPlanInfo)

    Bounded to MAX_SIZE entries; eldest evicted on overflow (LRU policy).
    """

    MAX_SIZE = 100

    def __init__(self):
        self._cache: OrderedDict[str, Tuple[CapabilityPlan, CompiledPlanInfo]] = (
            OrderedDict()
        )
        self._hits = 0
        self._misses = 0

    def get(
        self,
        capability_name: str,
        params: Dict[str, Any],
        version: int = 1,
    ) -> Optional[Tuple[CapabilityPlan, CompiledPlanInfo]]:
        key = self._make_key(capability_name, params, version)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            logger.debug(f"PlanCache HIT for '{capability_name}' v{version}")
            return self._cache[key]
        self._misses += 1
        return None

    def put(
        self,
        capability_name: str,
        params: Dict[str, Any],
        plan: CapabilityPlan,
        info: CompiledPlanInfo,
        version: int = 1,
    ):
        key = self._make_key(capability_name, params, version)
        self._cache[key] = (plan, info)
        self._cache.move_to_end(key)
        if len(self._cache) > self.MAX_SIZE:
            self._cache.popitem(last=False)
        logger.debug(
            f"PlanCache stored '{capability_name}' v{version} (size={len(self._cache)})"
        )

    def invalidate(self, capability_name: str = None):
        """Drop entries for a specific capability, or all if name is None."""
        if capability_name is None:
            self._cache.clear()
        else:
            to_drop = [k for k in self._cache if k.startswith(capability_name + ":")]
            for k in to_drop:
                del self._cache[k]

    def stats(self) -> Dict[str, Any]:
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(1, self._hits + self._misses), 2),
        }

    @staticmethod
    def _make_key(name: str, params: Dict[str, Any], version: int = 1) -> str:
        stable = json.dumps(params, sort_keys=True, default=str)
        h = hashlib.md5(stable.encode()).hexdigest()[:12]
        return f"{name}:v{version}:{h}"
