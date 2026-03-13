"""
agent/agent_state_machine.py — CAPABILITY 1: Agent State Machine
═══════════════════════════════════════════════════════════════════════════════
Deterministic cognitive state infrastructure for the InfraRely.

PROBLEM SOLVED:
  Most agent frameworks are stateless request-response loops.
  Agents forget goals. Agents cannot be interrupted and resumed.
  Long-running tasks collapse on first failure.

THIS IMPLEMENTATION:
  • Full state schema with persistent cognitive state
  • Deterministic state transitions with guard conditions
  • Checkpoint/restore protocol for crash recovery
  • StateStore backend (SQLite for single-node, in-memory LRU for active agents)
  • Namespace isolation for parallel agent execution

State Enum:
  IDLE → PLANNING → EXECUTING → WAITING → VERIFYING → COMPLETED
                                                     → FAILED → PLANNING (if retry budget)

RULES (from AOS spec):
  RULE 2 — STATE IS TRUTH
    No agent executes without a registered state.
    No state transitions without a validated guard condition.
    No execution is unrecoverable — every state can checkpoint and restore.
"""

from __future__ import annotations

import copy
import json
import os
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from infrarely.observability import logger


# ═══════════════════════════════════════════════════════════════════════════════
# STATE ENUM — Deterministic cognitive states
# ═══════════════════════════════════════════════════════════════════════════════


class AgentCognitiveState(Enum):
    """
    Every agent MUST be in exactly one of these states at all times.
    Transitions between states are deterministic and guarded.
    """

    IDLE = "idle"  # No active goal; waiting for input
    PLANNING = "planning"  # Goal received; generating/compiling plan
    EXECUTING = "executing"  # Plan compiled; executing graph nodes
    WAITING = "waiting"  # Blocked on external dependency/response
    VERIFYING = "verifying"  # All nodes done; running verification gates
    COMPLETED = "completed"  # All verification passed; task done
    FAILED = "failed"  # Unrecoverable failure or retry budget exceeded


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSITION RULES — All deterministic, no LLM
# ═══════════════════════════════════════════════════════════════════════════════

# Valid transitions: (from_state, to_state) → description
VALID_TRANSITIONS: Dict[Tuple[AgentCognitiveState, AgentCognitiveState], str] = {
    (
        AgentCognitiveState.IDLE,
        AgentCognitiveState.PLANNING,
    ): "goal received + validated",
    (
        AgentCognitiveState.PLANNING,
        AgentCognitiveState.EXECUTING,
    ): "plan compiled + all guards pass",
    (
        AgentCognitiveState.EXECUTING,
        AgentCognitiveState.WAITING,
    ): "tool requires external response",
    (
        AgentCognitiveState.WAITING,
        AgentCognitiveState.EXECUTING,
    ): "dependency resolved within timeout",
    (AgentCognitiveState.WAITING, AgentCognitiveState.FAILED): "timeout exceeded",
    (
        AgentCognitiveState.EXECUTING,
        AgentCognitiveState.VERIFYING,
    ): "all graph nodes reach SUCCESS or SKIPPED",
    (
        AgentCognitiveState.VERIFYING,
        AgentCognitiveState.COMPLETED,
    ): "all verification gates pass",
    (
        AgentCognitiveState.VERIFYING,
        AgentCognitiveState.FAILED,
    ): "verification gate failed",
    (
        AgentCognitiveState.FAILED,
        AgentCognitiveState.PLANNING,
    ): "retry_count < max_retries AND failure is recoverable",
    (
        AgentCognitiveState.FAILED,
        AgentCognitiveState.COMPLETED,
    ): "escalation path (human or fallback)",
    # Reset transitions (administrative)
    (
        AgentCognitiveState.COMPLETED,
        AgentCognitiveState.IDLE,
    ): "task completed, returning to idle",
    (
        AgentCognitiveState.FAILED,
        AgentCognitiveState.IDLE,
    ): "failure acknowledged, returning to idle",
}


def is_valid_transition(
    from_state: AgentCognitiveState, to_state: AgentCognitiveState
) -> bool:
    """Check if a state transition is valid according to the transition rules."""
    return (from_state, to_state) in VALID_TRANSITIONS


# ═══════════════════════════════════════════════════════════════════════════════
# GUARD CONDITIONS — must ALL be true before a transition fires
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class GuardCondition:
    """A single guard condition that must be satisfied for a transition."""

    name: str
    check_fn: Callable[["AgentStateRecord"], bool]
    error_message: str = ""

    def evaluate(self, state: "AgentStateRecord") -> Tuple[bool, str]:
        """Evaluate this guard. Returns (passed, error_message)."""
        try:
            result = self.check_fn(state)
            return result, (
                "" if result else self.error_message or f"Guard '{self.name}' failed"
            )
        except Exception as e:
            return False, f"Guard '{self.name}' raised exception: {e}"


@dataclass
class GuardResult:
    """Result of evaluating all guards for a transition."""

    passed: bool
    failed_guards: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class TransitionGuards:
    """
    Registry of guard conditions per transition.
    All guards must pass before a transition can fire.
    """

    def __init__(self):
        self._guards: Dict[
            Tuple[AgentCognitiveState, AgentCognitiveState], List[GuardCondition]
        ] = {}
        self._register_default_guards()

    def _register_default_guards(self):
        """Register the default guard conditions from the AOS spec."""

        # IDLE → PLANNING: goal must be set and valid
        self.register(
            AgentCognitiveState.IDLE,
            AgentCognitiveState.PLANNING,
            GuardCondition(
                name="goal_present",
                check_fn=lambda s: s.goal is not None and bool(s.goal),
                error_message="Goal must be set before transitioning to PLANNING",
            ),
        )

        # PLANNING → EXECUTING: plan must be compiled
        self.register(
            AgentCognitiveState.PLANNING,
            AgentCognitiveState.EXECUTING,
            GuardCondition(
                name="plan_compiled",
                check_fn=lambda s: s.plan is not None and bool(s.plan),
                error_message="Plan must be compiled before transitioning to EXECUTING",
            ),
        )

        # EXECUTING → VERIFYING: execution must have results
        self.register(
            AgentCognitiveState.EXECUTING,
            AgentCognitiveState.VERIFYING,
            GuardCondition(
                name="execution_has_results",
                check_fn=lambda s: s.execution_cursor >= 0,
                error_message="Execution must have produced results before VERIFYING",
            ),
        )

        # FAILED → PLANNING: retry budget must remain
        self.register(
            AgentCognitiveState.FAILED,
            AgentCognitiveState.PLANNING,
            GuardCondition(
                name="retry_budget_remaining",
                check_fn=lambda s: s.retry_count < s.max_retries,
                error_message="Retry budget exceeded — cannot re-plan",
            ),
        )
        self.register(
            AgentCognitiveState.FAILED,
            AgentCognitiveState.PLANNING,
            GuardCondition(
                name="failure_is_recoverable",
                check_fn=lambda s: s.failure_recoverable,
                error_message="Failure is not recoverable — cannot re-plan",
            ),
        )

        # Resource budget guard (applies to all transitions into EXECUTING)
        self.register(
            AgentCognitiveState.PLANNING,
            AgentCognitiveState.EXECUTING,
            GuardCondition(
                name="resource_budget_ok",
                check_fn=lambda s: s.token_budget_remaining > 0,
                error_message="Token budget exceeded — cannot execute",
            ),
        )

        # Waiting → Executing: dependency must be resolved
        self.register(
            AgentCognitiveState.WAITING,
            AgentCognitiveState.EXECUTING,
            GuardCondition(
                name="dependency_resolved",
                check_fn=lambda s: (
                    all(s.dependencies_met.values()) if s.dependencies_met else True
                ),
                error_message="Not all dependencies are resolved",
            ),
        )

    def register(
        self,
        from_state: AgentCognitiveState,
        to_state: AgentCognitiveState,
        guard: GuardCondition,
    ):
        """Register a guard condition for a specific transition."""
        key = (from_state, to_state)
        if key not in self._guards:
            self._guards[key] = []
        self._guards[key].append(guard)

    def evaluate(
        self,
        from_state: AgentCognitiveState,
        to_state: AgentCognitiveState,
        state: "AgentStateRecord",
    ) -> GuardResult:
        """Evaluate all guard conditions for a transition."""
        key = (from_state, to_state)
        guards = self._guards.get(key, [])

        failed = []
        errors = []
        for guard in guards:
            passed, error = guard.evaluate(state)
            if not passed:
                failed.append(guard.name)
                errors.append(error)

        return GuardResult(
            passed=len(failed) == 0,
            failed_guards=failed,
            errors=errors,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT STATE RECORD — The complete cognitive state of a single agent
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class AgentStateRecord:
    """
    Complete cognitive state for a single agent instance.

    This is THE source of truth for the agent's current state.
    Every field is required, every field is serializable, every
    field survives checkpoint/restore.
    """

    # Identity
    agent_id: str = field(default_factory=lambda: f"agent_{uuid.uuid4().hex[:8]}")

    # Cognitive state
    current_state: AgentCognitiveState = AgentCognitiveState.IDLE

    # Goal
    goal: Optional[Dict[str, Any]] = None  # structured goal object

    # Plan (compiled execution graph, serialized)
    plan: Optional[Dict[str, Any]] = None  # compiled plan (serializable)

    # Execution cursor
    execution_cursor: int = -1  # current step index (-1 = not started)

    # Memory references
    memory_refs: Dict[str, str] = field(
        default_factory=dict
    )  # pointers to knowledge layer

    # Dependencies
    dependencies_met: Dict[str, bool] = field(default_factory=dict)  # resolved deps

    # Retry tracking
    retry_count: int = 0
    max_retries: int = 3

    # Failure state
    failure_recoverable: bool = True
    last_failure_reason: str = ""

    # Resource tracking
    token_budget_remaining: int = 5000
    token_budget_used: int = 0
    execution_depth: int = 0
    max_execution_depth: int = 8

    # Checkpoint
    checkpoint: Optional[Dict[str, Any]] = None  # full serialized snapshot

    # Timestamps
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_updated: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Session tracking
    session_id: str = ""
    namespace: str = "default"  # namespace isolation

    # Transition history (audit trail)
    transition_history: List[Dict[str, Any]] = field(default_factory=list)

    # Execution results (accumulated during EXECUTING)
    step_results: Dict[str, Any] = field(default_factory=dict)

    # Verification results
    verification_results: Dict[str, Any] = field(default_factory=dict)

    # Waiting state
    waiting_for: Optional[str] = None  # what we're blocked on
    wait_timeout_ms: int = 30_000  # default 30s timeout
    wait_started_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for storage."""
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, AgentCognitiveState):
                d[k] = v.value
            elif isinstance(v, list):
                d[k] = copy.deepcopy(v)
            elif isinstance(v, dict):
                d[k] = copy.deepcopy(v)
            else:
                d[k] = v
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AgentStateRecord":
        """Deserialize from dict."""
        d = copy.deepcopy(d)
        if "current_state" in d and isinstance(d["current_state"], str):
            d["current_state"] = AgentCognitiveState(d["current_state"])
        return AgentStateRecord(
            **{k: v for k, v in d.items() if k in AgentStateRecord.__dataclass_fields__}
        )

    def make_checkpoint(self) -> Dict[str, Any]:
        """Create a full checkpoint snapshot."""
        snapshot = self.to_dict()
        snapshot["_checkpoint_ts"] = datetime.now(timezone.utc).isoformat()
        snapshot["_checkpoint_id"] = f"ckpt_{uuid.uuid4().hex[:8]}"
        self.checkpoint = snapshot
        return snapshot


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSITION EVENT — Immutable record of a state transition
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TransitionEvent:
    """Immutable record of a state transition for audit logging."""

    agent_id: str
    from_state: AgentCognitiveState
    to_state: AgentCognitiveState
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    guards_evaluated: List[str] = field(default_factory=list)
    guards_passed: bool = True
    reason: str = ""
    checkpoint_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "timestamp": self.timestamp,
            "guards_evaluated": self.guards_evaluated,
            "guards_passed": self.guards_passed,
            "reason": self.reason,
            "checkpoint_id": self.checkpoint_id,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT STATE MACHINE — Deterministic transition engine
# ═══════════════════════════════════════════════════════════════════════════════


class AgentStateMachine:
    """
    Deterministic state machine for a single agent.

    Every transition:
      1. Validates the (from, to) pair against VALID_TRANSITIONS
      2. Evaluates ALL guard conditions for that transition
      3. Takes a checkpoint snapshot BEFORE the transition
      4. Fires the transition atomically
      5. Records the transition event in the audit trail
      6. Persists to StateStore

    NO LLM is ever invoked in this module.
    """

    def __init__(
        self,
        state: AgentStateRecord,
        guards: Optional[TransitionGuards] = None,
        store: Optional["StateStore"] = None,
    ):
        self._state = state
        self._guards = guards or TransitionGuards()
        self._store = store
        self._lock = threading.Lock()

    @property
    def state(self) -> AgentStateRecord:
        return self._state

    @property
    def current_state(self) -> AgentCognitiveState:
        return self._state.current_state

    @property
    def agent_id(self) -> str:
        return self._state.agent_id

    def transition(
        self,
        to_state: AgentCognitiveState,
        reason: str = "",
    ) -> TransitionEvent:
        """
        Attempt a state transition. Fully deterministic.

        Returns TransitionEvent on success.
        Raises InvalidTransitionError on failure.
        """
        with self._lock:
            from_state = self._state.current_state

            # ── 1. Validate transition is structurally valid ──────────────
            if not is_valid_transition(from_state, to_state):
                raise InvalidTransitionError(
                    self._state.agent_id,
                    from_state,
                    to_state,
                    f"Transition {from_state.value} → {to_state.value} is not valid",
                )

            # ── 2. Evaluate guard conditions ──────────────────────────────
            guard_result = self._guards.evaluate(from_state, to_state, self._state)
            if not guard_result.passed:
                raise GuardConditionError(
                    self._state.agent_id,
                    from_state,
                    to_state,
                    guard_result.failed_guards,
                    guard_result.errors,
                )

            # ── 3. Checkpoint BEFORE transition ───────────────────────────
            checkpoint = self._state.make_checkpoint()
            checkpoint_id = checkpoint.get("_checkpoint_id", "")

            # ── 4. Fire the transition ────────────────────────────────────
            self._state.current_state = to_state
            self._state.last_updated = datetime.now(timezone.utc).isoformat()

            # ── 5. Record transition event ────────────────────────────────
            event = TransitionEvent(
                agent_id=self._state.agent_id,
                from_state=from_state,
                to_state=to_state,
                guards_evaluated=(
                    [g for g in guard_result.failed_guards]
                    if not guard_result.passed
                    else []
                ),
                guards_passed=True,
                reason=reason or VALID_TRANSITIONS.get((from_state, to_state), ""),
                checkpoint_id=checkpoint_id,
            )

            self._state.transition_history.append(event.to_dict())

            # Trim history to last 100 events
            if len(self._state.transition_history) > 100:
                self._state.transition_history = self._state.transition_history[-100:]

            # ── 6. Persist to StateStore ──────────────────────────────────
            if self._store:
                self._store.save(self._state)

            logger.info(
                f"StateMachine transition: {from_state.value} → {to_state.value}",
                agent=self._state.agent_id,
                reason=event.reason,
            )

            return event

    def set_goal(self, goal: Dict[str, Any]) -> None:
        """Set the agent's goal. Only valid in IDLE state."""
        if self._state.current_state != AgentCognitiveState.IDLE:
            raise InvalidStateError(
                self._state.agent_id,
                f"Cannot set goal in state {self._state.current_state.value} — must be IDLE",
            )
        self._state.goal = goal
        self._state.last_updated = datetime.now(timezone.utc).isoformat()
        if self._store:
            self._store.save(self._state)

    def set_plan(self, plan: Dict[str, Any]) -> None:
        """Set the compiled plan. Only valid in PLANNING state."""
        if self._state.current_state != AgentCognitiveState.PLANNING:
            raise InvalidStateError(
                self._state.agent_id,
                f"Cannot set plan in state {self._state.current_state.value} — must be PLANNING",
            )
        self._state.plan = plan
        self._state.last_updated = datetime.now(timezone.utc).isoformat()
        if self._store:
            self._store.save(self._state)

    def advance_cursor(self) -> int:
        """Advance the execution cursor. Only valid in EXECUTING state."""
        if self._state.current_state != AgentCognitiveState.EXECUTING:
            raise InvalidStateError(
                self._state.agent_id,
                f"Cannot advance cursor in state {self._state.current_state.value}",
            )
        self._state.execution_cursor += 1
        self._state.last_updated = datetime.now(timezone.utc).isoformat()
        if self._store:
            self._store.save(self._state)
        return self._state.execution_cursor

    def record_step_result(self, step_name: str, result: Dict[str, Any]) -> None:
        """Record a step execution result."""
        self._state.step_results[step_name] = result
        self._state.last_updated = datetime.now(timezone.utc).isoformat()
        if self._store:
            self._store.save(self._state)

    def record_failure(self, reason: str, recoverable: bool = True) -> None:
        """Record a failure with recoverability flag."""
        self._state.last_failure_reason = reason
        self._state.failure_recoverable = recoverable
        self._state.retry_count += 1
        self._state.last_updated = datetime.now(timezone.utc).isoformat()

    def enter_wait(self, waiting_for: str, timeout_ms: int = 30_000) -> None:
        """Enter WAITING state for an external dependency."""
        self._state.waiting_for = waiting_for
        self._state.wait_timeout_ms = timeout_ms
        self._state.wait_started_at = time.time()
        self.transition(
            AgentCognitiveState.WAITING, reason=f"waiting for: {waiting_for}"
        )

    def check_wait_timeout(self) -> bool:
        """Check if the wait has timed out. Returns True if timed out."""
        if self._state.wait_started_at is None:
            return False
        elapsed_ms = (time.time() - self._state.wait_started_at) * 1000
        return elapsed_ms > self._state.wait_timeout_ms

    def resolve_dependency(self, dep_name: str, value: Any = True) -> None:
        """Mark a dependency as resolved."""
        self._state.dependencies_met[dep_name] = bool(value)
        self._state.last_updated = datetime.now(timezone.utc).isoformat()

    def consume_tokens(self, count: int) -> bool:
        """Consume tokens from the budget. Returns False if budget exceeded."""
        if count > self._state.token_budget_remaining:
            return False
        self._state.token_budget_remaining -= count
        self._state.token_budget_used += count
        return True

    def restore_from_checkpoint(self) -> bool:
        """
        Restore agent state from the last checkpoint.
        Returns True if restore succeeded, False if no checkpoint available.
        """
        if self._state.checkpoint is None:
            # Try loading from store
            if self._store:
                loaded = self._store.load(self._state.agent_id)
                if loaded and loaded.checkpoint:
                    self._state = AgentStateRecord.from_dict(loaded.checkpoint)
                    logger.info(
                        f"Restored from store checkpoint",
                        agent=self._state.agent_id,
                    )
                    return True
            return False

        restored = AgentStateRecord.from_dict(self._state.checkpoint)
        self._state = restored
        logger.info(
            f"Restored from in-memory checkpoint",
            agent=self._state.agent_id,
            state=self._state.current_state.value,
        )
        return True

    def reset(self) -> None:
        """Full reset to IDLE with clean state."""
        self._state.current_state = AgentCognitiveState.IDLE
        self._state.goal = None
        self._state.plan = None
        self._state.execution_cursor = -1
        self._state.retry_count = 0
        self._state.step_results = {}
        self._state.verification_results = {}
        self._state.waiting_for = None
        self._state.wait_started_at = None
        self._state.failure_recoverable = True
        self._state.last_failure_reason = ""
        self._state.last_updated = datetime.now(timezone.utc).isoformat()
        if self._store:
            self._store.save(self._state)

    def snapshot(self) -> Dict[str, Any]:
        """Return a read-only snapshot of the current state."""
        return {
            "agent_id": self._state.agent_id,
            "current_state": self._state.current_state.value,
            "goal": self._state.goal,
            "execution_cursor": self._state.execution_cursor,
            "retry_count": self._state.retry_count,
            "max_retries": self._state.max_retries,
            "token_budget_remaining": self._state.token_budget_remaining,
            "token_budget_used": self._state.token_budget_used,
            "transition_count": len(self._state.transition_history),
            "step_results_count": len(self._state.step_results),
            "created_at": self._state.created_at,
            "last_updated": self._state.last_updated,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# STATE STORE — SQLite-backed persistent storage
# ═══════════════════════════════════════════════════════════════════════════════


class StateStore:
    """
    Persistent state storage for agent state records.

    Backends:
      - SQLite for single-node (fits in 8GB RAM, zero overhead)
      - In-memory LRU cache for active agents (500 agent cap ~ 200MB)

    Every state transition triggers a save.
    On system restart, agents restore from last valid checkpoint.
    """

    def __init__(self, db_path: Optional[str] = None, max_cache_size: int = 500):
        self._db_path = db_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data", "state_store.db"
        )
        self._cache: OrderedDict[str, AgentStateRecord] = OrderedDict()
        self._max_cache = max_cache_size
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database with the state table."""
        os.makedirs(os.path.dirname(os.path.abspath(self._db_path)), exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_states (
                    agent_id       TEXT PRIMARY KEY,
                    namespace      TEXT NOT NULL DEFAULT 'default',
                    current_state  TEXT NOT NULL,
                    state_data     TEXT NOT NULL,
                    checkpoint     TEXT,
                    created_at     TEXT NOT NULL,
                    last_updated   TEXT NOT NULL
                )
            """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transition_log (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id      TEXT NOT NULL,
                    from_state    TEXT NOT NULL,
                    to_state      TEXT NOT NULL,
                    reason        TEXT,
                    checkpoint_id TEXT,
                    timestamp     TEXT NOT NULL
                )
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_agent_namespace
                ON agent_states (namespace)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_transition_agent
                ON transition_log (agent_id)
            """
            )
            conn.commit()

    def save(self, state: AgentStateRecord) -> None:
        """Save agent state to both cache and SQLite."""
        with self._lock:
            # Update LRU cache
            self._cache[state.agent_id] = state
            self._cache.move_to_end(state.agent_id)

            # Evict oldest if cache full
            while len(self._cache) > self._max_cache:
                self._cache.popitem(last=False)

            # Persist to SQLite
            try:
                state_data = json.dumps(state.to_dict(), default=str)
                checkpoint_data = (
                    json.dumps(state.checkpoint, default=str)
                    if state.checkpoint
                    else None
                )

                with sqlite3.connect(self._db_path) as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO agent_states
                            (agent_id, namespace, current_state, state_data, checkpoint, created_at, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            state.agent_id,
                            state.namespace,
                            state.current_state.value,
                            state_data,
                            checkpoint_data,
                            state.created_at,
                            state.last_updated,
                        ),
                    )
                    conn.commit()
            except Exception as e:
                logger.error(f"StateStore.save failed: {e}", agent=state.agent_id)

    def load(self, agent_id: str) -> Optional[AgentStateRecord]:
        """Load agent state from cache or SQLite."""
        with self._lock:
            # Check cache first
            if agent_id in self._cache:
                self._cache.move_to_end(agent_id)
                return self._cache[agent_id]

            # Fall through to SQLite
            try:
                with sqlite3.connect(self._db_path) as conn:
                    row = conn.execute(
                        "SELECT state_data FROM agent_states WHERE agent_id = ?",
                        (agent_id,),
                    ).fetchone()
                    if row:
                        data = json.loads(row[0])
                        state = AgentStateRecord.from_dict(data)
                        self._cache[agent_id] = state
                        return state
            except Exception as e:
                logger.error(f"StateStore.load failed: {e}", agent=agent_id)
            return None

    def load_by_namespace(self, namespace: str) -> List[AgentStateRecord]:
        """Load all agents in a namespace."""
        results = []
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT state_data FROM agent_states WHERE namespace = ?",
                    (namespace,),
                ).fetchall()
                for row in rows:
                    data = json.loads(row[0])
                    results.append(AgentStateRecord.from_dict(data))
        except Exception as e:
            logger.error(f"StateStore.load_by_namespace failed: {e}")
        return results

    def delete(self, agent_id: str) -> bool:
        """Delete an agent's state from store."""
        with self._lock:
            self._cache.pop(agent_id, None)
            try:
                with sqlite3.connect(self._db_path) as conn:
                    conn.execute(
                        "DELETE FROM agent_states WHERE agent_id = ?", (agent_id,)
                    )
                    conn.commit()
                return True
            except Exception as e:
                logger.error(f"StateStore.delete failed: {e}", agent=agent_id)
                return False

    def log_transition(self, event: TransitionEvent) -> None:
        """Log a transition event to the audit table."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO transition_log
                        (agent_id, from_state, to_state, reason, checkpoint_id, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        event.agent_id,
                        event.from_state.value,
                        event.to_state.value,
                        event.reason,
                        event.checkpoint_id,
                        event.timestamp,
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"StateStore.log_transition failed: {e}")

    def get_transition_history(
        self, agent_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recent transition history for an agent."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    """
                    SELECT from_state, to_state, reason, checkpoint_id, timestamp
                    FROM transition_log
                    WHERE agent_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                """,
                    (agent_id, limit),
                ).fetchall()
                return [
                    {
                        "from_state": r[0],
                        "to_state": r[1],
                        "reason": r[2],
                        "checkpoint_id": r[3],
                        "timestamp": r[4],
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"StateStore.get_transition_history failed: {e}")
            return []

    def get_all_agents(self) -> List[Dict[str, str]]:
        """List all registered agents with their current state."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT agent_id, namespace, current_state, last_updated FROM agent_states"
                ).fetchall()
                return [
                    {
                        "agent_id": r[0],
                        "namespace": r[1],
                        "current_state": r[2],
                        "last_updated": r[3],
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"StateStore.get_all_agents failed: {e}")
            return []

    def agent_count(self) -> int:
        """Return total number of registered agents."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute("SELECT COUNT(*) FROM agent_states").fetchone()
                return row[0] if row else 0
        except Exception:
            return len(self._cache)

    def cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return {
            "cache_size": len(self._cache),
            "max_cache_size": self._max_cache,
            "cache_hit_agents": list(self._cache.keys())[:10],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════════


class InvalidTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    def __init__(
        self,
        agent_id: str,
        from_state: AgentCognitiveState,
        to_state: AgentCognitiveState,
        message: str = "",
    ):
        self.agent_id = agent_id
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(
            f"[{agent_id}] Invalid transition: {from_state.value} → {to_state.value}"
            + (f" — {message}" if message else "")
        )


class GuardConditionError(Exception):
    """Raised when guard conditions prevent a transition."""

    def __init__(
        self,
        agent_id: str,
        from_state: AgentCognitiveState,
        to_state: AgentCognitiveState,
        failed_guards: List[str],
        errors: List[str],
    ):
        self.agent_id = agent_id
        self.from_state = from_state
        self.to_state = to_state
        self.failed_guards = failed_guards
        self.errors = errors
        super().__init__(
            f"[{agent_id}] Guards failed for {from_state.value} → {to_state.value}: "
            + "; ".join(errors)
        )


class InvalidStateError(Exception):
    """Raised when an operation is invalid for the current state."""

    def __init__(self, agent_id: str, message: str):
        self.agent_id = agent_id
        super().__init__(f"[{agent_id}] {message}")


# ═══════════════════════════════════════════════════════════════════════════════
# STATE MACHINE MANAGER — Factory + lifecycle for multiple agents
# ═══════════════════════════════════════════════════════════════════════════════


class StateMachineManager:
    """
    Manages multiple agent state machines with namespace isolation.

    Provides:
      • Agent creation with automatic StateStore registration
      • Agent lookup and restore from checkpoint
      • Namespace-scoped agent listing
      • Parallel execution without state collision
    """

    def __init__(self, store: Optional[StateStore] = None):
        self._store = store or StateStore()
        self._guards = TransitionGuards()
        self._machines: Dict[str, AgentStateMachine] = {}
        self._lock = threading.Lock()

    def create_agent(
        self,
        agent_id: Optional[str] = None,
        namespace: str = "default",
        session_id: str = "",
        max_retries: int = 3,
        token_budget: int = 5000,
    ) -> AgentStateMachine:
        """
        Create a new agent with a registered state.
        RULE 2: No agent executes without a registered state in StateStore.
        """
        with self._lock:
            state = AgentStateRecord(
                agent_id=agent_id or f"agent_{uuid.uuid4().hex[:8]}",
                namespace=namespace,
                session_id=session_id,
                max_retries=max_retries,
                token_budget_remaining=token_budget,
            )

            machine = AgentStateMachine(state, self._guards, self._store)
            self._store.save(state)
            self._machines[state.agent_id] = machine

            logger.info(
                f"StateMachineManager: created agent",
                agent_id=state.agent_id,
                namespace=namespace,
            )
            return machine

    def get_agent(self, agent_id: str) -> Optional[AgentStateMachine]:
        """Get an agent's state machine. Restores from store if needed."""
        with self._lock:
            if agent_id in self._machines:
                return self._machines[agent_id]

            # Try restoring from StateStore
            state = self._store.load(agent_id)
            if state:
                machine = AgentStateMachine(state, self._guards, self._store)
                self._machines[agent_id] = machine
                logger.info(
                    f"StateMachineManager: restored agent from store",
                    agent_id=agent_id,
                    state=state.current_state.value,
                )
                return machine
            return None

    def list_agents(self, namespace: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all agents, optionally filtered by namespace."""
        agents = self._store.get_all_agents()
        if namespace:
            agents = [a for a in agents if a.get("namespace") == namespace]
        return agents

    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent and its state."""
        with self._lock:
            self._machines.pop(agent_id, None)
            return self._store.delete(agent_id)

    def snapshot(self) -> Dict[str, Any]:
        """Return a summary of all managed agents."""
        return {
            "active_machines": len(self._machines),
            "store_total": self._store.agent_count(),
            "cache_stats": self._store.cache_stats(),
            "agents": [
                {
                    "agent_id": m.agent_id,
                    "state": m.current_state.value,
                }
                for m in self._machines.values()
            ][:20],
        }
