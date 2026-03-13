"""
aos/_internal/state_bridge.py — SDK ↔ State Machine bridge
═══════════════════════════════════════════════════════════════════════════════
Maps SDK agent states to AOS AgentCognitiveState.
Manages state transitions, checkpoints, crash recovery.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentState(Enum):
    """SDK-level agent states (maps to AOS AgentCognitiveState)."""

    IDLE = "IDLE"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    WAITING = "WAITING"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# Valid transitions
_VALID_TRANSITIONS = {
    (AgentState.IDLE, AgentState.PLANNING),
    (AgentState.PLANNING, AgentState.EXECUTING),
    (AgentState.EXECUTING, AgentState.WAITING),
    (AgentState.WAITING, AgentState.EXECUTING),
    (AgentState.WAITING, AgentState.FAILED),
    (AgentState.EXECUTING, AgentState.VERIFYING),
    (AgentState.VERIFYING, AgentState.COMPLETED),
    (AgentState.VERIFYING, AgentState.FAILED),
    (AgentState.FAILED, AgentState.PLANNING),
    (AgentState.FAILED, AgentState.IDLE),
    (AgentState.COMPLETED, AgentState.IDLE),
}


@dataclass
class StateTransitionRecord:
    from_state: str
    to_state: str
    reason: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class StateMachine:
    """
    Deterministic state machine for one agent.
    Enforces valid transitions. Records full history.
    """

    def __init__(self, agent_id: str):
        self._agent_id = agent_id
        self._state = AgentState.IDLE
        self._history: List[StateTransitionRecord] = []
        self._goal: Optional[str] = None
        self._retry_count = 0
        self._max_retries = 3
        self._created_at = time.time()

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def state_name(self) -> str:
        return self._state.value

    @property
    def history(self) -> List[StateTransitionRecord]:
        return list(self._history)

    def transition(self, to_state: AgentState, reason: str = "") -> bool:
        """
        Attempt a state transition. Returns True if successful.
        Invalid transitions are rejected (never crashes).
        """
        if (self._state, to_state) in _VALID_TRANSITIONS:
            record = StateTransitionRecord(
                from_state=self._state.value,
                to_state=to_state.value,
                reason=reason,
            )
            self._history.append(record)
            self._state = to_state
            return True

        # Auto-recovery: if stuck, reset to IDLE
        if self._state in (AgentState.COMPLETED, AgentState.FAILED):
            self._state = AgentState.IDLE
            self._history.append(
                StateTransitionRecord(
                    from_state=self._state.value,
                    to_state="IDLE",
                    reason="auto-reset",
                )
            )
            return self.transition(to_state, reason)

        return False

    def set_goal(self, goal: str) -> None:
        self._goal = goal

    def reset(self) -> None:
        """Force reset to IDLE."""
        self._history.append(
            StateTransitionRecord(
                from_state=self._state.value,
                to_state="IDLE",
                reason="forced_reset",
            )
        )
        self._state = AgentState.IDLE
        self._goal = None
        self._retry_count = 0

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._created_at
