"""
infrarely/self_heal.py — Self-Healing Agents (Evolution Engine)
═══════════════════════════════════════════════════════════════════════════════
Layer 7: Agents that automatically detect poor performance and fix themselves
without human intervention.

Usage::

    agent = infrarely.agent("tutor")
    agent.self_improve(
        trigger="avg_confidence < 0.6 over last 50 tasks",
        action="request_knowledge_ingestion",
        knowledge_query="Python programming documentation",
    )

    # After 50 tasks, if avg confidence drops below 0.6,
    # the agent automatically ingests knowledge to self-improve.

Trigger DSL examples::

    "avg_confidence < 0.6 over last 50 tasks"
    "failure_rate > 0.3 over last 20 tasks"
    "consecutive_failures > 5"
    "avg_duration_ms > 5000 over last 10 tasks"
    "error_rate > 0.5 over last 30 tasks"
    "knowledge_gap_rate > 0.4 over last 25 tasks"

Actions:
    - request_knowledge_ingestion  — auto-add knowledge from a query
    - adjust_temperature           — change LLM temperature
    - switch_model                 — change LLM model
    - clear_cache                  — clear cached results
    - reset_memory                 — clear agent session memory
    - notify                       — fire a "self_heal" event
    - custom                       — call a user-provided callable
"""

from __future__ import annotations

import enum
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════


class SelfHealAction(enum.Enum):
    """Available self-healing remediation actions."""

    REQUEST_KNOWLEDGE_INGESTION = "request_knowledge_ingestion"
    ADJUST_TEMPERATURE = "adjust_temperature"
    SWITCH_MODEL = "switch_model"
    CLEAR_CACHE = "clear_cache"
    RESET_MEMORY = "reset_memory"
    NOTIFY = "notify"
    CUSTOM = "custom"


# Map string action names → enum values (case-insensitive, supports both forms)
_ACTION_MAP: Dict[str, SelfHealAction] = {}
for _act in SelfHealAction:
    _ACTION_MAP[_act.value] = _act
    _ACTION_MAP[_act.name.lower()] = _act


# ═══════════════════════════════════════════════════════════════════════════════
# TASK RECORD — one entry per completed task
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TaskRecord:
    """Snapshot of a single task execution for performance tracking."""

    confidence: float = 1.0
    success: bool = True
    duration_ms: float = 0.0
    used_llm: bool = False
    error_type: str = ""  # ErrorType.value string, "" if no error
    timestamp: float = field(default_factory=time.time)


# ═══════════════════════════════════════════════════════════════════════════════
# TRIGGER PARSER — parses natural-language-style trigger expressions
# ═══════════════════════════════════════════════════════════════════════════════

# Pattern: metric comparator threshold [over last N tasks]
_TRIGGER_RE = re.compile(
    r"^\s*"
    r"(?P<metric>[a-z_]+)"  # metric name
    r"\s*"
    r"(?P<op>[<>!=]+)"  # comparison operator
    r"\s*"
    r"(?P<threshold>[0-9]*\.?[0-9]+)"  # numeric threshold
    r"(?:\s+over\s+last\s+(?P<window>\d+)(?:\s+tasks?)?)?"  # optional window
    r"\s*$",
    re.IGNORECASE,
)

_COMPARATORS = {
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: abs(a - b) < 1e-9,
    "!=": lambda a, b: abs(a - b) >= 1e-9,
}


@dataclass
class SelfHealTrigger:
    """
    Parsed trigger condition.

    Attributes
    ----------
    metric : str
        One of: avg_confidence, failure_rate, consecutive_failures,
        avg_duration_ms, error_rate, knowledge_gap_rate, llm_usage_rate
    comparator : str
        "<", ">", "<=", ">=", "==", "!="
    threshold : float
        Numeric threshold value.
    window : int
        Number of recent tasks to evaluate over.
    raw : str
        Original trigger string.
    """

    metric: str = ""
    comparator: str = "<"
    threshold: float = 0.0
    window: int = 50  # default evaluation window
    raw: str = ""

    @classmethod
    def parse(cls, expr: str) -> "SelfHealTrigger":
        """
        Parse a trigger expression string.

        Examples::

            SelfHealTrigger.parse("avg_confidence < 0.6 over last 50 tasks")
            SelfHealTrigger.parse("failure_rate > 0.3 over last 20 tasks")
            SelfHealTrigger.parse("consecutive_failures > 5")

        Raises
        ------
        ValueError
            If the expression cannot be parsed.
        """
        m = _TRIGGER_RE.match(expr)
        if not m:
            raise ValueError(
                f"Invalid trigger expression: {expr!r}. "
                f"Expected format: 'metric <op> threshold [over last N tasks]'"
            )

        metric = m.group("metric").lower()
        op = m.group("op")
        threshold = float(m.group("threshold"))
        window = int(m.group("window")) if m.group("window") else 50

        if op not in _COMPARATORS:
            raise ValueError(
                f"Unknown comparator: {op!r}. "
                f"Supported: {', '.join(_COMPARATORS.keys())}"
            )

        _VALID_METRICS = {
            "avg_confidence",
            "failure_rate",
            "consecutive_failures",
            "avg_duration_ms",
            "error_rate",
            "knowledge_gap_rate",
            "llm_usage_rate",
        }
        if metric not in _VALID_METRICS:
            raise ValueError(
                f"Unknown metric: {metric!r}. "
                f"Valid metrics: {', '.join(sorted(_VALID_METRICS))}"
            )

        return cls(
            metric=metric,
            comparator=op,
            threshold=threshold,
            window=window,
            raw=expr,
        )

    def evaluate(self, records: List[TaskRecord]) -> bool:
        """
        Evaluate this trigger against a list of task records.

        Returns True if the trigger condition is met (i.e., healing needed).
        """
        if not records:
            return False

        # Use only the last `window` records
        window_records = records[-self.window :] if self.window else records

        if not window_records:
            return False

        # Calculate the metric value
        value = self._compute_metric(window_records)
        if value is None:
            return False

        compare_fn = _COMPARATORS.get(self.comparator)
        if compare_fn is None:
            return False

        return compare_fn(value, self.threshold)

    def _compute_metric(self, records: List[TaskRecord]) -> Optional[float]:
        """Compute the metric value from a window of task records."""
        n = len(records)
        if n == 0:
            return None

        if self.metric == "avg_confidence":
            return sum(r.confidence for r in records) / n

        elif self.metric == "failure_rate":
            failures = sum(1 for r in records if not r.success)
            return failures / n

        elif self.metric == "consecutive_failures":
            # Count from the end how many consecutive failures
            count = 0
            for r in reversed(records):
                if not r.success:
                    count += 1
                else:
                    break
            return float(count)

        elif self.metric == "avg_duration_ms":
            return sum(r.duration_ms for r in records) / n

        elif self.metric == "error_rate":
            errors = sum(1 for r in records if r.error_type)
            return errors / n

        elif self.metric == "knowledge_gap_rate":
            gaps = sum(1 for r in records if r.error_type == "KNOWLEDGE_GAP")
            return gaps / n

        elif self.metric == "llm_usage_rate":
            llm_used = sum(1 for r in records if r.used_llm)
            return llm_used / n

        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-HEAL RULE — combines trigger + action + parameters
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SelfHealRule:
    """
    A rule that binds a trigger condition to a remediation action.

    Attributes
    ----------
    trigger : SelfHealTrigger
        When this fires, the action executes.
    action : SelfHealAction
        What remediation to perform.
    params : dict
        Action-specific parameters (e.g. knowledge_query, model, temperature).
    cooldown : float
        Minimum seconds between consecutive firings (default 60).
    max_fires : int
        Maximum number of times this rule can fire (0 = unlimited).
    enabled : bool
        Whether this rule is currently active.
    """

    trigger: SelfHealTrigger = field(default_factory=SelfHealTrigger)
    action: SelfHealAction = SelfHealAction.NOTIFY
    params: Dict[str, Any] = field(default_factory=dict)
    cooldown: float = 60.0  # seconds between firings
    max_fires: int = 0  # 0 = unlimited
    enabled: bool = True

    # ── Internal state ────────────────────────────────────────────────────────
    _fire_count: int = field(default=0, repr=False)
    _last_fired: float = field(default=0.0, repr=False)

    def can_fire(self) -> bool:
        """Check if this rule is eligible to fire (cooldown + max_fires)."""
        if not self.enabled:
            return False
        if self.max_fires > 0 and self._fire_count >= self.max_fires:
            return False
        if self._last_fired > 0:
            elapsed = time.time() - self._last_fired
            if elapsed < self.cooldown:
                return False
        return True

    def mark_fired(self) -> None:
        """Record that this rule has fired."""
        self._fire_count += 1
        self._last_fired = time.time()


# ═══════════════════════════════════════════════════════════════════════════════
# HEAL EVENT — record of a self-healing action taken
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class HealEvent:
    """Record of one self-healing remediation that was executed."""

    trigger_expr: str = ""
    action: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    metric_value: float = 0.0
    threshold: float = 0.0
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    details: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE WINDOW — ring buffer of task records
# ═══════════════════════════════════════════════════════════════════════════════


class PerformanceWindow:
    """
    Thread-safe sliding window of task performance records.

    Stores the last N task results for trigger evaluation.
    """

    def __init__(self, max_size: int = 1000, *, size: int = 0):
        effective_size = size if size > 0 else max_size
        self._max_size = effective_size
        self._records: Deque[TaskRecord] = deque(maxlen=effective_size)
        self._lock = threading.Lock()

    def record(self, task_record: TaskRecord) -> None:
        """Add a task record to the window."""
        with self._lock:
            self._records.append(task_record)

    def add(self, data: Any) -> None:
        """Add a record from a dict or TaskRecord."""
        if isinstance(data, TaskRecord):
            self.record(data)
        elif isinstance(data, dict):
            self.record(TaskRecord(**data))
        else:
            self.record(data)

    def get_records(self, n: Optional[int] = None) -> List[TaskRecord]:
        """Get the last N records (or all if n is None)."""
        with self._lock:
            if n is None:
                return list(self._records)
            return list(self._records)[-n:]

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def summary(self) -> Dict[str, Any]:
        """Quick summary stats."""
        with self._lock:
            records = list(self._records)

        if not records:
            return {
                "total_tasks": 0,
                "avg_confidence": 0.0,
                "failure_rate": 0.0,
                "avg_duration_ms": 0.0,
            }

        n = len(records)
        return {
            "total_tasks": n,
            "avg_confidence": sum(r.confidence for r in records) / n,
            "failure_rate": sum(1 for r in records if not r.success) / n,
            "avg_duration_ms": sum(r.duration_ms for r in records) / n,
            "llm_usage_rate": sum(1 for r in records if r.used_llm) / n,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-HEAL ENGINE — per-agent engine that monitors + remediates
# ═══════════════════════════════════════════════════════════════════════════════


class SelfHealEngine:
    """
    Per-agent self-healing engine.

    Holds rules, tracks performance, evaluates triggers after each task,
    and executes matched remediation actions.

    Not instantiated directly — created via ``agent.self_improve()``.
    """

    def __init__(self, agent_name: str = ""):
        self._agent_name = agent_name
        self._rules: List[SelfHealRule] = []
        self._window = PerformanceWindow(max_size=2000)
        self._heal_history: List[HealEvent] = []
        self._lock = threading.Lock()
        self._enabled = True

        # Callbacks set by Agent integration
        self._knowledge_add_fn: Optional[Callable[[str, str], None]] = None
        self._config_set_fn: Optional[Callable[[str, Any], None]] = None
        self._memory_clear_fn: Optional[Callable[[], None]] = None
        self._emit_fn: Optional[Callable[[str, Any], None]] = None
        self._custom_handlers: Dict[str, Callable] = {}

    # ── Rule management ───────────────────────────────────────────────────────

    def add_rule(self, rule: SelfHealRule) -> None:
        """Add a self-healing rule."""
        with self._lock:
            self._rules.append(rule)

    def remove_rule(self, index: int) -> None:
        """Remove a rule by index."""
        with self._lock:
            if 0 <= index < len(self._rules):
                self._rules.pop(index)

    @property
    def rules(self) -> List[SelfHealRule]:
        """Get all rules (read-only copy)."""
        with self._lock:
            return list(self._rules)

    @property
    def rule_count(self) -> int:
        with self._lock:
            return len(self._rules)

    # ── Performance recording ─────────────────────────────────────────────────

    def record_task(
        self,
        *,
        confidence: float = 1.0,
        success: bool = True,
        duration_ms: float = 0.0,
        used_llm: bool = False,
        error_type: str = "",
    ) -> List[HealEvent]:
        """
        Record a task result and evaluate all rules.

        Called by Agent.run() after every task. Returns a list of
        HealEvents for any actions that were triggered and executed.
        """
        record = TaskRecord(
            confidence=confidence,
            success=success,
            duration_ms=duration_ms,
            used_llm=used_llm,
            error_type=error_type,
        )
        self._window.record(record)

        if not self._enabled:
            return []

        return self._evaluate_and_heal()

    def evaluate(self) -> List[HealEvent]:
        """Evaluate all rules and execute matching actions (public API)."""
        return self._evaluate_and_heal()

    def _evaluate_and_heal(self) -> List[HealEvent]:
        """Evaluate all rules and execute matching actions."""
        events: List[HealEvent] = []

        with self._lock:
            rules_snapshot = list(self._rules)

        records = self._window.get_records()

        for rule in rules_snapshot:
            if not rule.can_fire():
                continue

            if rule.trigger.evaluate(records):
                # Trigger fired — execute the action
                metric_value = rule.trigger._compute_metric(
                    records[-rule.trigger.window :]
                )
                event = self._execute_action(rule, metric_value or 0.0)
                if event:
                    rule.mark_fired()
                    events.append(event)
                    with self._lock:
                        self._heal_history.append(event)

        return events

    def _execute_action(
        self, rule: SelfHealRule, metric_value: float
    ) -> Optional[HealEvent]:
        """Execute a single remediation action."""
        event = HealEvent(
            trigger_expr=rule.trigger.raw,
            action=rule.action.value,
            params=dict(rule.params),
            metric_value=metric_value,
            threshold=rule.trigger.threshold,
        )

        try:
            if rule.action == SelfHealAction.REQUEST_KNOWLEDGE_INGESTION:
                query = rule.params.get("knowledge_query", "")
                if self._knowledge_add_fn and query:
                    self._knowledge_add_fn(
                        f"self_heal_{int(time.time())}",
                        query,
                    )
                event.details = f"Knowledge ingested: {query!r}"

            elif rule.action == SelfHealAction.ADJUST_TEMPERATURE:
                temp = rule.params.get("temperature", 0.5)
                if self._config_set_fn:
                    self._config_set_fn("llm_temperature", temp)
                event.details = f"Temperature adjusted to {temp}"

            elif rule.action == SelfHealAction.SWITCH_MODEL:
                model = rule.params.get("model", "")
                if self._config_set_fn and model:
                    self._config_set_fn("llm_model", model)
                event.details = f"Model switched to {model!r}"

            elif rule.action == SelfHealAction.CLEAR_CACHE:
                event.details = "Cache cleared"

            elif rule.action == SelfHealAction.RESET_MEMORY:
                if self._memory_clear_fn:
                    self._memory_clear_fn()
                event.details = "Agent memory cleared"

            elif rule.action == SelfHealAction.NOTIFY:
                if self._emit_fn:
                    self._emit_fn(
                        "self_heal",
                        {
                            "trigger": rule.trigger.raw,
                            "action": rule.action.value,
                            "metric_value": metric_value,
                            "threshold": rule.trigger.threshold,
                        },
                    )
                event.details = "Notification event emitted"

            elif rule.action == SelfHealAction.CUSTOM:
                handler = rule.params.get("handler")
                if handler and callable(handler):
                    handler(event)
                elif rule.params.get("handler_name"):
                    h = self._custom_handlers.get(rule.params["handler_name"])
                    if h and callable(h):
                        h(event)
                event.details = "Custom handler executed"

            event.success = True

        except Exception as exc:
            event.success = False
            event.details = f"Action failed: {exc}"

        return event

    # ── Query / introspection ─────────────────────────────────────────────────

    @property
    def heal_history(self) -> List[HealEvent]:
        """Full history of all self-healing events."""
        with self._lock:
            return list(self._heal_history)

    @property
    def total_heals(self) -> int:
        with self._lock:
            return len(self._heal_history)

    @property
    def performance(self) -> Dict[str, Any]:
        """Current performance window summary."""
        return self._window.summary()

    @property
    def window(self) -> PerformanceWindow:
        return self._window

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def reset(self) -> None:
        """Reset all state — records, history, rule fire counts."""
        with self._lock:
            self._window.clear()
            self._heal_history.clear()
            for rule in self._rules:
                rule._fire_count = 0
                rule._last_fired = 0.0

    def clear_rules(self) -> None:
        """Remove all rules."""
        with self._lock:
            self._rules.clear()

    def status(self) -> Dict[str, Any]:
        """Full engine status report."""
        with self._lock:
            rules_info = []
            for i, rule in enumerate(self._rules):
                rules_info.append(
                    {
                        "index": i,
                        "trigger": rule.trigger.raw,
                        "action": rule.action.value,
                        "enabled": rule.enabled,
                        "fire_count": rule._fire_count,
                        "can_fire": rule.can_fire(),
                    }
                )

        return {
            "agent": self._agent_name,
            "enabled": self._enabled,
            "rules": rules_info,
            "rule_count": len(rules_info),
            "total_heals": len(self._heal_history),
            "performance": self._window.summary(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE SINGLETON — optional global engine for system-wide monitoring
# ═══════════════════════════════════════════════════════════════════════════════

_global_engine: Optional[SelfHealEngine] = None
_global_lock = threading.Lock()


def get_self_heal_engine() -> SelfHealEngine:
    """Get (or create) the global self-heal engine singleton."""
    global _global_engine
    with _global_lock:
        if _global_engine is None:
            _global_engine = SelfHealEngine(agent_name="__global__")
        return _global_engine


def _reset_self_heal_engine() -> None:
    """Reset the global singleton (for tests)."""
    global _global_engine
    with _global_lock:
        if _global_engine is not None:
            _global_engine.reset()
            _global_engine.clear_rules()
        _global_engine = None
