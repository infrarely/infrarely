"""
aos/core.py — Agent class: the single entry point
═══════════════════════════════════════════════════════════════════════════════
Everything starts here.  ``agent = infrarely.agent("tutor")``

Philosophy 1: Progressive complexity — the simplest agent is ONE line.
Philosophy 5: Observable by default — every agent has state, traces, health.

Public API on Agent:
  agent.run(goal)           → Result  (the workhorse)
  agent.delegate(other, t)  → Result  (multi-agent)
  agent.broadcast(msg)      → None    (pub-sub)
  agent.state               → str     (current cognitive state)
  agent.health()            → HealthReport
  agent.get_trace(id)       → ExecutionTrace
  agent.explain()           → str     (last result explanation)
  agent.memory              → AgentMemory
  agent.tools               → dict
  agent.capabilities        → dict
  agent.reset()             → None
  agent.shutdown()          → None
"""

from __future__ import annotations

import threading
import time
import uuid
import weakref
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from infrarely.core.config import get_config, _ensure_configured, configure
from infrarely.core.result import Result, Error, ErrorType, _ok, _fail
from infrarely.memory.memory import AgentMemory
from infrarely.memory.knowledge import KnowledgeManager, get_knowledge_manager
from infrarely.core.decorators import (
    ToolRegistry,
    get_tool_registry,
    CapabilityRegistry,
    get_capability_registry,
)
from infrarely.observability.observability import (
    ExecutionTrace,
    TraceStore,
    HealthReport,
    MetricsCollector,
    get_metrics,
    get_logger,
)
from infrarely.internal.state_bridge import AgentState, StateMachine
from infrarely.internal.bridge import ExecutionEngine
from infrarely.platform.hitl import (
    HITLGate,
    ApprovalRule,
    ApprovalStatus,
    get_approval_manager,
)
from infrarely.security.security import (
    SecurityGuard,
    SecurityPolicy,
    get_security_guard,
)
from infrarely.core.streaming import StreamIterator, AsyncStreamIterator
from infrarely.security.input_sanitizer import InputSanitizer, get_input_sanitizer
from infrarely.platform.self_heal import (
    SelfHealEngine,
    SelfHealRule,
    SelfHealTrigger,
    SelfHealAction,
    HealEvent,
    _ACTION_MAP,
)
from infrarely.platform.acp import (
    ACPMessage,
    ACPResponse,
    ACPEndpoint,
    ACPIdentity,
    ACPTransport,
    ACPAdapter,
    ACPStatus,
    ACPExchange,
    ACP_VERSION,
    get_acp_transport,
    get_acp_registry,
)


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL AGENT REGISTRY — tracks all live agents via weak references
# ═══════════════════════════════════════════════════════════════════════════════

_MAX_REGISTRY_SIZE = 1000  # hard cap — oldest evicted first
_AGENTS: OrderedDict[str, weakref.ref] = OrderedDict()
_AGENTS_LOCK = threading.Lock()


def _register_agent(agent: "Agent") -> None:
    with _AGENTS_LOCK:
        # Evict stale (gc'd) refs while we're here
        dead = [k for k, ref in _AGENTS.items() if ref() is None]
        for k in dead:
            del _AGENTS[k]
        # Cap size — evict oldest if at limit
        while len(_AGENTS) >= _MAX_REGISTRY_SIZE:
            _AGENTS.popitem(last=False)
        _AGENTS[agent.name] = weakref.ref(agent)
        _AGENTS.move_to_end(agent.name)  # freshest last


def _unregister_agent(name: str) -> None:
    with _AGENTS_LOCK:
        _AGENTS.pop(name, None)


def _get_agent(name: str) -> Optional["Agent"]:
    with _AGENTS_LOCK:
        ref = _AGENTS.get(name)
        if ref is None:
            return None
        agent = ref()
        if agent is None:
            # GC'd — clean up stale entry
            del _AGENTS[name]
        return agent


def _all_agents() -> Dict[str, "Agent"]:
    with _AGENTS_LOCK:
        live: Dict[str, "Agent"] = {}
        dead: list[str] = []
        for k, ref in _AGENTS.items():
            agent = ref()
            if agent is not None:
                live[k] = agent
            else:
                dead.append(k)
        for k in dead:
            del _AGENTS[k]
        return live


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT CLASS
# ═══════════════════════════════════════════════════════════════════════════════


class Agent:
    """
    The core building block of AOS.

    Quick start::

        import infrarely

        agent = infrarely.agent("tutor")
        result = agent.run("What is photosynthesis?")
        print(result.output)

    With tools and knowledge::

        @infrarely.tool
        def lookup_student(student_id: str) -> dict:
            return db.get(student_id)

        tutor = infrarely.agent("tutor", tools=[lookup_student])
        tutor.knowledge.add_documents("./course_notes/")
        result = tutor.run("Summarize notes for CS301")
    """

    def __init__(
        self,
        name: str,
        *,
        tools: Optional[List[Callable]] = None,
        capabilities: Optional[List[Callable]] = None,
        config: Optional[Dict[str, Any]] = None,
        description: str = "",
        model: Optional[str] = None,
        memory_enabled: bool = True,
        knowledge_enabled: bool = True,
    ):
        """
        Create an agent.

        Parameters
        ----------
        name : str
            Unique agent name.  Two agents cannot share a name.
        tools : list[callable], optional
            Functions decorated with ``@infrarely.tool``.
        capabilities : list[callable], optional
            Functions decorated with ``@infrarely.capability``.
        config : dict, optional
            Per-agent config overrides (e.g. ``{"llm_model": "gpt-4o"}``).
        description : str
            What the agent does (used in multi-agent delegation).
        model : str, optional
            Shorthand for ``config={"llm_model": "..."}``
        memory_enabled : bool
            Whether this agent gets a memory store (default True).
        knowledge_enabled : bool
            Whether this agent gets a knowledge manager (default True).
        """
        _ensure_configured()

        self._name = name
        self._description = description or f"Agent '{name}'"
        self._id = f"agent_{name}_{uuid.uuid4().hex[:8]}"
        self._created_at = time.time()

        # ── Per-agent config overrides ────────────────────────────────────────
        self._config_overrides: Dict[str, Any] = config or {}
        if model:
            self._config_overrides["llm_model"] = model

        # ── State machine ─────────────────────────────────────────────────────
        self._state_machine = StateMachine(self._id)

        # ── Tool registry (per-agent view) ────────────────────────────────────
        self._tool_registry = get_tool_registry()
        self._tools: Dict[str, Callable] = {}
        for fn in tools or []:
            meta = self._tool_registry.get_meta(getattr(fn, "__name__", str(fn)))
            if meta:
                self._tools[meta.name] = fn
            else:
                # If not registered via @infrarely.tool, register on the fly
                fname = getattr(fn, "__name__", f"tool_{id(fn)}")
                self._tools[fname] = fn

        # ── Capability registry (per-agent view) ─────────────────────────────
        self._cap_registry = get_capability_registry()
        self._capabilities: Dict[str, Callable] = {}
        for fn in capabilities or []:
            meta = self._cap_registry.get_meta(getattr(fn, "__name__", str(fn)))
            if meta:
                self._capabilities[meta.name] = fn
            else:
                fname = getattr(fn, "__name__", f"cap_{id(fn)}")
                self._capabilities[fname] = fn

        # ── Memory ────────────────────────────────────────────────────────────
        self._memory: Optional[AgentMemory] = None
        if memory_enabled:
            cfg = get_config()
            db_path = cfg.get("memory_db_path", "./aos_memory.db")
            self._memory = AgentMemory(self._id, db_path=db_path)

        # ── Knowledge ─────────────────────────────────────────────────────────
        self._knowledge: Optional[KnowledgeManager] = None
        if knowledge_enabled:
            self._knowledge = get_knowledge_manager()

        # ── Trace store ───────────────────────────────────────────────────────
        cfg = get_config()
        trace_db = cfg.get("trace_db_path", "./aos_traces.db")
        self._trace_store = TraceStore(trace_db)

        # ── HITL gate ────────────────────────────────────────────────────────
        self._hitl_gate: Optional[HITLGate] = None

        # ── Security guard ───────────────────────────────────────────────────
        self._security_guard: Optional[SecurityGuard] = None
        security_cfg = get_config().get("security")
        if security_cfg is not None:
            self._security_guard = SecurityGuard(security_cfg)

        # ── Execution engine ──────────────────────────────────────────────────
        self._engine = ExecutionEngine(
            agent_name=self._name,
            tools=self._tools,
            capabilities=self._capabilities,
            knowledge=self._knowledge or get_knowledge_manager(),
            trace_store=self._trace_store,
            hitl_gate=self._hitl_gate,
        )

        # ── Runtime state ─────────────────────────────────────────────────────
        self._last_result: Optional[Result] = None
        self._task_count = 0
        self._agent_total_tasks = 0
        self._agent_successful_tasks = 0
        self._agent_failed_tasks = 0
        self._agent_llm_calls = 0
        self._logger = get_logger()
        self._metrics = get_metrics()
        self._listeners: Dict[str, List[Callable]] = {}
        self._message_handlers: List[Callable] = []
        self._alive = True

        # ── Self-Healing Engine (lazy — created on first self_improve()) ──────
        self._self_heal_engine: Optional[SelfHealEngine] = None

        # Register globally
        _register_agent(self)
        self._logger.info(
            f"Agent '{self._name}' created",
            agent=self._name,
            tools=list(self._tools.keys()),
            capabilities=list(self._capabilities.keys()),
        )

    # ═══════════════════════════════════════════════════════════════════
    # PROPERTIES
    # ═══════════════════════════════════════════════════════════════════

    @property
    def name(self) -> str:
        """Agent name."""
        return self._name

    @property
    def description(self) -> str:
        """Agent description."""
        return self._description

    @property
    def state(self) -> str:
        """Current cognitive state (IDLE, PLANNING, EXECUTING, etc.)."""
        return self._state_machine.state_name

    @property
    def memory(self) -> AgentMemory:
        """
        Access agent memory.

        Example::

            agent.memory.store("name", "Alice")
            agent.memory.get("name")  # → "Alice"
        """
        if self._memory is None:
            # Create on demand if wasn't enabled
            cfg = get_config()
            db_path = cfg.get("memory_db_path", "./aos_memory.db")
            self._memory = AgentMemory(self._id, db_path=db_path)
        return self._memory

    @property
    def knowledge(self) -> KnowledgeManager:
        """
        Access agent knowledge.

        Example::

            agent.knowledge.add_documents("./notes/")
            agent.knowledge.add_data("pi", "Pi is approximately 3.14159")
        """
        if self._knowledge is None:
            self._knowledge = get_knowledge_manager()
        return self._knowledge

    @property
    def tools(self) -> Dict[str, Callable]:
        """Registered tools."""
        return dict(self._tools)

    @property
    def capabilities(self) -> Dict[str, Callable]:
        """Registered capabilities."""
        return dict(self._capabilities)

    @property
    def alive(self) -> bool:
        """Whether the agent is active."""
        return self._alive

    # ═══════════════════════════════════════════════════════════════════
    # CORE: agent.run(goal) → Result
    # ═══════════════════════════════════════════════════════════════════

    def run(self, goal: str, *, context: Optional[Any] = None) -> Result:
        """
        Execute a goal and return a structured Result.

        The full AOS pipeline:
          1. Knowledge check (bypass LLM if confidence ≥ threshold)
          2. Intent classification (deterministic routing)
          3. Tool / capability / workflow execution
          4. LLM synthesis (last resort)
          5. Trace recording + metrics update

        Parameters
        ----------
        goal : str
            What the agent should accomplish.
        context : Any, optional
            Previous Result or extra context to ground the response.

        Returns
        -------
        Result
            Always a Result, never raises.  Check ``result.success``.

        Example
        -------
        ::

            result = agent.run("What is photosynthesis?")
            if result.success:
                print(result.output)
            else:
                print(result.errors[0].suggestion)
        """
        if not self._alive:
            return _fail(
                ErrorType.STATE_CORRUPTED,
                "Agent has been shut down",
                goal=goal,
                agent_name=self._name,
            )

        # ── Always-on input sanitization (GAP 1) ─────────────────────────
        sanitizer = get_input_sanitizer()
        goal, san_result = sanitizer.sanitize(goal)
        if san_result.blocked:
            return _fail(
                ErrorType.SECURITY_VIOLATION,
                f"Input blocked by sanitizer: {san_result.block_reason}",
                goal="[blocked]",
                agent_name=self._name,
            )

        # ── Security screening ────────────────────────────────────────────
        if self._security_guard is not None:
            allowed, processed_text, detection = self._security_guard.screen_input(
                goal, agent_name=self._name
            )
            if not allowed:
                return _fail(
                    ErrorType.SECURITY_VIOLATION,
                    f"Input blocked by security policy: {detection.details if detection else 'blocked'}",
                    goal=goal,
                    agent_name=self._name,
                )
            if processed_text is not None and processed_text != goal:
                goal = processed_text

        self._task_count += 1

        # Notify listeners
        self._emit("task_start", {"goal": goal, "task_number": self._task_count})

        # State: IDLE → PLANNING → EXECUTING
        self._state_machine.transition(AgentState.PLANNING, reason=f"goal: {goal[:60]}")
        self._state_machine.transition(AgentState.EXECUTING, reason="start_execution")

        try:
            # Delegate to execution engine
            result = self._engine.execute(goal, context=context)

            # State: EXECUTING → VERIFYING → COMPLETED|FAILED
            self._state_machine.transition(
                AgentState.VERIFYING, reason="verifying_result"
            )
            # Track per-agent metrics
            self._agent_total_tasks += 1
            if result.success:
                self._agent_successful_tasks += 1
                self._state_machine.transition(
                    AgentState.COMPLETED, reason="task_success"
                )
            else:
                self._agent_failed_tasks += 1
                self._state_machine.transition(AgentState.FAILED, reason="task_failed")
            if result.used_llm:
                self._agent_llm_calls += 1

            # Auto-store result in memory for context
            if self._memory and result.success:
                try:
                    self._memory.store(
                        f"_last_result_{self._task_count}",
                        {
                            "goal": goal,
                            "output": str(result.output)[:500],
                            "confidence": result.confidence,
                            "used_llm": result.used_llm,
                        },
                        scope="session",
                    )
                except Exception:
                    pass

            self._last_result = result

            # State: → IDLE (ready for next)
            self._state_machine.transition(AgentState.IDLE, reason="ready")

            # Notify listeners
            self._emit("task_complete", result)

            # ── Self-Healing: feed result into engine ─────────────────────
            if self._self_heal_engine is not None:
                err_type = ""
                if result.error and hasattr(result.error, "type"):
                    err_type = (
                        result.error.type.value
                        if hasattr(result.error.type, "value")
                        else str(result.error.type)
                    )
                try:
                    heal_events = self._self_heal_engine.record_task(
                        confidence=result.confidence,
                        success=result.success,
                        duration_ms=result.duration_ms,
                        used_llm=result.used_llm,
                        error_type=err_type,
                    )
                    if heal_events:
                        self._emit("self_heal", heal_events)
                except Exception:
                    pass  # never let self-heal crash the agent

            return result

        except Exception as e:
            # Should never reach here (engine catches all), but just in case
            self._state_machine.transition(AgentState.FAILED, reason=str(e))
            self._state_machine.reset()
            self._logger.error(
                f"Agent '{self._name}' unexpected error: {e}", agent=self._name
            )
            return _fail(
                ErrorType.UNKNOWN,
                f"Unexpected error: {e}",
                goal=goal,
                agent_name=self._name,
            )

    # ═══════════════════════════════════════════════════════════════════
    # MULTI-AGENT: delegation, broadcast, messaging
    # ═══════════════════════════════════════════════════════════════════

    def delegate(
        self,
        other: "Agent",
        task: str,
        *,
        context: Optional[Any] = None,
    ) -> Result:
        """
        Delegate a task to another agent.

        Example::

            researcher = infrarely.agent("researcher")
            writer = infrarely.agent("writer")
            facts = researcher.run("Find facts about Mars")
            summary = writer.delegate(researcher, "Summarize", context=facts)

        Parameters
        ----------
        other : Agent
            The agent to delegate to.
        task : str
            The task description.
        context : Any, optional
            Context to pass (e.g. a previous Result).

        Returns
        -------
        Result
            Result from the delegated agent.
        """
        if not isinstance(other, Agent):
            return _fail(
                ErrorType.DELEGATION_FAILED,
                f"Cannot delegate to {type(other).__name__}. Expected Agent.",
                goal=task,
                agent_name=self._name,
            )

        if not other.alive:
            return _fail(
                ErrorType.DELEGATION_FAILED,
                f"Agent '{other.name}' has been shut down",
                goal=task,
                agent_name=self._name,
            )

        self._logger.info(
            f"Agent '{self._name}' delegating to '{other.name}': {task[:60]}",
            agent=self._name,
            delegate=other.name,
        )

        # Pass our context along
        ctx = context
        if ctx is None and self._last_result is not None:
            ctx = self._last_result

        result = other.run(task, context=ctx)

        # Tag the result with delegation info
        if hasattr(result, "_metadata"):
            result._metadata["delegated_from"] = self._name
            result._metadata["delegated_to"] = other.name
        else:
            result._metadata = {
                "delegated_from": self._name,
                "delegated_to": other.name,
            }

        return result

    def broadcast(self, message: str, data: Any = None) -> int:
        """
        Send a message to all other live agents.

        Returns the number of agents that received the message.

        Example::

            coordinator.broadcast("New assignment posted")
        """
        agents = _all_agents()
        delivered = 0
        for name, agent in agents.items():
            if name == self._name:
                continue
            if not agent.alive:
                continue
            try:
                for handler in agent._message_handlers:
                    handler(self._name, message, data)
                delivered += 1
            except Exception as e:
                self._logger.warning(
                    f"Broadcast to '{name}' failed: {e}", agent=self._name
                )
        self._logger.info(
            f"Agent '{self._name}' broadcast to {delivered} agents",
            agent=self._name,
        )
        return delivered

    def on_message(self, handler: Callable) -> None:
        """
        Register a message handler for broadcast messages.

        Example::

            @agent.on_message
            def handle(from_agent, message, data):
                print(f"{from_agent} says: {message}")
        """
        self._message_handlers.append(handler)

    # ═══════════════════════════════════════════════════════════════════
    # ACP: Cross-Framework Delegation
    # ═══════════════════════════════════════════════════════════════════

    def delegate_external(
        self,
        endpoint: str,
        task: str,
        *,
        protocol: str = "ACP/1.0",
        context: Optional[Any] = None,
        timeout_ms: float = 30_000,
        auth_token: str = "",
        capabilities_required: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Result:
        """
        Delegate a task to an external agent via the Agent Collaboration Protocol.

        This enables AOS agents to collaborate with agents built on any framework
        (LangChain, CrewAI, AutoGPT, custom REST agents) — making AOS the
        infrastructure layer for multi-framework agent systems.

        Parameters
        ----------
        endpoint : str
            URL of the external agent (e.g. "http://other-service/agent").
        task : str
            The task to delegate.
        protocol : str
            Protocol version (default "ACP/1.0").
        context : Any, optional
            Context to send. If a Result, its output is extracted.
        timeout_ms : float
            Timeout in milliseconds (default 30000).
        auth_token : str
            Bearer token for authentication.
        capabilities_required : list[str], optional
            Capabilities the external agent must have.
        metadata : dict, optional
            Extra metadata to include in the ACP message.

        Returns
        -------
        Result
            Structured Result from the external agent.

        Example
        -------
        ::

            result = my_agent.delegate_external(
                endpoint="http://other-service/agent",
                protocol="ACP/1.0",
                task="Research this topic",
                context=my_context,
            )
            print(result.output)
        """
        if not self._alive:
            return _fail(
                ErrorType.STATE_CORRUPTED,
                "Agent has been shut down",
                goal=task,
                agent_name=self._name,
            )

        from infrarely.core.config import get_config

        cfg = get_config()
        if not cfg.get("acp_enabled", True):
            return _fail(
                ErrorType.CONFIGURATION_ERROR,
                "ACP is disabled via configuration (acp_enabled=False)",
                goal=task,
                agent_name=self._name,
            )

        # Build context dict
        ctx_dict: Dict[str, Any] = {}
        if context is not None:
            if isinstance(context, Result):
                ctx_dict = {
                    "previous_output": str(context.output)[:2000],
                    "confidence": context.confidence,
                    "sources": context.sources,
                }
            elif isinstance(context, dict):
                ctx_dict = context
            else:
                ctx_dict = {"data": str(context)[:2000]}

        # Build ACP endpoint
        acp_endpoint = ACPEndpoint(
            url=endpoint,
            name=f"external_{self._name}",
            auth_token=auth_token,
            timeout_ms=timeout_ms,
        )

        # Build ACP message
        message = ACPMessage(
            protocol=protocol,
            sender=ACPIdentity(name=self._name, framework="aos", version="0.1.0"),
            task=task,
            context=ctx_dict,
            timeout_ms=timeout_ms,
            capabilities_required=capabilities_required or [],
            metadata=metadata or {},
        )

        # Validate message
        validation_errors = message.validate()
        if isinstance(validation_errors, list) and len(validation_errors) > 0:
            return _fail(
                ErrorType.DELEGATION_FAILED,
                f"Invalid ACP message: {'; '.join(validation_errors)}",
                goal=task,
                agent_name=self._name,
            )

        self._logger.info(
            f"Agent '{self._name}' delegating externally via ACP to {endpoint}: {task[:60]}",
            agent=self._name,
        )
        self._emit(
            "acp_request",
            {
                "endpoint": endpoint,
                "task": task,
                "protocol": protocol,
                "message_id": message.message_id,
            },
        )

        # Send via transport
        transport = get_acp_transport()
        start_ms = time.time() * 1000
        try:
            acp_response = transport.send(acp_endpoint, message, timeout_ms=timeout_ms)
        except Exception as e:
            elapsed = time.time() * 1000 - start_ms
            self._logger.error(f"ACP delegation failed: {e}", agent=self._name)
            return _fail(
                ErrorType.DELEGATION_FAILED,
                f"ACP transport error: {e}",
                goal=task,
                agent_name=self._name,
                duration_ms=elapsed,
            )

        # Convert ACP response → AOS Result
        result = ACPAdapter.response_to_result(acp_response)

        # Tag with delegation metadata
        result._metadata = {
            "delegated_from": self._name,
            "delegated_to": endpoint,
            "protocol": protocol,
            "acp_message_id": message.message_id,
            "acp_reply_id": acp_response.message_id,
            "external_framework": acp_response.sender.framework,
        }

        self._emit(
            "acp_response",
            {
                "endpoint": endpoint,
                "status": acp_response.status,
                "duration_ms": acp_response.duration_ms,
                "message_id": message.message_id,
            },
        )

        return result

    # ═══════════════════════════════════════════════════════════════════
    # SELF-HEALING: agent.self_improve()
    # ═══════════════════════════════════════════════════════════════════

    def self_improve(
        self,
        trigger: str,
        action: str,
        *,
        knowledge_query: str = "",
        model: str = "",
        temperature: Optional[float] = None,
        handler: Optional[Callable] = None,
        on_heal: Optional[Callable] = None,
        cooldown: float = 60.0,
        max_fires: int = 0,
    ) -> "Agent":
        """
        Add a self-healing rule: the agent auto-detects poor performance
        and fixes itself without human intervention.

        Parameters
        ----------
        trigger : str
            Trigger expression, e.g. "avg_confidence < 0.6 over last 50 tasks".
        action : str
            Remediation action: "request_knowledge_ingestion", "adjust_temperature",
            "switch_model", "clear_cache", "reset_memory", "notify", "custom".
        knowledge_query : str
            Knowledge to ingest (for request_knowledge_ingestion action).
        model : str
            Model to switch to (for switch_model action).
        temperature : float, optional
            Temperature to set (for adjust_temperature action).
        handler : callable, optional
            Custom handler function (for custom action).
        on_heal : callable, optional
            Alias for handler.
        cooldown : float
            Minimum seconds between consecutive firings (default 60).
        max_fires : int
            Maximum number of times this rule can fire (0 = unlimited).

        Returns
        -------
        Agent
            Self, for chaining.

        Example
        -------
        ::

            agent.self_improve(
                trigger="avg_confidence < 0.6 over last 50 tasks",
                action="request_knowledge_ingestion",
                knowledge_query="Python programming documentation",
            )

            # Chain multiple rules:
            agent.self_improve(
                trigger="failure_rate > 0.5 over last 10 tasks",
                action="notify",
            ).self_improve(
                trigger="consecutive_failures > 5",
                action="reset_memory",
            )
        """
        from infrarely.core.config import get_config

        cfg = get_config()
        if not cfg.get("self_healing_enabled", True):
            self._logger.info("Self-healing disabled via config", agent=self._name)
            return self

        # Lazy-create the engine
        if self._self_heal_engine is None:
            self._self_heal_engine = SelfHealEngine(agent_name=self._name)
            # Wire up callbacks
            self._self_heal_engine._knowledge_add_fn = (
                lambda key, val: self.knowledge.add_data(key, val)
            )
            self._self_heal_engine._config_set_fn = lambda k, v: cfg.set(k, v)
            self._self_heal_engine._memory_clear_fn = lambda: (
                self._memory.clear() if self._memory else None
            )
            self._self_heal_engine._emit_fn = self._emit

        # Parse trigger
        parsed_trigger = SelfHealTrigger.parse(trigger)

        # Resolve action enum
        action_lower = action.lower().strip()
        action_enum = _ACTION_MAP.get(action_lower)
        if action_enum is None:
            raise ValueError(
                f"Unknown self-heal action: {action!r}. "
                f"Valid actions: {', '.join(sorted(set(a.value for a in SelfHealAction)))}"
            )

        # Build params
        params: Dict[str, Any] = {}
        if knowledge_query:
            params["knowledge_query"] = knowledge_query
        if model:
            params["model"] = model
        if temperature is not None:
            params["temperature"] = temperature
        # on_heal is alias for handler
        effective_handler = handler or on_heal
        if effective_handler is not None:
            params["handler"] = effective_handler

        rule = SelfHealRule(
            trigger=parsed_trigger,
            action=action_enum,
            params=params,
            cooldown=cooldown,
            max_fires=max_fires,
        )
        self._self_heal_engine.add_rule(rule)

        self._logger.info(
            f"Self-heal rule added: {trigger!r} → {action}",
            agent=self._name,
        )
        return self

    @property
    def self_heal(self) -> Optional[SelfHealEngine]:
        """
        Access the self-healing engine (None if no rules set).

        Example::

            agent.self_improve(trigger="...", action="...")
            print(agent.self_heal.status())
            print(agent.self_heal.heal_history)
        """
        return self._self_heal_engine

    # ═══════════════════════════════════════════════════════════════════
    # OBSERVABILITY
    # ═══════════════════════════════════════════════════════════════════

    def health(self) -> HealthReport:
        """
        Full health report.

        Example::

            print(agent.health())
        """
        metrics = self._metrics
        tool_reg = self._tool_registry

        # Count memory entries
        mem_count = 0
        if self._memory:
            try:
                mem_count = len(self._memory.list_keys())
            except Exception:
                pass

        # Count open circuit breakers
        cb_open = 0
        for tname in self._tools:
            meta = tool_reg.get_meta(tname)
            if meta and hasattr(meta, "circuit_breaker") and meta.circuit_breaker:
                if not meta.circuit_breaker.allow():
                    cb_open += 1

        return HealthReport(
            agent_name=self._name,
            state=self.state,
            uptime_seconds=time.time() - self._created_at,
            total_tasks=self._agent_total_tasks,
            successful_tasks=self._agent_successful_tasks,
            failed_tasks=self._agent_failed_tasks,
            avg_duration_ms=metrics.avg_task_duration(),
            memory_entries=mem_count,
            tools_registered=len(self._tools),
            llm_calls_total=self._agent_llm_calls,
            knowledge_queries_total=0,
            circuit_breakers_open=cb_open,
            last_error=metrics._tool_metrics.get("__last_error", {}).get("msg", ""),
        )

    def get_trace(self, trace_id: str) -> Optional[ExecutionTrace]:
        """
        Retrieve a full execution trace by ID.

        Example::

            result = agent.run("something")
            trace = agent.get_trace(result.trace_id)
            for step in trace.steps:
                print(step.name, step.duration_ms)
        """
        return self._trace_store.get(trace_id)

    def get_recent_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent execution traces for this agent."""
        return self._trace_store.list_recent(agent_name=self._name, limit=limit)

    def explain(self) -> str:
        """
        Human-readable explanation of the last execution.

        Example::

            result = agent.run("Calculate 2+2")
            print(agent.explain())
        """
        if self._last_result is None:
            return f"Agent '{self._name}' has not run any tasks yet."
        return self._last_result.explain()

    # ═══════════════════════════════════════════════════════════════════
    # EVENT SYSTEM
    # ═══════════════════════════════════════════════════════════════════

    def on(self, event: str, handler: Callable = None) -> Any:
        """
        Subscribe to agent events.

        Events: "task_start", "task_complete", "task_failed", "state_change"

        Can be used as a decorator or called directly::

            # As decorator
            @agent.on("task_complete")
            def on_done(result):
                print(f"Done! Confidence: {result.confidence}")

            # Direct call
            agent.on("task_complete", my_handler)
        """
        if handler is not None:
            # Direct call: agent.on("event", handler)
            if event not in self._listeners:
                self._listeners[event] = []
            self._listeners[event].append(handler)
            return handler

        # Decorator usage: @agent.on("event")
        def _decorator(fn: Callable) -> Callable:
            if event not in self._listeners:
                self._listeners[event] = []
            self._listeners[event].append(fn)
            return fn

        return _decorator

    def _emit(self, event: str, data: Any = None) -> None:
        """Fire an event to all listeners."""
        for handler in self._listeners.get(event, []):
            try:
                handler(data)
            except Exception as e:
                self._logger.warning(
                    f"Event handler failed for '{event}': {e}", agent=self._name
                )

    # ═══════════════════════════════════════════════════════════════════
    # HITL: Approval gates
    # ═══════════════════════════════════════════════════════════════════

    def require_approval_for(
        self,
        *,
        tools: Optional[List[str]] = None,
        when: Optional[Callable[..., bool]] = None,
        timeout: Optional[float] = None,
        auto_approve_if: Optional[Callable[..., bool]] = None,
        reason_template: str = "Agent wants to execute {tool_name} with args: {args}",
        notify: Optional[Callable] = None,
    ) -> "Agent":
        """
        Require human approval before executing specified tools.

        Example::

            agent.require_approval_for(
                tools=["process_payment"],
                when=lambda amount, **_: amount > 1000,
                timeout=3600,
            )

        Parameters
        ----------
        tools : list[str]
            Tool names that require approval.
        when : callable, optional
            Condition function receiving tool kwargs.
            If None, always require approval.
        timeout : float, optional
            Seconds to wait for approval before timing out.
        auto_approve_if : callable, optional
            Bypass condition — if True, auto-approve.
        reason_template : str
            Template for the approval reason string.
        notify : callable, optional
            Callback invoked when request is created.

        Returns
        -------
        Agent (self, for chaining)
        """
        if tools is None:
            tools = list(self._tools.keys())

        if self._hitl_gate is None:
            self._hitl_gate = HITLGate(self._name)

        cfg = get_config()
        t = timeout or cfg.get("hitl_default_timeout", 3600)

        rule = ApprovalRule(
            tools=tools,
            condition=when,
            timeout=t,
            auto_approve_if=auto_approve_if,
            reason_template=reason_template,
            notify=notify,
        )
        self._hitl_gate.add_rule(rule)
        self._rebuild_engine()
        self._logger.info(f"HITL rule added for tools: {tools}", agent=self._name)
        return self

    # ═══════════════════════════════════════════════════════════════════
    # STREAMING
    # ═══════════════════════════════════════════════════════════════════

    def stream(self, goal: str) -> StreamIterator:
        """
        Stream the agent's response token-by-token.

        Example::

            for chunk in agent.stream("Explain ML"):
                print(chunk, end="", flush=True)

        Returns
        -------
        StreamIterator
            Iterable of text chunks.
        """
        return StreamIterator(goal, self)

    def astream(self, goal: str) -> AsyncStreamIterator:
        """
        Async stream the agent's response.

        Example::

            async for chunk in agent.astream("Explain ML"):
                await ws.send(chunk)

        Returns
        -------
        AsyncStreamIterator
            Async iterable of text chunks.
        """
        return AsyncStreamIterator(goal, self)

    # ═══════════════════════════════════════════════════════════════════
    # ASYNC EXECUTION
    # ═══════════════════════════════════════════════════════════════════

    async def arun(self, goal: str, *, context: Optional[Any] = None) -> Result:
        """
        Run a goal asynchronously (non-blocking).

        Example::

            result = await agent.arun("What is 2+2?")

            # In FastAPI:
            @app.post("/ask")
            async def ask(q: str):
                result = await agent.arun(q)
                return {"answer": result.output}

        Returns
        -------
        Result
        """
        from infrarely.runtime.async_runner import async_run

        return await async_run(self, goal, context=context)

    # ═══════════════════════════════════════════════════════════════════
    # INTEGRATION: .use()
    # ═══════════════════════════════════════════════════════════════════

    def use(self, *integrations_or_tools) -> "Agent":
        """
        Add integrations or tools to this agent in one call.

        Example::

            from infrarely.integrations import slack, github
            agent = infrarely.agent("bot").use(slack, github)

            # Or add individual tools:
            agent.use(my_tool_function)

        Parameters
        ----------
        *integrations_or_tools : Integration | callable
            Integrations (provides .tools) or individual tool functions.

        Returns
        -------
        Agent (self, for chaining)
        """
        from infrarely.integrations import Integration

        for item in integrations_or_tools:
            if isinstance(item, Integration):
                for tool_fn in item.tools:
                    fname = getattr(tool_fn, "__name__", f"tool_{id(tool_fn)}")
                    self._tools[fname] = tool_fn
            elif callable(item):
                fname = getattr(item, "__name__", f"tool_{id(item)}")
                self._tools[fname] = item
            else:
                self._logger.warning(
                    f"Cannot use {type(item).__name__} as tool/integration",
                    agent=self._name,
                )

        self._rebuild_engine()
        return self

    # ═══════════════════════════════════════════════════════════════════
    # TOOL / CAPABILITY MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    def add_tool(self, fn: Callable) -> None:
        """
        Add a tool to this agent at runtime.

        Example::

            @infrarely.tool
            def search(query: str) -> str:
                return "results..."

            agent.add_tool(search)
        """
        fname = getattr(fn, "__name__", f"tool_{id(fn)}")
        self._tools[fname] = fn
        # Rebuild engine with updated tools
        self._rebuild_engine()
        self._logger.info(f"Tool '{fname}' added to agent '{self._name}'")

    def add_capability(self, fn: Callable) -> None:
        """Add a capability at runtime."""
        fname = getattr(fn, "__name__", f"cap_{id(fn)}")
        self._capabilities[fname] = fn
        self._rebuild_engine()
        self._logger.info(f"Capability '{fname}' added to agent '{self._name}'")

    def _rebuild_engine(self) -> None:
        """Rebuild the execution engine after tool/capability changes."""
        self._engine = ExecutionEngine(
            agent_name=self._name,
            tools=self._tools,
            capabilities=self._capabilities,
            knowledge=self._knowledge or get_knowledge_manager(),
            trace_store=self._trace_store,
            hitl_gate=self._hitl_gate,
        )

    # ═══════════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ═══════════════════════════════════════════════════════════════════

    def reset(self) -> None:
        """
        Reset agent state to IDLE. Clears session memory.

        Example::

            agent.reset()
            assert agent.state == "IDLE"
        """
        self._state_machine.reset()
        self._last_result = None
        self._task_count = 0
        if self._memory:
            try:
                self._memory.clear()
            except Exception:
                pass
        if self._self_heal_engine:
            try:
                self._self_heal_engine.reset()
            except Exception:
                pass
        self._logger.info(f"Agent '{self._name}' reset", agent=self._name)

    def shutdown(self) -> None:
        """
        Shut down the agent. Closes resources.

        Example::

            agent.shutdown()
        """
        self._alive = False
        self._state_machine.reset()
        if self._trace_store:
            try:
                self._trace_store.close()
            except Exception:
                pass
        if self._memory:
            try:
                self._memory.close()
            except Exception:
                pass
        _unregister_agent(self._name)
        self._logger.info(f"Agent '{self._name}' shut down", agent=self._name)

    # ═══════════════════════════════════════════════════════════════════
    # NATURAL LANGUAGE CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════

    @classmethod
    def from_description(
        cls,
        description: str,
        *,
        name: str = "",
        tools: Optional[List[Callable]] = None,
        capabilities: Optional[List[Callable]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> "Agent":
        """
        Create an agent from a plain English description.

        AOS parses the description and auto-generates the configuration
        including tools, capabilities, guardrails, and knowledge schema.

        Example::

            agent = Agent.from_description(\"\"\"
                A customer support agent for an e-commerce store.
                It should answer questions about orders, products, and returns.
                It must never discuss competitor products.
                It should escalate to a human if the customer is angry.
                It should always be polite and professional.
            \"\"\")

        Parameters
        ----------
        description : str
            Plain English description of the desired agent.
        name : str, optional
            Override auto-generated name.
        tools : list, optional
            Additional tools to add.
        capabilities : list, optional
            Additional capabilities to add.
        config : dict, optional
            Additional config overrides.

        Returns
        -------
        Agent
            A fully configured Agent with blueprint metadata.
        """
        from infrarely.platform.nlconfig import NLConfigurator

        configurator = NLConfigurator()
        return configurator.create_agent(
            description,
            name=name,
            tools=tools,
            capabilities=capabilities,
            config=config,
        )

    @property
    def blueprint(self) -> Any:
        """Return the NL blueprint if this agent was created from a description."""
        return getattr(self, "_blueprint", None)

    @property
    def guardrails(self) -> list:
        """Return guardrail rules if created from a description."""
        return getattr(self, "_guardrails", [])

    @property
    def escalation_triggers(self) -> list:
        """Return escalation triggers if created from a description."""
        return getattr(self, "_escalation_triggers", [])

    @property
    def personality(self) -> Any:
        """Return personality profile if created from a description."""
        return getattr(self, "_personality", None)

    @property
    def topic_scope(self) -> Any:
        """Return topic scope if created from a description."""
        return getattr(self, "_topic_scope", None)

    # ═══════════════════════════════════════════════════════════════════
    # DUNDER
    # ═══════════════════════════════════════════════════════════════════

    def __repr__(self) -> str:
        return (
            f"Agent(name={self._name!r}, state={self.state}, "
            f"tools={len(self._tools)}, capabilities={len(self._capabilities)})"
        )

    def __str__(self) -> str:
        return f"Agent('{self._name}')"

    def __del__(self):
        try:
            _unregister_agent(self._name)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL FACTORIES
# ═══════════════════════════════════════════════════════════════════════════════


def agent(
    name: str,
    *,
    tools: Optional[List[Callable]] = None,
    capabilities: Optional[List[Callable]] = None,
    config: Optional[Dict[str, Any]] = None,
    description: str = "",
    model: Optional[str] = None,
    memory_enabled: bool = True,
    knowledge_enabled: bool = True,
) -> Agent:
    """
    Create an agent. This is the primary entry point.

    Simplest usage::

        import infrarely
        agent = infrarely.agent("helper")
        result = agent.run("What is 2 + 2?")

    With tools::

        @infrarely.tool
        def weather(city: str) -> str:
            return f"Sunny in {city}"

        agent = infrarely.agent("weather_bot", tools=[weather])
        result = agent.run("What's the weather in NYC?")

    Parameters
    ----------
    name : str
        Unique agent name.
    tools : list, optional
        Functions decorated with @infrarely.tool.
    capabilities : list, optional
        Functions decorated with @infrarely.capability.
    config : dict, optional
        Override configuration for this agent.
    description : str
        Human description (used in multi-agent routing).
    model : str, optional
        LLM model override (e.g. "gpt-4o", "claude-3-sonnet").
    memory_enabled : bool
        Enable persistent memory (default True).
    knowledge_enabled : bool
        Enable knowledge layer (default True).

    Returns
    -------
    Agent
    """
    return Agent(
        name,
        tools=tools,
        capabilities=capabilities,
        config=config,
        description=description,
        model=model,
        memory_enabled=memory_enabled,
        knowledge_enabled=knowledge_enabled,
    )


def shutdown() -> None:
    """
    Shut down all agents and clean up resources.

    Call this when your application exits::

        infrarely.shutdown()
    """
    agents = _all_agents()
    logger = get_logger()
    for name, ag in agents.items():
        try:
            ag.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down agent '{name}': {e}")
    logger.info(f"AOS shutdown complete. {len(agents)} agents stopped.")
    logger.close()  # Flush and close log files
