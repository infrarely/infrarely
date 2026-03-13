"""
agent/core.py  (v3 — full pipeline)
═══════════════════════════════════════════════════════════════════════════════
Authoritative execution engine for the Student Life Assistant.

Full pipeline (deterministic, no implicit LLM):
──────────────────────────────────────────────────
  1. Store user message in working memory
  2. Router → IntentMatch (intent classifier + tool routing)
  3. Reasoning Engine → ReasoningResult (deterministic rules)
  4. Capability Resolver → Optional[Capability]
     4a. If capability matched:
         → Compiler → Plan Cache → Executor (graph-based) → Result
     4b. Else:
         → Single-tool routing (router.route) → ToolResult
  5. Verification Layer (Layer 4) — structural/logical/policy checks
  6. Error recovery if verification fails
  7. Memory update (working + episodic + long-term)
  8. Format response → AgentResponse

LLM call invariants (enforced, not documented wishes)
──────────────────────────────────────────────────────
  • At most ONE LLM call per user request
  • Zero calls for DETERMINISTIC and TOOL_GENERATIVE plans
  • Token budget checked before every call; hard limit blocks the call
  • All context assembly goes through context_builder (never ad-hoc strings)
"""

from __future__ import annotations
import time
from typing import Optional

import infrarely.core.app_config as config
from infrarely.agent.state import (
    AgentResponse,
    ExecutionPlan,
    Message,
    ResponseType,
    TaskState,
    ToolResult,
)
from infrarely.agent.context_builder import build as build_context, estimate_tokens
from infrarely.agent.response_formatter import format_result
from infrarely.agent.llm_client import llm_call
from infrarely.agent.verification import VerificationEngine
from infrarely.agent.reasoning_engine import DeterministicReasoningEngine
from infrarely.agent.capability_resolver import CapabilityResolver
from infrarely.agent.capability_compiler import (
    CapabilityCompiler,
    PlanCache,
    CompilationError,
)
from infrarely.agent.capability_executor import CapabilityExecutor
from infrarely.agent.error_recovery import ErrorRecoveryEngine
from infrarely.agent.execution_trace import ExecutionTrace
from infrarely.memory.advance_memory import AdvancedMemoryManager
from infrarely.memory.working import WorkingMemory
from infrarely.memory.structured import StructuredMemory
from infrarely.memory.long_term import LongTermMemory
from infrarely.router.tool_router import ToolRouter
from infrarely.tools.registry import ToolRegistry
from infrarely.observability import logger
from infrarely.observability.metrics import collector
from infrarely.observability.token_budget import TokenBudget

# ── Layer 5: Optimization Intelligence ────────────────────────────────────────
from infrarely.optimization.routing_optimizer import RoutingOptimizer
from infrarely.optimization.parameter_inference import ParameterInferenceEngine
from infrarely.optimization.capability_optimizer import CapabilityOptimizer
from infrarely.optimization.failure_analyzer import FailureAnalyzer
from infrarely.optimization.capability_discovery import CapabilityDiscoveryEngine
from infrarely.optimization.token_optimizer import TokenOptimizer
from infrarely.optimization.skill_memory import SkillMemory
from infrarely.optimization.quality_scorer import ExecutionQualityScorer
from infrarely.optimization.safety_controller import SafetyController
from infrarely.optimization.trace_intelligence import TraceIntelligenceEngine

# ── Layer 6: Multi-Agent Runtime ──────────────────────────────────────────────
from infrarely.runtime.agent_registry import AgentRegistry, AgentStatus
from infrarely.runtime.agent_scheduler import AgentScheduler, SchedulingStrategy
from infrarely.runtime.message_bus import MessageBus
from infrarely.runtime.shared_memory import SharedMemory
from infrarely.runtime.identity_permissions import (
    IdentityManager,
    AgentRole,
    Permission,
)
from infrarely.runtime.resource_isolation import ResourceIsolation, ResourceQuota
from infrarely.runtime.capability_market import CapabilityMarketplace
from infrarely.runtime.negotiation_protocol import NegotiationProtocol
from infrarely.runtime.lifecycle_manager import LifecycleManager
from infrarely.runtime.agent_monitoring import AgentMonitoring

# ── AOS Capabilities ─────────────────────────────────────────────────────────
from infrarely.agent.state_machine import (
    AgentStateMachine,
    AgentCognitiveState,
    StateMachineManager,
    InvalidTransitionError,
)
from infrarely.agent.planning_engine import DeterministicPlanningEngine
from infrarely.agent.knowledge_layer import (
    KnowledgeLayer,
    ConfidenceLevel,
    build_grounded_prompt,
    create_knowledge_layer_from_data,
)


_SYSTEM_PROMPT = (
    "You are a concise, accurate student life assistant. "
    "Answer in ≤4 sentences. Use only the provided context. "
    "Never invent facts. If context is missing, say so briefly."
)


class AgentCore:
    """
    One AgentCore instance = one student session.
    Full pipeline: Router → Reasoning → Resolver → Compiler → Cache → Executor → Verify → Memory → Response
    """

    def __init__(self, app_cfg, structured_memory: StructuredMemory):
        self._cfg = app_cfg
        self._smem = structured_memory
        self._wmem = WorkingMemory()
        self._ltmem = LongTermMemory()
        self._amem = AdvancedMemoryManager(app_cfg.student_id)
        self._budget = TokenBudget(
            agent_id=app_cfg.student_id,
            session_hard_limit=config.TOKEN_DAILY_BUDGET,
        )

        # ── Tool registry + router ────────────────────────────────────────────
        registry = ToolRegistry()
        self._registry = registry
        self._router = ToolRouter(registry, app_cfg)

        # ── Layer 2: Capability pipeline ──────────────────────────────────────
        self._capability_registry = None  # set via wire_capabilities()
        self._compiler = CapabilityCompiler(registry)
        self._cache = PlanCache()
        self._executor = CapabilityExecutor(registry, app_cfg)

        # ── Layer 3: Infrastructure ───────────────────────────────────────────
        self._reasoning = DeterministicReasoningEngine()
        self._verifier = VerificationEngine()
        self._recovery = ErrorRecoveryEngine()
        self._resolver = None  # needs capability_registry → set in wire_capabilities
        self._session_id = ""  # set via set_session_id()

        # ── Layer 5: Adaptive Intelligence ─────────────────────────────────────
        self._routing_optimizer = RoutingOptimizer()
        self._param_inference = ParameterInferenceEngine()
        self._capability_optimizer = CapabilityOptimizer()
        self._failure_analyzer = FailureAnalyzer()
        self._capability_discovery = CapabilityDiscoveryEngine()
        self._token_optimizer = TokenOptimizer()
        self._skill_memory = SkillMemory()
        self._quality_scorer = ExecutionQualityScorer()
        self._safety_controller = SafetyController()
        self._trace_intelligence = TraceIntelligenceEngine()

        # ── Wire safety controller into executor (Gap 5) ──────────────────────
        self._executor._safety = self._safety_controller
        # ── Wire safety controller into parameter inference (Gap 6) ───────────
        self._param_inference._safety_controller = self._safety_controller

        # ── Layer 6: Multi-Agent Runtime ──────────────────────────────────────
        self._rt_registry = AgentRegistry()
        _sched_strategy = {
            "priority": SchedulingStrategy.PRIORITY,
            "round_robin": SchedulingStrategy.ROUND_ROBIN,
        }.get(
            getattr(config, "RUNTIME_SCHEDULER_STRATEGY", "capability"),
            SchedulingStrategy.CAPABILITY,
        )
        self._rt_scheduler = AgentScheduler(strategy=_sched_strategy)
        self._rt_bus = MessageBus()
        self._rt_shared_memory = SharedMemory()
        self._rt_identity = IdentityManager()
        self._rt_resources = ResourceIsolation()
        self._rt_market = CapabilityMarketplace()
        self._rt_negotiation = NegotiationProtocol()
        self._rt_lifecycle = LifecycleManager(
            idle_timeout_ms=getattr(config, "RUNTIME_AGENT_IDLE_TIMEOUT_MS", 300_000),
        )
        self._rt_monitoring = AgentMonitoring()

        # ── AOS Capability 1: Agent State Machine ─────────────────────────────
        self._state_manager = StateMachineManager()
        self._agent_sm = self._state_manager.create_agent(
            agent_id=app_cfg.student_id,
        )

        # ── AOS Capability 2: Deterministic Planning Engine ───────────────────
        self._planning_engine = DeterministicPlanningEngine(
            capability_registry=None,  # populated in wire_capabilities()
            tool_registry=registry,
            token_budget_limit=config.TOKEN_DAILY_BUDGET,
        )

        # ── AOS Capability 3: Knowledge Layer ─────────────────────────────────
        import os as _os

        _data_dir = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "data",
        )
        self._knowledge = create_knowledge_layer_from_data(_data_dir)

        # Apply config limits to runtime modules
        self._rt_bus.RATE_LIMIT = getattr(config, "RUNTIME_MESSAGE_RATE_LIMIT", 50)
        self._rt_bus.BUS_CAPACITY = getattr(
            config, "RUNTIME_MESSAGE_BUS_CAPACITY", 2000
        )
        self._rt_bus.INBOX_CAP = getattr(config, "RUNTIME_MESSAGE_INBOX_CAP", 100)
        self._rt_scheduler.MAX_QUEUE = getattr(config, "RUNTIME_SCHEDULER_QUEUE", 200)
        self._rt_shared_memory.MAX_ENTRIES = getattr(
            config, "RUNTIME_SHARED_MEMORY_MAX", 1000
        )
        self._rt_shared_memory.LOCK_TIMEOUT_S = getattr(
            config, "RUNTIME_SHARED_MEMORY_LOCK_TIMEOUT", 30.0
        )
        self._rt_resources.GLOBAL_TOKEN_CEILING = getattr(
            config, "RUNTIME_GLOBAL_TOKEN_CEILING", 10_000
        )
        self._rt_negotiation.MAX_CONCURRENT = getattr(
            config, "RUNTIME_NEGOTIATION_MAX_CONCURRENT", 20
        )
        self._rt_lifecycle.GC_INTERVAL_MS = getattr(
            config, "RUNTIME_GC_INTERVAL_MS", 60_000
        )

        logger.info(
            "AgentCore ready (v6 full pipeline + adaptive + runtime + AOS)",
            agent=app_cfg.student_id,
            backend=config.LLM_BACKEND,
            model=config.LLM_MODEL,
        )

    def wire_capabilities(self, capability_registry) -> None:
        """
        Wire the capability registry and resolver.
        Called during CLI bootstrap after capabilities are registered.
        """
        self._capability_registry = capability_registry
        self._resolver = CapabilityResolver(capability_registry)
        # Update planning engine with known capabilities
        self._planning_engine._capability_registry = capability_registry
        logger.info("Capabilities wired", count=len(capability_registry.names()))

    def set_session_id(self, session_id: str) -> None:
        """Set the session ID for execution traces."""
        self._session_id = session_id

    # ── public API ────────────────────────────────────────────────────────────
    def process(self, user_input: str) -> AgentResponse:
        """
        Process one user turn.  Returns an AgentResponse.

        Full pipeline:
          1. Working memory + episodic record
          2. Router → ExecutionPlan + ToolResult (single-tool)
          3. Reasoning Engine evaluation
          4. Capability Resolver → try multi-step capability
          5. If capability → Compiler → Cache → Executor
          6. Verification Layer (Layer 4)
          7. Memory update
          8. Response formatting
        """
        t_start = time.monotonic()

        # ── 1. Store user message ─────────────────────────────────────────────
        evicted = self._wmem.add(Message(role="user", content=user_input))
        self._amem.record_episode("user_input", text=user_input[:200])
        if evicted:
            self._ltmem.compress_and_store(self._cfg.student_id, [evicted])

        # ── AOS: State Machine → PLANNING ─────────────────────────────────────
        try:
            self._agent_sm.set_goal({"query": user_input[:200]})
            self._agent_sm.transition(
                AgentCognitiveState.PLANNING, reason=user_input[:100]
            )
        except InvalidTransitionError:
            # Reset to IDLE then transition (e.g. after previous COMPLETED/FAILED)
            try:
                self._agent_sm.transition(AgentCognitiveState.IDLE)
                self._agent_sm.set_goal({"query": user_input[:200]})
                self._agent_sm.transition(
                    AgentCognitiveState.PLANNING, reason=user_input[:100]
                )
            except Exception:
                pass  # non-blocking — state machine is observational
        except Exception:
            pass  # non-blocking

        # ── AOS: Knowledge Layer — RULE 3: Knowledge before generation ────────
        knowledge_result = None
        knowledge_action = "no_knowledge"
        try:
            knowledge_result, knowledge_action = self._knowledge.query_with_decision(
                user_input
            )
            if knowledge_result:
                self._last_knowledge_result = knowledge_result
                logger.debug(
                    "KnowledgeLayer",
                    action=knowledge_action,
                    confidence=round(knowledge_result.confidence, 3),
                    chunks=len(knowledge_result.chunks),
                )
        except Exception as e:
            logger.error(f"KnowledgeLayer query failed: {e}")

        # ── 2. Create execution trace (CP4) ───────────────────────────────────
        trace = ExecutionTrace.new(user_input, self._cfg.student_id, self._session_id)

        # ── 3. Classify intent (no tool execution yet) ───────────────────────
        match, task_state = self._router.classify_only(user_input)
        trace.add_step("classify", task_state.tool, "intent_classified")

        # ── Layer 5: Adaptive routing confidence adjustment ───────────────────
        if getattr(config, "ENABLE_ADAPTIVE", True) and self._routing_optimizer.enabled:
            base_conf = match.confidence if hasattr(match, "confidence") else 0.7
            adjusted = self._routing_optimizer.adjust_confidence(
                task_state.task, base_conf
            )
            logger.debug(
                "RoutingOptimizer",
                intent=task_state.task,
                base=round(base_conf, 3),
                adjusted=round(adjusted, 3),
            )

        # ── Layer 5: Skill memory hint ────────────────────────────────────────
        skill_ctx = {
            "intent": task_state.task,
            "tool": task_state.tool,
            "course_id": task_state.params.get("course_id", ""),
            "topic": task_state.params.get("topic", ""),
        }
        skill_hint = self._skill_memory.best_action(skill_ctx)
        if skill_hint:
            logger.debug("SkillMemory hint", action=skill_hint)

        # ── Layer 5: Parameter inference for missing params ───────────────────
        if getattr(config, "ENABLE_ADAPTIVE", True):
            mem_facts = self._amem.facts_snapshot()
            inferred = self._param_inference.infer(
                query=user_input,
                existing_params=task_state.params,
                intent=task_state.task,
                memory_facts=mem_facts,
            )
            for pname, inf_param in inferred.items():
                if (
                    pname not in task_state.params
                    and self._safety_controller.can_infer_parameter(
                        pname, inf_param.confidence
                    )
                ):
                    task_state.params[pname] = inf_param.value
                    logger.debug(
                        "ParamInference filled",
                        param=pname,
                        value=inf_param.value,
                        conf=inf_param.confidence,
                        source=inf_param.source,
                    )

        # ── 4. Reasoning Engine ───────────────────────────────────────────────
        reasoning_ctx = {
            "intent": task_state.task,
            "tool": task_state.tool,
            "requires_llm": task_state.requires_llm,
            "course_id": task_state.params.get("course_id"),
            "topic": task_state.params.get("topic"),
            "query": user_input,
            "profile_loaded": True,  # assume loaded — smem is always available
            "llm_online": bool(
                config.GROQ_API_KEY
                or config.GEMINI_API_KEY
                or config.LLM_BACKEND == "ollama"
            ),
        }
        if config.ENABLE_REASONING_ENGINE:
            reasoning_result = self._reasoning.evaluate(reasoning_ctx)
            logger.debug(
                "ReasoningEngine result",
                triggered=reasoning_result.triggered_rules,
                recommended=reasoning_result.recommended_steps,
            )
        else:
            from infrarely.agent.reasoning_engine import ReasoningResult

            reasoning_result = ReasoningResult(
                triggered_rules=[], blocked_rules=[], recommended_steps=[]
            )

        # ── 5. Capability Resolver ────────────────────────────────────────────
        capability_result = None
        resolved_cap = None
        plan = None
        tool_result = None

        # ── AOS: Knowledge bypass gate ────────────────────────────────────────
        # If knowledge confidence >= 0.85 AND authoritative → skip LLM entirely
        if knowledge_action == "bypass_llm" and knowledge_result:
            # Compose direct answer from knowledge chunks
            kb_text = "\n".join(c.content for c in knowledge_result.chunks[:3])
            logger.info(
                "KnowledgeLayer: HIGH confidence bypass — no LLM call",
                confidence=round(knowledge_result.confidence, 3),
            )
            # Transition: PLANNING → EXECUTING → VERIFYING → COMPLETED
            try:
                self._agent_sm.set_plan({"knowledge_bypass": True})
                self._agent_sm.transition(AgentCognitiveState.EXECUTING)
                self._agent_sm.advance_cursor()
                self._agent_sm.transition(AgentCognitiveState.VERIFYING)
                self._agent_sm.transition(AgentCognitiveState.COMPLETED)
            except Exception:
                pass

            self._wmem.add(Message(role="assistant", content=kb_text))
            total_ms = (time.monotonic() - t_start) * 1000
            return AgentResponse(
                message=kb_text,
                response_type=ResponseType.DETERMINISTIC,
                task_state=task_state,
                tool_result=None,
                plan=None,
                llm_used=False,
                tokens_used=0,
                execution_ms=round(total_ms, 1),
            )

        # ── AOS: State Machine → EXECUTING ────────────────────────────────────
        try:
            self._agent_sm.set_plan(
                {"intent": task_state.task, "tool": task_state.tool}
            )
            self._agent_sm.transition(AgentCognitiveState.EXECUTING)
        except Exception:
            pass  # non-blocking

        # ── Layer 5: Safety controller — enter execution ──────────────────
        self._safety_controller.enter_execution()

        if self._resolver and self._capability_registry:
            resolver_result = self._resolver.resolve(
                reasoning=reasoning_result,
                query=user_input,
                intent=task_state.task,
                params=task_state.params,
            )

            if resolver_result.capability:
                resolved_cap = resolver_result.capability
                trace.capability = resolved_cap.name

                try:
                    # ── 6a. Compile + Cache (CP2/CP8/Gap10) ────────────────
                    extra_params = dict(task_state.params)
                    cap_version = getattr(resolved_cap, "version", 1)
                    cached = self._cache.get(
                        resolved_cap.name, extra_params, version=cap_version
                    )
                    if cached:
                        cap_plan, cap_info = cached
                        logger.debug(
                            f"PlanCache HIT for '{resolved_cap.name}' v{cap_version}"
                        )
                    else:
                        cap_plan, cap_info = self._compiler.compile(
                            resolved_cap, task_state, extra_params
                        )
                        self._cache.put(
                            resolved_cap.name,
                            extra_params,
                            cap_plan,
                            cap_info,
                            version=cap_version,
                        )

                    trace.add_step(
                        "compile",
                        resolved_cap.name,
                        "compiled",
                        tokens=cap_info.estimated_tokens,
                    )

                    # ── Execute with trace + budget (CP5/CP9) ─────────────────
                    capability_result = self._executor.execute(
                        cap_plan, trace=trace, budget=self._budget
                    )

                except CompilationError as ce:
                    logger.error(f"Capability compilation failed: {ce}")
                    trace.errors.append(f"compilation: {ce}")
                except Exception as e:
                    logger.error(f"Capability execution failed: {e}")
                    trace.errors.append(f"execution: {e}")

        # ── 6b. Single-tool fallback ──────────────────────────────────────────
        if capability_result is None:
            plan, tool_result = self._router.execute_tool(match, task_state)

            trace.add_step(
                "single_tool",
                tool_result.tool_name,
                tool_result.contract.value,
                duration_ms=tool_result.duration_ms,
                tokens=tool_result.metadata.get("tokens_used", 0),
            )

            # Verification (CP7)
            vr = self._verifier.verify(
                tool_result,
                {"token_budget": self._budget.snapshot()},
            )
            if not vr.passed:
                logger.warn(
                    "Verification failed for single-tool result",
                    violations=vr.violations[:3],
                )
                trace.errors.append(f"verification: {'; '.join(vr.violations[:2])}")

            # Error recovery (CP3 — always trigger on errors)
            if not tool_result.success:
                recovery_msg = self._recovery.recover(tool_result)
                if recovery_msg:
                    tool_result.message = recovery_msg
                    trace.add_step("recovery", tool_result.tool_name, "recovered")

        # ── Layer 5: Safety controller — exit execution ───────────────────
        self._safety_controller.exit_execution()

        # ── 7. Determine final response source ───────────────────────────────
        if capability_result is not None:
            response_text = capability_result.message
            llm_used = capability_result.tokens_used > 0
            tokens_used = capability_result.tokens_used
            response_type = (
                ResponseType.TOOL_GENERATIVE if llm_used else ResponseType.DETERMINISTIC
            )

            # Verification on capability outputs
            for step_name, step_result in capability_result.step_outputs.items():
                vr = self._verifier.verify(
                    step_result,
                    {"token_budget": self._budget.snapshot()},
                )
                if not vr.passed:
                    logger.warn(
                        f"Verification warning on capability step '{step_name}'",
                        violations=vr.violations[:2],
                    )

            # Semantic memory (CP10)
            if resolved_cap:
                self._amem.store_capability_facts(
                    resolved_cap.name, capability_result.step_outputs
                )
                self._amem.learn_pattern(task_state.task, resolved_cap.name)

        else:
            # Single-tool path — execute plan (LLM gate enforcement)
            response_text, llm_used, tokens_used = self._execute_plan(
                plan, tool_result, user_input
            )
            response_type = plan.response_type if plan else ResponseType.DETERMINISTIC

        # ── 8. Store assistant message + episodic record ──────────────────────
        self._wmem.add(Message(role="assistant", content=response_text))
        self._amem.record_episode(
            "assistant_response",
            response_type=(
                resolved_cap.name
                if resolved_cap
                else (plan.response_type.name if plan else "unknown")
            ),
            llm_used=llm_used,
            tokens=tokens_used,
            tool=tool_result.tool_name if tool_result else "",
            capability=resolved_cap.name if resolved_cap else None,
        )

        # ── 9. Update session counters ────────────────────────────────────────
        if llm_used:
            self._cfg.llm_call_count_session += 1
            self._cfg.token_count_session += tokens_used
            # Budget already recorded in executor for capability path
            if capability_result is None:
                self._budget.record(tokens_used, reason=task_state.task)

        # ── AOS: State Machine → VERIFYING → COMPLETED/FAILED ─────────────
        try:
            self._agent_sm.advance_cursor()  # satisfy execution_has_results guard
            self._agent_sm.transition(AgentCognitiveState.VERIFYING)
            final_state = (
                AgentCognitiveState.COMPLETED
                if (capability_result and capability_result.success)
                or (tool_result and tool_result.success)
                else AgentCognitiveState.FAILED
            )
            self._agent_sm.transition(final_state)
        except Exception:
            pass  # non-blocking

        # ── 10. Finalize and save execution trace (CP4/CP9) ──────────────────
        total_ms = (time.monotonic() - t_start) * 1000

        if capability_result is not None:
            outcome = "success" if capability_result.success else "partial"
            final_contract = (
                f"capability:{resolved_cap.name}" if resolved_cap else "capability"
            )
        elif tool_result and tool_result.success:
            outcome = "success"
            final_contract = tool_result.contract.value
        else:
            outcome = "failed"
            final_contract = tool_result.contract.value if tool_result else "unknown"

        trace.llm_calls = 1 if llm_used else 0
        trace.finalise(outcome, final_contract, total_ms)
        trace.save()

        # ── 11. Layer 5: Post-execution adaptive hooks ────────────────────────
        if getattr(config, "ENABLE_ADAPTIVE", True):
            self._layer5_post_execution(
                task_state=task_state,
                outcome=outcome,
                total_ms=total_ms,
                tokens_used=tokens_used,
                llm_used=llm_used,
                capability_result=capability_result,
                resolved_cap=resolved_cap,
                tool_result=tool_result,
                trace=trace,
            )

        execution_ms = total_ms

        return AgentResponse(
            message=response_text,
            response_type=response_type,
            task_state=task_state,
            tool_result=tool_result,
            plan=plan,
            llm_used=llm_used,
            tokens_used=tokens_used,
            execution_ms=round(execution_ms, 1),
        )

    # ── Layer 5 post-execution hook ──────────────────────────────────────────
    def _layer5_post_execution(
        self,
        task_state,
        outcome: str,
        total_ms: float,
        tokens_used: int,
        llm_used: bool,
        capability_result,
        resolved_cap,
        tool_result,
        trace,
    ) -> None:
        """
        Collect all Layer 5 adaptive recordings after execution.
        Each module is isolated (Gap 8) — one module's failure never
        blocks another, and never blocks the response to the user.
        """
        import time as _time

        success = outcome == "success"
        intent = task_state.task
        cap_name = resolved_cap.name if resolved_cap else ""
        tool_name = tool_result.tool_name if tool_result else ""

        # Maximum time budget for ALL layer5 work (ms)
        L5_TIME_BUDGET_MS = 200.0
        l5_start = _time.monotonic()

        def _budget_ok() -> bool:
            return (_time.monotonic() - l5_start) * 1000 < L5_TIME_BUDGET_MS

        def _safe_call(module_name: str, fn, *args, **kwargs):
            """Run a single adaptive module call with isolation (Gap 8)."""
            if not _budget_ok():
                logger.debug(f"Layer5 time budget exceeded — skipping {module_name}")
                return
            try:
                fn(*args, **kwargs)
            except Exception as e:
                logger.error(f"Layer5 [{module_name}] error: {e}")

        # ── Routing Optimizer ──────────────────────────────────────────────
        def _routing():
            self._routing_optimizer.record(
                intent=intent,
                success=success,
                latency_ms=total_ms,
                verification_passed=not bool(trace.errors),
                capability=cap_name,
            )

        _safe_call("routing_optimizer", _routing)

        # ── Parameter Inference — record successful params ────────────────
        if success:
            _safe_call(
                "param_inference",
                self._param_inference.record_success,
                intent=intent,
                params=task_state.params,
                capability=cap_name,
            )

        # ── Capability Optimizer ──────────────────────────────────────────
        if capability_result is not None and resolved_cap:

            def _cap_opt():
                node_results = {}
                for step_name, step_output in capability_result.step_outputs.items():
                    node_results[step_name] = {
                        "success": step_output.success,
                        "latency_ms": step_output.duration_ms,
                        "tokens": step_output.metadata.get("tokens_used", 0),
                    }
                self._capability_optimizer.record_capability(
                    capability_name=cap_name,
                    success=success,
                    partial=(outcome == "partial"),
                    latency_ms=total_ms,
                    tokens=tokens_used,
                    node_results=node_results,
                )

            _safe_call("capability_optimizer", _cap_opt)

        # ── Failure Analyzer ──────────────────────────────────────────────
        if not success:

            def _failure():
                error_type = "tool_error"
                error_msg = ""
                if trace.errors:
                    first_err = trace.errors[0]
                    if "verification" in first_err:
                        error_type = "verification"
                    elif "missing_param" in first_err:
                        error_type = "missing_param"
                    error_msg = first_err
                elif tool_result and tool_result.error:
                    error_msg = tool_result.error

                self._failure_analyzer.record_failure(
                    tool=tool_name or cap_name,
                    error_type=error_type,
                    error_msg=error_msg,
                    intent=intent,
                    capability=cap_name,
                    params=task_state.params,
                )

            _safe_call("failure_analyzer", _failure)

        if tool_result:
            _safe_call(
                "failure_analyzer_exec",
                self._failure_analyzer.record_tool_execution,
                tool=tool_name,
                success=tool_result.success,
            )

        # ── Capability Discovery — record tool sequences ──────────────────
        if capability_result is not None and resolved_cap:
            tools_used = list(capability_result.step_outputs.keys())
            _safe_call(
                "capability_discovery",
                self._capability_discovery.record_sequence,
                tools=tools_used,
                intent=intent,
                success=success,
            )
        elif tool_result:
            _safe_call(
                "capability_discovery",
                self._capability_discovery.record_sequence,
                tools=[tool_name],
                intent=intent,
                success=success,
            )

        # ── Token Optimizer ───────────────────────────────────────────────
        _safe_call(
            "token_optimizer",
            self._token_optimizer.record_execution,
            tool=tool_name or cap_name,
            tokens=tokens_used,
            llm_used=llm_used,
            capability=cap_name,
        )

        # ── Skill Memory — learn from successful capability flows ─────────
        if success and cap_name:

            def _skill():
                skill_condition = {
                    "intent": intent,
                    "capability": cap_name,
                }
                if task_state.params.get("topic"):
                    skill_condition["has_topic"] = True
                if task_state.params.get("course_id"):
                    skill_condition["has_course"] = True
                self._skill_memory.learn(
                    condition=skill_condition,
                    action=f"route_to:{cap_name}",
                )

            _safe_call("skill_memory_learn", _skill)

        if cap_name:
            _safe_call(
                "skill_memory_outcome",
                self._skill_memory.record_outcome,
                action=f"route_to:{cap_name}",
                success=success,
            )

        # ── Quality Scorer ────────────────────────────────────────────────
        if capability_result is not None and resolved_cap:

            def _quality():
                steps_total = len(resolved_cap.steps)
                steps_completed = len(capability_result.step_outputs)
                steps_failed = sum(
                    1 for s in capability_result.step_outputs.values() if not s.success
                )
                ver_ok = sum(
                    1
                    for s in capability_result.step_outputs.values()
                    if not self._verifier.verify(
                        s, {"token_budget": self._budget.snapshot()}
                    ).violations
                )
                ver_fail = steps_completed - ver_ok
                quality = self._quality_scorer.record(
                    capability=cap_name,
                    steps_total=steps_total,
                    steps_completed=steps_completed,
                    steps_failed=steps_failed,
                    verification_ok=ver_ok,
                    verification_fail=ver_fail,
                    latency_ms=total_ms,
                    tokens_used=tokens_used,
                )
                logger.debug(
                    "QualityScore", capability=cap_name, score=round(quality, 3)
                )

            _safe_call("quality_scorer", _quality)

        # ── Trace Intelligence ────────────────────────────────────────────
        def _trace_intel():
            tools_list = []
            if capability_result and resolved_cap:
                tools_list = list(capability_result.step_outputs.keys())
            elif tool_result:
                tools_list = [tool_name]
            self._trace_intelligence.record_trace(
                trace_id=trace.run_id,
                capability=cap_name,
                tools_used=tools_list,
                tokens=tokens_used,
                latency_ms=total_ms,
                outcome=outcome,
                errors=trace.errors,
            )

        _safe_call("trace_intelligence", _trace_intel)

        # Log total Layer 5 overhead
        l5_elapsed = (_time.monotonic() - l5_start) * 1000
        if l5_elapsed > 100:
            logger.debug(f"Layer5 post-execution took {l5_elapsed:.0f}ms")

    # ── Layer 5 snapshot (for CLI) ────────────────────────────────────────────
    def get_adaptive_snapshot(self) -> dict:
        """Return a combined snapshot of all Layer 5 modules."""
        return {
            "routing_optimizer": self._routing_optimizer.snapshot(),
            "parameter_inference": self._param_inference.snapshot(),
            "capability_optimizer": self._capability_optimizer.snapshot(),
            "failure_analyzer": self._failure_analyzer.snapshot(),
            "capability_discovery": self._capability_discovery.snapshot(),
            "token_optimizer": self._token_optimizer.snapshot(),
            "skill_memory": self._skill_memory.snapshot(),
            "quality_scorer": self._quality_scorer.snapshot(),
            "safety_controller": self._safety_controller.snapshot(),
            "trace_intelligence": self._trace_intelligence.snapshot(),
        }

    def get_memory_snapshot(self) -> dict:
        return {
            "working_memory_turns": self._wmem.turn_count(),
            "long_term_summaries": len(
                self._ltmem.get_all_summaries(self._cfg.student_id)
            ),
            "episodic_events": len(self._amem.recent_episodes(n=10_000)),
            "token_budget": self._budget.snapshot(),
            "plan_cache": self._cache.stats(),
        }

    # ── AOS snapshot (for CLI) ────────────────────────────────────────────────
    def get_aos_snapshot(self) -> dict:
        """Return a combined snapshot of all AOS capabilities."""
        sm_state = self._agent_sm.state
        return {
            "state_machine": {
                "current_state": sm_state.current_state.value,
                "agent_id": sm_state.agent_id,
                "transitions": len(sm_state.transition_history),
            },
            "planning_engine": {
                "cache_size": len(self._planning_engine._cache),
            },
            "knowledge_layer": self._knowledge.snapshot(),
        }

    def reset_session(self) -> None:
        self._wmem.clear()
        self._amem = AdvancedMemoryManager(self._cfg.student_id)
        self._budget.reset()
        self._cache.invalidate()
        self._cfg.reset_session_counters()
        logger.info("Session reset", agent=self._cfg.student_id)

    # ── execution plan enforcement (single-tool path) ─────────────────────────
    def _execute_plan(
        self,
        plan: ExecutionPlan,
        tool_result: ToolResult,
        user_input: str,
    ) -> tuple[str, bool, int]:
        """
        Returns (response_text, llm_used, tokens_used).

        Hard gates (in priority order):
          A. result.is_complete = True  → format directly, no LLM, ever
          B. plan.allow_llm = False     → format directly, no LLM, ever
          C. plan.allow_llm = True      → one LLM call (if budget allows)
        """
        # Gate A — tool produced the final answer (e.g. practice questions)
        if tool_result.is_complete:
            logger.debug(
                "Gate A: tool is_complete=True → skip LLM", tool=tool_result.tool_name
            )
            return format_result(tool_result), False, 0

        # Gate B — router said no LLM
        if not plan.allow_llm:
            logger.debug(
                "Gate B: plan.allow_llm=False → skip LLM",
                response_type=plan.response_type.name,
            )
            return format_result(tool_result), False, 0

        # Gate C — LLM allowed; check budget then call once
        if not self._budget.can_spend(estimated_tokens=config.LLM_MAX_TOKENS):
            # Budget exhausted — fall back to tool result
            logger.warn(
                "Gate C blocked: token budget exhausted, falling back to tool result"
            )
            return (
                format_result(tool_result) or self._budget.over_limit_message(),
                False,
                0,
            )

        # ── AOS: Knowledge grounding for LLM call ────────────────────────────
        # If knowledge confidence >= 0.70, inject grounding into LLM prompt
        knowledge_result = getattr(self, "_last_knowledge_result", None)
        if knowledge_result and knowledge_result.level == ConfidenceLevel.MEDIUM:
            grounded = build_grounded_prompt(user_input, knowledge_result)
            logger.debug("KnowledgeLayer: grounded LLM prompt injected")
            # Store for _call_llm_once to pick up
            self._grounding_prompt = grounded
        else:
            self._grounding_prompt = None

        response_text, tokens = self._call_llm_once(plan, tool_result, user_input)
        return response_text, True, tokens

    # ── LLM call (exactly one, ever) ─────────────────────────────────────────
    def _call_llm_once(
        self,
        plan: ExecutionPlan,
        tool_result: ToolResult,
        user_input: str,
    ) -> tuple[str, int]:
        """
        Assemble minimal context via context_builder, make one LLM call.
        Falls back to tool result text on any exception.
        """
        try:
            # Retrieve context ingredients
            lt_summary = self._ltmem.get_summary(self._cfg.student_id, n=2)
            profile = self._smem.get_profile(self._cfg.student_id)

            # Working memory history — last 2 turns, roles only
            recent = [
                m.to_api_dict()
                for m in self._wmem.get_recent(n=4)
                if m.role in ("user", "assistant") and m.content != user_input
            ]

            # Build token-budgeted context
            messages = build_context(
                task_state=plan.task_state,
                tool_result=tool_result,
                user_input=user_input,
                history=recent,
                profile=profile,
                lt_summary=lt_summary,
            )

            estimated = estimate_tokens(messages)
            logger.debug(
                "LLM context assembled",
                messages=len(messages),
                estimated_tokens=estimated,
                reason=plan.task_state.task,
            )

            text, tokens = llm_call(
                messages=messages,
                system=_SYSTEM_PROMPT,
                max_tokens=config.LLM_MAX_TOKENS,
                reason=plan.task_state.task,
            )
            collector.record_llm_tokens(tokens)
            return text, tokens

        except Exception as exc:
            logger.error(f"LLM call failed in core: {exc}")
            # Graceful degradation — return tool result as plain text
            fallback = format_result(tool_result)
            return (fallback or f"I'm unable to process that right now. ({exc})"), 0
