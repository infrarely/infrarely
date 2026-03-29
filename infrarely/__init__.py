"""
infrarely — Reliable Agent Infrastructure
═══════════════════════════════════════════════════════════════════════════════
Production-grade agent framework. One import, zero boilerplate.

Quick Start (3 lines)::

    import infrarely

    agent = infrarely.agent("helper")
    result = agent.run("What is 2+2?")

With tools::

    @infrarely.tool
    def weather(city: str) -> str:
        return f"Sunny in {city}"

    agent = infrarely.agent("bot", tools=[weather])
    result = agent.run("Weather in NYC?")
    print(result.output)

Multi-agent::

    researcher = infrarely.agent("researcher")
    writer = infrarely.agent("writer")
    facts = researcher.run("Find facts about Mars")
    article = writer.run("Write article", context=facts)

Knowledge::

    agent = infrarely.agent("tutor")
    agent.knowledge.add_documents("./notes/")
    result = agent.run("Explain photosynthesis")
    # LLM bypassed if knowledge confidence >= 85%

Observability::

    result = agent.run("task")
    print(result.explain())                    # human-readable trace
    print(agent.health())                      # health report
    print(infrarely.metrics.llm_bypass_rate()) # % tasks without LLM

Configuration::

    infrarely.configure(
        llm_provider="openai",
        api_key="sk-...",
        llm_model="gpt-4o",
    )

    # Or via environment variables:
    # INFRARELY_LLM_PROVIDER=openai
    # INFRARELY_API_KEY=sk-...

Workflow/DAG::

    wf = infrarely.workflow("pipeline", steps=[
        infrarely.step("fetch", fetch_data),
        infrarely.step("process", process, depends_on=["fetch"]),
        infrarely.step("report", generate_report, depends_on=["process"]),
    ])
    results = wf.execute()

Dashboard::

    infrarely.dashboard.start()  # http://localhost:8080

Shutdown::

    infrarely.shutdown()  # clean up all agents and resources
"""

from __future__ import annotations

try:
    from importlib.metadata import (
        PackageNotFoundError as _PkgNotFound,
    )
    from importlib.metadata import (
        version as _pkg_version,
    )

    __version__ = _pkg_version("infrarely")
except Exception:
    __version__ = "0.0.0"

__all__ = [
    # Core
    "agent",
    "Agent",
    "configure",
    "shutdown",
    # Decorators
    "tool",
    "capability",
    "route",
    "ResponseType",
    # Workflow
    "step",
    "workflow",
    "parallel",
    "Workflow",
    # Result & Error
    "Result",
    "Error",
    "ErrorType",
    "GuardrailViolation",
    "guardrail_registry",
    "replay",
    "list_agent_runs",
    "render_terminal",
    "render_html",
    # Knowledge
    "knowledge",
    # Memory
    "AgentMemory",
    # Observability
    "metrics",
    "dashboard",
    "health",
    # State
    "AgentState",
    # Security
    "SecurityPolicy",
    "SecurityGuard",
    "DetectionResult",
    # HITL
    "ApprovalManager",
    "ApprovalRequest",
    "ApprovalRule",
    "ApprovalStatus",
    # Evaluation
    "eval",
    # Versioning
    "versions",
    # Streaming
    "StreamChunk",
    "StreamResult",
    # Context
    "ContextStrategy",
    "ContextWindowManager",
    # Validation
    "SchemaValidator",
    "ValidationResult",
    # Async
    "async_run",
    "async_gather",
    # Sync
    "SyncSource",
    "SyncScheduler",
    # Testing
    "testing",
    # Integrations
    "integrations",
    # Approvals (alias)
    "approvals",
    # Token Tracking
    "TokenTracker",
    "TokenUsage",
    "ModelPricing",
    "token_tracker",
    # Sandbox
    "Sandbox",
    "SandboxGuard",
    "SandboxViolation",
    # Scaling
    "StateBackend",
    "MemoryBackend",
    "SQLiteBackend",
    "create_backend",
    # Compliance
    "ComplianceLog",
    "ComplianceEntry",
    "ActionType",
    "compliance_log",
    # Multi-Tenancy
    "TenantManager",
    "TenantConfig",
    "TenantContext",
    "tenant_manager",
    "create_tenant",
    # Events
    "EventBus",
    "WebhookRegistry",
    "ScheduleManager",
    "event_bus",
    "webhook_registry",
    "schedule_manager",
    # Input Sanitization
    "InputSanitizer",
    "SanitizationPolicy",
    "SanitizationResult",
    "input_sanitizer",
    # Key Rotation
    "KeyManager",
    "KeyRotationEvent",
    "key_manager",
    "rotate_api_key",
    # Tool Execution Sandbox
    "ToolSandboxPolicy",
    "ToolSandboxGuard",
    "ToolSandboxViolation",
    "ToolSandboxResult",
    "tool_sandbox_guard",
    # Self-Healing
    "SelfHealEngine",
    "SelfHealRule",
    "SelfHealTrigger",
    "SelfHealAction",
    "HealEvent",
    "PerformanceWindow",
    "self_heal_engine",
    # ACP
    "ACPMessage",
    "ACPResponse",
    "ACPEndpoint",
    "ACPIdentity",
    "ACPStatus",
    "ACPTransport",
    "ACPAdapter",
    "ACPRegistry",
    "ACPServer",
    "ACPExchange",
    "ACP_VERSION",
    "acp_registry",
    # Marketplace
    "PackageMeta",
    "PackageVersion",
    "PackageStatus",
    "PackageCategory",
    "PackageValidator",
    "InstalledPackage",
    "MarketplaceRegistry",
    "PackageManager",
    "Marketplace",
    "MARKETPLACE_VERSION",
    "marketplace",
    # NL Config
    "NLConfigurator",
    "DescriptionParser",
    "AgentBlueprint",
    "BlueprintCompiler",
    "GuardrailRule",
    "EscalationTrigger",
    "TopicScope",
    "PersonalityProfile",
    "SuggestedTool",
    "SuggestedCapability",
    "KnowledgeSchema",
    "nl_configurator",
    "NLCONFIG_VERSION",
    # Benchmarking
    "BenchmarkTask",
    "BenchmarkSuite",
    "BenchmarkRunner",
    "BenchmarkMetrics",
    "BenchmarkReport",
    "BenchmarkRegistry",
    "FrameworkBaseline",
    "TaskResult",
    "MetricDefinition",
    "run_benchmark",
    "get_benchmark_registry",
    "get_baseline",
    "list_baselines",
    "BENCHMARK_VERSION",
    "benchmark_registry",
]

# ─── Result & Error types ────────────────────────────────────────────────────
# ─── Integrations ─────────────────────────────────────────────────────────────
from infrarely import integrations
from infrarely.agent.state import ResponseType

# ─── Configuration ────────────────────────────────────────────────────────────
from infrarely.core.config import configure, get_config

# ─── Context Window ───────────────────────────────────────────────────────────
from infrarely.core.context import ContextStrategy, ContextWindowManager, OverflowAction

# ─── Decorators ───────────────────────────────────────────────────────────────
from infrarely.core.decorators import capability, tool

# ─── Events & Webhooks ───────────────────────────────────────────────────
from infrarely.core.events import (
    CronParser,
    Event,
    EventBus,
    ScheduleEntry,
    ScheduleManager,
    WebhookRegistry,
    WebhookRoute,
    get_event_bus,
    get_schedule_manager,
    get_webhook_registry,
)
from infrarely.core.guardrails import REGISTRY as guardrail_registry
from infrarely.core.guardrails import GuardrailViolation
from infrarely.core.replay import list_runs as list_agent_runs
from infrarely.core.replay import replay
from infrarely.core.result import Error, ErrorType, Result

# ─── Streaming ────────────────────────────────────────────────────────────────
from infrarely.core.streaming import (
    AsyncStreamIterator,
    StreamChunk,
    StreamIterator,
    StreamResult,
)

# ─── State ────────────────────────────────────────────────────────────────────
from infrarely.internal.state_bridge import AgentState

# ─── Knowledge (module-level singleton) ───────────────────────────────────────
from infrarely.memory import knowledge as _knowledge_module
from infrarely.memory.knowledge import get_knowledge_manager as _get_km

# ─── Memory ───────────────────────────────────────────────────────────────────
from infrarely.memory.memory import AgentMemory

# ─── Observability ────────────────────────────────────────────────────────────
from infrarely.observability.observability import (
    ExecutionTrace,
    HealthReport,
    MetricsCollector,
    TraceStore,
)
from infrarely.observability.observability import (
    get_logger as _get_logger,
)
from infrarely.observability.observability import (
    get_metrics as _get_metrics,
)
from infrarely.observability.trace_renderer import render_html, render_terminal

# ─── Testing ──────────────────────────────────────────────────────────────────
from infrarely.platform import testing

# ─── Evaluation ───────────────────────────────────────────────────────────────
from infrarely.platform.evaluation import EvalNamespace as _EvalNamespace

# ─── HITL (Human-in-the-Loop) ────────────────────────────────────────────────
from infrarely.platform.hitl import (
    ApprovalManager,
    ApprovalRequest,
    ApprovalRule,
    ApprovalStatus,
    HITLGate,
    get_approval_manager,
)

# ─── Multi-Tenancy ────────────────────────────────────────────────────────
from infrarely.platform.multitenancy import (
    AgentLimitExceeded,
    RateLimitExceeded,
    TenantConfig,
    TenantContext,
    TenantError,
    TenantManager,
    TenantNotFound,
    TokenBudgetExceeded,
    get_tenant_manager,
)

# ─── Knowledge Sync ──────────────────────────────────────────────────────────
from infrarely.platform.sync import SyncScheduler, SyncSource

# ─── Token Tracking ───────────────────────────────────────────────────────
from infrarely.platform.token_tracking import (
    ModelPricing,
    TokenTracker,
    TokenUsage,
    get_pricing,
    get_token_tracker,
    set_pricing,
)
from infrarely.platform.tool_sandbox import (
    ToolSandboxGuard as ToolSandboxGuardCls_,
)

# ─── Tool Execution Sandbox ──────────────────────────────────────────
from infrarely.platform.tool_sandbox import (
    ToolSandboxPolicy,
    ToolSandboxResult,
    ToolSandboxViolation,
    get_tool_sandbox_guard,
)

# ─── Validation ───────────────────────────────────────────────────────────────
from infrarely.platform.validation import (
    SchemaValidator,
    ValidationError,
    ValidationResult,
)

# ─── Versioning ───────────────────────────────────────────────────────────────
from infrarely.platform.versioning import VersionManager as _VersionManager
from infrarely.router.intent_classifier import register_route

# ─── Async Runner ─────────────────────────────────────────────────────────────
from infrarely.runtime.async_runner import (
    async_delegate,
    async_gather,
    async_parallel,
    async_run,
)

# ─── Sandbox ──────────────────────────────────────────────────────────────
from infrarely.runtime.sandbox import (
    ResourceMeter,
    Sandbox,
    SandboxViolation,
)
from infrarely.runtime.sandbox import (
    SandboxGuard as SandboxGuardCls,
)

# ─── Scaling ──────────────────────────────────────────────────────────────
from infrarely.runtime.scaling import (
    CoordinationManager,
    MemoryBackend,
    SQLiteBackend,
    StateBackend,
    create_backend,
)

# ─── Workflow ─────────────────────────────────────────────────────────────────
from infrarely.runtime.workflow import Workflow, parallel, step

# ─── Compliance ───────────────────────────────────────────────────────────
from infrarely.security.compliance import (
    ActionType,
    ComplianceEntry,
    ComplianceLog,
    get_compliance_log,
)

# ─── Input Sanitization ──────────────────────────────────────────────
from infrarely.security.input_sanitizer import (
    InputSanitizer,
    SanitizationPolicy,
    SanitizationResult,
    get_input_sanitizer,
)

# ─── Key Rotation ────────────────────────────────────────────────────
from infrarely.security.key_rotation import (
    KeyManager,
    KeyRotationEvent,
    get_key_manager,
    rotate_api_key,
)

# ─── Security ─────────────────────────────────────────────────────────────────
from infrarely.security.security import (
    DetectionResult,
    InjectionType,
    SecurityGuard,
    SecurityPolicy,
    ThreatLevel,
    get_audit_log,
    get_security_guard,
)

ToolSandboxGuard = ToolSandboxGuardCls_

# ─── Self-Healing ─────────────────────────────────────────────────────
# ─── Core (Agent + factories) ────────────────────────────────────────────────
from infrarely.core.agent import Agent, agent
from infrarely.core.agent import shutdown as _shutdown_fn

# ─── Dashboard ────────────────────────────────────────────────────────────────
from infrarely.observability.dashboard import Dashboard as _DashboardClass

# ─── Agent Collaboration Protocol (ACP) ───────────────────────────────
from infrarely.platform.acp import (
    ACP_VERSION,
    ACPAdapter,
    ACPEndpoint,
    ACPExchange,
    ACPFramework,
    ACPIdentity,
    ACPMessage,
    ACPRegistry,
    ACPResponse,
    ACPServer,
    ACPStatus,
    ACPTransport,
    get_acp_registry,
    get_acp_transport,
)

# ─── Agent Performance Benchmarking ────────────────────────────────────────
from infrarely.platform.benchmark import (
    BENCHMARK_VERSION,
    BenchmarkMetrics,
    BenchmarkRegistry,
    BenchmarkReport,
    BenchmarkRunner,
    BenchmarkSuite,
    BenchmarkTask,
    FrameworkBaseline,
    MetricDefinition,
    TaskResult,
    _reset_benchmark_registry,
    get_baseline,
    get_benchmark_registry,
    list_baselines,
    run_benchmark,
)

# ─── Agent Marketplace ─────────────────────────────────────────────────
from infrarely.platform.marketplace import (
    MARKETPLACE_VERSION,
    InstalledPackage,
    Marketplace,
    MarketplaceRegistry,
    PackageCategory,
    PackageManager,
    PackageMeta,
    PackageStatus,
    PackageValidator,
    PackageVersion,
    _reset_marketplace,
    get_marketplace,
)

# ─── Natural Language Agent Configuration ──────────────────────────────────
from infrarely.platform.nlconfig import (
    NLCONFIG_VERSION,
    AgentBlueprint,
    BlueprintCompiler,
    DescriptionParser,
    EscalationTrigger,
    GuardrailRule,
    KnowledgeSchema,
    NLConfigurator,
    PersonalityProfile,
    SuggestedCapability,
    SuggestedTool,
    TopicScope,
    _reset_nl_configurator,
    get_nl_configurator,
)
from infrarely.platform.self_heal import (
    HealEvent,
    PerformanceWindow,
    SelfHealAction,
    SelfHealEngine,
    SelfHealRule,
    SelfHealTrigger,
    get_self_heal_engine,
)

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL CONVENIENCE ACCESSORS
# ═══════════════════════════════════════════════════════════════════════════════

metrics: MetricsCollector = _get_metrics()
knowledge: _knowledge_module.KnowledgeManager = _get_km()
dashboard: _DashboardClass = _DashboardClass()
eval: _EvalNamespace = _EvalNamespace()
versions: _VersionManager = _VersionManager()
approvals: ApprovalManager = get_approval_manager()
security_guard: SecurityGuard = get_security_guard()
audit_log = get_audit_log()
token_tracker: TokenTracker = get_token_tracker()
compliance_log: ComplianceLog = get_compliance_log()
tenant_manager: TenantManager = get_tenant_manager()
event_bus: EventBus = get_event_bus()
webhook_registry: WebhookRegistry = get_webhook_registry()
schedule_manager: ScheduleManager = get_schedule_manager()
input_sanitizer: InputSanitizer = get_input_sanitizer()
key_manager: KeyManager = get_key_manager()
tool_sandbox_guard: ToolSandboxGuardCls_ = get_tool_sandbox_guard()
self_heal_engine: SelfHealEngine = get_self_heal_engine()
acp_registry: ACPRegistry = get_acp_registry()
marketplace: Marketplace = get_marketplace()
nl_configurator: NLConfigurator = get_nl_configurator()
benchmark_registry: BenchmarkRegistry = get_benchmark_registry()

# ── Attach from_description ──────────────────────────────────────────
agent.from_description = Agent.from_description


def health() -> dict:
    """System-wide health summary."""
    from infrarely.core.agent import _all_agents

    agents = _all_agents()
    report = {
        "sdk_version": __version__,
        "agents": len(agents),
        "agent_reports": {},
        "metrics": metrics.export(format="json"),
        "token_tracking": token_tracker.daily_summary(days=1),
        "tenants": tenant_manager.usage_all(),
    }
    for name, ag in agents.items():
        try:
            report["agent_reports"][name] = str(ag.health())
        except Exception as e:
            report["agent_reports"][name] = f"Error: {e}"
    return report


def create_tenant(
    tenant_id: str,
    *,
    config: dict | None = None,
    display_name: str = "",
    model: str = "",
    token_budget: int = 0,
    **kwargs,
) -> TenantContext:
    """Create a tenant. Convenience wrapper for ``tenant_manager.create()``."""
    return tenant_manager.create(
        tenant_id,
        display_name=display_name,
        model=model,
        token_budget=token_budget,
        config=config,
        **kwargs,
    )


def shutdown() -> None:
    """Shut down all agents and clean up InfraRely resources."""
    _shutdown_fn()


def workflow(
    name: str = "default",
    *steps_positional,
    steps: list | None = None,
) -> Workflow:
    """
    Create a workflow (DAG) from steps.

    Example::

        wf = infrarely.workflow("pipeline", steps=[
            infrarely.step("fetch", fetch_data),
            infrarely.step("process", process, depends_on=["fetch"]),
        ])
        results = wf.execute()
    """
    all_steps = list(steps_positional) + (steps or [])
    wf = Workflow(all_steps)
    wf.workflow_id = f"wf_{name}"
    return wf


def route(
    intent: str,
    tool: str,
    triggers: list[str],
    response_type: ResponseType = ResponseType.DETERMINISTIC,
    params: dict | None = None,
) -> None:
    """Register a deterministic routing rule for the intent classifier."""
    register_route(
        intent=intent,
        tool=tool,
        triggers=triggers,
        response_type=response_type,
        params=params,
    )
