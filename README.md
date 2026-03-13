[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Build Passing](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

---

# InfraRely

**Reliable Agent Infrastructure — Production-grade AI agent framework with zero boilerplate.**

---

## Why InfraRely?

Most AI agents today are unreliable when they move from demos to production.

### Common Problems
- **Non-deterministic execution** — behavior changes between runs, making debugging and incident response difficult
- **Hallucination (tool + response)** — models invent tool names, parameters, outputs, or unsupported claims
- **Poor observability** — limited traces make it hard to explain failures and regressions
- **Fragile multi-agent coordination** — delegation and message passing break under real workload pressure
- **Weak trust and accountability** — decisions are hard to audit, attribute, and defend in production
- **Identity breakdown** — unclear agent identity/permissions lead to unsafe cross-agent actions
- **Memory problems** — stale, conflicting, or ungrounded memory corrupts downstream decisions

InfraRely addresses these failures with infrastructure-first primitives:

- **Deterministic execution contracts** — router-first control flow with frozen plans and explicit fallbacks
- **Capability graphs** — dependency-aware workflows that compile and execute predictably
- **Verification layers** — structural, logical, knowledge, and policy checks on every result
- **Multi-agent runtime** — scheduler, message bus, shared memory, isolation, and deadlock-aware coordination
- **Identity and memory controls** — runtime identity/permissions plus scoped memory discipline for safer coordination

The result is an AI agent framework designed for reliability, auditability, and safe production deployment.

---

## Quick Start

```python
import infrarely

infrarely.configure(llm_provider="openai", api_key="sk-...")

agent = infrarely.agent("helper")
result = agent.run("What is 2+2?")
print(result.output)  # 4 (no LLM call — deterministic math)
```

### Install

```bash
pip install infrarely

# With LLM provider extras:
pip install infrarely[openai]
pip install infrarely[anthropic]
pip install infrarely[all-providers]
```

---

## Features

### Core Framework
- **3-line start** — `import infrarely` → `agent()` → `run()`
- **Errors-as-data** — `Result` objects with `.error`, never bare exceptions
- **LLM-as-last-resort** — Knowledge → Math → Tools → Capabilities → LLM
- **Observable by default** — traces, metrics, health checks on every agent

### 7-Layer Architecture

| Layer | Name | Description |
|-------|------|-------------|
| 1 | **Execution Contracts** | Deterministic routing, frozen execution plans, three-gate LLM isolation |
| 2 | **Capability Graphs** | Multi-step workflows with dependency resolution |
| 3 | **Infrastructure** | Execution depth guard, permissions, tool validation, sandboxing |
| 4 | **Verification** | Structural/logical/knowledge/policy checks on every result |
| 5 | **Adaptive Intelligence** | Self-optimizing routing, failure analysis, token optimization |
| 6 | **Multi-Agent Runtime** | OS-like kernel — scheduler, IPC, shared memory, RBAC, deadlock detection |
| 7 | **Autonomous Evolution** | Performance analysis, A/B testing, architecture proposals with policy guards |

### Tools & Knowledge

```python
@infrarely.tool
def weather(city: str) -> str:
    return f"Sunny in {city}"

agent = infrarely.agent("bot", tools=[weather])
result = agent.run("Weather in NYC?")
```

```python
agent = infrarely.agent("tutor")
agent.knowledge.add_documents("./notes/")
result = agent.run("Explain photosynthesis")
# LLM bypassed if knowledge confidence >= 85%
```

### Multi-Agent

```python
researcher = infrarely.agent("researcher")
writer = infrarely.agent("writer")
facts = researcher.run("Find facts about Mars")
article = writer.run("Write article", context=facts)
```

### Workflows (DAG)

```python
wf = infrarely.workflow("pipeline", steps=[
    infrarely.step("fetch", fetch_data),
    infrarely.step("process", process, depends_on=["fetch"]),
    infrarely.step("report", generate_report, depends_on=["process"]),
])
results = wf.execute()
```

### Streaming

```python
for chunk in agent.stream("Write a poem"):
    print(chunk.text, end="", flush=True)
```

### Security

- Prompt injection defense (7 injection types)
- Input sanitization (always-on)
- API key rotation
- Tool execution sandboxing
- Compliance audit logging

### Human-in-the-Loop

```python
agent.require_approval_for("send_email", auto_approve_after=300)
result = agent.run("Send welcome email")
# Pauses for human approval
```

### CLI

```bash
infrarely run "What is 2+2?"
infrarely health
infrarely metrics
infrarely deploy
infrarely verify
```

---

## InfraRely Architecture

```
                          Applications
                               │
                               │
                        AI Agents Layer
                 (Custom Agents Built by Developers)
                               │
                               │
                    InfraRely Agent Control Plane
            ┌─────────────────────────────────────────┐
            │                                         │
            │  Agent Pipeline                         │
            │  • Planning Engine                      │
            │  • Capability Graph                     │
            │  • Tool Router                          │
            │  • Verification Layer                   │
            │                                         │
            │  Platform Services                      │
            │  • Memory System                        │
            │  • Knowledge Engine                     │
            │  • Workflow DAG Engine                  │
            │  • Capability Registry                  │
            │                                         │
            │  Reliability Systems                    │
            │  • Retry & Circuit Breakers             │
            │  • Token Optimization                   │
            │  • Failure Recovery                     │
            │  • Self-Healing Execution               │
            │                                         │
            │  Observability                          │
            │  • Execution Traces                     │
            │  • Metrics & Telemetry                  │
            │  • Token Budget Monitoring              │
            │                                         │
            │  Security                               │
            │  • Input Sanitization                   │
            │  • Tool Sandbox                         │
            │  • Permission Policies                  │
            │  • Compliance Logging                   │
            └─────────────────────────────────────────┘
                               │
                               │
                       InfraRely Runtime
              (Scheduling, Isolation, State, Scaling)
                               │
                               │
                     External Systems / APIs
       Databases • SaaS APIs • Filesystems • LLM Providers
```

## Architecture

InfraRely is structured as a layered **Agent Operating System**.

1. **Applications**
   - Developer-built AI applications.

2. **Agents**
   - Logical workers that execute tasks and coordinate tools.

3. **InfraRely Control Plane**
   - Planning, routing, verification, and reliability systems.

4. **Runtime**
   - Execution environment responsible for scheduling, isolation, and scalability.

5. **External Systems**
   - APIs, databases, and LLM providers used by agents.

### Project Structure

```
infrarely/
├── core/           # Agent, Result, Config, Events, Decorators, Streaming
├── runtime/        # Workflow DAG, async runner, sandbox, scaling, multi-agent kernel
├── router/         # Rule-based intent classification, tool routing
├── agent/          # Execution pipeline, state machine, planning, verification
├── memory/         # Agent memory, knowledge engine, working/structured/long-term
├── security/       # Prompt injection defense, compliance, input sanitization
├── observability/  # Metrics, traces, logging, dashboard
├── optimization/   # Self-optimizing routing, failure analysis, token optimization
├── learning/       # A/B testing, architecture proposals, policy guards
├── platform/       # HITL, evaluation, versioning, marketplace, multitenancy, ACP
├── tools/          # Tool base classes, registry
├── capabilities/   # Multi-step capability definitions
├── integrations/   # GitHub, Gmail, Slack, Postgres, Notion, Webhooks, REST
├── internal/       # Execution engine bridges (private)
└── cli/            # CLI interface
```

## LLM Providers

| Provider | Model | Setup |
|----------|-------|-------|
| OpenAI | gpt-4o, gpt-4o-mini | `infrarely.configure(llm_provider="openai", api_key="sk-...")` |
| Anthropic | claude-sonnet-4-20250514 | `infrarely.configure(llm_provider="anthropic", api_key="...")` |
| Groq | llama-3.1-8b-instant | `infrarely.configure(llm_provider="groq", api_key="...")` |
| Google Gemini | gemini-1.5-flash | `infrarely.configure(llm_provider="gemini", api_key="...")` |
| Ollama | llama3.2 (local) | `infrarely.configure(llm_provider="ollama")` |

## Configuration

```python
infrarely.configure(
    llm_provider="openai",
    api_key="sk-...",
    llm_model="gpt-4o",
    knowledge_threshold=0.85,
    token_budget=10_000,
    log_level="INFO",
    max_agents=50,
)
```

Or via environment variables:
```bash
export INFRARELY_LLM_PROVIDER=openai
export INFRARELY_API_KEY=sk-...
```

## Documentation

- [Quickstart](docs/quickstart.md)
- [Core Concepts](docs/concepts.md)
- [Architecture](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Vision](docs/vision.md)
- [Runtime](docs/runtime.md)
- [Security Model](docs/security_model.md)
- [Multi-Agent Runtime](docs/multi_agent.md)
- [Verification](docs/verification.md)
- [Observability](docs/observability.md)

## License

MIT License — see [LICENSE](LICENSE) for details.
