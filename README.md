[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/infrarely/infrarely/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/infrarely/infrarely/actions/workflows/ci.yml?query=branch%3Amain)
[![PyPI version](https://img.shields.io/pypi/v/infrarely?color=0066CC&label=PyPI)](https://pypi.org/project/infrarely/)
[![Tests](https://img.shields.io/badge/tests-71%20passing-brightgreen)](https://github.com/infrarely/infrarely/actions)

---

# InfraRely

You shipped an AI agent last week.

It worked in testing.  
It worked in the demo.  
It worked the first three times in production.

Then it didn't.

Same input. Different output. No error. No log. No trace.  
Just a wrong answer, confidently delivered.

You spent 6 hours debugging a system that doesn't tell you what it did or why.

**That's not an AI problem. That's an infrastructure problem.**

---

## You've already felt this

Be honest. Have you ever:

- Run the same prompt twice and gotten different results, with no idea why
- Watched your agent call a tool with parameters you never defined
- Had a failure swallowed silently, only discovered downstream
- Written `try/except` around an entire agent because you didn't trust it
- Logged into production at 2am because your agent "just stopped working"
- Said the words **"let me just re-run it"** and hoped for a different result

If yes, you've already hit the ceiling of how agents are built today.

---

## What's actually broken

Every agent framework today is built around the same idea:

> Give the LLM a prompt, some tools, and hope it makes good decisions.

That worked fine for demos.

In production, "hope" is not a reliability strategy.

**Non-determinism.** The same input gives you different outputs on different runs. You can't reproduce bugs. You can't write stable tests. You can't explain to a stakeholder why the answer changed.

**Silent failures.** The agent doesn't crash, it just does the wrong thing. Calls the wrong tool. Hallucinates a parameter. Returns a confident answer with no basis. You find out when a user complains, not when the system breaks.

**No observability.** You can log the input and the final output. Everything in between is a black box. When something goes wrong, you're reading prompt text trying to reverse-engineer what happened.

**Tool chaos.** You give your agent 8 tools. It calls them in the wrong order. It invents arguments. It calls tools it shouldn't for the context it's in. You add more instructions to the prompt. It gets worse.

**Memory corruption.** One agent's context bleeds into another. Stale state from a previous run influences the current one. You build workarounds. The workarounds have bugs.

**It doesn't get better with more prompting.** You can't instruction-engineer your way out of an infrastructure problem.

---

## The industry's answer is wrong

The current playbook:

```
Agent breaks → add more instructions → test again → breaks differently → add more instructions
```

Every failure becomes a prompt patch.  
Every prompt patch is a hidden dependency.  
Every hidden dependency is a future incident.

You end up with a system that works until someone changes the prompt, rotates the model version, or sends an input you didn't anticipate.

This isn't engineering. It's archaeology, digging through inference outputs hoping to find what went wrong.

---

## InfraRely

InfraRely is **deterministic agent infrastructure**.

Not a new framework.  
Not a better prompt template.  
Not another abstraction over the OpenAI API.

A **control layer** that sits between your application and the LLM, enforcing how agents plan, route, execute, and verify.

The LLM is still there. It's just no longer in charge of everything.

```
Prompt → Hope → Retry → Patch → Repeat       (today)

Define → Control → Execute → Verify → Trust  (InfraRely)
```

---

## Get started in 60 seconds

```bash
pip install infrarely
```

```python
import infrarely

infrarely.configure(llm_provider="openai", api_key="sk-...")

@infrarely.tool(
    route=infrarely.route(
        match=["order", "status", "tracking"],
        required_params=["order_id"],
        param_types={"order_id": str},
    )
)
def get_order_status(order_id: str) -> dict:
    return db.query("SELECT * FROM orders WHERE id = ?", order_id)

agent = infrarely.agent("support-bot", tools=[get_order_status])
result = agent.run("What's the status of order #1042?")

print(result.output)         # "Order #1042 is out for delivery, arriving today."
print(result.error)          # None
print(result.llm_calls)      # 0  — routing was deterministic, LLM never consulted
print(result.trace.render())
```

```
┌─ Run: trace_8f3a21c [61ms]
│  Input: "What's the status of order #1042?"
│
├─ [1] Contract Router [2ms]
│     matched: get_order_status
│     params validated: {"order_id": "1042"} ✓
│     source: deterministic  (LLM not consulted)
│
├─ [2] Tool Execution: get_order_status [54ms]
│     input:  {"order_id": "1042"}
│     output: {"status": "out_for_delivery", "eta": "today"} ✓
│
├─ [3] Output Verification [3ms]
│     hallucination check: passed
│     schema validation:   passed
│
└─ Result: ✓ SUCCESS · 1 tool call · 0 LLM calls · $0.000
```

The routing was deterministic. The LLM was never consulted. That's the point.

---

## How a request moves through InfraRely

Most frameworks hand every request to the LLM immediately. InfraRely runs it through a resolution stack first. The LLM is the last resort, not the first call.

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────┐
│               Routing Contract Resolver              │
│                                                     │
│  Does input match a defined route?                  │
│  Are required params present and correctly typed?   │
│                                                     │
│  YES ───────────────────────────────────────────►  Tool Execution ──► Verification ──► Result
│  NO  ───────────────────────────────────────────►  Knowledge Base
└─────────────────────────────────────────────────────┘
                                │
                                ▼
               ┌────────────────────────────────────┐
               │           Knowledge Base            │
               │                                    │
               │  Does a document answer this       │
               │  with ≥ 85% confidence?            │
               │                                    │
               │  YES ──────────────────────────►  Output Verification ──► Result
               │  NO  ──────────────────────────►  LLM
               └────────────────────────────────────┘
                                │
                                ▼
               ┌────────────────────────────────────┐
               │               LLM                  │
               │    Consulted last, not first.       │
               └──────────────────┬─────────────────┘
                                  │
                                  ▼
               ┌────────────────────────────────────┐
               │         Output Verification        │
               │                                    │
               │  Hallucination check               │
               │  Schema validation                 │
               │  Tool call audit                   │
               └──────────────────┬─────────────────┘
                                  │
                                  ▼
                           Result object
                    (always structured, never
                     a bare exception)
```

Every step is traced. Every decision is logged. No black boxes.

---

## Execution trace anatomy

Every `agent.run()` produces a complete `Trace`. Here is what it captures:

```
Trace
├── run_id            "trace_8f3a21c"
├── duration_ms       61
├── input             "What's the status of order #1042?"
│
├── resolution_path
│   ├── contract_router     matched in 2ms (deterministic)
│   ├── knowledge_base      skipped (contract matched first)
│   └── llm                 skipped (0 calls)
│
├── tool_calls
│   └── [0] get_order_status
│       ├── input           {"order_id": "1042"}
│       ├── output          {"status": "out_for_delivery", ...}
│       ├── duration_ms     54
│       └── validated       true
│
├── verification
│   ├── hallucination_check passed
│   └── schema_validation   passed
│
└── result
    ├── output      "Order #1042 is out for delivery, arriving today."
    ├── error       null
    ├── cost_usd    0.000
    ├── tokens      0
    └── llm_calls   0
```

```python
result.trace.render()      # human-readable tree
result.trace.to_json()     # machine-readable for your logging pipeline
result.trace.export()      # structured for OpenTelemetry / Datadog / Grafana
```

---

## Why InfraRely vs everything else

| Capability | LangChain | LlamaIndex | CrewAI | InfraRely |
|---|:---:|:---:|:---:|:---:|
| Deterministic tool routing | ✗ | ✗ | ✗ | ✓ |
| Routing contracts (pre-LLM) | ✗ | ✗ | ✗ | ✓ |
| Structured Result, always | ✗ | ✗ | ✗ | ✓ |
| Output verification layer | ✗ | ✗ | ✗ | ✓ |
| Full execution traces | Partial | Partial | ✗ | ✓ |
| Memory isolation between agents | ✗ | ✗ | ✗ | ✓ |
| Human-in-the-loop native | ✗ | ✗ | Partial | ✓ |
| Workflow dependency graphs | Partial | Partial | ✓ | ✓ |
| Model-agnostic | ✓ | ✓ | ✓ | ✓ |
| Self-hosted / open-source | ✓ | ✓ | ✓ | ✓ |

LangChain, LlamaIndex, and CrewAI are orchestration frameworks. InfraRely is an infrastructure control layer. The difference is what's guaranteed at runtime, not what's possible with enough prompting.

---

## Core capabilities

### Deterministic routing contracts

Your agent stops guessing which tool to call. Tools are matched to intent by user-defined rules before the LLM is consulted.

```python
@infrarely.tool(
    route=infrarely.route(
        match=["order", "status", "tracking"],
        required_params=["order_id"],
        param_types={"order_id": str},
    )
)
def get_order_status(order_id: str) -> dict:
    return db.query("SELECT * FROM orders WHERE id = ?", order_id)

agent = infrarely.agent("support-bot", tools=[get_order_status])
result = agent.run("What's the status of order #1042?")
# Routing is deterministic
# Parameters validated against the contract
# No hallucinated parameters — contract violation = structured error
```

### Workflow dependency graphs

Stop chaining prompts and hoping context carries through. Define actual dependency graphs.

```python
wf = infrarely.workflow("report-pipeline", steps=[
    infrarely.step("fetch",   fetch_data),
    infrarely.step("process", clean_and_transform, depends_on=["fetch"]),
    infrarely.step("report",  generate_report,     depends_on=["process"]),
])

results = wf.execute()
# Steps execute in order, dependencies are guaranteed, failures are isolated
```

### Knowledge-first resolution

```python
agent = infrarely.agent("docs-agent")
agent.knowledge.add_documents("./product-docs/")

result = agent.run("What's the refund policy?")
# Answer comes from your documents, not from what the LLM guesses
# LLM is only called if document confidence is below 85%
```

### Multi-agent coordination without chaos

```python
researcher = infrarely.agent("researcher")
writer     = infrarely.agent("writer")

facts   = researcher.run("Summarize Q3 metrics")
article = writer.run("Draft exec summary", context=facts)
# Isolated agents, explicit message passing, no memory bleed
```

### Human-in-the-loop, native

```python
agent.require_approval_for("send_email", auto_approve_after=300)

result = agent.run("Send onboarding email to new users")
# Execution pauses. Waits for approval. Resumes on confirmation.
```

---

## Result object

Every `agent.run()` returns a structured `Result`. Always. Whether it succeeded or failed.

```python
result = agent.run("What's the status of order #1042?")

result.output        # str           — the agent's response
result.error         # str | None    — None on success, structured message on failure
result.trace         # Trace         — full execution trace
result.cost          # float         — USD cost of this run
result.tokens        # int           — total tokens used
result.tool_calls    # list[ToolCall]— tools invoked, in order
result.llm_calls     # int           — how many times LLM was actually consulted
result.duration_ms   # int           — total wall-clock time
```

No bare exceptions. No silent wrong answers. Errors are data.

---

## Full system architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                           Applications                                │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │
┌───────────────────────────────────▼──────────────────────────────────┐
│                               Agents                                  │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │
┌───────────────────────────────────▼──────────────────────────────────┐
│                       InfraRely Control Plane                         │
│                                                                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │  Planning Engine │  │ Capability Graph  │  │   Tool Router    │   │
│  │                  │  │                  │  │  (deterministic) │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘   │
│                                                                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │  Verification    │  │  Memory System   │  │    Security      │   │
│  │  Layer           │  │  (isolated)      │  │  injection       │   │
│  │                  │  │                  │  │  sandboxing      │   │
│  │                  │  │                  │  │  audit logs      │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘   │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │           Observability: Traces · Metrics · Telemetry         │   │
│  └───────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │
┌───────────────────────────────────▼──────────────────────────────────┐
│                 Runtime: Scheduling · Isolation · Scaling             │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │
┌───────────────────────────────────▼──────────────────────────────────┐
│                    External APIs · LLMs · Databases                   │
└──────────────────────────────────────────────────────────────────────┘
```

The LLM sits at the bottom, consulted only after every other resolution path is exhausted.

---

## Live observability

```bash
infrarely metrics   # live agent performance across all runs
infrarely health    # system status — providers, tools, memory
infrarely verify    # run verification checks against your agents
```

---

## LLM providers

InfraRely is model-agnostic. Swap providers with a single config change, zero code changes.

| Provider | Models |
|---|---|
| OpenAI | gpt-4o, gpt-4o-mini |
| Anthropic | claude-sonnet-4-20250514 |
| Groq | meta-llama/llama-4-scout-17b-16e-instruct |
| Google Gemini | gemini-1.5-flash |
| Ollama | local models (llama3.2, etc.) |

```python
infrarely.configure(
    llm_provider="anthropic",
    api_key="sk-ant-...",
    knowledge_threshold=0.85,
    token_budget=10_000,
)
```

---

## Install

```bash
pip install infrarely

pip install infrarely[openai]
pip install infrarely[anthropic]
pip install infrarely[all-providers]
```

**Requirements:** Python 3.10+

---

## Documentation

| Topic | Link |
|---|---|
| Quickstart | [docs/quickstart.md](./docs/quickstart.md) |
| Core concepts | [docs/concepts.md](./docs/concepts.md) |
| API reference | [docs/api_reference.md](./docs/api_reference.md) |
| Architecture | [docs/architecture.md](./docs/architecture.md) |
| Observability & traces | [docs/observability.md](./docs/observability.md) |
| Verification & validation | [docs/verification.md](./docs/verification.md) |
| Multi-agent orchestration | [docs/multi_agent.md](./docs/multi_agent.md) |
| Security model | [docs/security_model.md](./docs/security_model.md) |
| Runtime environment | [docs/runtime.md](./docs/runtime.md) |
| Migration guides | [docs/migration/](./docs/migration/) |
| Project vision | [docs/vision.md](./docs/vision.md) |

Full documentation at [infrarely.com/docs](https://infrarely.com/docs).

---

## Contributing

If you care about reliability over hype, control over magic, and software that behaves like software, contributions are welcome.

Read [CONTRIBUTING.md](./CONTRIBUTING.md) for setup and guidelines.  
Look for [`good first issue`](https://github.com/infrarely/infrarely/labels/good%20first%20issue) labels to get started.  
Open a [GitHub Discussion](https://github.com/infrarely/infrarely/discussions) before large changes.  
All PRs run against 71 tests. New capabilities need new tests.

---

## License

[MIT](https://opensource.org/licenses/MIT)
