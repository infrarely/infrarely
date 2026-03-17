[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Build Passing](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

---

# InfraRely

You shipped an AI agent last week.

It worked in testing.  
It worked in the demo.  
It worked the first three times in production.

Then it didn't.

Same input. Different output. No error. No log. No trace.  
Just a wrong answer — confidently delivered.

You spent 6 hours debugging a system that doesn't tell you what it did or why.

**That's not an AI problem. That's an infrastructure problem.**

---

## You've already felt this

Be honest. Have you ever:

- Run the same prompt twice and gotten different results — and had no idea why
- Watched your agent call a tool with parameters you never defined
- Had a failure swallowed silently, only discovered downstream
- Written `try/except` around an entire agent because you didn't trust it
- Logged into production at 2am because your agent "just stopped working"
- Said the words **"let me just re-run it"** and hoped for a different result

If yes — you've already hit the ceiling of how agents are built today.

---

## What's actually broken

Every agent framework today is built around the same idea:

> Give the LLM a prompt, some tools, and hope it makes good decisions.

That worked fine for demos.

In production, "hope" is not a reliability strategy.

Here's what really happens:

**Non-determinism.** The same input gives you different outputs on different runs. You can't reproduce bugs. You can't write stable tests. You can't explain to a stakeholder why the answer changed.

**Silent failures.** The agent doesn't crash — it just does the wrong thing. Calls the wrong tool. Hallucinates a parameter. Returns a confident answer with no basis. You find out when a user complains, not when the system breaks.

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

This isn't engineering. It's archaeology — digging through inference outputs hoping to find what went wrong.

---

## InfraRely

InfraRely is **deterministic agent infrastructure**.

Not a new framework.  
Not a better prompt template.  
Not another abstraction over the OpenAI API.

A **control layer** — between your application and the LLM — that enforces how agents plan, route, execute, and verify.

The LLM is still there. It's just no longer in charge of everything.

```
Prompt → Hope → Retry → Patch → Repeat  (today)

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

agent = infrarely.agent("helper")
result = agent.run("What is 2+2?")

print(result.output)   # 4  — no LLM call, deterministic math
print(result.trace)    # full execution trace, always
print(result.error)    # None — errors are data, never exceptions
```

That last line matters. **Errors are data.** You always get a structured result back — success or failure. No bare exceptions. No silent wrong answers. No guessing.

---

## What changes

### Your agent stops guessing which tool to call

InfraRely uses a router-first execution model. Tools are matched to intent by structured rules before the LLM is ever consulted. The LLM is the last resort — not the first.

```python
@infrarely.tool
def get_order_status(order_id: str) -> dict:
    return db.query("SELECT * FROM orders WHERE id = ?", order_id)

agent = infrarely.agent("support-bot", tools=[get_order_status])
result = agent.run("What's the status of order #1042?")
# Tool is called with validated, typed params — not hallucinated ones
```

### Your workflows stop being prompt chains

Stop chaining prompts and hoping context carries through. Define actual dependency graphs.

```python
wf = infrarely.workflow("report-pipeline", steps=[
    infrarely.step("fetch",   fetch_data),
    infrarely.step("process", clean_and_transform, depends_on=["fetch"]),
    infrarely.step("report",  generate_report,     depends_on=["process"]),
])

results = wf.execute()
# Steps execute in order, dependencies guaranteed, failures isolated
```

### Your knowledge base is consulted before the LLM

```python
agent = infrarely.agent("docs-agent")
agent.knowledge.add_documents("./product-docs/")

result = agent.run("What's the refund policy?")
# Answer comes from your documents — not from what the LLM guesses
# LLM is only called if confidence < 85%
```

### Multi-agent coordination without chaos

```python
researcher = infrarely.agent("researcher")
writer     = infrarely.agent("writer")

facts   = researcher.run("Summarize Q3 metrics")
article = writer.run("Draft exec summary", context=facts)
# Isolated agents, explicit message passing, no memory bleed
```

### You can ship human-in-the-loop without hacking

```python
agent.require_approval_for("send_email", auto_approve_after=300)
result = agent.run("Send onboarding email to new users")
# Execution pauses. Waits for approval. Resumes on confirmation.
```

---

## Everything is traced. Always.

Every agent run produces a full execution trace — what was considered, what was routed, what was called, what was verified, what was returned.

No more reading logs hoping to piece together what happened.  
No more re-running to reproduce a failure.  
No more "it works on my machine."

```bash
infrarely metrics   # live agent performance
infrarely health    # system status
infrarely verify    # run verification checks
```

---

## Architecture

```
Applications
     |
  Agents
     |
InfraRely Control Plane
  |-- Planning Engine
  |-- Capability Graph
  |-- Tool Router
  |-- Verification Layer
  |-- Memory System
  |-- Security (injection defense, sandboxing, audit logs)
  |-- Observability (traces, metrics, telemetry)
     |
Runtime (Scheduling, Isolation, Scaling)
     |
External APIs / LLMs / Databases
```

The LLM sits at the bottom — consulted only after every other resolution path is exhausted.

---

## LLM Providers

| Provider | Models |
|---|---|
| OpenAI | gpt-4o, gpt-4o-mini |
| Anthropic | claude-sonnet-4-20250514 |
| Groq | llama-3.1-8b-instant |
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

---

## Status

Early stage. Actively built.

If you've been burned by the problems above — try it, break it, and tell me what's missing.  
Every piece of feedback from the first 50 developers will directly shape what gets built next.

Open an issue. Start a discussion.

---

## Contributing

If you care about reliability over hype, control over magic, and software that behaves like software — this is worth contributing to.

---

## License

[MIT](https://opensource.org/licenses/MIT)
