# InfraRely SDK — Quickstart Guide

Get from zero to a working AI agent in under 5 minutes.

## Installation

```bash
pip install infrarely
```

Or install from source:

```bash
git clone <repo-url>
cd infrarely
pip install -e .
```

## Your First Agent (3 Lines)

```python
import infrarely

agent = infrarely.agent("my-agent")
result = agent.run("What is 42 * 17?")
print(result)  # 714
```

That's it. No API keys, no configuration, no boilerplate.

## How It Works

1. `infrarely.agent("my-agent")` creates a named agent with memory, state machine, and execution engine.
2. `agent.run(goal)` processes your request through a smart pipeline:
   - **Knowledge check** — answers from your data first (no LLM cost)
   - **Math evaluation** — deterministic math (no LLM needed)
   - **Tool routing** — calls your registered tools when relevant
   - **LLM synthesis** — only calls an LLM as last resort
3. The result is always a `Result` object — never an exception.

## Adding Tools

Wrap any function with `@infrarely.tool`:

```python
import infrarely

@infrarely.tool(description="Get current weather for a city")
def weather(city: str) -> str:
    return f"72°F and sunny in {city}"

agent = infrarely.agent("weather-bot", tools=[weather])
result = agent.run("What's the weather in Austin?")
print(result.output)   # "72°F and sunny in Austin"
print(result.used_llm)  # False — answered from tool, not LLM
```

Tools automatically get:
- **Retries** — configurable retry logic
- **Circuit breaker** — stops calling broken tools
- **Caching** — LRU + TTL cache for deterministic tools
- **Error wrapping** — tools never throw; errors become data

## Adding Knowledge

Give your agent facts to answer from:

```python
import infrarely

agent = infrarely.agent("study-buddy")

agent.knowledge.add_data("biology",
    "Mitochondria are the powerhouses of the cell. "
    "They generate most of the cell's ATP through oxidative phosphorylation."
)

result = agent.run("What do mitochondria do?")
print(result.output)      # Answer grounded in your data
print(result.used_llm)    # May be False if knowledge was sufficient
print(result.confidence)  # 0.0 to 1.0
```

## Using Memory

Agents remember things across calls:

```python
agent = infrarely.agent("assistant")

# Store facts
agent.memory.store("user_name", "Alice")
agent.memory.store("preference", "dark mode")

# Retrieve later
name = agent.memory.get("user_name")  # "Alice"

# Search across memory
results = agent.memory.search("preference")
```

Memory has three scopes:
- **session** (default) — cleared on reset
- **persistent** — survives restarts (SQLite)
- **shared** — visible to all agents

## Building Workflows

Chain tools into DAG workflows:

```python
import infrarely

@infrarely.tool()
def fetch_data(query: str) -> dict:
    return {"data": [1, 2, 3]}

@infrarely.tool()
def analyze(data: dict) -> str:
    return f"Found {len(data.get('data', []))} items"

@infrarely.tool()
def report(analysis: str) -> str:
    return f"Report: {analysis}"

wf = infrarely.workflow("pipeline",
    infrarely.step("fetch", fetch_data, required=True),
    infrarely.step("analyze", analyze, depends_on=["fetch"]),
    infrarely.step("report", report, depends_on=["analyze"]),
)

results = wf.execute({"query": "test data"})
print(results["report"].output)  # "Report: Found 3 items"
```

Steps run in parallel when their dependencies allow it.

## Multi-Agent Collaboration

```python
import infrarely

researcher = infrarely.agent("researcher")
writer = infrarely.agent("writer")

# Delegate work
result = researcher.delegate(writer, "Summarize this topic")

# Broadcast messages
researcher.broadcast("Data collection complete!")
```

## Using an LLM

For tasks that need actual AI reasoning, configure a provider:

```python
import infrarely

infrarely.configure(
    llm_provider="openai",     # or "anthropic", "groq", "gemini", "ollama"
    api_key="sk-...",          # or set OPENAI_API_KEY env var
    llm_model="gpt-4o-mini",   # auto-detected if omitted
)
# Shorthand aliases also work: provider="openai", model="gpt-4o-mini"

agent = infrarely.agent("smart-agent")
result = agent.run("Explain quantum entanglement in simple terms")
print(result.output)
```

Or use environment variables (auto-detected):

```bash
export OPENAI_API_KEY=sk-...
```

```python
import infrarely
agent = infrarely.agent("smart-agent")
result = agent.run("Explain quantum entanglement")  # Just works
```

## Error Handling

Results never throw. Check success with `if result`:

```python
result = agent.run("something that might fail")

if result:
    print(result.output)
else:
    print(f"Failed: {result.error.message}")
    print(f"Suggestion: {result.error.suggestion}")
```

## Observability

```python
# Health check
health = infrarely.health()
print(health["status"])  # "healthy"

# Metrics
metrics = infrarely.metrics.summary()
print(f"LLM bypass rate: {metrics['llm_bypass_rate']:.0f}%")

# Execution traces
traces = agent.get_recent_traces(limit=5)
for t in traces:
    print(f"{t.goal} → {'OK' if t.success else 'FAIL'} ({t.duration_ms:.0f}ms)")
```

## Next Steps

- [Core Concepts](concepts.md) — understand agents, tools, knowledge, memory, workflows
- [API Reference](api_reference.md) — complete API documentation
- [Examples](../examples/) — runnable example scripts
