# InfraRely SDK — Core Concepts

Understanding the building blocks of the Agent Operating System SDK.

---

## 1. Agents

An **agent** is the central object. It has a name, memory, knowledge, tools, a state machine, and an execution engine.

```python
import infrarely

agent = infrarely.agent("my-agent", description="Does useful things")
```

### Lifecycle

Agents follow a strict state machine:

```
IDLE → PLANNING → EXECUTING → VERIFYING → COMPLETED → IDLE
                                        → FAILED → IDLE
```

- **IDLE** — ready for work
- **PLANNING** — analyzing the goal, selecting strategy
- **EXECUTING** — running tools, knowledge queries, or LLM calls
- **VERIFYING** — checking the result quality
- **COMPLETED / FAILED** — terminal states, auto-returns to IDLE

### Agent Registry

Every agent is globally registered by name. You can retrieve agents anywhere:

```python
agent = infrarely.agent("worker")  # Creates or retrieves by name
```

Call `infrarely.shutdown()` to clean up all agents at the end of your program.

---

## 2. Tools

A **tool** is a Python function decorated with `@infrarely.tool`. Tools are the primary way agents interact with the outside world (APIs, databases, files).

```python
@infrarely.tool(
    description="Fetch weather data",
    retries=3,           # Retry up to 3 times on failure
    timeout=5.0,         # 5-second timeout
    cache=True,          # Cache results
    cache_ttl=300,       # Cache for 5 minutes
    fallback=lambda: "Weather unavailable",
    tags=["weather", "api"],
)
def get_weather(city: str) -> str:
    # Call real API here
    return f"72°F in {city}"
```

### Built-in Resilience

Every tool automatically gets:

| Feature | Behavior |
|---------|----------|
| **Circuit Breaker** | After 3 consecutive failures, stops calling for 30s |
| **Retries** | Configurable retry count with backoff |
| **Timeout** | Thread-based timeout enforcement |
| **Caching** | LRU cache with TTL for deterministic tools |
| **Error Wrapping** | Never raises — returns error dict on failure |

### Tool Registry

Tools are registered globally. When an agent receives a goal, the intent classifier matches it to registered tools by name, description, and parameter names.

---

## 3. Capabilities

A **capability** is a higher-level operation — a named function that represents something the agent "can do." Capabilities are matched when tools don't cover the request.

```python
@infrarely.capability(
    name="summarize",
    description="Summarize text content",
    tags=["text", "nlp"],
)
def summarize_text(text: str, max_length: int = 100) -> str:
    return text[:max_length] + "..."
```

---

## 4. Knowledge

The **knowledge** system lets agents answer questions from your data without calling an LLM.

```python
agent.knowledge.add_data("topic", "Your facts here...")
agent.knowledge.add_documents("label", ["doc1.txt", "doc2.txt"])
```

### Decision Gate

When `agent.run(goal)` is called, the knowledge system returns a **decision**:

| Decision | Meaning | Action |
|----------|---------|--------|
| `bypass_llm` | High-confidence answer from knowledge | Skip LLM entirely |
| `ground_llm` | Partial match — use as context | Send to LLM with knowledge context |
| `low_confidence` | Weak match | Fall through to other strategies |
| `no_knowledge` | No relevant data | Skip knowledge entirely |

This is how the SDK achieves **LLM cost reduction** — most factual queries are answered without any API call.

### How It Works

Knowledge uses a **TF-IDF vector index** built from pure Python (no numpy, no external dependencies). Documents are chunked, vectorized, and searched by cosine similarity.

---

## 5. Memory

**Memory** gives agents persistence across calls and sessions.

```python
# Store
agent.memory.store("key", "value", scope="session")

# Retrieve
value = agent.memory.get("key")  # Returns None if not found

# Search
results = agent.memory.search("keyword")

# Other operations
agent.memory.has("key")       # bool
agent.memory.forget("key")    # delete
agent.memory.list_keys()      # all keys
agent.memory.clear()          # clear all scopes
```

### Scopes

| Scope | Lifetime | Storage | Visible To |
|-------|----------|---------|------------|
| `session` | Until `agent.reset()` | In-memory | This agent |
| `persistent` | Survives restarts | SQLite | This agent |
| `shared` | Survives restarts | SQLite | All agents |

### Shared Memory

Agents can communicate through shared memory:

```python
agent_a.memory.store("status", "done", scope="shared")
status = agent_b.memory.get("status")  # "done"
```

---

## 6. Workflows

A **workflow** is a DAG (directed acyclic graph) of steps that execute in dependency order with automatic parallelization.

```python
wf = infrarely.workflow("my-pipeline",
    infrarely.step("fetch", fetch_fn, required=True),
    infrarely.step("parse", parse_fn, depends_on=["fetch"]),
    infrarely.step("analyze", analyze_fn, depends_on=["fetch"]),  # Runs parallel with parse
    infrarely.step("report", report_fn, depends_on=["parse", "analyze"]),
)

results = wf.execute(initial_input)
```

### Step Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | Unique step identifier |
| `fn` | callable | required | Function to execute |
| `depends_on` | list[str] | `[]` | Steps that must complete first |
| `required` | bool | `True` | If False, failure is non-fatal |
| `timeout` | float | None | Per-step timeout in seconds |
| `fallback` | str | None | Name of fallback step on failure |

### Execution Model

1. Steps are sorted into **levels** via topological sort
2. Steps at the same level run in **parallel** (ThreadPoolExecutor)
3. Required step failure **aborts** the workflow
4. Optional step failure **skips** the step (marked `skipped=True`)
5. Fallback step runs if the primary step fails

### Parallel Execution

Use `infrarely.parallel()` for simple parallel function calls:

```python
results = infrarely.parallel(
    lambda: fetch("url1"),
    lambda: fetch("url2"),
    lambda: fetch("url3"),
)
```

---

## 7. Multi-Agent Patterns

### Delegation

One agent can delegate a task to another:

```python
result = manager.delegate(worker, "Do this task")
```

The worker runs the task and returns a `Result` to the manager.

### Broadcasting

An agent can send a message to all other agents:

```python
@worker.on_message
def handle(from_agent, message, data):
    print(f"Got: {message} from {from_agent}")

manager.broadcast("Phase 1 complete!", data={"phase": 1})
```

### Event System

Hook into agent lifecycle events:

```python
@agent.on("task_start")
def on_start(data):
    print(f"Starting: {data['goal']}")

@agent.on("task_complete")
def on_complete(data):
    print(f"Done in {data['duration_ms']}ms")

@agent.on("error")
def on_error(data):
    print(f"Error: {data['error']}")
```

---

## 8. Observability

### Metrics

```python
summary = infrarely.metrics.summary()
# {
#   "total_tasks": 42,
#   "success_rate": 0.95,
#   "llm_bypass_rate": 0.73,
#   "avg_duration_ms": 120.5,
#   ...
# }
```

### Execution Traces

Every `agent.run()` creates a trace with:
- Goal, duration, success/failure
- Steps executed (knowledge, tool, LLM calls)
- State transitions
- Token usage

```python
traces = agent.get_recent_traces(limit=10)
trace = agent.get_trace(trace_id)
```

### Health Checks

```python
health = infrarely.health()
# {"status": "healthy", "agents": 3, "uptime_seconds": 120, ...}

agent_health = agent.health()
# HealthReport with status, task_count, error_count, etc.
```

### Dashboard

```python
infrarely.dashboard.start(port=8080)  # Web UI with real-time metrics
```

---

## 9. Error Philosophy

**Errors are data, not exceptions.** Agent operations never throw.

```python
result = agent.run("some task")

if not result:
    err = result.error
    print(err.type)        # ErrorType enum (e.g., TOOL_ERROR)
    print(err.message)     # Human-readable message
    print(err.suggestion)  # Auto-generated fix suggestion
    print(err.recovered)   # True if error was auto-recovered
```

### Error Types

| Type | When |
|------|------|
| `TOOL_ERROR` | Tool function failed |
| `LLM_ERROR` | LLM API call failed |
| `TIMEOUT` | Operation timed out |
| `PERMISSION` | Action not allowed |
| `VALIDATION` | Invalid input |
| `KNOWLEDGE_ERROR` | Knowledge system failed |
| `MEMORY_ERROR` | Memory operation failed |
| `WORKFLOW_ERROR` | Workflow step failed |
| `DELEGATION_ERROR` | Agent delegation failed |
| `CONFIG_ERROR` | Bad configuration |
| `UNKNOWN` | Unexpected error |

---

## 10. Configuration

```python
infrarely.configure(
    provider="openai",          # LLM provider
    model="gpt-4o-mini",        # Model name
    api_key="sk-...",           # API key (or use env vars)
    log_level="INFO",           # DEBUG, INFO, WARNING, ERROR
    memory_backend="sqlite",    # "memory" or "sqlite"
    max_retries=3,              # Default retries for tools
    temperature=0.7,            # LLM temperature
)
```

Most settings have sensible defaults. The SDK auto-detects API keys from environment variables:
- `InfraRely_API_KEY` / `InfraRely_PROVIDER`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GROQ_API_KEY`
- `GEMINI_API_KEY`

---

## Next Steps

- [Quickstart](quickstart.md) — get running in 5 minutes
- [API Reference](api_reference.md) — complete API documentation
- [Examples](../examples/) — runnable code samples
