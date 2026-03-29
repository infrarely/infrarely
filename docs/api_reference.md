# InfraRely SDK — API Reference

Complete reference for all public APIs in the `infrarely` package.

---

## Module: `infrarely`

The top-level module. Import everything from here:

```python
import infrarely
```

### `infrarely.agent(name, *, description="", tools=None, capabilities=None) → Agent`

Create or retrieve a named agent.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | Unique agent name |
| `description` | str | `""` | Human-readable description |
| `tools` | list[callable] | `None` | Pre-registered tool functions |
| `capabilities` | list[callable] | `None` | Pre-registered capabilities |

Returns the same agent instance if called with an existing name.

---

### `infrarely.configure(**kwargs) → None`

Set global SDK configuration. Call once at startup.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `llm_provider` | str | `"openai"` | LLM provider: `"openai"`, `"anthropic"`, `"groq"`, `"gemini"`, `"ollama"` |
| `provider` | str | — | Alias for `llm_provider` (convenience) |
| `llm_model` | str | auto | Model name (auto-detected from provider) |
| `model` | str | — | Alias for `llm_model` (convenience) |
| `api_key` | str | auto | API key (auto-detected from env) |
| `log_level` | str | `"WARNING"` | Logging level |
| `memory_backend` | str | `"memory"` | `"memory"` or `"sqlite"` |
| `max_retries` | int | `2` | Default tool retry count |
| `temperature` | float | `0.7` | LLM temperature |
| `base_url` | str | `None` | Custom API base URL |

---

### `infrarely.shutdown() → None`

Shut down all registered agents and clean up resources.

---

### `infrarely.health() → dict`

Global health check across all agents.

Returns:
```python
{
    "status": "healthy",      # "healthy", "degraded", or "unhealthy"
    "agents": 3,              # Number of active agents
    "uptime_seconds": 120.5,
    "metrics": {
        "total_tasks": 42,
        "success_rate": 0.95,
        "llm_bypass_rate": 0.73,
    }
}
```

---

### `infrarely.metrics` — `MetricsCollector`

Global metrics singleton. See [MetricsCollector](#metricscollector).

### `infrarely.knowledge` — `KnowledgeManager`

Global knowledge singleton. See [KnowledgeManager](#knowledgemanager).

### `infrarely.dashboard` — `Dashboard`

Global dashboard singleton. See [Dashboard](#dashboard).

---

## Decorators

### `@infrarely.tool(**kwargs) → callable`

Register a function as a tool.

```python
@infrarely.tool(
    description="Fetch weather",
    retries=3,
    timeout=5.0,
    cache=True,
    cache_ttl=300,
    fallback=lambda: "default",
    deterministic=False,
    tags=["weather"],
)
def get_weather(city: str) -> str:
    return f"72°F in {city}"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `description` | str | `""` | Tool description for intent matching |
| `retries` | int | `0` | Max retry attempts |
| `timeout` | float | `None` | Timeout in seconds |
| `cache` | bool | `False` | Enable result caching |
| `cache_ttl` | int | `60` | Cache TTL in seconds |
| `fallback` | callable | `None` | Fallback function on failure |
| `deterministic` | bool | `False` | Hint for optimizer |
| `tags` | list[str] | `[]` | Tags for categorization |

**Circuit breaker** is always active (threshold=3 failures, 30s recovery).

---

### `@infrarely.capability(**kwargs) → callable`

Register a higher-level capability.

```python
@infrarely.capability(
    name="summarize",
    description="Summarize text",
    tags=["nlp"],
)
def summarize(text: str) -> str:
    return text[:100] + "..."
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | function name | Capability name |
| `description` | str | `""` | Description for matching |
| `tags` | list[str] | `[]` | Tags for categorization |

---

## Workflow API

### `infrarely.step(name, fn, **kwargs) → Step`

Create a workflow step.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | Unique step name |
| `fn` | callable | required | Function to execute |
| `depends_on` | list[str] | `[]` | Dependency step names |
| `required` | bool | `True` | Whether failure aborts workflow |
| `timeout` | float | `None` | Step timeout in seconds |
| `fallback` | str | `None` | Name of fallback step |

---

### `infrarely.workflow(name, *steps) → Workflow`

Create a DAG workflow from steps.

```python
wf = infrarely.workflow("pipeline",
    infrarely.step("a", fn_a),
    infrarely.step("b", fn_b, depends_on=["a"]),
)
results = wf.execute(initial_input)
```

---

### `Workflow.execute(initial_input=None) → dict[str, StepResult]`

Execute the workflow. Returns a dict mapping step name to `StepResult`.

---

### `infrarely.parallel(*fns) → list`

Execute functions in parallel. Returns list of results.

```python
results = infrarely.parallel(fn1, fn2, fn3)
```

---

## Class: `Agent`

### `agent.run(goal, context=None) → Result`

Execute a goal. The agent pipeline:
1. Knowledge check → may bypass LLM
2. Intent classification → route to tool/capability
3. Math evaluation → deterministic math
4. Tool/capability execution
5. LLM synthesis (last resort)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `goal` | str | required | What to accomplish |
| `context` | any | `None` | Additional context |

Returns: `Result`

---

### `agent.delegate(other_agent, task, context=None) → Result`

Delegate a task to another agent.

---

### `agent.broadcast(message, data=None) → int`

Send a message to all other agents. Returns number of recipients.

---

### `agent.on_message(handler) → None`

Register a broadcast message handler.

```python
@agent.on_message
def handle(from_agent, message, data):
    print(f"{from_agent}: {message}")
```

---

### `agent.on(event, handler) → None`

Register an event handler. Can be used as a decorator.

Events: `"task_start"`, `"task_complete"`, `"error"`

```python
@agent.on("task_complete")
def done(data):
    print(f"Done in {data['duration_ms']}ms")
```

---

### `agent.add_tool(fn) → None`

Add a tool function at runtime.

### `agent.add_capability(fn) → None`

Add a capability function at runtime.

---

### `agent.health() → HealthReport`

Agent-level health report.

```python
report = agent.health()
report.agent_name      # Agent name
report.state           # Current FSM state (e.g. "IDLE")
report.total_tasks     # Total tasks run
report.failed_tasks    # Total failures
report.uptime_seconds  # Seconds since creation
```

---

### `agent.get_trace(trace_id) → ExecutionTrace`

Get a specific execution trace by ID.

### `agent.get_recent_traces(limit=10) → list[ExecutionTrace]`

Get recent execution traces.

### `agent.explain() → str`

Human-readable summary of the agent's current status, capabilities, and recent activity.

---

### `agent.memory` — `AgentMemory`

Per-agent memory instance. See [AgentMemory](#agentmemory).

### `agent.knowledge` — `KnowledgeManager`

Per-agent knowledge instance. See [KnowledgeManager](#knowledgemanager).

---

### `agent.reset() → None`

Reset agent state: clears session memory, resets state machine to IDLE.

### `agent.shutdown() → None`

Shut down this agent and remove from global registry.

---

## Class: `Result`

Returned by `agent.run()`. Never raises exceptions.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `output` | any | The result value |
| `success` | bool | Whether the operation succeeded |
| `confidence` | float | 0.0 to 1.0 confidence score |
| `used_llm` | bool | Whether an LLM was called |
| `sources` | list[str] | Where the answer came from |
| `duration_ms` | float | Execution time in ms |
| `trace_id` | str | Unique trace identifier |
| `error` | Error \| None | Error details if failed |

### Methods

- `bool(result)` — returns `result.success`
- `str(result)` — returns string of `result.output`
- `result.explain()` — formatted execution summary

---

## Class: `Error`

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `type` | ErrorType | Error category enum |
| `message` | str | Human-readable message |
| `step` | str | Which step failed |
| `recovered` | bool | Whether auto-recovery succeeded |
| `suggestion` | str | Auto-generated fix suggestion |

### `ErrorType` Enum

`TOOL_FAILURE`, `PLAN_INVALID`, `KNOWLEDGE_GAP`, `VERIFICATION_FAILED`, `BUDGET_EXCEEDED`, `TIMEOUT`, `STATE_CORRUPTED`, `PERMISSION_DENIED`, `CONFIGURATION_ERROR`, `DELEGATION_FAILED`, `SECURITY_VIOLATION`, `VALIDATION`, `APPROVAL_TIMEOUT`, `APPROVAL_REJECTED`, `UNKNOWN`

---

## Class: `AgentMemory`

### `memory.store(key, value, scope="session") → None`

Store a value. Scope: `"session"`, `"persistent"`, `"shared"`.

### `memory.get(key, default=None) → any`

Retrieve a value. Checks session → persistent → shared.

### `memory.forget(key) → None`

Delete a key from all scopes.

### `memory.has(key) → bool`

Check if a key exists.

### `memory.list_keys() → list[str]`

List all keys across all scopes.

### `memory.search(query) → list[MemorySearchResult]`

Full-text search across memory entries.

### `memory.clear() → None`

Clear all scopes. Takes no arguments.

---

## Class: `KnowledgeManager`

### `knowledge.add_data(label, text) → None`

Add raw text data as a knowledge source.

### `knowledge.add_documents(label, paths) → None`

Add documents (file paths) as knowledge.

### `knowledge.add_database(label, connection_string) → None`

Register a database as a knowledge source.

### `knowledge.add_api(label, url, headers=None) → None`

Register an API endpoint as a knowledge source.

### `knowledge.query(question) → KnowledgeResult`

Query knowledge. Returns:

```python
result.confidence    # float — 0.0 to 1.0
result.decision      # str — "bypass_llm", "ground_llm", "low_confidence", "no_knowledge"
result.source_names  # list[str] — source labels
result.chunks        # list[KnowledgeChunk] — matched chunks
result.query         # str — the original query
result.duration_ms   # float — query time in ms
```

**Decision thresholds** (default threshold = 0.85, from `knowledge_threshold` config):
- `confidence >= threshold` → `bypass_llm` (answer directly from knowledge)
- `confidence >= 0.25` → `ground_llm` (use knowledge to ground LLM response)
- `confidence < 0.25` → `low_confidence` (LLM answers independently)

---

## Class: `MetricsCollector`

### `metrics.summary() → dict`

Get metrics summary:

```python
{
    "total_tasks": 42,
    "successful_tasks": 40,
    "failed_tasks": 2,
    "success_rate": 0.952,
    "llm_calls": 10,
    "llm_bypass_rate": 0.762,
    "avg_duration_ms": 120.5,
    "total_tokens": 5000,
}
```

### `metrics.record_task(success, duration_ms, used_llm, tokens=0) → None`

Record a completed task.

### `metrics.export(format="json") → str`

Export metrics as JSON string.

---

## Class: `Dashboard`

### `dashboard.start(port=8080) → None`

Start the web dashboard (dark theme, auto-refreshing).

### `dashboard.stop() → None`

Stop the dashboard server.

---

## Class: `ExecutionTrace`

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `trace_id` | str | Unique identifier |
| `goal` | str | The goal that was executed |
| `success` | bool | Whether it succeeded |
| `duration_ms` | float | Total duration |
| `steps` | list[TraceStep] | Execution steps |
| `llm_calls` | list[TraceLLMCall] | LLM API calls made |
| `knowledge_queries` | list[TraceKnowledgeQuery] | Knowledge lookups |
| `state_transitions` | list[TraceStateTransition] | State changes |
| `tool_calls` | list[TraceToolCall] | Tool invocations |

---

## Class: `HealthReport`

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `agent_name` | str | Agent name |
| `state` | str | Current state machine state |
| `total_tasks` | int | Total tasks run |
| `successful_tasks` | int | Tasks that succeeded |
| `failed_tasks` | int | Tasks that failed |
| `avg_duration_ms` | float | Average task duration (ms) |
| `uptime_seconds` | float | Time since creation |
| `memory_entries` | int | Number of memory entries |
| `tools_registered` | int | Number of tools |
| `llm_calls_total` | int | Total LLM API calls |
| `knowledge_queries_total` | int | Knowledge queries made |
| `circuit_breakers_open` | int | Open circuit breakers |
| `last_error` | str | Most recent error message |

---

## Class: `StepResult`

Returned in workflow execution results dict.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `output` | any | Step result value |
| `success` | bool | Whether step succeeded |
| `duration_ms` | float | Step duration |
| `skipped` | bool | True if optional step was skipped |
| `error` | str \| None | Error message if failed |
