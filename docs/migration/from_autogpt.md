# Migrating from AutoGPT to InfraRely

## Why Switch?

| Feature | AutoGPT | InfraRely |
|---------|---------|-----|
| Setup complexity | Docker + config files | `pip install infrarely` |
| Execution model | Autonomous loop | Structured pipeline |
| Cost control | Hard to predict | Token budget + LLM bypass |
| Error handling | Crashes/loops | Structured Results |
| Observability | Console logs | Full traces + dashboard |
| State management | File-based | SQLite + memory scopes |

---

## Side-by-Side: Autonomous Task

### AutoGPT
```python
# Requires Docker, .env file, config.yaml
# agents/<agent_name>/settings.yaml:
# ai_name: ResearchAgent
# ai_role: Research assistant
# ai_goals:
#   - Research the topic of quantum computing
#   - Write a summary document

# Run via CLI:
# docker compose run --rm auto-gpt --continuous
# → Agent loops until task complete (or forever)
# → Unpredictable cost
# → No programmatic control
```

### InfraRely
```python
import infrarely

researcher = infrarely.agent("researcher")
researcher.knowledge.add_documents("./quantum_docs/")

result = researcher.run("Research quantum computing and write a summary")
print(result.output)
print(f"Cost: {'$0' if not result.used_llm else 'LLM call made'}")
print(f"Confidence: {result.confidence}")
print(f"Duration: {result.duration_ms}ms")
```

---

## Side-by-Side: Tool Usage

### AutoGPT
```yaml
# ai_settings.yaml
plugins:
  - AutoGPTWebSearch
  - AutoGPTFileOps

# Plugin configuration in .env
GOOGLE_API_KEY=...
CUSTOM_SEARCH_ENGINE_ID=...
```

### InfraRely
```python
import infrarely

@infrarely.tool
def web_search(query: str) -> str:
    """Search the web."""
    return api.search(query)

@infrarely.tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    with open(path, "w") as f:
        f.write(content)
    return f"Written to {path}"

agent = infrarely.agent("assistant", tools=[web_search, write_file])
```

---

## Side-by-Side: Memory / State

### AutoGPT
```
# Memory managed via Pinecone or local JSON
# Configuration in .env:
MEMORY_BACKEND=pinecone
PINECONE_API_KEY=...

# No programmatic access to memory state
```

### InfraRely
```python
import infrarely

agent = infrarely.agent("assistant")

# Memory is built-in
agent.memory.store("user_name", "Alice")
agent.memory.store("preferences", {"theme": "dark"})

name = agent.memory.get("user_name")  # "Alice"

# Export/import memory
snapshot = agent.memory.export()
```

---

## Key Differences

1. **AutoGPT runs autonomously** — it loops until it decides it's done (or runs forever). InfraRely executes a single goal and returns.

2. **AutoGPT has unpredictable costs** — each loop iteration makes LLM calls. InfraRely bypasses LLM for factual queries (knowledge match, math, tool calls).

3. **AutoGPT requires Docker** — InfraRely is `pip install infrarely`.

4. **AutoGPT has no programmatic API** — it's a CLI tool. InfraRely is a Python library you import and call.

---

## Migration Checklist

- [ ] Remove Docker setup for AutoGPT
- [ ] Replace `ai_settings.yaml` with Python code
- [ ] Replace plugins with `@infrarely.tool` functions
- [ ] Replace `.env` configuration with `infrarely.configure()`
- [ ] Replace autonomous loops with structured `agent.run()` calls
- [ ] Replace Pinecone/JSON memory with built-in `agent.memory`
- [ ] Add `agent.knowledge.add_documents()` for RAG
- [ ] Use `result.success` and `result.error` instead of try/catch
