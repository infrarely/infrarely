# Changelog

All notable changes to InfraRely will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> Note: Repository history was consolidated during v0.1.7 restructure.
> All code is intact. Version history documented below.

## v0.1.7 - 2026-03-29

### Added
- **Deterministic Routing Contracts**: User-definable `@infrarely.route()` for tool matching
  - Match keywords define when a tool is routed
  - Required parameters validation before execution
  - Type-safe parameter enforcement
  - Fallback strategy control (LLM_RESOLVE, named tool, or FAIL)
- **Visual Execution Traces**: ASCII-formatted trace rendering
  - `result.trace` now shows formatted execution flow
  - `ExecutionTrace.render()` produces readable box-drawing output
  - Shows step timing, tool calls, parameters, and outcomes
  - Human-readable without post-processing

### Changed
- **Tool decorator** now accepts `route=infrarely.route(...)` parameter
- **Result.trace** property: Returns rendered trace string instead of just trace_id
  - Backward compatible: falls back to trace_id if full trace unavailable
- Updated README examples to show deterministic routing in action

### What's working
- Deterministic tool routing via user-defined contracts
- Full visual execution traces with structured formatting
- Type-safe parameter validation before tool execution
- Agent creation and execution via infrarely.agent()
- Three-scope memory system (session, permanent, shared)
- Workflow DAG with dependency resolution and parallel execution
- Prompt injection detection and input sanitization
- Multi-agent delegation and broadcast
- LLM providers: OpenAI, Anthropic, Groq, Gemini, Ollama

### Known limitations
- No replay system (v0.2.0)
- Runtime guardrail violations not fully structured (v0.2.0)

## v0.1.6 - 2026-03-29

### What's working
- Agent creation and execution via infrarely.agent()
- Three-scope memory system (session, permanent, shared)
- Workflow DAG with dependency resolution and parallel execution
- Prompt injection detection and input sanitization
- Multi-agent delegation and broadcast
- LLM providers: OpenAI, Anthropic, Groq, Gemini, Ollama

### Known limitations
- Routing contracts are not user-definable yet (v0.1.7) ✓ SHIPPED
- No replay system (v0.1.7)
- No visual execution trace (v0.1.7) ✓ SHIPPED
- Runtime guardrail violations not structured (v0.1.7)

## [0.1.1] through [0.1.5] - 2026-03-12 to 2026-03-28
- Iterative bug fixes and stability improvements
- Groq provider updates
- Intent classifier refinements
- Memory system hardening
- Security layer improvements

## [0.1.0] - 2026-03-12

### Added
- Initial release of InfraRely (rebranded from legacy SDK + Student Agent)
- **Core framework**: Agent, Result, Config, Events, Decorators, Streaming
- **7-layer architecture**: Execution contracts → Capability graphs → Infrastructure → Verification → Adaptive intelligence → Multi-agent runtime → Autonomous evolution
- **Rule-based intent classifier**: Zero-token intent classification with weighted keyword matching
- **Multi-agent runtime**: OS-like kernel with scheduler, IPC, shared memory, RBAC, deadlock detection
- **Adaptive intelligence**: Self-optimizing routing, failure analysis, token optimization
- **Autonomous evolution**: Performance analysis, A/B testing, architecture proposals with policy guards
- **Security**: Prompt injection defense (7 types), input sanitization, key rotation, compliance logging
- **Memory**: Session/persistent/shared (SDK) + working/structured/long-term (execution)
- **Knowledge engine**: TF-IDF vector search, LLM bypass when confidence ≥ 85%
- **Workflow DAG engine**: Topological sort, auto-parallel execution, fallback steps
- **CLI**: `infrarely run`, `infrarely health`, `infrarely metrics`, `infrarely deploy`, `infrarely verify`
- **Integrations**: GitHub, Gmail, Slack, Postgres, Notion, Webhooks, REST API
- **Advanced**: HITL approval gates, evaluation suites, versioning, marketplace, multitenancy, ACP
- **LLM providers**: OpenAI, Anthropic, Groq, Google Gemini, Ollama (local)
- Zero external dependencies for core functionality