# Changelog

All notable changes to InfraRely will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-12

### Added
- Initial release of InfraRely (rebranded from AOS SDK + Student Agent)
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
