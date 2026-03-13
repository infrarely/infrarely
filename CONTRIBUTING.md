# Contributing to InfraRely

Thank you for your interest in contributing to InfraRely! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, constructive, and professional. We are building infrastructure that developers rely on — quality and reliability matter.

## Getting Started

### Prerequisites

- Python 3.10+
- Git

### Setup

```bash
git clone https://github.com/infrarely/infrarely.git
cd infrarely
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

### Linting

```bash
ruff check .
ruff format --check .
mypy infrarely/
```

## Project Structure

```
infrarely/
├── core/           # Agent core, config, decorators, streaming
├── runtime/        # Scheduling, message bus, sandbox, workflows
├── router/         # Intent classification, tool routing
├── memory/         # Working, long-term, structured memory
├── optimization/   # Routing optimizer, failure analysis, token optimization
├── platform/       # HITL, evaluation, marketplace, multitenancy, ACP
├── learning/       # Performance analysis, A/B testing, architecture
├── observability/  # Logging, metrics, dashboards, token budgets
├── security/       # Input sanitization, key rotation, compliance
├── tools/          # Base tool class, tool registry
├── agent/          # Agent pipeline, capability graph, execution
├── integrations/   # Provider integrations (OpenAI, Anthropic, etc.)
├── cli/            # Command-line interface
└── internal/       # Internal bridges and state management
```

## How to Contribute

### Reporting Bugs

- Use GitHub Issues
- Include: Python version, OS, InfraRely version, minimal reproduction steps
- Check existing issues before opening a new one

### Suggesting Features

- Open a GitHub Discussion or Issue
- Describe the use case, not just the solution
- Explain how it fits into agent infrastructure

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make your changes
4. Add or update tests
5. Run the full test suite: `pytest`
6. Run linting: `ruff check . && mypy infrarely/`
7. Commit with a clear message: `git commit -m "feat: add X to Y"`
8. Push and open a PR

### Commit Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — New feature
- `fix:` — Bug fix
- `docs:` — Documentation only
- `refactor:` — Code change that neither fixes a bug nor adds a feature
- `test:` — Adding or updating tests
- `chore:` — Maintenance tasks

### PR Guidelines

- Keep PRs focused — one feature or fix per PR
- Include tests for new functionality
- Update documentation if behavior changes
- Ensure CI passes before requesting review

## Architecture Principles

InfraRely is an **infrastructure SDK**, not an application. Contributions should:

1. **Be generic** — No application-specific logic (e.g., no student tools, no domain data)
2. **Be composable** — Components should work independently and together
3. **Be reliable** — Include error handling, circuit breakers, and graceful degradation
4. **Be observable** — Emit metrics and traces for debugging
5. **Be zero-boilerplate** — Minimize setup required by end users

## Areas for Contribution

- Runtime improvements (scheduling, isolation, persistence)
- New provider integrations
- Observability enhancements
- Security hardening
- Documentation and examples
- Performance optimization
- CLI improvements

## Questions?

Open a GitHub Discussion or reach out at **contribute@infrarely.dev**.
